//! Another model plugin's trunk that a plugin runs itself (Kev's and D1's decision trunks, MOSS-Image's text
//! encoder): the pack parsed and loaded through that plugin's table and forwarded the way the engine's driver
//! forwards it (`engines/mlx/plugin/driver.zig`), over a `Cache` the caller owns.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const abi = @import("../../abi/plugin.zig");
const common = @import("../../abi/common.zig");
const quant_abi = @import("../../abi/quant.zig");
const run = @import("run.zig");
const kv_cache = @import("../kv/cache.zig");
const cache_ops = @import("../kv/cache_ops.zig");
const resident = @import("../weights/resident.zig");
const weights_load = @import("../weights/load.zig");
const mergeConfigJson = @import("../config/merge.zig").mergeConfigJson;
const FsPackDir = @import("../util/pack_dir.zig").FsPackDir;

const Array = mlx.mlx_array;
const Ops = cache_ops.For(cache_ops.Entry);

pub const Trunk = struct {
    table: *const abi.ModelTable,
    parsed: abi.Parsed = .{},
    model: *anyopaque = undefined,
    s: mlx.mlx_stream,
    /// No recurrent layer: a cache's position is its KV step, else its own count (the driver's `Positions`).
    kv_step: bool = true,
    next_slot: u32 = 0,

    /// The pack at `dir` parsed by the one of `tables` that claims it (as the registry picks), `overrides`
    /// deep-merged into its config.json first; `load` loads it.
    pub fn parse(io: std.Io, a: std.mem.Allocator, dir: []const u8, tables: []const *const abi.ModelTable, overrides: []const u8) !Trunk {
        var fs = try FsPackDir.open(io, dir);
        defer fs.close();
        const pd = fs.packDir();
        const buf = try a.alloc(u8, 1 << 20);
        defer a.free(buf);
        var why: abi.Verdict = .{};
        const raw = pd.readFile("config.json", buf, &why) orelse return error.ConfigUnreadable;
        const cfg = try mergeConfigJson(a, raw, overrides);
        defer a.free(cfg);
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const model_type = try @import("../plugin.zig").modelTypeOf(arena.allocator(), cfg);
        var claims: [8]common.Claim = undefined;
        for (tables, claims[0..tables.len]) |tb, *c| c.* = tb.claims(.of(model_type), .of(cfg), &pd);
        const table = tables[(common.pick(claims[0..tables.len]) catch return error.TrunkUnsupported) orelse return error.TrunkUnsupported];
        var t: Trunk = .{ .table = table, .s = undefined };
        if (!table.parse(&pd, .of(cfg), .{}, null, false, &t.parsed).isOk()) return error.TrunkUnsupported;
        t.kv_step = t.parsed.cache.ssm_state.entries == 0;
        return t;
    }

    /// Loads the parsed pack on `s` from `weights` (handed over: the caller's renames of the pack's tensors), else
    /// from the pack's own tensors, as the driver loads a served pack.
    pub fn load(self: *Trunk, io: std.Io, a: std.mem.Allocator, dir: []const u8, s: mlx.mlx_stream, weights: ?resident.Weights) !void {
        var res = resident.Resident.adopt(weights orelse try weights_load.loadWeightsOpt(io, a, dir, .{ .keep_f16 = self.parsed.shell.act_dtype == .f16 }));
        // The model holds its own references: the map's go once the plugin has read it.
        defer res.deinit();
        const src = res.source();
        var fs = try FsPackDir.open(io, dir);
        defer fs.close();
        const pd = fs.packDir();
        var why: abi.Verdict = .{};
        self.model = self.table.load(&.{
            .model_dir = .of(dir),
            .dir = &pd,
            .parsed = &self.parsed,
            .settings_json = .of("{}"),
            .stream = .{ .ctx = s.ctx },
            .weights = &src,
        }, &why) orelse return if (std.mem.eql(u8, why.why.slice(), "MissingWeight")) error.MissingWeight else error.TrunkLoadFailed;
        self.s = s;
    }

    /// Unloads the model (`loaded`: `load` returned) and frees the parse.
    pub fn close(self: *Trunk, loaded: bool) void {
        if (loaded) self.table.unload(self.model);
        self.table.free_parsed(&self.parsed);
    }

    /// The trunk's embedding table or LM head as loaded (borrowed until `close`).
    pub fn weight(self: *const Trunk, which: abi.TrunkWeight) !quant_abi.Weight {
        const get = (self.table.spec orelse return error.NoTrunkWeight).trunk_weight orelse return error.NoTrunkWeight;
        var w: quant_abi.Weight = .{};
        if (!get(self.model, which, &w).isOk()) return error.NoTrunkWeight;
        return w;
    }

    pub const Step = struct {
        /// The final-normed hidden rows instead of the logits.
        hidden: bool = true,
        capture_layers: []const u32 = &.{},
        /// Filled with each captured layer's rows (new references), `capture_layers.len` of them.
        captured: []Array = &.{},
        mask_add: Array = .{ .ctx = null },
    };

    /// One forward of `ids` (`[1, rows]`) after what `cache` holds: the hidden rows `[1, rows, hidden]` (or the
    /// logits `[1, rows, vocab]`), a new reference.
    pub fn forward(self: *Trunk, cache: *Cache, ids: Array, step: Step) !Array {
        const rows: u32 = @intCast(mlx.getShape(ids)[1]);
        cache.slot = .{ .kv = &cache.kv, .ssm = cache.entries };
        if (cache.id == 0) {
            self.next_slot +%= 1;
            if (self.next_slot == 0) self.next_slot = 1;
            cache.id = self.next_slot;
        }
        const handles = [_]abi.Handle{cache.slot.handle()};
        const slot_ids = [_]u32{cache.id};
        const base: usize = if (self.kv_step) cache.kv.step else cache.pos;
        const pos = [_]u32{@intCast(base)};
        var captured: [64]abi.Array = undefined;
        if (step.capture_layers.len > captured.len or step.captured.len != step.capture_layers.len) return error.TrunkCaptureMismatch;
        @memset(captured[0..step.capture_layers.len], .{});
        const f: abi.Forward = .{
            .mode = if (rows == 1) .decode else .prefill,
            .logits = if (step.hidden) .none else .all_rows,
            .want_hidden = step.hidden,
            .rows = rows,
            .tokens = .{},
            .ids = .{ .ctx = ids.ctx },
            .slots = .of(&slot_ids),
            .positions = .of(&pos),
            .caches = .of(&handles),
            .capture_layers = .of(step.capture_layers),
            .cache_ops = &Ops.ops,
            .eval_every = self.evalEvery(rows, base + rows),
            .mask_add = .{ .ctx = step.mask_add.ctx },
        };
        var out: abi.ForwardOut = .{ .captured = &captured };
        const v = self.table.forward(self.model, &f, &out);
        if (!v.isOk()) {
            for ([_]abi.Array{ out.logits, out.hidden }) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(.{ .ctx = x.ctx });
            };
            for (captured[0..step.capture_layers.len]) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(.{ .ctx = x.ctx });
            };
            return error.TrunkForwardFailed;
        }
        cache.pos += rows;
        for (step.captured, captured[0..step.capture_layers.len]) |*dst, x| dst.* = .{ .ctx = x.ctx };
        if (step.hidden) {
            if (out.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
            return .{ .ctx = out.hidden.ctx };
        }
        if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
        const logits: Array = .{ .ctx = out.logits.ctx };
        defer _ = mlx.mlx_array_free(logits);
        var shaped = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&shaped, logits, &[_]c_int{ 1, @intCast(rows), -1 }, 3, self.s));
        return shaped;
    }

    /// The driver's eval cadence for a forward of `rows` over `total_kv` keys (dense KV).
    fn evalEvery(self: *const Trunk, rows: u32, total_kv: u64) u32 {
        if (rows < run.PREFILL_EVAL_MIN_SEQ) return 0;
        const sh = self.parsed.shell;
        const cadence = if (self.kv_step) run.EVAL_EVERY_N_LAYERS else run.PREFILL_EVAL_CADENCE_DEFAULT;
        return run.prefillEvalCadence(cadence, sh.head_dim, sh.head_dim, sh.num_attention_heads, sh.num_key_value_heads, rows, total_kv, false);
    }
};

/// One sequence's state: a dense `KVCache` and one recurrent entry per layer, and its position.
pub const Cache = struct {
    a: std.mem.Allocator,
    kv: kv_cache.KVCache,
    entries: []cache_ops.Entry,
    pos: usize = 0,
    id: u32 = 0,
    slot: Ops.Slot = undefined,

    pub fn init(a: std.mem.Allocator, layers: u32) !Cache {
        var kv = try kv_cache.KVCache.init(a, layers);
        errdefer kv.deinit();
        const entries = try a.alloc(cache_ops.Entry, layers);
        @memset(entries, .{});
        return .{ .a = a, .kv = kv, .entries = entries };
    }

    pub fn deinit(self: *Cache) void {
        for (self.entries) |*e| for ([_]Array{ e.conv_state, e.ssm_state, e.spec_state_seq, e.spec_conv_input, e.spec_state_in }) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        };
        self.a.free(self.entries);
        self.kv.deinit();
    }

    pub const Snapshot = struct {
        a: std.mem.Allocator,
        kv: kv_cache.KVCacheSnapshot,
        states: []State,
        pos: usize,

        const State = struct { conv: Array, ssm: Array, initialized: bool };

        pub fn deinit(self: *Snapshot) void {
            for (self.states) |st| for ([_]Array{ st.conv, st.ssm }) |x| {
                if (x.ctx != null) _ = mlx.mlx_array_free(x);
            };
            self.a.free(self.states);
            self.kv.deinit();
        }
    };

    /// What a later `restore` puts back: the KV rows, each layer's recurrent state and the position.
    pub fn snapshot(self: *const Cache) !Snapshot {
        var kv = try self.kv.snapshot();
        errdefer kv.deinit();
        const states = try self.a.alloc(Snapshot.State, self.entries.len);
        for (states, self.entries) |*st, e| st.* = .{ .conv = ref(e.conv_state), .ssm = ref(e.ssm_state), .initialized = e.initialized };
        return .{ .a = self.a, .kv = kv, .states = states, .pos = self.pos };
    }

    pub fn restore(self: *Cache, snap: *const Snapshot) !void {
        try self.kv.restore(&snap.kv);
        for (self.entries, snap.states) |*e, st| {
            for ([_]*Array{ &e.conv_state, &e.ssm_state }) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x.*);
                x.* = .{ .ctx = null };
            };
            e.conv_state = ref(st.conv);
            e.ssm_state = ref(st.ssm);
            e.initialized = st.initialized;
        }
        self.pos = snap.pos;
    }

    fn ref(x: Array) Array {
        if (x.ctx == null) return x;
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, x);
        return r;
    }
};
