//! The conformance suite: what every plugin table holds, run on each registered plugin and template in `zig build
//! test` (from the generated registry) and on an outside plugin's dylib (`mlx-serve plugin check`, a `Subject` from
//! its manifest). MLX-free by construction for claims and parse (the build checks each `config.zig`'s imports);
//! `expectExactness` runs a plugin's declared exactness contracts on an engine serving it.

const std = @import("std");
const env = @import("../../env.zig");
const common = @import("../../abi/common.zig");
const model_abi = @import("../../abi/plugin.zig");
const media_abi = @import("../../abi/media.zig");
const quant_abi = @import("../../abi/quant.zig");
const engine_abi = @import("../../abi/engine.zig");
const exactness = @import("../../abi/exactness_test.zig");
const sdk_plugin = @import("../plugin.zig");
const fixture = @import("fixture.zig");
const test_cache = @import("cache_ops.zig");
const mlx = @import("../../mlx/ffi.zig");
const Resident = @import("../weights/resident.zig").Resident;
const loadWeightsOpt = @import("../weights/load.zig").loadWeightsOpt;
const streamed = @import("../weights/streamed.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const Str = common.Str;
const Verdict = common.Verdict;
const Array = common.Array;
const Claim = common.Claim;

/// Every template and registered plugin of the generated registry `R`, then one claimant per fixture pack.
pub fn expectRegistry(comptime R: type) !void {
    inline for (R.templates) |M| try expectPlugin(M);
    inline for (R.plugins) |M| try expectPlugin(M);
    try expectOneClaimant(&R.entries, &R.model_configs);
}

/// A plugin as the suite sees it: a built-in from its root (`subjectOf`) or an outside one from its dylib's manifest.
pub const Subject = struct {
    name: []const u8,
    /// The `model_type`s it declares.
    claims: []const []const u8 = &.{},
    table: Table,
    fixture: fixture.Fixture,

    pub const Table = union(sdk_plugin.Kind) {
        model: *const model_abi.ModelTable,
        media: *const media_abi.MediaTable,
        quant: *const quant_abi.QuantTable,
        engine: *const engine_abi.Engine,
    };
};

/// Plugin root `M` as a subject; its fixture is built in `a`.
pub fn subjectOf(comptime M: type, a: Allocator) !Subject {
    return .{
        .name = M.plugin.name,
        .claims = M.plugin.claims,
        .table = @unionInit(Subject.Table, @tagName(M.plugin.kind), &M.table),
        .fixture = try M.conformanceFixture(a),
    };
}

pub fn expectPlugin(comptime M: type) !void {
    if (@import("media_pack.zig").ramShort(M)) return;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try expectSubject(testing.io, arena.allocator(), try subjectOf(M, arena.allocator()));
}

/// The suite on one subject; `a` should be an arena (the suite frees nothing it allocates in it).
pub fn expectSubject(io: std.Io, a: Allocator, s: Subject) !void {
    // An error an earlier test left latched would fail this plugin's job; drop it, saying whose it was not.
    var buf: [512]u8 = undefined;
    if (mlx.takeError(&buf)) |stale| std.debug.print("conformance: plugin {s}: dropped an earlier test's MLX error ({s})\n", .{ s.name, stale });
    switch (s.table) {
        .model => |t| try expectModel(io, a, s, t),
        .media => |t| try expectMedia(io, a, s, t),
        .quant => |t| try expectQuant(s, t),
        .engine => |t| try expectEngine(io, a, s, t),
    }
}

/// An engine plugin on its fixture staged on disk: it probes and opens it on its own thread (an outside engine never
/// calls MLX), serves one request (prefilled, tokens, one `finished`) and closes. The full engine contract
/// (`abi/contract_test.zig`) runs on the kit's example engine in `zig build test`.
fn expectEngine(io: std.Io, a: Allocator, s: Subject, t: *const engine_abi.Engine) !void {
    if (t.size != @sizeOf(engine_abi.Engine) or t.version != engine_abi.version) return fail(s.name, "table size or version is not this ABI's", .{});
    var staged = try Staged.init(io, a, s.fixture);
    defer staged.deinit();
    const cfg = s.fixture.file("config.json") orelse "";
    var v: Verdict = .{};
    if (!t.probe(.of(staged.path), .of(cfg), &v)) return fail(s.name, "does not probe its own fixture", v.why);
    var run: EngineRun = .{};
    const host: engine_abi.Host = .{ .ctx = &run, .log = EngineRun.log, .wake = EngineRun.wake };
    const o: engine_abi.Open = .{ .model_dir = .of(staged.path), .model_type = .{}, .config_json = .of(cfg), .context = 0, .max_concurrent = 1, .memory_budget = 1 << 30, .settings_json = .{}, .options_json = .{}, .host = &host };
    var info: engine_abi.Info = .{ .name = .{}, .version = .{}, .caps = .{}, .threading = .host_gpu_thread, .context_window = 0, .max_concurrent = 0, .embedding_dims = 0, .eos = .{}, .startup = .{} };
    const e = t.open(&o, &info, &v) orelse return fail(s.name, "does not open its own fixture", v.why);
    defer t.close(e);
    if (info.size != @sizeOf(engine_abi.Info) or info.name.len == 0 or info.context_window == 0 or info.max_concurrent == 0) return fail(s.name, "info lacks its size, name, window or concurrency", .{});
    if (info.threading != .own_thread or t.step != null) return fail(s.name, "an outside engine runs its own runtime on its own thread (`Threading.own_thread`, no `step`)", .{});
    const prompt = [_]u32{ 1, 2, 3 };
    const r: engine_abi.Request = .{ .prompt = .of(&prompt), .max_tokens = 4 };
    const sv = t.submit(e, 1, &r, .{ .ctx = &run, .event = EngineRun.event });
    if (!sv.isOk()) return fail(s.name, "refuses a plain request", sv.why);
    var waited: u32 = 0;
    while (run.finished.load(.acquire) == 0) : (waited += 1) {
        if (waited == 30_000) return fail(s.name, "a plain request never finished", .{});
        io.sleep(.fromMilliseconds(1), .awake) catch {};
    }
    if (run.bad.load(.acquire)) return fail(s.name, "a request's events are out of order (prefilled, tokens, one finished)", .{});
    if (run.reason != .length or run.tokens.load(.acquire) != 4) return fail(s.name, "a plain request does not run to its max_tokens", .{});
}

/// What one request's events looked like, recorded on the engine's thread.
const EngineRun = struct {
    prefilled: std.atomic.Value(u32) = .init(0),
    tokens: std.atomic.Value(u32) = .init(0),
    finished: std.atomic.Value(u32) = .init(0),
    bad: std.atomic.Value(bool) = .init(false),
    reason: @FieldType(engine_abi.Event, "reason") = .stop,

    fn log(_: *anyopaque, _: u8, _: Str) callconv(.c) void {}
    fn wake(_: *anyopaque) callconv(.c) void {}
    fn event(ctx: *anyopaque, id: u64, ev: *const engine_abi.Event) callconv(.c) engine_abi.Action {
        const r: *EngineRun = @ptrCast(@alignCast(ctx));
        if (id != 1 or ev.size < @sizeOf(engine_abi.Event) or r.finished.load(.acquire) > 0) r.bad.store(true, .release);
        switch (ev.kind) {
            .prefilled => if (r.prefilled.fetchAdd(1, .acq_rel) > 0 or r.tokens.load(.acquire) > 0) r.bad.store(true, .release),
            .tokens => {
                if (r.prefilled.load(.acquire) == 0 or ev.tokens.len == 0) r.bad.store(true, .release);
                _ = r.tokens.fetchAdd(@intCast(ev.tokens.len), .acq_rel);
            },
            .audio => r.bad.store(true, .release),
            .finished => {
                r.reason = ev.reason;
                _ = r.finished.fetchAdd(1, .release);
            },
        }
        return .go;
    }
};

fn fail(name: []const u8, comptime what: []const u8, why: Str) error{Conformance} {
    std.debug.print("conformance: plugin {s}: " ++ what ++ " ({s})\n", .{ name, why.slice() });
    return error.Conformance;
}

fn expectOk(name: []const u8, comptime what: []const u8, v: Verdict) !void {
    if (v.isOk()) return;
    var buf: [512]u8 = undefined;
    if (mlx.takeError(&buf)) |text| std.debug.print("conformance: plugin {s}: latched MLX error: {s}\n", .{ name, text });
    return fail(name, what, v.why);
}

fn expectHead(s: Subject, comptime T: type, version: u32, t: *const T) !void {
    if (t.size != @sizeOf(T) or t.version != version) return fail(s.name, "table size or version is not this ABI's", .{});
    if (!std.mem.eql(u8, t.name.slice(), s.name)) return fail(s.name, "table name differs from its declaration", t.name);
}

/// A plugin's fixture staged on disk under `$TMPDIR` (MLX loads by path) and as a `PackDir` over the same files.
const Staged = struct {
    io: std.Io,
    path: []const u8,
    dir: common.PackDir,

    fn init(io: std.Io, a: Allocator, fx: fixture.Fixture) !Staged {
        var rnd: [8]u8 = undefined;
        io.random(&rnd);
        const base = std.mem.trimEnd(u8, if (env.get(.TMPDIR)) |t| std.mem.span(t) else "/tmp", "/");
        const path = try std.fmt.allocPrint(a, "{s}/mlx-serve-conformance-{x}", .{ base, std.mem.readInt(u64, &rnd, .little) });
        var dir = try std.Io.Dir.cwd().createDirPathOpen(io, path, .{});
        defer dir.close(io);
        errdefer std.Io.Dir.cwd().deleteTree(io, path) catch {};
        for (fx.files) |f| {
            if (std.fs.path.dirname(f.name)) |d| try dir.createDirPath(io, d);
            try dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        }
        const mem = try a.create(fixture.MemDir);
        mem.* = .{ .files = fx.files };
        return .{ .io = io, .path = path, .dir = mem.packDir() };
    }
    fn deinit(self: *Staged) void {
        std.Io.Dir.cwd().deleteTree(self.io, self.path) catch {};
    }
};

/// A model plugin's fixture, parsed (full) and loaded on the CPU stream. For the suite and plugin tests (`Pack`).
pub const Loaded = struct {
    name: []const u8,
    table: *const model_abi.ModelTable,
    staged: Staged,
    stream: mlx.mlx_stream,
    parsed: model_abi.Parsed = .{},
    model: *anyopaque = undefined,
    /// The map's zeroed experts (`Opts.zero_experts`), held to ask whether anything evaluated them.
    zeroed: std.ArrayList(mlx.mlx_array) = .empty,

    pub const Opts = struct {
        /// The load's settings (`LoadCtx.settings_json`, the parse's too).
        settings: []const u8 = "",
        /// The fixture's routed experts affine 8-bit on disk (`quantizeExperts`).
        quantize_experts: bool = false,
        /// The engine's map holds those experts zeroed: only a load that reads them from the pack serves them.
        zero_experts: bool = false,
        /// On the GPU stream (the stream's reads land behind Metal events), not the CPU's.
        gpu: bool = false,
    };

    pub fn open(io: std.Io, a: Allocator, name: []const u8, table: *const model_abi.ModelTable, fx: fixture.Fixture) !Loaded {
        return openWith(io, a, name, table, fx, .{});
    }

    pub fn openWith(io: std.Io, a: Allocator, name: []const u8, table: *const model_abi.ModelTable, fx: fixture.Fixture, opts: Opts) !Loaded {
        const cfg = fx.file("config.json") orelse return error.FixtureWithoutConfig;
        var self: Loaded = .{ .name = name, .table = table, .staged = try Staged.init(io, a, fx), .stream = if (opts.gpu) mlx.mlx_default_gpu_stream_new() else mlx.mlx_default_cpu_stream_new() };
        errdefer {
            _ = mlx.mlx_stream_free(self.stream);
            self.staged.deinit();
        }
        if (opts.quantize_experts) try quantizeExperts(io, a, self.staged.path, fx);
        try expectOk(name, "parse of its fixture", table.parse(&self.staged.dir, .of(cfg), .of(opts.settings), null, false, &self.parsed));
        errdefer table.free_parsed(&self.parsed);
        // The engine's loader reads the fixture's tensors, as it does for a served pack.
        // The towers its parse declares load too, as a served load keeps them.
        const caps = self.parsed.caps;
        var weights = Resident.adopt(try loadWeightsOpt(io, std.heap.c_allocator, self.staged.path, .{ .vision = caps.images or caps.video or caps.audio, .audio_tower = caps.audio }));
        defer weights.deinit();
        if (opts.zero_experts) {
            var it = weights.weights.map.iterator();
            while (it.next()) |e| if (streamed.expertTensor(e.key_ptr.*)) {
                var z = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_zeros(&z, mlx.mlx_array_shape(e.value_ptr.*), mlx.mlx_array_ndim(e.value_ptr.*), mlx.mlx_array_dtype(e.value_ptr.*), self.stream));
                _ = mlx.mlx_array_free(e.value_ptr.*);
                e.value_ptr.* = z;
                var held = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&held, z));
                try self.zeroed.append(std.heap.c_allocator, held);
            };
        }
        const src = weights.source();
        var why: Verdict = .{};
        self.model = table.load(&.{ .model_dir = .of(self.staged.path), .dir = &self.staged.dir, .parsed = &self.parsed, .settings_json = .of(opts.settings), .stream = .{ .ctx = self.stream.ctx }, .weights = &src }, &why) orelse {
            // A plugin whose kernels this device cannot run says so by name; the suite has nothing to check here.
            if (why.code == .unsupported) return error.UnsupportedOnThisDevice;
            return fail(name, "load of its fixture", why.why);
        };
        return self;
    }
    pub fn close(self: *Loaded) void {
        // Work still queued on the stream reads the staged files' mappings: a removed file is a SIGBUS there.
        _ = mlx.mlx_synchronize(self.stream);
        self.table.unload(self.model);
        self.table.free_parsed(&self.parsed);
        for (self.zeroed.items) |z| _ = mlx.mlx_array_free(z);
        self.zeroed.deinit(std.heap.c_allocator);
        _ = mlx.mlx_stream_free(self.stream);
        self.staged.deinit();
    }
};

/// Rewrites the staged fixture's float routed-expert weights as affine 8-bit at group 32 (scales and biases bf16, as
/// mlx-lm stores them) and indexes every shard (`model.safetensors.index.json`), as a downloaded pack is.
fn quantizeExperts(io: std.Io, a: Allocator, dir: []const u8, fx: fixture.Fixture) !void {
    // A safetensors load evaluates on the CPU only.
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var index: std.Io.Writer.Allocating = .init(a);
    try index.writer.writeAll("{\"metadata\":{},\"weight_map\":{");
    var first = true;
    for (fx.files) |f| if (std.mem.endsWith(u8, f.name, ".safetensors")) {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, f.name }, 0);
        var in = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(in);
        var meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_load_safetensors(&in, &meta, path.ptr, s));
        const out = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(out);
        const all = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(all);
        const it = mlx.mlx_map_string_to_array_iterator_new(in);
        defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
        var key: ?[*:0]const u8 = null;
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        while (mlx.mlx_map_string_to_array_iterator_next(&key, &v, it) == 0) {
            const k = std.mem.span(key.?);
            const sh = mlx.mlx_array_shape(v);
            const nd = mlx.mlx_array_ndim(v);
            const float = switch (mlx.mlx_array_dtype(v)) {
                .float32, .float16, .bfloat16 => true,
                else => false,
            };
            const parts: [3]mlx.mlx_array = if (std.mem.endsWith(u8, k, ".weight") and streamed.expertTensor(k) and float and nd >= 2 and @mod(sh[nd - 1], 32) == 0) blk: {
                var q = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(q);
                try mlx.check(mlx.mlx_quantize(&q, v, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
                var r: [3]mlx.mlx_array = undefined;
                for (&r, 0..) |*x, i| {
                    var raw = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_vector_array_get(&raw, q, i));
                    x.* = if (i == 0) raw else c: {
                        var c = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_astype(&c, raw, .bfloat16, s));
                        _ = mlx.mlx_array_free(raw);
                        break :c c;
                    };
                }
                break :blk r;
            } else .{ v, .{ .ctx = null }, .{ .ctx = null } };
            const base = k[0 .. k.len - (if (parts[1].ctx != null) ".weight".len else 0)];
            for (parts, [_][]const u8{ if (parts[1].ctx != null) ".weight" else "", ".scales", ".biases" }) |x, suffix| if (x.ctx != null) {
                const name = try std.fmt.allocPrintSentinel(a, "{s}{s}", .{ base, suffix }, 0);
                try mlx.check(mlx.mlx_map_string_to_array_insert(out, name.ptr, x));
                try mlx.check(mlx.mlx_vector_array_append_value(all, x));
                try index.writer.print("{s}\"{s}\":\"{s}\"", .{ if (first) "" else ",", name, f.name });
                first = false;
                if (parts[1].ctx != null) _ = mlx.mlx_array_free(x);
            };
        }
        // Every array read before the file under it is replaced.
        try mlx.check(mlx.mlx_eval(all));
        try mlx.check(mlx.mlx_save_safetensors(path.ptr, out, meta));
    };
    try index.writer.writeAll("}}");
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
}

/// Plugin root `M`'s fixture, loaded (`Loaded`).
pub fn Pack(comptime M: type) type {
    return struct {
        pub fn open(a: Allocator) !Loaded {
            return Loaded.open(testing.io, a, M.plugin.name, &M.table, try M.conformanceFixture(a)) catch |e|
                if (e == error.UnsupportedOnThisDevice) error.SkipZigTest else e;
        }
    };
}

/// The values of `arr` as f32 (any float dtype); consumes the reference.
pub fn consumeF32(a: Allocator, arr: Array) ![]f32 {
    const x: mlx.mlx_array = .{ .ctx = arr.ctx };
    defer _ = mlx.mlx_array_free(x);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const data = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return a.dupe(f32, data[0..n]);
}

// ── model ──

fn expectModel(io: std.Io, a: Allocator, s: Subject, t: *const model_abi.ModelTable) !void {
    const name = s.name;
    try expectHead(s, model_abi.ModelTable, model_abi.version, t);
    const fx = s.fixture;
    const cfg = fx.file("config.json") orelse return error.FixtureWithoutConfig;
    const mem: fixture.MemDir = .{ .files = fx.files };
    const dir = mem.packDir();
    const none: fixture.MemDir = .{ .files = &.{} };
    const empty = none.packDir();

    // Claims: its fixture and every declared type, never a stranger.
    if (t.claims(.of(try sdk_plugin.modelTypeOf(a, cfg)), .of(cfg), &dir) == .no) return fail(name, "does not claim its own fixture", .{});
    for (s.claims) |mt| if (t.claims(.of(mt), .of(cfg), &dir) == .no) return fail(name, "does not claim a declared model_type", .of(mt));
    if (t.claims(.of("conformance_stranger"), .of("{}"), &empty) != .no) return fail(name, "claims a stranger's pack", .{});

    // Stub parse (discovery): sized structs, a sane shell; a broken config is refused by name with nothing to free.
    var stub: model_abi.Parsed = .{};
    try expectOk(name, "stub parse of its fixture", t.parse(&dir, .of(cfg), .{}, null, true, &stub));
    defer t.free_parsed(&stub);
    try expectParsed(name, &stub);
    var bad: model_abi.Parsed = .{};
    const v = t.parse(&dir, .of("{"), .{}, null, true, &bad);
    if (v.isOk() or v.why.len == 0 or bad.private != null) return fail(name, "a broken config.json is not refused by name", .{});

    var pack = Loaded.open(io, a, name, t, fx) catch |e| if (e == error.UnsupportedOnThisDevice) return else return e;
    defer pack.close();
    try expectParsed(name, &pack.parsed);

    // Cache layout: one sized entry per layer; it decides whether the bill may price KV.
    const n_layers = pack.parsed.shell.num_hidden_layers;
    const layers = try a.alloc(model_abi.LayerCache, n_layers + 1);
    const n = if (t.cache.layout) |f| f(&pack.parsed, null, layers.ptr, @intCast(layers.len)) else n_layers;
    if (n != n_layers) return fail(name, "cache layout length is not num_hidden_layers", .{});
    var attention: u32 = if (t.cache.layout == null) n else 0;
    if (t.cache.layout != null) for (layers[0..n]) |l| {
        if (l.size != @sizeOf(model_abi.LayerCache)) return fail(name, "a cache layout entry has another size", .{});
        if (l.kind == .full or l.kind == .sliding) attention += 1;
    };

    // Bill: sized, deterministic, no KV without an attention layer.
    var b1: model_abi.Bill = .{};
    var b2: model_abi.Bill = .{};
    const shape: model_abi.Shape = .{ .context = 4096 };
    t.bill(&pack.parsed, &shape, &b1);
    t.bill(&pack.parsed, &shape, &b2);
    if (b1.size != @sizeOf(model_abi.Bill)) return fail(name, "bill has another size", .{});
    inline for (@typeInfo(model_abi.Bill).@"struct".field_names) |f| if (@field(b1, f) != @field(b2, f)) return fail(name, "bill is not deterministic: " ++ f, .{});
    if (attention == 0 and b1.kv_per_token != 0) return fail(name, "bills KV with no attention layer", .{});

    inline for (@typeInfo(model_abi.Mode).@"enum".field_names) |f| try expectForward(a, &pack, @field(model_abi.Mode, f));
    if (pack.parsed.caps.experts_stream) try expectStreamedExperts(io, a, name, t, fx);
}

/// A plugin that streams its experts (`Caps.experts_stream`) serves the same logits streamed as in memory, over its
/// fixture with the routed experts affine 8-bit (what the stream reads). The streamed load's map holds them zeroed, so
/// only a load that reads them from the pack matches.
pub fn expectStreamedExperts(io: std.Io, a: Allocator, name: []const u8, t: *const model_abi.ModelTable, fx: fixture.Fixture) !void {
    // Said, not skipped: the registry runs every plugin in one test, and a skip would end it at the first.
    if (!streamed.built or mlx.noGpuBackend() or mlx.virtualGpu()) {
        std.debug.print("conformance: plugin {s}: streamed experts unchecked here (no expert streamer or no real GPU)\n", .{name});
        return;
    }
    var resident = try Loaded.openWith(io, a, name, t, fx, .{ .quantize_experts = true, .gpu = true });
    defer resident.close();
    var live = try Loaded.openWith(io, a, name, t, fx, .{ .quantize_experts = true, .zero_experts = true, .gpu = true, .settings = "{\"stream_experts\":true}" });
    defer live.close();
    try expectSameLogits(name, try promptThenStep(a, &resident), try promptThenStep(a, &live), "streamed experts serve other logits than the same experts in memory");
    // A batched decode over two row groups of 8 (GLM's) serves streamed what it serves in memory; a streamed load may
    // run single-flight (refused `unsupported`). A fixture's few experts all fit the transient rows, so this cannot
    // see a fetch rewrite rows a pending group still reads (glm5_next evaluates each group first).
    if (model_abi.declaresMode(resident.parsed.caps, .batched_rows)) {
        const want = (try batchedStep(a, &resident, 16)) orelse return fail(name, "a declared batched decode is refused", .{});
        if (try batchedStep(a, &live, 16)) |got| try expectSameLogits(name, want, got, "a batched decode over streamed experts serves other logits than in memory");
    }
    // Logits cannot see a load that keeps its experts too: the zeroed copies the map holds stay unread.
    for (live.zeroed.items) |z| {
        var read = false;
        _ = mlx._mlx_array_is_available(&read, z);
        if (read) return fail(name, "a streamed load made the experts it streams resident", .{});
    }
}

fn expectSameLogits(name: []const u8, want: []const f32, got: []const f32, comptime why: []const u8) !void {
    var top: f32 = 0;
    var diff: f32 = 0;
    for (want, got) |w, g| {
        top = @max(top, @abs(w));
        diff = @max(diff, @abs(w - g));
    }
    if (!(diff <= 1e-3 * @max(1, top))) {
        std.debug.print("conformance: plugin {s}: streamed logits differ by {d} (largest {d})\n", .{ name, diff, top });
        return fail(name, why, .{});
    }
}

/// The logits of one batched decode step over `n` fresh slots, a token each; null when the load refuses the mode.
fn batchedStep(a: Allocator, pack: *Loaded, n: u32) !?[]f32 {
    const vocab = pack.parsed.shell.vocab_size;
    const tokens = try a.alloc(u32, n);
    const slot_ids = try a.alloc(u32, n);
    const positions = try a.alloc(u32, n);
    for (tokens, slot_ids, 0..) |*tok, *s, i| {
        tok.* = @intCast((i * 7 + 1) % vocab);
        s.* = @intCast(i);
    }
    @memset(positions, 0);
    const caches = try a.alloc(test_cache.TestCache, n);
    const handles = try a.alloc(model_abi.Handle, n);
    for (caches, handles) |*c, *h| {
        c.* = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        h.* = c.handle();
    }
    defer for (caches) |*c| c.deinit();
    var fo: model_abi.ForwardOut = .{};
    const v = pack.table.forward(pack.model, &.{ .mode = .batched_rows, .logits = .last_row, .rows = 1, .tokens = .of(tokens), .slots = .of(slot_ids), .positions = .of(positions), .caches = .of(handles), .cache_ops = &test_cache.ops }, &fo);
    if (v.code == .unsupported and fo.logits.ctx == null) return null;
    // A fixture's bank (a row or two) is too small for MLX's row-block class a wide batch takes; a pack's holds dozens.
    if (std.mem.eql(u8, v.why.slice(), "StreamedBankTooFewRows")) return null;
    try expectOk(pack.name, "batched decode over streamed experts", v);
    if (fo.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.hidden.ctx });
    const l = try consumeF32(a, fo.logits);
    if (l.len != n * vocab) return fail(pack.name, "batched logits are not [slots, vocab]", .{});
    return l;
}

/// The logits of a four-token prompt (last row) then of one decode step on its cache, back to back.
fn promptThenStep(a: Allocator, pack: *Loaded) ![]f32 {
    const vocab = pack.parsed.shell.vocab_size;
    var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const out = try a.alloc(f32, 2 * vocab);
    for ([_]u32{ 4, 1 }, [_]u32{ 0, 4 }, 0..) |rows, pos, k| {
        const tokens = try a.alloc(u32, rows);
        for (tokens, 0..) |*tok, i| tok.* = @intCast((pos + i) * 7 % vocab + 1);
        const handles = [_]model_abi.Handle{cache.handle()};
        var fo: model_abi.ForwardOut = .{};
        const mode: model_abi.Mode = if (rows > 1) .prefill else .decode;
        try expectOk(pack.name, "forward over streamed experts", pack.table.forward(pack.model, &.{ .mode = mode, .rows = rows, .tokens = .of(tokens), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{pos}), .caches = .of(&handles), .cache_ops = &test_cache.ops }, &fo));
        if (fo.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.hidden.ctx });
        const l = try consumeF32(a, fo.logits);
        if (l.len != vocab) return fail(pack.name, "logits are not [1, vocab]", .{});
        @memcpy(out[k * vocab ..][0..vocab], l);
    }
    return out;
}

fn expectParsed(name: []const u8, p: *const model_abi.Parsed) !void {
    inline for (.{ "shell", "chat", "tokenizer", "cache", "spec", "placeholder_ids", "embed", "weights" }) |field| {
        if (@field(p, field).size != @sizeOf(@TypeOf(@field(p, field)))) return fail(name, "parsed " ++ field ++ " has another size", .{});
    }
    if (p.size != @sizeOf(model_abi.Parsed)) return fail(name, "parsed has another size", .{});
    const s = p.shell;
    if (s.vocab_size == 0 or s.hidden_size == 0 or s.num_hidden_layers == 0) return fail(name, "shell lacks its dimensions", .{});
    if (s.eos_count > model_abi.max_eos or p.tokenizer.extra_eos_count > model_abi.max_eos) return fail(name, "an eos count past its array", .{});
}

/// One forward per mode: a declared mode returns [R, vocab] finite logits (an encoder: its hidden rows), an
/// undeclared one is refused `unsupported`.
fn expectForward(a: Allocator, pack: *Loaded, mode: model_abi.Mode) !void {
    const name = pack.name;
    const vocab = pack.parsed.shell.vocab_size;
    const slots: u32, const rows: u32, const logits: model_abi.Logits = switch (mode) {
        .prefill => .{ 1, 4, .last_row },
        .decode => .{ 1, 1, .last_row },
        .verify => .{ 1, 3, .all_rows },
        .batched_rows => .{ 2, 1, .last_row },
    };
    const tokens = try a.alloc(u32, slots * rows);
    for (tokens, 0..) |*tok, i| tok.* = @intCast((i * 7 + 1) % vocab);
    const slot_ids = try a.alloc(u32, slots);
    for (slot_ids, 0..) |*s, i| s.* = @intCast(i);
    const positions = try a.alloc(u32, slots);
    @memset(positions, 0);
    // Each slot reads and writes a plain cache through `cache_ops`, as the engine's slots do.
    const caches = try a.alloc(test_cache.TestCache, slots);
    const handles = try a.alloc(model_abi.Handle, slots);
    for (caches, handles) |*c, *h| {
        c.* = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        h.* = c.handle();
    }
    defer for (caches) |*c| c.deinit();
    var out: model_abi.ForwardOut = .{};
    const v = pack.table.forward(pack.model, &.{ .mode = mode, .logits = logits, .rows = rows, .tokens = .of(tokens), .slots = .of(slot_ids), .positions = .of(positions), .caches = .of(handles), .cache_ops = &test_cache.ops }, &out);
    if (!model_abi.declaresMode(pack.parsed.caps, mode)) {
        if (v.code != .unsupported or out.logits.ctx != null) return fail(name, "an undeclared forward mode is not refused unsupported", .{});
        return;
    }
    try expectOk(name, "forward in a declared mode", v);
    if (pack.parsed.caps.embed) return expectHidden(name, a, out, slots, rows, pack.parsed.shell.hidden_size);
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    const x: mlx.mlx_array = .{ .ctx = out.logits.ctx };
    if (x.ctx == null) return fail(name, "forward returned no logits", .{});
    const want_rows = if (logits == .last_row) slots else slots * rows;
    const shape = mlx.mlx_array_shape(x);
    if (mlx.mlx_array_ndim(x) != 2 or shape[0] != want_rows or shape[1] != vocab) {
        _ = mlx.mlx_array_free(x);
        return fail(name, "logits are not [rows, vocab]", .of(@tagName(mode)));
    }
    for (try consumeF32(a, out.logits)) |l| if (!std.math.isFinite(l)) return fail(name, "non-finite logits", .of(@tagName(mode)));
}

/// An encoder's prefill: finite hidden rows `[slots, rows, hidden]` (logits, if any, are freed unread).
fn expectHidden(name: []const u8, a: Allocator, out: model_abi.ForwardOut, slots: u32, rows: u32, hidden: u32) !void {
    if (out.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
    const x: mlx.mlx_array = .{ .ctx = out.hidden.ctx };
    if (x.ctx == null) return fail(name, "an encoder forward returned no hidden rows", .{});
    const shape = mlx.mlx_array_shape(x);
    if (mlx.mlx_array_ndim(x) != 3 or shape[0] != slots or shape[1] != rows or shape[2] != hidden) {
        _ = mlx.mlx_array_free(x);
        return fail(name, "hidden rows are not [slots, rows, hidden]", .{});
    }
    for (try consumeF32(a, out.hidden)) |h| if (!std.math.isFinite(h)) return fail(name, "non-finite hidden rows", .{});
}

/// The exactness contracts a plugin declares in its caps (drafted == plain, resumed == cold, concurrent == solo), on
/// an engine serving its pack. `expectModel` calls it once an engine serves plugins (P8: the MLX table over
/// `Pack.staged.path`, and the gate model env-gated).
pub fn expectExactness(engine: *const engine_abi.Engine, fx: exactness.Fixture, caps: model_abi.Caps) !void {
    try exactness.expect(engine, fx, .declaredBy(caps));
}

// ── media ──

/// Counts the host callbacks of a job; `leave` makes the client go away at the first boundary.
const Recorder = struct {
    reports: u32 = 0,
    boundaries: u32 = 0,
    leave: bool = false,

    fn progress(self: *Recorder) media_abi.Progress {
        return .{ .ctx = self, .report = report, .boundary = boundary };
    }
    fn report(ctx: ?*anyopaque, _: Str, _: u32, _: u32) callconv(.c) void {
        const self: *Recorder = @ptrCast(@alignCast(ctx.?));
        self.reports += 1;
    }
    fn boundary(ctx: ?*anyopaque) callconv(.c) bool {
        const self: *Recorder = @ptrCast(@alignCast(ctx.?));
        self.boundaries += 1;
        return self.leave;
    }
};

fn expectMedia(io: std.Io, a: Allocator, subject: Subject, t: *const media_abi.MediaTable) !void {
    const name = subject.name;
    try expectHead(subject, media_abi.MediaTable, media_abi.version, t);
    const fx = subject.fixture;
    var staged = try Staged.init(io, a, fx);
    defer staged.deinit();
    const dir = &staged.dir;
    const none: fixture.MemDir = .{ .files = &.{} };
    const empty = none.packDir();

    const mt = if (fx.file("config.json")) |cfg| try sdk_plugin.modelTypeOf(a, cfg) else "";
    if (t.claims(.of(mt), dir) == .no) return fail(name, "does not claim its own fixture", .{});
    if (t.claims(.of("conformance_stranger"), &empty) != .no) return fail(name, "claims a stranger's pack", .{});
    if (t.required_file.len != 0 and !dir.has(t.required_file.slice())) return fail(name, "its fixture lacks the required file", t.required_file);

    var caps: media_abi.MediaCaps = .{};
    t.caps(dir, &caps);
    if (caps.size != @sizeOf(media_abi.MediaCaps) or caps.routes.size != @sizeOf(media_abi.Routes)) return fail(name, "caps have another size", .{});
    const machine: media_abi.Machine = .{};
    if (t.bill_load(dir, &machine) == 0) return fail(name, "bills a load at zero bytes", .{});
    if (t.bill_request(dir, &machine, .of(fx.body)) != t.bill_request(dir, &machine, .of(fx.body))) return fail(name, "request bill is not deterministic", .{});

    // A virtual GPU refuses the job's kernels (its threadgroup caps): the checks above are the proof there.
    if (mlx.virtualGpu()) return;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var why: Verdict = .{};
    const e = t.load(&.{ .model_dir = .of(staged.path), .dir = dir, .stream = .{ .ctx = s.ctx } }, &why) orelse return fail(name, "load of its fixture", why.why);
    defer t.unload(e);
    if (t.set_loras) |f| try expectOk(name, "clearing the LoRA stack", f(e, .{}));

    var msg_buf: [512]u8 = undefined;
    var msg_len: usize = 0;
    const inputs = try a.alloc(Str, fx.inputs.len);
    for (inputs, fx.inputs) |*i, x| i.* = .of(x);
    const req: media_abi.Request = .{ .body = .of(fx.body), .inputs = .of(inputs), .model_id = .of(name), .message = .{ .ptr = &msg_buf, .cap = msg_buf.len, .len = &msg_len }, .dir = dir };
    var rec: Recorder = .{};
    try expectOk(name, "a job on its fixture", t.step(e, &req, &rec.progress()));
    if (rec.reports == 0 or rec.boundaries == 0) return fail(name, "a job never reports progress or yields at a boundary", .{});
    var bytes: Str = .{};
    try expectOk(name, "encode", t.encode(e, &bytes));
    if (bytes.len == 0) return fail(name, "encode returned no bytes", .{});

    if (t.step_batch) |batch| {
        const reqs = [_]media_abi.Request{ req, req };
        var out: [2]Verdict = undefined;
        var results: [2]Str = undefined;
        var brec: Recorder = .{};
        batch(e, .of(&reqs), &out, &results, &brec.progress());
        for (out, results) |bv, r| {
            try expectOk(name, "a merged job on its fixture", bv);
            if (r.len == 0) return fail(name, "a merged job returned no bytes", .{});
        }
    }

    var gone: Recorder = .{ .leave = true };
    const v = t.step(e, &req, &gone.progress());
    if (v.isOk() or v.why.len == 0) return fail(name, "a job whose client left is not stopped by name", .{});
}

// ── quant ──

fn expectQuant(s: Subject, t: *const quant_abi.QuantTable) !void {
    const name = s.name;
    try expectHead(s, quant_abi.QuantTable, quant_abi.version, t);
    const fx = s.fixture;
    const mem: fixture.MemDir = .{ .files = fx.files };
    const dir = mem.packDir();
    const config = fx.file("config.json") orelse "{}";
    if (t.claims(.of(fx.block), .of(config)) == .no) return fail(name, "does not claim its own block", .of(fx.block));
    if (t.claims(.of("{}"), .of("{}")) != .no) return fail(name, "claims an empty block", .{});
    try expectOk(name, "validate of its own block", t.validate(.of(fx.block), .of(config), &dir));
}

// ── claims across plugins ──

/// Every pack has at most one claimant at the top rank (`by_marker` > `by_type`); two fail, naming them.
pub fn expectOneClaimant(list: []const sdk_plugin.Entry, packs: []const fixture.File) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ranks = try a.alloc(Claim, list.len);
    for (packs) |pack| {
        const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = pack.bytes }} };
        const dir = mem.packDir();
        const got = sdk_plugin.claimsOf(list, try sdk_plugin.modelTypeOf(a, pack.bytes), pack.bytes, &dir, ranks);
        _ = common.pick(got) catch {
            const top = std.mem.max(u8, std.mem.sliceAsBytes(got));
            std.debug.print("claims: {s} is claimed at one rank by", .{pack.name});
            for (list, got) |e, c| if (@intFromEnum(c) == top) std.debug.print(" {s}", .{e.plugin.name});
            std.debug.print("\n", .{});
            return error.TwoClaimants;
        };
    }
}

// ── tests ──

fn fakeModel(comptime name: []const u8, comptime claims: *const fn (Str, Str, *const common.PackDir) callconv(.c) Claim) model_abi.ModelTable {
    return .{ .name = .of(name), .claims = claims, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = undefined, .weights = undefined };
}

fn claimsByMarker(_: Str, config_json: Str, _: *const common.PackDir) callconv(.c) Claim {
    return if (std.mem.indexOf(u8, config_json.slice(), "\"num_attention_heads_per_layer\"") != null) .by_marker else .no;
}

const laguna_pack: fixture.File = .{ .name = "laguna.json", .bytes = @embedFile("../../fixtures/model-configs/laguna.json") };
const gemma_pack: fixture.File = .{ .name = "gemma3-text.json", .bytes = @embedFile("../../fixtures/model-configs/gemma3-text.json") };

test "conformance claims: two type claims on one fixture pack fail; a marker claim outranks a type claim" {
    const by_type = comptime fakeModel("laguna", sdk_plugin.claimsByType(&.{"laguna"}));
    const twin = comptime fakeModel("laguna_twin", sdk_plugin.claimsByType(&.{"laguna"}));
    const marker = comptime fakeModel("flint_like", claimsByMarker);
    const gemma = comptime fakeModel("gemma3", sdk_plugin.claimsByType(&.{"gemma3_text"}));
    const packs = [_]fixture.File{ laguna_pack, gemma_pack };

    try expectOneClaimant(&.{ .{ .plugin = .{ .name = "laguna", .kind = .model }, .model = &by_type }, .{ .plugin = .{ .name = "gemma3", .kind = .model }, .model = &gemma } }, &packs);
    try testing.expectError(error.TwoClaimants, expectOneClaimant(&.{
        .{ .plugin = .{ .name = "laguna", .kind = .model }, .model = &by_type },
        .{ .plugin = .{ .name = "laguna_twin", .kind = .model }, .model = &twin },
    }, &packs));

    // The marker claimant wins laguna.json over the type claimant, and the claims pick names it.
    const list = [_]sdk_plugin.Entry{
        .{ .plugin = .{ .name = "laguna", .kind = .model }, .model = &by_type },
        .{ .plugin = .{ .name = "flint_like", .kind = .model }, .model = &marker },
    };
    try expectOneClaimant(&list, &packs);
    var ranks: [2]Claim = undefined;
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = laguna_pack.bytes }} };
    const dir = mem.packDir();
    try testing.expectEqual(@as(?usize, 1), try common.pick(sdk_plugin.claimsOf(&list, "laguna", laguna_pack.bytes, &dir, &ranks)));
}
