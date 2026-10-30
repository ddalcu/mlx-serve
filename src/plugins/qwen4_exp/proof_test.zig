//! Parity proof for the qwen4_exp plugin: on a tiny random pack, the table's forward (prefill, decode across the QSA
//! budget, verify, a later chunk, layer and stream captures, batched rows) is bit-equal to the legacy forward the
//! engine ran (`Transformer.forwardWith` / `forwardMoeBatchedDecode`, recorded on the tapes), on the engine's slot caches and on the
//! model's own.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const model = @import("../../sdk/config/model.zig");
const sdk = @import("../../sdk/root.zig");
const engine_cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const Trunk = @import("trunk.zig").Trunk;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const nextSsmMemberGen = @import("../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const q4 = @import("plugin.zig");
const fixture = @import("fixture.zig");
const Oracle = @import("oracle_test.zig").Oracle;

const testing = std.testing;
const Array = mlx.mlx_array;

/// A slot's engine caches: what `Slot.init` allocates for a GDN trunk.
const Caches = struct {
    kv: KVCache,
    ssm: []SSMCacheEntry,
    pos: u32 = 0,
    slot: engine_cache_ops.Slot = undefined,

    fn init(layers: u32) !Caches {
        const ssm = try testing.allocator.alloc(SSMCacheEntry, layers);
        const gen = nextSsmMemberGen();
        for (ssm) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false, .member_gen = gen };
        return .{ .kv = try KVCache.init(testing.allocator, layers), .ssm = ssm };
    }

    fn deinit(c: *Caches, xfm: *Trunk) void {
        xfm.ssmGroupDrop(c.ssm);
        for (c.ssm) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        testing.allocator.free(c.ssm);
        c.kv.deinit();
    }

    fn handle(c: *Caches) abi.Handle {
        c.slot = .{ .kv = &c.kv, .ssm = c.ssm };
        return c.slot.handle();
    }
};

/// A pack on disk, loaded twice on the GPU stream: a bare trunk module and the plugin's table.
const Twin = struct {
    tmp: testing.TmpDir,
    cfg: model.ModelConfig,
    weights: model.Weights,
    xfm: Trunk,
    parsed: abi.Parsed = .{},
    m: *anyopaque,

    fn open(files: []const sdk.fixture.File) !*Twin {
        const a = testing.allocator;
        const io = testing.io;
        const t = try a.create(Twin);
        errdefer a.destroy(t);
        t.tmp = testing.tmpDir(.{});
        errdefer t.tmp.cleanup();
        for (files) |f| try t.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir = buf[0..try t.tmp.dir.realPath(io, &buf)];
        t.cfg = try model.parseConfigLegacy(io, a, dir);
        errdefer t.cfg.deinit(a);
        t.weights = try model.loadModelWeights(io, a, dir, &t.cfg, false);
        errdefer t.weights.deinit();
        model.resolveWeightPrefix(&t.cfg, &t.weights);
        t.xfm = try Trunk.init(io, a, t.cfg, &t.weights);
        errdefer t.xfm.deinit();
        t.xfm.prepare();
        const mem: sdk.fixture.MemDir = .{ .files = files };
        const pd = mem.packDir();
        try testing.expect(q4.table.parse(&pd, .of(files[0].bytes), .{}, null, false, &t.parsed).isOk());
        var why: abi.Verdict = .{};
        t.m = q4.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .stream = .{ .ctx = mlx.gpuStream().ctx } }, &why) orelse {
            std.debug.print("load: {s}\n", .{why.why.slice()});
            return error.PluginLoad;
        };
        return t;
    }

    fn close(t: *Twin) void {
        q4.table.unload(t.m);
        q4.table.free_parsed(&t.parsed);
        t.xfm.deinit();
        t.weights.deinit();
        t.cfg.deinit(testing.allocator);
        t.tmp.cleanup();
        testing.allocator.destroy(t);
    }

    /// The plugin's trunk module: the engine slots it batched are dropped from its group.
    fn inner(t: *Twin) *Trunk {
        return &@as(*q4.Model, @ptrCast(@alignCast(t.m))).trunk;
    }

    fn ids(toks: []const u32) Array {
        var buf: [32]i32 = undefined;
        for (toks, 0..) |id, k| buf[k] = @intCast(id);
        return mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, @intCast(toks.len) }, 2, .int32);
    }

    /// The legacy forward's logits `[S, V]` on `c`, as the engine calls it; the stream and layer 2 when asked.
    fn legacy(t: *Twin, c: *Caches, mode: abi.Mode, toks: []const u32, stream: ?*Array, layer2: ?*Array) !Array {
        const x = ids(toks);
        defer _ = mlx.mlx_array_free(x);
        var off: usize = c.pos;
        var cap_out = [_]Array{mlx.mlx_array_new()};
        defer _ = mlx.mlx_array_free(cap_out[0]);
        var cl: CaptureLayers = .{ .ids = &.{2}, .out = &cap_out };
        var ctx: ForwardCtx = .{ .cache = &c.kv, .moe_seq_offset = &off, .ssm_entries = c.ssm, .capture_hidden = null, .capture_hidden_all = stream, .capture_layers = if (layer2 != null) &cl else null, .vision_embeddings = null, .spec_verify = mode == .verify, .capture_ssm_seq = mode == .verify };
        const l = try t.xfm.forwardWith(&ctx, x);
        defer _ = mlx.mlx_array_free(l);
        if (layer2) |dst| _ = mlx.mlx_array_set(dst, cap_out[0]);
        c.pos += @intCast(toks.len);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&o, l, &[_]c_int{ @intCast(toks.len), -1 }, 2, mlx.gpuStream()));
        return o;
    }

    /// The table's logits `[S, V]`: on the engine's caches `c`, or (null) the model's own slot 0.
    fn plugin(t: *Twin, c: ?*Caches, pos: u32, mode: abi.Mode, toks: []const u32, out: *abi.ForwardOut, capture: bool) !Array {
        const h = [_]abi.Handle{if (c) |cc| cc.handle() else null};
        const v = q4.table.forward(t.m, &.{
            .mode = mode,
            .logits = .all_rows,
            .want_hidden = capture,
            .rows = @intCast(toks.len),
            .tokens = .of(toks),
            .positions = .of(&.{pos}),
            .caches = if (c != null) .of(&h) else .{},
            .cache_ops = if (c != null) &engine_cache_ops.ops else null,
            .capture_layers = if (capture) .of(&.{2}) else .{},
        }, out);
        if (!v.isOk()) {
            std.debug.print("forward: {s}\n", .{v.why.slice()});
            return error.PluginForward;
        }
        if (c) |cc| cc.pos += @intCast(toks.len);
        return .{ .ctx = out.logits.ctx };
    }
};

const Step = struct { mode: abi.Mode, ids: []const u32, capture: bool = false };

/// Prefill below the QSA budget, decodes across it, a verify block, a later chunk past it.
const steps = [_]Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 9, 7, 25, 1 }, .capture = true },
    .{ .mode = .decode, .ids = &.{8} },
    .{ .mode = .decode, .ids = &.{55} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 20 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .prefill, .ids = &.{ 4, 33, 7, 40, 90, 101, 2, 64, 77 }, .capture = true },
    .{ .mode = .decode, .ids = &.{2} },
    .{ .mode = .verify, .ids = &.{ 6, 70 } },
};

fn expectSteps(precision: fixture.Precision, own: bool, oracle: *Oracle) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.files(arena.allocator(), 5, precision));
    defer t.close();
    const layers = t.cfg.num_hidden_layers;
    var lc = try Caches.init(layers);
    defer lc.deinit(&t.xfm);
    var pc = try Caches.init(layers);
    defer pc.deinit(t.inner());
    for (steps, 0..) |st, k| {
        errdefer std.debug.print("qwen4_exp proof: step {d} ({s})\n", .{ k, @tagName(st.mode) });
        var stream = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stream);
        var layer2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(layer2);
        const pos = lc.pos;
        const want = try t.legacy(&lc, st.mode, st.ids, if (st.capture) &stream else null, if (st.capture) &layer2 else null);
        defer _ = mlx.mlx_array_free(want);
        var captured = [_]abi.Array{.{}};
        var out: abi.ForwardOut = .{ .captured = &captured };
        const got = try t.plugin(if (own) null else &pc, pos, st.mode, st.ids, &out, st.capture);
        defer _ = mlx.mlx_array_free(got);
        try oracle.check(got, want, false);
        if (st.capture) {
            const hs: Array = .{ .ctx = out.hidden.ctx };
            defer _ = mlx.mlx_array_free(hs);
            const l2: Array = .{ .ctx = captured[0].ctx };
            defer _ = mlx.mlx_array_free(l2);
            try oracle.check(hs, stream, false);
            try oracle.check(l2, layer2, false);
        }
    }
    try oracle.close();
}

test "qwen4_exp plugin: bf16 pack, every step on the engine's slot caches equals the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var oracle = try Oracle.open("proof-bf16");
    try expectSteps(.bf16, false, &oracle);
}

test "qwen4_exp plugin: 4-bit pack, every step on the engine's slot caches equals the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var oracle = try Oracle.open("proof-q4");
    try expectSteps(.q4, false, &oracle);
}

test "qwen4_exp plugin: on the model's own slot state every step equals the legacy forward bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var oracle = try Oracle.open("proof-q4-own");
    try expectSteps(.q4, true, &oracle);
}

test "qwen4_exp plugin: batched rows equal the legacy batched GDN tick per slot, and a cold group steps serially" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.files(arena.allocator(), 9, .q4));
    defer t.close();
    var oracle = try Oracle.open("proof-batched");
    const layers = t.cfg.num_hidden_layers;
    var lc: [2]Caches = .{ try Caches.init(layers), try Caches.init(layers) };
    defer for (&lc) |*c| c.deinit(&t.xfm);
    var pc: [2]Caches = .{ try Caches.init(layers), try Caches.init(layers) };
    defer for (&pc) |*c| c.deinit(t.inner());
    const prompts = [_][]const u32{ &.{ 3, 17, 42, 9, 7, 25, 1, 50, 60, 70 }, &.{ 11, 4, 99 } };
    for (prompts, 0..) |p, i| {
        const want = try t.legacy(&lc[i], .prefill, p, null, null);
        _ = mlx.mlx_array_free(want);
        var out: abi.ForwardOut = .{};
        const got = try t.plugin(&pc[i], 0, .prefill, p, &out, false);
        _ = mlx.mlx_array_free(got);
    }
    const ticks = [_][2]u32{ .{ 8, 21 }, .{ 55, 3 }, .{ 13, 88 } };
    for (ticks) |toks| {
        var lctx: [2]ForwardCtx = undefined;
        var lptr: [2]*ForwardCtx = undefined;
        var offs: [2]usize = undefined;
        const pos = [_]u32{ lc[0].pos, lc[1].pos };
        for (0..2) |i| {
            offs[i] = pos[i];
            lctx[i] = .{ .cache = &lc[i].kv, .moe_seq_offset = &offs[i], .ssm_entries = lc[i].ssm, .capture_hidden = null, .vision_embeddings = null };
            lptr[i] = &lctx[i];
        }
        // Both slots' recurrent state is live: the engine's gate takes the batched tick, not the serial fallback.
        try testing.expect(t.xfm.batchedGdnReady(&lptr));
        const want = try t.xfm.forwardMoeBatchedDecode(&toks, &lptr, &pos, null);
        defer {
            for (want) |w| _ = mlx.mlx_array_free(w);
            testing.allocator.free(want);
        }
        const h = [_]abi.Handle{ pc[0].handle(), pc[1].handle() };
        var out: abi.ForwardOut = .{};
        try testing.expect(q4.table.forward(t.m, &.{ .mode = .batched_rows, .rows = 1, .tokens = .of(&toks), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &engine_cache_ops.ops }, &out).isOk());
        const got: Array = .{ .ctx = out.logits.ctx };
        defer _ = mlx.mlx_array_free(got);
        for (want, 0..) |w, i| {
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, got, &[_]c_int{ @intCast(i), 0 }, 2, &[_]c_int{ @intCast(i + 1), @intCast(t.cfg.vocab_size) }, 2, &[_]c_int{ 1, 1 }, 2, mlx.gpuStream()));
            var w2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w2);
            try mlx.check(mlx.mlx_reshape(&w2, w, &[_]c_int{ 1, -1 }, 2, mlx.gpuStream()));
            try oracle.check(row, w2, false);
        }
        for (0..2) |i| {
            lc[i].pos += 1;
            pc[i].pos += 1;
        }
    }
    try oracle.close();
}

test "qwen4_exp plugin: a position that does not follow the slot's state is refused by name" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.files(arena.allocator(), 3, .bf16));
    defer t.close();
    var out: abi.ForwardOut = .{};
    const l = try t.plugin(null, 0, .prefill, &.{ 1, 2, 3 }, &out, false);
    _ = mlx.mlx_array_free(l);
    var bad: abi.ForwardOut = .{};
    const v = q4.table.forward(t.m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&.{4}), .positions = .of(&.{5}) }, &bad);
    try testing.expect(!v.isOk() and bad.logits.ctx == null);
}

/// The routed experts' output of layer 0 on `x` (shared expert skipped), evaluated.
fn routedOut(t: *Trunk, x: Array) !Array {
    const y = try t.moeMLP2WithRouter(x, x, &t.moe_layers[0].mlp.moe, null, true);
    try mlx.check(mlx.mlx_array_eval(y));
    return y;
}

/// `x * k` in `x`'s dtype, evaluated.
fn scaled(x: Array, k: f32) !Array {
    const s = mlx.gpuStream();
    const kk = mlx.mlx_array_new_float(k);
    defer _ = mlx.mlx_array_free(kk);
    var kd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kd);
    try mlx.check(mlx.mlx_astype(&kd, kk, mlx.mlx_array_dtype(x), s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&y, x, kd, s));
    try mlx.check(mlx.mlx_array_eval(y));
    return y;
}

test "qwen4_exp plugin: a per-expert scale of 2 doubles the routed experts exactly, at decode, verify and prefill widths" {
    // ModelOpt packs carry each expert's up x down weight_scale_2 as `mlp.switch_mlp.expert_scale`, on the routing
    // weight; dropping it decodes garbage. Doubling every routing weight doubles the routed sum bit for bit.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    for ([_]fixture.Precision{ .bf16, .q4 }) |prec| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const t = try Twin.open(try fixture.filesWith(arena.allocator(), 61, prec, .{ .expert_scale = 2.0 }));
        defer t.close();
        // The plugin's own load binds the scale on every layer.
        for (t.inner().moe_layers) |*lw| try testing.expect(lw.mlp.moe.per_expert_scale != null);
        const mw = &t.xfm.moe_layers[0].mlp.moe;
        const bound = mw.*;
        var prng = std.Random.DefaultPrng.init(0x5CA1E);
        for ([_]c_int{ 1, 3, 9 }) |rows| {
            const x = try randRows(prng.random(), rows);
            defer _ = mlx.mlx_array_free(x);
            const with = try routedOut(&t.xfm, x);
            defer _ = mlx.mlx_array_free(with);
            mw.per_expert_scale = null;
            const without = routedOut(&t.xfm, x);
            mw.* = bound;
            const plain = try without;
            defer _ = mlx.mlx_array_free(plain);
            const want = try scaled(plain, 2.0);
            defer _ = mlx.mlx_array_free(want);
            try ops.expectBitEqual(with, want);
        }
        // A grouped verify's joined rows equal each row's own MoE with the scale bound.
        var inputs: [2]Array = undefined;
        for (&inputs) |*in| in.* = try randRows(prng.random(), 3);
        defer for (inputs) |in| {
            _ = mlx.mlx_array_free(in);
        };
        const rows = try t.xfm.moeVerifyRows(&inputs, mw);
        defer {
            for (rows) |r| _ = mlx.mlx_array_free(r);
            testing.allocator.free(rows);
        }
        for (inputs, rows) |in, got| {
            const solo = try t.xfm.moeMLP(in, mw);
            defer _ = mlx.mlx_array_free(solo);
            try mlx.check(mlx.mlx_array_eval(got));
            try mlx.check(mlx.mlx_array_eval(solo));
            try ops.expectBitEqual(got, solo);
        }
    }
}

test "qwen4_exp plugin: a per-expert gate scale equals the gate projection scaled by it" {
    // ModelOpt's gate weight_scale_2 (`gate_proj.expert_scale`) multiplies the gate before the SiLU; on a bf16 bank a
    // scale of 2 is the bank times 2, exactly. Prompt widths: both sides take the sorted path.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.filesWith(arena.allocator(), 63, .bf16, .{ .gate_expert_scale = 2.0 }));
    defer t.close();
    for (t.inner().moe_layers) |*lw| try testing.expect(lw.mlp.moe.gate_expert_scale.ctx != null);
    const mw = &t.xfm.moe_layers[0].mlp.moe;
    const bound = mw.*;
    const doubled = try scaled(mw.switch_gate_w, 2.0);
    defer _ = mlx.mlx_array_free(doubled);
    var prng = std.Random.DefaultPrng.init(0x6A7E);
    for ([_]c_int{ 1, 3, 9 }) |rows| {
        const x = try randRows(prng.random(), rows);
        defer _ = mlx.mlx_array_free(x);
        const with = try routedOut(&t.xfm, x);
        defer _ = mlx.mlx_array_free(with);
        mw.gate_expert_scale = .{ .ctx = null };
        mw.switch_gate_w = doubled;
        const folded = routedOut(&t.xfm, x);
        mw.* = bound;
        const want = try folded;
        defer _ = mlx.mlx_array_free(want);
        if (rows == 1) try testing.expect(try relErr(with, want) < 1e-2) else try ops.expectBitEqual(with, want);
    }
}

/// `rows` random bf16 hidden rows `[1, rows, 64]`.
fn randRows(r: std.Random, rows: c_int) !Array {
    var buf: [9 * 64]f32 = undefined;
    for (buf[0..@intCast(rows * 64)]) |*v| v.* = r.floatNorm(f32) * 0.5;
    const xf = mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, rows, 64 }, 3, .float32);
    defer _ = mlx.mlx_array_free(xf);
    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, mlx.gpuStream()));
    return x;
}

/// Columns `[lo, lo + n)` of a `[in, out]` weight.
fn cols(w: Array, lo: c_int, n: c_int) !Array {
    const sh = mlx.getShape(w);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, w, &[_]c_int{ 0, lo }, 2, &[_]c_int{ sh[0], lo + n }, 2, &[_]c_int{ 1, 1 }, 2, mlx.gpuStream()));
    return out;
}

/// A fresh recurrent entry.
fn freshEntry() SSMCacheEntry {
    return .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
}

fn freeEntry(e: *SSMCacheEntry) void {
    _ = mlx.mlx_array_free(e.conv_state);
    _ = mlx.mlx_array_free(e.ssm_state);
    ssmFreeQsaState(e);
}

/// `|got - want|` over `|want|`, in f32.
fn relErr(got: Array, want: Array) !f32 {
    const read = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
    const a = try read(testing.allocator, got, mlx.gpuStream());
    defer testing.allocator.free(a);
    const b = try read(testing.allocator, want, mlx.gpuStream());
    defer testing.allocator.free(b);
    var err: f64 = 0;
    var ref: f64 = 0;
    for (a, b) |x, y| {
        try testing.expect(std.math.isFinite(x));
        err += (x - y) * (x - y);
        ref += y * y;
    }
    return @floatCast(@sqrt(err / @max(ref, 1e-30)));
}

test "qwen4_exp plugin: a GDN layer with the combined in_proj_qkvz/in_proj_ba projections equals the separate ones" {
    // The binder takes `linear_attn.in_proj_qkvz` (qwen3_next's interleaved layout) whenever a pack ships it.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.files(arena.allocator(), 71, .bf16));
    defer t.close();
    const la = &t.xfm.moe_layers[0].attn.linear;
    try testing.expect(la.qkv_s.ctx == null and la.in.w.ctx == null); // bf16: [in, out], no joined rows
    const cfg = &t.xfm.config;
    const hk: c_int = @intCast(cfg.linear_num_key_heads);
    const dk: c_int = @intCast(cfg.linear_key_head_dim);
    const dv: c_int = @intCast(cfg.linear_value_head_dim);
    const vph: c_int = @divExact(@as(c_int, @intCast(cfg.linear_num_value_heads)), hk);
    const key_dim = hk * dk;
    // Per key head: [q, k, v, z] and [b, a].
    var qkvz_parts: std.ArrayList(Array) = .empty;
    defer {
        for (qkvz_parts.items) |p| _ = mlx.mlx_array_free(p);
        qkvz_parts.deinit(testing.allocator);
    }
    var ba_parts: std.ArrayList(Array) = .empty;
    defer {
        for (ba_parts.items) |p| _ = mlx.mlx_array_free(p);
        ba_parts.deinit(testing.allocator);
    }
    var g: c_int = 0;
    while (g < hk) : (g += 1) {
        try qkvz_parts.append(testing.allocator, try cols(la.qkv_w, g * dk, dk));
        try qkvz_parts.append(testing.allocator, try cols(la.qkv_w, key_dim + g * dk, dk));
        try qkvz_parts.append(testing.allocator, try cols(la.qkv_w, 2 * key_dim + g * vph * dv, vph * dv));
        try qkvz_parts.append(testing.allocator, try cols(la.z_w, g * vph * dv, vph * dv));
        try ba_parts.append(testing.allocator, try cols(la.b_w, g * vph, vph));
        try ba_parts.append(testing.allocator, try cols(la.a_w, g * vph, vph));
    }
    const qkvz = try @import("../../mlx/array.zig").concatAxis1(mlx.gpuStream(), qkvz_parts.items);
    defer _ = mlx.mlx_array_free(qkvz);
    const ba = try @import("../../mlx/array.zig").concatAxis1(mlx.gpuStream(), ba_parts.items);
    defer _ = mlx.mlx_array_free(ba);
    var comb = la.*;
    comb.combined_proj = true;
    comb.qkv_w = qkvz;
    comb.b_w = ba;
    comb.z_w = .{ .ctx = null };
    comb.a_w = .{ .ctx = null };

    var e_sep = freshEntry();
    defer freeEntry(&e_sep);
    var e_comb = freshEntry();
    defer freeEntry(&e_comb);
    var prng = std.Random.DefaultPrng.init(0xC0B);
    for ([_]c_int{ 5, 1, 3 }, [_]bool{ true, false, false }) |rows, prefill| {
        var buf: [5 * 64]f32 = undefined;
        for (buf[0..@intCast(rows * 64)]) |*v| v.* = prng.random().floatNorm(f32) * 0.5;
        const xf = mlx.mlx_array_new_data(&buf, &[_]c_int{ 1, rows, 64 }, 3, .float32);
        defer _ = mlx.mlx_array_free(xf);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, mlx.gpuStream()));
        const want = try t.xfm.gatedDeltaNet(x, la, &e_sep, 0, 1, rows, prefill);
        defer _ = mlx.mlx_array_free(want);
        const got = try t.xfm.gatedDeltaNet(x, &comb, &e_comb, 0, 1, rows, prefill);
        defer _ = mlx.mlx_array_free(got);
        try testing.expect(try relErr(got, want) < 1e-2);
        try testing.expect(try relErr(e_comb.ssm_state, e_sep.ssm_state) < 1e-2);
    }
}
