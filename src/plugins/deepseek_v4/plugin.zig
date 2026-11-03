//! DeepSeek-V4-Flash: the module-owned native arch (`deepseek_v4.zig`) behind the model table. Serial, single-flight,
//! no prefix cache: the request's state lives on the model, and position 0 starts a new one.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const dsv4 = @import("deepseek_v4.zig");
const bill_mod = @import("bill.zig");
const fixture = @import("fixture.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const Weights = @import("../../sdk/weights/resident.zig").Weights;
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill_mod.bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout },
    .weights = &.{},
};

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// Ratio-0 layers keep a sliding window of latents; compressed layers grow their compressed history. Every layer is
/// one shared KV latent of `head_dim`, held by the module, never by the engine's cache.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], first..) |*l, li| {
        const ratio = if (li < c.dsv4_n_compress_ratios) c.dsv4_compress_ratios[li] else 0;
        l.* = .{ .kind = if (ratio == 0) .sliding else .full, .window = if (ratio == 0) c.sliding_window else 0, .kv_heads = c.num_key_value_heads, .k_dim = c.head_dim };
    }
    return n;
}

pub const Module = dsv4.Dsv4Model;

/// The module a loaded model runs on (`engines/mlx/plugin/intree.zig`).
pub fn moduleOf(mp: *anyopaque) *Module {
    const m: *Model = @ptrCast(@alignCast(mp));
    return m.mdl;
}

pub const Model = struct {
    cfg: config.Config,
    weights: Weights,
    mdl: *dsv4.Dsv4Model,
    caps: abi.Caps,
    s: mlx.mlx_stream,
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = switch (e) {
            error.OutOfMemory => Verdict.refuse(.failed, "deepseek_v4: out of memory"),
            error.MissingWeight => Verdict.refuse(.failed, "deepseek_v4: a weight is missing"),
            else => Verdict.refuse(.failed, "deepseek_v4: the pack does not load"),
        };
        return null;
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Model {
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.cfg = config.of(ctx.parsed).*;
    m.weights = (if (ctx.weights) |w| Resident.take(w) else null) orelse
        try @import("../../sdk/weights/load.zig").loadWeights(std.Io.Threaded.global_single_threaded.io(), gpa, ctx.model_dir.slice());
    errdefer m.weights.deinit();
    m.s = .{ .ctx = ctx.stream.ctx };
    m.caps = ctx.parsed.caps;
    var dw = try dsv4.loadDsv4Weights(gpa, &m.cfg, &m.weights);
    errdefer dw.deinit();
    m.mdl = try gpa.create(dsv4.Dsv4Model);
    errdefer gpa.destroy(m.mdl);
    m.mdl.* = try dsv4.initModel(gpa, &m.cfg, dw, m.s);
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    m.mdl.deinit();
    gpa.destroy(m.mdl);
    m.weights.deinit();
    gpa.destroy(m);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "deepseek_v4: forward mode not declared");
    if (f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "deepseek_v4: no layer capture");
    // Before any forward: a position-0 run would reset the live state for rows nobody reads.
    if (f.want_hidden) return Verdict.refuse(.unsupported, "deepseek_v4: no hidden-state output (embeddings)");
    if (f.slots.len > 1 or f.rows == 0) return Verdict.refuse(.unsupported, "deepseek_v4: one slot's rows at a time");
    const in = Ids.of(f) catch return Verdict.refuse(.failed, "deepseek_v4: the token ids do not read back");
    defer in.deinit();
    const pos: usize = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    if (pos != 0 and (m.mdl.dec_state == null or m.mdl.dec_state.?.n != pos)) return Verdict.refuse(.failed, "deepseek_v4: the position does not follow the model's state");
    if (f.logits == .all_rows and f.rows > dsv4.prefillSub()) return Verdict.refuse(.too_large, "deepseek_v4: every row's logits past one prefill sub-chunk");
    for (in.host) |t| if (t >= m.cfg.vocab_size) return Verdict.refuse(.failed, "deepseek_v4: token id past the vocabulary");
    const logits = run(m, in.host, pos, f.logits == .all_rows, in.dev) catch return Verdict.refuse(.failed, "deepseek_v4: the forward failed");
    if (f.logits == .none) _ = mlx.mlx_array_free(logits) else out.logits = .{ .ctx = logits.ctx };
    return .ok;
}

/// The legacy seam's arms: a new request at position 0; a decode token lazily on the GPU when the state allows it,
/// else on the host; a later chunk extends the state. `all` returns every row's logits (a verify block); `dev` is a
/// device token (one decode row) that stays on the GPU where the lazy arm takes it.
fn run(m: *Model, host_ids: []const u32, pos: usize, all: bool, dev: mlx.mlx_array) !mlx.mlx_array {
    const mdl = m.mdl;
    const vocab: c_int = @intCast(m.cfg.vocab_size);
    if (pos == 0) {
        if (mdl.dec_state) |*ds| dsv4.deinitDecodeState(ds);
        mdl.dec_state = null;
        mdl.dec_state = try dsv4.initDecodeState(mdl, gpa);
    }
    const st = &mdl.dec_state.?;
    var one: [1]u32 = undefined;
    const lazy = !all and pos != 0 and (dev.ctx != null or host_ids.len == 1) and dsv4.lazyDecodeReady(mdl, st);
    if (lazy) {
        const arr = if (dev.ctx != null) try deviceToken(dev, &.{1}, m.s) else blk: {
            const id: i32 = @intCast(host_ids[0]);
            break :blk mlx.mlx_array_new_data(&id, &[_]c_int{1}, 1, .int32);
        };
        defer _ = mlx.mlx_array_free(arr);
        const lz = try dsv4.decodeStepLazy(mdl, gpa, st, arr);
        defer _ = mlx.mlx_array_free(lz);
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_reshape(&o, lz, &[_]c_int{ 1, vocab }, 2, m.s));
        return o;
    }
    const ids = if (dev.ctx != null) blk: {
        one[0] = try hostToken(dev);
        if (one[0] >= m.cfg.vocab_size) return error.TokenPastVocabulary;
        break :blk one[0..];
    } else host_ids;
    const host = if (all)
        try dsv4.extendStateAllLogits(mdl, gpa, st, ids)
    else if (pos == 0)
        try dsv4.prefillIntoState(mdl, gpa, st, ids)
    else if (ids.len == 1)
        try dsv4.decodeStep(mdl, gpa, st, ids[0])
    else
        try dsv4.extendState(mdl, gpa, st, ids);
    defer gpa.free(host);
    const rows: c_int = @intCast(host.len / m.cfg.vocab_size);
    return mlx.mlx_array_new_data(host.ptr, &[_]c_int{ rows, vocab }, 2, .float32);
}

/// A device token id as an int32 array of `shape` (lazy: nothing is evaluated).
pub fn deviceToken(dev: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, dev, shape.ptr, shape.len, s));
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_astype(&o, r, .int32, s));
    return o;
}

/// A forward's token ids: one decode row's device token stays on the GPU (`dev`, the pipelined decode's still-lazy
/// sample); any other device ids are read back to the host, as the legacy seam reads them.
pub const Ids = struct {
    host: []const u32,
    dev: mlx.mlx_array = .{ .ctx = null },
    owned: bool = false,

    pub fn of(f: *const abi.Forward) !Ids {
        if (f.ids.ctx == null) return if (f.tokens.len == f.rows) .{ .host = f.tokens.slice() } else error.RowsMismatch;
        const dev: mlx.mlx_array = .{ .ctx = f.ids.ctx };
        if (f.rows == 1 and f.logits != .all_rows) return .{ .host = &.{}, .dev = dev };
        var i32s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(i32s);
        try mlx.check(mlx.mlx_astype(&i32s, dev, .int32, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(i32s));
        const n = mlx.mlx_array_size(i32s);
        if (n != f.rows) return error.RowsMismatch;
        const data = mlx.mlx_array_data_int32(i32s) orelse return error.NoData;
        const host = try gpa.alloc(u32, n);
        for (host, data[0..n]) |*o, id| o.* = @intCast(id);
        return .{ .host = host, .owned = true };
    }

    pub fn deinit(in: Ids) void {
        if (in.owned) gpa.free(in.host);
    }
};

/// A device token id read back to the host (an eval: the arms that need it on the host).
pub fn hostToken(dev: mlx.mlx_array) !u32 {
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_eval(dev));
    try mlx.check(mlx.mlx_array_item_int32(&v, dev));
    return @intCast(v);
}

/// The tiny pack the conformance suite runs: the oracle's four-layer miniature (every attention arm), trunk only.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.v4Files(a, 11) };
}

// ── tests ──

test {
    _ = config;
    _ = bill_mod;
    _ = fixture;
    _ = @import("contract.zig");
    _ = @import("contract_kit.zig");
}
