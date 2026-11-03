//! The MLX-native formats (`affine`, `fp8`, `mxfp8`, `nvfp4`): one load and one dispatch for all four, delegating to
//! the engine's per-weight resolution (`computeQuantParams`), `qmatmulBits` and `gatherExpertMm`. The formats differ
//! in what they claim, validate and bill (`block.zig`), and in the defaults their block declares.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const block = @import("block.zig");
const params = @import("params.zig");
const sushi_pack = @import("exl3/sushi_pack.zig");
const modelopt_pack = @import("nvfp4/modelopt_pack.zig");
const fp8_block128 = @import("fp8_block128.zig");
const qmatmulBits = @import("../blocks/linear/qmatmul.zig").qmatmulBits;
const gatherExpertMm = @import("../blocks/moe/experts.zig").gatherExpertMm;
const transposeBf16Weight = @import("../../mlx/array.zig").transposeBf16Weight;

const Str = abi.Str;
const Verdict = abi.Verdict;
const QuantMode = @import("params.zig").QuantMode;

/// `Weight.layout`: dense (pre-transposed `[in, out]`), an MLX mode, row-scaled e4m3 (`sushi_pack.fp8Linear`), or
/// e4m3 on a 64 x 128 scale grid (`fp8_block128.linear`).
pub const Layout = enum(u8) { dense, affine, nvfp4, mxfp4, mxfp8, fp8_rows, fp8_block128 };

/// The per-load state: the block's declaration as the engine's resolver reads it.
pub const State = struct {
    model: params.QuantParams = .{ .bits = 0, .group_size = 0 },
    /// Hadamard: the rotation block.
    block: u32 = 0,
};

pub fn arr(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}
pub fn wrap(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}
fn stream(s: abi.Stream) mlx.mlx_stream {
    return .{ .ctx = s.ctx };
}

/// A Zig error as a refusal (an error's name is static text).
pub fn failed(e: anyerror) Verdict {
    return .{ .code = .failed, .why = Str.of(@errorName(e)) };
}

pub fn modeOf(w: *const abi.Weight) QuantMode {
    return switch (@as(Layout, @enumFromInt(w.layout))) {
        .nvfp4 => .nvfp4,
        .mxfp4 => .mxfp4,
        .mxfp8 => .mxfp8,
        else => .affine,
    };
}

pub fn state(ctx: *const abi.LoadCtx) ?*const State {
    return @ptrCast(@alignCast(ctx.state orelse return null));
}

pub fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const d = block.declared(arena.allocator(), ctx.block_json.slice());
    const mode = QuantMode.fromString(d.mode) orelse return Verdict.refuse(.unsupported, "quantization mode other than affine, nvfp4, mxfp4, mxfp8");
    const st = std.heap.page_allocator.create(State) catch return failed(error.OutOfMemory);
    st.* = .{};
    st.model = .{ .bits = d.bits, .group_size = d.group_size, .mode = mode };
    if (block.hadamardBlock(arena.allocator(), ctx.config_json.slice())) |b| st.block = b;
    ctx.state = st;
    return .ok;
}

pub fn close(ctx: *abi.LoadCtx) callconv(.c) void {
    if (ctx.state) |p| std.heap.page_allocator.destroy(@as(*State, @ptrCast(@alignCast(p))));
    ctx.state = null;
}

/// A new handle on a source array (the source keeps its own).
fn share(a: mlx.mlx_array) !mlx.mlx_array {
    if (a.ctx == null) return a;
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    return h;
}

/// `<base>.<leaf>` from the source, empty when absent.
pub fn tensor(ctx: *const abi.LoadCtx, base: []const u8, leaf: []const u8) mlx.mlx_array {
    var nb: [320]u8 = undefined;
    const name = std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, leaf }) catch return .{};
    return arr(ctx.source.tensor(name));
}

/// The triple at `base`, its quant params resolved per weight as the engine's `quantParamsHinted` does.
pub fn loadInto(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, out: *abi.Weight) !void {
    const st = state(ctx) orelse return error.QuantFormatNotOpened;
    const w = tensor(ctx, base, "weight");
    if (w.ctx == null) return error.MissingWeight;
    if (mlx.getShape(w).len < 2) return error.NotAProjection;
    const sc = tensor(ctx, base, "scales");
    const bi = tensor(ctx, base, "biases");
    out.* = .{};
    errdefer release(out);
    const grid = tensor(ctx, base, "weight_scale_inv");
    if (sc.ctx == null and fp8_block128.isBlockScaled(w, grid)) {
        out.w = wrap(try share(w));
        out.scales = wrap(try share(grid));
        out.layout = @intFromEnum(Layout.fp8_block128);
        out.bits = 8;
        out.group_size = fp8_block128.block_cols;
        out.block = fp8_block128.block_rows;
        return;
    }
    if (sc.ctx == null) {
        out.w = wrap(try transposeBf16Weight(w, stream(ctx.stream)));
        out.layout = @intFromEnum(Layout.dense);
        return;
    }
    out.w = wrap(try share(w));
    out.scales = wrap(try share(sc));
    out.biases = wrap(try share(bi));
    if (sushi_pack.isRowScaledFp8(w, sc)) {
        out.layout = @intFromEnum(Layout.fp8_rows);
        out.bits = 8;
        return;
    }
    const qp = params.computeQuantParams(st.model, w, sc, if (in_dim > 0) in_dim else null);
    out.bits = @intCast(qp.bits);
    out.group_size = qp.group_size;
    out.layout = @intFromEnum(switch (qp.mode) {
        .affine => Layout.affine,
        .nvfp4 => Layout.nvfp4,
        .mxfp4 => Layout.mxfp4,
        .mxfp8 => Layout.mxfp8,
        .gguf => return error.GgufIsNotAnMlxMode,
    });
}

pub fn loadTensor(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    loadInto(ctx, name.slice(), in_dim, out) catch |e| return failed(e);
    return .ok;
}

pub fn release(w: *abi.Weight) callconv(.c) void {
    for ([_]abi.Array{ w.w, w.scales, w.biases, w.side }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(arr(a));
    };
    w.* = .{};
}

pub fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const y = qmatmulBits(arr(x), arr(w.w), arr(w.scales), arr(w.biases), w.bits, w.group_size, modeOf(w), stream(s)) catch |e| return failed(e);
    out.* = wrap(y);
    return .ok;
}

pub fn moe(bank: *const abi.Weight, x: abi.Array, lhs: abi.Array, rhs: abi.Array, sorted: bool, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (bank.layout == @intFromEnum(Layout.fp8_rows) or bank.layout == @intFromEnum(Layout.fp8_block128)) return Verdict.refuse(.unsupported, "fp8: e4m3 expert banks");
    var res = mlx.mlx_array_new();
    gatherExpertMm(&res, arr(x), arr(bank.w), arr(bank.scales), arr(bank.biases), arr(lhs), arr(rhs), bank.bits, bank.group_size, modeOf(bank), sorted, stream(s)) catch |e| {
        _ = mlx.mlx_array_free(res);
        return failed(e);
    };
    out.* = wrap(res);
    return .ok;
}

/// Stored as the pack holds it.
pub fn billStored(_: Str, _: Str, stored: u64) callconv(.c) u64 {
    return stored;
}

/// A ModelOpt pack decodes its csf-compressed expert scales at load.
fn billNvfp4(block_json: Str, model_dir: Str, stored: u64) callconv(.c) u64 {
    if (!block.isModelOpt(block_json.slice())) return stored;
    return stored +| modelopt_pack.bankExpansionBytes(model_dir.slice());
}

fn builtin(comptime name: []const u8) block.Builtin {
    for (block.builtins) |b| if (std.mem.eql(u8, b.name, name)) return b;
    unreachable;
}

/// The table of the built-in `name` over the shared load and dispatch.
pub fn table(comptime name: []const u8, comptime bill: anytype) abi.QuantTable {
    const b = comptime builtin(name);
    return .{
        .name = Str.of(name),
        .claims = b.claims,
        .validate = b.validate,
        .bill = bill,
        .open = &open,
        .close = &close,
        .load_tensor = &loadTensor,
        .load_bank = &loadTensor,
        .release = &release,
        .matmul = &matmul,
        .moe = &moe,
    };
}

pub const affine = table("affine", &billStored);
pub const fp8 = table("fp8", &billStored);
pub const mxfp8 = table("mxfp8", &billStored);
pub const nvfp4 = table("nvfp4", &billNvfp4);

// ── tests ──

const testing = std.testing;
const ops = @import("../../mlx/ops.zig");
const parity = @import("../testing/parity.zig");
const resident = @import("../weights/resident.zig");

pub const Triple = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array = .{},
    bi: mlx.mlx_array = .{},
};

/// `w` quantized by MLX in `mode` (affine carries biases; the fp modes do not).
pub fn quantized(w: mlx.mlx_array, mode: [*:0]const u8, bits: u32, gs: u32, s: mlx.mlx_stream) !Triple {
    var v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_quantize(&v, w, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), mode, .{}, s));
    var t: Triple = .{ .w = mlx.mlx_array_new(), .sc = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&t.w, v, 0));
    try mlx.check(mlx.mlx_vector_array_get(&t.sc, v, 1));
    if (mlx.mlx_vector_array_size(v) > 2) {
        t.bi = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&t.bi, v, 2));
    }
    return t;
}

/// Hands `t` to `weights` under `base` (the map owns it from here).
pub fn putTriple(weights: *resident.Weights, base: []const u8, t: Triple) !void {
    var nb: [256]u8 = undefined;
    try weights.put(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}), t.w);
    if (t.sc.ctx != null) try weights.put(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}), t.sc);
    if (t.bi.ctx != null) try weights.put(try std.fmt.bufPrint(&nb, "{s}.biases", .{base}), t.bi);
}

pub const Case = struct { base: []const u8, layout: Layout, bits: u8 = 0, group_size: u32 = 0 };

/// Loads every case of `r` through `t` and checks the resolved params, then that `t.matmul` at 1, 3 and 9 rows is
/// bit-equal to the direct path (`quantParamsOrDense` from the activation, then `qmatmulBits`).
fn expectParity(t: *const abi.QuantTable, r: *const resident.Resident, block_json: []const u8, cases: []const Case, k: c_int, s: mlx.mlx_stream) !void {
    const src = r.source();
    var ctx: abi.LoadCtx = .{ .block_json = .of(block_json), .source = &src, .stream = .{ .ctx = s.ctx } };
    try testing.expect(t.open(&ctx).isOk());
    defer t.close(&ctx);
    var prng = std.Random.DefaultPrng.init(0x9a41);
    for (cases) |c| {
        var w: abi.Weight = .{};
        const v = t.load_tensor(&ctx, .of(c.base), @intCast(k), &w);
        if (!v.isOk()) std.debug.print("{s}: {s}\n", .{ c.base, v.why.slice() });
        try testing.expect(v.isOk());
        defer t.release(&w);
        try testing.expectEqual(@intFromEnum(c.layout), w.layout);
        if (c.bits != 0) try testing.expectEqual(c.bits, w.bits);
        if (c.group_size != 0) try testing.expectEqual(c.group_size, w.group_size);
        const direct = tensor(&ctx, c.base, "weight");
        const dense_t = if (c.layout == .dense) try transposeBf16Weight(direct, s) else direct;
        defer if (c.layout == .dense) {
            _ = mlx.mlx_array_free(dense_t);
        };
        for ([_]c_int{ 1, 3, 9 }) |rows| {
            const x = try parity.testRandWeightBf16(prng.random(), &.{ 1, rows, k }, s);
            defer _ = mlx.mlx_array_free(x);
            var got: abi.Array = .{};
            try testing.expect(t.matmul(&w, wrap(x), .{ .ctx = s.ctx }, &got).isOk());
            defer _ = mlx.mlx_array_free(arr(got));
            const sc = tensor(&ctx, c.base, "scales");
            const qp = params.quantParamsOrDense(state(&ctx).?.model, dense_t, sc, @intCast(k));
            const want = try qmatmulBits(x, dense_t, sc, tensor(&ctx, c.base, "biases"), qp.bits, qp.group_size, qp.mode, s);
            defer _ = mlx.mlx_array_free(want);
            try ops.expectBitEqual(arr(got), want);
        }
    }
}

const N: c_int = 128;
const K: c_int = 256;

fn randW(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return parity.testRandWeightBf16(rnd, shape, s);
}

/// One weight quantized in `mode` under `base` in `weights`.
fn addQuantized(weights: *resident.Weights, rnd: std.Random, base: []const u8, mode: [*:0]const u8, bits: u32, gs: u32, s: mlx.mlx_stream) !void {
    const w = try randW(rnd, &.{ N, K }, s);
    defer _ = mlx.mlx_array_free(w);
    try putTriple(weights, base, try quantized(w, mode, bits, gs, s));
}

// Bar: loading through the format resolves each weight as the engine does and its matmul is bit-equal to the
// engine's direct path, at every affine width MLX ships and every group size, beside a dense tensor and a legacy
// MXFP8 tensor in an affine pack.
test "quant affine: every shipped width and group size loads and multiplies bit-equal to the direct path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(1);
    var weights = resident.Weights.init(testing.allocator);
    var cases: std.ArrayList(Case) = .empty;
    defer cases.deinit(testing.allocator);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| testing.allocator.free(n);
        names.deinit(testing.allocator);
    }
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]u32{ 32, 64, 128 }) |gs| {
        const base = try std.fmt.allocPrint(testing.allocator, "l.b{d}g{d}", .{ bits, gs });
        try names.append(testing.allocator, base);
        try addQuantized(&weights, prng.random(), base, "affine", bits, gs, s);
        try cases.append(testing.allocator, .{ .base = base, .layout = .affine, .bits = bits, .group_size = gs });
    };
    try weights.put("l.dense.weight", try randW(prng.random(), &.{ N, K }, s));
    try cases.append(testing.allocator, .{ .base = "l.dense", .layout = .dense });
    try addQuantized(&weights, prng.random(), "l.mx", "mxfp8", 8, 32, s);
    try cases.append(testing.allocator, .{ .base = "l.mx", .layout = .mxfp8, .bits = 8, .group_size = 32 });
    var r = resident.Resident.adopt(weights);
    defer r.deinit();
    try expectParity(&affine, &r, "{\"bits\":4,\"group_size\":64}", cases.items, K, s);
}

// Bar: MXFP8 g32 packs, bit-equal to the direct path.
test "quant mxfp8: an MXFP8 pack loads and multiplies bit-equal to the direct path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(2);
    var weights = resident.Weights.init(testing.allocator);
    try addQuantized(&weights, prng.random(), "l.q", "mxfp8", 8, 32, s);
    var r = resident.Resident.adopt(weights);
    defer r.deinit();
    try expectParity(&mxfp8, &r, "{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\"}", &.{.{ .base = "l.q", .layout = .mxfp8, .bits = 8, .group_size = 32 }}, K, s);
}

// Bar: NVFP4 g16 and MXFP4 g32 packs, with the overrides shipped packs carry (an affine 8-bit shared MLP inside an
// NVFP4 QAT pack, an MXFP8 embedding inside an MXFP4 pack), and a ModelOpt pack's MXFP8 linear under its NVFP4
// declaration; each resolves per weight and multiplies bit-equal to the direct path.
test "quant nvfp4: NVFP4, MXFP4, their overrides and ModelOpt's MXFP8 linears match the direct path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(3);
    {
        var weights = resident.Weights.init(testing.allocator);
        try addQuantized(&weights, prng.random(), "l.nv", "nvfp4", 4, 16, s);
        try addQuantized(&weights, prng.random(), "l.shared", "affine", 8, 64, s);
        var r = resident.Resident.adopt(weights);
        defer r.deinit();
        try expectParity(&nvfp4, &r, "{\"bits\":4,\"group_size\":16,\"mode\":\"nvfp4\"}", &.{
            .{ .base = "l.nv", .layout = .nvfp4, .bits = 4, .group_size = 16 },
            .{ .base = "l.shared", .layout = .affine, .bits = 8, .group_size = 64 },
        }, K, s);
    }
    {
        var weights = resident.Weights.init(testing.allocator);
        try addQuantized(&weights, prng.random(), "l.mx4", "mxfp4", 4, 32, s);
        try addQuantized(&weights, prng.random(), "model.embed_tokens", "mxfp8", 8, 32, s);
        var r = resident.Resident.adopt(weights);
        defer r.deinit();
        try expectParity(&nvfp4, &r, "{\"bits\":4,\"group_size\":32,\"mode\":\"mxfp4\",\"model.embed_tokens\":{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\"}}", &.{
            .{ .base = "l.mx4", .layout = .mxfp4, .bits = 4, .group_size = 32 },
            .{ .base = "model.embed_tokens", .layout = .mxfp8, .bits = 8, .group_size = 32 },
        }, K, s);
    }
    {
        var weights = resident.Weights.init(testing.allocator);
        try addQuantized(&weights, prng.random(), "l.o_proj", "mxfp8", 8, 32, s);
        var r = resident.Resident.adopt(weights);
        defer r.deinit();
        try expectParity(&nvfp4, &r, "{\"quant_method\":\"modelopt\"}", &.{.{ .base = "l.o_proj", .layout = .mxfp8, .bits = 8, .group_size = 32 }}, K, s);
    }
}

// Bar: a ModelOpt NVFP4 linear that loads dense carries `weight_scale_2` as MLX's global scale amax = s2 * 448 * 6,
// bit-equal to MLX's own dequantization at that amax, and multiplies through the format as the direct path does.
test "quant nvfp4: ModelOpt's weight_scale_2 reaches MLX as amax / (448 * 6)" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(4);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(testing.io, &path_buf)];
    try modelopt_pack.writeMiniPack(a, testing.io, tmp.dir, .{ .layers = 1, .experts = 1 });

    const w = try randW(prng.random(), &.{ N, K }, s);
    defer _ = mlx.mlx_array_free(w);
    const q = try quantized(w, "nvfp4", 4, 16, s);
    defer _ = mlx.mlx_array_free(q.w);
    defer _ = mlx.mlx_array_free(q.sc);
    var bytes = mlx.mlx_array_new(); // ModelOpt stores the nibbles as u8
    try mlx.check(mlx.mlx_view(&bytes, q.w, .uint8, s));
    const s2 = [_]f32{0.0125};
    const base = "model.visual.blocks.0.attn.proj";
    var weights = resident.Weights.init(a);
    try weights.put(base ++ ".weight", bytes);
    try weights.put(base ++ ".weight_scale", try share(q.sc));
    try weights.put(base ++ ".weight_scale_2", mlx.mlx_array_new_data(&s2, &[_]c_int{1}, 1, .float32));
    try modelopt_pack.adapt(&weights, dir, null, false, null, null);
    var r = resident.Resident.adopt(weights);
    defer r.deinit();

    const amax_v = [_]f32{s2[0] * (448.0 * 6.0)};
    const amax = mlx.mlx_array_new_data(&amax_v, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(amax);
    // On the stream the load dequantizes on.
    const ls = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(ls);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_dequantize(&ref, q.w, q.sc, .{}, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", amax, .{ .value = .bfloat16, .has_value = true }, ls));
    try ops.expectBitEqual(r.weights.get(base ++ ".weight").?, ref);
    try expectParity(&nvfp4, &r, "{\"quant_method\":\"modelopt\"}", &.{.{ .base = base, .layout = .dense }}, K, s);
}

/// A row-scaled e4m3 weight as Sushi's adaptation leaves a QKV part: codes `[n, k]` (finite, both signs) and one f32
/// scale row per code row.
fn rowScaledFp8(rnd: std.Random, n: c_int, k: c_int) !Triple {
    const codes = try testing.allocator.alloc(u8, @intCast(n * k));
    defer testing.allocator.free(codes);
    for (codes) |*c| c.* = rnd.uintLessThan(u8, 0x78) | (if (rnd.boolean()) @as(u8, 0x80) else 0);
    const sv = try testing.allocator.alloc(f32, @intCast(n * @divExact(k, 128)));
    defer testing.allocator.free(sv);
    for (sv) |*v| v.* = 0.001 + 0.01 * rnd.float(f32);
    return .{
        .w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ n, k }, 2, .uint8),
        .sc = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ n, @divExact(k, 128) }, 2, .float32),
    };
}

// Bar: Sushi's row-scaled e4m3 QKV parts load as fp8 rows and multiply bit-equal to the direct path at GEMV widths
// and past them, beside the layer-0 MLP the adaptation decoded dense.
test "quant fp8: row-scaled e4m3 weights load and multiply bit-equal to the direct path" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(5);
    var weights = resident.Weights.init(testing.allocator);
    try putTriple(&weights, "l.q_proj", try rowScaledFp8(prng.random(), N, K));
    try weights.put("l.gate_proj.weight", try randW(prng.random(), &.{ N, K }, s));
    var r = resident.Resident.adopt(weights);
    defer r.deinit();
    try expectParity(&fp8, &r, "{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}", &.{
        .{ .base = "l.q_proj", .layout = .fp8_rows, .bits = 8 },
        .{ .base = "l.gate_proj", .layout = .dense },
    }, K, s);
}

// Bar: routed experts through the format's MoE dispatch are bit-equal to the engine's gather, sorted or not, for
// every bank layout the gather serves.
test "quant moe: expert banks gather bit-equal to the engine's dispatch" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(6);
    const E: c_int = 4;
    const R: c_int = 8;
    const Bank = struct { t: *const abi.QuantTable, block: []const u8, mode: ?[*:0]const u8, bits: u32, gs: u32 };
    const banks = [_]Bank{
        .{ .t = &affine, .block = "{\"bits\":4,\"group_size\":64}", .mode = "affine", .bits = 4, .gs = 64 },
        .{ .t = &affine, .block = "{\"bits\":3,\"group_size\":32}", .mode = "affine", .bits = 3, .gs = 32 },
        .{ .t = &nvfp4, .block = "{\"bits\":4,\"group_size\":16,\"mode\":\"nvfp4\"}", .mode = "nvfp4", .bits = 4, .gs = 16 },
        .{ .t = &nvfp4, .block = "{\"bits\":4,\"group_size\":32,\"mode\":\"mxfp4\"}", .mode = "mxfp4", .bits = 4, .gs = 32 },
        .{ .t = &mxfp8, .block = "{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\"}", .mode = "mxfp8", .bits = 8, .gs = 32 },
        .{ .t = &affine, .block = "{\"bits\":4,\"group_size\":64}", .mode = null, .bits = 0, .gs = 0 },
    };
    const ids = [_]u32{ 0, 0, 1, 1, 2, 3, 3, 3 };
    const rhs = mlx.mlx_array_new_data(&ids, &[_]c_int{R}, 1, .uint32);
    defer _ = mlx.mlx_array_free(rhs);
    const x = try randW(prng.random(), &.{ R, 1, K }, s);
    defer _ = mlx.mlx_array_free(x);
    for (banks) |b| {
        const raw = try randW(prng.random(), &.{ E, N, K }, s);
        var weights = resident.Weights.init(testing.allocator);
        if (b.mode) |m| {
            defer _ = mlx.mlx_array_free(raw);
            try putTriple(&weights, "mlp.switch_mlp.up_proj", try quantized(raw, m, b.bits, b.gs, s));
        } else try weights.put("mlp.switch_mlp.up_proj.weight", raw);
        var r = resident.Resident.adopt(weights);
        defer r.deinit();
        const src = r.source();
        var ctx: abi.LoadCtx = .{ .block_json = .of(b.block), .source = &src, .stream = .{ .ctx = s.ctx } };
        try testing.expect(b.t.open(&ctx).isOk());
        defer b.t.close(&ctx);
        var bank: abi.Weight = .{};
        try testing.expect(b.t.load_bank(&ctx, .of("mlp.switch_mlp.up_proj"), @intCast(K), &bank).isOk());
        defer b.t.release(&bank);
        for ([_]bool{ true, false }) |sorted| {
            var got: abi.Array = .{};
            try testing.expect(b.t.moe(&bank, wrap(x), .{}, wrap(rhs), sorted, .{ .ctx = s.ctx }, &got).isOk());
            defer _ = mlx.mlx_array_free(arr(got));
            var want = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(want);
            const mode: QuantMode = if (b.mode) |m| QuantMode.fromString(std.mem.span(m)).? else .affine;
            const w = tensor(&ctx, "mlp.switch_mlp.up_proj", "weight");
            const wt = if (b.mode == null) try transposeBf16Weight(w, s) else w;
            defer if (b.mode == null) {
                _ = mlx.mlx_array_free(wt);
            };
            try gatherExpertMm(&want, x, wt, tensor(&ctx, "mlp.switch_mlp.up_proj", "scales"), tensor(&ctx, "mlp.switch_mlp.up_proj", "biases"), .{}, rhs, b.bits, b.gs, mode, sorted, s);
            try ops.expectBitEqual(arr(got), want);
        }
    }
}

test "quant native: a missing weight and an unopened load are refused by name" {
    var weights = resident.Weights.init(testing.allocator);
    var r = resident.Resident.adopt(weights);
    defer r.deinit();
    _ = &weights;
    const src = r.source();
    var ctx: abi.LoadCtx = .{ .block_json = .of("{\"bits\":4}"), .source = &src };
    var w: abi.Weight = .{};
    try testing.expectEqualStrings("QuantFormatNotOpened", affine.load_tensor(&ctx, .of("x"), 0, &w).why.slice());
    try testing.expect(affine.open(&ctx).isOk());
    defer affine.close(&ctx);
    try testing.expectEqualStrings("MissingWeight", affine.load_tensor(&ctx, .of("x"), 0, &w).why.slice());
    try testing.expectEqual(@as(u64, 7), affine.bill(.of("{}"), .of("/nowhere"), 7));
}
