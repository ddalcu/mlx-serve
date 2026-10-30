//! The GGUF quant format: a GGUF file's tensors as stored, raw ggml blocks in a uint8 weight beside a `scales`
//! sentinel naming the block type (`kernels.Info`), multiplied by our own Metal kernels (`kernels.zig`). The file's
//! rebuilt config carries `quantization.mode: gguf`; its tensor types were vetted when the file was found servable.

const std = @import("std");
const abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const block = @import("../block.zig");
const kernels = @import("kernels.zig");

const Str = abi.Str;
const Verdict = abi.Verdict;

fn arr(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}
fn wrap(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn failed(e: anyerror) Verdict {
    return Verdict.named(.failed, @errorName(e));
}

fn open(_: *abi.LoadCtx) callconv(.c) Verdict {
    return .ok;
}

fn close(_: *abi.LoadCtx) callconv(.c) void {}

fn loadTensor(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    loadInto(ctx, name.slice(), in_dim, out) catch |e| return failed(e);
    return .ok;
}

/// `<base>.weight` (uint8 [rows, row bytes], or [experts, rows, row bytes] for a bank) and its `<base>.scales`
/// sentinel; `in_dim` must match the weight's own input width.
fn loadInto(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, out: *abi.Weight) !void {
    const w = tensor(ctx, base, "weight");
    const sc = tensor(ctx, base, "scales");
    const info = kernels.infoOf(w, sc) orelse return error.NotAGgufWeight;
    const ws = mlx.getShape(w);
    const in = @as(usize, @intCast(ws[ws.len - 1])) / info.ty.blockBytes() * info.ty.blockElems();
    if (in_dim != 0 and in_dim != in) return error.BadShape;
    out.* = .{};
    errdefer release(out);
    out.w = wrap(try share(w));
    out.scales = wrap(try share(sc));
    out.layout = @intCast(@backingInt(info.ty));
}

fn tensor(ctx: *const abi.LoadCtx, base: []const u8, leaf: []const u8) mlx.mlx_array {
    var nb: [320]u8 = undefined;
    const name = std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, leaf }) catch return .{};
    return arr(ctx.source.tensor(name));
}

fn share(a: mlx.mlx_array) !mlx.mlx_array {
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    return h;
}

fn release(w: *abi.Weight) callconv(.c) void {
    for ([_]abi.Array{ w.w, w.scales, w.biases, w.side }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(arr(a));
    };
    w.* = .{};
}

fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const info = kernels.infoOf(arr(w.w), arr(w.scales)) orelse return Verdict.refuse(.failed, "gguf: not a GGUF weight");
    out.* = wrap(kernels.linear(info, arr(x), arr(w.w), .{ .ctx = s.ctx }) catch |e| return failed(e));
    return .ok;
}

/// `sorted` is a hint the kernels do not need: each picks its arm by the row count.
fn moe(bank: *const abi.Weight, x: abi.Array, lhs: abi.Array, rhs: abi.Array, _: bool, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const info = kernels.infoOf(arr(bank.w), arr(bank.scales)) orelse return Verdict.refuse(.failed, "gguf: not a GGUF bank");
    out.* = wrap(kernels.gatherLinear(info, arr(x), arr(bank.w), arr(lhs), arr(rhs), .{ .ctx = s.ctx }) catch |e| return failed(e));
    return .ok;
}

/// Tensors load as stored.
fn bill(_: Str, _: Str, stored: u64) callconv(.c) u64 {
    return stored;
}

fn builtin() block.Builtin {
    for (block.builtins) |b| if (std.mem.eql(u8, b.name, "gguf")) return b;
    unreachable;
}

pub const table: abi.QuantTable = .{
    .name = Str.of("gguf"),
    .claims = builtin().claims,
    .validate = builtin().validate,
    .bill = &bill,
    .open = &open,
    .close = &close,
    .load_tensor = &loadTensor,
    .load_bank = &loadTensor,
    .release = &release,
    .matmul = &matmul,
    .moe = &moe,
};

// ── tests ──

const testing = std.testing;
const ops = @import("../../../mlx/ops.zig");
const quants = @import("quants.zig");
const resident = @import("../../weights/resident.zig");
const parity = @import("../../testing/parity.zig");

/// The ggml fixture's 4 blocks as a [2 rows x 2 blocks] weight in the kernels' layout, and as a bank of 2 experts.
fn putFixture(w: *resident.Weights, comptime ty: quants.GgmlType, s: mlx.mlx_stream) !void {
    const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
    const bytes = data[0 .. 4 * ty.blockBytes()];
    const in = 2 * ty.blockElems();
    const rb = kernels.rowBytes(ty, in);
    var rows: [4 * 256]u8 = undefined;
    if (kernels.isSplit(ty, in)) kernels.packRows(ty, in, rows[0 .. 2 * rb], bytes) else @memcpy(rows[0..bytes.len], bytes);
    for ([_][]const c_int{ &.{ 2, @intCast(rb) }, &.{ 2, 1, @intCast(rb) } }, [_][]const u8{ "l", "e" }) |shape, base| {
        var nb: [64]u8 = undefined;
        try w.put(try std.fmt.bufPrint(&nb, "{s}.{s}.weight", .{ base, @tagName(ty) }), mlx.mlx_array_new_data(&rows, shape.ptr, @intCast(shape.len), .uint8));
        try w.put(try std.fmt.bufPrint(&nb, "{s}.{s}.scales", .{ base, @tagName(ty) }), try kernels.sentinel(.{ .ty = ty }, s));
    }
}

// Bar: every block type loads through the table and multiplies bit-equal to the kernels the engine calls directly,
// on the matvec, tile and dequant arms, and through an expert bank at decode and prefill widths.
test "quant gguf: every block type multiplies through the table bit-equal to the direct kernels" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var w = resident.Weights.init(testing.allocator);
    inline for (kernels.all_types) |ty| try putFixture(&w, ty, s);
    var r = resident.Resident.adopt(w);
    defer r.deinit();
    const src = r.source();
    var ctx: abi.LoadCtx = .{ .block_json = .of("{\"group_size\":32,\"bits\":4,\"mode\":\"gguf\"}"), .source = &src, .stream = .{ .ctx = s.ctx } };
    try testing.expect(table.open(&ctx).isOk());
    defer table.close(&ctx);
    var prng = std.Random.DefaultPrng.init(0x66);
    inline for (kernels.all_types) |ty| {
        const in: c_int = @intCast(2 * ty.blockElems());
        var lin: abi.Weight = .{};
        try testing.expect(table.load_tensor(&ctx, .of("l." ++ @tagName(ty)), @intCast(in), &lin).isOk());
        defer table.release(&lin);
        try testing.expectEqualStrings("BadShape", table.load_tensor(&ctx, .of("l." ++ @tagName(ty)), @intCast(in + 1), &lin).why.slice());
        const wl = src.tensor("l." ++ @tagName(ty) ++ ".weight");
        for ([_]c_int{ 1, 3, kernels.MATVEC_MAX_ROWS + 3, 32 }) |m| {
            const x = try parity.testRandWeightBf16(prng.random(), &.{ 1, m, in }, s);
            defer _ = mlx.mlx_array_free(x);
            var got: abi.Array = .{};
            try testing.expect(table.matmul(&lin, wrap(x), .{ .ctx = s.ctx }, &got).isOk());
            defer _ = mlx.mlx_array_free(arr(got));
            const want = try kernels.linear(.{ .ty = ty }, x, arr(wl), s);
            defer _ = mlx.mlx_array_free(want);
            try ops.expectBitEqual(arr(got), want);
        }
        var bank: abi.Weight = .{};
        try testing.expect(table.load_bank(&ctx, .of("e." ++ @tagName(ty)), @intCast(in), &bank).isOk());
        defer table.release(&bank);
        const wb = src.tensor("e." ++ @tagName(ty) ++ ".weight");
        for ([_]c_int{ 1, 16 }) |m| {
            const x = try parity.testRandWeightBf16(prng.random(), &.{ m, 1, in }, s);
            defer _ = mlx.mlx_array_free(x);
            const ids = [_]u32{ 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0 };
            const experts = mlx.mlx_array_new_data(&ids, &[_]c_int{m}, 1, .uint32);
            defer _ = mlx.mlx_array_free(experts);
            var got: abi.Array = .{};
            try testing.expect(table.moe(&bank, wrap(x), .{}, wrap(experts), true, .{ .ctx = s.ctx }, &got).isOk());
            defer _ = mlx.mlx_array_free(arr(got));
            const want = try kernels.gatherLinear(.{ .ty = ty }, x, arr(wb), .{ .ctx = null }, experts, s);
            defer _ = mlx.mlx_array_free(want);
            try ops.expectBitEqual(arr(got), want);
        }
    }
    var dense: abi.Weight = .{};
    try testing.expectEqualStrings("NotAGgufWeight", table.load_tensor(&ctx, .of("missing"), 0, &dense).why.slice());
}
