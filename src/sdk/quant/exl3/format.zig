//! The EXL3 quant format: Sushi's `expert_quant` routed-expert banks of 16x16 trellis tiles, `format: exl3` (MCG,
//! MUL1) and its STQ codebook family (`format: stq`: STD1, SPEC1, PAIR1 under a named, digest-pinned table). The
//! decode is `sushi/` (Sushi 1.3.0 as upstream ships it, module `sushi_exl3`). Experts only: a layer's three banks
//! run as one routed SwiGLU dispatch, never per projection.

const std = @import("std");
/// Sushi's decode itself, for the plugins that run the routed chain's stages on their own (mlx-stream's GLM-5.3).
pub const sushi = @import("sushi_exl3");
const abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const block = @import("../block.zig");

const Str = abi.Str;
const Verdict = abi.Verdict;

pub const Spec = sushi.Spec;
pub const Bank = sushi.Bank;
pub const Proj = sushi.Proj;
pub const Decode = sushi.format.Decode;
pub const Codebook = sushi.format.Codebook;
pub const Window = sushi.format.Window;
/// The host's row planning around `routed` (`DECODE_ROWS_MAX`, `usesPrefillArm`, `rowsOfShape`).
pub const kernels = sushi.kernels;

/// `expert_quant` of a config root, refused by name when this build has no kernels for its codebook, window, table
/// or for `topk` routed experts.
pub fn parse(root: std.json.ObjectMap, topk: u32) !Spec {
    const spec = try sushi.parseExpertQuant(root);
    try sushi.admitTopK(topk);
    return spec;
}

/// What a dispatch decodes with: an STQ table rides along (a table-keyed codebook without it is refused).
pub fn decode(spec: Spec) Decode {
    return .{ .codebook = spec.codebook, .window = spec.window, .table = spec.table };
}

/// One projection's bank at load: its suh/svh present, a trellis the kernels decode at the pack's rate, and a rate
/// the pack's codebook table has a row for.
pub fn admit(p: Proj, experts: u32, in_dim: u32, out_dim: u32, spec: Spec) !void {
    if (p.suh.ctx == null or p.svh.ctx == null) return error.MissingWeight;
    const shape = mlx.getShape(p.trellis);
    if (!sushi.trellisAdmitted(shape, experts, in_dim, out_dim, spec.rate)) return error.Exl3TrellisGeometry;
    const rate = sushi.format.kFromPackedDim(@intCast(shape[3])) orelse return error.Exl3TrellisGeometry;
    if (!rate.in(spec.rates)) return error.Exl3CodebookTableRateMissing;
}

pub const Routed = struct {
    /// > 0: GLM's clamped SwiGLU at this limit.
    limit: c_int = 0,
    verify_rows: bool = false,
};

/// Routed SwiGLU of `x` [B, S, D] through `bank`'s experts `inds` [B, S, K], weighted by `scores`.
pub fn routed(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, spec: Spec, opts: Routed) !mlx.mlx_array {
    const dec = decode(spec);
    return if (opts.limit > 0)
        sushi.moeClamped(s, x, bank, inds, scores, dec, opts.limit)
    else
        sushi.moe(s, x, bank, inds, scores, dec, opts.verify_rows);
}

// ── the quant table ──

fn arr(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}
fn wrap(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn failed(e: anyerror) Verdict {
    return Verdict.named(.failed, @errorName(e));
}

fn specOf(ctx: *const abi.LoadCtx) ?*const Spec {
    return @ptrCast(@alignCast(ctx.state orelse return null));
}

fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const root = block.expertQuantRoot(arena.allocator(), ctx.block_json.slice()) orelse return Verdict.refuse(.failed, "expert_quant block does not parse");
    const spec = sushi.parseExpertQuant(root) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const st = std.heap.page_allocator.create(Spec) catch |e| return failed(e);
    st.* = spec;
    ctx.state = st;
    return .ok;
}

fn close(ctx: *abi.LoadCtx) callconv(.c) void {
    if (ctx.state) |p| std.heap.page_allocator.destroy(@as(*Spec, @ptrCast(@alignCast(p))));
    ctx.state = null;
}

fn noTensor(_: *const abi.LoadCtx, _: Str, _: u32, _: *abi.Weight) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "exl3: routed expert banks only");
}

fn loadBank(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    loadBankInto(ctx, name.slice(), in_dim, out) catch |e| return failed(e);
    return .ok;
}

/// `<base>.{trellis,suh,svh}`: trellis U16 [E, in/16, out/16, n], suh F16 [E, in], svh F16 [E, out]. The weight
/// carries its decode: `layout` the codebook, `flags` the window's bits, `block` the rate (halfwords per tile),
/// `state` the pack's table index + 1 (null: none).
fn loadBankInto(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, out: *abi.Weight) !void {
    const spec = specOf(ctx) orelse return error.QuantFormatNotOpened;
    const t = tensor(ctx, base, "trellis");
    if (t.ctx == null) return error.MissingWeight;
    const shape = mlx.getShape(t);
    if (shape.len != 4) return error.Exl3TrellisGeometry;
    const p: Proj = .{ .trellis = t, .suh = tensor(ctx, base, "suh"), .svh = tensor(ctx, base, "svh") };
    const experts: u32 = @intCast(shape[0]);
    const in: u32 = @intCast(shape[1] * 16);
    if (in_dim != 0 and in_dim != in) return error.Exl3TrellisGeometry;
    try admit(p, experts, in, @intCast(shape[2] * 16), spec.*);
    out.* = .{};
    errdefer release(out);
    out.w = wrap(try share(p.trellis));
    out.scales = wrap(try share(p.suh));
    out.biases = wrap(try share(p.svh));
    out.layout = @backingInt(spec.codebook);
    out.flags = @intCast(spec.window.bits());
    out.block = @intCast(shape[3]);
    out.state = if (spec.table) |i| @ptrFromInt(@as(usize, i) + 1) else null;
}

fn tensor(ctx: *const abi.LoadCtx, base: []const u8, leaf: []const u8) mlx.mlx_array {
    var nb: [320]u8 = undefined;
    const name = std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, leaf }) catch return .{};
    return arr(ctx.source.tensor(name));
}

/// A new handle on a source array (the source keeps its own).
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

fn noMatmul(_: *const abi.Weight, _: abi.Array, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "exl3: routed expert banks only");
}

fn noMoe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "exl3: a layer's banks run as one routed SwiGLU (swiglu)");
}

fn projOf(w: *const abi.Weight) Proj {
    return .{ .trellis = arr(w.w), .suh = arr(w.scales), .svh = arr(w.biases) };
}

/// The decode spec a loaded bank carries; the three banks of a layer share it.
fn decodeOf(w: *const abi.Weight) ?Decode {
    const window = Window.fromBits(w.flags) orelse return null;
    const index: ?u8 = if (w.state) |p| @intCast(@intFromPtr(p) - 1) else null;
    return .{ .codebook = @fromBackingInt(@intCast(w.layout)), .window = window, .table = index };
}

fn swiglu(gate: *const abi.Weight, up: *const abi.Weight, down: *const abi.Weight, x: abi.Array, inds: abi.Array, scores: abi.Array, opts: *const abi.Swiglu, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(gate) orelse return Verdict.refuse(.failed, "exl3: a bank without its decode spec");
    if (!std.meta.eql(decodeOf(up), dec) or !std.meta.eql(decodeOf(down), dec)) return Verdict.refuse(.unsupported, "exl3: a layer's banks decode differently");
    const bank: Bank = .{ .gate = projOf(gate), .up = projOf(up), .down = projOf(down) };
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = (if (opts.limit > 0)
        sushi.moeClamped(st, arr(x), bank, arr(inds), arr(scores), dec, opts.limit)
    else
        sushi.moe(st, arr(x), bank, arr(inds), arr(scores), dec, opts.verify_rows)) catch |e| return failed(e);
    out.* = wrap(y);
    return .ok;
}

/// Banks load as stored.
fn bill(_: Str, _: Str, stored: u64) callconv(.c) u64 {
    return stored;
}

fn builtin() block.Builtin {
    for (block.builtins) |b| if (std.mem.eql(u8, b.name, "exl3")) return b;
    unreachable;
}

pub const table: abi.QuantTable = .{
    .name = Str.of("exl3"),
    .claims = builtin().claims,
    .validate = builtin().validate,
    .bill = &bill,
    .open = &open,
    .close = &close,
    .load_tensor = &noTensor,
    .load_bank = &loadBank,
    .release = &release,
    .matmul = &noMatmul,
    .moe = &noMoe,
    .swiglu = &swiglu,
};

// ── tests ──

const testing = std.testing;
const ops = @import("../../../mlx/ops.zig");
const resident = @import("../../weights/resident.zig");

const D: c_int = 128;
const I: c_int = 128;
const E: c_int = 4;
const K: c_int = 2;

/// One projection's random bank under `<base>.{trellis,suh,svh}` in `w`, at `n` halfwords per tile.
fn putBank(w: *resident.Weights, rnd: std.Random, base: []const u8, in: c_int, out: c_int, n: c_int) !void {
    const words = try testing.allocator.alloc(u16, @intCast(E * @divExact(in, 16) * @divExact(out, 16) * n));
    defer testing.allocator.free(words);
    for (words) |*v| v.* = rnd.int(u16);
    var nb: [128]u8 = undefined;
    try w.put(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base}), mlx.mlx_array_new_data(words.ptr, &[_]c_int{ E, @divExact(in, 16), @divExact(out, 16), n }, 4, .uint16));
    for ([_]struct { leaf: []const u8, dim: c_int }{ .{ .leaf = "suh", .dim = in }, .{ .leaf = "svh", .dim = out } }) |side| {
        const h = try testing.allocator.alloc(u16, @intCast(E * side.dim));
        defer testing.allocator.free(h);
        for (h) |*v| v.* = sushi.format.f32ToF16Bits(if (rnd.boolean()) 0.125 else -0.125);
        try w.put(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, side.leaf }), mlx.mlx_array_new_data(h.ptr, &[_]c_int{ E, side.dim }, 2, .float16));
    }
}

const projs = [_][]const u8{ "l.gate_proj", "l.up_proj", "l.down_proj" };

const Case = struct {
    block: []const u8,
    /// Halfwords per tile of the gate, up and down banks.
    n: [3]c_int,
    /// What the engine built before the format: `Decode{ codebook, window }`, plus the table STQ needs.
    want: Decode,
    limit: c_int = 0,
};

// Bar: a layer loaded through the table runs bit-equal to Sushi's own dispatch with the decode the engine passed
// before the switch, at decode, verify and prefill widths, for MCG, MUL1, GLM's clamped SwiGLU and an STQ table.
test "quant exl3: routed SwiGLU through the table is bit-equal to the direct dispatch" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const cases = [_]Case{
        .{ .block = "{\"format\":\"exl3\",\"k\":3,\"codebook\":\"mcg\"}", .n = .{ 48, 48, 48 }, .want = .{ .codebook = .mcg } },
        .{ .block = "{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mul1\"}", .n = .{ 32, 32, 32 }, .want = .{ .codebook = .mul1 } },
        .{ .block = "{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}", .n = .{ 40, 36, 40 }, .want = .{ .codebook = .mcg, .window = .w14 }, .limit = 10 },
        .{ .block = "{\"format\":\"stq\",\"k\":3,\"codebook\":\"std1\",\"window\":15,\"codebook_table\":\"std1\",\"codebook_digest\":\"c5dba171\"}", .n = .{ 48, 32, 48 }, .want = Decode.tabled(.std1, .w15) },
    };
    var prng = std.Random.DefaultPrng.init(0xe713);
    for (cases) |c| {
        var w = resident.Weights.init(testing.allocator);
        for (projs, c.n, 0..) |base, n, i| try putBank(&w, prng.random(), base, if (i == 2) I else D, if (i == 2) D else I, n);
        var r = resident.Resident.adopt(w);
        defer r.deinit();
        const src = r.source();
        var ctx: abi.LoadCtx = .{ .block_json = .of(c.block), .source = &src, .stream = .{ .ctx = s.ctx } };
        try testing.expect(table.open(&ctx).isOk());
        defer table.close(&ctx);
        var banks: [3]abi.Weight = @splat(.{});
        defer for (&banks) |*b| table.release(b);
        for (projs, &banks, 0..) |base, *b, i| try testing.expect(table.load_bank(&ctx, .of(base), @intCast(if (i == 2) I else D), b).isOk());
        const direct: Bank = .{ .gate = projOf(&banks[0]), .up = projOf(&banks[1]), .down = projOf(&banks[2]) };
        const spec = specOf(&ctx).?.*;
        for ([_]struct { rows: c_int, verify: bool }{ .{ .rows = 1, .verify = false }, .{ .rows = 3, .verify = true }, .{ .rows = 3, .verify = false }, .{ .rows = kernels.DECODE_ROWS_MAX + 5, .verify = false } }) |shape| {
            if (c.limit > 0 and shape.verify) continue;
            const x = try @import("../../testing/parity.zig").testRandWeightBf16(prng.random(), &.{ 1, shape.rows, D }, s);
            defer _ = mlx.mlx_array_free(x);
            const ids = try testing.allocator.alloc(i32, @intCast(shape.rows * K));
            defer testing.allocator.free(ids);
            const sc = try testing.allocator.alloc(f32, ids.len);
            defer testing.allocator.free(sc);
            for (ids, sc, 0..) |*id, *v, i| {
                id.* = @intCast((i / @as(usize, K) + i % @as(usize, K)) % @as(usize, E));
                v.* = prng.random().float(f32);
            }
            const inds = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, shape.rows, K }, 3, .int32);
            defer _ = mlx.mlx_array_free(inds);
            const scores = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{ 1, shape.rows, K }, 3, .float32);
            defer _ = mlx.mlx_array_free(scores);
            const want = if (c.limit > 0)
                try sushi.moeClamped(s, x, direct, inds, scores, c.want, c.limit)
            else
                try sushi.moe(s, x, direct, inds, scores, c.want, shape.verify);
            defer _ = mlx.mlx_array_free(want);
            const opts: abi.Swiglu = .{ .limit = c.limit, .verify_rows = shape.verify };
            var got: abi.Array = .{};
            const v = table.swiglu.?(&banks[0], &banks[1], &banks[2], wrap(x), wrap(inds), wrap(scores), &opts, .{ .ctx = s.ctx }, &got);
            if (!v.isOk()) std.debug.print("{s}: {s}\n", .{ c.block, v.why.slice() });
            try testing.expect(v.isOk());
            defer _ = mlx.mlx_array_free(arr(got));
            try ops.expectBitEqual(arr(got), want);
            const engine = try routed(s, x, direct, inds, scores, spec, .{ .limit = c.limit, .verify_rows = shape.verify });
            defer _ = mlx.mlx_array_free(engine);
            try ops.expectBitEqual(engine, want);
        }
    }
}

// Bar: what the format cannot serve is refused by name, at open or at the bank's load, never at a dispatch.
test "quant exl3: an unknown table, a bank rate outside the table and per-projection calls are refused by name" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xe714);
    var w = resident.Weights.init(testing.allocator);
    try putBank(&w, prng.random(), "l.gate_proj", D, I, 16);
    try putBank(&w, prng.random(), "l.up_proj", D, I, 48);
    var r = resident.Resident.adopt(w);
    defer r.deinit();
    const src = r.source();

    var bad: abi.LoadCtx = .{ .block_json = .of("{\"format\":\"stq\",\"k\":2,\"codebook\":\"spec1\",\"window\":15}"), .source = &src, .stream = .{ .ctx = s.ctx } };
    const v0 = table.open(&bad);
    try testing.expectEqualStrings("Exl3CodebookTableUnknown", v0.why.slice());

    var ctx: abi.LoadCtx = .{ .block_json = .of("{\"format\":\"stq\",\"k\":3,\"codebook\":\"std1\",\"window\":15,\"codebook_table\":\"std1\",\"codebook_digest\":\"c5dba171\"}"), .source = &src, .stream = .{ .ctx = s.ctx } };
    try testing.expect(table.open(&ctx).isOk());
    defer table.close(&ctx);
    var b: abi.Weight = .{};
    try testing.expectEqualStrings("Exl3CodebookTableRateMissing", table.load_bank(&ctx, .of("l.gate_proj"), D, &b).why.slice());
    try testing.expectEqualStrings("Exl3TrellisGeometry", table.load_bank(&ctx, .of("l.up_proj"), I + 16, &b).why.slice());
    try testing.expectEqualStrings("MissingWeight", table.load_bank(&ctx, .of("l.down_proj"), I, &b).why.slice());
    try testing.expect(table.load_bank(&ctx, .of("l.up_proj"), D, &b).isOk());
    defer table.release(&b);
    try testing.expectEqual(Verdict.Code.unsupported, table.load_tensor(&ctx, .of("l.up_proj"), D, &b).code);
    var out: abi.Array = .{};
    try testing.expectEqual(Verdict.Code.unsupported, table.matmul(&b, .{}, .{ .ctx = s.ctx }, &out).code);
    try testing.expectEqual(Verdict.Code.unsupported, table.moe(&b, .{}, .{}, .{}, false, .{ .ctx = s.ctx }, &out).code);
}

// Bar: validation refuses what Sushi's parse refuses, by its error's name, and admits the shipped blocks.
test "quant exl3: validation is Sushi's parse plus the reduce bank's top-k" {
    const dir: abi.PackDir = undefined;
    for ([_][]const u8{
        "{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}",
        "{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mul1\"}",
        "{\"format\":\"stq\",\"k\":2.5,\"codebook\":\"spec1\",\"window\":15,\"codebook_table\":\"spec1-w15\",\"codebook_digest\":\"37f72431\"}",
        "{\"format\":\"stq\",\"k\":3,\"codebook\":\"pair1\",\"window\":16,\"codebook_table\":\"pair1-w16-v1\",\"codebook_digest\":\"c089b4eb\"}",
    }) |b| try testing.expect(table.validate(.of(b), .of("{\"num_experts_per_tok\":8}"), &dir).isOk());
    for ([_]struct { block: []const u8, config: []const u8, want: []const u8 }{
        .{ .block = "{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mul2\"}", .config = "{}", .want = "ExpertLayoutUnsupported" },
        .{ .block = "{\"format\":\"stq\",\"k\":2,\"codebook\":\"spec1\",\"window\":15}", .config = "{}", .want = "Exl3CodebookTableUnknown" },
        .{ .block = "{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mcg\",\"window\":7}", .config = "{}", .want = "Exl3WindowUnsupported" },
        .{ .block = "{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mcg\"}", .config = "{\"text_config\":{\"num_experts_per_tok\":33}}", .want = "Exl3TopKExceedsReduceBank" },
    }) |c| try testing.expectEqualStrings(c.want, table.validate(.of(c.block), .of(c.config), &dir).why.slice());
}
