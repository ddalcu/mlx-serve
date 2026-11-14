//! GLM-5.3-Flash's and Ling's MoE layers run on the shared `Moe` block (weights built by `trunk.zig`): GLM's
//! one-kernel decode (`glm5.moeDecode`, affine banks, up to `MOE_DECODE_MAX_ROWS` rows) is the block's `fused` arm.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const glm5 = @import("glm5_next.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");

const Array = mlx.mlx_array;

/// GLM's routed + shared experts in two dispatches; null (another quant mode, geometry or device) takes the block's
/// arms.
pub fn fusedDecode(f: *const moe.Fused, w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array, inds: Array, scores: Array) !?Array {
    const se = &(w.shared orelse return null);
    inline for (.{ w.gate.qp, w.up.qp, w.down.qp }) |qp| if (qp.mode != .affine) return null;
    inline for (.{ se.gate, se.up, se.down }) |l| if (l.mode != .affine or l.s.ctx == null) return null;
    const k: c_int = @intCast(mlx.mlx_array_size(inds));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, inds, &[_]c_int{k}, 1, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, flat, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{k}, 1, s));
    const y = (try glm5.moeDecode(s, x, ids, sc, qb(w.gate), qb(w.up), qb(w.down), ql(se.gate), ql(se.up), ql(se.down), f.limit)) orelse return null;
    defer _ = mlx.mlx_array_free(y);
    const xs = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xs.ptr, xs.len, s));
    return out;
}

fn qb(b: moe.Bank) glm5.QBank {
    return .{ .w = b.w, .s = b.s, .b = b.b, .bits = b.qp.bits, .gs = b.qp.group_size };
}

fn ql(l: Linear) glm5.QBank {
    return .{ .w = l.w, .s = l.s, .b = l.b, .bits = l.bits, .gs = l.group_size };
}

/// The block over `x` (owned by the caller).
pub fn forward(w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array) !Array {
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    return g.escape(try moe.mlp(w, &g, x));
}

/// `forward` with each row given its own one-row call's bytes (`moe.mlpRows`).
pub fn forwardRows(w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array) !Array {
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    return g.escape(try moe.mlpRows(w, &g, x));
}

/// The rows the EXL3 formats' routed SwiGLU sums each row by itself (their decode chain): `forwardLastRow`'s bound.
pub const last_row_max_rows: c_int = @intCast(exl3.kernels.DECODE_ROWS_MAX);

/// `forward`'s output at `x`'s last row alone (`moe.mlpLastRow`, owned by the caller), or null where the block takes
/// another arm and the caller runs `forward`.
pub fn forwardLastRow(w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array) !?Array {
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    const y = (try moe.mlpLastRow(w, &g, x, last_row_max_rows)) orelse return null;
    return g.escape(y);
}

/// Whether a split prompt forward's spans of `rows_a` and `rows_b` rows run this layer's routed experts in one call
/// (`moe.splitServes`).
pub const splitServes = moe.splitServes;

/// The block over a split prompt forward's two spans, `x_a` then `x_b` (`moe.mlpSplit`: their routed experts in one
/// call of the streamed bank's source), each output as `forward` gives it for that span alone (both owned by the caller).
pub fn forwardSplit(w: *const moe.Moe.Weights, s: mlx.mlx_stream, x_a: Array, x_b: Array) ![2]Array {
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    const ys = try moe.mlpSplit(w, &g, x_a, x_b);
    return .{ g.escape(ys[0]), g.escape(ys[1]) };
}

// ── tests ──

const testing = std.testing;
const quant_abi = @import("../../abi/quant.zig");
const resident = @import("../../sdk/weights/resident.zig");
const randBf16 = @import("../../sdk/testing/parity.zig").testRandUniformBf16;

/// GLM-5.3-Flash's MTP bank geometry (hidden 4096, inter 2048, 8 routes a row, K4 MUL1 codewords, clamped at 10) over
/// 16 experts: a row's bytes read only its own routes.
const t_hidden: c_int = 4096;
const t_inter: c_int = 2048;
const t_experts: c_int = 16;
const t_topk: c_int = 8;
const t_limit: c_int = 10;

/// A random EXL3 projection bank under `<base>.{trellis,suh,svh}`: codewords [E, in/16, out/16, 64], suh and svh f16
/// in [0.5, 1.5] (svh over sqrt(in)).
fn putBank(w: *resident.Weights, rnd: std.Random, base: []const u8, in: c_int, out: c_int) !void {
    const n: c_int = 64;
    const words = try testing.allocator.alloc(u16, @intCast(t_experts * @divExact(in, 16) * @divExact(out, 16) * n));
    defer testing.allocator.free(words);
    rnd.bytes(std.mem.sliceAsBytes(words));
    var nb: [128]u8 = undefined;
    try w.put(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base}), mlx.mlx_array_new_data(words.ptr, &[_]c_int{ t_experts, @divExact(in, 16), @divExact(out, 16), n }, 4, .uint16));
    const sides = [_]struct { leaf: []const u8, dim: c_int, scale: f32 }{ .{ .leaf = "suh", .dim = in, .scale = 1 }, .{ .leaf = "svh", .dim = out, .scale = 1 / @sqrt(@as(f32, @floatFromInt(in))) } };
    for (sides) |side| {
        const h = try testing.allocator.alloc(u16, @intCast(t_experts * side.dim));
        defer testing.allocator.free(h);
        for (h) |*v| v.* = @bitCast(@as(f16, @floatCast((0.5 + rnd.float(f32)) * side.scale)));
        try w.put(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, side.leaf }), mlx.mlx_array_new_data(h.ptr, &[_]c_int{ t_experts, side.dim }, 2, .float16));
    }
}

/// A layer's three banks loaded through the in-tree EXL3 table, as the head's format banks are. Opened in place: the
/// load context points at `src`, which borrows `r`.
const TestBank = struct {
    r: resident.Resident = undefined,
    src: quant_abi.Source = undefined,
    ctx: quant_abi.LoadCtx = undefined,
    w: [3]quant_abi.Weight = @splat(.{}),

    fn open(b: *TestBank, rnd: std.Random, s: mlx.mlx_stream) !void {
        var w = resident.Weights.init(testing.allocator);
        const names = [_][]const u8{ "l.gate_proj", "l.up_proj", "l.down_proj" };
        const ins = [_]c_int{ t_hidden, t_hidden, t_inter };
        for (names, ins) |name, in| try putBank(&w, rnd, name, in, if (in == t_hidden) t_inter else t_hidden);
        b.r = resident.Resident.adopt(w);
        b.src = b.r.source();
        b.ctx = .{ .block_json = .of("{\"format\":\"exl3\",\"k\":4,\"codebook\":\"mul1\"}"), .source = &b.src, .stream = .{ .ctx = s.ctx } };
        try testing.expect(exl3.table.open(&b.ctx).isOk());
        for (names, ins, &b.w) |name, in, *x| try testing.expect(exl3.table.load_bank(&b.ctx, .of(name), @intCast(in), x).isOk());
    }

    fn deinit(b: *TestBank) void {
        for (&b.w) |*x| exl3.table.release(x);
        exl3.table.close(&b.ctx);
        b.r.deinit();
    }

    fn banks(b: *const TestBank) moe.FormatBanks {
        return .{ .table = &exl3.table, .gate = b.w[0], .up = b.w[1], .down = b.w[2] };
    }
};

/// Row `r` of `x` [1, rows, w]: [1, 1, w].
fn rowOf(s: mlx.mlx_stream, x: Array, r: c_int) !Array {
    const sh = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

/// The routes of `rows` rows over a shuffled bank, consecutive rows sharing four experts; weights bf16 in [0.05, 0.4].
fn routesOf(rnd: std.Random, s: mlx.mlx_stream, rows: c_int) ![2]Array {
    const e: usize = @intCast(t_experts);
    const k: usize = @intCast(t_topk);
    var perm: [e]i32 = undefined;
    for (&perm, 0..) |*p, i| p.* = @intCast(i);
    rnd.shuffle(i32, &perm);
    var ids: [16 * k]i32 = undefined;
    const n: usize = @intCast(rows);
    for (0..n) |r| for (0..k) |j| {
        ids[r * k + j] = perm[(4 * r + j) % e];
    };
    const inds = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, rows, t_topk }, 3, .int32);
    errdefer _ = mlx.mlx_array_free(inds);
    return .{ inds, try randBf16(rnd, &.{ 1, rows, t_topk }, 0.05, 0.4, s) };
}

// Bar: the head's format bank sums each row by itself up to the decode chain's rows, so the last row alone is exact.
test "glm5_next moe: the MTP bank's routed experts over one row give that row of the call over every row bit for bit at 2 to 16 rows" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(20261109);
    const rnd = prng.random();
    var bank: TestBank = .{};
    try bank.open(rnd, s);
    defer bank.deinit();
    const opts: quant_abi.Swiglu = .{ .limit = t_limit };
    const call = struct {
        fn run(b: *const TestBank, st: mlx.mlx_stream, o: *const quant_abi.Swiglu, x: Array, inds: Array, scores: Array) !Array {
            var y: quant_abi.Array = .{};
            const v = exl3.table.swiglu.?(&b.w[0], &b.w[1], &b.w[2], .{ .ctx = x.ctx }, .{ .ctx = inds.ctx }, .{ .ctx = scores.ctx }, o, .{ .ctx = st.ctx }, &y);
            if (!v.isOk()) return error.TestUnexpectedResult;
            return .{ .ctx = y.ctx };
        }
    }.run;
    for ([_]c_int{ 2, 3, 4, 5, 8, last_row_max_rows }) |rows| {
        const x = try randBf16(rnd, &.{ 1, rows, t_hidden }, -2, 2, s);
        defer _ = mlx.mlx_array_free(x);
        const rt = try routesOf(rnd, s, rows);
        defer for (rt) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const want = try call(&bank, s, &opts, x, rt[0], rt[1]);
        defer _ = mlx.mlx_array_free(want);
        var r: c_int = 0;
        while (r < rows) : (r += 1) {
            var parts: [4]Array = undefined;
            for (&parts, [_]Array{ x, rt[0], rt[1], want }) |*p, a| p.* = try rowOf(s, a, r);
            defer for (parts) |p| {
                _ = mlx.mlx_array_free(p);
            };
            const got = try call(&bank, s, &opts, parts[0], parts[1], parts[2]);
            defer _ = mlx.mlx_array_free(got);
            try ops.expectBitEqual(got, parts[3]);
        }
    }
}

// Bar: the head's last-row MoE is the last row of `forward`, bit for bit.
test "glm5_next moe: the last-row block equals forward's last row, bit for bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(20261110);
    const rnd = prng.random();
    var bank: TestBank = .{};
    try bank.open(rnd, s);
    defer bank.deinit();
    const shared_inter: c_int = 1024;
    var held: [6]Array = undefined;
    held[0] = try randBf16(rnd, &.{ t_hidden, t_experts }, -0.05, 0.05, s);
    var router = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&router, held[0], .float32, s));
    held[1] = router;
    held[2] = try randBf16(rnd, &.{ t_hidden, shared_inter }, -0.03, 0.03, s);
    held[3] = try randBf16(rnd, &.{ t_hidden, shared_inter }, -0.03, 0.03, s);
    held[4] = try randBf16(rnd, &.{ shared_inter, t_hidden }, -0.03, 0.03, s);
    var zeros = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{t_experts}, 1, .float32, s));
    held[5] = zeros;
    defer for (held) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const qp: @import("../../sdk/quant/params.zig").QuantParams = .{ .bits = 4, .group_size = 64 };
    const w: moe.Moe.Weights = .{
        .router = .{ .w = held[1] },
        .route = .{ .sigmoid_bias = .{ .expert_bias = held[5], .norm = true, .scale = 2.5 } },
        .gate = .{ .w = .{}, .qp = qp },
        .up = .{ .w = .{}, .qp = qp },
        .down = .{ .w = .{}, .qp = qp },
        .shared = .{ .gate = .{ .w = held[2] }, .up = .{ .w = held[3] }, .down = .{ .w = held[4] } },
        .top_k = t_topk,
        .experts = t_experts,
        .clamp = t_limit,
        .clamp_limit = @floatFromInt(t_limit),
        .format_banks = bank.banks(),
    };
    for ([_]c_int{ 1, 2, 3, 4, last_row_max_rows + 1 }) |rows| {
        const x = try randBf16(rnd, &.{ 1, rows, t_hidden }, -1, 1, s);
        defer _ = mlx.mlx_array_free(x);
        const want = try forward(&w, s, x);
        defer _ = mlx.mlx_array_free(want);
        // the last row alone: 2 rows to the decode chain's; one row and a wider call decline
        const got = try forwardLastRow(&w, s, x);
        if (rows < 2 or rows > last_row_max_rows) {
            try testing.expect(got == null);
            continue;
        }
        const last = got orelse return error.TestUnexpectedResult;
        defer _ = mlx.mlx_array_free(last);
        const want_last = try rowOf(s, want, rows - 1);
        defer _ = mlx.mlx_array_free(want_last);
        try ops.expectBitEqual(last, want_last);
    }
}
