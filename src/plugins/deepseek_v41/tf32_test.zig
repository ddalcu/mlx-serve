//! The served f32 GEMMs under MLX's TF32 switch (`MLX_ENABLE_TF32`, on by default on NAX: an f32 GEMM's operands
//! rounded to a 10-bit mantissa where M and N exceed 1). `DSV41_TF32_FIXTURE` names a directory of operands saved
//! from a served run, `<site>-<i>.safetensors` (`a`, `b` as the matmul reads them). Each one runs on this process's GPU
//! path against float64. The trunk's sites must read operands TF32 leaves unchanged (bf16 or f16 data held in f32);
//! the DSpark drafter's attention (draft-only) is measured, not held to that.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const device = @import("../../mlx/device.zig");
const resident = @import("../../sdk/weights/resident.zig");

const A = mlx.mlx_array;

const Site = struct { name: []const u8, exact: bool };
/// `exact`: a trunk site, whose operands must be TF32-exact.
const sites = [_]Site{
    .{ .name = "router", .exact = true },
    .{ .name = "lin_f32", .exact = true },
    .{ .name = "hc_collapse", .exact = false },
    .{ .name = "dspark_qk", .exact = false },
    .{ .name = "dspark_pv", .exact = false },
};

extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;

fn prod(sh: []const c_int) usize {
    var p: usize = 1;
    for (sh) |d| p *= @intCast(d);
    return p;
}

/// `x` contiguous, in f32 on the host (`gpa` owns the slice).
fn hostOf(gpa: std.mem.Allocator, x: A, s: mlx.mlx_stream) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return gpa.dupe(f32, d[0..mlx.mlx_array_size(f)]);
}

/// f32 values whose low 13 mantissa bits are set: the ones TF32 rounds.
fn tf32Rounded(v: []const f32) usize {
    var n: usize = 0;
    for (v) |x| {
        if (@as(u32, @bitCast(x)) & 0x1fff != 0) n += 1;
    }
    return n;
}

const Err = struct { rel_l2: f64, max_abs: f64, rounded_a: usize, rounded_b: usize };

/// `a @ b` on this process's GPU path against float64 over the same operands, rows sampled to about 6e7 multiply-adds.
fn measure(gpa: std.mem.Allocator, file: []const u8, a: A, b: A, s: mlx.mlx_stream) !Err {
    const sa = mlx.getShape(a);
    const sb = mlx.getShape(b);
    const M: usize = @intCast(sa[sa.len - 2]);
    const K: usize = @intCast(sa[sa.len - 1]);
    const N: usize = @intCast(sb[sb.len - 1]);
    const Ba = prod(sa[0 .. sa.len - 2]);
    const Bb = prod(sb[0 .. sb.len - 2]);
    if (Bb != 1 and Bb != Ba) return error.Tf32FixtureBatch;
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, a, b, s));
    const hy = try hostOf(gpa, y, s);
    defer gpa.free(hy);
    const ha = try hostOf(gpa, a, s);
    defer gpa.free(ha);
    const hb = try hostOf(gpa, b, s);
    defer gpa.free(hb);
    // b transposed per batch, [Bb][N][K], for row-major dots.
    const bt = try gpa.alloc(f32, hb.len);
    defer gpa.free(bt);
    for (0..Bb) |bi| {
        for (0..K) |kk| {
            for (0..N) |j| bt[(bi * N + j) * K + kk] = hb[(bi * K + kk) * N + j];
        }
    }
    const rows = Ba * M;
    const budget: usize = 60_000_000;
    const step = @max(1, (rows * N * K + budget - 1) / budget);
    var se: f64 = 0;
    var sr: f64 = 0;
    var max_abs: f64 = 0;
    var peak: f64 = 0;
    var done: usize = 0;
    var r: usize = 0;
    while (r < rows) : (r += step) {
        const arow = ha[r * K ..][0..K];
        const bb = if (Bb == 1) 0 else r / M;
        for (0..N) |j| {
            const col = bt[(bb * N + j) * K ..][0..K];
            var acc: f64 = 0;
            for (arow, col) |x, w| acc += @as(f64, x) * @as(f64, w);
            const d = @as(f64, hy[r * N + j]) - acc;
            se += d * d;
            sr += acc * acc;
            max_abs = @max(max_abs, @abs(d));
            peak = @max(peak, @abs(acc));
        }
        done += 1;
    }
    const e: Err = .{ .rel_l2 = if (sr > 0) @sqrt(se / sr) else 0, .max_abs = max_abs, .rounded_a = tf32Rounded(ha), .rounded_b = tf32Rounded(hb) };
    const path = if (M <= 1 or N <= 1) "gemv" else if (device.tf32Matmul()) "TF32 on" else "TF32 off";
    std.debug.print("[tf32] {s}: M {d} N {d} K {d} batch {d}, {s}: rows {d} of {d}: rel l2 {e:.3}, max abs {e:.3}, max abs / peak {e:.3}; TF32 rounds a {d} of {d}, b {d} of {d}\n", .{
        file, M, N, K, Ba, path, done, rows, e.rel_l2, max_abs, if (peak > 0) max_abs / peak else 0, e.rounded_a, ha.len, e.rounded_b, hb.len,
    });
    return e;
}

test "dsv41 tf32: the served f32 GEMMs against float64; the trunk's read TF32-exact operands (DSV41_TF32_FIXTURE)" {
    const dir = env.span(.DSV41_TF32_FIXTURE) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const s = mlx.gpuStream();
    var seen: usize = 0;
    for (sites) |site| {
        for (0..3) |i| {
            const file = try std.fmt.allocPrint(gpa, "{s}-{d}.safetensors", .{ site.name, i });
            defer gpa.free(file);
            const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, file }, 0);
            defer gpa.free(path);
            if (access(path.ptr, 0) != 0) continue;
            var w = try resident.loadWeightsFile(gpa, dir, file);
            defer w.deinit();
            const e = try measure(gpa, file, w.get("a") orelse return error.Tf32FixtureNoA, w.get("b") orelse return error.Tf32FixtureNoB, s);
            if (site.exact) {
                try std.testing.expectEqual(@as(usize, 0), e.rounded_a);
                try std.testing.expectEqual(@as(usize, 0), e.rounded_b);
                try std.testing.expect(e.rel_l2 < 1e-5);
            }
            seen += 1;
        }
    }
    if (seen == 0) return error.SkipZigTest;
}
