//! `MLX_SERVE_DSV41_FP8_FOLD` (`decode.fp8Fold`) and `MLX_SERVE_DSV41_FP4_READS` (`moe_fp4.fp4Reads`) read three mxfp8
//! codes apart from MLX: e8m0 0xFF as +Inf, e8m0 247..254 as an overflow, e4m3 S.1111.111 (a NaN) as 480. Each serves
//! only where the load found none of them in a tensor it reads (`settle`): FP8_FOLD every mxfp8 linear of a layer,
//! FP4_READS each layer's shared down. Else it declines by name and MLX's reads serve.

const std = @import("std");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

/// The levers `settle` left serving: the last load's verdict (one model loads at a time), one copy across `-Dunits`.
const State = struct { fold: bool = true, reads: bool = true };
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_fp8_codes", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

/// A code an mxfp8 linear holds that the levers read apart from MLX.
pub const Found = enum {
    none,
    e8m0_high,
    e4m3_nan,

    fn what(f: Found) []const u8 {
        return switch (f) {
            .none => "no such code",
            .e8m0_high => "an e8m0 scale code of 247 or more",
            .e4m3_nan => "an e4m3 NaN code (S.1111.111)",
        };
    }
};

/// A layer's linears by their pack names; `down` the shared down FP4_READS reads.
const Item = struct { name: []const u8, lin: ?*const dsv41.Lin, down: bool = false };
const max_items = 11;

fn items(ly: *const dsv41.Layer) [max_items]Item {
    return .{
        .{ .name = "attn.wq_a", .lin = &ly.wq_a },
        .{ .name = "attn.wkv", .lin = &ly.wkv },
        .{ .name = "attn.wq_a+wkv", .lin = if (ly.q_kv) |*q| q else null },
        .{ .name = "attn.wq_b", .lin = &ly.wq_b },
        .{ .name = "attn.indexer.wq_b", .lin = if (ly.idx) |*x| &x.wq_b else null },
        .{ .name = "attn.wo_a", .lin = &ly.wo_a },
        .{ .name = "attn.wo_b", .lin = &ly.wo_b },
        .{ .name = "ffn.shared_experts.w1", .lin = &ly.shared[0] },
        .{ .name = "ffn.shared_experts.w3", .lin = &ly.shared[1] },
        .{ .name = "ffn.shared_experts.w1+w3", .lin = if (ly.shared_gu) |*g| g else null },
        .{ .name = "ffn.shared_experts.w2", .lin = &ly.shared[2], .down = true },
    };
}

/// An mxfp8 linear in MLX's arrays: e4m3 codes as u32 words beside u8 e8m0 scales.
fn mxfp8(l: *const dsv41.Lin) bool {
    if (l.ext != null or l.w.ctx == null or l.s.ctx == null or l.qp.mode != .mxfp8) return false;
    return mlx.mlx_array_dtype(l.w) == .uint32 and mlx.mlx_array_dtype(l.s) == .uint8;
}

/// What each of `lins` holds, in one evaluation: the largest scale code and the largest code's low seven bits.
fn scan(gpa: std.mem.Allocator, s: mlx.mlx_stream, lins: []const *const dsv41.Lin, out: []Found) !void {
    var sc = dsv41.Scope.init(gpa, s);
    defer sc.deinit();
    const low7: u8 = 0x7F;
    const m7 = try sc.keep(mlx.mlx_array_new_data(&low7, &[_]c_int{}, 0, .uint8));
    var maxes: [2 * max_items]A = undefined;
    for (lins, 0..) |l, k| {
        var o = mlx.mlx_array_new();
        maxes[2 * k] = try sc.res(mlx.mlx_max(&o, l.s, false, s), &o);
        const bits = try sc.op2(mlx.mlx_bitwise_and, try sc.view(l.w, .uint8), m7);
        var p = mlx.mlx_array_new();
        maxes[2 * k + 1] = try sc.res(mlx.mlx_max(&p, bits, false, s), &p);
    }
    const v = mlx.mlx_vector_array_new_data(&maxes, 2 * lins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
    for (out, 0..) |*f, k| {
        const smax = (mlx.mlx_array_data_uint8(maxes[2 * k]) orelse return error.NoData)[0];
        const wmax = (mlx.mlx_array_data_uint8(maxes[2 * k + 1]) orelse return error.NoData)[0];
        f.* = if (smax >= 247) .e8m0_high else if (wmax == 0x7F) .e4m3_nan else .none;
    }
}

/// At load, once the weights are evaluated: every layer's mxfp8 linears scanned a layer at a time, and each lever
/// that reads a tensor holding one of the codes declined, with one line naming it and the tensor.
pub fn settle(m: *const dsv41.Dsv41Model) !void {
    const st = ps();
    st.* = .{};
    const fold = lev.onUnlessZero(.FP8_FOLD);
    const reads = lev.onUnlessZero(.FP4_READS);
    if (!fold and !reads) return;
    for (m.layers, 0..) |*ly, li| {
        if (!(fold and st.fold) and !(reads and st.reads)) return;
        var lins: [max_items]*const dsv41.Lin = undefined;
        var held: [max_items]Item = undefined;
        var n: usize = 0;
        const all = items(ly);
        for (all) |it| if (it.lin) |l| if (mxfp8(l)) {
            lins[n] = l;
            held[n] = it;
            n += 1;
        };
        if (n == 0) continue;
        var found: [max_items]Found = undefined;
        try scan(m.gpa, m.s, lins[0..n], found[0..n]);
        const stage = li >= m.n_layers;
        const pfx: []const u8 = if (stage) "mtp" else "layers";
        const at = if (stage) li - m.n_layers else li;
        for (held[0..n], found[0..n]) |it, f| if (f != .none) {
            if (fold and st.fold) {
                st.fold = false;
                log.info("deepseek_v41: MLX_SERVE_DSV41_FP8_FOLD declined: {s}.{d}.{s} holds {s}, which the fold reads apart from MLX; MLX's mxfp8 reads serve\n", .{ pfx, at, it.name, f.what() });
            }
            if (reads and it.down and st.reads) {
                st.reads = false;
                log.info("deepseek_v41: MLX_SERVE_DSV41_FP4_READS declined: {s}.{d}.{s} holds {s}, which its reads take apart from MLX; MLX's reads serve\n", .{ pfx, at, it.name, f.what() });
            }
        };
    }
}

// ── tests ──

const testing = std.testing;
const fixture = @import("fixture.zig");

fn gpuOk() bool {
    return mlx.metalKernelsAvailable() and !mlx.noGpuBackend() and !mlx.virtualGpu();
}

test "dsv41 fp8 codes: the scan finds e8m0 0xFF and 247 (not 246) and e4m3 0x7F and 0xFF (not 0x7E or 0xFE), one linear apart from the next" {
    if (!gpuOk()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    const Case = struct { s: u8, w: u8, want: Found };
    const cases = [_]Case{
        .{ .s = 127, .w = 0x38, .want = .none },
        .{ .s = 246, .w = 0x7E, .want = .none },
        .{ .s = 127, .w = 0xFE, .want = .none },
        .{ .s = 247, .w = 0x38, .want = .e8m0_high },
        .{ .s = 0xFF, .w = 0x38, .want = .e8m0_high },
        .{ .s = 127, .w = 0x7F, .want = .e4m3_nan },
        .{ .s = 127, .w = 0xFF, .want = .e4m3_nan },
    };
    var lins: [cases.len]dsv41.Lin = undefined;
    var ptrs: [cases.len]*const dsv41.Lin = undefined;
    var arrs: [2 * cases.len]A = undefined;
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (cases, 0..) |c, k| {
        // A [4, 64] linear: 64 words of filler codes, 8 scales, one code planted in its last row.
        var words: [4 * 16]u32 = @splat(0x30303030);
        var sc: [4 * 2]u8 = @splat(127);
        words[4 * 16 - 1] = (0x30303030 & ~@as(u32, 0xFF)) | c.w;
        sc[4 * 2 - 1] = c.s;
        arrs[2 * k] = mlx.mlx_array_new_data(&words, &[_]c_int{ 4, 16 }, 2, .uint32);
        arrs[2 * k + 1] = mlx.mlx_array_new_data(&sc, &[_]c_int{ 4, 2 }, 2, .uint8);
        lins[k] = .{ .w = arrs[2 * k], .s = arrs[2 * k + 1], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
        ptrs[k] = &lins[k];
    }
    var got: [cases.len]Found = undefined;
    try scan(gpa, s, &ptrs, &got);
    for (cases, got) |c, g| try testing.expectEqual(c.want, g);
}

const Arm = struct { bytes: []f32, verdict: State };

/// Layer 0 with `plant` (`fixture.Opts.plant`) after one decode step: its output and the load's verdict, under the
/// switches `off` turns off; `lift` sets the verdict aside before the step (the levers as they read before the scan).
fn stepArm(gpa: std.mem.Allocator, plant: []const u8, off: struct { fold: bool = false, reads: bool = false }, lift: bool) !Arm {
    const o1 = env.override(.MLX_SERVE_DSV41_FP8_FOLD, if (off.fold) "0" else null);
    defer o1.restore();
    const o2 = env.override(.MLX_SERVE_DSV41_FP4_READS, if (off.reads) "0" else null);
    defer o2.restore();
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .experts = 8, .experts_nvfp4 = true, .fused = true, .plant = plant });
    defer fx.deinit();
    const verdict = ps().*;
    if (lift) ps().* = .{};
    const x = try fx.stream(1);
    defer _ = mlx.mlx_array_free(x);
    const pre = try fx.cpre(1);
    defer _ = mlx.mlx_array_free(pre);
    const out = try dsv41.workerStep(fx.m, gpa, x, pre, 8, false, false);
    defer for (out) |o| {
        _ = mlx.mlx_array_free(o);
    };
    return .{ .bytes = try dsv41.hostF32(out[0]), .verdict = verdict };
}

test "dsv41 fp8 codes: a zero 32-group under e8m0 code 250 (MLX reads 0, the levers 0 x +Inf) declines FP8_FOLD in wo_b and both levers in the shared down; a decode step is then the bytes of the declined switches off, and with the decline lifted it differs" {
    if (!gpuOk()) return error.SkipZigTest;
    const gpa = testing.allocator;
    defer ps().* = .{};
    const Case = struct { plant: []const u8, want: State };
    for ([_]Case{
        .{ .plant = "layers.0.attn.wo_b", .want = .{ .fold = false, .reads = true } },
        .{ .plant = "layers.0.ffn.shared_experts.w2", .want = .{ .fold = false, .reads = false } },
    }) |c| {
        const mlx_reads = try stepArm(gpa, c.plant, .{ .fold = !c.want.fold, .reads = !c.want.reads }, false);
        defer gpa.free(mlx_reads.bytes);
        const got = try stepArm(gpa, c.plant, .{}, false);
        defer gpa.free(got.bytes);
        const lifted = try stepArm(gpa, c.plant, .{}, true);
        defer gpa.free(lifted.bytes);
        std.debug.print("[fp8 codes] {s}: verdict fold={} reads={}, lifted non-finite {d}\n", .{ c.plant, got.verdict.fold, got.verdict.reads, nonFinite(lifted.bytes) });
        try testing.expectEqual(c.want, got.verdict);
        try testing.expectEqual(@as(usize, 0), nonFinite(mlx_reads.bytes));
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx_reads.bytes), std.mem.sliceAsBytes(got.bytes));
        try testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(mlx_reads.bytes), std.mem.sliceAsBytes(lifted.bytes)));
    }
}

fn nonFinite(xs: []const f32) usize {
    var n: usize = 0;
    for (xs) |v| n += @intFromBool(!std.math.isFinite(v));
    return n;
}
