//! `qwen_vision.zig`'s live parity test against a real checkpoint: it loads the pack through the engine, so it stays out
//! of the tower's import graph.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const qwen_vision = @import("qwen_vision.zig");
const QwenVision = qwen_vision.QwenVision;
const readBinF32 = qwen_vision.readBinF32;

// Live parity vs the mlx-vlm reference vision tower. Feeds the REFERENCE's own
// pixel_values straight into QwenVision (isolating the ViT math from the
// preprocessing), then compares post-merger embeddings. Build the fixture first:
//   python3 tests/build_qwen_vision_fixture.py --model <dir> --image <img> --out <fix>
// then run via tests/test_qwen_vision_parity.sh (sets the env vars below).
test "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" {
    const model_raw = env.get(.QWEN_VISION_TEST_MODEL) orelse return error.SkipZigTest;
    const fix_raw = env.get(.QWEN_VISION_FIXTURE) orelse return error.SkipZigTest;
    const gh_raw = env.get(.QV_GH) orelse return error.SkipZigTest;
    const gw_raw = env.get(.QV_GW) orelse return error.SkipZigTest;
    const dir = std.mem.sliceTo(model_raw, 0);
    const fix = std.mem.sliceTo(fix_raw, 0);
    if (dir.len == 0 or fix.len == 0) return error.SkipZigTest;
    const grid_h = try std.fmt.parseInt(u32, std.mem.sliceTo(gh_raw, 0), 10);
    const grid_w = try std.fmt.parseInt(u32, std.mem.sliceTo(gw_raw, 0), 10);

    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const model_mod = @import("../config/model.zig");
    const config = try model_mod.parseConfig(io, alloc, dir);
    try std.testing.expect(config.qwen_vision);
    var weights = try model_mod.loadWeightsWithVision(io, alloc, dir);
    defer weights.deinit();
    var qv = try QwenVision.init(alloc, .of(&config), &weights);
    defer qv.deinit();

    // Reference pixel_values [N, C*tps*ps*ps] and post-merger embeddings.
    const px_path = try std.fmt.allocPrint(alloc, "{s}/pixel_values.bin", .{fix});
    defer alloc.free(px_path);
    const ref_path = try std.fmt.allocPrint(alloc, "{s}/ref_embeds.bin", .{fix});
    defer alloc.free(ref_path);
    const px = try readBinF32(io, alloc, px_path);
    defer alloc.free(px);
    const ref = try readBinF32(io, alloc, ref_path);
    defer alloc.free(ref);

    const n = grid_h * grid_w;
    const feat = px.len / n;
    const px_shape = [_]c_int{ @intCast(n), @intCast(feat) };
    const px_arr = mlx.mlx_array_new_data(px.ptr, &px_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(px_arr);

    const out = try qv.forward(px_arr, grid_h, grid_w);
    defer _ = mlx.mlx_array_free(out);
    var out_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_f32);
    try mlx.check(mlx.mlx_astype(&out_f32, out, .float32, qv.s));
    try mlx.check(mlx.mlx_array_eval(out_f32));
    const od = mlx.mlx_array_data_float32(out_f32) orelse return error.TestUnexpectedNullData;

    try std.testing.expectEqual(ref.len, n / 4 * config.qv_out_hidden);
    const dim = config.qv_out_hidden;
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    var sum_abs: f64 = 0;
    var gt_010: usize = 0;
    var gt_030: usize = 0;
    // Per-token max diff to see clustering (a structural bug clusters by token).
    var worst_tok_diff: f64 = 0;
    var worst_tok: usize = 0;
    var tok_sum: f64 = 0;
    var cur_tok: usize = 0;
    for (0..ref.len) |i| {
        const d = @abs(od[i] - ref[i]);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
        if (d > 0.10) gt_010 += 1;
        if (d > 0.30) gt_030 += 1;
        sum_abs += d;
        const tok = i / dim;
        if (tok != cur_tok) {
            if (tok_sum > worst_tok_diff) {
                worst_tok_diff = tok_sum;
                worst_tok = cur_tok;
            }
            tok_sum = 0;
            cur_tok = tok;
        }
        tok_sum += d;
    }
    const mean_abs = sum_abs / @as(f64, @floatFromInt(ref.len));
    std.debug.print("\nqwen vision parity: max_abs={d:.5} mean_abs={d:.6} (N_merged={d}, dim={d})\n", .{ max_abs, mean_abs, n / 4, dim });
    std.debug.print("  max@ token={d} chan={d} ours={d:.4} ref={d:.4}\n", .{ max_idx / dim, max_idx % dim, od[max_idx], ref[max_idx] });
    std.debug.print("  diffs>0.10: {d}/{d}  diffs>0.30: {d}  worst-token meanrow-diff token={d} sum={d:.3}\n", .{ gt_010, ref.len, gt_030, worst_tok, worst_tok_diff / @as(f64, @floatFromInt(dim)) });
    // bf16 ViT through 12 blocks with different reduction orders than the
    // reference: tolerate accumulated rounding + a handful of outliers, catch
    // real bugs (a structural bug blows up mean_abs by 10-100x).
    try std.testing.expect(mean_abs < 0.02);
    try std.testing.expect(gt_030 < ref.len / 1000); // <0.1% of elements may exceed 0.30
}
