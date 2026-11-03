//! sdk/blocks/embedding.
//! Owns gatherQuantizedRows.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const mlx_gguf = @import("../quant/gguf/root.zig");
const QuantMode = @import("../quant/params.zig").QuantMode;
const testing = std.testing;
const maxAbsDiffF32 = @import("../testing/parity.zig").maxAbsDiffF32;

/// Gather rows out of a QUANTIZED table and dequantize only those rows.
///
/// The naive shape — dequantize the whole table, then `take` — reads the packed
/// weights AND writes a full float copy of the table on every call. For a
/// 157184x2560 8-bit embedding that is ~0.4 GB in + ~0.8 GB out to look up a
/// handful of rows, i.e. ~15% of a decode token on a 157k-vocab model spent
/// producing 157183 rows nobody reads.
///
/// Quantization here is per-ROW-group along the last axis, so a row's scales
/// and biases are exactly the ones at the same index — gathering first is
/// arithmetically identical, not an approximation, and the guard test pins that
/// as BIT equality. `ids` may have any shape; the result is
/// `ids.shape ++ [in_dim]` (mlx `take` semantics on axis 0).
///
/// `global_scale` is per-TENSOR (mxfp/nvfp4 families), so it rides through
/// unchanged; `biases` may be a null handle for bias-less modes.
pub fn gatherQuantizedRows(
    s: mlx.mlx_stream,
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    ids: mlx.mlx_array,
    group_size: u32,
    bits: u32,
    mode: QuantMode,
    global_scale: mlx.mlx_array,
    out_dtype: mlx.mlx_optional_dtype,
) !mlx.mlx_array {
    var taken_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken_w);
    try mlx.check(mlx.mlx_take_axis(&taken_w, w, ids, 0, s));

    var taken_s = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken_s);
    try mlx.check(mlx.mlx_take_axis(&taken_s, scales, ids, 0, s));

    // `mlx_take_axis` on a null handle aborts, so bias-less modes gather none.
    var taken_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken_b);
    if (biases.ctx != null) {
        try mlx.check(mlx.mlx_take_axis(&taken_b, biases, ids, 0, s));
    }

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_dequantize(
        &out,
        taken_w,
        taken_s,
        taken_b,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        mode.cstr(),
        global_scale,
        out_dtype,
        s,
    ));
    return out;
}

// ── tests ──

test "gatherQuantizedRows is BIT-identical to dequantizing the whole table and taking rows" {
    // The whole point of the helper is that it is not an approximation:
    // quantization groups run along the LAST axis, so a row's scales/biases are
    // exactly the ones at the same index and gathering first cannot change a
    // single bit. Anything weaker than bit equality would leave room for a
    // "close enough" lookup table, which is how a checkpoint's embedding drifts
    // from its lm_head.
    const s = mlx.gpuStream();
    const V = 96;
    const IN = 256;
    const gs = 64;
    const bits = 8;

    const host = try std.testing.allocator.alloc(f32, V * IN);
    defer std.testing.allocator.free(host);
    for (host, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 251)) - 125)) * 0.013;
    const sh = [_]c_int{ V, IN };
    const w_f32 = mlx.mlx_array_new_data(host.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(w_f32);
    var w_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf);
    try mlx.check(mlx.mlx_astype(&w_bf, w_f32, .bfloat16, s));

    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&q, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));

    // A 2-D id array, because that is the live shape ([batch, seq]) and the
    // result must come back as ids.shape ++ [IN], not flattened.
    const ids_host = [_]i32{ 0, 7, 95, 42, 42, 13 };
    const ids_sh = [_]c_int{ 2, 3 };
    const ids = mlx.mlx_array_new_data(&ids_host, &ids_sh, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    var ids32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids32);
    try mlx.check(mlx.mlx_astype(&ids32, ids, .uint32, s));

    const got = try gatherQuantizedRows(s, q, sc, bi, ids32, gs, bits, .affine, .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true });
    defer _ = mlx.mlx_array_free(got);

    // Reference: the shape the helper replaces — dequantize everything, take.
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_dequantize(&deq, q, sc, bi, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_take_axis(&want, deq, ids32, 0, s));

    const shape_got = mlx.getShape(got);
    try testing.expectEqual(@as(usize, 3), shape_got.len);
    try testing.expectEqual(@as(c_int, 2), shape_got[0]);
    try testing.expectEqual(@as(c_int, 3), shape_got[1]);
    try testing.expectEqual(@as(c_int, IN), shape_got[2]);

    const n = mlx.mlx_array_size(want);
    const flat = [_]c_int{@intCast(n)};
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_reshape(&gf, got, &flat, 1, s));
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_reshape(&wf, want, &flat, 1, s));
    try testing.expectEqual(@as(f32, 0.0), try maxAbsDiffF32(gf, wf, s));
}
