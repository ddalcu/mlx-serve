//! sdk/vision/splice.
//! Owns spliceVisionRows.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const testing = std.testing;

/// Scatter vision/audio embedding rows into `h` at placeholder-token
/// positions. `row_offset` is the number of placeholder tokens earlier
/// chunks of this prefill already consumed (chunked vision prefill): the
/// row lookup is a cumsum over THIS forward's mask, so the offset is what
/// keeps a chunk boundary inside an image from restarting the scatter at
/// row 0. Frees `h` and returns the spliced replacement.
pub fn spliceVisionRows(h: mlx.mlx_array, token_ids: mlx.mlx_array, vision_emb: mlx.mlx_array, image_token_id: u32, audio_token_id: u32, video_token_id: u32, row_offset: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    return spliceRows(h, token_ids, vision_emb, &.{ image_token_id, audio_token_id, video_token_id }, row_offset, s);
}

/// `spliceVisionRows` over any set of placeholder ids (0 = unset).
pub fn spliceRows(h: mlx.mlx_array, token_ids: mlx.mlx_array, vision_emb: mlx.mlx_array, placeholder_ids: []const u32, row_offset: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const h_shape = mlx.getShape(h);

    // mask = token_ids is any placeholder id. Every modality splices through
    // this one channel: the embedding tensor concatenates each item's rows
    // in prompt order, so a single sequence-order scatter lands each row in
    // its slot. An id of 0 is unset (an audio-only model has no image
    // token), never token 0.
    var mask_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_2d);
    const ids_shape = mlx.getShape(token_ids);
    try mlx.check(mlx.mlx_zeros(&mask_2d, ids_shape.ptr, ids_shape.len, .bool_, s));
    for (placeholder_ids) |extra_id| {
        if (extra_id == 0) continue;
        const extra_arr = mlx.mlx_array_new_int(@intCast(extra_id));
        defer _ = mlx.mlx_array_free(extra_arr);
        var extra_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(extra_mask);
        try mlx.check(mlx.mlx_equal(&extra_mask, token_ids, extra_arr, s));
        var combined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_logical_or(&combined, mask_2d, extra_mask, s));
        _ = mlx.mlx_array_free(mask_2d);
        mask_2d = combined;
    }

    // Expand mask to [B, seq_len, hidden] via broadcast
    const expand_shape = [_]c_int{ h_shape[0], h_shape[1], 1 };
    var mask_3d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_3d);
    try mlx.check(mlx.mlx_reshape(&mask_3d, mask_2d, &expand_shape, 3, s));

    // Broadcast to full shape via logical and with ones
    var mask_expanded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_expanded);
    {
        var ones_h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones_h);
        try mlx.check(mlx.mlx_ones(&ones_h, h_shape.ptr, 3, .bool_, s));
        try mlx.check(mlx.mlx_multiply(&mask_expanded, mask_3d, ones_h, s));
    }

    // Flatten everything
    const total = h_shape[0] * h_shape[1] * h_shape[2];
    const flat_shape = [_]c_int{total};

    var mask_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_flat);
    try mlx.check(mlx.mlx_reshape(&mask_flat, mask_expanded, &flat_shape, 1, s));

    // mask_int = mask_flat.astype(int32)
    var mask_int = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_int);
    try mlx.check(mlx.mlx_astype(&mask_int, mask_flat, .int32, s));

    // indices = cumsum(mask_int, axis=0) - 1 + row_offset*hidden. The
    // cumsum counts masked ELEMENTS of this forward (hidden consecutive
    // per placeholder token); earlier chunks consumed row_offset rows of
    // the flat source, i.e. row_offset*hidden elements.
    var cumsum_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cumsum_arr);
    try mlx.check(mlx.mlx_cumsum(&cumsum_arr, mask_int, 0, false, true, s));
    const elem_offset: i64 = @as(i64, @intCast(row_offset)) * @as(i64, @intCast(h_shape[2]));
    const off_i = mlx.mlx_array_new_int(@intCast(elem_offset - 1));
    defer _ = mlx.mlx_array_free(off_i);
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_add(&indices, cumsum_arr, off_i, s));

    // source = vision_emb.flatten()
    const ve_shape = mlx.getShape(vision_emb);
    const source_size = ve_shape[0] * ve_shape[1] * ve_shape[2];
    const source_shape = [_]c_int{source_size};
    var source_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(source_flat);
    try mlx.check(mlx.mlx_reshape(&source_flat, vision_emb, &source_shape, 1, s));

    // indices_mod = indices % source_size
    const source_size_arr = mlx.mlx_array_new_int(source_size);
    defer _ = mlx.mlx_array_free(source_size_arr);
    var indices_mod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices_mod);
    try mlx.check(mlx.mlx_remainder(&indices_mod, indices, source_size_arr, s));

    // aligned = source[indices_mod]
    var aligned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aligned);
    try mlx.check(mlx.mlx_take(&aligned, source_flat, indices_mod, s));

    // Cast aligned to h's dtype: a mismatch promotes the whole residual (f32 on a Prism pack's f16 stream).
    var aligned_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(aligned_bf);
    try mlx.check(mlx.mlx_astype(&aligned_bf, aligned, mlx.mlx_array_dtype(h), s));

    // input_flat = h.flatten()
    var input_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(input_flat);
    try mlx.check(mlx.mlx_reshape(&input_flat, h, &flat_shape, 1, s));

    // result_flat = where(mask_flat, aligned_bf, input_flat)
    var result_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(result_flat);
    try mlx.check(mlx.mlx_where(&result_flat, mask_flat, aligned_bf, input_flat, s));

    // Reshape back to [B, seq_len, hidden]
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&result, result_flat, h_shape.ptr, 3, s));
    _ = mlx.mlx_array_free(h);
    return result;
}
