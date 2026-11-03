//! The legacy `Transformer`'s methods over `sdk/blocks/embedding.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const mlx_gguf = @import("../../../sdk/quant/gguf/root.zig");
const Transformer = @import("../transformer.zig").Transformer;

/// Raw embedding-table lookup: gather + dequant to bf16 ONLY — no
/// normed-embeddings RMS norm, no emb_scale. The DFlash noise embeds are
/// contractually the raw table rows (the reference calls `F.embedding`
/// on the weight directly — the un-foldable embed norm is exactly what
/// makes this hook possible); the trunk's own forward goes through
/// `embedding`, which layers the norm/scale on top of this. Returns
/// `[B, L, hidden]` bf16, caller frees.
pub fn rawEmbedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
    if (self.emb_w.ctx == null) return error.MissingWeight;
    const id_shape = mlx.getShape(token_ids);
    const batch = id_shape[0];
    const seq_len = id_shape[1];

    const flat_shape = [_]c_int{batch * seq_len};
    var flat_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_ids);
    try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

    var taken_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken_w);
    try mlx.check(mlx.mlx_take_axis(&taken_w, self.emb_w, flat_ids, 0, self.s));

    var emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb);
    if (self.emb_s.ctx == null) {
        // Dense bf16 embedding table: the gathered rows ARE the embeddings.
        try mlx.check(mlx.mlx_astype(&emb, taken_w, .bfloat16, self.s));
    } else if (mlx_gguf.kernels.infoOf(self.emb_w, self.emb_s)) |info| {
        emb = try mlx_gguf.kernels.dequantize(info.ty, taken_w, self.actDtype(), self.s);
    } else {
        var taken_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_s);
        try mlx.check(mlx.mlx_take_axis(&taken_s, self.emb_s, flat_ids, 0, self.s));
        // Bias-less modes (nvfp4/mxfp4/mxfp8) have a null-ctx emb_b —
        // mlx_take_axis on a null handle aborts, so gather only when present.
        var taken_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_b);
        if (self.emb_b.ctx != null) {
            try mlx.check(mlx.mlx_take_axis(&taken_b, self.emb_b, flat_ids, 0, self.s));
        }
        const emb_qp = self.quantParamsHinted(self.emb_w, self.emb_s, self.config.hidden_size);
        try mlx.check(mlx.mlx_dequantize(
            &emb,
            taken_w,
            taken_s,
            taken_b,
            mlx.mlx_optional_int.some(@intCast(emb_qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(emb_qp.bits)),
            emb_qp.mode.cstr(),
            .{}, // global_scale (null)
            .{ .value = self.actDtype(), .has_value = true },
            self.s,
        ));
    }

    const out_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
    var reshaped = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&reshaped, emb, &out_shape, 3, self.s));
    return self.unrotateEmbedding(reshaped);
}

pub fn embedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const raw = try self.rawEmbedding(token_ids);

    if (self.emb_scale) |scale| {
        defer _ = mlx.mlx_array_free(raw);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, raw, scale, self.s));
        return scaled;
    }
    return raw;
}

pub fn dequantTake(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array) !mlx.mlx_array {
    var tw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tw);
    try mlx.check(mlx.mlx_take_axis(&tw, w, ids, 0, self.s));
    if (sc.ctx == null) {
        // Dense bf16: gathered rows are the embeddings; no dequantize.
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&result, tw, .bfloat16, self.s));
        return result;
    }
    if (mlx_gguf.kernels.infoOf(w, sc)) |info| return mlx_gguf.kernels.dequantize(info.ty, tw, .bfloat16, self.s);
    var ts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ts);
    try mlx.check(mlx.mlx_take_axis(&ts, sc, ids, 0, self.s));
    // Bias-less modes ship a null-ctx bi — gather only when present.
    var tb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tb);
    if (bi.ctx != null) {
        try mlx.check(mlx.mlx_take_axis(&tb, bi, ids, 0, self.s));
    }
    var result = mlx.mlx_array_new();
    const qp = self.quantParamsFor(w, sc);
    try mlx.check(mlx.mlx_dequantize(
        &result,
        tw,
        ts,
        tb,
        mlx.mlx_optional_int.some(@intCast(qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(qp.bits)),
        qp.mode.cstr(),
        .{}, // global_scale (null)
        .{ .value = .bfloat16, .has_value = true },
        self.s,
    ));
    return result;
}
