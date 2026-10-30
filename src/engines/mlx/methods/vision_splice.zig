//! The legacy `Transformer`'s methods over `sdk/vision/splice.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;
const spliceVisionRows = @import("../../../sdk/vision/splice.zig").spliceVisionRows;

/// Splice vision embeddings into text embeddings at image_token_id positions.
/// h: [B, seq_len, hidden] text embeddings
/// token_ids: [B, seq_len] original token IDs
/// vision_emb: [B, N_img, hidden] vision embeddings
/// Returns new h with vision embeddings replacing image token positions.
/// masked_scatter: replaces image token positions with vision features.
/// Matches Python reference: cumsum-based indexing into flattened source.
pub fn spliceVisionEmbeddings(self: *Transformer, h: mlx.mlx_array, token_ids: mlx.mlx_array, vision_emb: mlx.mlx_array, image_token_id: u32, audio_token_id: u32, video_token_id: u32, row_offset: usize) !mlx.mlx_array {
    return spliceVisionRows(h, token_ids, vision_emb, image_token_id, audio_token_id, video_token_id, row_offset, self.s);
}

/// Apply vision embeddings to text embeddings during prefill.
/// Handles scaling and splicing at image_token_id positions.
/// Returns the (potentially modified) h; caller should replace their h with the result.
pub fn applyVisionEmbeddingsWith(self: *Transformer, ctx: *ForwardCtx, h: mlx.mlx_array, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const cfg = &self.config;
    const h_shape = mlx.getShape(h);
    // Only during prefill (seq_len > 1)
    if (h_shape[1] <= 1) return h;
    const ve = ctx.vision_embeddings orelse return h;
    if (cfg.image_token_id == 0 and cfg.audio_token_id == 0 and cfg.video_token_id == 0) return h;

    // Vision embeddings come out of the MultimodalEmbedder already in text-hidden space;
    // mlx-vlm does NOT re-scale them by sqrt(hidden) the way text embeddings are scaled
    // at LM embedding time. Splice directly — scaling here corrupts the MoE router's
    // magnitude assumptions (visible as "please provide an image" responses on 26B MoE).
    return self.spliceVisionEmbeddings(h, token_ids, ve, cfg.image_token_id, cfg.audio_token_id, cfg.video_token_id, ctx.vision_splice_offset);
}
