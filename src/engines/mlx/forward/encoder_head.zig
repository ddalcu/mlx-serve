//! engines/mlx/forward/encoder_head.
//! Owns forwardEmbedding, forwardEmbeddingMasked, hasEmbedProjection, embedProjection.

const mlx = @import("../../../mlx/ffi.zig");
const Transformer = @import("../transformer.zig").Transformer;
const plugin_driver = @import("../plugin/driver.zig");

/// Forward pass that returns hidden states (after final_norm, before lm_head).
/// Output shape: [1, seq_len, hidden_size]. Caller must free.
pub fn forwardEmbedding(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
    return self.forwardEmbeddingMasked(token_ids, null, null);
}

/// Embedding forward for a padded [B, T] batch: `key_pad_mask` (additive,
/// [B, 1, 1, T]) keeps padded positions out of encoder attention. Null
/// mask == plain forwardEmbedding. `vision` is one prompt's image/video
/// soft-token rows, spliced in at the placeholder ids (EmbeddingGemma 2).
pub fn forwardEmbeddingMasked(self: *Transformer, token_ids: mlx.mlx_array, key_pad_mask: ?mlx.mlx_array, vision: ?mlx.mlx_array) !mlx.mlx_array {
    self.embedding_mode = true;
    defer self.embedding_mode = false;
    var ctx = self.defaultCtx();
    ctx.key_pad_mask = key_pad_mask;
    ctx.vision_embeddings = vision;
    return self.forwardWith(&ctx, token_ids);
}

/// True when the checkpoint ships a sentence-transformers Dense head
/// (EmbeddingGemma dense.0/dense.1, EmbeddingGemma 2's embedding_projection
/// as the lone dense.0), applied between pool and normalize.
pub fn hasEmbedProjection(self: *const Transformer) bool {
    const b = self.plugin orelse return false;
    const e = b.table.embed orelse return false;
    return e.project != null;
}

/// Sentence-transformers Dense head: pooled [B, H] → dense.0 (→ dense.1). Caller frees.
pub fn embedProjection(self: *const Transformer, pooled: mlx.mlx_array) !mlx.mlx_array {
    const b = self.plugin orelse return error.NoEmbedProjection;
    return plugin_driver.embedProject(b, pooled);
}
