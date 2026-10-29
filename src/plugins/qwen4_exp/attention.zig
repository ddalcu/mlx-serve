//! qwen4_exp's full-attention layer: gated q/k/v projection, the QSA indexer's mask or block picks, and the
//! attention arm that reads them (dense, gathered, or the fused decode kernel).

const mlx = @import("../../mlx/ffi.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../../engines/mlx/layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Trunk = @import("trunk.zig").Trunk;
const AttnVerifyInputs = @import("trunk/verify.zig").AttnVerifyInputs;
const qwen4Standin = @import("../../sdk/diag/standin.zig").qwen4Standin;
const Qwen4Trace = @import("mtp.zig").Qwen4Trace;
const Qwen4AttnProf = @import("prof.zig").Qwen4AttnProf;
const headPlaceVisibility = @import("qsa/head_place.zig").headPlaceVisibility;
const qsaMaskFromBlocks = @import("../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;

/// Full-attention layer with the QSA mask threaded through
/// `gatedFullAttnWith` (q-gate, QK norm, partial RoPE, KV cache all shared).
pub fn qwen4AttnWith(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, is_prefill: bool) !mlx.mlx_array {
    return self.qwen4AttnProjected(ctx, x, fa, entry, layer, cache_len, pos_base, batch, seq_len, is_prefill, null, false);
}

pub fn qwen4AttnProjected(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, is_prefill: bool, projected: ?*const AttnVerifyInputs, skip_output: bool) !mlx.mlx_array {
    // Registered above the fallible builders: `qsaMask` can throw with `ctx.qsa_blocks` already written.
    defer {
        if (ctx.qsa_mask.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_mask);
        ctx.qsa_mask = .{ .ctx = null };
        if (ctx.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_blocks);
        ctx.qsa_blocks = .{ .ctx = null };
        if (ctx.batch_slots) |slots| {
            for (slots) |sc| {
                if (sc.qsa_mask.ctx != null) _ = mlx.mlx_array_free(sc.qsa_mask);
                sc.qsa_mask = .{ .ctx = null };
                if (sc.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(sc.qsa_blocks);
                sc.qsa_blocks = .{ .ctx = null };
            }
        }
    }
    const prof = Qwen4AttnProf.begin(x, seq_len);
    defer Qwen4AttnProf.active = false;
    if (fa.idx_qk_w.ctx != null and !qwen4Standin().attn_qsa) {
        ctx.qsa_mask = try self.qsaMask(ctx, x, fa, entry, layer, cache_len, pos_base, batch, seq_len);
    }
    if (prof) Qwen4AttnProf.lap(if (ctx.qsa_blocks.ctx != null) ctx.qsa_blocks else ctx.qsa_mask, .indexer);
    if (ctx.head_place) |p| {
        const vis = try headPlaceVisibility(self.s, ctx.qsa_mask, p, @intCast(self.config.indexer_compress_ratio), cache_len + seq_len);
        if (ctx.qsa_mask.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_mask);
        ctx.qsa_mask = vis;
    }
    if (layer == 3) if (Trunk.qwen4_trace) |tr| {
        if (ctx.qsa_mask.ctx != null) {
            Qwen4Trace.set(&tr.qsa_mask, ctx.qsa_mask);
        } else if (ctx.qsa_blocks.ctx != null) {
            const m = try qsaMaskFromBlocks(self.s, ctx.qsa_blocks, cache_len + seq_len, @intCast(self.config.indexer_compress_ratio));
            defer _ = mlx.mlx_array_free(m);
            Qwen4Trace.set(&tr.qsa_mask, m);
        }
    };
    const out = try self.gatedFullAttnProjected(ctx, x, fa, layer, pos_base + cache_len, batch, seq_len, is_prefill, projected, skip_output);
    if (prof) Qwen4AttnProf.lap(out, .tail);
    return out;
}
