//! engines/mlx/capture.
//! Owns forwardCaptureHidden, forwardWithCapture, forwardWithCaptureAll, captureHidden.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ForwardCtx = @import("forward_ctx.zig").ForwardCtx;
const Transformer = @import("transformer.zig").Transformer;

/// Run a forward pass and ALSO capture the post-final-norm hidden state
/// at the LAST position into `*out_hidden`. Used by PLD verify-fusion
/// (which re-uses the captured hidden as part of partial-accept rollback)
/// and by the Gemma 4 assistant drafter (which needs `h_prev` as a seed
/// for the next drafter step). Caller owns the captured array (must
/// `mlx_array_free`). Both `forwardStandard` and `forwardMoe` honor the
/// capture; other families fall through to a regular forward and leave
/// `*out_hidden` as a default `mlx_array_new()`.
pub fn forwardCaptureHidden(
    self: *Transformer,
    token_ids: mlx.mlx_array,
    out_hidden: *mlx.mlx_array,
) !mlx.mlx_array {
    std.debug.assert(self.capture_hidden == null); // re-entrant call
    var ctx = self.defaultCtx();
    ctx.capture_hidden = out_hidden;
    return self.forwardWith(&ctx, token_ids);
}

/// Variant of `forwardWith` that overrides `ctx.capture_hidden` for this
/// call only (saved and restored on exit). Used by per-slot generators
/// (Phase 2) so the capture target is request-local without mutating
/// shared state on the ctx.
pub fn forwardWithCapture(
    self: *Transformer,
    ctx: *ForwardCtx,
    token_ids: mlx.mlx_array,
    out_hidden: *mlx.mlx_array,
) !mlx.mlx_array {
    const saved = ctx.capture_hidden;
    ctx.capture_hidden = out_hidden;
    defer ctx.capture_hidden = saved;
    return self.forwardWith(ctx, token_ids);
}

/// Like `forwardWithCapture` but ALSO captures the full `[B, L, H]`
/// post-final-norm hidden (all positions) into `out_hidden_all`. The MTP
/// verify forward needs the last-position hidden (next round's h_prev)
/// and every position's hidden (committed-history re-append) in one pass.
pub fn forwardWithCaptureAll(
    self: *Transformer,
    ctx: *ForwardCtx,
    token_ids: mlx.mlx_array,
    out_hidden: *mlx.mlx_array,
    out_hidden_all: *mlx.mlx_array,
) !mlx.mlx_array {
    const saved = ctx.capture_hidden;
    const saved_all = ctx.capture_hidden_all;
    ctx.capture_hidden = out_hidden;
    ctx.capture_hidden_all = out_hidden_all;
    defer {
        ctx.capture_hidden = saved;
        ctx.capture_hidden_all = saved_all;
    }
    return self.forwardWith(ctx, token_ids);
}

/// `ctx.capture_hidden` gets the LAST row of `src` `[B, 1, H]`,
/// `ctx.capture_hidden_all` the whole `[B, L, H]` (refcount-shared).
pub fn captureHidden(ctx: *ForwardCtx, src: mlx.mlx_array, s: mlx.mlx_stream) !void {
    if (ctx.capture_hidden) |target| {
        const shape = mlx.getShape(src);
        const last = shape[1] - 1;
        const start = [_]c_int{ 0, last, 0 };
        const stop = [_]c_int{ shape[0], shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sliced, src, &start, 3, &stop, 3, &strides, 3, s));
        _ = mlx.mlx_array_set(target, sliced);
        _ = mlx.mlx_array_free(sliced);
    }
    if (ctx.capture_hidden_all) |target_all| _ = mlx.mlx_array_set(target_all, src);
}
