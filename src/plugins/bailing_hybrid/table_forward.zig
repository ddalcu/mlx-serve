//! Ling 3.0's forward behind the model table: KDA layers on the slot's recurrent state and MLA layers on its KV, both
//! through `CacheOps`, then the MoE (or the dense first layer). One slot's rows per call; a verify captures its KDA
//! rows.
//! copied_from: engines/mlx/forward/layered.zig (`forwardMoeWith`, its bailing arms).

const mlx = @import("../../mlx/ffi.zig");
const slots_mod = @import("../glm5_next/slots.zig");
const trunk = @import("../glm5_next/trunk.zig");
const glm_moe = @import("../glm5_next/moe.zig");
const mla = @import("mla.zig");

const Array = mlx.mlx_array;

pub const Step = struct {
    slots: slots_mod.Slots,
    capture: bool = false,
    capture_ids: []const u32 = &.{},
    capture_out: []Array = &.{},
    eval_every: u32 = 0,
};

/// The final-normed hidden `[1, S, H]`; the head is the caller's.
pub fn hidden(m: *trunk.Model, st: *const Step, token_ids: Array) !Array {
    const offset: c_int = @intCast(st.slots.positions[0]);
    var h = try m.embedding(token_ids);
    defer _ = mlx.mlx_array_free(h);
    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;

    // The previous layer's tail may already have produced this layer's input norm.
    var carried_normed: ?Array = null;
    errdefer if (carried_normed) |c| {
        _ = mlx.mlx_array_free(c);
    };
    const n_layers = m.layers.len;
    for (m.layers, 0..) |*lw, layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const normed = if (carried_normed) |c| c else try m.rmsNorm(h, lw.input_norm);
        carried_normed = null;
        defer _ = mlx.mlx_array_free(normed);

        const attn_out = switch (lw.mixer) {
            .kda => |*la| try slots_mod.kda(m, st.slots, 0, normed, la, li, batch, seq_len, is_prefill, st.capture),
            .mla => |*w| try mla.mlaAttn(m, st.slots, normed, w, li, offset, batch, seq_len, is_prefill),
            .dsa => return error.LingLayerIsDsa,
        };
        defer _ = mlx.mlx_array_free(attn_out);

        // Residual + post-attention norm as one kernel where one serves.
        var ff_normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ff_normed);
        if (try m.narrowAddNorm(h, attn_out, lw.post_norm)) |fused| {
            _ = mlx.mlx_array_free(h);
            h = fused.h;
            _ = mlx.mlx_array_free(ff_normed);
            ff_normed = fused.normed;
        } else {
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, attn_out, m.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;
            _ = mlx.mlx_array_free(ff_normed);
            ff_normed = try m.rmsNorm(h, lw.post_norm);
        }
        const mlp_out = switch (lw.mlp) {
            .moe => |*mw| try glm_moe.forward(mw, m.s, ff_normed),
            .dense => |*dw| try m.denseMlp(dw, ff_normed),
        };
        defer _ = mlx.mlx_array_free(mlp_out);

        const last = layer_idx + 1 == n_layers;
        const next_norm = if (last) m.final_norm else m.layers[layer_idx + 1].input_norm;
        if (try m.narrowAddNorm(h, mlp_out, next_norm)) |fused| {
            _ = mlx.mlx_array_free(h);
            h = fused.h;
            carried_normed = fused.normed;
        } else {
            var h_next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_next, h, mlp_out, m.s));
            _ = mlx.mlx_array_free(h);
            h = h_next;
        }

        // DFlash taps `hidden_states[li + 1]`, the layer's output.
        for (st.capture_ids, st.capture_out) |cid, *slot| if (cid == li) {
            _ = mlx.mlx_array_set(slot, h);
        };
        if (is_prefill and st.eval_every > 0 and ((layer_idx + 1) % st.eval_every == 0 or last)) {
            try st.slots.evalWith(h, li + 1);
        }
    }
    if (carried_normed) |c| {
        carried_normed = null;
        return c;
    }
    return m.rmsNorm(h, m.final_norm);
}
