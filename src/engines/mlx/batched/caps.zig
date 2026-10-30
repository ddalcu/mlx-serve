//! engines/mlx/batched/caps.
//! Owns hasRecurrentLayers, batchedGdnReady, supportsBatchedGdnDecode.

const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;

/// Can these slots take ONE batched GDN decode tick? Every slot must
/// already carry initialized state for every linear layer — that is true
/// after any prefill, and false for a slot that has not run one, whose
/// zero-state would otherwise be merged at the wrong width.
/// Does a slot of this trunk carry per-layer recurrent state (GDN, PLE window)
/// the batched path merges? MiMo's attention + MoE trunk carries none.
pub fn hasRecurrentLayers(self: *const Transformer) bool {
    if (self.plugin) |b| return b.positions != .kv_step;
    const ml = self.moe_layers orelse return false;
    for (ml) |*lw| if (lw.is_linear or lw.ple != null) return true;
    return false;
}

pub fn batchedGdnReady(self: *const Transformer, ctxs: []const *ForwardCtx) bool {
    if (self.plugin) |b| {
        for (ctxs) |c| {
            const entries = c.ssm_entries orelse return b.positions == .kv_step;
            for (entries, 0..) |*e, i| if (b.recurrentAt(i) and !e.initialized) return false;
        }
        return true;
    }
    const ml = self.moe_layers orelse return false;
    if (!self.hasRecurrentLayers()) return true;
    for (ctxs) |c| {
        const entries = c.ssm_entries orelse return false;
        if (entries.len != ml.len) return false;
        for (ml, 0..) |*lw, i| {
            // The PLE layer merges its conv window too and reads the
            // slot's token history — both exist after any prefill.
            if (lw.ple != null and (entries[i].aux_state.ctx == null or !entries[i].ple_prev_valid)) return false;
            if (!lw.is_linear) continue;
            if (!entries[i].initialized) return false;
            if (entries[i].conv_state.ctx == null or entries[i].ssm_state.ctx == null) return false;
        }
    }
    return true;
}

/// Does this transformer support `forwardMoeBatchedDecode`? The GDN trunk
/// shape only — the archs that share `forwardMoeWith` but add per-layer
/// state or routing of their own (inkling short-convs, laguna, MLA,
/// gemma4 MoE) are NOT covered and must keep the serial path. Asked by
/// the scheduler's batching gate, so a new arch on this forward defaults
/// to serial instead of silently riding a path that never modelled it.
pub fn supportsBatchedGdnDecode(self: *const Transformer) bool {
    // Arch question: ONE predicate, shared with server.zig's
    // --max-concurrent clamp so the two cannot disagree about whether
    // this model batches.
    if (!self.config.supportsBatchedGdnDecode()) return false;
    if (self.plugin != null) return true;
    // Built-state question: this path reads `moe_layers` and the GDN
    // ssm entries, so the trunk must actually be that shape.
    if (self.moe_layers == null) return false;
    if (self.mlx_stream_ext != null) return false;
    // Every layer must be one of the two shapes this path handles.
    for (self.moe_layers.?) |*lw| {
        switch (lw.mlp) {
            .dense => {},
            // Routed experts are row-generic (`moeMLP` sorts B*S rows); the
            // config predicate already named which MoE trunks' per-slot state
            // this path merges.
            .moe => {},
        }
    }
    return true;
}
