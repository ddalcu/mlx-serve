//! The legacy `Transformer`'s methods over `sdk/blocks/moe/routing.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const Transformer = @import("../transformer.zig").Transformer;
const RouterGate = @import("../../../sdk/blocks/moe/router_kernel.zig").RouterGate;
const moeRouterTopK = @import("../../../sdk/blocks/moe/router_kernel.zig").moeRouterTopK;
const MoeRouting = @import("../../../sdk/blocks/moe/routing.zig").MoeRouting;
const groupLimitedRouting = @import("../../../sdk/blocks/moe/routing.zig").groupLimitedRouting;
const hy3RoutingChain = @import("../../../sdk/blocks/moe/routing.zig").hy3RoutingChain;
const kolibriRoutingChain = @import("../../../sdk/blocks/moe/routing.zig").kolibriRoutingChain;
const moeRoutingChain = @import("../../../sdk/blocks/moe/routing.zig").moeRoutingChain;

/// Compile MoE routing (negate→argpartition→slice→softmax→take_along_axis→sum→expand→divide)
/// into a single fused kernel. Input: router_logits. Outputs: inds, norm_scores.
/// shapeless=false: slice bounds derive from input ndim, so the closure must
/// re-trace per input shape. MoE inference only sees two shapes in practice
/// (decode seq_len=1, prefill seq_len=N), so the trace cost amortizes after
/// the first prefill + first decode.
pub fn compileMoeRouting(self: *Transformer) void {
    if (self.config.moe_sigmoid_router) {
        // hy_v3: the sigmoid+bias chain takes a second input (the per-layer
        // expert_bias weight), so it compiles as its own closure.
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &hy3RoutingClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, false);
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_hy3_routing = compiled;
            log.info("MoE routing compiled (hy3 sigmoid variant)\n", .{});
        }
        return;
    }
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &moeRoutingClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, false);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_moe_routing = compiled;
        log.info("MoE routing compiled (kernel fusion enabled)\n", .{});
    }
}

/// Pure subgraph for MoE routing. Inputs:
///   [0] router_logits — shape [..., num_experts]
/// Outputs:
///   [0] inds         — shape [..., K], int32 expert indices (top-K)
///   [1] norm_scores  — shape [..., K], renormalized top-K softmax weights
///
/// The sigma-MoE per-expert-scale path stays outside the closure because it
/// branches on per-layer weights at model-load time.
pub fn moeRoutingClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    const k: c_int = @intCast(self.config.num_experts_per_tok);

    var router_logits = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&router_logits, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(router_logits);

    const inds_norm = self.moeRoutingUncompiled(router_logits, k) catch return -1;
    defer _ = mlx.mlx_array_free(inds_norm.inds);
    defer _ = mlx.mlx_array_free(inds_norm.norm_scores);

    const out_arr = [_]mlx.mlx_array{ inds_norm.inds, inds_norm.norm_scores };
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 2);
    return 0;
}

/// Reference implementation of the MoE routing chain (used both as fallback
/// and as the body the compiled closure traces). Returns owned `inds` +
/// `norm_scores` arrays — caller must free both.
pub fn moeRoutingUncompiled(self: *const Transformer, router_logits: mlx.mlx_array, k: c_int) !MoeRouting {
    return moeRoutingChain(router_logits, k, self.s);
}

/// Hy3 sigmoid routing closure body. Inputs:
///   [0] router_logits — [..., num_experts]
///   [1] expert_bias   — f32 [num_experts]
/// route_norm / route_scale / k are model-constant (from config), so they
/// bake into the trace.
pub fn hy3RoutingClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    const k: c_int = @intCast(self.config.num_experts_per_tok);

    var router_logits = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&router_logits, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(router_logits);
    var expert_bias = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&expert_bias, input, 1) != 0) return -1;
    defer _ = mlx.mlx_array_free(expert_bias);

    const routed = hy3RoutingChain(
        router_logits,
        expert_bias,
        k,
        self.config.moe_route_norm,
        self.config.router_scaling_factor,
        self.s,
    ) catch return -1;
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    const out_arr = [_]mlx.mlx_array{ routed.inds, routed.norm_scores };
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 2);
    return 0;
}

/// Hy3 sigmoid+bias routing — compiled closure when available, else the
/// direct chain. Returns owned `inds` + `norm_scores`.
pub fn computeHy3Routing(self: *const Transformer, router_logits: mlx.mlx_array, expert_bias: mlx.mlx_array) !MoeRouting {
    if (self.config.moe_logit_select) {
        const k: c_int = @intCast(self.config.num_experts_per_tok);
        if (try moeRouterTopK(self.s, router_logits, expert_bias, k, .logit_bias, false, 1.0, .float32, 0, 0, null)) |fused| return fused;
        return kolibriRoutingChain(router_logits, expert_bias, k, self.s);
    }
    // Group-limited ("noaux_tc") selection is the same sigmoid+bias chain
    // with a group mask between the bias and the top-k, so it shares this
    // entry point rather than a parallel one. `groupLimitedRouting` has its
    // own fused-kernel attempt and its own fallback chain.
    if (self.config.moe_n_group > 1) {
        return groupLimitedRouting(
            router_logits,
            expert_bias,
            @intCast(self.config.num_experts_per_tok),
            @intCast(self.config.moe_n_group),
            @intCast(self.config.moe_topk_group),
            self.config.moe_route_norm,
            self.config.router_scaling_factor,
            self.s,
        );
    }
    if (try moeRouterTopK(
        self.s,
        router_logits,
        expert_bias,
        @intCast(self.config.num_experts_per_tok),
        .sigmoid_bias,
        self.config.moe_route_norm,
        self.config.router_scaling_factor,
        .bfloat16,
        0,
        0,
        null,
    )) |fused| return fused;
    if (self.compiled_hy3_routing) |compiled| {
        const in_arr = [_]mlx.mlx_array{ router_logits, expert_bias };
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 2);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));

        var inds = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(inds);
        try mlx.check(mlx.mlx_vector_array_get(&inds, out_vec, 0));
        var norm_scores = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(norm_scores);
        try mlx.check(mlx.mlx_vector_array_get(&norm_scores, out_vec, 1));
        return .{ .inds = inds, .norm_scores = norm_scores };
    }
    return hy3RoutingChain(
        router_logits,
        expert_bias,
        @intCast(self.config.num_experts_per_tok),
        self.config.moe_route_norm,
        self.config.router_scaling_factor,
        self.s,
    );
}

/// Apply the compiled MoE routing closure if available, else fall back.
/// Returns owned `inds` + `norm_scores` — caller must free both.
pub fn computeMoeRouting(self: *const Transformer, router_logits: mlx.mlx_array, gate: ?RouterGate) !MoeRouting {
    const k: c_int = @intCast(self.config.num_experts_per_tok);
    if (try moeRouterTopK(
        self.s,
        router_logits,
        .{ .ctx = null },
        k,
        .softmax,
        true,
        1.0,
        mlx.mlx_array_dtype(router_logits),
        0,
        0,
        gate,
    )) |fused| return fused;
    if (self.compiled_moe_routing) |compiled| {
        const in_arr = [_]mlx.mlx_array{router_logits};
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));

        var inds = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(inds);
        try mlx.check(mlx.mlx_vector_array_get(&inds, out_vec, 0));
        var norm_scores = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(norm_scores);
        try mlx.check(mlx.mlx_vector_array_get(&norm_scores, out_vec, 1));
        return .{ .inds = inds, .norm_scores = norm_scores };
    }
    return self.moeRoutingUncompiled(router_logits, k);
}
