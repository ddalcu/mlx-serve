//! engines/mlx/load/layers_layered.
//! Owns initMoeLayers.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const moe_affine4 = @import("../../../sdk/blocks/moe/moe_affine4.zig");
const Weights = @import("../../../sdk/config/model.zig").Weights;
const concatRows = @import("../../../mlx/array.zig").concatRows;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const hy3ExpertContainer = @import("../../../plugins/hy_v3/load.zig").hy3ExpertContainer;
const loadHcWeights = @import("../../../plugins/qwen4_exp/load.zig").loadHcWeights;
const loadPleWeights = @import("../../../plugins/qwen4_exp/load.zig").loadPleWeights;
const fuseRowsInPlace = @import("../../../sdk/blocks/linear/fused_rows.zig").fuseRowsInPlace;
const joinableCols = @import("../../../sdk/blocks/linear/fused_rows.zig").joinableCols;
const checkExl3Bank = @import("exl3_bank.zig").checkExl3Bank;
const MoeLayerWeights = @import("../layer_weights.zig").MoeLayerWeights;
const QW = @import("../layer_weights.zig").QW;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const bf16Scalar = @import("weights.zig").bf16Scalar;
const dequantizeSmall = @import("weights.zig").dequantizeSmall;
const getLayerBias = @import("weights.zig").getLayerBias;
const getLayerScaleOrEmpty = @import("weights.zig").getLayerScaleOrEmpty;
const getLayerWeight = @import("weights.zig").getLayerWeight;
const getLayerWeightAt = @import("weights.zig").getLayerWeightAt;
const getLayerWeightAtOpt = @import("weights.zig").getLayerWeightAtOpt;
const getLayerWeightOpt = @import("weights.zig").getLayerWeightOpt;
const lastDim = @import("weights.zig").lastDim;
const layerBase = @import("weights.zig").layerBase;
const loadQW = @import("weights.zig").loadQW;
const maybeTransposeForBf16 = @import("weights.zig").maybeTransposeForBf16;
const moeExpertSuffix = @import("weights.zig").moeExpertSuffix;
const splitPackedGateUp = @import("weights.zig").splitPackedGateUp;

pub fn initMoeLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { moe_layers: []MoeLayerWeights, ssm_entries: []SSMCacheEntry, owned_bf16: []mlx.mlx_array } {
    log.info("Precomputing MoE layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const moe_layers = try allocator.alloc(MoeLayerWeights, config.num_hidden_layers);
    const ssm_entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
    var owned_bf16: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer {
        for (owned_bf16.items) |a| _ = mlx.mlx_array_free(a);
        owned_bf16.deinit(allocator);
    }
    const is_gemma4 = config.isGemma4Layers();
    const is_hy3 = std.mem.eql(u8, config.model_type, "hy_v3");
    const is_qwen4 = config.isQwen4();
    // Attention projections that may ship dense bf16 beside quantized experts.
    const probe_attn_scales = config.modelopt;

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &moe_layers[i];
        const is_linear = config.isLinearLayer(li);

        // qwen4_exp has none: the hyper-connection read normalizes.
        lw.input_norm = if (is_qwen4)
            mlx.mlx_array_new()
        else
            try getLayerWeight(weights, name_buf, prefix, li, "input_layernorm.weight");
        lw.post_attn_norm = if (is_qwen4)
            mlx.mlx_array_new()
        else
            try getLayerWeight(weights, name_buf, prefix, li, "post_attention_layernorm.weight");
        lw.is_linear = is_linear;

        // `moe_layers` comes from `allocator.alloc` which skips struct defaults, so every
        // optional must be initialized before the conditional Gemma-4-only assignments;
        // otherwise the eval loop reads uninitialized memory as valid handles (segfaults
        // with 0xaa...aa on Qwen3-Next and similar non-Gemma MoE models).
        lw.pre_ff_norm = null;
        lw.post_ff_norm = null;
        lw.pre_ff_norm_2 = null;
        lw.post_ff_norm_1 = null;
        lw.post_ff_norm_2 = null;
        lw.layer_scalar = null;
        lw.shared_mlp = null;
        lw.hc_attn = null;
        lw.hc_mlp = null;
        lw.ple = null;
        if (is_qwen4) {
            var hb: [96]u8 = undefined;
            lw.hc_attn = try loadHcWeights(weights, std.fmt.bufPrint(&hb, "{s}.layers.{d}.attn_hyper_connection", .{ prefix, li }) catch unreachable, true, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            lw.hc_mlp = try loadHcWeights(weights, std.fmt.bufPrint(&hb, "{s}.layers.{d}.mlp_hyper_connection", .{ prefix, li }) catch unreachable, true, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            if (config.pleLocalLayer() == li) {
                lw.ple = try loadPleWeights(weights, name_buf, prefix, li, config.hc_count, config.hidden_size, &owned_bf16, allocator, s);
            }
        }

        // Gemma 4 MoE: extra feedforward norms, layer scalar, shared expert MLP
        if (is_gemma4) {
            lw.pre_ff_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "pre_feedforward_layernorm.weight");
            lw.post_ff_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm.weight");
            lw.pre_ff_norm_2 = getLayerWeightOpt(weights, name_buf, prefix, li, "pre_feedforward_layernorm_2.weight");
            lw.post_ff_norm_1 = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm_1.weight");
            lw.post_ff_norm_2 = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm_2.weight");
            lw.layer_scalar = getLayerWeightOpt(weights, name_buf, prefix, li, "layer_scalar");
            lw.shared_mlp = .{
                .gate_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                .gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                .gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                .up_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                .up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                .down_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                .down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
            };
            const sm = &lw.shared_mlp.?;
            try maybeTransposeForBf16(&sm.gate_w, sm.gate_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&sm.up_w, sm.up_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&sm.down_w, sm.down_s, &owned_bf16, allocator, s);
            if (sm.gate_s.ctx != null) {
                var parts = [_][3]*mlx.mlx_array{ .{ &sm.gate_w, &sm.gate_s, &sm.gate_b }, .{ &sm.up_w, &sm.up_s, &sm.up_b } };
                const names = [_][]const u8{ "mlp.gate_proj", "mlp.up_proj" };
                sm.gu = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
            }
        }

        if (is_linear) {
            // Detect combined (qkvz+ba) vs separate (qkv+z+a+b) projections.
            // Each projection's `*_s`/`*_b` are loaded optionally — Unsloth Dynamic
            // checkpoints (e.g. Qwen3.6 UD) leave linear_attn projections as plain
            // bf16 with no scales/biases tensors, even though the rest of the model
            // is quantized. Null-ctx scales triggers a transpose-on-load so that
            // `qmatmulBits` can dispatch to plain `mlx_matmul`.
            const combined = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.weight") != null;
            if (combined) {
                lw.attn = .{ .linear = .{
                    .combined_proj = true,
                    .qkv_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.weight"),
                    .qkv_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.scales") orelse mlx.mlx_array_new(),
                    .qkv_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.biases") orelse mlx.mlx_array_new(),
                    .z_w = mlx.mlx_array_new(),
                    .z_s = mlx.mlx_array_new(),
                    .z_b = mlx.mlx_array_new(),
                    .a_w = mlx.mlx_array_new(),
                    .a_s = mlx.mlx_array_new(),
                    .a_b = mlx.mlx_array_new(),
                    .b_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.weight"),
                    .b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.scales") orelse mlx.mlx_array_new(),
                    .b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.biases") orelse mlx.mlx_array_new(),
                    .conv1d_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                    .A_log = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                    .dt_bias = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                    .norm_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                    .out_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                    .out_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.scales") orelse mlx.mlx_array_new(),
                    .out_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.biases") orelse mlx.mlx_array_new(),
                } };
                const la = &lw.attn.linear;
                try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, allocator, s);
            } else {
                lw.attn = .{ .linear = .{
                    .qkv_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.weight"),
                    .qkv_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.scales") orelse mlx.mlx_array_new(),
                    .qkv_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.biases") orelse mlx.mlx_array_new(),
                    .z_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_z.weight"),
                    .z_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.scales") orelse mlx.mlx_array_new(),
                    .z_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.biases") orelse mlx.mlx_array_new(),
                    .a_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_a.weight"),
                    .a_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.scales") orelse mlx.mlx_array_new(),
                    .a_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.biases") orelse mlx.mlx_array_new(),
                    .b_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_b.weight"),
                    .b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.scales") orelse mlx.mlx_array_new(),
                    .b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.biases") orelse mlx.mlx_array_new(),
                    .conv1d_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                    .A_log = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                    .dt_bias = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                    .norm_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                    .out_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                    .out_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.scales") orelse mlx.mlx_array_new(),
                    .out_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.biases") orelse mlx.mlx_array_new(),
                } };
                const la = &lw.attn.linear;
                try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.z_w, la.z_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.a_w, la.a_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, allocator, s);
                if (la.qkv_s.ctx != null) {
                    var parts = [_][3]*mlx.mlx_array{ .{ &la.qkv_w, &la.qkv_s, &la.qkv_b }, .{ &la.z_w, &la.z_s, &la.z_b }, .{ &la.a_w, &la.a_s, &la.a_b }, .{ &la.b_w, &la.b_s, &la.b_b } };
                    const names = [_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b" };
                    la.in = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
                }
            }
        } else {
            // MiMo and ModelOpt packs may ship attention DENSE bf16 (no scales)
            // beside quantized experts. Probe per-tensor: scales
            // present → the quant path (quantParamsHinted resolves bits from
            // geometry); absent → bf16 (maybeTransposeForBf16). Other archs keep
            // mandating scales (config.quant_bits) so a genuinely-missing scale
            // still errors loudly. getLayerBias already tolerates absence.
            const k_s = if (probe_attn_scales)
                (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.scales") orelse mlx.mlx_array_new())
            else
                try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.k_proj.scales", config.quant_bits);
            const k_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
            const k_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config);
            // Gemma 4 MoE: global layers use K=V (no separate v_proj)
            const v_w = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.weight") orelse k_w;
            const v_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.scales") orelse k_s;
            const v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.biases") orelse k_b;
            const v_aliases_k = v_w.ctx == k_w.ctx;
            lw.attn = .{
                .full = .{
                    .q_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                    .q_s = if (probe_attn_scales)
                        (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.scales") orelse mlx.mlx_array_new())
                    else
                        try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.q_proj.scales", config.quant_bits),
                    .q_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config),
                    .k_w = k_w,
                    .k_s = k_s,
                    .k_b = k_b,
                    .v_w = v_w,
                    .v_s = v_s,
                    .v_b = v_b,
                    .o_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                    .o_s = if (probe_attn_scales)
                        (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.scales") orelse mlx.mlx_array_new())
                    else
                        try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.o_proj.scales", config.quant_bits),
                    .o_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.o_proj.biases", &config),
                    // An arch that DECLARES QK norm must ship it (a missing weight
                    // there is a broken checkpoint, not a variant). One that
                    // declares it off (MiMo) has no q_norm/k_norm tensors at
                    // all, and the mandatory getter's `unreachable` would kill the
                    // process on load rather than report anything.
                    .q_norm = if (config.has_qk_norm)
                        try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight")
                    else
                        (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_norm.weight") orelse mlx.mlx_array_new()),
                    .k_norm = if (config.has_qk_norm)
                        try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight")
                    else
                        (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_norm.weight") orelse mlx.mlx_array_new()),
                },
            };
            {
                // Dense bf16 (null-ctx scales): pre-transpose [out,in]→[in,out] so
                // qmatmulBits dispatches to a plain matmul. No-op on quantized weights.
                const fa = &lw.attn.full;
                try maybeTransposeForBf16(&fa.q_w, fa.q_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.k_w, fa.k_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.o_w, fa.o_s, &owned_bf16, allocator, s);
                if (v_aliases_k) {
                    // K=V share one weight — re-alias V to the transposed K (don't
                    // create a second copy or double-free).
                    fa.v_w = fa.k_w;
                } else {
                    try maybeTransposeForBf16(&fa.v_w, fa.v_s, &owned_bf16, allocator, s);
                }
                if (is_qwen4 and config.indexer_n_heads > 0) {
                    fa.idx_qk_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.weight");
                    fa.idx_qk_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.scales") orelse mlx.mlx_array_new();
                    fa.idx_qk_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.biases") orelse mlx.mlx_array_new();
                    fa.idx_q_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.q_layernorm.weight");
                    fa.idx_k_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.k_layernorm.weight");
                    try maybeTransposeForBf16(&fa.idx_qk_w, fa.idx_qk_s, &owned_bf16, allocator, s);
                }
                // Row-exact archs on NAX: one lane matmul serves q|k|v at every
                // width (a column's bits follow the joined shape, the same at 1 row and 16).
                const lane_join = naxAvailable() and config.rowExactArch();
                if ((is_gemma4 or lane_join) and !v_aliases_k and fa.q_s.ctx != null) {
                    var parts = [_][3]*mlx.mlx_array{ .{ &fa.q_w, &fa.q_s, &fa.q_b }, .{ &fa.k_w, &fa.k_s, &fa.k_b }, .{ &fa.v_w, &fa.v_s, &fa.v_b } };
                    const names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
                    fa.qkv = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
                }
            }
        }

        // hy_v3: layers [0, first_k_dense_replace) are DENSE (layer 0 on the
        // 295B) — they take the dense binding arm below. Every established
        // arch has first_k_dense_replace == 0, so layer_is_moe == isMoe() there.
        const layer_is_moe = config.isMoe() and li >= config.first_k_dense_replace;
        if (layer_is_moe and is_gemma4) {
            // Gemma 4 MoE: different weight naming, Sigma-MoE routing, no shared expert gate.
            // Each `*_s`/`*_b` is loaded optionally for Unsloth Dynamic compatibility —
            // bf16 layers carry only the weight, no scales/biases. The post-construction
            // `maybeTransposeForBf16` calls are no-ops for already-quantized weights.
            //
            // Expert naming variants:
            //   Gemma 4 26B-A4B:  experts.switch_glu.{gate,up,down}_proj.*
            //   packed:           experts.gate_up_proj.* (PACKED [E, 2M, X],
            //                     gate rows first) + experts.down_proj.*
            // The packed variant is sliced into gate/up halves at load
            // (splitPackedGateUp) so both feed the same MoeMlpWeights fields.
            const packed_gu_w = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.weight");
            var sw_gate_w: mlx.mlx_array = undefined;
            var sw_gate_s: mlx.mlx_array = undefined;
            var sw_gate_b: mlx.mlx_array = undefined;
            var sw_up_w: mlx.mlx_array = undefined;
            var sw_up_s: mlx.mlx_array = undefined;
            var sw_up_b: mlx.mlx_array = undefined;
            var sw_down_w: mlx.mlx_array = undefined;
            var sw_down_s: mlx.mlx_array = undefined;
            var sw_down_b: mlx.mlx_array = undefined;
            if (packed_gu_w) |gu_w| {
                const w_pair = try splitPackedGateUp(gu_w, s);
                try owned_bf16.append(allocator, w_pair.gate);
                try owned_bf16.append(allocator, w_pair.up);
                sw_gate_w = w_pair.gate;
                sw_up_w = w_pair.up;
                if (getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.scales")) |gu_s| {
                    const s_pair = try splitPackedGateUp(gu_s, s);
                    try owned_bf16.append(allocator, s_pair.gate);
                    try owned_bf16.append(allocator, s_pair.up);
                    sw_gate_s = s_pair.gate;
                    sw_up_s = s_pair.up;
                } else {
                    sw_gate_s = mlx.mlx_array_new();
                    sw_up_s = mlx.mlx_array_new();
                }
                if (getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.biases")) |gu_b| {
                    const b_pair = try splitPackedGateUp(gu_b, s);
                    try owned_bf16.append(allocator, b_pair.gate);
                    try owned_bf16.append(allocator, b_pair.up);
                    sw_gate_b = b_pair.gate;
                    sw_up_b = b_pair.up;
                } else {
                    sw_gate_b = mlx.mlx_array_new();
                    sw_up_b = mlx.mlx_array_new();
                }
                sw_down_w = try getLayerWeight(weights, name_buf, prefix, li, "experts.down_proj.weight");
                sw_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.down_proj.scales") orelse mlx.mlx_array_new();
                sw_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.down_proj.biases") orelse mlx.mlx_array_new();
            } else {
                sw_gate_w = try getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.weight");
                sw_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.scales") orelse mlx.mlx_array_new();
                sw_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.biases") orelse mlx.mlx_array_new();
                sw_up_w = try getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.weight");
                sw_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.scales") orelse mlx.mlx_array_new();
                sw_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.biases") orelse mlx.mlx_array_new();
                sw_down_w = try getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.weight");
                sw_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.scales") orelse mlx.mlx_array_new();
                sw_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.biases") orelse mlx.mlx_array_new();
            }
            lw.mlp = .{
                .moe = .{
                    .router_w = try getLayerWeight(weights, name_buf, prefix, li, "router.proj.weight"),
                    .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "router.proj.scales") orelse mlx.mlx_array_new(),
                    .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "router.proj.biases") orelse mlx.mlx_array_new(),
                    .router_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "router.scale"),
                    .per_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "router.per_expert_scale"),
                    .switch_gate_w = sw_gate_w,
                    .switch_gate_s = sw_gate_s,
                    .switch_gate_b = sw_gate_b,
                    .switch_up_w = sw_up_w,
                    .switch_up_s = sw_up_s,
                    .switch_up_b = sw_up_b,
                    .switch_down_w = sw_down_w,
                    .switch_down_s = sw_down_s,
                    .switch_down_b = sw_down_b,
                    // Shared expert handled via lw.shared_mlp for Gemma 4 (separate branch in forward)
                    .shared_gate_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                    .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_up_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                    .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_down_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                    .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
                },
            };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
            }
            // Pre-fold the sigma-MoE router norm scale: at runtime the router does
            // `rms_norm(x, router_scale * hidden_size^-0.5, eps)`. Multiplying once
            // at load time saves the per-layer multiply (3 ops × num_layers).
            if (lw.mlp.moe.router_scale) |rs| {
                const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
                const root_scalar = bf16Scalar(root_size, s);
                defer _ = mlx.mlx_array_free(root_scalar);
                var folded = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&folded, rs, root_scalar, s));
                try owned_bf16.append(allocator, folded);
                lw.mlp.moe.router_scale = folded;
            }
        } else if (layer_is_moe and is_hy3) {
            // Hy3 (hy_v3): stacked experts (already [E, out, packed] in MLX
            // conversions — never per-expert), the QUANTIZED router under
            // mlp.router.gate.* (8-bit on shipped checkpoints; scales optional
            // for a bf16 build — mlx-community oQ2e ships a bf16 router), the
            // f32 selection bias mlp.expert_bias, and an UNGATED shared expert
            // under mlp.shared_mlp.* (absent on 0-shared configs — binds empty
            // and moeMLP skips the branch). route_norm and router_scaling_factor
            // ride on the weights struct so moeMLP2 needs no config re-derivation
            // per call. The expert container name varies by converter
            // (`mlp.experts.*` in ox-ox builds, `mlp.switch_mlp.*` in mlx-lm
            // builds) — probe once and thread the resolved name through.
            const ex = hy3ExpertContainer(weights, name_buf, prefix, li);
            var exbuf: [64]u8 = undefined;
            lw.mlp = .{
                .moe = .{
                    .router_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.router.gate.weight"),
                    .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.router.gate.scales") orelse mlx.mlx_array_new(),
                    .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.router.gate.biases") orelse mlx.mlx_array_new(),
                    .switch_gate_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.weight")),
                    .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.biases")) orelse mlx.mlx_array_new(),
                    .switch_up_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.weight")),
                    .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.biases")) orelse mlx.mlx_array_new(),
                    .switch_down_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.weight")),
                    .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.biases")) orelse mlx.mlx_array_new(),
                    .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.biases") orelse mlx.mlx_array_new(),
                    .expert_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.expert_bias") orelse blk: {
                        // moe_router_enable_expert_bias=false checkpoints ship no
                        // bias tensor; the reference keeps zeros — selection
                        // reduces to plain sigmoid top-k. Non-null keeps moeMLP2
                        // on the sigmoid chain (null would fall back to softmax).
                        var zeros = mlx.mlx_array_new();
                        const zshape = [_]c_int{@intCast(config.num_experts)};
                        try mlx.check(mlx.mlx_zeros(&zeros, &zshape, 1, .float32, s));
                        try owned_bf16.append(allocator, zeros);
                        break :blk zeros;
                    },
                    .route_norm = config.moe_route_norm,
                    .route_scale = config.router_scaling_factor,
                    .shared_ungated = true,
                },
            };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
            }
        } else if (layer_is_moe) {
            // Qwen3.5 MoE — also serves Qwen3-30B-A3B (`qwen3_moe`), which shares
            // this exact router/switch_mlp layout. Each `*_s`/`*_b` is loaded
            // optionally — Unsloth Dynamic checkpoints (e.g. Qwen3.6-A3B UD) leave
            // the router (`mlp.gate`) and the shared-expert gate
            // (`mlp.shared_expert_gate`) as plain bf16, with no scales/biases.
            // The shared-expert WEIGHTS themselves are also optional: qwen3_moe
            // (Qwen3-30B-A3B / Coder) dropped the shared expert entirely
            // (shared_expert_intermediate_size: 0, no mlp.shared_expert.*). When
            // absent they bind to empty handles and `shared_expert_gate_w` stays
            // null, which makes moeMLP early-return the routed-expert sum without
            // ever reading them — so no MISSING WEIGHT crash. The
            // `maybeTransposeForBf16` calls below pre-transpose bf16 weights from
            // `[out, in]` → `[in, out]` so `qmatmulBits` can dispatch to plain
            // `mlx_matmul`; they no-op on already-quantized AND on empty handles.
            // An EXL3 pack (src/sdk/quant/exl3) binds trellis/suh/svh in the same slots.
            const leaf: [3][]const u8 = if (config.exl3 != null) .{ "trellis", "suh", "svh" } else .{ "weight", "scales", "biases" };
            var exbuf: [64]u8 = undefined;
            lw.mlp = .{ .moe = .{
                .router_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.weight"),
                .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.scales") orelse mlx.mlx_array_new(),
                .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.biases") orelse mlx.mlx_array_new(),
                .switch_gate_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", leaf[0])),
                .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", leaf[1])) orelse mlx.mlx_array_new(),
                .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", leaf[2])) orelse mlx.mlx_array_new(),
                .switch_up_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", leaf[0])),
                .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", leaf[1])) orelse mlx.mlx_array_new(),
                .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", leaf[2])) orelse mlx.mlx_array_new(),
                .switch_down_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", leaf[0])),
                .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", leaf[1])) orelse mlx.mlx_array_new(),
                .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", leaf[2])) orelse mlx.mlx_array_new(),
                .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight") orelse mlx.mlx_array_new(),
                .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales") orelse mlx.mlx_array_new(),
                .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases") orelse mlx.mlx_array_new(),
                .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight") orelse mlx.mlx_array_new(),
                .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales") orelse mlx.mlx_array_new(),
                .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases") orelse mlx.mlx_array_new(),
                .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight") orelse mlx.mlx_array_new(),
                .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales") orelse mlx.mlx_array_new(),
                .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases") orelse mlx.mlx_array_new(),
                .shared_expert_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.weight"),
                .shared_expert_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.scales") orelse mlx.mlx_array_new(),
                .shared_expert_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.biases") orelse mlx.mlx_array_new(),
                .per_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.expert_scale"),
                .gate_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.expert_scale") orelse .{ .ctx = null },
                .up_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.expert_scale") orelse .{ .ctx = null },
            } };
            if (config.exl3) |spec| try checkExl3Bank(&lw.mlp.moe, config, spec);
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
                if (mw.shared_expert_gate_w) |*seg_w_ptr| {
                    try dequantizeSmall(seg_w_ptr, &mw.shared_expert_gate_s.?, &mw.shared_expert_gate_b.?, config.hidden_size, &owned_bf16, allocator, s);
                    try maybeTransposeForBf16(seg_w_ptr, mw.shared_expert_gate_s.?, &owned_bf16, allocator, s);
                    if (mw.shared_expert_gate_s.?.ctx == null and mlx.mlx_array_size(seg_w_ptr.*) == config.hidden_size) {
                        var flat = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(flat);
                        try mlx.check(mlx.mlx_reshape(&flat, seg_w_ptr.*, &[_]c_int{@intCast(config.hidden_size)}, 1, s));
                        var vec = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_contiguous(&vec, flat, false, s));
                        try owned_bf16.append(allocator, vec);
                        mw.shared_gate_vec = vec;
                    }
                }
                const Bk = moe_affine4.Bank;
                mw.bank_views = try moe_affine4.bindViews(Bk{ .w = mw.switch_gate_w, .s = mw.switch_gate_s, .b = mw.switch_gate_b }, Bk{ .w = mw.switch_up_w, .s = mw.switch_up_s, .b = mw.switch_up_b }, Bk{ .w = mw.switch_down_w, .s = mw.switch_down_s, .b = mw.switch_down_b }, &owned_bf16, allocator, s);
            }
        } else {
            lw.mlp = .{ .dense = .{
                .gate_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                .gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                .gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                .up_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                .up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                .down_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                .down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
            } };
            {
                const dw = &lw.mlp.dense;
                try maybeTransposeForBf16(&dw.gate_w, dw.gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dw.up_w, dw.up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dw.down_w, dw.down_s, &owned_bf16, allocator, s);
                // Hadamard packs: `aliasFusedRotations` binds the joined buffer's signs.
                if (dw.gate_s.ctx != null) {
                    var parts = [_][3]*mlx.mlx_array{ .{ &dw.gate_w, &dw.gate_s, &dw.gate_b }, .{ &dw.up_w, &dw.up_s, &dw.up_b } };
                    const names = [_][]const u8{ "mlp.gate_proj", "mlp.up_proj" };
                    dw.gu = try fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, allocator, s);
                }
            }
        }

        ssm_entries[i] = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };
    }

    // The PLE is placed by exact equality inside the loop above and nothing downstream notices
    // its absence; a silently PLE-less qwen4 emits plausible text.
    if (config.isQwen4()) {
        const has_ple = try allocator.alloc(bool, moe_layers.len);
        defer allocator.free(has_ple);
        for (moe_layers, 0..) |*lw, i| has_ple[i] = lw.ple != null;
        if (!model_mod.qwen4PleInstalledAt(has_ple, if (config.pleLocalLayer()) |l| @intCast(l) else -1)) {
            log.err("[qwen4] the PLE was not installed at layer {d} ({d} layers carry PLE weights)\n", .{ config.ple_layer_idx, std.mem.count(bool, has_ple, &.{true}) });
            return error.Qwen4PleNotInstalled;
        }
    }

    return .{
        .moe_layers = moe_layers,
        .ssm_entries = ssm_entries,
        .owned_bf16 = try owned_bf16.toOwnedSlice(allocator),
    };
}
