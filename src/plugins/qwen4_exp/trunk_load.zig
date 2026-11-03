//! Builds the trunk from a pack's loaded map: the embedding and head, the layers (the engine's MoE layer binder),
//! the final mixer, the n-gram state and table, the MTP head, the YaRN tables, then one batched eval of every
//! weight. The map stays the caller's (the trunk borrows its arrays).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const model = @import("../../sdk/config/model.zig");
const qwen4_mod = @import("qwen4_exp.zig");
const ple_gpu = @import("ple_gpu.zig");
const distributed_mod = @import("../../engines/mlx/cluster/distributed.zig");
const fused_rows = @import("../../sdk/blocks/linear/fused_rows.zig");
const Trunk = @import("trunk.zig").Trunk;
const loadHcWeights = @import("load.zig").loadHcWeights;
const loadQwen4Mtp = @import("mtp.zig").loadQwen4Mtp;
const qsaNaxArm = @import("qsa/gather.zig").qsaNaxArm;
const qsaScoreFusedArm = @import("../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreFusedArm;
const checkQsaCompressRatio = @import("qsa/history.zig").checkQsaCompressRatio;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const initMoeLayers = @import("../../engines/mlx/load/layers_layered.zig").initMoeLayers;
const weights_util = @import("../../engines/mlx/load/weights.zig");
const quantizeHeadMxfp8 = @import("../../sdk/quant/mxfp8.zig").quantizeHeadMxfp8;
const computeYarnFreqs = @import("../../sdk/blocks/rope/yarn.zig").computeYarnFreqs;
const yarnSpec = @import("../../engines/mlx/methods/yarn.zig").yarnSpec;
const mtp_nax_profile = @import("../../engines/mlx/spec/mtp_nax_profile.zig");
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const ternaryKernelCandidate = @import("../../sdk/quant/ternary.zig").ternaryKernelCandidate;
const firstNonTernaryBias = @import("../../sdk/quant/ternary.zig").firstNonTernaryBias;

/// A config field or bound weight the trunk's forward never reads (its arms are another family's), named; null when
/// none. The plugin's config and the layer binder produce none of these for a qwen4_exp pack, so a hit is a layout
/// the trunk has to learn before it serves it.
pub fn unservedLayout(config: *const model.ModelConfig, layers: []const MoeLayerWeights) ?[]const u8 {
    if (config.hadamard_block != 0) return "Hadamard rotations";
    if (config.norm_groups > 1) return "grouped norms";
    if (config.moe_sigmoid_router or config.moe_logit_select) return "a sigmoid router";
    if (config.kda_vector_gate or config.kda_gate_lower_bound != 0) return "a KDA gate";
    if (!config.kda_sigmoid_out_gate) return "a swish output gate";
    if (config.swiglu_clamp or config.swiglu_limit != 0) return "a clamped SwiGLU";
    if (config.hidden_act != .silu) return "a non-SiLU activation";
    if (config.scale_embeddings) return "scaled embeddings";
    for (layers) |*lw| {
        if (lw.shared_mlp != null) return "another family's layer blocks";
        switch (lw.mlp) {
            .moe => |*mw| {
                if (mw.expert_bias != null) return "an expert selection bias";
                if (mw.router_scale != null) return "a router norm scale";
                if (mw.shared_ungated) return "an ungated shared expert";
            },
            .dense => {},
        }
    }
    return null;
}

pub const Options = struct {
    /// Quantize a dense bf16 head to MXFP8 at load (the lossy `lm_head_mxfp8` setting).
    head_mxfp8: bool = false,
};

/// `config` is borrowed for the trunk's life; `weights` must outlive it.
pub fn load(io: std.Io, a: std.mem.Allocator, config: *const model.ModelConfig, weights: *model.Weights, opts: Options) !Trunk {
    const s = mlx.gpuStream();
    if (config.indexer_budget != 0) {
        qsaScoreFusedArm();
        qsaNaxArm();
    }
    const prefix = config.weight_prefix;
    var name_buf: [256]u8 = undefined;

    // Dense tables (float dtype) carry no scales; bias-less quant modes carry no biases.
    const bias_mandatory = config.quant_bits > 0 and config.quant_mode.hasBiases();
    const emb_w = weights_util.getNamedWeight(weights, &name_buf, prefix, "embed_tokens", "weight") orelse {
        log.err("MISSING WEIGHT: {s}.embed_tokens.weight\n", .{prefix});
        return error.MissingWeight;
    };
    const emb_dense = weights_util.floatDtypeTable(mlx.mlx_array_dtype(emb_w));
    const emb_s = if (emb_dense) mlx.mlx_array_new() else weights_util.getNamedWeight(weights, &name_buf, prefix, "embed_tokens", "scales") orelse {
        log.err("MISSING WEIGHT: {s}.embed_tokens.scales\n", .{prefix});
        return error.MissingWeight;
    };
    const emb_b = if (emb_dense) mlx.mlx_array_new() else weights_util.getNamedWeight(weights, &name_buf, prefix, "embed_tokens", "biases") orelse blk: {
        if (bias_mandatory) {
            log.err("MISSING WEIGHT: {s}.embed_tokens.biases\n", .{prefix});
            return error.MissingWeight;
        }
        break :blk mlx.mlx_array_new();
    };

    // A nested decoder's head is a sibling of `model`.
    var head_key_buf: [160]u8 = undefined;
    var head_key: ?[]const u8 = null;
    var lm_head_w: mlx.mlx_array = undefined;
    var lm_head_s: mlx.mlx_array = undefined;
    var lm_head_b: mlx.mlx_array = undefined;
    {
        const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
        if (weights_util.getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.weight", lm_prefix)) |w| {
            lm_head_w = w;
            const head_dense = weights_util.floatDtypeTable(mlx.mlx_array_dtype(w));
            lm_head_s = if (head_dense) mlx.mlx_array_new() else try weights_util.getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix);
            lm_head_b = if (head_dense)
                mlx.mlx_array_new()
            else if (bias_mandatory)
                try weights_util.getWeightFmt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix)
            else
                weights_util.getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix) orelse mlx.mlx_array_new();
            head_key = std.fmt.bufPrint(&head_key_buf, "{s}.lm_head.weight", .{lm_prefix}) catch null;
        } else if (weights.get("lm_head.weight")) |w| {
            lm_head_w = w;
            head_key = "lm_head.weight";
            lm_head_s = weights.get("lm_head.scales") orelse emb_s;
            lm_head_b = weights.get("lm_head.biases") orelse emb_b;
        } else if (config.tie_word_embeddings) {
            lm_head_w = emb_w;
            lm_head_s = emb_s;
            lm_head_b = emb_b;
        } else {
            log.err("MISSING WEIGHT: lm_head.weight\n", .{});
            return error.MissingWeight;
        }
    }
    if (opts.head_mxfp8) {
        if (head_key != null and lm_head_s.ctx == null) {
            lm_head_s = try quantizeHeadMxfp8(weights, head_key.?, lm_head_w, s);
            lm_head_w = weights.get(head_key.?).?;
            const shape = mlx.getShape(lm_head_s);
            log.info("[lm-head] mxfp8 (lossy): the {d}x{d} bf16 head quantized at load\n", .{ shape[0], shape[1] * 32 });
        } else log.info("[lm-head] mxfp8 declined: the head is {s}\n", .{if (lm_head_s.ctx != null) "already quantized" else "the embedding"});
    }

    const ml = try initMoeLayers(a, config.*, weights, &name_buf, s);
    var owned: std.ArrayList(mlx.mlx_array) = .fromOwnedSlice(ml.owned_bf16);
    errdefer {
        for (owned.items) |arr| _ = mlx.mlx_array_free(arr);
        owned.deinit(a);
    }
    errdefer a.free(ml.moe_layers);
    if (unservedLayout(config, ml.moe_layers)) |what| {
        log.err("[qwen4] this pack carries {s}, which the trunk does not run\n", .{what});
        return error.Qwen4UnservedLayout;
    }
    // The layers' recurrent state lives on each slot; the binder's template entries are not kept.
    for (ml.ssm_entries) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(e);
    }
    a.free(ml.ssm_entries);
    if (fused_rows.fused_row_groups > 0) log.info("[load] row-joined projection groups: {d}\n", .{fused_rows.fused_row_groups});
    fused_rows.fused_row_groups = 0;

    const ones_hidden = blk: {
        const one_val = weights_util.bf16Scalar(1.0, s);
        defer _ = mlx.mlx_array_free(one_val);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_full(&o, &[_]c_int{@intCast(config.hidden_size)}, 1, one_val, .bfloat16, s));
        break :blk o;
    };
    errdefer _ = mlx.mlx_array_free(ones_hidden);

    // YaRN: the mlx_fast_rope denominators, the [head_dim] mscale (attention_factor on the rotated dims) and the host
    // inverse frequencies the M-RoPE tables and the QSA pooled-key ladder are filled from.
    var rope_freqs_yarn: ?mlx.mlx_array = null;
    var yarn_mscale: ?mlx.mlx_array = null;
    var yarn_inv_freq: ?[]const f64 = null;
    errdefer if (yarn_inv_freq) |f| a.free(f);
    if (config.rope_yarn) {
        const spec = yarnSpec(config);
        const rotary_dim: u32 = config.yarnRotaryDim();
        const half: usize = rotary_dim / 2;
        const freqs_f64 = try a.alloc(f64, half);
        defer a.free(freqs_f64);
        computeYarnFreqs(freqs_f64, config.head_dim, config.yarnPartial(), config.rope_theta, config.yarn_factor, config.yarn_beta_fast, config.yarn_beta_slow, config.yarn_orig_max_pos, config.yarn_truncate);
        const inv_f64 = try a.alloc(f64, half);
        spec.invFreq(inv_f64);
        yarn_inv_freq = inv_f64;
        const freqs_f32 = try a.alloc(f32, half);
        defer a.free(freqs_f32);
        for (freqs_f64, 0..) |v, i| freqs_f32[i] = @floatCast(v);
        rope_freqs_yarn = mlx.mlx_array_new_data(freqs_f32.ptr, &[_]c_int{@intCast(half)}, 1, .float32);
        const mscale_f32 = try a.alloc(f32, config.head_dim);
        defer a.free(mscale_f32);
        for (mscale_f32, 0..) |*m, i| m.* = if (i < rotary_dim) config.yarn_attention_factor else 1.0;
        yarn_mscale = mlx.mlx_array_new_data(mscale_f32.ptr, &[_]c_int{@intCast(config.head_dim)}, 1, .float32);
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, rope_freqs_yarn.?);
        _ = mlx.mlx_vector_array_append_value(eval_vec, yarn_mscale.?);
        try mlx.check(mlx.mlx_eval(eval_vec));
        const cr = spec.correctionRange();
        log.info("[rope] YaRN: factor {d:.4} (window {d} -> {d}), rotary {d} dims, ramp [{d:.0}, {d:.0}], mscale {d:.6}\n", .{ config.yarn_factor, config.yarn_orig_max_pos, config.contextCap(), rotary_dim, cr.low, cr.high, config.yarn_attention_factor });
    }

    // One batched eval of every weight the trunk reads.
    {
        const eval_start = std.Io.Timestamp.now(io, .awake);
        const all_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(all_vec);
        inline for (.{ emb_w, emb_s, emb_b, lm_head_w, lm_head_s, lm_head_b }) |arr| {
            if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, arr);
        }
        for (ml.moe_layers) |lw| {
            if (lw.hc_attn) |*hw| weights_util.appendStructArrays(all_vec, hw);
            if (lw.hc_mlp) |*hw| weights_util.appendStructArrays(all_vec, hw);
            if (lw.ple) |*pw| weights_util.appendStructArrays(all_vec, pw);
            weights_util.appendHybridMlpWeights(all_vec, &lw.mlp);
            switch (lw.attn) {
                .full => |fa| weights_util.appendFullAttnWeights(all_vec, &fa),
                .linear => |la| weights_util.appendLinearAttnWeights(all_vec, &la),
            }
        }
        try mlx.check(mlx.mlx_eval(all_vec));
        const eval_ms: i64 = @intCast(@divTrunc(eval_start.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
        log.info("Batch eval all weights: {d}ms\n", .{eval_ms});
    }

    // The final hyper-connection read and the module-owned n-gram hash + mmapped table.
    var mb: [160]u8 = undefined;
    const mixer = try loadHcWeights(weights, std.fmt.bufPrint(&mb, "{s}.hyper_connection_mixer", .{prefix}) catch unreachable, false, config.hc_count, config.hidden_size, &owned, a, s);
    // The `0` is the PLE's ordinal among the config's injection points, not `ple_layer_idx`.
    try checkQsaCompressRatio(config.indexer_compress_ratio);
    const st = try a.create(qwen4_mod.Qwen4State);
    errdefer a.destroy(st);
    st.* = .{
        .hash = try qwen4_mod.NgramHash.init(config.vocab_size, config.ngram_size, config.heads_per_ngram, config.ngram_vocab_base, config.ngram_vocab_divisor, config.ngram_seed, 0, config.ngram_eos),
        .table = if (config.embedded_ple_payload_bytes != null)
            try qwen4_mod.NgramTable.openEmbedded(std.fs.path.dirname(config.ngram_table_path orelse return error.MissingNgramTable) orelse return error.MissingNgramTable, try model.qwen4EmbeddedSpec(config))
        else
            try qwen4_mod.NgramTable.open(config.ngram_table_path orelse return error.MissingNgramTable),
    };
    errdefer st.deinit();
    if (config.embedded_ple_payload_bytes) |bytes| {
        if (st.table.embeddedPayloadBytes() != bytes) return error.EmbeddedNgramTableChanged;
    }
    if (st.table.rows != st.hash.total_rows or st.table.dim * st.hash.n_heads != config.ple_embed_dim) {
        log.err("[qwen4] ngram_table.bin geometry {d}x{d} does not match the config ({d} rows, {d} heads x dim)\n", .{ st.table.rows, st.table.dim, st.hash.total_rows, st.hash.n_heads });
        return error.NgramTableMismatch;
    }
    var weights_bytes: usize = 0;
    _ = mlx.mlx_get_active_memory(&weights_bytes);
    // A pipeline rank without the PLE layer, or a tensor worker, never reads the table.
    if (config.pleLocalLayer() != null and !(if (config.dist_shard) |sh| sh.rank != 0 else false)) {
        st.gpu = ple_gpu.load(&st.table, ple_gpu.pleGpu().enabled, weights_bytes);
        if (st.gpu == null) st.table.calibrateArm();
        // The weights load just evicted the table from page cache; an embedded pack's GPU arm holds its own copy.
        if (st.gpu == null or st.table.embedded == null) st.table.startWarm();
    }
    // A pipeline split's head would read the last layer's stream, which lives on the worker.
    var head = if (config.dist_span != null) null else try loadQwen4Mtp(a, config.*, weights, &name_buf, s);
    errdefer if (head) |*h| h.deinit(a);
    log.info("[qwen4] n-gram table {d} rows x {d} ({d}-bit, mmapped), PLE at layer {d}, QSA budget {d}/{d}\n", .{ st.table.rows, st.table.dim, st.table.bits, config.ple_layer_idx, config.indexer_budget, config.indexer_compress_ratio });

    // Prism's ternary codec writes every affine bias as exactly -scale; only then do the 2-bit kernels serve.
    var ternary_2bit = false;
    if (ternaryKernelCandidate(config.hadamard_block, .of(config))) {
        // An embedding that is not also the lm_head is only ever gathered.
        const gather_only = if (lm_head_b.ctx == emb_b.ctx) mlx.mlx_array{ .ctx = null } else emb_b;
        if (try firstNonTernaryBias(weights, gather_only, s)) |key| {
            log.info("[qmv2] 2-bit pack is not ternary ({s}): stock qmm\n", .{key});
        } else {
            ternary_2bit = true;
            log.info("[qmv2] ternary 2-bit pack (bias == -scale): ternary kernels armed\n", .{});
        }
    }

    const head_n: c_int = blk: {
        const sh = mlx.getShape(lm_head_w);
        break :blk if (sh.len == 2) sh[0] else 0;
    };
    const calibrated = mtp_nax_profile.mtpNaxCalibratedModelFrom(config, head_n) and config.quant_group_size == 64 and config.quant_mode == .affine;
    const uniform_bits = config.quant_bits == 4 or config.quant_bits == 6 or config.quant_bits == 8;
    return .{
        .config = config.*,
        .s = s,
        .allocator = a,
        .emb_w = emb_w,
        .emb_s = emb_s,
        .emb_b = emb_b,
        .lm_head_w = lm_head_w,
        .lm_head_s = lm_head_s,
        .lm_head_b = lm_head_b,
        .moe_layers = ml.moe_layers,
        .ternary_2bit = ternary_2bit,
        .moe_owned_bf16 = try owned.toOwnedSlice(a),
        .qwen4_mixer = mixer,
        .qwen4 = st,
        .qwen4_mtp = head,
        .ones_hidden = ones_hidden,
        .one = weights_util.bf16Scalar(1.0, s),
        .rms_eps_arr = mlx.mlx_array_new_float(config.rms_norm_eps),
        .rope_freqs_yarn = rope_freqs_yarn,
        .yarn_mscale = yarn_mscale,
        .yarn_inv_freq = yarn_inv_freq,
        .dist_peer = if (config.dist_span) |sp| if (sp.first()) distributed_mod.peer() else null else null,
        .dist_tp = if (config.dist_shard) |sh| if (sh.rank == 0) distributed_mod.peer() else null else null,
        .mtp_uniform_affine_trunk = calibrated and uniform_bits and mtp_nax_profile.mtpNaxUniformAffineTrunkFrom(config, ml.moe_layers),
        .mtp_oqe_affine_trunk = calibrated and config.quant_bits == 4 and mtp_nax_profile.mtpNaxOqeAffineTrunkFrom(config, ml.moe_layers),
    };
}

test "qwen4_exp trunk: a layout the trunk does not run is refused by name, the plugin's own config is not" {
    const testing = std.testing;
    const toModelConfig = @import("model_config.zig").toModelConfig;
    const config = @import("config.zig");
    const mem: @import("../../sdk/testing/fixture.zig").MemDir = .{ .files = &.{} };
    const pd = mem.packDir();
    var p: @import("../../abi/plugin.zig").Parsed = .{};
    try testing.expect(config.parse(&pd, .of(config.tiny_text), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    var c = try toModelConfig(testing.allocator, config.of(&p), null);
    defer c.deinit(testing.allocator);
    try testing.expect(unservedLayout(&c, &.{}) == null);
    inline for (.{ .{ "hadamard_block", 512 }, .{ "norm_groups", 4 }, .{ "moe_sigmoid_router", true }, .{ "kda_vector_gate", true }, .{ "kda_sigmoid_out_gate", false }, .{ "swiglu_limit", 7.0 }, .{ "scale_embeddings", true } }) |m| {
        var bad = c;
        @field(bad, m[0]) = m[1];
        try testing.expect(unservedLayout(&bad, &.{}) != null);
    }
}
