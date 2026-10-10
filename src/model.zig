const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const mlx_gguf = @import("arch/mlx_gguf.zig");
const sushi_exl3 = @import("sushi_exl3");
const sushi_pack = @import("sushi_pack.zig");
const model_discovery = @import("model_discovery.zig");
const tokenizer_mod = @import("tokenizer.zig");
const qwen4_exp = @import("qwen4_exp.zig");
const qwen4_ple = @import("qwen4_ple.zig");
const kv_quant_mod = @import("kv_quant.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");

pub const HiddenAct = enum { gelu_approx, gelu, silu, relu_sq };

/// MLX quantization mode from config.json's `quantization.mode`. All
/// non-affine modes store NO `.biases` tensors (per-group fp8-encoded uint8
/// scales only) but share the packed-u32 weight layout, so supporting them is
/// a matter of skipping the biases fetch and passing the right mode string to
/// the mlx quantized ops. Tag names match the mlx-c mode strings exactly.
/// Upper bound on a vision tower's per-layer type table (muse ships 50).
pub const MAX_VISION_LAYERS = 64;

/// `MuseGlimmerImageProcessor.max_image_tokens` — MERGED tokens, not pixels.
pub const MUSE_MAX_IMAGE_TOKENS = 4096;

pub const QuantMode = enum {
    affine,
    nvfp4,
    mxfp4,
    mxfp8,
    /// Raw ggml blocks (lib/mlx-serve-gguf). Never reaches an MLX quantized op:
    /// the per-tensor type rides on the weight, see `mlx_gguf.kernels.Info`.
    gguf,

    pub fn fromString(name: []const u8) ?QuantMode {
        return std.meta.stringToEnum(QuantMode, name);
    }

    /// Mode string for mlx_quantized_matmul / mlx_gather_qmm / mlx_dequantize.
    pub fn cstr(self: QuantMode) [*:0]const u8 {
        return switch (self) {
            .affine => "affine",
            .nvfp4 => "nvfp4",
            .mxfp4 => "mxfp4",
            .mxfp8 => "mxfp8",
            .gguf => "gguf",
        };
    }

    /// Affine is the only mode whose checkpoints carry per-group biases.
    pub fn hasBiases(self: QuantMode) bool {
        return self == .affine;
    }
};

pub const LayerBlockType = enum { attention, gated_conv, mamba2, mlp, moe };

/// Sentence-transformers pooling operation for embedding requests (issue
/// #116): masked mean over real positions, the CLS token (position 0), or the
/// last real (non-padding) token. Every mode is followed by L2 normalization.
pub const PoolingMode = enum {
    mean,
    cls,
    last_token,

    pub fn fromString(s: []const u8) ?PoolingMode {
        if (std.mem.eql(u8, s, "mean")) return .mean;
        if (std.mem.eql(u8, s, "cls")) return .cls;
        if (std.mem.eql(u8, s, "last_token")) return .last_token;
        return null;
    }
};

/// Parse a sentence-transformers `1_Pooling/config.json`. Returns the pooling
/// mode when the file declares one we implement, null when the content isn't a
/// pooling config at all (malformed JSON, unrelated object — best-effort, like
/// generation_config.json), and `error.UnsupportedPoolingMode` when the file
/// DOES declare pooling but only modes we don't implement (weighted-mean,
/// max): serving those checkpoints mean-pooled would be silent corruption.
pub fn parsePoolingSidecar(content: []const u8) !?PoolingMode {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), content, .{}) catch return null;
    if (parsed != .object) return null;
    const obj = parsed.object;

    const getBool = struct {
        fn get(o: std.json.ObjectMap, key: []const u8) bool {
            if (o.get(key)) |v| {
                if (v == .bool) return v.bool;
            }
            return false;
        }
    }.get;
    // ST configs set exactly one mode true; check ours most-specific first.
    if (getBool(obj, "pooling_mode_lasttoken")) return .last_token;
    if (getBool(obj, "pooling_mode_cls_token")) return .cls;
    if (getBool(obj, "pooling_mode_mean_tokens")) return .mean;
    // ST 6 names the mode in one string instead of a boolean per mode.
    if (obj.get("pooling_mode")) |v| if (v == .string) {
        if (std.mem.eql(u8, v.string, "mean")) return .mean;
        if (std.mem.eql(u8, v.string, "cls")) return .cls;
        if (std.mem.eql(u8, v.string, "lasttoken")) return .last_token;
        return error.UnsupportedPoolingMode;
    };
    // Declares pooling, but none we support → refuse rather than mean-pool.
    var it = obj.iterator();
    while (it.next()) |e| {
        if (std.mem.startsWith(u8, e.key_ptr.*, "pooling_mode_")) return error.UnsupportedPoolingMode;
    }
    return null;
}

/// Known-family pooling fallback for checkpoints that ship neither an explicit
/// `pooling_mode` nor the ST sidecar (the mlx-community conversions strip it).
/// Gated on the arch so a directory name can never flip an unrelated model:
/// qwen3* named *embedding* → last-token (Qwen3-Embedding's contract), BERT
/// bge-/mxbai-embed → CLS (their model cards' contract). Everything else null
/// → the mean default.
pub fn poolingFromDirName(dir_basename: []const u8, model_type: []const u8) ?PoolingMode {
    var lower_buf: [256]u8 = undefined;
    if (dir_basename.len > lower_buf.len) return null;
    const lower = std.ascii.lowerString(&lower_buf, dir_basename);
    if (std.mem.startsWith(u8, model_type, "qwen3")) {
        if (std.mem.indexOf(u8, lower, "embedding") != null) return .last_token;
        return null;
    }
    if (std.mem.eql(u8, model_type, "bert")) {
        if (std.mem.indexOf(u8, lower, "bge-") != null) return .cls;
        if (std.mem.indexOf(u8, lower, "mxbai-embed") != null) return .cls;
        return null;
    }
    return null;
}

pub const Qwen4NormConvention = enum { delta, folded };

/// Llama 3.x `rope_type: "llama3"` (HF `_compute_llama3_parameters`).
pub const Llama3Rope = struct {
    factor: f32,
    low_freq_factor: f32 = 1.0,
    high_freq_factor: f32 = 4.0,
    original_max_position_embeddings: f32 = 8192.0,
};

pub const ModelConfig = struct {
    // Architecture identity
    model_type: []const u8 = "gemma3",
    weight_prefix: []const u8 = "language_model.model",

    // Core dimensions
    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    /// Whether `intermediate_size` came from the JSON or is the struct default
    /// above. MoE checkpoints routinely omit the key (DSV4 ships none at all),
    /// and a consumer that cannot tell the two apart bills the 15360 default as
    /// though the model declared it — 3.96 GB of phantom MLP transient in the
    /// prefill guard. Only readers that need the DISTINCTION should look here;
    /// everyone else keeps using `intermediate_size` and its fallback value.
    intermediate_size_declared: bool = false,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    rms_norm_eps: f32 = 1e-6,
    /// K2-Horizon: every RMS norm normalizes `hidden_size / norm_groups`-wide channel groups on their own rms, then applies the full weight.
    norm_groups: u32 = 1,
    /// Prism Hadamard packs: every `<linear>.signs` weight reads `H_block(signs * x)` (rht.zig); 0 = none.
    hadamard_block: u32 = 0,

    // RoPE
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,
    rope_proportional: bool = false, // Gemma 4: full attention uses proportional RoPE
    rope_proportional_factor: f32 = 1.0,
    rope_llama3: ?Llama3Rope = null,

    // Sliding window attention
    has_sliding_window: bool = true,
    sliding_window: u32 = 1024,
    sliding_window_pattern: u32 = 6,

    // Quantization. 0 = dense bf16 (config.json has no "quantization" key);
    // quantized checkpoints always set this from that key (see parseConfig).
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,

    // Attention scale: 1/sqrt(query_pre_attn_scalar) for Gemma, 1/sqrt(head_dim) for others
    query_pre_attn_scalar: u32 = 256,

    // Architectural differences between model families
    tie_word_embeddings: bool = false,
    hidden_act: HiddenAct = .gelu_approx,
    norm_has_offset: bool = true,
    scale_embeddings: bool = true,
    has_pre_ff_norm: bool = true,
    has_qk_norm: bool = true,
    // Learned per-head attention sinks (gpt_oss `self_attn.sinks`, [n_heads]):
    // one extra logit column that lands in the softmax DENOMINATOR only, so
    // every head can attend to "nothing". mlx's fused SDPA takes them
    // natively (`mlx_fast_scaled_dot_product_attention`'s `sinks` argument);
    // this flag is what makes the loader fetch the weight and the forward
    // pass it instead of the null array.
    has_attn_sinks: bool = false,
    // gpt_oss clamped SwiGLU. Non-zero limit selects
    //   clip(gate, max=limit) * sigmoid(alpha*gate) * (clip(up, ±limit) + 1)
    // over the standard silu(gate)*up. The `+ 1` on the linear branch and the
    // asymmetric clip are both load-bearing; `hidden_act` in a gpt_oss
    // config.json says "silu" and is never read by the reference.
    swiglu_limit: f32 = 0.0,
    swiglu_alpha: f32 = 1.702,
    /// GLM-5-Next's form under the same limit: silu(min(gate, limit)) * clip(up, ±limit).
    swiglu_clamp: bool = false,

    // MoE
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    // DeepSeek-V3-style sigmoid routing (hy_v3): scores = sigmoid(logits) in
    // f32; top-k SELECTED on scores + expert_bias but WEIGHTED by the unbiased
    // scores; optional renorm (/(sum+1e-20)) then × router_scaling_factor.
    moe_sigmoid_router: bool = false,
    // kolibri1: top-k SELECTED on logits + expert_bias (no sigmoid before the
    // bias), WEIGHTED by sigmoid(logits) of the picks; requires moe_sigmoid_router.
    moe_logit_select: bool = false,
    moe_route_norm: bool = true,
    router_scaling_factor: f32 = 1.0,
    // Layers [0, first_k_dense_replace) use a dense MLP instead of MoE
    // (hy_v3: layer 0 dense at intermediate_size, the rest MoE).
    first_k_dense_replace: u32 = 0,
    // Laguna: MoE router logit soft-capping (tanh). 0 = off (the shipped
    // Laguna-S-2.1 checkpoint sets moe_router_logit_softcapping: 0.0).
    moe_router_logit_softcapping: f32 = 0.0,

    // Grouped ("noaux_tc") expert routing: split the biased scores into
    // moe_n_group equal groups, keep the moe_topk_group best by their top-2
    // sum, then take the global top-k inside the survivors. 1/1 = ungrouped.
    moe_n_group: u32 = 1,
    moe_topk_group: u32 = 1,

    // Linear attention (GatedDeltaNet)
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,

    // KDA (Kimi Delta Attention, bailing_hybrid) variations on the
    // GatedDeltaNet recurrence:
    //   - the forget gate is PER CHANNEL ([B,T,H,Dk]) rather than per head, so
    //     the fused kernel indexes `g` by the key channel (kda_vector_gate);
    //   - a non-zero lower bound replaces the softplus gate entirely with
    //     `g = bound * sigmoid(exp(A_log) * (a + dt_bias))`, which is bounded
    //     in (bound, 0) instead of (-inf, 0) — it is NOT a clamp on the
    //     softplus form (fla/ops/kda/fused_recurrent.py);
    //   - the output gate is a plain sigmoid, not SiLU/swish.
    kda_vector_gate: bool = false,
    kda_gate_lower_bound: f32 = 0.0, // 0 = plain -exp(A_log)·softplus form
    kda_sigmoid_out_gate: bool = false,

    // Multi-head Latent Attention (bailing_hybrid's full-attention layers,
    // DeepSeek-V3 shape): low-rank Q (q_a_proj → q_a_layernorm → q_b_proj) and
    // a single compressed KV latent (kv_a_proj_with_mqa → kv_lora_rank latent
    // + qk_rope_head_dim shared rope key) expanded per head by kv_b_proj.
    // Query/key head dim is nope+rope; the value head dim is SMALLER, so the
    // KV cache holds asymmetric K/V (MLX's SDPA has a 192/128 vector kernel).
    // mla_head_gate: per-head sigmoid gate on the attention output (the
    // checkpoint's `head_wise` gated_attention_proj_granularity_type).
    mla_q_lora_rank: u32 = 0, // 0 = not an MLA arch
    mla_kv_lora_rank: u32 = 0,
    mla_qk_nope_head_dim: u32 = 0,
    mla_qk_rope_head_dim: u32 = 0,
    mla_v_head_dim: u32 = 0,
    mla_head_gate: bool = false,
    // GLM-5-Next DeepSeek sparse attention: absorbed NoPE MLA over a `dsa_kv_lora_rank`
    // latent (per-head q/k and v widths `dsa_head_dim`) plus a pooled indexer that picks
    // `dsa_index_topk` tokens in `dsa_index_kpool`-token pools.
    dsa_q_lora_rank: u32 = 0,
    dsa_kv_lora_rank: u32 = 0,
    dsa_head_dim: u32 = 0,
    dsa_index_topk: u32 = 0,
    dsa_index_kpool: u32 = 0,
    dsa_index_heads: u32 = 0,
    dsa_index_head_dim: u32 = 0,
    // RoPE rotates ADJACENT PAIRS (x[2i], x[2i+1]) instead of halves — mlx's
    // `traditional` rope. Set by rope_interleave.
    rope_interleaved_pairs: bool = false,

    // Hybrid attention
    full_attention_interval: u32 = 0,
    // Hybrid archs: layers at or past this index are ALWAYS full
    // attention, whatever the interval says. The reference's rule is
    // `(idx+1) % group == 0 OR idx >= n_layers // group * group`, i.e. the
    // ragged tail after the last WHOLE group never gets linear attention.
    // 0 = no tail bound, which is every other hybrid arch (qwen3_next, lfm2).
    linear_attn_tail_from: u32 = 0,
    partial_rotary_factor: f32 = 1.0,
    attn_output_gate: bool = false,

    // Qwen4-Exp (Qwen3.8-Flash-Next): gated residual streams ("hyper
    // connections", hc_count x hidden wide), a hashed n-gram embedding
    // injected at ONE layer (PLE), and Qwen Sparse Attention (indexer-selected
    // 4-token blocks past `indexer_budget` tokens). hc_count 0 = none.
    hc_count: u32 = 0,
    hc_lowrank: u32 = 0,
    ple_layer_idx: i32 = -1, // 0-based; the config lists 1-based ids
    ple_embed_dim: u32 = 0,
    ple_conv_kernel: u32 = 4,
    ngram_size: u32 = 3,
    heads_per_ngram: u32 = 8,
    ngram_vocab_base: u64 = 20_000_000,
    ngram_vocab_divisor: u32 = 128,
    ngram_seed: u64 = 1234,
    split_ngram_parts: u32 = 128,
    embedded_ple_payload_bytes: ?u64 = null,
    qwen4_norm_convention: ?Qwen4NormConvention = null,
    indexer_n_heads: u32 = 0, // 0 = dense attention
    indexer_head_dim: u32 = 0,
    indexer_budget: u32 = 0,
    indexer_compress_ratio: u32 = 0,
    /// Routed experts in EXL3 (Sushi packs, `expert_quant` in config.json),
    /// served by lib/sushi; null = the pack's own affine banks.
    exl3: ?sushi_exl3.Spec = null,
    /// The TEXT config's own eos (its first entry): the n-gram hash's segment
    /// reset token, independent of the generation-time stop set.
    ngram_eos: u32 = 0,
    /// `<model_dir>/ngram_table.bin` for the PLE table (mmapped by the
    /// engine, never mlx-loaded). Set by `parseConfig`; lives as long as the
    /// config does.
    ngram_table_path: ?[]const u8 = null,
    /// deepseek_v41: the pack's directory, where its Engram tables sit beside
    /// the weights. Set by `parseConfig`.
    dsv41_dir: ?[]const u8 = null,

    // Laguna: softplus per-head attention output gate. self_attn.g_proj →
    // softplus(fp32) → per-head scalar × attn output (reshaped [..,H,D]) before
    // o_proj. Distinct from attn_output_gate (qwen3-next sigmoid + doubled
    // q_proj) — separate weight, softplus activation, per-head broadcast.
    laguna_attn_gate: bool = false,
    // Laguna: per-layer Q-head count (full-attention layers 48, sliding 72;
    // KV heads uniform at num_key_value_heads). 0 = uniform num_attention_heads.
    num_attention_heads_per_layer: [128]u32 = @splat(0),
    has_per_layer_heads: bool = false,

    // MuseGlimmer (muse_glimmer): weight-less shared QK RMS-norm with Q scaled
    // by qk_scale_factor (folded into attnScale — a scalar commutes through
    // RoPE and QK^T), elementwise sigmoid attention output gate from a
    // separate self_attn.gate_proj read off the post-input-norm hidden,
    // RMS-normed embeddings (NO sqrt(hidden) scale), Gemma2-centered sandwich
    // norms whose POST norms use post_norm_eps while the FINAL norm is
    // plain-scale (ones-init), logits scaled by output_multiplier before the
    // tanh softcap, and NoPE on layers whose layer_rope_theta entry is 0
    // (exactly the full-attention layers in the released checkpoint).
    qk_scale_factor: f32 = 0.0, // 0 = off
    output_multiplier: f32 = 0.0, // 0 = off
    post_norm_eps: f32 = 0.0, // 0 = same as rms_norm_eps
    final_norm_plain: bool = false, // final norm skips the (1+w) fold
    qk_norm_weightless: bool = false, // param-free RMS on Q and K heads
    normed_embeddings: bool = false, // param-free RMS after embedding lookup
    attn_sigmoid_gate: bool = false, // attn_out *= sigmoid(gate_proj(normed))
    attn_gate_headwise: bool = false, // the gate is ONE scalar per head (g_proj [heads, hidden])
    attn_fused_qkv: bool = false, // checkpoint ships self_attn.q_k_v_proj = [q | k | v] rows
    layer_no_rope: [128]bool = @splat(false),

    // Laguna YaRN RoPE (full-attention layers only; sliding layers use default
    // RoPE at rope_local_base_freq). rope_yarn gates the freqs + mscale
    // precompute at model load; the sliding/full split is by isGlobalLayer.
    rope_yarn: bool = false,
    yarn_factor: f32 = 1.0,
    yarn_orig_max_pos: u32 = 0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
    yarn_attention_factor: f32 = 1.0,
    /// HF's `truncate` (default true): floor/ceil the ramp correction bounds to
    /// whole dims. Only the flat-`rope_parameters` readers set it.
    yarn_truncate: bool = true,

    // Inkling (inkling_mm_model, Thinking Machines Inkling Small). NO RoPE:
    // position = the RelativeLogits bias (per-layer wr_du → [heads, d_rel]
    // relative states × a learned [d_rel, extent] profile bank → additive bias
    // over backward distances) + four depthwise causal short-convolutions per
    // layer + log-scaling on global layers past inkling_log_n_floor tokens.
    inkling_d_rel: u32 = 0, // 0 = not an inkling arch
    inkling_rel_extent: u32 = 0, // global-layer bias extent; sliding layers use their window
    inkling_log_n_floor: u32 = 0, // 0 = log-scaling off (exact no-op below the floor)
    inkling_log_alpha: f32 = 0.1,
    inkling_sconv_kernel: u32 = 0, // 0 = no short convolutions
    // Router-gated stacked shared experts (the routing "sink": their weights
    // come from the same softmax as the routed top-k). Distinct from qwen/hy3
    // shared experts (ungated always-added).
    inkling_n_shared_experts: u32 = 0,
    // muP logit scaling: hidden /= this before the unembed matmul (1 = off).
    logits_mup_width_multiplier: f32 = 1.0,
    // Slice logits to the first N rows (vocab padding; 0 = full vocab).
    unpadded_vocab_size: u32 = 0,

    // DeepSeek V4 Flash (deepseek_v4). MQA over ONE head_dim-wide latent
    // (num_key_value_heads == 1): low-rank Q (wq_a → q_norm → wq_b, then an
    // UNWEIGHTED per-head RMS), grouped low-rank O (o_groups slabs of
    // o_lora_rank), rope on the last dsv4_rope_head_dim dims with INVERSE
    // rope on the attention output; per-head attn_sink joins the softmax
    // denominator only. Every layer slides over the last sliding_window raw
    // latents; layers with dsv4_compress_ratios[i] != 0 add learned
    // gated-pooling compression of the history (ratio 4 = overlapping windows
    // + a top-dsv4_index_topk indexer over its own fp4/Hadamard-simulated
    // compressed keys; other ratios plain, all compressed slots visible).
    // YaRN applies ONLY on compressed layers at dsv4_compress_rope_theta
    // (ratio-0 layers run plain rope_theta, no yarn, and there is NO yarn
    // mscale anywhere — the reference applies none). The residual stream is
    // dsv4_hc_mult copies mixed per token by Sinkhorn-normalized
    // hyper-connections. The first dsv4_hash_layers MoE layers route by
    // TOKEN ID (gate.tid2eid). Reference: the release's own
    // inference/{model,kernel}.py; full notes in memory dsv4-port.
    dsv4_q_lora_rank: u32 = 0, // 0 = not a deepseek_v4 arch
    dsv4_o_lora_rank: u32 = 0,
    dsv4_o_groups: u32 = 0,
    dsv4_rope_head_dim: u32 = 0,
    dsv4_hash_layers: u32 = 0,
    dsv4_index_n_heads: u32 = 0,
    dsv4_index_head_dim: u32 = 0,
    dsv4_index_topk: u32 = 0,
    dsv4_hc_mult: u32 = 0,
    dsv4_hc_sinkhorn_iters: u32 = 0,
    dsv4_hc_eps: f32 = 1e-6,
    dsv4_swiglu_limit: f32 = 0.0,
    dsv4_compress_rope_theta: f32 = 0.0,
    // Per-layer compression ratio (0 = pure sliding window). Entries beyond
    // num_hidden_layers describe the MTP module(s).
    dsv4_compress_ratios: [128]u8 = @splat(0),
    dsv4_n_compress_ratios: u32 = 0,
    dsv4_mtp_layers: u32 = 0,
    // DSpark block-parallel speculative decoding (0731 and later). The draft
    // stages live under the SAME `mtp.*` namespace as the preview's single
    // MTP module — `dspark_block_size != 0` is what tells the two apart:
    // stage 0 projects the concatenated hidden states of
    // `dspark_target_layer_ids` (main_proj/main_norm, replacing the preview's
    // e_proj/h_proj) and drafts a whole block of `dspark_block_size` slots
    // seeded with `dspark_noise_token_id`, the last stage adding a rank-
    // `dspark_markov_rank` bigram bias plus a confidence head. Parsed here so
    // the engine can tell a DSpark checkpoint from a preview one BEFORE
    // touching weights; the draft path itself is not wired yet.
    dsv4_dspark_block_size: u32 = 0,
    dsv4_dspark_noise_token_id: u32 = 0,
    dsv4_dspark_markov_rank: u32 = 0,
    dsv4_dspark_target_layers: [8]u8 = @splat(0),
    dsv4_n_dspark_target_layers: u32 = 0,

    // DeepSeek-V4.1 (deepseek_v41) on top of the dsv4_* block: only the
    // `kv_sources` layers compress KV and only the `index_sources` run an
    // indexer; every other compressed layer reads the latest source at or
    // above it. The candidate source pools its scores into blocks that later
    // index sources mask with. Engram adds an n-gram table lookup to the
    // stream before its layers; the DSpark head has its own MoE size.
    dsv41_kv_sources: u64 = 0, // bit i = layer i
    dsv41_index_sources: u64 = 0,
    dsv41_candidate_source: i32 = -1,
    dsv41_candidate_topk_blocks: u32 = 0,
    dsv41_candidate_block_size: u32 = 0,
    dsv41_engram_layers: [4]u8 = @splat(0),
    dsv41_engram_rows: [4]u64 = @splat(0),
    dsv41_n_engram_layers: u32 = 0,
    dsv41_engram_max_ngram: u32 = 0,
    dsv41_engram_vocab_size: u64 = 0,
    dsv41_engram_n_heads: u32 = 0,
    dsv41_engram_head_dim: u32 = 0,
    dsv41_engram_pad_id: u32 = 0,
    dsv41_engram_compressed_vocab: u32 = 0,
    dsv41_dspark_experts: u32 = 0,
    dsv41_dspark_top_k: u32 = 0,
    /// The pack's routed experts are an EXL3 bank (`experts.bin`, the OpensourceWTF
    /// streaming repack): mlx-stream serves it (`arch/mlx_stream.zig`) and loads its
    /// weights. Set by `parseConfig`.
    dsv41_stream: bool = false,

    // BERT encoder-only
    is_encoder_only: bool = false,
    layer_norm_eps: f32 = 1e-12,
    type_vocab_size: u32 = 0,

    /// Sentence-transformers pooling for /v1/embeddings (issue #116). null =
    /// no explicit signal → masked mean (the historical behavior, correct for
    /// MiniLM-class BERTs and EmbeddingGemma). Set from config.json
    /// `pooling_mode`, the ST `1_Pooling/config.json` sidecar, or the
    /// known-family name fallback (`poolingFromDirName`). A non-null mode on a
    /// decoder arch (Qwen3-Embedding) also advertises the `embeddings`
    /// capability WITHOUT flipping `is_encoder_only` — the forward stays the
    /// arch's own causal pass.
    pooling_mode: ?PoolingMode = null,

    // Bidirectional-attention embedding models (EmbeddingGemma): a decoder
    // arch (gemma3_text) trained as an encoder. Implies is_encoder_only.
    use_bidirectional_attention: bool = false,
    // BOS id from config.json (embedding models wrap inputs <bos>…<eos>).
    bos_token_id: ?u32 = null,

    // Context length from config.json (0 = unknown)
    max_position_embeddings: u32 = 0,

    /// Auto-context, FROZEN at model-load time (`server.pinAutoContext`).
    /// 0 = not pinned yet.
    ///
    /// Without `--ctx-size` the effective context used to be recomputed from
    /// LIVE memory on every request, so the number the server advertised drifted
    /// as other processes took RAM (measured: 92,387–94,883 across one session).
    /// Agent CLIs budget their own `max_tokens` against that advertised value,
    /// so it has to hold still for the model's whole residency. Explicit
    /// `--ctx-size` still wins over this.
    pinned_context: u32 = 0,

    /// Per-model settings from `model-settings.json`, set at the load
    /// construction site. `ctx_override` 0 = the process `--ctx-size`/auto;
    /// `kv_quant_override` null = the process `--kv-quant`; `mtp_override`
    /// true = head loaded AND on by default (the `--mtp` force, per model).
    ctx_override: u32 = 0,
    kv_quant_override: ?kv_quant_mod.KVQuantConfig = null,
    mtp_override: ?bool = null,
    /// null = the process `--mtp-typical`/`--mtp-tokenv3` (exact when neither).
    mtp_acceptance_override: ?mtp_acceptance_mod.Mode = null,
    /// null = the process `--mtp-greedy-tail` (off when absent).
    mtp_greedy_tail_override: ?bool = null,
    /// null = the process default (`MLX_SERVE_BONSAI_INT8_PREFILL`).
    int8_prefill_override: ?bool = null,
    /// Dense context K/V a loaded DFlash drafter keeps per trunk token, per request. Stamped at load.
    drafter_ctx_bytes_per_token: u64 = 0,
    /// `drafter` setting: null or "auto" = the in-dir probe, "off", or a path. Owned.
    drafter_override: ?[]const u8 = null,
    /// Every quantized projection fits the row-exact kernels (`Transformer.init`).
    row_exact_covered: bool = false,
    /// A DFlash drafter is bound (`DflashModel.bind`): its block tree needs exact verify rows.
    dflash_bound: bool = false,

    /// The prefill chunk this model was sized for, FROZEN at load
    /// (`server.pinPrefillChunk`). 0 = not pinned yet, which keeps the
    /// launch/base chunk.
    ///
    /// The chunk is the multiplier on the biggest transient in the memory bill
    /// (`8 x chunk x max(hidden, ffn) x 2`, three of them). Nothing used to size
    /// it to the MACHINE, so a 16 GB Mac reserved the same 5-7 GB envelope a
    /// 128 GB one does, which is most of its budget: the sizer then reported a
    /// 1024-token context and the admission guard refused prompts whose real
    /// peak was a third of the bill. The sizer, `checkAttentionMemory` and
    /// `generate.effectivePrefillChunk` all read THIS field, so the bill and the
    /// forward can never disagree. Explicit `--prefill-chunk` still wins.
    pinned_prefill_chunk: u32 = 0,

    // Stop tokens (populated from config.json)
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,

    // Model-author sampling recommendations from generation_config.json
    // (e.g. Qwen 3.6: temp 1.0 / top_p 0.95 / top_k 20; Gemma 4: top_k 64).
    // null = the file or key is absent. Used as defaults for request fields
    // the client OMITTED — Claude Code sends no sampling params at all, and
    // pre-2026-06 it sampled the full untruncated distribution at temp 1.0,
    // well outside the model card's intended envelope.
    gen_temperature: ?f32 = null,
    gen_top_p: ?f32 = null,
    gen_top_k: ?u32 = null,
    gen_min_p: ?f32 = null,

    // The checkpoint's OWN thinking default, from generation_config.json's
    // `default_chat_template_kwargs.enable_thinking`. null = the file or key
    // is absent. Read by `defaultEnableThinking` for requests that name no
    // thinking preference; an explicit request value still outranks it.
    gen_enable_thinking: ?bool = null,

    // Gemma 4: explicit layer type map (bit = 1 means full/global attention)
    has_explicit_layer_types: bool = false,
    layer_is_global: [128]bool = @splat(false),

    // Vision encoder (Gemma 4 SigLIP)
    has_vision: bool = false,
    /// EmbeddingGemma 2's processors (`processor_config.json`): soft tokens per image and per video FRAME, and
    /// the most frames kept from a video. The processors' own defaults until the file says otherwise.
    gv_image_tokens: u32 = 280,
    gv_video_tokens: u32 = 70,
    gv_video_frames: u32 = 32,
    vision_hidden_size: u32 = 768,
    vision_num_layers: u32 = 16,
    vision_num_heads: u32 = 12,
    vision_head_dim: u32 = 64,
    vision_intermediate_size: u32 = 3072,
    vision_patch_size: u32 = 16,
    vision_pooling_kernel: u32 = 3,
    vision_soft_tokens: u32 = 280,
    vision_position_embedding_size: u32 = 10240,
    vision_rope_theta: f32 = 100.0,
    vision_use_clipped_linears: bool = true,
    image_token_id: u32 = 0, // 0 = no image token
    boi_token_id: u32 = 0, // beginning of image
    eoi_token_id: u32 = 0, // end of image

    // Gemma 4 12B "unified" (encoder-free) multimodal. Instead of the SigLIP
    // transformer tower, vision is a single patch embedder
    // (LN → Dense → LN → +factorized 2D posemb → LN → RMSNorm → Linear) and
    // audio is raw 640-sample frames projected straight to text space. Set
    // when model_type is gemma4_unified*. See src/vision.zig (UnifiedEmbedder).
    is_gemma4_unified: bool = false,
    vision_mm_embed_dim: u32 = 0, // unified: mm_embed_dim (3840 for 12B = text hidden)
    vision_model_patch_size: u32 = 0, // unified: 48px merged "model patch" (16px teacher × 3 pool)
    vision_mm_posemb_size: u32 = 0, // unified: factorized position table size per axis (1120)
    // Audio (gemma4_unified). embed_audio projects audio_embed_dim → text hidden.
    audio_token_id: u32 = 0, // 0 = no audio token
    boa_token_id: u32 = 0, // beginning of audio
    eoa_token_id: u32 = 0, // end of audio
    audio_embed_dim: u32 = 0, // unified: raw samples per token (640)
    audio_samples_per_token: u32 = 640, // 40ms @ 16kHz

    // Qwen3.5/3.6 vision (Qwen3-VL ViT). Distinct from the Gemma SigLIP fields
    // above: Qwen ships a fused-qkv ViT with a patch merger, and the text trunk
    // uses INTERLEAVED M-RoPE (image tokens get 2D grid positions). Populated in
    // the qwen3_5 arm below. Encoder: src/qwen_vision.zig; M-RoPE: src/mrope.zig.
    qwen_vision: bool = false,
    qv_depth: u32 = 0, // ViT transformer blocks
    qv_hidden: u32 = 0, // ViT hidden size
    qv_heads: u32 = 0, // ViT attention heads
    qv_head_dim: u32 = 0, // = qv_hidden / qv_heads
    qv_intermediate: u32 = 0, // ViT MLP intermediate
    qv_patch: u32 = 16, // pixel patch size
    qv_temporal_patch: u32 = 2, // frames folded per patch (still image duplicated)
    qv_merge: u32 = 2, // spatial merge: merge×merge patches → one LLM token
    qv_num_pos_emb: u32 = 0, // learned pos table entries (e.g. 2304 = 48×48)
    qv_out_hidden: u32 = 0, // merger output dim (= text hidden_size)
    // Image-area bounds from processor_config.json / preprocessor_config.json.
    // 0 means absent: the Qwen processor defaults remain the fallback.
    qv_min_pixels: u32 = 0,
    qv_max_pixels: u32 = 0,
    // Muse-Glimmer vision (src/muse_vision.zig). Shares the qv_* geometry but
    // NOT the Qwen ViT: split qkv, learned pos table resampled per image,
    // window/full attention per layer, and plain 1D text positions (no M-RoPE).
    muse_vision: bool = false,
    mv_pos_side: u32 = 0, // learned pos table is pos_side x pos_side
    mv_projector_hidden: u32 = 0, // vision_adapter width
    mv_ln_eps: f32 = 1e-5,
    mv_rope_theta: f64 = 10000.0,
    mv_max_image_tokens: u32 = 0, // processor cap, in MERGED tokens
    mv_full_attn: [MAX_VISION_LAYERS]bool = @splat(false),
    // LFM2-VL vision (src/lfm2_vision.zig). The tower's geometry comes from the
    // generic vision_* fields above (it is a stock SigLIP2); these are LFM2-VL's
    // own wrapper — the projector, and the NaFlex processor's token budget.
    lfm2_vision: bool = false,
    lv_pos_side: u32 = 0, // learned pos table is pos_side x pos_side
    lv_downsample: u32 = 2, // projector pixel-unshuffle factor
    lv_projector_hidden: u32 = 0,
    lv_ln_eps: f32 = 1e-6,
    lv_min_image_tokens: u32 = 64,
    lv_max_image_tokens: u32 = 256,
    lv_tile_size: u32 = 512,
    lv_min_tiles: u32 = 2,
    lv_max_tiles: u32 = 10,
    lv_split_images: bool = true,
    lv_use_thumbnail: bool = true,
    lv_pixels_tolerance: f32 = 2.0,
    lv_thumbnail_token_id: u32 = 0,
    lv_row_col_base_id: u32 = 0, // id of `<|img_row_1_col_1|>`; the block is row-major
    // Interleaved M-RoPE sections [t, h, w]; sum = rotary_dim/2 (e.g. [11,11,10]).
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
    // Qwen vision token ids (top-level config.json). image_token_id reuses the
    // shared field above (parsed generically at the image_token_id block).
    video_token_id: u32 = 0,
    vision_start_token_id: u32 = 0,
    vision_end_token_id: u32 = 0,

    // Gemma 4: dual head dimensions and KV sharing
    global_head_dim: u32 = 0, // 0 = same as head_dim
    v_head_dim: u32 = 0, // 0 = same as the layer's head_dim (MiMo: K 192 / V 128)
    attention_value_scale: f32 = 1.0, // V multiplied before caching (MiMo-V2.6: 0.707)
    num_global_key_value_heads: u32 = 0, // 0 = same as num_key_value_heads
    num_kv_shared_layers: u32 = 0,
    final_logit_softcapping: f32 = 0.0, // 0 = disabled
    hidden_size_per_layer_input: u32 = 0, // >0 enables PLE
    partial_rotary_factor_global: f32 = 1.0, // for global/full attention layers
    has_v_norm: bool = false, // parameter-free RMS norm on values
    // Gemma 4 (31B): full_attention layers share V with K (no v_proj stored)
    attention_k_eq_v: bool = false,

    // Block diffusion (DiffusionGemma). canvas_length > 0 marks a diffusion
    // checkpoint: generation runs the canvas-denoising loop in
    // src/diffusion.zig instead of autoregressive decode. The knobs mirror
    // the checkpoint's embedded `generation_config` object; defaults match
    // google/diffusiongemma-26B-A4B-it.
    canvas_length: u32 = 0,
    diffusion_max_steps: u32 = 48,
    diffusion_t_min: f32 = 0.4,
    diffusion_t_max: f32 = 0.8,
    diffusion_entropy_bound: f32 = 0.1,
    diffusion_confidence_threshold: f32 = 0.005,
    diffusion_stability_threshold: u32 = 1,
    diffusion_pad_token: u32 = 0,

    // Hybrid layers (LFM2, Nemotron-H): per-layer type dispatch
    has_hybrid_layers: bool = false,
    layer_block_types: [128]LayerBlockType = @splat(.attention),
    has_embedding_norm: bool = false, // LFM2: RMS norm applied to embeddings
    has_final_norm: bool = true, // false for LFM2 (no model.norm.weight)

    // LFM2 gated convolution
    lfm_conv_kernel: u32 = 3,
    /// LFM2.5-8B-A1B (`lfm2_moe`): the hybrid trunk's per-layer MLP is a
    /// sparse MoE from `num_dense_layers` on. `model_type` collapses to
    /// "lfm2" (same conv/attention mixers), so this flag is what tells the
    /// layer loader which feed-forward to bind.
    lfm2_moe: bool = false,
    /// First N layers keep a DENSE feed-forward; the rest are MoE.
    num_dense_layers: u32 = 0,
    lfm_conv_dim: u32 = 0, // 0 = hidden_size

    // Mamba2 SSM (Nemotron-H)
    mamba_num_heads: u32 = 0,
    mamba_head_dim: u32 = 0,
    mamba_n_groups: u32 = 8,
    ssm_state_size: u32 = 128,
    mamba_conv_kernel: u32 = 4,
    mamba_expand: u32 = 2,
    time_step_min: f32 = 0.0,
    time_step_max: f32 = std.math.inf(f32),
    mamba_chunk_size: u32 = 256,
    mamba_mlp_act: HiddenAct = .relu_sq, // Nemotron-H MLP uses ReLU^2

    /// How many keys ONE query actually reads during prefill at this prompt
    /// length. `seq` (dense causal) unless the architecture BOUNDS its
    /// attention — the prefill admission guard's score term multiplies by this,
    /// and billing a dense key axis for a sparse arch is a spurious 400.
    ///
    /// deepseek_v4 reads a raw sliding window plus at most ONE compressed arm
    /// per layer (`deepseek_v4.zig`: `tk = wk + n_sel`, `wk = @min(m.window,
    /// seq_total)`), so the widest layer is what the guard must bill:
    ///   - `compress_ratios[i] == 0` → window only
    ///   - `== 4` → top-`index_topk` of the `seq/4` compressed slots (the
    ///     literal 4 mirrors the engine's own `if (ratio == 4)` branch)
    ///   - otherwise → ALL `seq/ratio` slots, visibility-masked
    /// plus one sink column. The all-visible arm is seq-scaled, so it overtakes
    /// the top-k arm at long context and the bound must track it rather than
    /// freezing at `index_topk`. A checkpoint declaring no ratios stays dense —
    /// an arch we cannot bound must never be billed as though we had.
    pub fn prefillAttnKeys(self: *const ModelConfig, seq: u64) u64 {
        if (self.isDsv41()) return @min(seq, @as(u64, self.sliding_window) + self.dsv4_index_topk + 1);
        if (!std.mem.eql(u8, self.model_type, "deepseek_v4")) return seq;
        const n = @min(self.dsv4_n_compress_ratios, self.dsv4_compress_ratios.len);
        if (n == 0) return seq;
        const window: u64 = @min(@as(u64, self.sliding_window), seq);
        var widest: u64 = 0;
        for (self.dsv4_compress_ratios[0..n]) |r| {
            if (r == 0) continue;
            const slots: u64 = seq / r;
            const cols: u64 = if (r == 4) @min(@as(u64, self.dsv4_index_topk), slots) else slots;
            widest = @max(widest, cols);
        }
        return @min(seq, window + widest + 1);
    }

    pub fn isGlobalLayer(self: ModelConfig, layer_idx: u32) bool {
        if (!self.has_sliding_window) return true;
        if (self.has_explicit_layer_types and layer_idx < 128) {
            return self.layer_is_global[layer_idx];
        }
        // HF/mlx-lm convention (Gemma 3): the GLOBAL layer closes each group —
        // global when `(idx + 1) % pattern == 0` (layers 5, 11, … for pattern 6).
        return (layer_idx % self.sliding_window_pattern) == self.sliding_window_pattern - 1;
    }

    /// For Gemma 4 KV sharing: get the source layer index for a shared layer.
    /// Returns null if the layer computes its own KV (not shared).
    pub fn getKVSourceLayer(self: ModelConfig, layer_idx: u32) ?u32 {
        if (self.num_kv_shared_layers == 0) return null;
        const first_shared = self.num_hidden_layers - self.num_kv_shared_layers;
        if (layer_idx < first_shared) return null;
        const is_global = self.isGlobalLayer(layer_idx);
        // Find last concrete layer of the same type (scanning downward)
        var j: u32 = first_shared;
        while (j > 0) {
            j -= 1;
            if (self.isGlobalLayer(j) == is_global) return j;
        }
        return null;
    }

    /// Get effective head_dim for a layer (global layers may use global_head_dim).
    pub fn layerHeadDim(self: ModelConfig, layer_idx: u32) u32 {
        if (self.global_head_dim > 0 and self.isGlobalLayer(layer_idx)) {
            return self.global_head_dim;
        }
        return self.head_dim;
    }

    /// Per-layer Q-head count (Laguna: 48 on full-attention layers, 72 on
    /// sliding). Every other arch has uniform heads, so this falls back to
    /// num_attention_heads. KV heads stay uniform (layerKVHeads).
    pub fn layerNumHeads(self: ModelConfig, layer_idx: u32) u32 {
        if (self.has_per_layer_heads and layer_idx < 128 and self.num_attention_heads_per_layer[layer_idx] > 0) {
            return self.num_attention_heads_per_layer[layer_idx];
        }
        return self.num_attention_heads;
    }

    /// The cached V width of a layer (MiMo stores K 192 / V 128).
    pub fn layerVHeadDim(self: ModelConfig, layer_idx: u32) u32 {
        return if (self.v_head_dim > 0) self.v_head_dim else self.layerHeadDim(layer_idx);
    }

    /// Leading channels of a head that RoPE rotates: `int(head_dim * partial)`.
    pub fn layerRopeDims(self: ModelConfig, layer_idx: u32) u32 {
        const partial = if (self.isGlobalLayer(layer_idx)) self.partial_rotary_factor_global else self.partial_rotary_factor;
        return @intFromFloat(@as(f32, @floatFromInt(self.layerHeadDim(layer_idx))) * partial);
    }

    /// Get effective num_kv_heads for a layer.
    pub fn layerKVHeads(self: ModelConfig, layer_idx: u32) u32 {
        if (self.num_global_key_value_heads > 0 and self.isGlobalLayer(layer_idx)) {
            return self.num_global_key_value_heads;
        }
        return self.num_key_value_heads;
    }

    pub fn isLinearLayer(self: ModelConfig, layer_idx: u32) bool {
        if (self.full_attention_interval == 0) return false;
        if (self.linear_attn_tail_from != 0 and layer_idx >= self.linear_attn_tail_from) return false;
        return ((layer_idx + 1) % self.full_attention_interval) != 0;
    }

    /// Which `partial_rotary_factor` the YaRN table covers: laguna/gemma4 scale
    /// only their full-attention layers and spell that one
    /// `partial_rotary_factor_global`; every other YaRN arch (qwen4_exp) has a
    /// single rope for the whole trunk.
    pub fn yarnPartial(self: *const ModelConfig) f32 {
        return if (self.isQwen4()) self.partial_rotary_factor else self.partial_rotary_factor_global;
    }

    /// `int(head_dim × yarnPartial())` — the rotating slice of a head, i.e. the
    /// dims the YaRN frequency table covers (qwen4_exp: 256 × 0.25 = 64, whose
    /// 32 frequencies are what `mrope_section` [11,11,10] sums to).
    pub fn yarnRotaryDim(self: *const ModelConfig) u32 {
        return @intFromFloat(@as(f32, @floatFromInt(self.head_dim)) * self.yarnPartial());
    }

    /// The longest sequence the rope can actually resolve. Plain:
    /// `max_position_embeddings`. YaRN: `original_max_position_embeddings ×
    /// factor` — the window HF and vLLM both derive `max_model_len` from — since
    /// a position past it aliases back inside the ramp. 0 = no rope-derived cap.
    pub fn contextCap(self: *const ModelConfig) u32 {
        const declared = self.max_position_embeddings;
        if (!self.rope_yarn) return declared;
        const orig: f64 = @floatFromInt(self.yarn_orig_max_pos);
        const factor: f64 = @floatCast(self.yarn_factor);
        const scaled: f64 = @floor(orig * factor);
        const max_u32: f64 = @floatFromInt(std.math.maxInt(u32));
        const window: u32 = if (scaled >= max_u32) std.math.maxInt(u32) else @intFromFloat(scaled);
        return if (declared == 0) window else @min(window, declared);
    }

    /// How many layers hold an attention KV cache. A hybrid arch interleaves
    /// linear-attention layers, which carry a FIXED-SIZE recurrent state
    /// instead of a per-token cache — billing them as attention layers made
    /// the memory model charge a uniform arch's footprint for a model
    /// carrying a fraction of it (bailing_hybrid: 6 of 24).
    pub fn attnCacheLayerCount(self: *const ModelConfig) u32 {
        // A `layer_block_types` hybrid (LFM2 via `layer_types`, Nemotron-H via
        // `hybrid_override_pattern`) never sets `full_attention_interval`, so
        // the interval arm below counted EVERY layer: lfm2 caches 8 of 30 and
        // was billed 3.75x, Nemotron-H worse. Only the `.attention` blocks
        // reach `ctx.cache` in the hybrid forward — gated_conv and mamba2 hold
        // a fixed-size recurrent state in `ssm_entries` instead. Layers past
        // the 128-entry table keep the array's `.attention` default, which is
        // the direction that over-bills rather than OOMs.
        if (self.has_hybrid_layers) {
            var n: u32 = if (self.num_hidden_layers > self.layer_block_types.len)
                self.num_hidden_layers - @as(u32, self.layer_block_types.len)
            else
                0;
            var li: u32 = 0;
            while (li < self.num_hidden_layers and li < self.layer_block_types.len) : (li += 1) {
                if (self.layer_block_types[li] == .attention) n += 1;
            }
            return n;
        }
        if (self.full_attention_interval == 0) return self.num_hidden_layers;
        var n: u32 = 0;
        var i: u32 = 0;
        while (i < self.num_hidden_layers) : (i += 1) {
            if (!self.isLinearLayer(i)) n += 1;
        }
        return n;
    }

    /// Dense (bf16) KV-cache bytes ONE token occupies across the whole model.
    /// The uniform `layers × 2 × kv_heads × head_dim` formula is wrong on a
    /// hybrid MLA arch in both terms: only `attnCacheLayerCount` layers cache
    /// at all, and MLA's key (nope+rope) is WIDER than its value. Every
    /// memory estimate that sizes a KV cache reads this one helper so the
    /// auto-context sizer and the prefill admission guard cannot disagree.
    /// Whether the prefill chunk is resolved per request (by the admission bill) instead of
    /// once at load. qwen4_exp only: a 1M session's load-time reserve pins every ordinary
    /// prompt to a narrow rung.
    pub fn perRequestPrefillChunk(self: *const ModelConfig) bool {
        return self.longCtxGated();
    }

    /// Rows a sliding layer keeps past its window on an arch that caches only the window
    /// (`slidingRing`): spec rollback plus the prefix-cache matches that land near an entry's end.
    pub const SLIDING_RING_SLACK: u32 = 2048;

    /// Sliding layers cache a bounded ring of rows, as mlx-lm's `RotatingKVCache`, not every token.
    pub fn slidingRing(self: *const ModelConfig) bool {
        return self.isMimo();
    }

    pub fn slidingKeepRows(self: *const ModelConfig) u32 {
        return self.sliding_window + SLIDING_RING_SLACK;
    }

    /// Dense bytes of ONE row across every ring layer (0 off `slidingRing`).
    pub fn slidingRowBytes(self: *const ModelConfig) u64 {
        if (!self.slidingRing()) return 0;
        var sum: u64 = 0;
        for (0..self.num_hidden_layers) |i| {
            const li: u32 = @intCast(i);
            if (!self.isGlobalLayer(li)) sum += self.layerKvRowBytes(li);
        }
        return sum;
    }

    fn layerKvRowBytes(self: *const ModelConfig, li: u32) u64 {
        return @as(u64, self.layerKVHeads(li)) * (@as(u64, self.layerHeadDim(li)) + self.layerVHeadDim(li)) * 2;
    }

    pub fn kvBytesPerToken(self: *const ModelConfig) u64 {
        // DeepSeek-V4.1 grows only its KV sources' compressed latents and index
        // keys (bf16); the window is a fixed ring per slot.
        if (self.isDsv41()) {
            var sum: u64 = 0;
            for (0..self.num_hidden_layers) |l| {
                const r: u64 = self.dsv4_compress_ratios[l];
                if (r == 0 or (self.dsv41_kv_sources >> @intCast(l)) & 1 == 0) continue;
                sum += self.head_dim / r;
                if ((self.dsv41_index_sources >> @intCast(l)) & 1 != 0) sum += self.dsv4_index_head_dim / r;
            }
            return sum * 2;
        }
        // GLM-5-Next stores one latent row plus the indexer's key and gate per token.
        if (self.isGlm5()) return @as(u64, self.attnCacheLayerCount()) * (self.dsa_kv_lora_rank + 2 * self.dsa_index_head_dim) * 2;
        // An arch that declares its own V width (MiMo) also varies KV heads per
        // layer type: billed layer by layer. Ring layers are a fixed per-slot term.
        if (self.v_head_dim > 0) {
            var sum: u64 = 0;
            for (0..self.num_hidden_layers) |i| {
                const li: u32 = @intCast(i);
                if (self.slidingRing() and !self.isGlobalLayer(li)) continue;
                sum += self.layerKvRowBytes(li);
            }
            return sum;
        }
        const widths: u64 = if (self.isMla())
            @as(u64, self.mlaQkHeadDim()) + @as(u64, self.mla_v_head_dim)
        else
            2 * @as(u64, self.head_dim);
        // MLA decompresses its latent to EVERY attention head before the write
        // (`mlaAttnWith` broadcasts the MQA rope key to `num_attention_heads`
        // and caches `[B, num_attention_heads, S, qk_dim]`), so its cache has
        // no grouping to save on — `num_key_value_heads` is the GQA question
        // and this arch never asks it. Equal on Ling 3.0 (16/16), so the
        // spelling is invisible today and would UNDER-bill the first MLA
        // checkpoint that groups — the direction that ends in an uncatchable
        // Metal OOM rather than a 400.
        const heads: u64 = if (self.isMla())
            @as(u64, self.num_attention_heads)
        else
            @as(u64, self.num_key_value_heads);
        return @as(u64, self.attnCacheLayerCount()) * heads * widths * 2;
    }

    /// Dense bf16 bytes of QSA indexer history ONE token occupies: the pooled
    /// blocks `[kv/ratio, idx_hd]` per full-attn layer. The raw keys are a fixed
    /// ring (`qsaRingBytes`, billed once per slot), not per token. Not
    /// kv-quantized. Zero on archs without an indexer. ONE copy; the billed width
    /// (copies + score bank) is `server.statePerTokenBilled`.
    pub fn qsaHistoryBytesPerToken(self: *const ModelConfig) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        const ratio = @max(@as(u64, self.indexer_compress_ratio), 1);
        return n * hd * 2 / ratio;
    }

    /// The raw indexer keys every live slot holds: `QSA_RING_ROWS` rows per
    /// full-attn layer, context-independent, billed once per slot.
    pub fn qsaRingBytes(self: *const ModelConfig) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        const rows = @as(u64, @intCast(@import("transformer.zig").QSA_RING_ROWS));
        return n * rows * hd * 2;
    }

    /// f32 bytes per token of the QSA block-score operand a live slot holds
    /// (`SSMCacheEntry.qsa_score_bank`). Never in an entry. Zero without an indexer.
    pub fn qsaScoreBankBytesPerToken(self: *const ModelConfig) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        if (@import("transformer.zig").qsaScoreFusedActiveFor(1, @intCast(self.indexer_n_heads), @intCast(self.indexer_head_dim))) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        const ratio = @max(@as(u64, self.indexer_compress_ratio), 1);
        return n * hd * 4 / ratio;
    }

    /// Bytes one SSM checkpoint holds: recurrent state + conv window of every linear layer.
    /// The QSA key history is not here (it lands on the newest checkpoint only).
    pub fn ssmCheckpointBytes(self: *const ModelConfig) u64 {
        if (self.linear_num_value_heads == 0) return 0;
        const linear_layers: u64 = @as(u64, self.num_hidden_layers) -| self.attnCacheLayerCount();
        if (linear_layers == 0) return 0;
        const state_elem: u64 = if (self.ssmStateDtype() == .float32) 4 else 2;
        const state: u64 = @as(u64, self.linear_num_value_heads) *
            @as(u64, self.linear_value_head_dim) * @as(u64, self.linear_key_head_dim) * state_elem;
        const conv_dim: u64 = 2 * @as(u64, self.linear_num_key_heads) * self.linear_key_head_dim +
            @as(u64, self.linear_num_value_heads) * self.linear_value_head_dim;
        const conv: u64 = @as(u64, self.linear_conv_kernel_dim) -| 1;
        return linear_layers * (state + conv * conv_dim * 2);
    }

    /// Activation dtype: f16 for a Prism Hadamard pack (its own contract: f16
    /// activations over f16 scales), bf16 everywhere else.
    pub fn actDtype(self: *const ModelConfig) mlx.mlx_dtype {
        return if (self.hadamard_block > 0) .float16 else .bfloat16;
    }

    /// GatedDeltaNet recurrent state dtype: f32 on Hadamard packs, as the
    /// reference runtime keeps it; bf16 elsewhere.
    pub fn ssmStateDtype(self: *const ModelConfig) mlx.mlx_dtype {
        return if (self.hadamard_block > 0 or self.isGlm5()) .float32 else .bfloat16;
    }

    /// Configured MTP depth (0 = auto). Hadamard packs run a grafted head that
    /// over-speculates in auto mode; 2 measured best (code 71 vs 58 tok/s).
    pub fn mtpDepth(self: *const ModelConfig, configured: u32) u32 {
        if (configured == 0 and self.hadamard_block > 0) return 2;
        // Nemotron-H MoE: every verify row routes to more experts, so the
        // round cost climbs with depth while the head's acceptance decays;
        // depth 2 beats both 1 and 3+, the adaptive default cap (6) loses.
        if (configured == 0 and std.mem.eql(u8, self.model_type, "nemotron_h")) return 2;
        // glm5_next: the head's acceptance decays past three drafts while every verify row
        // costs a MoE expert read; the adaptive default settled at one draft and lost.
        if (configured == 0 and self.isGlm5()) return 3;
        return configured;
    }

    /// SSD prefix-cache layout marker (`kv_disk_cache.modelFingerprintWithLayout`).
    /// Nemotron-H keys were RoPE-rotated before its attention became NoPE; the
    /// marker gives it a fresh root. Null keeps every other arch's root.
    pub fn cacheLayoutNamespace(self: *const ModelConfig) ?[]const u8 {
        if (std.mem.eql(u8, self.model_type, "nemotron_h")) return "nemotron-h-nope-v1";
        if (self.rope_llama3 != null) return "llama3-rope-v1";
        // A pool's completing indexer row stores the pool's key in its gate half.
        if (self.isGlm5()) return "glm5-pooled-keys-v1";
        return null;
    }

    /// Decode-width projections go through the row-exact kernels, whose rows
    /// get the same bits at any window width: a verify row reproduces serial
    /// decoding. Only where every projection fits them AND a DFlash drafter is
    /// bound: serial and MTP keep the faster stock paths otherwise.
    pub fn rowExactDecode(self: *const ModelConfig) bool {
        return self.row_exact_covered and self.dflash_bound and self.rowExactArch();
    }

    /// MTP acceptance for this load: exact while a DFlash drafter is bound, so
    /// the MTP rounds it yields to stay byte-exact too.
    pub fn mtpAcceptance(self: *const ModelConfig, default: mtp_acceptance_mod.Mode) mtp_acceptance_mod.Mode {
        if (self.dflash_bound) return .exact;
        return self.mtp_acceptance_override orelse default;
    }

    pub fn rowExactArch(self: *const ModelConfig) bool {
        if (std.mem.eql(u8, self.model_type, "nemotron_h")) return true;
        return std.mem.startsWith(u8, self.model_type, "qwen3_5") and !self.isMoe() and self.hadamard_block == 0;
    }

    /// The MTP head's hidden input is the trunk's final-normed hidden, not the
    /// residual (Nemotron-H: more drafts kept at depth 1 and 2).
    pub fn mtpReadsFinalNorm(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "nemotron_h") or self.isMimo();
    }

    /// Vocab rows a spec drafter proposes from (0 = all): Qwen3.8's tokenizer
    /// puts 99.64% of committed tokens below id 98304, so a draft reads 40% of
    /// the head. A token past it is never drafted: speed only, never output.
    pub fn draftVocab(self: *const ModelConfig) c_int {
        if (self.rowExactDecode() and std.mem.startsWith(u8, self.model_type, "qwen3_5") and self.vocab_size >= 248320) return 98304;
        return 0;
    }

    pub fn isGlm5(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "glm5_next");
    }

    pub fn isDsv41(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "deepseek_v41");
    }

    pub fn isMimo(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "mimo_v2");
    }

    pub fn isMoe(self: *const ModelConfig) bool {
        return self.num_experts > 0;
    }

    /// True when the full-attention layers are Multi-head Latent Attention
    /// (compressed KV latent + low-rank Q), not plain GQA projections.
    pub fn isMla(self: *const ModelConfig) bool {
        return self.mla_kv_lora_rank > 0;
    }

    /// Does Q go through a low-rank pair, or straight from the hidden state?
    ///
    /// `q_lora_rank: null` is DeepSeek-V3's documented option and what the
    /// whole Ling 3.0 FLASH line ships (tiny ships 256). Those checkpoints
    /// carry a plain `attention.q_proj` instead of
    /// q_a_proj/q_a_layernorm/q_b_proj. 0 is the signal, since a real rank is
    /// always positive.
    pub fn mlaHasQLora(self: *const ModelConfig) bool {
        return self.mla_q_lora_rank > 0;
    }

    /// MLA query/key head dim = the non-positional part plus the rope part.
    /// This — not head_dim — is what the attention scale and the cached K's
    /// last dim are measured in.
    pub fn mlaQkHeadDim(self: *const ModelConfig) u32 {
        return self.mla_qk_nope_head_dim + self.mla_qk_rope_head_dim;
    }

    /// Which of fla's two KDA gate arms this checkpoint declares. A non-zero
    /// `kda_lower_bound` REPLACES the softplus form with the bounded sigmoid;
    /// absent (0) means the softplus form, which the shared GatedDeltaNet chain
    /// already computes elementwise and therefore serves a per-channel gate
    /// unchanged. Feeding bound 0 to the bounded chain yields exp(0) = 1 — a
    /// gate that never forgets — so the arm must be chosen, never defaulted.
    pub fn kdaUsesBoundedGate(self: *const ModelConfig) bool {
        return self.kda_vector_gate and self.kda_gate_lower_bound != 0.0;
    }

    /// The pooling op /v1/embeddings runs: the explicit signal, else masked
    /// mean (the historical default — correct for MiniLM and EmbeddingGemma).
    pub fn effectivePooling(self: *const ModelConfig) PoolingMode {
        return self.pooling_mode orelse .mean;
    }

    /// Whether this model serves /v1/embeddings meaningfully: encoder-only
    /// (BERT, EmbeddingGemma) or a decoder with a declared pooling contract
    /// (Qwen3-Embedding). Drives capability advertising, never dispatch.
    pub fn hasEmbeddingCapability(self: *const ModelConfig) bool {
        return self.is_encoder_only or self.pooling_mode != null;
    }

    pub fn isInkling(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "inkling_mm_model");
    }

    /// Google EmbeddingGemma 2 (`embedding_gemma2`): a bidirectional Gemma 4 text trunk with
    /// projection-only per-layer inputs and an `embedding_projection` head.
    pub fn isEmbeddingGemma2(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "embedding_gemma2");
    }

    /// Qwen3.8-Flash-Next (`qwen4_exp`): the qwen3_5 GDN + MoE trunk wrapped
    /// in hyper-connection residual streams, with the n-gram PLE and QSA.
    pub fn isQwen4(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "qwen4_exp");
    }

    /// The long-context blast-radius predicate: every long-context mechanism (KV
    /// reservation, pad-waste cap, checkpoint thinning, admission terms, chunk bar) was
    /// measured on qwen4_exp only, so they are opt-in by arch. Never hand-roll it at a site.
    pub fn longCtxGated(self: *const ModelConfig) bool {
        return self.isQwen4();
    }

    pub fn batchedEffectiveKvLen(self: *const ModelConfig, kv: u32, gather_on: bool, gather_min_kv: u32) u32 {
        if (!self.isQwen4() or !gather_on) return kv;
        if (kv <= gather_min_kv) return kv;
        const cap = self.indexer_budget + self.indexer_compress_ratio;
        if (cap == 0) return kv;
        return @min(kv, cap);
    }

    /// SSD-first prefix cache arch predicate; delegates to `longCtxGated`.
    pub fn ssdFirstCapable(self: *const ModelConfig) bool {
        return self.longCtxGated();
    }

    /// True when per-request SSM/conv cache entries must exist: hybrid
    /// recurrence (LFM2/Nemotron/GDN) or Inkling's four per-layer short
    /// convolutions. Shared by Transformer.init and the scheduler's per-slot
    /// allocation — the two predicates MUST agree or slots crash on a null
    /// `ctx.ssm_entries` (the Qwen3.5-MoE class).
    pub fn needsSsmEntries(self: *const ModelConfig) bool {
        return self.has_hybrid_layers or self.full_attention_interval > 0 or self.isInkling();
    }

    /// Block-diffusion checkpoint (DiffusionGemma): generation is the canvas
    /// denoising loop, not autoregressive decode.
    pub fn isDiffusion(self: *const ModelConfig) bool {
        return self.canvas_length > 0;
    }

    /// Pure-config half of "can this arch ride the batched GatedDeltaNet
    /// decode kernel?" (`Transformer.forwardMoeBatchedDecode`) — a dense
    /// GDN trunk with periodic full attention, i.e. the qwen3_5 family.
    ///
    /// This exists because the answer is needed in TWO places that see
    /// different things: `server.zig` decides whether `--max-concurrent`
    /// clamps to 1 with only a ModelConfig in hand, while
    /// `Transformer.supportsBatchedGdnDecode` also checks the built layer
    /// set. Both MUST read this predicate — when they were hand-rolled
    /// separately, the server kept clamping qwen3_5 to serial decode while
    /// the scheduler was happily batching it, so `--max-concurrent 4` (the
    /// obvious serving config) silently DISABLED the batched path.
    ///
    /// Says nothing about MoE/hybrid archs that merely share the same
    /// forward — those stay serial, by name, in both callers.
    pub fn supportsBatchedGdnDecode(self: *const ModelConfig) bool {
        // MiMo: attention + row-generic MoE with no per-slot recurrent state to merge.
        if (self.isMimo()) return true;
        // GLM-5-Next: a batched tick's tokens ride as rows of one window; each slot's KDA state and
        // DSA cache are advanced by its own call (`Transformer.forwardGlmBatchedDecode`).
        if (self.isGlm5()) return true;
        if (self.full_attention_interval == 0) return false; // not a GDN trunk
        if (self.has_hybrid_layers) return false; // lfm2 / nemotron_h
        if (self.is_encoder_only) return false;
        // Routed experts are row-generic; a MoE trunk batches when its per-slot
        // state is what the path merges (GDN pair, qwen4's PLE window + QSA keys).
        if (self.isMoe() and !self.isQwen4() and !std.mem.eql(u8, self.model_type, "qwen3_5_moe")) return false;
        if (self.isInkling() or self.isMla() or self.isGemma4Layers()) return false;
        if (self.isDiffusion()) return false;
        if (self.kda_vector_gate) return false; // bailing KDA: its own gate shape
        if (std.mem.eql(u8, self.model_type, "laguna")) return false;
        if (std.mem.eql(u8, self.model_type, "deepseek_v4")) return false;
        return true;
    }

    /// True when the trunk uses the Gemma 4 layer structure (dual FFN with
    /// shared-expert branch, sigma-MoE router, 7 norms, layer_scalar, v_norm,
    /// proportional RoPE on full layers). DiffusionGemma reuses the Gemma 4
    /// 26B-A4B decoder verbatim, so transformer.zig's gemma4 forward/binding
    /// paths key on this rather than on the model_type string.
    pub fn isGemma4Layers(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "gemma4") or
            std.mem.eql(u8, self.model_type, "diffusion_gemma");
    }

    /// Takes the sampling recommendations + extra eos ids of a generation_config.json.
    fn applyGenerationDefaults(self: *ModelConfig, gen_content: []const u8) void {
        const gd = parseGenerationDefaultsFromJson(gen_content);
        self.gen_temperature = gd.temperature;
        self.gen_top_p = gd.top_p;
        self.gen_top_k = gd.top_k;
        self.gen_min_p = gd.min_p;
        self.gen_enable_thinking = gd.enable_thinking;
        self.mergeEosTokens(gd.eos_token_ids[0..gd.num_eos]);
    }

    /// Additive + dedup-guarded, like every terminator merge here.
    pub fn mergeEosTokens(self: *ModelConfig, ids: []const u32) void {
        for (ids) |id| if (!self.isEosToken(id)) self.addEosToken(id);
    }

    pub fn addEosToken(self: *ModelConfig, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }

    /// Gemma's chat template always ends turns with `<end_of_turn>` (id 106)
    /// and emits `<eos>` (id 1) at sequence end, so both must be stop tokens
    /// for EVERY Gemma family (gemma3 / gemma4 / diffusion_gemma). Some
    /// checkpoints declare only a SCALAR `eos_token_id: 1` (e.g. the
    /// abliterated text-only `-lm-` builds) — gating the 106 add on
    /// `num_eos_tokens == 0` then leaves it out and it leaks into output as
    /// repeated `<end_of_turn>`. Merge both ADDITIVELY + dedup-guarded (never
    /// removes a config-declared stop). Same leak class as the Qwen2.5-Coder
    /// `<|im_end|>` merge performed at load time (main.zig / scheduler doLoad).
    pub fn ensureGemmaTerminators(self: *ModelConfig) void {
        if (!self.isEosToken(1)) self.addEosToken(1);
        if (!self.isEosToken(106)) self.addEosToken(106);
    }

    /// MuseGlimmer terminators: <|end_of_text|> = 200001 and <|eot|> = 200008
    /// (the chat template's turn terminator; <|eom|> 200007 is deliberately
    /// NOT an eos — generation continues across channel segments). Additive +
    /// dedup-guarded like ensureGemmaTerminators.
    pub fn ensureMuseTerminators(self: *ModelConfig) void {
        if (!self.isEosToken(200001)) self.addEosToken(200001);
        if (!self.isEosToken(200008)) self.addEosToken(200008);
    }

    /// NoPE layers (muse_glimmer: layer_rope_theta[i] == 0). The released
    /// checkpoint's NoPE layers are exactly its full-attention layers, but the
    /// two facts stay independently parsed — layer_types drives masking,
    /// layer_rope_theta drives rotation.
    pub fn layerSkipsRope(self: *const ModelConfig, layer_idx: u32) bool {
        return layer_idx < 128 and self.layer_no_rope[layer_idx];
    }

    /// Post-attention / post-feedforward norm epsilon (muse_glimmer separates
    /// it from rms_norm_eps; everyone else shares one value).
    pub fn postNormEps(self: *const ModelConfig) f32 {
        return if (self.post_norm_eps > 0) self.post_norm_eps else self.rms_norm_eps;
    }

    /// SDPA softmax scale for the standard dense forward. MuseGlimmer
    /// multiplies the unit-RMS Q by qk_scale_factor on top of the standard
    /// 1/sqrt(head_dim); Gemma 4's QK-norm handles normalization (scale 1.0);
    /// everything else keys on query_pre_attn_scalar.
    pub fn attnScale(self: *const ModelConfig) f32 {
        if (self.qk_scale_factor > 0)
            return self.qk_scale_factor / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        if (std.mem.eql(u8, self.model_type, "gemma4") or self.isEmbeddingGemma2()) return 1.0;
        return 1.0 / @sqrt(@as(f32, @floatFromInt(self.query_pre_attn_scalar)));
    }

    /// Hy3 (hy_v3) family terminator: <｜hy_eos:opensource｜> = 120025. Real
    /// MLX conversions (ox-ox 2-bit) ship NO eos in config.json and NO
    /// generation_config.json, so without this merge generation never halts.
    /// Additive + dedup-guarded like ensureGemmaTerminators — never gate a
    /// known chat-terminator on "config provided no eos".
    pub fn ensureHy3Terminators(self: *ModelConfig) void {
        if (!self.isEosToken(120025)) self.addEosToken(120025);
    }

    /// gpt_oss / harmony terminators, merged ADDITIVELY onto whatever the
    /// config declared. A harmony assistant turn can end two ways and the
    /// config names only one of them:
    ///   <|return|> (200002) — the declared eos, ends a normal answer.
    ///   <|call|>   (200012) — ends a TOOL CALL. Never the eos, so a
    ///                         tools request that stops only on eos runs to
    ///                         max_tokens with the call already complete.
    /// <|end|> (200007) is deliberately NOT here: it closes the analysis
    /// channel MID-turn, immediately before `<|start|>assistant<|channel|>final`
    /// opens. Terminating on it would truncate every thinking response to its
    /// reasoning and never emit an answer.
    pub fn ensureGptOssTerminators(self: *ModelConfig) void {
        if (!self.isEosToken(200002)) self.addEosToken(200002);
        if (!self.isEosToken(200012)) self.addEosToken(200012);
    }

    /// The head width the PREFILL SCORE tensor is actually built at. Normally
    /// `head_dim`, but an arch can score at a different width than it stores
    /// values at (an MLA q.k can contract over nope+rope widths while
    /// `head_dim` stays the value width) — reading `head_dim` there puts such
    /// an arch under the `<= 128` "fused SDPA covers it" early-out, so the
    /// score budget that exists for exactly this materializing path never
    /// applies. A new arch scoring wider than it stores adds its arm here.
    /// 0 = no score tensor at all: MiMo's prefill attention is always a fused
    /// kernel (`fusedSinkAttnPrefill`), whatever its width.
    pub fn prefillScoreHeadDim(self: *const ModelConfig) u32 {
        if (self.isMimo()) return 0;
        // GLM-5-Next: the sparse rows never form a score matrix (indexed kernel), and the
        // dense ones end at the indexer budget (~2k), whatever the chunk.
        if (self.isGlm5()) return 0;
        if (self.isMla()) return self.mlaQkHeadDim();
        return self.head_dim;
    }

    /// Whether a chat request that names NO thinking preference should render
    /// with thinking on. Our server always passes `enable_thinking` explicitly,
    /// so a template whose own default is 'on' is silently overridden to off
    /// for every client that omits the field — the vendor's default mode
    /// becomes unreachable without a vendor-specific flag. An EXPLICIT request
    /// value always outranks this (see `server.resolveEnableThinking`); it
    /// only fills a silent request.
    ///
    /// First the checkpoint's OWN declaration
    /// (`generation_config.json` -> `default_chat_template_kwargs.enable_thinking`),
    /// then the per-arch allowlist below, which stays opt-in and only where the
    /// vendor documents thinking-on AND the shipped template agrees — never
    /// inferred from "the template mentions enable_thinking".
    pub fn defaultEnableThinking(self: *const ModelConfig, has_tools: bool) bool {
        // The checkpoint's own declared default outranks the arch allowlist:
        // it is the model author speaking, not our guess about the family.
        if (self.gen_enable_thinking) |v| return v;
        // muse_glimmer: tool turns keep thinking (a tool call is a `to=<fn>`
        // header, so the recipient must stay free and the reasoning is
        // delivered rather than paid-and-dropped). A plain chat request
        // defaults to the prompt-committed to=user channel instead
        // (chat.noThinkTailSuffix) — no reasoning pass runs at all.
        // gpt_oss: UNCONDITIONALLY on. Harmony has no thinking-off mode — the
        // template's `Reasoning: low|medium|high` sets depth, not presence, and
        // the model opens `<|channel|>analysis<|message|>` on every turn no
        // matter what we ask. Defaulting a silent request to off did not stop
        // the reasoning pass, it just routed the analysis channel down the
        // flush-text streaming branch, which leaked `<|channel|>analysis` and
        // the reasoning itself into visible content (live 2026-08-12).
        // Thinking-off here would have to be enforced in the PROMPT, and
        // harmony offers no way to do it.
        if (std.mem.eql(u8, self.model_type, "gpt_oss")) return true;
        if (has_tools and std.mem.eql(u8, self.model_type, "muse_glimmer")) return true;
        // bailing_hybrid (Ling 3.0): thinking-on with or without tools. The
        // checkpoint's own template normalizes an undefined `enable_thinking`
        // to `thinking_option = 'on'` unconditionally, and unlike muse there
        // is no prompt-committed no-think channel to fall back to — so a
        // tool-less silent request gated OFF just makes a reasoner answer
        // without reasoning ("17 - 9 = 8" where the thinking arm works the
        // word problem and answers "9 sheep are left").
        if (std.mem.eql(u8, self.model_type, "bailing_hybrid")) return true;
        // mimo_v2: the template thinks unless told `enable_thinking` false.
        if (self.isMimo()) return true;
        // k2_horizon: the template opens a think marker on every assistant
        // turn; thinking-off is the prompt-committed closer (chat.contentChannelTail).
        if (std.mem.eql(u8, self.model_type, "k2_horizon")) return true;
        // glm5_next: the template opens `<think>` on every assistant turn too.
        if (self.isGlm5()) return true;
        // deepseek_v41: DeepSeek's template defaults `thinking_mode` to thinking.
        if (self.isDsv41()) return true;

        return false;
    }

    /// Fill still-null sampling recommendations with the FAMILY's documented
    /// upstream defaults. Community re-quants/distills routinely ship no
    /// generation_config.json (live 2026-07-13: a Qwen3.6-35B distill served
    /// to pi resolved omitted fields to the hardcoded 1.0/1.0/off — full
    /// untruncated tail sampling on a 4-bit MoE — and a 16K-token agent turn
    /// degenerated into word salad). Same pattern as the gemma3 head-count
    /// gotcha: when resolution relies on per-arch defaults a minimal
    /// checkpoint may omit, fill them explicitly.
    ///
    /// Deliberately fills ONLY the truncation knobs (top_k/top_p — what keeps
    /// the tail out of the sample space), never temperature: a null temp stays
    /// the neutral 1.0, and explicit request/flag/file values always win
    /// (this runs AFTER generation_config.json parse, nulls only).
    pub fn applyFamilySamplingDefaults(self: *ModelConfig) void {
        const t = self.model_type;
        // Qwen 3.x family only — Qwen2.5's upstream defaults differ (top_p
        // 0.8); never guess numbers the family didn't document.
        const is_qwen = std.mem.eql(u8, t, "qwen3") or
            std.mem.eql(u8, t, "qwen3_moe") or
            std.mem.eql(u8, t, "qwen3_5_moe") or
            std.mem.eql(u8, t, "qwen4_exp") or
            std.mem.eql(u8, t, "qwen3_next");
        const is_gemma = std.mem.eql(u8, t, "gemma3") or
            std.mem.eql(u8, t, "gemma4") or
            std.mem.eql(u8, t, "diffusion_gemma");
        if (is_qwen) {
            if (self.gen_top_k == null) self.gen_top_k = 20;
            if (self.gen_top_p == null) self.gen_top_p = 0.95;
        } else if (is_gemma) {
            if (self.gen_top_k == null) self.gen_top_k = 64;
            if (self.gen_top_p == null) self.gen_top_p = 0.95;
        } else if (std.mem.eql(u8, t, "inkling_mm_model")) {
            // Thinking Machines publishes NO recommendation (no
            // generation_config.json in any Inkling repo; their bundled
            // tooling samples greedily), so top_p 0.95 is OUR choice to cut
            // the untruncated tail — the first real pi agent session
            // (2026-07-30) ran the hardcoded 1.0/1.0/off and degenerated
            // into duplicated tool calls.
            if (self.gen_top_p == null) self.gen_top_p = 0.95;
        }
    }

    /// DeepSeek-V4 releases ship generation_config.json with the WILD
    /// signature (temp 1.0 / top_p 1.0) that their own inference/generate.py
    /// IGNORES — its default is temperature 0.6, the value our converter
    /// writes into our mirrors. External conversions (pipenetwork REAP) copy
    /// the source file verbatim, and an agent CLI that omits temperature then
    /// samples the untruncated tail (live 2026-08-01: pi against REAP37
    /// degenerated into token loops on its FIRST turn). When the reference
    /// implementation deliberately ignores a config field, that field is not
    /// the source of truth (the laguna YaRN class): the EXACT untouched
    /// signature resolves to the reference's default; anything an author
    /// actually tuned is left alone, and request/flag values always win.
    pub fn applyDsv4ReferenceSampling(self: *ModelConfig) void {
        if (!std.mem.eql(u8, self.model_type, "deepseek_v4")) return;
        const t = self.gen_temperature orelse return;
        const p = self.gen_top_p orelse return;
        if (t == 1.0 and p == 1.0) {
            log.info("deepseek_v4: generation_config carries the source's wild 1.0/1.0 signature — resolving to the reference default temp 0.6\n", .{});
            self.gen_temperature = 0.6;
        }
    }

    pub fn isEosToken(self: *const ModelConfig, id: u32) bool {
        for (self.eos_token_ids[0..self.num_eos_tokens]) |eos| {
            if (id == eos) return true;
        }
        return false;
    }

    pub fn eosTokenSlice(self: *const ModelConfig) []const u32 {
        return self.eos_token_ids[0..self.num_eos_tokens];
    }

    /// Everything a load derives from the tokenizer; both load paths call it.
    pub fn applyTokenizer(
        self: *ModelConfig,
        tok: *const tokenizer_mod.Tokenizer,
        eos_token: ?[]const u8,
    ) void {
        // Always merge the chat terminator, even when config.json names an EOS:
        // Qwen2.5-Coder's config says <|endoftext|> but turns end on <|im_end|>.
        if (eos_token) |eos_str| {
            if (tok.special_tokens.get(eos_str)) |eos_id| {
                if (!self.isEosToken(eos_id)) {
                    self.addEosToken(eos_id);
                    log.info("EOS token from tokenizer: {s} (id={d})\n", .{ eos_str, eos_id });
                }
            }
        }
        if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
            if (!self.isEosToken(eot_id)) self.addEosToken(eot_id);
        }
        // Id 0 can be produced spuriously under long/confusing prompts.
        if (tok.special_tokens.get("<pad>")) |pad_id| {
            if (pad_id > 0 and !self.isEosToken(pad_id)) {
                self.addEosToken(pad_id);
                log.info("Added <pad> as stop token (id={d})\n", .{pad_id});
            }
        }
        self.populateLfm2ImageTokens(tok);
    }

    /// LFM2-VL wraps its image-token run in `<|image_start|>`/`<|image_end|>`,
    /// labels every tile with `<|img_row_R_col_C|>` and marks the thumbnail
    /// with `<|img_thumbnail|>`. NONE of those ids appear in config.json — the
    /// tokenizer is the only place they exist — so they are resolved by STRING
    /// at load. A missing marker leaves its id 0,
    /// which every consumer reads as "this checkpoint has no such token".
    pub fn populateLfm2ImageTokens(self: *ModelConfig, tok: *const tokenizer_mod.Tokenizer) void {
        if (!self.lfm2_vision) return;
        if (tok.special_tokens.get("<|image_start|>")) |id| self.boi_token_id = id;
        if (tok.special_tokens.get("<|image_end|>")) |id| self.eoi_token_id = id;
        if (tok.special_tokens.get("<|img_thumbnail|>")) |id| self.lv_thumbnail_token_id = id;
        // The row/col markers are one contiguous block laid out row-major over
        // the max tile grid, so the first one plus (row, col) locates them all.
        if (tok.special_tokens.get("<|img_row_1_col_1|>")) |id| self.lv_row_col_base_id = id;
        if (self.image_token_id == 0) {
            if (tok.special_tokens.get("<image>")) |id| self.image_token_id = id;
        }
        log.info("LFM2-VL image tokens: <image>={d} start={d} end={d} thumbnail={d} row_col_base={d}\n", .{
            self.image_token_id, self.boi_token_id, self.eoi_token_id, self.lv_thumbnail_token_id, self.lv_row_col_base_id,
        });
    }

    /// Free the allocator-owned fields (`ngram_table_path`, allocPrint'd by
    /// `parseConfig`, and `drafter_override`); everything else is plain data or a borrowed slice. Every
    /// `destroy` of a parsed config pairs with this, or a qwen4 load leaks the
    /// path. Idempotent.
    pub fn deinit(self: *ModelConfig, allocator: std.mem.Allocator) void {
        if (self.ngram_table_path) |p| allocator.free(p);
        self.ngram_table_path = null;
        if (self.dsv41_dir) |p| allocator.free(p);
        self.dsv41_dir = null;
        if (self.drafter_override) |p| allocator.free(p);
        self.drafter_override = null;
    }
};

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    const content = (try mlx_gguf.sidecar(io, allocator, model_dir, .config)) orelse blk: {
        const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
        defer allocator.free(path);

        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    };
    defer allocator.free(content);

    var config = try parseConfigFromJson(allocator, content);
    errdefer config.deinit(allocator);
    if (config.isDsv41()) {
        config.dsv41_dir = try allocator.dupe(u8, model_dir);
        const bank = try std.fmt.allocPrint(allocator, "{s}/experts.bin", .{model_dir});
        defer allocator.free(bank);
        config.dsv41_stream = if (std.Io.Dir.accessAbsolute(io, bank, .{})) |_| true else |_| false;
    }
    if (config.isQwen4()) {
        config.ngram_table_path = try std.fmt.allocPrint(allocator, "{s}/ngram_table.bin", .{model_dir});
        if (try qwen4_exp.inspectEmbedded(model_dir, try qwen4EmbeddedSpec(&config))) |info| {
            config.embedded_ple_payload_bytes = info.payload_bytes;
        }
    }

    // Model-author sampling recommendations ride in a sibling file. Optional —
    // any failure (missing file, bad JSON) leaves the fields null.
    const gen_path = try std.fmt.allocPrint(allocator, "{s}/generation_config.json", .{model_dir});
    defer allocator.free(gen_path);
    const gen_sidecar: ?[]u8 = mlx_gguf.sidecar(io, allocator, model_dir, .generation_config) catch null;
    if (gen_sidecar) |gen_content| {
        defer allocator.free(gen_content);
        config.applyGenerationDefaults(gen_content);
    } else if (std.Io.Dir.openFileAbsolute(io, gen_path, .{})) |gen_file| {
        defer gen_file.close(io);
        var gen_buf: [4096]u8 = undefined;
        var gen_reader = gen_file.reader(io, &gen_buf);
        if (gen_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |gen_content| {
            defer allocator.free(gen_content);
            config.applyGenerationDefaults(gen_content);
        } else |_| {}
    } else |_| {}
    // Pooling (issue #116), priority: explicit config.json `pooling_mode`
    // (already parsed) > the ST `1_Pooling/config.json` sidecar > the
    // known-family name fallback. A sidecar declaring only unsupported modes
    // fails the load here — explicitly, never a silent mean-pool.
    if (config.pooling_mode == null) {
        const pool_path = try std.fmt.allocPrint(allocator, "{s}/1_Pooling/config.json", .{model_dir});
        defer allocator.free(pool_path);
        if (std.Io.Dir.openFileAbsolute(io, pool_path, .{})) |pool_file| {
            defer pool_file.close(io);
            var pool_buf: [4096]u8 = undefined;
            var pool_reader = pool_file.reader(io, &pool_buf);
            if (pool_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |pool_content| {
                defer allocator.free(pool_content);
                config.pooling_mode = try parsePoolingSidecar(pool_content);
                if (config.pooling_mode) |m|
                    log.info("[embed] pooling from 1_Pooling/config.json: {s}\n", .{@tagName(m)});
            } else |_| {}
        } else |_| {}
    }
    if (config.pooling_mode == null) {
        if (poolingFromDirName(std.fs.path.basename(model_dir), config.model_type)) |m| {
            config.pooling_mode = m;
            log.info("[embed] pooling inferred from checkpoint name: {s}\n", .{@tagName(m)});
        }
    }

    // Community re-quants often ship NO generation_config.json; fill the
    // still-null truncation knobs with the family's documented defaults so
    // omitted-field resolution never bottoms out at untruncated sampling.
    config.applyFamilySamplingDefaults();
    // ... and a dsv4 generation_config carrying the source's verbatim wild
    // signature resolves to the reference implementation's own default.
    config.applyDsv4ReferenceSampling();

    // Qwen image sizing is processor metadata rather than an architecture
    // constant. Prefer processor_config.json and fill any missing field from
    // the older preprocessor_config.json layout.
    if (config.qwen_vision or config.muse_vision) {
        var vision_defaults = VisionProcessorDefaults{};
        const processor_files = [_][]const u8{
            "processor_config.json",
            "preprocessor_config.json",
        };
        for (processor_files) |name| {
            const processor_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, name });
            defer allocator.free(processor_path);
            if (std.Io.Dir.openFileAbsolute(io, processor_path, .{})) |processor_file| {
                defer processor_file.close(io);
                var processor_buf: [4096]u8 = undefined;
                var processor_reader = processor_file.reader(io, &processor_buf);
                if (processor_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |processor_content| {
                    defer allocator.free(processor_content);
                    const parsed_defaults = parseVisionProcessorDefaultsFromJson(processor_content);
                    if (vision_defaults.min_pixels == null)
                        vision_defaults.min_pixels = parsed_defaults.min_pixels;
                    if (vision_defaults.max_pixels == null)
                        vision_defaults.max_pixels = parsed_defaults.max_pixels;
                    if (vision_defaults.max_image_tokens == null)
                        vision_defaults.max_image_tokens = parsed_defaults.max_image_tokens;
                } else |_| {}
            } else |_| {}
        }
        if (vision_defaults.min_pixels != null and
            vision_defaults.max_pixels != null and
            vision_defaults.min_pixels.? > vision_defaults.max_pixels.?)
        {
            vision_defaults = .{};
        }
        config.qv_min_pixels = vision_defaults.min_pixels orelse 0;
        config.qv_max_pixels = vision_defaults.max_pixels orelse 0;
        config.mv_max_image_tokens = vision_defaults.max_image_tokens orelse MUSE_MAX_IMAGE_TOKENS;
    }

    if (config.isEmbeddingGemma2()) {
        const processor_path = try std.fmt.allocPrint(allocator, "{s}/processor_config.json", .{model_dir});
        defer allocator.free(processor_path);
        if (std.Io.Dir.openFileAbsolute(io, processor_path, .{})) |processor_file| {
            defer processor_file.close(io);
            var processor_buf: [4096]u8 = undefined;
            var processor_reader = processor_file.reader(io, &processor_buf);
            if (processor_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |processor_content| {
                defer allocator.free(processor_content);
                const budgets = parseSoftTokenBudgetsFromJson(allocator, processor_content);
                config.gv_image_tokens = budgets.image orelse config.gv_image_tokens;
                config.gv_video_tokens = budgets.video orelse config.gv_video_tokens;
                config.gv_video_frames = budgets.max_frames orelse config.gv_video_frames;
            } else |_| {}
        } else |_| {}
        log.info("[vision] EmbeddingGemma 2 processors: {d} soft tokens per image, {d} per video frame, at most {d} frames\n", .{ config.gv_image_tokens, config.gv_video_tokens, config.gv_video_frames });
    }

    return config;
}

/// Sampling recommendations parsed out of a model's generation_config.json.
pub const GenerationDefaults = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    min_p: ?f32 = null,
    /// `default_chat_template_kwargs.enable_thinking` — the checkpoint's own
    /// thinking default. null when absent or not a bool.
    enable_thinking: ?bool = null,
    /// `eos_token_id` (scalar or list): HF stops generation on these, and a
    /// checkpoint may name the chat terminator ONLY here (K2-Horizon's
    /// `<|ifm|im_end|>` rides beside config.json's `<|ifm|endoftext|>`).
    eos_token_ids: [8]u32 = @splat(0),
    num_eos: usize = 0,
};

/// Image-area limits parsed from a Qwen processor configuration.
pub const VisionProcessorDefaults = struct {
    min_pixels: ?u32 = null,
    max_pixels: ?u32 = null,
    /// Muse: the cap is on MERGED tokens, not pixels.
    max_image_tokens: ?u32 = null,
};

fn positiveJsonU32(value: ?std.json.Value) ?u32 {
    const actual = value orelse return null;
    return switch (actual) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

/// Parse both processor layouts used by Qwen checkpoints:
/// `image_processor.{min_pixels,max_pixels}` and
/// `size.{shortest_edge,longest_edge}`.
pub fn parseVisionProcessorDefaultsFromJson(content: []const u8) VisionProcessorDefaults {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};

    const root = parsed.value.object;
    const processor = if (root.get("image_processor")) |value|
        if (value == .object) value.object else root
    else
        root;

    var defaults = VisionProcessorDefaults{
        .min_pixels = positiveJsonU32(processor.get("min_pixels")),
        .max_pixels = positiveJsonU32(processor.get("max_pixels")),
        .max_image_tokens = positiveJsonU32(processor.get("max_image_tokens")),
    };
    if (processor.get("size")) |value| {
        if (value == .object) {
            if (defaults.min_pixels == null)
                defaults.min_pixels = positiveJsonU32(value.object.get("shortest_edge"));
            if (defaults.max_pixels == null)
                defaults.max_pixels = positiveJsonU32(value.object.get("longest_edge"));
        }
    }
    if (defaults.min_pixels != null and
        defaults.max_pixels != null and
        defaults.min_pixels.? > defaults.max_pixels.?)
    {
        return .{};
    }
    return defaults;
}

pub const SoftTokenBudgets = struct { image: ?u32 = null, video: ?u32 = null, max_frames: ?u32 = null };

/// The soft-token budget each EmbeddingGemma 2 processor accepts; any other value makes the processor raise.
fn validSoftTokenBudget(value: ?std.json.Value) ?u32 {
    const n = positiveJsonU32(value) orelse return null;
    return switch (n) {
        70, 140, 280, 560, 1120 => n,
        else => null,
    };
}

/// `processor_config.json` of an EmbeddingGemma 2 checkpoint: the image and video processors' budgets and the
/// video frame cap. A missing or unusable field stays null so the processors' defaults hold.
pub fn parseSoftTokenBudgetsFromJson(allocator: std.mem.Allocator, content: []const u8) SoftTokenBudgets {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};
    const root = parsed.value.object;
    var out = SoftTokenBudgets{};
    if (root.get("image_processor")) |v| {
        if (v == .object) out.image = validSoftTokenBudget(v.object.get("max_soft_tokens"));
    }
    if (root.get("video_processor")) |v| {
        if (v == .object) {
            out.video = validSoftTokenBudget(v.object.get("max_soft_tokens"));
            out.max_frames = positiveJsonU32(v.object.get("max_frames"));
        }
    }
    return out;
}

/// Pure parser for generation_config.json content. Total: malformed JSON or
/// out-of-range values yield nulls — a corrupt config must never pin
/// sampling to an extreme. (`do_sample` is deliberately ignored: HF uses it
/// for greedy-vs-sample mode selection, which the request's own temperature
/// already expresses.)
pub fn parseGenerationDefaultsFromJson(content: []const u8) GenerationDefaults {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};
    const root = parsed.value.object;

    var gd = GenerationDefaults{};
    if (root.get("temperature")) |v| {
        const t: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (t) |tv| {
            if (tv >= 0.0 and tv <= 2.0) gd.temperature = tv;
        }
    }
    if (root.get("top_p")) |v| {
        const p: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (p) |pv| {
            if (pv > 0.0 and pv <= 1.0) gd.top_p = pv;
        }
    }
    if (root.get("top_k")) |v| {
        switch (v) {
            .integer => |i| if (i > 0 and i <= 1000) {
                gd.top_k = @intCast(i);
            },
            else => {},
        }
    }
    if (root.get("min_p")) |v| {
        const m: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (m) |mv| {
            if (mv >= 0.0 and mv <= 1.0) gd.min_p = mv;
        }
    }
    if (root.get("eos_token_id")) |v| {
        switch (v) {
            .integer => |i| if (i >= 0) {
                gd.eos_token_ids[0] = @intCast(i);
                gd.num_eos = 1;
            },
            .array => |arr| for (arr.items) |item| {
                if (item == .integer and item.integer >= 0 and gd.num_eos < gd.eos_token_ids.len) {
                    gd.eos_token_ids[gd.num_eos] = @intCast(item.integer);
                    gd.num_eos += 1;
                }
            },
            else => {},
        }
    }
    // The checkpoint's own chat-template kwargs. Only a real bool counts —
    // anything else leaves the field null and the arch default in charge.
    if (root.get("default_chat_template_kwargs")) |v| {
        if (v == .object) {
            if (v.object.get("enable_thinking")) |et| {
                if (et == .bool) gd.enable_thinking = et.bool;
            }
        }
    }
    return gd;
}

/// One qwen4_exp integer bound, read strictly: wrong-typed or negative refuses.
fn qwen4ConfigU64(cfg_obj: std.json.ObjectMap, key: []const u8) !?u64 {
    const v = jsonField(cfg_obj, key) orelse return null;
    if (v != .integer or v.integer < 0) return error.InvalidQwen4ConfigField;
    return @intCast(v.integer);
}

fn qwen4ConfigU32(cfg_obj: std.json.ObjectMap, key: []const u8) !?u32 {
    const v = try qwen4ConfigU64(cfg_obj, key) orelse return null;
    if (v > std.math.maxInt(u32)) return error.InvalidQwen4ConfigField;
    return @intCast(v);
}

/// Range-check every qwen4_exp bound the forward indexes a fixed array with or divides by.
/// Names travel to the client as "Model load failed: <name>".
fn validateQwen4Config(config: *const ModelConfig) !void {
    // `NgramHash.multipliers` is [MAX_NGRAM_SIZE]i64; `ple_prev` is written ngram_size-1 deep.
    if (config.ngram_size < 2 or config.ngram_size > qwen4_exp.MAX_NGRAM_SIZE) {
        return error.InvalidQwen4NgramSize;
    }
    // `vocab`/`offsets` are [MAX_HEADS]i64, written n_heads deep.
    if (config.heads_per_ngram == 0) return error.InvalidQwen4NgramHeads;
    if (@as(u64, config.ngram_size - 1) * config.heads_per_ngram > qwen4_exp.MAX_HEADS) {
        return error.InvalidQwen4NgramHeads;
    }
    if (config.ngram_vocab_divisor == 0 or config.ngram_vocab_base < 2) {
        return error.InvalidQwen4NgramVocab;
    }
    if (config.ple_embed_dim == 0 or config.ple_embed_dim % ((config.ngram_size - 1) * config.heads_per_ngram) != 0 or config.split_ngram_parts == 0) {
        return error.InvalidQwen4PleGeometry;
    }
    // The forward divides kv by the ratio and selects `budget / ratio` blocks.
    if (config.indexer_n_heads > 0) {
        if (config.indexer_head_dim == 0) return error.InvalidQwen4Indexer;
        if (config.indexer_compress_ratio == 0) return error.InvalidQwen4Indexer;
        if (config.indexer_budget < config.indexer_compress_ratio) return error.InvalidQwen4Indexer;
    }
    if (config.ple_layer_idx < 0 or config.ple_layer_idx >= @as(i64, config.num_hidden_layers)) {
        return error.InvalidQwen4PleLayer;
    }
}

pub fn qwen4EmbeddedSpec(config: *const ModelConfig) !qwen4_exp.EmbeddedSpec {
    const hash = try qwen4_exp.NgramHash.init(config.vocab_size, config.ngram_size, config.heads_per_ngram, config.ngram_vocab_base, config.ngram_vocab_divisor, config.ngram_seed, 0, config.ngram_eos);
    if (config.ple_embed_dim == 0 or config.ple_embed_dim % hash.n_heads != 0) return error.InvalidQwen4PleGeometry;
    if (config.ple_layer_idx < 0) return error.InvalidQwen4PleLayer;
    return .{ .rows = hash.total_rows, .dim = config.ple_embed_dim / hash.n_heads, .shards = config.split_ngram_parts, .layer_index = @intCast(config.ple_layer_idx) };
}

/// True when the layer loop installed the PLE on exactly the layer the config names. A negative
/// index asks for no PLE (the MTP head's own layer) and is satisfied by a loop that installed none.
pub fn qwen4PleInstalledAt(has_ple: []const bool, ple_layer_idx: i32) bool {
    if (ple_layer_idx < 0) return std.mem.indexOfScalar(bool, has_ple, true) == null;
    if (ple_layer_idx >= has_ple.len) return false;
    const want: usize = @intCast(ple_layer_idx);
    for (has_ple, 0..) |p, i| if (p != (i == want)) return false;
    return true;
}

/// I/O-free variant for unit tests and for callers that already have the
/// config.json bytes in memory. The full I/O-bound `parseConfig` delegates here.
/// Qwen3-VL-family vision + M-RoPE fields, shared by the qwen3_5 and
/// qwen4_exp arms (same `vision_config` keys, `rope_parameters.mrope_*`,
/// vision token ids). The generic vision_config block already set
/// `has_vision`; this reads Qwen's own keys into `qv_*`.
/// One block for every packed module; mixed blocks are not a layout we serve.
fn prismHadamardBlock(root: std.json.ObjectMap) !u32 {
    const modules = (try jsonValue(.array, root.get("modules") orelse return error.UnsupportedHadamardLayout)).items;
    var block: u32 = 0;
    for (modules) |m| {
        const module = try jsonValue(.object, m);
        const b = try jsonU32(module.get("block") orelse return error.UnsupportedHadamardLayout);
        if (b <= 0 or (block != 0 and b != block)) return error.UnsupportedHadamardLayout;
        block = b;
    }
    if (block == 0) return error.UnsupportedHadamardLayout;
    return block;
}

fn parseQwenVisionFields(config: *ModelConfig, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    if (jsonField(root, "vision_config")) |vc_val| {
        if (vc_val == .object) {
            const vc = vc_val.object;
            config.qwen_vision = true;
            if (jsonField(vc, "depth")) |v| {
                if (v == .integer) config.qv_depth = try jsonU32(v);
            }
            if (jsonField(vc, "hidden_size")) |v| {
                if (v == .integer) config.qv_hidden = try jsonU32(v);
            }
            if (jsonField(vc, "num_heads")) |v| {
                if (v == .integer) config.qv_heads = try jsonU32(v);
            }
            if (jsonField(vc, "intermediate_size")) |v| {
                if (v == .integer) config.qv_intermediate = try jsonU32(v);
            }
            if (jsonField(vc, "patch_size")) |v| {
                if (v == .integer) config.qv_patch = try jsonU32(v);
            }
            if (jsonField(vc, "temporal_patch_size")) |v| {
                if (v == .integer) config.qv_temporal_patch = try jsonU32(v);
            }
            if (jsonField(vc, "spatial_merge_size")) |v| {
                if (v == .integer) config.qv_merge = try jsonU32(v);
            }
            if (jsonField(vc, "num_position_embeddings")) |v| {
                if (v == .integer) config.qv_num_pos_emb = try jsonU32(v);
            }
            if (jsonField(vc, "out_hidden_size")) |v| {
                if (v == .integer) config.qv_out_hidden = try jsonU32(v);
            }
            if (config.qv_heads != 0) config.qv_head_dim = config.qv_hidden / config.qv_heads;
            if (config.qv_out_hidden == 0) config.qv_out_hidden = config.hidden_size;
        }
    }
    // Interleaved M-RoPE sections (text_config.rope_parameters). rope_theta /
    // partial_rotary_factor already parsed in the generic rope block above.
    if (jsonField(cfg_obj, "rope_parameters")) |rp| {
        if (rp == .object) {
            if (jsonField(rp.object, "mrope_interleaved")) |v| {
                if (v == .bool) config.mrope_interleaved = v.bool;
            }
            if (jsonField(rp.object, "mrope_section")) |v| {
                if (v == .array) {
                    for (v.array.items, 0..) |item, i| {
                        if (i >= 3) break;
                        if (item == .integer) config.mrope_section[i] = try jsonU32(item);
                    }
                }
            }
        }
    }
    // Qwen vision token ids (top-level).
    if (jsonField(root, "video_token_id")) |v| {
        if (v == .integer) config.video_token_id = try jsonU32(v);
    }
    if (jsonField(root, "vision_start_token_id")) |v| {
        if (v == .integer) config.vision_start_token_id = try jsonU32(v);
    }
    if (jsonField(root, "vision_end_token_id")) |v| {
        if (v == .integer) config.vision_end_token_id = try jsonU32(v);
    }
}

/// Flat HF `rope_parameters` carrying `rope_type: "yarn"` — the YaRN context
/// extension, i.e. exactly what vLLM's `--hf-overrides` recipe for Qwen3.5
/// writes:
///
///   {"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
///     "original_max_position_embeddings": 262144, "rope_theta": 10000000,
///     "partial_rotary_factor": 0.25, "mrope_interleaved": true,
///     "mrope_section": [11,11,10]}}}
///
/// `factor` may be omitted, in which case HF derives it from
/// `max_position_embeddings / original_max_position_embeddings` (as vLLM's
/// `_get_and_verify_max_len` does). `attention_factor` is HF's key and
/// REPLACES the computed mscale; `attn_factor` is vLLM's and MULTIPLIES
/// `yarnMscale(factor)`. Neither is present in a vendor config, and per HF's
/// default the mscale is then COMPUTED as 0.1·ln(factor)+1 — the value the
/// scaling was calibrated with. Nested per-layer-type `rope_parameters`
/// (laguna/gemma4) never reach here: they have no top-level `rope_type`.
fn parseYarnRopeParameters(config: *ModelConfig, cfg_obj: std.json.ObjectMap) !void {
    const rp_val = jsonField(cfg_obj, "rope_parameters") orelse return;
    if (rp_val != .object) return;
    const rp = rp_val.object;
    const rt = jsonField(rp, "rope_type") orelse return;
    if (!(rt == .string and std.mem.eql(u8, rt.string, "yarn"))) return;

    if (jsonField(rp, "original_max_position_embeddings")) |v| {
        if (v == .integer) config.yarn_orig_max_pos = try jsonU32(v);
    }
    // A YaRN block with no window to scale FROM is not a scaling we can
    // reproduce: the ramp bounds (and so every mid-band frequency) come from
    // it. Refuse the load rather than serve a silently-wrong rotation.
    if (config.yarn_orig_max_pos == 0) return error.YarnRopeNeedsOriginalMaxPos;
    if (jsonField(rp, "factor")) |v| config.yarn_factor = try jsonFloat(v);
    if (jsonField(rp, "beta_fast")) |v| config.yarn_beta_fast = try jsonFloat(v);
    if (jsonField(rp, "beta_slow")) |v| config.yarn_beta_slow = try jsonFloat(v);
    if (jsonField(rp, "truncate")) |v| {
        if (v == .bool) config.yarn_truncate = v.bool;
    }
    if (config.yarn_factor <= 0.0) return error.InvalidRopeScalingFactor;
    // HF: `factor = max_position_embeddings / original_max_position_embeddings`
    // when the block leaves it out (the config then only states the window).
    if (jsonField(rp, "factor") == null and config.max_position_embeddings > config.yarn_orig_max_pos) {
        config.yarn_factor = @as(f32, @floatFromInt(config.max_position_embeddings)) /
            @as(f32, @floatFromInt(config.yarn_orig_max_pos));
    }
    // HF `attention_factor` replaces; vLLM `attn_factor` multiplies the
    // computed 0.1·ln(factor)+1. Both present → HF wins.
    if (jsonField(rp, "attention_factor")) |v| {
        config.yarn_attention_factor = try jsonFloat(v);
    } else if (jsonField(rp, "attn_factor")) |v| {
        config.yarn_attention_factor = yarnMscale(config.yarn_factor) * try jsonFloat(v);
    } else {
        config.yarn_attention_factor = yarnMscale(config.yarn_factor);
    }
    if (!std.math.isFinite(config.yarn_attention_factor)) return error.InvalidConfigField;
    config.rope_yarn = true;
}

/// HF's default YaRN mscale (`attention_factor`) for a scaling `factor`.
fn yarnMscale(factor: f32) f32 {
    if (factor <= 1.0) return 1.0;
    return 0.1 * @log(@as(f32, factor)) + 1.0;
}

/// Launch-time JSON deep-merged into every `config.json` before it is parsed,
/// set once from `--config-overrides`. vLLM's `--hf-overrides` analogue: the
/// only way to re-shape a checkpoint's declared geometry — most often to scale
/// its rope and widen the context — without editing the model directory, and
/// therefore the way to A/B a scaling experiment on identical weights.
var config_overrides: ?[]const u8 = null;

pub fn setConfigOverrides(raw: ?[]const u8) void {
    config_overrides = raw;
}

pub fn getConfigOverrides() ?[]const u8 {
    return config_overrides;
}

/// Deep-merge `overrides` into a config.json document: objects merge key by key
/// — so `{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4}}}`
/// keeps every sibling it passes through, exactly like vLLM's
/// `_apply_dict_overrides` — and anything else replaces. The whole merge lives
/// in an arena that dies before this returns; only the re-serialized bytes (in
/// `allocator`) escape.
fn mergeConfigJson(allocator: std.mem.Allocator, base: []const u8, overrides: []const u8) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dst = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    const src = try std.json.parseFromSliceLeaky(std.json.Value, a, overrides, .{});
    if (dst != .object or src != .object) return error.ConfigOverridesMustBeObject;
    if (src.object.contains("qwen4_norm_convention")) {
        log.err("[qwen4] qwen4_norm_convention cannot be set through server-wide --config-overrides; remove that override and use a checkpoint-local config.json marker.\n", .{});
        return error.GlobalQwen4NormOverride;
    }
    try mergeObjects(a, &dst.object, src.object);
    var out: std.Io.Writer.Allocating = .init(a);
    var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try dst.jsonStringify(&jws);
    return allocator.dupe(u8, out.written());
}

fn mergeObjects(a: std.mem.Allocator, dst: *std.json.ObjectMap, src: std.json.ObjectMap) !void {
    var it = src.iterator();
    while (it.next()) |e| {
        if (dst.getPtr(e.key_ptr.*)) |p| {
            if (p.* == .object and e.value_ptr.* == .object) {
                // The handle is copied, so a rehash inside the recursion would
                // be lost — merge through the copy and store it back. `p` stays
                // valid: `dst` itself is not written during the recursion.
                var child = p.object;
                try mergeObjects(a, &child, e.value_ptr.object);
                p.* = .{ .object = child };
                continue;
            }
        }
        // Keys and values are arena-owned by the override document, which
        // outlives this merge.
        try dst.put(a, e.key_ptr.*, e.value_ptr.*);
    }
}

/// EmbeddingGemma 2's context window as its model card states it. Kept in sync with
/// `model_discovery.parseStubMeta`, which cannot import this file.
pub const EMBEDDING_GEMMA2_WINDOW: u32 = 8192;

/// `per_layer_config` gives the full-attention layers a head width and KV head count of their own. The loader
/// keeps one such geometry (`global_head_dim`, `num_global_key_value_heads`), so entries that name a sliding
/// layer, disagree with each other or leave a full layer out are refused. An absent block is HF's default.
fn parseEmbeddingGemma2Geometry(cfg_obj: std.json.ObjectMap, config: *ModelConfig) !void {
    if (config.num_hidden_layers > config.layer_is_global.len) return error.UnsupportedLayerGeometry;
    config.global_head_dim = 512;
    config.num_global_key_value_heads = 1;
    const block = jsonField(cfg_obj, "per_layer_config") orelse return;
    var listed: @TypeOf(config.layer_is_global) = @splat(false);
    var first = true;
    var it = (try jsonValue(.object, block)).iterator();
    while (it.next()) |e| {
        const layer = std.fmt.parseInt(u32, e.key_ptr.*, 10) catch return error.UnsupportedLayerGeometry;
        if (layer >= config.num_hidden_layers or !config.isGlobalLayer(layer)) return error.UnsupportedLayerGeometry;
        const entry = try jsonValue(.object, e.value_ptr.*);
        const head_dim = try jsonU32(jsonField(entry, "head_dim") orelse return error.UnsupportedLayerGeometry);
        const kv_heads = try jsonU32(jsonField(entry, "num_key_value_heads") orelse return error.UnsupportedLayerGeometry);
        if (!first and (head_dim != config.global_head_dim or kv_heads != config.num_global_key_value_heads)) return error.UnsupportedLayerGeometry;
        config.global_head_dim = head_dim;
        config.num_global_key_value_heads = kv_heads;
        listed[layer] = true;
        first = false;
    }
    for (0..config.num_hidden_layers) |layer| {
        if (!first and config.isGlobalLayer(@intCast(layer)) and !listed[layer]) return error.UnsupportedLayerGeometry;
    }
}

/// The MiMo release (FP8 trunk, per-expert MXFP4 bytes) is not a pack; mlx packs
/// carry an mlx-style quantization_config with no `quant_method`.
fn refuseUnconvertedMimo(cfg_obj: std.json.ObjectMap) !void {
    const qc = jsonField(cfg_obj, "quantization_config") orelse return;
    if (qc == .object and jsonField(qc.object, "quant_method") != null) {
        log.err("mimo_v2: this is the original release; convert it first with tests/convert_mimo_v2.py\n", .{});
        return error.UnconvertedMimoCheckpoint;
    }
}

/// A Sushi pack's `expert_quant` makes its routed experts EXL3 banks.
fn parseSushiExperts(root: std.json.ObjectMap, config: *ModelConfig) !void {
    if (jsonField(root, "expert_quant") == null) return;
    config.exl3 = try sushi_exl3.parseExpertQuant(root);
    try sushi_exl3.admitTopK(config.num_experts_per_tok);
}

pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !ModelConfig {
    // The launch-time overrides apply to EVERY parse (primary load, on-demand
    // load, discovery stubs), so the advertised context and the loaded model
    // can never disagree about what window the checkpoint has.
    const merged: ?[]const u8 = if (config_overrides) |ov| blk: {
        break :blk try mergeConfigJson(allocator, content, ov);
    } else null;
    defer if (merged) |m| allocator.free(m);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, merged orelse content, .{});
    defer parsed.deinit();

    const root = try jsonValue(.object, parsed.value);
    var config = ModelConfig{};

    // Detect model_type from top-level (always present)
    const model_type = if (jsonField(root, "model_type")) |v| try jsonValue(.string, v) else "gemma3";

    // Determine which object to read config from: text_config (nested) or root (flat)
    const cfg_obj = if (jsonField(root, "text_config")) |tc_val| try jsonValue(.object, tc_val) else root;

    // Parse common fields
    if (jsonField(cfg_obj, "vocab_size")) |v| config.vocab_size = try jsonU32(v);
    if (jsonField(cfg_obj, "hidden_size")) |v| config.hidden_size = try jsonU32(v);
    if (jsonField(cfg_obj, "intermediate_size")) |v| {
        config.intermediate_size = try jsonU32(v);
        config.intermediate_size_declared = true;
    }
    if (jsonField(cfg_obj, "num_hidden_layers")) |v| config.num_hidden_layers = try jsonU32(v);
    if (jsonField(cfg_obj, "num_attention_heads")) |v| config.num_attention_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "num_key_value_heads")) |v| config.num_key_value_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "head_dim")) |v| config.head_dim = try jsonU32(v);
    if (jsonField(cfg_obj, "max_position_embeddings")) |v| config.max_position_embeddings = try jsonU32(v);
    if (jsonField(cfg_obj, "rms_norm_eps")) |v| config.rms_norm_eps = try jsonFloat(v);
    if (jsonField(cfg_obj, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
    if (jsonField(cfg_obj, "query_pre_attn_scalar")) |v| config.query_pre_attn_scalar = try jsonU32(v);

    // MoE fields (guard against JSON null values)
    if (jsonField(cfg_obj, "num_experts")) |v| {
        if (v == .integer) config.num_experts = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_experts_per_tok")) |v| {
        if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "top_k_experts")) |v| {
        if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "moe_intermediate_size")) |v| {
        if (v == .integer) config.moe_intermediate_size = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "shared_expert_intermediate_size")) |v| {
        if (v == .integer) config.shared_expert_intermediate_size = try jsonU32(v);
    }

    // Linear attention (GatedDeltaNet) fields
    if (jsonField(cfg_obj, "linear_num_key_heads")) |v| config.linear_num_key_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "linear_num_value_heads")) |v| config.linear_num_value_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "linear_key_head_dim")) |v| config.linear_key_head_dim = try jsonU32(v);
    if (jsonField(cfg_obj, "linear_value_head_dim")) |v| config.linear_value_head_dim = try jsonU32(v);
    if (jsonField(cfg_obj, "linear_conv_kernel_dim")) |v| config.linear_conv_kernel_dim = try jsonU32(v);

    // Hybrid attention
    if (jsonField(cfg_obj, "full_attention_interval")) |v| config.full_attention_interval = try jsonU32(v);
    if (jsonField(cfg_obj, "attn_output_gate")) |v| {
        if (v == .bool) config.attn_output_gate = v.bool;
    }

    // Bidirectional-attention embedding models (EmbeddingGemma): a decoder
    // arch trained as an encoder. Routes to the encoder forward + the
    // /v1/embeddings surface; chat surfaces reject it.
    if (jsonField(cfg_obj, "use_bidirectional_attention")) |v| {
        if (v == .bool and v.bool) {
            config.use_bidirectional_attention = true;
            config.is_encoder_only = true;
        }
    }
    // Explicit pooling contract (issue #116): "mean" | "cls" | "last_token" in
    // config.json marks a checkpoint as an embedding model and picks the pool
    // op. An unknown value is a parse error, never a silent mean-pool —
    // wrong-semantics vectors are harder to detect than a refused load.
    if (jsonField(root, "pooling_mode")) |v| {
        if (v == .string) {
            config.pooling_mode = PoolingMode.fromString(v.string) orelse
                return error.UnsupportedPoolingMode;
        }
    }
    if (jsonField(cfg_obj, "bos_token_id")) |v| {
        if (v == .integer and v.integer >= 0) config.bos_token_id = try jsonU32(v);
    }

    // Rope parameters (nested for Qwen3.5)
    if (jsonField(cfg_obj, "rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            if (jsonField(rp_val.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
            if (jsonField(rp_val.object, "partial_rotary_factor")) |v| config.partial_rotary_factor = try jsonFloat(v);
        }
    }

    // Sliding window
    if (cfg_obj.get("sliding_window")) |v| {
        if (v == .null) {
            config.has_sliding_window = false;
        } else {
            config.sliding_window = try jsonU32(v);
            config.has_sliding_window = true;
        }
    }
    if (jsonField(cfg_obj, "sliding_window_pattern")) |v| config.sliding_window_pattern = try jsonU32(v);

    // Gemma-specific: dual RoPE bases
    if (jsonField(cfg_obj, "rope_local_base_freq")) |v| config.rope_local_base_freq = try jsonFloat(v);
    if (jsonField(cfg_obj, "rope_scaling")) |rs_val| {
        if (rs_val == .object) {
            if (jsonField(rs_val.object, "factor")) |v| config.rope_scaling_factor = try jsonFloat(v);
        }
    }

    // Gemma 4: explicit layer_types array
    if (jsonField(cfg_obj, "layer_types")) |lt_val| {
        if (lt_val == .array) {
            config.has_explicit_layer_types = true;
            for (lt_val.array.items, 0..) |item, i| {
                if (i >= 128) break;
                if (item == .string) config.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
            }
        }
    }

    // Gemma 4: dual head dimensions and KV sharing
    if (jsonField(cfg_obj, "global_head_dim")) |v| {
        if (v == .integer) config.global_head_dim = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_global_key_value_heads")) |v| {
        if (v == .integer) config.num_global_key_value_heads = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_kv_shared_layers")) |v| {
        if (v == .integer) config.num_kv_shared_layers = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "attention_k_eq_v")) |v| {
        if (v == .bool) config.attention_k_eq_v = v.bool;
    }
    if (jsonField(cfg_obj, "final_logit_softcapping")) |v| {
        config.final_logit_softcapping = try jsonFloat(v);
    }
    if (jsonField(cfg_obj, "hidden_size_per_layer_input")) |v| {
        if (v == .integer) config.hidden_size_per_layer_input = try jsonU32(v);
    }

    // Gemma 4: nested rope_parameters with per-attention-type config
    if (jsonField(cfg_obj, "rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            // Gemma 4 style: { "full_attention": {...}, "sliding_attention": {...} }
            if (jsonField(rp_val.object, "full_attention")) |fa| {
                if (fa == .object) {
                    if (jsonField(fa.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
                    if (jsonField(fa.object, "partial_rotary_factor")) |v| config.partial_rotary_factor_global = try jsonFloat(v);
                    if (jsonField(fa.object, "rope_type")) |v| {
                        if (v == .string and std.mem.eql(u8, v.string, "proportional")) {
                            config.rope_proportional = true;
                            if (jsonField(fa.object, "factor")) |fv| config.rope_proportional_factor = try jsonFloat(fv);
                        }
                    }
                }
            }
            if (jsonField(rp_val.object, "sliding_attention")) |sa| {
                if (sa == .object) {
                    if (jsonField(sa.object, "rope_theta")) |v| config.rope_local_base_freq = try jsonFloat(v);
                }
            }
            // Qwen3.5 style: { "rope_theta": ..., "partial_rotary_factor": ... }
            if (jsonField(rp_val.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
            if (jsonField(rp_val.object, "partial_rotary_factor")) |v| config.partial_rotary_factor = try jsonFloat(v);
        }
    }

    // Tie word embeddings
    if (jsonField(root, "tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }
    if (jsonField(cfg_obj, "tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }

    // Check root level for max_position_embeddings (may not be in text_config)
    if (config.max_position_embeddings == 0) {
        if (jsonField(root, "max_position_embeddings")) |v| {
            if (v == .integer) config.max_position_embeddings = try jsonU32(v);
        }
    }

    // Parse quantization from top level
    if (jsonField(root, "quantization")) |q_val| {
        const q = try jsonValue(.object, q_val);
        if (jsonField(q, "bits")) |v| config.quant_bits = try jsonU32(v);
        if (jsonField(q, "group_size")) |v| config.quant_group_size = try jsonU32(v);
        if (jsonField(q, "mode")) |v| {
            if (v == .string) {
                config.quant_mode = QuantMode.fromString(v.string) orelse {
                    log.err("unsupported quantization mode '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)\n", .{v.string});
                    return error.UnsupportedQuantMode;
                };
            }
        }
        // MLX ships affine kernels only for bits {2,3,4,5,6,8} (ops.cpp
        // rejects the rest at quantize() time, but an ALREADY-quantized
        // checkpoint skips that check and dies at Metal kernel load during
        // warmup — an uncatchable process kill). Reject at parse instead.
        if (config.quant_mode == .affine and config.quant_bits != 0) {
            switch (config.quant_bits) {
                2, 3, 4, 5, 6, 8 => {},
                else => {
                    log.err("unsupported affine quantization: {d}-bit (this MLX runtime supports 2, 3, 4, 5, 6, 8)\n", .{config.quant_bits});
                    return error.UnsupportedQuantBits;
                },
            }
        }
    }

    // EOS tokens
    if (jsonField(root, "eos_token_id")) |v| {
        switch (v) {
            .integer => |i| config.addEosToken(std.math.cast(u32, i) orelse return error.InvalidConfigField),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .integer) config.addEosToken(try jsonU32(item));
                }
            },
            else => {},
        }
    }

    // Vision config (Gemma 4 SigLIP)
    if (jsonField(root, "vision_config")) |vc_val| {
        if (vc_val == .object) {
            config.has_vision = true;
            const vc = vc_val.object;
            if (jsonField(vc, "hidden_size")) |v| {
                if (v == .integer) config.vision_hidden_size = try jsonU32(v);
            }
            if (jsonField(vc, "num_hidden_layers")) |v| {
                if (v == .integer) config.vision_num_layers = try jsonU32(v);
            }
            if (jsonField(vc, "num_attention_heads")) |v| {
                if (v == .integer) config.vision_num_heads = try jsonU32(v);
            }
            if (jsonField(vc, "head_dim")) |v| {
                if (v == .integer) config.vision_head_dim = try jsonU32(v);
            }
            if (jsonField(vc, "global_head_dim")) |v| {
                if (v == .integer) config.vision_head_dim = try jsonU32(v);
            }
            if (jsonField(vc, "intermediate_size")) |v| {
                if (v == .integer) config.vision_intermediate_size = try jsonU32(v);
            }
            if (jsonField(vc, "patch_size")) |v| {
                if (v == .integer) config.vision_patch_size = try jsonU32(v);
            }
            if (jsonField(vc, "pooling_kernel_size")) |v| {
                if (v == .integer) config.vision_pooling_kernel = try jsonU32(v);
            }
            if (jsonField(vc, "default_output_length")) |v| {
                if (v == .integer) config.vision_soft_tokens = try jsonU32(v);
            }
            if (jsonField(vc, "position_embedding_size")) |v| {
                if (v == .integer) config.vision_position_embedding_size = try jsonU32(v);
            }
            if (jsonField(vc, "rope_parameters")) |rp| {
                if (rp == .object) {
                    if (jsonField(rp.object, "rope_theta")) |v| config.vision_rope_theta = try jsonFloat(v);
                }
            }
            if (jsonField(vc, "use_clipped_linears")) |v| {
                if (v == .bool) config.vision_use_clipped_linears = v.bool;
            }
            // vision_config.standardize is presence-only — the actual `std_scale`/`std_bias`
            // safetensors presence drives behavior in `VisionEncoder.init`, so the config
            // flag needs no field.

            // Gemma 4 12B unified (encoder-free) vision fields. Distinct names
            // from the SigLIP tower: mm_embed_dim (vs hidden_size),
            // model_patch_size (48px merged patch vs 16px teacher patch_size),
            // num_soft_tokens (vs default_output_length), mm_posemb_size.
            if (jsonField(vc, "mm_embed_dim")) |v| {
                if (v == .integer) config.vision_mm_embed_dim = try jsonU32(v);
            }
            if (jsonField(vc, "model_patch_size")) |v| {
                if (v == .integer) config.vision_model_patch_size = try jsonU32(v);
            }
            if (jsonField(vc, "num_soft_tokens")) |v| {
                if (v == .integer) config.vision_soft_tokens = try jsonU32(v);
            }
            if (jsonField(vc, "mm_posemb_size")) |v| {
                if (v == .integer) config.vision_mm_posemb_size = try jsonU32(v);
            }
        }
    }
    // Audio config (Gemma 4 12B unified — raw-waveform projection, no conformer)
    if (jsonField(root, "audio_config")) |ac_val| {
        if (ac_val == .object) {
            const ac = ac_val.object;
            if (jsonField(ac, "audio_embed_dim")) |v| {
                if (v == .integer) config.audio_embed_dim = try jsonU32(v);
            }
            // audio_samples_per_token lives in processor_config, not config.json;
            // default 640 (40ms @ 16kHz) matches the only shipped unified checkpoint.
            if (jsonField(ac, "audio_samples_per_token")) |v| {
                if (v == .integer) config.audio_samples_per_token = try jsonU32(v);
            }
        }
    }
    if (jsonField(root, "audio_token_id")) |v| {
        if (v == .integer) config.audio_token_id = try jsonU32(v);
    }
    if (jsonField(root, "boa_token_id")) |v| {
        if (v == .integer) config.boa_token_id = try jsonU32(v);
    }
    // eoa lives under `eoa_token_index` in the unified config.json.
    if (jsonField(root, "eoa_token_index")) |v| {
        if (v == .integer) config.eoa_token_id = try jsonU32(v);
    }
    if (jsonField(root, "eoa_token_id")) |v| {
        if (v == .integer and config.eoa_token_id == 0) config.eoa_token_id = try jsonU32(v);
    }
    // Image token ID (top-level or in mm_tokens_per_image config)
    if (jsonField(root, "image_token_id")) |v| {
        if (v == .integer) config.image_token_id = try jsonU32(v);
    }
    if (jsonField(root, "image_token_index")) |v| {
        if (v == .integer and config.image_token_id == 0) config.image_token_id = try jsonU32(v);
    }
    if (jsonField(root, "boi_token_id")) |v| {
        if (v == .integer) config.boi_token_id = try jsonU32(v);
    }
    if (jsonField(root, "eoi_token_id")) |v| {
        if (v == .integer) config.eoi_token_id = try jsonU32(v);
    }

    // Set model-family defaults based on model_type
    if (std.mem.eql(u8, model_type, "gemma3") or
        std.mem.eql(u8, model_type, "gemma3_text"))
    {
        // "gemma3_text" is the FLAT text-only checkpoint (Gemma3ForCausalLM,
        // e.g. the abliterated -lm- builds): no vision tower, weights under
        // "model.*". The multimodal "gemma3" nests them under
        // "language_model.model.*" and carries a text_config. Collapse both
        // onto "gemma3" so every downstream gemma3 comparison fires.
        config.model_type = "gemma3";
        // gemma-3-4b-it's text_config omits num_attention_heads / num_key_value_heads
        // / head_dim and leans on the HF Gemma3TextConfig defaults (8 q-heads,
        // 4 kv-heads, head_dim 256). Our struct defaults are the 12b/27b shape
        // (16/8), so apply the HF defaults explicitly when the config is silent —
        // otherwise the Q projection (8*256) reshapes against 16 heads and the
        // model crashes at warmup (issue #43). The 12b ships these fields, so its
        // values are read at lines 458-460 and these fills never fire.
        if (jsonField(cfg_obj, "num_attention_heads") == null) config.num_attention_heads = 8;
        if (jsonField(cfg_obj, "num_key_value_heads") == null) config.num_key_value_heads = 4;
        if (jsonField(cfg_obj, "head_dim") == null) config.head_dim = 256;
        // Multimodal checkpoint → "language_model.model"; flat text-only
        // (gemma3_text, no text_config) → "model" (mirrors the LFM2 VL split).
        config.weight_prefix = if (jsonField(root, "text_config") != null) "language_model.model" else "model";
        // Gemma always ties word embeddings; the abliterated text-only build
        // omits the flag (would default false) and ships no lm_head tensor, so
        // force it on. An explicit lm_head tensor, if present, still wins in
        // transformer.zig's resolution.
        config.tie_word_embeddings = true;
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = true;
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        if (config.rope_scaling_factor == 1.0) {
            if (jsonField(cfg_obj, "rope_scaling")) |rs_val| {
                if (rs_val == .object) {
                    if (jsonField(rs_val.object, "factor")) |_| {} else {
                        config.rope_scaling_factor = 8.0;
                    }
                } else {
                    config.rope_scaling_factor = 8.0;
                }
            } else {
                config.rope_scaling_factor = 8.0;
            }
        }
        config.ensureGemmaTerminators();
    } else if (std.mem.eql(u8, model_type, "gemma4") or
        std.mem.eql(u8, model_type, "gemma4_text") or
        std.mem.eql(u8, model_type, "gemma4_unified") or
        std.mem.eql(u8, model_type, "gemma4_unified_text"))
    {
        // Per the Gemma 4 12B developer guide, `gemma4_unified` "contains the
        // same advanced decoder structure as the Gemma 4 31B Dense model" —
        // the unified-ness lives in a tiny vision/audio embedder we don't
        // wire here. So we collapse the internal tag onto plain "gemma4" so
        // every downstream model_type comparison in transformer.zig/drafter.zig
        // (attn_scale gate, recommendedBlockSize, etc.) treats it identically
        // to 31B Dense without per-arch fan-out.
        // Detect the unified (12B encoder-free multimodal) variant before we
        // collapse the tag — drives vision_embedder/embed_audio weight loading
        // and the encoder-free forward in src/vision.zig.
        config.is_gemma4_unified = std.mem.indexOf(u8, model_type, "unified") != null;
        config.model_type = "gemma4";
        config.weight_prefix = "language_model.model";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = false; // Gemma 4 norms have NO offset (plain weight, not 1+weight)
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        config.has_v_norm = true; // Parameter-free RMS norm on values
        config.rope_scaling_factor = 1.0; // No scaling, uses proportional RoPE via theta
        // [1, 106, 50] from the config array (when present) plus an additive
        // guarantee of <eos>(1) + <end_of_turn>(106) for checkpoints that
        // declare only a scalar eos_token_id.
        config.ensureGemmaTerminators();
        // Note on `attention_k_eq_v` for the 12B unified checkpoint: the
        // weights ship separate v_proj for SLIDING layers and omit them for
        // FULL_ATTENTION layers — i.e. K==V alias is a per-layer choice
        // keyed on global-layer-ness. The existing bindModelWeights logic
        // (transformer.zig:5677) already encodes that:
        //   `k_eq_v = config.attention_k_eq_v and isGlobalLayer(li)`.
        // So we leave the parsed flag intact.
    } else if (std.mem.eql(u8, model_type, "embedding_gemma2")) {
        // Google EmbeddingGemma 2: the Gemma 4 text trunk trained as a bidirectional encoder, with per-layer
        // inputs derived from the embeddings alone (no `embed_tokens_per_layer`) and an `embedding_projection`
        // head. The image tower is a Gemma 4 SigLIP tower; the audio tower is not served.
        config.model_type = "embedding_gemma2";
        config.weight_prefix = "language_model";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = false;
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        config.has_v_norm = true;
        config.tie_word_embeddings = true;
        config.is_encoder_only = true;
        config.use_bidirectional_attention = true;
        // The model card's window, not the trunk's 262144-position rope table.
        config.max_position_embeddings = if (config.max_position_embeddings == 0) EMBEDDING_GEMMA2_WINDOW else @min(config.max_position_embeddings, EMBEDDING_GEMMA2_WINDOW);
        try parseEmbeddingGemma2Geometry(cfg_obj, &config);
        if (jsonField(root, "video_token_id")) |v| {
            if (v == .integer) config.video_token_id = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "eos_token_id")) |v| switch (v) {
            .integer => |i| config.addEosToken(std.math.cast(u32, i) orelse return error.InvalidConfigField),
            .array => |arr| for (arr.items) |item| {
                if (item == .integer) config.addEosToken(try jsonU32(item));
            },
            else => {},
        };
    } else if (std.mem.eql(u8, model_type, "diffusion_gemma")) {
        // DiffusionGemma (block diffusion, June 2026). The trunk is the
        // Gemma 4 26B-A4B MoE decoder verbatim — same dual-FFN layer
        // structure, sigma-MoE router, v_norm, dual head geometry,
        // proportional RoPE — under weight prefix `model.decoder`. The
        // model_type stays distinct because GENERATION is different: a
        // bidirectional canvas-denoising loop (src/diffusion.zig), not
        // autoregressive decode.
        config.model_type = "diffusion_gemma";
        config.weight_prefix = "model.decoder";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = false;
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        config.has_v_norm = true;
        config.rope_scaling_factor = 1.0;
        // Full-attention layers ship NO v_proj — V is the param-free-normed
        // k_proj output. Same per-layer alias the Gemma 4 31B/12B binder uses.
        config.attention_k_eq_v = true;
        // canvas_length: top-level; presence is what flags diffusion.
        if (jsonField(root, "canvas_length")) |v| {
            if (v == .integer) config.canvas_length = try jsonU32(v);
        }
        if (config.canvas_length == 0) config.canvas_length = 256;
        // Diffusion knobs from the embedded generation_config object.
        if (jsonField(root, "generation_config")) |gc_val| {
            if (gc_val == .object) {
                const gc = gc_val.object;
                if (jsonField(gc, "max_denoising_steps")) |v| {
                    if (v == .integer) config.diffusion_max_steps = try jsonU32(v);
                }
                if (jsonField(gc, "t_min")) |v| config.diffusion_t_min = try jsonFloat(v);
                if (jsonField(gc, "t_max")) |v| config.diffusion_t_max = try jsonFloat(v);
                if (jsonField(gc, "confidence_threshold")) |v| config.diffusion_confidence_threshold = try jsonFloat(v);
                if (jsonField(gc, "stability_threshold")) |v| {
                    if (v == .integer) config.diffusion_stability_threshold = try jsonU32(v);
                }
                if (jsonField(gc, "pad_token_id")) |v| {
                    if (v == .integer) config.diffusion_pad_token = try jsonU32(v);
                }
                if (jsonField(gc, "sampler_config")) |sc_val| {
                    if (sc_val == .object) {
                        if (jsonField(sc_val.object, "entropy_bound")) |v| config.diffusion_entropy_bound = try jsonFloat(v);
                    }
                }
                // EOS may also live here (mirrors the top-level list).
                if (config.num_eos_tokens == 0) {
                    if (jsonField(gc, "eos_token_id")) |v| {
                        switch (v) {
                            .integer => |i| config.addEosToken(std.math.cast(u32, i) orelse return error.InvalidConfigField),
                            .array => |arr| for (arr.items) |item| {
                                if (item == .integer) config.addEosToken(try jsonU32(item));
                            },
                            else => {},
                        }
                    }
                }
            }
        }
        // The model.encoder.vision_tower is not wired yet (dropped at load by
        // shouldKeepWeightKey) — never advertise vision for this arch.
        config.has_vision = false;
        config.ensureGemmaTerminators();
    } else if (std.mem.eql(u8, model_type, "muse_glimmer") or
        std.mem.eql(u8, model_type, "muse_glimmer_text"))
    {
        // Muse-Glimmer-30B (meta-models). Dense GQA trunk (32/2 heads, hd 128)
        // with Gemma2-style sandwich-norm layers; every 4th layer counted
        // backward from the last is full-attention AND NoPE (layer_rope_theta
        // 0), the rest slide at 2048. "muse_glimmer_text" is the flat
        // text-only sibling (bare "model" prefix, no text_config).
        config.model_type = "muse_glimmer";
        config.weight_prefix = if (jsonField(root, "text_config") != null) "model.language_model" else "model";
        config.hidden_act = .silu;
        config.norm_has_offset = true; // sandwich norms are Gemma2-centered (1+w)…
        config.final_norm_plain = true; // …but model.norm is plain-scale (ones-init)
        config.scale_embeddings = false; // embeddings are RMS-normed, not sqrt(hidden)-scaled
        config.has_pre_ff_norm = true;
        config.has_qk_norm = false; // no q_norm/k_norm tensors in the checkpoint —
        config.qk_norm_weightless = true; // shared weight-less RMS on Q and K instead
        config.normed_embeddings = true;
        config.attn_sigmoid_gate = true;
        // Reference-config class defaults, overridden by explicit keys below.
        config.qk_scale_factor = 3.87;
        config.output_multiplier = 0.19611613513818404;
        config.post_norm_eps = 1e-8;
        if (jsonField(cfg_obj, "qk_scale_factor")) |v| config.qk_scale_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "output_multiplier")) |v| config.output_multiplier = try jsonFloat(v);
        if (jsonField(cfg_obj, "post_norm_eps")) |v| config.post_norm_eps = try jsonFloat(v);
        // ONE theta for every roped layer: sliding layers read
        // rope_local_base_freq in the forward, but muse ships its base only as
        // rope_parameters.rope_theta — without this the Gemma-flavored 10000
        // default mis-rotates all 39 roped layers (the 2026-08-11 first-turn
        // repetition-loop root cause; global layers are NoPE so EVERY rotated
        // layer ran at the wrong base).
        if (jsonField(cfg_obj, "rope_local_base_freq") == null)
            config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "layer_rope_theta")) |lrt| {
            if (lrt == .array) {
                for (lrt.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    config.layer_no_rope[i] = try jsonFloat(item) == 0;
                }
            }
        }
        // Vision tower (src/muse_vision.zig). Muse names its geometry keys its
        // own way and the window/full pattern is a per-layer list, not a stride.
        if (jsonField(root, "vision_config")) |vc_val| {
            if (vc_val == .object) {
                const vc = vc_val.object;
                config.muse_vision = true;
                if (jsonField(vc, "num_hidden_layers")) |v| {
                    if (v == .integer) config.qv_depth = try jsonU32(v);
                }
                if (jsonField(vc, "hidden_size")) |v| {
                    if (v == .integer) config.qv_hidden = try jsonU32(v);
                }
                if (jsonField(vc, "num_attention_heads")) |v| {
                    if (v == .integer) config.qv_heads = try jsonU32(v);
                }
                if (jsonField(vc, "intermediate_size")) |v| {
                    if (v == .integer) config.qv_intermediate = try jsonU32(v);
                }
                if (jsonField(vc, "patch_size")) |v| {
                    if (v == .integer) config.qv_patch = try jsonU32(v);
                }
                if (jsonField(vc, "patch_temporal")) |v| {
                    if (v == .integer) config.qv_temporal_patch = try jsonU32(v);
                }
                if (jsonField(vc, "merge_size")) |v| {
                    if (v == .integer) config.qv_merge = try jsonU32(v);
                }
                if (jsonField(vc, "pos_emb_height")) |v| {
                    if (v == .integer) config.mv_pos_side = try jsonU32(v);
                }
                if (jsonField(vc, "layer_norm_eps")) |v| config.mv_ln_eps = try jsonFloat(v);
                if (jsonField(vc, "rope_parameters")) |rp| {
                    if (rp == .object) {
                        if (jsonField(rp.object, "rope_theta")) |v| config.mv_rope_theta = try jsonFloat(v);
                    }
                }
                if (jsonField(vc, "layer_types")) |lt| {
                    if (lt == .array) for (lt.array.items, 0..) |item, i| {
                        if (i >= MAX_VISION_LAYERS) break;
                        if (item == .string) config.mv_full_attn[i] = std.mem.eql(u8, item.string, "full_attention");
                    };
                }
                if (config.qv_heads != 0) config.qv_head_dim = config.qv_hidden / config.qv_heads;
                config.qv_out_hidden = config.hidden_size;
                if (jsonField(root, "projector_hidden_size")) |v| {
                    if (v == .integer) config.mv_projector_hidden = try jsonU32(v);
                }
                // The processor wraps the pad run in <|image_start|>/<|image_end|>;
                // config.json carries neither, so the ids come from the vocab.
                config.boi_token_id = 200080;
                config.eoi_token_id = 200081;
            }
        }
        config.ensureMuseTerminators();
    } else if (std.mem.eql(u8, model_type, "spark2_5")) {
        // XHToken Spark-X2.5 (1.7B / 4B): dense GQA, hd 256, 3:1 sliding(512)/
        // full layers with per-type RoPE (sliding: full rotary at 1e4; full:
        // 25% rotary at 5e6), exact-erf GELU gated MLP, plain RMS norms,
        // per-head sigmoid attention output gate, fused q_k_v_proj, tied head.
        config.model_type = "spark2_5";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.hidden_act = .gelu;
        config.attn_sigmoid_gate = true;
        config.attn_gate_headwise = true;
        config.attn_fused_qkv = true;
        config.rope_scaling_factor = 1.0;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "qwen3_5_moe") or
        std.mem.eql(u8, model_type, "qwen3_5") or
        std.mem.eql(u8, model_type, "qwen3_5_moe_text") or
        std.mem.eql(u8, model_type, "qwen3_5_text") or
        std.mem.eql(u8, model_type, "prism_hadamard_qwen35"))
    {
        config.model_type = "qwen3_5_moe";
        if (std.mem.eql(u8, model_type, "prism_hadamard_qwen35")) config.hadamard_block = try prismHadamardBlock(root);
        config.weight_prefix = "language_model.model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        try parseQwenVisionFields(&config, root, cfg_obj);
    } else if (std.mem.eql(u8, model_type, "qwen4_exp") or
        std.mem.eql(u8, model_type, "qwen4_exp_text"))
    {
        config.model_type = "qwen4_exp";
        config.weight_prefix = "language_model.model";
        config.norm_has_offset = false;
        if (root.get("qwen4_norm_convention")) |v| {
            if (v != .string) return error.InvalidQwen4NormConvention;
            config.qwen4_norm_convention = std.meta.stringToEnum(Qwen4NormConvention, v.string) orelse return error.InvalidQwen4NormConvention;
        }
        config.has_final_norm = false; // hyper_connection_mixer replaces model.norm
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.kda_sigmoid_out_gate = true; // output_gate_type "sigmoid"
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        config.hc_count = 4;
        config.hc_lowrank = 320;
        config.ple_embed_dim = config.hidden_size;
        try parseQwenVisionFields(&config, root, cfg_obj);
        // YaRN (262144 → e.g. 1048576) rides ONE rotary table for the whole
        // trunk: attention, the QSA indexer and the MTP head all read it, so
        // a scaled rotation cannot desync the block selector from attention.
        try parseYarnRopeParameters(&config, cfg_obj);
        // Read strictly; range-checked in `validateQwen4Config` once every field is in.
        if (try qwen4ConfigU32(cfg_obj, "hc_count")) |v| config.hc_count = v;
        if (try qwen4ConfigU32(cfg_obj, "hc_lowrank")) |v| config.hc_lowrank = v;
        {
            const v = jsonField(cfg_obj, "ple_layer_ids") orelse return error.InvalidQwen4PleLayer;
            if (v != .array or v.array.items.len != 1 or v.array.items[0] != .integer) {
                return error.InvalidQwen4PleLayer;
            }
            const id = v.array.items[0].integer;
            if (id < 1 or id > @as(i64, config.num_hidden_layers)) return error.InvalidQwen4PleLayer;
            config.ple_layer_idx = std.math.cast(i32, id - 1) orelse return error.InvalidQwen4PleLayer;
        }
        if (try qwen4ConfigU32(cfg_obj, "ple_embed_dim")) |v| config.ple_embed_dim = v;
        if (try qwen4ConfigU32(cfg_obj, "ple_conv_kernel_size")) |v| config.ple_conv_kernel = v;
        if (try qwen4ConfigU32(cfg_obj, "ngram_size")) |v| config.ngram_size = v;
        if (try qwen4ConfigU32(cfg_obj, "heads_per_ngram")) |v| config.heads_per_ngram = v;
        if (try qwen4ConfigU64(cfg_obj, "ngram_vocab_size_base")) |v| config.ngram_vocab_base = v;
        if (try qwen4ConfigU32(cfg_obj, "make_ngram_vocab_size_divisible_by")) |v| config.ngram_vocab_divisor = v;
        if (try qwen4ConfigU64(cfg_obj, "seed")) |v| config.ngram_seed = v;
        if (try qwen4ConfigU32(cfg_obj, "split_ngram_parts")) |v| config.split_ngram_parts = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_n_heads")) |v| config.indexer_n_heads = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_head_dim")) |v| config.indexer_head_dim = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_budget")) |v| config.indexer_budget = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_compress_ratio")) |v| config.indexer_compress_ratio = v;
        if (jsonField(cfg_obj, "eos_token_id")) |v| {
            switch (v) {
                .integer => |i| config.ngram_eos = std.math.cast(u32, i) orelse return error.InvalidConfigField,
                .array => |arr| if (arr.items.len > 0 and arr.items[0] == .integer) {
                    config.ngram_eos = try jsonU32(arr.items[0]);
                },
                else => {},
            }
            if (config.num_eos_tokens == 0) config.addEosToken(config.ngram_eos);
        }
        try validateQwen4Config(&config);
        if (jsonField(root, "expert_quant") != null) {
            config.exl3 = try sushi_exl3.parseExpertQuant(root);
            try sushi_exl3.admitTopK(config.num_experts_per_tok);
        }
    } else if (std.mem.eql(u8, model_type, "qwen3_moe") or
        std.mem.eql(u8, model_type, "qwen3_moe_text"))
    {
        // Qwen3-30B-A3B / Qwen3-Coder-30B-A3B. Shares qwen3_5_moe's weight
        // layout (`mlp.gate` router + stacked `mlp.switch_mlp.*` experts) and
        // its MoE forward, but differs in three ways that make it its OWN
        // model_type rather than a remap onto qwen3_5_moe:
        //   1. No GatedDeltaNet — every layer is full attention
        //      (full_attention_interval stays 0 ⇒ isLinearLayer == false).
        //   2. No attention output gate (attn_output_gate stays false; the
        //      qwen3_5 split-Q path would mis-shape the projection here).
        //   3. No shared expert (shared_expert_intermediate_size: 0, no
        //      mlp.shared_expert.* weights). The MoE binding in
        //      transformer.zig loads those optionally and the forward skips
        //      the shared branch when shared_expert_gate_w is null.
        // weight_prefix is plain "model" (no language_model nesting).
        config.model_type = "qwen3_moe";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "gpt_oss")) {
        // OpenAI gpt-oss (20B-A3.6B / 120B-A5.1B). A plain dense-attention MoE
        // that rides the qwen3_moe forward arms — no linear/SSM layers, no
        // module-owned decode state, so prefix cache, batched decode and spec
        // decode all apply normally. Family-specific pieces:
        //   1. Learned per-head attention SINKS (self_attn.sinks) — an extra
        //      softmax-denominator column. mlx's fused SDPA takes them.
        //   2. Clamped SwiGLU (swiglu_limit, alpha 1.702, +1 on the linear
        //      branch) instead of silu(gate)*up.
        //   3. Additive biases everywhere: q/k/v/o_proj.bias, mlp.router.bias
        //      and per-expert gate/up/down bias — all living BESIDE the affine
        //      quantizer's `.biases` tensors in the same checkpoint.
        //   4. No QK norm.
        // Router is softmax-over-top-k, which is algebraically identical to
        // the existing softmax-all → top-k → renorm chain, so moe_route_norm
        // stays true and moe_sigmoid_router stays false.
        config.model_type = "gpt_oss";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.hidden_act = .silu;
        config.has_attn_sinks = true;
        // ONE theta for both layer types. The Gemma-flavored default of 10000
        // would silently mis-base every sliding layer — the muse
        // first-turn-repetition class (deterministic "coherent then loops").
        config.rope_local_base_freq = config.rope_theta;
        config.rope_scaling_factor = 1.0;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        // Expert count rides `num_local_experts`; the generic block only knows
        // `num_experts`. Top-k has two spellings and both appear in shipped
        // configs (`num_experts_per_tok` is generic, `experts_per_token` is not).
        if (jsonField(cfg_obj, "num_local_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "experts_per_token")) |v| {
            if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
        }
        // There is no moe_intermediate_size key: the expert width IS
        // intermediate_size (2880 on both sizes).
        if (config.moe_intermediate_size == 0) {
            config.moe_intermediate_size = config.intermediate_size;
        }
        if (jsonField(cfg_obj, "swiglu_limit")) |v| config.swiglu_limit = try jsonFloat(v);
        // Flat YaRN block. mscale is COMPUTED, never read: the config ships no
        // "attention_factor" at all, and mlx-lm's YarnRoPE defaults
        // (mscale 1 / mscale_all_dim 0) give 0.1*ln(factor) + 1. Laguna
        // precedent — but note dsv4 is the same shape with the OPPOSITE
        // answer, so this stays a per-arch decision.
        if (jsonField(cfg_obj, "rope_scaling")) |rs| {
            if (rs == .object) {
                const is_yarn = if (jsonField(rs.object, "rope_type")) |rt|
                    (rt == .string and std.mem.eql(u8, rt.string, "yarn"))
                else
                    false;
                if (is_yarn) {
                    config.rope_yarn = true;
                    if (jsonField(rs.object, "factor")) |x| config.yarn_factor = try jsonFloat(x);
                    if (jsonField(rs.object, "beta_fast")) |x| config.yarn_beta_fast = try jsonFloat(x);
                    if (jsonField(rs.object, "beta_slow")) |x| config.yarn_beta_slow = try jsonFloat(x);
                    if (jsonField(rs.object, "original_max_position_embeddings")) |x| {
                        if (x == .integer) config.yarn_orig_max_pos = try jsonU32(x);
                    }
                    if (config.yarn_factor > 1.0) {
                        config.yarn_attention_factor = 0.1 * @log(config.yarn_factor) + 1.0;
                    }
                }
            }
        }
        config.ensureGptOssTerminators();
    } else if (std.mem.eql(u8, model_type, "glm5_next") or std.mem.eql(u8, model_type, "glm5_next_text")) {
        // Z.ai GLM-5.3-Flash (Glm5NextForConditionalGeneration) in mlx-vlm's glm5_next layout
        // (TensorFold's MLX packs). Three Kimi-Delta-Attention layers per DeepSeek sparse
        // attention layer, every sublayer inside DeepSeek-V4's 4-stream Sinkhorn
        // hyper-connection. The sparse attention is absorbed NoPE MLA over a 512 latent; a
        // pooled indexer picks the tokens each query reads. Sigmoid noaux_tc MoE with one
        // ungated shared expert and a clamped SwiGLU; dense MLP on the leading layers.
        config.model_type = "glm5_next";
        config.weight_prefix = if (jsonField(root, "text_config") != null) "language_model.model" else "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        config.moe_sigmoid_router = true;
        if (jsonField(cfg_obj, "n_routed_experts")) |v| config.num_experts = try jsonU32(v);
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "n_group")) |v| config.moe_n_group = try jsonU32(v);
        if (jsonField(cfg_obj, "topk_group")) |v| config.moe_topk_group = try jsonU32(v);
        if (jsonField(cfg_obj, "n_shared_experts")) |v| {
            if (v == .integer) config.shared_expert_intermediate_size =
                std.math.mul(u32, try jsonU32(v), config.moe_intermediate_size) catch return error.InvalidConfigField;
        }
        if (jsonField(cfg_obj, "swiglu_limit")) |v| {
            if (v != .null) config.swiglu_limit = try jsonFloat(v);
        }
        config.swiglu_clamp = true;

        // Sparse attention: NoPE, q/k and v at one width.
        if (jsonField(cfg_obj, "qk_rope_head_dim")) |v| {
            if (v == .integer and v.integer != 0) {
                log.err("glm5_next: qk_rope_head_dim {d} (only NoPE sparse attention is served)\n", .{v.integer});
                return error.UnsupportedGlm5Config;
            }
        }
        if (jsonField(cfg_obj, "q_lora_rank")) |v| config.dsa_q_lora_rank = try jsonU32(v);
        if (jsonField(cfg_obj, "kv_lora_rank")) |v| config.dsa_kv_lora_rank = try jsonU32(v);
        if (jsonField(cfg_obj, "qk_nope_head_dim")) |v| config.dsa_head_dim = try jsonU32(v);
        if (jsonField(cfg_obj, "v_head_dim")) |v| {
            if ((try jsonU32(v)) != config.dsa_head_dim) {
                log.err("glm5_next: v_head_dim differs from qk_nope_head_dim (not supported)\n", .{});
                return error.UnsupportedGlm5Config;
            }
        }
        config.head_dim = config.dsa_head_dim;
        if (config.num_key_value_heads != config.num_attention_heads) config.num_key_value_heads = config.num_attention_heads;
        if (jsonField(cfg_obj, "index_topk")) |v| config.dsa_index_topk = try jsonU32(v);
        if (jsonField(cfg_obj, "index_kpool")) |v| config.dsa_index_kpool = try jsonU32(v);
        if (jsonField(cfg_obj, "index_n_heads")) |v| config.dsa_index_heads = try jsonU32(v);
        if (jsonField(cfg_obj, "index_head_dim")) |v| config.dsa_index_head_dim = try jsonU32(v);
        inline for (.{ "index_kpool_compress", "index_kpool_always_select_tail", "mla_use_nope" }) |key| {
            if (jsonField(cfg_obj, key)) |v| {
                if (v == .bool and !v.bool) {
                    log.err("glm5_next: {s}=false (not supported)\n", .{key});
                    return error.UnsupportedGlm5Config;
                }
            }
        }
        if (config.dsa_index_kpool == 0 or config.dsa_index_topk % config.dsa_index_kpool != 0) {
            log.err("glm5_next: index_topk must be a multiple of index_kpool\n", .{});
            return error.UnsupportedGlm5Config;
        }
        // Every indexer scores its own layer (`shared` reuses the previous one's top-k).
        if (jsonField(cfg_obj, "indexer_types")) |v| {
            for ((try jsonValue(.array, v)).items) |t| {
                if (!std.mem.eql(u8, try jsonValue(.string, t), "full")) {
                    log.err("glm5_next: a shared indexer layer (not supported)\n", .{});
                    return error.UnsupportedGlm5Config;
                }
            }
        }

        // Layer pattern: sparse attention every 4th layer, the rest KDA.
        config.full_attention_interval = 4;
        if (jsonField(cfg_obj, "layer_types")) |v| {
            for ((try jsonValue(.array, v)).items, 0..) |t, i| {
                const name = try jsonValue(.string, t);
                const sparse = std.mem.eql(u8, name, "deepseek_sparse_attention") or std.mem.eql(u8, name, "full_attention");
                if (sparse != !config.isLinearLayer(@intCast(i))) {
                    log.err("glm5_next: layer {d} is {s}, off the every-4th pattern (not supported)\n", .{ i, name });
                    return error.UnsupportedGlm5Config;
                }
            }
        }
        if (jsonField(cfg_obj, "mlp_layer_types")) |v| {
            config.first_k_dense_replace = 0;
            for ((try jsonValue(.array, v)).items, 0..) |t, i| {
                if (std.mem.eql(u8, try jsonValue(.string, t), "dense")) {
                    if (config.first_k_dense_replace != i) {
                        log.err("glm5_next: dense layer {d} after a MoE layer (not supported)\n", .{i});
                        return error.UnsupportedGlm5Config;
                    }
                    config.first_k_dense_replace += 1;
                }
            }
        } else if (jsonField(cfg_obj, "first_k_dense_replace")) |v| config.first_k_dense_replace = try jsonU32(v);

        // KDA: per-channel bounded gate, sigmoid output gate, low-rank gate projections.
        config.linear_num_key_heads = config.num_attention_heads;
        config.linear_key_head_dim = 128;
        if (jsonField(cfg_obj, "linear_attn_config")) |lac_v| {
            const lac = try jsonValue(.object, lac_v);
            if (jsonField(lac, "num_heads")) |v| config.linear_num_key_heads = try jsonU32(v);
            if (jsonField(lac, "head_dim")) |v| config.linear_key_head_dim = try jsonU32(v);
            if (jsonField(lac, "short_conv_kernel_size")) |v| config.linear_conv_kernel_dim = try jsonU32(v);
            if (jsonField(lac, "gate_lower_bound")) |v| {
                if (v != .null) config.kda_gate_lower_bound = try jsonFloat(v);
            }
        }
        if (config.kda_gate_lower_bound == 0.0) config.kda_gate_lower_bound = -5.0;
        config.linear_num_value_heads = config.linear_num_key_heads;
        config.linear_value_head_dim = config.linear_key_head_dim;
        config.kda_vector_gate = true;
        config.kda_sigmoid_out_gate = true;

        // DeepSeek-V4's Sinkhorn hyper-connection around every sublayer.
        config.dsv4_hc_mult = 4;
        if (jsonField(cfg_obj, "hc_mult")) |v| config.dsv4_hc_mult = try jsonU32(v);
        if (config.dsv4_hc_mult != 4) {
            log.err("glm5_next: hc_mult {d} (only 4 is served)\n", .{config.dsv4_hc_mult});
            return error.UnsupportedGlm5Config;
        }
        config.dsv4_hc_sinkhorn_iters = 20;
        if (jsonField(cfg_obj, "hc_sinkhorn_iters")) |v| config.dsv4_hc_sinkhorn_iters = try jsonU32(v);
        if (jsonField(cfg_obj, "hc_eps")) |v| config.dsv4_hc_eps = try jsonFloat(v);
        // The GLM-5-Next vision tower is not wired.
        config.has_vision = false;
        try parseSushiExperts(root, &config);
    } else if (std.mem.eql(u8, model_type, "mimo_v2") or std.mem.eql(u8, model_type, "mimo_v2_flash")) {
        // Xiaomi MiMo-V2.6-Flash (309B-A15B) in the mlx-lm layout (tests/convert_mimo_v2.py,
        // the community packs); `mimo_v2_flash` is MiMo-V2-Flash and TensorFold's V2.6 label.
        // gpt_oss-shaped attention (learned sinks, 128-token sliding window) with
        // per-layer-type geometry: global layers 4 KV heads at theta 1e7, sliding
        // layers 8 KV heads at theta 1e4 with sinks; K 192 / V 128, RoPE on the
        // first int(192 * 0.334) = 64 channels. Experts: laguna's sigmoid routing
        // with a selection-only bias and no shared expert; dense layers lead.
        // A Sushi pack's `quant_method: fp8` describes its trunk; its experts are EXL3.
        if (jsonField(root, "expert_quant") == null) try refuseUnconvertedMimo(cfg_obj);
        config.model_type = "mimo_v2";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.hidden_act = .silu;
        config.has_attn_sinks = true;
        config.moe_sigmoid_router = true;
        if (jsonField(cfg_obj, "layernorm_epsilon")) |v| config.rms_norm_eps = try jsonFloat(v);
        if (jsonField(cfg_obj, "attention_value_scale")) |v| {
            if (v != .null) config.attention_value_scale = try jsonFloat(v);
        }
        if (jsonField(cfg_obj, "n_routed_experts")) |v| config.num_experts = try jsonU32(v);
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| {
            if (v != .null) config.router_scaling_factor = try jsonFloat(v);
        }
        if (jsonField(cfg_obj, "n_group")) |v| config.moe_n_group = try jsonU32(v);
        if (jsonField(cfg_obj, "topk_group")) |v| config.moe_topk_group = try jsonU32(v);
        if (jsonField(cfg_obj, "n_shared_experts")) |v| {
            if (v == .integer and v.integer != 0) {
                log.err("mimo_v2: n_shared_experts {d} not supported\n", .{v.integer});
                return error.UnsupportedMimoConfig;
            }
        }

        // Pattern 0 = global, 1 = sliding; the dense MLP layers lead the stack.
        const pattern = try jsonValue(.array, jsonField(cfg_obj, "hybrid_layer_pattern") orelse return error.UnsupportedMimoConfig);
        const freq = try jsonValue(.array, jsonField(cfg_obj, "moe_layer_freq") orelse return error.UnsupportedMimoConfig);
        if (pattern.items.len != config.num_hidden_layers or freq.items.len != config.num_hidden_layers or config.num_hidden_layers > 128) {
            log.err("mimo_v2: hybrid_layer_pattern/moe_layer_freq must list all {d} layers\n", .{config.num_hidden_layers});
            return error.UnsupportedMimoConfig;
        }
        config.has_explicit_layer_types = true;
        for (pattern.items, freq.items, 0..) |p, f, i| {
            config.layer_is_global[i] = (try jsonU32(p)) == 0;
            const moe = (try jsonU32(f)) != 0;
            if (!moe and config.first_k_dense_replace != i) {
                log.err("mimo_v2: dense layer {d} after a MoE layer is not supported\n", .{i});
                return error.UnsupportedMimoConfig;
            }
            if (!moe) config.first_k_dense_replace += 1;
        }

        // Root KV heads / theta are the GLOBAL layers'; `swa_*` the sliding ones.
        config.num_global_key_value_heads = config.num_key_value_heads;
        if (jsonField(cfg_obj, "swa_num_key_value_heads")) |v| config.num_key_value_heads = try jsonU32(v);
        if (jsonField(cfg_obj, "swa_rope_theta")) |v| config.rope_local_base_freq = try jsonFloat(v);
        if (jsonField(cfg_obj, "v_head_dim")) |v| config.v_head_dim = try jsonU32(v);
        const same = .{ .{ "swa_head_dim", config.head_dim }, .{ "swa_v_head_dim", config.v_head_dim }, .{ "swa_num_attention_heads", config.num_attention_heads } };
        inline for (same) |kv| {
            if (jsonField(cfg_obj, kv[0])) |v| {
                if ((try jsonU32(v)) != kv[1]) {
                    log.err("mimo_v2: {s} differs from the global layers' (not supported)\n", .{kv[0]});
                    return error.UnsupportedMimoConfig;
                }
            }
        }
        config.partial_rotary_factor_global = config.partial_rotary_factor;
        config.query_pre_attn_scalar = config.head_dim;
        config.rope_scaling_factor = 1.0;
        // The MiMo-ViT tower is not wired: the generic vision_config block must not arm SigLIP.
        config.has_vision = false;
        try parseSushiExperts(root, &config);
    } else if (std.mem.eql(u8, model_type, "hy_v3")) {
        // Tencent Hunyuan 3 (Hy3, 295B-A21B MoE; July 2026). Pure
        // full-attention MoE that rides the qwen3_moe forward arms: GQA with
        // per-head QK RMS-norm, full rotary, scale = head_dim^-0.5, no output
        // gate, plain "model" prefix. Family-specific pieces handled
        // explicitly: DeepSeek-V3-style SIGMOID router with expert bias
        // (mlp.expert_bias, f32) + top-k renorm + router_scaling_factor, an
        // UNGATED always-added shared expert (mlp.shared_mlp.*), and
        // first_k_dense_replace dense bottom layers. Reference: mlx-lm PR
        // #1211 (converted repos ship hy_v3.py alongside the weights).
        // NOTE: tencent's generation_config documents temp 0.9 / top_k -1 /
        // top_p 1 — untruncated by design, so applyFamilySamplingDefaults
        // deliberately has no hy_v3 fill.
        config.model_type = "hy_v3";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        config.moe_sigmoid_router = true;
        // qk_norm / route_norm default TRUE when absent (mlx-lm ModelArgs
        // defaults) but an explicit false must win.
        if (jsonField(cfg_obj, "qk_norm")) |v| {
            if (v == .bool) config.has_qk_norm = v.bool;
        }
        if (jsonField(cfg_obj, "route_norm")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "router_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "first_k_dense_replace")) |v| {
            if (v == .integer) config.first_k_dense_replace = try jsonU32(v);
        }
        // Expert width may ride as expert_hidden_dim when moe_intermediate_size
        // is absent (both = 1536 on the 295B).
        if (config.moe_intermediate_size == 0) {
            if (jsonField(cfg_obj, "expert_hidden_dim")) |v| {
                if (v == .integer) config.moe_intermediate_size = try jsonU32(v);
            }
        }
        // No explicit shared_expert_intermediate_size key in hy_v3 configs:
        // derive num_shared_experts × expert width. 0 shared experts leaves it
        // 0 and the binder/forward skip the shared branch.
        if (jsonField(cfg_obj, "num_shared_experts")) |v| {
            if (v == .integer) config.shared_expert_intermediate_size =
                std.math.mul(u32, try jsonU32(v), config.moe_intermediate_size) catch return error.InvalidConfigField;
        }
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        config.ensureHy3Terminators();
    } else if (std.mem.eql(u8, model_type, "bailing_hybrid")) {
        // inclusionAI Ling 3.0 (bailing_hybrid; BailingMoeV3ForCausalLM). A
        // KDA + MLA hybrid MoE: three Kimi-Delta-Attention linear layers for
        // every Multi-head-Latent-Attention layer (layer_group_size 4, so
        // layers 3/7/11/… are full attention), DeepSeek-V3-style grouped
        // sigmoid routing with an expert bias, one ungated shared expert, and
        // a dense MLP on the bottom first_k_dense_replace layers.
        //
        // Three things here are NOT the qwen3.5 GDN defaults and each has its
        // own config field: the forget gate is per CHANNEL (kda_vector_gate),
        // it uses the bounded sigmoid form rather than softplus
        // (kda_gate_lower_bound), and the output gate is a plain sigmoid.
        // RoPE covers only the qk_rope_head_dim slice of each query/key head
        // and rotates ADJACENT PAIRS (rope_interleave).
        //
        // References: the checkpoint's own modeling_bailing_moe_v3.py, and
        // fla/ops/kda/fused_recurrent.py for the gate. Mirror:
        // rapid-mlx/Ling-3.0-tiny-MLX-4bit.
        config.model_type = "bailing_hybrid";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false; // MLA norms the LATENTS, not the heads
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;

        // Hybrid layout. `layer_group_size` counts layers per group with the
        // LAST one full — exactly isLinearLayer's `(idx+1) % interval != 0`.
        if (jsonField(cfg_obj, "layer_group_size")) |v| {
            if (v == .integer) config.full_attention_interval = try jsonU32(v);
        }

        // KDA: one linear head per attention head, key dim = value dim = head_dim.
        config.linear_num_key_heads = config.num_attention_heads;
        config.linear_num_value_heads = config.num_attention_heads;
        config.linear_key_head_dim = config.head_dim;
        config.linear_value_head_dim = config.head_dim;
        if (jsonField(cfg_obj, "short_conv_kernel_size")) |v| {
            if (v == .integer) config.linear_conv_kernel_dim = try jsonU32(v);
        }
        // A per-head KDA (`num_kv_heads_for_linear_attn`) would give the linear
        // layers their own head count instead of the attention heads' — the
        // three lines above would then be wrong, so refuse rather than size the
        // recurrent state off the wrong geometry.
        if (jsonField(cfg_obj, "num_kv_heads_for_linear_attn")) |v| {
            if (v == .integer and v.integer != 0 and v.integer != @as(i64, config.num_attention_heads)) {
                log.err("bailing_hybrid: num_kv_heads_for_linear_attn {d} != num_attention_heads {d} (per-head KDA not supported)\n", .{ v.integer, config.num_attention_heads });
                return error.UnsupportedBailingConfig;
            }
        }
        config.kda_vector_gate = true;
        config.kda_sigmoid_out_gate = true;
        // fla's kernel has TWO gate arms and this key selects between them: a
        // negative bound takes `exp(bound·σ(exp(A_log)·(a+dt_bias)))`, an absent
        // key the plain `exp(-exp(A_log)·softplus(·))` (which the shared
        // GatedDeltaNet chain already serves, elementwise, so a per-channel gate
        // needs nothing new). A bound of exactly 0 would degenerate the bounded
        // form to exp(0) = 1 — a gate that never forgets — so it is refused
        // rather than served as the other arm by accident.
        if (jsonField(cfg_obj, "kda_lower_bound")) |v| {
            if (v != .null) {
                const lb = try jsonFloat(v);
                if (lb >= 0.0) {
                    log.err("bailing_hybrid: kda_lower_bound must be negative (got {d})\n", .{lb});
                    return error.UnsupportedBailingConfig;
                }
                config.kda_gate_lower_bound = lb;
            }
        }
        // `kda_safe_gate` is a numerics detail of the reference's own kernel
        // launch, not a change of formula — both arms above are already
        // evaluated in f32 here, so it is read as satisfied and ignored.
        //
        // The KDA LoRA gate variants (f_a_proj/f_b_proj, g_a_proj/g_b_proj)
        // are a different weight layout; the shipped checkpoint sets
        // no_kda_lora. Refuse rather than fail with a MISSING WEIGHT crash —
        // and accept BOTH spellings, since a checkpoint that states only the
        // positive one otherwise slips straight through to that crash.
        if (jsonField(cfg_obj, "no_kda_lora")) |v| {
            if (v == .bool and !v.bool) {
                log.err("bailing_hybrid: low-rank KDA gates (no_kda_lora=false) not supported\n", .{});
                return error.UnsupportedBailingConfig;
            }
        }
        if (jsonField(cfg_obj, "use_kda_lora")) |v| {
            if (v == .bool and v.bool) {
                log.err("bailing_hybrid: low-rank KDA gates (use_kda_lora=true) not supported\n", .{});
                return error.UnsupportedBailingConfig;
            }
        }

        // MLA.
        if (jsonField(cfg_obj, "q_lora_rank")) |v| {
            if (v == .integer) config.mla_q_lora_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "kv_lora_rank")) |v| {
            if (v == .integer) config.mla_kv_lora_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "qk_nope_head_dim")) |v| {
            if (v == .integer) config.mla_qk_nope_head_dim = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "qk_rope_head_dim")) |v| {
            if (v == .integer) config.mla_qk_rope_head_dim = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "v_head_dim")) |v| {
            if (v == .integer) config.mla_v_head_dim = try jsonU32(v);
        }
        if (config.mla_v_head_dim == 0) config.mla_v_head_dim = config.head_dim;
        // `q_lora_rank: null` is a plain q_proj — a different weight layout,
        // now served (see `mlaHasQLora`). The KV latent has no such fallback.
        if (config.mla_kv_lora_rank == 0) {
            log.err("bailing_hybrid: kv_lora_rank is required\n", .{});
            return error.UnsupportedBailingConfig;
        }
        // The declared qk_head_dim must agree with nope+rope: everything
        // downstream (the cached K's last dim, the attention scale, the q_b
        // split) is derived from the two halves.
        const qk_head_dim = std.math.add(u32, config.mla_qk_nope_head_dim, config.mla_qk_rope_head_dim) catch return error.InvalidConfigField;
        if (jsonField(cfg_obj, "qk_head_dim")) |v| {
            if (v == .integer and try jsonU32(v) != qk_head_dim) {
                log.err("bailing_hybrid: qk_head_dim {d} != qk_nope_head_dim + qk_rope_head_dim ({d})\n", .{ v.integer, qk_head_dim });
                return error.UnsupportedBailingConfig;
            }
        }
        // Attention scale is 1/sqrt(qk_head_dim) — the FULL query width,
        // wider than head_dim. Not derivable from head_dim on this arch.
        config.query_pre_attn_scalar = qk_head_dim;
        if (jsonField(cfg_obj, "gated_attention_proj_granularity_type")) |v| {
            if (v == .string) {
                if (std.mem.eql(u8, v.string, "head_wise")) {
                    config.mla_head_gate = true;
                } else {
                    // element_wise gating is a differently-shaped g_proj.
                    log.err("bailing_hybrid: only head_wise attention gating supported (got '{s}')\n", .{v.string});
                    return error.UnsupportedBailingConfig;
                }
            }
        }
        if (jsonField(cfg_obj, "rope_interleave")) |v| {
            if (v == .bool) config.rope_interleaved_pairs = v.bool;
        }
        // `use_mla_nope` makes the MLA layers positionless (Kimi-Linear ships
        // exactly that: `rotary_emb=None`). `mlaAttnWith` always ropes the rope
        // slice, so a NoPE checkpoint would be served with positions its
        // reference never applies — refuse by name instead.
        if (jsonField(cfg_obj, "use_mla_nope")) |v| {
            if (v == .bool and v.bool) {
                log.err("bailing_hybrid: NoPE MLA (use_mla_nope=true) not supported\n", .{});
                return error.UnsupportedBailingConfig;
            }
        }
        // partial_rotary_factor is stated against head_dim but the reference's
        // rotary module overrides it to 1.0 over qk_rope_head_dim — rope covers
        // that slice ENTIRELY, so the generic partial factor must not leak in.
        config.partial_rotary_factor = 1.0;

        // Three optional norms the forward does NOT implement. Each is a real
        // BailingMoeV3 switch and each is FALSE in every shipped checkpoint, so
        // they cost nothing here — but a variant flipping one would be served
        // silently without it, which is the failure mode this whole block of
        // named refusals exists to prevent.
        const unsupported_flags = [_][]const u8{ "value_norm", "up_proj_norm", "use_nGPT" };
        for (unsupported_flags) |key| {
            if (jsonField(cfg_obj, key)) |v| {
                if (v == .bool and v.bool) {
                    log.err("bailing_hybrid: {s}=true not supported\n", .{key});
                    return error.UnsupportedBailingConfig;
                }
            }
        }
        // The KDA conv activation. True everywhere shipped; the shared
        // GatedDeltaNet path applies silu after the causal conv unconditionally,
        // so a checkpoint declaring otherwise would get an activation it never
        // trained with.
        if (jsonField(cfg_obj, "linear_silu")) |v| {
            if (v == .bool and !v.bool) {
                log.err("bailing_hybrid: linear_silu=false not supported (the conv activation is silu)\n", .{});
                return error.UnsupportedBailingConfig;
            }
        }

        // MoE: grouped sigmoid routing (noaux_tc) + one ungated shared expert.
        config.moe_sigmoid_router = true;
        if (jsonField(cfg_obj, "first_k_dense_replace")) |v| {
            if (v == .integer) config.first_k_dense_replace = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "n_group")) |v| {
            if (v == .integer) config.moe_n_group = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "topk_group")) |v| {
            if (v == .integer) config.moe_topk_group = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        // Shared expert width = num_shared_experts × its own intermediate size
        // (which falls back to the routed expert width when absent).
        if (config.shared_expert_intermediate_size == 0) {
            var shared_width: u32 = config.moe_intermediate_size;
            if (jsonField(cfg_obj, "moe_shared_expert_intermediate_size")) |v| {
                if (v == .integer) shared_width = try jsonU32(v);
            }
            var n_shared: u32 = 0;
            if (jsonField(cfg_obj, "num_shared_experts")) |v| {
                if (v == .integer) n_shared = try jsonU32(v);
            }
            config.shared_expert_intermediate_size = std.math.mul(u32, shared_width, n_shared) catch return error.InvalidConfigField;
        }
        // Softmax routing is a different score function; only sigmoid ships.
        if (jsonField(cfg_obj, "score_function")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "sigmoid")) {
                log.err("bailing_hybrid: only sigmoid score_function supported (got '{s}')\n", .{v.string});
                return error.UnsupportedBailingConfig;
            }
        }
        // The MTP head ships disabled (num_nextn_predict_layers 0) and no
        // mtp.* weights are in the checkpoint. The single terminator
        // (`<|role_end|>`) rides the root eos_token_id — no additive merge.
    } else if (std.mem.eql(u8, model_type, "laguna")) {
        // poolside Laguna S 2.1 (117.6B-A8.5B MoE coder; nvfp4 experts, 256K
        // ctx). Pure-attention MoE that rides the qwen3.5/hy_v3 MoE forward
        // arms. Family-specific pieces: (1) per-layer Q-head counts
        // (num_attention_heads_per_layer: 48 full / 72 sliding; KV uniform 8),
        // (2) a softplus per-head attention OUTPUT gate (self_attn.g_proj), and
        // (3) YaRN RoPE on full-attention layers (theta 5e5, factor 32) with
        // default RoPE (theta 1e4, full rotary) on the 512-window sliding
        // layers. MoE routing is DeepSeek-V3 SIGMOID + expert bias
        // (mlp.gate.e_score_correction_bias) exactly like hy_v3, but the weight
        // NAMING matches qwen3_moe (mlp.gate router, mlp.switch_mlp experts,
        // mlp.shared_expert UNGATED always-added). Dense MLP on mlp_only_layers
        // (layer 0 on the shipped checkpoint), resolved by per-layer weight
        // presence probe at load. Reference: modeling_laguna.py (Apache-2.0).
        config.model_type = "laguna";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.laguna_attn_gate = true;
        config.moe_sigmoid_router = true;
        config.rope_scaling_factor = 1.0;
        // scale = head_dim^-0.5 (query_pre_attn_scalar absent in Laguna config).
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        // Router: norm_topk_prob → route_norm; moe_routed_scaling_factor → scale.
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "moe_routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "moe_router_logit_softcapping")) |v| config.moe_router_logit_softcapping = try jsonFloat(v);
        // Router logit soft-capping is off on the shipped checkpoint and untested;
        // reject a >0 value rather than silently ignore it (honest reject).
        if (config.moe_router_logit_softcapping > 0.0) {
            log.err("laguna: moe_router_logit_softcapping > 0 not supported in v1 (got {d})\n", .{config.moe_router_logit_softcapping});
            return error.UnsupportedLagunaConfig;
        }
        // Only per-head gating is implemented (assert uniform; honest reject).
        if (jsonField(cfg_obj, "gating")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "per-head") and !std.mem.eql(u8, v.string, "per_head")) {
                log.err("laguna: only per-head attention gating supported (got '{s}')\n", .{v.string});
                return error.UnsupportedLagunaConfig;
            }
        }
        // Per-layer Q-head count (cap 128 like layer_is_global).
        if (jsonField(cfg_obj, "num_attention_heads_per_layer")) |v| {
            if (v == .array) {
                config.has_per_layer_heads = true;
                for (v.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item == .integer) config.num_attention_heads_per_layer[i] = try jsonU32(item);
                }
            }
        }
        // YaRN on full-attention layers. The generic nested-rope block above
        // already set rope_theta (5e5), partial_rotary_factor_global (0.5) from
        // full_attention and rope_local_base_freq (1e4) from sliding_attention;
        // here we pull the YaRN-specific fields and flag the precompute.
        if (jsonField(cfg_obj, "rope_parameters")) |rp| {
            if (rp == .object) {
                if (jsonField(rp.object, "full_attention")) |fa_val| {
                    if (fa_val == .object) {
                        const fa = fa_val.object;
                        const is_yarn = if (jsonField(fa, "rope_type")) |rt|
                            (rt == .string and std.mem.eql(u8, rt.string, "yarn"))
                        else
                            false;
                        if (is_yarn) {
                            config.rope_yarn = true;
                            if (jsonField(fa, "factor")) |x| config.yarn_factor = try jsonFloat(x);
                            if (jsonField(fa, "beta_fast")) |x| config.yarn_beta_fast = try jsonFloat(x);
                            if (jsonField(fa, "beta_slow")) |x| config.yarn_beta_slow = try jsonFloat(x);
                            // mscale is COMPUTED, never read from the config's
                            // "attention_factor". Both vendored MLX Laguna
                            // implementations drop that field and take MLX's
                            // YaRN default (mscale 1 / mscale_all_dim 0 =>
                            // 0.1*ln(factor) + 1); poolside's fused kernel
                            // hardcodes the result. S ships the computed value
                            // literally, XS ships 1.0 — honouring the field
                            // would run XS's full-attention layers unscaled.
                            // Generic YaRN readers still honour it; only this
                            // arch pins the value the checkpoint was trained on.
                            if (config.yarn_factor > 1.0) {
                                config.yarn_attention_factor = 0.1 * @log(config.yarn_factor) + 1.0;
                            }
                            if (jsonField(fa, "original_max_position_embeddings")) |x| {
                                if (x == .integer) config.yarn_orig_max_pos = try jsonU32(x);
                            }
                        }
                    }
                }
            }
        }
        // eos [2, 24] (〈|EOS|〉, </assistant>) parsed generically from
        // eos_token_id above; no additive terminator merge needed.
    } else if (std.mem.eql(u8, model_type, "kolibri1")) {
        // Aleph Alpha Kolibri-1: the laguna attention arm minus the output gate, with NoPE
        // full-attention layers, sandwich norms and an ungated shared expert on every layer.
        // Reference: Aleph-Alpha/aleph-alpha-inference kolibri1.py.
        config.model_type = "kolibri1";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.moe_sigmoid_router = true;
        config.moe_logit_select = true;
        config.moe_route_norm = false;
        config.router_scaling_factor = 1.0;
        config.rope_scaling_factor = 1.0;
        config.query_pre_attn_scalar = config.head_dim;
        // Sliding layers rotate at rope_theta (the config has no local base).
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool and v.bool) {
                log.err("kolibri1: norm_topk_prob=true not supported\n", .{});
                return error.UnsupportedKolibriConfig;
            }
        }
        // Full-attention layers carry no positional encoding.
        if (config.has_explicit_layer_types) {
            for (0..@min(config.num_hidden_layers, 128)) |i| config.layer_no_rope[i] = config.layer_is_global[i];
        }
    } else if (std.mem.eql(u8, model_type, "inkling_mm_model")) {
        // Thinking Machines Inkling Small (276B-A12B MoE, natively multimodal;
        // REAP builds prune n_routed_experts). NO RoPE anywhere: position =
        // RelativeLogits bias + 4 short convs/layer + log-scaling on global
        // layers. Per-head q/k RMSNorm with scale 1/head_dim; hybrid
        // sliding(512)/global from local_layer_ids; dense SwiGLU bottom layers
        // then sigmoid-routed MoE whose selected+shared logits share one
        // logsigmoid-softmax (the shared-expert "sink"); untied quantized
        // embed/unembed with muP logit scaling and a padded vocab. Reference:
        // the checkpoint's bundled inkling_mlx/ (Apache-2.0, parity-validated).
        config.model_type = "inkling_mm_model";
        config.weight_prefix = "model.llm";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.rope_scaling_factor = 1.0;
        // q/k are per-head RMS-normalized → scale = 1/head_dim, expressed via
        // the shared 1/sqrt(query_pre_attn_scalar) convention.
        config.query_pre_attn_scalar = std.math.mul(u32, config.head_dim, config.head_dim) catch return error.InvalidConfigField;
        // The checkpoint labels the MoE expert width `intermediate_size` (read
        // by the generic block above) and the dense bottom-layer width
        // `dense_intermediate_size` — opposite of our field meanings. Swap.
        config.moe_intermediate_size = config.intermediate_size;
        if (jsonField(cfg_obj, "dense_intermediate_size")) |v| {
            if (v == .integer) {
                config.intermediate_size = try jsonU32(v);
                config.intermediate_size_declared = true;
            }
        }
        if (jsonField(cfg_obj, "dense_mlp_idx")) |v| {
            if (v == .integer) config.first_k_dense_replace = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "n_routed_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "n_shared_experts")) |v| {
            if (v == .integer) config.inkling_n_shared_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "route_scale")) |v| config.router_scaling_factor = try jsonFloat(v);
        // Position machinery.
        if (jsonField(cfg_obj, "d_rel")) |v| {
            if (v == .integer) config.inkling_d_rel = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "rel_extent")) |v| {
            if (v == .integer) config.inkling_rel_extent = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "log_scaling_n_floor")) |v| {
            if (v == .integer) config.inkling_log_n_floor = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "log_scaling_alpha")) |v| config.inkling_log_alpha = try jsonFloat(v);
        if (jsonField(cfg_obj, "sconv_kernel_size")) |v| {
            if (v == .integer) config.inkling_sconv_kernel = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "use_sconv")) |v| {
            if (v == .bool and !v.bool) config.inkling_sconv_kernel = 0;
        }
        // Embedding norm (use_embed_norm, default true for this family).
        config.has_embedding_norm = true;
        if (jsonField(cfg_obj, "use_embed_norm")) |v| {
            if (v == .bool) config.has_embedding_norm = v.bool;
        }
        // Hybrid sliding/global: the config names LOCAL (sliding) layers and
        // uses `sliding_window_size` (the generic block reads `sliding_window`).
        if (jsonField(cfg_obj, "sliding_window_size")) |v| {
            if (v == .integer) {
                config.sliding_window = try jsonU32(v);
                config.has_sliding_window = true;
            }
        }
        if (jsonField(cfg_obj, "local_layer_ids")) |v| {
            if (v == .array) {
                config.has_explicit_layer_types = true;
                for (config.layer_is_global[0..@min(config.num_hidden_layers, 128)]) |*g| g.* = true;
                for (v.array.items) |item| {
                    if (item == .integer and item.integer >= 0 and item.integer < 128) {
                        config.layer_is_global[@intCast(item.integer)] = false;
                    }
                }
            }
        }
        // muP logits + padded vocab.
        if (jsonField(cfg_obj, "logits_mup_width_multiplier")) |v| config.logits_mup_width_multiplier = try jsonFloat(v);
        if (jsonField(cfg_obj, "unpadded_vocab_size")) |v| {
            if (v == .integer) config.unpadded_vocab_size = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "model_max_length")) |v| {
            if (v == .integer) config.max_position_embeddings = try jsonU32(v);
        }
        // v1 is text-only: the hMLP vision_config must not arm the SigLIP path
        // (the generic vision_config block above set has_vision = true).
        config.has_vision = false;
        // Honest rejects: the forward implements exactly the shipped geometry
        // and router formula. A checkpoint that diverges must refuse to load,
        // not run silently wrong.
        const swa_heads: u32 = if (jsonField(cfg_obj, "swa_num_attention_heads")) |v| try jsonU32(v) else config.num_attention_heads;
        const swa_kv: u32 = if (jsonField(cfg_obj, "swa_num_key_value_heads")) |v| try jsonU32(v) else config.num_key_value_heads;
        const swa_hd: u32 = if (jsonField(cfg_obj, "swa_head_dim")) |v| try jsonU32(v) else config.head_dim;
        if (swa_heads != config.num_attention_heads or swa_kv != config.num_key_value_heads or swa_hd != config.head_dim) {
            log.err("inkling: sliding-attention geometry {d}/{d}/{d} differs from global {d}/{d}/{d} — not supported\n", .{ swa_heads, swa_kv, swa_hd, config.num_attention_heads, config.num_key_value_heads, config.head_dim });
            return error.UnsupportedInklingConfig;
        }
        if (jsonField(cfg_obj, "gate_activation")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "sigmoid")) {
                log.err("inkling: gate_activation '{s}' not supported (sigmoid only)\n", .{v.string});
                return error.UnsupportedInklingConfig;
            }
        }
    } else if (std.mem.eql(u8, model_type, "deepseek_v4") or std.mem.eql(u8, model_type, "deepseek_v41")) {
        // DeepSeek V4 Flash (284B-A13B, 1M ctx). See the dsv4_* field block
        // for the architecture summary; reference is the release's own
        // inference/{model,kernel}.py (torch). Loaded from OUR converted
        // mixed-quant mirror (tests/convert_dsv4_weights.py) — bare
        // inference-style tensor names, stacked expert banks. V4.1 shares
        // these fields and adds the dsv41_* block below.
        const is_v41 = std.mem.eql(u8, model_type, "deepseek_v41");
        config.model_type = if (is_v41) "deepseek_v41" else "deepseek_v4";
        config.weight_prefix = ""; // release ships bare names (embed.weight, layers.N....)
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false; // q-norm is on the lora rank + unweighted per-head RMS, handled in-arch
        config.hidden_act = .silu;
        if (jsonField(cfg_obj, "n_routed_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "num_hash_layers")) |v| {
            if (v == .integer) config.dsv4_hash_layers = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "q_lora_rank")) |v| {
            if (v == .integer) config.dsv4_q_lora_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "o_lora_rank")) |v| {
            if (v == .integer) config.dsv4_o_lora_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "o_groups")) |v| {
            if (v == .integer) config.dsv4_o_groups = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "qk_rope_head_dim")) |v| {
            if (v == .integer) config.dsv4_rope_head_dim = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "index_n_heads")) |v| {
            if (v == .integer) config.dsv4_index_n_heads = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "index_head_dim")) |v| {
            if (v == .integer) config.dsv4_index_head_dim = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "index_topk")) |v| {
            if (v == .integer) config.dsv4_index_topk = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "hc_mult")) |v| {
            if (v == .integer) config.dsv4_hc_mult = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "hc_sinkhorn_iters")) |v| {
            if (v == .integer) config.dsv4_hc_sinkhorn_iters = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "hc_eps")) |v| config.dsv4_hc_eps = try jsonFloat(v);
        if (jsonField(cfg_obj, "swiglu_limit")) |v| config.dsv4_swiglu_limit = try jsonFloat(v);
        if (jsonField(cfg_obj, "compress_rope_theta")) |v| config.dsv4_compress_rope_theta = try jsonFloat(v);
        if (jsonField(cfg_obj, "num_nextn_predict_layers")) |v| {
            if (v == .integer) config.dsv4_mtp_layers = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "dspark_block_size")) |v| {
            if (v == .integer) config.dsv4_dspark_block_size = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "dspark_noise_token_id")) |v| {
            if (v == .integer) config.dsv4_dspark_noise_token_id = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "dspark_markov_rank")) |v| {
            if (v == .integer) config.dsv4_dspark_markov_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "dspark_target_layer_ids")) |v| {
            if (v == .array) {
                for (v.array.items, 0..) |item, i| {
                    if (i >= config.dsv4_dspark_target_layers.len) break;
                    if (item == .integer) config.dsv4_dspark_target_layers[i] = std.math.cast(u8, try jsonU32(item)) orelse return error.InvalidConfigField;
                }
                config.dsv4_n_dspark_target_layers = @intCast(@min(v.array.items.len, config.dsv4_dspark_target_layers.len));
            }
        }
        if (jsonField(cfg_obj, "compress_ratios")) |v| {
            if (v == .array) {
                for (v.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item == .integer) config.dsv4_compress_ratios[i] = std.math.cast(u8, try jsonU32(item)) orelse return error.InvalidConfigField;
                }
                config.dsv4_n_compress_ratios = @intCast(@min(v.array.items.len, 128));
            }
        }
        // YaRN on compressed layers only; the reference applies NO mscale
        // (softmax scale stays head_dim^-0.5 everywhere), so
        // yarn_attention_factor stays 1.0 — do not compute the 0.1·ln(f)+1
        // default here (laguna-class trap in the other direction).
        if (jsonField(cfg_obj, "rope_scaling")) |rs| {
            if (rs == .object) {
                config.rope_yarn = true;
                if (jsonField(rs.object, "factor")) |x| config.yarn_factor = try jsonFloat(x);
                if (jsonField(rs.object, "beta_fast")) |x| config.yarn_beta_fast = try jsonFloat(x);
                if (jsonField(rs.object, "beta_slow")) |x| config.yarn_beta_slow = try jsonFloat(x);
                if (jsonField(rs.object, "original_max_position_embeddings")) |x| {
                    if (x == .integer) config.yarn_orig_max_pos = try jsonU32(x);
                }
            }
        }
        // Honest rejects: the forward implements exactly sqrt(softplus)
        // scoring with selection-only bias (noaux_tc), ONE always-on shared
        // expert, and a single shared KV latent. Divergent checkpoints must
        // refuse to load, not run silently wrong.
        if (jsonField(cfg_obj, "scoring_func")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "sqrtsoftplus")) {
                log.err("{s}: scoring_func '{s}' not supported (sqrtsoftplus only)\n", .{ config.model_type, v.string });
                return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
            }
        }
        if (jsonField(cfg_obj, "topk_method")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "noaux_tc")) {
                log.err("{s}: topk_method '{s}' not supported (noaux_tc only)\n", .{ config.model_type, v.string });
                return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
            }
        }
        if (jsonField(cfg_obj, "n_shared_experts")) |v| {
            if (v == .integer and v.integer != 1) {
                log.err("{s}: n_shared_experts {d} not supported (exactly 1)\n", .{ config.model_type, v.integer });
                return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
            }
        }
        if (config.num_key_value_heads != 1) {
            log.err("{s}: num_key_value_heads {d} not supported (single shared KV latent)\n", .{ config.model_type, config.num_key_value_heads });
            return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
        }
        // The July-31 release supersedes the preview, and the preview's
        // single next-token MTP module is no longer supported — its draft
        // path (e_proj/h_proj over one stage) shares nothing with DSpark's
        // block-parallel stages beyond the `mtp.*` namespace, so carrying it
        // would mean maintaining a second architecture for a checkpoint the
        // vendor withdrew. A preview config announces itself by declaring MTP
        // layers with no DSpark descriptor; say so instead of loading a model
        // whose draft weights we would silently ignore.
        if (config.dsv4_mtp_layers > 0 and config.dsv4_dspark_block_size == 0) {
            log.err("{s}: this is the superseded PREVIEW checkpoint (num_nextn_predict_layers={d}, no dspark_* config). Use DeepSeek-V4-Flash-0731 or later.\n", .{ config.model_type, config.dsv4_mtp_layers });
            return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
        }
        if (is_v41) {
            // Text-only: the vision tower is never loaded.
            config.has_vision = false;
            try parseDsv41Fields(cfg_obj, &config);
        }
    } else if (std.mem.eql(u8, model_type, "qwen3_next")) {
        config.model_type = "qwen3_next";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "partial_rotary_factor")) |v| config.partial_rotary_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "lfm2") or std.mem.startsWith(u8, model_type, "lfm2")) {
        config.model_type = "lfm2";
        // VL variant nests text weights under language_model.model (like Gemma 4)
        config.weight_prefix = if (jsonField(root, "text_config") != null) "language_model.model" else "model";
        config.hidden_act = .silu;
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.has_sliding_window = false;
        config.has_hybrid_layers = true;
        config.has_embedding_norm = false;
        config.has_final_norm = true; // embedding_norm IS the final norm
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (config.head_dim == 256) { // default from gemma3, override
            if (config.num_attention_heads == 0) return error.InvalidConfigField;
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }
        config.query_pre_attn_scalar = config.head_dim;
        // tie_embedding (LFM2 name) -> tie_word_embeddings; default true for LFM2
        config.tie_word_embeddings = true;
        if (jsonField(cfg_obj, "tie_embedding")) |v| {
            if (v == .bool) config.tie_word_embeddings = v.bool;
        }
        if (jsonField(cfg_obj, "norm_eps")) |v| config.rms_norm_eps = try jsonFloat(v);
        if (jsonField(cfg_obj, "conv_L_cache")) |v| {
            if (v == .integer) config.lfm_conv_kernel = try jsonU32(v);
        }
        if (std.mem.eql(u8, model_type, "lfm2_moe")) {
            config.lfm2_moe = true;
            if (jsonField(cfg_obj, "num_experts")) |v| {
                if (v == .integer) config.num_experts = try jsonU32(v);
            }
            if (jsonField(cfg_obj, "num_experts_per_tok")) |v| {
                if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
            }
            if (jsonField(cfg_obj, "moe_intermediate_size")) |v| {
                if (v == .integer) config.moe_intermediate_size = try jsonU32(v);
            }
            if (jsonField(cfg_obj, "num_dense_layers")) |v| {
                if (v == .integer) config.num_dense_layers = try jsonU32(v);
            }
            if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
                if (v == .bool) config.moe_route_norm = v.bool;
            }
            if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
            if (config.num_experts == 0 or config.num_experts_per_tok == 0 or config.moe_intermediate_size == 0) {
                return error.IncompleteLfm2MoeConfig;
            }
        }
        if (jsonField(cfg_obj, "conv_dim")) |v| config.lfm_conv_dim = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 0,
        };
        // Parse layer_types array: ["conv", "full_attention", ...]
        if (jsonField(cfg_obj, "layer_types")) |lt_val| {
            if (lt_val == .array) {
                for (lt_val.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item != .string) continue;
                    config.layer_block_types[i] = if (std.mem.eql(u8, item.string, "conv"))
                        .gated_conv
                    else
                        .attention;
                }
            }
        }
        if (config.num_eos_tokens == 0) {
            if (jsonField(cfg_obj, "eos_token_id")) |v| {
                if (v == .integer) config.addEosToken(try jsonU32(v));
            }
        }
        // LFM2-VL: a stock `siglip2_vision_model` tower (src/lfm2_vision.zig)
        // plus LFM2-VL's own projector. The generic vision_config block above
        // already read the tower's geometry; everything here is the wrapper.
        // A `lfm2` checkpoint with no vision_config stays text-only.
        if (jsonField(root, "vision_config")) |vc_val| {
            if (vc_val == .object and std.mem.eql(u8, model_type, "lfm2_vl")) {
                config.lfm2_vision = true;
                const vc = vc_val.object;
                config.lv_ln_eps = 1e-6;
                if (jsonField(vc, "layer_norm_eps")) |v| config.lv_ln_eps = try jsonFloat(v);
                // The stored table is square: num_patches = pos_side².
                var num_patches: u32 = 256;
                if (jsonField(vc, "num_patches")) |v| {
                    if (v == .integer) num_patches = try jsonU32(v);
                }
                config.lv_pos_side = std.math.sqrt(num_patches);
                if (jsonField(root, "downsample_factor")) |v| {
                    if (v == .integer) config.lv_downsample = try jsonU32(v);
                }
                if (jsonField(root, "projector_hidden_size")) |v| {
                    if (v == .integer) config.lv_projector_hidden = try jsonU32(v);
                }
                if (jsonField(root, "min_image_tokens")) |v| {
                    if (v == .integer) config.lv_min_image_tokens = try jsonU32(v);
                }
                if (jsonField(root, "max_image_tokens")) |v| {
                    if (v == .integer) config.lv_max_image_tokens = try jsonU32(v);
                }
                if (jsonField(root, "tile_size")) |v| {
                    if (v == .integer) config.lv_tile_size = try jsonU32(v);
                }
                if (jsonField(root, "min_tiles")) |v| {
                    if (v == .integer) config.lv_min_tiles = try jsonU32(v);
                }
                if (jsonField(root, "max_tiles")) |v| {
                    if (v == .integer) config.lv_max_tiles = try jsonU32(v);
                }
                if (jsonField(root, "do_image_splitting")) |v| {
                    if (v == .bool) config.lv_split_images = v.bool;
                }
                if (jsonField(root, "use_thumbnail")) |v| {
                    if (v == .bool) config.lv_use_thumbnail = v.bool;
                }
                if (jsonField(root, "max_pixels_tolerance")) |v| config.lv_pixels_tolerance = try jsonFloat(v);
                if (config.lv_projector_hidden == 0) config.lv_projector_hidden = config.hidden_size;
            } else {
                // `vision_config` present without the VL tag (mlx-community's
                // text-only LFM2.5 packs ship an EMPTY one): never advertise a
                // tower we have no weights for.
                config.has_vision = false;
            }
        }
    } else if (std.mem.eql(u8, model_type, "nemotron_h")) {
        config.model_type = "nemotron_h";
        config.weight_prefix = "backbone";
        config.hidden_act = .silu;
        config.mamba_mlp_act = .relu_sq;
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.has_sliding_window = false;
        config.has_hybrid_layers = true;
        config.has_final_norm = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        config.query_pre_attn_scalar = config.head_dim;
        // NoPE: the reference attention never rotates q/k, whatever
        // rope_theta the config carries. Covers the MTP head's layer too.
        config.layer_no_rope = @splat(true);
        if (jsonField(cfg_obj, "rms_norm_eps")) |v| {
            config.rms_norm_eps = try jsonFloat(v);
        } else if (jsonField(cfg_obj, "layer_norm_epsilon")) |v| {
            config.rms_norm_eps = try jsonFloat(v);
        }
        // Mamba2-specific config
        if (jsonField(cfg_obj, "mamba_num_heads")) |v| config.mamba_num_heads = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 0,
        };
        if (jsonField(cfg_obj, "mamba_head_dim")) |v| config.mamba_head_dim = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 0,
        };
        if (jsonField(cfg_obj, "n_groups")) |v| config.mamba_n_groups = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 8,
        };
        if (jsonField(cfg_obj, "ssm_state_size")) |v| config.ssm_state_size = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 128,
        };
        if (jsonField(cfg_obj, "conv_kernel")) |v| config.mamba_conv_kernel = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 4,
        };
        if (jsonField(cfg_obj, "expand")) |v| config.mamba_expand = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 2,
        };
        // time_step_limit: Python defaults to (0.0, inf) if not in config.
        // config.json may have time_step_min/time_step_max fields but Python ignores them
        // for SSM clipping — only time_step_limit (a 2-element array) is used.
        if (jsonField(cfg_obj, "time_step_limit")) |v| {
            if (v == .array) {
                const items = v.array.items;
                if (items.len >= 2) {
                    config.time_step_min = try jsonFloat(items[0]);
                    config.time_step_max = try jsonFloat(items[1]);
                }
            }
        }
        if (jsonField(cfg_obj, "chunk_size")) |v| config.mamba_chunk_size = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 256,
        };
        // Parse hybrid_override_pattern: "M-M-M-MM-M-M*-..."
        if (jsonField(cfg_obj, "hybrid_override_pattern")) |v| {
            if (v == .string) {
                for (v.string, 0..) |ch, i| {
                    if (i >= 128) break;
                    config.layer_block_types[i] = switch (ch) {
                        'M' => .mamba2,
                        '-' => .mlp,
                        '*' => .attention,
                        'E' => .moe,
                        else => .attention,
                    };
                }
            }
        }
        // Nemotron 3.5 configs spell the same pattern as a list of names; the
        // string wins when both are present (mlx-lm's order).
        if (jsonField(cfg_obj, "layers_block_type")) |v| {
            if (v == .array and jsonField(cfg_obj, "hybrid_override_pattern") == null) {
                for (v.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item != .string) continue;
                    const name = item.string;
                    config.layer_block_types[i] = if (std.mem.eql(u8, name, "mamba"))
                        .mamba2
                    else if (std.mem.eql(u8, name, "mlp"))
                        .mlp
                    else if (std.mem.eql(u8, name, "moe"))
                        .moe
                    else
                        .attention;
                }
            }
        }
        // MoE blocks ('E' / "moe"): sigmoid router with a selection-only
        // score-correction bias, ReLU^2 routed experts, one shared expert.
        if (jsonField(cfg_obj, "n_routed_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "moe_shared_expert_intermediate_size")) |v| {
            if (v == .integer) config.shared_expert_intermediate_size = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "n_group")) |v| {
            if (v == .integer) config.moe_n_group = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "topk_group")) |v| {
            if (v == .integer) config.moe_topk_group = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "moe_latent_size")) |v| {
            if (v == .integer) return error.UnsupportedNemotronLatentMoe;
        }
        if (config.num_eos_tokens == 0) {
            if (jsonField(cfg_obj, "eos_token_id")) |v| {
                if (v == .integer) config.addEosToken(try jsonU32(v));
            }
        }
    } else if (std.mem.eql(u8, model_type, "deepseek_v4")) {
        // MLX-format DSV4 is not supported in this build. Users should load
        // the GGUF checkpoint via the ds4 engine (`*.gguf` early-branch in
        // main.zig / Swift app). Fall through to the unknown-arch error
        // path so the failure message points at the right thing.
        return error.UnsupportedDsv4MlxFormat;
    } else if (std.mem.eql(u8, model_type, "bert")) {
        config.model_type = "bert";
        config.is_encoder_only = true;
        config.weight_prefix = "";
        config.hidden_act = .gelu_approx;
        config.tie_word_embeddings = true;
        config.has_sliding_window = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.scale_embeddings = false;
        config.norm_has_offset = false;
        if (config.num_attention_heads == 0) return error.InvalidConfigField;
        config.head_dim = config.hidden_size / config.num_attention_heads;
        config.num_key_value_heads = config.num_attention_heads;
        config.query_pre_attn_scalar = config.head_dim;
        if (jsonField(cfg_obj, "layer_norm_eps")) |v| config.layer_norm_eps = try jsonFloat(v);
        if (jsonField(cfg_obj, "type_vocab_size")) |v| config.type_vocab_size = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 2,
        };
    } else {
        // Llama-family defaults (qwen3, llama, mistral, etc.)
        if (std.mem.eql(u8, model_type, "qwen3")) {
            config.model_type = "qwen3";
        } else if (std.mem.eql(u8, model_type, "qwen2")) {
            // Qwen2.5 family: dense Llama-style attention, NO QK-norm (left
            // false below), additive qkv-projection biases applied in the
            // forward when present.
            config.model_type = "qwen2";
        } else if (std.mem.eql(u8, model_type, "llama")) {
            config.model_type = "llama";
        } else if (std.mem.eql(u8, model_type, "mistral")) {
            config.model_type = "mistral";
        } else if (std.mem.eql(u8, model_type, "k2_horizon")) {
            // IFM K2-Horizon dense (0.9B/3.7B/7B/32B): a Llama trunk whose
            // RMS norms are GROUPED (`layernorm_num_groups`). The MoVA MoE
            // sizes (`mova_num_experts` > 0) are a different attention and
            // are not served.
            config.model_type = "k2_horizon";
            if (jsonField(cfg_obj, "layernorm_num_groups")) |v| {
                if (v == .integer and v.integer > 1) config.norm_groups = try jsonU32(v);
            }
        } else {
            config.model_type = "unknown";
        }
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.rope_scaling_factor = 1.0;
        config.rope_llama3 = try parseLlama3Rope(cfg_obj);
        config.rope_local_base_freq = config.rope_theta;
        // Every layer attends the whole context, as in mlx-lm's llama/qwen2/mistral:
        // the struct default is Gemma's 5-sliding-to-1-global pattern at 1024.
        config.has_sliding_window = false;
        // Llama-family models (qwen2, llama, mistral) usually omit `head_dim`;
        // the HF default is hidden_size / num_attention_heads. Without this the
        // stale 256 sentinel (line 53) would corrupt attention for any such
        // checkpoint that doesn't ship an explicit head_dim (e.g. Qwen2.5).
        // qwen3 ships an explicit head_dim, so this leaves it untouched.
        if (jsonField(cfg_obj, "head_dim") == null) {
            if (config.num_attention_heads == 0) return error.InvalidConfigField;
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }

        if (jsonField(cfg_obj, "hidden_act")) |v| {
            if (v == .string) {
                if (std.mem.eql(u8, v.string, "silu")) {
                    config.hidden_act = .silu;
                } else if (std.mem.eql(u8, v.string, "gelu_pytorch_tanh")) {
                    config.hidden_act = .gelu_approx;
                }
            }
        }

        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }

        if (std.mem.eql(u8, model_type, "qwen3")) {
            config.has_qk_norm = true;
        }
    }

    if (jsonField(cfg_obj, "linear_num_key_heads") != null or jsonField(cfg_obj, "linear_num_value_heads") != null or
        jsonField(cfg_obj, "linear_key_head_dim") != null or jsonField(cfg_obj, "linear_value_head_dim") != null or
        jsonField(cfg_obj, "linear_conv_kernel_dim") != null)
    {
        if (config.linear_num_key_heads == 0 or config.linear_num_value_heads == 0 or
            config.linear_num_value_heads % config.linear_num_key_heads != 0)
            return error.InvalidConfigField;
    }
    return config;
}

/// V4.1's fields beyond the shared dsv4 block, refused by name where the
/// forward could not serve them: every compressed layer must find a source of
/// its own ratio at or above it (the reference reads whichever source ran
/// last, so a mismatch would silently attend over the wrong cache).
fn parseDsv41Fields(cfg_obj: std.json.ObjectMap, config: *ModelConfig) !void {
    const refuse = struct {
        fn f(comptime fmt: []const u8, args: anytype) error{UnsupportedDsv41Config} {
            log.err("deepseek_v41: " ++ fmt ++ "\n", args);
            return error.UnsupportedDsv41Config;
        }
    }.f;
    const n_layers = config.num_hidden_layers;
    if (n_layers == 0 or n_layers > 64 or config.dsv4_n_compress_ratios < n_layers) return refuse("{d} layers with {d} compress ratios", .{ n_layers, config.dsv4_n_compress_ratios });
    if (config.dsv4_hc_mult != 4) return refuse("hc_mult {d} (only 4 is served)", .{config.dsv4_hc_mult});
    const layerMask = struct {
        fn f(obj: std.json.ObjectMap, key: []const u8, n: u32) !u64 {
            var mask: u64 = 0;
            const v = jsonField(obj, key) orelse return 0;
            for ((try jsonValue(.array, v)).items) |item| {
                const l = try jsonU32(item);
                if (l >= n) return refuse("{s} names layer {d} of {d}", .{ key, l, n });
                mask |= @as(u64, 1) << @intCast(l);
            }
            return mask;
        }
    }.f;
    config.dsv41_kv_sources = try layerMask(cfg_obj, "kv_source_layer_ids", n_layers);
    config.dsv41_index_sources = try layerMask(cfg_obj, "index_source_layer_ids", n_layers);
    if (jsonField(cfg_obj, "candidate_source_layer_id")) |v| config.dsv41_candidate_source = std.math.cast(i32, try jsonValue(.integer, v)) orelse return error.InvalidConfigField;
    if (jsonField(cfg_obj, "candidate_topk_blocks")) |v| config.dsv41_candidate_topk_blocks = try jsonU32(v);
    if (jsonField(cfg_obj, "candidate_block_size")) |v| config.dsv41_candidate_block_size = try jsonU32(v);

    const ratios = config.dsv4_compress_ratios[0..n_layers];
    const has = struct {
        fn f(mask: u64, l: usize) bool {
            return (mask >> @intCast(l)) & 1 != 0;
        }
    }.f;
    var last_kv: ?usize = null;
    var last_owner: ?usize = null;
    var last_index: ?usize = null;
    for (ratios, 0..) |r, l| {
        if (r > 2) return refuse("layer {d}: compress ratio {d} (modes 0, 1, 2 are served)", .{ l, r });
        const kv = has(config.dsv41_kv_sources, l);
        const ix = has(config.dsv41_index_sources, l);
        if (r == 0) {
            if (kv or ix) return refuse("layer {d}: a source without compression", .{l});
            continue;
        }
        if (kv) last_kv = l;
        if (kv and ix) last_owner = l;
        if (ix) last_index = l;
        const src = last_kv orelse return refuse("layer {d}: no KV source at or above it", .{l});
        if (ratios[src] != r) return refuse("layer {d} (ratio {d}) would read layer {d}'s ratio-{d} cache", .{ l, r, src, ratios[src] });
        if (ix) {
            const owner = last_owner orelse return refuse("layer {d}: no index-key owner at or above it", .{l});
            if (ratios[owner] != r) return refuse("layer {d} would score layer {d}'s keys", .{ l, owner });
        } else if (last_index == null) return refuse("layer {d}: no index source at or above it", .{l});
    }
    if (config.dsv41_candidate_source >= 0) {
        const c: usize = @intCast(config.dsv41_candidate_source);
        if (c >= n_layers or !has(config.dsv41_index_sources, c)) return refuse("candidate source {d} is not an index source", .{c});
        if (config.dsv41_candidate_topk_blocks == 0 or config.dsv41_candidate_block_size == 0) return refuse("candidate source without blocks", .{});
    }

    if (jsonField(cfg_obj, "engram_layer_ids")) |v| {
        const items = (try jsonValue(.array, v)).items;
        if (items.len > config.dsv41_engram_layers.len) return refuse("{d} engram layers", .{items.len});
        for (items, 0..) |item, i| {
            const l = try jsonU32(item);
            if (l >= n_layers) return refuse("engram layer {d} of {d}", .{ l, n_layers });
            config.dsv41_engram_layers[i] = @intCast(l);
        }
        config.dsv41_n_engram_layers = @intCast(items.len);
    }
    if (config.dsv41_n_engram_layers > 0) {
        const rows_v = jsonField(cfg_obj, "engram_num_embeddings") orelse return refuse("engram without engram_num_embeddings", .{});
        const rows = (try jsonValue(.array, rows_v)).items;
        if (rows.len != config.dsv41_n_engram_layers) return refuse("{d} engram tables for {d} layers", .{ rows.len, config.dsv41_n_engram_layers });
        for (rows, 0..) |item, i| config.dsv41_engram_rows[i] = std.math.cast(u64, try jsonValue(.integer, item)) orelse return error.InvalidConfigField;
        if (jsonField(cfg_obj, "engram_max_ngram_size")) |x| config.dsv41_engram_max_ngram = try jsonU32(x);
        if (jsonField(cfg_obj, "engram_vocab_size")) |x| config.dsv41_engram_vocab_size = std.math.cast(u64, try jsonValue(.integer, x)) orelse return error.InvalidConfigField;
        if (jsonField(cfg_obj, "engram_n_heads")) |x| config.dsv41_engram_n_heads = try jsonU32(x);
        if (jsonField(cfg_obj, "engram_head_dim")) |x| config.dsv41_engram_head_dim = try jsonU32(x);
        if (jsonField(cfg_obj, "engram_pad_token_id")) |x| config.dsv41_engram_pad_id = try jsonU32(x);
        if (jsonField(cfg_obj, "engram_compressed_vocab_size")) |x| config.dsv41_engram_compressed_vocab = try jsonU32(x);
        if (config.dsv41_engram_max_ngram < 2 or config.dsv41_engram_max_ngram > 8 or
            config.dsv41_engram_n_heads == 0 or config.dsv41_engram_n_heads > 16 or
            config.dsv41_engram_head_dim == 0 or config.dsv41_engram_head_dim % 32 != 0 or
            config.dsv41_engram_compressed_vocab == 0 or config.dsv41_engram_vocab_size == 0)
            return refuse("engram geometry ({d}-grams, {d} heads x {d})", .{ config.dsv41_engram_max_ngram, config.dsv41_engram_n_heads, config.dsv41_engram_head_dim });
    }
    if (jsonField(cfg_obj, "dspark_n_routed_experts")) |v| config.dsv41_dspark_experts = try jsonU32(v);
    if (jsonField(cfg_obj, "dspark_num_experts_per_tok")) |v| config.dsv41_dspark_top_k = try jsonU32(v);
    for (config.dsv4_dspark_target_layers[0..config.dsv4_n_dspark_target_layers]) |l| {
        if (l >= n_layers) return refuse("DSpark target layer {d} of {d}", .{ l, n_layers });
    }
}

fn jsonValue(comptime tag: std.meta.Tag(std.json.Value), v: std.json.Value) !@TypeOf(@field(@as(std.json.Value, undefined), @tagName(tag))) {
    if (v != tag) return error.InvalidConfigField;
    return @field(v, @tagName(tag));
}

fn jsonU32(v: std.json.Value) !u32 {
    return std.math.cast(u32, try jsonValue(.integer, v)) orelse error.InvalidConfigField;
}

// Optional config nulls have the same fallback and precedence as omitted keys.
fn jsonField(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

/// `rope_scaling` (or v5's `rope_parameters`) when it names the llama3 schedule.
fn parseLlama3Rope(cfg_obj: std.json.ObjectMap) !?Llama3Rope {
    const rs = jsonField(cfg_obj, "rope_scaling") orelse jsonField(cfg_obj, "rope_parameters") orelse return null;
    if (rs != .object) return null;
    const t = jsonField(rs.object, "rope_type") orelse jsonField(rs.object, "type") orelse return null;
    if (t != .string or !std.mem.eql(u8, t.string, "llama3")) return null;
    var r = Llama3Rope{ .factor = try jsonFloat(jsonField(rs.object, "factor") orelse return error.InvalidConfigField) };
    if (jsonField(rs.object, "low_freq_factor")) |v| r.low_freq_factor = try jsonFloat(v);
    if (jsonField(rs.object, "high_freq_factor")) |v| r.high_freq_factor = try jsonFloat(v);
    if (jsonField(rs.object, "original_max_position_embeddings")) |v| r.original_max_position_embeddings = try jsonFloat(v);
    if (r.factor <= 0 or r.high_freq_factor <= r.low_freq_factor) return error.InvalidConfigField;
    return r;
}

fn jsonFloat(v: std.json.Value) !f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => 0.0,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfigField;
    return f;
}

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    /// Hand the map a new array under `name`, freeing the one it held (load-time
    /// weight fusion parks its row views here so the originals go away).
    pub fn replace(self: *Weights, name: []const u8, arr: mlx.mlx_array) void {
        if (self.map.getPtr(name)) |p| {
            _ = mlx.mlx_array_free(p.*);
            p.* = arr;
        }
    }

    /// Drop `name` and free its array (a loader that copied it elsewhere lets the original go).
    pub fn remove(self: *Weights, name: []const u8) void {
        const kv = self.map.fetchRemove(name) orelse return;
        _ = mlx.mlx_array_free(kv.value);
        self.allocator.free(kv.key);
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// The generic nestings a text trunk ships under: flat, mlx-community's
/// re-nest, and meta's VL original (Muse-Glimmer). `parseConfigFromJson`
/// picks from config KEYS; this probe corrects it from the checkpoint.
const FLAT_PREFIX = "model";
const NESTED_PREFIX = "language_model.model";
const VL_NESTED_PREFIX = "model.language_model";

fn hasWeightsUnder(weights: *const Weights, prefix: []const u8) bool {
    var it = weights.map.keyIterator();
    while (it.next()) |k| {
        const key = k.*;
        if (key.len > prefix.len and key[prefix.len] == '.' and std.mem.startsWith(u8, key, prefix)) return true;
    }
    return false;
}

/// Re-point `config.weight_prefix` at the nesting the CHECKPOINT actually uses.
///
/// Which of the two a converter emits is not reliably declared in config.json,
/// so `parseConfigFromJson` guesses from `text_config` presence — wrong for any
/// checkpoint that nests without declaring one (mlx-community LFM2.5-2.6B:
/// `Lfm2ForCausalLM`, an EMPTY `vision_config`, every weight under
/// `language_model.model.*`; the guess picked `model` and the load died on
/// `MISSING WEIGHT: model.embed_tokens.weight`). The class has now shipped in
/// both directions, so the weights get the last word.
///
/// Conservative by construction: only the generic spellings participate
/// (never an arch with its own — `backbone`, `model.llm`, `""`), and a swap
/// happens only when the configured one holds NOTHING, so every checkpoint
/// that already loaded binds byte-identically. Scan order puts the most
/// specific spelling first: a `model.language_model.*` checkpoint also
/// satisfies the bare "model" probe.
/// Prism Hadamard packs run f16 activations over f16 scales (the pack's own
/// contract) and ship the unpacked tensors f32 (norms, conv, the dense GDN a/b
/// rows, A_log, dt_bias): an f32 table widens the f16 residual, so those
/// narrow to f16. Signs stay f32: the rotation runs in f32.
pub fn narrowHadamardPackTables(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    if (config.hadamard_block == 0) return;
    var it = weights.map.iterator();
    while (it.next()) |kv| {
        const v = kv.value_ptr.*;
        if (mlx.mlx_array_dtype(v) != .float32) continue;
        if (std.mem.endsWith(u8, kv.key_ptr.*, ".signs")) continue;
        var cast = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast, v, config.actDtype(), s));
        _ = mlx.mlx_array_free(v);
        kv.value_ptr.* = cast;
    }
}

pub fn resolveWeightPrefix(config: *ModelConfig, weights: *const Weights) void {
    const candidates = [_][]const u8{ NESTED_PREFIX, VL_NESTED_PREFIX, FLAT_PREFIX };
    var known = false;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) known = true;
    }
    if (!known) return;

    if (hasWeightsUnder(weights, config.weight_prefix)) return;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) continue;
        if (!hasWeightsUnder(weights, p)) continue;
        log.info("weight prefix: config implies \"{s}\", checkpoint uses \"{s}\" — using the checkpoint's\n", .{ config.weight_prefix, p });
        config.weight_prefix = p;
        return;
    }
}

/// Load all safetensors files from model_dir.
/// When `load_vision` is true, vision_tower and multi_modal_projector weights are included.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, .{});
}

/// How a load treats stored dtypes. `keep_f16`: a pack whose activation dtype
/// is f16 (Prism Hadamard packs) keeps its f16 side tensors and tables as
/// stored; narrowing them to bf16 drops 3 mantissa bits of every group scale.
/// `dsv41` also reads past the page cache (`nocache_reader`): its weights all load at init, and a pack near the
/// RAM size would have them compressed under the file cache.
/// `index_owners`: a tensor two shards carry loads from the shard the index names.
pub const LoadOpts = struct { vision: bool = false, keep_f16: bool = false, embedded_ple: bool = false, defer_qwen4_norms: bool = false, dsv41: bool = false, index_owners: bool = false, shard: []const u8 = "", owners: ?*const Owners = null };

pub const Owners = std.StringHashMapUnmanaged([]const u8);

/// tensor -> shard from the index's `weight_map`, strings owned by `arena`.
pub fn indexOwners(io: std.Io, arena: std.mem.Allocator, dir: std.Io.Dir) ?Owners {
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", arena, .limited(64 * 1024 * 1024)) catch return null;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, raw, .{}) catch return null;
    if (parsed != .object) return null;
    const wm = parsed.object.get("weight_map") orelse return null;
    if (wm != .object) return null;
    var out: Owners = .empty;
    var it = wm.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .string) continue;
        // An owner shard that is not on disk claims nothing.
        _ = dir.statFile(io, e.value_ptr.string, .{}) catch continue;
        out.put(arena, e.key_ptr.*, e.value_ptr.string) catch return null;
    }
    return out;
}

/// The text model's weights for `config`.
pub fn loadModelWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *ModelConfig, load_vision: bool) !Weights {
    if (config.dsv41_stream) return Weights.init(allocator); // mlx-stream loads its own
    var gguf_weights = Weights.init(allocator);
    errdefer gguf_weights.deinit();
    if (try mlx_gguf.loadWeights(io, allocator, model_dir, &gguf_weights.map)) return gguf_weights;
    if (config.embedded_ple_payload_bytes != null) {
        const info = (try qwen4_exp.inspectEmbedded(model_dir, try qwen4EmbeddedSpec(config))) orelse return error.MissingEmbeddedNgramTable;
        if (info.payload_bytes != config.embedded_ple_payload_bytes.?) return error.EmbeddedNgramTableChanged;
    }
    var weights = try loadWeightsOpt(io, allocator, model_dir, .{ .vision = load_vision, .keep_f16 = config.actDtype() == .float16, .embedded_ple = config.isQwen4() and config.embedded_ple_payload_bytes != null, .defer_qwen4_norms = config.isQwen4(), .dsv41 = config.isDsv41(), .index_owners = config.exl3 != null });
    errdefer weights.deinit();
    if (config.isDsv41()) try loadDsv41Residents(io, allocator, &weights, model_dir);
    if (config.isQwen4()) {
        defer reportF16Narrowing();
        resolveWeightPrefix(config, &weights);
        const s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        try resolveAndFoldQwen4Norms(config, &weights, s, model_dir);
    }
    if (config.exl3 != null) {
        const s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        try sushi_pack.adapt(config, &weights, s);
    }
    return weights;
}

/// deepseek_v41 serves text only and reads its Engram tables from disk per
/// lookup (~100 GB a layer), so neither the vision tower nor those tables
/// enter the weight map.
fn dsv41DropsKey(key: []const u8) bool {
    return dsv41VisionKey(key) or std.mem.indexOf(u8, key, ".engram.embed.") != null;
}

pub fn dsv41VisionKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "vision.") or std.mem.startsWith(u8, key, "aligner.") or std.mem.startsWith(u8, key, "image_");
}

/// The repack keeps the Engram projections beside its tables, under `engram/`.
fn loadDsv41Residents(io: std.Io, allocator: std.mem.Allocator, weights: *Weights, model_dir: []const u8) !void {
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/engram/engram-residents.safetensors", .{model_dir}, 0);
    defer allocator.free(path);
    std.Io.Dir.accessAbsolute(io, path, .{}) catch return;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try loadSafetensorsFile(allocator, weights, path, s, .{});
}

/// Load ONE safetensors file (absolute path) into a Weights map — for
/// sidecar files that live beside the trunk shards (e.g. a root-level
/// `mtp.safetensors`), where a directory scan would sweep in the trunk.
pub fn loadWeightsSingleFile(allocator: std.mem.Allocator, abs_path: []const u8) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const pathz = try allocator.dupeSentinel(u8, abs_path, 0);
    defer allocator.free(pathz);
    try loadSafetensorsFile(allocator, &weights, pathz, s, .{});

    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} — corrupt or empty safetensors file?\n", .{abs_path});
        return error.NoWeightFiles;
    }
    return weights;
}

pub fn loadWeightsWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, .{ .vision = true });
}

fn loadWeightsOpt(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, opts: LoadOpts) !Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    return loadWeightsFromOpenDir(io, allocator, dir, model_dir, opts);
}

/// Load every `*.safetensors` in an already-open `dir` into a Weights map.
/// `model_dir` is the on-disk path string, used both to build the per-file
/// absolute path for `mlx_load_safetensors` and to phrase the error message.
/// Split out of `loadWeightsOpt` so the incomplete-checkpoint guard below is
/// unit-testable against a `tmpDir` (mirrors `model_discovery.discoverModelsInDir`).
fn loadWeightsFromOpenDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8, opts: LoadOpts) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    // The index names the shards; anything else is dead weight (issue #274:
    // a pack shipped two shards no weight_map entry names — RAM for nothing)
    // or a foreign file whose parse failure would be an uncatchable MLX abort.
    var referenced = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);

    var owners_arena = std.heap.ArenaAllocator.init(allocator);
    defer owners_arena.deinit();
    var owners: ?Owners = if (opts.index_owners) indexOwners(io, owners_arena.allocator(), dir) else null;

    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        // Accept regular files AND symlinks: HuggingFace cache snapshots store
        // every weight file as a symlink into ../../blobs/<hash>. mlx_load_safetensors
        // resolves the link at the OS level, so a symlinked *.safetensors loads fine.
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) {
            log.warn("skipping {s}: not named by model.safetensors.index.json\n", .{entry.name});
            continue;
        };

        const path_slice = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        defer allocator.free(path_slice);
        const path = try allocator.dupeSentinel(u8, path_slice, 0);
        defer allocator.free(path);

        log.info("Loading {s}...\n", .{entry.name});
        var shard_opts = opts;
        shard_opts.shard = entry.name;
        if (owners) |*o| shard_opts.owners = o;
        try loadSafetensorsFile(allocator, &weights, path, s, shard_opts);
        file_count += 1;
    }

    // Incomplete-checkpoint guard. A dir with config/tokenizer but no (or no
    // usable) *.safetensors is the classic interrupted-download shape: the
    // small files land first, the multi-GB weight shards never finalize. Before
    // this guard the loader returned an empty map and the caller crashed with a
    // misleading `MISSING WEIGHT: <prefix>.embed_tokens.weight` (the first
    // weight looked up) + `unreachable`, pointing at the model arch instead of
    // the download. Fail here with an actionable message, mirroring the
    // tokenizer path's "incomplete download?" hint (see main.zig).
    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} ({d} *.safetensors file(s) found) — the checkpoint looks like an incomplete download (config/tokenizer present, weight shards missing). Re-download the model (e.g. `mlx-serve pull <model>`) or delete the dir and re-fetch.\n", .{ model_dir, file_count });
        return error.NoWeightFiles;
    }

    log.info("Loaded {d} weights from {d} file(s)\n", .{ weights.count(), file_count });
    if (!opts.defer_qwen4_norms) reportF16Narrowing();
    return weights;
}

/// Whether a just-loaded f16 tensor must be narrowed to the engine's bf16
/// activation dtype.
///
/// Two shapes qualify, for the same underlying reason — an f16 value that
/// meets a bf16 activation promotes the RESULT to f32:
///
///   - Quant SIDE tensors (scales/biases), which can be 2-D so they are keyed
///     on the suffix. f16 side tensors force gather_qmm/qmatmul onto a ~4x
///     slower mixed-dtype path (hy_v3 2-bit live, 2026-07-14: 0.70 vs 0.18 ms
///     per 8-expert gather — 1.2 tok/s on the 295B instead of ~15+).
///   - ANY 1-D f16 tensor: a per-channel table (norm weight, bias, A_log,
///     dt_bias) that is multiplied or added straight into the residual. Leave
///     one f16 and the residual turns f32 at the first layer and STAYS f32,
///     so every later weight read is upcast — the Laguna YaRN-mscale class,
///     one level up. Measured on prism-ml/Ternary-Bonsai-27B-mlx-2bit (the
///     only f16 checkpoint on hand, qwen3_5 GDN hybrid): 27.99 -> 23.88
///     ms/forward, 14.7%, three paired boots with cooldown.
///
/// Plain multi-dimensional WEIGHTS keep their dtype. They are matmul
/// OPERANDS, and MLX selects its kernel off that dtype, so narrowing one is a
/// kernel-selection change rather than a promotion fix — measured as a wash
/// here (23.15 vs 23.47 ms, inside boot-to-boot drift), so the minimal rule
/// is the one that ships.
///
/// The cast node stays lazy, so the load-time batch eval materializes bf16
/// directly. Delta from the 3 dropped mantissa bits: cos 0.99999994 — far
/// below any quant noise floor.
pub fn narrowsLoadedF16(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    if (dtype != .float16) return false;
    if (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases")) return true;
    return ndim == 1;
}

/// Kill switch for the 1-D arm (`MLX_SERVE_F16_NARROW_1D=0`). A load-time
/// dtype normalization is invisible once the model is up, so a one-boot A/B
/// switch is the only way to attribute a future f16-checkpoint regression to
/// it. The side-tensor arm predates this and is not switchable.
var narrow_1d_env: ?bool = null;

/// ON for Metal and CUDA: an f16 norm left beside bf16 activations widens the
/// whole residual to f32. OFF for the Omarchy Vulkan backend, which has no
/// bf16 GPU kernel.
fn narrow1dDefault(darwin: bool, cuda: bool) bool {
    return darwin or cuda;
}

test "f16 1-D tables narrow to bf16 on Metal and CUDA, not on Vulkan" {
    try std.testing.expect(narrow1dDefault(true, false));
    try std.testing.expect(narrow1dDefault(false, true));
    try std.testing.expect(!narrow1dDefault(false, false));
}
fn narrow1dEnabled() bool {
    if (narrow_1d_env) |v| return v;
    const on = blk: {
        const raw = std.c.getenv("MLX_SERVE_F16_NARROW_1D") orelse
            break :blk narrow1dDefault(builtin.os.tag.isDarwin(), mlx.cudaAvailable());
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    narrow_1d_env = on;
    return on;
}

/// Count of 1-D f16 tables narrowed this load — reported once per model so a
/// declined normalization is nameable from the log instead of silently
/// reading as "this checkpoint just isn't f16".
var narrowed_1d: usize = 0;

pub fn reportF16Narrowing() void {
    if (narrowed_1d == 0) return;
    log.info("[dtype] narrowed {d} 1-D f16 tables to bf16 (MLX_SERVE_F16_NARROW_1D=0 disables)\n", .{narrowed_1d});
    narrowed_1d = 0;
}

/// Load ONE safetensors file into a Weights map, tensors as stored (lazy).
/// Safetensors load runs on a CPU stream (Load::eval_gpu is Not Implemented —
/// the GPU-stream path kills the whole server). The iterator hands a +1
/// reference in `value`; it transfers straight into the map.
pub fn loadWeightsFile(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8) !Weights {
    var w = Weights.init(allocator);
    errdefer w.deinit();
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
    defer allocator.free(path);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, mlx.mlx_default_cpu_stream_new()));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const owned_key = allocator.dupe(u8, std.mem.span(key.?)) catch |e| {
            _ = mlx.mlx_array_free(value);
            return e;
        };
        w.map.put(owned_key, value) catch |e| {
            allocator.free(owned_key);
            _ = mlx.mlx_array_free(value);
            return e;
        };
    }
    log.info("[weights] loaded {d} tensors from {s}\n", .{ w.count(), file });
    return w;
}

pub fn loadSafetensorsFile(
    allocator: std.mem.Allocator,
    weights: *Weights,
    path: [*:0]const u8,
    s: mlx.mlx_stream,
    opts: LoadOpts,
) !void {
    const load_vision = opts.vision;
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);

    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

    if (opts.dsv41 and comptime builtin.os.tag.isDarwin())
        try @import("nocache_reader.zig").loadSafetensors(&tensor_map, &meta_map, path, s)
    else
        try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);

    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();

        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }

        const key_str = std.mem.span(key.?);

        const foreign = if (opts.owners) |o| (if (o.get(key_str)) |owner| !std.mem.eql(u8, owner, opts.shard) else false) else false;
        if (foreign or !shouldKeepWeightKey(key_str, load_vision) or (opts.embedded_ple and qwen4_exp.embeddedTensorName(key_str)) or
            (opts.dsv41 and dsv41DropsKey(key_str)))
        {
            _ = mlx.mlx_array_free(value);
            continue;
        }

        // Read the shape BEFORE the cast frees `value` — a freed handle's
        // ndim is a use-after-free, not a zero.
        const ndim = mlx.mlx_array_ndim(value);
        var final_value = value;
        if (!(opts.defer_qwen4_norms and qwen4NormNeedsFold(key_str)) and
            !opts.keep_f16 and narrowsLoadedF16(key_str, ndim, mlx.mlx_array_dtype(value)) and
            (ndim != 1 or narrow1dEnabled()))
        {
            var cast = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&cast, value, .bfloat16, s));
            _ = mlx.mlx_array_free(value);
            final_value = cast;
            if (ndim == 1) narrowed_1d += 1;
        }

        // oMLX's V4.1 packs nest the text model under `language_model.`: kept under the release's names.
        const name = if (opts.dsv41 and std.mem.startsWith(u8, key_str, dsv41_text_prefix)) key_str[dsv41_text_prefix.len..] else key_str;
        const owned_key = try allocator.dupe(u8, name);
        try weights.map.put(owned_key, final_value);
    }
}

pub const dsv41_text_prefix = "language_model.";

fn foldQwen4Norm(value: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var shifted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shifted);
    try mlx.check(mlx.mlx_add(&shifted, value, one, s));
    var folded = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(folded);
    try mlx.check(mlx.mlx_astype(&folded, shifted, .bfloat16, s));
    return folded;
}

fn inferLoadedQwen4Norms(config: *const ModelConfig, weights: *const Weights, s: mlx.mlx_stream) !Qwen4NormConvention {
    if (config.num_hidden_layers > 4096 or config.indexer_head_dim == 0) return error.InvalidQwen4NormAnchor;
    var inferred: ?Qwen4NormConvention = null;
    var total_bytes: u64 = 0;
    for (0..config.num_hidden_layers) |layer| {
        if (config.isLinearLayer(@intCast(layer))) continue;
        for ([_][]const u8{ "q_layernorm.weight", "k_layernorm.weight" }) |suffix| {
            var name_buf: [256]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "{s}.layers.{d}.self_attn.indexer.{s}", .{ config.weight_prefix, layer, suffix }) catch return error.InvalidQwen4NormAnchor;
            const value = weights.get(name) orelse {
                log.err("[qwen4] missing norm anchor {s}\n", .{name});
                return error.MissingQwen4NormAnchor;
            };
            const shape = mlx.getShape(value);
            if (shape.len != 1 or shape[0] != config.indexer_head_dim) return error.InvalidQwen4NormAnchor;
            const dtype = mlx.mlx_array_dtype(value);
            if (dtype != .float32 and dtype != .float16 and dtype != .bfloat16) return error.InvalidQwen4NormAnchor;
            const bytes = std.math.mul(u64, @as(u64, @intCast(mlx.mlx_array_size(value))), @as(u64, @intCast(mlx.mlx_array_itemsize(value)))) catch return error.InvalidQwen4NormAnchor;
            total_bytes = std.math.add(u64, total_bytes, bytes) catch return error.InvalidQwen4NormAnchor;
            if (total_bytes > 1024 * 1024) return error.InvalidQwen4NormAnchor;

            var f32_value = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32_value);
            try mlx.check(mlx.mlx_astype(&f32_value, value, .float32, s));
            try mlx.check(mlx.mlx_array_eval(f32_value));
            const data = mlx.mlx_array_data_float32(f32_value) orelse return error.InvalidQwen4NormAnchor;
            var sum: f64 = 0;
            for (data[0..@intCast(config.indexer_head_dim)]) |number| {
                if (!std.math.isFinite(number)) return error.AmbiguousQwen4NormConvention;
                sum += number;
            }
            const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(config.indexer_head_dim)));
            const current: Qwen4NormConvention = if (mean >= -0.2 and mean <= 0.2) .delta else if (mean >= 0.8 and mean <= 1.2) .folded else return error.AmbiguousQwen4NormConvention;
            if (inferred) |prior| {
                if (prior != current) return error.AmbiguousQwen4NormConvention;
            } else inferred = current;
        }
    }
    return inferred orelse error.InvalidQwen4NormAnchor;
}

fn resolveAndFoldQwen4Norms(config: *ModelConfig, weights: *Weights, s: mlx.mlx_stream, model_dir: []const u8) !void {
    const marked = config.qwen4_norm_convention != null;
    const inferred = config.qwen4_norm_convention == null and config.embedded_ple_payload_bytes != null;
    const convention = if (config.qwen4_norm_convention) |marker| marker else if (inferred)
        inferLoadedQwen4Norms(config, weights, s) catch |err| {
            log.err("[qwen4] cannot infer norm convention for {s}: {s}; set qwen4_norm_convention to delta or folded in this checkpoint's config.json.\n", .{ model_dir, @errorName(err) });
            return err;
        }
    else
        Qwen4NormConvention.folded;

    var it = weights.map.iterator();
    while (it.next()) |entry| {
        if (!qwen4NormNeedsFold(entry.key_ptr.*)) continue;
        const value = entry.value_ptr.*;
        if (convention == .delta) {
            entry.value_ptr.* = try foldQwen4Norm(value, s);
        } else if (config.actDtype() != .float16 and narrowsLoadedF16(entry.key_ptr.*, mlx.mlx_array_ndim(value), mlx.mlx_array_dtype(value)) and narrow1dEnabled()) {
            var cast = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(cast);
            try mlx.check(mlx.mlx_astype(&cast, value, .bfloat16, s));
            entry.value_ptr.* = cast;
            narrowed_1d += 1;
        } else continue;
        _ = mlx.mlx_array_free(value);
    }
    config.qwen4_norm_convention = convention;
    if (inferred) {
        log.info("[qwen4] inferred {s} norm convention from loaded trunk indexer norms: {s}\n", .{ @tagName(convention), model_dir });
    } else if (!marked) {
        log.info("[qwen4] using folded norm convention (legacy external-table default): {s}\n", .{model_dir});
    } else {
        log.info("[qwen4] using checkpoint-local {s} norm convention: {s}\n", .{ @tagName(convention), model_dir });
    }
}

pub fn qwen4NormNeedsFold(key: []const u8) bool {
    const suffixes = [_][]const u8{
        "hc_norm.weight",      "q_norm.weight",         "k_norm.weight",        "q_layernorm.weight",           "k_layernorm.weight",
        "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
    };
    for (suffixes) |suffix| if (std.mem.endsWith(u8, key, suffix)) return true;
    return false;
}

test "Qwen4 norm convention and embedded table layout are independent" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{root_buf[0..root_len]}, 0);
    defer allocator.free(path);
    const norms = [_][:0]const u8{
        "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight",
        "language_model.model.layers.1.self_attn.q_norm.weight",
        "language_model.model.layers.1.self_attn.k_norm.weight",
        "language_model.model.layers.1.self_attn.indexer.q_layernorm.weight",
        "language_model.model.layers.1.self_attn.indexer.k_layernorm.weight",
        "language_model.model.layers.1.ple.norm_key.weight",
        "language_model.model.layers.1.ple.norm_query.weight",
        "language_model.model.layers.1.ple.norm_conv.weight",
        "language_model.model.mtp.pre_fc_norm_embedding.weight",
        "language_model.model.mtp.pre_fc_norm_hidden.weight",
    };
    const gate = "language_model.model.layers.1.linear_attn.norm.weight";
    const multiplier = "language_model.model.layers.1.ple.ple_embedding.layer_multipliers";
    const shard = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight";
    const scale = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale";
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const shape = [_]c_int{1};
        const data = [_]f32{0.5};
        const value = mlx.mlx_array_new_data(&data, &shape, 1, .float32);
        defer _ = mlx.mlx_array_free(value);
        var raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(raw);
        try mlx.check(mlx.mlx_astype(&raw, value, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(raw));
        for (norms) |norm| _ = mlx.mlx_map_string_to_array_insert(map, norm, raw);
        _ = mlx.mlx_map_string_to_array_insert(map, gate, raw);
        _ = mlx.mlx_map_string_to_array_insert(map, multiplier, raw);
        _ = mlx.mlx_map_string_to_array_insert(map, shard, raw);
        _ = mlx.mlx_map_string_to_array_insert(map, scale, raw);
        try mlx.check(mlx.mlx_save_safetensors(path.ptr, map, meta));
    }
    var weights = Weights.init(allocator);
    defer weights.deinit();
    try loadSafetensorsFile(allocator, &weights, path, s, .{ .embedded_ple = true, .defer_qwen4_norms = true });
    var embedded_delta_config = ModelConfig{ .model_type = "qwen4_exp", .qwen4_norm_convention = .delta, .embedded_ple_payload_bytes = 120 };
    try resolveAndFoldQwen4Norms(&embedded_delta_config, &weights, s, "fixture");
    try testing.expect(weights.get(shard) == null);
    try testing.expect(weights.get(scale) == null);
    try testing.expect(weights.get(multiplier) != null);
    var f32_gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_gate);
    try mlx.check(mlx.mlx_astype(&f32_gate, weights.get(gate).?, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32_gate));
    try testing.expectEqual(@as(f32, 0.5), mlx.mlx_array_data_float32(f32_gate).?[0]);
    try testing.expect(!qwen4NormNeedsFold(gate));
    var converted = Weights.init(allocator);
    defer converted.deinit();
    try loadSafetensorsFile(allocator, &converted, path, s, .{ .defer_qwen4_norms = true });
    var external_folded_config = ModelConfig{ .model_type = "qwen4_exp" };
    try resolveAndFoldQwen4Norms(&external_folded_config, &converted, s, "fixture");
    try testing.expect(converted.get(shard) != null);
    try testing.expect(converted.get(scale) != null);
    var external_delta = Weights.init(allocator);
    defer external_delta.deinit();
    try loadSafetensorsFile(allocator, &external_delta, path, s, .{ .defer_qwen4_norms = true });
    var external_delta_config = ModelConfig{ .model_type = "qwen4_exp", .qwen4_norm_convention = .delta };
    try resolveAndFoldQwen4Norms(&external_delta_config, &external_delta, s, "fixture");
    try testing.expect(external_delta.get(shard) != null);
    try testing.expect(external_delta.get(scale) != null);
    var embedded_folded = Weights.init(allocator);
    defer embedded_folded.deinit();
    try loadSafetensorsFile(allocator, &embedded_folded, path, s, .{ .embedded_ple = true, .defer_qwen4_norms = true });
    var embedded_folded_config = ModelConfig{ .model_type = "qwen4_exp", .qwen4_norm_convention = .folded, .embedded_ple_payload_bytes = 120 };
    try resolveAndFoldQwen4Norms(&embedded_folded_config, &embedded_folded, s, "fixture");
    try testing.expect(embedded_folded.get(shard) == null);
    try testing.expect(embedded_folded.get(scale) == null);
    const cases = [_]struct { weights: *Weights, expected: f32 }{
        .{ .weights = &weights, .expected = 1.5 },
        .{ .weights = &external_delta, .expected = 1.5 },
        .{ .weights = &converted, .expected = 0.5 },
        .{ .weights = &embedded_folded, .expected = 0.5 },
    };
    for (norms) |norm| {
        try testing.expect(qwen4NormNeedsFold(norm));
        for (cases) |case| {
            const value = case.weights.get(norm).?;
            if (case.expected == 1.5) try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(value));
            var f32_value = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32_value);
            try mlx.check(mlx.mlx_astype(&f32_value, value, .float32, s));
            try mlx.check(mlx.mlx_array_eval(f32_value));
            try testing.expectEqual(case.expected, mlx.mlx_array_data_float32(f32_value).?[0]);
        }
    }
}

fn addQwen4NormTestArray(weights: *Weights, name: []const u8, values: []const f32, shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !void {
    const raw = mlx.mlx_array_new_data(values.ptr, shape.ptr, @intCast(shape.len), .float32);
    var value = raw;
    if (dtype != .float32) {
        value = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&value, raw, dtype, s));
        _ = mlx.mlx_array_free(raw);
    }
    errdefer _ = mlx.mlx_array_free(value);
    const key = try weights.allocator.dupe(u8, name);
    errdefer weights.allocator.free(key);
    try weights.map.put(key, value);
}

test "Qwen4 loaded indexer norms infer only unanimous finite full-attention anchors" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const names = [_][]const u8{
        "language_model.model.layers.0.self_attn.indexer.q_layernorm.weight",
        "language_model.model.layers.0.self_attn.indexer.k_layernorm.weight",
        "language_model.model.layers.1.self_attn.indexer.q_layernorm.weight",
        "language_model.model.layers.1.self_attn.indexer.k_layernorm.weight",
    };
    const shape = [_]c_int{4};
    for ([_]struct { value: f32, dtype: mlx.mlx_dtype, expected: Qwen4NormConvention }{
        .{ .value = 0.05, .dtype = .float32, .expected = .delta },
        .{ .value = 0.05, .dtype = .float16, .expected = .delta },
        .{ .value = 0.95, .dtype = .bfloat16, .expected = .folded },
    }) |case| {
        var weights = Weights.init(testing.allocator);
        defer weights.deinit();
        for (names) |name| {
            const values = [_]f32{ case.value, case.value, case.value, case.value };
            try addQwen4NormTestArray(&weights, name, &values, &shape, case.dtype, s);
        }
        const config = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 2, .full_attention_interval = 1, .indexer_head_dim = 4 };
        try testing.expectEqual(case.expected, try inferLoadedQwen4Norms(&config, &weights, s));
    }

    var weights = Weights.init(testing.allocator);
    defer weights.deinit();
    const values = [_]f32{ 0.05, 0.05, 0.05, 0.05 };
    try addQwen4NormTestArray(&weights, names[2], &values, &shape, .float32, s);
    try addQwen4NormTestArray(&weights, names[3], &values, &shape, .float32, s);
    const hybrid = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 2, .full_attention_interval = 2, .indexer_head_dim = 4 };
    try testing.expectEqual(Qwen4NormConvention.delta, try inferLoadedQwen4Norms(&hybrid, &weights, s));
}

test "Qwen4 loaded indexer norms refuse missing malformed mixed and nonfinite anchors" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const names = [_][]const u8{
        "language_model.model.layers.0.self_attn.indexer.q_layernorm.weight",
        "language_model.model.layers.0.self_attn.indexer.k_layernorm.weight",
        "language_model.model.layers.1.self_attn.indexer.q_layernorm.weight",
        "language_model.model.layers.1.self_attn.indexer.k_layernorm.weight",
    };
    const config = ModelConfig{ .model_type = "qwen4_exp", .num_hidden_layers = 2, .full_attention_interval = 1, .indexer_head_dim = 4 };
    for ([_]struct { bad: usize, value: f32 = 0.05, dtype: mlx.mlx_dtype = .float32, shape: []const c_int = &.{4}, omit: bool = false, expected: anyerror }{
        .{ .bad = 0, .omit = true, .expected = error.MissingQwen4NormAnchor },
        .{ .bad = 3, .omit = true, .expected = error.MissingQwen4NormAnchor },
        .{ .bad = 1, .value = 0.95, .expected = error.AmbiguousQwen4NormConvention },
        .{ .bad = 3, .value = 0.5, .expected = error.AmbiguousQwen4NormConvention },
        .{ .bad = 2, .value = std.math.nan(f32), .expected = error.AmbiguousQwen4NormConvention },
        .{ .bad = 0, .value = std.math.inf(f32), .expected = error.AmbiguousQwen4NormConvention },
        .{ .bad = 1, .dtype = .int32, .expected = error.InvalidQwen4NormAnchor },
        .{ .bad = 2, .shape = &.{ 2, 2 }, .expected = error.InvalidQwen4NormAnchor },
    }) |case| {
        var weights = Weights.init(testing.allocator);
        defer weights.deinit();
        for (names, 0..) |name, i| {
            if (i == case.bad and case.omit) continue;
            const value = if (i == case.bad) case.value else @as(f32, 0.05);
            const values = [_]f32{ value, value, value, value };
            try addQwen4NormTestArray(&weights, name, &values, if (i == case.bad) case.shape else &.{4}, if (i == case.bad) case.dtype else .float32, s);
        }
        try testing.expectError(case.expected, inferLoadedQwen4Norms(&config, &weights, s));
    }
    var empty = Weights.init(testing.allocator);
    defer empty.deinit();
    var no_full = config;
    no_full.num_hidden_layers = 0;
    try testing.expectError(error.InvalidQwen4NormAnchor, inferLoadedQwen4Norms(&no_full, &empty, s));
}

test "Qwen4 delta folds f16 before bf16 narrowing and resolves the shared MTP map once" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var weights = Weights.init(testing.allocator);
    defer weights.deinit();
    const shape = [_]c_int{1};
    const values = [_]f32{0.0117};
    const trunk = "language_model.model.layers.1.self_attn.indexer.q_layernorm.weight";
    const mtp = "language_model.model.mtp.pre_fc_norm_hidden.weight";
    try addQwen4NormTestArray(&weights, trunk, &values, &shape, .float16, s);
    try addQwen4NormTestArray(&weights, mtp, &values, &shape, .float16, s);
    var config = ModelConfig{ .model_type = "qwen4_exp", .qwen4_norm_convention = .delta, .embedded_ple_payload_bytes = 120 };
    try resolveAndFoldQwen4Norms(&config, &weights, s, "fixture");
    try testing.expectEqual(Qwen4NormConvention.delta, config.qwen4_norm_convention.?);
    for ([_][]const u8{ trunk, mtp }) |name| {
        const arr = weights.get(name).?;
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(arr));
        var f32_value = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32_value);
        try mlx.check(mlx.mlx_astype(&f32_value, arr, .float32, s));
        try mlx.check(mlx.mlx_array_eval(f32_value));
        try testing.expectEqual(@as(f32, 1.0078125), mlx.mlx_array_data_float32(f32_value).?[0]);
    }
}

test "Qwen4 folded f16 keeps generic narrowing after convention resolution" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    defer reportF16Narrowing();
    const shape = [_]c_int{1};
    const values = [_]f32{0.95};
    const name = "language_model.model.layers.1.self_attn.indexer.k_layernorm.weight";
    for ([_]?Qwen4NormConvention{ .folded, null }) |marker| {
        var weights = Weights.init(testing.allocator);
        defer weights.deinit();
        try addQwen4NormTestArray(&weights, name, &values, &shape, .float16, s);
        var config = ModelConfig{ .model_type = "qwen4_exp", .qwen4_norm_convention = marker };
        try resolveAndFoldQwen4Norms(&config, &weights, s, "fixture");
        try testing.expectEqual(Qwen4NormConvention.folded, config.qwen4_norm_convention.?);
        try testing.expectEqual(if (narrow1dEnabled()) mlx.mlx_dtype.bfloat16 else mlx.mlx_dtype.float16, mlx.mlx_array_dtype(weights.get(name).?));
    }
}

/// True if the safetensors weight `key` should be retained for the text
/// forward pass. Audio is always dropped; vision is dropped unless
/// `load_vision` is set. MTP-style head tensors (`*.mtp.*`) on Qwen3.5/3.6
/// checkpoints are kept (the binder ignores them, but the loader doesn't
/// need to know that).
pub fn shouldKeepWeightKey(key: []const u8, load_vision: bool) bool {
    // Gemma 4 12B `gemma4_unified` is encoder-free: it ships a tiny vision
    // patch embedder (`vision_embedder.*` + `embed_vision.*`) and a raw-waveform
    // audio projection (`embed_audio.*`) instead of the SigLIP vision tower and
    // conformer audio tower of earlier Gemma 4 variants. Those embedders are
    // wired in src/vision.zig (UnifiedEmbedder), so keep them under the same
    // `load_vision` gate as the SigLIP weights (`--no-vision` → text only).
    const is_vision = std.mem.startsWith(u8, key, "vision_tower.") or
        std.mem.startsWith(u8, key, "embed_vision.") or
        std.mem.startsWith(u8, key, "vision_embedder.") or
        std.mem.startsWith(u8, key, "embed_audio.") or
        std.mem.startsWith(u8, key, "multi_modal_projector.") or
        std.mem.startsWith(u8, key, "language_model.multi_modal_projector.");
    // The heavy SigLIP-era conformer audio tower is still not wired — drop it.
    const is_audio_tower = std.mem.startsWith(u8, key, "audio_tower.") or
        std.mem.startsWith(u8, key, "language_model.audio_multi_modal_projector.");
    if (is_audio_tower) return false;
    // DiffusionGemma nests its (not-yet-wired) vision tower under
    // model.encoder.* — always drop it so a 26B text load doesn't carry
    // ~1 GB of dead tower weights. The encoder LAYER SCALARS
    // (model.encoder.language_model.layers.N.layer_scalar) must survive:
    // they're the only untied encoder text params and the causal encoder
    // pass multiplies by them instead of the decoder's layer_scalar.
    if (std.mem.startsWith(u8, key, "model.encoder.vision_tower.") or
        std.mem.startsWith(u8, key, "model.encoder.embed_vision.")) return false;
    // Muse-Glimmer nests its tower/adapter/projection under "model."; the
    // mlx-community re-nest drops that prefix (its bare "vision_tower." already
    // rides the is_vision gate above). Both follow --no-vision.
    if (!load_vision and (std.mem.startsWith(u8, key, "model.vision_tower.") or
        std.mem.startsWith(u8, key, "model.vision_adapter.") or
        std.mem.startsWith(u8, key, "model.vision_projection.") or
        std.mem.startsWith(u8, key, "vision_adapter.") or
        std.mem.startsWith(u8, key, "vision_projection.") or
        // avlp12's Qwen3.8 "Alis" packs spell the Qwen3-VL tower
        // `model.visual.` (pure rename of `vision_tower.`).
        std.mem.startsWith(u8, key, "model.visual."))) return false;
    if (is_vision and !load_vision) return false;
    return true;
}

// ── Tests ──

const testing = std.testing;

test "ModelConfig defaults" {
    const config = ModelConfig{};
    try testing.expectEqual(@as(u32, 0), config.num_eos_tokens);
    try testing.expectEqual(@as(u32, 0), config.max_position_embeddings);
    try testing.expectEqual(@as(u32, 0), config.quant_bits); // 0 = dense bf16 (no "quantization" key)
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    try testing.expect(!config.tie_word_embeddings);
}

test "loadWeights casts f16 quant scales/biases to bf16 (mixed-dtype qmm slow-path class)" {
    // hy_v3 2-bit (ox-ox) ships F16 scales/biases beside bf16 activations —
    // MLX's gather_qmm/qmatmul take a ~4x slower mixed-dtype path (measured
    // 2026-07-14: 0.70 vs 0.18 ms per 8-expert gather; 1.2 tok/s on the 295B
    // instead of ~15+). The loader must cast quant SIDE tensors to bf16 once;
    // weights and non-quant tensors keep their dtype. Dequant delta from the
    // 3 dropped mantissa bits: cos 0.99999994 — under the 2-bit noise floor.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);

    // Build a tiny map: an f16 "scales", an f16 "biases", an f16 plain weight
    // (must NOT be cast), and a bf16 scales (no-op).
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);

        const shape = [_]c_int{ 4, 4 };
        const data: [16]f32 = @splat(0.5);
        const f32_arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var f16_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f16_arr);
        try mlx.check(mlx.mlx_astype(&f16_arr, f32_arr, .float16, s));
        var bf16_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bf16_arr);
        try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(f16_arr));
        try mlx.check(mlx.mlx_array_eval(bf16_arr));

        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.gate_proj.scales", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.gate_proj.biases", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.up_proj.weight", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.down_proj.scales", bf16_arr);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    var weights = try loadWeights(io, allocator, dir_path);
    defer weights.deinit();

    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.gate_proj.scales").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.gate_proj.biases").?));
    // A plain WEIGHT stays f16 (dense-f16 tables are legitimate — only the
    // quant side tensors force the mixed-dtype qmm path).
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.up_proj.weight").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.down_proj.scales").?));
}

test "loadWeights on a weightless dir (incomplete download) errors clearly, not empty map" {
    // Reproduces the live misdiagnosis: an interrupted `hf download`/`mlx-serve
    // pull` lands config + tokenizer but never finalizes the *.safetensors
    // weight shards. Before the guard, loadWeights returned an empty map and
    // the caller crashed with a misleading "MISSING WEIGHT:
    // model.embed_tokens.weight" (the first weight looked up) + `unreachable`,
    // pointing at the model arch instead of the incomplete checkpoint.
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"mistral\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    // The index file names the shards but is NOT itself a weight file — it must
    // not be mistaken for one (it ends in .json, not .safetensors).
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{}" });

    try std.testing.expectError(
        error.NoWeightFiles,
        loadWeightsFromOpenDir(io, allocator, tmp.dir, "/incomplete-model", .{}),
    );
}

test "loadWeights reads only the shards the index names (issue #274)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // One real (hand-built) shard + one garbage file that mlx would abort on.
    const hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var st: [8 + hdr.len + 4]u8 = undefined;
    std.mem.writeInt(u64, st[0..8], hdr.len, .little);
    @memcpy(st[8 .. 8 + hdr.len], hdr);
    @memset(st[8 + hdr.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001.safetensors", .data = &st });
    try tmp.dir.writeFile(io, .{ .sub_path = "stray.safetensors", .data = "not a safetensors file" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model-00001.safetensors\"}}" });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir);
    var w = try loadWeightsFromOpenDir(io, allocator, tmp.dir, dir, .{});
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 1), w.count());
}

test "loadWeights ignores an index that names no shard on disk (re-sharded upload, stale index)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var st: [8 + hdr.len + 4]u8 = undefined;
    std.mem.writeInt(u64, st[0..8], hdr.len, .little);
    @memcpy(st[8 .. 8 + hdr.len], hdr);
    @memset(st[8 + hdr.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = &st });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model-00001-of-00005.safetensors\"}}" });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir);
    var w = try loadWeightsFromOpenDir(io, allocator, tmp.dir, dir, .{});
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 1), w.count());
}

test "resolveWeightPrefix: the CHECKPOINT decides the nesting, not the config keys" {
    // mlx-community/LFM2.5-2.6B-{8bit,nvfp4} declare `Lfm2ForCausalLM` with NO
    // text_config (just an empty `vision_config`), yet ship every weight under
    // `language_model.model.*`. The config-key guess picked "model" and the
    // load died on `MISSING WEIGHT: model.embed_tokens.weight` (live
    // 2026-08-04). The same class shipped in the opposite direction before, so
    // the probe corrects either way.
    const allocator = testing.allocator;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // Nested checkpoint, flat guess → re-pointed (the LFM2.5 crash).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.layers.0.self_attn.q_proj.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // Flat checkpoint, nested guess → re-pointed the other way.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model", config.weight_prefix);
    }
    // Both present (a real VL checkpoint) → the configured prefix stands, so
    // nothing that loads today can be re-pointed by this probe.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // An arch with its OWN prefix is never touched, even when it holds nothing
    // (a genuinely broken checkpoint must stay a clear MISSING WEIGHT).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "nemotron_h", .weight_prefix = "backbone" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("backbone", config.weight_prefix);
    }
    // A prefix that is a strict PREFIX of the key's first segment must not
    // count as a hit ("model" vs "model_extra.*").
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model_extra.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // mlx-community/Muse-Glimmer-30B-4bit (live 2026-08-11): meta's config
    // keeps text_config, so the guess is the VL-original "model.language_model"
    // — but mlx_lm convert re-nests every text weight under
    // "language_model.model.*". The third spelling joins the probe.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        try put(&w, allocator, "language_model.lm_head.weight");
        try put(&w, allocator, "vision_tower.layers.0.norm1.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "model.language_model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // Our own mirror layout (meta-original nesting) stays put.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.language_model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "model.language_model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    }
    // Ordering: a "model.language_model.*" checkpoint ALSO matches the bare
    // "model" probe (the '.' check passes at "model.language_model"), so the
    // most specific spelling must win the scan.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.language_model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    }
}

test "applyDsv4ReferenceSampling: the source's wild signature resolves to the reference's temp 0.6" {
    // DeepSeek-V4 releases ship generation_config.json with temp 1.0/top_p
    // 1.0 — the wild signature their own inference/generate.py IGNORES (its
    // default is 0.6, which our converter writes into our mirrors). External
    // conversions (pipenetwork REAP) copy the file verbatim; pi (omits
    // temperature) against REAP37 degenerated into token loops on its first
    // turn (live 2026-08-01). The exact untouched signature resolves to the
    // reference default; anything an author actually tuned is untouched.
    var wild = ModelConfig{ .model_type = "deepseek_v4" };
    wild.gen_temperature = 1.0;
    wild.gen_top_p = 1.0;
    wild.applyDsv4ReferenceSampling();
    try testing.expectEqual(@as(?f32, 0.6), wild.gen_temperature);
    try testing.expectEqual(@as(?f32, 1.0), wild.gen_top_p);

    // A tuned config is not the signature — untouched.
    var tuned = ModelConfig{ .model_type = "deepseek_v4" };
    tuned.gen_temperature = 1.0;
    tuned.gen_top_p = 0.9;
    tuned.applyDsv4ReferenceSampling();
    try testing.expectEqual(@as(?f32, 1.0), tuned.gen_temperature);

    // Other archs never touched, even with the signature values.
    var other = ModelConfig{ .model_type = "llama" };
    other.gen_temperature = 1.0;
    other.gen_top_p = 1.0;
    other.applyDsv4ReferenceSampling();
    try testing.expectEqual(@as(?f32, 1.0), other.gen_temperature);

    // No generation_config at all (both null) — nothing to resolve.
    var bare = ModelConfig{ .model_type = "deepseek_v4" };
    bare.applyDsv4ReferenceSampling();
    try testing.expectEqual(@as(?f32, null), bare.gen_temperature);
}

test "applyFamilySamplingDefaults: qwen family gets top_k 20 / top_p 0.95 when the checkpoint ships no generation_config" {
    // Live soak capture 2026-07-13 (stamsam Qwen3.6-35B distill, served to pi):
    // the community re-quant ships NO generation_config.json, so omitted-field
    // sampling resolution bottomed out at the hardcoded 1.0/1.0/off — full
    // untruncated tail sampling on a 4-bit MoE. A 16K-token turn degenerated
    // into word salad (zero tool calls) and burned the client's whole output
    // budget. Qwen's own recommendation for the family is top_k 20/top_p 0.95;
    // fill exactly the truncation knobs, never temperature.
    var qwen = ModelConfig{ .model_type = "qwen3_5_moe" };
    qwen.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 20), qwen.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.95), qwen.gen_top_p);
    try testing.expectEqual(@as(?f32, null), qwen.gen_temperature);

    var gemma = ModelConfig{ .model_type = "gemma4" };
    gemma.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 64), gemma.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.95), gemma.gen_top_p);

    // Families without a documented upstream recommendation stay null.
    var llama = ModelConfig{ .model_type = "llama" };
    llama.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, null), llama.gen_top_k);
    try testing.expectEqual(@as(?f32, null), llama.gen_top_p);

    // Inkling ships no generation_config.json anywhere and TM publishes no
    // sampling recommendation (their own tooling is greedy-only), so top_p
    // 0.95 is OUR tail cut — the first real pi agent session (2026-07-30) ran
    // wild-sampled at 1.0/1.0/off and degenerated into duplicate calls.
    var inkling = ModelConfig{ .model_type = "inkling_mm_model" };
    inkling.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, null), inkling.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.95), inkling.gen_top_p);
    try testing.expectEqual(@as(?f32, null), inkling.gen_temperature);
}

test "applyFamilySamplingDefaults never overrides explicit generation_config values" {
    // The checkpoint's own generation_config.json (parsed before this runs)
    // always wins — the family fallback fills NULLS only.
    var config = ModelConfig{ .model_type = "qwen3_5_moe" };
    config.gen_top_k = 40;
    config.gen_top_p = 0.8;
    config.gen_temperature = 0.6;
    config.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 40), config.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.8), config.gen_top_p);
    try testing.expectEqual(@as(?f32, 0.6), config.gen_temperature);
    // Partial file: only the missing knob is filled.
    var partial = ModelConfig{ .model_type = "qwen3" };
    partial.gen_top_p = 0.8;
    partial.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 20), partial.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.8), partial.gen_top_p);
}

test "defaultEnableThinking: opt-in per arch, and every existing arch stays off" {
    // No prior arch opts in — including the families whose templates merely
    // MENTION enable_thinking, which is not evidence of a thinking-on default.
    for ([_][]const u8{ "qwen3", "qwen3_5_moe", "gemma4", "gemma3_text", "laguna", "deepseek_v4", "inkling_mm_model", "llama", "hy_v3", "lfm2" }) |t| {
        const c = ModelConfig{ .model_type = t };
        try testing.expect(!c.defaultEnableThinking(false));
        try testing.expect(!c.defaultEnableThinking(true));
    }
    // muse_glimmer opts in only WITH tools: recipient selection is where its
    // reasoning earns its keep. A plain chat request defaults to the
    // prompt-committed to=user channel (chat.noThinkTailSuffix) — a real
    // skip, so there is nothing to deliver or drop.
    const muse = ModelConfig{ .model_type = "muse_glimmer" };
    try testing.expect(!muse.defaultEnableThinking(false));
    try testing.expect(muse.defaultEnableThinking(true));
    // bailing_hybrid opts IN on BOTH arms: Ling 3.0 ships as a reasoner and
    // its template normalizes an undefined enable_thinking to 'on' with no
    // reference to tools. Gating it on has_tools (as muse is) contradicted the
    // checkpoint and left a tool-less silent request answering without
    // reasoning — muse can do that because its prompt commits a to=user
    // channel, and this arch has no such fallback.
    const ling = ModelConfig{ .model_type = "bailing_hybrid" };
    try testing.expect(ling.defaultEnableThinking(false));
    try testing.expect(ling.defaultEnableThinking(true));
    // k2_horizon: the template opens a think marker on every assistant turn
    // and the pack declares no default; a declared off still wins.
    const k2 = ModelConfig{ .model_type = "k2_horizon" };
    try testing.expect(k2.defaultEnableThinking(false));
    try testing.expect(k2.defaultEnableThinking(true));
    const k2_off = ModelConfig{ .model_type = "k2_horizon", .gen_enable_thinking = false };
    try testing.expect(!k2_off.defaultEnableThinking(false));
    // gpt_oss opts in with AND without tools. Unlike muse there is no
    // thinking-off prompt to commit: harmony's `Reasoning: low|medium|high`
    // sets depth, not presence, so the model opens an analysis channel on
    // every turn. Defaulting a silent request off did not skip the reasoning
    // pass — it sent the analysis channel down the flush-text streaming
    // branch, leaking `<|channel|>analysis` and the reasoning into content.
    const goss = ModelConfig{ .model_type = "gpt_oss" };
    try testing.expect(goss.defaultEnableThinking(false));
    try testing.expect(goss.defaultEnableThinking(true));
    // deepseek_v41: DeepSeek's own template defaults `thinking_mode` to thinking.
    const v41 = ModelConfig{ .model_type = "deepseek_v41" };
    try testing.expect(v41.defaultEnableThinking(false));
    try testing.expect(v41.defaultEnableThinking(true));
}

test "defaultEnableThinking: the checkpoint's own generation_config default outranks the arch allowlist" {
    // A thinking model whose arch is not on the allowlist still thinks when
    // its own generation_config declares it — this is the case a silent
    // request used to lose (3 tokens and no reasoning where the same weights
    // reason for ~1000 tokens under a runner that obeys the template).
    var on = ModelConfig{ .model_type = "qwen3" };
    on.gen_enable_thinking = true;
    try testing.expect(on.defaultEnableThinking(false));
    try testing.expect(on.defaultEnableThinking(true));
    // And a checkpoint that declares thinking OFF turns an opted-in arch off.
    var off = ModelConfig{ .model_type = "bailing_hybrid" };
    off.gen_enable_thinking = false;
    try testing.expect(!off.defaultEnableThinking(false));
    try testing.expect(!off.defaultEnableThinking(true));
}

test "parseGenerationDefaultsFromJson: reads default_chat_template_kwargs.enable_thinking" {
    const on = parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": true}}",
    );
    try testing.expectEqual(@as(?bool, true), on.enable_thinking);
    const off = parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": false}}",
    );
    try testing.expectEqual(@as(?bool, false), off.enable_thinking);
    // Absent, wrong shape, or a non-bool value: null, arch default stays.
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson("{\"top_k\": 20}").enable_thinking);
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": \"on\"}",
    ).enable_thinking);
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": \"yes\"}}",
    ).enable_thinking);
}

test "parseGenerationDefaultsFromJson: eos_token_id list merges additively into the stop set" {
    const gd = parseGenerationDefaultsFromJson("{\"eos_token_id\": [1, 250019]}");
    try testing.expectEqual(@as(usize, 2), gd.num_eos);
    var config = ModelConfig{};
    config.addEosToken(1);
    config.mergeEosTokens(gd.eos_token_ids[0..gd.num_eos]);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expect(config.isEosToken(250019));
    const scalar = parseGenerationDefaultsFromJson("{\"eos_token_id\": 7}");
    try testing.expectEqual(@as(u32, 7), scalar.eos_token_ids[0]);
    try testing.expectEqual(@as(usize, 0), parseGenerationDefaultsFromJson("{\"eos_token_id\": \"x\"}").num_eos);
}

test "ModelConfig addEosToken" {
    var config = ModelConfig{};
    config.addEosToken(1);
    config.addEosToken(106);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expect(config.isEosToken(1));
    try testing.expect(config.isEosToken(106));
    try testing.expect(!config.isEosToken(42));
}

test "ModelConfig addEosToken max capacity" {
    var config = ModelConfig{};
    // Fill all 8 slots
    for (0..8) |i| {
        config.addEosToken(@intCast(i + 100));
    }
    try testing.expectEqual(@as(u32, 8), config.num_eos_tokens);
    // 9th should be silently dropped
    config.addEosToken(999);
    try testing.expectEqual(@as(u32, 8), config.num_eos_tokens);
    try testing.expect(!config.isEosToken(999));
}

test "EOS merge: chat-terminator added even when config already provided an eos" {
    // Regression for the Qwen2.5-Coder-7B leak: its config.json sets
    // eos_token_id=<|endoftext|> (151643), but its chat template ends turns
    // with <|im_end|> (151645). The load path (main.zig / scheduler doLoad)
    // must ALWAYS merge the tokenizer's chat-terminator EOS — additively and
    // dedup-guarded — not only when config provided none; otherwise <|im_end|>
    // is never a stop token and leaks into the output (broke structured JSON /
    // tool calling). This pins the merge invariant those call sites implement.
    var config = ModelConfig{};
    config.addEosToken(151643); // from config.json eos_token_id
    try testing.expectEqual(@as(u32, 1), config.num_eos_tokens);

    // Merge step the fix performs: add the chat terminator if absent.
    const chat_eos: u32 = 151645; // <|im_end|>, from tokenizer_config eos_token
    if (!config.isEosToken(chat_eos)) config.addEosToken(chat_eos);

    try testing.expect(config.isEosToken(151645)); // now stops on <|im_end|>
    try testing.expect(config.isEosToken(151643)); // original preserved
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);

    // Idempotent: re-running the merge must not duplicate.
    if (!config.isEosToken(chat_eos)) config.addEosToken(chat_eos);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
}

test "ModelConfig eosTokenSlice" {
    var config = ModelConfig{};
    config.addEosToken(10);
    config.addEosToken(20);
    const slice = config.eosTokenSlice();
    try testing.expectEqual(@as(usize, 2), slice.len);
    try testing.expectEqual(@as(u32, 10), slice[0]);
    try testing.expectEqual(@as(u32, 20), slice[1]);
}

test "ModelConfig isGlobalLayer with sliding window" {
    // Gemma 3 convention (HF + mlx-lm): every Nth layer is global, with the
    // pattern anchored at the END of each group — global when
    // `(idx + 1) % pattern == 0`, i.e. layers 5, 11, 17… for pattern 6.
    // The old `% pattern == 0` phase made layer 0 global and layer 5 local —
    // every layer got the wrong RoPE base/scale and attention scope, which
    // surfaced as fluent-but-wrong output (spaced digits, broken arithmetic)
    // on gemma-3-12b. Gemma 4 ships explicit layer_types and never hits this
    // fallback.
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.sliding_window_pattern = 6;
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(1));
    try testing.expect(config.isGlobalLayer(5));
    try testing.expect(!config.isGlobalLayer(6));
    try testing.expect(config.isGlobalLayer(11));
    try testing.expect(!config.isGlobalLayer(12));
}

test "ModelConfig isGlobalLayer without sliding window" {
    var config = ModelConfig{};
    config.has_sliding_window = false;
    // All layers should be global
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(1));
    try testing.expect(config.isGlobalLayer(5));
}

test "ModelConfig isLinearLayer" {
    var config = ModelConfig{};
    config.full_attention_interval = 4;
    // Layer 0: (0+1) % 4 == 1 != 0 → linear
    try testing.expect(config.isLinearLayer(0));
    // Layer 3: (3+1) % 4 == 0 → NOT linear (full attention)
    try testing.expect(!config.isLinearLayer(3));
    // Layer 7: (7+1) % 4 == 0 → NOT linear
    try testing.expect(!config.isLinearLayer(7));
    // Layer 4: (4+1) % 4 == 1 → linear
    try testing.expect(config.isLinearLayer(4));
}

test "linear_attn_tail_from forces full attention past the last whole group" {
    // A layer count that is NOT a multiple of the group size is where the
    // reference's second clause bites: with 40 layers, 40//6*6 = 36, so layers
    // 36..39 are ALL full attention even though (idx+1) % 6 != 0. Dropping the
    // clause would run four layers through the wrong attention type silently.
    var config = ModelConfig{};
    config.num_hidden_layers = 40;
    config.full_attention_interval = 6;
    config.linear_attn_tail_from = 40 / 6 * 6; // 36
    try testing.expect(config.isLinearLayer(34));
    try testing.expect(!config.isLinearLayer(35)); // (35+1) % 6 == 0
    try testing.expect(!config.isLinearLayer(36)); // tail clause
    try testing.expect(!config.isLinearLayer(37));
    try testing.expect(!config.isLinearLayer(38));
    try testing.expect(!config.isLinearLayer(39));
}

test "linear_attn_tail_from is off by default so no existing arch moves" {
    // qwen3_next/lfm2 set full_attention_interval without a tail bound.
    var config = ModelConfig{};
    config.full_attention_interval = 4;
    try testing.expectEqual(@as(u32, 0), config.linear_attn_tail_from);
    try testing.expect(config.isLinearLayer(100));
    try testing.expect(config.isLinearLayer(1000));
}

test "ModelConfig isLinearLayer disabled" {
    var config = ModelConfig{};
    config.full_attention_interval = 0;
    try testing.expect(!config.isLinearLayer(0));
    try testing.expect(!config.isLinearLayer(5));
}

test "ModelConfig isMoe" {
    var config = ModelConfig{};
    try testing.expect(!config.isMoe());
    config.num_experts = 8;
    try testing.expect(config.isMoe());
}

test "jsonFloat converts integer" {
    const val = std.json.Value{ .integer = 42 };
    try testing.expectApproxEqAbs(@as(f32, 42.0), try jsonFloat(val), 0.001);
}

test "jsonFloat converts float" {
    const val = std.json.Value{ .float = 3.14 };
    try testing.expectApproxEqAbs(@as(f32, 3.14), try jsonFloat(val), 0.01);
}

test "ModelConfig isGlobalLayer with explicit layer_types" {
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    // Set layer 4 and 9 as global (like Gemma 4 E2B pattern)
    config.layer_is_global[4] = true;
    config.layer_is_global[9] = true;
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(3));
    try testing.expect(config.isGlobalLayer(4));
    try testing.expect(!config.isGlobalLayer(5));
    try testing.expect(config.isGlobalLayer(9));
}

test "ModelConfig getKVSourceLayer" {
    var config = ModelConfig{};
    config.num_hidden_layers = 35;
    config.num_kv_shared_layers = 20;
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    // E2B pattern: every 5th layer starting from 4 is global
    for (0..35) |i| {
        config.layer_is_global[i] = (i % 5 == 4);
    }
    // Layers 0-14 are concrete (no source)
    try testing.expect(config.getKVSourceLayer(0) == null);
    try testing.expect(config.getKVSourceLayer(14) == null);
    // Layer 15 (sliding) -> should map to layer 13 (last concrete sliding)
    try testing.expectEqual(@as(?u32, 13), config.getKVSourceLayer(15));
    // Layer 19 (full) -> should map to layer 14 (last concrete full)
    try testing.expectEqual(@as(?u32, 14), config.getKVSourceLayer(19));
    // Layer 20 (sliding) -> should also map to layer 13
    try testing.expectEqual(@as(?u32, 13), config.getKVSourceLayer(20));
}

test "ModelConfig layerHeadDim" {
    var config = ModelConfig{};
    config.head_dim = 256;
    config.global_head_dim = 512;
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    config.layer_is_global[4] = true;
    try testing.expectEqual(@as(u32, 256), config.layerHeadDim(0));
    try testing.expectEqual(@as(u32, 512), config.layerHeadDim(4));
}

test "ModelConfig BERT defaults" {
    var config = ModelConfig{};
    config.is_encoder_only = true;
    config.model_type = "bert";
    config.hidden_size = 384;
    config.num_attention_heads = 12;
    config.head_dim = 384 / 12;
    config.num_key_value_heads = 12;

    try testing.expect(config.is_encoder_only);
    try testing.expectEqual(@as(u32, 32), config.head_dim);
    try testing.expectEqual(@as(u32, 12), config.num_key_value_heads);
    try testing.expectApproxEqAbs(@as(f32, 1e-12), config.layer_norm_eps, 1e-15);
}

test "ModelConfig BERT is not MoE" {
    var config = ModelConfig{};
    config.is_encoder_only = true;
    config.model_type = "bert";
    try testing.expect(!config.isMoe());
}

test "ModelConfig BERT has no sliding window" {
    var config = ModelConfig{};
    config.is_encoder_only = true;
    config.has_sliding_window = false;
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(5));
}

test "shouldKeepWeightKey accepts orphan MTP head weights on Qwen3.5/3.6 checkpoints" {
    // Some Qwen3.5/3.6 checkpoints embed `*.mtp.*` tensors in the MAIN
    // shards (the sidecar-based MTP head in src/mtp.zig loads separately).
    // The safetensors iterator must let them through (they're neither vision
    // nor audio) so the model loads cleanly; the trunk binder ignores them.
    try testing.expect(shouldKeepWeightKey("language_model.model.mtp.0.eh_proj.weight", true));
    try testing.expect(shouldKeepWeightKey("language_model.model.mtp.0.eh_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.mtp.0.shared_head.head.weight", false));
}

test "shouldKeepWeightKey filters audio and gated vision weights" {
    // Regression: the existing filter should still reject audio and reject
    // vision when load_vision is false.
    try testing.expect(!shouldKeepWeightKey("audio_tower.encoder.layer.0.weight", true));
    try testing.expect(!shouldKeepWeightKey("vision_tower.encoder.layer.0.weight", false));
    // qwen4_exp / Alis packs spell the Qwen3-VL tower `model.visual.` — --no-vision drops it too.
    try testing.expect(!shouldKeepWeightKey("model.visual.blocks.0.attn.qkv.weight", false));
    try testing.expect(shouldKeepWeightKey("model.visual.blocks.0.attn.qkv.weight", true));
    try testing.expect(shouldKeepWeightKey("vision_tower.encoder.layer.0.weight", true));
    try testing.expect(shouldKeepWeightKey("language_model.model.layers.0.self_attn.q_proj.weight", false));
}

test "shouldKeepWeightKey keeps Gemma 4 12B unified embedder weights when vision enabled" {
    // gemma4_unified is encoder-free: vision_embedder.* (patch embedder),
    // embed_vision.* and embed_audio.* (raw projections) ARE wired in
    // src/vision.zig (UnifiedEmbedder), so they must be kept under load_vision.
    // The heavy SigLIP-era conformer audio_tower.* stays dropped.
    try testing.expect(shouldKeepWeightKey("vision_embedder.patch_dense.weight", true));
    try testing.expect(shouldKeepWeightKey("embed_vision.embedding_projection.weight", true));
    try testing.expect(shouldKeepWeightKey("embed_audio.embedding_projection.weight", true));
    // Gated off by --no-vision.
    try testing.expect(!shouldKeepWeightKey("vision_embedder.patch_dense.weight", false));
    try testing.expect(!shouldKeepWeightKey("embed_audio.embedding_projection.weight", false));
    // The conformer audio tower is never wired — always dropped.
    try testing.expect(!shouldKeepWeightKey("audio_tower.encoder.layer.0.weight", true));
}

test "shouldKeepWeightKey gates Muse-Glimmer vision on load_vision in both nestings" {
    // Ours nests the tower under `model.`; mlx-community re-nests it bare.
    // Both spellings ride the same gate --no-vision flips.
    try testing.expect(!shouldKeepWeightKey("model.vision_tower.layers.0.norm1.weight", false));
    try testing.expect(!shouldKeepWeightKey("model.vision_adapter.fc1.weight", false));
    try testing.expect(!shouldKeepWeightKey("model.vision_projection.weight", false));
    try testing.expect(!shouldKeepWeightKey("vision_adapter.fc1.weight", false));
    try testing.expect(!shouldKeepWeightKey("vision_tower.layers.0.norm1.weight", false));
    try testing.expect(shouldKeepWeightKey("model.vision_tower.layers.0.norm1.weight", true));
    try testing.expect(shouldKeepWeightKey("model.vision_adapter.fc1.weight", true));
    try testing.expect(shouldKeepWeightKey("model.vision_projection.weight", true));
    try testing.expect(shouldKeepWeightKey("vision_adapter.fc1.weight", true));
    try testing.expect(shouldKeepWeightKey("vision_tower.layers.0.norm1.weight", true));
    // avlp12 Alis spells the Qwen3-VL tower `model.visual.` — same gate, or
    // --no-vision cannot drop it and we hold ~0.9 GB we never read.
    try testing.expect(!shouldKeepWeightKey("model.visual.blocks.0.norm1.weight", false));
    try testing.expect(shouldKeepWeightKey("model.visual.blocks.0.norm1.weight", true));
    // Text weights are never touched either way.
    try testing.expect(shouldKeepWeightKey("model.language_model.embed_tokens.weight", false));
    try testing.expect(shouldKeepWeightKey("language_model.model.embed_tokens.weight", false));
    try testing.expect(shouldKeepWeightKey("language_model.lm_head.weight", false));
}

test "ModelConfig parses gemma4_unified text_config" {
    // Gemma 4 12B base ships `model_type: gemma4_unified` with text_config
    // carrying the language tower. The dispatch arm must:
    //   - tag as gemma4_unified
    //   - inherit the gemma4 weight prefix + norm/scale flags
    //   - pass attention_k_eq_v through unchanged so the per-layer binder
    //     (transformer.zig:5677) aliases V to K on global layers but uses
    //     the shipped v_proj on sliding layers
    //   - pass through gemma4 fields like global_head_dim, final_logit_softcapping.
    const json =
        \\{
        \\  "model_type": "gemma4_unified",
        \\  "text_config": {
        \\    "model_type": "gemma4_unified_text",
        \\    "hidden_size": 3840,
        \\    "intermediate_size": 15360,
        \\    "num_hidden_layers": 48,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "head_dim": 256,
        \\    "global_head_dim": 512,
        \\    "num_global_key_value_heads": 8,
        \\    "num_kv_shared_layers": 0,
        \\    "hidden_size_per_layer_input": 0,
        \\    "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        \\    "rope_parameters": {
        \\      "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "factor": 1.0},
        \\      "sliding_attention": {"rope_theta": 10000.0}
        \\    },
        \\    "attention_k_eq_v": true,
        \\    "final_logit_softcapping": 30.0,
        \\    "hidden_activation": "gelu_pytorch_tanh",
        \\    "rms_norm_eps": 1e-06,
        \\    "max_position_embeddings": 8192,
        \\    "sliding_window": 1024
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    // Collapsed onto "gemma4" so downstream code paths (attn_scale gate,
    // recommendedBlockSize) match the 31B Dense decoder it inherits.
    try testing.expectEqualStrings("gemma4", config.model_type);
    try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    try testing.expect(config.has_v_norm);
    try testing.expect(config.has_pre_ff_norm);
    try testing.expect(config.has_qk_norm);
    try testing.expect(!config.norm_has_offset);
    try testing.expect(config.scale_embeddings);
    // 12B's k_proj/v_proj layout is mixed: full_attention layers omit v_proj
    // (K=V alias), sliding layers ship it. The existing per-layer binder
    // (transformer.zig:5677) keys the alias on isGlobalLayer; we must keep
    // the config flag intact so that AND-clause fires on global layers only.
    try testing.expect(config.attention_k_eq_v);
    try testing.expectEqual(@as(u32, 512), config.global_head_dim);
    try testing.expectEqual(@as(u32, 0), config.hidden_size_per_layer_input);
    try testing.expectApproxEqAbs(@as(f32, 30.0), config.final_logit_softcapping, 0.001);
    try testing.expectEqual(@as(u32, 3840), config.hidden_size);
    try testing.expectEqual(@as(u32, 48), config.num_hidden_layers);
    // Unified flag drives the encoder-free vision/audio embedder path.
    try testing.expect(config.is_gemma4_unified);
}

test "ModelConfig parses kolibri1: logit-bias routing, NoPE on full layers, sandwich-norm MoE" {
    // Trimmed copy of Kolibri-1's config.json: 4 layers keep the full/sliding mix.
    const json =
        \\{
        \\  "model_type": "kolibri1",
        \\  "hidden_size": 2560,
        \\  "num_hidden_layers": 4,
        \\  "num_attention_heads": 48,
        \\  "num_key_value_heads": 4,
        \\  "head_dim": 128,
        \\  "rms_norm_eps": 1e-06,
        \\  "vocab_size": 128000,
        \\  "max_position_embeddings": 262144,
        \\  "rope_theta": 10000.0,
        \\  "tie_word_embeddings": false,
        \\  "use_sliding_window": true,
        \\  "sliding_window": 513,
        \\  "num_experts": 384,
        \\  "num_experts_per_tok": 6,
        \\  "moe_intermediate_size": 512,
        \\  "shared_expert_intermediate_size": 512,
        \\  "norm_topk_prob": false,
        \\  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        \\  "quantization": {"group_size": 64, "bits": 3, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("kolibri1", config.model_type);
    try testing.expect(config.moe_sigmoid_router and config.moe_logit_select);
    try testing.expect(!config.moe_route_norm);
    try testing.expectEqual(@as(u32, 384), config.num_experts);
    try testing.expectEqual(@as(u32, 6), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 513), config.sliding_window);
    // Only the full-attention layer is NoPE; sliding layers rotate at rope_theta.
    try testing.expect(!config.layerSkipsRope(0));
    try testing.expect(config.layerSkipsRope(3));
    try testing.expect(config.isGlobalLayer(3) and !config.isGlobalLayer(0));
    try testing.expectApproxEqAbs(@as(f32, 10000.0), config.rope_local_base_freq, 1e-3);
    try testing.expectEqual(@as(u32, 128), config.query_pre_attn_scalar);
}

test "ModelConfig parses laguna (poolside Laguna-S-2.1): per-layer heads, softplus gate, YaRN, sigmoid MoE" {
    // Trimmed but faithful copy of poolside/Laguna-S-2.1-NVFP4-mlx config.json.
    // 4 layers = one full/sliding group (full@0, sliding@1..3) so per-layer
    // head counts and layer_types both exercise a global/local boundary.
    const json =
        \\{
        \\  "model_type": "laguna",
        \\  "hidden_size": 3072,
        \\  "intermediate_size": 12288,
        \\  "num_hidden_layers": 4,
        \\  "num_attention_heads": 48,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "rms_norm_eps": 1e-06,
        \\  "vocab_size": 100352,
        \\  "max_position_embeddings": 262144,
        \\  "tie_word_embeddings": false,
        \\  "eos_token_id": [2, 24],
        \\  "bos_token_id": 2,
        \\  "gating": "per-head",
        \\  "sliding_window": 512,
        \\  "num_experts": 256,
        \\  "num_experts_per_tok": 10,
        \\  "moe_intermediate_size": 1024,
        \\  "shared_expert_intermediate_size": 1024,
        \\  "moe_routed_scaling_factor": 2.5,
        \\  "norm_topk_prob": true,
        \\  "moe_router_logit_softcapping": 0.0,
        \\  "mlp_only_layers": [0],
        \\  "num_attention_heads_per_layer": [48, 72, 72, 72],
        \\  "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        \\  "rope_parameters": {
        \\    "full_attention": {
        \\      "rope_theta": 500000.0, "rope_type": "yarn", "factor": 32.0,
        \\      "original_max_position_embeddings": 8192, "beta_slow": 1.0, "beta_fast": 32.0,
        \\      "attention_factor": 1.3465735902799727, "partial_rotary_factor": 0.5
        \\    },
        \\    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}
        \\  },
        \\  "quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("laguna", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    // Qwen/hy3-style norm + embedding flags.
    try testing.expect(!config.norm_has_offset);
    try testing.expect(!config.scale_embeddings);
    try testing.expect(!config.has_pre_ff_norm);
    try testing.expect(config.has_qk_norm);
    // Laguna-specific attention gate + sigmoid router.
    try testing.expect(config.laguna_attn_gate);
    try testing.expect(config.moe_sigmoid_router);
    try testing.expect(config.moe_route_norm);
    try testing.expectApproxEqAbs(@as(f32, 2.5), config.router_scaling_factor, 1e-6);
    // MoE dims.
    try testing.expectEqual(@as(u32, 256), config.num_experts);
    try testing.expectEqual(@as(u32, 10), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 1024), config.moe_intermediate_size);
    try testing.expectEqual(@as(u32, 1024), config.shared_expert_intermediate_size);
    // Attention shape + scale (query_pre_attn_scalar defaults to head_dim).
    try testing.expectEqual(@as(u32, 128), config.head_dim);
    try testing.expectEqual(@as(u32, 8), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 128), config.query_pre_attn_scalar);
    // Per-layer Q-heads: 48 on full (layer 0), 72 on sliding (layer 1+).
    try testing.expect(config.has_per_layer_heads);
    try testing.expectEqual(@as(u32, 48), config.layerNumHeads(0));
    try testing.expectEqual(@as(u32, 72), config.layerNumHeads(1));
    // Layer types: full@0 = global, sliding@1 = local.
    try testing.expect(config.has_explicit_layer_types);
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(1));
    try testing.expect(config.has_sliding_window);
    try testing.expectEqual(@as(u32, 512), config.sliding_window);
    // RoPE: full-attn YaRN (theta 5e5, partial 0.5) + sliding default (theta 1e4, full rotary).
    try testing.expectApproxEqAbs(@as(f32, 500000.0), config.rope_theta, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), config.partial_rotary_factor_global, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 10000.0), config.rope_local_base_freq, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), config.partial_rotary_factor, 1e-6);
    try testing.expect(config.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 32.0), config.yarn_factor, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 32.0), config.yarn_beta_fast, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), config.yarn_beta_slow, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.3465735902799727), config.yarn_attention_factor, 1e-9);
    try testing.expectEqual(@as(u32, 8192), config.yarn_orig_max_pos);
    // Quant: nvfp4, gs16.
    try testing.expectEqual(QuantMode.nvfp4, config.quant_mode);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 16), config.quant_group_size);
    // EOS pair (〈|EOS|〉=2, </assistant>=24).
    const eos = config.eosTokenSlice();
    try testing.expectEqual(@as(usize, 2), eos.len);
    try testing.expectEqual(@as(u32, 2), eos[0]);
    try testing.expectEqual(@as(u32, 24), eos[1]);
}

/// MiMo-V2.6-Flash in the mlx-lm layout (`tests/convert_mimo_v2.py`, mlx-community's
/// mxfp4-q8), layers trimmed to 6.
const mimo_v2_pack_json =
    \\{
    \\  "model_type": "mimo_v2",
    \\  "add_full_attention_sink_bias": false, "add_swa_attention_sink_bias": true,
    \\  "attention_projection_layout": "fused_qkv", "attention_value_scale": 0.707,
    \\  "eos_token_id": 151645, "head_dim": 192, "hidden_size": 4096,
    \\  "hybrid_layer_pattern": [0, 1, 1, 1, 1, 0],
    \\  "intermediate_size": 16384, "layernorm_epsilon": 1e-06,
    \\  "max_position_embeddings": 1048576, "moe_intermediate_size": 2048,
    \\  "moe_layer_freq": [0, 1, 1, 1, 1, 1],
    \\  "n_group": 1, "n_routed_experts": 256, "n_shared_experts": null, "norm_topk_prob": true,
    \\  "num_attention_heads": 64, "num_experts_per_tok": 8, "num_hidden_layers": 6,
    \\  "num_key_value_heads": 4, "num_nextn_predict_layers": 3, "partial_rotary_factor": 0.334,
    \\  "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default"},
    \\  "rope_theta": 10000000.0, "routed_scaling_factor": null, "scoring_func": "sigmoid",
    \\  "sliding_window": 128, "swa_head_dim": 192, "swa_num_attention_heads": 64,
    \\  "swa_num_key_value_heads": 8, "swa_rope_theta": 10000.0, "swa_v_head_dim": 128,
    \\  "tie_word_embeddings": false, "topk_group": 1, "topk_method": "noaux_tc",
    \\  "v_head_dim": 128, "vocab_size": 152576,
    \\  "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"},
    \\  "quantization_config": {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    \\}
;

test "ModelConfig: mimo_v2 pack config parse" {
    const config = try parseConfigFromJson(testing.allocator, mimo_v2_pack_json);
    try testing.expectEqualStrings("mimo_v2", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expect(!config.norm_has_offset and !config.scale_embeddings and !config.has_pre_ff_norm and !config.has_qk_norm);
    try testing.expectEqual(QuantMode.mxfp4, config.quant_mode);
    // Experts: sigmoid scores, selection-only bias, renormalized top-8; layer 0 dense.
    try testing.expectEqual(@as(u32, 256), config.num_experts);
    try testing.expectEqual(@as(u32, 8), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 2048), config.moe_intermediate_size);
    try testing.expectEqual(@as(u32, 1), config.first_k_dense_replace);
    try testing.expect(config.moe_sigmoid_router and config.moe_route_norm);
    try testing.expectEqual(@as(f32, 1.0), config.router_scaling_factor);
    // Pattern 0 = global (4 KV heads, theta 1e7), 1 = sliding (8 KV heads, theta 1e4, sinks).
    try testing.expect(config.isGlobalLayer(0) and !config.isGlobalLayer(1) and config.isGlobalLayer(5));
    try testing.expectEqual(@as(u32, 4), config.layerKVHeads(0));
    try testing.expectEqual(@as(u32, 8), config.layerKVHeads(1));
    try testing.expectEqual(@as(u32, 128), config.sliding_window);
    try testing.expect(config.has_attn_sinks);
    try testing.expectEqual(@as(f32, 1e7), config.rope_theta);
    try testing.expectEqual(@as(f32, 1e4), config.rope_local_base_freq);
    // K 192 / V 128; rope covers int(192 * 0.334) = 64 dims; scale on the 192-wide key.
    try testing.expectEqual(@as(u32, 192), config.layerHeadDim(1));
    try testing.expectEqual(@as(u32, 128), config.layerVHeadDim(1));
    try testing.expectEqual(@as(u32, 64), config.layerRopeDims(0));
    try testing.expectEqual(@as(u32, 64), config.layerRopeDims(1));
    try testing.expectEqual(@as(u32, 192), config.query_pre_attn_scalar);
    // V is scaled before caching, at runtime, as mlx-lm does.
    try testing.expectEqual(@as(f32, 0.707), config.attention_value_scale);
    // KV per token: the global layers only; a sliding layer holds a bounded ring of rows.
    try testing.expectEqual(@as(u64, 2 * 4 * (192 + 128) * 2), config.kvBytesPerToken());
    try testing.expectEqual(@as(u64, 4 * 8 * (192 + 128) * 2), config.slidingRowBytes());
    try testing.expectEqual(@as(u32, 128 + ModelConfig.SLIDING_RING_SLACK), config.slidingKeepRows());
}

/// GLM-5.3-Flash (TensorFold's MLX packs), layers trimmed to 8.
const glm5_next_pack_json =
    \\{
    \\  "model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
    \\  "tie_word_embeddings": false,
    \\  "text_config": {
    \\    "model_type": "glm5_next_text", "vocab_size": 154880, "hidden_size": 4096,
    \\    "intermediate_size": 12288, "moe_intermediate_size": 2048, "num_hidden_layers": 8,
    \\    "num_nextn_predict_layers": 1, "num_attention_heads": 64, "num_key_value_heads": 64,
    \\    "n_shared_experts": 1, "n_routed_experts": 288, "routed_scaling_factor": 2.5,
    \\    "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 0, "v_head_dim": 256,
    \\    "qk_nope_head_dim": 256, "qk_head_dim": 256, "head_dim": 0, "n_group": 1, "topk_group": 1,
    \\    "num_experts_per_tok": 8, "norm_topk_prob": true, "max_position_embeddings": 1048576,
    \\    "rms_norm_eps": 1e-05, "eos_token_id": [154820, 154827, 154829], "pad_token_id": 154820,
    \\    "mlp_layer_types": ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
    \\    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
    \\                    "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
    \\    "indexer_types": ["full", "full", "full", "full", "full", "full", "full", "full"],
    \\    "index_topk": 2048, "index_head_dim": 128, "index_n_heads": 32, "index_kpool": 4,
    \\    "index_kpool_always_select_tail": true, "index_kpool_compress": true, "mla_use_nope": true,
    \\    "swiglu_limit": 10.0, "hc_mult": 4, "hc_eps": 1e-06, "hc_sinkhorn_iters": 20, "mhc": true,
    \\    "first_k_dense_replace": 3, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    \\    "moe_router_dtype": "float32",
    \\    "linear_attn_config": {"num_heads": 64, "head_dim": 128, "gate_lower_bound": -5.0,
    \\                           "short_conv_kernel_size": 4, "full_attn_layers": [3, 7]}
    \\  },
    \\  "vision_config": {"model_type": "glm5_next_vision", "depth": 24}
    \\}
;

test "ModelConfig: glm5_next binds KDA, sparse latent attention, mHC and the clamped SwiGLU" {
    var config = try parseConfigFromJson(testing.allocator, glm5_next_pack_json);
    defer config.deinit(testing.allocator);
    try testing.expect(config.isGlm5());
    // Its indexer rows carry pooled keys: an older SSD cache root never restores into it.
    try testing.expectEqualStrings("glm5-pooled-keys-v1", config.cacheLayoutNamespace().?);
    try testing.expect(!config.isMla());
    try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    // Layers 3 and 7 are sparse attention, the rest KDA.
    try testing.expect(config.isLinearLayer(0) and config.isLinearLayer(2) and !config.isLinearLayer(3) and !config.isLinearLayer(7));
    try testing.expectEqual(@as(u32, 2), config.attnCacheLayerCount());
    try testing.expectEqual(@as(u32, 64), config.linear_num_key_heads);
    try testing.expectEqual(@as(u32, 128), config.linear_key_head_dim);
    try testing.expect(config.kda_vector_gate and config.kda_sigmoid_out_gate and config.kdaUsesBoundedGate());
    try testing.expectEqual(@as(f32, -5.0), config.kda_gate_lower_bound);
    try testing.expectEqual(@as(u32, 1536), config.dsa_q_lora_rank);
    try testing.expectEqual(@as(u32, 512), config.dsa_kv_lora_rank);
    try testing.expectEqual(@as(u32, 256), config.dsa_head_dim);
    try testing.expectEqual(@as(u32, 2048), config.dsa_index_topk);
    try testing.expectEqual(@as(u32, 4), config.dsa_index_kpool);
    try testing.expectEqual(@as(u32, 32), config.dsa_index_heads);
    try testing.expectEqual(@as(u32, 128), config.dsa_index_head_dim);
    try testing.expectEqual(@as(u32, 4), config.dsv4_hc_mult);
    try testing.expectEqual(@as(u32, 20), config.dsv4_hc_sinkhorn_iters);
    try testing.expect(config.swiglu_clamp);
    try testing.expectEqual(@as(f32, 10.0), config.swiglu_limit);
    try testing.expectEqual(@as(u32, 288), config.num_experts);
    try testing.expectEqual(@as(u32, 8), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 3), config.first_k_dense_replace);
    try testing.expectEqual(@as(u32, 2048), config.shared_expert_intermediate_size);
    try testing.expectEqual(@as(f32, 2.5), config.router_scaling_factor);
    try testing.expect(config.moe_sigmoid_router and config.moe_route_norm);
    // Per sparse-attention layer per token: the 512 latent plus the indexer's key and gate.
    try testing.expectEqual(@as(u64, 2 * (512 + 128 + 128) * 2), config.kvBytesPerToken());
    try testing.expect(!config.has_vision);
    // The template opens `<think>` on every assistant turn.
    try testing.expect(config.defaultEnableThinking(false) and config.defaultEnableThinking(true));
    // The reference keeps the KDA state in f32.
    try testing.expectEqual(mlx.mlx_dtype.float32, config.ssmStateDtype());
}

test "ModelConfig: glm5_next refuses what it does not serve, by name" {
    for ([_][]const u8{ "\"shared\", \"full\"", "\"full\", \"shared\"" }) |types| {
        var buf: [256]u8 = undefined;
        const body = try std.fmt.bufPrint(&buf, "{{\"model_type\": \"glm5_next_text\", \"num_hidden_layers\": 2, \"kv_lora_rank\": 64, \"indexer_types\": [{s}]}}", .{types});
        try testing.expectError(error.UnsupportedGlm5Config, parseConfigFromJson(testing.allocator, body));
    }
    try testing.expectError(error.UnsupportedGlm5Config, parseConfigFromJson(testing.allocator, "{\"model_type\": \"glm5_next_text\", \"num_hidden_layers\": 2, \"kv_lora_rank\": 64, \"qk_rope_head_dim\": 64}"));
}

test "ModelConfig: mimo_v2 source release is refused with the converter's name" {
    // The release stores FP8 + per-expert MXFP4 (`quant_method: fp8`); mlx packs
    // carry an mlx-style quantization_config and load.
    const src = try std.mem.replaceOwned(u8, testing.allocator, mimo_v2_pack_json,
        \\"quantization_config": {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    ,
        \\"quantization_config": {"quant_method": "fp8", "store_dtype": "mxfp4"}
    );
    defer testing.allocator.free(src);
    try testing.expectError(error.UnconvertedMimoCheckpoint, parseConfigFromJson(testing.allocator, src));
}

test "ModelConfig: a Sushi pack's expert_quant selects EXL3 experts on mimo_v2 and glm5_next" {
    const eq = "{\"expert_quant\":{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14},";
    const glm = try parseConfigFromJson(testing.allocator, eq ++ glm5_next_pack_json[1..]);
    try testing.expectEqual(@as(u32, 40), glm.exl3.?.rate.n);
    try testing.expectEqual(sushi_exl3.format.Window.w14, glm.exl3.?.window);
    try testing.expect((try parseConfigFromJson(testing.allocator, glm5_next_pack_json)).exl3 == null);

    // The pack's `quant_method: fp8` names its trunk, so the release refusal does not fire.
    const fp8 = try std.mem.replaceOwned(u8, testing.allocator, mimo_v2_pack_json,
        \\"quantization_config": {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    ,
        \\"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
    );
    defer testing.allocator.free(fp8);
    const with_eq = try std.mem.concat(testing.allocator, u8, &.{ eq, fp8[1..] });
    defer testing.allocator.free(with_eq);
    const mimo = try parseConfigFromJson(testing.allocator, with_eq);
    try testing.expect(mimo.isMimo());
    try testing.expectEqual(sushi_exl3.format.Codebook.mcg, mimo.exl3.?.codebook);
    try testing.expectError(error.UnconvertedMimoCheckpoint, parseConfigFromJson(testing.allocator, fp8));
}

test "indexOwners: the index names the shard a duplicated tensor loads from, an absent shard claims nothing" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "affine.safetensors", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"lm_head.weight\":\"affine.safetensors\",\"x.weight\":\"gone.safetensors\"}}" });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const owners = indexOwners(io, arena.allocator(), tmp.dir).?;
    try testing.expectEqualStrings("affine.safetensors", owners.get("lm_head.weight").?);
    try testing.expect(owners.get("x.weight") == null);
}

test "ModelConfig: mimo_v2_flash (MiMo-V2-Flash, TensorFold/Vontra V2.6 packs) parses as mimo_v2" {
    const flash = try std.mem.replaceOwned(u8, testing.allocator, mimo_v2_pack_json, "\"model_type\": \"mimo_v2\"", "\"model_type\": \"mimo_v2_flash\"");
    defer testing.allocator.free(flash);
    const config = try parseConfigFromJson(testing.allocator, flash);
    try testing.expect(config.isMimo());
    try testing.expectEqual(@as(f32, 0.707), config.attention_value_scale);
    // MiMo-V2-Flash itself declares no V scale.
    const v2 = try std.mem.replaceOwned(u8, testing.allocator, flash, "\"attention_value_scale\": 0.707,", "");
    defer testing.allocator.free(v2);
    try testing.expectEqual(@as(f32, 1.0), (try parseConfigFromJson(testing.allocator, v2)).attention_value_scale);
}

test "ModelConfig: gpt_oss (OpenAI gpt-oss-20b) config parse" {
    // Trimmed from mlx-community/gpt-oss-20b-MXFP4-Q8/config.json. The 120B is
    // the same shape (36 layers, 128 experts), so one block serves both.
    const json =
        \\{
        \\  "architectures": ["GptOssForCausalLM"],
        \\  "model_type": "gpt_oss",
        \\  "attention_bias": true,
        \\  "head_dim": 64,
        \\  "hidden_act": "silu",
        \\  "hidden_size": 2880,
        \\  "initial_context_length": 4096,
        \\  "intermediate_size": 2880,
        \\  "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        \\  "max_position_embeddings": 131072,
        \\  "num_attention_heads": 64,
        \\  "num_hidden_layers": 24,
        \\  "num_key_value_heads": 8,
        \\  "num_local_experts": 32,
        \\  "num_experts_per_tok": 4,
        \\  "experts_per_token": 4,
        \\  "rms_norm_eps": 1e-05,
        \\  "rope_scaling": {
        \\    "beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0,
        \\    "original_max_position_embeddings": 4096,
        \\    "rope_type": "yarn", "truncate": false
        \\  },
        \\  "rope_theta": 150000,
        \\  "sliding_window": 128,
        \\  "swiglu_limit": 7.0,
        \\  "tie_word_embeddings": false,
        \\  "vocab_size": 201088,
        \\  "eos_token_id": 200002,
        \\  "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("gpt_oss", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    // Qwen-style norm/embedding flags; gpt-oss has NO QK norm.
    try testing.expect(!config.norm_has_offset);
    try testing.expect(!config.scale_embeddings);
    try testing.expect(!config.has_pre_ff_norm);
    try testing.expect(!config.has_qk_norm);
    // MoE dims: num_local_experts is the spelling, and the expert width is
    // plain intermediate_size (no moe_intermediate_size key).
    try testing.expectEqual(@as(u32, 32), config.num_experts);
    try testing.expectEqual(@as(u32, 4), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 2880), config.moe_intermediate_size);
    // Router is a plain softmax-over-top-k with an ADDITIVE bias, not hy3's
    // sigmoid+selection-bias chain.
    try testing.expect(!config.moe_sigmoid_router);
    try testing.expect(config.moe_route_norm);
    // Attention shape: hd 64, GQA 64/8, scale = head_dim^-0.5.
    try testing.expectEqual(@as(u32, 64), config.head_dim);
    try testing.expectEqual(@as(u32, 64), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 64), config.query_pre_attn_scalar);
    // Learned per-head attention sinks (self_attn.sinks) — the softmax
    // denominator gets an extra column; SDPA takes them natively.
    try testing.expect(config.has_attn_sinks);
    // Alternating layer types, sliding FIRST (layer 0 is local).
    try testing.expect(config.has_explicit_layer_types);
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(1));
    try testing.expect(config.has_sliding_window);
    try testing.expectEqual(@as(u32, 128), config.sliding_window);
    // ONE theta for both layer types. Leaving rope_local_base_freq at the
    // Gemma-flavored 10000 default would mis-base every sliding layer — the
    // muse first-turn-repetition class (deterministic "coherent then loops").
    try testing.expectApproxEqAbs(@as(f32, 150000.0), config.rope_theta, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 150000.0), config.rope_local_base_freq, 1.0);
    // YaRN: mscale is COMPUTED (0.1*ln(32)+1), matching mlx-lm's YarnRoPE
    // defaults (mscale 1 / mscale_all_dim 0) — the config ships no
    // attention_factor at all. Laguna precedent.
    try testing.expect(config.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 32.0), config.yarn_factor, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 32.0), config.yarn_beta_fast, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), config.yarn_beta_slow, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.3465735902799727), config.yarn_attention_factor, 1e-9);
    try testing.expectEqual(@as(u32, 4096), config.yarn_orig_max_pos);
    // Clamped SwiGLU: clip(gate, max=limit) * sigmoid(alpha*gate) * (clip(up, ±limit) + 1).
    // `hidden_act: "silu"` in the config is a lie — the reference never uses it.
    try testing.expectApproxEqAbs(@as(f32, 7.0), config.swiglu_limit, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.702), config.swiglu_alpha, 1e-6);
    // Quant: mxfp4 gs32 for the expert banks (attention/embed/lm_head ride
    // per-tensor affine-8 overrides resolved at weight-load time).
    try testing.expectEqual(QuantMode.mxfp4, config.quant_mode);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 32), config.quant_group_size);
    // Terminators merge ADDITIVELY: <|return|> (200002, the declared eos) and
    // <|call|> (200012, which ends a tool call and is NEVER the eos).
    // <|end|> (200007) is deliberately absent — it closes the analysis channel
    // MID-generation, before the final channel opens.
    const eos = config.eosTokenSlice();
    try testing.expectEqual(@as(usize, 2), eos.len);
    try testing.expectEqual(@as(u32, 200002), eos[0]);
    try testing.expectEqual(@as(u32, 200012), eos[1]);
}

test "ModelConfig: laguna YaRN mscale is COMPUTED, never read from attention_factor (Laguna-XS ships 1.0)" {
    // Laguna-XS-2.1-NVFP4-mlx's config.json carries "attention_factor": 1.0,
    // but both vendored MLX Laguna implementations deliberately drop that field
    // and let MLX compute its default mscale (0.1*ln(factor) + 1); poolside's
    // own fused kernel hardcodes the result, 1.3465735912322998f. Reading the
    // field verbatim would run 10 of XS's 40 layers with unscaled YaRN RoPE.
    // S got away with it only because its config happens to ship the computed
    // value; the two must agree by construction, not by luck.
    const json =
        \\{
        \\  "model_type": "laguna",
        \\  "hidden_size": 2048,
        \\  "intermediate_size": 8192,
        \\  "num_hidden_layers": 4,
        \\  "num_attention_heads": 64,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "rms_norm_eps": 1e-06,
        \\  "vocab_size": 100352,
        \\  "tie_word_embeddings": false,
        \\  "gating": "per-head",
        \\  "sliding_window": 512,
        \\  "num_experts": 256,
        \\  "num_experts_per_tok": 8,
        \\  "moe_intermediate_size": 512,
        \\  "mlp_only_layers": [0],
        \\  "num_attention_heads_per_layer": [48, 64, 64, 64],
        \\  "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        \\  "rope_parameters": {
        \\    "full_attention": {
        \\      "rope_theta": 500000.0, "rope_type": "yarn", "factor": 32.0,
        \\      "original_max_position_embeddings": 8192, "beta_slow": 1.0, "beta_fast": 32.0,
        \\      "attention_factor": 1.0, "partial_rotary_factor": 0.5
        \\    },
        \\    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}
        \\  },
        \\  "quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 32.0), config.yarn_factor, 1e-6);
    // 0.1 * ln(32) + 1 — the same value S's config ships literally.
    try testing.expectApproxEqAbs(@as(f32, 1.3465735902799727), config.yarn_attention_factor, 1e-6);
}

test "ModelConfig parses spark2_5 (Spark-X2.5): fused qkv, headwise sigmoid gate, exact gelu, dual rope" {
    const json =
        \\{
        \\  "model_type": "spark2_5",
        \\  "hidden_size": 2560, "intermediate_size": 10240, "num_hidden_layers": 8,
        \\  "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
        \\  "hidden_act": "gelu", "rms_norm_eps": 1e-06, "vocab_size": 131072,
        \\  "gate_attn_act_mode": "sigmoid", "headwise_attn_output_gate": true,
        \\  "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
        \\  "max_position_embeddings": 1048576, "tie_word_embeddings": true,
        \\  "sliding_window": 512,
        \\  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        \\                  "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        \\  "rope_parameters": {
        \\    "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
        \\    "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000}
        \\  },
        \\  "quantization": {"group_size": 64, "bits": 8, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("spark2_5", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expectEqual(HiddenAct.gelu, config.hidden_act);
    try testing.expect(config.attn_sigmoid_gate);
    try testing.expect(config.attn_gate_headwise);
    try testing.expect(config.attn_fused_qkv);
    try testing.expect(!config.norm_has_offset);
    try testing.expect(!config.has_pre_ff_norm);
    try testing.expect(!config.has_qk_norm);
    try testing.expect(!config.scale_embeddings);
    try testing.expect(config.tie_word_embeddings);
    try testing.expectEqual(@as(u32, 256), config.query_pre_attn_scalar);
    try testing.expect(config.has_explicit_layer_types);
    try testing.expect(config.layer_is_global[3] and !config.layer_is_global[2]);
    try testing.expectEqual(@as(u32, 512), config.sliding_window);
    try testing.expectApproxEqAbs(@as(f32, 5000000.0), config.rope_theta, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 10000.0), config.rope_local_base_freq, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.25), config.partial_rotary_factor_global, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), config.rope_scaling_factor, 1e-6);
    try testing.expect(config.isEosToken(1));
}

test "ModelConfig parses muse_glimmer (Muse-Glimmer-30B): NoPE full layers, qk scale, mixed norm offsets" {
    // Trimmed but faithful copy of meta-models/Muse-Glimmer-30B config.json.
    // 8 layers = two sliding/full groups; full attention every 4th layer
    // counted backward from the last (i%4==3 for 8 layers), and exactly those
    // layers have layer_rope_theta 0 (NoPE).
    const json =
        \\{
        \\  "model_type": "muse_glimmer",
        \\  "image_token_id": 200092,
        \\  "text_config": {
        \\    "model_type": "muse_glimmer_text",
        \\    "hidden_size": 6656,
        \\    "intermediate_size": 19968,
        \\    "num_hidden_layers": 8,
        \\    "num_attention_heads": 32,
        \\    "num_key_value_heads": 2,
        \\    "head_dim": 128,
        \\    "hidden_activation": "silu",
        \\    "rms_norm_eps": 1e-05,
        \\    "post_norm_eps": 1e-08,
        \\    "qk_scale_factor": 3.87,
        \\    "output_multiplier": 0.19611613513818404,
        \\    "final_logit_softcapping": 20.0,
        \\    "vocab_size": 202048,
        \\    "max_position_embeddings": 131072,
        \\    "tie_word_embeddings": false,
        \\    "bos_token_id": 200000,
        \\    "eos_token_id": 200001,
        \\    "sliding_window": 2048,
        \\    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        \\                    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        \\    "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0, 500000.0, 500000.0, 500000.0, 0],
        \\    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}
        \\  },
        \\  "vision_config": {"model_type": "muse_glimmer_vision"},
        \\  "quantization": {"group_size": 64, "bits": 8}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("muse_glimmer", config.model_type);
    try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    try testing.expectEqual(@as(u32, 32), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 2), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 128), config.head_dim);
    try testing.expect(config.has_sliding_window);
    try testing.expectEqual(@as(u32, 2048), config.sliding_window);
    try testing.expect(config.has_explicit_layer_types);
    try testing.expect(config.layer_is_global[3]);
    try testing.expect(config.layer_is_global[7]);
    try testing.expect(!config.layer_is_global[2]);
    // NoPE: exactly the full-attention layers skip RoPE (layer_rope_theta 0).
    try testing.expect(config.layerSkipsRope(3));
    try testing.expect(config.layerSkipsRope(7));
    try testing.expect(!config.layerSkipsRope(0));
    try testing.expectApproxEqAbs(@as(f32, 500000.0), config.rope_theta, 1e-3);
    // Sliding layers read rope_local_base_freq in EVERY forward path, and muse
    // ships ONE theta for all roped layers (rope_parameters.rope_theta) — the
    // Gemma-flavored 10000 default silently mis-rotated all 39 roped layers
    // (2026-08-11 first-turn repetition-loop root cause).
    try testing.expectApproxEqAbs(@as(f32, 500000.0), config.rope_local_base_freq, 1e-3);
    // Attention scale folds the post-qk-norm Q multiplier into 1/sqrt(head_dim).
    try testing.expectApproxEqAbs(@as(f32, 3.87 / 11.313708), config.attnScale(), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.19611613513818404), config.output_multiplier, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 20.0), config.final_logit_softcapping, 1e-6);
    // Sandwich norms are Gemma2-centered (1+w) at eps 1e-5 / post-norms 1e-8;
    // the FINAL norm is plain-scale (Gemma4 style, ones-init).
    try testing.expect(config.norm_has_offset);
    try testing.expect(config.final_norm_plain);
    try testing.expectApproxEqAbs(@as(f32, 1e-08), config.postNormEps(), 1e-12);
    try testing.expect(config.has_pre_ff_norm);
    // Weight-less shared qk-norm (no q_norm/k_norm tensors in the checkpoint),
    // RMS-normed embeddings with NO sqrt(hidden) scale, sigmoid attn out gate.
    try testing.expect(!config.has_qk_norm);
    try testing.expect(config.qk_norm_weightless);
    try testing.expect(!config.scale_embeddings);
    try testing.expect(config.normed_embeddings);
    try testing.expect(config.attn_sigmoid_gate);
    try testing.expect(!config.tie_word_embeddings);
    try testing.expectEqual(HiddenAct.silu, config.hidden_act);
    // <|end_of_text|>(200001) from config + <|eot|>(200008), the template's
    // turn terminator, merged additively.
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expectEqual(@as(u32, 200001), config.eos_token_ids[0]);
    try testing.expectEqual(@as(u32, 200008), config.eos_token_ids[1]);
    try testing.expectEqual(@as(u32, 8), config.quant_bits);
}

test "ModelConfig parses the muse_glimmer vision tower (own key spellings, window/full pattern)" {
    // Trimmed copy of the real vision_config: muse names its geometry keys
    // differently from Qwen (patch_temporal, merge_size, pos_emb_*), and the
    // window/full pattern is per-layer, not a stride.
    const json =
        \\{
        \\  "model_type": "muse_glimmer",
        \\  "image_token_id": 200092,
        \\  "out_hidden_size": 6144,
        \\  "projector_hidden_size": 4096,
        \\  "projector_hidden_act": "gelu",
        \\  "text_config": {"model_type": "muse_glimmer_text", "hidden_size": 6656, "head_dim": 128,
        \\                  "num_attention_heads": 32, "num_key_value_heads": 2, "rms_norm_eps": 1e-05},
        \\  "vision_config": {
        \\    "model_type": "muse_glimmer_vision",
        \\    "hidden_act": "gelu",
        \\    "hidden_size": 1536,
        \\    "intermediate_size": 8960,
        \\    "layer_norm_eps": 1e-05,
        \\    "layer_types": ["window_attention", "window_attention", "window_attention", "full_attention",
        \\                    "window_attention", "full_attention"],
        \\    "merge_size": 2,
        \\    "num_attention_heads": 16,
        \\    "num_hidden_layers": 6,
        \\    "patch_size": 14,
        \\    "patch_temporal": 2,
        \\    "pos_emb_height": 32,
        \\    "pos_emb_width": 32,
        \\    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
        \\  }
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.muse_vision);
    try testing.expect(config.has_vision);
    try testing.expect(!config.qwen_vision); // no M-RoPE, no vision_start/end
    try testing.expectEqual(@as(u32, 6), config.qv_depth);
    try testing.expectEqual(@as(u32, 1536), config.qv_hidden);
    try testing.expectEqual(@as(u32, 16), config.qv_heads);
    try testing.expectEqual(@as(u32, 96), config.qv_head_dim);
    try testing.expectEqual(@as(u32, 8960), config.qv_intermediate);
    try testing.expectEqual(@as(u32, 14), config.qv_patch);
    try testing.expectEqual(@as(u32, 2), config.qv_temporal_patch);
    try testing.expectEqual(@as(u32, 2), config.qv_merge);
    try testing.expectEqual(@as(u32, 32), config.mv_pos_side);
    try testing.expectEqual(@as(u32, 4096), config.mv_projector_hidden);
    // The tower's own output width is the TEXT hidden size — `out_hidden_size`
    // is the adapter's INPUT (hidden x merge^2), not the spliced width.
    try testing.expectEqual(@as(u32, 6656), config.qv_out_hidden);
    try testing.expectApproxEqAbs(@as(f32, 1e-05), config.mv_ln_eps, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10000.0), config.mv_rope_theta, 1e-6);
    // Every 4th layer is full attention; the rest window. Read per layer, since
    // the released 50-layer tower ends on an off-stride full layer.
    try testing.expect(!config.mv_full_attn[0]);
    try testing.expect(config.mv_full_attn[3]);
    try testing.expect(!config.mv_full_attn[4]);
    try testing.expect(config.mv_full_attn[5]);
    // muse wraps the pad run with <|image_start|>/<|image_end|>, so the generic
    // BOI/EOI inserter needs no muse arm.
    try testing.expectEqual(@as(u32, 200092), config.image_token_id);
    try testing.expectEqual(@as(u32, 200080), config.boi_token_id);
    try testing.expectEqual(@as(u32, 200081), config.eoi_token_id);
}

test "ModelConfig muse_glimmer_text flat sibling collapses onto muse_glimmer with bare prefix" {
    const json =
        \\{
        \\  "model_type": "muse_glimmer_text",
        \\  "hidden_size": 6656,
        \\  "num_hidden_layers": 8,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 128,
        \\  "vocab_size": 202048,
        \\  "sliding_window": 2048
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("muse_glimmer", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
}

test "ModelConfig parses inkling_mm_model (Thinking Machines Inkling Small REAP25)" {
    // Real shape of pipenetwork/Inkling-Small-MLX-REAP25-4bit's config.json
    // (REAP-pruned 192/256 routed experts; the full builds differ only in
    // n_routed_experts). NO RoPE anywhere — position comes from the
    // relative-logits bias + per-layer short convolutions + log-scaling; the
    // checkpoint labels the MoE expert width `intermediate_size` and the dense
    // bottom-layer width `dense_intermediate_size` (opposite of our field
    // meanings, swapped in the arm). Scale is 1/head_dim (per-head q/k RMSNorm),
    // not 1/sqrt(head_dim).
    const json =
        \\{
        \\  "architectures": ["InklingForConditionalGeneration"],
        \\  "model_type": "inkling_mm_model",
        \\  "eos_token_id": 200006,
        \\  "text_config": {
        \\    "model_max_length": 1048576,
        \\    "hidden_size": 4096,
        \\    "num_hidden_layers": 42,
        \\    "vocab_size": 201024,
        \\    "num_attention_heads": 32,
        \\    "num_key_value_heads": 8,
        \\    "head_dim": 128,
        \\    "d_rel": 16,
        \\    "rel_extent": 1024,
        \\    "log_scaling_n_floor": 128000,
        \\    "log_scaling_alpha": 0.1,
        \\    "rms_norm_eps": 1e-06,
        \\    "use_embed_norm": true,
        \\    "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22,24,25,26,27,28,30,31,32,33,34,36,37,38,39,40],
        \\    "dense_mlp_idx": 2,
        \\    "use_sconv": true,
        \\    "sconv_kernel_size": 4,
        \\    "unpadded_vocab_size": 200058,
        \\    "logits_mup_width_multiplier": 16.0,
        \\    "swa_head_dim": 128,
        \\    "swa_num_attention_heads": 32,
        \\    "swa_num_key_value_heads": 8,
        \\    "sliding_window_size": 512,
        \\    "n_routed_experts": 192,
        \\    "num_experts_per_tok": 6,
        \\    "n_shared_experts": 2,
        \\    "shared_expert_sink": true,
        \\    "dense_intermediate_size": 16384,
        \\    "intermediate_size": 2048,
        \\    "route_scale": 8.0,
        \\    "use_gate_bias": true,
        \\    "gate_activation": "sigmoid",
        \\    "norm_after_topk": true,
        \\    "use_global_scale": true
        \\  },
        \\  "audio_config": {"n_mel_bins": 80, "mel_vocab_size": 16},
        \\  "vision_config": {"vision_encoder_type": "hmlp", "patch_size": 40, "n_layers": 4},
        \\  "mtp_config": {"num_nextn_predict_layers": 8},
        \\  "quantization": {"group_size": 64, "bits": 4, "recipe": "uniform"},
        \\  "reap": {"kept_experts": 192}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("inkling_mm_model", config.model_type);
    try testing.expectEqualStrings("model.llm", config.weight_prefix);
    try testing.expectEqual(@as(u32, 201024), config.vocab_size);
    try testing.expectEqual(@as(u32, 200058), config.unpadded_vocab_size);
    try testing.expectEqual(@as(u32, 4096), config.hidden_size);
    try testing.expectEqual(@as(u32, 42), config.num_hidden_layers);
    try testing.expectEqual(@as(u32, 32), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 128), config.head_dim);
    try testing.expectEqual(@as(u32, 1048576), config.max_position_embeddings);
    // scale = 1/head_dim, expressed through 1/sqrt(query_pre_attn_scalar)
    try testing.expectEqual(@as(u32, 128 * 128), config.query_pre_attn_scalar);
    // Hybrid sliding/global from local_layer_ids: every 6th layer global.
    try testing.expect(config.has_sliding_window);
    try testing.expectEqual(@as(u32, 512), config.sliding_window);
    try testing.expect(config.has_explicit_layer_types);
    try testing.expect(config.isGlobalLayer(5));
    try testing.expect(config.isGlobalLayer(41));
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(40));
    // Dense bottom layers vs MoE: widths swapped from the checkpoint labels.
    try testing.expectEqual(@as(u32, 2), config.first_k_dense_replace);
    try testing.expectEqual(@as(u32, 16384), config.intermediate_size);
    try testing.expectEqual(@as(u32, 2048), config.moe_intermediate_size);
    try testing.expectEqual(@as(u32, 192), config.num_experts);
    try testing.expectEqual(@as(u32, 6), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 2), config.inkling_n_shared_experts);
    try testing.expectApproxEqAbs(@as(f32, 8.0), config.router_scaling_factor, 1e-6);
    // Position machinery: rel-logits bias + short conv + log-scaling.
    try testing.expectEqual(@as(u32, 16), config.inkling_d_rel);
    try testing.expectEqual(@as(u32, 1024), config.inkling_rel_extent);
    try testing.expectEqual(@as(u32, 128000), config.inkling_log_n_floor);
    try testing.expectApproxEqAbs(@as(f32, 0.1), config.inkling_log_alpha, 1e-6);
    try testing.expectEqual(@as(u32, 4), config.inkling_sconv_kernel);
    try testing.expect(config.has_embedding_norm);
    try testing.expect(config.has_qk_norm);
    try testing.expectEqual(HiddenAct.silu, config.hidden_act);
    try testing.expectApproxEqAbs(@as(f32, 16.0), config.logits_mup_width_multiplier, 1e-6);
    // v1 is text-only: the hMLP vision_config must NOT arm the SigLIP path.
    try testing.expect(!config.has_vision);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    const eos = config.eosTokenSlice();
    try testing.expectEqual(@as(usize, 1), eos.len);
    try testing.expectEqual(@as(u32, 200006), eos[0]);
}

test "ModelConfig parses deepseek_v4 (DeepSeek-V4-Flash-0731 mirror)" {
    // Shape of our converted mirror's config.json: the deepseek-ai release
    // config minus quantization_config (fp8 source), plus the converter's
    // per-weight `quantization` dict. MQA over ONE 512-dim latent, low-rank
    // Q/grouped-low-rank O, sliding-window 128 + per-layer compression
    // (ratio 4 overlapping w/ indexer, 128 plain), Sinkhorn hyper-connections,
    // hash routing on the first 3 layers, sqrt(softplus) scoring — all
    // identical between the preview and 0731, which differ only by the DSpark
    // draft module (3 stages ⇒ 46 compress_ratios, + the dspark_* block).
    const json =
        \\{
        \\  "architectures": ["DeepseekV4ForCausalLM"],
        \\  "model_type": "deepseek_v4",
        \\  "bos_token_id": 0,
        \\  "eos_token_id": 1,
        \\  "head_dim": 512,
        \\  "hidden_act": "silu",
        \\  "hidden_size": 4096,
        \\  "index_head_dim": 128,
        \\  "index_n_heads": 64,
        \\  "index_topk": 512,
        \\  "max_position_embeddings": 1048576,
        \\  "moe_intermediate_size": 2048,
        \\  "n_routed_experts": 256,
        \\  "n_shared_experts": 1,
        \\  "norm_topk_prob": true,
        \\  "num_attention_heads": 64,
        \\  "num_experts_per_tok": 6,
        \\  "num_hidden_layers": 43,
        \\  "num_hash_layers": 3,
        \\  "num_key_value_heads": 1,
        \\  "num_nextn_predict_layers": 3,
        \\  "dspark_block_size": 5,
        \\  "dspark_noise_token_id": 128799,
        \\  "dspark_target_layer_ids": [40, 41, 42],
        \\  "dspark_markov_rank": 256,
        \\  "o_groups": 8,
        \\  "o_lora_rank": 1024,
        \\  "q_lora_rank": 1024,
        \\  "qk_rope_head_dim": 64,
        \\  "hc_eps": 1e-06,
        \\  "hc_mult": 4,
        \\  "hc_sinkhorn_iters": 20,
        \\  "rms_norm_eps": 1e-06,
        \\  "rope_scaling": {
        \\    "beta_fast": 32,
        \\    "beta_slow": 1,
        \\    "factor": 16,
        \\    "original_max_position_embeddings": 65536,
        \\    "type": "yarn"
        \\  },
        \\  "rope_theta": 10000,
        \\  "routed_scaling_factor": 1.5,
        \\  "scoring_func": "sqrtsoftplus",
        \\  "sliding_window": 128,
        \\  "swiglu_limit": 10.0,
        \\  "tie_word_embeddings": false,
        \\  "topk_method": "noaux_tc",
        \\  "torch_dtype": "bfloat16",
        \\  "vocab_size": 129280,
        \\  "compress_rope_theta": 160000,
        \\  "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0],
        \\  "quantization": {"group_size": 64, "bits": 8, "mode": "affine",
        \\    "layers.0.ffn.experts.w1": {"group_size": 64, "bits": 2, "mode": "affine"}}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("deepseek_v4", config.model_type);
    try testing.expectEqualStrings("", config.weight_prefix);
    try testing.expectEqual(@as(u32, 129280), config.vocab_size);
    try testing.expectEqual(@as(u32, 4096), config.hidden_size);
    try testing.expectEqual(@as(u32, 43), config.num_hidden_layers);
    try testing.expectEqual(@as(u32, 64), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 1), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 512), config.head_dim);
    try testing.expectEqual(@as(u32, 1048576), config.max_position_embeddings);
    try testing.expect(config.has_sliding_window);
    try testing.expectEqual(@as(u32, 128), config.sliding_window);
    // MoE: 256 experts top-6, shared expert at moe width, sum-normalized
    // weights × 1.5; hash routing on the first 3 layers.
    try testing.expectEqual(@as(u32, 256), config.num_experts);
    try testing.expectEqual(@as(u32, 6), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 2048), config.moe_intermediate_size);
    try testing.expect(config.moe_route_norm);
    try testing.expectApproxEqAbs(@as(f32, 1.5), config.router_scaling_factor, 1e-6);
    try testing.expectEqual(@as(u32, 3), config.dsv4_hash_layers);
    // Attention geometry.
    try testing.expectEqual(@as(u32, 1024), config.dsv4_q_lora_rank);
    try testing.expectEqual(@as(u32, 1024), config.dsv4_o_lora_rank);
    try testing.expectEqual(@as(u32, 8), config.dsv4_o_groups);
    try testing.expectEqual(@as(u32, 64), config.dsv4_rope_head_dim);
    // Indexer + compression.
    try testing.expectEqual(@as(u32, 64), config.dsv4_index_n_heads);
    try testing.expectEqual(@as(u32, 128), config.dsv4_index_head_dim);
    try testing.expectEqual(@as(u32, 512), config.dsv4_index_topk);
    try testing.expectApproxEqAbs(@as(f32, 160000.0), config.dsv4_compress_rope_theta, 1e-3);
    try testing.expectEqual(@as(u32, 46), config.dsv4_n_compress_ratios);
    try testing.expectEqual(@as(u8, 0), config.dsv4_compress_ratios[0]);
    try testing.expectEqual(@as(u8, 4), config.dsv4_compress_ratios[2]);
    try testing.expectEqual(@as(u8, 128), config.dsv4_compress_ratios[3]);
    try testing.expectEqual(@as(u8, 128), config.dsv4_compress_ratios[41]);
    // Layer 42 IS compressed (ratio 4, with indexer) — only layers 0/1 and
    // the MTP module run pure sliding-window attention.
    try testing.expectEqual(@as(u8, 4), config.dsv4_compress_ratios[42]);
    // The three trailing entries are DSpark's draft stages: pure sliding
    // window, like layers 0/1.
    try testing.expectEqual(@as(u8, 0), config.dsv4_compress_ratios[43]);
    try testing.expectEqual(@as(u8, 0), config.dsv4_compress_ratios[45]);
    // DSpark descriptor — what tells a 0731 checkpoint from the preview.
    try testing.expectEqual(@as(u32, 5), config.dsv4_dspark_block_size);
    try testing.expectEqual(@as(u32, 128799), config.dsv4_dspark_noise_token_id);
    try testing.expectEqual(@as(u32, 256), config.dsv4_dspark_markov_rank);
    try testing.expectEqual(@as(u32, 3), config.dsv4_n_dspark_target_layers);
    try testing.expectEqual(@as(u8, 40), config.dsv4_dspark_target_layers[0]);
    try testing.expectEqual(@as(u8, 42), config.dsv4_dspark_target_layers[2]);
    // Hyper-connections + clipped SwiGLU.
    try testing.expectEqual(@as(u32, 4), config.dsv4_hc_mult);
    try testing.expectEqual(@as(u32, 20), config.dsv4_hc_sinkhorn_iters);
    try testing.expectApproxEqAbs(@as(f32, 1e-6), config.dsv4_hc_eps, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 10.0), config.dsv4_swiglu_limit, 1e-6);
    // YaRN applies only on compressed layers (at compress_rope_theta);
    // ratio-0 layers run plain rope_theta. The forward picks per layer.
    try testing.expect(config.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 16.0), config.yarn_factor, 1e-6);
    try testing.expectEqual(@as(u32, 65536), config.yarn_orig_max_pos);
    try testing.expectApproxEqAbs(@as(f32, 10000.0), config.rope_theta, 1e-3);
    // In-checkpoint draft stages (mtp.0/1/2.*, all ratio 0).
    try testing.expectEqual(@as(u32, 3), config.dsv4_mtp_layers);
    try testing.expectEqual(HiddenAct.silu, config.hidden_act);
    try testing.expectEqual(@as(u32, 8), config.quant_bits);
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    const eos = config.eosTokenSlice();
    try testing.expectEqual(@as(usize, 1), eos.len);
    try testing.expectEqual(@as(u32, 1), eos[0]);
}

/// DeepSeek's V4.1-Flash release config (rev dba1be0a), text_config abridged
/// to the keys the engine reads, vision_config dropped (text-only).
pub const dsv41_release_config =
    \\{"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
    \\ "bos_token_id": 0, "eos_token_id": 1,
    \\ "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 129280, "hidden_size": 5120,
    \\  "moe_intermediate_size": 2304, "num_hidden_layers": 40, "num_attention_heads": 64,
    \\  "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280,
    \\  "o_lora_rank": 1024, "o_groups": 8, "hidden_act": "silu", "swiglu_limit": 10.0,
    \\  "rms_norm_eps": 1e-20, "max_position_embeddings": 1048576, "rope_theta": 10000,
    \\  "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
    \\   "original_max_position_embeddings": 65536},
    \\  "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6,
    \\  "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "norm_topk_prob": true,
    \\  "routed_scaling_factor": 1.5, "sliding_window": 128,
    \\  "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    \\  "compress_rope_theta": 160000, "kv_source_layer_ids": [2, 8, 14, 20],
    \\  "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36], "index_n_heads": 32,
    \\  "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 20,
    \\  "candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4,
    \\  "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1, 14],
    \\  "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4,
    \\  "engram_vocab_size": 16000000, "engram_n_heads": 8, "engram_head_dim": 256,
    \\  "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092,
    \\  "num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_noise_token_id": 128799,
    \\  "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256,
    \\  "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3},
    \\ "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32}}
;

test "ModelConfig parses deepseek_v41 (DeepSeek-V4.1-Flash release)" {
    const config = try parseConfigFromJson(testing.allocator, dsv41_release_config);
    try testing.expectEqualStrings("deepseek_v41", config.model_type);
    try testing.expectEqualStrings("", config.weight_prefix);
    try testing.expect(!config.has_vision); // the tower is not served
    try testing.expectEqual(@as(u32, 5120), config.hidden_size);
    try testing.expectEqual(@as(u32, 40), config.num_hidden_layers);
    try testing.expectEqual(@as(u32, 512), config.head_dim);
    try testing.expectEqual(@as(u32, 384), config.num_experts);
    try testing.expectEqual(@as(u32, 6), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 2304), config.moe_intermediate_size);
    try testing.expectApproxEqAbs(@as(f32, 1.5), config.router_scaling_factor, 1e-6);
    try testing.expectEqual(@as(u32, 1280), config.dsv4_q_lora_rank);
    try testing.expectEqual(@as(u32, 32), config.dsv4_index_n_heads);
    try testing.expectEqual(@as(u32, 43), config.dsv4_n_compress_ratios);
    try testing.expectEqual(@as(u8, 2), config.dsv4_compress_ratios[19]);
    try testing.expectEqual(@as(u8, 1), config.dsv4_compress_ratios[20]);
    try testing.expectEqual(@as(u32, 0), config.dsv4_hash_layers);
    try testing.expectEqual(@as(u64, (1 << 2) | (1 << 8) | (1 << 14) | (1 << 20)), config.dsv41_kv_sources);
    try testing.expectEqual(@as(u64, (1 << 2) | (1 << 8) | (1 << 14) | (1 << 20) | (1 << 24) | (1 << 28) | (1 << 32) | (1 << 36)), config.dsv41_index_sources);
    try testing.expectEqual(@as(i32, 20), config.dsv41_candidate_source);
    try testing.expectEqual(@as(u32, 2048), config.dsv41_candidate_topk_blocks);
    try testing.expectEqual(@as(u32, 8), config.dsv41_candidate_block_size);
    try testing.expectEqual(@as(u32, 2), config.dsv41_n_engram_layers);
    try testing.expectEqual(@as(u8, 14), config.dsv41_engram_layers[1]);
    try testing.expectEqual(@as(u64, 384016682), config.dsv41_engram_rows[1]);
    try testing.expectEqual(@as(u32, 4), config.dsv41_engram_max_ngram);
    try testing.expectEqual(@as(u64, 16000000), config.dsv41_engram_vocab_size);
    try testing.expectEqual(@as(u32, 8), config.dsv41_engram_n_heads);
    try testing.expectEqual(@as(u32, 256), config.dsv41_engram_head_dim);
    try testing.expectEqual(@as(u32, 2), config.dsv41_engram_pad_id);
    try testing.expectEqual(@as(u32, 99092), config.dsv41_engram_compressed_vocab);
    try testing.expectEqual(@as(u32, 5), config.dsv4_dspark_block_size);
    try testing.expectEqual(@as(u8, 37), config.dsv4_dspark_target_layers[0]);
    try testing.expectEqual(@as(u32, 128), config.dsv41_dspark_experts);
    try testing.expectEqual(@as(u32, 3), config.dsv41_dspark_top_k);
    try testing.expect(config.rope_yarn);
    try testing.expectEqual(@as(u32, 65536), config.yarn_orig_max_pos);
    // A single-flight arch that never batches decode.
    try testing.expect(!config.supportsBatchedGdnDecode());
}

test "deepseek_v41 bills its compressed caches per token, the window per slot" {
    const config = try parseConfigFromJson(testing.allocator, dsv41_release_config);
    // KV sources 2/8/14 keep a 512-wide latent per two tokens and 20 one per
    // token; the same four own index keys (128 wide), all bf16.
    try testing.expectEqual(@as(u64, 2 * (3 * 256 + 512 + 3 * 64 + 128)), config.kvBytesPerToken());
    // Every layer reads its 128-token window plus top-512 compressed entries.
    try testing.expectEqual(@as(u64, 128 + 512 + 1), config.prefillAttnKeys(1 << 20));
    try testing.expectEqual(@as(u64, 32), config.prefillAttnKeys(32));
}

test "parseConfig: a deepseek_v41 pack with an EXL3 bank is mlx-stream's, and the host loads none of its weights" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = dsv41_release_config });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    var plain = try parseConfig(io, testing.allocator, dir);
    defer plain.deinit(testing.allocator);
    try testing.expect(!plain.dsv41_stream);
    try tmp.dir.writeFile(io, .{ .sub_path = "experts.bin", .data = "" });
    var repack = try parseConfig(io, testing.allocator, dir);
    defer repack.deinit(testing.allocator);
    try testing.expect(repack.dsv41_stream);
    var w = try loadModelWeights(io, testing.allocator, dir, &repack, false);
    defer w.deinit();
    try testing.expectEqual(@as(u32, 0), w.count());
}

test "loadWeightsOpt deepseek_v41: tensors nested under language_model. (oMLX's packs) load under the release's names" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const hdr = "{\"language_model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}," ++
        "\"vision.norm.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,16]}}";
    var shard: [8 + hdr.len + 16]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], hdr.len, .little);
    @memcpy(shard[8..][0..hdr.len], hdr);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &shard });
    var buf: [512]u8 = undefined;
    var w = try loadWeightsOpt(io, testing.allocator, buf[0..try tmp.dir.realPath(io, &buf)], .{ .dsv41 = true });
    defer w.deinit();
    try testing.expect(w.get("norm.weight") != null);
    try testing.expectEqual(@as(u32, 1), w.count());
}

test "ModelConfig deepseek_v41: shapes the forward does not implement are refused by name" {
    const Case = struct { from: []const u8, to: []const u8 };
    const cases = [_]Case{
        .{ .from = "\"num_key_value_heads\": 1", .to = "\"num_key_value_heads\": 2" },
        .{ .from = "\"sqrtsoftplus\"", .to = "\"softmax\"" },
        .{ .from = "\"noaux_tc\"", .to = "\"greedy\"" },
        // compress mode 3 (any ratio above 2)
        .{ .from = "[0, 0, 2, 2,", .to = "[0, 0, 3, 3," },
        // layer 2 compresses but no KV source sits at or above it
        .{ .from = "\"kv_source_layer_ids\": [2, 8, 14, 20]", .to = "\"kv_source_layer_ids\": [8, 14, 20]" },
        // the ratio-1 layers would read a ratio-2 source
        .{ .from = "\"kv_source_layer_ids\": [2, 8, 14, 20]", .to = "\"kv_source_layer_ids\": [2, 8, 14]" },
        // candidate source that is not an index source
        .{ .from = "\"candidate_source_layer_id\": 20", .to = "\"candidate_source_layer_id\": 21" },
        .{ .from = "\"hc_mult\": 4", .to = "\"hc_mult\": 2" },
    };
    for (cases) |c| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, dsv41_release_config, c.from, c.to);
        defer testing.allocator.free(json);
        try testing.expect(!std.mem.eql(u8, json, dsv41_release_config));
        testing.expectError(error.UnsupportedDsv41Config, parseConfigFromJson(testing.allocator, json)) catch |e| {
            std.debug.print("not refused: {s} -> {s}\n", .{ c.from, c.to });
            return e;
        };
    }
}

test "prefillAttnKeys: dense archs bill the whole prompt, deepseek_v4 bills its sparse bound" {
    // The admission guard's score term asks ONE question: how many keys does a
    // single query actually read during prefill? Dense causal attention reads
    // the whole prompt; DSV4 reads a 128-wide raw window plus ONE compressed
    // arm, and that difference is the whole 10.3 GB spurious-400 (2026-07-31).
    var dense = ModelConfig{};
    dense.model_type = "qwen3_5";
    try testing.expectEqual(@as(u64, 100_000), dense.prefillAttnKeys(100_000));

    var cfg = ModelConfig{};
    cfg.model_type = "deepseek_v4";
    cfg.sliding_window = 128;
    cfg.dsv4_index_topk = 512;
    cfg.dsv4_n_compress_ratios = 4;
    cfg.dsv4_compress_ratios[0] = 0; // window only
    cfg.dsv4_compress_ratios[1] = 4; // top-k indexer arm
    cfg.dsv4_compress_ratios[2] = 128; // all-visible arm, seq/128 slots
    cfg.dsv4_compress_ratios[3] = 0;

    // 5806 tokens: ratio-4 layers select top-512 of 1451 slots; ratio-128
    // layers see 45. Widest layer = 128 + 512 + 1 sink.
    try testing.expectEqual(@as(u64, 641), cfg.prefillAttnKeys(5806));

    // At 1M the ALL-VISIBLE arm overtakes the top-k one (1M/128 = 8192 slots),
    // so the bound must track it rather than freezing at index_topk — this is
    // the term that keeps the guard honest at long context.
    try testing.expectEqual(@as(u64, 128 + 8192 + 1), cfg.prefillAttnKeys(1_048_576));

    // Never wider than the prompt: a 32-token prompt has 32 keys, not 641.
    try testing.expectEqual(@as(u64, 32), cfg.prefillAttnKeys(32));

    // A config that declares no ratios at all falls back to dense — an arch we
    // cannot bound must never be billed as if we had bounded it.
    var bare = ModelConfig{};
    bare.model_type = "deepseek_v4";
    bare.sliding_window = 128;
    bare.dsv4_n_compress_ratios = 0;
    try testing.expectEqual(@as(u64, 100_000), bare.prefillAttnKeys(100_000));
}

test "ModelConfig deepseek_v4: the superseded PREVIEW checkpoint is rejected" {
    // The preview's single next-token MTP module shares nothing with DSpark's
    // block-parallel stages beyond the `mtp.*` namespace, and the vendor
    // withdrew it — supporting both would mean two draft architectures. A
    // preview config is exactly "declares MTP layers, carries no dspark_*
    // descriptor"; loading it would silently ignore its draft weights, so it
    // has to fail at parse with a message naming the fix.
    const allocator = testing.allocator;
    const json =
        \\{
        \\  "model_type": "deepseek_v4", "num_hidden_layers": 43, "hidden_size": 4096,
        \\  "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
        \\  "qk_rope_head_dim": 64, "q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8,
        \\  "sliding_window": 128, "index_n_heads": 64, "index_head_dim": 128, "index_topk": 512,
        \\  "n_routed_experts": 256, "num_experts_per_tok": 6, "moe_intermediate_size": 2048,
        \\  "n_shared_experts": 1, "num_hash_layers": 3, "routed_scaling_factor": 1.5,
        \\  "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "norm_topk_prob": true,
        \\  "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "swiglu_limit": 10.0,
        \\  "rms_norm_eps": 1e-6, "vocab_size": 129280, "max_position_embeddings": 1048576,
        \\  "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
        \\  "num_nextn_predict_layers": 1,
        \\  "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],
        \\  "bos_token_id": 0, "eos_token_id": 1
        \\}
    ;
    try testing.expectError(error.UnsupportedDsv4Config, parseConfigFromJson(allocator, json));
}

test "ModelConfig deepseek_v4 rejects unsupported scoring/shared-expert shapes" {
    // The forward implements exactly sqrt(softplus) scoring with
    // selection-only bias and ONE always-on shared expert. A checkpoint that
    // diverges must refuse to load, not run silently wrong.
    const bad_scoring =
        \\{"model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": 43,
        \\ "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
        \\ "vocab_size": 129280, "n_routed_experts": 256, "num_experts_per_tok": 6,
        \\ "n_shared_experts": 1, "moe_intermediate_size": 2048,
        \\ "scoring_func": "softmax", "topk_method": "noaux_tc"}
    ;
    try testing.expectError(error.UnsupportedDsv4Config, parseConfigFromJson(testing.allocator, bad_scoring));
    const bad_shared =
        \\{"model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": 43,
        \\ "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
        \\ "vocab_size": 129280, "n_routed_experts": 256, "num_experts_per_tok": 6,
        \\ "n_shared_experts": 2, "moe_intermediate_size": 2048,
        \\ "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc"}
    ;
    try testing.expectError(error.UnsupportedDsv4Config, parseConfigFromJson(testing.allocator, bad_shared));
}

test "ModelConfig inkling_mm_model rejects a sliding-attention geometry that differs from global" {
    // The config carries separate swa_* head fields; the shipped checkpoints
    // are uniform (32/8/128 both classes) and the forward implements exactly
    // that. A future checkpoint that diverges must be an honest reject, not a
    // silently wrong forward.
    const json =
        \\{
        \\  "model_type": "inkling_mm_model",
        \\  "text_config": {
        \\    "hidden_size": 4096, "num_hidden_layers": 42, "vocab_size": 201024,
        \\    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        \\    "swa_head_dim": 128, "swa_num_attention_heads": 32, "swa_num_key_value_heads": 16,
        \\    "sliding_window_size": 512, "sconv_kernel_size": 4,
        \\    "d_rel": 16, "rel_extent": 1024
        \\  }
        \\}
    ;
    try testing.expectError(error.UnsupportedInklingConfig, parseConfigFromJson(testing.allocator, json));
}

test "ModelConfig: use_bidirectional_attention marks an embedding encoder (EmbeddingGemma, issue #79)" {
    // Real shape of mlx-community/embeddinggemma-300m-8bit's config.json: a
    // gemma3_text DECODER config trained bidirectionally. Without the flag
    // routing it to the encoder path, it loads as a causal chat model
    // (garbage output) and /v1/embeddings rejects it.
    const json =
        \\{
        \\  "model_type": "gemma3_text",
        \\  "use_bidirectional_attention": true,
        \\  "hidden_size": 768,
        \\  "num_hidden_layers": 24,
        \\  "num_attention_heads": 3,
        \\  "num_key_value_heads": 1,
        \\  "head_dim": 256,
        \\  "intermediate_size": 1152,
        \\  "sliding_window": 512,
        \\  "bos_token_id": 2,
        \\  "eos_token_id": 1,
        \\  "pad_token_id": 0,
        \\  "max_position_embeddings": 2048,
        \\  "rope_theta": 1000000.0,
        \\  "rope_local_base_freq": 10000.0,
        \\  "query_pre_attn_scalar": 256,
        \\  "vocab_size": 262144,
        \\  "quantization": {"group_size": 64, "bits": 8}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.use_bidirectional_attention);
    // Implies encoder-only: /v1/embeddings accepts it, chat surfaces 400 it,
    // discovery advertises the embeddings capability.
    try testing.expect(config.is_encoder_only);
    try testing.expectEqual(@as(?u32, 2), config.bos_token_id);
    try testing.expectEqual(@as(u32, 8), config.quant_bits);

    // A chat gemma3_text WITHOUT the flag must stay a generation model.
    const chat_json =
        \\{
        \\  "model_type": "gemma3_text",
        \\  "hidden_size": 768,
        \\  "num_hidden_layers": 24,
        \\  "num_attention_heads": 3,
        \\  "num_key_value_heads": 1,
        \\  "head_dim": 256,
        \\  "vocab_size": 262144
        \\}
    ;
    const chat_config = try parseConfigFromJson(testing.allocator, chat_json);
    try testing.expect(!chat_config.use_bidirectional_attention);
    try testing.expect(!chat_config.is_encoder_only);
}

const embedding_gemma2_config_json =
    \\{
    \\  "architectures": ["EmbeddingGemma2Model"],
    \\  "model_type": "embedding_gemma2",
    \\  "audio_config": {"model_type": "gemma4_audio", "hidden_size": 1024, "output_proj_dims": 1536},
    \\  "audio_token_id": 258881,
    \\  "image_token_id": 258880,
    \\  "text_config": {
    \\    "model_type": "embedding_gemma2_text",
    \\    "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
    \\    "embedding_dim": 768, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh",
    \\    "hidden_size": 512, "hidden_size_per_layer_input": 512, "intermediate_size": 2048,
    \\    "layer_types": [
    \\      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    \\      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    \\      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    \\      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
    \\    ],
    \\    "max_position_embeddings": 262144,
    \\    "num_attention_heads": 4, "num_hidden_layers": 24, "num_key_value_heads": 2,
    \\    "per_layer_config": {
    \\      "05": {"head_dim": 512, "num_key_value_heads": 1}, "11": {"head_dim": 512, "num_key_value_heads": 1},
    \\      "17": {"head_dim": 512, "num_key_value_heads": 1}, "23": {"head_dim": 512, "num_key_value_heads": 1}
    \\    },
    \\    "rms_norm_eps": 1e-06,
    \\    "rope_parameters": {
    \\      "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
    \\      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
    \\    },
    \\    "sliding_window": 512, "vocab_size": 262144
    \\  },
    \\  "boi_token_id": 255999, "eoi_token_id": 258882,
    \\  "video_token_id": 258884,
    \\  "vision_config": {
    \\    "model_type": "gemma4_vision", "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 16,
    \\    "num_attention_heads": 12, "num_key_value_heads": 12, "head_dim": 64, "global_head_dim": 64,
    \\    "patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 280, "position_embedding_size": 10240,
    \\    "use_clipped_linears": false, "standardize": false, "rms_norm_eps": 1e-06,
    \\    "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"}
    \\  },
    \\  "quantization": {"group_size": 64, "bits": 8, "mode": "affine"}
    \\}
;

test "ModelConfig: embedding_gemma2 is a bidirectional Gemma 4 trunk with projection-only per-layer inputs" {
    const config = try parseConfigFromJson(testing.allocator, embedding_gemma2_config_json);
    try testing.expect(config.isEmbeddingGemma2());
    try testing.expectEqualStrings("language_model", config.weight_prefix);
    // An encoder: /v1/embeddings accepts it, chat 400s it, and the input is wrapped <bos>…<eos>.
    try testing.expect(config.is_encoder_only);
    try testing.expect(config.use_bidirectional_attention);
    try testing.expect(config.hasEmbeddingCapability());
    try testing.expectEqual(PoolingMode.mean, config.effectivePooling());
    try testing.expectEqual(@as(?u32, 2), config.bos_token_id);
    try testing.expectEqual(@as(u32, 1), config.num_eos_tokens);
    try testing.expectEqual(@as(u32, 1), config.eos_token_ids[0]);
    try testing.expectEqual(@as(u32, 8), config.quant_bits);
    // The trained window (model card), not the trunk's 262144 rope table.
    try testing.expectEqual(@as(u32, 8192), config.max_position_embeddings);
    // Gemma 4 trunk numerics: plain norms, QK/V norms, unit attention scale, sqrt(hidden) embeddings.
    try testing.expect(!config.norm_has_offset);
    try testing.expect(config.has_qk_norm and config.has_v_norm and config.has_pre_ff_norm);
    try testing.expect(config.scale_embeddings);
    try testing.expectEqual(@as(f32, 1.0), config.attnScale());
    try testing.expectEqual(@as(u32, 512), config.hidden_size_per_layer_input);
    try testing.expectEqual(@as(f32, 1_000_000.0), config.rope_theta);
    try testing.expectEqual(@as(f32, 10_000.0), config.rope_local_base_freq);
    try testing.expect(!config.rope_proportional);
    // The window is the inclusive RADIUS of a symmetric band.
    try testing.expect(config.has_sliding_window);
    try testing.expectEqual(@as(u32, 512), config.sliding_window);
    // Five sliding layers then one full layer; the full layers are wider (hd 512, one KV head).
    for (0..24) |i| {
        const li: u32 = @intCast(i);
        const full = i % 6 == 5;
        try testing.expectEqual(full, config.isGlobalLayer(li));
        try testing.expectEqual(@as(u32, if (full) 512 else 256), config.layerHeadDim(li));
        try testing.expectEqual(@as(u32, if (full) 1 else 2), config.layerKVHeads(li));
    }
    // The image tower is a plain (unclipped, unstandardized) Gemma 4 SigLIP tower; the audio tower is not served.
    try testing.expect(config.has_vision);
    try testing.expectEqual(@as(u32, 768), config.vision_hidden_size);
    try testing.expectEqual(@as(u32, 16), config.vision_num_layers);
    try testing.expectEqual(@as(u32, 12), config.vision_num_heads);
    try testing.expectEqual(@as(u32, 64), config.vision_head_dim);
    try testing.expectEqual(@as(u32, 16), config.vision_patch_size);
    try testing.expectEqual(@as(u32, 3), config.vision_pooling_kernel);
    try testing.expectEqual(@as(u32, 10240), config.vision_position_embedding_size);
    try testing.expectEqual(@as(f32, 100.0), config.vision_rope_theta);
    try testing.expect(!config.vision_use_clipped_linears);
    try testing.expectEqual(@as(u32, 258880), config.image_token_id);
    try testing.expectEqual(@as(u32, 258884), config.video_token_id);
    try testing.expectEqual(@as(u32, 255999), config.boi_token_id);
    try testing.expectEqual(@as(u32, 258882), config.eoi_token_id);
    // Without a processor_config.json the budgets are the processors' own defaults.
    try testing.expectEqual(@as(u32, 280), config.gv_image_tokens);
    try testing.expectEqual(@as(u32, 70), config.gv_video_tokens);
    try testing.expectEqual(@as(u32, 32), config.gv_video_frames);
}

test "parseSoftTokenBudgetsFromJson reads the image and video processors of an EmbeddingGemma 2 processor_config" {
    // The release's file: images 280 tokens, video frames 140, at most 32 frames.
    const release =
        \\{"image_processor": {"image_processor_type": "Gemma4ImageProcessor", "max_soft_tokens": 280, "patch_size": 16},
        \\ "video_processor": {"max_soft_tokens": 140, "max_frames": 32, "fps": 1}, "processor_class": "EmbeddingGemma2Processor"}
    ;
    const got = parseSoftTokenBudgetsFromJson(testing.allocator, release);
    try testing.expectEqual(@as(?u32, 280), got.image);
    try testing.expectEqual(@as(?u32, 140), got.video);
    try testing.expectEqual(@as(?u32, 32), got.max_frames);

    // The processors accept only these budgets; anything else keeps the default instead of failing every request.
    const odd = parseSoftTokenBudgetsFromJson(testing.allocator,
        \\{"image_processor": {"max_soft_tokens": 300}, "video_processor": {"max_soft_tokens": 1121, "max_frames": 0}}
    );
    try testing.expectEqual(@as(?u32, null), odd.image);
    try testing.expectEqual(@as(?u32, null), odd.video);
    try testing.expectEqual(@as(?u32, null), odd.max_frames);
    for ([_]u32{ 70, 140, 280, 560, 1120 }) |budget| {
        var buf: [96]u8 = undefined;
        const json = try std.fmt.bufPrint(&buf, "{{\"image_processor\": {{\"max_soft_tokens\": {d}}}}}", .{budget});
        try testing.expectEqual(@as(?u32, budget), parseSoftTokenBudgetsFromJson(testing.allocator, json).image);
    }
    try testing.expectEqual(@as(?u32, null), parseSoftTokenBudgetsFromJson(testing.allocator, "not json").image);

    // The file as Google ships it, audio feature extractor and all: a toy-sized buffer does not hold its parse.
    const real = parseSoftTokenBudgetsFromJson(testing.allocator, @embedFile("fixtures/embedding_gemma2_processor_config.json"));
    try testing.expectEqual(@as(?u32, 280), real.image);
    try testing.expectEqual(@as(?u32, 140), real.video);
    try testing.expectEqual(@as(?u32, 32), real.max_frames);
}

test "parseConfig: an EmbeddingGemma 2 directory's processor_config.json sets the image and video budgets" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = embedding_gemma2_config_json });
    try tmp.dir.writeFile(io, .{ .sub_path = "processor_config.json", .data =
        \\{"image_processor": {"max_soft_tokens": 560}, "video_processor": {"max_soft_tokens": 140, "max_frames": 16}}
    });
    var path_buf: [512]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const config = try parseConfig(io, testing.allocator, path);
    try testing.expectEqual(@as(u32, 560), config.gv_image_tokens);
    try testing.expectEqual(@as(u32, 140), config.gv_video_tokens);
    try testing.expectEqual(@as(u32, 16), config.gv_video_frames);
}

test "ModelConfig: embedding_gemma2 reads a list-shaped eos_token_id" {
    // HF types it `int | list[int]`; the first id is the one input wrapping appends.
    const json = try std.mem.replaceOwned(u8, testing.allocator, embedding_gemma2_config_json, "\"eos_token_id\": 1,", "\"eos_token_id\": [1, 106],");
    defer testing.allocator.free(json);
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expectEqual(@as(u32, 1), config.eos_token_ids[0]);
}

test "ModelConfig: embedding_gemma2 keeps a smaller trained window than the card" {
    const json = try std.mem.replaceOwned(u8, testing.allocator, embedding_gemma2_config_json, "\"max_position_embeddings\": 262144", "\"max_position_embeddings\": 64");
    defer testing.allocator.free(json);
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 64), config.max_position_embeddings);
}

test "ModelConfig: embedding_gemma2 refuses per-layer geometry it does not model" {
    // The loader reads one geometry for every full-attention layer and the base one for the rest.
    const full = "\"head_dim\": 512, \"num_key_value_heads\": 1}";
    const cases = [_]struct { from: []const u8, to: []const u8 }{
        // a sliding layer with a geometry of its own
        .{ .from = "\"05\": {", .to = "\"04\": {" ++ full ++ ", \"05\": {" },
        // a full layer left on the base geometry
        .{ .from = ", \"23\": {" ++ full, .to = "" },
        // two full layers that disagree
        .{ .from = "\"11\": {\"head_dim\": 512", .to = "\"11\": {\"head_dim\": 256" },
    };
    for (cases) |c| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, embedding_gemma2_config_json, c.from, c.to);
        defer testing.allocator.free(json);
        try testing.expectError(error.UnsupportedLayerGeometry, parseConfigFromJson(testing.allocator, json));
    }
}

test "ModelConfig fills HF gemma3 defaults when text_config omits head counts" {
    // gemma-3-4b-it-4bit's text_config carries hidden_size/num_hidden_layers but
    // OMITS num_attention_heads/num_key_value_heads/head_dim, relying on the HF
    // Gemma3TextConfig defaults (8 q-heads / 4 kv-heads / head_dim 256). Our
    // struct defaults are the 12b/27b shape (16 q / 8 kv), so without an explicit
    // fill the Q projection (8*256=2048) gets reshaped against 16 heads and the
    // model crashes at warmup with "Cannot reshape array of size 2048 into shape
    // (1,1,16,256)" (issue #43). The 12b config ships these fields explicitly, so
    // it was never affected.
    const json =
        \\{
        \\  "model_type": "gemma3",
        \\  "text_config": {
        \\    "model_type": "gemma3_text",
        \\    "hidden_size": 2560,
        \\    "intermediate_size": 10240,
        \\    "num_hidden_layers": 34,
        \\    "sliding_window": 1024,
        \\    "rms_norm_eps": 1e-06
        \\  },
        \\  "vision_config": {"hidden_size": 1152},
        \\  "quantization": {"bits": 4, "group_size": 32}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("gemma3", config.model_type);
    // HF Gemma3TextConfig defaults — NOT our 12b-shaped struct defaults (16/8).
    try testing.expectEqual(@as(u32, 8), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 4), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 256), config.head_dim);
    // Fields the 4b text_config DOES carry must still win.
    try testing.expectEqual(@as(u32, 2560), config.hidden_size);
    try testing.expectEqual(@as(u32, 34), config.num_hidden_layers);
}

test "ModelConfig parses Qwen3.5 vision tower + interleaved M-RoPE" {
    // A minimal qwen3_5 VL config.json: distinct vision_config keys (depth/num_heads/
    // spatial_merge_size/...) land in qv_*, the interleaved M-RoPE sections come from
    // text_config.rope_parameters, and the three Qwen vision token ids are read from
    // the top level. has_vision must be set (the model IS multimodal now).
    const json =
        \\{
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 248056,
        \\  "video_token_id": 248057,
        \\  "vision_start_token_id": 248053,
        \\  "vision_end_token_id": 248054,
        \\  "text_config": {
        \\    "hidden_size": 1024,
        \\    "head_dim": 256,
        \\    "num_attention_heads": 8,
        \\    "num_key_value_heads": 2,
        \\    "full_attention_interval": 4,
        \\    "rope_parameters": {
        \\      "rope_theta": 10000000,
        \\      "partial_rotary_factor": 0.25,
        \\      "mrope_interleaved": true,
        \\      "mrope_section": [11, 11, 10]
        \\    }
        \\  },
        \\  "vision_config": {
        \\    "model_type": "qwen3_5",
        \\    "depth": 12,
        \\    "hidden_size": 768,
        \\    "num_heads": 12,
        \\    "intermediate_size": 3072,
        \\    "patch_size": 16,
        \\    "temporal_patch_size": 2,
        \\    "spatial_merge_size": 2,
        \\    "num_position_embeddings": 2304,
        \\    "out_hidden_size": 1024
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 32}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.has_vision);
    try testing.expect(config.qwen_vision);
    try testing.expectEqual(@as(u32, 12), config.qv_depth);
    try testing.expectEqual(@as(u32, 768), config.qv_hidden);
    try testing.expectEqual(@as(u32, 12), config.qv_heads);
    try testing.expectEqual(@as(u32, 64), config.qv_head_dim); // 768 / 12
    try testing.expectEqual(@as(u32, 2), config.qv_merge);
    try testing.expectEqual(@as(u32, 2), config.qv_temporal_patch);
    try testing.expectEqual(@as(u32, 2304), config.qv_num_pos_emb);
    try testing.expectEqual(@as(u32, 1024), config.qv_out_hidden);
    try testing.expect(config.mrope_interleaved);
    try testing.expectEqual([3]u32{ 11, 11, 10 }, config.mrope_section);
    try testing.expectEqual(@as(u32, 248056), config.image_token_id);
    try testing.expectEqual(@as(u32, 248057), config.video_token_id);
    try testing.expectEqual(@as(u32, 248053), config.vision_start_token_id);
    try testing.expectEqual(@as(u32, 248054), config.vision_end_token_id);
    // partial_rotary_factor → rotary_dim = 256*0.25 = 64.
    try testing.expectApproxEqAbs(@as(f32, 0.25), config.partial_rotary_factor, 1e-6);
}

test "parseVisionProcessorDefaultsFromJson supports current and legacy Qwen layouts" {
    const current = parseVisionProcessorDefaultsFromJson(
        \\{"image_processor":{"min_pixels":65536,"max_pixels":16777216}}
    );
    try testing.expectEqual(@as(?u32, 65536), current.min_pixels);
    try testing.expectEqual(@as(?u32, 16777216), current.max_pixels);

    const legacy = parseVisionProcessorDefaultsFromJson(
        \\{"size":{"shortest_edge":3136,"longest_edge":1003520}}
    );
    try testing.expectEqual(@as(?u32, 3136), legacy.min_pixels);
    try testing.expectEqual(@as(?u32, 1003520), legacy.max_pixels);
}

test "parseVisionProcessorDefaultsFromJson rejects invalid values and ranges" {
    const reversed = parseVisionProcessorDefaultsFromJson(
        \\{"image_processor":{"min_pixels":4096,"max_pixels":1024}}
    );
    try testing.expectEqual(@as(?u32, null), reversed.min_pixels);
    try testing.expectEqual(@as(?u32, null), reversed.max_pixels);

    const invalid = parseVisionProcessorDefaultsFromJson(
        \\{"image_processor":{"min_pixels":0,"max_pixels":4294967296}}
    );
    try testing.expectEqual(@as(?u32, null), invalid.min_pixels);
    try testing.expectEqual(@as(?u32, null), invalid.max_pixels);

    const malformed = parseVisionProcessorDefaultsFromJson("not json");
    try testing.expectEqual(@as(?u32, null), malformed.min_pixels);
    try testing.expectEqual(@as(?u32, null), malformed.max_pixels);
}

test "parseConfig prefers processor_config and fills missing Qwen bounds from preprocessor_config" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "qwen3_5",
        \\  "text_config": {"hidden_size": 1024, "head_dim": 128},
        \\  "vision_config": {
        \\    "depth": 1,
        \\    "hidden_size": 64,
        \\    "num_heads": 1,
        \\    "patch_size": 16,
        \\    "temporal_patch_size": 2,
        \\    "spatial_merge_size": 2,
        \\    "out_hidden_size": 1024
        \\  }
        \\}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
    try tmp.dir.writeFile(io, .{
        .sub_path = "processor_config.json",
        .data = "{\"image_processor\":{\"min_pixels\":65536}}",
    });
    try tmp.dir.writeFile(io, .{
        .sub_path = "preprocessor_config.json",
        .data = "{\"size\":{\"shortest_edge\":3136,\"longest_edge\":16777216}}",
    });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const config = try parseConfig(io, testing.allocator, path_buf[0..path_len]);
    try testing.expect(config.qwen_vision);
    try testing.expectEqual(@as(u32, 65536), config.qv_min_pixels);
    try testing.expectEqual(@as(u32, 16777216), config.qv_max_pixels);
}

test "ModelConfig prism_hadamard_qwen35 is qwen3_5 with the module Hadamard block" {
    const json =
        \\{
        \\  "model_type": "prism_hadamard_qwen35",
        \\  "text_config": {"model_type": "qwen3_5_text", "hidden_size": 5120, "num_hidden_layers": 64},
        \\  "modules": [{"path": "lm_head", "block": 1024}, {"path": "model.layers.0.mlp.up_proj", "block": 1024}],
        \\  "quantization": {"bits": 2, "group_size": 128, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("qwen3_5_moe", config.model_type);
    try testing.expectEqual(@as(u32, 1024), config.hadamard_block);
    const mixed =
        \\{"model_type": "prism_hadamard_qwen35", "text_config": {"hidden_size": 5120},
        \\ "modules": [{"path": "a", "block": 1024}, {"path": "b", "block": 512}]}
    ;
    try testing.expectError(error.UnsupportedHadamardLayout, parseConfigFromJson(testing.allocator, mixed));
}

test "ModelConfig text-only qwen3_5 has no qwen_vision" {
    const json =
        \\{
        \\  "model_type": "qwen3_5_text",
        \\  "hidden_size": 1024,
        \\  "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(!config.qwen_vision);
    try testing.expect(!config.has_vision);
}

test "ModelConfig keeps explicit gemma3 head counts (12b)" {
    // Regression guard for the fix above: a gemma3 text_config that DOES ship
    // head counts must keep them, never get clobbered by the HF-default fill.
    const json =
        \\{
        \\  "model_type": "gemma3",
        \\  "text_config": {
        \\    "model_type": "gemma3_text",
        \\    "hidden_size": 3840,
        \\    "num_hidden_layers": 48,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "head_dim": 256,
        \\    "sliding_window": 1024
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 32}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 16), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 256), config.head_dim);
}

test "ModelConfig routes flat gemma3_text (Gemma3ForCausalLM) onto gemma3, model prefix, tied" {
    // mlx-community/gemma-3-12b-it-qat-abliterated-lm-4bit ships a FLAT config
    // (no text_config) with top-level model_type "gemma3_text", architectures
    // ["Gemma3ForCausalLM"], weights under "model.*", tied embeddings (no
    // lm_head tensor, tie_word_embeddings omitted). Before the fix the
    // top-level "gemma3_text" matched no arm and fell through to the
    // llama-family else branch → model_type "unknown", tie=false, no QK-norm —
    // and crashed at load with "MISSING WEIGHT: lm_head.weight".
    const json =
        \\{
        \\  "model_type": "gemma3_text",
        \\  "architectures": ["Gemma3ForCausalLM"],
        \\  "hidden_size": 3840,
        \\  "intermediate_size": 15360,
        \\  "num_hidden_layers": 48,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 256,
        \\  "sliding_window": 1024,
        \\  "rms_norm_eps": 1e-06,
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    // Collapsed onto "gemma3" so transformer.zig's gemma3 forward/binding fire.
    try testing.expectEqualStrings("gemma3", config.model_type);
    // Flat checkpoint → "model.*" prefix (NOT "language_model.model").
    try testing.expectEqualStrings("model", config.weight_prefix);
    // Gemma always ties; the abliterated checkpoint omits the flag, so default
    // it on — lm_head then resolves to the embedding table instead of crashing.
    try testing.expect(config.tie_word_embeddings);
    // Full gemma3 numeric arm, not the llama-family fallback.
    try testing.expect(config.has_qk_norm);
    try testing.expect(config.scale_embeddings);
    try testing.expect(config.has_pre_ff_norm);
    try testing.expect(config.norm_has_offset);
    // Explicit head counts preserved.
    try testing.expectEqual(@as(u32, 16), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 256), config.head_dim);
}

test "ModelConfig gemma3 merges <end_of_turn> (106) even with scalar eos_token_id: 1" {
    // The abliterated text-only checkpoint declares a SCALAR eos_token_id: 1
    // (and tokenizer_config eos_token <eos>=1), but its chat template ends
    // turns with <end_of_turn> (106). Gating the 106 add on num_eos_tokens==0
    // (defeated by the scalar 1) left 106 out of the stop set, so generation
    // leaked repeated "<end_of_turn>" into the visible content. The gemma3 arm
    // must merge 106 additively (Qwen2.5-Coder <|im_end|> leak class).
    const json =
        \\{
        \\  "model_type": "gemma3_text",
        \\  "hidden_size": 3840,
        \\  "num_hidden_layers": 48,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 256,
        \\  "eos_token_id": 1,
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.isEosToken(1)); // config-declared eos preserved
    try testing.expect(config.isEosToken(106)); // <end_of_turn> merged in
}

test "ensureGemmaTerminators is additive and dedup-guarded" {
    var c = ModelConfig{};
    c.addEosToken(1); // config-provided scalar eos
    c.ensureGemmaTerminators();
    try testing.expect(c.isEosToken(1));
    try testing.expect(c.isEosToken(106));
    try testing.expectEqual(@as(u32, 2), c.num_eos_tokens);
    // Idempotent: re-running adds nothing.
    c.ensureGemmaTerminators();
    try testing.expectEqual(@as(u32, 2), c.num_eos_tokens);
}

test "ModelConfig multimodal gemma3 keeps language_model.model prefix" {
    // The gemma3 weight prefix is now conditional on text_config presence;
    // guard that a multimodal checkpoint (vision_config + nested text_config)
    // still nests its weights under "language_model.model".
    const json =
        \\{
        \\  "model_type": "gemma3",
        \\  "text_config": {"model_type": "gemma3_text", "hidden_size": 3840, "num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256, "sliding_window": 1024},
        \\  "vision_config": {"hidden_size": 1152},
        \\  "quantization": {"bits": 4, "group_size": 32}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("gemma3", config.model_type);
    try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    try testing.expect(config.tie_word_embeddings);
}

test "ModelConfig parses gemma4_unified vision + audio multimodal fields" {
    // The 12B unified config.json carries top-level vision_config/audio_config
    // with encoder-free dims plus image/audio/boi/boa/eoi/eoa token ids. These
    // drive the UnifiedEmbedder forward and placeholder insertion.
    const json =
        \\{
        \\  "model_type": "gemma4_unified",
        \\  "image_token_id": 258880,
        \\  "audio_token_id": 258881,
        \\  "boi_token_id": 255999,
        \\  "eoi_token_id": 258882,
        \\  "boa_token_id": 256000,
        \\  "eoa_token_index": 258883,
        \\  "vision_config": {
        \\    "model_type": "gemma4_unified_vision",
        \\    "mm_embed_dim": 3840,
        \\    "mm_posemb_size": 1120,
        \\    "model_patch_size": 48,
        \\    "patch_size": 16,
        \\    "pooling_kernel_size": 3,
        \\    "num_soft_tokens": 280,
        \\    "output_proj_dims": 3840,
        \\    "rms_norm_eps": 1e-06
        \\  },
        \\  "audio_config": {
        \\    "model_type": "gemma4_unified_audio",
        \\    "audio_embed_dim": 640,
        \\    "output_proj_dims": 640,
        \\    "rms_norm_eps": 1e-06
        \\  },
        \\  "text_config": {
        \\    "hidden_size": 3840,
        \\    "num_hidden_layers": 48,
        \\    "head_dim": 256
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.is_gemma4_unified);
    try testing.expect(config.has_vision);
    // Vision (encoder-free) dims.
    try testing.expectEqual(@as(u32, 3840), config.vision_mm_embed_dim);
    try testing.expectEqual(@as(u32, 48), config.vision_model_patch_size);
    try testing.expectEqual(@as(u32, 1120), config.vision_mm_posemb_size);
    try testing.expectEqual(@as(u32, 280), config.vision_soft_tokens);
    try testing.expectEqual(@as(u32, 3), config.vision_pooling_kernel);
    // Audio.
    try testing.expectEqual(@as(u32, 640), config.audio_embed_dim);
    // Multimodal token ids.
    try testing.expectEqual(@as(u32, 258880), config.image_token_id);
    try testing.expectEqual(@as(u32, 258881), config.audio_token_id);
    try testing.expectEqual(@as(u32, 255999), config.boi_token_id);
    try testing.expectEqual(@as(u32, 258882), config.eoi_token_id);
    try testing.expectEqual(@as(u32, 256000), config.boa_token_id);
    try testing.expectEqual(@as(u32, 258883), config.eoa_token_id);
}

test "parseConfigFromJson mistral honors explicit head_dim (≠ hidden/heads), flat prefix, quant" {
    // Mistral-Small-24B-Instruct-2501-4bit. The distinguishing trait vs the
    // llama-family default is that head_dim (128) is EXPLICIT and does NOT
    // equal hidden_size/num_attention_heads (5120/32 = 160). The mistral arm
    // must HONOR the explicit value (null-check, never recompute) — recomputing
    // to 160 would corrupt the Q/K/V reshape. Red-on-revert if the arm ever
    // unconditionally sets head_dim = hidden/heads. Also pins flat "model"
    // prefix (text-only, not the multimodal "language_model.model"), quant_bits
    // from the top-level "quantization" block, layer count, and untied lm_head.
    const json =
        \\{
        \\  "model_type": "mistral",
        \\  "hidden_size": 5120,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "num_hidden_layers": 40,
        \\  "intermediate_size": 32768,
        \\  "vocab_size": 131072,
        \\  "rms_norm_eps": 1e-05,
        \\  "rope_theta": 100000000.0,
        \\  "tie_word_embeddings": false,
        \\  "quantization": {"group_size": 64, "bits": 4}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("mistral", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expectEqual(@as(u32, 128), config.head_dim); // honored, NOT 5120/32=160
    try testing.expectEqual(@as(u32, 32), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 40), config.num_hidden_layers);
    try testing.expectEqual(@as(u32, 131072), config.vocab_size);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expect(!config.tie_word_embeddings);
}

test "parseConfigFromJson dense bf16 qwen3_5_moe → quant_bits 0" {
    // A fully-dense bf16 checkpoint (e.g. Qwen3.6-35B-A3B-bf16) has NO
    // "quantization" key. quant_bits must stay 0 so the loader skips every
    // .scales/.biases fetch and the forward pass dispatches to plain matmul.
    const json =
        \\{
        \\  "model_type": "qwen3_5_moe",
        \\  "text_config": {
        \\    "hidden_size": 2048,
        \\    "head_dim": 256,
        \\    "num_hidden_layers": 40,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 2,
        \\    "num_experts": 256,
        \\    "num_experts_per_tok": 8,
        \\    "moe_intermediate_size": 512,
        \\    "attn_output_gate": true,
        \\    "tie_word_embeddings": false
        \\  }
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 0), config.quant_bits);
    try testing.expectEqualStrings("qwen3_5_moe", config.model_type);
    try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    try testing.expect(config.attn_output_gate);
    try testing.expect(config.isMoe());
    try testing.expectEqual(@as(u32, 256), config.num_experts);
}

test "parseConfigFromJson bailing_hybrid (Ling 3.0) KDA/MLA/MoE fields" {
    // rapid-mlx/Ling-3.0-tiny-MLX-4bit's config.json, trimmed to the keys the
    // arch actually reads. Every derived quantity here is load-bearing: the
    // linear_* block sizes the KDA state, the mla_* block sizes the MLA
    // projections and the KV cache's asymmetric K/V head dims, and the MoE
    // block picks the grouped (noaux_tc) router.
    const json =
        \\{
        \\  "model_type": "bailing_hybrid",
        \\  "hidden_size": 1536,
        \\  "intermediate_size": 4608,
        \\  "num_hidden_layers": 24,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 16,
        \\  "head_dim": 128,
        \\  "layer_group_size": 4,
        \\  "short_conv_kernel_size": 4,
        \\  "kda_lower_bound": -5,
        \\  "kda_safe_gate": true,
        \\  "q_lora_rank": 256,
        \\  "kv_lora_rank": 512,
        \\  "qk_nope_head_dim": 128,
        \\  "qk_rope_head_dim": 64,
        \\  "qk_head_dim": 192,
        \\  "v_head_dim": 128,
        \\  "gated_attention_proj_granularity_type": "head_wise",
        \\  "rope_interleave": true,
        \\  "rope_theta": 6000000,
        \\  "rms_norm_eps": 1e-06,
        \\  "num_experts": 128,
        \\  "num_experts_per_tok": 8,
        \\  "moe_intermediate_size": 512,
        \\  "moe_shared_expert_intermediate_size": 512,
        \\  "num_shared_experts": 1,
        \\  "first_k_dense_replace": 1,
        \\  "n_group": 8,
        \\  "topk_group": 4,
        \\  "norm_topk_prob": true,
        \\  "routed_scaling_factor": 2.5,
        \\  "score_function": "sigmoid",
        \\  "vocab_size": 157184,
        \\  "tie_word_embeddings": false,
        \\  "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("bailing_hybrid", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);

    // Hybrid layout: layer_group_size 4 ⇒ layers 3/7/11/15/19/23 are MLA,
    // every other layer is KDA.
    try testing.expectEqual(@as(u32, 4), config.full_attention_interval);
    try testing.expect(config.isLinearLayer(0));
    try testing.expect(config.isLinearLayer(2));
    try testing.expect(!config.isLinearLayer(3));
    try testing.expect(!config.isLinearLayer(23));
    try testing.expect(config.needsSsmEntries());

    // KDA geometry: one head per attention head, key dim == value dim == head_dim.
    try testing.expectEqual(@as(u32, 16), config.linear_num_key_heads);
    try testing.expectEqual(@as(u32, 16), config.linear_num_value_heads);
    try testing.expectEqual(@as(u32, 128), config.linear_key_head_dim);
    try testing.expectEqual(@as(u32, 128), config.linear_value_head_dim);
    try testing.expectEqual(@as(u32, 4), config.linear_conv_kernel_dim);
    try testing.expect(config.kda_vector_gate);
    try testing.expectEqual(@as(f32, -5), config.kda_gate_lower_bound);

    // MLA geometry.
    try testing.expectEqual(@as(u32, 256), config.mla_q_lora_rank);
    try testing.expectEqual(@as(u32, 512), config.mla_kv_lora_rank);
    try testing.expectEqual(@as(u32, 128), config.mla_qk_nope_head_dim);
    try testing.expectEqual(@as(u32, 64), config.mla_qk_rope_head_dim);
    try testing.expectEqual(@as(u32, 128), config.mla_v_head_dim);
    try testing.expectEqual(@as(u32, 192), config.mlaQkHeadDim());
    try testing.expect(config.mla_head_gate);
    try testing.expect(config.rope_interleaved_pairs);
    try testing.expect(config.isMla());

    // MoE: sigmoid + expert bias + group-limited (noaux_tc) routing.
    try testing.expect(config.isMoe());
    try testing.expect(config.moe_sigmoid_router);
    try testing.expectEqual(@as(u32, 128), config.num_experts);
    try testing.expectEqual(@as(u32, 8), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 512), config.moe_intermediate_size);
    try testing.expectEqual(@as(u32, 512), config.shared_expert_intermediate_size);
    try testing.expectEqual(@as(u32, 1), config.first_k_dense_replace);
    try testing.expectEqual(@as(u32, 8), config.moe_n_group);
    try testing.expectEqual(@as(u32, 4), config.moe_topk_group);
    try testing.expect(config.moe_route_norm);
    try testing.expectEqual(@as(f32, 2.5), config.router_scaling_factor);

    // Attention scale is over the FULL qk head dim (192), not head_dim.
    try testing.expectEqual(@as(u32, 192), config.query_pre_attn_scalar);

    // A negative bound selects fla's bounded-sigmoid arm.
    try testing.expect(config.kdaUsesBoundedGate());
}

test "bailing_hybrid gate arm is SELECTED by the bound, never defaulted" {
    // `kdaGateChain` with bound 0 computes exp(0) = 1: a decay that never
    // forgets, on a checkpoint that merely omitted the key. The arms are fla's
    // two, and the absent case belongs to the softplus chain (which is
    // elementwise, so it serves a per-channel gate unchanged).
    var bounded = ModelConfig{ .model_type = "bailing_hybrid" };
    bounded.kda_vector_gate = true;
    bounded.kda_gate_lower_bound = -5;
    try testing.expect(bounded.kdaUsesBoundedGate());

    var unbounded = ModelConfig{ .model_type = "bailing_hybrid" };
    unbounded.kda_vector_gate = true; // per-channel gate, softplus form
    try testing.expect(!unbounded.kdaUsesBoundedGate());

    // A per-HEAD gate is never the bounded arm regardless of the field.
    var per_head = ModelConfig{ .model_type = "qwen3_5_moe" };
    per_head.kda_gate_lower_bound = -5;
    try testing.expect(!per_head.kdaUsesBoundedGate());
}

test "parseConfigFromJson bailing_hybrid refuses by NAME every variant it cannot serve" {
    // The arch's policy is refuse-loudly over serve-wrong: each key below
    // selects math this port does not implement, and each is at its harmless
    // value in every shipped mirror — so the ONLY thing standing between a
    // future variant and silently wrong output is this list. `use_kda_lora` is
    // the positive spelling of `no_kda_lora` (a checkpoint stating only that one
    // otherwise runs straight into a MISSING WEIGHT crash), and
    // `kda_lower_bound: 0` is the degenerate gate.
    const cases = [_][]const u8{
        "\"use_mla_nope\": true",
        "\"value_norm\": true",
        "\"up_proj_norm\": true",
        "\"use_nGPT\": true",
        "\"linear_silu\": false",
        "\"use_kda_lora\": true",
        "\"no_kda_lora\": false",
        "\"kda_lower_bound\": 0",
        "\"num_kv_heads_for_linear_attn\": 4",
        "\"score_function\": \"softmax\"",
        "\"gated_attention_proj_granularity_type\": \"element_wise\"",
        "\"qk_head_dim\": 256",
    };
    for (cases) |extra| {
        const json = try std.fmt.allocPrint(testing.allocator,
            \\{{
            \\  "model_type": "bailing_hybrid",
            \\  "hidden_size": 1536, "num_hidden_layers": 24,
            \\  "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 128,
            \\  "layer_group_size": 4,
            \\  "q_lora_rank": 256, "kv_lora_rank": 512,
            \\  "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            \\  "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
            \\  "vocab_size": 157184, {s}
            \\}}
        , .{extra});
        defer testing.allocator.free(json);
        try testing.expectError(error.UnsupportedBailingConfig, parseConfigFromJson(testing.allocator, json));
    }

    // And the shipped shape still loads: an ABSENT kda_lower_bound is the
    // softplus arm, not a refusal.
    const softplus =
        \\{
        \\  "model_type": "bailing_hybrid",
        \\  "hidden_size": 1536, "num_hidden_layers": 24,
        \\  "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 128,
        \\  "layer_group_size": 4,
        \\  "q_lora_rank": 256, "kv_lora_rank": 512,
        \\  "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        \\  "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        \\  "vocab_size": 157184, "num_kv_heads_for_linear_attn": 0
        \\}
    ;
    const ok = try parseConfigFromJson(testing.allocator, softplus);
    try testing.expect(ok.kda_vector_gate);
    try testing.expect(!ok.kdaUsesBoundedGate());
}

test "parseConfigFromJson quantized qwen3_5_moe → quant_bits from key" {
    // Same arch but with a "quantization" block: quant_bits must reflect it so
    // the mandatory scale/bias fetches still fire (a missing scale is a clear
    // MISSING WEIGHT error, not a silent dense fallback). Guards the default flip.
    const json =
        \\{
        \\  "model_type": "qwen3_5_moe",
        \\  "text_config": {"hidden_size": 2048, "num_experts": 256},
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
}

test "a qwen3_5_moe trunk batches decode: its only per-slot state is the GDN pair" {
    // Bar: routed experts are row-generic (the sorted gather path takes B*S rows),
    // so a qwen3_5 MoE batches like the dense trunk; MoE trunks with other
    // per-slot state (hy3, laguna, lfm2_moe, bailing) stay refused.
    const json =
        \\{
        \\  "model_type": "qwen3_5_moe",
        \\  "text_config": {"hidden_size": 2048, "num_experts": 256, "full_attention_interval": 4}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.isMoe());
    try testing.expect(config.supportsBatchedGdnDecode());

    var laguna = std.mem.zeroes(ModelConfig);
    laguna.model_type = "laguna";
    laguna.num_experts = 64;
    laguna.full_attention_interval = 4;
    try testing.expect(!laguna.supportsBatchedGdnDecode());
}

test "parseConfigFromJson rejects affine bits MLX has no kernels for" {
    // A checkpoint declaring an affine bit-width outside MLX's kernel set
    // ({2,3,4,5,6,8}) must fail at PARSE, not at warmup: mlx only validates
    // bits inside quantize(), so an already-quantized 1-bit checkpoint sails
    // through load and dies with an uncatchable Metal kernel-load error
    // ("Unable to load kernel affine_dequantize_..._b_1") that kills the
    // whole server. Live bite: prism-ml/Bonsai-27B-mlx-1bit.
    const json_1bit =
        \\{
        \\  "model_type": "qwen3_5",
        \\  "text_config": {"hidden_size": 5120},
        \\  "quantization": {"bits": 1, "group_size": 128}
        \\}
    ;
    try testing.expectError(error.UnsupportedQuantBits, parseConfigFromJson(testing.allocator, json_1bit));

    const json_7bit =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"bits": 7, "group_size": 64}
        \\}
    ;
    try testing.expectError(error.UnsupportedQuantBits, parseConfigFromJson(testing.allocator, json_7bit));
}

test "parseConfigFromJson nvfp4 quantization mode" {
    // NVFP4 checkpoints (issue #24): {"group_size": 16, "bits": 4, "mode": "nvfp4"}.
    // The mode must land on config.quant_mode so the loader skips the .biases
    // fetches (nvfp4 stores no biases tensors) and the matmul call sites pass
    // "nvfp4" to mlx instead of "affine".
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(QuantMode.nvfp4, config.quant_mode);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 16), config.quant_group_size);
    try testing.expect(!config.quant_mode.hasBiases());
}

test "parseConfigFromJson explicit affine mode keeps biases" {
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
    try testing.expect(config.quant_mode.hasBiases());
}

test "parseConfigFromJson unknown quantization mode → error" {
    // An unrecognized mode must fail loudly at config parse — not crash later
    // in the weight loader with a misleading MISSING WEIGHT error.
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 32, "bits": 4, "mode": "fp99"}
        \\}
    ;
    try testing.expectError(error.UnsupportedQuantMode, parseConfigFromJson(testing.allocator, json));
}

test "parseConfigFromJson qwen3_moe (Qwen3-30B-A3B) → MoE, no shared expert, no output gate" {
    // Qwen3-Coder-30B-A3B / Qwen3-30B-A3B ship model_type "qwen3_moe": a pure
    // full-attention MoE (no GatedDeltaNet) that DROPPED the shared expert that
    // Qwen2-MoE / Qwen3.5-MoE carry (shared_expert_intermediate_size: 0, no
    // mlp.shared_expert.* weights). It must NOT be remapped onto qwen3_5_moe
    // (which assumes a shared expert and an attention output gate) — doing so
    // crashed at load with "MISSING WEIGHT: ...mlp.shared_expert.gate_proj.weight".
    const json =
        \\{
        \\  "model_type": "qwen3_moe",
        \\  "hidden_size": 2048,
        \\  "head_dim": 128,
        \\  "num_hidden_layers": 48,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 4,
        \\  "num_experts": 128,
        \\  "num_experts_per_tok": 8,
        \\  "moe_intermediate_size": 768,
        \\  "shared_expert_intermediate_size": 0,
        \\  "use_qk_norm": true,
        \\  "use_sliding_window": false,
        \\  "rope_theta": 10000000,
        \\  "tie_word_embeddings": false,
        \\  "quantization": {"bits": 8, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("qwen3_moe", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expect(config.isMoe());
    try testing.expectEqual(@as(u32, 128), config.num_experts);
    try testing.expectEqual(@as(u32, 8), config.num_experts_per_tok);
    // qwen3 attention: QK-norm on, NO output gate (that's a qwen3_5 thing).
    try testing.expect(config.has_qk_norm);
    try testing.expect(!config.attn_output_gate);
    // Full attention everywhere — no GatedDeltaNet/linear layers.
    try testing.expect(!config.isLinearLayer(0));
    try testing.expect(!config.isLinearLayer(3));
    try testing.expect(!config.has_hybrid_layers);
    try testing.expect(!config.has_sliding_window);
    try testing.expectEqual(@as(u32, 8), config.quant_bits);
}

test "parseConfigFromJson hy_v3 (Tencent Hunyuan 3 295B-A21B) → sigmoid-router MoE, first-k dense, shared expert" {
    // tencent/Hy3 (July 2026): pure full-attention MoE, GQA 64/8 hd-128 with
    // QK-norm, 192 experts top-8 + 1 ungated shared expert, DeepSeek-V3-style
    // sigmoid router with expert bias + top-k renorm + scaling factor, and
    // layer 0 dense (first_k_dense_replace). Real checkpoints (ox-ox MLX
    // conversion) ship NO eos/bos in config.json and NO generation_config.json
    // — the eos (<｜hy_eos:opensource｜> = 120025) must be filled by the arm or
    // generation never stops. qk_norm/route_norm are ABSENT in the real config
    // and default true (mlx-lm hy_v3 ModelArgs defaults).
    const json =
        \\{
        \\  "model_type": "hy_v3",
        \\  "vocab_size": 120832,
        \\  "hidden_size": 4096,
        \\  "intermediate_size": 13312,
        \\  "num_hidden_layers": 80,
        \\  "num_attention_heads": 64,
        \\  "num_key_value_heads": 8,
        \\  "head_dim": 128,
        \\  "max_position_embeddings": 262144,
        \\  "rms_norm_eps": 1e-05,
        \\  "rope_theta": 11158840.0,
        \\  "tie_word_embeddings": false,
        \\  "num_experts": 192,
        \\  "num_experts_per_tok": 8,
        \\  "moe_intermediate_size": 1536,
        \\  "num_shared_experts": 1,
        \\  "first_k_dense_replace": 1,
        \\  "router_scaling_factor": 2.826,
        \\  "moe_router_use_sigmoid": true,
        \\  "moe_router_enable_expert_bias": true,
        \\  "num_nextn_predict_layers": 1,
        \\  "expert_hidden_dim": 1536,
        \\  "quantization": {"bits": 2, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("hy_v3", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expect(config.isMoe());
    try testing.expectEqual(@as(u32, 192), config.num_experts);
    try testing.expectEqual(@as(u32, 8), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 1536), config.moe_intermediate_size);
    // 1 shared expert × expert_hidden_dim — there is no explicit
    // shared_expert_intermediate_size key in hy_v3 configs.
    try testing.expectEqual(@as(u32, 1536), config.shared_expert_intermediate_size);
    try testing.expectEqual(@as(u32, 1), config.first_k_dense_replace);
    // Sigmoid router + bias + renorm + scaling factor.
    try testing.expect(config.moe_sigmoid_router);
    try testing.expect(config.moe_route_norm);
    try testing.expectApproxEqAbs(@as(f32, 2.826), config.router_scaling_factor, 1e-6);
    // Attention: qwen3-shaped — QK-norm (default-true when key absent), no
    // output gate, full rotary, scale = head_dim^-0.5.
    try testing.expect(config.has_qk_norm);
    try testing.expect(!config.attn_output_gate);
    try testing.expect(!config.has_sliding_window);
    try testing.expect(!config.has_hybrid_layers);
    try testing.expect(!config.isLinearLayer(0));
    try testing.expectEqual(@as(u32, 128), config.head_dim);
    try testing.expectEqual(@as(u32, 128), config.query_pre_attn_scalar);
    try testing.expectApproxEqAbs(@as(f32, 1.0), config.partial_rotary_factor, 1e-6);
    try testing.expectEqual(HiddenAct.silu, config.hidden_act);
    try testing.expect(!config.scale_embeddings);
    try testing.expect(!config.tie_word_embeddings);
    // eos fallback: config carries none; the arm must add 120025 or generation
    // never halts (same class as ensureGemmaTerminators).
    try testing.expect(config.isEosToken(120025));
    try testing.expectEqual(@as(u32, 2), config.quant_bits);
    try testing.expectEqual(@as(u32, 262144), config.max_position_embeddings);
}

test "parseConfigFromJson hy_v3 explicit route_norm/qk_norm false are honored" {
    // The arm defaults route_norm/qk_norm TRUE when absent; explicit false in a
    // future checkpoint must win (never bake the default over a declared value).
    const json =
        \\{
        \\  "model_type": "hy_v3",
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 4,
        \\  "num_attention_heads": 8,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 128,
        \\  "num_experts": 16,
        \\  "num_experts_per_tok": 2,
        \\  "moe_intermediate_size": 256,
        \\  "num_shared_experts": 2,
        \\  "route_norm": false,
        \\  "qk_norm": false,
        \\  "eos_token_id": [7, 9]
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(!config.moe_route_norm);
    try testing.expect(!config.has_qk_norm);
    try testing.expect(config.moe_sigmoid_router);
    // 2 shared experts × 256.
    try testing.expectEqual(@as(u32, 512), config.shared_expert_intermediate_size);
    // Declared eos survives; the 120025 merge is additive, never a replace.
    try testing.expect(config.isEosToken(7));
    try testing.expect(config.isEosToken(9));
    try testing.expect(config.isEosToken(120025));
    // router_scaling_factor absent → neutral 1.0.
    try testing.expectApproxEqAbs(@as(f32, 1.0), config.router_scaling_factor, 1e-6);
}

test "parseConfigFromJson qwen2 (Qwen2.5) → dense, no QK-norm, silu" {
    // Qwen2.5-Coder / Qwen2.5-Instruct ship model_type "qwen2": a dense
    // full-attention Llama-family arch that, unlike qwen3, has NO QK-norm and
    // DOES carry additive qkv-projection biases (q/k/v_proj.bias). The forward
    // applies those biases when present; here we pin the config classification.
    const json =
        \\{
        \\  "model_type": "qwen2",
        \\  "hidden_size": 5120,
        \\  "num_hidden_layers": 64,
        \\  "num_attention_heads": 40,
        \\  "num_key_value_heads": 8,
        \\  "intermediate_size": 27648,
        \\  "rms_norm_eps": 1e-6,
        \\  "rope_theta": 1000000.0,
        \\  "hidden_act": "silu",
        \\  "tie_word_embeddings": false,
        \\  "quantization": {"bits": 8, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("qwen2", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expect(!config.isMoe());
    // KEY difference from qwen3: no QK-norm.
    try testing.expect(!config.has_qk_norm);
    try testing.expect(!config.has_pre_ff_norm);
    try testing.expect(!config.scale_embeddings);
    try testing.expectEqual(HiddenAct.silu, config.hidden_act);
    try testing.expectEqual(@as(u32, 128), config.head_dim); // 5120 / 40
    try testing.expectEqual(@as(u32, 8), config.quant_bits);
}

test "ModelConfig parses diffusion_gemma (DiffusionGemma 26B-A4B block diffusion)" {
    // Faithful subset of mlx-community/diffusiongemma-26B-A4B-it-4bit's
    // config.json. The trunk is the Gemma 4 26B-A4B MoE decoder (dual FFN,
    // sigma-MoE router, v_norm, K=V alias on full layers, proportional RoPE)
    // under weight prefix `model.decoder`; the diffusion-specific knobs ride
    // in the embedded `generation_config` object plus top-level canvas_length.
    const json =
        \\{
        \\  "model_type": "diffusion_gemma",
        \\  "canvas_length": 256,
        \\  "eos_token_id": [1, 106, 50],
        \\  "tie_word_embeddings": true,
        \\  "generation_config": {
        \\    "confidence_threshold": 0.005,
        \\    "max_denoising_steps": 48,
        \\    "pad_token_id": 0,
        \\    "sampler_config": {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1},
        \\    "stability_threshold": 1,
        \\    "t_max": 0.8,
        \\    "t_min": 0.4
        \\  },
        \\  "text_config": {
        \\    "model_type": "diffusion_gemma_text",
        \\    "vocab_size": 262144,
        \\    "hidden_size": 2816,
        \\    "intermediate_size": 2112,
        \\    "moe_intermediate_size": 704,
        \\    "num_experts": 128,
        \\    "top_k_experts": 8,
        \\    "num_hidden_layers": 30,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "num_global_key_value_heads": 2,
        \\    "head_dim": 256,
        \\    "global_head_dim": 512,
        \\    "final_logit_softcapping": 30.0,
        \\    "hidden_activation": "gelu_pytorch_tanh",
        \\    "rms_norm_eps": 1e-06,
        \\    "max_position_embeddings": 262144,
        \\    "sliding_window": 1024,
        \\    "tie_word_embeddings": true,
        \\    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        \\    "rope_parameters": {
        \\      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        \\      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        \\    },
        \\    "use_bidirectional_attention": "vision"
        \\  },
        \\  "vision_config": {"model_type": "gemma4_vision", "hidden_size": 1152, "num_hidden_layers": 27},
        \\  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    // model_type stays distinct — it drives diffusion generation dispatch —
    // but the trunk inherits every gemma4 layer-structure flag.
    try testing.expectEqualStrings("diffusion_gemma", config.model_type);
    try testing.expectEqualStrings("model.decoder", config.weight_prefix);
    try testing.expect(config.isDiffusion());
    try testing.expect(config.has_v_norm);
    try testing.expect(config.has_pre_ff_norm);
    try testing.expect(config.has_qk_norm);
    try testing.expect(!config.norm_has_offset);
    try testing.expect(config.scale_embeddings);
    try testing.expect(config.tie_word_embeddings);
    // Full-attention layers ship no v_proj: V = param-free-norm(k_proj out).
    try testing.expect(config.attention_k_eq_v);
    // MoE trunk
    try testing.expect(config.isMoe());
    try testing.expectEqual(@as(u32, 128), config.num_experts);
    try testing.expectEqual(@as(u32, 8), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 704), config.moe_intermediate_size);
    // Dual head geometry + per-type RoPE
    try testing.expectEqual(@as(u32, 512), config.global_head_dim);
    try testing.expectEqual(@as(u32, 2), config.num_global_key_value_heads);
    try testing.expect(config.rope_proportional);
    try testing.expectApproxEqAbs(@as(f32, 0.25), config.partial_rotary_factor_global, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1000000.0), config.rope_theta, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 10000.0), config.rope_local_base_freq, 0.5);
    try testing.expect(config.has_explicit_layer_types);
    try testing.expect(config.isGlobalLayer(5));
    try testing.expect(!config.isGlobalLayer(4));
    try testing.expectEqual(@as(u32, 1024), config.sliding_window);
    try testing.expectApproxEqAbs(@as(f32, 30.0), config.final_logit_softcapping, 0.001);
    // EOS set {eos, end_of_turn, +1}
    try testing.expectEqual(@as(u32, 3), config.num_eos_tokens);
    try testing.expect(config.isEosToken(1));
    try testing.expect(config.isEosToken(106));
    try testing.expect(config.isEosToken(50));
    // Diffusion generation knobs from the embedded generation_config
    try testing.expectEqual(@as(u32, 256), config.canvas_length);
    try testing.expectEqual(@as(u32, 48), config.diffusion_max_steps);
    try testing.expectApproxEqAbs(@as(f32, 0.4), config.diffusion_t_min, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.8), config.diffusion_t_max, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.1), config.diffusion_entropy_bound, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.005), config.diffusion_confidence_threshold, 0.000001);
    try testing.expectEqual(@as(u32, 1), config.diffusion_stability_threshold);
    try testing.expectEqual(@as(u32, 0), config.diffusion_pad_token);
    // Vision tower (model.encoder.vision_tower.*) is not wired yet — the
    // diffusion arm must NOT advertise vision, or image requests would splice
    // embeddings into a tower-less forward.
    try testing.expect(!config.has_vision);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
}

test "shouldKeepWeightKey drops DiffusionGemma encoder vision tower (text-only v1)" {
    // DiffusionGemma nests its vision tower under model.encoder.* — distinct
    // from the bare vision_tower.* prefixes of earlier checkpoints. Until the
    // tower is wired, those tensors must be dropped even with load_vision on,
    // and ALWAYS dropped when vision is off.
    try testing.expect(!shouldKeepWeightKey("model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", false));
    try testing.expect(!shouldKeepWeightKey("model.encoder.embed_vision.embedding_projection.weight", false));
    // Trunk + diffusion weights always survive.
    try testing.expect(shouldKeepWeightKey("model.decoder.layers.0.experts.gate_up_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.decoder.self_conditioning.gate_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.encoder.language_model.layers.0.layer_scalar", false));
}

test "narrowsLoadedF16 catches per-channel tables, not matmul operands" {
    // Quant side tensors: the pre-existing rule, keyed on the suffix because
    // they can be 2-D.
    try testing.expect(narrowsLoadedF16("model.layers.0.mlp.down_proj.scales", 2, .float16));
    try testing.expect(narrowsLoadedF16("model.layers.0.mlp.down_proj.biases", 2, .float16));

    // Any 1-D f16 tensor is a PER-CHANNEL table — a norm weight, a bias, a
    // gate table. It gets multiplied or added straight into the activation
    // stream, so leaving it f16 beside a bf16 residual promotes the residual
    // (and therefore every later weight read) to f32.
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.input_layernorm.weight", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.linear_attn.A_log", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.linear_attn.dt_bias", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.norm.weight", 1, .float16));

    // A 2-D dense f16 weight is a MATMUL OPERAND, not a table. MLX picks its
    // kernel off that dtype, so narrowing it is a kernel-selection change and
    // not this rule's business — it stays per-site.
    try testing.expect(!narrowsLoadedF16("vision_tower.blocks.0.attn.qkv.weight", 2, .float16));
    try testing.expect(!narrowsLoadedF16("language_model.model.layers.0.linear_attn.conv1d.weight", 3, .float16));

    // Everything already in the engine's dtype, and packed weights, are left
    // alone.
    try testing.expect(!narrowsLoadedF16("model.layers.0.input_layernorm.weight", 1, .bfloat16));
    try testing.expect(!narrowsLoadedF16("model.layers.0.mlp.down_proj.weight", 2, .uint32));
    try testing.expect(!narrowsLoadedF16("model.layers.0.mlp.down_proj.scales", 2, .bfloat16));
}

test "parseGenerationDefaultsFromJson: reads model sampling recommendations" {
    // Verbatim shape of Qwen3.6 / Gemma 4 checkpoints' generation_config.json.
    const json =
        \\{"bos_token_id": 248044, "do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
    ;
    const gd = parseGenerationDefaultsFromJson(json);
    try testing.expectEqual(@as(?f32, 1.0), gd.temperature);
    try testing.expectEqual(@as(?f32, 0.95), gd.top_p);
    try testing.expectEqual(@as(?u32, 20), gd.top_k);
}

test "parseGenerationDefaultsFromJson: min_p reads in range, rejects out of range" {
    try testing.expectEqual(@as(?f32, 0.05), parseGenerationDefaultsFromJson("{\"min_p\":0.05}").min_p);
    try testing.expectEqual(@as(?f32, 0), parseGenerationDefaultsFromJson("{\"min_p\":0}").min_p);
    try testing.expectEqual(@as(?f32, null), parseGenerationDefaultsFromJson("{\"min_p\":-0.1}").min_p);
    try testing.expectEqual(@as(?f32, null), parseGenerationDefaultsFromJson("{\"min_p\":1.5}").min_p);
}

test "pooling: config.json pooling_mode key parses; unknown value rejected at parse" {
    // Explicit converter/operator contract for checkpoints whose config alone
    // can't reveal pooling (Qwen3-Embedding declares plain `qwen3`).
    const base = "{{\"model_type\":\"qwen3\",\"hidden_size\":64,\"num_attention_heads\":8,\"num_hidden_layers\":2,\"pooling_mode\":\"{s}\"}}";
    inline for (.{ .{ "last_token", PoolingMode.last_token }, .{ "cls", PoolingMode.cls }, .{ "mean", PoolingMode.mean } }) |case| {
        const json = try std.fmt.allocPrint(testing.allocator, base, .{case[0]});
        defer testing.allocator.free(json);
        const config = try parseConfigFromJson(testing.allocator, json);
        try testing.expectEqual(@as(?PoolingMode, case[1]), config.pooling_mode);
        try testing.expect(config.hasEmbeddingCapability());
        try testing.expect(!config.is_encoder_only); // pooling never flips the arch
    }
    // An unknown mode is a parse error, not a silent mean-pool: wrong-semantics
    // vectors are harder to detect than a refused load.
    const bad = try std.fmt.allocPrint(testing.allocator, base, .{"weighted_mean"});
    defer testing.allocator.free(bad);
    try testing.expectError(error.UnsupportedPoolingMode, parseConfigFromJson(testing.allocator, bad));
}

test "pooling: sentence-transformers 1_Pooling sidecar parses all three modes" {
    // Verbatim shape of ST `1_Pooling/config.json` (Qwen3-Embedding sets
    // lasttoken, bge/mxbai set cls_token, MiniLM sets mean_tokens).
    const last =
        \\{"word_embedding_dimension": 2560, "pooling_mode_cls_token": false,
        \\ "pooling_mode_mean_tokens": false, "pooling_mode_max_tokens": false,
        \\ "pooling_mode_mean_sqrt_len_tokens": false, "pooling_mode_lasttoken": true}
    ;
    try testing.expectEqual(@as(?PoolingMode, .last_token), try parsePoolingSidecar(last));
    const cls =
        \\{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false, "pooling_mode_lasttoken": false}
    ;
    try testing.expectEqual(@as(?PoolingMode, .cls), try parsePoolingSidecar(cls));
    const mean =
        \\{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}
    ;
    try testing.expectEqual(@as(?PoolingMode, .mean), try parsePoolingSidecar(mean));
}

test "pooling: a sentence-transformers 6 sidecar spells the mode as a string" {
    // google/embeddinggemma-2's own 1_Pooling/config.json; the other spellings are ST 6's names for the modes.
    const mean =
        \\{"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": true}
    ;
    try testing.expectEqual(@as(?PoolingMode, .mean), try parsePoolingSidecar(mean));
    try testing.expectEqual(@as(?PoolingMode, .cls), try parsePoolingSidecar("{\"pooling_mode\": \"cls\"}"));
    try testing.expectEqual(@as(?PoolingMode, .last_token), try parsePoolingSidecar("{\"pooling_mode\": \"lasttoken\"}"));
    // Modes we do not implement refuse the load, as the boolean spelling does.
    for ([_][]const u8{ "max", "mean_sqrt_len", "weightedmean" }) |mode| {
        var buf: [64]u8 = undefined;
        const json = try std.fmt.bufPrint(&buf, "{{\"pooling_mode\": \"{s}\"}}", .{mode});
        try testing.expectError(error.UnsupportedPoolingMode, parsePoolingSidecar(json));
    }
}

test "pooling: sidecar demanding an unsupported mode errors; non-pooling JSON is ignored" {
    // A sidecar that DOES declare pooling but none we implement (weighted-mean,
    // max) must refuse the load — mean-pooling it anyway is silent corruption.
    const unsupported =
        \\{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": false,
        \\ "pooling_mode_max_tokens": true, "pooling_mode_lasttoken": false}
    ;
    try testing.expectError(error.UnsupportedPoolingMode, parsePoolingSidecar(unsupported));
    // Malformed / unrelated JSON: best-effort null, like generation_config.json.
    try testing.expectEqual(@as(?PoolingMode, null), try parsePoolingSidecar("not json"));
    try testing.expectEqual(@as(?PoolingMode, null), try parsePoolingSidecar("{\"dimension\": 384}"));
}

test "pooling: known-family directory-name fallback" {
    // The mlx-community conversions ship NO sidecar and a plain chat
    // model_type, so a metadata-less checkpoint falls back to the family
    // table — gated on the arch so a name can never flip an unrelated model.
    try testing.expectEqual(@as(?PoolingMode, .last_token), poolingFromDirName("Qwen3-Embedding-4B-4bit-DWQ", "qwen3"));
    try testing.expectEqual(@as(?PoolingMode, .last_token), poolingFromDirName("qwen3-embedding-0.6b", "qwen3"));
    try testing.expectEqual(@as(?PoolingMode, null), poolingFromDirName("Qwen3-8B-4bit", "qwen3"));
    try testing.expectEqual(@as(?PoolingMode, null), poolingFromDirName("Qwen3-Embedding-4B", "llama"));
    // bge / mxbai are CLS-pooling BERTs (their cards say so); MiniLM stays mean.
    try testing.expectEqual(@as(?PoolingMode, .cls), poolingFromDirName("bge-small-en-v1.5-8bit", "bert"));
    try testing.expectEqual(@as(?PoolingMode, .cls), poolingFromDirName("mxbai-embed-large-v1", "bert"));
    try testing.expectEqual(@as(?PoolingMode, null), poolingFromDirName("all-MiniLM-L6-v2", "bert"));
    // EmbeddingGemma is mean-pooled via its own bidirectional path — the name
    // fallback must not touch non-qwen3 archs on the "embedding" substring.
    try testing.expectEqual(@as(?PoolingMode, null), poolingFromDirName("embeddinggemma-300m-8bit", "gemma3_text"));
}

test "pooling: effectivePooling defaults to mean; encoder capability unions" {
    var config = ModelConfig{};
    try testing.expectEqual(PoolingMode.mean, config.effectivePooling());
    try testing.expect(!config.hasEmbeddingCapability());
    config.is_encoder_only = true;
    try testing.expect(config.hasEmbeddingCapability());
    config.is_encoder_only = false;
    config.pooling_mode = .last_token;
    try testing.expectEqual(PoolingMode.last_token, config.effectivePooling());
    try testing.expect(config.hasEmbeddingCapability());
}

test "parseGenerationDefaultsFromJson: missing keys and malformed input give nulls" {
    const partial = parseGenerationDefaultsFromJson("{\"eos_token_id\": [1, 2]}");
    try testing.expectEqual(@as(?f32, null), partial.temperature);
    try testing.expectEqual(@as(?f32, null), partial.top_p);
    try testing.expectEqual(@as(?u32, null), partial.top_k);

    const broken = parseGenerationDefaultsFromJson("not json at all");
    try testing.expectEqual(@as(?f32, null), broken.temperature);

    // Out-of-range values are dropped, not clamped — a corrupt config must
    // not silently pin sampling to an extreme.
    const insane = parseGenerationDefaultsFromJson("{\"temperature\": 99.0, \"top_p\": 7.0, \"top_k\": -5}");
    try testing.expectEqual(@as(?f32, null), insane.temperature);
    try testing.expectEqual(@as(?f32, null), insane.top_p);
    try testing.expectEqual(@as(?u32, null), insane.top_k);
}

test "attnCacheLayerCount: a layer_block_types hybrid counts only its ATTENTION layers" {
    // LFM2.5-2.6B's real shape: 30 layers, 22 gated-conv + 8 full-attention,
    // 8 KV heads at head_dim 64. `isLinearLayer` keys on
    // `full_attention_interval`, which this family never sets (it populates
    // `layer_block_types` instead), so every memory estimate billed a KV cache
    // for all 30 — 3.75x the bytes the model can ever store, spent out of the
    // auto-context budget on exactly the arch small Macs are pointed at.
    // Nemotron-H is the same class through `hybrid_override_pattern` (only its
    // `*` layers cache) and over-bills harder still.
    var config = ModelConfig{};
    config.num_hidden_layers = 30;
    config.num_key_value_heads = 8;
    config.head_dim = 64;
    config.has_hybrid_layers = true;
    // The shipped LFM2.5-2.6B layer_types, verbatim.
    const lfm2_attn = [_]u32{ 2, 5, 9, 13, 17, 21, 24, 27 };
    for (0..30) |i| config.layer_block_types[i] = .gated_conv;
    for (lfm2_attn) |i| config.layer_block_types[i] = .attention;
    try testing.expectEqual(@as(u32, 8), config.attnCacheLayerCount());
    try testing.expectEqual(@as(u64, 8 * 8 * 2 * 64 * 2), config.kvBytesPerToken());

    // Nemotron-H: mamba2 and mlp blocks hold a fixed-size recurrent state, not
    // a per-token cache — only the attention blocks are billed.
    var nemo = ModelConfig{};
    nemo.num_hidden_layers = 12;
    nemo.num_key_value_heads = 8;
    nemo.head_dim = 128;
    nemo.has_hybrid_layers = true;
    const pattern = [_]LayerBlockType{ .mamba2, .mlp, .mamba2, .attention, .mamba2, .mlp, .mamba2, .mlp, .mamba2, .attention, .mamba2, .mlp };
    for (pattern, 0..) |b, i| nemo.layer_block_types[i] = b;
    try testing.expectEqual(@as(u32, 2), nemo.attnCacheLayerCount());

    // A hybrid checkpoint that ships no layer_types leaves the array at its
    // `.attention` default and keeps the whole-model bill — the safe direction.
    var bare = ModelConfig{};
    bare.num_hidden_layers = 16;
    bare.num_key_value_heads = 4;
    bare.head_dim = 128;
    bare.has_hybrid_layers = true;
    try testing.expectEqual(@as(u32, 16), bare.attnCacheLayerCount());
}

test "nemotron_h: a layers_block_type LIST sets the per-layer blocks like hybrid_override_pattern" {
    // Nemotron 3.5 configs ship `layers_block_type` as a list of names and no
    // pattern string; unread, every layer keeps the `.attention` default.
    const json =
        \\{
        \\  "model_type": "nemotron_h",
        \\  "hidden_size": 2688, "num_hidden_layers": 7,
        \\  "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        \\  "vocab_size": 131072,
        \\  "layers_block_type": ["mamba", "moe", "mamba", "attention", "mlp", "mamba", "moe"],
        \\  "mtp_layers_block_type": ["attention", "moe"]
        \\}
    ;
    const cfg = try parseConfigFromJson(testing.allocator, json);
    const want = [_]LayerBlockType{ .mamba2, .moe, .mamba2, .attention, .mlp, .mamba2, .moe };
    for (want, 0..) |b, i| try testing.expectEqual(b, cfg.layer_block_types[i]);
    try testing.expectEqual(@as(u32, 1), cfg.attnCacheLayerCount());
}

test "nemotron_h: attention is NoPE in every trunk layer and the MTP head" {
    // The config ships rope_theta/partial_rotary_factor, but the reference
    // attention (HF NemotronHAttention, mlx-lm) never rotates q or k.
    const json =
        \\{
        \\  "model_type": "nemotron_h",
        \\  "hidden_size": 2688, "num_hidden_layers": 4,
        \\  "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        \\  "vocab_size": 131072, "rope_theta": 10000, "partial_rotary_factor": 1.0,
        \\  "layers_block_type": ["mamba", "attention", "moe", "attention"]
        \\}
    ;
    const cfg = try parseConfigFromJson(testing.allocator, json);
    // Index num_hidden_layers is the MTP head's attention layer.
    for (0..cfg.num_hidden_layers + 1) |i| try testing.expect(cfg.layerSkipsRope(@intCast(i)));
}

test "nemotron_h: MoE routing fields parse; a latent MoE is refused by name" {
    // Nemotron-3.5-Lightning-30B-A3B's shipped values.
    const json =
        \\{
        \\  "model_type": "nemotron_h",
        \\  "hidden_size": 2688, "num_hidden_layers": 2,
        \\  "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        \\  "vocab_size": 131072, "layers_block_type": ["mamba", "moe"],
        \\  "n_routed_experts": 128, "num_experts_per_tok": 6,
        \\  "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712,
        \\  "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
        \\  "routed_scaling_factor": 2.5, "moe_latent_size": null
        \\}
    ;
    const cfg = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 128), cfg.num_experts);
    try testing.expectEqual(@as(u32, 6), cfg.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 1856), cfg.moe_intermediate_size);
    try testing.expectEqual(@as(u32, 3712), cfg.shared_expert_intermediate_size);
    try testing.expectEqual(@as(u32, 1), cfg.moe_n_group);
    try testing.expectEqual(@as(u32, 1), cfg.moe_topk_group);
    try testing.expect(cfg.moe_route_norm);
    try testing.expectEqual(@as(f32, 2.5), cfg.router_scaling_factor);
    // MoE bills and gates key on this; batching still declines on has_hybrid_layers first.
    try testing.expect(cfg.isMoe());
    try testing.expect(cfg.has_hybrid_layers);

    // The latent variant projects into a smaller expert space
    // (fc1/fc2_latent_proj) that the hybrid MoE op does not carry.
    const latent =
        \\{
        \\  "model_type": "nemotron_h",
        \\  "hidden_size": 4096, "num_hidden_layers": 2,
        \\  "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        \\  "vocab_size": 131072, "layers_block_type": ["mamba", "moe"],
        \\  "n_routed_experts": 512, "num_experts_per_tok": 22, "moe_latent_size": 1024
        \\}
    ;
    try testing.expectError(error.UnsupportedNemotronLatentMoe, parseConfigFromJson(testing.allocator, latent));
}

test "bailing_hybrid: a null q_lora_rank is the direct-q_proj arm, not a refusal" {
    // Ling 3.0 FLASH ships `"q_lora_rank": null` where tiny ships 256, and its
    // MLA layers carry a plain `attention.q_proj` instead of the
    // q_a_proj/q_a_layernorm/q_b_proj triple. That is DeepSeek-V3's documented
    // option, not a broken export — refusing it meant the whole flash line was
    // unloadable while tiny worked.
    const json =
        \\{
        \\  "model_type": "bailing_hybrid",
        \\  "hidden_size": 2560, "num_hidden_layers": 42,
        \\  "num_attention_heads": 32, "num_key_value_heads": 32, "head_dim": 128,
        \\  "layer_group_size": 6,
        \\  "q_lora_rank": null, "kv_lora_rank": 512,
        \\  "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        \\  "num_experts": 512, "num_experts_per_tok": 8, "moe_intermediate_size": 768,
        \\  "vocab_size": 157184, "kda_lower_bound": -5.0
        \\}
    ;
    const cfg = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(cfg.isMla());
    // 0 IS the signal: no low-rank Q, project straight from the hidden state.
    try testing.expectEqual(@as(u32, 0), cfg.mla_q_lora_rank);
    try testing.expect(!cfg.mlaHasQLora());
    try testing.expectEqual(@as(u32, 512), cfg.mla_kv_lora_rank);
    // The attention scale still comes from the FULL query width, unchanged.
    try testing.expectEqual(@as(u32, 192), cfg.mlaQkHeadDim());
    try testing.expectEqual(@as(u32, 192), cfg.query_pre_attn_scalar);

    // kv_lora_rank is still genuinely required — the latent has no fallback.
    const no_kv =
        \\{
        \\  "model_type": "bailing_hybrid",
        \\  "hidden_size": 2560, "num_hidden_layers": 42,
        \\  "num_attention_heads": 32, "num_key_value_heads": 32, "head_dim": 128,
        \\  "layer_group_size": 6, "q_lora_rank": null,
        \\  "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        \\  "num_experts": 512, "num_experts_per_tok": 8, "moe_intermediate_size": 768,
        \\  "vocab_size": 157184
        \\}
    ;
    try testing.expectError(error.UnsupportedBailingConfig, parseConfigFromJson(testing.allocator, no_kv));

    // And tiny's low-rank arm is untouched.
    const tiny =
        \\{
        \\  "model_type": "bailing_hybrid",
        \\  "hidden_size": 1536, "num_hidden_layers": 24,
        \\  "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 128,
        \\  "layer_group_size": 4,
        \\  "q_lora_rank": 256, "kv_lora_rank": 512,
        \\  "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        \\  "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        \\  "vocab_size": 157184
        \\}
    ;
    const t = try parseConfigFromJson(testing.allocator, tiny);
    try testing.expect(t.mlaHasQLora());
    try testing.expectEqual(@as(u32, 256), t.mla_q_lora_rank);
}

test "parseConfigFromJson: qwen4_exp (Qwen3.8-Flash-Next) reads the hyper-connection, PLE, QSA and text-config eos fields" {
    const json =
        \\{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp",
        \\ "text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,
        \\ "full_attention_interval":4,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        \\ "hc_count":4,"hc_lowrank":320,"ple_layer_ids":[2],"ple_embed_dim":2560,"ple_conv_kernel_size":4,
        \\ "ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,
        \\ "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,
        \\ "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\ "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,
        \\ "eos_token_id":248044,"vocab_size":248320,"rms_norm_eps":1e-6,"output_gate_type":"sigmoid",
        \\ "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}},
        \\ "quantization":{"group_size":64,"bits":4,"mode":"affine"}}
    ;
    const c = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(c.isQwen4());
    try testing.expectEqualStrings("language_model.model", c.weight_prefix);
    try testing.expectEqual(@as(u32, 4), c.hc_count);
    try testing.expectEqual(@as(u32, 320), c.hc_lowrank);
    try testing.expectEqual(@as(i32, 1), c.ple_layer_idx); // 1-based [2] → layer 1
    try testing.expectEqual(@as(u32, 2560), c.ple_embed_dim);
    try testing.expectEqual(@as(u32, 4), c.indexer_n_heads);
    try testing.expectEqual(@as(u32, 2048), c.indexer_budget);
    try testing.expectEqual(@as(u32, 4), c.indexer_compress_ratio);
    try testing.expectEqual(@as(u32, 248044), c.ngram_eos);
    try testing.expectEqual(@as(u32, 4), c.full_attention_interval);
    try testing.expect(c.isLinearLayer(0) and !c.isLinearLayer(3));
    try testing.expectEqual(@as(u32, 12), c.attnCacheLayerCount());
    try testing.expectEqual(@as(u64, 12 * 128 * 2 / 4), c.qsaHistoryBytesPerToken());
    try testing.expectEqual(@as(u64, 12 * @as(u64, @intCast(@import("transformer.zig").QSA_RING_ROWS)) * 128 * 2), c.qsaRingBytes());
    try testing.expect(c.attn_output_gate and c.kda_sigmoid_out_gate and !c.has_final_norm and !c.norm_has_offset);
    try testing.expect(c.isMoe() and c.supportsBatchedGdnDecode()); // per-slot state on the SSMCacheEntry: batches
    try testing.expectEqual(@as(f32, 0.25), c.partial_rotary_factor);
    try testing.expectEqual(@as(f32, 10000000.0), c.rope_theta);
    try testing.expect(!c.qwen_vision and !c.has_vision);
}

test "parseConfigFromJson: qwen4_exp with vision_config reads the Qwen3-VL tower, M-RoPE and vision token ids" {
    const json =
        \\{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp",
        \\ "image_token_id":248056,"video_token_id":248057,"vision_start_token_id":248053,"vision_end_token_id":248054,
        \\ "vision_config":{"depth":27,"hidden_size":1152,"num_heads":16,"intermediate_size":4304,"patch_size":16,
        \\   "temporal_patch_size":2,"spatial_merge_size":2,"num_position_embeddings":2304,"out_hidden_size":2560,"model_type":"qwen4_exp_vision"},
        \\ "text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,
        \\ "full_attention_interval":4,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        \\ "ple_layer_ids":[2],"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,
        \\ "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,
        \\ "eos_token_id":248044,"vocab_size":248320,"rms_norm_eps":1e-6,
        \\ "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}},
        \\ "quantization":{"group_size":64,"bits":4,"mode":"affine"}}
    ;
    const c = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(c.isQwen4() and c.has_vision and c.qwen_vision);
    try testing.expectEqual(@as(u32, 27), c.qv_depth);
    try testing.expectEqual(@as(u32, 1152), c.qv_hidden);
    try testing.expectEqual(@as(u32, 16), c.qv_heads);
    try testing.expectEqual(@as(u32, 72), c.qv_head_dim);
    try testing.expectEqual(@as(u32, 4304), c.qv_intermediate);
    try testing.expectEqual(@as(u32, 16), c.qv_patch);
    try testing.expectEqual(@as(u32, 2), c.qv_temporal_patch);
    try testing.expectEqual(@as(u32, 2), c.qv_merge);
    try testing.expectEqual(@as(u32, 2304), c.qv_num_pos_emb);
    try testing.expectEqual(@as(u32, 2560), c.qv_out_hidden);
    try testing.expect(c.mrope_interleaved);
    try testing.expectEqual([3]u32{ 11, 11, 10 }, c.mrope_section);
    try testing.expectEqual(@as(u32, 248056), c.image_token_id);
    try testing.expectEqual(@as(u32, 248057), c.video_token_id);
    try testing.expectEqual(@as(u32, 248053), c.vision_start_token_id);
    try testing.expectEqual(@as(u32, 248054), c.vision_end_token_id);
}

// ── qwen4_exp YaRN context extension (262144 → 1048576) ──────────────────
//
// Both documents below are the SHIPPED checkpoint's text config (Qwen3.8-Flash-
// Next, `model_type: qwen4_exp`) — one as it ships (plain rope, 262144) and one
// with the YaRN block vLLM's `--hf-overrides` recipe writes. Keeping them as
// literals means the parser is tested against the real file shape, braces and
// all, rather than a synthesized one.

/// The checkpoint as it ships: `rope_type: "default"`, a 262144 window.
const QWEN4_SHIPPED =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
    \\    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    \\    "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    \\    "ple_layer_ids": [2],
    \\    "vocab_size": 248320, "eos_token_id": 248044, "max_position_embeddings": 262144,
    \\    "rope_parameters": {
    \\      "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    \\      "mrope_section": [11, 11, 10], "mrope_interleaved": true
    \\    }
    \\  }
    \\}
;

/// The same checkpoint with its rope scaled 4× and the window widened — exactly
/// `vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {...}}}'
/// --max-model-len 1010000` expressed as config instead of a flag.
const QWEN4_YARN =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
    \\    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    \\    "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    \\    "ple_layer_ids": [2],
    \\    "vocab_size": 248320, "eos_token_id": 248044, "max_position_embeddings": 1048576,
    \\    "rope_parameters": {
    \\      "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144,
    \\      "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    \\      "mrope_section": [11, 11, 10], "mrope_interleaved": true
    \\    }
    \\  }
    \\}
;

test "parseConfigFromJson: qwen4_exp YaRN rope_parameters extends 262144 to 1048576" {
    const c = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expect(c.isQwen4());
    // The scaling is recognised and lands where the engine reads it —
    // transformer.yarnSpec() consumes exactly these fields.
    try testing.expect(c.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 4.0), c.yarn_factor, 1e-9);
    try testing.expectEqual(@as(u32, 262_144), c.yarn_orig_max_pos);
    try testing.expectApproxEqAbs(@as(f32, 32.0), c.yarn_beta_fast, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 1.0), c.yarn_beta_slow, 1e-9);
    try testing.expect(c.yarn_truncate); // HF's default, absent from the block
    // The mscale is COMPUTED (no `attention_factor` in the block, so HF's
    // default applies): 0.1·ln 4 + 1 — the value the extension is calibrated to.
    try testing.expectApproxEqAbs(@as(f32, 1.138629436111989), c.yarn_attention_factor, 1e-6);
    // qwen4_exp has ONE rope for the trunk, so the YaRN table spans exactly the
    // 64 dims attention rotates: `partial_rotary_factor`, NOT laguna's
    // `partial_rotary_factor_global` (1.0 here — reading it would scale all 256
    // dims and rotate the pass-through slice).
    try testing.expectApproxEqAbs(@as(f32, 0.25), c.yarnPartial(), 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 1.0), c.partial_rotary_factor_global, 1e-9);
    try testing.expectEqual(@as(u32, 64), c.yarnRotaryDim());
    try testing.expectApproxEqAbs(@as(f32, 10_000_000.0), c.rope_theta, 1.0);
    // The 32 frequencies of the scaled table are the 32 halves the interleaved
    // M-RoPE selector splits [11,11,10] across. If they disagreed, half the
    // table would rotate against an axis the position table doesn't have.
    try testing.expectEqual(
        c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2],
        c.yarnRotaryDim() / 2,
    );
    // The window the server may advertise: original × factor, and the config's
    // own declaration agrees.
    try testing.expectEqual(@as(u32, 1_048_576), c.max_position_embeddings);
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // Cost of that window: only the 12 interval-full layers bill KV, so
    // 12 layers × 2 kv heads × (K+V) × 256 dims × 2 bytes = 24 KiB per token.
    try testing.expectEqual(@as(u32, 12), c.attnCacheLayerCount());
    try testing.expectEqual(@as(u64, 24_576), c.kvBytesPerToken());
}

test "parseConfigFromJson: a Sushi qwen4_exp pack's expert_quant selects EXL3 banks" {
    const eq = "{\"expert_quant\":{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mcg\",\"window\":15},";
    const c = try parseConfigFromJson(testing.allocator, eq ++ QWEN4_SHIPPED[1..]);
    try testing.expect(c.exl3 != null);
    try testing.expectEqual(@as(u32, 32), c.exl3.?.rate.n);
    try testing.expectEqual(sushi_exl3.format.Codebook.mcg, c.exl3.?.codebook);
    try testing.expect((try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED)).exl3 == null);
    const bad = "{\"expert_quant\":{\"format\":\"exl3\",\"k\":2,\"codebook\":\"mul2\"},";
    try testing.expectError(error.ExpertLayoutUnsupported, parseConfigFromJson(testing.allocator, bad ++ QWEN4_SHIPPED[1..]));
}

test "parseConfigFromJson: the shipped (unscaled) qwen4_exp config is untouched" {
    // The regression guard for every checkpoint that predates the extension:
    // no YaRN, and `contextCap` is just max_position_embeddings, so no server
    // sizing path can shift for a model that did not ask to be scaled.
    const c = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(c.isQwen4());
    try testing.expect(!c.rope_yarn);
    try testing.expectEqual(@as(f32, 1.0), c.yarn_factor);
    try testing.expectEqual(@as(u32, 262_144), c.max_position_embeddings);
    try testing.expectEqual(c.max_position_embeddings, c.contextCap());
    try testing.expectApproxEqAbs(@as(f32, 1.0), c.yarn_attention_factor, 1e-9);
    // Same geometry otherwise — YaRN is a rotation, not an architecture change.
    const y = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expectEqual(c.kvBytesPerToken(), y.kvBytesPerToken());
    try testing.expectEqual(c.num_hidden_layers, y.num_hidden_layers);
    try testing.expectEqual(c.head_dim, y.head_dim);
    try testing.expectEqual(c.mrope_section, y.mrope_section);
}

test "parseConfigFromJson: YaRN reads beta_fast/beta_slow/truncate and honours a pinned mscale" {
    defer setConfigOverrides(null);
    // HF's `attention_factor` REPLACES the computed 0.1·ln(factor)+1.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,"attention_factor":1.25}}}
    );
    const pinned = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(pinned.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 1.25), pinned.yarn_attention_factor, 1e-9);
    // Still reads theta/partial from the merged block (the base config's values).
    try testing.expectApproxEqAbs(@as(f32, 0.25), pinned.yarnPartial(), 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 10_000_000.0), pinned.rope_theta, 1.0);

    // vLLM's `attn_factor` MULTIPLIES the computed 0.1·ln(factor)+1.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,"attn_factor":0.5}}}
    );
    const vl = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 0.5 * 1.138629436111989), vl.yarn_attention_factor, 1e-6);

    // Both keys present: HF's attention_factor wins (replace, not multiply).
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,
        \\  "attention_factor":1.25,"attn_factor":0.5}}}
    );
    const both = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 1.25), both.yarn_attention_factor, 1e-9);

    // The ramp knobs are read too — they move the blend, and so every frequency
    // between the bands.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,
        \\  "beta_fast":16,"beta_slow":2,"truncate":false}}}
    );
    const tuned = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 16.0), tuned.yarn_beta_fast, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 2.0), tuned.yarn_beta_slow, 1e-9);
    try testing.expect(!tuned.yarn_truncate);
    // With no pinned mscale, the computed default returns.
    try testing.expectApproxEqAbs(@as(f32, 1.138629436111989), tuned.yarn_attention_factor, 1e-6);
}

test "parseConfigFromJson: YaRN derives factor from the window when the block omits it (HF)" {
    defer setConfigOverrides(null);
    // HF: `factor = max_position_embeddings / original_max_position_embeddings`
    // when the block names only the window. Here the override widens the
    // declared window to 2M out of 262144 → factor 8, mscale 0.1·ln 8 + 1.
    setConfigOverrides(
        \\{"text_config":{"max_position_embeddings":2097152,
        \\  "rope_parameters":{"rope_type":"yarn","original_max_position_embeddings":262144}}}
    );
    const c = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(c.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 8.0), c.yarn_factor, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.2079441541679836), c.yarn_attention_factor, 1e-6);
    try testing.expectEqual(@as(u32, 2_097_152), c.contextCap());
}

test "parseConfigFromJson: Llama 3.x llama3 rope_scaling is read, and keys it changes get their own SSD root" {
    const llama32 =
        \\{"model_type":"llama","hidden_size":3072,"num_hidden_layers":28,"num_attention_heads":24,
        \\ "num_key_value_heads":8,"intermediate_size":8192,"vocab_size":128256,"rope_theta":500000.0,
        \\ "rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
        \\   "original_max_position_embeddings":8192,"rope_type":"llama3"}}
    ;
    const c = try parseConfigFromJson(testing.allocator, llama32);
    const l3 = c.rope_llama3 orelse return error.TestExpectedLlama3Rope;
    try testing.expectEqual(@as(f32, 32.0), l3.factor);
    try testing.expectEqual(@as(f32, 1.0), l3.low_freq_factor);
    try testing.expectEqual(@as(f32, 4.0), l3.high_freq_factor);
    try testing.expectEqual(@as(f32, 8192.0), l3.original_max_position_embeddings);
    try testing.expectEqual(@as(f32, 1.0), c.rope_scaling_factor);
    try testing.expectEqualStrings("llama3-rope-v1", c.cacheLayoutNamespace().?);
    // No `sliding_window` key: every layer attends the whole context (the Gemma
    // default made 5 of 6 layers 1024-token windows and broke prompts past 1k).
    try testing.expect(!c.has_sliding_window);
    for (0..28) |li| try testing.expect(c.isGlobalLayer(@intCast(li)));

    // The rest of the Llama family, which reads its attention through the same branch.
    for ([_][]const u8{ "llama", "mistral", "qwen2", "qwen3", "k2_horizon" }) |mt| {
        var buf: [256]u8 = undefined;
        const cfg = try std.fmt.bufPrint(&buf, "{{\"model_type\":\"{s}\",\"hidden_size\":64,\"num_hidden_layers\":12,\"num_attention_heads\":4,\"vocab_size\":100}}", .{mt});
        const plain = try parseConfigFromJson(testing.allocator, cfg);
        try testing.expect(plain.rope_llama3 == null);
        try testing.expect(plain.cacheLayoutNamespace() == null);
        for (0..12) |li| try testing.expect(plain.isGlobalLayer(@intCast(li)));
    }
}

test "parseConfigFromJson: YaRN with no pre-trained window, or a zero factor, fails the load" {
    defer setConfigOverrides(null);
    // The ramp bounds come from `original_max_position_embeddings`. Without it
    // every blended frequency is a guess — refuse the load rather than serve a
    // rope that looks fine at short contexts and decays beyond the window.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0}}}
    );
    try testing.expectError(
        error.YarnRopeNeedsOriginalMaxPos,
        parseConfigFromJson(testing.allocator, QWEN4_SHIPPED),
    );
    // A zero factor is not "no scaling", it is a divide-by-zero waiting to run.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":0.0,
        \\  "original_max_position_embeddings":262144}}}
    );
    try testing.expectError(
        error.InvalidRopeScalingFactor,
        parseConfigFromJson(testing.allocator, QWEN4_SHIPPED),
    );
}

test "ModelConfig.contextCap: the rope-derived window binds what the server advertises" {
    const scaled = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    // Over-advertised: a config claiming 2M tokens on a factor-4 ramp out of
    // 262144 still cannot resolve past 1048576 — past there positions alias back
    // inside the window, which is the failure this clamps against.
    var c = scaled;
    c.max_position_embeddings = 2_000_000;
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // Under-advertised: serving LESS than the scaled window is legal — the ramp
    // is fixed by the pre-trained length, not by what you choose to run.
    c.max_position_embeddings = 400_000;
    try testing.expectEqual(@as(u32, 400_000), c.contextCap());
    // Declaring nothing: the ramp still says how far the rope reaches.
    c.max_position_embeddings = 0;
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // A fractional factor floors (vLLM's `int()` of the same product).
    c.yarn_factor = 3.5;
    try testing.expectEqual(@as(u32, 917_504), c.contextCap()); // floor(262144*3.5)
}

test "parseConfigFromJson: --config-overrides deep-merges a nested block without clobbering siblings" {
    // The merge is what makes the flag usable for rope at all: `rope_parameters`
    // is written as a whole object, and a REPLACE would drop the
    // `partial_rotary_factor` / `mrope_section` keys beside it — silently
    // rotating 256 dims instead of 64, or the wrong axes. vLLM has the same
    // rule (`_update_nested` merges, `_apply_dict_overrides` only replaces
    // non-config values), and the same trap is documented in its source.
    defer setConfigOverrides(null);
    // Pre-override: the shipped config really does have no scaling.
    try testing.expect(!(try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED)).rope_yarn);
    setConfigOverrides(
        \\{"text_config":{"max_position_embeddings":1048576,
        \\  "rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\    "original_max_position_embeddings":262144}}}
    );
    const c = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(c.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 4.0), c.yarn_factor, 1e-9);
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // Keys the override never mentioned survived at BOTH levels of the merge.
    try testing.expectApproxEqAbs(@as(f32, 0.25), c.partial_rotary_factor, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 10_000_000.0), c.rope_theta, 1.0);
    try testing.expect(c.mrope_interleaved);
    try testing.expectEqual([3]u32{ 11, 11, 10 }, c.mrope_section);
    try testing.expectEqual(@as(u32, 262_144), c.yarn_orig_max_pos);
    // The result is indistinguishable from the hand-written extended config.
    const written = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expectApproxEqAbs(written.yarn_factor, c.yarn_factor, 1e-9);
    try testing.expectEqual(written.contextCap(), c.contextCap());
    try testing.expectApproxEqAbs(written.yarn_attention_factor, c.yarn_attention_factor, 1e-9);
}

test "parseConfigFromJson: --config-overrides replaces scalars and arrays, creates new keys, rejects junk" {
    defer setConfigOverrides(null);
    // Scalars and arrays replace wholesale (vLLM's base case); an array nested
    // in an object that is otherwise merged still replaces the array it meets.
    setConfigOverrides(
        \\{"text_config":{"num_hidden_layers":8,"head_dim":128,
        \\  "rope_parameters":{"mrope_section":[9,9,9]}}}
    );
    const c = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expectEqual(@as(u32, 8), c.num_hidden_layers);
    try testing.expectEqual(@as(u32, 128), c.head_dim);
    try testing.expectEqual([3]u32{ 9, 9, 9 }, c.mrope_section);
    try testing.expect(c.rope_yarn); // the block's other keys survived
    try testing.expectApproxEqAbs(@as(f32, 4.0), c.yarn_factor, 1e-9);

    // A key the document never had is created at the level the parser reads
    // (qwen4's fields come from `text_config`, so that's where it must land).
    setConfigOverrides(
        \\{"text_config":{"ngram_size":5}}
    );
    const n = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectEqual(@as(u32, 5), n.ngram_size);

    // Only an object is a document; a bare array must not half-apply.
    setConfigOverrides(
        \\[1,2,3]
    );
    try testing.expectError(
        error.ConfigOverridesMustBeObject,
        parseConfigFromJson(testing.allocator, QWEN4_SHIPPED),
    );
    // Clearing the seam restores the shipped document exactly.
    setConfigOverrides(null);
    const clean = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectEqual(@as(u32, 48), clean.num_hidden_layers);
    try testing.expectEqual(@as(u32, 256), clean.head_dim);
    try testing.expectEqual(@as(u32, 3), clean.ngram_size);
    try testing.expectEqual([3]u32{ 11, 11, 10 }, clean.mrope_section);
    try testing.expect(!clean.rope_yarn);
}

test "ModelConfig.longCtxGated: the long-context blast radius is ONE predicate, qwen4_exp only" {
    const t = std.testing;
    var qwen4 = ModelConfig{ .model_type = "qwen4_exp" };
    try t.expect(qwen4.longCtxGated());
    try t.expect(qwen4.ssdFirstCapable());

    for ([_][]const u8{
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_next",
        "lfm2",
        "nemotron_h",
        "bailing_hybrid",
        "llama",
        "mistral",
        "gemma3",
        "gemma4",
        "deepseek_v4",
        "muse_glimmer",
    }) |mt| {
        var cfg = ModelConfig{ .model_type = mt };
        try t.expect(!cfg.longCtxGated());
        try t.expect(!cfg.ssdFirstCapable());
    }
}

/// One qwen4_exp config document with `extra` fields spliced in.
fn qwen4CaseJson(comptime extra: []const u8) []const u8 {
    return "{\"model_type\":\"qwen4_exp\",\"hidden_size\":2560,\"num_hidden_layers\":48," ++
        "\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":2,\"head_dim\":256," ++
        "\"hc_count\":4,\"hc_lowrank\":320,\"ple_embed_dim\":2560,\"ple_conv_kernel_size\":4," ++
        "\"num_experts\":512,\"num_experts_per_tok\":10,\"moe_intermediate_size\":640," ++
        "\"eos_token_id\":248044,\"vocab_size\":248320,\"rms_norm_eps\":1e-6," ++
        extra ++ "}";
}

const QWEN4_GOOD_FIELDS =
    "\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":8," ++
    "\"ngram_vocab_size_base\":20000000,\"make_ngram_vocab_size_divisible_by\":128," ++
    "\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2048,\"indexer_compress_ratio\":4";

test "qwen4 norm convention is an explicit root marker with strict values" {
    const unmarked = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS));
    try testing.expectEqual(@as(?Qwen4NormConvention, null), unmarked.qwen4_norm_convention);
    const delta = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"qwen4_norm_convention\":\"delta\""));
    try testing.expectEqual(Qwen4NormConvention.delta, delta.qwen4_norm_convention.?);
    const folded = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"qwen4_norm_convention\":\"folded\""));
    try testing.expectEqual(Qwen4NormConvention.folded, folded.qwen4_norm_convention.?);
    for ([_][]const u8{
        qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"qwen4_norm_convention\":null"),
        qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"qwen4_norm_convention\":1"),
        qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"qwen4_norm_convention\":\"other\""),
    }) |json| try testing.expectError(error.InvalidQwen4NormConvention, parseConfigFromJson(testing.allocator, json));
    defer setConfigOverrides(null);
    setConfigOverrides("{\"max_position_embeddings\":131072}");
    const overridden = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"qwen4_norm_convention\":\"folded\""));
    try testing.expectEqual(Qwen4NormConvention.folded, overridden.qwen4_norm_convention.?);
    try testing.expectEqual(@as(u32, 131072), overridden.max_position_embeddings);
}

test "unmarked embedded qwen4 defers norm convention until weights load" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try @import("qwen4_ple.zig").writeFixture(&td, .valid);
    const json =
        \\{"model_type":"qwen4_exp","hidden_size":64,"num_hidden_layers":2,
        \\"num_attention_heads":1,"num_key_value_heads":1,"head_dim":64,
        \\"vocab_size":1,"ple_layer_ids":[2],"ple_embed_dim":64,
        \\"ngram_size":3,"heads_per_ngram":1,"ngram_vocab_size_base":2,
        \\"make_ngram_vocab_size_divisible_by":6,"split_ngram_parts":3,
        \\"indexer_n_heads":1,"indexer_head_dim":4,"indexer_budget":8,"indexer_compress_ratio":2}
    ;
    try td.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const len = try td.dir.realPath(io, &path);
    var unmarked = try parseConfig(io, testing.allocator, path[0..len]);
    defer unmarked.deinit(testing.allocator);
    try testing.expectEqual(@as(?Qwen4NormConvention, null), unmarked.qwen4_norm_convention);
    try testing.expectEqual(@as(?u64, 120), unmarked.embedded_ple_payload_bytes);
    const marked_json = try std.fmt.allocPrint(testing.allocator, "{s},\"qwen4_norm_convention\":\"delta\"}}", .{std.mem.trimEnd(u8, json, " \n\r\t}")});
    defer testing.allocator.free(marked_json);
    try td.dir.writeFile(io, .{ .sub_path = "config.json", .data = marked_json });
    var accepted = try parseConfig(io, testing.allocator, path[0..len]);
    defer accepted.deinit(testing.allocator);
    try testing.expectEqual(Qwen4NormConvention.delta, accepted.qwen4_norm_convention.?);
    try testing.expectEqual(@as(?u64, 120), accepted.embedded_ple_payload_bytes);
}

test "local oQ Qwen4 metadata defers convention until loaded weights" {
    const raw = std.c.getenv("QWEN4_EMBEDDED_TEST_MODEL") orelse return error.SkipZigTest;
    const path = std.mem.span(raw);
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try parseConfig(io, testing.allocator, path);
    defer config.deinit(testing.allocator);
    try testing.expect(config.isQwen4());
    try testing.expectEqual(@as(?Qwen4NormConvention, null), config.qwen4_norm_convention);
    try testing.expect(config.embedded_ple_payload_bytes.? > 0);
}

test "Qwen4 norm global override is rejected independently of the checkpoint marker" {
    defer setConfigOverrides(null);
    setConfigOverrides("{\"qwen4_norm_convention\":\"delta\"}");
    try testing.expectError(error.GlobalQwen4NormOverride, parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS)));
    try testing.expectError(error.GlobalQwen4NormOverride, parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"qwen4_norm_convention\":\"folded\"")));
}

test "local legacy Qwen4 metadata keeps external folded default" {
    const raw = std.c.getenv("QWEN4_EXTERNAL_TEST_MODEL") orelse return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try parseConfig(io, testing.allocator, std.mem.span(raw));
    defer config.deinit(testing.allocator);
    try testing.expect(config.isQwen4());
    try testing.expectEqual(@as(?Qwen4NormConvention, null), config.qwen4_norm_convention);
    try testing.expectEqual(@as(?u64, null), config.embedded_ple_payload_bytes);
}

test "Qwen4 norm selection stays checkpoint-local across mixed model loads" {
    const raw = std.c.getenv("QWEN4_EMBEDDED_TEST_MODEL") orelse return error.SkipZigTest;
    const legacy = std.c.getenv("QWEN4_EXTERNAL_TEST_MODEL") orelse return error.SkipZigTest;
    const paths = [_][]const u8{ std.mem.span(raw), std.mem.span(legacy) };
    const io = std.Io.Threaded.global_single_threaded.io();
    for ([_]usize{ 0, 1, 1, 0 }) |i| {
        var config = try parseConfig(io, testing.allocator, paths[i]);
        defer config.deinit(testing.allocator);
        try testing.expectEqual(@as(?Qwen4NormConvention, null), config.qwen4_norm_convention);
        try testing.expectEqual(i == 0, config.embedded_ple_payload_bytes != null);
    }
}

test "qwen4_exp config: an n-gram bound past the fixed arrays is a named load error" {
    const good = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS));
    try testing.expectEqual(@as(u32, 3), good.ngram_size);
    try testing.expectEqual(@as(u32, 8), good.heads_per_ngram);

    try testing.expectError(error.InvalidQwen4NgramSize, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":9,\"heads_per_ngram\":8"),
    ));
    try testing.expectError(error.InvalidQwen4NgramSize, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":1,\"heads_per_ngram\":8"),
    ));
    try testing.expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":0"),
    ));
    try testing.expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":5,\"heads_per_ngram\":16"),
    ));
    try testing.expectError(error.InvalidQwen4NgramVocab, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"make_ngram_vocab_size_divisible_by\":0"),
    ));
}

test "qwen4_exp config: a wrong-typed or negative bound is a refusal, never a silent default" {
    try testing.expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":-1"),
    ));
    try testing.expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":\"3\""),
    ));
    try testing.expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"heads_per_ngram\":3.5"),
    ));
    try testing.expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_compress_ratio\":-4"),
    ));
}

test "qwen4_exp config: an armed QSA indexer must carry a usable budget and ratio" {
    try testing.expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2048"),
    ));
    try testing.expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2,\"indexer_compress_ratio\":4"),
    ));
    try testing.expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_budget\":2048,\"indexer_compress_ratio\":4"),
    ));
    const dense = try parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":8"),
    );
    try testing.expectEqual(@as(u32, 0), dense.indexer_n_heads);
    try testing.expectEqual(@as(u32, 0), dense.indexer_compress_ratio);
}

test "qwen4_exp config: the PLE layer id must name exactly one layer that exists" {
    try testing.expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ngram_size\":3"),
    ));
    try testing.expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[0]"),
    ));
    try testing.expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[49]"),
    ));
    try testing.expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2,5]"),
    ));
    try testing.expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[]"),
    ));
    try testing.expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":2"),
    ));
    const c = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS));
    try testing.expectEqual(@as(i32, 1), c.ple_layer_idx);
}

test "qwen4 PLE placement: the layer loop must install exactly one PLE, at the configured layer" {
    try testing.expect(qwen4PleInstalledAt(&.{ false, true, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, false, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ true, true, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, 2));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, 4));
    // A negative index is a build that asks for no PLE (`loadQwen4Mtp` sets -1 for the head's layer).
    try testing.expect(qwen4PleInstalledAt(&.{false}, -1));
    try testing.expect(!qwen4PleInstalledAt(&.{true}, -1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, -1));
    try testing.expect(qwen4PleInstalledAt(&.{ false, false }, -1));
    // ...while a config that DOES name a layer is unchanged.
    try testing.expect(!qwen4PleInstalledAt(&.{false}, 0));
    try testing.expect(qwen4PleInstalledAt(&.{true}, 0));
}

test "ModelConfig parses k2_horizon (K2-Horizon-7B): llama trunk with grouped RMS norms" {
    const json =
        \\{
        \\  "model_type": "k2_horizon",
        \\  "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36,
        \\  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        \\  "hidden_act": "silu", "rms_norm_eps": 1e-06, "vocab_size": 250624,
        \\  "layernorm_num_groups": 4, "query_key_norm": false, "attention_gate_func": null,
        \\  "num_experts": 0, "sliding_window": null, "tie_word_embeddings": false,
        \\  "max_position_embeddings": 524288, "rope_head_dim": 128,
        \\  "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("k2_horizon", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expectEqual(@as(u32, 4), config.norm_groups);
    try testing.expectEqual(HiddenAct.silu, config.hidden_act);
    try testing.expectEqual(@as(u32, 128), config.head_dim);
    try testing.expectEqual(@as(f32, 10000000.0), config.rope_theta);
    try testing.expect(!config.has_qk_norm);
    try testing.expect(!config.has_sliding_window);
    try testing.expect(!config.tie_word_embeddings);
    try testing.expect(!config.norm_has_offset);
    try testing.expect(!config.has_pre_ff_norm);
}

test "rowExactDecode: a covered trunk decodes exact only while a DFlash drafter is bound" {
    var cfg = ModelConfig{};
    cfg.model_type = "qwen3_5";
    cfg.row_exact_covered = true;
    // Serial and MTP keep the stock paths: exact mode costs MTP ~30%.
    try std.testing.expect(!cfg.rowExactDecode());
    cfg.dflash_bound = true;
    try std.testing.expect(cfg.rowExactDecode());
    cfg.row_exact_covered = false;
    try std.testing.expect(!cfg.rowExactDecode());
}

test "mtpAcceptance: exact while a DFlash drafter is bound, else the model setting, else the launch default" {
    var cfg = ModelConfig{};
    const typical: mtp_acceptance_mod.Mode = .{ .typical = .{ .delta = 0.2 } };
    try std.testing.expect(std.meta.activeTag(cfg.mtpAcceptance(typical)) == .typical);
    cfg.mtp_acceptance_override = .{ .tokenv3 = 0.95 };
    try std.testing.expect(std.meta.activeTag(cfg.mtpAcceptance(typical)) == .tokenv3);
    cfg.dflash_bound = true;
    try std.testing.expect(cfg.mtpAcceptance(typical) == .exact);
}

test "parseConfigFromJson rejects invalid field types and ranges" {
    const bad = [_][]const u8{
        "[]",
        "{\"model_type\":3}",
        "{\"text_config\":[]}",
        "{\"hidden_size\":\"4096\"}",
        "{\"num_hidden_layers\":1.5}",
        "{\"vocab_size\":4294967296}",
        "{\"sliding_window\":\"bad\"}",
        "{\"num_experts\":-1}",
        "{\"eos_token_id\":-1}",
        "{\"eos_token_id\":[2,4294967296]}",
        "{\"rope_theta\":1e100}",
        "{\"quantization\":[]}",
        "{\"quantization\":{\"bits\":\"8\"}}",
        "{\"vision_config\":{\"hidden_size\":-1}}",
        "{\"model_type\":\"qwen3_5\",\"vision_config\":{\"depth\":-1}}",
        "{\"model_type\":\"qwen3_5\",\"rope_parameters\":{\"mrope_section\":[11,-1,10]}}",
        "{\"model_type\":\"prism_hadamard_qwen35\",\"modules\":{}}",
        "{\"model_type\":\"prism_hadamard_qwen35\",\"modules\":[{\"block\":4294967296}]}",
        "{\"model_type\":\"llama\",\"num_attention_heads\":0}",
        "{\"model_type\":\"inkling_mm_model\",\"head_dim\":65536}",
        "{\"model_type\":\"hy_v3\",\"num_shared_experts\":4294967295,\"moe_intermediate_size\":2}",
        "{\"text_config\":{\"linear_num_key_heads\":-1}}",
        "{\"model_type\":\"qwen3_5\",\"linear_num_key_heads\":0}",
        "{\"model_type\":\"qwen3_5\",\"text_config\":{\"linear_num_key_heads\":3,\"linear_num_value_heads\":8}}",
        "{\"model_type\":\"laguna\",\"num_attention_heads_per_layer\":[-1]}",
        "{\"model_type\":\"deepseek_v4\",\"compress_ratios\":[256]}",
        "{\"model_type\":\"deepseek_v4\",\"dspark_target_layer_ids\":[256]}",
        "{\"model_type\":\"lfm2\",\"num_attention_heads\":0}",
        "{\"model_type\":\"bert\",\"num_attention_heads\":0}",
        "{\"model_type\":\"nemotron_h\",\"mamba_num_heads\":-1}",
        "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":1,\"qk_nope_head_dim\":4294967295,\"qk_rope_head_dim\":1}",
        "{\"model_type\":\"qwen4_exp\",\"rope_parameters\":{\"rope_type\":\"yarn\",\"original_max_position_embeddings\":1024,\"factor\":1e30,\"attn_factor\":1e38}}",
    };
    for (bad) |content| {
        if (parseConfigFromJson(testing.allocator, content)) |parsed_config| {
            var config = parsed_config;
            defer config.deinit(testing.allocator);
            std.debug.print("accepted invalid config: {s}\n", .{content});
            return error.TestExpectedError;
        } else |err| try testing.expectEqual(error.InvalidConfigField, err);
    }

    try testing.expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(testing.allocator,
        \\{"model_type":"qwen4_exp","ple_layer_ids":[1],"ngram_size":3,"heads_per_ngram":2147483648}
    ));
    try testing.expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(testing.allocator,
        \\{"model_type":"qwen4_exp","num_hidden_layers":4294967295,"ple_layer_ids":[4294967295]}
    ));

    const valid = try parseConfigFromJson(testing.allocator,
        \\{"model_type":"qwen3_5","hidden_size":4294967295,"num_experts":null,
        \\ "sliding_window":null,"eos_token_id":[0,4294967295],"rope_theta":10000,
        \\ "final_logit_softcapping":null,"unknown_vendor_field":{"future":true}}
    );
    try testing.expectEqual(std.math.maxInt(u32), valid.hidden_size);
    try testing.expectEqual(@as(u32, 0), valid.num_experts);
    try testing.expect(!valid.has_sliding_window);
    try testing.expectEqual(std.math.maxInt(u32), valid.eos_token_ids[1]);
    try testing.expectEqual(@as(f32, 10000), valid.rope_theta);
    try testing.expectEqual(@as(f32, 0), valid.final_logit_softcapping);
}

test "parseConfigFromJson preserves optional nulls and skipped fields" {
    const cases = [_][2][]const u8{
        .{ "{}", "{\"head_dim\":null,\"rope_theta\":null,\"hidden_size\":null,\"intermediate_size\":null}" },
        .{ "{}", "{\"bos_token_id\":-1}" },
        .{ "{\"model_type\":\"k2_horizon\"}", "{\"model_type\":\"k2_horizon\",\"layernorm_num_groups\":-1}" },
        .{ "{}", "{\"model_type\":null,\"text_config\":null,\"quantization\":null,\"linear_num_key_heads\":null}" },
        .{ "{}", "{\"num_experts\":\"unused\",\"attn_output_gate\":1,\"use_bidirectional_attention\":1,\"pooling_mode\":false}" },
        .{ "{\"eos_token_id\":[2]}", "{\"eos_token_id\":[2,\"unused\",null]}" },
        .{ "{\"model_type\":\"llama\"}", "{\"model_type\":\"llama\",\"partial_rotary_factor\":null,\"rope_parameters\":{\"rope_theta\":null,\"partial_rotary_factor\":null}}" },
        .{ "{\"model_type\":\"qwen3_5\",\"rope_parameters\":{\"rope_type\":\"yarn\",\"original_max_position_embeddings\":1024,\"attn_factor\":2}}", "{\"model_type\":\"qwen3_5\",\"rope_parameters\":{\"rope_type\":\"yarn\",\"original_max_position_embeddings\":1024,\"factor\":null,\"attention_factor\":null,\"attn_factor\":2,\"beta_fast\":null,\"beta_slow\":null}}" },
        .{ "{\"model_type\":\"nemotron_h\"}", "{\"model_type\":\"nemotron_h\",\"mamba_num_heads\":\"unused\"}" },
    };
    for (cases) |pair| {
        var expected = try parseConfigFromJson(testing.allocator, pair[0]);
        defer expected.deinit(testing.allocator);
        var actual = try parseConfigFromJson(testing.allocator, pair[1]);
        defer actual.deinit(testing.allocator);
        try testing.expectEqualDeep(expected, actual);
    }
}

test "parseConfigFromJson accepts real checkpoint configs" {
    const names = .{ "gemma3-text", "gemma4-text", "gemma4-unified-text", "mistral", "nemotron-h", "laguna" };
    var failed = false;
    inline for (names) |name| {
        if (parseConfigFromJson(testing.allocator, @embedFile("fixtures/model-configs/" ++ name ++ ".json"))) |parsed_config| {
            var config = parsed_config;
            defer config.deinit(testing.allocator);
            if (std.mem.startsWith(u8, name, "gemma"))
                try testing.expectEqual(@as(f32, 1e6), config.rope_theta);
            if (std.mem.eql(u8, name, "mistral"))
                try testing.expectEqual(@as(u32, 128), config.head_dim);
        } else |err| {
            std.debug.print("{s}: {s}\n", .{ name, @errorName(err) });
            failed = true;
        }
    }
    try testing.expect(!failed);
}
