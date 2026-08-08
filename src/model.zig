const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const tokenizer_mod = @import("tokenizer.zig");

pub const HiddenAct = enum { gelu_approx, silu, relu_sq };

/// MLX quantization mode from config.json's `quantization.mode`. All
/// non-affine modes store NO `.biases` tensors (per-group fp8-encoded uint8
/// scales only) but share the packed-u32 weight layout, so supporting them is
/// a matter of skipping the biases fetch and passing the right mode string to
/// the mlx quantized ops. Tag names match the mlx-c mode strings exactly.
pub const QuantMode = enum {
    affine,
    nvfp4,
    mxfp4,
    mxfp8,

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

    // RoPE
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,
    rope_proportional: bool = false, // Gemma 4: full attention uses proportional RoPE
    rope_proportional_factor: f32 = 1.0,

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

    // MoE
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    // DeepSeek-V3-style sigmoid routing (hy_v3): scores = sigmoid(logits) in
    // f32; top-k SELECTED on scores + expert_bias but WEIGHTED by the unbiased
    // scores; optional renorm (/(sum+1e-20)) then × router_scaling_factor.
    moe_sigmoid_router: bool = false,
    moe_route_norm: bool = true,
    router_scaling_factor: f32 = 1.0,
    // Layers [0, first_k_dense_replace) use a dense MLP instead of MoE
    // (hy_v3: layer 0 dense at intermediate_size, the rest MoE).
    first_k_dense_replace: u32 = 0,
    // Laguna: MoE router logit soft-capping (tanh). 0 = off (the shipped
    // Laguna-S-2.1 checkpoint sets moe_router_logit_softcapping: 0.0).
    moe_router_logit_softcapping: f32 = 0.0,

    // Linear attention (GatedDeltaNet)
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,

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

    // Laguna: softplus per-head attention output gate. self_attn.g_proj →
    // softplus(fp32) → per-head scalar × attn output (reshaped [..,H,D]) before
    // o_proj. Distinct from attn_output_gate (qwen3-next sigmoid + doubled
    // q_proj) — separate weight, softplus activation, per-head broadcast.
    laguna_attn_gate: bool = false,
    // Laguna: per-layer Q-head count (full-attention layers 48, sliding 72;
    // KV heads uniform at num_key_value_heads). 0 = uniform num_attention_heads.
    num_attention_heads_per_layer: [128]u32 = @splat(0),
    has_per_layer_heads: bool = false,

    // Laguna YaRN RoPE (full-attention layers only; sliding layers use default
    // RoPE at rope_local_base_freq). rope_yarn gates the freqs + mscale
    // precompute at model load; the sliding/full split is by isGlobalLayer.
    rope_yarn: bool = false,
    yarn_factor: f32 = 1.0,
    yarn_orig_max_pos: u32 = 0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
    yarn_attention_factor: f32 = 1.0,

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

    // Gemma 4: explicit layer type map (bit = 1 means full/global attention)
    has_explicit_layer_types: bool = false,
    layer_is_global: [128]bool = @splat(false),

    // Vision encoder (Gemma 4 SigLIP)
    has_vision: bool = false,
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
    // Interleaved M-RoPE sections [t, h, w]; sum = rotary_dim/2 (e.g. [11,11,10]).
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
    // Qwen vision token ids (top-level config.json). image_token_id reuses the
    // shared field above (parsed generically at the image_token_id block).
    video_token_id: u32 = 0,
    vision_start_token_id: u32 = 0,
    vision_end_token_id: u32 = 0,

    // Token IDs that mark the start of a user turn in the rendered prompt.
    // Populated at startup by encoding a chat-template-specific prefix string
    // (e.g. "<|turn>user\n" for Gemma 4, "<|im_start|>user\n" for Qwen ChatML).
    // Used by insertImageTokens to locate the latest user turn — a hard-coded
    // ID search would silently break across architectures and quantizations.
    user_turn_marker_ids: [16]u32 = @splat(0),
    user_turn_marker_len: u8 = 0,

    // Gemma 4: dual head dimensions and KV sharing
    global_head_dim: u32 = 0, // 0 = same as head_dim
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

    /// How many layers hold an attention KV cache. A hybrid MLA arch can
    /// carry attention on a fraction of its layers; every current arch
    /// caches on all of them. Used by the prefill preflight, which otherwise
    /// bills a uniform arch's footprint for a model carrying less.
    pub fn attnCacheLayerCount(self: *const ModelConfig) u32 {
        return self.num_hidden_layers;
    }

    pub fn isMoe(self: *const ModelConfig) bool {
        return self.num_experts > 0;
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

    /// True when the trunk uses the Gemma 4 layer structure (dual FFN with
    /// shared-expert branch, sigma-MoE router, 7 norms, layer_scalar, v_norm,
    /// proportional RoPE on full layers). DiffusionGemma reuses the Gemma 4
    /// 26B-A4B decoder verbatim, so transformer.zig's gemma4 forward/binding
    /// paths key on this rather than on the model_type string.
    pub fn isGemma4Layers(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "gemma4") or
            std.mem.eql(u8, self.model_type, "diffusion_gemma");
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

    /// Hy3 (hy_v3) family terminator: <｜hy_eos:opensource｜> = 120025. Real
    /// MLX conversions (ox-ox 2-bit) ship NO eos in config.json and NO
    /// generation_config.json, so without this merge generation never halts.
    /// Additive + dedup-guarded like ensureGemmaTerminators — never gate a
    /// known chat-terminator on "config provided no eos".
    pub fn ensureHy3Terminators(self: *ModelConfig) void {
        if (!self.isEosToken(120025)) self.addEosToken(120025);
    }

    /// The head width the PREFILL SCORE tensor is actually built at. Normally
    /// `head_dim`, but an arch can score at a different width than it stores
    /// values at (an MLA q.k can contract over nope+rope widths while
    /// `head_dim` stays the value width) — reading `head_dim` there puts such
    /// an arch under the `<= 128` "fused SDPA covers it" early-out, so the
    /// score budget that exists for exactly this materializing path never
    /// applies. A new arch scoring wider than it stores adds its arm here.
    pub fn prefillScoreHeadDim(self: *const ModelConfig) u32 {
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
    /// Opt-in per arch, and only where the vendor documents thinking-on AND
    /// the shipped template agrees — never inferred from "the template mentions
    /// enable_thinking".
    pub fn defaultEnableThinking(self: *const ModelConfig) bool {
        _ = self;
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

    pub fn userTurnMarkerSlice(self: *const ModelConfig) []const u32 {
        return self.user_turn_marker_ids[0..self.user_turn_marker_len];
    }

    /// Encode the architecture-appropriate user-turn prefix and store the IDs
    /// on the config. Selects the prefix by matching marker tokens that appear
    /// in `chat_template`, so a model that ships an unusual template still
    /// gets the right tokenization. No-op (leaves length=0) when no known
    /// pattern matches — insertImageTokens then falls back to its end-anchored
    /// heuristic.
    pub fn populateUserTurnMarker(
        self: *ModelConfig,
        allocator: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        chat_template: []const u8,
    ) !void {
        const prefix = pickUserTurnPrefix(chat_template) orelse return;
        const ids = try tok.encode(allocator, prefix);
        defer allocator.free(ids);
        const cap = self.user_turn_marker_ids.len;
        if (ids.len == 0 or ids.len > cap) {
            log.warn("user turn marker '{s}' encoded to {d} tokens (cap {d}); skipping\n", .{ prefix, ids.len, cap });
            return;
        }
        @memcpy(self.user_turn_marker_ids[0..ids.len], ids);
        self.user_turn_marker_len = @intCast(ids.len);
        log.info("User turn marker: \"{s}\" -> {d} tokens\n", .{ prefix, ids.len });
    }
};

/// Pick the user-turn prefix string for a model based on what its chat template
/// emits at the start of a user turn. Order matters — the more specific Gemma 4
/// `<|turn>` is checked before the older `<start_of_turn>` so a tokenizer that
/// happens to register both still picks the one its template actually uses.
pub fn pickUserTurnPrefix(chat_template: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, chat_template, "<|turn>") != null) {
        return "<|turn>user\n"; // Gemma 4
    }
    if (std.mem.indexOf(u8, chat_template, "<start_of_turn>") != null) {
        return "<start_of_turn>user\n"; // Gemma 3
    }
    if (std.mem.indexOf(u8, chat_template, "<|im_start|>") != null) {
        return "<|im_start|>user\n"; // Qwen / generic ChatML
    }
    if (std.mem.indexOf(u8, chat_template, "<|start_header_id|>") != null) {
        return "<|start_header_id|>user<|end_header_id|>\n\n"; // Llama 3
    }
    return null;
}

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    defer allocator.free(content);

    var config = try parseConfigFromJson(allocator, content);

    // Model-author sampling recommendations ride in a sibling file. Optional —
    // any failure (missing file, bad JSON) leaves the fields null.
    const gen_path = try std.fmt.allocPrint(allocator, "{s}/generation_config.json", .{model_dir});
    defer allocator.free(gen_path);
    if (std.Io.Dir.openFileAbsolute(io, gen_path, .{})) |gen_file| {
        defer gen_file.close(io);
        var gen_buf: [4096]u8 = undefined;
        var gen_reader = gen_file.reader(io, &gen_buf);
        if (gen_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |gen_content| {
            defer allocator.free(gen_content);
            const gd = parseGenerationDefaultsFromJson(gen_content);
            config.gen_temperature = gd.temperature;
            config.gen_top_p = gd.top_p;
            config.gen_top_k = gd.top_k;
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
    if (config.qwen_vision) {
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
    }

    return config;
}

/// Sampling recommendations parsed out of a model's generation_config.json.
pub const GenerationDefaults = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
};

/// Image-area limits parsed from a Qwen processor configuration.
pub const VisionProcessorDefaults = struct {
    min_pixels: ?u32 = null,
    max_pixels: ?u32 = null,
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
    return gd;
}

/// I/O-free variant for unit tests and for callers that already have the
/// config.json bytes in memory. The full I/O-bound `parseConfig` delegates here.
pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !ModelConfig {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    var config = ModelConfig{};

    // Detect model_type from top-level (always present)
    const model_type = if (root.get("model_type")) |v| v.string else "gemma3";

    // Determine which object to read config from: text_config (nested) or root (flat)
    const cfg_obj = if (root.get("text_config")) |tc_val| tc_val.object else root;

    // Parse common fields
    if (cfg_obj.get("vocab_size")) |v| config.vocab_size = @intCast(v.integer);
    if (cfg_obj.get("hidden_size")) |v| config.hidden_size = @intCast(v.integer);
    if (cfg_obj.get("intermediate_size")) |v| {
        config.intermediate_size = @intCast(v.integer);
        config.intermediate_size_declared = true;
    }
    if (cfg_obj.get("num_hidden_layers")) |v| config.num_hidden_layers = @intCast(v.integer);
    if (cfg_obj.get("num_attention_heads")) |v| config.num_attention_heads = @intCast(v.integer);
    if (cfg_obj.get("num_key_value_heads")) |v| config.num_key_value_heads = @intCast(v.integer);
    if (cfg_obj.get("head_dim")) |v| config.head_dim = @intCast(v.integer);
    if (cfg_obj.get("max_position_embeddings")) |v| config.max_position_embeddings = @intCast(v.integer);
    if (cfg_obj.get("rms_norm_eps")) |v| config.rms_norm_eps = jsonFloat(v);
    if (cfg_obj.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
    if (cfg_obj.get("query_pre_attn_scalar")) |v| config.query_pre_attn_scalar = @intCast(v.integer);

    // MoE fields (guard against JSON null values)
    if (cfg_obj.get("num_experts")) |v| { if (v == .integer) config.num_experts = @intCast(v.integer); }
    if (cfg_obj.get("num_experts_per_tok")) |v| { if (v == .integer) config.num_experts_per_tok = @intCast(v.integer); }
    if (cfg_obj.get("top_k_experts")) |v| { if (v == .integer) config.num_experts_per_tok = @intCast(v.integer); }
    if (cfg_obj.get("moe_intermediate_size")) |v| { if (v == .integer) config.moe_intermediate_size = @intCast(v.integer); }
    if (cfg_obj.get("shared_expert_intermediate_size")) |v| { if (v == .integer) config.shared_expert_intermediate_size = @intCast(v.integer); }

    // Linear attention (GatedDeltaNet) fields
    if (cfg_obj.get("linear_num_key_heads")) |v| config.linear_num_key_heads = @intCast(v.integer);
    if (cfg_obj.get("linear_num_value_heads")) |v| config.linear_num_value_heads = @intCast(v.integer);
    if (cfg_obj.get("linear_key_head_dim")) |v| config.linear_key_head_dim = @intCast(v.integer);
    if (cfg_obj.get("linear_value_head_dim")) |v| config.linear_value_head_dim = @intCast(v.integer);
    if (cfg_obj.get("linear_conv_kernel_dim")) |v| config.linear_conv_kernel_dim = @intCast(v.integer);

    // Hybrid attention
    if (cfg_obj.get("full_attention_interval")) |v| config.full_attention_interval = @intCast(v.integer);
    if (cfg_obj.get("attn_output_gate")) |v| {
        if (v == .bool) config.attn_output_gate = v.bool;
    }

    // Bidirectional-attention embedding models (EmbeddingGemma): a decoder
    // arch trained as an encoder. Routes to the encoder forward + the
    // /v1/embeddings surface; chat surfaces reject it.
    if (cfg_obj.get("use_bidirectional_attention")) |v| {
        if (v == .bool and v.bool) {
            config.use_bidirectional_attention = true;
            config.is_encoder_only = true;
        }
    }
    // Explicit pooling contract (issue #116): "mean" | "cls" | "last_token" in
    // config.json marks a checkpoint as an embedding model and picks the pool
    // op. An unknown value is a parse error, never a silent mean-pool —
    // wrong-semantics vectors are harder to detect than a refused load.
    if (root.get("pooling_mode")) |v| {
        if (v == .string) {
            config.pooling_mode = PoolingMode.fromString(v.string) orelse
                return error.UnsupportedPoolingMode;
        }
    }
    if (cfg_obj.get("bos_token_id")) |v| {
        if (v == .integer and v.integer >= 0) config.bos_token_id = @intCast(v.integer);
    }

    // Rope parameters (nested for Qwen3.5)
    if (cfg_obj.get("rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            if (rp_val.object.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
            if (rp_val.object.get("partial_rotary_factor")) |v| config.partial_rotary_factor = jsonFloat(v);
        }
    }

    // Sliding window
    if (cfg_obj.get("sliding_window")) |v| {
        if (v == .null) {
            config.has_sliding_window = false;
        } else {
            config.sliding_window = @intCast(v.integer);
            config.has_sliding_window = true;
        }
    }
    if (cfg_obj.get("sliding_window_pattern")) |v| config.sliding_window_pattern = @intCast(v.integer);

    // Gemma-specific: dual RoPE bases
    if (cfg_obj.get("rope_local_base_freq")) |v| config.rope_local_base_freq = jsonFloat(v);
    if (cfg_obj.get("rope_scaling")) |rs_val| {
        if (rs_val == .object) {
            if (rs_val.object.get("factor")) |v| config.rope_scaling_factor = jsonFloat(v);
        }
    }

    // Gemma 4: explicit layer_types array
    if (cfg_obj.get("layer_types")) |lt_val| {
        if (lt_val == .array) {
            config.has_explicit_layer_types = true;
            for (lt_val.array.items, 0..) |item, i| {
                if (i >= 128) break;
                if (item == .string) {
                    config.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
                }
            }
        }
    }

    // Gemma 4: dual head dimensions and KV sharing
    if (cfg_obj.get("global_head_dim")) |v| {
        if (v == .integer) config.global_head_dim = @intCast(v.integer);
    }
    if (cfg_obj.get("num_global_key_value_heads")) |v| {
        if (v == .integer) config.num_global_key_value_heads = @intCast(v.integer);
    }
    if (cfg_obj.get("num_kv_shared_layers")) |v| {
        if (v == .integer) config.num_kv_shared_layers = @intCast(v.integer);
    }
    if (cfg_obj.get("attention_k_eq_v")) |v| {
        if (v == .bool) config.attention_k_eq_v = v.bool;
    }
    if (cfg_obj.get("final_logit_softcapping")) |v| {
        config.final_logit_softcapping = jsonFloat(v);
    }
    if (cfg_obj.get("hidden_size_per_layer_input")) |v| {
        if (v == .integer) config.hidden_size_per_layer_input = @intCast(v.integer);
    }

    // Gemma 4: nested rope_parameters with per-attention-type config
    if (cfg_obj.get("rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            // Gemma 4 style: { "full_attention": {...}, "sliding_attention": {...} }
            if (rp_val.object.get("full_attention")) |fa| {
                if (fa == .object) {
                    if (fa.object.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
                    if (fa.object.get("partial_rotary_factor")) |v| config.partial_rotary_factor_global = jsonFloat(v);
                    if (fa.object.get("rope_type")) |v| {
                        if (v == .string and std.mem.eql(u8, v.string, "proportional")) {
                            config.rope_proportional = true;
                            if (fa.object.get("factor")) |fv| config.rope_proportional_factor = jsonFloat(fv);
                        }
                    }
                }
            }
            if (rp_val.object.get("sliding_attention")) |sa| {
                if (sa == .object) {
                    if (sa.object.get("rope_theta")) |v| config.rope_local_base_freq = jsonFloat(v);
                }
            }
            // Qwen3.5 style: { "rope_theta": ..., "partial_rotary_factor": ... }
            if (rp_val.object.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
            if (rp_val.object.get("partial_rotary_factor")) |v| config.partial_rotary_factor = jsonFloat(v);
        }
    }

    // Tie word embeddings
    if (root.get("tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }
    if (cfg_obj.get("tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }

    // Check root level for max_position_embeddings (may not be in text_config)
    if (config.max_position_embeddings == 0) {
        if (root.get("max_position_embeddings")) |v| {
            if (v == .integer) config.max_position_embeddings = @intCast(v.integer);
        }
    }

    // Parse quantization from top level
    if (root.get("quantization")) |q_val| {
        const q = q_val.object;
        if (q.get("bits")) |v| config.quant_bits = @intCast(v.integer);
        if (q.get("group_size")) |v| config.quant_group_size = @intCast(v.integer);
        if (q.get("mode")) |v| {
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
    if (root.get("eos_token_id")) |v| {
        switch (v) {
            .integer => |i| config.addEosToken(@intCast(i)),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .integer) config.addEosToken(@intCast(item.integer));
                }
            },
            else => {},
        }
    }

    // Vision config (Gemma 4 SigLIP)
    if (root.get("vision_config")) |vc_val| {
        if (vc_val == .object) {
            config.has_vision = true;
            const vc = vc_val.object;
            if (vc.get("hidden_size")) |v| { if (v == .integer) config.vision_hidden_size = @intCast(v.integer); }
            if (vc.get("num_hidden_layers")) |v| { if (v == .integer) config.vision_num_layers = @intCast(v.integer); }
            if (vc.get("num_attention_heads")) |v| { if (v == .integer) config.vision_num_heads = @intCast(v.integer); }
            if (vc.get("head_dim")) |v| { if (v == .integer) config.vision_head_dim = @intCast(v.integer); }
            if (vc.get("global_head_dim")) |v| { if (v == .integer) config.vision_head_dim = @intCast(v.integer); }
            if (vc.get("intermediate_size")) |v| { if (v == .integer) config.vision_intermediate_size = @intCast(v.integer); }
            if (vc.get("patch_size")) |v| { if (v == .integer) config.vision_patch_size = @intCast(v.integer); }
            if (vc.get("pooling_kernel_size")) |v| { if (v == .integer) config.vision_pooling_kernel = @intCast(v.integer); }
            if (vc.get("default_output_length")) |v| { if (v == .integer) config.vision_soft_tokens = @intCast(v.integer); }
            if (vc.get("position_embedding_size")) |v| { if (v == .integer) config.vision_position_embedding_size = @intCast(v.integer); }
            if (vc.get("rope_parameters")) |rp| {
                if (rp == .object) {
                    if (rp.object.get("rope_theta")) |v| config.vision_rope_theta = jsonFloat(v);
                }
            }
            if (vc.get("use_clipped_linears")) |v| { if (v == .bool) config.vision_use_clipped_linears = v.bool; }
            // vision_config.standardize is presence-only — the actual `std_scale`/`std_bias`
            // safetensors presence drives behavior in `VisionEncoder.init`, so the config
            // flag needs no field.

            // Gemma 4 12B unified (encoder-free) vision fields. Distinct names
            // from the SigLIP tower: mm_embed_dim (vs hidden_size),
            // model_patch_size (48px merged patch vs 16px teacher patch_size),
            // num_soft_tokens (vs default_output_length), mm_posemb_size.
            if (vc.get("mm_embed_dim")) |v| { if (v == .integer) config.vision_mm_embed_dim = @intCast(v.integer); }
            if (vc.get("model_patch_size")) |v| { if (v == .integer) config.vision_model_patch_size = @intCast(v.integer); }
            if (vc.get("num_soft_tokens")) |v| { if (v == .integer) config.vision_soft_tokens = @intCast(v.integer); }
            if (vc.get("mm_posemb_size")) |v| { if (v == .integer) config.vision_mm_posemb_size = @intCast(v.integer); }
        }
    }
    // Audio config (Gemma 4 12B unified — raw-waveform projection, no conformer)
    if (root.get("audio_config")) |ac_val| {
        if (ac_val == .object) {
            const ac = ac_val.object;
            if (ac.get("audio_embed_dim")) |v| { if (v == .integer) config.audio_embed_dim = @intCast(v.integer); }
            // audio_samples_per_token lives in processor_config, not config.json;
            // default 640 (40ms @ 16kHz) matches the only shipped unified checkpoint.
            if (ac.get("audio_samples_per_token")) |v| { if (v == .integer) config.audio_samples_per_token = @intCast(v.integer); }
        }
    }
    if (root.get("audio_token_id")) |v| {
        if (v == .integer) config.audio_token_id = @intCast(v.integer);
    }
    if (root.get("boa_token_id")) |v| {
        if (v == .integer) config.boa_token_id = @intCast(v.integer);
    }
    // eoa lives under `eoa_token_index` in the unified config.json.
    if (root.get("eoa_token_index")) |v| {
        if (v == .integer) config.eoa_token_id = @intCast(v.integer);
    }
    if (root.get("eoa_token_id")) |v| {
        if (v == .integer and config.eoa_token_id == 0) config.eoa_token_id = @intCast(v.integer);
    }
    // Image token ID (top-level or in mm_tokens_per_image config)
    if (root.get("image_token_id")) |v| {
        if (v == .integer) config.image_token_id = @intCast(v.integer);
    }
    if (root.get("image_token_index")) |v| {
        if (v == .integer and config.image_token_id == 0) config.image_token_id = @intCast(v.integer);
    }
    if (root.get("boi_token_id")) |v| {
        if (v == .integer) config.boi_token_id = @intCast(v.integer);
    }
    if (root.get("eoi_token_id")) |v| {
        if (v == .integer) config.eoi_token_id = @intCast(v.integer);
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
        if (cfg_obj.get("num_attention_heads") == null) config.num_attention_heads = 8;
        if (cfg_obj.get("num_key_value_heads") == null) config.num_key_value_heads = 4;
        if (cfg_obj.get("head_dim") == null) config.head_dim = 256;
        // Multimodal checkpoint → "language_model.model"; flat text-only
        // (gemma3_text, no text_config) → "model" (mirrors the LFM2 VL split).
        config.weight_prefix = if (root.get("text_config") != null) "language_model.model" else "model";
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
            if (cfg_obj.get("rope_scaling")) |rs_val| {
                if (rs_val == .object) {
                    if (rs_val.object.get("factor")) |_| {} else {
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
        if (root.get("canvas_length")) |v| {
            if (v == .integer) config.canvas_length = @intCast(v.integer);
        }
        if (config.canvas_length == 0) config.canvas_length = 256;
        // Diffusion knobs from the embedded generation_config object.
        if (root.get("generation_config")) |gc_val| {
            if (gc_val == .object) {
                const gc = gc_val.object;
                if (gc.get("max_denoising_steps")) |v| { if (v == .integer) config.diffusion_max_steps = @intCast(v.integer); }
                if (gc.get("t_min")) |v| config.diffusion_t_min = jsonFloat(v);
                if (gc.get("t_max")) |v| config.diffusion_t_max = jsonFloat(v);
                if (gc.get("confidence_threshold")) |v| config.diffusion_confidence_threshold = jsonFloat(v);
                if (gc.get("stability_threshold")) |v| { if (v == .integer) config.diffusion_stability_threshold = @intCast(v.integer); }
                if (gc.get("pad_token_id")) |v| { if (v == .integer) config.diffusion_pad_token = @intCast(v.integer); }
                if (gc.get("sampler_config")) |sc_val| {
                    if (sc_val == .object) {
                        if (sc_val.object.get("entropy_bound")) |v| config.diffusion_entropy_bound = jsonFloat(v);
                    }
                }
                // EOS may also live here (mirrors the top-level list).
                if (config.num_eos_tokens == 0) {
                    if (gc.get("eos_token_id")) |v| {
                        switch (v) {
                            .integer => |i| config.addEosToken(@intCast(i)),
                            .array => |arr| for (arr.items) |item| {
                                if (item == .integer) config.addEosToken(@intCast(item.integer));
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
    } else if (std.mem.eql(u8, model_type, "qwen3_5_moe") or
        std.mem.eql(u8, model_type, "qwen3_5") or
        std.mem.eql(u8, model_type, "qwen3_5_moe_text") or
        std.mem.eql(u8, model_type, "qwen3_5_text"))
    {
        config.model_type = "qwen3_5_moe";
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
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        // Qwen3-VL vision tower (separate fields from the Gemma SigLIP block;
        // see src/qwen_vision.zig). The generic vision_config block above already
        // set has_vision; here we read Qwen's distinct keys into qv_*.
        if (root.get("vision_config")) |vc_val| {
            if (vc_val == .object) {
                const vc = vc_val.object;
                config.qwen_vision = true;
                if (vc.get("depth")) |v| { if (v == .integer) config.qv_depth = @intCast(v.integer); }
                if (vc.get("hidden_size")) |v| { if (v == .integer) config.qv_hidden = @intCast(v.integer); }
                if (vc.get("num_heads")) |v| { if (v == .integer) config.qv_heads = @intCast(v.integer); }
                if (vc.get("intermediate_size")) |v| { if (v == .integer) config.qv_intermediate = @intCast(v.integer); }
                if (vc.get("patch_size")) |v| { if (v == .integer) config.qv_patch = @intCast(v.integer); }
                if (vc.get("temporal_patch_size")) |v| { if (v == .integer) config.qv_temporal_patch = @intCast(v.integer); }
                if (vc.get("spatial_merge_size")) |v| { if (v == .integer) config.qv_merge = @intCast(v.integer); }
                if (vc.get("num_position_embeddings")) |v| { if (v == .integer) config.qv_num_pos_emb = @intCast(v.integer); }
                if (vc.get("out_hidden_size")) |v| { if (v == .integer) config.qv_out_hidden = @intCast(v.integer); }
                if (config.qv_heads != 0) config.qv_head_dim = config.qv_hidden / config.qv_heads;
                if (config.qv_out_hidden == 0) config.qv_out_hidden = config.hidden_size;
            }
        }
        // Interleaved M-RoPE sections (text_config.rope_parameters). rope_theta /
        // partial_rotary_factor already parsed in the generic rope block above.
        if (cfg_obj.get("rope_parameters")) |rp| {
            if (rp == .object) {
                if (rp.object.get("mrope_interleaved")) |v| { if (v == .bool) config.mrope_interleaved = v.bool; }
                if (rp.object.get("mrope_section")) |v| {
                    if (v == .array) {
                        for (v.array.items, 0..) |item, i| {
                            if (i >= 3) break;
                            if (item == .integer) config.mrope_section[i] = @intCast(item.integer);
                        }
                    }
                }
            }
        }
        // Qwen vision token ids (top-level).
        if (root.get("video_token_id")) |v| { if (v == .integer) config.video_token_id = @intCast(v.integer); }
        if (root.get("vision_start_token_id")) |v| { if (v == .integer) config.vision_start_token_id = @intCast(v.integer); }
        if (root.get("vision_end_token_id")) |v| { if (v == .integer) config.vision_end_token_id = @intCast(v.integer); }
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
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
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
        if (cfg_obj.get("qk_norm")) |v| { if (v == .bool) config.has_qk_norm = v.bool; }
        if (cfg_obj.get("route_norm")) |v| { if (v == .bool) config.moe_route_norm = v.bool; }
        if (cfg_obj.get("router_scaling_factor")) |v| config.router_scaling_factor = jsonFloat(v);
        if (cfg_obj.get("first_k_dense_replace")) |v| { if (v == .integer) config.first_k_dense_replace = @intCast(v.integer); }
        // Expert width may ride as expert_hidden_dim when moe_intermediate_size
        // is absent (both = 1536 on the 295B).
        if (config.moe_intermediate_size == 0) {
            if (cfg_obj.get("expert_hidden_dim")) |v| { if (v == .integer) config.moe_intermediate_size = @intCast(v.integer); }
        }
        // No explicit shared_expert_intermediate_size key in hy_v3 configs:
        // derive num_shared_experts × expert width. 0 shared experts leaves it
        // 0 and the binder/forward skip the shared branch.
        if (cfg_obj.get("num_shared_experts")) |v| {
            if (v == .integer) config.shared_expert_intermediate_size =
                @as(u32, @intCast(v.integer)) * config.moe_intermediate_size;
        }
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        config.ensureHy3Terminators();
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
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        // Router: norm_topk_prob → route_norm; moe_routed_scaling_factor → scale.
        if (cfg_obj.get("norm_topk_prob")) |v| { if (v == .bool) config.moe_route_norm = v.bool; }
        if (cfg_obj.get("moe_routed_scaling_factor")) |v| config.router_scaling_factor = jsonFloat(v);
        if (cfg_obj.get("moe_router_logit_softcapping")) |v| config.moe_router_logit_softcapping = jsonFloat(v);
        // Router logit soft-capping is off on the shipped checkpoint and untested;
        // reject a >0 value rather than silently ignore it (honest reject).
        if (config.moe_router_logit_softcapping > 0.0) {
            log.err("laguna: moe_router_logit_softcapping > 0 not supported in v1 (got {d})\n", .{config.moe_router_logit_softcapping});
            return error.UnsupportedLagunaConfig;
        }
        // Only per-head gating is implemented (assert uniform; honest reject).
        if (cfg_obj.get("gating")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "per-head") and !std.mem.eql(u8, v.string, "per_head")) {
                log.err("laguna: only per-head attention gating supported (got '{s}')\n", .{v.string});
                return error.UnsupportedLagunaConfig;
            }
        }
        // Per-layer Q-head count (cap 128 like layer_is_global).
        if (cfg_obj.get("num_attention_heads_per_layer")) |v| {
            if (v == .array) {
                config.has_per_layer_heads = true;
                for (v.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item == .integer) config.num_attention_heads_per_layer[i] = @intCast(item.integer);
                }
            }
        }
        // YaRN on full-attention layers. The generic nested-rope block above
        // already set rope_theta (5e5), partial_rotary_factor_global (0.5) from
        // full_attention and rope_local_base_freq (1e4) from sliding_attention;
        // here we pull the YaRN-specific fields and flag the precompute.
        if (cfg_obj.get("rope_parameters")) |rp| {
            if (rp == .object) {
                if (rp.object.get("full_attention")) |fa_val| {
                    if (fa_val == .object) {
                        const fa = fa_val.object;
                        const is_yarn = if (fa.get("rope_type")) |rt|
                            (rt == .string and std.mem.eql(u8, rt.string, "yarn"))
                        else
                            false;
                        if (is_yarn) {
                            config.rope_yarn = true;
                            if (fa.get("factor")) |x| config.yarn_factor = jsonFloat(x);
                            if (fa.get("beta_fast")) |x| config.yarn_beta_fast = jsonFloat(x);
                            if (fa.get("beta_slow")) |x| config.yarn_beta_slow = jsonFloat(x);
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
                            if (fa.get("original_max_position_embeddings")) |x| {
                                if (x == .integer) config.yarn_orig_max_pos = @intCast(x.integer);
                            }
                        }
                    }
                }
            }
        }
        // eos [2, 24] (〈|EOS|〉, </assistant>) parsed generically from
        // eos_token_id above; no additive terminator merge needed.
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
        config.query_pre_attn_scalar = config.head_dim * config.head_dim;
        // The checkpoint labels the MoE expert width `intermediate_size` (read
        // by the generic block above) and the dense bottom-layer width
        // `dense_intermediate_size` — opposite of our field meanings. Swap.
        config.moe_intermediate_size = config.intermediate_size;
        if (cfg_obj.get("dense_intermediate_size")) |v| {
            if (v == .integer) {
                config.intermediate_size = @intCast(v.integer);
                config.intermediate_size_declared = true;
            }
        }
        if (cfg_obj.get("dense_mlp_idx")) |v| { if (v == .integer) config.first_k_dense_replace = @intCast(v.integer); }
        if (cfg_obj.get("n_routed_experts")) |v| { if (v == .integer) config.num_experts = @intCast(v.integer); }
        if (cfg_obj.get("n_shared_experts")) |v| { if (v == .integer) config.inkling_n_shared_experts = @intCast(v.integer); }
        if (cfg_obj.get("route_scale")) |v| config.router_scaling_factor = jsonFloat(v);
        // Position machinery.
        if (cfg_obj.get("d_rel")) |v| { if (v == .integer) config.inkling_d_rel = @intCast(v.integer); }
        if (cfg_obj.get("rel_extent")) |v| { if (v == .integer) config.inkling_rel_extent = @intCast(v.integer); }
        if (cfg_obj.get("log_scaling_n_floor")) |v| { if (v == .integer) config.inkling_log_n_floor = @intCast(v.integer); }
        if (cfg_obj.get("log_scaling_alpha")) |v| config.inkling_log_alpha = jsonFloat(v);
        if (cfg_obj.get("sconv_kernel_size")) |v| { if (v == .integer) config.inkling_sconv_kernel = @intCast(v.integer); }
        if (cfg_obj.get("use_sconv")) |v| { if (v == .bool and !v.bool) config.inkling_sconv_kernel = 0; }
        // Embedding norm (use_embed_norm, default true for this family).
        config.has_embedding_norm = true;
        if (cfg_obj.get("use_embed_norm")) |v| { if (v == .bool) config.has_embedding_norm = v.bool; }
        // Hybrid sliding/global: the config names LOCAL (sliding) layers and
        // uses `sliding_window_size` (the generic block reads `sliding_window`).
        if (cfg_obj.get("sliding_window_size")) |v| {
            if (v == .integer) {
                config.sliding_window = @intCast(v.integer);
                config.has_sliding_window = true;
            }
        }
        if (cfg_obj.get("local_layer_ids")) |v| {
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
        if (cfg_obj.get("logits_mup_width_multiplier")) |v| config.logits_mup_width_multiplier = jsonFloat(v);
        if (cfg_obj.get("unpadded_vocab_size")) |v| { if (v == .integer) config.unpadded_vocab_size = @intCast(v.integer); }
        if (cfg_obj.get("model_max_length")) |v| { if (v == .integer) config.max_position_embeddings = @intCast(v.integer); }
        // v1 is text-only: the hMLP vision_config must not arm the SigLIP path
        // (the generic vision_config block above set has_vision = true).
        config.has_vision = false;
        // Honest rejects: the forward implements exactly the shipped geometry
        // and router formula. A checkpoint that diverges must refuse to load,
        // not run silently wrong.
        const swa_heads: u32 = if (cfg_obj.get("swa_num_attention_heads")) |v| @intCast(v.integer) else config.num_attention_heads;
        const swa_kv: u32 = if (cfg_obj.get("swa_num_key_value_heads")) |v| @intCast(v.integer) else config.num_key_value_heads;
        const swa_hd: u32 = if (cfg_obj.get("swa_head_dim")) |v| @intCast(v.integer) else config.head_dim;
        if (swa_heads != config.num_attention_heads or swa_kv != config.num_key_value_heads or swa_hd != config.head_dim) {
            log.err("inkling: sliding-attention geometry {d}/{d}/{d} differs from global {d}/{d}/{d} — not supported\n", .{ swa_heads, swa_kv, swa_hd, config.num_attention_heads, config.num_key_value_heads, config.head_dim });
            return error.UnsupportedInklingConfig;
        }
        if (cfg_obj.get("gate_activation")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "sigmoid")) {
                log.err("inkling: gate_activation '{s}' not supported (sigmoid only)\n", .{v.string});
                return error.UnsupportedInklingConfig;
            }
        }
    } else if (std.mem.eql(u8, model_type, "deepseek_v4")) {
        // DeepSeek V4 Flash (284B-A13B, 1M ctx). See the dsv4_* field block
        // for the architecture summary; reference is the release's own
        // inference/{model,kernel}.py (torch). Loaded from OUR converted
        // mixed-quant mirror (tests/convert_dsv4_weights.py) — bare
        // inference-style tensor names, stacked expert banks.
        config.model_type = "deepseek_v4";
        config.weight_prefix = ""; // release ships bare names (embed.weight, layers.N....)
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false; // q-norm is on the lora rank + unweighted per-head RMS, handled in-arch
        config.hidden_act = .silu;
        if (cfg_obj.get("n_routed_experts")) |v| { if (v == .integer) config.num_experts = @intCast(v.integer); }
        if (cfg_obj.get("num_hash_layers")) |v| { if (v == .integer) config.dsv4_hash_layers = @intCast(v.integer); }
        if (cfg_obj.get("routed_scaling_factor")) |v| config.router_scaling_factor = jsonFloat(v);
        if (cfg_obj.get("norm_topk_prob")) |v| { if (v == .bool) config.moe_route_norm = v.bool; }
        if (cfg_obj.get("q_lora_rank")) |v| { if (v == .integer) config.dsv4_q_lora_rank = @intCast(v.integer); }
        if (cfg_obj.get("o_lora_rank")) |v| { if (v == .integer) config.dsv4_o_lora_rank = @intCast(v.integer); }
        if (cfg_obj.get("o_groups")) |v| { if (v == .integer) config.dsv4_o_groups = @intCast(v.integer); }
        if (cfg_obj.get("qk_rope_head_dim")) |v| { if (v == .integer) config.dsv4_rope_head_dim = @intCast(v.integer); }
        if (cfg_obj.get("index_n_heads")) |v| { if (v == .integer) config.dsv4_index_n_heads = @intCast(v.integer); }
        if (cfg_obj.get("index_head_dim")) |v| { if (v == .integer) config.dsv4_index_head_dim = @intCast(v.integer); }
        if (cfg_obj.get("index_topk")) |v| { if (v == .integer) config.dsv4_index_topk = @intCast(v.integer); }
        if (cfg_obj.get("hc_mult")) |v| { if (v == .integer) config.dsv4_hc_mult = @intCast(v.integer); }
        if (cfg_obj.get("hc_sinkhorn_iters")) |v| { if (v == .integer) config.dsv4_hc_sinkhorn_iters = @intCast(v.integer); }
        if (cfg_obj.get("hc_eps")) |v| config.dsv4_hc_eps = jsonFloat(v);
        if (cfg_obj.get("swiglu_limit")) |v| config.dsv4_swiglu_limit = jsonFloat(v);
        if (cfg_obj.get("compress_rope_theta")) |v| config.dsv4_compress_rope_theta = jsonFloat(v);
        if (cfg_obj.get("num_nextn_predict_layers")) |v| { if (v == .integer) config.dsv4_mtp_layers = @intCast(v.integer); }
        if (cfg_obj.get("dspark_block_size")) |v| { if (v == .integer) config.dsv4_dspark_block_size = @intCast(v.integer); }
        if (cfg_obj.get("dspark_noise_token_id")) |v| { if (v == .integer) config.dsv4_dspark_noise_token_id = @intCast(v.integer); }
        if (cfg_obj.get("dspark_markov_rank")) |v| { if (v == .integer) config.dsv4_dspark_markov_rank = @intCast(v.integer); }
        if (cfg_obj.get("dspark_target_layer_ids")) |v| {
            if (v == .array) {
                for (v.array.items, 0..) |item, i| {
                    if (i >= config.dsv4_dspark_target_layers.len) break;
                    if (item == .integer) config.dsv4_dspark_target_layers[i] = @intCast(item.integer);
                }
                config.dsv4_n_dspark_target_layers = @intCast(@min(v.array.items.len, config.dsv4_dspark_target_layers.len));
            }
        }
        if (cfg_obj.get("compress_ratios")) |v| {
            if (v == .array) {
                for (v.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item == .integer) config.dsv4_compress_ratios[i] = @intCast(item.integer);
                }
                config.dsv4_n_compress_ratios = @intCast(@min(v.array.items.len, 128));
            }
        }
        // YaRN on compressed layers only; the reference applies NO mscale
        // (softmax scale stays head_dim^-0.5 everywhere), so
        // yarn_attention_factor stays 1.0 — do not compute the 0.1·ln(f)+1
        // default here (laguna-class trap in the other direction).
        if (cfg_obj.get("rope_scaling")) |rs| {
            if (rs == .object) {
                config.rope_yarn = true;
                if (rs.object.get("factor")) |x| config.yarn_factor = jsonFloat(x);
                if (rs.object.get("beta_fast")) |x| config.yarn_beta_fast = jsonFloat(x);
                if (rs.object.get("beta_slow")) |x| config.yarn_beta_slow = jsonFloat(x);
                if (rs.object.get("original_max_position_embeddings")) |x| {
                    if (x == .integer) config.yarn_orig_max_pos = @intCast(x.integer);
                }
            }
        }
        // Honest rejects: the forward implements exactly sqrt(softplus)
        // scoring with selection-only bias (noaux_tc), ONE always-on shared
        // expert, and a single shared KV latent. Divergent checkpoints must
        // refuse to load, not run silently wrong.
        if (cfg_obj.get("scoring_func")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "sqrtsoftplus")) {
                log.err("deepseek_v4: scoring_func '{s}' not supported (sqrtsoftplus only)\n", .{v.string});
                return error.UnsupportedDsv4Config;
            }
        }
        if (cfg_obj.get("topk_method")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "noaux_tc")) {
                log.err("deepseek_v4: topk_method '{s}' not supported (noaux_tc only)\n", .{v.string});
                return error.UnsupportedDsv4Config;
            }
        }
        if (cfg_obj.get("n_shared_experts")) |v| {
            if (v == .integer and v.integer != 1) {
                log.err("deepseek_v4: n_shared_experts {d} not supported (exactly 1)\n", .{v.integer});
                return error.UnsupportedDsv4Config;
            }
        }
        if (config.num_key_value_heads != 1) {
            log.err("deepseek_v4: num_key_value_heads {d} not supported (single shared KV latent)\n", .{config.num_key_value_heads});
            return error.UnsupportedDsv4Config;
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
            log.err("deepseek_v4: this is the superseded PREVIEW checkpoint (num_nextn_predict_layers={d}, no dspark_* config). Use DeepSeek-V4-Flash-0731 or later.\n", .{config.dsv4_mtp_layers});
            return error.UnsupportedDsv4Config;
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
        if (cfg_obj.get("partial_rotary_factor")) |v| config.partial_rotary_factor = jsonFloat(v);
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "lfm2") or std.mem.startsWith(u8, model_type, "lfm2")) {
        config.model_type = "lfm2";
        // VL variant nests text weights under language_model.model (like Gemma 4)
        config.weight_prefix = if (root.get("text_config") != null) "language_model.model" else "model";
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
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }
        config.query_pre_attn_scalar = config.head_dim;
        // tie_embedding (LFM2 name) -> tie_word_embeddings; default true for LFM2
        config.tie_word_embeddings = true;
        if (cfg_obj.get("tie_embedding")) |v| {
            if (v == .bool) config.tie_word_embeddings = v.bool;
        }
        if (cfg_obj.get("norm_eps")) |v| config.rms_norm_eps = jsonFloat(v);
        if (cfg_obj.get("conv_dim")) |v| config.lfm_conv_dim = switch (v) {
            .integer => |i| @intCast(i),
            else => 0,
        };
        // Parse layer_types array: ["conv", "full_attention", ...]
        if (cfg_obj.get("layer_types")) |lt_val| {
            if (lt_val == .array) {
                for (lt_val.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item == .string) {
                        config.layer_block_types[i] = if (std.mem.eql(u8, item.string, "conv"))
                            .gated_conv
                        else
                            .attention;
                    }
                }
            }
        }
        if (config.num_eos_tokens == 0) {
            if (cfg_obj.get("eos_token_id")) |v| {
                if (v == .integer) config.addEosToken(@intCast(v.integer));
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
        if (cfg_obj.get("rms_norm_eps")) |v| {
            config.rms_norm_eps = jsonFloat(v);
        } else if (cfg_obj.get("layer_norm_epsilon")) |v| {
            config.rms_norm_eps = jsonFloat(v);
        }
        // Mamba2-specific config
        if (cfg_obj.get("mamba_num_heads")) |v| config.mamba_num_heads = switch (v) { .integer => |i| @intCast(i), else => 0 };
        if (cfg_obj.get("mamba_head_dim")) |v| config.mamba_head_dim = switch (v) { .integer => |i| @intCast(i), else => 0 };
        if (cfg_obj.get("n_groups")) |v| config.mamba_n_groups = switch (v) { .integer => |i| @intCast(i), else => 8 };
        if (cfg_obj.get("ssm_state_size")) |v| config.ssm_state_size = switch (v) { .integer => |i| @intCast(i), else => 128 };
        if (cfg_obj.get("conv_kernel")) |v| config.mamba_conv_kernel = switch (v) { .integer => |i| @intCast(i), else => 4 };
        if (cfg_obj.get("expand")) |v| config.mamba_expand = switch (v) { .integer => |i| @intCast(i), else => 2 };
        // time_step_limit: Python defaults to (0.0, inf) if not in config.
        // config.json may have time_step_min/time_step_max fields but Python ignores them
        // for SSM clipping — only time_step_limit (a 2-element array) is used.
        if (cfg_obj.get("time_step_limit")) |v| {
            if (v == .array) {
                const items = v.array.items;
                if (items.len >= 2) {
                    config.time_step_min = jsonFloat(items[0]);
                    config.time_step_max = jsonFloat(items[1]);
                }
            }
        }
        if (cfg_obj.get("chunk_size")) |v| config.mamba_chunk_size = switch (v) { .integer => |i| @intCast(i), else => 256 };
        // Parse hybrid_override_pattern: "M-M-M-MM-M-M*-..."
        if (cfg_obj.get("hybrid_override_pattern")) |v| {
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
        if (config.num_eos_tokens == 0) {
            if (cfg_obj.get("eos_token_id")) |v| {
                if (v == .integer) config.addEosToken(@intCast(v.integer));
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
        config.head_dim = config.hidden_size / config.num_attention_heads;
        config.num_key_value_heads = config.num_attention_heads;
        config.query_pre_attn_scalar = config.head_dim;
        if (cfg_obj.get("layer_norm_eps")) |v| config.layer_norm_eps = jsonFloat(v);
        if (cfg_obj.get("type_vocab_size")) |v| config.type_vocab_size = switch (v) {
            .integer => |i| @intCast(i),
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
        } else {
            config.model_type = "unknown";
        }
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        // Llama-family models (qwen2, llama, mistral) usually omit `head_dim`;
        // the HF default is hidden_size / num_attention_heads. Without this the
        // stale 256 sentinel (line 53) would corrupt attention for any such
        // checkpoint that doesn't ship an explicit head_dim (e.g. Qwen2.5).
        // qwen3 ships an explicit head_dim, so this leaves it untouched.
        if (cfg_obj.get("head_dim") == null) {
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }

        if (cfg_obj.get("hidden_act")) |v| {
            if (v == .string) {
                if (std.mem.eql(u8, v.string, "silu")) {
                    config.hidden_act = .silu;
                } else if (std.mem.eql(u8, v.string, "gelu_pytorch_tanh")) {
                    config.hidden_act = .gelu_approx;
                }
            }
        }

        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }

        if (std.mem.eql(u8, model_type, "qwen3")) {
            config.has_qk_norm = true;
        }
    }

    return config;
}

fn jsonFloat(v: std.json.Value) f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => 0.0,
    };
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

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// The two nestings a text trunk ships under. `parseConfigFromJson` picks
/// between them from config KEYS; this probe corrects it from the checkpoint.
const FLAT_PREFIX = "model";
const NESTED_PREFIX = "language_model.model";

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
/// Conservative by construction: it only ever alternates between those two
/// prefixes (never an arch with its own — `backbone`, `model.llm`, `""`), and
/// only when the configured one holds NOTHING, so every checkpoint that
/// already loaded binds byte-identically.
pub fn resolveWeightPrefix(config: *ModelConfig, weights: *const Weights) void {
    const alt = if (std.mem.eql(u8, config.weight_prefix, FLAT_PREFIX))
        NESTED_PREFIX
    else if (std.mem.eql(u8, config.weight_prefix, NESTED_PREFIX))
        FLAT_PREFIX
    else
        return;

    if (hasWeightsUnder(weights, config.weight_prefix)) return;
    if (!hasWeightsUnder(weights, alt)) return;

    log.info("weight prefix: config implies \"{s}\", checkpoint uses \"{s}\" — using the checkpoint's\n", .{ config.weight_prefix, alt });
    config.weight_prefix = alt;
}

/// Load all safetensors files from model_dir.
/// When `load_vision` is true, vision_tower and multi_modal_projector weights are included.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, false);
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
    try loadSafetensorsFile(allocator, &weights, pathz, s, false);

    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} — corrupt or empty safetensors file?\n", .{abs_path});
        return error.NoWeightFiles;
    }
    return weights;
}

pub fn loadWeightsWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, true);
}

fn loadWeightsOpt(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, load_vision: bool) !Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    return loadWeightsFromOpenDir(io, allocator, dir, model_dir, load_vision);
}

/// Load every `*.safetensors` in an already-open `dir` into a Weights map.
/// `model_dir` is the on-disk path string, used both to build the per-file
/// absolute path for `mlx_load_safetensors` and to phrase the error message.
/// Split out of `loadWeightsOpt` so the incomplete-checkpoint guard below is
/// unit-testable against a `tmpDir` (mirrors `model_discovery.discoverModelsInDir`).
fn loadWeightsFromOpenDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8, load_vision: bool) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        // Accept regular files AND symlinks: HuggingFace cache snapshots store
        // every weight file as a symlink into ../../blobs/<hash>. mlx_load_safetensors
        // resolves the link at the OS level, so a symlinked *.safetensors loads fine.
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;

        const path_slice = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        defer allocator.free(path_slice);
        const path = try allocator.dupeSentinel(u8, path_slice, 0);
        defer allocator.free(path);

        log.info("Loading {s}...\n", .{entry.name});
        try loadSafetensorsFile(allocator, &weights, path, s, load_vision);
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
    reportF16Narrowing();
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
fn narrow1dEnabled() bool {
    if (narrow_1d_env) |v| return v;
    const on = blk: {
        const raw = std.c.getenv("MLX_SERVE_F16_NARROW_1D") orelse break :blk true;
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

pub fn loadSafetensorsFile(
    allocator: std.mem.Allocator,
    weights: *Weights,
    path: [*:0]const u8,
    s: mlx.mlx_stream,
    load_vision: bool,
) !void {
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);

    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

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

        if (!shouldKeepWeightKey(key_str, load_vision)) {
            _ = mlx.mlx_array_free(value);
            continue;
        }

        // Read the shape BEFORE the cast frees `value` — a freed handle's
        // ndim is a use-after-free, not a zero.
        const ndim = mlx.mlx_array_ndim(value);
        var final_value = value;
        if (narrowsLoadedF16(key_str, ndim, mlx.mlx_array_dtype(value)) and
            (ndim != 1 or narrow1dEnabled()))
        {
            var cast = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&cast, value, .bfloat16, s));
            _ = mlx.mlx_array_free(value);
            final_value = cast;
            if (ndim == 1) narrowed_1d += 1;
        }

        const owned_key = try allocator.dupe(u8, key_str);
        try weights.map.put(owned_key, final_value);
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
        loadWeightsFromOpenDir(io, allocator, tmp.dir, "/incomplete-model", false),
    );
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
    // No arch opts in today — including the families whose templates merely
    // MENTION enable_thinking, which is not evidence of a thinking-on default.
    for ([_][]const u8{ "qwen3", "qwen3_5_moe", "gemma4", "gemma3_text", "laguna", "deepseek_v4", "inkling_mm_model", "llama", "hy_v3", "lfm2" }) |t| {
        const c = ModelConfig{ .model_type = t };
        try testing.expect(!c.defaultEnableThinking());
    }
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

test "pickUserTurnPrefix Gemma 4 wins over older patterns" {
    // Gemma 4 templates also contain "<start_of_turn>" inside fallback comments
    // in some checkpoints — make sure we still pick the Gemma 4 marker first.
    const tmpl = "{{- '<|turn>' + role + '\n' }} {# legacy: <start_of_turn> #}";
    try testing.expectEqualStrings("<|turn>user\n", pickUserTurnPrefix(tmpl).?);
}

test "pickUserTurnPrefix Gemma 3" {
    const tmpl = "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>";
    try testing.expectEqualStrings("<start_of_turn>user\n", pickUserTurnPrefix(tmpl).?);
}

test "pickUserTurnPrefix Qwen ChatML" {
    const tmpl = "<|im_start|>user\n{{ message['content'] }}<|im_end|>";
    try testing.expectEqualStrings("<|im_start|>user\n", pickUserTurnPrefix(tmpl).?);
}

test "pickUserTurnPrefix Llama 3" {
    const tmpl = "<|start_header_id|>user<|end_header_id|>\n\n{{ content }}<|eot_id|>";
    try testing.expectEqualStrings("<|start_header_id|>user<|end_header_id|>\n\n", pickUserTurnPrefix(tmpl).?);
}

test "pickUserTurnPrefix unknown template returns null" {
    try testing.expect(pickUserTurnPrefix("[INST] {{ content }} [/INST]") == null);
    try testing.expect(pickUserTurnPrefix("") == null);
}

test "ModelConfig userTurnMarkerSlice respects length" {
    var config = ModelConfig{};
    config.user_turn_marker_ids[0] = 105;
    config.user_turn_marker_ids[1] = 2364;
    config.user_turn_marker_ids[2] = 107;
    config.user_turn_marker_len = 3;
    const slice = config.userTurnMarkerSlice();
    try testing.expectEqual(@as(usize, 3), slice.len);
    try testing.expectEqual(@as(u32, 105), slice[0]);
    try testing.expectEqual(@as(u32, 2364), slice[1]);
    try testing.expectEqual(@as(u32, 107), slice[2]);
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
    try testing.expectApproxEqAbs(@as(f32, 42.0), jsonFloat(val), 0.001);
}

test "jsonFloat converts float" {
    const val = std.json.Value{ .float = 3.14 };
    try testing.expectApproxEqAbs(@as(f32, 3.14), jsonFloat(val), 0.01);
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
    try testing.expect(config.isEosToken(1));   // config-declared eos preserved
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
