//! DeepSeek-V4.1-Flash (GGUF arch "deepseek41", ISTA-DASLab's GSQ-RCO sets): a rename onto the deepseek_v41 plugin's
//! names (an MLX pack's once its `language_model.` prefix is stripped). Every tensor is stored in HF's orientation;
//! `attn_output_a` is 3-D, the `[o_groups, o_lora_rank, in]` view the plugin reshapes `wo_a` to anyway. Two layouts:
//! the 4-shard sets keep the Engram tables in their own GGUF beside the set (`engram_file_key`); the 6-shard re-split
//! (llama.cpp's names, the Coder's expert-pruned sets) carries them as shard 6, one `ple_table` whose buckets the
//! `ple.*` keys list. Either way the tables are read in place, and the hash the plugin derives from the config bit
//! for bit (`plugins/deepseek_v41/engram.zig`).
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");

pub const arch_name = "deepseek41";
const p = arch_name ++ ".";

/// The Engram tables' GGUF, named by the set's first shard and found beside it.
pub const engram_file_key = p ++ "engram.file";
/// One table per Engram layer (`blk.<layer>.`), in that file.
pub const engram_leaf = "engram_embd.weight";
/// The re-split's Engram tables: one tensor, each layer's rows after the ones before it (`pleRowStart`).
pub const ple_table = "per_layer_token_embd.weight";
/// The gating id the 4-shard sets pair with `expert_scoring_func` sqrtsoftplus; the re-split keeps only the id.
const sqrt_softplus_gating = 4;

const MAX_LAYERS = 64;
const MAX_ENGRAM = 4;

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    vocab: u32,
    ctx_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_dims: u32,
    q_lora: u32,
    o_lora: u32,
    o_groups: u32,
    window: u32,
    rms_eps: f64,
    rope_theta: f64,
    compress_rope_theta: f64,
    /// YaRN on the compressed layers, factor 0 = none.
    yarn_factor: f64,
    yarn_orig_ctx: u32,
    yarn_beta_fast: f64,
    yarn_beta_slow: f64,
    n_experts: u32,
    n_experts_used: u32,
    n_shared: u32,
    expert_ffn: u32,
    route_scale: f64,
    route_norm: bool,
    /// One clamp for routed and shared experts: the plugin has a single `swiglu_limit`.
    swiglu_limit: f64,
    hash_layers: u32,
    idx_heads: u32,
    idx_dim: u32,
    idx_topk: u32,
    hc: u32,
    hc_iters: u32,
    hc_eps: f64,
    /// Per trunk layer, 0 = window only. The file's list runs on over the DSpark stages it does not ship.
    compress: [MAX_LAYERS]u8,
    /// Bit i = layer i.
    kv_sources: u64,
    index_sources: u64,
    candidate_source: u32,
    candidate_topk: u32,
    candidate_block: u32,
    n_engram: u32,
    engram_layers: [MAX_ENGRAM]u32,
    engram_rows: [MAX_ENGRAM]u64,
    engram_ngram: u32,
    engram_heads: u32,
    engram_dim: u32,
    engram_vocab: u32,
    engram_compressed: u32,
    engram_pad: u32,
    bos: u32,
    eos: u32,

    pub fn read(f: *const gguf.File) !Hparams {
        const n_layers = try arch.u32Key(f, p ++ "block_count");
        if (n_layers == 0 or n_layers > MAX_LAYERS) return error.BadMetadata;
        // The forward serves exactly this routing (`deepseek_v4/config.zig` refuses the rest).
        if (f.getString(p ++ "expert_scoring_func")) |func| {
            if (!std.mem.eql(u8, func, "sqrtsoftplus")) return error.BadMetadata;
        } else if (try arch.u32Key(f, p ++ "expert_gating_func") != sqrt_softplus_gating) return error.BadMetadata;
        var hp = Hparams{
            .n_layers = n_layers,
            .hidden = try arch.u32Key(f, p ++ "embedding_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .ctx_len = try arch.u32Key(f, p ++ "context_length"),
            .n_heads = try arch.u32Key(f, p ++ "attention.head_count"),
            .n_kv_heads = try arch.u32Key(f, p ++ "attention.head_count_kv"),
            .head_dim = try arch.u32Key(f, p ++ "attention.key_length"),
            .rope_dims = try arch.u32Key(f, p ++ "rope.dimension_count"),
            .q_lora = try arch.u32Key(f, p ++ "attention.q_lora_rank"),
            .o_lora = try arch.u32Key(f, p ++ "attention.output_lora_rank"),
            .o_groups = try arch.u32Key(f, p ++ "attention.output_group_count"),
            .window = try arch.u32Key(f, p ++ "attention.sliding_window"),
            .rms_eps = try float(f, "attention.layer_norm_rms_epsilon"),
            .rope_theta = try float(f, "rope.freq_base"),
            .compress_rope_theta = try float(f, "attention.compress_rope_freq_base"),
            .yarn_factor = 0,
            .yarn_orig_ctx = 0,
            .yarn_beta_fast = 0,
            .yarn_beta_slow = 0,
            .n_experts = try arch.u32Key(f, p ++ "expert_count"),
            .n_experts_used = try arch.u32Key(f, p ++ "expert_used_count"),
            .n_shared = try arch.u32Key(f, p ++ "expert_shared_count"),
            .expert_ffn = try arch.u32Key(f, p ++ "expert_feed_forward_length"),
            .route_scale = try float(f, "expert_weights_scale"),
            .route_norm = f.getBool(p ++ "expert_weights_norm") orelse return error.MissingMetadata,
            .swiglu_limit = try swigluLimit(f, n_layers),
            .hash_layers = try arch.u32Key(f, p ++ "hash_layer_count"),
            .idx_heads = try arch.u32Key(f, p ++ "attention.indexer.head_count"),
            .idx_dim = try arch.u32Key(f, p ++ "attention.indexer.key_length"),
            .idx_topk = try arch.u32Key(f, p ++ "attention.indexer.top_k"),
            .hc = try arch.u32Key(f, p ++ "hyper_connection.count"),
            .hc_iters = try arch.u32Key(f, p ++ "hyper_connection.sinkhorn_iterations"),
            .hc_eps = try float(f, "hyper_connection.epsilon"),
            .compress = @splat(0),
            .kv_sources = try layerMask(f, keyOf(f, "attention.kv_source_layers", "kv_source_layer_ids"), n_layers),
            .index_sources = try layerMask(f, keyOf(f, "attention.indexer.source_layers", "index_source_layer_ids"), n_layers),
            .candidate_source = try arch.u32Key(f, keyOf(f, "attention.indexer.candidate_source_layer", "attention.candidate_source_layer_id")),
            .candidate_topk = try arch.u32Key(f, keyOf(f, "attention.indexer.candidate_top_k_blocks", "attention.candidate_topk_blocks")),
            .candidate_block = try arch.u32Key(f, keyOf(f, "attention.indexer.candidate_block_size", "attention.candidate_block_size")),
            .n_engram = 0,
            .engram_layers = @splat(0),
            .engram_rows = @splat(0),
            .engram_ngram = 0,
            .engram_heads = 0,
            .engram_dim = 0,
            .engram_vocab = 0,
            .engram_compressed = 0,
            .engram_pad = 0,
            .bos = try arch.u32Key(f, "tokenizer.ggml.bos_token_id"),
            .eos = try arch.u32Key(f, "tokenizer.ggml.eos_token_id"),
        };
        if (try arch.u32Key(f, p ++ "attention.value_length") != hp.head_dim) return error.BadMetadata;
        const ratios = f.getArray(p ++ "attention.compress_ratios") orelse return error.MissingMetadata;
        if (ratios.len < n_layers) return error.BadMetadata;
        for (hp.compress[0..n_layers], 0..) |*r, i| r.* = std.math.cast(u8, ratios.int(i)) orelse return error.BadMetadata;
        if (f.getString(p ++ "rope.scaling.type")) |ty| {
            if (!std.mem.eql(u8, ty, "yarn")) return error.BadMetadata;
            hp.yarn_factor = try float(f, "rope.scaling.factor");
            hp.yarn_orig_ctx = try arch.u32Key(f, p ++ "rope.scaling.original_context_length");
            hp.yarn_beta_fast = try float(f, "rope.scaling.yarn_beta_fast");
            hp.yarn_beta_slow = try float(f, "rope.scaling.yarn_beta_slow");
        }
        if (f.getBool(p ++ "engram.present") orelse false) try hp.readEngram(f) else if (f.getArray(p ++ "ple.layers") != null) try hp.readPle(f);
        return hp;
    }

    /// The re-split's Engram (`ple.*`): per layer, (ngram - 1) * heads prime buckets back to back in `ple_table`.
    /// The plugin derives the buckets as the primes from its vocab on, so the vocab is the first bucket (a gap in
    /// the file's primes fails its row check); the compressed vocab is the token map's.
    fn readPle(hp: *Hparams, f: *const gguf.File) !void {
        const layers = f.getArray(p ++ "ple.layers") orelse return error.MissingMetadata;
        const sizes = f.getArray(p ++ "ple.head_vocab_sizes") orelse return error.MissingMetadata;
        const offsets = f.getArray(p ++ "ple.head_offsets") orelse return error.MissingMetadata;
        const map = f.getArray(p ++ "engram.token_map") orelse return error.MissingMetadata;
        hp.engram_ngram = try arch.u32Key(f, p ++ "ple.ngram_size");
        hp.engram_heads = try arch.u32Key(f, p ++ "ple.heads_per_ngram");
        hp.engram_dim = try arch.u32Key(f, p ++ "embedding_length_per_layer_input");
        hp.engram_pad = try arch.u32Key(f, p ++ "engram.pad_token_id");
        if (layers.len == 0 or layers.len > MAX_ENGRAM or hp.engram_ngram < 2 or map.len != hp.vocab) return error.BadMetadata;
        const cols = (hp.engram_ngram - 1) * hp.engram_heads;
        if (sizes.len != layers.len * cols or offsets.len != sizes.len) return error.BadMetadata;
        var at: u64 = 0;
        for (0..layers.len) |l| {
            hp.engram_layers[l] = std.math.cast(u32, layers.int(l)) orelse return error.BadMetadata;
            if (hp.engram_layers[l] >= hp.n_layers) return error.BadMetadata;
            const first = at;
            for (l * cols..(l + 1) * cols) |i| {
                if (offsets.int(i) != at) return error.BadMetadata;
                at += std.math.cast(u64, sizes.int(i)) orelse return error.BadMetadata;
            }
            hp.engram_rows[l] = at - first;
        }
        hp.n_engram = @intCast(layers.len);
        hp.engram_vocab = std.math.cast(u32, sizes.int(0)) orelse return error.BadMetadata;
        var hi: i64 = 0;
        for (0..map.len) |i| hi = @max(hi, map.int(i));
        hp.engram_compressed = std.math.cast(u32, hi + 1) orelse return error.BadMetadata;
    }

    fn readEngram(hp: *Hparams, f: *const gguf.File) !void {
        const layers = f.getArray(p ++ "engram.layer_ids") orelse return error.MissingMetadata;
        const rows = f.getArray(p ++ "engram.num_embeddings") orelse return error.MissingMetadata;
        if (layers.len == 0 or layers.len > MAX_ENGRAM or rows.len != layers.len) return error.BadMetadata;
        if (f.getString(engram_file_key) == null) return error.MissingMetadata;
        for (0..layers.len) |i| {
            hp.engram_layers[i] = std.math.cast(u32, layers.int(i)) orelse return error.BadMetadata;
            hp.engram_rows[i] = std.math.cast(u64, rows.int(i)) orelse return error.BadMetadata;
            if (hp.engram_layers[i] >= hp.n_layers) return error.BadMetadata;
        }
        hp.n_engram = @intCast(layers.len);
        hp.engram_ngram = try arch.u32Key(f, p ++ "engram_max_ngram_size");
        hp.engram_heads = try arch.u32Key(f, p ++ "engram_n_heads");
        hp.engram_dim = try arch.u32Key(f, p ++ "engram_head_dim");
        hp.engram_vocab = try arch.u32Key(f, p ++ "engram_vocab_size");
        hp.engram_compressed = try arch.u32Key(f, p ++ "engram_compressed_vocab_size");
        hp.engram_pad = try arch.u32Key(f, p ++ "engram_pad_token_id");
    }
};

fn float(f: *const gguf.File, comptime key: []const u8) !f64 {
    return f.getFloat(p ++ key) orelse error.MissingMetadata;
}

/// The per-layer clamps of the routed and the shared experts, all one value.
fn swigluLimit(f: *const gguf.File, n_layers: u32) !f64 {
    var limit: ?f64 = null;
    for ([_][]const u8{ p ++ "swiglu_clamp_exp", p ++ "swiglu_clamp_shexp" }) |key| {
        const a = f.getArray(key) orelse return error.MissingMetadata;
        if (a.len != n_layers or a.elem_type != .f32) return error.BadMetadata;
        for (0..a.len) |i| {
            if (limit != null and limit.? != a.float(i)) return error.BadMetadata;
            limit = a.float(i);
        }
    }
    return limit.?;
}

/// The re-split's name for a key if the file has it, else the 4-shard sets' `old`.
fn keyOf(f: *const gguf.File, comptime key: []const u8, comptime old: []const u8) []const u8 {
    return if (f.kv.get(p ++ key) != null) p ++ key else p ++ old;
}

/// The first row of Engram table `index` in the re-split's `ple_table`.
pub fn pleRowStart(f: *const gguf.File, index: usize) !u64 {
    const offsets = f.getArray(p ++ "ple.head_offsets") orelse return error.MissingMetadata;
    const n = (f.getArray(p ++ "ple.layers") orelse return error.MissingMetadata).len;
    if (n == 0 or index >= n or offsets.len % n != 0) return error.BadMetadata;
    return std.math.cast(u64, offsets.int(index * (offsets.len / n))) orelse error.BadMetadata;
}

fn layerMask(f: *const gguf.File, key: []const u8, n_layers: u32) !u64 {
    const a = f.getArray(key) orelse return error.MissingMetadata;
    var mask: u64 = 0;
    for (0..a.len) |i| {
        const l = std.math.cast(u6, a.int(i)) orelse return error.BadMetadata;
        if (l >= n_layers) return error.BadMetadata;
        mask |= @as(u64, 1) << l;
    }
    return mask;
}

fn sep(list: []const u8) []const u8 {
    return if (list.len > 0) "," else "";
}

/// The config.json of our V4.1 MLX packs (`ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw`), text only: the
/// set ships no tower and no DSpark stages.
pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var ratios: std.ArrayList(u8) = .empty;
    defer ratios.deinit(allocator);
    var kv: std.ArrayList(u8) = .empty;
    defer kv.deinit(allocator);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    var elayers: std.ArrayList(u8) = .empty;
    defer elayers.deinit(allocator);
    var erows: std.ArrayList(u8) = .empty;
    defer erows.deinit(allocator);
    for (hp.compress[0..hp.n_layers], 0..) |r, l| {
        try ratios.print(allocator, "{s}{d}", .{ sep(ratios.items), r });
        if ((hp.kv_sources >> @intCast(l)) & 1 != 0) try kv.print(allocator, "{s}{d}", .{ sep(kv.items), l });
        if ((hp.index_sources >> @intCast(l)) & 1 != 0) try index.print(allocator, "{s}{d}", .{ sep(index.items), l });
    }
    for (hp.engram_layers[0..hp.n_engram], hp.engram_rows[0..hp.n_engram]) |l, n| {
        try elayers.print(allocator, "{s}{d}", .{ sep(elayers.items), l });
        try erows.print(allocator, "{s}{d}", .{ sep(erows.items), n });
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    // Several prints: one format call takes at most 32 arguments.
    try out.print(allocator,
        \\{{"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41","bos_token_id":{[bos]d},
        \\"eos_token_id":{[eos]d},"tie_word_embeddings":false,"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}},
        \\"text_config":{{"model_type":"deepseek_v41_text","vocab_size":{[vocab]d},"hidden_size":{[hidden]d},
        \\"num_hidden_layers":{[n_layers]d},"num_attention_heads":{[n_heads]d},"num_key_value_heads":{[n_kv]d},
        \\"head_dim":{[head_dim]d},"qk_rope_head_dim":{[rope_dims]d},"q_lora_rank":{[q_lora]d},"o_lora_rank":{[o_lora]d},
        \\"o_groups":{[o_groups]d},"rms_norm_eps":{[eps]e},"max_position_embeddings":{[ctx]d},"rope_theta":{[theta]d},
        \\"sliding_window":{[window]d},"compress_rope_theta":{[ctheta]d},"compress_ratios":[{[ratios]s}],
        \\"kv_source_layer_ids":[{[kv]s}],"index_source_layer_ids":[{[index]s}],
    , .{
        .bos = hp.bos,
        .eos = hp.eos,
        .vocab = hp.vocab,
        .hidden = hp.hidden,
        .n_layers = hp.n_layers,
        .n_heads = hp.n_heads,
        .n_kv = hp.n_kv_heads,
        .head_dim = hp.head_dim,
        .rope_dims = hp.rope_dims,
        .q_lora = hp.q_lora,
        .o_lora = hp.o_lora,
        .o_groups = hp.o_groups,
        .eps = hp.rms_eps,
        .ctx = hp.ctx_len,
        .theta = hp.rope_theta,
        .window = hp.window,
        .ctheta = hp.compress_rope_theta,
        .ratios = ratios.items,
        .kv = kv.items,
        .index = index.items,
    });
    if (hp.yarn_factor > 0) try out.print(allocator,
        \\"rope_scaling":{{"rope_type":"yarn","factor":{d},"beta_fast":{d},"beta_slow":{d},"original_max_position_embeddings":{d}}},
    , .{ hp.yarn_factor, hp.yarn_beta_fast, hp.yarn_beta_slow, hp.yarn_orig_ctx });
    if (hp.n_engram > 0) try out.print(allocator,
        \\"engram_layer_ids":[{[layers]s}],"engram_num_embeddings":[{[rows]s}],"engram_max_ngram_size":{[ngram]d},
        \\"engram_vocab_size":{[vocab]d},"engram_n_heads":{[heads]d},"engram_head_dim":{[dim]d},
        \\"engram_pad_token_id":{[pad]d},"engram_compressed_vocab_size":{[compressed]d},
    , .{
        .layers = elayers.items,
        .rows = erows.items,
        .ngram = hp.engram_ngram,
        .vocab = hp.engram_vocab,
        .heads = hp.engram_heads,
        .dim = hp.engram_dim,
        .pad = hp.engram_pad,
        .compressed = hp.engram_compressed,
    });
    try out.print(allocator,
        \\"n_routed_experts":{[experts]d},"n_shared_experts":{[shared]d},"num_experts_per_tok":{[used]d},
        \\"moe_intermediate_size":{[ffn]d},"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","norm_topk_prob":{[norm]},
        \\"routed_scaling_factor":{[scale]d},"swiglu_limit":{[swiglu]d},"num_hash_layers":{[hash]d},
        \\"index_n_heads":{[idx_heads]d},"index_head_dim":{[idx_dim]d},"index_topk":{[idx_topk]d},
        \\"candidate_source_layer_id":{[cand]d},"candidate_topk_blocks":{[cand_topk]d},"candidate_block_size":{[cand_block]d},
        \\"hc_mult":{[hc]d},"hc_sinkhorn_iters":{[hc_iters]d},"hc_eps":{[hc_eps]e}}}}}
    , .{
        .experts = hp.n_experts,
        .shared = hp.n_shared,
        .used = hp.n_experts_used,
        .ffn = hp.expert_ffn,
        .norm = hp.route_norm,
        .scale = hp.route_scale,
        .swiglu = hp.swiglu_limit,
        .hash = hp.hash_layers,
        .idx_heads = hp.idx_heads,
        .idx_dim = hp.idx_dim,
        .idx_topk = hp.idx_topk,
        .cand = hp.candidate_source,
        .cand_topk = hp.candidate_topk,
        .cand_block = hp.candidate_block,
        .hc = hp.hc,
        .hc_iters = hp.hc_iters,
        .hc_eps = hp.hc_eps,
    });
    return out.toOwnedSlice(allocator);
}

/// HF name + load transform for a GGUF tensor, null when the tensor is unknown.
pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "token_embd.weight", "embed.weight" },
        .{ "output.weight", "head.weight" },
        .{ "output_norm.weight", "norm.weight" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return .{ .name = g[1], .transform = .none };
    // The Engram hash's token map, primes, offsets and multipliers: the plugin derives the same ones.
    for ([_][]const u8{ "engram_token_map.weight", "engram_primes.weight", "engram_offsets.weight", "engram_multipliers.weight" }) |h|
        if (std.mem.eql(u8, name, h)) return arch.skip;

    if (std.mem.eql(u8, name, ple_table)) return arch.skip;

    const blk = arch.splitLayer(name, hp.n_layers) orelse return null;
    if (std.mem.eql(u8, blk.leaf, engram_leaf)) return arch.skip;
    const table = [_]struct { []const u8, []const u8 }{
        .{ "attn_norm.weight", "attn_norm.weight" },
        .{ "attn_sinks.weight", "attn.attn_sink" },
        .{ "attn_q_a.weight", "attn.wq_a.weight" },
        .{ "attn_q_a_norm.weight", "attn.q_norm.weight" },
        .{ "attn_q_b.weight", "attn.wq_b.weight" },
        .{ "attn_kv.weight", "attn.wkv.weight" },
        .{ "attn_kv_a_norm.weight", "attn.kv_norm.weight" },
        .{ "attn_output_a.weight", "attn.wo_a.weight" },
        .{ "attn_output_b.weight", "attn.wo_b.weight" },
        .{ "attn_compressor_kv.weight", "attn.compressor.wkv.weight" },
        .{ "attn_compressor_gate.weight", "attn.compressor.wgate.weight" },
        .{ "attn_compressor_norm.weight", "attn.compressor.norm.weight" },
        .{ "indexer.attn_q_b.weight", "attn.indexer.wq_b.weight" },
        .{ "indexer.attn_k.weight", "attn.indexer.wk.weight" },
        .{ "indexer.k_norm.weight", "attn.indexer.k_norm.weight" },
        .{ "indexer.proj.weight", "attn.indexer.weights_proj.weight" },
        .{ "hc_attn_fn.weight", "hc_attn_fn" },
        .{ "hc_attn_base.weight", "hc_attn_base" },
        .{ "hc_attn_scale.weight", "hc_attn_scale" },
        .{ "hc_ffn_fn.weight", "hc_ffn_fn" },
        .{ "hc_ffn_base.weight", "hc_ffn_base" },
        .{ "hc_ffn_scale.weight", "hc_ffn_scale" },
        .{ "ffn_norm.weight", "ffn_norm.weight" },
        .{ "ffn_gate_inp.weight", "ffn.gate.weight" },
        .{ "exp_probs_b.bias", "ffn.gate.bias" },
        .{ "exp_probs_b_vl.bias", "ffn.gate.bias_vl" },
        .{ "ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight" },
        .{ "ffn_up_shexp.weight", "ffn.shared_experts.w3.weight" },
        .{ "ffn_down_shexp.weight", "ffn.shared_experts.w2.weight" },
        .{ "ffn_gate_exps.weight", "ffn.experts.w1.weight" },
        .{ "ffn_up_exps.weight", "ffn.experts.w3.weight" },
        .{ "ffn_down_exps.weight", "ffn.experts.w2.weight" },
        .{ "engram_wkv.weight", "engram.wkv.weight" },
        .{ "engram_q.weight", "engram.q_weight" },
        .{ "engram_k.weight", "engram.k_weight" },
        // The re-split's names: `engram_gate` is `q_weight * k_weight` folded (f32), as the plugin's `qk`.
        .{ "engram_kv.weight", "engram.wkv.weight" },
        .{ "engram_gate.weight", "engram.qk" },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        const f32_leaf = std.mem.startsWith(u8, row[0], "hc_") or std.mem.startsWith(u8, row[0], "exp_probs_b") or
            std.mem.eql(u8, row[0], "attn_sinks.weight") or std.mem.eql(u8, row[0], "engram_gate.weight");
        return .{ .name = full, .transform = if (f32_leaf) .keep_f32 else .none };
    };
    return null;
}

/// The 2.00 bpw set's metadata (shard 1 of `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS`), as a File (tests only;
/// the plugin's own parser reads its `configJson` in `plugins/deepseek_v41/config.zig`).
pub fn testFile() !gguf.File {
    var f: gguf.File = .{ .arena = .init(std.testing.allocator), .version = 3 };
    errdefer f.deinit();
    const a = f.arena.allocator();
    const ints = [_]struct { []const u8, u64 }{
        .{ p ++ "block_count", 40 },                             .{ p ++ "context_length", 1048576 },
        .{ p ++ "embedding_length", 5120 },                      .{ p ++ "attention.head_count", 64 },
        .{ p ++ "attention.head_count_kv", 1 },                  .{ p ++ "attention.key_length", 512 },
        .{ p ++ "attention.value_length", 512 },                 .{ p ++ "rope.dimension_count", 64 },
        .{ p ++ "attention.q_lora_rank", 1280 },                 .{ p ++ "attention.output_lora_rank", 1024 },
        .{ p ++ "attention.output_group_count", 8 },             .{ p ++ "attention.sliding_window", 128 },
        .{ p ++ "rope.scaling.original_context_length", 65536 }, .{ p ++ "expert_count", 384 },
        .{ p ++ "expert_used_count", 6 },                        .{ p ++ "expert_shared_count", 1 },
        .{ p ++ "expert_feed_forward_length", 2304 },            .{ p ++ "hash_layer_count", 0 },
        .{ p ++ "attention.indexer.head_count", 32 },            .{ p ++ "attention.indexer.key_length", 128 },
        .{ p ++ "attention.indexer.top_k", 512 },                .{ p ++ "hyper_connection.count", 4 },
        .{ p ++ "hyper_connection.sinkhorn_iterations", 20 },    .{ p ++ "attention.candidate_source_layer_id", 20 },
        .{ p ++ "attention.candidate_topk_blocks", 2048 },       .{ p ++ "attention.candidate_block_size", 8 },
        .{ p ++ "engram_max_ngram_size", 4 },                    .{ p ++ "engram_n_heads", 8 },
        .{ p ++ "engram_head_dim", 256 },                        .{ p ++ "engram_vocab_size", 16000000 },
        .{ p ++ "engram_compressed_vocab_size", 99092 },         .{ p ++ "engram_pad_token_id", 2 },
        .{ "tokenizer.ggml.bos_token_id", 0 },                   .{ "tokenizer.ggml.eos_token_id", 1 },
    };
    for (ints) |kv| try f.kv.put(a, kv[0], .{ .uint = kv[1] });
    const floats = [_]struct { []const u8, f32 }{
        .{ p ++ "attention.layer_norm_rms_epsilon", 1e-20 },   .{ p ++ "rope.freq_base", 10000 },
        .{ p ++ "attention.compress_rope_freq_base", 160000 }, .{ p ++ "rope.scaling.factor", 16 },
        .{ p ++ "rope.scaling.yarn_beta_fast", 32 },           .{ p ++ "rope.scaling.yarn_beta_slow", 1 },
        .{ p ++ "expert_weights_scale", 1.5 },                 .{ p ++ "hyper_connection.epsilon", 1e-6 },
    };
    for (floats) |kv| try f.kv.put(a, kv[0], .{ .float = kv[1] });
    try f.kv.put(a, p ++ "expert_weights_norm", .{ .bool = true });
    try f.kv.put(a, p ++ "engram.present", .{ .bool = true });
    try f.kv.put(a, p ++ "expert_scoring_func", .{ .string = "sqrtsoftplus" });
    try f.kv.put(a, p ++ "rope.scaling.type", .{ .string = "yarn" });
    try f.kv.put(a, engram_file_key, .{ .string = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" });
    try f.kv.put(a, "tokenizer.ggml.tokens", .{ .array = .{ .elem_type = .string, .len = 129280, .bytes = &.{} } });
    const clamp = try a.alloc(f32, 40);
    @memset(clamp, 10);
    for ([_][]const u8{ p ++ "swiglu_clamp_exp", p ++ "swiglu_clamp_shexp" }) |key|
        try f.kv.put(a, key, .{ .array = .{ .elem_type = .f32, .len = 40, .bytes = std.mem.sliceAsBytes(clamp) } });
    const ratios = try a.alloc(i32, 43);
    for (ratios, 0..) |*r, i| r.* = if (i < 2 or i >= 40) 0 else if (i < 20) 2 else 1;
    const lists = [_]struct { []const u8, []const i32 }{
        .{ p ++ "attention.compress_ratios", ratios },
        .{ p ++ "kv_source_layer_ids", &.{ 2, 8, 14, 20 } },
        .{ p ++ "index_source_layer_ids", &.{ 2, 8, 14, 20, 24, 28, 32, 36 } },
        .{ p ++ "engram.layer_ids", &.{ 1, 14 } },
        .{ p ++ "engram.num_embeddings", &.{ 384006168, 384016682 } },
    };
    for (lists) |l| try f.kv.put(a, l[0], .{ .array = .{ .elem_type = .i32, .len = l[1].len, .bytes = std.mem.sliceAsBytes(l[1]) } });
    return f;
}

/// The Coder's 6-shard re-split (shard 1 of `DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS`, 192 experts kept):
/// `testFile`'s geometry under llama.cpp's names, the Engram as its `ple.*` buckets (tests only).
pub fn testFileResplit() !gguf.File {
    var f = try testFile();
    errdefer f.deinit();
    const a = f.arena.allocator();
    inline for (.{ "engram.present", "engram.layer_ids", "engram.num_embeddings", "engram_max_ngram_size", "engram_n_heads", "engram_head_dim", "engram_vocab_size", "engram_compressed_vocab_size", "engram_pad_token_id", "engram.file", "expert_scoring_func", "kv_source_layer_ids", "index_source_layer_ids", "attention.candidate_source_layer_id", "attention.candidate_topk_blocks", "attention.candidate_block_size" }) |k|
        _ = f.kv.remove(p ++ k);
    const ints = [_]struct { []const u8, u64 }{
        .{ p ++ "expert_count", 192 },                               .{ p ++ "expert_gating_func", sqrt_softplus_gating },
        .{ p ++ "attention.indexer.candidate_source_layer", 20 },    .{ p ++ "attention.indexer.candidate_top_k_blocks", 2048 },
        .{ p ++ "attention.indexer.candidate_block_size", 8 },       .{ p ++ "ple.ngram_size", 4 },
        .{ p ++ "ple.heads_per_ngram", 8 },                          .{ p ++ "embedding_length_per_layer_input", 256 },
        .{ p ++ "engram.pad_token_id", 2 },
    };
    for (ints) |kv| try f.kv.put(a, kv[0], .{ .uint = kv[1] });
    // The 48 consecutive primes from 16000057: 24 buckets a layer.
    const sizes = [_]u64{
        16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183, 16000189, 16000207, 16000211, 16000253,
        16000277, 16000289, 16000307, 16000321, 16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463,
        16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627, 16000667, 16000669, 16000693, 16000697,
        16000711, 16000729, 16000759, 16000769, 16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
    };
    const offsets = try a.alloc(u64, sizes.len);
    var at: u64 = 0;
    for (offsets, sizes) |*o, n| {
        o.* = at;
        at += n;
    }
    const token_map = try a.alloc(i32, 129280);
    for (token_map, 0..) |*t, i| t.* = @intCast(i % 99092);
    try f.kv.put(a, p ++ "ple.head_vocab_sizes", .{ .array = .{ .elem_type = .u64, .len = sizes.len, .bytes = std.mem.sliceAsBytes(try a.dupe(u64, &sizes)) } });
    try f.kv.put(a, p ++ "ple.head_offsets", .{ .array = .{ .elem_type = .u64, .len = offsets.len, .bytes = std.mem.sliceAsBytes(offsets) } });
    const lists = [_]struct { []const u8, []const i32 }{
        .{ p ++ "attention.kv_source_layers", &.{ 2, 8, 14, 20 } },
        .{ p ++ "attention.indexer.source_layers", &.{ 2, 8, 14, 20, 24, 28, 32, 36 } },
        .{ p ++ "ple.layers", &.{ 1, 14 } },
        .{ p ++ "engram.token_map", token_map },
    };
    for (lists) |l| try f.kv.put(a, l[0], .{ .array = .{ .elem_type = .i32, .len = l[1].len, .bytes = std.mem.sliceAsBytes(l[1]) } });
    return f;
}

test "Hparams.read: the 6-shard re-split's names give the 4-shard set's hyperparameters, its Engram from ple.*" {
    var old = try testFile();
    defer old.deinit();
    var new = try testFileResplit();
    defer new.deinit();
    var want = try Hparams.read(&old);
    const got = try Hparams.read(&new);
    // The plugin derives the buckets as the primes from its vocab on: from 16000000 or from the first, the same ones.
    try std.testing.expectEqual(@as(u32, 16000057), got.engram_vocab);
    want.engram_vocab = got.engram_vocab;
    want.n_experts = 192;
    try std.testing.expectEqualDeep(want, got);
    try std.testing.expectEqual(@as(u64, 0), try pleRowStart(&new, 0));
    try std.testing.expectEqual(@as(u64, 384006168), try pleRowStart(&new, 1));

    // Buckets that do not run back to back, and another gating, are refused.
    const a = new.arena.allocator();
    const offsets = new.getArray(p ++ "ple.head_offsets").?;
    const bad = try a.alloc(u64, offsets.len);
    for (bad, 0..) |*o, i| o.* = @intCast(offsets.int(i));
    bad[30] += 1;
    try new.kv.put(a, p ++ "ple.head_offsets", .{ .array = .{ .elem_type = .u64, .len = bad.len, .bytes = std.mem.sliceAsBytes(bad) } });
    try std.testing.expectError(error.BadMetadata, Hparams.read(&new));
    var other = try testFileResplit();
    defer other.deinit();
    try other.kv.put(other.arena.allocator(), p ++ "expert_gating_func", .{ .uint = 2 });
    try std.testing.expectError(error.BadMetadata, Hparams.read(&other));
}

test "Hparams.read: the release's geometry, the trunk's ratios only, Engram and YaRN" {
    var f = try testFile();
    defer f.deinit();
    const hp = try Hparams.read(&f);
    try std.testing.expectEqual(@as(u32, 40), hp.n_layers);
    try std.testing.expectEqual(@as(u8, 2), hp.compress[19]);
    try std.testing.expectEqual(@as(u8, 1), hp.compress[39]);
    try std.testing.expectEqual(@as(u64, (1 << 2) | (1 << 8) | (1 << 14) | (1 << 20)), hp.kv_sources);
    try std.testing.expectEqual(@as(u32, 2), hp.n_engram);
    try std.testing.expectEqual(@as(u64, 384016682), hp.engram_rows[1]);
    try std.testing.expectEqual(@as(f64, 10), hp.swiglu_limit);
    try std.testing.expectEqual(@as(f64, 16), hp.yarn_factor);

    // A shared-expert clamp apart from the routed one has no field in the plugin's config.
    const a = f.arena.allocator();
    const clamp = try a.alloc(f32, 40);
    @memset(clamp, 10);
    clamp[7] = 8;
    try f.kv.put(a, p ++ "swiglu_clamp_shexp", .{ .array = .{ .elem_type = .f32, .len = 40, .bytes = std.mem.sliceAsBytes(clamp) } });
    try std.testing.expectError(error.BadMetadata, Hparams.read(&f));
}

test "Hparams.read refuses another routing and a source past the trunk" {
    var f = try testFile();
    defer f.deinit();
    const a = f.arena.allocator();
    try f.kv.put(a, p ++ "index_source_layer_ids", .{ .array = .{ .elem_type = .i32, .len = 1, .bytes = std.mem.sliceAsBytes(&[_]i32{40}) } });
    try std.testing.expectError(error.BadMetadata, Hparams.read(&f));
    var g = try testFile();
    defer g.deinit();
    try g.kv.put(g.arena.allocator(), p ++ "expert_scoring_func", .{ .string = "softmax" });
    try std.testing.expectError(error.BadMetadata, Hparams.read(&g));
}

test "mapTensor: every family of the set onto the plugin's names; the Engram's hash and tables skipped" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 40;
    var buf: [160]u8 = undefined;
    const want = [_]struct { []const u8, []const u8 }{
        .{ "token_embd.weight", "embed.weight" },
        .{ "output.weight", "head.weight" },
        .{ "output_norm.weight", "norm.weight" },
        .{ "blk.0.attn_sinks.weight", "layers.0.attn.attn_sink" },
        .{ "blk.0.attn_q_a.weight", "layers.0.attn.wq_a.weight" },
        .{ "blk.0.attn_q_a_norm.weight", "layers.0.attn.q_norm.weight" },
        .{ "blk.0.attn_kv_a_norm.weight", "layers.0.attn.kv_norm.weight" },
        .{ "blk.39.attn_output_a.weight", "layers.39.attn.wo_a.weight" },
        .{ "blk.39.attn_output_b.weight", "layers.39.attn.wo_b.weight" },
        .{ "blk.2.attn_compressor_gate.weight", "layers.2.attn.compressor.wgate.weight" },
        .{ "blk.20.attn_compressor_norm.weight", "layers.20.attn.compressor.norm.weight" },
        .{ "blk.36.indexer.attn_q_b.weight", "layers.36.attn.indexer.wq_b.weight" },
        .{ "blk.36.indexer.proj.weight", "layers.36.attn.indexer.weights_proj.weight" },
        .{ "blk.20.indexer.attn_k.weight", "layers.20.attn.indexer.wk.weight" },
        .{ "blk.20.indexer.k_norm.weight", "layers.20.attn.indexer.k_norm.weight" },
        .{ "blk.5.hc_attn_fn.weight", "layers.5.hc_attn_fn" },
        .{ "blk.5.hc_ffn_scale.weight", "layers.5.hc_ffn_scale" },
        .{ "blk.5.ffn_gate_inp.weight", "layers.5.ffn.gate.weight" },
        .{ "blk.5.exp_probs_b.bias", "layers.5.ffn.gate.bias" },
        .{ "blk.5.ffn_up_shexp.weight", "layers.5.ffn.shared_experts.w3.weight" },
        .{ "blk.5.ffn_gate_exps.weight", "layers.5.ffn.experts.w1.weight" },
        .{ "blk.5.ffn_down_exps.weight", "layers.5.ffn.experts.w2.weight" },
        .{ "blk.14.engram_wkv.weight", "layers.14.engram.wkv.weight" },
        .{ "blk.14.engram_q.weight", "layers.14.engram.q_weight" },
        .{ "blk.1.engram_k.weight", "layers.1.engram.k_weight" },
        .{ "blk.14.engram_kv.weight", "layers.14.engram.wkv.weight" },
        .{ "blk.1.engram_gate.weight", "layers.1.engram.qk" },
        .{ "blk.5.exp_probs_b_vl.bias", "layers.5.ffn.gate.bias_vl" },
        .{ "per_layer_token_embd.weight", "" },
        .{ "engram_token_map.weight", "" },
        .{ "engram_multipliers.weight", "" },
        .{ "blk.14.engram_embd.weight", "" },
    };
    for (want) |w| {
        const m = mapTensor(&buf, w[0], hp) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(w[1], m.name);
        const f32_leaf = std.mem.indexOf(u8, w[0], ".hc_") != null or std.mem.endsWith(u8, w[0], "attn_sinks.weight") or
            std.mem.indexOf(u8, w[0], "exp_probs_b") != null or std.mem.endsWith(u8, w[0], "engram_gate.weight");
        try std.testing.expect(m.transform == if (f32_leaf) arch.Transform.keep_f32 else arch.Transform.none);
    }
    // DSpark stages past the trunk and another converter's leaves are unknown.
    for ([_][]const u8{ "blk.40.attn_q_a.weight", "blk.0.attn_q.weight", "rope_freqs.weight" }) |n|
        try std.testing.expect(mapTensor(&buf, n, hp) == null);
}

