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
/// The DS4 layout's Engram rows: 256 e4m3 codes then 8 e8m0 scales, one per 32 (the release's own fp8 table).
pub const engram_encoding = "e4m3_e8m0_32_row264";
/// ggml's I8, a type the loader does not serve: the DS4 layout stores each Engram table as I8 [row bytes, rows].
pub const engram_row_type: @import("../quants.zig").GgmlType = @enumFromInt(24);
/// The gating id the 4-shard sets pair with `expert_scoring_func` sqrtsoftplus; the re-split keeps only the id.
const sqrt_softplus_gating = 4;

const MAX_LAYERS = 64;
const MAX_ENGRAM = 4;
const MAX_STAGES = 8;

/// The DSpark stage bundle beside a set (Unsloth's `dspark-*.gguf`) names its arch `dflash`.
pub const dspark_arch = "dflash";
const dp = dspark_arch ++ ".";
/// The DS4 converter's bundle (`*-DSpark-support-*.gguf`): `dspark.*` keys, its tensors under `mtp.<stage>.`.
pub const ds4_dspark_arch = "deepseek41-dspark";
const ds4p = "dspark.";

pub fn isDsparkArch(name: []const u8) bool {
    return std.mem.eql(u8, name, dspark_arch) or std.mem.eql(u8, name, ds4_dspark_arch);
}

/// A stage bundle's geometry: the `dspark_*` keys of our packs' config and its stages' compress ratios.
pub const Dspark = struct {
    stages: u32,
    block: u32,
    targets: [MAX_STAGES]u8,
    n_targets: u32,
    noise: u32,
    markov_rank: u32,
    experts: u32,
    top_k: u32,
    compress: [MAX_STAGES]u8,

    /// The bundle's keys; refused unless its stages run the shapes of the trunk `hp` they draft for.
    pub fn read(f: *const gguf.File, hp: Hparams) !Dspark {
        const name = f.getString("general.architecture") orelse "";
        if (std.mem.eql(u8, name, ds4_dspark_arch)) return readDs4(f, hp);
        if (!std.mem.eql(u8, name, dspark_arch)) return error.UnsupportedArch;
        const stages = try arch.u32Key(f, dp ++ "block_count");
        if (stages == 0 or stages > MAX_STAGES or hp.n_layers + stages > MAX_LAYERS) return error.BadMetadata;
        const same = [_]struct { []const u8, u32 }{
            .{ dp ++ "embedding_length", hp.hidden },             .{ dp ++ "attention.head_count", hp.n_heads },
            .{ dp ++ "attention.key_length", hp.head_dim },       .{ dp ++ "attention.q_lora_rank", hp.q_lora },
            .{ dp ++ "attention.output_lora_rank", hp.o_lora },   .{ dp ++ "attention.output_group_count", hp.o_groups },
            .{ dp ++ "expert_feed_forward_length", hp.expert_ffn },
        };
        for (same) |kv| if (try arch.u32Key(f, kv[0]) != kv[1]) return error.BadMetadata;
        const markov = f.tensors.get("markov_w1.weight") orelse return error.MissingTensor;
        var d: Dspark = .{
            .stages = stages,
            .block = try arch.u32Key(f, dp ++ "block_size"),
            .targets = @splat(0),
            .n_targets = 0,
            .noise = try arch.u32Key(f, "tokenizer.ggml.mask_token_id"),
            .markov_rank = std.math.cast(u32, markov.dims[0]) orelse return error.BadMetadata,
            .experts = try arch.u32Key(f, dp ++ "expert_count"),
            .top_k = try arch.u32Key(f, dp ++ "expert_used_count"),
            .compress = @splat(0),
        };
        try d.readTargets(f.getArray(dp ++ "target_layers") orelse return error.MissingMetadata, hp);
        const ratios = f.getArray(dp ++ "attention.compress_ratios") orelse return error.MissingMetadata;
        if (ratios.len != stages) return error.BadMetadata;
        for (d.compress[0..stages], 0..) |*r, i| r.* = std.math.cast(u8, ratios.int(i)) orelse return error.BadMetadata;
        return d;
    }

    /// The DS4 converter's bundle: its stages counted and checked against the trunk by their tensors; they ship no
    /// compressor (the release's ratios end in zeros for them).
    fn readDs4(f: *const gguf.File, hp: Hparams) !Dspark {
        var d: Dspark = .{
            .stages = 0,
            .block = try arch.u32Key(f, ds4p ++ "block_size"),
            .targets = @splat(0),
            .n_targets = 0,
            .noise = try arch.u32Key(f, ds4p ++ "noise_token_id"),
            .markov_rank = try arch.u32Key(f, ds4p ++ "markov_rank"),
            .experts = try arch.u32Key(f, ds4p ++ "n_routed_experts"),
            .top_k = try arch.u32Key(f, ds4p ++ "num_experts_per_tok"),
            .compress = @splat(0),
        };
        var nb: [64]u8 = undefined;
        while (f.tensors.get(std.fmt.bufPrint(&nb, "mtp.{d}.attn_q_b.weight", .{d.stages}) catch unreachable)) |q| : (d.stages += 1) {
            const e = f.tensors.get(std.fmt.bufPrint(&nb, "mtp.{d}.ffn_gate_exps.weight", .{d.stages}) catch unreachable) orelse return error.MissingTensor;
            if (d.stages == MAX_STAGES or q.dims[0] != hp.q_lora or q.dims[1] != hp.n_heads * hp.head_dim or
                e.dims[0] != hp.hidden or e.dims[1] != hp.expert_ffn or e.dims[2] != d.experts) return error.BadMetadata;
        }
        if (d.stages == 0 or hp.n_layers + d.stages > MAX_LAYERS) return error.BadMetadata;
        try d.readTargets(f.getArray(ds4p ++ "target_layer_ids") orelse return error.MissingMetadata, hp);
        return d;
    }

    fn readTargets(d: *Dspark, targets: gguf.Array, hp: Hparams) !void {
        if (targets.len == 0 or targets.len > MAX_STAGES or d.block == 0) return error.BadMetadata;
        for (0..targets.len) |i| {
            d.targets[i] = std.math.cast(u8, targets.int(i)) orelse return error.BadMetadata;
            if (d.targets[i] >= hp.n_layers) return error.BadMetadata;
        }
        d.n_targets = @intCast(targets.len);
    }
};

/// The DS4 converter's vision tower (`*-Vision.gguf`, ds4's `--vision`): the release's tower under its own names.
pub const vision_arch = "deepseek4-vision";
const vp = vision_arch ++ ".";
/// llama.cpp's mmproj of the same tower (`general.architecture` `clip`): the projector it names, and its key prefix.
pub const clip_projector = "deepseek41v";
const cvp = "clip.vision.";

/// A tower GGUF's geometry, the `vision_config` of our packs; refused unless it was built for the trunk it sits beside.
pub const Vision = struct {
    layers: u32,
    hidden: u32,
    ffn: u32,
    heads: u32,
    patch: u32,
    downsample: u32,
    max_tokens: u32,
    min_pixels: u32,
    /// 0 = no cap (the release's null).
    max_wh_ratio: u32,
    rope_theta: f64,
    image_token: u32,

    /// `revision` is the trunk's `general.source.revision`: ds4 pairs a tower with its language GGUF by it, and the
    /// September tower has the 1017 trunk's shapes.
    pub fn read(f: *const gguf.File, hp: Hparams, revision: ?[]const u8) !Vision {
        if (std.mem.eql(u8, f.getString("general.architecture") orelse "", "clip")) return readClip(f, hp);
        if (!std.mem.eql(u8, f.getString("general.architecture") orelse "", vision_arch)) return error.UnsupportedArch;
        if (!std.mem.eql(u8, f.getString(vp ++ "checkpoint_variant") orelse "", "v4.1-flash")) return error.BadMetadata;
        if (revision) |r| if (f.getString("general.source.revision")) |own| if (!std.mem.eql(u8, r, own)) return error.OtherCheckpoint;
        if (try arch.u32Key(f, vp ++ "projection_length") != hp.hidden or
            try arch.u32Key(f, vp ++ "language.block_count") != hp.n_layers or
            try arch.u32Key(f, vp ++ "language.expert_count") != hp.n_experts) return error.BadMetadata;
        try knownTensors(f);
        return .{
            .layers = try arch.u32Key(f, vp ++ "block_count"),
            .hidden = try arch.u32Key(f, vp ++ "embedding_length"),
            .ffn = try arch.u32Key(f, vp ++ "feed_forward_length"),
            .heads = try arch.u32Key(f, vp ++ "attention.head_count"),
            .patch = try arch.u32Key(f, vp ++ "patch_size"),
            .downsample = try arch.u32Key(f, vp ++ "downsample_ratio"),
            .max_tokens = try arch.u32Key(f, vp ++ "image.max_tokens"),
            .min_pixels = try arch.u32Key(f, vp ++ "image.min_pixels"),
            .max_wh_ratio = try arch.u32Key(f, vp ++ "image.max_width_height_ratio"),
            .rope_theta = f.getFloat(vp ++ "rope.freq_base") orelse return error.MissingMetadata,
            .image_token = try arch.u32Key(f, vp ++ "image.token_id"),
        };
    }

    /// The mmproj names no checkpoint, token budget, aspect cap, rope base or image token: those are the release's.
    fn readClip(f: *const gguf.File, hp: Hparams) !Vision {
        if (!std.mem.eql(u8, f.getString("clip.projector_type") orelse "", clip_projector)) return error.UnsupportedArch;
        if (try arch.u32Key(f, cvp ++ "projection_dim") != hp.hidden) return error.BadMetadata;
        try knownTensors(f);
        return .{
            .layers = try arch.u32Key(f, cvp ++ "block_count"),
            .hidden = try arch.u32Key(f, cvp ++ "embedding_length"),
            .ffn = try arch.u32Key(f, cvp ++ "feed_forward_length"),
            .heads = try arch.u32Key(f, cvp ++ "attention.head_count"),
            .patch = try arch.u32Key(f, cvp ++ "patch_size"),
            .downsample = try arch.u32Key(f, cvp ++ "projector.scale_factor"),
            .max_tokens = 1024,
            .min_pixels = try arch.u32Key(f, cvp ++ "image_min_pixels"),
            .max_wh_ratio = 0,
            .rope_theta = 10000,
            // `<｜deepseek_image｜>` in every V4.1 trunk.
            .image_token = 129264,
        };
    }

    /// Every tensor one the tower loads: anything else would fail the whole load after vision was advertised.
    fn knownTensors(f: *const gguf.File) !void {
        var buf: [96]u8 = undefined;
        for (f.tensors.keys(), f.tensors.values()) |name, t| switch (t.ty) {
            .f32, .f16, .bf16 => if (mapVisionTensor(&buf, name) == null) return error.UnsupportedTensor,
            else => return error.UnsupportedTensor,
        };
    }
};

/// A tower GGUF's tensor under the name `plugins/deepseek_v41/vision.zig` reads (the release's own), null when unknown.
/// An mmproj's gate and up are `mlp.w1_gate` / `mlp.w1_up`, which `weights.loadVision` joins into `mlp.w1`.
pub fn mapVisionTensor(buf: []u8, name: []const u8) ?arch.Mapped {
    const ours = std.mem.startsWith(u8, name, "vision.") or std.mem.startsWith(u8, name, "aligner.") or
        std.mem.eql(u8, name, "image_start") or std.mem.eql(u8, name, "image_end") or std.mem.eql(u8, name, "image_newline");
    if (ours) return .{ .name = name, .transform = .none };
    const whole = [_][2][]const u8{
        .{ "v.post_ln.weight", "vision.norm.weight" },             .{ "v.patch_embd.weight", "vision.patch_embed.proj.weight" },
        .{ "v.patch_embd.bias", "vision.patch_embed.proj.bias" },   .{ "v.token_embd.img_start", "image_start" },
        .{ "v.token_embd.img_end", "image_end" },                   .{ "v.image_newline", "image_newline" },
        .{ "mm.1.weight", "aligner.w1.weight" },                    .{ "mm.1.bias", "aligner.w1.bias" },
        .{ "mm.2.weight", "aligner.w2.weight" },                    .{ "mm.2.bias", "aligner.w2.bias" },
    };
    for (whole) |w| if (std.mem.eql(u8, name, w[0])) return .{ .name = w[1], .transform = .none };
    if (!std.mem.startsWith(u8, name, "v.blk.")) return null;
    const rest = name["v.blk.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const leaf = rest[dot + 1 ..];
    const block_leaves = [_][2][]const u8{
        .{ "attn_qkv.", "attn.wqkv." }, .{ "attn_out.", "attn.wo." }, .{ "ffn_down.", "mlp.w2." }, .{ "ln1.", "norm1." },
        .{ "ln2.", "norm2." },          .{ "ffn_gate.", "mlp.w1_gate." }, .{ "ffn_up.", "mlp.w1_up." },
    };
    for (block_leaves) |l| if (std.mem.startsWith(u8, leaf, l[0])) {
        const n = std.fmt.bufPrint(buf, "vision.blocks.{s}.{s}{s}", .{ rest[0..dot], l[1], leaf[l[0].len..] }) catch return null;
        return .{ .name = n, .transform = .none };
    };
    return null;
}

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
        if (f.getString(p ++ "config")) |json| return readEmbedded(f, json);
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

    /// The DS4 layout: HF's keys, the release's config.json whole in `deepseek41.config`, read from its `text_config`;
    /// the Engram tables sit in the file in `engram_encoding`.
    fn readEmbedded(f: *const gguf.File, json: []const u8) !Hparams {
        if (f.getString(p ++ "engram.encoding")) |enc| if (!std.mem.eql(u8, enc, engram_encoding)) return error.BadMetadata;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const root = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.BadMetadata;
        if (root != .object) return error.BadMetadata;
        const c = std.json.parseFromValueLeaky(Embedded, a, root.object.get("text_config") orelse root, .{ .ignore_unknown_fields = true }) catch return error.BadMetadata;
        const n_layers = c.num_hidden_layers;
        if (n_layers == 0 or n_layers > MAX_LAYERS or c.compress_ratios.len < n_layers) return error.BadMetadata;
        if (!std.mem.eql(u8, c.scoring_func, "sqrtsoftplus")) return error.BadMetadata;
        var hp = Hparams{
            .n_layers = n_layers,
            .hidden = c.hidden_size,
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .ctx_len = c.max_position_embeddings,
            .n_heads = c.num_attention_heads,
            .n_kv_heads = c.num_key_value_heads,
            .head_dim = c.head_dim,
            .rope_dims = c.qk_rope_head_dim,
            .q_lora = c.q_lora_rank,
            .o_lora = c.o_lora_rank,
            .o_groups = c.o_groups,
            .window = c.sliding_window,
            .rms_eps = c.rms_norm_eps,
            .rope_theta = c.rope_theta,
            .compress_rope_theta = c.compress_rope_theta,
            .yarn_factor = 0,
            .yarn_orig_ctx = 0,
            .yarn_beta_fast = 0,
            .yarn_beta_slow = 0,
            .n_experts = c.n_routed_experts,
            .n_experts_used = c.num_experts_per_tok,
            .n_shared = c.n_shared_experts,
            .expert_ffn = c.moe_intermediate_size,
            .route_scale = c.routed_scaling_factor,
            .route_norm = c.norm_topk_prob,
            .swiglu_limit = c.swiglu_limit,
            .hash_layers = c.num_hash_layers,
            .idx_heads = c.index_n_heads,
            .idx_dim = c.index_head_dim,
            .idx_topk = c.index_topk,
            .hc = c.hc_mult,
            .hc_iters = c.hc_sinkhorn_iters,
            .hc_eps = c.hc_eps,
            .compress = @splat(0),
            .kv_sources = 0,
            .index_sources = 0,
            .candidate_source = c.candidate_source_layer_id,
            .candidate_topk = c.candidate_topk_blocks,
            .candidate_block = c.candidate_block_size,
            .n_engram = @intCast(c.engram_layer_ids.len),
            .engram_layers = @splat(0),
            .engram_rows = @splat(0),
            .engram_ngram = c.engram_max_ngram_size,
            .engram_heads = c.engram_n_heads,
            .engram_dim = c.engram_head_dim,
            .engram_vocab = c.engram_vocab_size,
            .engram_compressed = c.engram_compressed_vocab_size,
            .engram_pad = c.engram_pad_token_id,
            .bos = try arch.u32Key(f, "tokenizer.ggml.bos_token_id"),
            .eos = try arch.u32Key(f, "tokenizer.ggml.eos_token_id"),
        };
        for (hp.compress[0..n_layers], c.compress_ratios[0..n_layers]) |*r, v| r.* = std.math.cast(u8, v) orelse return error.BadMetadata;
        for (c.kv_source_layer_ids) |l| hp.kv_sources |= try layerBit(l, n_layers);
        for (c.index_source_layer_ids) |l| hp.index_sources |= try layerBit(l, n_layers);
        if (c.rope_scaling) |y| {
            if (!std.mem.eql(u8, y.rope_type, "yarn")) return error.BadMetadata;
            hp.yarn_factor = y.factor;
            hp.yarn_orig_ctx = y.original_max_position_embeddings;
            hp.yarn_beta_fast = y.beta_fast;
            hp.yarn_beta_slow = y.beta_slow;
        }
        if (hp.n_engram > MAX_ENGRAM or c.engram_num_embeddings.len != hp.n_engram) return error.BadMetadata;
        for (c.engram_layer_ids, c.engram_num_embeddings, 0..) |l, rows, i| {
            if (l >= n_layers) return error.BadMetadata;
            hp.engram_layers[i] = l;
            hp.engram_rows[i] = rows;
        }
        return hp;
    }
};

/// The `text_config` fields of the release's config.json that `Hparams.readEmbedded` reads.
const Embedded = struct {
    hidden_size: u32,
    num_hidden_layers: u32,
    max_position_embeddings: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    qk_rope_head_dim: u32,
    q_lora_rank: u32,
    o_lora_rank: u32,
    o_groups: u32,
    sliding_window: u32,
    rms_norm_eps: f64,
    rope_theta: f64,
    compress_rope_theta: f64,
    rope_scaling: ?struct { rope_type: []const u8, factor: f64, beta_fast: f64, beta_slow: f64, original_max_position_embeddings: u32 } = null,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    n_shared_experts: u32,
    moe_intermediate_size: u32,
    routed_scaling_factor: f64,
    norm_topk_prob: bool,
    scoring_func: []const u8,
    swiglu_limit: f64,
    num_hash_layers: u32 = 0,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    hc_mult: u32,
    hc_sinkhorn_iters: u32,
    hc_eps: f64,
    compress_ratios: []const u32,
    kv_source_layer_ids: []const u32,
    index_source_layer_ids: []const u32,
    candidate_source_layer_id: u32,
    candidate_topk_blocks: u32,
    candidate_block_size: u32,
    engram_layer_ids: []const u32 = &.{},
    engram_num_embeddings: []const u64 = &.{},
    engram_max_ngram_size: u32 = 0,
    engram_n_heads: u32 = 0,
    engram_head_dim: u32 = 0,
    engram_vocab_size: u32 = 0,
    engram_compressed_vocab_size: u32 = 0,
    engram_pad_token_id: u32 = 0,
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
    for (0..a.len) |i| mask |= try layerBit(a.int(i), n_layers);
    return mask;
}

fn layerBit(layer: i64, n_layers: u32) !u64 {
    const l = std.math.cast(u6, layer) orelse return error.BadMetadata;
    if (l >= n_layers) return error.BadMetadata;
    return @as(u64, 1) << l;
}

fn sep(list: []const u8) []const u8 {
    return if (list.len > 0) "," else "";
}

/// The config.json of our V4.1 MLX packs (`ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw`), text only (the
/// set ships no tower), with the DSpark keys and the stages' ratios when a stage bundle `ds` sits beside it.
pub fn configJson(allocator: std.mem.Allocator, hp: Hparams, ds: ?Dspark) ![]u8 {
    return configJsonWith(allocator, hp, ds, null);
}

/// `configJson`, with the packs' `vision_config` and image token when a tower GGUF `vision` sits beside the set.
pub fn configJsonWith(allocator: std.mem.Allocator, hp: Hparams, ds: ?Dspark, vision: ?Vision) ![]u8 {
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
    var targets: std.ArrayList(u8) = .empty;
    defer targets.deinit(allocator);
    for (hp.compress[0..hp.n_layers], 0..) |r, l| {
        try ratios.print(allocator, "{s}{d}", .{ sep(ratios.items), r });
        if ((hp.kv_sources >> @intCast(l)) & 1 != 0) try kv.print(allocator, "{s}{d}", .{ sep(kv.items), l });
        if ((hp.index_sources >> @intCast(l)) & 1 != 0) try index.print(allocator, "{s}{d}", .{ sep(index.items), l });
    }
    if (ds) |d| {
        for (d.compress[0..d.stages]) |r| try ratios.print(allocator, ",{d}", .{r});
        for (d.targets[0..d.n_targets]) |t| try targets.print(allocator, "{s}{d}", .{ sep(targets.items), t });
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
    if (ds) |d| try out.print(allocator,
        \\"num_nextn_predict_layers":{[stages]d},"dspark_block_size":{[block]d},"dspark_noise_token_id":{[noise]d},
        \\"dspark_markov_rank":{[rank]d},"dspark_target_layer_ids":[{[targets]s}],"dspark_n_routed_experts":{[experts]d},
        \\"dspark_num_experts_per_tok":{[top_k]d},
    , .{ .stages = d.stages, .block = d.block, .noise = d.noise, .rank = d.markov_rank, .targets = targets.items, .experts = d.experts, .top_k = d.top_k });
    try out.print(allocator,
        \\"n_routed_experts":{[experts]d},"n_shared_experts":{[shared]d},"num_experts_per_tok":{[used]d},
        \\"moe_intermediate_size":{[ffn]d},"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","norm_topk_prob":{[norm]},
        \\"routed_scaling_factor":{[scale]d},"swiglu_limit":{[swiglu]d},"num_hash_layers":{[hash]d},
        \\"index_n_heads":{[idx_heads]d},"index_head_dim":{[idx_dim]d},"index_topk":{[idx_topk]d},
        \\"candidate_source_layer_id":{[cand]d},"candidate_topk_blocks":{[cand_topk]d},"candidate_block_size":{[cand_block]d},
        \\"hc_mult":{[hc]d},"hc_sinkhorn_iters":{[hc_iters]d},"hc_eps":{[hc_eps]e}}}
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
    if (vision) |v| try out.print(allocator,
        \\,"image_token_id":{[token]d},"vision_config":{{"model_type":"deepseek_v41_vision","num_hidden_layers":{[layers]d},
        \\"hidden_size":{[hidden]d},"num_attention_heads":{[heads]d},"intermediate_size":{[ffn]d},"patch_size":{[patch]d},
        \\"rope_theta":{[theta]d},"downsample_ratio":{[ds]d},"max_image_tokens":{[max]d},"min_pixels":{[min]d},
        \\"max_wh_ratio":{[ratio]d}}}
    , .{
        .token = v.image_token,
        .layers = v.layers,
        .hidden = v.hidden,
        .heads = v.heads,
        .ffn = v.ffn,
        .patch = v.patch,
        .theta = v.rope_theta,
        .ds = v.downsample,
        .max = v.max_tokens,
        .min = v.min_pixels,
        .ratio = v.max_wh_ratio,
    });
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator);
}

/// A block's GGUF leaves and the plugin's names under its layer (trunk layers and DSpark stages alike).
const leaves = [_]struct { []const u8, []const u8 }{
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
    // The DS4 layout's names.
    .{ "engram_q_norm.weight", "engram.q_weight" },
    .{ "engram_k_norm.weight", "engram.k_weight" },
};

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
    return leafOf(buf, "layers", blk.layer, blk.leaf);
}

fn leafOf(buf: []u8, comptime prefix: []const u8, layer: u32, leaf: []const u8) ?arch.Mapped {
    for (leaves) |row| if (std.mem.eql(u8, leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, prefix ++ ".{d}.{s}", .{ layer, row[1] }) catch return null;
        const f32_leaf = std.mem.startsWith(u8, row[0], "hc_") or std.mem.startsWith(u8, row[0], "exp_probs_b") or
            std.mem.eql(u8, row[0], "attn_sinks.weight") or std.mem.eql(u8, row[0], "engram_gate.weight");
        return .{ .name = full, .transform = if (f32_leaf) .keep_f32 else .none };
    };
    return null;
}

/// A stage bundle's heads: Unsloth's name, the plugin's under `mtp.<stage>.` and whether that is the last stage (else
/// stage 0). The DS4 bundle ships them under the plugin's names already.
const dspark_heads = [_]struct { []const u8, []const u8, bool }{
    .{ "fc.weight", "main_proj.weight", false },
    .{ "enc.output_norm.weight", "main_norm.weight", false },
    .{ "output_norm.weight", "norm.weight", true },
    .{ "markov_w1.weight", "markov_head.embed.weight", true },
    .{ "markov_w2.weight", "markov_head.head.weight", true },
    .{ "conf_proj.weight", "confidence_head.proj.weight", true },
};

/// A stage bundle's tensor onto the plugin's DSpark names (`mtp.<stage>.*`, the heads on stage 0 and the last),
/// null when the tensor is unknown.
pub fn mapDsparkTensor(buf: []u8, name: []const u8, ds: Dspark) ?arch.Mapped {
    const last = ds.stages - 1;
    for (dspark_heads) |h| if (std.mem.eql(u8, name, h[0])) {
        return .{ .name = std.fmt.bufPrint(buf, "mtp.{d}.{s}", .{ if (h[2]) last else 0, h[1] }) catch return null, .transform = .none };
    };
    if (std.mem.startsWith(u8, name, "mtp.")) {
        const rest = name["mtp.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const stage = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
        if (stage >= ds.stages) return null;
        for (dspark_heads) |h| if (std.mem.eql(u8, rest[dot + 1 ..], h[1])) {
            if (stage != if (h[2]) last else 0) return null;
            return .{ .name = std.fmt.bufPrint(buf, "{s}", .{name}) catch return null, .transform = .none };
        };
        return leafOf(buf, "mtp", stage, rest[dot + 1 ..]);
    }
    const blk = arch.splitLayer(name, ds.stages) orelse return null;
    return leafOf(buf, "mtp", blk.layer, blk.leaf);
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

/// The release's config.json as the DS4 converter embeds it (`deepseek41.config`), its tower and fp8 block left out.
pub const release_config =
    \\{"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41","text_config":{"attention_bias":false,
    \\"attention_dropout":0.0,"candidate_block_size":8,"candidate_source_layer_id":20,"candidate_topk_blocks":2048,
    \\"compress_ratios":[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    \\"compress_rope_theta":160000,"dspark_block_size":5,"dspark_markov_rank":256,"dspark_n_routed_experts":128,
    \\"dspark_noise_token_id":128799,"dspark_num_experts_per_tok":3,"dspark_target_layer_ids":[37,38,39],
    \\"engram_compressed_vocab_size":99092,"engram_head_dim":256,"engram_layer_ids":[1,14],"engram_max_ngram_size":4,
    \\"engram_n_heads":8,"engram_num_embeddings":[384006168,384016682],"engram_pad_token_id":2,
    \\"engram_vocab_size":16000000,"hc_eps":1e-06,"hc_mult":4,"hc_sinkhorn_iters":20,"head_dim":512,
    \\"hidden_act":"silu","hidden_size":5120,"index_head_dim":128,"index_n_heads":32,"index_source_layer_ids":[2,8,14,
    \\20,24,28,32,36],"index_topk":512,"initializer_range":0.02,"kv_source_layer_ids":[2,8,14,20],
    \\"max_position_embeddings":1048576,"model_type":"deepseek_v41_text","moe_intermediate_size":2304,
    \\"n_routed_experts":384,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,
    \\"num_experts_per_tok":6,"num_hidden_layers":40,"num_key_value_heads":1,"num_nextn_predict_layers":3,
    \\"o_groups":8,"o_lora_rank":1024,"q_lora_rank":1280,"qk_rope_head_dim":64,"rms_norm_eps":1e-20,
    \\"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,
    \\"rope_type":"yarn"},"rope_theta":10000,"routed_scaling_factor":1.5,"scoring_func":"sqrtsoftplus",
    \\"sliding_window":128,"swiglu_limit":10.0,"tie_word_embeddings":false,"topk_method":"noaux_tc","use_cache":true,
    \\"vocab_size":129280}}
;

/// The DS4 layout's metadata (ivanfioravanti's `DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf`), `config` the
/// embedded config.json (tests only).
pub fn testFileDs4(config: []const u8) !gguf.File {
    var f: gguf.File = .{ .arena = .init(std.testing.allocator), .version = 3 };
    errdefer f.deinit();
    const a = f.arena.allocator();
    try f.kv.put(a, p ++ "config", .{ .string = try a.dupe(u8, config) });
    try f.kv.put(a, p ++ "engram.encoding", .{ .string = engram_encoding });
    try f.kv.put(a, "tokenizer.ggml.bos_token_id", .{ .uint = 0 });
    try f.kv.put(a, "tokenizer.ggml.eos_token_id", .{ .uint = 1 });
    try f.kv.put(a, "tokenizer.ggml.tokens", .{ .array = .{ .elem_type = .string, .len = 129280, .bytes = &.{} } });
    return f;
}

test "Hparams.read: the DS4 layout's embedded config.json gives the 4-shard set's hyperparameters" {
    var old = try testFile();
    defer old.deinit();
    var ds4 = try testFileDs4(release_config);
    defer ds4.deinit();
    var want = try Hparams.read(&old);
    const got = try Hparams.read(&ds4);
    // The 4-shard set's epsilons are f32 keys: the same values at the f32 the plugin reads.
    for ([_]*f64{ &want.rms_eps, &want.hc_eps }, [_]f64{ got.rms_eps, got.hc_eps }) |w, g| {
        try std.testing.expectEqual(@as(f32, @floatCast(w.*)), @as(f32, @floatCast(g)));
        w.* = g;
    }
    try std.testing.expectEqualDeep(want, got);
}

test "Hparams.read refuses a DS4 layout with Engram rows in another encoding, another routing or a source past the trunk" {
    const a = std.testing.allocator;
    var enc = try testFileDs4(release_config);
    defer enc.deinit();
    try enc.kv.put(enc.arena.allocator(), p ++ "engram.encoding", .{ .string = "e5m2_e8m0_32_row264" });
    try std.testing.expectError(error.BadMetadata, Hparams.read(&enc));
    for ([_][2][]const u8{
        .{ "\"sqrtsoftplus\"", "\"softmax\"" },
        .{ "\"kv_source_layer_ids\":[2,8,14,20]", "\"kv_source_layer_ids\":[2,8,14,40]" },
        .{ "\"engram_layer_ids\":[1,14]", "\"engram_layer_ids\":[1,14,20]" },
        .{ "\"rope_type\":\"yarn\"", "\"rope_type\":\"linear\"" },
    }) |edit| {
        const json = try std.mem.replaceOwned(u8, a, release_config, edit[0], edit[1]);
        defer a.free(json);
        var f = try testFileDs4(json);
        defer f.deinit();
        try std.testing.expectError(error.BadMetadata, Hparams.read(&f));
    }
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
        .{ "blk.1.engram_q_norm.weight", "layers.1.engram.q_weight" },
        .{ "blk.14.engram_k_norm.weight", "layers.14.engram.k_weight" },
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


/// Unsloth's stage bundle (`dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`) metadata, as a File (tests only).
pub fn testDsparkFile() !gguf.File {
    var f: gguf.File = .{ .arena = .init(std.testing.allocator), .version = 3 };
    errdefer f.deinit();
    const a = f.arena.allocator();
    try f.kv.put(a, "general.architecture", .{ .string = dspark_arch });
    const ints = [_]struct { []const u8, u64 }{
        .{ dp ++ "block_count", 3 },                   .{ dp ++ "embedding_length", 5120 },
        .{ dp ++ "attention.head_count", 64 },         .{ dp ++ "attention.key_length", 512 },
        .{ dp ++ "attention.q_lora_rank", 1280 },      .{ dp ++ "attention.output_lora_rank", 1024 },
        .{ dp ++ "attention.output_group_count", 8 },  .{ dp ++ "expert_feed_forward_length", 2304 },
        .{ dp ++ "block_size", 5 },                    .{ dp ++ "expert_count", 128 },
        .{ dp ++ "expert_used_count", 3 },             .{ "tokenizer.ggml.mask_token_id", 128799 },
    };
    for (ints) |kv| try f.kv.put(a, kv[0], .{ .uint = kv[1] });
    const lists = [_]struct { []const u8, []const i32 }{
        .{ dp ++ "target_layers", &.{ 37, 38, 39 } },
        .{ dp ++ "attention.compress_ratios", &.{ 0, 0, 0 } },
    };
    for (lists) |l| try f.kv.put(a, l[0], .{ .array = .{ .elem_type = .i32, .len = l[1].len, .bytes = std.mem.sliceAsBytes(l[1]) } });
    try f.tensors.put(a, "markov_w1.weight", .{ .dims = .{ 256, 129280, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = &.{} });
    return f;
}

test "Dspark.read: Unsloth's stage bundle gives our packs' dspark keys; another trunk's shapes or targets are refused" {
    var t = try testFile();
    defer t.deinit();
    const hp = try Hparams.read(&t);
    var f = try testDsparkFile();
    defer f.deinit();
    const d = try Dspark.read(&f, hp);
    try std.testing.expectEqual(@as(u32, 3), d.stages);
    try std.testing.expectEqual(@as(u32, 5), d.block);
    try std.testing.expectEqualSlices(u8, &.{ 37, 38, 39 }, d.targets[0..d.n_targets]);
    try std.testing.expectEqual(@as(u32, 128799), d.noise);
    try std.testing.expectEqual(@as(u32, 256), d.markov_rank);
    try std.testing.expectEqual(@as(u32, 128), d.experts);
    try std.testing.expectEqual(@as(u32, 3), d.top_k);

    const a = f.arena.allocator();
    try f.kv.put(a, dp ++ "embedding_length", .{ .uint = 4096 });
    try std.testing.expectError(error.BadMetadata, Dspark.read(&f, hp));
    var g = try testDsparkFile();
    defer g.deinit();
    try g.kv.put(g.arena.allocator(), dp ++ "target_layers", .{ .array = .{ .elem_type = .i32, .len = 1, .bytes = std.mem.sliceAsBytes(&[_]i32{40}) } });
    try std.testing.expectError(error.BadMetadata, Dspark.read(&g, hp));
    var h = try testFile();
    defer h.deinit();
    try std.testing.expectError(error.UnsupportedArch, Dspark.read(&h, hp));
}

test "mapDsparkTensor: a stage block's leaves under mtp.<stage>, the heads on stage 0 and the last" {
    var f = try testDsparkFile();
    defer f.deinit();
    var t = try testFile();
    defer t.deinit();
    const d = try Dspark.read(&f, try Hparams.read(&t));
    var buf: [160]u8 = undefined;
    const want = [_]struct { []const u8, []const u8 }{
        .{ "blk.0.attn_q_b.weight", "mtp.0.attn.wq_b.weight" },
        .{ "blk.2.ffn_gate_exps.weight", "mtp.2.ffn.experts.w1.weight" },
        .{ "blk.1.exp_probs_b.bias", "mtp.1.ffn.gate.bias" },
        .{ "blk.1.hc_ffn_fn.weight", "mtp.1.hc_ffn_fn" },
        .{ "fc.weight", "mtp.0.main_proj.weight" },
        .{ "enc.output_norm.weight", "mtp.0.main_norm.weight" },
        .{ "output_norm.weight", "mtp.2.norm.weight" },
        .{ "markov_w1.weight", "mtp.2.markov_head.embed.weight" },
        .{ "markov_w2.weight", "mtp.2.markov_head.head.weight" },
        .{ "conf_proj.weight", "mtp.2.confidence_head.proj.weight" },
    };
    for (want) |w| try std.testing.expectEqualStrings(w[1], (mapDsparkTensor(&buf, w[0], d) orelse return error.TestUnexpectedResult).name);
    for ([_][]const u8{ "blk.3.attn_q_b.weight", "token_embd.weight", "blk.0.attn_q.weight" }) |n|
        try std.testing.expect(mapDsparkTensor(&buf, n, d) == null);
}

/// The DS4 converter's stage bundle (ivanfioravanti's `DeepSeek-V4.1-Flash-1017-DSpark-support-MXFP4.gguf`) metadata
/// and the stage tensors `Dspark.read` looks at, as a File (tests only).
pub fn testDsparkFileDs4() !gguf.File {
    var f: gguf.File = .{ .arena = .init(std.testing.allocator), .version = 3 };
    errdefer f.deinit();
    const a = f.arena.allocator();
    try f.kv.put(a, "general.architecture", .{ .string = ds4_dspark_arch });
    const ints = [_]struct { []const u8, u64 }{
        .{ ds4p ++ "block_size", 5 },       .{ ds4p ++ "markov_rank", 256 },        .{ ds4p ++ "noise_token_id", 128799 },
        .{ ds4p ++ "n_routed_experts", 128 }, .{ ds4p ++ "num_experts_per_tok", 3 },
    };
    for (ints) |kv| try f.kv.put(a, kv[0], .{ .uint = kv[1] });
    try f.kv.put(a, ds4p ++ "target_layer_ids", .{ .array = .{ .elem_type = .i32, .len = 3, .bytes = std.mem.sliceAsBytes(&[_]i32{ 37, 38, 39 }) } });
    for (0..3) |s| {
        try f.tensors.put(a, try std.fmt.allocPrint(a, "mtp.{d}.attn_q_b.weight", .{s}), .{ .dims = .{ 1280, 32768, 1, 1 }, .n_dims = 2, .ty = .q8_0, .data = &.{} });
        try f.tensors.put(a, try std.fmt.allocPrint(a, "mtp.{d}.ffn_gate_exps.weight", .{s}), .{ .dims = .{ 5120, 2304, 128, 1 }, .n_dims = 3, .ty = .mxfp4, .data = &.{} });
    }
    return f;
}

test "Dspark.read: the DS4 converter's stage bundle gives Unsloth's geometry; stages of another shape are refused" {
    var t = try testFile();
    defer t.deinit();
    const hp = try Hparams.read(&t);
    var u = try testDsparkFile();
    defer u.deinit();
    var d = try testDsparkFileDs4();
    defer d.deinit();
    try std.testing.expectEqualDeep(try Dspark.read(&u, hp), try Dspark.read(&d, hp));

    const a = d.arena.allocator();
    try d.tensors.put(a, "mtp.1.ffn_gate_exps.weight", .{ .dims = .{ 5120, 2304, 64, 1 }, .n_dims = 3, .ty = .mxfp4, .data = &.{} });
    try std.testing.expectError(error.BadMetadata, Dspark.read(&d, hp));
    var q = try testDsparkFileDs4();
    defer q.deinit();
    try q.tensors.put(q.arena.allocator(), "mtp.2.attn_q_b.weight", .{ .dims = .{ 1024, 32768, 1, 1 }, .n_dims = 2, .ty = .q8_0, .data = &.{} });
    try std.testing.expectError(error.BadMetadata, Dspark.read(&q, hp));
}

test "mapDsparkTensor: the DS4 bundle's stage leaves onto the plugin's names, its heads already under them" {
    var f = try testDsparkFileDs4();
    defer f.deinit();
    var t = try testFile();
    defer t.deinit();
    const d = try Dspark.read(&f, try Hparams.read(&t));
    var buf: [160]u8 = undefined;
    const want = [_]struct { []const u8, []const u8 }{
        .{ "mtp.0.attn_q_b.weight", "mtp.0.attn.wq_b.weight" },
        .{ "mtp.2.ffn_gate_exps.weight", "mtp.2.ffn.experts.w1.weight" },
        .{ "mtp.1.exp_probs_b_vl.bias", "mtp.1.ffn.gate.bias_vl" },
        .{ "mtp.1.hc_ffn_fn.weight", "mtp.1.hc_ffn_fn" },
        .{ "mtp.0.main_proj.weight", "mtp.0.main_proj.weight" },
        .{ "mtp.0.main_norm.weight", "mtp.0.main_norm.weight" },
        .{ "mtp.2.norm.weight", "mtp.2.norm.weight" },
        .{ "mtp.2.markov_head.embed.weight", "mtp.2.markov_head.embed.weight" },
        .{ "mtp.2.markov_head.head.weight", "mtp.2.markov_head.head.weight" },
        .{ "mtp.2.confidence_head.proj.weight", "mtp.2.confidence_head.proj.weight" },
    };
    for (want) |w| {
        const m = mapDsparkTensor(&buf, w[0], d) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(w[1], m.name);
        try std.testing.expect(m.transform == if (std.mem.indexOf(u8, w[0], ".hc_") != null or std.mem.indexOf(u8, w[0], "exp_probs_b") != null) arch.Transform.keep_f32 else arch.Transform.none);
    }
    for ([_][]const u8{ "mtp.3.attn_q_b.weight", "mtp.0.attn_q.weight", "mtp.x.norm.weight", "mtp.1.norm.weight", "mtp.1.fc.weight", "token_embd.weight" }) |n|
        try std.testing.expect(mapDsparkTensor(&buf, n, d) == null);
}

/// The DS4 converter's tower (`DeepSeek-V4.1-Flash-1017-Vision.gguf`) metadata and two of its tensors, as a File
/// (tests only).
pub fn testVisionFileDs4() !gguf.File {
    var f: gguf.File = .{ .arena = .init(std.testing.allocator), .version = 3 };
    errdefer f.deinit();
    const a = f.arena.allocator();
    try f.kv.put(a, "general.architecture", .{ .string = vision_arch });
    try f.kv.put(a, "general.source.revision", .{ .string = "1bae0e1088993b9b82de08f59de5be5a34e42170" });
    try f.kv.put(a, vp ++ "checkpoint_variant", .{ .string = "v4.1-flash" });
    try f.kv.put(a, vp ++ "rope.freq_base", .{ .float = 10000 });
    const ints = [_]struct { []const u8, u64 }{
        .{ vp ++ "block_count", 32 },              .{ vp ++ "embedding_length", 1024 },     .{ vp ++ "feed_forward_length", 2816 },
        .{ vp ++ "attention.head_count", 16 },     .{ vp ++ "projection_length", 5120 },    .{ vp ++ "patch_size", 14 },
        .{ vp ++ "downsample_ratio", 3 },          .{ vp ++ "image.max_tokens", 1024 },     .{ vp ++ "image.min_pixels", 295936 },
        .{ vp ++ "image.max_width_height_ratio", 0 }, .{ vp ++ "image.token_id", 129264 }, .{ vp ++ "language.block_count", 40 },
        .{ vp ++ "language.expert_count", 384 },
    };
    for (ints) |kv| try f.kv.put(a, kv[0], .{ .uint = kv[1] });
    try f.tensors.put(a, "vision.patch_embed.proj.weight", .{ .dims = .{ 588, 1024, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = &.{} });
    try f.tensors.put(a, "aligner.w1.weight", .{ .dims = .{ 9216, 5120, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = &.{} });
    return f;
}

test "Vision.read: the DS4 converter's tower gives our packs' vision_config; another checkpoint, trunk or type is refused" {
    var t = try testFileDs4(release_config);
    defer t.deinit();
    const hp = try Hparams.read(&t);
    var v = try testVisionFileDs4();
    defer v.deinit();
    const rev = v.getString("general.source.revision").?;
    try std.testing.expectEqualDeep(Vision{
        .layers = 32, .hidden = 1024, .ffn = 2816, .heads = 16, .patch = 14, .downsample = 3, .max_tokens = 1024,
        .min_pixels = 295936, .max_wh_ratio = 0, .rope_theta = 10000, .image_token = 129264,
    }, try Vision.read(&v, hp, rev));
    // A set that names no revision pairs by geometry alone.
    _ = try Vision.read(&v, hp, null);
    try std.testing.expectError(error.OtherCheckpoint, Vision.read(&v, hp, "df42c109f1defefcbfcedbe7d905718a12266e40"));

    const a = v.arena.allocator();
    try v.kv.put(a, vp ++ "language.expert_count", .{ .uint = 256 });
    try std.testing.expectError(error.BadMetadata, Vision.read(&v, hp, rev));
    var e = try testVisionFileDs4();
    defer e.deinit();
    try e.kv.put(e.arena.allocator(), vp ++ "checkpoint_variant", .{ .string = "vision-exp" });
    try std.testing.expectError(error.BadMetadata, Vision.read(&e, hp, rev));
    var q = try testVisionFileDs4();
    defer q.deinit();
    try q.tensors.put(q.arena.allocator(), "aligner.w2.weight", .{ .dims = .{ 5120, 5120, 1, 1 }, .n_dims = 2, .ty = .q8_0, .data = &.{} });
    try std.testing.expectError(error.UnsupportedTensor, Vision.read(&q, hp, rev));
    var g = try testVisionFileDs4();
    defer g.deinit();
    try g.kv.put(g.arena.allocator(), "general.architecture", .{ .string = "clip" });
    try std.testing.expectError(error.UnsupportedArch, Vision.read(&g, hp, rev));
    // A tensor the tower does not name: declined here, so the set serves text instead of failing its load.
    var x = try testVisionFileDs4();
    defer x.deinit();
    try x.tensors.put(x.arena.allocator(), "mm_projector.weight", .{ .dims = .{ 1024, 1024, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = &.{} });
    try std.testing.expectError(error.UnsupportedTensor, Vision.read(&x, hp, rev));
}

/// llama.cpp's mmproj for the same tower (`mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf`, projector `deepseek41v`): its
/// metadata and two of its tensors, as a File (tests only).
pub fn testVisionFileClip() !gguf.File {
    var f: gguf.File = .{ .arena = .init(std.testing.allocator), .version = 3 };
    errdefer f.deinit();
    const a = f.arena.allocator();
    try f.kv.put(a, "general.architecture", .{ .string = "clip" });
    try f.kv.put(a, "clip.projector_type", .{ .string = clip_projector });
    const ints = [_]struct { []const u8, u64 }{
        .{ cvp ++ "block_count", 32 },  .{ cvp ++ "embedding_length", 1024 }, .{ cvp ++ "feed_forward_length", 2816 },
        .{ cvp ++ "attention.head_count", 16 }, .{ cvp ++ "projection_dim", 5120 }, .{ cvp ++ "patch_size", 14 },
        .{ cvp ++ "projector.scale_factor", 3 }, .{ cvp ++ "image_min_pixels", 295936 }, .{ cvp ++ "image_size", 672 },
    };
    for (ints) |kv| try f.kv.put(a, kv[0], .{ .uint = kv[1] });
    try f.tensors.put(a, "v.patch_embd.weight", .{ .dims = .{ 14, 14, 3, 1024 }, .n_dims = 4, .ty = .f32, .data = &.{} });
    try f.tensors.put(a, "mm.1.weight", .{ .dims = .{ 9216, 5120, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = &.{} });
    return f;
}

test "Vision.read: llama.cpp's mmproj of the tower gives the DS4 tower's vision_config; another projector or trunk is refused" {
    var t = try testFileDs4(release_config);
    defer t.deinit();
    const hp = try Hparams.read(&t);
    var c = try testVisionFileClip();
    defer c.deinit();
    var d = try testVisionFileDs4();
    defer d.deinit();
    try std.testing.expectEqualDeep(try Vision.read(&d, hp, null), try Vision.read(&c, hp, "1bae0e1088993b9b82de08f59de5be5a34e42170"));
    var o = try testVisionFileClip();
    defer o.deinit();
    try o.kv.put(o.arena.allocator(), "clip.projector_type", .{ .string = "qwen3vl_merger" });
    try std.testing.expectError(error.UnsupportedArch, Vision.read(&o, hp, null));
    var w = try testVisionFileClip();
    defer w.deinit();
    try w.kv.put(w.arena.allocator(), cvp ++ "projection_dim", .{ .uint = 4096 });
    try std.testing.expectError(error.BadMetadata, Vision.read(&w, hp, null));
}

test "mapVisionTensor: llama.cpp's mmproj names onto the tower's; gate and up land apart for loadVision to join" {
    var buf: [96]u8 = undefined;
    const want = [_][2][]const u8{
        .{ "mm.1.weight", "aligner.w1.weight" },                         .{ "mm.2.bias", "aligner.w2.bias" },
        .{ "v.blk.0.attn_qkv.weight", "vision.blocks.0.attn.wqkv.weight" }, .{ "v.blk.31.attn_out.bias", "vision.blocks.31.attn.wo.bias" },
        .{ "v.blk.7.ffn_down.weight", "vision.blocks.7.mlp.w2.weight" },  .{ "v.blk.7.ln1.weight", "vision.blocks.7.norm1.weight" },
        .{ "v.blk.7.ln2.weight", "vision.blocks.7.norm2.weight" },        .{ "v.blk.7.ffn_gate.weight", "vision.blocks.7.mlp.w1_gate.weight" },
        .{ "v.blk.7.ffn_up.weight", "vision.blocks.7.mlp.w1_up.weight" }, .{ "v.post_ln.weight", "vision.norm.weight" },
        .{ "v.patch_embd.weight", "vision.patch_embed.proj.weight" },     .{ "v.patch_embd.bias", "vision.patch_embed.proj.bias" },
        .{ "v.token_embd.img_start", "image_start" },                     .{ "v.token_embd.img_end", "image_end" },
        .{ "v.image_newline", "image_newline" },
    };
    for (want) |w| {
        const m = mapVisionTensor(&buf, w[0]) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(w[1], m.name);
    }
    for ([_][]const u8{ "v.blk.0.ffn_norm.weight", "mm.0.weight", "v.position_embd.weight" }) |n| try std.testing.expect(mapVisionTensor(&buf, n) == null);
}

test "mapVisionTensor: the tower keeps the release's names, which the plugin's tower reads; anything else is unknown" {
    var buf: [96]u8 = undefined;
    for ([_][]const u8{ "vision.blocks.31.mlp.w1.weight", "vision.patch_embed.proj.bias", "vision.norm.weight", "aligner.w2.bias", "image_start", "image_end", "image_newline" }) |n| {
        const m = mapVisionTensor(&buf, n) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(n, m.name);
        try std.testing.expect(m.transform == .none);
    }
    for ([_][]const u8{ "image_pad", "layers.0.ffn.gate.bias_vl", "token_embd.weight" }) |n|
        try std.testing.expect(mapVisionTensor(&buf, n) == null);
}
