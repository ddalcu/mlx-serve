//! Qwen3.8-Flash-Next (GGUF arch "qwen4exp"): the qwen35 GatedDeltaNet + MoE trunk (value heads tiled the same way,
//! see arch/qwen35.zig) inside 4 hyper-connection streams, QSA sparse attention on the full layers and an n-gram PLE
//! on one layer. llama.cpp's converter splits the indexer's q|k projection in two (joined again at load) and stores
//! every zero-centred norm folded (w + 1). Its affine-coded blocks load as MLX affine (`affine.zig`), so the plugin's
//! affine kernels serve them. The n-gram table (`per_layer_token_embd`) stays in the file: the qwen4_exp plugin reads
//! its rows from there.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");
const qwen35 = @import("qwen35.zig");

pub const arch_name = "qwen4exp";
const p = arch_name ++ ".";

/// The n-gram table: every PLE head's rows, IQ4_NL or any other ggml type.
pub const table_name = "per_layer_token_embd.weight";

/// Names follow our qwen4_exp packs.
const prefix = "language_model.model.";

pub const Hparams = struct {
    base: qwen35.Hparams,
    hc_count: u32,
    hc_lowrank: u32,
    idx_heads: u32,
    idx_dim: u32,
    idx_budget: u32,
    compress: u32,
    ple_layer: u32,
    ngram_size: u32,
    heads_per_ngram: u32,
    ple_conv: u32,
    ple_eos: u32,
    /// One head's row width; the PLE embedding is `n_heads` rows side by side.
    ple_row: u32,
    /// The hash's vocab base and row divisor, recovered from the stored head sizes and table rows.
    ngram_base: u64,
    ngram_divisor: u64,
    image_token: u32,

    pub fn read(f: *const gguf.File) !Hparams {
        const base = try qwen35.Hparams.read(f, arch_name);
        const ratios = f.getArray(p ++ "attention.compress_ratios") orelse return error.MissingMetadata;
        if (ratios.len != base.n_layers) return error.BadMetadata;
        var compress: u32 = 0;
        for (0..ratios.len) |i| {
            const want_full = (i + 1) % base.full_attn_interval == 0;
            const v = std.math.cast(u32, ratios.int(i)) orelse return error.BadMetadata;
            if (want_full != (v != 0) or (compress != 0 and v != 0 and v != compress)) return error.BadMetadata;
            if (v != 0) compress = v;
        }
        const layers = f.getArray(p ++ "ple.layers") orelse return error.MissingMetadata;
        if (layers.len != 1) return error.BadMetadata;
        const hp = Hparams{
            .base = base,
            .hc_count = try arch.u32Key(f, p ++ "hyper_connection.count"),
            .hc_lowrank = try arch.u32Key(f, p ++ "hyper_connection.low_rank"),
            .idx_heads = try arch.u32Key(f, p ++ "attention.indexer.head_count"),
            .idx_dim = try arch.u32Key(f, p ++ "attention.indexer.key_length"),
            .idx_budget = try arch.u32Key(f, p ++ "attention.indexer.top_k"),
            .compress = compress,
            .ple_layer = std.math.cast(u32, layers.int(0)) orelse return error.BadMetadata,
            .ngram_size = try arch.u32Key(f, p ++ "ple.ngram_size"),
            .heads_per_ngram = try arch.u32Key(f, p ++ "ple.heads_per_ngram"),
            .ple_conv = try arch.u32Key(f, p ++ "ple.conv_kernel"),
            .ple_eos = try arch.u32Key(f, p ++ "ple.eos_token_id"),
            .ple_row = try arch.u32Key(f, p ++ "embedding_length_per_layer_input"),
            .ngram_base = 0,
            .ngram_divisor = 0,
            .image_token = if (f.getInt(p ++ "ple.image_token_id") != null) try arch.u32Key(f, p ++ "ple.image_token_id") else 0,
        };
        if (hp.ngram_size < 2 or hp.ple_layer >= base.n_layers or compress == 0) return error.BadMetadata;
        var out = hp;
        try out.readHash(f);
        return out;
    }

    /// The hash's vocab sizes are consecutive primes from its base, back to back, padded to a row multiple:
    /// any base in (previous prime, first size] and any divisor that pads the sum to the table's rows rebuild them.
    fn readHash(self: *Hparams, f: *const gguf.File) !void {
        const sizes = f.getArray(p ++ "ple.head_vocab_sizes") orelse return error.MissingMetadata;
        const offsets = f.getArray(p ++ "ple.head_offsets") orelse return error.MissingMetadata;
        if (sizes.len != self.nHeads() or offsets.len != sizes.len) return error.BadMetadata;
        var total: u64 = 0;
        for (0..sizes.len) |i| {
            if (offsets.int(i) != @as(i64, @intCast(total)) or sizes.int(i) < 2) return error.BadMetadata;
            total += @intCast(sizes.int(i));
        }
        const t = f.tensors.get(table_name) orelse return error.MissingMetadata;
        if (t.n_dims != 2 or t.dims[0] != self.ple_row or t.dims[1] < total) return error.BadMetadata;
        const rows = t.dims[1];
        var d: u64 = 1;
        while (d <= 1 << 24) : (d *= 2) {
            if (rows % d == 0 and rows - total < d) break;
        } else return error.BadMetadata;
        self.ngram_base = @intCast(sizes.int(0));
        self.ngram_divisor = d;
    }

    pub fn nHeads(self: Hparams) u32 {
        return (self.ngram_size - 1) * self.heads_per_ngram;
    }
};

/// The config.json of our qwen4_exp packs, text only (no MTP head, no tower: the file carries neither).
pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    const b = hp.base;
    var layer_types: std.ArrayList(u8) = .empty;
    defer layer_types.deinit(allocator);
    for (0..b.n_layers) |i| {
        if (i > 0) try layer_types.append(allocator, ',');
        try layer_types.appendSlice(allocator, if ((i + 1) % b.full_attn_interval == 0) "\"full_attention\"" else "\"linear_attention\"");
    }
    const rotary = @as(f64, @floatFromInt(b.rope_dims)) / @as(f64, @floatFromInt(b.head_dim));
    // Three prints: one format call takes at most 32 arguments.
    const ple = try std.fmt.allocPrint(allocator,
        \\"heads_per_ngram":{[heads_per_ngram]d},"make_ngram_vocab_size_divisible_by":{[divisor]d},"ngram_size":{[ngram]d},
        \\"ngram_vocab_size_base":{[base]d},"ple_conv_kernel_size":{[ple_conv]d},"ple_embed_dim":{[ple_dim]d},"ple_layer_ids":[{[ple_id]d}],
        \\"hc_count":{[hc]d},"hc_lowrank":{[hc_lowrank]d},"indexer_budget":{[budget]d},"indexer_compress_ratio":{[compress]d},
        \\"indexer_head_dim":{[idx_dim]d},"indexer_n_heads":{[idx_heads]d},
        \\"rope_parameters":{{"mrope_interleaved":true,"mrope_section":[{[m0]d},{[m1]d},{[m2]d}],"partial_rotary_factor":{[rotary]d},
        \\"rope_theta":{[theta]d},"rope_type":"default"}}
    , .{
        .heads_per_ngram = hp.heads_per_ngram,
        .divisor = hp.ngram_divisor,
        .ngram = hp.ngram_size,
        .base = hp.ngram_base,
        .ple_conv = hp.ple_conv,
        .ple_dim = hp.ple_row * hp.nHeads(),
        .ple_id = hp.ple_layer + 1,
        .hc = hp.hc_count,
        .hc_lowrank = hp.hc_lowrank,
        .budget = hp.idx_budget,
        .compress = hp.compress,
        .idx_dim = hp.idx_dim,
        .idx_heads = hp.idx_heads,
        .m0 = b.mrope[0],
        .m1 = b.mrope[1],
        .m2 = b.mrope[2],
        .rotary = rotary,
        .theta = b.rope_theta,
    });
    defer allocator.free(ple);
    const trunk = try std.fmt.allocPrint(allocator,
        \\"model_type":"qwen4_exp_text","attention_bias":false,"dtype":"bfloat16","eos_token_id":{[ple_eos]d},
        \\"full_attention_interval":{[interval]d},"head_dim":{[head_dim]d},"hidden_act":"silu","hidden_size":{[hidden]d},
        \\"layer_types":[{[layer_types]s}],"linear_conv_kernel_dim":{[conv]d},"linear_key_head_dim":{[dk]d},
        \\"linear_num_key_heads":{[nk]d},"linear_num_value_heads":{[nv]d},"linear_value_head_dim":{[dv]d},
        \\"mamba_ssm_dtype":"float32","max_position_embeddings":{[ctx]d},"moe_intermediate_size":{[expert_ffn]d},
        \\"num_attention_heads":{[n_heads]d},"num_experts":{[n_experts]d},"num_experts_per_tok":{[topk]d},
        \\"num_hidden_layers":{[n_layers]d},"num_key_value_heads":{[n_kv]d},"output_gate_type":"sigmoid",
        \\"partial_rotary_factor":{[rotary]d},"rms_norm_eps":{[eps]e},"shared_expert_intermediate_size":{[shared_ffn]d},
        \\"tie_word_embeddings":{[tied]},"vocab_size":{[vocab]d}
    , .{
        .ple_eos = hp.ple_eos,
        .interval = b.full_attn_interval,
        .head_dim = b.head_dim,
        .hidden = b.hidden,
        .layer_types = layer_types.items,
        .conv = b.conv_kernel,
        .dk = b.dk,
        .nk = b.nk,
        .nv = b.nv,
        .dv = b.dv,
        .ctx = b.ctx_len,
        .expert_ffn = b.expert_ffn,
        .n_heads = b.n_heads,
        .n_experts = b.n_experts,
        .topk = b.n_experts_used,
        .n_layers = b.n_layers,
        .n_kv = b.n_kv_heads,
        .rotary = rotary,
        .eps = b.rms_eps,
        .shared_ffn = b.shared_ffn,
        .tied = b.tied,
        .vocab = b.vocab,
    });
    defer allocator.free(trunk);
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp","tie_word_embeddings":{[tied]},
        \\"eos_token_id":{[eos_id]d},"image_token_id":{[image]d},"qwen4_norm_convention":"folded",
        \\"quantization":{{"group_size":32,"bits":4,"mode":"affine"}},"text_config":{{{[trunk]s},{[ple]s}}}}}
    , .{ .tied = b.tied, .eos_id = b.eos_id, .image = hp.image_token, .trunk = trunk, .ple = ple });
}

/// The indexer's two halves, joined into `self_attn.indexer.index_qk_proj` by `joinIndexer`.
pub const idx_q = "self_attn.indexer.index_q_proj.weight";
pub const idx_k = "self_attn.indexer.index_k_proj.weight";

/// HF name + load transform for a GGUF tensor, null when the tensor is unknown.
pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "output_hc_norm.weight", prefix ++ "hyper_connection_mixer.hc_norm.weight" },
        .{ "output_hc_up.weight", prefix ++ "hyper_connection_mixer.input_mix_weight_up.weight" },
        .{ "output_hc_down.weight", prefix ++ "hyper_connection_mixer.input_mix_weight_down.weight" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return .{ .name = g[1], .transform = .none };
    if (std.mem.eql(u8, name, table_name)) return arch.skip;

    const blk = arch.splitLayer(name, hp.base.n_layers) orelse return qwen35.mapTensor(buf, name, hp.base);
    const table = [_]struct { []const u8, []const u8, arch.Transform }{
        .{ "hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight", .none },
        .{ "hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight", .none },
        .{ "hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight", .none },
        .{ "hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight", .none },
        .{ "hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight", .none },
        .{ "hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight", .none },
        .{ "hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight", .none },
        .{ "hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight", .none },
        .{ "ple_conv1d.weight", "ple.conv1d.weight", .trailing_axis },
        .{ "ple_key.weight", "ple.key_proj.weight", .none },
        .{ "ple_value.weight", "ple.value_proj.weight", .none },
        .{ "ple_norm_conv.weight", "ple.norm_conv.weight", .none },
        .{ "ple_norm_key.weight", "ple.norm_key.weight", .none },
        .{ "ple_norm_query.weight", "ple.norm_query.weight", .none },
        .{ "indexer.q_proj.weight", idx_q, .none },
        .{ "indexer.k_proj.weight", idx_k, .none },
        .{ "indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight", .none },
        .{ "indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight", .none },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, prefix ++ "layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = row[2] };
    };
    return qwen35.mapTensor(buf, name, hp.base);
}

/// The draft head's prefix in our packs.
pub const mtp_prefix = "language_model.mtp.";
/// The head's [embedding | hidden] combiner, split into `fc_embedding` and `fc_hidden` by the loader.
pub const mtp_eh_proj = mtp_prefix ++ "eh_proj.weight";

/// HF name + transform for a tensor of an MTP sidecar (`mtp-*.gguf`, the head as the block after the trunk's
/// `n_layers`), null when unknown. A shared head borrows the trunk's embedding and output.
pub fn mapMtpTensor(buf: []u8, name: []const u8, base: qwen35.Hparams) ?arch.Mapped {
    // A self-contained head's own copies of the trunk's embedding and output.
    if (std.mem.eql(u8, name, "token_embd.weight") or std.mem.eql(u8, name, "output.weight")) return arch.skip;
    const blk = arch.splitLayer(name, base.n_layers + 1) orelse return null;
    if (blk.layer != base.n_layers) return null;
    const nextn = [_]struct { []const u8, []const u8 }{
        .{ "nextn.eh_proj.weight", "eh_proj.weight" },
        .{ "nextn.enorm.weight", "pre_fc_norm_embedding.weight" },
        .{ "nextn.hnorm.weight", "pre_fc_norm_hidden.weight" },
        .{ "nextn.hc_head_norm.weight", "hyper_connection_mixer.hc_norm.weight" },
        .{ "nextn.hc_head_up.weight", "hyper_connection_mixer.input_mix_weight_up.weight" },
        .{ "nextn.hc_head_down.weight", "hyper_connection_mixer.input_mix_weight_down.weight" },
    };
    for (nextn) |n| if (std.mem.eql(u8, blk.leaf, n[0])) {
        return .{ .name = std.fmt.bufPrint(buf, mtp_prefix ++ "{s}", .{n[1]}) catch return null, .transform = .none };
    };
    // The head's decoder layer reads like a trunk layer: map it as layer 0 and move it under the head.
    var hp = std.mem.zeroes(Hparams);
    hp.base = base;
    var tmp: [160]u8 = undefined;
    var inner: [160]u8 = undefined;
    const as_trunk = std.fmt.bufPrint(&tmp, "blk.0.{s}", .{blk.leaf}) catch return null;
    const m = mapTensor(&inner, as_trunk, hp) orelse return null;
    const trunk_layer = prefix ++ "layers.0.";
    if (!std.mem.startsWith(u8, m.name, trunk_layer)) return null;
    return .{ .name = std.fmt.bufPrint(buf, mtp_prefix ++ "layers.0.{s}", .{m.name[trunk_layer.len..]}) catch return null, .transform = m.transform };
}

test "mapMtpTensor: the sidecar's block 48 is the head, its decoder layer moves under language_model.mtp" {
    var base = std.mem.zeroes(qwen35.Hparams);
    base.n_layers = 48;
    base.nk = 16;
    base.nv = 48;
    var buf: [160]u8 = undefined;
    try std.testing.expectEqualStrings(mtp_eh_proj, mapMtpTensor(&buf, "blk.48.nextn.eh_proj.weight", base).?.name);
    try std.testing.expectEqualStrings("language_model.mtp.hyper_connection_mixer.input_mix_weight_up.weight", mapMtpTensor(&buf, "blk.48.nextn.hc_head_up.weight", base).?.name);
    try std.testing.expectEqualStrings("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight", mapMtpTensor(&buf, "blk.48.ffn_gate_exps.weight", base).?.name);
    try std.testing.expectEqualStrings("language_model.mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", mapMtpTensor(&buf, "blk.48.hc_attn_inject.weight", base).?.name);
    try std.testing.expect(mapMtpTensor(&buf, "blk.48.ffn_gate_inp_shexp.weight", base).?.transform == .shared_gate);
    try std.testing.expectEqualStrings("language_model.mtp.layers.0." ++ idx_q, mapMtpTensor(&buf, "blk.48.indexer.q_proj.weight", base).?.name);
    try std.testing.expect(mapMtpTensor(&buf, "blk.47.attn_q.weight", base) == null);
    try std.testing.expectEqual(@as(usize, 0), mapMtpTensor(&buf, "token_embd.weight", base).?.name.len);
    try std.testing.expect(mapMtpTensor(&buf, "output_norm.weight", base) == null);
}

test "mapTensor: hyper connections, PLE, the split indexer and the trunk's qwen35 leaves" {
    var hp = std.mem.zeroes(Hparams);
    hp.base.n_layers = 4;
    hp.base.nk = 16;
    hp.base.nv = 48;
    hp.base.dk = 128;
    hp.base.dv = 128;
    var buf: [160]u8 = undefined;
    try std.testing.expectEqualStrings("language_model.model.layers.3.mlp_hyper_connection.block_inject_weight.weight", mapTensor(&buf, "blk.3.hc_ffn_inject.weight", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.1.ple_conv1d.weight", hp).?.transform == .trailing_axis);
    try std.testing.expectEqualStrings("language_model.model.layers.3." ++ idx_k, mapTensor(&buf, "blk.3.indexer.k_proj.weight", hp).?.name);
    try std.testing.expectEqualStrings("language_model.model.hyper_connection_mixer.input_mix_weight_down.weight", mapTensor(&buf, "output_hc_down.weight", hp).?.name);
    try std.testing.expectEqual(@as(usize, 0), mapTensor(&buf, table_name, hp).?.name.len);
    // The trunk keeps qwen35's value-head un-tiling (nv = 3 nk).
    try std.testing.expect(mapTensor(&buf, "blk.0.ssm_out.weight", hp).?.transform == .tiled_input);
    try std.testing.expectEqualStrings("language_model.lm_head.weight", mapTensor(&buf, "output.weight", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.9.hc_ffn_up.weight", hp) == null);
    try std.testing.expect(mapTensor(&buf, "blk.0.hc_ffn_gate.weight", hp) == null);
}

test "configJson parses with the qwen4_exp geometry" {
    var hp = std.mem.zeroes(Hparams);
    hp.base.n_layers = 8;
    hp.base.full_attn_interval = 4;
    hp.base.head_dim = 256;
    hp.base.rope_dims = 64;
    hp.base.rms_eps = 1e-6;
    hp.base.rope_theta = 1e7;
    hp.ngram_size = 3;
    hp.heads_per_ngram = 8;
    hp.ple_row = 160;
    hp.ple_layer = 1;
    hp.ngram_base = 20000003;
    hp.ngram_divisor = 512;
    const json = try configJson(std.testing.allocator, hp);
    defer std.testing.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("qwen4_exp", parsed.value.object.get("model_type").?.string);
    const tc = parsed.value.object.get("text_config").?.object;
    try std.testing.expectEqual(@as(i64, 2560), tc.get("ple_embed_dim").?.integer);
    try std.testing.expectEqual(@as(i64, 2), tc.get("ple_layer_ids").?.array.items[0].integer);
    try std.testing.expectEqualStrings("full_attention", tc.get("layer_types").?.array.items[7].string);
}
