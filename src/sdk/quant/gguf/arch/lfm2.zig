//! LFM2 (GGUF arch "lfm2") and LFM2-MoE ("lfm2moe"): short-conv layers mixed
//! with GQA attention layers, told apart by the per-layer KV head count (0 =
//! conv). The GGUF stores the tensors as HF does, so loading is a rename plus
//! the conv kernel getting HF's trailing axis.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");

const MAX_LAYERS = 128;

pub fn handles(name: []const u8) bool {
    return std.mem.eql(u8, name, "lfm2") or std.mem.eql(u8, name, "lfm2moe");
}

pub const Hparams = struct {
    moe: bool,
    n_layers: u32,
    hidden: u32,
    ffn: u32,
    n_heads: u32,
    n_kv_heads: u32,
    /// True for the attention layers.
    attention: [MAX_LAYERS]bool,
    conv_cache: u32,
    rope_theta: f64,
    rms_eps: f64,
    ctx_len: u32,
    vocab: u32,
    eos: u32,
    /// MoE only.
    n_experts: u32,
    n_experts_used: u32,
    expert_ffn: u32,
    dense_layers: u32,

    pub fn read(f: *const gguf.File, name: []const u8) !Hparams {
        const k = Keys{ .f = f, .arch = name };
        const n_layers = try k.int("block_count");
        if (n_layers == 0 or n_layers > MAX_LAYERS) return error.BadMetadata;
        var attention: [MAX_LAYERS]bool = @splat(false);
        var n_kv_heads: u32 = 0;
        var kbuf: [64]u8 = undefined;
        const kv_key = try std.fmt.bufPrint(&kbuf, "{s}.attention.head_count_kv", .{name});
        if (f.getArray(kv_key)) |arr| {
            if (arr.len != n_layers) return error.BadMetadata;
            for (0..n_layers) |i| {
                const kv = std.math.cast(u32, arr.int(i)) orelse return error.BadMetadata;
                attention[i] = kv > 0;
                if (kv > 0) n_kv_heads = kv;
            }
        } else {
            n_kv_heads = try k.int("attention.head_count_kv");
            attention = @splat(n_kv_heads > 0);
        }
        const moe = std.mem.eql(u8, name, "lfm2moe");
        return .{
            .moe = moe,
            .n_layers = n_layers,
            .hidden = try k.int("embedding_length"),
            .ffn = try k.int("feed_forward_length"),
            .n_heads = try k.int("attention.head_count"),
            .n_kv_heads = n_kv_heads,
            .attention = attention,
            .conv_cache = try k.int("shortconv.l_cache"),
            .rope_theta = try k.float("rope.freq_base"),
            .rms_eps = try k.float("attention.layer_norm_rms_epsilon"),
            .ctx_len = try k.int("context_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .eos = try arch.u32Key(f, "tokenizer.ggml.eos_token_id"),
            .n_experts = if (moe) try k.int("expert_count") else 0,
            .n_experts_used = if (moe) try k.int("expert_used_count") else 0,
            .expert_ffn = if (moe) try k.int("expert_feed_forward_length") else 0,
            .dense_layers = if (moe) try k.int("leading_dense_block_count") else 0,
        };
    }
};

const Keys = struct {
    f: *const gguf.File,
    arch: []const u8,

    fn int(self: Keys, suffix: []const u8) !u32 {
        var buf: [96]u8 = undefined;
        return arch.u32Key(self.f, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ self.arch, suffix }));
    }

    fn float(self: Keys, suffix: []const u8) !f64 {
        var buf: [96]u8 = undefined;
        return self.f.getFloat(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ self.arch, suffix })) orelse error.MissingMetadata;
    }
};

pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var layer_types: std.ArrayList(u8) = .empty;
    defer layer_types.deinit(allocator);
    for (hp.attention[0..hp.n_layers], 0..) |attn, i| {
        if (i > 0) try layer_types.append(allocator, ',');
        try layer_types.appendSlice(allocator, if (attn) "\"full_attention\"" else "\"conv\"");
    }
    var moe_buf: [256]u8 = undefined;
    const moe = if (hp.moe) try std.fmt.bufPrint(&moe_buf,
        \\,"num_experts":{d},"num_experts_per_tok":{d},"moe_intermediate_size":{d},"num_dense_layers":{d},"norm_topk_prob":true,"routed_scaling_factor":1.0,"use_expert_bias":true
    , .{ hp.n_experts, hp.n_experts_used, hp.expert_ffn, hp.dense_layers }) else "";
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["{[architecture]s}"],"model_type":"{[model_type]s}","bos_token_id":1,"eos_token_id":{[eos]d},
        \\"conv_L_cache":{[conv_cache]d},"conv_bias":false,"hidden_size":{[hidden]d},"intermediate_size":{[ffn]d},"block_ff_dim":{[ffn]d},
        \\"block_auto_adjust_ff_dim":false,"layer_types":[{[layer_types]s}],"max_position_embeddings":{[ctx_len]d},"norm_eps":{[rms_eps]e},
        \\"num_attention_heads":{[n_heads]d},"num_hidden_layers":{[n_layers]d},"num_key_value_heads":{[n_kv_heads]d},"rope_theta":{[rope_theta]d},
        \\"tie_embedding":true,"vocab_size":{[vocab]d}{[moe]s},"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}}}}
    , .{
        .architecture = if (hp.moe) "Lfm2MoeForCausalLM" else "Lfm2ForCausalLM",
        .model_type = if (hp.moe) "lfm2_moe" else "lfm2",
        .eos = hp.eos,
        .conv_cache = hp.conv_cache,
        .hidden = hp.hidden,
        .ffn = hp.ffn,
        .layer_types = layer_types.items,
        .ctx_len = hp.ctx_len,
        .rms_eps = hp.rms_eps,
        .n_heads = hp.n_heads,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .rope_theta = hp.rope_theta,
        .vocab = hp.vocab,
        .moe = moe,
    });
}

pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "token_embd.weight", "model.embed_tokens.weight" },
        .{ "token_embd_norm.weight", "model.embedding_norm.weight" },
        .{ "output.weight", "lm_head.weight" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return .{ .name = g[1], .transform = .none };
    const blk = arch.splitLayer(name, hp.n_layers) orelse return null;
    const table = [_]struct { []const u8, []const u8, arch.Transform }{
        .{ "attn_norm.weight", "operator_norm.weight", .none },
        // [channels, kernel] in GGUF, [channels, kernel, 1] in HF.
        .{ "shortconv.conv.weight", "conv.conv.weight", .trailing_axis },
        .{ "shortconv.in_proj.weight", "conv.in_proj.weight", .none },
        .{ "shortconv.out_proj.weight", "conv.out_proj.weight", .none },
        .{ "attn_q.weight", "self_attn.q_proj.weight", .none },
        .{ "attn_k.weight", "self_attn.k_proj.weight", .none },
        .{ "attn_v.weight", "self_attn.v_proj.weight", .none },
        .{ "attn_output.weight", "self_attn.out_proj.weight", .none },
        .{ "attn_q_norm.weight", "self_attn.q_layernorm.weight", .none },
        .{ "attn_k_norm.weight", "self_attn.k_layernorm.weight", .none },
        .{ "ffn_norm.weight", "ffn_norm.weight", .none },
        .{ "ffn_gate.weight", "feed_forward.w1.weight", .none },
        .{ "ffn_up.weight", "feed_forward.w3.weight", .none },
        .{ "ffn_down.weight", "feed_forward.w2.weight", .none },
        .{ "ffn_gate_inp.weight", "feed_forward.gate.weight", .none },
        .{ "ffn_gate_exps.weight", "feed_forward.switch_mlp.gate_proj.weight", .none },
        .{ "ffn_up_exps.weight", "feed_forward.switch_mlp.up_proj.weight", .none },
        .{ "ffn_down_exps.weight", "feed_forward.switch_mlp.down_proj.weight", .none },
        .{ "exp_probs_b.bias", "feed_forward.expert_bias", .none },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = row[2] };
    };
    return null;
}

test "mapTensor renames conv, attention and expert tensors" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 4;
    var buf: [128]u8 = undefined;
    const conv = mapTensor(&buf, "blk.0.shortconv.conv.weight", hp).?;
    try std.testing.expectEqualStrings("model.layers.0.conv.conv.weight", conv.name);
    try std.testing.expectEqual(arch.Transform.trailing_axis, conv.transform);
    try std.testing.expectEqualStrings("model.layers.2.self_attn.k_layernorm.weight", mapTensor(&buf, "blk.2.attn_k_norm.weight", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.3.feed_forward.switch_mlp.down_proj.weight", mapTensor(&buf, "blk.3.ffn_down_exps.weight", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.3.feed_forward.expert_bias", mapTensor(&buf, "blk.3.exp_probs_b.bias", hp).?.name);
    try std.testing.expectEqualStrings("model.embedding_norm.weight", mapTensor(&buf, "token_embd_norm.weight", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.4.attn_q.weight", hp) == null);
}

test "configJson carries the layer types and the MoE block only for lfm2moe" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 3;
    hp.attention[1] = true;
    hp.rms_eps = 1e-5;
    hp.rope_theta = 1e6;
    const a = std.testing.allocator;
    const dense = try configJson(a, hp);
    defer a.free(dense);
    const dp = try std.json.parseFromSlice(std.json.Value, a, dense, .{});
    defer dp.deinit();
    try std.testing.expectEqualStrings("lfm2", dp.value.object.get("model_type").?.string);
    try std.testing.expectEqualStrings("full_attention", dp.value.object.get("layer_types").?.array.items[1].string);
    try std.testing.expect(dp.value.object.get("num_experts") == null);

    hp.moe = true;
    hp.n_experts = 32;
    hp.n_experts_used = 4;
    hp.expert_ffn = 1792;
    hp.dense_layers = 2;
    const moe = try configJson(a, hp);
    defer a.free(moe);
    const mp = try std.json.parseFromSlice(std.json.Value, a, moe, .{});
    defer mp.deinit();
    try std.testing.expectEqualStrings("lfm2_moe", mp.value.object.get("model_type").?.string);
    try std.testing.expectEqual(@as(i64, 4), mp.value.object.get("num_experts_per_tok").?.integer);
    try std.testing.expectEqual(@as(i64, 2), mp.value.object.get("num_dense_layers").?.integer);
}
