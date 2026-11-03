//! Recursive Flint (GGUF arch "flint"): sliding RoPE layers and NoPE full layers with learned sinks, leading dense
//! layers, then sigmoid-routed experts beside an ungated shared expert. llama.cpp's converter only renames (NeoX
//! RoPE, so no Q/K permutation; plain RMSNorm weights; experts stacked in id order), so loading is a rename onto
//! the mlx-lm / RecursiveSI flint-int4 names.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");

pub const arch_name = "flint";

const MAX_LAYERS = 128;
const MAX_EOS = 8;

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    /// The leading dense layers' MLP width.
    ffn: u32,
    dense_layers: u32,
    n_experts: u32,
    n_experts_used: u32,
    expert_ffn: u32,
    shared_ffn: u32,
    route_norm: bool,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    sliding_window: u32,
    /// Per layer: sliding attention (else full), and RoPE (else NoPE).
    sliding: [MAX_LAYERS]bool,
    rope: [MAX_LAYERS]bool,
    rope_theta: f64,
    rms_eps: f64,
    ctx_len: u32,
    vocab: u32,
    bos: u32,
    /// Every token that ends a reply (the GGUF's end-of-generation list).
    eos: [MAX_EOS]u32,
    n_eos: u32,
    tied: bool,

    pub fn read(f: *const gguf.File) !Hparams {
        const n_layers = try arch.u32Key(f, "flint.block_count");
        if (n_layers == 0 or n_layers > MAX_LAYERS) return error.BadMetadata;
        // llama.cpp's gating func 2 = sigmoid, the only routing Flint has.
        if (try arch.u32Key(f, "flint.expert_gating_func") != 2) return error.BadMetadata;
        var hp = Hparams{
            .n_layers = n_layers,
            .hidden = try arch.u32Key(f, "flint.embedding_length"),
            .ffn = try arch.u32Key(f, "flint.feed_forward_length"),
            .dense_layers = try arch.u32Key(f, "flint.leading_dense_block_count"),
            .n_experts = try arch.u32Key(f, "flint.expert_count"),
            .n_experts_used = try arch.u32Key(f, "flint.expert_used_count"),
            .expert_ffn = try arch.u32Key(f, "flint.expert_feed_forward_length"),
            .shared_ffn = try arch.u32Key(f, "flint.expert_shared_feed_forward_length"),
            .route_norm = f.getBool("flint.expert_weights_norm") orelse return error.MissingMetadata,
            .n_heads = try arch.u32Key(f, "flint.attention.head_count"),
            .n_kv_heads = try arch.u32Key(f, "flint.attention.head_count_kv"),
            .head_dim = try arch.u32Key(f, "flint.attention.key_length"),
            .sliding_window = try arch.u32Key(f, "flint.attention.sliding_window"),
            .sliding = @splat(false),
            .rope = @splat(false),
            .rope_theta = f.getFloat("flint.rope.freq_base") orelse return error.MissingMetadata,
            .rms_eps = f.getFloat("flint.attention.layer_norm_rms_epsilon") orelse return error.MissingMetadata,
            .ctx_len = try arch.u32Key(f, "flint.context_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .bos = try arch.u32Key(f, "tokenizer.ggml.bos_token_id"),
            .eos = undefined,
            .n_eos = 1,
            .tied = f.tensors.get("output.weight") == null,
        };
        if (hp.dense_layers > n_layers) return error.BadMetadata;
        for ([_]struct { []const u8, []bool }{
            .{ "flint.attention.sliding_window_pattern", &hp.sliding },
            .{ "flint.attention.rope_pattern", &hp.rope },
        }) |p| {
            const arr = f.getArray(p[0]) orelse return error.MissingMetadata;
            if (arr.len != n_layers or arr.elem_type != .bool) return error.BadMetadata;
            for (p[1][0..n_layers], 0..) |*v, i| v.* = arr.boolean(i);
        }
        if (f.getArray("tokenizer.ggml.eog_token_ids")) |arr| {
            if (arr.len == 0 or arr.len > MAX_EOS) return error.BadMetadata;
            hp.n_eos = @intCast(arr.len);
            for (hp.eos[0..arr.len], 0..) |*id, i| id.* = std.math.cast(u32, arr.int(i)) orelse return error.BadMetadata;
        } else hp.eos[0] = try arch.u32Key(f, "tokenizer.ggml.eos_token_id");
        return hp;
    }
};

fn sep(list: []const u8) []const u8 {
    return if (list.len > 0) "," else "";
}

/// The config.json of RecursiveSI/flint-int4, the keys the flint plugin and the engine read.
pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var layer_types: std.ArrayList(u8) = .empty;
    defer layer_types.deinit(allocator);
    var nope: std.ArrayList(u8) = .empty;
    defer nope.deinit(allocator);
    var dense: std.ArrayList(u8) = .empty;
    defer dense.deinit(allocator);
    var eos: std.ArrayList(u8) = .empty;
    defer eos.deinit(allocator);
    for (0..hp.n_layers) |i| {
        try layer_types.print(allocator, "{s}\"{s}\"", .{ sep(layer_types.items), if (hp.sliding[i]) "sliding_attention" else "full_attention" });
        if (!hp.rope[i]) try nope.print(allocator, "{s}{d}", .{ sep(nope.items), i });
    }
    for (0..hp.dense_layers) |i| try dense.print(allocator, "{s}{d}", .{ sep(dense.items), i });
    for (hp.eos[0..hp.n_eos]) |id| try eos.print(allocator, "{s}{d}", .{ sep(eos.items), id });
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["FlintForCausalLM"],"model_type":"flint","attention_bias":false,"bos_token_id":{[bos]d},
        \\"eos_token_id":[{[eos]s}],"head_dim":{[head_dim]d},"hidden_act":"silu","hidden_size":{[hidden]d},
        \\"intermediate_size":{[ffn]d},"layer_types":[{[layer_types]s}],"max_position_embeddings":{[ctx_len]d},
        \\"mlp_only_layers":[{[dense]s}],"moe_intermediate_size":{[expert_ffn]d},"no_rope_layer_indices":[{[nope]s}],
        \\"norm_topk_prob":{[route_norm]},"num_attention_heads":{[n_heads]d},"num_experts":{[n_experts]d},
        \\"num_experts_per_tok":{[n_experts_used]d},"num_hidden_layers":{[n_layers]d},"num_key_value_heads":{[n_kv_heads]d},
        \\"rms_norm_eps":{[rms_eps]e},"rope_parameters":{{"rope_theta":{[rope_theta]d},"rope_type":"default"}},
        \\"rope_theta":{[rope_theta]d},"router_score_function":"sigmoid","shared_expert_intermediate_size":{[shared_ffn]d},
        \\"sliding_window":{[sliding_window]d},"softmax_type":"learnable","tie_word_embeddings":{[tied]},
        \\"use_sliding_window":true,"vocab_size":{[vocab]d},"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}}}}
    , .{
        .bos = hp.bos,
        .eos = eos.items,
        .head_dim = hp.head_dim,
        .hidden = hp.hidden,
        .ffn = hp.ffn,
        .layer_types = layer_types.items,
        .ctx_len = hp.ctx_len,
        .dense = dense.items,
        .expert_ffn = hp.expert_ffn,
        .nope = nope.items,
        .route_norm = hp.route_norm,
        .n_heads = hp.n_heads,
        .n_experts = hp.n_experts,
        .n_experts_used = hp.n_experts_used,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .rms_eps = hp.rms_eps,
        .rope_theta = hp.rope_theta,
        .shared_ffn = hp.shared_ffn,
        .sliding_window = hp.sliding_window,
        .tied = hp.tied,
        .vocab = hp.vocab,
    });
}

pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "token_embd.weight", "model.embed_tokens.weight" },
        .{ "output_norm.weight", "model.norm.weight" },
        .{ "output.weight", "lm_head.weight" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return .{ .name = g[1], .transform = .none };
    const blk = arch.splitLayer(name, hp.n_layers) orelse return null;
    const table = [_]struct { []const u8, []const u8 }{
        .{ "attn_norm.weight", "input_layernorm.weight" },
        .{ "attn_q.weight", "self_attn.q_proj.weight" },
        .{ "attn_k.weight", "self_attn.k_proj.weight" },
        .{ "attn_v.weight", "self_attn.v_proj.weight" },
        .{ "attn_output.weight", "self_attn.o_proj.weight" },
        .{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
        .{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
        .{ "attn_sinks.weight", "self_attn.softmax_offset" },
        // The pre-FFN norm: HF's post_attention_layernorm.
        .{ "ffn_norm.weight", "post_attention_layernorm.weight" },
        .{ "ffn_gate.weight", "mlp.gate_proj.weight" },
        .{ "ffn_up.weight", "mlp.up_proj.weight" },
        .{ "ffn_down.weight", "mlp.down_proj.weight" },
        .{ "ffn_gate_inp.weight", "mlp.gate.weight" },
        // Only written for a checkpoint with `router_expert_bias`.
        .{ "exp_probs_b.bias", "mlp.gate.expert_bias" },
        .{ "ffn_gate_exps.weight", "mlp.switch_mlp.gate_proj.weight" },
        .{ "ffn_up_exps.weight", "mlp.switch_mlp.up_proj.weight" },
        .{ "ffn_down_exps.weight", "mlp.switch_mlp.down_proj.weight" },
        .{ "ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight" },
        .{ "ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight" },
        .{ "ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight" },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = .none };
    };
    return null;
}

test "mapTensor renames every flint leaf onto flint-int4's names" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 49;
    var buf: [128]u8 = undefined;
    const want = [_]struct { []const u8, []const u8 }{
        .{ "token_embd.weight", "model.embed_tokens.weight" },
        .{ "output_norm.weight", "model.norm.weight" },
        .{ "output.weight", "lm_head.weight" },
        .{ "blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight" },
        .{ "blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight" },
        .{ "blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight" },
        .{ "blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight" },
        .{ "blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight" },
        .{ "blk.0.attn_q_norm.weight", "model.layers.0.self_attn.q_norm.weight" },
        .{ "blk.0.attn_k_norm.weight", "model.layers.0.self_attn.k_norm.weight" },
        .{ "blk.0.attn_sinks.weight", "model.layers.0.self_attn.softmax_offset" },
        .{ "blk.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight" },
        .{ "blk.1.ffn_gate.weight", "model.layers.1.mlp.gate_proj.weight" },
        .{ "blk.1.ffn_up.weight", "model.layers.1.mlp.up_proj.weight" },
        .{ "blk.1.ffn_down.weight", "model.layers.1.mlp.down_proj.weight" },
        .{ "blk.3.ffn_gate_inp.weight", "model.layers.3.mlp.gate.weight" },
        .{ "blk.3.exp_probs_b.bias", "model.layers.3.mlp.gate.expert_bias" },
        .{ "blk.3.ffn_gate_exps.weight", "model.layers.3.mlp.switch_mlp.gate_proj.weight" },
        .{ "blk.3.ffn_up_exps.weight", "model.layers.3.mlp.switch_mlp.up_proj.weight" },
        .{ "blk.3.ffn_down_exps.weight", "model.layers.3.mlp.switch_mlp.down_proj.weight" },
        .{ "blk.48.ffn_gate_shexp.weight", "model.layers.48.mlp.shared_expert.gate_proj.weight" },
        .{ "blk.48.ffn_up_shexp.weight", "model.layers.48.mlp.shared_expert.up_proj.weight" },
        .{ "blk.48.ffn_down_shexp.weight", "model.layers.48.mlp.shared_expert.down_proj.weight" },
    };
    for (want) |w| {
        const m = mapTensor(&buf, w[0], hp).?;
        try std.testing.expectEqualStrings(w[1], m.name);
        try std.testing.expect(m.transform == .none);
    }
    try std.testing.expect(mapTensor(&buf, "blk.49.attn_q.weight", hp) == null);
    try std.testing.expect(mapTensor(&buf, "blk.0.attn_qkv.weight", hp) == null);
    try std.testing.expect(mapTensor(&buf, "rope_freqs.weight", hp) == null);
}

test "configJson carries the sliding and NoPE layers, the dense layers and every end-of-generation id" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 6;
    hp.dense_layers = 2;
    hp.sliding = @splat(true);
    hp.rope = @splat(true);
    for ([_]usize{ 0, 1, 5 }) |i| {
        hp.sliding[i] = false;
        hp.rope[i] = false;
    }
    hp.route_norm = true;
    hp.sliding_window = 513;
    hp.rope_theta = 10000;
    hp.rms_eps = 9.999999974752427e-7;
    hp.eos[0..4].* = .{ 150001, 150003, 150005, 150018 };
    hp.n_eos = 4;
    const a = std.testing.allocator;
    const json = try configJson(a, hp);
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("flint", o.get("model_type").?.string);
    const lt = o.get("layer_types").?.array.items;
    try std.testing.expectEqual(@as(usize, 6), lt.len);
    try std.testing.expectEqualStrings("full_attention", lt[1].string);
    try std.testing.expectEqualStrings("sliding_attention", lt[2].string);
    const nope = o.get("no_rope_layer_indices").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), nope.len);
    try std.testing.expectEqual(@as(i64, 5), nope[2].integer);
    try std.testing.expectEqual(@as(usize, 2), o.get("mlp_only_layers").?.array.items.len);
    try std.testing.expectEqual(@as(i64, 150018), o.get("eos_token_id").?.array.items[3].integer);
    try std.testing.expect(o.get("norm_topk_prob").?.bool);
    try std.testing.expect(!o.get("tie_word_embeddings").?.bool);
    try std.testing.expectEqual(@as(i64, 513), o.get("sliding_window").?.integer);
    try std.testing.expectEqualStrings("gguf", o.get("quantization").?.object.get("mode").?.string);
}
