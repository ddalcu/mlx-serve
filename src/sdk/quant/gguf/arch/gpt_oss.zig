//! gpt-oss (GGUF arch "gpt-oss"): 20B / 120B MoE with attention sinks,
//! alternating sliding / full attention, biased projections and per-expert
//! biases, experts in MXFP4. Every tensor is stored as HF does, so loading
//! is a rename.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");

pub const arch_name = "gpt-oss";

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    ffn: u32,
    expert_ffn: u32,
    n_experts: u32,
    n_experts_used: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    sliding_window: u32,
    /// Every `swa_period`-th layer is full attention, the rest sliding.
    swa_period: u32,
    rope_theta: f64,
    rope_factor: f64,
    rope_original_ctx: u32,
    rms_eps: f64,
    ctx_len: u32,
    vocab: u32,
    eos: u32,

    pub fn read(f: *const gguf.File) !Hparams {
        return .{
            .n_layers = try arch.u32Key(f, "gpt-oss.block_count"),
            .hidden = try arch.u32Key(f, "gpt-oss.embedding_length"),
            .ffn = try arch.u32Key(f, "gpt-oss.feed_forward_length"),
            .expert_ffn = try arch.u32Key(f, "gpt-oss.expert_feed_forward_length"),
            .n_experts = try arch.u32Key(f, "gpt-oss.expert_count"),
            .n_experts_used = try arch.u32Key(f, "gpt-oss.expert_used_count"),
            .n_heads = try arch.u32Key(f, "gpt-oss.attention.head_count"),
            .n_kv_heads = try arch.u32Key(f, "gpt-oss.attention.head_count_kv"),
            .head_dim = try arch.u32Key(f, "gpt-oss.attention.key_length"),
            .sliding_window = try arch.u32Key(f, "gpt-oss.attention.sliding_window"),
            .swa_period = if (f.getInt("gpt-oss.attention.sliding_window_pattern") != null) try arch.u32Key(f, "gpt-oss.attention.sliding_window_pattern") else 2,
            .rope_theta = f.getFloat("gpt-oss.rope.freq_base") orelse return error.MissingMetadata,
            .rope_factor = f.getFloat("gpt-oss.rope.scaling.factor") orelse 1.0,
            .rope_original_ctx = if (f.getInt("gpt-oss.rope.scaling.original_context_length") != null) try arch.u32Key(f, "gpt-oss.rope.scaling.original_context_length") else 4096,
            .rms_eps = f.getFloat("gpt-oss.attention.layer_norm_rms_epsilon") orelse return error.MissingMetadata,
            .ctx_len = try arch.u32Key(f, "gpt-oss.context_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .eos = try arch.u32Key(f, "tokenizer.ggml.eos_token_id"),
        };
    }
};

pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var layer_types: std.ArrayList(u8) = .empty;
    defer layer_types.deinit(allocator);
    for (0..hp.n_layers) |i| {
        if (i > 0) try layer_types.append(allocator, ',');
        // llama.cpp's swa pattern: layer i slides unless i % period == period - 1.
        try layer_types.appendSlice(allocator, if (hp.swa_period > 0 and i % hp.swa_period == hp.swa_period - 1) "\"full_attention\"" else "\"sliding_attention\"");
    }
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss","attention_bias":true,"eos_token_id":{[eos]d},
        \\"experts_per_token":{[n_experts_used]d},"head_dim":{[head_dim]d},"hidden_act":"silu","hidden_size":{[hidden]d},
        \\"initial_context_length":{[rope_original_ctx]d},"intermediate_size":{[expert_ffn]d},"layer_types":[{[layer_types]s}],
        \\"max_position_embeddings":{[ctx_len]d},"num_attention_heads":{[n_heads]d},"num_experts_per_tok":{[n_experts_used]d},
        \\"num_hidden_layers":{[n_layers]d},"num_key_value_heads":{[n_kv_heads]d},"num_local_experts":{[n_experts]d},
        \\"rms_norm_eps":{[rms_eps]e},"rope_scaling":{{"beta_fast":32.0,"beta_slow":1.0,"factor":{[rope_factor]d:.1},
        \\"original_max_position_embeddings":{[rope_original_ctx]d},"rope_type":"yarn","truncate":false}},"rope_theta":{[rope_theta]d},
        \\"sliding_window":{[sliding_window]d},"swiglu_limit":7.0,"tie_word_embeddings":false,"vocab_size":{[vocab]d},
        \\"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}}}}
    , .{
        .eos = hp.eos,
        .n_experts_used = hp.n_experts_used,
        .head_dim = hp.head_dim,
        .hidden = hp.hidden,
        .rope_original_ctx = hp.rope_original_ctx,
        .expert_ffn = hp.expert_ffn,
        .layer_types = layer_types.items,
        .ctx_len = hp.ctx_len,
        .n_heads = hp.n_heads,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .n_experts = hp.n_experts,
        .rms_eps = hp.rms_eps,
        .rope_factor = hp.rope_factor,
        .rope_theta = hp.rope_theta,
        .sliding_window = hp.sliding_window,
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
        .{ "post_attention_norm.weight", "post_attention_layernorm.weight" },
        .{ "attn_q.weight", "self_attn.q_proj.weight" },
        .{ "attn_q.bias", "self_attn.q_proj.bias" },
        .{ "attn_k.weight", "self_attn.k_proj.weight" },
        .{ "attn_k.bias", "self_attn.k_proj.bias" },
        .{ "attn_v.weight", "self_attn.v_proj.weight" },
        .{ "attn_v.bias", "self_attn.v_proj.bias" },
        .{ "attn_output.weight", "self_attn.o_proj.weight" },
        .{ "attn_output.bias", "self_attn.o_proj.bias" },
        .{ "attn_sinks.weight", "self_attn.sinks" },
        .{ "ffn_gate_inp.weight", "mlp.router.weight" },
        .{ "ffn_gate_inp.bias", "mlp.router.bias" },
        .{ "ffn_gate_exps.weight", "mlp.experts.gate_proj.weight" },
        .{ "ffn_gate_exps.bias", "mlp.experts.gate_proj.bias" },
        .{ "ffn_up_exps.weight", "mlp.experts.up_proj.weight" },
        .{ "ffn_up_exps.bias", "mlp.experts.up_proj.bias" },
        .{ "ffn_down_exps.weight", "mlp.experts.down_proj.weight" },
        .{ "ffn_down_exps.bias", "mlp.experts.down_proj.bias" },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = .none };
    };
    return null;
}

test "mapTensor renames sinks, biases and expert banks" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 24;
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("model.layers.3.self_attn.sinks", mapTensor(&buf, "blk.3.attn_sinks.weight", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.0.mlp.experts.up_proj.bias", mapTensor(&buf, "blk.0.ffn_up_exps.bias", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.23.mlp.router.bias", mapTensor(&buf, "blk.23.ffn_gate_inp.bias", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.1.self_attn.o_proj.bias", mapTensor(&buf, "blk.1.attn_output.bias", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.24.attn_q.weight", hp) == null);
}

test "configJson alternates sliding and full attention and carries the yarn scaling" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 4;
    hp.swa_period = 2;
    hp.rope_factor = 32;
    hp.rope_original_ctx = 4096;
    hp.rms_eps = 1e-5;
    const a = std.testing.allocator;
    const json = try configJson(a, hp);
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    const lt = o.get("layer_types").?.array.items;
    try std.testing.expectEqualStrings("sliding_attention", lt[0].string);
    try std.testing.expectEqualStrings("full_attention", lt[1].string);
    try std.testing.expectEqualStrings("yarn", o.get("rope_scaling").?.object.get("rope_type").?.string);
    try std.testing.expectEqual(@as(f64, 32), o.get("rope_scaling").?.object.get("factor").?.float);
}
