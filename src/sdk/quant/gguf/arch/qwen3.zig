//! Qwen3, dense (GGUF arch "qwen3") and MoE ("qwen3moe": routed experts, softmax top-k, no shared expert):
//! hparams, the config.json mlx-lm's `qwen3` / `qwen3_moe` conversions carry, and the GGUF -> HF tensor map.
//! llama.cpp ropes Qwen in NEOX order, so Q/K rows are stored as HF does and loading is a rename.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");
const Keys = @import("qwen35.zig").Keys;

pub fn handles(arch_name: []const u8) bool {
    return std.mem.eql(u8, arch_name, "qwen3") or std.mem.eql(u8, arch_name, "qwen3moe");
}

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    ffn: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_theta: f64,
    rms_eps: f64,
    ctx_len: u32,
    vocab: u32,
    eos_id: u32,
    tied: bool,
    /// Routed experts, 0 = dense; then experts per token and the expert MLP width.
    n_experts: u32,
    n_experts_used: u32,
    expert_ffn: u32,

    pub fn read(f: *const gguf.File, arch_name: []const u8) !Hparams {
        const k = Keys{ .f = f, .arch_name = arch_name };
        const n_experts = k.int("expert_count") catch 0;
        return .{
            .n_layers = try k.int("block_count"),
            .hidden = try k.int("embedding_length"),
            .ffn = try k.int("feed_forward_length"),
            .n_heads = try k.int("attention.head_count"),
            .n_kv_heads = try k.int("attention.head_count_kv"),
            .head_dim = try k.int("attention.key_length"),
            .rope_theta = try k.float("rope.freq_base"),
            .rms_eps = try k.float("attention.layer_norm_rms_epsilon"),
            .ctx_len = try k.int("context_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .eos_id = try arch.u32Key(f, "tokenizer.ggml.eos_token_id"),
            .tied = f.tensors.get("output.weight") == null,
            .n_experts = n_experts,
            .n_experts_used = if (n_experts == 0) 0 else try k.int("expert_used_count"),
            .expert_ffn = if (n_experts == 0) 0 else try k.int("expert_feed_forward_length"),
        };
    }
};

pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    const moe = hp.n_experts > 0;
    var moe_buf: [200]u8 = undefined;
    const moe_fields = if (moe) std.fmt.bufPrint(&moe_buf,
        \\"num_experts":{d},"num_experts_per_tok":{d},"moe_intermediate_size":{d},"norm_topk_prob":true,"decoder_sparse_step":1,"mlp_only_layers":[],
    , .{ hp.n_experts, hp.n_experts_used, hp.expert_ffn }) catch unreachable else "";
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["{[architecture]s}"],"model_type":"{[model_type]s}","attention_bias":false,"eos_token_id":{[eos_id]d},
        \\"head_dim":{[head_dim]d},"hidden_act":"silu","hidden_size":{[hidden]d},"intermediate_size":{[ffn]d},{[moe_fields]s}
        \\"max_position_embeddings":{[ctx_len]d},"num_attention_heads":{[n_heads]d},"num_hidden_layers":{[n_layers]d},
        \\"num_key_value_heads":{[n_kv_heads]d},"rms_norm_eps":{[rms_eps]e},"rope_theta":{[rope_theta]d},
        \\"tie_word_embeddings":{[tied]},"vocab_size":{[vocab]d},"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}}}}
    , .{
        .architecture = if (moe) "Qwen3MoeForCausalLM" else "Qwen3ForCausalLM",
        .model_type = if (moe) "qwen3_moe" else "qwen3",
        .eos_id = hp.eos_id,
        .head_dim = hp.head_dim,
        .hidden = hp.hidden,
        .ffn = hp.ffn,
        .moe_fields = moe_fields,
        .ctx_len = hp.ctx_len,
        .n_heads = hp.n_heads,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .rms_eps = hp.rms_eps,
        .rope_theta = hp.rope_theta,
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
        .{ "ffn_norm.weight", "post_attention_layernorm.weight" },
        .{ "attn_q.weight", "self_attn.q_proj.weight" },
        .{ "attn_k.weight", "self_attn.k_proj.weight" },
        .{ "attn_v.weight", "self_attn.v_proj.weight" },
        .{ "attn_output.weight", "self_attn.o_proj.weight" },
        .{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
        .{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
        .{ "ffn_gate.weight", "mlp.gate_proj.weight" },
        .{ "ffn_up.weight", "mlp.up_proj.weight" },
        .{ "ffn_down.weight", "mlp.down_proj.weight" },
        .{ "ffn_gate_inp.weight", "mlp.gate.weight" },
        .{ "ffn_gate_exps.weight", "mlp.switch_mlp.gate_proj.weight" },
        .{ "ffn_up_exps.weight", "mlp.switch_mlp.up_proj.weight" },
        .{ "ffn_down_exps.weight", "mlp.switch_mlp.down_proj.weight" },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = .none };
    };
    return null;
}

test "mapTensor renames the dense and MoE leaves" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 48;
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("model.layers.47.post_attention_layernorm.weight", mapTensor(&buf, "blk.47.ffn_norm.weight", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.0.self_attn.k_norm.weight", mapTensor(&buf, "blk.0.attn_k_norm.weight", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.3.mlp.gate.weight", mapTensor(&buf, "blk.3.ffn_gate_inp.weight", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.3.mlp.switch_mlp.down_proj.weight", mapTensor(&buf, "blk.3.ffn_down_exps.weight", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.1.mlp.up_proj.weight", mapTensor(&buf, "blk.1.ffn_up.weight", hp).?.name);
    try std.testing.expectEqualStrings("lm_head.weight", mapTensor(&buf, "output.weight", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.48.attn_q.weight", hp) == null);
    try std.testing.expect(mapTensor(&buf, "blk.0.attn_qkv.weight", hp) == null);
}

test "configJson carries the dense and MoE geometry, head_dim always" {
    const a = std.testing.allocator;
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 36;
    hp.head_dim = 128;
    hp.rms_eps = 1e-6;
    hp.rope_theta = 5e6;
    hp.tied = true;
    const dense = try configJson(a, hp);
    defer a.free(dense);
    const d = try std.json.parseFromSlice(std.json.Value, a, dense, .{});
    defer d.deinit();
    try std.testing.expectEqualStrings("qwen3", d.value.object.get("model_type").?.string);
    try std.testing.expectEqual(@as(i64, 128), d.value.object.get("head_dim").?.integer);
    try std.testing.expect(d.value.object.get("num_experts") == null);

    hp.n_experts = 128;
    hp.n_experts_used = 8;
    hp.expert_ffn = 768;
    const moe = try configJson(a, hp);
    defer a.free(moe);
    const m = try std.json.parseFromSlice(std.json.Value, a, moe, .{});
    defer m.deinit();
    try std.testing.expectEqualStrings("qwen3_moe", m.value.object.get("model_type").?.string);
    try std.testing.expectEqual(@as(i64, 768), m.value.object.get("moe_intermediate_size").?.integer);
    try std.testing.expectEqual(@as(i64, 128), m.value.object.get("head_dim").?.integer);
}
