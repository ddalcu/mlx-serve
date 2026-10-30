//! Qwen3-Next (GGUF arch "qwen3next"): GatedDeltaNet + gated full attention
//! every 4th layer, 512 routed experts + a shared one. Unlike `qwen35`,
//! llama.cpp's converter keeps HF's grouped value-head order here, so no
//! un-tiling; it stores the norms as w + 1 (what mlx-lm's checkpoints carry
//! too) and fuses the GatedDeltaNet b / a projections per key head, which
//! mlx-serve wants as two tensors.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");

pub const arch_name = "qwen3next";

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_dims: u32,
    rope_theta: f64,
    rms_eps: f64,
    ctx_len: u32,
    full_attn_interval: u32,
    conv_kernel: u32,
    nk: u32,
    nv: u32,
    dk: u32,
    dv: u32,
    n_experts: u32,
    n_experts_used: u32,
    expert_ffn: u32,
    shared_ffn: u32,
    vocab: u32,
    eos_id: u32,
    tied: bool,

    pub fn read(f: *const gguf.File) !Hparams {
        const nv = try arch.u32Key(f, "qwen3next.ssm.time_step_rank");
        const hp = Hparams{
            .n_layers = try arch.u32Key(f, "qwen3next.block_count"),
            .hidden = try arch.u32Key(f, "qwen3next.embedding_length"),
            .n_heads = try arch.u32Key(f, "qwen3next.attention.head_count"),
            .n_kv_heads = try arch.u32Key(f, "qwen3next.attention.head_count_kv"),
            .head_dim = try arch.u32Key(f, "qwen3next.attention.key_length"),
            .rope_dims = try arch.u32Key(f, "qwen3next.rope.dimension_count"),
            .rope_theta = f.getFloat("qwen3next.rope.freq_base") orelse return error.MissingMetadata,
            .rms_eps = f.getFloat("qwen3next.attention.layer_norm_rms_epsilon") orelse return error.MissingMetadata,
            .ctx_len = try arch.u32Key(f, "qwen3next.context_length"),
            .full_attn_interval = if (f.getInt("qwen3next.full_attention_interval") != null) try arch.u32Key(f, "qwen3next.full_attention_interval") else 4,
            .conv_kernel = try arch.u32Key(f, "qwen3next.ssm.conv_kernel"),
            .nk = try arch.u32Key(f, "qwen3next.ssm.group_count"),
            .nv = nv,
            .dk = try arch.u32Key(f, "qwen3next.ssm.state_size"),
            .dv = (try arch.u32Key(f, "qwen3next.ssm.inner_size")) / @max(nv, 1),
            .n_experts = try arch.u32Key(f, "qwen3next.expert_count"),
            .n_experts_used = try arch.u32Key(f, "qwen3next.expert_used_count"),
            .expert_ffn = try arch.u32Key(f, "qwen3next.expert_feed_forward_length"),
            .shared_ffn = try arch.u32Key(f, "qwen3next.expert_shared_feed_forward_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .eos_id = try arch.u32Key(f, "tokenizer.ggml.eos_token_id"),
            .tied = f.tensors.get("output.weight") == null,
        };
        if (hp.nk == 0 or hp.nv % hp.nk != 0 or hp.full_attn_interval == 0 or hp.head_dim == 0) return error.BadMetadata;
        return hp;
    }

    /// Value heads per key head.
    pub fn r(self: Hparams) u32 {
        return self.nv / self.nk;
    }
};

pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var layer_types: std.ArrayList(u8) = .empty;
    defer layer_types.deinit(allocator);
    for (0..hp.n_layers) |i| {
        if (i > 0) try layer_types.append(allocator, ',');
        try layer_types.appendSlice(allocator, if ((i + 1) % hp.full_attn_interval == 0) "\"full_attention\"" else "\"linear_attention\"");
    }
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["Qwen3NextForCausalLM"],"model_type":"qwen3_next","attention_bias":false,"attn_output_gate":true,
        \\"bos_token_id":151643,"decoder_sparse_step":1,"eos_token_id":{[eos_id]d},"full_attention_interval":{[full_attn_interval]d},
        \\"head_dim":{[head_dim]d},"hidden_act":"silu","hidden_size":{[hidden]d},"layer_types":[{[layer_types]s}],
        \\"linear_conv_kernel_dim":{[conv_kernel]d},"linear_key_head_dim":{[dk]d},"linear_num_key_heads":{[nk]d},
        \\"linear_num_value_heads":{[nv]d},"linear_value_head_dim":{[dv]d},"max_position_embeddings":{[ctx_len]d},"mlp_only_layers":[],
        \\"moe_intermediate_size":{[expert_ffn]d},"norm_topk_prob":true,"num_attention_heads":{[n_heads]d},"num_experts":{[n_experts]d},
        \\"num_experts_per_tok":{[n_experts_used]d},"num_hidden_layers":{[n_layers]d},"num_key_value_heads":{[n_kv_heads]d},
        \\"partial_rotary_factor":{[partial]d},"rms_norm_eps":{[rms_eps]e},"rope_scaling":null,"rope_theta":{[rope_theta]d},
        \\"shared_expert_intermediate_size":{[shared_ffn]d},"tie_word_embeddings":{[tied]},"use_sliding_window":false,"vocab_size":{[vocab]d},
        \\"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}}}}
    , .{
        .eos_id = hp.eos_id,
        .full_attn_interval = hp.full_attn_interval,
        .head_dim = hp.head_dim,
        .hidden = hp.hidden,
        .layer_types = layer_types.items,
        .conv_kernel = hp.conv_kernel,
        .dk = hp.dk,
        .nk = hp.nk,
        .nv = hp.nv,
        .dv = hp.dv,
        .ctx_len = hp.ctx_len,
        .expert_ffn = hp.expert_ffn,
        .n_heads = hp.n_heads,
        .n_experts = hp.n_experts,
        .n_experts_used = hp.n_experts_used,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .partial = @as(f64, @floatFromInt(hp.rope_dims)) / @as(f64, @floatFromInt(hp.head_dim)),
        .rms_eps = hp.rms_eps,
        .rope_theta = hp.rope_theta,
        .shared_ffn = hp.shared_ffn,
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
    // The MTP head is one more block past the trunk; mlx-serve does not use it.
    const blk = arch.splitLayer(name, hp.n_layers + 1) orelse return null;
    if (blk.layer == hp.n_layers) return arch.skip;
    const table = [_]struct { []const u8, []const u8, arch.Transform }{
        .{ "attn_norm.weight", "input_layernorm.weight", .none },
        .{ "post_attention_norm.weight", "post_attention_layernorm.weight", .none },
        .{ "ffn_gate_inp.weight", "mlp.gate.weight", .none },
        .{ "ffn_gate_exps.weight", "mlp.switch_mlp.gate_proj.weight", .none },
        .{ "ffn_up_exps.weight", "mlp.switch_mlp.up_proj.weight", .none },
        .{ "ffn_down_exps.weight", "mlp.switch_mlp.down_proj.weight", .none },
        .{ "ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight", .none },
        .{ "ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight", .none },
        .{ "ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight", .none },
        .{ "ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight", .shared_gate },
        .{ "attn_q.weight", "self_attn.q_proj.weight", .none },
        .{ "attn_k.weight", "self_attn.k_proj.weight", .none },
        .{ "attn_v.weight", "self_attn.v_proj.weight", .none },
        .{ "attn_output.weight", "self_attn.o_proj.weight", .none },
        .{ "attn_q_norm.weight", "self_attn.q_norm.weight", .none },
        .{ "attn_k_norm.weight", "self_attn.k_norm.weight", .none },
        .{ "attn_qkv.weight", "linear_attn.in_proj_qkv.weight", .none },
        .{ "attn_gate.weight", "linear_attn.in_proj_z.weight", .none },
        .{ "ssm_ba.weight", "linear_attn.in_proj_ba.weight", .{ .split_ba = .{ .nk = hp.nk, .r = hp.r() } } },
        .{ "ssm_a", "linear_attn.A_log", .neg_log },
        .{ "ssm_dt.bias", "linear_attn.dt_bias", .none },
        .{ "ssm_conv1d.weight", "linear_attn.conv1d.weight", .trailing_axis },
        .{ "ssm_norm.weight", "linear_attn.norm.weight", .none },
        .{ "ssm_out.weight", "linear_attn.out_proj.weight", .none },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = row[2] };
    };
    return null;
}

test "mapTensor renames the GatedDeltaNet and expert tensors and splits b/a" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 48;
    hp.nk = 16;
    hp.nv = 32;
    var buf: [128]u8 = undefined;
    const ba = mapTensor(&buf, "blk.0.ssm_ba.weight", hp).?;
    try std.testing.expectEqualStrings("model.layers.0.linear_attn.in_proj_ba.weight", ba.name);
    try std.testing.expectEqual(@as(u32, 2), ba.transform.split_ba.r);
    try std.testing.expectEqualStrings("model.layers.3.self_attn.q_proj.weight", mapTensor(&buf, "blk.3.attn_q.weight", hp).?.name);
    try std.testing.expectEqualStrings("model.layers.1.mlp.switch_mlp.up_proj.weight", mapTensor(&buf, "blk.1.ffn_up_exps.weight", hp).?.name);
    try std.testing.expectEqualStrings("", mapTensor(&buf, "blk.48.nextn.enorm.weight", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.49.attn_norm.weight", hp) == null);
}

test "configJson marks every 4th layer full attention" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 8;
    hp.full_attn_interval = 4;
    hp.head_dim = 256;
    hp.rope_dims = 64;
    hp.rms_eps = 1e-6;
    hp.rope_theta = 1e7;
    const a = std.testing.allocator;
    const json = try configJson(a, hp);
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("qwen3_next", o.get("model_type").?.string);
    try std.testing.expectEqualStrings("full_attention", o.get("layer_types").?.array.items[3].string);
    try std.testing.expectEqualStrings("linear_attention", o.get("layer_types").?.array.items[4].string);
    try std.testing.expectEqual(@as(f64, 0.25), o.get("partial_rotary_factor").?.float);
}
