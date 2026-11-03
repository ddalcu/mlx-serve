//! Qwen3.5 / Qwen3.6, dense (GGUF arch "qwen35") and MoE ("qwen35moe", same
//! attention + GatedDeltaNet, the MLP is 256 routed experts + a shared one):
//! hparams, the HF-style config.json mlx-serve parses, and the GGUF -> HF tensor map.
//!
//! llama.cpp's converter stores the GatedDeltaNet value heads TILED
//! ([K0v0, K1v0, .., K0v1, K1v1, ..]) where HF groups them by key head
//! ([K0v0, K0v1, .., K1v0, ..]). Every transform here only undoes that and the
//! `ssm_a = -exp(A_log)` rewrite, nothing is requantized.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");
const Transform = arch.Transform;

/// GGUF arch names served here, the metadata keys carry the same prefix.
pub fn handles(arch_name: []const u8) bool {
    return std.mem.eql(u8, arch_name, "qwen35") or std.mem.eql(u8, arch_name, "qwen35moe");
}

/// Names follow the mlx-community Qwen3.5 checkpoints.
const prefix = "language_model.model.";

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    ffn: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_dims: u32,
    rope_theta: f64,
    rms_eps: f64,
    ctx_len: u32,
    full_attn_interval: u32,
    conv_kernel: u32,
    /// GatedDeltaNet: key heads, value heads, key and value head dims.
    nk: u32,
    nv: u32,
    dk: u32,
    dv: u32,
    vocab: u32,
    eos_id: u32,
    tied: bool,
    mrope: [3]u32,
    /// Routed experts, 0 = dense. Then experts per token and the expert / shared expert MLP widths.
    n_experts: u32,
    n_experts_used: u32,
    expert_ffn: u32,
    shared_ffn: u32,

    pub fn read(f: *const gguf.File, arch_name: []const u8) !Hparams {
        const k = Keys{ .f = f, .arch_name = arch_name };
        const nv = try k.int("ssm.time_step_rank");
        const sections = k.array("rope.dimension_sections") orelse return error.MissingMetadata;
        if (sections.len < 3) return error.MissingMetadata;
        const n_experts = k.int("expert_count") catch 0;
        const hp = Hparams{
            // block_count includes the MTP head block(s) llama.cpp keeps after the trunk; mlx-serve does not use them.
            .n_layers = (try k.int("block_count")) - (k.int("nextn_predict_layers") catch 0),
            .hidden = try k.int("embedding_length"),
            .ffn = if (n_experts == 0) try k.int("feed_forward_length") else 0,
            .n_experts = n_experts,
            .n_experts_used = if (n_experts == 0) 0 else try k.int("expert_used_count"),
            .expert_ffn = if (n_experts == 0) 0 else try k.int("expert_feed_forward_length"),
            .shared_ffn = if (n_experts == 0) 0 else try k.int("expert_shared_feed_forward_length"),
            .n_heads = try k.int("attention.head_count"),
            .n_kv_heads = try k.int("attention.head_count_kv"),
            .head_dim = try k.int("attention.key_length"),
            .rope_dims = try k.int("rope.dimension_count"),
            .rope_theta = try k.float("rope.freq_base"),
            .rms_eps = try k.float("attention.layer_norm_rms_epsilon"),
            .ctx_len = try k.int("context_length"),
            .full_attn_interval = try k.int("full_attention_interval"),
            .conv_kernel = try k.int("ssm.conv_kernel"),
            .nk = try k.int("ssm.group_count"),
            .nv = nv,
            .dk = try k.int("ssm.state_size"),
            .dv = (try k.int("ssm.inner_size")) / @max(nv, 1),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .eos_id = try arch.u32Key(f, "tokenizer.ggml.eos_token_id"),
            .tied = f.tensors.get("output.weight") == null,
            .mrope = .{ @intCast(sections.int(0)), @intCast(sections.int(1)), @intCast(sections.int(2)) },
        };
        if (hp.nk == 0 or hp.nv % hp.nk != 0 or hp.full_attn_interval == 0) return error.BadMetadata;
        return hp;
    }

    /// Value heads per key head.
    pub fn r(self: Hparams) u32 {
        return self.nv / self.nk;
    }
};

/// Metadata lookups under "<arch_name>.".
const Keys = struct {
    f: *const gguf.File,
    arch_name: []const u8,

    fn key(self: Keys, buf: []u8, suffix: []const u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}.{s}", .{ self.arch_name, suffix }) catch unreachable;
    }

    fn int(self: Keys, suffix: []const u8) !u32 {
        var buf: [96]u8 = undefined;
        return arch.u32Key(self.f, self.key(&buf, suffix));
    }

    fn float(self: Keys, suffix: []const u8) !f64 {
        var buf: [96]u8 = undefined;
        return self.f.getFloat(self.key(&buf, suffix)) orelse error.MissingMetadata;
    }

    fn array(self: Keys, suffix: []const u8) ?gguf.Array {
        var buf: [96]u8 = undefined;
        return self.f.getArray(self.key(&buf, suffix));
    }
};

/// The config.json an mlx-community conversion of the same model would carry.
pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var layer_types: std.ArrayList(u8) = .empty;
    defer layer_types.deinit(allocator);
    for (0..hp.n_layers) |i| {
        if (i > 0) try layer_types.append(allocator, ',');
        try layer_types.appendSlice(allocator, if ((i + 1) % hp.full_attn_interval == 0) "\"full_attention\"" else "\"linear_attention\"");
    }
    const moe = hp.n_experts > 0;
    var mlp_buf: [200]u8 = undefined;
    const mlp = if (moe) std.fmt.bufPrint(&mlp_buf,
        \\"num_experts":{d},"num_experts_per_tok":{d},"moe_intermediate_size":{d},"shared_expert_intermediate_size":{d}
    , .{ hp.n_experts, hp.n_experts_used, hp.expert_ffn, hp.shared_ffn }) catch unreachable else std.fmt.bufPrint(&mlp_buf,
        \\"intermediate_size":{d}
    , .{hp.ffn}) catch unreachable;
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["{[architecture]s}"],"model_type":"{[model_type]s}","tie_word_embeddings":{[tied]},
        \\"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}},
        \\"text_config":{{"model_type":"{[model_type]s}_text","attention_bias":false,"attn_output_gate":true,"dtype":"bfloat16",
        \\"eos_token_id":{[eos_id]d},"full_attention_interval":{[full_attn_interval]d},"head_dim":{[head_dim]d},"hidden_act":"silu",
        \\"hidden_size":{[hidden]d},{[mlp]s},"linear_conv_kernel_dim":{[conv_kernel]d},
        \\"linear_key_head_dim":{[dk]d},"linear_num_key_heads":{[nk]d},"linear_num_value_heads":{[nv]d},"linear_value_head_dim":{[dv]d},
        \\"max_position_embeddings":{[ctx_len]d},"mlp_only_layers":[],"num_attention_heads":{[n_heads]d},
        \\"num_hidden_layers":{[n_layers]d},"num_key_value_heads":{[n_kv_heads]d},"rms_norm_eps":{[rms_eps]e},
        \\"tie_word_embeddings":{[tied]},"vocab_size":{[vocab]d},"mamba_ssm_dtype":"float32","layer_types":[{[layer_types]s}],
        \\"rope_parameters":{{"mrope_interleaved":true,"mrope_section":[{[m0]d},{[m1]d},{[m2]d}],"rope_type":"default",
        \\"rope_theta":{[rope_theta]d},"partial_rotary_factor":{[rotary]d}}}}}}}
    , .{
        .architecture = if (moe) "Qwen3_5MoeForConditionalGeneration" else "Qwen3_5ForConditionalGeneration",
        .model_type = if (moe) "qwen3_5_moe" else "qwen3_5",
        .mlp = mlp,
        .tied = hp.tied,
        .eos_id = hp.eos_id,
        .full_attn_interval = hp.full_attn_interval,
        .head_dim = hp.head_dim,
        .hidden = hp.hidden,
        .conv_kernel = hp.conv_kernel,
        .dk = hp.dk,
        .nk = hp.nk,
        .nv = hp.nv,
        .dv = hp.dv,
        .ctx_len = hp.ctx_len,
        .n_heads = hp.n_heads,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .rms_eps = hp.rms_eps,
        .vocab = hp.vocab,
        .layer_types = layer_types.items,
        .m0 = hp.mrope[0],
        .m1 = hp.mrope[1],
        .m2 = hp.mrope[2],
        .rope_theta = hp.rope_theta,
        .rotary = @as(f64, @floatFromInt(hp.rope_dims)) / @as(f64, @floatFromInt(hp.head_dim)),
    });
}

/// HF name + load transform for a GGUF tensor, null when the tensor is unknown.
pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "token_embd.weight", prefix ++ "embed_tokens.weight" },
        .{ "output_norm.weight", prefix ++ "norm.weight" },
        .{ "output.weight", "language_model.lm_head.weight" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return .{ .name = g[1], .transform = .none };

    // The MTP head is the block right after the trunk.
    if (arch.splitLayer(name, hp.n_layers + 1)) |b| if (b.layer == hp.n_layers) return arch.skip;
    const blk = arch.splitLayer(name, hp.n_layers) orelse return null;

    const v_rows = Transform{ .untile_rows = .{ .start = 0, .unit = hp.dv } };
    const head_rows = Transform{ .untile_rows = .{ .start = 0, .unit = 1 } };
    const table = [_]struct { []const u8, []const u8, Transform }{
        .{ "attn_norm.weight", "input_layernorm.weight", .none },
        .{ "post_attention_norm.weight", "post_attention_layernorm.weight", .none },
        .{ "ffn_gate.weight", "mlp.gate_proj.weight", .none },
        .{ "ffn_up.weight", "mlp.up_proj.weight", .none },
        .{ "ffn_down.weight", "mlp.down_proj.weight", .none },
        // MoE: router, the expert banks ([experts, rows, row bytes]) and the gated shared expert.
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
        .{ "attn_qkv.weight", "linear_attn.in_proj_qkv.weight", .{ .untile_rows = .{ .start = 2 * hp.nk * hp.dk, .unit = hp.dv } } },
        .{ "attn_gate.weight", "linear_attn.in_proj_z.weight", v_rows },
        .{ "ssm_alpha.weight", "linear_attn.in_proj_a.weight", head_rows },
        .{ "ssm_beta.weight", "linear_attn.in_proj_b.weight", head_rows },
        .{ "ssm_a", "linear_attn.A_log", .a_log },
        .{ "ssm_dt.bias", "linear_attn.dt_bias", .untile_vec },
        .{ "ssm_conv1d.weight", "linear_attn.conv1d.weight", .conv1d },
        .{ "ssm_norm.weight", "linear_attn.norm.weight", .none },
        .{ "ssm_out.weight", "linear_attn.out_proj.weight", .tiled_input },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, prefix ++ "layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        // With one value head per key head the tiled and grouped orders coincide.
        const t: Transform = if (hp.r() == 1) switch (row[2]) {
            .a_log => .a_log,
            .conv1d => .conv1d,
            .shared_gate => .shared_gate,
            else => .none,
        } else row[2];
        return .{ .name = full, .transform = t };
    };
    return null;
}

/// Grouped position of tiled value head `t`: tiled index j*nk+g holds head g*r+j.
pub fn groupedIndex(t: u32, nk: u32, r: u32) u32 {
    return (t % nk) * r + t / nk;
}

test "groupedIndex undoes llama.cpp's tiling" {
    // nk=2, r=3: tiled order is [g0j0, g1j0, g0j1, g1j1, g0j2, g1j2].
    const want = [_]u32{ 0, 3, 1, 4, 2, 5 };
    for (want, 0..) |w, t| try std.testing.expectEqual(w, groupedIndex(@intCast(t), 2, 3));
}

test "mapTensor names layers and picks the un-tiling transform" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 4;
    hp.nk = 16;
    hp.nv = 32;
    hp.dk = 128;
    hp.dv = 128;
    var buf: [128]u8 = undefined;
    const qkv = mapTensor(&buf, "blk.2.attn_qkv.weight", hp).?;
    try std.testing.expectEqualStrings("language_model.model.layers.2.linear_attn.in_proj_qkv.weight", qkv.name);
    try std.testing.expectEqual(@as(u32, 4096), qkv.transform.untile_rows.start);
    try std.testing.expect(mapTensor(&buf, "blk.2.ssm_out.weight", hp).?.transform == .tiled_input);
    try std.testing.expect(mapTensor(&buf, "blk.9.ffn_up.weight", hp) == null);
    try std.testing.expect(mapTensor(&buf, "blk.1.nextn.eh_proj.weight", hp) == null);

    try std.testing.expectEqualStrings("language_model.model.layers.1.mlp.switch_mlp.down_proj.weight", mapTensor(&buf, "blk.1.ffn_down_exps.weight", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.1.ffn_gate_inp_shexp.weight", hp).?.transform == .shared_gate);

    hp.nv = 16;
    try std.testing.expect(mapTensor(&buf, "blk.2.attn_qkv.weight", hp).?.transform == .none);
    try std.testing.expect(mapTensor(&buf, "blk.1.ffn_gate_inp_shexp.weight", hp).?.transform == .shared_gate);
    try std.testing.expect(mapTensor(&buf, "blk.2.ssm_a", hp).?.transform == .a_log);
}

test "configJson is valid JSON with the GDN geometry" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 8;
    hp.full_attn_interval = 4;
    hp.head_dim = 256;
    hp.rope_dims = 64;
    hp.nk = 16;
    hp.nv = 32;
    hp.rms_eps = 1e-6;
    hp.rope_theta = 1e7;
    hp.tied = true;
    const json = try configJson(std.testing.allocator, hp);
    defer std.testing.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    const tc = parsed.value.object.get("text_config").?.object;
    try std.testing.expectEqual(@as(i64, 32), tc.get("linear_num_value_heads").?.integer);
    try std.testing.expectEqualStrings("full_attention", tc.get("layer_types").?.array.items[3].string);
    try std.testing.expectEqualStrings("linear_attention", tc.get("layer_types").?.array.items[4].string);
    try std.testing.expectEqual(@as(f64, 0.25), tc.get("rope_parameters").?.object.get("partial_rotary_factor").?.float);
    try std.testing.expectEqualStrings("qwen3_5_text", tc.get("model_type").?.string);
    try std.testing.expect(tc.get("num_experts") == null);

    hp.n_experts = 256;
    hp.n_experts_used = 8;
    hp.expert_ffn = 512;
    hp.shared_ffn = 512;
    const moe_json = try configJson(std.testing.allocator, hp);
    defer std.testing.allocator.free(moe_json);
    const moe = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, moe_json, .{});
    defer moe.deinit();
    try std.testing.expectEqualStrings("qwen3_5_moe", moe.value.object.get("model_type").?.string);
    const mtc = moe.value.object.get("text_config").?.object;
    try std.testing.expectEqualStrings("qwen3_5_moe_text", mtc.get("model_type").?.string);
    try std.testing.expectEqual(@as(i64, 8), mtc.get("num_experts_per_tok").?.integer);
    try std.testing.expect(mtc.get("intermediate_size") == null);
}
