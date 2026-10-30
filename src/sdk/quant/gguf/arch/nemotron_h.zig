//! Nemotron-H (GGUF arch "nemotron_h") and Nemotron 3 Nano ("nemotron_h_moe"):
//! Mamba2 layers with attention and MLP / MoE layers in between, told apart
//! per layer by the KV head count (attention) and the MLP width (0 = Mamba).
//! Tensors are stored as HF does, except `ssm_a = -exp(A_log)` and the
//! Mamba vectors carrying a leading axis of 1.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");

const MAX_LAYERS = 128;

pub fn handles(name: []const u8) bool {
    return std.mem.eql(u8, name, "nemotron_h") or std.mem.eql(u8, name, "nemotron_h_moe");
}

const Kind = enum { mamba, attention, mlp, moe };

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    /// MLP width of the dense MLP layers (0 when every MLP layer is MoE).
    ffn: u32,
    kinds: [MAX_LAYERS]Kind,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    mamba_heads: u32,
    mamba_head_dim: u32,
    n_groups: u32,
    state_size: u32,
    conv_kernel: u32,
    expand: u32,
    rope_theta: f64,
    rms_eps: f64,
    ctx_len: u32,
    vocab: u32,
    eos: u32,
    n_experts: u32,
    n_experts_used: u32,
    expert_ffn: u32,
    shared_ffn: u32,
    routed_scale: f64,
    norm_topk: bool,

    pub fn read(f: *const gguf.File, name: []const u8) !Hparams {
        const k = Keys{ .f = f, .arch = name };
        const n_layers = try k.int("block_count");
        if (n_layers == 0 or n_layers > MAX_LAYERS) return error.BadMetadata;
        const n_experts = k.int("expert_count") catch 0;
        var kinds: [MAX_LAYERS]Kind = @splat(.mamba);
        var n_kv_heads: u32 = 0;
        var ffn: u32 = 0;
        for (0..n_layers) |i| {
            const kv = try k.perLayer("attention.head_count_kv", i);
            const ff = try k.perLayer("feed_forward_length", i);
            kinds[i] = if (kv > 0) .attention else if (ff > 0) (if (n_experts > 0) .moe else .mlp) else .mamba;
            if (kv > 0) n_kv_heads = kv;
            if (ff > 0 and n_experts == 0) ffn = ff;
        }
        const inner = try k.int("ssm.inner_size");
        const mamba_heads = try k.int("ssm.time_step_rank");
        const hidden = try k.int("embedding_length");
        if (mamba_heads == 0 or inner % mamba_heads != 0) return error.BadMetadata;
        return .{
            .n_layers = n_layers,
            .hidden = hidden,
            .ffn = ffn,
            .kinds = kinds,
            .n_heads = try k.int("attention.head_count"),
            .n_kv_heads = n_kv_heads,
            .head_dim = try k.int("attention.key_length"),
            .mamba_heads = mamba_heads,
            .mamba_head_dim = inner / mamba_heads,
            .n_groups = try k.int("ssm.group_count"),
            .state_size = try k.int("ssm.state_size"),
            .conv_kernel = try k.int("ssm.conv_kernel"),
            // HF's `expand` (2 on every Nemotron-H); the real inner width is heads x head dim.
            .expand = 2,
            .rope_theta = try k.float("rope.freq_base"),
            .rms_eps = try k.float("attention.layer_norm_rms_epsilon"),
            .ctx_len = try k.int("context_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .eos = try arch.u32Key(f, "tokenizer.ggml.eos_token_id"),
            .n_experts = n_experts,
            .n_experts_used = if (n_experts > 0) try k.int("expert_used_count") else 0,
            .expert_ffn = if (n_experts > 0) try k.int("expert_feed_forward_length") else 0,
            .shared_ffn = if (n_experts > 0) k.int("expert_shared_feed_forward_length") catch 0 else 0,
            .routed_scale = if (n_experts > 0) k.float("expert_weights_scale") catch 1.0 else 1.0,
            .norm_topk = if (n_experts > 0) k.boolean("expert_weights_norm") else false,
        };
    }
};

const Keys = struct {
    f: *const gguf.File,
    arch: []const u8,

    fn key(self: Keys, buf: []u8, suffix: []const u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}.{s}", .{ self.arch, suffix }) catch unreachable;
    }

    fn int(self: Keys, suffix: []const u8) !u32 {
        var buf: [96]u8 = undefined;
        return arch.u32Key(self.f, self.key(&buf, suffix));
    }

    fn float(self: Keys, suffix: []const u8) !f64 {
        var buf: [96]u8 = undefined;
        return self.f.getFloat(self.key(&buf, suffix)) orelse error.MissingMetadata;
    }

    fn boolean(self: Keys, suffix: []const u8) bool {
        var buf: [96]u8 = undefined;
        return self.f.getBool(self.key(&buf, suffix)) orelse false;
    }

    /// One number for every layer or an array with one per layer.
    fn perLayer(self: Keys, suffix: []const u8, layer: usize) !u32 {
        var buf: [96]u8 = undefined;
        const name = self.key(&buf, suffix);
        if (self.f.getInt(name) != null) return arch.u32Key(self.f, name);
        const arr = self.f.getArray(name) orelse return error.MissingMetadata;
        if (layer >= arr.len) return error.BadMetadata;
        return std.math.cast(u32, arr.int(layer)) orelse error.BadMetadata;
    }
};

pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var pattern: [MAX_LAYERS]u8 = undefined;
    for (hp.kinds[0..hp.n_layers], 0..) |kind, i| pattern[i] = switch (kind) {
        .mamba => 'M',
        .attention => '*',
        .mlp => '-',
        .moe => 'E',
    };
    var moe_buf: [320]u8 = undefined;
    const moe = if (hp.n_experts > 0) try std.fmt.bufPrint(&moe_buf,
        \\,"n_routed_experts":{d},"num_experts_per_tok":{d},"moe_intermediate_size":{d},"moe_shared_expert_intermediate_size":{d},
        \\"n_shared_experts":{d},"norm_topk_prob":{},"routed_scaling_factor":{d},"n_group":1,"topk_group":1
    , .{ hp.n_experts, hp.n_experts_used, hp.expert_ffn, hp.shared_ffn, @intFromBool(hp.shared_ffn > 0), hp.norm_topk, hp.routed_scale }) else "";
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["NemotronHForCausalLM"],"model_type":"nemotron_h","bos_token_id":1,"eos_token_id":{[eos]d},
        \\"hidden_size":{[hidden]d},"intermediate_size":{[ffn]d},"num_hidden_layers":{[n_layers]d},"hybrid_override_pattern":"{[pattern]s}",
        \\"num_attention_heads":{[n_heads]d},"num_key_value_heads":{[n_kv_heads]d},"head_dim":{[head_dim]d},"attention_bias":false,
        \\"mamba_num_heads":{[mamba_heads]d},"mamba_head_dim":{[mamba_head_dim]d},"n_groups":{[n_groups]d},"ssm_state_size":{[state_size]d},
        \\"conv_kernel":{[conv_kernel]d},"expand":{[expand]d},"chunk_size":128,"mamba_hidden_act":"silu","mlp_hidden_act":"relu2","mlp_bias":false,
        \\"layer_norm_epsilon":{[rms_eps]e},"rms_norm_eps":{[rms_eps]e},"rope_theta":{[rope_theta]d},"max_position_embeddings":{[ctx_len]d},
        \\"tie_word_embeddings":false,"vocab_size":{[vocab]d}{[moe]s},"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}}}}
    , .{
        .eos = hp.eos,
        .hidden = hp.hidden,
        .ffn = if (hp.ffn > 0) hp.ffn else hp.expert_ffn,
        .n_layers = hp.n_layers,
        .pattern = pattern[0..hp.n_layers],
        .n_heads = hp.n_heads,
        .n_kv_heads = hp.n_kv_heads,
        .head_dim = hp.head_dim,
        .mamba_heads = hp.mamba_heads,
        .mamba_head_dim = hp.mamba_head_dim,
        .n_groups = hp.n_groups,
        .state_size = hp.state_size,
        .conv_kernel = hp.conv_kernel,
        .expand = hp.expand,
        .rms_eps = hp.rms_eps,
        .rope_theta = hp.rope_theta,
        .ctx_len = hp.ctx_len,
        .vocab = hp.vocab,
        .moe = moe,
    });
}

pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "token_embd.weight", "backbone.embeddings.weight" },
        .{ "output_norm.weight", "backbone.norm_f.weight" },
        .{ "output.weight", "lm_head.weight" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return .{ .name = g[1], .transform = .none };
    // The MTP head is one more block past the trunk; mlx-serve does not use it.
    const blk = arch.splitLayer(name, hp.n_layers + 1) orelse return null;
    if (blk.layer == hp.n_layers) return arch.skip;
    const table = [_]struct { []const u8, []const u8, arch.Transform }{
        .{ "attn_norm.weight", "norm.weight", .none },
        .{ "ssm_in.weight", "mixer.in_proj.weight", .none },
        .{ "ssm_conv1d.weight", "mixer.conv1d.weight", .trailing_axis },
        .{ "ssm_conv1d.bias", "mixer.conv1d.bias", .none },
        .{ "ssm_a", "mixer.A_log", .neg_log },
        .{ "ssm_d", "mixer.D", .flatten },
        .{ "ssm_dt.bias", "mixer.dt_bias", .none },
        .{ "ssm_norm.weight", "mixer.norm.weight", .flatten },
        .{ "ssm_out.weight", "mixer.out_proj.weight", .none },
        .{ "attn_q.weight", "mixer.q_proj.weight", .none },
        .{ "attn_k.weight", "mixer.k_proj.weight", .none },
        .{ "attn_v.weight", "mixer.v_proj.weight", .none },
        .{ "attn_output.weight", "mixer.o_proj.weight", .none },
        .{ "ffn_up.weight", "mixer.up_proj.weight", .none },
        .{ "ffn_down.weight", "mixer.down_proj.weight", .none },
        .{ "ffn_gate_inp.weight", "mixer.gate.weight", .none },
        .{ "exp_probs_b.bias", "mixer.gate.e_score_correction_bias", .none },
        .{ "ffn_up_exps.weight", "mixer.switch_mlp.fc1.weight", .none },
        .{ "ffn_down_exps.weight", "mixer.switch_mlp.fc2.weight", .none },
        .{ "ffn_up_shexp.weight", "mixer.shared_experts.up_proj.weight", .none },
        .{ "ffn_down_shexp.weight", "mixer.shared_experts.down_proj.weight", .none },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "backbone.layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = row[2] };
    };
    return null;
}

test "mapTensor renames mamba, attention and expert tensors and skips the MTP block" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 52;
    var buf: [128]u8 = undefined;
    const a = mapTensor(&buf, "blk.0.ssm_a", hp).?;
    try std.testing.expectEqualStrings("backbone.layers.0.mixer.A_log", a.name);
    try std.testing.expectEqual(arch.Transform.neg_log, a.transform);
    try std.testing.expectEqualStrings("backbone.layers.1.mixer.switch_mlp.fc1.weight", mapTensor(&buf, "blk.1.ffn_up_exps.weight", hp).?.name);
    try std.testing.expectEqualStrings("backbone.layers.5.mixer.o_proj.weight", mapTensor(&buf, "blk.5.attn_output.weight", hp).?.name);
    try std.testing.expectEqualStrings("backbone.norm_f.weight", mapTensor(&buf, "output_norm.weight", hp).?.name);
    try std.testing.expectEqualStrings("", mapTensor(&buf, "blk.52.nextn.enorm.weight", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.53.attn_norm.weight", hp) == null);
}

test "configJson spells the layer pattern and the MoE block" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 6;
    hp.kinds[1] = .moe;
    hp.kinds[3] = .moe;
    hp.kinds[5] = .attention;
    hp.n_experts = 128;
    hp.n_experts_used = 6;
    hp.expert_ffn = 1856;
    hp.shared_ffn = 3712;
    hp.routed_scale = 2.5;
    hp.norm_topk = true;
    hp.rms_eps = 1e-5;
    hp.rope_theta = 10000;
    const a = std.testing.allocator;
    const json = try configJson(a, hp);
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("MEMEM*", o.get("hybrid_override_pattern").?.string);
    try std.testing.expectEqual(@as(i64, 128), o.get("n_routed_experts").?.integer);
    try std.testing.expectEqual(@as(i64, 1), o.get("n_shared_experts").?.integer);
    try std.testing.expectEqual(@as(f64, 2.5), o.get("routed_scaling_factor").?.float);
}
