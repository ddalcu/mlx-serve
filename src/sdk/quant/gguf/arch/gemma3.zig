//! Gemma 3 (GGUF arch "gemma3"), text only: the 1B / 4B / 12B / 27B decoders.
//! llama.cpp's converter stores every norm weight as HF's weight + 1 (its
//! RMSNorm multiplies by 1 + w), so those come back down by one at load.
//! mlx-serve's flat `gemma3_text` config puts the weights under `model.`.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");

pub const arch_name = "gemma3";

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    ffn: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    /// 0 = no sliding layers.
    sliding_window: u32,
    /// Every `swa_pattern`-th layer is global.
    swa_pattern: u32,
    rope_theta: f64,
    rope_theta_swa: f64,
    /// Linear rope scaling factor, 1 = none.
    rope_factor: f64,
    softcap: f64,
    rms_eps: f64,
    ctx_len: u32,
    vocab: u32,
    eos: [2]u32,
    eos_len: usize,

    pub fn read(f: *const gguf.File) !Hparams {
        const n_layers = try arch.u32Key(f, "gemma3.block_count");
        var hp: Hparams = .{
            .n_layers = n_layers,
            .hidden = try arch.u32Key(f, "gemma3.embedding_length"),
            .ffn = try arch.u32Key(f, "gemma3.feed_forward_length"),
            .n_heads = try arch.u32Key(f, "gemma3.attention.head_count"),
            .n_kv_heads = try arch.u32Key(f, "gemma3.attention.head_count_kv"),
            .head_dim = try arch.u32Key(f, "gemma3.attention.key_length"),
            .sliding_window = if (f.getInt("gemma3.attention.sliding_window") != null) try arch.u32Key(f, "gemma3.attention.sliding_window") else 0,
            .swa_pattern = if (f.getInt("gemma3.attention.sliding_window_pattern") != null) try arch.u32Key(f, "gemma3.attention.sliding_window_pattern") else 6,
            .rope_theta = f.getFloat("gemma3.rope.freq_base") orelse return error.MissingMetadata,
            .rope_theta_swa = f.getFloat("gemma3.rope.freq_base_swa") orelse 10000.0,
            .rope_factor = f.getFloat("gemma3.rope.scaling.factor") orelse 1.0,
            .softcap = f.getFloat("gemma3.final_logit_softcapping") orelse 0.0,
            .rms_eps = f.getFloat("gemma3.attention.layer_norm_rms_epsilon") orelse return error.MissingMetadata,
            .ctx_len = try arch.u32Key(f, "gemma3.context_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .eos = undefined,
            .eos_len = 0,
        };
        // The GGUF's eos (<end_of_turn>) and <eos>, like the HF config.
        hp.addEos(try arch.u32Key(f, "tokenizer.ggml.eos_token_id"));
        var it = (f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).strings();
        var id: u32 = 0;
        while (it.next()) |tok| : (id += 1) if (std.mem.eql(u8, tok, "<eos>")) {
            hp.addEos(id);
            break;
        };
        return hp;
    }

    fn addEos(self: *Hparams, id: u32) void {
        if (std.mem.indexOfScalar(u32, self.eos[0..self.eos_len], id) != null) return;
        self.eos[self.eos_len] = id;
        self.eos_len += 1;
    }

    /// Attention scale: 1/sqrt(head_dim), except the 27B (62 layers), which
    /// llama.cpp and HF scale by hidden / heads.
    fn queryPreAttnScalar(self: Hparams) u32 {
        return if (self.n_layers == 62) self.hidden / self.n_heads else self.head_dim;
    }
};

pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var eos: std.ArrayList(u8) = .empty;
    defer eos.deinit(allocator);
    for (hp.eos[0..hp.eos_len], 0..) |id, i| try eos.print(allocator, "{s}{d}", .{ if (i > 0) "," else "", id });
    var softcap_buf: [32]u8 = undefined;
    const softcap = if (hp.softcap == 0) "null" else try std.fmt.bufPrint(&softcap_buf, "{d}", .{hp.softcap});
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["Gemma3ForCausalLM"],"model_type":"gemma3_text","attention_bias":false,"attn_logit_softcapping":null,
        \\"bos_token_id":2,"eos_token_id":[{[eos]s}],"final_logit_softcapping":{[softcap]s},"head_dim":{[head_dim]d},
        \\"hidden_activation":"gelu_pytorch_tanh","hidden_size":{[hidden]d},"intermediate_size":{[ffn]d},
        \\"max_position_embeddings":{[ctx_len]d},"num_attention_heads":{[n_heads]d},"num_hidden_layers":{[n_layers]d},
        \\"num_key_value_heads":{[n_kv_heads]d},"query_pre_attn_scalar":{[qpa]d},"rms_norm_eps":{[rms_eps]e},
        \\"rope_local_base_freq":{[rope_theta_swa]d},"rope_scaling":{{"factor":{[rope_factor]d:.1},"rope_type":"linear"}},"rope_theta":{[rope_theta]d},
        \\"sliding_window":{[sliding_window]d},"sliding_window_pattern":{[swa_pattern]d},"tie_word_embeddings":true,"vocab_size":{[vocab]d},
        \\"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}}}}
    , .{
        .eos = eos.items,
        .softcap = softcap,
        .head_dim = hp.head_dim,
        .hidden = hp.hidden,
        .ffn = hp.ffn,
        .ctx_len = hp.ctx_len,
        .n_heads = hp.n_heads,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .qpa = hp.queryPreAttnScalar(),
        .rms_eps = hp.rms_eps,
        .rope_theta_swa = hp.rope_theta_swa,
        .rope_factor = hp.rope_factor,
        .rope_theta = hp.rope_theta,
        .sliding_window = hp.sliding_window,
        .swa_pattern = hp.swa_pattern,
        .vocab = hp.vocab,
    });
}

pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    if (std.mem.eql(u8, name, "token_embd.weight")) return .{ .name = "model.embed_tokens.weight", .transform = .none };
    if (std.mem.eql(u8, name, "output_norm.weight")) return .{ .name = "model.norm.weight", .transform = .minus_one };
    if (std.mem.eql(u8, name, "output.weight")) return .{ .name = "lm_head.weight", .transform = .none };
    if (std.mem.eql(u8, name, "rope_freqs.weight")) return arch.skip;

    const blk = arch.splitLayer(name, hp.n_layers) orelse return null;
    const table = [_]struct { []const u8, []const u8 }{
        .{ "attn_norm.weight", "input_layernorm.weight" },
        .{ "attn_q.weight", "self_attn.q_proj.weight" },
        .{ "attn_k.weight", "self_attn.k_proj.weight" },
        .{ "attn_v.weight", "self_attn.v_proj.weight" },
        .{ "attn_output.weight", "self_attn.o_proj.weight" },
        .{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
        .{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
        .{ "post_attention_norm.weight", "post_attention_layernorm.weight" },
        .{ "ffn_norm.weight", "pre_feedforward_layernorm.weight" },
        .{ "ffn_gate.weight", "mlp.gate_proj.weight" },
        .{ "ffn_up.weight", "mlp.up_proj.weight" },
        .{ "ffn_down.weight", "mlp.down_proj.weight" },
        .{ "post_ffw_norm.weight", "post_feedforward_layernorm.weight" },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = if (std.mem.endsWith(u8, row[0], "norm.weight")) .minus_one else .none };
    };
    return null;
}

test "mapTensor renames layers and marks every norm for the -1 shift" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 4;
    var buf: [128]u8 = undefined;
    const qn = mapTensor(&buf, "blk.3.attn_q_norm.weight", hp).?;
    try std.testing.expectEqualStrings("model.layers.3.self_attn.q_norm.weight", qn.name);
    try std.testing.expectEqual(arch.Transform.minus_one, qn.transform);
    const up = mapTensor(&buf, "blk.0.ffn_up.weight", hp).?;
    try std.testing.expectEqualStrings("model.layers.0.mlp.up_proj.weight", up.name);
    try std.testing.expectEqual(arch.Transform.none, up.transform);
    try std.testing.expectEqual(arch.Transform.minus_one, mapTensor(&buf, "output_norm.weight", hp).?.transform);
    try std.testing.expect(mapTensor(&buf, "blk.4.attn_q.weight", hp) == null);
    try std.testing.expect(mapTensor(&buf, "blk.0.ffn_gate_inp.weight", hp) == null);
}

test "configJson is valid JSON with the HF gemma3_text fields" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 26;
    hp.hidden = 1152;
    hp.n_heads = 4;
    hp.head_dim = 256;
    hp.swa_pattern = 6;
    hp.rope_factor = 1;
    hp.rms_eps = 1e-6;
    hp.addEos(106);
    hp.addEos(1);
    const a = std.testing.allocator;
    const json = try configJson(a, hp);
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("gemma3_text", o.get("model_type").?.string);
    try std.testing.expectEqual(@as(i64, 256), o.get("query_pre_attn_scalar").?.integer);
    try std.testing.expectEqual(@as(usize, 2), o.get("eos_token_id").?.array.items.len);
    try std.testing.expect(o.get("final_logit_softcapping").? == .null);
    try std.testing.expectEqual(@as(f64, 1), o.get("rope_scaling").?.object.get("factor").?.float);
    hp.n_layers = 62;
    hp.hidden = 5376;
    hp.n_heads = 32;
    const big = try configJson(a, hp);
    defer a.free(big);
    const bp = try std.json.parseFromSlice(std.json.Value, a, big, .{});
    defer bp.deinit();
    try std.testing.expectEqual(@as(i64, 168), bp.value.object.get("query_pre_attn_scalar").?.integer);
}
