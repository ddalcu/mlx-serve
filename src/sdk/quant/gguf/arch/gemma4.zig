//! Gemma 4 (GGUF arch "gemma4"): E2B / E4B with per-layer embeddings (PLE) and
//! shared-KV layers, and the dense 12B+ whose full-attention layers reuse K as V.
//! The GGUF stores every tensor exactly like the HF checkpoint (no norm offset,
//! no head reorder), so loading is a rename.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");

pub const arch_name = "gemma4";

/// Names follow the mlx-community Gemma 4 checkpoints.
const prefix = "language_model.model.";

const MAX_LAYERS = 128;

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    /// MLP width of layer 0. `double_wide`: the KV-shared layers are 2x that.
    ffn: u32,
    double_wide: bool,
    n_heads: u32,
    /// KV heads and head dim of the sliding layers, then of the full-attention ones.
    n_kv_heads: u32,
    head_dim: u32,
    n_global_kv_heads: u32,
    global_head_dim: u32,
    sliding_window: u32,
    sliding: [MAX_LAYERS]bool,
    shared_kv_layers: u32,
    /// Per-layer embedding width, 0 = the model has none.
    ple_dim: u32,
    softcap: f64,
    rope_theta: f64,
    rope_theta_swa: f64,
    rms_eps: f64,
    ctx_len: u32,
    vocab: u32,
    eos: Eos,
    /// Full-attention layers have no v_proj, V = K.
    k_eq_v: bool,

    pub fn read(f: *const gguf.File) !Hparams {
        const n_layers = try arch.u32Key(f, "gemma4.block_count");
        const pattern = f.getArray("gemma4.attention.sliding_window_pattern") orelse return error.MissingMetadata;
        if (n_layers == 0 or n_layers > MAX_LAYERS or pattern.len != n_layers) return error.BadMetadata;
        var sliding: [MAX_LAYERS]bool = @splat(false);
        for (0..n_layers) |i| sliding[i] = pattern.boolean(i);
        const first_swa = std.mem.indexOfScalar(bool, sliding[0..n_layers], true) orelse return error.BadMetadata;
        const first_full = std.mem.indexOfScalar(bool, sliding[0..n_layers], false) orelse return error.BadMetadata;

        var name_buf: [64]u8 = undefined;
        const full_v = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{first_full}) catch unreachable;
        const ffn = try perLayer(f, "gemma4.feed_forward_length", 0);
        return .{
            .n_layers = n_layers,
            .hidden = try arch.u32Key(f, "gemma4.embedding_length"),
            .ffn = ffn,
            .double_wide = try perLayer(f, "gemma4.feed_forward_length", n_layers - 1) == 2 * ffn,
            .n_heads = try arch.u32Key(f, "gemma4.attention.head_count"),
            .n_kv_heads = try perLayer(f, "gemma4.attention.head_count_kv", first_swa),
            .head_dim = try arch.u32Key(f, "gemma4.attention.key_length_swa"),
            .n_global_kv_heads = try perLayer(f, "gemma4.attention.head_count_kv", first_full),
            .global_head_dim = try arch.u32Key(f, "gemma4.attention.key_length"),
            .sliding_window = try arch.u32Key(f, "gemma4.attention.sliding_window"),
            .sliding = sliding,
            .shared_kv_layers = try arch.u32Key(f, "gemma4.attention.shared_kv_layers"),
            .ple_dim = try arch.u32Key(f, "gemma4.embedding_length_per_layer_input"),
            .softcap = f.getFloat("gemma4.final_logit_softcapping") orelse return error.MissingMetadata,
            .rope_theta = f.getFloat("gemma4.rope.freq_base") orelse return error.MissingMetadata,
            .rope_theta_swa = f.getFloat("gemma4.rope.freq_base_swa") orelse return error.MissingMetadata,
            .rms_eps = f.getFloat("gemma4.attention.layer_norm_rms_epsilon") orelse return error.MissingMetadata,
            .ctx_len = try arch.u32Key(f, "gemma4.context_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .eos = try Eos.read(f),
            .k_eq_v = f.tensors.get(full_v) == null,
        };
    }

    /// A key that is one number for every layer or an array with one per layer.
    fn perLayer(f: *const gguf.File, key: []const u8, layer: usize) !u32 {
        if (f.getInt(key) != null) return arch.u32Key(f, key);
        const arr = f.getArray(key) orelse return error.MissingMetadata;
        if (layer >= arr.len) return error.BadMetadata;
        return std.math.cast(u32, arr.int(layer)) orelse error.BadMetadata;
    }
};

/// Every token that ends a reply: the GGUF's eos (`<turn|>` in practice) and,
/// like the HF configs, `<eos>` and `<|tool_response>` (the model opens one
/// right after a tool call, which is where generation has to stop).
const Eos = struct {
    ids: [3]u32 = undefined,
    len: usize = 0,

    fn read(f: *const gguf.File) !Eos {
        var eos = Eos{};
        eos.add(try arch.u32Key(f, "tokenizer.ggml.eos_token_id"));
        var it = (f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).strings();
        var id: u32 = 0;
        while (it.next()) |tok| : (id += 1) {
            if (std.mem.eql(u8, tok, "<eos>") or std.mem.eql(u8, tok, "<|tool_response>")) eos.add(id);
            if (eos.len == eos.ids.len) break;
        }
        return eos;
    }

    fn add(self: *Eos, id: u32) void {
        if (std.mem.indexOfScalar(u32, self.ids[0..self.len], id) != null) return;
        self.ids[self.len] = id;
        self.len += 1;
    }
};

/// The 12B+ dense models are HF's `gemma4_unified`, the PLE ones plain `gemma4`.
pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var layer_types: std.ArrayList(u8) = .empty;
    defer layer_types.deinit(allocator);
    for (hp.sliding[0..hp.n_layers], 0..) |swa, i| {
        if (i > 0) try layer_types.append(allocator, ',');
        try layer_types.appendSlice(allocator, if (swa) "\"sliding_attention\"" else "\"full_attention\"");
    }
    var eos: std.ArrayList(u8) = .empty;
    defer eos.deinit(allocator);
    for (hp.eos.ids[0..hp.eos.len], 0..) |id, i| try eos.print(allocator, "{s}{d}", .{ if (i > 0) "," else "", id });

    const unified = hp.ple_dim == 0;
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["{[architecture]s}"],"model_type":"{[model_type]s}","tie_word_embeddings":true,
        \\"eos_token_id":[{[eos]s}],"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}},
        \\"text_config":{{"model_type":"{[model_type]s}_text","attention_bias":false,"attention_k_eq_v":{[k_eq_v]},"dtype":"bfloat16",
        \\"enable_moe_block":false,"final_logit_softcapping":{[softcap]d},"global_head_dim":{[global_head_dim]d},"head_dim":{[head_dim]d},
        \\"hidden_activation":"gelu_pytorch_tanh","hidden_size":{[hidden]d},"hidden_size_per_layer_input":{[ple_dim]d},
        \\"intermediate_size":{[ffn]d},"layer_types":[{[layer_types]s}],"max_position_embeddings":{[ctx_len]d},
        \\"num_attention_heads":{[n_heads]d},"num_global_key_value_heads":{[n_global_kv_heads]d},"num_hidden_layers":{[n_layers]d},
        \\"num_key_value_heads":{[n_kv_heads]d},"num_kv_shared_layers":{[shared_kv_layers]d},"rms_norm_eps":{[rms_eps]e},
        \\"rope_parameters":{{"full_attention":{{"partial_rotary_factor":0.25,"rope_theta":{[rope_theta]d},"rope_type":"proportional"}},
        \\"sliding_attention":{{"rope_theta":{[rope_theta_swa]d},"rope_type":"default"}}}},"sliding_window":{[sliding_window]d},
        \\"tie_word_embeddings":true,"use_double_wide_mlp":{[double_wide]},"vocab_size":{[vocab]d},
        \\"vocab_size_per_layer_input":{[ple_vocab]d}}}}}
    , .{
        .architecture = if (unified) "Gemma4UnifiedForConditionalGeneration" else "Gemma4ForConditionalGeneration",
        .model_type = if (unified) "gemma4_unified" else "gemma4",
        .eos = eos.items,
        .k_eq_v = hp.k_eq_v,
        .softcap = hp.softcap,
        .global_head_dim = hp.global_head_dim,
        .head_dim = hp.head_dim,
        .hidden = hp.hidden,
        .ple_dim = hp.ple_dim,
        .ffn = hp.ffn,
        .layer_types = layer_types.items,
        .ctx_len = hp.ctx_len,
        .n_heads = hp.n_heads,
        .n_global_kv_heads = hp.n_global_kv_heads,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .shared_kv_layers = hp.shared_kv_layers,
        .rms_eps = hp.rms_eps,
        .rope_theta = hp.rope_theta,
        .rope_theta_swa = hp.rope_theta_swa,
        .sliding_window = hp.sliding_window,
        .double_wide = hp.double_wide,
        .vocab = hp.vocab,
        .ple_vocab = if (unified) 0 else hp.vocab,
    });
}

pub fn mapTensor(buf: []u8, name: []const u8, hp: Hparams) ?arch.Mapped {
    const globals = [_]struct { []const u8, []const u8 }{
        .{ "token_embd.weight", prefix ++ "embed_tokens.weight" },
        .{ "output_norm.weight", prefix ++ "norm.weight" },
        .{ "per_layer_token_embd.weight", prefix ++ "embed_tokens_per_layer.weight" },
        .{ "per_layer_model_proj.weight", prefix ++ "per_layer_model_projection.weight" },
        .{ "per_layer_proj_norm.weight", prefix ++ "per_layer_projection_norm.weight" },
    };
    for (globals) |g| if (std.mem.eql(u8, name, g[0])) return .{ .name = g[1], .transform = .none };
    // The proportional-rope frequency table, mlx-serve builds its own from the config.
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
        .{ "layer_output_scale.weight", "layer_scalar" },
        .{ "inp_gate.weight", "per_layer_input_gate.weight" },
        .{ "proj.weight", "per_layer_projection.weight" },
        .{ "post_norm.weight", "post_per_layer_input_norm.weight" },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, prefix ++ "layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = .none };
    };
    return null;
}

test "mapTensor renames layers, PLE tensors and skips the rope table" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 4;
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("language_model.model.layers.3.layer_scalar", mapTensor(&buf, "blk.3.layer_output_scale.weight", hp).?.name);
    try std.testing.expectEqualStrings("language_model.model.layers.0.pre_feedforward_layernorm.weight", mapTensor(&buf, "blk.0.ffn_norm.weight", hp).?.name);
    try std.testing.expectEqualStrings("language_model.model.layers.1.per_layer_input_gate.weight", mapTensor(&buf, "blk.1.inp_gate.weight", hp).?.name);
    try std.testing.expectEqualStrings("language_model.model.embed_tokens_per_layer.weight", mapTensor(&buf, "per_layer_token_embd.weight", hp).?.name);
    try std.testing.expectEqualStrings("", mapTensor(&buf, "rope_freqs.weight", hp).?.name);
    try std.testing.expect(mapTensor(&buf, "blk.4.attn_q.weight", hp) == null);
    try std.testing.expect(mapTensor(&buf, "blk.0.ffn_gate_inp.weight", hp) == null);
}

test "configJson is valid JSON for the PLE and the unified flavour" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 6;
    hp.sliding = @splat(true);
    hp.sliding[5] = false;
    hp.ple_dim = 256;
    hp.vocab = 1000;
    hp.double_wide = true;
    hp.rms_eps = 1e-6;
    hp.eos.add(106);
    hp.eos.add(1);
    hp.eos.add(106);

    const a = std.testing.allocator;
    const json = try configJson(a, hp);
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("gemma4", parsed.value.object.get("model_type").?.string);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.object.get("eos_token_id").?.array.items.len);
    const tc = parsed.value.object.get("text_config").?.object;
    try std.testing.expectEqualStrings("gemma4_text", tc.get("model_type").?.string);
    try std.testing.expectEqualStrings("full_attention", tc.get("layer_types").?.array.items[5].string);
    try std.testing.expectEqual(@as(i64, 1000), tc.get("vocab_size_per_layer_input").?.integer);
    try std.testing.expectEqualStrings("proportional", tc.get("rope_parameters").?.object.get("full_attention").?.object.get("rope_type").?.string);

    hp.ple_dim = 0;
    hp.k_eq_v = true;
    const unified = try configJson(a, hp);
    defer a.free(unified);
    const up = try std.json.parseFromSlice(std.json.Value, a, unified, .{});
    defer up.deinit();
    try std.testing.expectEqualStrings("gemma4_unified_text", up.value.object.get("text_config").?.object.get("model_type").?.string);
    try std.testing.expect(up.value.object.get("text_config").?.object.get("attention_k_eq_v").?.bool);
}
