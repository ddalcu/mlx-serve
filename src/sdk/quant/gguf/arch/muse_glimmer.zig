//! Muse-Glimmer-30B (GGUF arch "muse-glimmer"), text only, served by the muse_glimmer plugin as the flat
//! `muse_glimmer_text` config (weights under `model.`). llama.cpp's converter stores the four sandwich norms as HF's
//! weight + 1 (the final norm stays plain), writes `qk_scale_factor` into every q_norm weight (k_norm = ones), and
//! interleaves each Q/K head's rope halves for ggml's NORM rope where HF keeps them rotate-half.
const std = @import("std");
const gguf = @import("../gguf.zig");
const arch = @import("../arch.zig");
const Transform = arch.Transform;

pub const arch_name = "muse-glimmer";

const MAX_LAYERS = 128;

pub const Hparams = struct {
    n_layers: u32,
    hidden: u32,
    ffn: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    sliding_window: u32,
    /// Per layer: sliding and roped, else full attention and NoPE (llama.cpp's graph).
    sliding: [MAX_LAYERS]bool,
    /// The base of every roped layer, all of them sliding.
    rope_theta: f32,
    rms_eps: f32,
    /// 0 = none.
    softcap: f32,
    /// HF's `output_multiplier`.
    logit_scale: f32,
    /// HF's `qk_scale_factor`, the value of every q_norm weight.
    qk_scale: f32,
    ctx_len: u32,
    vocab: u32,
    bos: u32,
    eos: u32,
    eot: u32,
    tied: bool,

    pub fn read(f: *const gguf.File) !Hparams {
        const p = arch_name ++ ".";
        const n_layers = try arch.u32Key(f, p ++ "block_count");
        if (n_layers == 0 or n_layers > MAX_LAYERS) return error.BadMetadata;
        const head_dim = try arch.u32Key(f, p ++ "attention.key_length");
        // The q/k norms map to nothing (the plugin's are weight-less), so they must hold only the scale and ones.
        const qk_scale = try uniform(f, "blk.0.attn_q_norm.weight", head_dim);
        for (0..n_layers) |i| for ([_][]const u8{ "q", "k" }, [_]f32{ qk_scale, 1 }) |qk, want| {
            var nb: [48]u8 = undefined;
            const name = std.fmt.bufPrint(&nb, "blk.{d}.attn_{s}_norm.weight", .{ i, qk }) catch unreachable;
            if (try uniform(f, name, head_dim) != want) return error.BadMetadata;
        };
        const theta = try float(f, p ++ "rope.freq_base", null);
        const eos = try arch.u32Key(f, "tokenizer.ggml.eos_token_id");
        const eot_key = "tokenizer.ggml.eot_token_id";
        return .{
            .n_layers = n_layers,
            .hidden = try arch.u32Key(f, p ++ "embedding_length"),
            .ffn = try arch.u32Key(f, p ++ "feed_forward_length"),
            .n_heads = try arch.u32Key(f, p ++ "attention.head_count"),
            .n_kv_heads = try arch.u32Key(f, p ++ "attention.head_count_kv"),
            .head_dim = head_dim,
            .sliding_window = try arch.u32Key(f, p ++ "attention.sliding_window"),
            .sliding = try slidingLayers(f, n_layers),
            .rope_theta = try float(f, p ++ "rope.freq_base_swa", theta),
            .rms_eps = try float(f, p ++ "attention.layer_norm_rms_epsilon", null),
            .softcap = try float(f, p ++ "final_logit_softcapping", 0),
            .logit_scale = try float(f, p ++ "logit_scale", null),
            .qk_scale = qk_scale,
            .ctx_len = try arch.u32Key(f, p ++ "context_length"),
            .vocab = @intCast((f.getArray("tokenizer.ggml.tokens") orelse return error.MissingMetadata).len),
            .bos = try arch.u32Key(f, "tokenizer.ggml.bos_token_id"),
            .eos = eos,
            .eot = if (f.getInt(eot_key) != null) try arch.u32Key(f, eot_key) else eos,
            .tied = f.tensors.get("output.weight") == null,
        };
    }
};

/// llama.cpp's reading: one bool per layer, else a period (default 4) whose last layer is full attention.
fn slidingLayers(f: *const gguf.File, n_layers: u32) ![MAX_LAYERS]bool {
    const key = arch_name ++ ".attention.sliding_window_pattern";
    var sliding: [MAX_LAYERS]bool = @splat(false);
    if (f.getArray(key)) |arr| {
        if (arr.elem_type != .bool or arr.len != n_layers) return error.BadMetadata;
        for (0..n_layers) |i| sliding[i] = arr.boolean(i);
        return sliding;
    }
    const n: u32 = if (f.getInt(key) != null) try arch.u32Key(f, key) else 4;
    for (0..n_layers) |i| sliding[i] = n == 0 or i % n < n - 1;
    return sliding;
}

/// `key` as f32, `default` when absent (null = required).
fn float(f: *const gguf.File, key: []const u8, default: ?f32) !f32 {
    const v = f.getFloat(key) orelse return default orelse error.MissingMetadata;
    return @floatCast(v);
}

/// The one value every entry of the F32 vector `name` (`n` long) holds.
fn uniform(f: *const gguf.File, name: []const u8, n: u32) !f32 {
    const t = f.tensors.get(name) orelse return error.MissingMetadata;
    if (t.ty != .f32 or t.elems() != n) return error.BadMetadata;
    const v = std.mem.bytesAsSlice(f32, t.data);
    for (v) |x| if (x != v[0]) return error.BadMetadata;
    return v[0];
}

/// The text config of meta-models/Muse-Glimmer-30B, flat. llama.cpp's graph fixes `post_norm_eps` at 1e-8 (no key).
pub fn configJson(allocator: std.mem.Allocator, hp: Hparams) ![]u8 {
    var layer_types: std.ArrayList(u8) = .empty;
    defer layer_types.deinit(allocator);
    var thetas: std.ArrayList(u8) = .empty;
    defer thetas.deinit(allocator);
    for (hp.sliding[0..hp.n_layers], 0..) |swa, i| {
        const sep = if (i > 0) "," else "";
        try layer_types.print(allocator, "{s}\"{s}\"", .{ sep, if (swa) "sliding_attention" else "full_attention" });
        try thetas.print(allocator, "{s}{d}", .{ sep, if (swa) hp.rope_theta else 0 });
    }
    var eot_buf: [16]u8 = undefined;
    const eot = if (hp.eot == hp.eos) "" else std.fmt.bufPrint(&eot_buf, ",{d}", .{hp.eot}) catch unreachable;
    return std.fmt.allocPrint(allocator,
        \\{{"architectures":["MuseGlimmerForCausalLM"],"model_type":"muse_glimmer_text","attention_bias":false,
        \\"bos_token_id":{[bos]d},"eos_token_id":[{[eos]d}{[eot]s}],"final_logit_softcapping":{[softcap]d},
        \\"head_dim":{[head_dim]d},"hidden_activation":"silu","hidden_size":{[hidden]d},"intermediate_size":{[ffn]d},
        \\"layer_rope_theta":[{[thetas]s}],"layer_types":[{[layer_types]s}],"max_position_embeddings":{[ctx_len]d},
        \\"num_attention_heads":{[n_heads]d},"num_hidden_layers":{[n_layers]d},"num_key_value_heads":{[n_kv_heads]d},
        \\"output_multiplier":{[logit_scale]d},"post_norm_eps":1e-8,"qk_scale_factor":{[qk_scale]d},
        \\"rms_norm_eps":{[rms_eps]e},"rope_parameters":{{"rope_theta":{[rope_theta]d},"rope_type":"default"}},
        \\"sliding_window":{[sliding_window]d},"tie_word_embeddings":{[tied]},"vocab_size":{[vocab]d},
        \\"quantization":{{"group_size":32,"bits":4,"mode":"gguf"}}}}
    , .{
        .bos = hp.bos,
        .eos = hp.eos,
        .eot = eot,
        .softcap = hp.softcap,
        .head_dim = hp.head_dim,
        .hidden = hp.hidden,
        .ffn = hp.ffn,
        .thetas = thetas.items,
        .layer_types = layer_types.items,
        .ctx_len = hp.ctx_len,
        .n_heads = hp.n_heads,
        .n_layers = hp.n_layers,
        .n_kv_heads = hp.n_kv_heads,
        .logit_scale = hp.logit_scale,
        .qk_scale = hp.qk_scale,
        .rms_eps = hp.rms_eps,
        .rope_theta = hp.rope_theta,
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
    // Read into `qk_scale` (see Hparams.read).
    for ([_][]const u8{ "attn_q_norm.weight", "attn_k_norm.weight" }) |n| if (std.mem.eql(u8, blk.leaf, n)) return arch.skip;
    const table = [_]struct { []const u8, []const u8, Transform }{
        .{ "attn_q.weight", "self_attn.q_proj.weight", .{ .rope_rows = hp.head_dim } },
        .{ "attn_k.weight", "self_attn.k_proj.weight", .{ .rope_rows = hp.head_dim } },
        .{ "attn_norm.weight", "input_layernorm.weight", .minus_one },
        .{ "attn_v.weight", "self_attn.v_proj.weight", .none },
        .{ "attn_gate.weight", "self_attn.gate_proj.weight", .none },
        .{ "attn_output.weight", "self_attn.o_proj.weight", .none },
        .{ "post_attention_norm.weight", "post_attention_layernorm.weight", .minus_one },
        .{ "ffn_norm.weight", "pre_feedforward_layernorm.weight", .minus_one },
        .{ "ffn_gate.weight", "mlp.gate_proj.weight", .none },
        .{ "ffn_up.weight", "mlp.up_proj.weight", .none },
        .{ "ffn_down.weight", "mlp.down_proj.weight", .none },
        .{ "post_ffw_norm.weight", "post_feedforward_layernorm.weight", .minus_one },
    };
    for (table) |row| if (std.mem.eql(u8, blk.leaf, row[0])) {
        const full = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ blk.layer, row[1] }) catch return null;
        return .{ .name = full, .transform = row[2] };
    };
    return null;
}

/// Muse-Glimmer-30B-UD-Q4_K_XL's metadata (sliding pattern as its scalar period), q/k norms as the converter writes
/// them, and its head.
fn testFile(q_norm: []const f32, k_norm: []const f32) !gguf.File {
    var f: gguf.File = .{ .arena = .init(std.testing.allocator), .version = 3 };
    errdefer f.deinit();
    const a = f.arena.allocator();
    const p = arch_name ++ ".";
    const ints = [_]struct { []const u8, u64 }{
        .{ p ++ "block_count", 52 },                     .{ p ++ "context_length", 131072 },
        .{ p ++ "embedding_length", 6656 },              .{ p ++ "feed_forward_length", 19968 },
        .{ p ++ "attention.head_count", 32 },            .{ p ++ "attention.head_count_kv", 2 },
        .{ p ++ "attention.key_length", 128 },           .{ p ++ "attention.sliding_window", 2048 },
        .{ p ++ "attention.sliding_window_pattern", 4 }, .{ "tokenizer.ggml.bos_token_id", 200000 },
        .{ "tokenizer.ggml.eos_token_id", 200001 },      .{ "tokenizer.ggml.eot_token_id", 200008 },
    };
    for (ints) |kv| try f.kv.put(a, kv[0], .{ .uint = kv[1] });
    const floats = [_]struct { []const u8, f32 }{
        .{ p ++ "rope.freq_base", 500000 },      .{ p ++ "attention.layer_norm_rms_epsilon", 1e-5 },
        .{ p ++ "final_logit_softcapping", 20 }, .{ p ++ "logit_scale", 0.19611613513818404 },
    };
    for (floats) |kv| try f.kv.put(a, kv[0], .{ .float = kv[1] });
    try f.kv.put(a, "tokenizer.ggml.tokens", .{ .array = .{ .elem_type = .string, .len = 202048, .bytes = &.{} } });
    for (0..52) |i| for ([_][]const u8{ "q", "k" }, [_][]const f32{ q_norm, k_norm }) |qk, w| {
        const name = try std.fmt.allocPrint(a, "blk.{d}.attn_{s}_norm.weight", .{ i, qk });
        try f.tensors.put(a, name, .{ .dims = .{ w.len, 1, 1, 1 }, .n_dims = 1, .ty = .f32, .data = std.mem.sliceAsBytes(w) });
    };
    try f.tensors.put(a, "output.weight", .{ .dims = .{ 6656, 202048, 1, 1 }, .n_dims = 2, .ty = .q5_k, .data = &.{} });
    return f;
}

test "Hparams.read takes the period-4 pattern, the q_norm scale and the head from the file" {
    const q: [128]f32 = @splat(3.87);
    const ones: [128]f32 = @splat(1);
    var f = try testFile(&q, &ones);
    defer f.deinit();
    const hp = try Hparams.read(&f);
    try std.testing.expect(hp.sliding[0] and hp.sliding[2] and !hp.sliding[3] and hp.sliding[48] and !hp.sliding[51]);
    try std.testing.expectEqual(@as(f32, 3.87), hp.qk_scale);
    try std.testing.expectEqual(@as(f32, 500000), hp.rope_theta);
    try std.testing.expectEqual(@as(u32, 200008), hp.eot);
    try std.testing.expect(!hp.tied);

    // The bool-per-layer pattern the current converter writes reads the same.
    var pattern: [52]u8 = undefined;
    for (&pattern, 0..) |*b, i| b.* = @intFromBool(i % 4 != 3);
    const key = arch_name ++ ".attention.sliding_window_pattern";
    try f.kv.put(f.arena.allocator(), key, .{ .array = .{ .elem_type = .bool, .len = 52, .bytes = &pattern } });
    try std.testing.expectEqualSlices(bool, hp.sliding[0..52], (try Hparams.read(&f)).sliding[0..52]);
}

test "Hparams.read refuses q/k norms that are more than the scale and ones" {
    var q: [128]f32 = @splat(3.87);
    const ones: [128]f32 = @splat(1);
    q[5] = 3.5;
    var bad_q = try testFile(&q, &ones);
    defer bad_q.deinit();
    try std.testing.expectError(error.BadMetadata, Hparams.read(&bad_q));
    const scale: [128]f32 = @splat(3.87);
    var bad_k = try testFile(&scale, &scale);
    defer bad_k.deinit();
    try std.testing.expectError(error.BadMetadata, Hparams.read(&bad_k));
}

test "mapTensor renames every leaf, shifts the sandwich norms and skips the q/k norms" {
    var hp = std.mem.zeroes(Hparams);
    hp.n_layers = 52;
    hp.head_dim = 128;
    var buf: [128]u8 = undefined;
    const want = [_]struct { []const u8, []const u8, Transform }{
        .{ "token_embd.weight", "model.embed_tokens.weight", .none },
        .{ "output_norm.weight", "model.norm.weight", .none },
        .{ "output.weight", "lm_head.weight", .none },
        .{ "blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight", .minus_one },
        .{ "blk.3.attn_v.weight", "model.layers.3.self_attn.v_proj.weight", .none },
        .{ "blk.3.attn_gate.weight", "model.layers.3.self_attn.gate_proj.weight", .none },
        .{ "blk.48.attn_output.weight", "model.layers.48.self_attn.o_proj.weight", .none },
        .{ "blk.1.post_attention_norm.weight", "model.layers.1.post_attention_layernorm.weight", .minus_one },
        .{ "blk.1.ffn_norm.weight", "model.layers.1.pre_feedforward_layernorm.weight", .minus_one },
        .{ "blk.2.ffn_gate.weight", "model.layers.2.mlp.gate_proj.weight", .none },
        .{ "blk.2.ffn_up.weight", "model.layers.2.mlp.up_proj.weight", .none },
        .{ "blk.2.ffn_down.weight", "model.layers.2.mlp.down_proj.weight", .none },
        .{ "blk.51.post_ffw_norm.weight", "model.layers.51.post_feedforward_layernorm.weight", .minus_one },
        .{ "blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight", .{ .rope_rows = 128 } },
        .{ "blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight", .{ .rope_rows = 128 } },
        .{ "blk.0.attn_q_norm.weight", "", .none },
        .{ "blk.51.attn_k_norm.weight", "", .none },
    };
    for (want) |w| {
        const m = mapTensor(&buf, w[0], hp).?;
        try std.testing.expectEqualStrings(w[1], m.name);
        try std.testing.expectEqual(w[2], m.transform);
    }
    const unknown = [_][]const u8{ "blk.52.attn_v.weight", "rope_freqs.weight" };
    for (unknown) |n| try std.testing.expect(mapTensor(&buf, n, hp) == null);
}

test "configJson is the flat text config with NoPE full layers and the file's scales" {
    const q: [128]f32 = @splat(3.87);
    const ones: [128]f32 = @splat(1);
    var f = try testFile(&q, &ones);
    defer f.deinit();
    const a = std.testing.allocator;
    const json = try configJson(a, try Hparams.read(&f));
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("muse_glimmer_text", o.get("model_type").?.string);
    try std.testing.expect(o.get("text_config") == null and o.get("vision_config") == null);
    const lt = o.get("layer_types").?.array.items;
    const thetas = o.get("layer_rope_theta").?.array.items;
    try std.testing.expectEqual(@as(usize, 52), lt.len);
    try std.testing.expectEqualStrings("full_attention", lt[51].string);
    try std.testing.expectEqualStrings("sliding_attention", lt[50].string);
    try std.testing.expectEqual(@as(i64, 0), thetas[3].integer);
    try std.testing.expectEqual(@as(i64, 500000), thetas[0].integer);
    try std.testing.expectEqual(@as(f32, 3.87), @as(f32, @floatCast(o.get("qk_scale_factor").?.float)));
    try std.testing.expectEqual(@as(f32, 0.19611613513818404), @as(f32, @floatCast(o.get("output_multiplier").?.float)));
    try std.testing.expectEqual(@as(f32, 1e-5), @as(f32, @floatCast(o.get("rms_norm_eps").?.float)));
    try std.testing.expectEqual(@as(usize, 2), o.get("eos_token_id").?.array.items.len);
    try std.testing.expect(!o.get("tie_word_embeddings").?.bool);
    try std.testing.expectEqualStrings("gguf", o.get("quantization").?.object.get("mode").?.string);
}
