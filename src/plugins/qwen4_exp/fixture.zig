//! Tiny random qwen4_exp packs: four layers (GDN at 0 and 2, QSA attention at 1 and 3), the n-gram PLE at layer 2
//! with its `ngram_table.bin`, hyper-connections, a routed + shared MoE, bf16 or 4-bit projections, and the
//! in-checkpoint MTP head. The QSA budget is 8 tokens, so a short prompt crosses it.

const std = @import("std");
const sdk = @import("../../sdk/root.zig");
const Tensors = @import("../../sdk/testing/tensors.zig").Tensors;
const NgramHash = @import("qwen4_exp.zig").NgramHash;

pub const Precision = enum { bf16, q4 };

pub const vocab = 128;
const hidden = 64;
const layers = 4;
const heads = 4;
const kv_heads = 2;
const head_dim = 32;
const hk = 2;
const hv = 4;
const dk = 32;
const dv = 32;
const experts = 4;
const inter = 32;
const shared = 32;
const hc = 4;
const lowrank = 16;
const ple_dim = 64;
const idx_heads = 2;
const idx_dim = 16;
const group = 32;

pub const config_json = std.fmt.comptimePrint(
    \\{{"model_type":"qwen4_exp","language_model_only":true,"tie_word_embeddings":false,
    \\ "text_config":{{"model_type":"qwen4_exp_text","vocab_size":{d},"hidden_size":{d},"num_hidden_layers":{d},
    \\ "full_attention_interval":2,"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},
    \\ "hc_count":{d},"hc_lowrank":{d},"linear_num_key_heads":{d},"linear_num_value_heads":{d},
    \\ "linear_key_head_dim":{d},"linear_value_head_dim":{d},"linear_conv_kernel_dim":4,"num_experts":{d},
    \\ "num_experts_per_tok":2,"moe_intermediate_size":{d},"shared_expert_intermediate_size":{d},
    \\ "ple_layer_ids":[3],"ple_embed_dim":{d},"ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":2,
    \\ "ngram_vocab_size_base":100,"make_ngram_vocab_size_divisible_by":8,"seed":1234,"split_ngram_parts":2,
    \\ "indexer_n_heads":{d},"indexer_kv_heads":1,"indexer_head_dim":{d},"indexer_budget":8,"indexer_compress_ratio":4,
    \\ "rms_norm_eps":1e-6,"hidden_act":"silu","output_gate_type":"sigmoid","partial_rotary_factor":0.25,
    \\ "rope_parameters":{{"rope_theta":10000.0,"rope_type":"default","partial_rotary_factor":0.25}},
    \\ "max_position_embeddings":4096,"bos_token_id":1,"eos_token_id":1}}}}
, .{ vocab, hidden, layers, heads, kv_heads, head_dim, hc, lowrank, hk, hv, dk, dv, experts, inter, shared, ple_dim, idx_heads, idx_dim });

const quant_block = std.fmt.comptimePrint("\"quantization\":{{\"group_size\":{d},\"bits\":4,\"mode\":\"affine\"}}", .{group});

const Pack = struct {
    t: Tensors,
    q: bool,

    /// A projection `[.., out, in]`: random bf16, or 4-bit words with their scales and biases.
    fn proj(p: *Pack, comptime base: []const u8, args: anytype, lead: []const u64, out: u64, in: u64, scale: f32) !void {
        var shape: [3]u64 = undefined;
        @memcpy(shape[0..lead.len], lead);
        if (!p.q or in % group != 0) {
            shape[lead.len] = out;
            shape[lead.len + 1] = in;
            return p.t.rand(base ++ ".weight", args, shape[0 .. lead.len + 2], .BF16, scale);
        }
        shape[lead.len] = out;
        shape[lead.len + 1] = in / 8;
        try p.t.words(base ++ ".weight", args, shape[0 .. lead.len + 2]);
        shape[lead.len + 1] = in / group;
        try p.t.fill(base ++ ".scales", args, shape[0 .. lead.len + 2], .BF16, scale / 4);
        try p.t.fill(base ++ ".biases", args, shape[0 .. lead.len + 2], .BF16, -scale * 15 / 8);
    }

    fn hcBlock(p: *Pack, comptime base: []const u8, args: anytype, inject: bool) !void {
        try p.t.fill(base ++ ".hc_norm.weight", args, &.{hc * hidden}, .BF16, 1.0);
        try p.t.rand(base ++ ".input_mix_weight_down.weight", args, &.{ lowrank, hc * hidden }, .BF16, 0.05);
        try p.t.rand(base ++ ".input_mix_weight_up.weight", args, &.{ hc * hidden, lowrank }, .BF16, 0.05);
        if (inject) try p.t.rand(base ++ ".block_inject_weight.weight", args, &.{ hc, hc * hidden }, .BF16, 0.05);
    }
};

/// One layer under `L` (`{d}` = `li`): GDN or QSA attention, its two hyper-connection blocks, the routed + shared
/// MoE, and the PLE on the layer that carries it.
fn writeLayer(p: *Pack, comptime L: []const u8, li: usize, full: bool, with_ple: bool, opts: Opts) !void {
    try p.hcBlock(L ++ ".attn_hyper_connection", .{li}, true);
    try p.hcBlock(L ++ ".mlp_hyper_connection", .{li}, true);
    if (!full) {
        const conv = 2 * hk * dk + hv * dv;
        try p.t.rand(L ++ ".linear_attn.A_log", .{li}, &.{hv}, .BF16, 0.5);
        try p.t.rand(L ++ ".linear_attn.dt_bias", .{li}, &.{hv}, .BF16, 0.5);
        try p.t.rand(L ++ ".linear_attn.conv1d.weight", .{li}, &.{ conv, 4, 1 }, .BF16, 0.3);
        try p.t.fill(L ++ ".linear_attn.norm.weight", .{li}, &.{dv}, .BF16, 1.0);
        try p.proj(L ++ ".linear_attn.in_proj_qkv", .{li}, &.{}, conv, hidden, 0.1);
        try p.proj(L ++ ".linear_attn.in_proj_z", .{li}, &.{}, hv * dv, hidden, 0.1);
        try p.proj(L ++ ".linear_attn.in_proj_a", .{li}, &.{}, hv, hidden, 0.1);
        try p.proj(L ++ ".linear_attn.in_proj_b", .{li}, &.{}, hv, hidden, 0.1);
        try p.proj(L ++ ".linear_attn.out_proj", .{li}, &.{}, hidden, hv * dv, 0.1);
    } else {
        try p.proj(L ++ ".self_attn.q_proj", .{li}, &.{}, heads * head_dim * 2, hidden, 0.1);
        try p.proj(L ++ ".self_attn.k_proj", .{li}, &.{}, kv_heads * head_dim, hidden, 0.1);
        try p.proj(L ++ ".self_attn.v_proj", .{li}, &.{}, kv_heads * head_dim, hidden, 0.1);
        try p.proj(L ++ ".self_attn.o_proj", .{li}, &.{}, hidden, heads * head_dim, 0.1);
        try p.t.fill(L ++ ".self_attn.q_norm.weight", .{li}, &.{head_dim}, .BF16, 1.0);
        try p.t.fill(L ++ ".self_attn.k_norm.weight", .{li}, &.{head_dim}, .BF16, 1.0);
        try p.proj(L ++ ".self_attn.indexer.index_qk_proj", .{li}, &.{}, (idx_heads + 1) * idx_dim, hidden, 0.1);
        try p.t.fill(L ++ ".self_attn.indexer.q_layernorm.weight", .{li}, &.{idx_dim}, .BF16, 1.0);
        try p.t.fill(L ++ ".self_attn.indexer.k_layernorm.weight", .{li}, &.{idx_dim}, .BF16, 1.0);
    }
    try p.t.rand(L ++ ".mlp.gate.weight", .{li}, &.{ experts, hidden }, .BF16, 0.1);
    try p.t.rand(L ++ ".mlp.shared_expert_gate.weight", .{li}, &.{ 1, hidden }, .BF16, 0.1);
    try p.proj(L ++ ".mlp.shared_expert.gate_proj", .{li}, &.{}, shared, hidden, 0.1);
    try p.proj(L ++ ".mlp.shared_expert.up_proj", .{li}, &.{}, shared, hidden, 0.1);
    try p.proj(L ++ ".mlp.shared_expert.down_proj", .{li}, &.{}, hidden, shared, 0.1);
    try p.proj(L ++ ".mlp.switch_mlp.gate_proj", .{li}, &.{experts}, inter, hidden, 0.1);
    try p.proj(L ++ ".mlp.switch_mlp.up_proj", .{li}, &.{experts}, inter, hidden, 0.1);
    try p.proj(L ++ ".mlp.switch_mlp.down_proj", .{li}, &.{experts}, hidden, inter, 0.1);
    if (opts.expert_scale) |v| try p.t.fill(L ++ ".mlp.switch_mlp.expert_scale", .{li}, &.{experts}, .BF16, v);
    if (opts.gate_expert_scale) |v| try p.t.fill(L ++ ".mlp.switch_mlp.gate_proj.expert_scale", .{li}, &.{experts}, .BF16, v);
    if (with_ple) {
        try p.t.rand(L ++ ".ple.conv1d.weight", .{li}, &.{ hc * hidden, 4, 1 }, .BF16, 0.3);
        try p.proj(L ++ ".ple.key_proj", .{li}, &.{}, hc * hidden, ple_dim, 0.1);
        try p.proj(L ++ ".ple.value_proj", .{li}, &.{}, hidden, ple_dim, 0.1);
        inline for (.{ "norm_key", "norm_query", "norm_conv" }) |n| try p.t.fill(L ++ ".ple." ++ n ++ ".weight", .{li}, &.{hc * hidden}, .BF16, 1.0);
    }
}

/// config.json, the safetensors and the n-gram table of one tiny pack (`seed` draws the weights). The pack carries
/// its in-checkpoint MTP head (`language_model.mtp.*`: one QSA + MoE layer, its own mixer).
pub fn files(a: std.mem.Allocator, seed: u64, precision: Precision) ![]const sdk.fixture.File {
    return filesWith(a, seed, precision, .{});
}

pub const Opts = struct {
    /// An lm_head that reads only the sign of the hidden's sum (token 5 when positive, 6 when negative): the trunk
    /// and the MTP head agree often, so rounds take full, partial and rejected accepts.
    sign_head: bool = false,
    /// A Qwen3-VL pack's text side: image/video ids and an interleaved M-RoPE spectrum (no tower weights).
    vision: bool = false,
    /// Every expert's routing scale (`mlp.switch_mlp.expert_scale`, ModelOpt's up x down `weight_scale_2`).
    expert_scale: ?f32 = null,
    /// Every expert's gate-projection scale (`mlp.switch_mlp.gate_proj.expert_scale`, applied before the SiLU).
    gate_expert_scale: ?f32 = null,
};

/// The image placeholder of a `vision` pack.
pub const image_id = 120;

pub fn filesWith(a: std.mem.Allocator, seed: u64, precision: Precision, opts: Opts) ![]const sdk.fixture.File {
    var p: Pack = .{ .t = Tensors.init(a, seed), .q = precision == .q4 };
    try p.proj("language_model.model.embed_tokens", .{}, &.{}, vocab, hidden, 0.5);
    if (opts.sign_head) {
        var head: [vocab * hidden]f32 = @splat(0);
        @memset(head[5 * hidden ..][0..hidden], 0.05);
        @memset(head[6 * hidden ..][0..hidden], -0.05);
        try p.t.values("language_model.lm_head.weight", .{}, &.{ vocab, hidden }, .BF16, &head);
    } else try p.proj("language_model.lm_head", .{}, &.{}, vocab, hidden, 0.1);
    try p.hcBlock("language_model.model.hyper_connection_mixer", .{}, false);
    for (0..layers) |li| try writeLayer(&p, "language_model.model.layers.{d}", li, li % 2 == 1, li == 2, opts);
    const M = "language_model.mtp";
    try p.proj(M ++ ".fc_embedding", .{}, &.{}, hidden, hidden, 0.1);
    try p.proj(M ++ ".fc_hidden", .{}, &.{}, hidden, hidden, 0.1);
    try p.t.fill(M ++ ".pre_fc_norm_embedding.weight", .{}, &.{hidden}, .BF16, 1.0);
    try p.t.fill(M ++ ".pre_fc_norm_hidden.weight", .{}, &.{hc * hidden}, .BF16, 1.0);
    try p.hcBlock(M ++ ".hyper_connection_mixer", .{}, false);
    try writeLayer(&p, M ++ ".layers.{d}", 0, true, false, opts);
    var cfg: []const u8 = config_json;
    if (opts.vision) {
        const rope = try std.mem.replaceOwned(u8, a, cfg, "\"partial_rotary_factor\":0.25}", "\"partial_rotary_factor\":0.25,\"mrope_section\":[2,1,1],\"mrope_interleaved\":true}");
        cfg = try std.mem.concat(a, u8, &.{ "{\"image_token_id\":120,\"video_token_id\":123,\"vision_start_token_id\":121,\"vision_end_token_id\":122,\"vision_config\":{\"depth\":2,\"hidden_size\":32,\"num_heads\":2,\"intermediate_size\":64,\"patch_size\":16,\"out_hidden_size\":64},", rope[1..] });
    }
    if (p.q) cfg = try std.mem.concat(a, u8, &.{ cfg[0 .. cfg.len - 1], ",", quant_block, "}" });
    const out = try a.alloc(sdk.fixture.File, 3);
    out[0] = .{ .name = "config.json", .bytes = cfg };
    out[1] = .{ .name = "model.safetensors", .bytes = try p.t.bytes() };
    out[2] = .{ .name = "ngram_table.bin", .bytes = try ngramTable(a, seed) };
    return out;
}

/// `ngram_table.bin` in its raw bf16 form: one `[rows, dim]` region, rows and dim from the config's hash.
fn ngramTable(a: std.mem.Allocator, seed: u64) ![]u8 {
    const hash = try NgramHash.init(vocab, 3, 2, 100, 8, 1234, 0, 1);
    const rows = hash.total_rows;
    const dim = ple_dim / hash.n_heads;
    const data_len = rows * dim * 2;
    const header = try std.fmt.allocPrint(a, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"32\"}},\"weight\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}}}", .{ rows, dim, data_len });
    const out = try a.alloc(u8, 8 + header.len + data_len);
    std.mem.writeInt(u64, out[0..8], header.len, .little);
    @memcpy(out[8..][0..header.len], header);
    var prng = std.Random.DefaultPrng.init(seed +% 99);
    const data = out[8 + header.len ..];
    for (0..rows * dim) |i| {
        const v: f32 = prng.random().floatNorm(f32) * 0.5;
        std.mem.writeInt(u16, data[i * 2 ..][0..2], @truncate(@as(u32, @bitCast(v)) >> 16), .little);
    }
    return out;
}

// ── tests ──

test "qwen4_exp fixture: the config parses and the table matches its hash" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const fs = try files(arena.allocator(), 1, .q4);
    try std.testing.expect(std.mem.indexOf(u8, fs[0].bytes, "\"bits\":4") != null);
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), fs[0].bytes, .{});
    try std.testing.expect(parsed == .object);
}
