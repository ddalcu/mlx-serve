//! The KDA-hybrid plugins' tiny random packs, MLX-free: a GLM-5.3-Flash miniature (KDA at 128-wide heads, the
//! release's sparse attention geometry (64 heads of 256 over a 512 latent, a 4-token pool), 8 experts) and a Ling 3.0 one (KDA + MLA, grouped
//! routing), every linear 4-bit affine in groups of 64, norms bf16, the reference's f32 tables f32.

const std = @import("std");
const sdk = @import("../../sdk/root.zig");
const Tensors = @import("../../sdk/testing/tensors.zig").Tensors;

const File = sdk.fixture.File;

/// A 4-bit, group-64 affine linear `<base>.{weight,scales,biases}` over `in` inputs, rows `lead`.
pub fn q4(t: *Tensors, comptime fmt: []const u8, args: anytype, lead: []const u64, in: u64) !void {
    var shape: [3]u64 = undefined;
    @memcpy(shape[0..lead.len], lead);
    shape[lead.len] = in / 8;
    try t.words(fmt ++ ".weight", args, shape[0 .. lead.len + 1]);
    shape[lead.len] = in / 64;
    // Scales near 1/16 of a code, biases centering the codes: weights within about +-0.5 / sqrt(in) scale.
    const amp: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
    try t.fill(fmt ++ ".scales", args, shape[0 .. lead.len + 1], .BF16, amp / 8.0);
    try t.fill(fmt ++ ".biases", args, shape[0 .. lead.len + 1], .BF16, -amp);
}

fn norm(t: *Tensors, comptime fmt: []const u8, args: anytype, n: u64) !void {
    try t.fill(fmt, args, &.{n}, .BF16, 1.0);
}

// ── GLM-5.3-Flash ──

pub const glm = struct {
    pub const H = 256;
    pub const V = 16384;
    pub const LAYERS = 4;
    pub const HEADS = 64;
    pub const HD = 256;
    pub const LATENT = 512;
    pub const Q_LORA = 128;
    pub const IH = 32;
    pub const IHD = 128;
    pub const KPOOL = 4;
    pub const HV = 2;
    pub const DK = 128;
    pub const E = 8;
    pub const M = 128;
    pub const INTER = 256;
};

pub const glm_config =
    \\{"model_type":"glm5_next_text","vocab_size":16384,"hidden_size":256,"intermediate_size":256,"num_hidden_layers":4,
    \\ "num_attention_heads":64,"num_key_value_heads":64,"head_dim":0,"rms_norm_eps":1e-5,"max_position_embeddings":4096,
    \\ "q_lora_rank":128,"kv_lora_rank":512,"qk_nope_head_dim":256,"qk_rope_head_dim":0,"v_head_dim":256,
    \\ "index_topk":8,"index_kpool":4,"index_n_heads":32,"index_head_dim":128,"index_kpool_compress":true,
    \\ "n_routed_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":128,"n_shared_experts":1,
    \\ "routed_scaling_factor":2.5,"n_group":1,"topk_group":1,"norm_topk_prob":true,"swiglu_limit":10.0,
    \\ "mlp_layer_types":["dense","sparse","sparse","sparse"],
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","deepseek_sparse_attention"],
    \\ "linear_attn_config":{"num_heads":2,"head_dim":128,"short_conv_kernel_size":4},
    \\ "hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-6,"eos_token_id":[1],"tie_word_embeddings":false,"image_token_id":500,
    \\ "quantization":{"group_size":64,"bits":4}}
;

fn hc(t: *Tensors, li: usize, comptime which: []const u8) !void {
    try t.rand("model.layers.{d}." ++ which ++ ".fn", .{li}, &.{ 24, 4 * glm.H }, .BF16, 0.02);
    try t.rand("model.layers.{d}." ++ which ++ ".base", .{li}, &.{24}, .F32, 0.5);
    try t.fill("model.layers.{d}." ++ which ++ ".scale", .{li}, &.{3}, .F32, 0.5);
}

fn routed(t: *Tensors, comptime base: []const u8, args: anytype, E: u64, M: u64, H: u64, exl3: bool) !void {
    if (exl3) {
        try exl3Bank(t, base ++ ".switch_mlp.gate_proj", args, E, H, M);
        try exl3Bank(t, base ++ ".switch_mlp.up_proj", args, E, H, M);
        try exl3Bank(t, base ++ ".switch_mlp.down_proj", args, E, M, H);
    } else {
        try q4(t, base ++ ".switch_mlp.gate_proj", args, &.{ E, M }, H);
        try q4(t, base ++ ".switch_mlp.up_proj", args, &.{ E, M }, H);
        try q4(t, base ++ ".switch_mlp.down_proj", args, &.{ E, H }, M);
    }
    try q4(t, base ++ ".shared_experts.gate_proj", args, &.{M}, H);
    try q4(t, base ++ ".shared_experts.up_proj", args, &.{M}, H);
    try q4(t, base ++ ".shared_experts.down_proj", args, &.{H}, M);
}

/// One EXL3 expert bank as a Sushi pack stores it: `trellis` codes at `EXL3_N` halfwords per 16x16 tile and the
/// input/output sign vectors `suh`/`svh` (f16, +-1/8).
fn exl3Bank(t: *Tensors, comptime base: []const u8, args: anytype, E: u64, in: u64, out: u64) !void {
    try t.halfwords(base ++ ".trellis", args, &.{ E, in / 16, out / 16, EXL3_N });
    try t.values(base ++ ".suh", args, &.{ E, in }, .F16, try signs(t, E * in));
    try t.values(base ++ ".svh", args, &.{ E, out }, .F16, try signs(t, E * out));
}

fn signs(t: *Tensors, n: u64) ![]const f32 {
    const vs = try t.a.alloc(f32, n);
    for (vs) |*v| v.* = if (t.prng.random().boolean()) 0.125 else -0.125;
    return vs;
}

/// Halfwords per tile at the Sushi GLM pack's rate (k 2.5).
const EXL3_N = 40;

/// A sparse-attention layer and its routed MoE under `base` (a trunk layer, or the MTP head's block).
fn dsaMoe(t: *Tensors, comptime base: []const u8, args: anytype, exl3: bool) !void {
    const g = glm;
    try q4(t, base ++ ".self_attn.q_a_proj", args, &.{g.Q_LORA}, g.H);
    try norm(t, base ++ ".self_attn.q_a_layernorm.weight", args, g.Q_LORA);
    try q4(t, base ++ ".self_attn.q_b_proj", args, &.{g.HEADS * g.HD}, g.Q_LORA);
    try q4(t, base ++ ".self_attn.kv_a_proj_with_mqa", args, &.{g.LATENT}, g.H);
    try norm(t, base ++ ".self_attn.kv_a_layernorm.weight", args, g.LATENT);
    try q4(t, base ++ ".self_attn.embed_q", args, &.{ g.HEADS, g.LATENT }, g.HD);
    try q4(t, base ++ ".self_attn.unembed_out", args, &.{ g.HEADS, g.HD }, g.LATENT);
    try q4(t, base ++ ".self_attn.o_proj", args, &.{g.H}, g.HEADS * g.HD);
    try q4(t, base ++ ".self_attn.indexer.wq_b", args, &.{g.IH * g.IHD}, g.Q_LORA);
    try q4(t, base ++ ".self_attn.indexer.wk", args, &.{g.IHD}, g.H);
    try norm(t, base ++ ".self_attn.indexer.k_norm.weight", args, g.IHD);
    try t.rand(base ++ ".self_attn.indexer.k_norm.bias", args, &.{g.IHD}, .BF16, 0.05);
    try q4(t, base ++ ".self_attn.indexer.weights_proj", args, &.{g.IH}, g.H);
    try t.rand(base ++ ".self_attn.indexer.index_kpool_compress_gate", args, &.{ g.IHD, g.H }, .BF16, 0.05);
    try t.rand(base ++ ".self_attn.indexer.index_kpool_compress_ape", args, &.{ g.KPOOL, g.IHD }, .BF16, 0.5);
    try t.rand(base ++ ".mlp.gate.weight", args, &.{ g.E, g.H }, .BF16, 0.1);
    try t.rand(base ++ ".mlp.gate.e_score_correction_bias", args, &.{g.E}, .F32, 0.1);
    try routed(t, base ++ ".mlp", args, g.E, g.M, g.H, exl3);
}

fn dense(t: *Tensors, li: usize, inter: u64, H: u64) !void {
    try q4(t, "model.layers.{d}.mlp.gate_proj", .{li}, &.{inter}, H);
    try q4(t, "model.layers.{d}.mlp.up_proj", .{li}, &.{inter}, H);
    try q4(t, "model.layers.{d}.mlp.down_proj", .{li}, &.{H}, inter);
}

pub fn glmTensors(a: std.mem.Allocator, seed: u64, exl3: bool) ![]u8 {
    const g = glm;
    var t = Tensors.init(a, seed);
    try q4(&t, "model.embed_tokens", .{}, &.{g.V}, g.H);
    try norm(&t, "model.norm.weight", .{}, g.H);
    try q4(&t, "lm_head", .{}, &.{g.V}, g.H);
    for (0..g.LAYERS) |li| {
        try norm(&t, "model.layers.{d}.input_layernorm.weight", .{li}, g.H);
        try norm(&t, "model.layers.{d}.post_attention_layernorm.weight", .{li}, g.H);
        try hc(&t, li, "attn_hc");
        try hc(&t, li, "ffn_hc");
        if (li == 3) {
            try dsaMoe(&t, "model.layers.{d}", .{li}, exl3);
        } else {
            const C = g.HV * g.DK;
            inline for (.{ "q", "k", "v" }) |p| {
                try q4(&t, "model.layers.{d}.self_attn." ++ p ++ "_proj", .{li}, &.{C}, g.H);
                try t.rand("model.layers.{d}.self_attn." ++ p ++ "_conv1d.weight", .{li}, &.{ C, 1, 4 }, .BF16, 0.3);
            }
            try q4(&t, "model.layers.{d}.self_attn.f_a_proj", .{li}, &.{g.DK}, g.H);
            try q4(&t, "model.layers.{d}.self_attn.f_b_proj", .{li}, &.{C}, g.DK);
            try q4(&t, "model.layers.{d}.self_attn.g_a_proj", .{li}, &.{g.DK}, g.H);
            try q4(&t, "model.layers.{d}.self_attn.g_b_proj", .{li}, &.{C}, g.DK);
            try q4(&t, "model.layers.{d}.self_attn.b_proj", .{li}, &.{g.HV}, g.H);
            try q4(&t, "model.layers.{d}.self_attn.o_proj", .{li}, &.{g.H}, C);
            try norm(&t, "model.layers.{d}.self_attn.o_norm.weight", .{li}, g.DK);
            try t.rand("model.layers.{d}.self_attn.A_log", .{li}, &.{g.HV}, .F32, 0.5);
            try t.rand("model.layers.{d}.self_attn.dt_bias", .{li}, &.{C}, .F32, 0.5);
        }
        if (li == 0) {
            try dense(&t, li, g.INTER, g.H);
        } else if (li != 3) {
            try t.rand("model.layers.{d}.mlp.gate.weight", .{li}, &.{ g.E, g.H }, .BF16, 0.1);
            try t.rand("model.layers.{d}.mlp.gate.e_score_correction_bias", .{li}, &.{g.E}, .F32, 0.1);
            try routed(&t, "model.layers.{d}.mlp", .{li}, g.E, g.M, g.H, exl3);
        }
    }
    // The MTP head (`mtp.0`): eh_proj over [enorm(embed), hnorm(hidden)], one plain DSA + MoE block, its norm.
    try norm(&t, "model.mtp.0.enorm.weight", .{}, g.H);
    try norm(&t, "model.mtp.0.hnorm.weight", .{}, g.H);
    try norm(&t, "model.mtp.0.norm.weight", .{}, g.H);
    try q4(&t, "model.mtp.0.eh_proj", .{}, &.{g.H}, 2 * g.H);
    try norm(&t, "model.mtp.0.block.input_layernorm.weight", .{}, g.H);
    try norm(&t, "model.mtp.0.block.post_attention_layernorm.weight", .{}, g.H);
    try dsaMoe(&t, "model.mtp.0.block", .{}, exl3);
    return t.bytes();
}

pub fn glmFiles(a: std.mem.Allocator, seed: u64) ![]const File {
    const files = try a.alloc(File, 2);
    files[0] = .{ .name = "config.json", .bytes = glm_config };
    files[1] = .{ .name = "model.safetensors", .bytes = try glmTensors(a, seed, false) };
    return files;
}

/// The Sushi pack's `expert_quant` block (`beamster/GLM-5.3-Flash-Sushi-*`).
pub const sushi_config = "{\"expert_quant\":{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}," ++ glm_config[1..];

/// The same pack with its routed experts in EXL3, as a Sushi pack stores them.
pub fn glmSushiFiles(a: std.mem.Allocator, seed: u64) ![]const File {
    const files = try a.alloc(File, 2);
    files[0] = .{ .name = "config.json", .bytes = sushi_config };
    files[1] = .{ .name = "model.safetensors", .bytes = try glmTensors(a, seed, true) };
    return files;
}

// ── Ling 3.0 ──

pub const ling = struct {
    pub const H = 256;
    pub const V = 512;
    pub const LAYERS = 4;
    pub const HEADS = 4;
    pub const HD = 128;
    pub const NOPE = 64;
    pub const ROPE = 64;
    pub const VD = 64;
    pub const Q_LORA = 64;
    pub const KV_LORA = 64;
    pub const E = 8;
    pub const M = 64;
    pub const INTER = 256;
};

pub const ling_config =
    \\{"model_type":"bailing_hybrid","vocab_size":512,"hidden_size":256,"intermediate_size":256,"num_hidden_layers":4,
    \\ "num_attention_heads":4,"num_key_value_heads":4,"head_dim":128,"rms_norm_eps":1e-6,"rope_theta":600000,
    \\ "max_position_embeddings":4096,"layer_group_size":4,"short_conv_kernel_size":4,"kda_lower_bound":-5,
    \\ "no_kda_lora":true,"q_lora_rank":64,"kv_lora_rank":64,"qk_nope_head_dim":64,"qk_rope_head_dim":64,
    \\ "qk_head_dim":128,"v_head_dim":64,"gated_attention_proj_granularity_type":"head_wise","rope_interleave":true,
    \\ "first_k_dense_replace":1,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":64,
    \\ "num_shared_experts":1,"moe_shared_expert_intermediate_size":64,"n_group":4,"topk_group":2,
    \\ "norm_topk_prob":true,"routed_scaling_factor":2.5,"score_function":"sigmoid","eos_token_id":1,
    \\ "tie_word_embeddings":false,"quantization":{"group_size":64,"bits":4}}
;

pub fn lingTensors(a: std.mem.Allocator, seed: u64) ![]u8 {
    const g = ling;
    var t = Tensors.init(a, seed);
    try q4(&t, "model.word_embeddings", .{}, &.{g.V}, g.H);
    try norm(&t, "model.norm.weight", .{}, g.H);
    try q4(&t, "lm_head", .{}, &.{g.V}, g.H);
    for (0..g.LAYERS) |li| {
        try norm(&t, "model.layers.{d}.input_layernorm.weight", .{li}, g.H);
        try norm(&t, "model.layers.{d}.post_attention_layernorm.weight", .{li}, g.H);
        if (li == 3) {
            const qk = g.NOPE + g.ROPE;
            try q4(&t, "model.layers.{d}.attention.q_a_proj", .{li}, &.{g.Q_LORA}, g.H);
            try norm(&t, "model.layers.{d}.attention.q_a_layernorm.weight", .{li}, g.Q_LORA);
            try q4(&t, "model.layers.{d}.attention.q_b_proj", .{li}, &.{g.HEADS * qk}, g.Q_LORA);
            try q4(&t, "model.layers.{d}.attention.kv_a_proj_with_mqa", .{li}, &.{g.KV_LORA + g.ROPE}, g.H);
            try norm(&t, "model.layers.{d}.attention.kv_a_layernorm.weight", .{li}, g.KV_LORA);
            try q4(&t, "model.layers.{d}.attention.kv_b_proj", .{li}, &.{g.HEADS * (g.NOPE + g.VD)}, g.KV_LORA);
            try q4(&t, "model.layers.{d}.attention.dense", .{li}, &.{g.H}, g.HEADS * g.VD);
            try q4(&t, "model.layers.{d}.attention.g_proj", .{li}, &.{g.HEADS}, g.H);
        } else {
            const C = g.HEADS * g.HD;
            inline for (.{ "q", "k", "v" }) |p| {
                try q4(&t, "model.layers.{d}.attention." ++ p ++ "_proj", .{li}, &.{C}, g.H);
                try t.rand("model.layers.{d}.attention." ++ p ++ "_conv1d.weight", .{li}, &.{ C, 1, 4 }, .BF16, 0.3);
            }
            try q4(&t, "model.layers.{d}.attention.f_proj", .{li}, &.{C}, g.H);
            try q4(&t, "model.layers.{d}.attention.g_proj", .{li}, &.{C}, g.H);
            try q4(&t, "model.layers.{d}.attention.b_proj", .{li}, &.{g.HEADS}, g.H);
            try q4(&t, "model.layers.{d}.attention.o_proj", .{li}, &.{g.H}, C);
            try norm(&t, "model.layers.{d}.attention.o_norm.weight", .{li}, g.HD);
            try t.rand("model.layers.{d}.attention.A_log", .{li}, &.{g.HEADS}, .F32, 0.5);
            try t.rand("model.layers.{d}.attention.dt_bias", .{li}, &.{C}, .F32, 0.5);
        }
        if (li == 0) {
            try dense(&t, li, g.INTER, g.H);
        } else {
            try t.rand("model.layers.{d}.mlp.gate.weight", .{li}, &.{ g.E, g.H }, .BF16, 0.1);
            try t.rand("model.layers.{d}.mlp.gate.expert_bias", .{li}, &.{g.E}, .F32, 0.1);
            try routed(&t, "model.layers.{d}.mlp", .{li}, g.E, g.M, g.H, false);
        }
    }
    return t.bytes();
}

pub fn lingFiles(a: std.mem.Allocator, seed: u64) ![]const File {
    const files = try a.alloc(File, 2);
    files[0] = .{ .name = "config.json", .bytes = ling_config };
    files[1] = .{ .name = "model.safetensors", .bytes = try lingTensors(a, seed) };
    return files;
}
