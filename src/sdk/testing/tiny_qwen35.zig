//! A fixture's tiny qwen3_5 trunk (the decision packs Kev and Clef ride one): two layers, a GatedDeltaNet layer then
//! gated attention, at the geometry the fused kernels take (hd 256 attention, 128-wide linear-attention heads).
//! MLX-free: the config and the weights as `Tensors` entries.

const std = @import("std");
const tensors = @import("tensors.zig");

pub const hidden = 256;
pub const vocab = 264;
const inter = 512;
const heads = 4;
const kv_heads = 2;
const hd = 256;
const hk = 2;
const hv = 8;
const dk = 128;
const dv = 128;

pub const config =
    \\{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5","hidden_size":256,"intermediate_size":512,
    \\ "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":256,"vocab_size":264,
    \\ "rms_norm_eps":1e-6,"full_attention_interval":2,"linear_num_key_heads":2,"linear_num_value_heads":8,
    \\ "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"max_position_embeddings":4096,
    \\ "rope_parameters":{"rope_theta":100000.0,"partial_rotary_factor":0.25},"tie_word_embeddings":false},
    \\ "tie_word_embeddings":false}
;

/// The trunk's weights (bf16) under `language_model.`, as mlx-lm names them.
pub fn weights(t: *tensors.Tensors) !void {
    const p = "language_model.model.";
    try t.rand(p ++ "embed_tokens.weight", .{}, &.{ vocab, hidden }, .BF16, 0.5);
    try t.rand(p ++ "norm.weight", .{}, &.{hidden}, .BF16, 0.1);
    try t.rand("language_model.lm_head.weight", .{}, &.{ vocab, hidden }, .BF16, 0.1);
    for (0..2) |i| {
        try t.rand(p ++ "layers.{d}.input_layernorm.weight", .{i}, &.{hidden}, .BF16, 0.1);
        try t.rand(p ++ "layers.{d}.post_attention_layernorm.weight", .{i}, &.{hidden}, .BF16, 0.1);
        try t.rand(p ++ "layers.{d}.mlp.gate_proj.weight", .{i}, &.{ inter, hidden }, .BF16, 0.06);
        try t.rand(p ++ "layers.{d}.mlp.up_proj.weight", .{i}, &.{ inter, hidden }, .BF16, 0.06);
        try t.rand(p ++ "layers.{d}.mlp.down_proj.weight", .{i}, &.{ hidden, inter }, .BF16, 0.04);
    }
    const qkv = 2 * hk * dk + hv * dv;
    try t.rand(p ++ "layers.0.linear_attn.in_proj_qkv.weight", .{}, &.{ qkv, hidden }, .BF16, 0.06);
    try t.rand(p ++ "layers.0.linear_attn.in_proj_z.weight", .{}, &.{ hv * dv, hidden }, .BF16, 0.06);
    try t.rand(p ++ "layers.0.linear_attn.in_proj_a.weight", .{}, &.{ hv, hidden }, .BF16, 0.06);
    try t.rand(p ++ "layers.0.linear_attn.in_proj_b.weight", .{}, &.{ hv, hidden }, .BF16, 0.06);
    try t.rand(p ++ "layers.0.linear_attn.conv1d.weight", .{}, &.{ qkv, 4, 1 }, .BF16, 0.5);
    try t.rand(p ++ "layers.0.linear_attn.A_log", .{}, &.{hv}, .BF16, 0.5);
    try t.rand(p ++ "layers.0.linear_attn.dt_bias", .{}, &.{hv}, .BF16, 0.5);
    try t.rand(p ++ "layers.0.linear_attn.norm.weight", .{}, &.{dv}, .BF16, 0.1);
    try t.rand(p ++ "layers.0.linear_attn.out_proj.weight", .{}, &.{ hidden, hv * dv }, .BF16, 0.03);
    try t.rand(p ++ "layers.1.self_attn.q_proj.weight", .{}, &.{ 2 * heads * hd, hidden }, .BF16, 0.06);
    try t.rand(p ++ "layers.1.self_attn.k_proj.weight", .{}, &.{ kv_heads * hd, hidden }, .BF16, 0.06);
    try t.rand(p ++ "layers.1.self_attn.v_proj.weight", .{}, &.{ kv_heads * hd, hidden }, .BF16, 0.06);
    try t.rand(p ++ "layers.1.self_attn.o_proj.weight", .{}, &.{ hidden, heads * hd }, .BF16, 0.03);
    try t.rand(p ++ "layers.1.self_attn.q_norm.weight", .{}, &.{hd}, .BF16, 0.1);
    try t.rand(p ++ "layers.1.self_attn.k_norm.weight", .{}, &.{hd}, .BF16, 0.1);
}
