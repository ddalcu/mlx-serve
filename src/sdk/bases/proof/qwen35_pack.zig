//! A tiny qwen3_5 / qwen3_5_moe pack (hd-128 linear-attention heads, hd-256 attention) for the plugin proofs.

const std = @import("std");
const fixture = @import("fixture.zig");

const H: c_int = 256;
const VOCAB: c_int = 256;
const LAYERS = 4;
const HEADS: c_int = 4;
const KV_HEADS: c_int = 2;
const HD: c_int = 256;
const INTER: c_int = 512;
const HK: c_int = 2;
const HV: c_int = 8;
const DK: c_int = 128;
const DV: c_int = 128;
const KERNEL: c_int = 4;
const MOE_INTER: c_int = 128;

pub fn config(comptime moe_pack: bool, comptime quantized: bool, comptime experts: c_int) []const u8 {
    const mt = if (moe_pack) "qwen3_5_moe" else "qwen3_5";
    const moe_keys = if (moe_pack) "\"num_experts\":" ++ std.fmt.comptimePrint("{d}", .{experts}) ++ ",\"num_experts_per_tok\":2,\"moe_intermediate_size\":128,\"shared_expert_intermediate_size\":128," else "";
    return "{\"model_type\":\"" ++ mt ++ "\",\"text_config\":{\"model_type\":\"" ++ mt ++ "\",\"hidden_size\":256,\"intermediate_size\":512," ++
        "\"num_hidden_layers\":4,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":256,\"vocab_size\":256," ++ moe_keys ++
        "\"rms_norm_eps\":1e-6,\"full_attention_interval\":2,\"linear_num_key_heads\":2,\"linear_num_value_heads\":8," ++
        "\"linear_key_head_dim\":128,\"linear_value_head_dim\":128,\"linear_conv_kernel_dim\":4,\"max_position_embeddings\":4096," ++
        "\"rope_parameters\":{\"rope_theta\":100000.0,\"partial_rotary_factor\":0.25},\"tie_word_embeddings\":false}," ++
        "\"tie_word_embeddings\":false" ++ (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}}" else "}");
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.{s}", .{ layer, name });
}

/// Linear-attention layers 0 and 2, gated attention 1 and 3; 4-bit or bf16 (dense attention: decode reads the
/// requantized side copies). The shared-expert gate stays bf16.
pub fn writePack(p: *fixture.Pack, comptime moe_pack: bool, q: bool, experts: c_int) !void {
    var buf: [160]u8 = undefined;
    try p.weight("language_model.model.embed_tokens", VOCAB, H, q);
    try p.norm("language_model.model.norm.weight", H);
    try p.weight("language_model.lm_head", VOCAB, H, q);
    for (0..LAYERS) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        if (moe_pack) {
            try p.weight(try at(&buf, i, "mlp.gate"), experts, H, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.gate_proj"), &.{ experts, MOE_INTER, H }, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.up_proj"), &.{ experts, MOE_INTER, H }, q);
            try p.tensor(try at(&buf, i, "mlp.switch_mlp.down_proj"), &.{ experts, H, MOE_INTER }, q);
            try p.weight(try at(&buf, i, "mlp.shared_expert.gate_proj"), MOE_INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.shared_expert.up_proj"), MOE_INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.shared_expert.down_proj"), H, MOE_INTER, q);
            try p.weight(try at(&buf, i, "mlp.shared_expert_gate"), 1, H, false);
        } else {
            try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.down_proj"), H, INTER, q);
        }
        if (i % 2 == 0) {
            try p.weight(try at(&buf, i, "linear_attn.in_proj_qkv"), 2 * HK * DK + HV * DV, H, q);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_z"), HV * DV, H, q);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_a"), HV, H, q);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_b"), HV, H, q);
            try p.dense(try at(&buf, i, "linear_attn.conv1d.weight"), &.{ 2 * HK * DK + HV * DV, KERNEL, 1 }, 0.5, 0);
            try p.dense(try at(&buf, i, "linear_attn.A_log"), &.{HV}, 0.5, 0.2);
            try p.dense(try at(&buf, i, "linear_attn.dt_bias"), &.{HV}, 0.5, 0);
            try p.norm(try at(&buf, i, "linear_attn.norm.weight"), DV);
            try p.weight(try at(&buf, i, "linear_attn.out_proj"), H, HV * DV, q);
        } else {
            try p.weight(try at(&buf, i, "self_attn.q_proj"), 2 * HEADS * HD, H, q);
            try p.weight(try at(&buf, i, "self_attn.k_proj"), KV_HEADS * HD, H, q);
            try p.weight(try at(&buf, i, "self_attn.v_proj"), KV_HEADS * HD, H, q);
            try p.weight(try at(&buf, i, "self_attn.o_proj"), H, HEADS * HD, q);
            try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), HD);
            try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), HD);
        }
    }
}
