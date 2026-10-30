//! Qwen3-Next (80B-A3B): the qwen3_5 family's trunk (`plugins/qwen3_5`'s bases and load) with q|k|v|z and b|a
//! projections combined per key head, flat `model.*` weights and its own pipelined-prefill table. Decodes serial.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const q35 = @import("../qwen3_5/plugin.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &q35.bill,
    .load = &q35.load,
    .unload = &q35.unload,
    .forward = &q35.forward,
    .cache = &.{ .layout = &config.layout },
    .spec = &.{ .head = &q35.head_hooks, .trunk_weight = &q35.trunkWeight },
    .weights = &.{},
};

/// The tiny pack the conformance suite runs: four layers (three GDN, one attention), width 64, eight experts.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v = 64;
    const h = 64;
    const heads = 2;
    const kv_heads = 1;
    const hd = 32;
    const hk = 1;
    const hv = 2;
    const dk = 32;
    const dv = 32;
    const experts = 8;
    const inter = 32;
    const conv = 2 * hk * dk + hv * dv;
    var tensors: std.ArrayList(sdk.fixture.Tensor) = .empty;
    var seed: u64 = 80;
    const put = struct {
        fn f(al: std.mem.Allocator, list: *std.ArrayList(sdk.fixture.Tensor), sd: *u64, name: []const u8, shape: []const u64, scale: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            sd.* += 1;
            try list.append(al, .{ .name = name, .shape = shape, .data = try sdk.fixture.randomF32(al, sd.*, n, scale) });
        }
    }.f;
    const ones = try a.alloc(f32, h);
    @memset(ones, 1);
    try put(a, &tensors, &seed, "model.embed_tokens.weight", &.{ v, h }, 0.5);
    try put(a, &tensors, &seed, "lm_head.weight", &.{ v, h }, 0.1);
    try tensors.append(a, .{ .name = "model.norm.weight", .shape = &.{h}, .data = ones });
    for (0..4) |i| {
        const L = struct {
            fn n(al: std.mem.Allocator, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "model.layers.{d}.{s}", .{ li, leaf });
            }
        };
        try tensors.append(a, .{ .name = try L.n(a, i, "input_layernorm.weight"), .shape = &.{h}, .data = ones });
        try tensors.append(a, .{ .name = try L.n(a, i, "post_attention_layernorm.weight"), .shape = &.{h}, .data = ones });
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.gate.weight"), &.{ experts, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.switch_mlp.gate_proj.weight"), &.{ experts, inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.switch_mlp.up_proj.weight"), &.{ experts, inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.switch_mlp.down_proj.weight"), &.{ experts, h, inter }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.shared_expert.gate_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.shared_expert.up_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.shared_expert.down_proj.weight"), &.{ h, inter }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.shared_expert_gate.weight"), &.{ 1, h }, 0.1);
        if (i % 4 != 3) {
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.in_proj_qkvz.weight"), &.{ conv + hv * dv, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.in_proj_ba.weight"), &.{ 2 * hv, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.conv1d.weight"), &.{ conv, 4, 1 }, 0.3);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.A_log"), &.{hv}, 0.5);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.dt_bias"), &.{hv}, 0.5);
            try tensors.append(a, .{ .name = try L.n(a, i, "linear_attn.norm.weight"), .shape = &.{dv}, .data = ones[0..dv] });
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.out_proj.weight"), &.{ h, hv * dv }, 0.1);
        } else {
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.q_proj.weight"), &.{ 2 * heads * hd, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.k_proj.weight"), &.{ kv_heads * hd, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.v_proj.weight"), &.{ kv_heads * hd, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.o_proj.weight"), &.{ h, heads * hd }, 0.1);
            try tensors.append(a, .{ .name = try L.n(a, i, "self_attn.q_norm.weight"), .shape = &.{hd}, .data = ones[0..hd] });
            try tensors.append(a, .{ .name = try L.n(a, i, "self_attn.k_norm.weight"), .shape = &.{hd}, .data = ones[0..hd] });
        }
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"qwen3_next","hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,"num_attention_heads":2,
        \\ "num_key_value_heads":1,"head_dim":32,"vocab_size":64,"rms_norm_eps":1e-6,"rope_theta":10000000,
        \\ "partial_rotary_factor":0.25,"full_attention_interval":4,"linear_num_key_heads":1,"linear_num_value_heads":2,
        \\ "linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,"num_experts":8,
        \\ "num_experts_per_tok":2,"moe_intermediate_size":32,"shared_expert_intermediate_size":32,
        \\ "max_position_embeddings":4096,"hidden_act":"silu","eos_token_id":2,"tie_word_embeddings":false}
        },
        .{ .name = "model.safetensors", .bytes = try sdk.fixture.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
