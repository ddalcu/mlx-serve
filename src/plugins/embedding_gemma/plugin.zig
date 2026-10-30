//! EmbeddingGemma: the Gemma 3 trunk as a bidirectional encoder (`gemma4/bidirectional.zig`), its sentence-transformers
//! projection head (`dense.0`, `dense.1` at the pack's root) behind `EmbedHooks.project`. Served on `/v1/embeddings`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const encoder = @import("../gemma4/bidirectional.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = &config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &encoder.bill,
    .load = encoder.loadAs(.embedding_gemma),
    .unload = &encoder.unload,
    .forward = &encoder.tableForward,
    .cache = &.{ .layout = &config.layout },
    .embed = &.{ .project = &encoder.project },
    .weights = &.{},
};

/// The tiny pack the conformance suite runs: two layers (sliding, global), width 64, seeded random f32 weights and a
/// two-layer projection head.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v = 64;
    const h = 64;
    const inter = 128;
    const hd = 32;
    var tensors: std.ArrayList(sdk.fixture.Tensor) = .empty;
    var seed: u64 = 23;
    const put = struct {
        fn f(al: std.mem.Allocator, list: *std.ArrayList(sdk.fixture.Tensor), sd: *u64, name: []const u8, shape: []const u64, scale: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            sd.* += 1;
            try list.append(al, .{ .name = name, .shape = try al.dupe(u64, shape), .data = try sdk.fixture.randomF32(al, sd.*, n, scale) });
        }
    }.f;
    try put(a, &tensors, &seed, "model.embed_tokens.weight", &.{ v, h }, 0.5);
    try put(a, &tensors, &seed, "model.norm.weight", &.{h}, 0.1);
    try put(a, &tensors, &seed, "dense.0.weight", &.{ 2 * h, h }, 0.1);
    try put(a, &tensors, &seed, "dense.1.weight", &.{ h, 2 * h }, 0.1);
    for (0..2) |i| {
        const L = struct {
            fn n(al: std.mem.Allocator, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "model.layers.{d}.{s}", .{ li, leaf });
            }
        };
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |nm|
            try put(a, &tensors, &seed, try L.n(a, i, nm), &.{h}, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.q_norm.weight"), &.{hd}, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.k_norm.weight"), &.{hd}, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.q_proj.weight"), &.{ 2 * hd, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.k_proj.weight"), &.{ hd, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.v_proj.weight"), &.{ hd, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.o_proj.weight"), &.{ h, 2 * hd }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.gate_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.up_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.down_proj.weight"), &.{ h, inter }, 0.1);
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"gemma3_text","use_bidirectional_attention":true,"hidden_size":64,"intermediate_size":128,
        \\ "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":64,
        \\ "sliding_window":2,"sliding_window_pattern":2,"query_pre_attn_scalar":32,"rms_norm_eps":1e-6,
        \\ "max_position_embeddings":64,"bos_token_id":2,"eos_token_id":1,"pooling_mode":"mean"}
        },
        .{ .name = "model.safetensors", .bytes = try sdk.fixture.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
