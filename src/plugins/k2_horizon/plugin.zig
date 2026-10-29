//! IFM K2-Horizon dense: llama's decoder with grouped RMS norms on every residual-width norm,
//! `Decoder(TableEmbed, GroupedRmsNorm, Attention, SwiGlu, ResidualTail, LmHead)`, loaded by llama's loader.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const llama = @import("../llama/plugin.zig");
const GroupedRmsNorm = @import("norm.zig").GroupedRmsNorm;
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;

pub const Dense = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = GroupedRmsNorm,
    .mixer = b.attention.Attention,
    .mlp = b.mlp.SwiGlu,
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
});

comptime {
    if (Dense.batches_decode != config.batches_decode) @compileError("k2_horizon: config.batches_decode disagrees with the base");
}

const T = llama.Table(Dense);

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &T.unload,
    .forward = &T.forward,
    .cache = &.{},
    .weights = &.{},
};

/// Every term is the engine's default arithmetic over the shell (full attention at the shell's geometry).
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return llama.loadDense(Dense, ctx, groupedNorm) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

fn groupedNorm(o: *llama.loader.Owned, w: mlx.mlx_array, c: *const config.Config, s: mlx.mlx_stream) !GroupedRmsNorm.Weights {
    const groups: c_int = @intCast(c.norm_groups);
    const gw: c_int = @intCast(c.hidden_size / c.norm_groups);
    // The weight-less group norm's ones, one group wide in the activation dtype (a few KB per norm).
    var ones = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{gw}, 1, .bfloat16, s));
    try o.own(ones);
    return .{ .w = w, .groups = groups, .hidden = @intCast(c.hidden_size), .ones = ones };
}

/// The tiny pack the conformance suite runs: llama's, with two norm groups.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const fx = try llama.conformanceFixture(a);
    const files = try a.dupe(sdk.fixture.File, fx.files);
    for (files) |*f| if (std.mem.eql(u8, f.name, "config.json")) {
        f.bytes =
            \\{"model_type":"k2_horizon","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":2,
            \\ "num_key_value_heads":1,"head_dim":32,"vocab_size":64,"rms_norm_eps":1e-5,"max_position_embeddings":4096,
            \\ "hidden_act":"silu","eos_token_id":2,"layernorm_num_groups":2,"num_experts":0,
            \\ "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
        ;
    };
    return .{ .files = files };
}

// ── tests ──

test {
    _ = config;
}
