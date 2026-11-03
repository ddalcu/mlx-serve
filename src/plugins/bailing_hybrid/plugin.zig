//! Ling 3.0 (`bailing_hybrid`, BailingMoeV3): KDA linear attention (a per-channel bounded-sigmoid gate) beside
//! naive MLA every `layer_group_size`-th layer, a grouped sigmoid MoE with one ungated shared expert. Its own forward
//! (`table_forward.zig`) over the trunk GLM-5.3-Flash's plugin loads (`glm5_next/trunk.zig`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const host = @import("../glm5_next/host.zig");
const fwd = @import("table_forward.zig");
const fixture = @import("../glm5_next/fixture.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &host.unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout },
    .weights = &.{},
    .spec = &.{ .trunk_weight = &host.trunkWeight },
};

/// KDA layers keep a recurrent state; an MLA layer caches every head's expanded key (nope | rope) and value.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], first..) |*l, li| l.* = if (c.isLinearLayer(@intCast(li)))
        .{ .kind = .linear }
    else
        .{ .kind = .full, .kv_heads = c.num_attention_heads, .k_dim = c.mla_qk_nope_head_dim + c.mla_qk_rope_head_dim, .v_dim = c.mla_v_head_dim };
    return n;
}

/// MLA decompresses its latent to every attention head before the write, so the cache bills per head.
fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const c = config.of(p);
    const qk: u32 = c.mla_qk_nope_head_dim + c.mla_qk_rope_head_dim;
    out.* = .{
        .kv_per_token = @as(u64, c.attnLayerCount()) * c.num_attention_heads * (qk + c.mla_v_head_dim) * 2,
        .ssm_checkpoint = c.ssmCheckpointBytes(2),
        .prefill_score_width = qk,
    };
}

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return host.load(plugin.name, ctx, out);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m = host.of(mp);
    if (!abi.declaresMode(m.caps, f.mode)) return abi.Verdict.refuse(.unsupported, "bailing_hybrid: forward mode not declared");
    const ops = f.cache_ops orelse return abi.Verdict.refuse(.unsupported, "bailing_hybrid: a forward without cache_ops");
    return host.forwardOne(plugin.name, fwd, m, ops, f, out);
}

/// The tiny pack the conformance suite runs: KDA, MLA, the dense first layer and the grouped MoE.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.lingFiles(a, 11) };
}

// ── tests ──

test {
    _ = config;
}
