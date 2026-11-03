//! Prism Bonsai packs (`prism_hadamard_qwen35`): `plugins/qwen3_5`'s dense trunk with every projection stored in a
//! Hadamard-rotated basis (the `hadamard` quant format), f16 activations and an f32 GDN state. Its seams fuse the next
//! norm's rotation (`seam.zig`); its MTP head is grafted from Qwen3.8-27B at load (`mtp_graft.zig`, the engine's).

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

/// qwen3_5's conformance pack under this type: no weight rotated (the format's rotated arm needs block-wide rows).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const base = try q35.conformanceFixture(a);
    const files = try a.dupe(sdk.fixture.File, base.files);
    for (files) |*f| if (std.mem.eql(u8, f.name, "config.json")) {
        f.bytes = try std.mem.replaceOwned(u8, a, f.bytes, "{\"model_type\":\"qwen3_5\",", "{\"model_type\":\"prism_hadamard_qwen35\",\"modules\":[{\"path\":\"lm_head\",\"block\":1024}],");
    };
    return .{ .files = files };
}

// ── tests ──

test {
    _ = config;
    _ = @import("seam.zig");
}
