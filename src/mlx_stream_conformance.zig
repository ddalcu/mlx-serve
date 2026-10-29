//! mlx-stream's conformance suite (CPU lane): `sdk.testing` over this plugin's declaration, its arch's table, and the
//! EXL3 quant and expert source the arch binds. It is its own test build's root (`zig build mlx-stream-conformance`),
//! so the last check sees only this suite's device use.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk/weights/streamed/sdk_ext.zig");
const plugin = @import("mlx_stream.zig").plugin;



comptime {
    _ = @import("sdk/weights/streamed/sdk_ext/kinds.zig");
}

test "mlx-stream conformance: the plugin negotiates with this host and registers one macOS-only arch" {
    var buf: [64]u8 = undefined;
    try sdk.negotiate(plugin, try sdk.host(&buf));
    try std.testing.expectEqualStrings("mlx-stream", plugin.name);
    try std.testing.expect(plugin.macos_only);
    try std.testing.expect(plugin.provides.arch != null and plugin.provides.source == null and plugin.provides.engine == null);
}

test "mlx-stream conformance: the arch and the kinds it binds decline what is not theirs" {
    const near_misses = [_]sdk.testing.ClaimCase{
        .{ .config = "{}", .want = null },
        .{ .config = "{\"model_type\":\"__no_such_model__\"}", .want = null },
    };
    const arch = comptime sdk.Arch.of(plugin.provides.arch.?);
    try sdk.testing.expectClaims(arch.claims, &near_misses);
    const source = comptime sdk_ext.ExpertSource.of(@import("sdk/quant/exl3/stream/source.zig"));
    try sdk.testing.expectClaims(source.claims, &near_misses);
    // a weight group no quant's format describes
    const group_near_misses = [_]sdk.testing.GroupClaimCase{
        .{ .quantization = "{}", .hidden = 5120, .inter = 2304, .want = null },
        .{ .quantization = "{\"mode\":\"__no_such_format__\",\"bits\":4,\"group_size\":32}", .hidden = 5120, .inter = 2304, .want = null },
    };
    const quant = comptime sdk_ext.Quant.of(@import("sdk/quant/exl3/stream/quant.zig"));
    try sdk.testing.expectGroupClaims(quant.claims, &group_near_misses);
}

test "mlx-stream conformance: the EXL3 quant is pinned by the kernel registry's manifest" {
    const quant = comptime sdk_ext.Quant.of(@import("sdk/quant/exl3/stream/quant.zig"));
    try std.testing.expectEqualStrings("exl3-mul1-k3", quant.name);
    try std.testing.expectEqualStrings(@import("sdk/quant/exl3/stream/kernels.zig").manifest_sha256, quant.kernels.?.manifest_sha256);
}

test "mlx-stream conformance: the EXL3 source's capabilities, and the arch claims the one reader for the process" {
    const k = comptime sdk_ext.ExpertSource.of(@import("sdk/quant/exl3/stream/source.zig"));
    try std.testing.expectEqualStrings("exl3-stream", k.name);
    try std.testing.expect(k.caps.two_phase and k.caps.transient_release and k.caps.event_gates and !k.caps.construction_reset);
    try std.testing.expect(@import("sdk/weights/streamed/expert/stream.zig").uses_reader);
    const arch = comptime sdk.Arch.of(plugin.provides.arch.?);
    try std.testing.expect(arch.claim_process != null and arch.release_process != null);
    try arch.claim_process.?();
    try std.testing.expectError(error.ExpertReaderInUse, arch.claim_process.?());
    arch.release_process.?();
    try arch.claim_process.?();
    arch.release_process.?();
}

test "mlx-stream conformance: every arch the glue dispatches declines what is not its own, and claims the one reader" {
    const near_misses = [_]sdk.testing.ClaimCase{
        .{ .config = "{}", .want = null },
        .{ .config = "{\"model_type\":\"__no_such_model__\"}", .want = null },
    };
    inline for (@import("mlx_stream.zig").archs) |A| {
        const arch = comptime sdk.Arch.of(A);
        try sdk.testing.expectClaims(arch.claims, &near_misses);
        try std.testing.expect(arch.claim_process != null and arch.release_process != null);
    }
    const glm = comptime sdk.Arch.of(@import("plugins/glm_moe_dsa/arch.zig"));
    try sdk.testing.expectClaims(glm.claims, &.{
        .{ .config = "{\"model_type\":\"glm_moe_dsa\"}", .want = .native },
        .{ .config = @embedFile("sdk/weights/streamed/fixtures/mimo_v2_flash_q4_config.json"), .want = null },
    });
    try std.testing.expect(glm.spec == .draft_lane and glm.caps.owns_decode_state);
}

// Declared last so it runs after every other conformance test (the CPU lane's bar).
test "mlx-stream conformance: the CPU lane created no Metal device" {
    try sdk.testing.expectNoDevice();
}
