//! The one way engine, host and sdk code reaches the DeepSeek-V4.1 plugin (`plugins/deepseek_v41`). The plugin is macOS
//! only (its config.zig's `macos_only`): the generated registry leaves it out of every other build, and there each
//! const here is its stub in `dsv41_absent.zig`, so no file of the plugin is analyzed or linked.

const std = @import("std");
const absent = @import("dsv41_absent.zig");

/// The build registers the plugin: macOS, and only macOS.
pub const on = @import("../../../plugins/registry.zig").has("deepseek_v41");

pub const deepseek_v41 = if (on) @import("../../../plugins/deepseek_v41/deepseek_v41.zig") else absent.deepseek_v41;
pub const plugin = if (on) @import("../../../plugins/deepseek_v41/plugin.zig") else absent.plugin;
pub const config = if (on) @import("../../../plugins/deepseek_v41/config.zig") else absent.config;
pub const bill = if (on) @import("../../../plugins/deepseek_v41/bill.zig") else absent.bill;
pub const vision = if (on) @import("../../../plugins/deepseek_v41/vision.zig") else absent.vision;
pub const keep_wide = if (on) @import("../../../plugins/deepseek_v41/prompt/keep_wide.zig") else absent.keep_wide;
pub const contract = if (on) @import("../../../plugins/deepseek_v41/contract.zig") else absent.contract;

/// The plugin's `Shared` states, which `src/shared_all.zig` references so the host unit exports each.
pub const shared = if (on) .{
    @import("../../../plugins/deepseek_v41/decode/engram_decode.zig").shared_state,
    @import("../../../plugins/deepseek_v41/decode/fp8_codes.zig").shared_state,
    @import("../../../plugins/deepseek_v41/fuse.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/engram_direct.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/flash_attn.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/hc_v2.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/idx_nax.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/pass_async.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/scratch.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/scratch_attn.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/scratch_x1.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/transient_keep.zig").shared_state,
    @import("../../../plugins/deepseek_v41/prompt/keep_wide.zig").shared_state,
    @import("../../../plugins/deepseek_v41/levers.zig").shared_state,
    @import("../../../plugins/deepseek_v41/deepseek_v41.zig").clusterHooks_shared,
} else .{};

// ── tests ──

/// Each pub declaration of `Stub` that `Real` lacks, or a function whose parameter count differs, as "path: why";
/// a struct type declared in both, and no generic's instance (`Parked`), is compared one level down (its methods).
fn drift(comptime path: []const u8, comptime Stub: type, comptime Real: type, comptime depth: u32) []const []const u8 {
    var out: []const []const u8 = &.{};
    inline for (comptime std.meta.declarations(Stub)) |name| {
        const at = path ++ "." ++ name;
        if (!@hasDecl(Real, name)) {
            out = out ++ [_][]const u8{at ++ ": no such declaration"};
        } else {
            const s = @field(Stub, name);
            const r = @field(Real, name);
            const S = @TypeOf(s);
            const R = @TypeOf(r);
            if (@typeInfo(S) == .@"fn") {
                if (@typeInfo(R) != .@"fn") {
                    out = out ++ [_][]const u8{at ++ ": not a function"};
                } else if (@typeInfo(S).@"fn".param_types.len != @typeInfo(R).@"fn".param_types.len) {
                    out = out ++ [_][]const u8{at ++ ": parameter count"};
                }
            } else if (S == type and R == type and depth > 0 and @typeInfo(s) == .@"struct" and @typeInfo(r) == .@"struct" and
                std.mem.indexOfScalar(u8, @typeName(s), '(') == null)
            {
                out = out ++ drift(at, s, r, depth - 1);
            }
        }
    }
    const done = out;
    return done;
}

test "dsv41_link: every absent stub names a declaration of the real plugin module, with its parameter count" {
    if (!on) return error.SkipZigTest;
    const pairs = .{
        .{ "deepseek_v41", absent.deepseek_v41, deepseek_v41 },
        .{ "plugin", absent.plugin, plugin },
        .{ "config", absent.config, config },
        .{ "bill", absent.bill, bill },
        .{ "vision", absent.vision, vision },
        .{ "keep_wide", absent.keep_wide, keep_wide },
        .{ "contract", absent.contract, contract },
    };
    var n: usize = 0;
    inline for (pairs) |p| {
        const bad = comptime drift(p[0], p[1], p[2], 1);
        for (bad) |b| std.debug.print("dsv41_absent drift: {s}\n", .{b});
        n += bad.len;
    }
    try std.testing.expectEqual(@as(usize, 0), n);
}
