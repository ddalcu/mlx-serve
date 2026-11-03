//! The built-in quant formats' tables, in `block.builtins` order: the index `block.claimant` returns names the table
//! here. Discovery reads `block.zig` (MLX-free); the loader reads this list.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const block = @import("block.zig");
const native = @import("native.zig");
const hadamard = @import("hadamard/format.zig");
const exl3 = @import("exl3/format.zig");
const gguf = @import("gguf/format.zig");

pub const tables = [_]*const abi.QuantTable{ &native.affine, &native.fp8, &native.mxfp8, &native.nvfp4, &hadamard.table, &exl3.table, &gguf.table };

/// The table serving `block_json` of `config_json`; null when no built-in claims it.
pub fn claimant(block_json: []const u8, config_json: []const u8) error{Ambiguous}!?*const abi.QuantTable {
    const i = try block.claimant(block_json, config_json) orelse return null;
    return tables[i];
}

// ── tests ──

const testing = std.testing;
const sdk = @import("../root.zig");
const conformance = @import("../testing/conformance.zig");
const fixture = @import("../testing/fixture.zig");

test "quant formats: the tables follow block.builtins, each with its own claims and validation" {
    try testing.expectEqual(block.builtins.len, tables.len);
    for (block.builtins, tables) |b, t| {
        try testing.expectEqualStrings(b.name, t.name.slice());
        try testing.expect(b.claims == t.claims and b.validate == t.validate);
    }
    try testing.expectEqual(&native.affine, (try claimant("{\"bits\":4}", "{}")).?);
    try testing.expectEqual(@as(?*const abi.QuantTable, null), try claimant("{\"quant_method\":\"bitsandbytes\"}", "{}"));
}

/// A built-in format as the conformance suite reads a quant plugin.
fn Builtin(comptime i: usize, comptime block_json: []const u8, comptime config_json: []const u8) type {
    return struct {
        pub const plugin = sdk.Plugin{ .name = block.builtins[i].name, .kind = .quant };
        pub const table = tables[i].*;
        pub fn conformanceFixture(_: std.mem.Allocator) !fixture.Fixture {
            return .{ .files = &.{.{ .name = "config.json", .bytes = config_json }}, .block = block_json };
        }
    };
}

test "quant formats: every built-in format passes the quant plugin conformance" {
    try conformance.expectPlugin(Builtin(0, "{\"bits\":4,\"group_size\":64}", "{}"));
    try conformance.expectPlugin(Builtin(1, "{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}", "{}"));
    try conformance.expectPlugin(Builtin(2, "{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\"}", "{}"));
    try conformance.expectPlugin(Builtin(3, "{\"bits\":4,\"group_size\":16,\"mode\":\"nvfp4\"}", "{}"));
    try conformance.expectPlugin(Builtin(4, "{\"bits\":2,\"group_size\":128}", "{\"modules\":[{\"path\":\"lm_head\",\"block\":1024}]}"));
    try conformance.expectPlugin(Builtin(5, "{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}", "{}"));
    try conformance.expectPlugin(Builtin(6, "{\"group_size\":32,\"bits\":4,\"mode\":\"gguf\"}", "{}"));
}

test {
    _ = block;
    _ = native;
    _ = hadamard;
    _ = exl3;
    _ = gguf;
    _ = @import("../weights/resident.zig");
    _ = @import("../weights/streamed.zig");
    _ = @import("fp8_block128.zig");
}
