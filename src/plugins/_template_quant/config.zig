//! The template quant format's claim, validation and bill: MLX-free (the config parse and the load preflight run
//! them); the build checks this file's import closure.

const std = @import("std");
const abi = @import("../../abi/quant.zig");

pub const mode = "_template_quant";

const Block = struct { mode: []const u8 = "", bits: u8 = 0, group_size: u32 = 0 };

fn parseBlock(json: []const u8) ?Block {
    var buf: [1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    return std.json.parseFromSliceLeaky(Block, fba.allocator(), json, .{ .ignore_unknown_fields = true }) catch null;
}

pub fn claims(block_json: abi.Str, config_json: abi.Str) callconv(.c) abi.Claim {
    _ = config_json;
    const b = parseBlock(block_json.slice()) orelse return .no;
    return if (std.mem.eql(u8, b.mode, mode)) .by_type else .no;
}

pub fn validate(block_json: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Verdict {
    _ = config_json;
    _ = dir;
    const b = parseBlock(block_json.slice()) orelse return abi.Verdict.refuse(.failed, "_template_quant: block does not parse");
    if (b.bits != 4 and b.bits != 8) return abi.Verdict.refuse(.unsupported, "_template_quant: bits other than 4 or 8");
    if (b.group_size == 0 or b.group_size % 32 != 0) return abi.Verdict.refuse(.unsupported, "_template_quant: group size not a multiple of 32");
    return .ok;
}

/// Stored as the pack holds it.
pub fn bill(block_json: abi.Str, model_dir: abi.Str, stored: u64) callconv(.c) u64 {
    _ = block_json;
    _ = model_dir;
    return stored;
}

// ── tests ──

const testing = std.testing;

test "_template_quant config: claims its mode only; validates bits and group size by name" {
    const none = abi.Str.of("{}");
    try testing.expectEqual(abi.Claim.by_type, claims(.of("{\"mode\":\"_template_quant\",\"bits\":4}"), none));
    try testing.expectEqual(abi.Claim.no, claims(.of("{\"mode\":\"affine\"}"), none));
    try testing.expectEqual(abi.Claim.no, claims(.of("{"), none));
    const dir: abi.PackDir = undefined;
    try testing.expect(validate(.of("{\"mode\":\"_template_quant\",\"bits\":4,\"group_size\":64}"), none, &dir).isOk());
    try testing.expectEqual(abi.Verdict.Code.unsupported, validate(.of("{\"bits\":3,\"group_size\":64}"), none, &dir).code);
    try testing.expectEqual(abi.Verdict.Code.unsupported, validate(.of("{\"bits\":4,\"group_size\":48}"), none, &dir).code);
}
