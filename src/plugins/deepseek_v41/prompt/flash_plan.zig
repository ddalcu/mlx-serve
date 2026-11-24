//! `MLX_SERVE_DSV41_FLASH_ATTN`'s MLX-free parts (flash_attn.zig is the switch's MLX side): the setting, the
//! kernel's text and its heads-region twin. Std only, tested by `zig build test`.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const twins = @import("attn_twins.zig");

pub const Mode = enum { off, on };

pub const Error = error{
    /// MLX_SERVE_DSV41_FLASH_ATTN is neither 0 nor 1 (unset or empty: on).
    FlashAttnSetting,
    /// MLX_SERVE_DSV41_FLASH_TRIM is neither 0 nor 1 (unset or empty: FLASH_ATTN's setting).
    FlashTrimSetting,
    /// MLX_SERVE_DSV41_FLASH_TRIM=1 beside MLX_SERVE_DSV41_FLASH_ATTN=0: the flash kernel asked for and turned off.
    FlashTrimOff,
    /// MLX_SERVE_DSV41_FLASH_PSUM is neither 0 nor 1 (unset or empty: off).
    FlashPsumSetting,
    /// MLX_SERVE_DSV41_FLASH_PSUM=1 beside MLX_SERVE_DSV41_FLASH_ATTN=0: the flash kernel asked for and turned off.
    FlashPsumOff,
};

/// MLX_SERVE_DSV41_FLASH_TRIM: on by default (unset or empty), `1` asked for, `0` off (the flash spans then run
/// `source` without `definesTrim`). The default follows FLASH_ATTN: with the flash kernel off it is off too; only an
/// asked-for `1` beside FLASH_ATTN=0 is refused (`FlashTrimOff`).
pub const Trim = enum { off, default, on };
pub fn parseTrim(value: ?[]const u8) Error!Trim {
    const v = value orelse return .default;
    if (v.len == 0) return .default;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "1")) return .on;
    return error.FlashTrimSetting;
}

/// MLX_SERVE_DSV41_FLASH_PSUM: off unless `1` (the flash spans then run `source` under `definesPsum`).
pub fn parsePsum(value: ?[]const u8) Error!bool {
    const v = value orelse return false;
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.FlashPsumSetting;
}

/// The defines a flash text's header ends with: FA_TRIM (MLX_SERVE_DSV41_FLASH_TRIM), FA_PSUM
/// (MLX_SERVE_DSV41_FLASH_PSUM) and `plant` (tests only: 1 reads the window's keys a row late).
pub fn defines(comptime trim: bool, comptime psum: bool, comptime plant: u1) [:0]const u8 {
    return std.fmt.comptimePrint("#define FA_TRIM {d}\n#define FA_PSUM {d}\n#define FA_PLANT {d}\n", .{ @intFromBool(trim), @intFromBool(psum), plant });
}
pub fn definesTrim(comptime plant: u1) [:0]const u8 {
    return defines(true, false, plant);
}
pub fn definesPsum(comptime plant: u1) [:0]const u8 {
    return defines(false, true, plant);
}

/// On unless the setting says otherwise (: flash attention judged on speed and quality, not words).
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    return error.FlashAttnSetting;
}

/// Whether the setting names on itself (`1`), not by default: a wide span the kernel cannot take is then refused, not
/// served by the gather.
pub fn explicitOn(value: ?[]const u8) bool {
    const v = value orelse return false;
    return std.mem.eql(u8, v, "1");
}

/// Groups of 16 heads a threadgroup (the fa1u microbench's choice: fa4).
pub const hgn: c_int = 4;

/// The kernel's text; its heads-region twin replaces the grouped store's address with the run region's row layout
/// (`[rows, og, in]`, the span's first row `soff[1]`), as the heads pv twin does.
pub const source = kernelSource("plugins/deepseek_v41/kernels/attn_flash.metal");
pub const region_store = twins.Sub{
    .old = "  return o + (size_t(h / hgr) * L + l) * size_t(hgr * D) + size_t(h % hgr) * D + d;\n",
    .new = "  return o + (size_t(soff[1] + l) * G + size_t(h / hgr)) * size_t(hgr * D) + size_t(h % hgr) * D + d;\n",
};

/// The heads-region twin's text (the caller frees it): the region bound whole as `o_dst`, a carrier word, `dep`, `soff`
/// and `dep2` added (attn_twins.zig).
pub fn regionTwin(a: std.mem.Allocator, header: []const u8) ![:0]u8 {
    return twins.deriveWith(a, source, header, &.{.{ .name = "o", .mtype = "OT", .off = null }}, &.{}, &.{region_store}, &.{"dep2"});
}

// ── tests ──

const testing = std.testing;

test "flash_attn setting: on unset, empty or 1; 0 off; anything else refused by name" {
    try testing.expectEqual(Mode.on, try parse(null));
    try testing.expectEqual(Mode.on, try parse(""));
    try testing.expectEqual(Mode.on, try parse("1"));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expect(explicitOn("1"));
    for ([_]?[]const u8{ null, "", "0" }) |v| try testing.expect(!explicitOn(v));
    for ([_][]const u8{ "on", "2", "true", "check", " 1" }) |v| try testing.expectError(error.FlashAttnSetting, parse(v));
}

test "flash_attn region twin: the grouped store's address moved to the run region's rows, the rest of the text kept" {
    const t = try regionTwin(testing.allocator, "");
    defer testing.allocator.free(t);
    try testing.expect(std.mem.indexOf(u8, t, region_store.new) != null);
    try testing.expect(std.mem.indexOf(u8, t, region_store.old) == null);
    try testing.expect(std.mem.startsWith(u8, t, "device OT* o = (device OT*) o_dst;\n"));
}

test "flash_trim setting: on by default unset or empty, 0 off, 1 asked for; anything else refused by name" {
    for ([_]?[]const u8{ null, "" }) |v| try testing.expectEqual(Trim.default, try parseTrim(v));
    try testing.expectEqual(Trim.off, try parseTrim("0"));
    try testing.expectEqual(Trim.on, try parseTrim("1"));
    for ([_][]const u8{ "on", "2", "true", " 1" }) |v| try testing.expectError(error.FlashTrimSetting, parseTrim(v));
}

test "flash_psum setting: off unset, empty or 0; 1 on; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "0" }) |v| try testing.expect(!try parsePsum(v));
    try testing.expect(try parsePsum("1"));
    for ([_][]const u8{ "on", "2", "true", " 1" }) |v| try testing.expectError(error.FlashPsumSetting, parsePsum(v));
}
