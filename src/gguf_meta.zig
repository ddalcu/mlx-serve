//! Minimal GGUF header reader — just enough to decide which embedded engine
//! should serve a given `.gguf` file. We don't materialize tensors or full
//! metadata, only walk the KV pairs at the top of the file looking for two
//! keys:
//!
//!   - `general.architecture` (string) — what arch the file claims to be
//!   - `deepseek4.attention.output_lora_rank` (any numeric) — present only on
//!     the antirez/ds4-style DSV4-Flash GGUFs that the embedded ds4 engine
//!     can actually load (vanilla llama.cpp deepseek4 quants omit it)
//!
//! Routing rule (`preferredEngine`):
//!   `general.architecture == "deepseek4"` AND lora-rank key present → ds4
//!   otherwise → llama.cpp
//!
//! This replaces the basename-only heuristic in `model_discovery.zig`, which
//! mis-routed both real-world cases reported in issue #15 (a vanilla
//! deepseek4 GGUF whose name started with `deepseek-v4-flash` went to ds4
//! and crashed on missing metadata; a true ds4 GGUF whose name started with
//! `Huihui-` went to llama.cpp and crashed on unknown arch).
//!
//! Why we don't reuse libllama's metadata API: `llama_model_load_from_file`
//! rejects unknown architectures BEFORE metadata becomes readable, which is
//! the exact path we're trying to avoid taking by mistake. ds4's C API also
//! offers no metadata-peek. A purpose-built reader keeps the routing
//! decision cheap and side-effect free.

const std = @import("std");

pub const Engine = enum { ds4, llama };

pub const Info = struct {
    /// Owned dupe of `general.architecture` value; null if the key was
    /// absent or not a string.
    architecture: ?[]u8 = null,
    has_ds4_lora_rank: bool = false,

    pub fn deinit(self: *Info, allocator: std.mem.Allocator) void {
        if (self.architecture) |a| allocator.free(a);
        self.* = .{};
    }
};

pub fn preferredEngine(info: Info) Engine {
    if (info.architecture) |a| {
        if (std.mem.eql(u8, a, "deepseek4") and info.has_ds4_lora_rank) return .ds4;
    }
    return .llama;
}

pub const Error = error{
    BadMagic,
    UnsupportedVersion,
    UnsupportedType,
    Truncated,
    KeyTooLong,
} || std.mem.Allocator.Error;

// GGUF value types — kept private to this file.
const TY_U8: u32 = 0;
const TY_I8: u32 = 1;
const TY_U16: u32 = 2;
const TY_I16: u32 = 3;
const TY_U32: u32 = 4;
const TY_I32: u32 = 5;
const TY_F32: u32 = 6;
const TY_BOOL: u32 = 7;
const TY_STRING: u32 = 8;
const TY_ARRAY: u32 = 9;
const TY_U64: u32 = 10;
const TY_I64: u32 = 11;
const TY_F64: u32 = 12;

// Defensive caps. A key over 1 KiB or a value-string over 16 MiB indicates
// a malformed file (or a malicious one); GGUFs in the wild have keys under
// 100 bytes and string values under a few KiB. Vocab arrays go through a
// separate path that doesn't materialize anything.
const MAX_KEY_LEN: u64 = 1024;
const MAX_STR_VALUE_LEN: u64 = 16 * 1024 * 1024;

/// Stream-parse Info from a *Reader. Both runtime (file) and test (fixed
/// bytes) callers go through here. Short-circuits as soon as both probe
/// keys have been resolved.
pub fn parseInfo(allocator: std.mem.Allocator, r: *std.Io.Reader) Error!Info {
    var info: Info = .{};
    errdefer info.deinit(allocator);

    // Magic + version + counts.
    const magic = takeBytes(r, 4) catch return error.Truncated;
    if (!std.mem.eql(u8, magic, "GGUF")) return error.BadMagic;
    const version = takeIntT(r, u32) catch return error.Truncated;
    if (version < 2 or version > 3) return error.UnsupportedVersion;
    _ = takeIntT(r, u64) catch return error.Truncated; // tensor_count, unused
    const kv_count = takeIntT(r, u64) catch return error.Truncated;

    var seen_arch = false;
    var i: u64 = 0;
    while (i < kv_count) : (i += 1) {
        // Key.
        const key_len = takeIntT(r, u64) catch return error.Truncated;
        if (key_len > MAX_KEY_LEN) return error.KeyTooLong;
        const key_buf = takeBytes(r, @intCast(key_len)) catch return error.Truncated;

        // We need an owned copy if we're matching against it AND continuing
        // to read, because the next take call invalidates the buffer slice.
        // Match first, then advance.
        const is_arch = std.mem.eql(u8, key_buf, "general.architecture");
        const is_ds4_lora = std.mem.eql(u8, key_buf, "deepseek4.attention.output_lora_rank");

        const value_type = takeIntT(r, u32) catch return error.Truncated;

        if (is_arch and value_type == TY_STRING) {
            const v_len = takeIntT(r, u64) catch return error.Truncated;
            if (v_len > MAX_STR_VALUE_LEN) return error.Truncated;
            const v_buf = takeBytes(r, @intCast(v_len)) catch return error.Truncated;
            info.architecture = try allocator.dupe(u8, v_buf);
            seen_arch = true;
        } else if (is_ds4_lora and isNumeric(value_type)) {
            // Presence is what matters; the actual value (rank) isn't used
            // for routing. Skip past the value cleanly.
            try skipValue(r, value_type);
            info.has_ds4_lora_rank = true;
        } else {
            try skipValue(r, value_type);
        }

        // Short-circuit once both probe keys are resolved — saves walking
        // the (potentially huge) tokenizer/vocab arrays that come later.
        if (seen_arch and info.has_ds4_lora_rank) break;
    }

    return info;
}

/// Open the GGUF file and parse Info via the file's buffered reader.
/// Caller owns the returned Info (call `.deinit(allocator)`).
pub fn readFromFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Info {
    var file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rbuf: [16 * 1024]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    return parseInfo(allocator, &rs.interface);
}

// ── private ──

fn isNumeric(ty: u32) bool {
    return switch (ty) {
        TY_U8, TY_I8, TY_U16, TY_I16, TY_U32, TY_I32, TY_F32, TY_U64, TY_I64, TY_F64 => true,
        else => false,
    };
}

fn fixedTypeSize(ty: u32) ?u64 {
    return switch (ty) {
        TY_U8, TY_I8, TY_BOOL => 1,
        TY_U16, TY_I16 => 2,
        TY_U32, TY_I32, TY_F32 => 4,
        TY_U64, TY_I64, TY_F64 => 8,
        else => null,
    };
}

fn skipValue(r: *std.Io.Reader, ty: u32) Error!void {
    if (fixedTypeSize(ty)) |n| {
        r.discardAll64(n) catch return error.Truncated;
        return;
    }
    switch (ty) {
        TY_STRING => {
            const n = takeIntT(r, u64) catch return error.Truncated;
            r.discardAll64(n) catch return error.Truncated;
        },
        TY_ARRAY => {
            const inner_ty = takeIntT(r, u32) catch return error.Truncated;
            const inner_count = takeIntT(r, u64) catch return error.Truncated;
            if (fixedTypeSize(inner_ty)) |sz| {
                // count * sz could overflow u64 in theory; fail-safe.
                const total = std.math.mul(u64, inner_count, sz) catch return error.Truncated;
                r.discardAll64(total) catch return error.Truncated;
            } else if (inner_ty == TY_STRING) {
                var k: u64 = 0;
                while (k < inner_count) : (k += 1) {
                    const sl = takeIntT(r, u64) catch return error.Truncated;
                    r.discardAll64(sl) catch return error.Truncated;
                }
            } else if (inner_ty == TY_ARRAY) {
                var k: u64 = 0;
                while (k < inner_count) : (k += 1) try skipValue(r, TY_ARRAY);
            } else {
                return error.UnsupportedType;
            }
        },
        else => return error.UnsupportedType,
    }
}

// Reader-method wrappers that translate the std error set into our Error.
// `takeArray(n)` would want a comptime n; we use `take(n)` for runtime n
// then convert to a slice.
fn takeBytes(r: *std.Io.Reader, n: usize) ![]const u8 {
    return r.take(n);
}

fn takeIntT(r: *std.Io.Reader, comptime T: type) !T {
    return r.takeInt(T, .little);
}

// ── tests ──

const testing = std.testing;

const Value = union(enum) {
    str: []const u8,
    u32_v: u32,
    u64_v: u64,
    str_array: []const []const u8,
};

const KV = struct { key: []const u8, value: Value };

fn appendU32(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u32) !void {
    var tmp: [4]u8 = undefined;
    std.mem.writeInt(u32, &tmp, v, .little);
    try buf.appendSlice(allocator, &tmp);
}

fn appendU64(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u64) !void {
    var tmp: [8]u8 = undefined;
    std.mem.writeInt(u64, &tmp, v, .little);
    try buf.appendSlice(allocator, &tmp);
}

fn appendStr(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try appendU64(buf, allocator, s.len);
    try buf.appendSlice(allocator, s);
}

/// Build a minimal but valid GGUF v3 header from a KV list. tensor_count
/// is zero — we never read past the metadata section in parseInfo.
fn buildHeader(allocator: std.mem.Allocator, kvs: []const KV) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.appendSlice(allocator, "GGUF");
    try appendU32(&buf, allocator, 3); // version
    try appendU64(&buf, allocator, 0); // tensor_count
    try appendU64(&buf, allocator, @intCast(kvs.len)); // kv_count
    for (kvs) |kv| {
        try appendStr(&buf, allocator, kv.key);
        switch (kv.value) {
            .str => |s| {
                try appendU32(&buf, allocator, TY_STRING);
                try appendStr(&buf, allocator, s);
            },
            .u32_v => |v| {
                try appendU32(&buf, allocator, TY_U32);
                try appendU32(&buf, allocator, v);
            },
            .u64_v => |v| {
                try appendU32(&buf, allocator, TY_U64);
                try appendU64(&buf, allocator, v);
            },
            .str_array => |arr| {
                try appendU32(&buf, allocator, TY_ARRAY);
                try appendU32(&buf, allocator, TY_STRING);
                try appendU64(&buf, allocator, arr.len);
                for (arr) |s| try appendStr(&buf, allocator, s);
            },
        }
    }
    return buf.toOwnedSlice(allocator);
}

fn parseBytes(allocator: std.mem.Allocator, bytes: []const u8) Error!Info {
    var r = std.Io.Reader.fixed(bytes);
    return parseInfo(allocator, &r);
}

test "preferredEngine: llama arch → llama" {
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "llama" } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqualStrings("llama", info.architecture.?);
    try testing.expect(!info.has_ds4_lora_rank);
    try testing.expectEqual(Engine.llama, preferredEngine(info));
}

test "preferredEngine: deepseek4 + lora_rank → ds4" {
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqualStrings("deepseek4", info.architecture.?);
    try testing.expect(info.has_ds4_lora_rank);
    try testing.expectEqual(Engine.ds4, preferredEngine(info));
}

test "preferredEngine: deepseek4 without lora_rank → llama" {
    // Models like Preyazz/DeepSeek-V4-Flash-GGUF (vanilla llama.cpp quant)
    // declare arch=deepseek4 but lack the antirez/ds4 MLA metadata — these
    // must NOT route to ds4 (would crash on missing key at engine open).
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "general.name", .value = .{ .str = "DeepSeek V4 Flash" } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqualStrings("deepseek4", info.architecture.?);
    try testing.expect(!info.has_ds4_lora_rank);
    try testing.expectEqual(Engine.llama, preferredEngine(info));
}

test "preferredEngine: lora_rank present but wrong arch → llama" {
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "qwen2" } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqual(Engine.llama, preferredEngine(info));
}

test "parseInfo: skips string arrays between probe keys" {
    // Real GGUFs carry `tokenizer.ggml.tokens` as a giant string array. The
    // parser must walk past it cleanly to find the lora_rank key that comes
    // after.
    const tokens = [_][]const u8{ "<s>", "</s>", "<unk>", "hello", "world" };
    const bytes = try buildHeader(testing.allocator, &.{
        .{ .key = "general.architecture", .value = .{ .str = "deepseek4" } },
        .{ .key = "tokenizer.ggml.tokens", .value = .{ .str_array = &tokens } },
        .{ .key = "deepseek4.attention.output_lora_rank", .value = .{ .u32_v = 1024 } },
    });
    defer testing.allocator.free(bytes);
    var info = try parseBytes(testing.allocator, bytes);
    defer info.deinit(testing.allocator);
    try testing.expectEqual(Engine.ds4, preferredEngine(info));
}

test "parseInfo: short-circuits once both keys found" {
    // Add a malformed KV AFTER the two we care about. parseInfo must NOT
    // reach it; reaching it would error. This pins the early-return.
    // Build a header with a deliberately broken 3rd KV after the two we
    // care about. parseInfo must NOT reach it; reaching it would error.
    // This pins the early-return.
    var bad: std.ArrayList(u8) = .empty;
    defer bad.deinit(testing.allocator);
    try bad.appendSlice(testing.allocator, "GGUF");
    try appendU32(&bad, testing.allocator, 3);
    try appendU64(&bad, testing.allocator, 0);
    try appendU64(&bad, testing.allocator, 3); // kv_count=3
    try appendStr(&bad, testing.allocator, "general.architecture");
    try appendU32(&bad, testing.allocator, TY_STRING);
    try appendStr(&bad, testing.allocator, "deepseek4");
    try appendStr(&bad, testing.allocator, "deepseek4.attention.output_lora_rank");
    try appendU32(&bad, testing.allocator, TY_U32);
    try appendU32(&bad, testing.allocator, 1024);
    // Third KV: bogus type 99 — would trip UnsupportedType if reached.
    try appendStr(&bad, testing.allocator, "junk");
    try appendU32(&bad, testing.allocator, 99);

    var info = try parseBytes(testing.allocator, bad.items);
    defer info.deinit(testing.allocator);
    try testing.expectEqual(Engine.ds4, preferredEngine(info));
}

test "parseInfo: bad magic → BadMagic" {
    const bytes = "NOPE\x03\x00\x00\x00" ++ ([_]u8{0} ** 16);
    try testing.expectError(error.BadMagic, parseBytes(testing.allocator, bytes));
}

test "parseInfo: unsupported version → UnsupportedVersion" {
    var bytes: [16]u8 = undefined;
    @memcpy(bytes[0..4], "GGUF");
    std.mem.writeInt(u32, bytes[4..8], 1, .little); // v1 — unsupported
    std.mem.writeInt(u64, bytes[8..16], 0, .little);
    // Truncated; we never get past version check.
    try testing.expectError(error.UnsupportedVersion, parseBytes(testing.allocator, &bytes));
}

test "parseInfo: truncated mid-KV → Truncated" {
    // Header announces 5 KVs but body is empty.
    var bytes: [24]u8 = undefined;
    @memcpy(bytes[0..4], "GGUF");
    std.mem.writeInt(u32, bytes[4..8], 3, .little);
    std.mem.writeInt(u64, bytes[8..16], 0, .little); // tensor_count
    std.mem.writeInt(u64, bytes[16..24], 5, .little); // kv_count
    try testing.expectError(error.Truncated, parseBytes(testing.allocator, &bytes));
}
