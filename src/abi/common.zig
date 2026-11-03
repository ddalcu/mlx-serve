//! The ABI primitives every table shares (engine, model, media, quant): slices, verdicts, MLX handles, claims and
//! the pack directory discovery reads. Extern layouts and the C calling convention; nothing here holds state.

const std = @import("std");

/// A borrowed slice: pointer and length. Freed, if ever, by whoever allocated it.
pub fn Span(comptime T: type) type {
    return extern struct {
        ptr: ?[*]const T = null,
        len: usize = 0,

        pub fn of(s: []const T) @This() {
            return .{ .ptr = s.ptr, .len = s.len };
        }
        pub fn slice(self: @This()) []const T {
            return if (self.ptr) |p| p[0..self.len] else &.{};
        }
    };
}

/// UTF-8 text, never NUL-terminated.
pub const Str = Span(u8);

/// The answer of every fallible call: a code the host maps to HTTP, and the reason by name.
pub const Verdict = extern struct {
    code: Code = .ok,
    why: Str = .{}, // names the reason; static text, safe to read from any thread at any time

    pub const Code = enum(u8) { ok, unsupported, too_large, busy, failed };

    pub const ok: Verdict = .{};

    /// A refusal; `why` is comptime so it is static text by construction.
    pub fn refuse(code: Code, comptime why: []const u8) Verdict {
        return .{ .code = code, .why = Str.of(why) };
    }
    /// A refusal whose text is chosen at run time (an `@errorName`, an entry of a static table). `why` must be static
    /// or live as long as the plugin: the host reads it after the call returns.
    pub fn named(code: Code, why: []const u8) Verdict {
        return .{ .code = code, .why = Str.of(why) };
    }
    pub fn isOk(self: Verdict) bool {
        return self.code == .ok;
    }
};

/// An mlx-c `mlx_array` by value (one context pointer, the same layout). Who owns the reference is stated at each use.
pub const Array = extern struct { ctx: ?*anyopaque = null };
/// An mlx-c `mlx_stream` by value.
pub const Stream = extern struct { ctx: ?*anyopaque = null };

/// How a plugin claims a pack, in rank order: a marker file or field beats a `model_type` match.
pub const Claim = enum(u8) { no = 0, by_type = 1, by_marker = 2 };

/// The one claimant of a pack: the highest rank wins, two claims at that rank are ambiguous, null = nobody claims it.
pub fn pick(claims: []const Claim) error{Ambiguous}!?usize {
    var best: ?usize = null;
    var tie = false;
    for (claims, 0..) |c, i| {
        if (c == .no) continue;
        const b = best orelse {
            best = i;
            continue;
        };
        const rank = @intFromEnum(c);
        const top = @intFromEnum(claims[b]);
        if (rank > top) {
            best = i;
            tie = false;
        } else if (rank == top) tie = true;
    }
    if (tie) return error.Ambiguous;
    return best;
}

/// A pack directory as discovery sees it: existence and bounded reads, paths relative to the pack root, no MLX.
pub const PackDir = extern struct {
    size: u32 = @sizeOf(PackDir),
    ctx: ?*anyopaque,
    /// Whether `rel` exists (a file or a folder).
    exists: *const fn (ctx: ?*anyopaque, rel: Str) callconv(.c) bool,
    /// Reads `rel` whole into `buf`, setting `len`; a file longer than `cap` is `too_large`, a missing one `failed`.
    read: *const fn (ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict,
    /// Reads the JSON header of the safetensors file `rel` into `buf` (the same bounds as `read`).
    safetensors_header: *const fn (ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict,
    /// The entries of folder `rel` ("" = the root) as NUL-separated names, a folder's name ending in '/'; symlinks
    /// followed. The same bounds as `read`.
    list: *const fn (ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict,
    /// `rel`'s size in bytes through symlinks; 0 when it is missing or not a file.
    file_bytes: *const fn (ctx: ?*anyopaque, rel: Str) callconv(.c) u64,

    pub fn has(self: *const PackDir, rel: []const u8) bool {
        return self.exists(self.ctx, Str.of(rel));
    }
    /// `rel`'s bytes, read into `buf`; null with `why` set when the read is refused.
    pub fn readFile(self: *const PackDir, rel: []const u8, buf: []u8, why: *Verdict) ?[]u8 {
        var len: usize = 0;
        const v = self.read(self.ctx, Str.of(rel), buf.ptr, buf.len, &len);
        if (!v.isOk()) {
            why.* = v;
            return null;
        }
        return buf[0..len];
    }
    pub fn fileBytes(self: *const PackDir, rel: []const u8) u64 {
        return self.file_bytes(self.ctx, Str.of(rel));
    }
    /// The entries of folder `rel` (see `list`), read into `buf`; null when the folder cannot be listed.
    pub fn entries(self: *const PackDir, rel: []const u8, buf: []u8) ?Entries {
        var len: usize = 0;
        if (!self.list(self.ctx, Str.of(rel), buf.ptr, buf.len, &len).isOk()) return null;
        return .{ .rest = buf[0..len] };
    }
};

/// A `PackDir.list` result, one name at a time (a folder's name ends in '/').
pub const Entries = struct {
    rest: []const u8,

    pub fn next(self: *Entries) ?[]const u8 {
        while (self.rest.len > 0) {
            const end = std.mem.indexOfScalar(u8, self.rest, 0) orelse self.rest.len;
            const name = self.rest[0..end];
            self.rest = if (end < self.rest.len) self.rest[end + 1 ..] else &.{};
            if (name.len > 0) return name;
        }
        return null;
    }
};

/// The symbol a plugin's table is exported under: `mlx_serve_plugin_<name>_v<version>`.
pub fn exportName(comptime name: []const u8, comptime version: u32) []const u8 {
    return std.fmt.comptimePrint("mlx_serve_plugin_{s}_v{d}", .{ name, version });
}

// ── tests ──

const testing = std.testing;

test "abi common: a span round-trips a slice; an empty span is an empty slice" {
    const s = Str.of("abc");
    try testing.expectEqualStrings("abc", s.slice());
    try testing.expectEqual(@as(usize, 0), (Str{}).slice().len);
    try testing.expectEqual(@as(usize, 2 * @sizeOf(usize)), @sizeOf(Str));
}

test "abi common: a refusal names its reason; ok is the default" {
    try testing.expect((Verdict{}).isOk() and Verdict.ok.isOk());
    const v = Verdict.refuse(.too_large, "prompt past the window");
    try testing.expect(!v.isOk() and v.code == .too_large);
    try testing.expectEqualStrings("prompt past the window", v.why.slice());
    const n = Verdict.named(.failed, @errorName(error.MissingWeight));
    try testing.expect(n.code == .failed and std.mem.eql(u8, "MissingWeight", n.why.slice()));
}

test "abi common: a marker claim beats a type claim; two claims at the top rank are ambiguous" {
    try testing.expectEqual(@as(?usize, null), try pick(&.{ .no, .no }));
    try testing.expectEqual(@as(?usize, 1), try pick(&.{ .no, .by_type }));
    try testing.expectEqual(@as(?usize, 2), try pick(&.{ .by_type, .no, .by_marker }));
    try testing.expectEqual(@as(?usize, 0), try pick(&.{ .by_marker, .by_type, .by_type }));
    try testing.expectError(error.Ambiguous, pick(&.{ .by_type, .by_type }));
    try testing.expectError(error.Ambiguous, pick(&.{ .by_marker, .by_type, .by_marker }));
}

test "abi common: the export symbol carries the name and the table version" {
    try testing.expectEqualStrings("mlx_serve_plugin_flint_v1", exportName("flint", 1));
}
