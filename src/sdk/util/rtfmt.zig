//! Runtime formatting for log lines (plan L12). `Writer.print` is one instance per format string and argument types,
//! so every log call site used to carry its own copy of the formatter. Here a call site packs its arguments into
//! `Arg`s and one non-generic `print` walks the format. Output is byte-identical to `std.fmt` over what `supported`
//! accepts; anything else stays on `Writer.print`.

const std = @import("std");
const Writer = std.Io.Writer;

/// One packed argument. Integers go through `isize`/`usize` exactly as `Writer.printInt` funnels them, so the sign
/// rule (a `+` on a positive signed value under a width) is the same; floats keep their width (the shortest
/// round-trip spelling of an f32 is not its f64's).
pub const Arg = union(enum) {
    i: isize,
    u: usize,
    f32: f32,
    f64: f64,
    s: []const u8,
    b: bool,
};

/// Whether `fmt` over `Args` takes the runtime path: placeholders without positions or named widths, and one of
/// `{}`/`{d}` (integers, floats, bools for `{}`), `{x}`/`{X}` (integers), `{e}` (floats), `{c}` (u8), `{s}` (byte
/// strings), `{t}` (enums, tagged unions, errors).
pub fn supported(comptime fmt: []const u8, comptime Args: type) bool {
    const info = @typeInfo(Args);
    if (info != .@"struct" or !info.@"struct".is_tuple) return false;
    const types = info.@"struct".field_types;
    @setEvalBranchQuota(fmt.len * 1000 + 1000);
    comptime var n: usize = 0;
    comptime var i: usize = 0;
    inline while (i < fmt.len) {
        if (fmt[i] == '{' and i + 1 < fmt.len and fmt[i + 1] == '{') {
            i += 2;
            continue;
        }
        if (fmt[i] == '}') {
            i += 2;
            continue;
        }
        if (fmt[i] != '{') {
            i += 1;
            continue;
        }
        const end = comptime std.mem.indexOfScalarPos(u8, fmt, i, '}') orelse return false;
        const body = fmt[i + 1 .. end];
        if (comptime std.mem.indexOfScalar(u8, body, '[') != null) return false;
        const ph = comptime std.fmt.Placeholder.parse(body);
        if (ph.arg != .none or ph.width == .named or ph.precision == .named) return false;
        if (n >= types.len or !argFits(ph.specifier_arg, types[n])) return false;
        n += 1;
        i = end + 1;
    }
    return n == types.len;
}

fn argFits(comptime spec: []const u8, comptime T: type) bool {
    const ti = @typeInfo(T);
    const int = (ti == .int and ti.int.bits <= 64) or ti == .comptime_int;
    const float = T == f32 or T == f64;
    if (spec.len == 0) return int or float or T == bool;
    if (spec.len != 1) return false;
    return switch (spec[0]) {
        'd' => int or float,
        'x', 'X' => int,
        'e' => float,
        'c' => T == u8,
        's' => isString(T),
        't' => ti == .@"enum" or ti == .error_set or (ti == .@"union" and ti.@"union".tag_type != null),
        else => false,
    };
}

fn isString(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |p| switch (p.size) {
            .slice => p.child == u8,
            .one => switch (@typeInfo(p.child)) {
                .array => |a| a.child == u8,
                else => false,
            },
            .many, .c => p.child == u8 and p.sentinel() == 0,
        },
        else => false,
    };
}

pub fn Packed(comptime Args: type) type {
    return [@typeInfo(Args).@"struct".field_types.len]Arg;
}

pub fn pack(args: anytype) Packed(@TypeOf(args)) {
    var out: Packed(@TypeOf(args)) = undefined;
    inline for (args, 0..) |v, k| out[k] = one(v);
    return out;
}

fn one(v: anytype) Arg {
    const T = @TypeOf(v);
    return switch (@typeInfo(T)) {
        .comptime_int => if (v >= 0) .{ .u = v } else .{ .i = v },
        .int => |t| if (t.signedness == .signed) .{ .i = v } else .{ .u = v },
        .float => if (T == f32) .{ .f32 = v } else .{ .f64 = v },
        .bool => .{ .b = v },
        .@"enum", .@"union" => .{ .s = @tagName(v) },
        .error_set => .{ .s = @errorName(v) },
        .pointer => |p| switch (p.size) {
            .many, .c => .{ .s = std.mem.span(v) },
            else => .{ .s = v },
        },
        else => unreachable, // `supported` admits nothing else
    };
}

/// `w.print(fmt, args)` for a `fmt` and `args` that `supported` accepted.
pub fn print(w: *Writer, fmt: []const u8, args: []const Arg) Writer.Error!void {
    var i: usize = 0;
    var n: usize = 0;
    while (i < fmt.len) {
        const c = fmt[i];
        if ((c == '{' or c == '}') and i + 1 < fmt.len and fmt[i + 1] == c) {
            try w.writeByte(c);
            i += 2;
            continue;
        }
        if (c != '{') {
            const stop = std.mem.indexOfAnyPos(u8, fmt, i, "{}") orelse fmt.len;
            try w.writeAll(fmt[i..stop]);
            i = stop;
            continue;
        }
        const end = std.mem.indexOfScalarPos(u8, fmt, i, '}').?;
        try printArg(w, fmt[i + 1 .. end], args[n]);
        n += 1;
        i = end + 1;
    }
}

/// One placeholder body (`d:>8.2`) applied to `arg`, read as `std.fmt.Placeholder.parse` reads it.
fn printArg(w: *Writer, body: []const u8, arg: Arg) Writer.Error!void {
    const colon = std.mem.indexOfScalar(u8, body, ':') orelse body.len;
    const spec = body[0..colon];
    var o: std.fmt.Options = .{};
    if (colon < body.len) {
        var p = body[colon + 1 ..];
        var fill: ?u8 = null;
        if (p.len > 1 and isAlign(p[1])) {
            fill = p[0];
            p = p[1..];
        }
        if (p.len > 0 and isAlign(p[0])) {
            o.alignment = switch (p[0]) {
                '<' => .left,
                '^' => .center,
                else => .right,
            };
            p = p[1..];
        } else if (fill == null and p.len > 0 and p[0] == '0') {
            fill = '0';
        }
        if (fill) |f| o.fill = f;
        var k: usize = 0;
        while (k < p.len and std.ascii.isDigit(p[k])) k += 1;
        if (k > 0) o.width = std.fmt.parseInt(usize, p[0..k], 10) catch unreachable;
        if (k < p.len and p[k] == '.') {
            const q = p[k + 1 ..];
            o.precision = std.fmt.parseInt(usize, q, 10) catch unreachable;
        }
    }
    const kind: u8 = if (spec.len == 0) 0 else spec[0];
    switch (arg) {
        .i => |v| try w.printInt(v, if (kind == 'x' or kind == 'X') 16 else 10, if (kind == 'X') .upper else .lower, o),
        .u => |v| if (kind == 'c') try w.printAsciiChar(@intCast(v), o) else try w.printInt(v, if (kind == 'x' or kind == 'X') 16 else 10, if (kind == 'X') .upper else .lower, o),
        .f32 => |v| try w.printFloat(v, o.toNumber(if (kind == 'e') .scientific else .decimal, .lower)),
        .f64 => |v| try w.printFloat(v, o.toNumber(if (kind == 'e') .scientific else .decimal, .lower)),
        .s => |v| try w.alignBufferOptions(v, o),
        .b => |v| try w.alignBufferOptions(if (v) "true" else "false", o),
    }
}

fn isAlign(c: u8) bool {
    return c == '<' or c == '^' or c == '>';
}

// ── tests ──

const E = enum { alpha, beta };
const U = union(enum) { one: u8, two: void };

/// Every accepted shape prints what `std.fmt` prints.
fn same(comptime fmt: []const u8, args: anytype) !void {
    comptime std.debug.assert(supported(fmt, @TypeOf(args)));
    var want_buf: [256]u8 = undefined;
    const want = try std.fmt.bufPrint(&want_buf, fmt, args);
    var got_buf: [256]u8 = undefined;
    var w = Writer.fixed(&got_buf);
    const a = pack(args);
    try print(&w, fmt, &a);
    try std.testing.expectEqualStrings(want, w.buffered());
}

test "rtfmt: every accepted placeholder prints what std.fmt prints" {
    const c_str: [*:0]const u8 = "cstr";
    const slice: []const u8 = "slice";
    try same("plain text, no args", .{});
    try same("{{literal}} braces {d}", .{@as(u32, 7)});
    try same("{d} {d} {d} {d}", .{ @as(i32, -5), @as(i64, 9_000_000_000), @as(u8, 255), @as(usize, 0) });
    try same("{d:>6}|{d:<6}|{d:^7}|{d:06}|{d:>6}", .{ @as(i32, 42), @as(u16, 42), @as(i8, -3), @as(u32, 42), @as(i32, -42) });
    try same("{d:5} signed under a width {d:5}", .{ @as(i32, 3), @as(u32, 3) });
    try same("{d} {d} comptime", .{ 12, -12 });
    try same("{x} {X} {x:0>4} {X:8}", .{ @as(u32, 0xbeef), @as(u64, 0xbeef), @as(u8, 0xa), @as(i32, -255) });
    try same("{d:.2} {d:.0} {d} {d}", .{ @as(f64, 3.14159), @as(f32, 2.5), @as(f32, 0.1), @as(f64, 0.1) });
    try same("{d:>8.3}|{e}|{e:.2}", .{ @as(f64, -1.5), @as(f64, 12345.678), @as(f32, 0.000123) });
    try same("{d} {d} {d}", .{ std.math.inf(f64), -std.math.inf(f32), std.math.nan(f64) });
    try same("{} {} {}", .{ @as(u32, 5), @as(f64, 1.25), true });
    try same("{} {:>6}", .{ false, true });
    try same("{c}{c:>3}", .{ @as(u8, 'z'), @as(u8, 'q') });
    try same("{s} {s} {s} {s:>8}|{s:-<7}|{s:^9}", .{ "lit", slice, c_str, "r", "l", "mid" });
    try same("{t} {t} {t} {t:>8}", .{ E.beta, error.OutOfMemory, U{ .two = {} }, E.alpha });
}

test "rtfmt: positions, named widths, other specifiers and other types stay on Writer.print" {
    try std.testing.expect(!supported("{[0]d}", struct { u32 }));
    try std.testing.expect(!supported("{d:[w]}", struct { u32 }));
    try std.testing.expect(!supported("{any}", struct { u32 }));
    try std.testing.expect(!supported("{s}", struct { u32 }));
    try std.testing.expect(!supported("{d}", struct { E }));
    try std.testing.expect(!supported("{d}", struct { u128 }));
    try std.testing.expect(!supported("{?d}", struct { ?u32 }));
    try std.testing.expect(!supported("{B}", struct { u64 }));
    try std.testing.expect(!supported("{d} {d}", struct { u32 }));
    try std.testing.expect(!supported("{d}", struct { u32, u32 }));
    try std.testing.expect(supported("{d} {s}", struct { u32, []const u8 }));
}
