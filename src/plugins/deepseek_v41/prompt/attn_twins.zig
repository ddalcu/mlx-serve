//! The ring twins (scratch_attn.zig): an attention gather launch's stock text (attn_qk, Q_TAIL's attn_qk_t,
//! attn_sm, attn_pv) rewritten so the scores and probabilities it writes or reads live in a slot of a per-prompt region
//! (scratch_attn.zig), derived at load from the stock texts the plugin compiles, never stored.
//!
//! MLX declares a custom kernel's inputs and outputs itself, by the names the call gives, so a stock text reads and
//! writes them by name and never declares them. The twin binds a region whole as an input and declares the stock
//! name itself before the stock statements, at the slot's first word (`soff[0]`, one more input): an output `o` the
//! stock text writes as `device T* o = (device T*) o_dst + soff[0];` (`into`), an input `i` it reads as
//! `const device T* i = i_rg + soff[0];` (`from`). It adds one `[1]` uint32 output, `carrier`, the grid's first thread
//! writes (a reader binding it is ordered after every write of the launch), and one input it never reads, `dep` (the
//! launch it must follow: its carrier). Every stock statement is unchanged, so the twin computes the stock words.
//!
//! The derivation refuses by name rather than guess: no `#pragma METAL fp contract` line (the gather texts have none:
//! the prologue goes first), every `into` and `from` name named by the stock text, and none of the names the twin adds
//! (`<name>_dst`, `<name>_rg`, `carrier`, `dep`, `soff`) nor a rebound name's `_shape` / `_strides` (a region's are
//! not the stock array's) in the text or its header. Std only: tested without MLX.

const std = @import("std");

pub const Error = error{
    /// The stock text has a `#pragma METAL fp contract` line (the prologue would go after it; the gather texts have none).
    AttnTwinPragma,
    /// The stock text never names a rebound array.
    AttnTwinUnused,
    /// A replaced statement is not in the stock text exactly once.
    AttnTwinStatement,
    /// The stock text or its header names a name the twin adds, or a rebound array's strides or shape.
    AttnTwinNameTaken,
};

/// One array the twin rebinds to a region's slot: its stock name, Metal type (MLX's: `float`, `float16_t`, ...; or a
/// template type the stock text has, `OT`) and the `soff` word that offsets it (null: the region's first word).
pub const Io = struct { name: []const u8, mtype: []const u8, off: ?u8 = 0 };
/// One stock statement the twin replaces (it must occur once): the heads pv twin moves its
/// grouped store to the run region's row layout.
pub const Sub = struct { old: []const u8, new: []const u8 };

const pragma_key = "#pragma METAL fp contract";

fn isIdent(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// `text` names identifier `a ++ b` (not as part of a longer identifier).
fn hasWord(text: []const u8, a: []const u8, b: []const u8) bool {
    const n = a.len + b.len;
    if (n == 0 or text.len < n) return false;
    var i: usize = 0;
    while (i + n <= text.len) : (i += 1) {
        if (!std.mem.eql(u8, text[i..][0..a.len], a) or !std.mem.eql(u8, text[i + a.len ..][0..b.len], b)) continue;
        if (i > 0 and isIdent(text[i - 1])) continue;
        if (i + n < text.len and isIdent(text[i + n])) continue;
        return true;
    }
    return false;
}

/// Every anchor checked.
pub fn check(stock: []const u8, header: []const u8, into: []const Io, from: []const Io) Error!void {
    return checkWith(stock, header, into, from, &.{}, &.{});
}
/// `check`, with the statements the twin replaces (each once in the stock text) and more names it adds (`extra`).
pub fn checkWith(stock: []const u8, header: []const u8, into: []const Io, from: []const Io, subs: []const Sub, extra: []const []const u8) Error!void {
    for (subs) |sb| if (std.mem.count(u8, stock, sb.old) != 1) return error.AttnTwinStatement;
    for (extra) |w| {
        if (hasWord(stock, w, "") or hasWord(header, w, "")) return error.AttnTwinNameTaken;
    }
    if (std.mem.indexOf(u8, stock, pragma_key) != null) return error.AttnTwinPragma;
    for ([_][]const Io{ into, from }, [_][]const u8{ "_dst", "_rg" }) |list, added| {
        for (list) |io| {
            if (!hasWord(stock, io.name, "")) return error.AttnTwinUnused;
            for ([_][]const u8{ added, "_shape", "_strides" }) |suffix| {
                if (hasWord(stock, io.name, suffix) or hasWord(header, io.name, suffix)) return error.AttnTwinNameTaken;
            }
        }
    }
    for ([_][]const u8{ "carrier", "dep", "soff" }) |w| {
        if (hasWord(stock, w, "") or hasWord(header, w, "")) return error.AttnTwinNameTaken;
    }
}

/// The twin's text (the caller frees it): the prologue (each `into` output and `from` input declared at its slot, the
/// carrier word), then the stock text unchanged.
pub fn derive(a: std.mem.Allocator, stock: []const u8, header: []const u8, into: []const Io, from: []const Io) (Error || error{OutOfMemory})![:0]u8 {
    return deriveWith(a, stock, header, into, from, &.{}, &.{});
}
/// `derive`, with stock statements replaced (`subs`, each once) and more names the twin adds checked (`extra`).
pub fn deriveWith(a: std.mem.Allocator, stock: []const u8, header: []const u8, into: []const Io, from: []const Io, subs: []const Sub, extra: []const []const u8) (Error || error{OutOfMemory})![:0]u8 {
    try checkWith(stock, header, into, from, subs, extra);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (into) |o| {
        if (o.off) |k| try out.print(a, "device {s}* {s} = (device {s}*) {s}_dst + soff[{d}];\n", .{ o.mtype, o.name, o.mtype, o.name, k }) else try out.print(a, "device {s}* {s} = (device {s}*) {s}_dst;\n", .{ o.mtype, o.name, o.mtype, o.name });
    }
    for (from) |i| {
        if (i.off) |k| try out.print(a, "const device {s}* {s} = {s}_rg + soff[{d}];\n", .{ i.mtype, i.name, i.name, k }) else try out.print(a, "const device {s}* {s} = {s}_rg;\n", .{ i.mtype, i.name, i.name });
    }
    try out.appendSlice(a, "if (thread_position_in_grid.x == 0u && thread_position_in_grid.y == 0u && thread_position_in_grid.z == 0u) carrier[0] = 1u;\n");
    var body = stock;
    var owned: ?[]u8 = null;
    defer if (owned) |o| a.free(o);
    for (subs) |sb| {
        const next = try std.mem.replaceOwned(u8, a, body, sb.old, sb.new);
        if (owned) |o| a.free(o);
        owned = next;
        body = next;
    }
    try out.appendSlice(a, body);
    return out.toOwnedSliceSentinel(a, 0);
}

const testing = std.testing;

test "attn twins: outputs into a slot, inputs from a slot, the carrier word, then the stock text unchanged" {
    const stock = "const size_t rw = size_t(l) * NK;\np[rw] = PT(s[rw] * 2.0f);\n";
    const t = try derive(testing.allocator, stock, "", &.{.{ .name = "p", .mtype = "float16_t" }}, &.{.{ .name = "s", .mtype = "float" }});
    defer testing.allocator.free(t);
    try testing.expectEqualStrings("device float16_t* p = (device float16_t*) p_dst + soff[0];\n" ++
        "const device float* s = s_rg + soff[0];\n" ++
        "if (thread_position_in_grid.x == 0u && thread_position_in_grid.y == 0u && thread_position_in_grid.z == 0u) carrier[0] = 1u;\n" ++
        stock, t);
    try testing.expectEqual(@as(u8, 0), t.ptr[t.len]);
}

test "attn twins: a dep-only twin (no rebound output) keeps its outputs MLX's" {
    const stock = "o[0] = OT(p[0]);\n";
    const t = try derive(testing.allocator, stock, "/* header */", &.{}, &.{.{ .name = "p", .mtype = "float16_t" }});
    defer testing.allocator.free(t);
    try testing.expect(std.mem.startsWith(u8, t, "const device float16_t* p = p_rg + soff[0];\n"));
    try testing.expect(std.mem.endsWith(u8, t, stock));
}

test "attn twins: a replaced statement and a rebound output at no offset (the heads twin)" {
    const stock = "const int L = meta[7];\n    device OT* dst = o + (h * L + l) * W + d;\n    *dst = OT(p[l]);\n";
    const subs = [_]Sub{.{ .old = "    device OT* dst = o + (h * L + l) * W + d;\n", .new = "    device OT* dst = o + (size_t(soff[1] + l) * G + h) * W + d;\n" }};
    const t = try deriveWith(testing.allocator, stock, "", &.{.{ .name = "o", .mtype = "OT", .off = null }}, &.{.{ .name = "p", .mtype = "float16_t" }}, &subs, &.{"dep2"});
    defer testing.allocator.free(t);
    try testing.expectEqualStrings("device OT* o = (device OT*) o_dst;\n" ++
        "const device float16_t* p = p_rg + soff[0];\n" ++
        "if (thread_position_in_grid.x == 0u && thread_position_in_grid.y == 0u && thread_position_in_grid.z == 0u) carrier[0] = 1u;\n" ++
        "const int L = meta[7];\n    device OT* dst = o + (size_t(soff[1] + l) * G + h) * W + d;\n    *dst = OT(p[l]);\n", t);
    // the statement must be there once; `dep2` must not be taken
    try testing.expectError(error.AttnTwinStatement, checkWith("x = 1;\n", "", &.{}, &.{}, &subs, &.{}));
    try testing.expectError(error.AttnTwinNameTaken, checkWith(stock, "#define dep2 1", &.{}, &.{}, &subs, &.{"dep2"}));
}

test "attn twins: a pragma, an unused name, or a name the twin adds is refused" {
    const s_out = [_]Io{.{ .name = "s", .mtype = "float" }};
    try testing.expectError(error.AttnTwinPragma, check("#pragma METAL fp contract(off)\ns[0] = 0;\n", "", &s_out, &.{}));
    // `s` itself is never named: `scores` and `s2` are other identifiers
    try testing.expectError(error.AttnTwinUnused, check("scores[0] = s2;\n", "", &s_out, &.{}));
    try testing.expectError(error.AttnTwinNameTaken, check("s[0] = s_shape[1];\n", "", &s_out, &.{}));
    try testing.expectError(error.AttnTwinNameTaken, check("s[0] = 0;\n", "#define s_dst x", &s_out, &.{}));
    try testing.expectError(error.AttnTwinNameTaken, check("s[0] = s_rg[0];\n", "", &.{}, &s_out));
    try testing.expectError(error.AttnTwinNameTaken, check("s[0] = soff;\n", "", &s_out, &.{}));
    try testing.expectError(error.AttnTwinNameTaken, check("s[0] = 0;\n", "inline void dep() {}", &s_out, &.{}));
    // `deps`, `carriers` and `soffset` are other identifiers
    try check("s[0] = deps + carriers + soffset;\n", "", &s_out, &.{});
}
