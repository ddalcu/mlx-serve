//! Unicode NFC (canonical decomposition, canonical ordering, canonical
//! composition; UAX #15), the HF `NFC` normalizer. Tables: nfc_table.zig.

const std = @import("std");
const table = @import("nfc_table.zig");

const S_BASE = 0xAC00;
const L_BASE = 0x1100;
const V_BASE = 0x1161;
const T_BASE = 0x11A7;
const V_COUNT = 21;
const T_COUNT = 28;
const S_COUNT = 19 * V_COUNT * T_COUNT;

/// `s` in NFC, or null when it is NFC already (always for ASCII) or not
/// valid UTF-8 (left as is). Caller frees a non-null result.
pub fn normalize(a: std.mem.Allocator, s: []const u8) !?[]u8 {
    for (s) |b| {
        if (b >= 0x80) break;
    } else return null;
    var view = std.unicode.Utf8View.init(s) catch return null;
    var cps: std.ArrayList(u21) = .empty;
    defer cps.deinit(a);
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| try decompose(a, &cps, cp);
    reorder(cps.items);
    const n = compose(cps.items);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.ensureTotalCapacity(a, s.len);
    for (cps.items[0..n]) |cp| {
        var buf: [4]u8 = undefined;
        try out.appendSlice(a, buf[0..try std.unicode.utf8Encode(cp, &buf)]);
    }
    if (std.mem.eql(u8, out.items, s)) {
        out.deinit(a);
        return null;
    }
    return try out.toOwnedSlice(a);
}

fn decompose(a: std.mem.Allocator, out: *std.ArrayList(u21), cp: u21) !void {
    if (cp >= S_BASE and cp < S_BASE + S_COUNT) {
        const i = cp - S_BASE;
        try out.append(a, L_BASE + i / (V_COUNT * T_COUNT));
        try out.append(a, V_BASE + (i % (V_COUNT * T_COUNT)) / T_COUNT);
        if (i % T_COUNT != 0) try out.append(a, T_BASE + i % T_COUNT);
        return;
    }
    const d = decompOf(cp) orelse return out.append(a, cp);
    try decompose(a, out, d[1]);
    if (d[2] != 0) try decompose(a, out, d[2]);
}

/// Stable sort of each run of nonzero-class marks by class (insertion sort: runs are short).
fn reorder(cps: []u21) void {
    var i: usize = 1;
    while (i < cps.len) : (i += 1) {
        const cc = class(cps[i]);
        if (cc == 0) continue;
        var j = i;
        while (j > 0) : (j -= 1) {
            const prev = class(cps[j - 1]);
            if (prev == 0 or prev <= cc) break;
            std.mem.swap(u21, &cps[j], &cps[j - 1]);
        }
    }
}

/// UAX #15 canonical composition in place; returns the new length.
fn compose(cps: []u21) usize {
    if (cps.len == 0) return 0;
    var starter: usize = 0;
    var last: u16 = class(cps[0]);
    if (last != 0) last = 256; // a leading mark has no starter to join
    var n: usize = 1;
    for (cps[1..]) |cp| {
        const cc = class(cp);
        if (pair(cps[starter], cp)) |c| if (last < cc or last == 0) {
            cps[starter] = c;
            continue;
        };
        if (cc == 0) starter = n;
        last = cc;
        cps[n] = cp;
        n += 1;
    }
    return n;
}

fn pair(first: u21, second: u21) ?u21 {
    if (first >= L_BASE and first < L_BASE + 19 and second >= V_BASE and second < V_BASE + V_COUNT)
        return S_BASE + ((first - L_BASE) * V_COUNT + (second - V_BASE)) * T_COUNT;
    if (first >= S_BASE and first < S_BASE + S_COUNT and (first - S_BASE) % T_COUNT == 0 and second > T_BASE and second < T_BASE + T_COUNT)
        return first + (second - T_BASE);
    var lo: usize = 0;
    var hi: usize = table.comp.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const e = table.comp[mid];
        if (e[0] == first and e[1] == second) return e[2];
        if (e[0] < first or (e[0] == first and e[1] < second)) lo = mid + 1 else hi = mid;
    }
    return null;
}

fn class(cp: u21) u16 {
    var lo: usize = 0;
    var hi: usize = table.ccc.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const e = table.ccc[mid];
        if (cp < e[0]) hi = mid else if (cp > e[1]) lo = mid + 1 else return @intCast(e[2]);
    }
    return 0;
}

fn decompOf(cp: u21) ?[3]u21 {
    var lo: usize = 0;
    var hi: usize = table.decomp.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const e = table.decomp[mid];
        if (e[0] == cp) return e;
        if (e[0] < cp) lo = mid + 1 else hi = mid;
    }
    return null;
}

test "nfc: composes, reorders marks, splits precomposed + mark, keeps exclusions" {
    const a = std.testing.allocator;
    // Expected values are Python unicodedata.normalize("NFC", input).
    const cases = [_][2][]const u8{
        .{ "e\u{301}te\u{301} a\u{300} c\u{327}a o\u{302}", "été à ça ô" },
        .{ "e\u{323}\u{302}", "\u{1EC7}" }, // canonical order
        .{ "e\u{302}\u{323}", "\u{1EC7}" }, // non-canonical order: reorder first
        .{ "\u{E9}\u{323}", "\u{1EB9}\u{301}" }, // precomposed + mark: decompose first
        .{ "\u{212B}", "\u{C5}" }, // singleton decomposition
        .{ "\u{958}", "\u{915}\u{93C}" }, // composition exclusion stays split
        .{ "か\u{3099}", "が" },
        .{ "\u{1112}\u{1161}\u{11AB}", "한" },
        .{ "\u{301}e", "\u{301}e" }, // a leading mark does not compose
    };
    for (cases) |c| {
        const got = (try normalize(a, c[0])) orelse c[0];
        defer if (got.ptr != c[0].ptr) a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
    // NFC already, ASCII, invalid UTF-8: unchanged, nothing allocated.
    for ([_][]const u8{ "déjà vu", "plain ascii", "한국어", "\xff\xfe" }) |s|
        try std.testing.expect((try normalize(a, s)) == null);
}
