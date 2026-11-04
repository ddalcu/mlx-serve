//! Named, immutable codebook constant tables: what a pack's `expert_quant` points at with `codebook_table` and
//! `codebook_digest`. One row of u32 constants per rate n = 16K, every even n of the table's own range (within 16 to
//! 128), per codebook.
//!
//! A SPEC1 table's window is the one it was tuned at, nothing more: any window the engine admits for MCG decodes it,
//! and the kernels read the row of each bank's own rate from the table (`Decode.table`). PAIR1's pair codec reads
//! the whole 16-bit window, so a PAIR1 table holds only at its own window, and its rows are the ones the kernels
//! have compiled (`format.pair1Pair`).
//!
//! STD1 and SPEC1 are table-driven only: a pack names one of `shipped`, and a pack naming none this build holds is
//! refused. `std1` is one pair at every rate.
//!
//! To add a SPEC1 table: append one entry to `shipped` (name, window, rows, digest); no kernel changes.
const std = @import("std");
const builtin = @import("builtin");
const format = @import("expert_exl3.zig");

/// The range of the tables from before K8 (n = 24..64), which PAIR1's compiled rows still cover.
const k4_lo: u32 = 24;
const k4_count = (64 - k4_lo) / 2 + 1;

pub const Refusal = error{
    Exl3CodebookTableUnknown,
    Exl3CodebookTableDigestMismatch,
    Exl3CodebookTableWindowMismatch,
    Exl3CodebookTableCodebookMismatch,
    Exl3CodebookTableRateMissing,
    Exl3CodebookTableLegacyConfig,
    Exl3CodebookTableUndecodable,
};

pub const Row = [4]u32;

pub const Table = struct {
    name: []const u8,
    codebook: format.Codebook,
    window: format.Window,
    /// First 8 hex digits of sha256 over the canonical text, pinned so a shipped table cannot change.
    digest: []const u8,
    /// How many of a row's constants the codebook uses ((std1 and spec1 2, pair1 4)).
    fields: usize,
    /// The rate of `rows[0]`; row i is rate n_lo + 2i.
    n_lo: u32 = k4_lo,
    rows: []const Row,
    /// Digests of this table's earlier, shorter versions: their rows are this table's at their rates, and a pack
    /// stamped with one is refused as a legacy config.
    superseded: []const []const u8 = &.{},

    pub fn row(self: *const Table, rate: format.Rate) ?Row {
        if (rate.n % 2 != 0 or rate.n < self.n_lo) return null;
        const i = (rate.n - self.n_lo) / 2;
        return if (i < self.rows.len) self.rows[i] else null;
    }

    pub fn rateOf(self: *const Table, i: usize) format.Rate {
        return .{ .n = self.n_lo + 2 * @as(u32, @intCast(i)) };
    }

    /// The rates this table has a row for, as the bit set `Rate.in` reads (bit n/2).
    pub fn rates(self: *const Table) u128 {
        var set: u128 = 0;
        for (0..self.rows.len) |i| set |= @as(u128, 1) << @intCast(self.rateOf(i).n / 2);
        return set;
    }

    /// Whether a pack at `window` may name this table: STD1 and SPEC1 decode at any window, PAIR1 only at its own.
    pub fn admits(self: *const Table, window: format.Window) bool {
        return self.codebook.tabledHashMask() or self.window == window;
    }

    /// Whether the kernels decode this table's rows: STD1's and SPEC1's are data, PAIR1's are the ones compiled in.
    pub fn decodable(self: *const Table) bool {
        if (self.codebook.tabledHashMask()) return true;
        for (self.rows, 0..) |r, i| {
            const want = compiledRow(self.codebook, self.rateOf(i)) orelse return false;
            if (!std.mem.eql(u32, r[0..self.fields], want[0..self.fields])) return false;
        }
        return true;
    }
};

fn compiledRow(codebook: format.Codebook, rate: format.Rate) ?Row {
    switch (codebook) {
        .pair1 => {
            const p = format.pair1Pair(rate) orelse return null;
            return .{ p.mult, p.rot, p.mask0, p.mask1 };
        },
        else => return null,
    }
}

fn compiledRows(codebook: format.Codebook) [k4_count]Row {
    var out: [k4_count]Row = undefined;
    for (&out, 0..) |*r, i| r.* = compiledRow(codebook, .{ .n = k4_lo + 2 * @as(u32, @intCast(i)) }).?;
    return out;
}

const Pair = struct { u32, u32 };

/// `pair1-w16-v1` is, by definition, what the kernels have compiled since before the tables existed.
const shipped = [_]Table{
    .{ .name = "pair1-w16-v1", .codebook = .pair1, .window = .w16, .digest = "c089b4eb", .fields = 4, .rows = &compiledRows(.pair1) },
    .{ .name = "spec1-w15", .codebook = .spec1, .window = .w15, .digest = "37f72431", .fields = 2, .n_lo = 16, .rows = &tabledRows(w15Pairs()), .superseded = &.{"2ac2594d"} },
    .{ .name = "spec1-w16", .codebook = .spec1, .window = .w16, .digest = "753ffa03", .fields = 2, .rows = &tabledRows(w16Pairs()) },
    .{ .name = "std1", .codebook = .std1, .window = .w15, .digest = "c5dba171", .fields = 2, .rows = &tabledRows(@as([k4_count]Pair, @splat(global_pair))) },
};

/// The set at window 15, n = 16..128. Its rows
/// n = 24..64 are the 21-row table it extends (digest `2ac2594d`), so a pack stamped with that digest decodes the same.
fn w15Pairs() [57]Pair {
    return .{
        .{ 0x84d1cf89, 0x8fe08947 },
        .{ 0x84c1cf49, 0x8fe48a47 },
        .{ 0x84d1cfe9, 0x8fe08a47 },
        .{ 0x84d1cfc9, 0x8ff08b67 },
        .{ 0x84c1cf49, 0x8fe08b47 },
        .{ 0x84c1cf99, 0x8fa08b66 },
        .{ 0x862c1d37, 0x8ff28c50 },
        .{ 0x84641fb7, 0x8fa08c40 },
        .{ 0x846c1fa7, 0x8fe08c80 },
        .{ 0x844c1fd7, 0x8ffe8c98 },
        .{ 0x846c1fa7, 0x8ffe8d10 },
        .{ 0x845c1f77, 0x8ffa8e10 },
        .{ 0x843c1fa7, 0x8ffe8e40 },
        .{ 0x843c1fa7, 0x8ffe8e40 },
        .{ 0x842c1fc7, 0x8fff8f00 },
        .{ 0x843c1fb7, 0x8ffe8f10 },
        .{ 0x84181fb5, 0x8fe08f80 },
        .{ 0x84081f75, 0x8ffa8f72 },
        .{ 0x84081f75, 0x8fd68fd0 },
        .{ 0x84241f27, 0x8ffc8fc0 },
        .{ 0x84081f73, 0x8fe08f80 },
        .{ 0x84081f75, 0x8fbe8ff0 },
        .{ 0x3fed4cef, 0x8fe08fe0 },
        .{ 0x3fed4cef, 0x8fc18ff2 },
        .{ 0x3fed4cef, 0x8ff08ff0 },
        .{ 0x3fed4cef, 0x8fe28ff0 },
        .{ 0x3fed4cef, 0x8fe08fe0 },
        .{ 0x80081b7b, 0x8ff08fc8 },
        .{ 0x80081b7b, 0x8ff48fc0 },
        .{ 0x400036cb, 0x8f448fff },
        .{ 0x400036cb, 0x8fe48fff },
        .{ 0x400036cb, 0x8f448ff9 },
        .{ 0x8000387b, 0x8fe08ff7 },
        .{ 0x8000387b, 0x8fc08fee },
        .{ 0x80003779, 0x8fe88ffe },
        .{ 0x80003779, 0x8fc88ffc },
        .{ 0x80003779, 0x8fca8ffc },
        .{ 0x80003779, 0x8fcc8ffe },
        .{ 0x80003879, 0x8ffc8fef },
        .{ 0x50003851, 0x8ff98fff },
        .{ 0xd0003751, 0x8ff98fff },
        .{ 0x50003851, 0x8ff98ff6 },
        .{ 0xd0003931, 0x8ff98fff },
        .{ 0xd0003931, 0x8ff98ffa },
        .{ 0xd0003931, 0x8ff48ffa },
        .{ 0x300031b1, 0x8ffb8ffc },
        .{ 0x300031b1, 0x8ffa8ffe },
        .{ 0x300031b1, 0x8ffb8ffc },
        .{ 0xc00031f1, 0x8ffe8ffa },
        .{ 0x400031f1, 0x8ffe8ffe },
        .{ 0x400031f1, 0x8ffe8ffe },
        .{ 0x40003931, 0x8ffc8ffc },
        .{ 0x40003931, 0x8ffc8ffc },
        .{ 0x40003931, 0x8ff88ffe },
        .{ 0x40003931, 0x8ff88ffc },
        .{ 0x80003811, 0x8ffc8ffc },
        .{ 0x80003811, 0x8ffc8ffc },
    };
}

/// The set at window 16, n = 24..64.
fn w16Pairs() [k4_count]Pair {
    return .{
        .{ 0x862d1c19, 0x8fee8c30 },
        .{ 0x862d1d37, 0x8fee8c30 },
        .{ 0x862d1d37, 0x8ffe8c10 },
        .{ 0x842f1f37, 0x8ffc8d10 },
        .{ 0x840f1f67, 0x8ffc8d10 },
        .{ 0x844f1f77, 0x8ffe8d40 },
        .{ 0x840d1f37, 0x8ffe8e10 },
        .{ 0x84091f33, 0x8fe88e00 },
        .{ 0x844d1f77, 0x8fe08e00 },
        .{ 0x843c1fa7, 0x8ffe8e40 },
        .{ 0x844c1f7f, 0x8ffc8f02 },
        .{ 0x841c1fb9, 0x8fe88f04 },
        .{ 0x842c1fc7, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8ffe8fc1 },
        .{ 0x83fc1fa3, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8fee8fc0 },
        .{ 0x84061f77, 0x8fd08f80 },
        .{ 0x84081f6f, 0x8ffe8fc0 },
        .{ 0x84081f25, 0x8ffe8fe0 },
        .{ 0x83f81f6f, 0x8ff48fc0 },
    };
}

/// The one pair `std1` decodes with, at every rate.
const global_pair: Pair = .{ 0x840C1F37, 0x8FFE8FD0 };

fn tabledRows(pairs: anytype) [pairs.len]Row {
    var out: [pairs.len]Row = undefined;
    for (&out, pairs) |*r, p| r.* = .{ p[0], p[1], 0, 0 };
    return out;
}

/// Test-only SPEC1 table whose pair differs at every rate, so a decoder that read one pair for every bank fails.
fn syntheticPairs() [57]Pair {
    const masks = [_]u32{ 0x8FFE8FD0, 0x8FE08FC0, 0x8FF08FE0, 0x8F8E8FD8 };
    var out: [57]Pair = undefined;
    for (&out, 0..) |*p, i| p.* = .{ 0x840C1F37 +% (16 + 2 * @as(u32, @intCast(i))) *% 0x2E0B5, masks[i % masks.len] };
    return out;
}

const test_only = [_]Table{
    .{ .name = "spec1-test-per-rate", .codebook = .spec1, .window = .w12, .digest = "a2f2fe40", .fields = 2, .n_lo = 16, .rows = &tabledRows(syntheticPairs()) },
};

pub const tables = shipped ++ if (builtin.is_test) test_only else [0]Table{};

pub fn indexOf(comptime name: []const u8) u8 {
    return comptime for (tables, 0..) |t, i| {
        if (std.mem.eql(u8, t.name, name)) break @intCast(i);
    } else @compileError("no codebook table " ++ name);
}

pub fn index(t: *const Table) u8 {
    return @intCast(t - &tables[0]);
}

/// The SPEC1 pair a bank at `rate` decodes with under table `which`; null where the table has no row.
pub fn tabledPair(which: u8, rate: format.Rate) ?format.HashMask {
    const r = tables[which].row(rate) orelse return null;
    return .{ .mult = r[0], .mask = r[1] };
}

pub fn lookup(name: []const u8) ?*const Table {
    for (&tables) |*t| if (std.mem.eql(u8, t.name, name)) return t;
    return null;
}

/// First 8 hex digits of sha256 over one line per row, ascending n: `<n decimal>,<each field as 8 lowercase hex
/// digits, comma separated>\n`.
pub fn digestOf(fields: usize, ns: []const u32, rows: []const Row) [8]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (ns, rows) |n, r| {
        var line: [64]u8 = undefined;
        var len = (std.fmt.bufPrint(&line, "{d}", .{n}) catch unreachable).len;
        for (r[0..fields]) |v| len += (std.fmt.bufPrint(line[len..], ",{x:0>8}", .{v}) catch unreachable).len;
        line[len] = '\n';
        h.update(line[0 .. len + 1]);
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return std.fmt.bytesToHex(out[0..4].*, .lower);
}

pub fn digestOfTable(t: *const Table) [8]u8 {
    var ns: [(format.Rate.max_n - format.Rate.min_n) / 2 + 1]u32 = undefined;
    for (ns[0..t.rows.len], 0..) |*n, i| n.* = t.rateOf(i).n;
    return digestOf(t.fields, ns[0..t.rows.len], t.rows);
}

/// The table `block` names for `codebook` at `window`, refusing by name. Only std1, spec1 and pair1 have tables.
pub fn resolve(block: std.json.ObjectMap, codebook: format.Codebook, window: format.Window) Refusal!*const Table {
    if (!codebook.hasTable()) return error.Exl3CodebookTableCodebookMismatch;
    const name = block.get("codebook_table") orelse return error.Exl3CodebookTableUnknown;
    if (name != .string) return error.Exl3CodebookTableUnknown;
    const table = lookup(name.string) orelse return error.Exl3CodebookTableUnknown;
    const digest = block.get("codebook_digest") orelse return error.Exl3CodebookTableDigestMismatch;
    if (digest != .string) return error.Exl3CodebookTableDigestMismatch;
    for (table.superseded) |old| if (std.mem.eql(u8, digest.string, old)) return error.Exl3CodebookTableLegacyConfig;
    if (!std.mem.eql(u8, digest.string, table.digest)) return error.Exl3CodebookTableDigestMismatch;
    if (table.codebook != codebook) return error.Exl3CodebookTableCodebookMismatch;
    if (!table.admits(window)) return error.Exl3CodebookTableWindowMismatch;
    if (!table.decodable()) return error.Exl3CodebookTableUndecodable;
    return table;
}

test "the canonical digest is pinned by a fixed vector" {
    const t = std.testing;
    const rows = [_]Row{ .{ 1, 2, 3, 4 }, .{ 0xFFFFFFFF, 0, 0x80000000, 0x7FFFFFFF } };
    try t.expectEqualStrings("72fc0906", &digestOf(4, &.{ 24, 64 }, &rows));
}

test "every table keeps its pinned digest, and the tables from before K8 one row per even n from 24 to 64" {
    const t = std.testing;
    for (&tables) |*table| {
        try t.expectEqualStrings(table.digest, &digestOfTable(table));
        try t.expect(table.decodable());
    }
    for ([_][]const u8{ "pair1-w16-v1", "spec1-w16", "std1" }) |name| {
        try t.expectEqual(@as(usize, 21), lookup(name).?.rows.len);
        try t.expectEqual(@as(u32, 24), lookup(name).?.n_lo);
    }
    try t.expectEqualStrings("c089b4eb", lookup("pair1-w16-v1").?.digest);
    try t.expectEqualStrings("37f72431", lookup("spec1-w15").?.digest);
    try t.expectEqualStrings("c5dba171", lookup("std1").?.digest);
    try t.expectEqualStrings("753ffa03", lookup("spec1-w16").?.digest);
    const s1 = lookup("spec1-w15").?;
    try t.expectEqual(Row{ 0x84c1cf49, 0x8fe08b47, 0, 0 }, s1.row(.{ .n = 24 }).?);
    try t.expectEqual(Row{ 0x3fed4cef, 0x8ff08ff0, 0, 0 }, s1.row(.{ .n = 64 }).?);
    try t.expect(s1.row(.{ .n = 14 }) == null and s1.row(.{ .n = 25 }) == null and s1.row(.{ .n = 130 }) == null);
    try t.expectEqual(Row{ 0x58F41BF3, 3, 0x8ED78C13, 0x8EEE8C03 }, lookup("pair1-w16-v1").?.row(.{ .n = 24 }).?);
    try t.expectEqual(Row{ 0xE2C7FC0D, 19, 0x8F408FA0, 0x8F118FC9 }, lookup("pair1-w16-v1").?.row(.{ .n = 64 }).?);
}

test "a PAIR1 table the kernels have not compiled is undecodable, never decoded with another table's constants" {
    const t = std.testing;
    var other = lookup("pair1-w16-v1").?.*;
    try t.expect(other.decodable());
    var rows = other.rows[0..k4_count].*;
    rows[3][0] +%= 1;
    other.rows = &rows;
    try t.expect(!other.decodable());
}

test "a SPEC1 table is data: any rows decode, and the test table's pair differs at every rate" {
    const t = std.testing;
    var other = lookup("spec1-w15").?.*;
    var rows = other.rows[0..k4_count].*;
    rows[3][0] +%= 1;
    other.rows = &rows;
    try t.expect(other.decodable());
    const synthetic = lookup("spec1-test-per-rate").?;
    for (synthetic.rows[1..], synthetic.rows[0 .. synthetic.rows.len - 1]) |b, a| try t.expect(a[0] != b[0] and !std.mem.eql(u32, a[0..2], b[0..2]));
    try t.expect(tabledPair(index(synthetic), .{ .n = 14 }) == null and tabledPair(index(synthetic), .{ .n = 130 }) == null);
    try t.expectEqual(synthetic.rows[12][0], tabledPair(index(synthetic), .{ .n = 40 }).?.mult);
}

fn resolveText(raw: []const u8, codebook: format.Codebook, window: format.Window) Refusal!*const Table {
    const t = std.testing;
    const parsed = std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{}) catch unreachable;
    defer parsed.deinit();
    return resolve(parsed.value.object, codebook, window);
}

test "a config names its table and every refusal is named" {
    const t = std.testing;
    const ok_pair1 = "{\"codebook_table\":\"pair1-w16-v1\",\"codebook_digest\":\"c089b4eb\"";
    try t.expectEqualStrings("pair1-w16-v1", (try resolveText(ok_pair1 ++ "}", .pair1, .w16)).name);
    // A SPEC1 table's window is provenance: every admitted window decodes it.
    for (std.enums.values(format.Window)) |w| {
        try t.expectEqualStrings("spec1-w15", (try resolveText("{\"codebook_table\":\"spec1-w15\",\"codebook_digest\":\"37f72431\"}", .spec1, w)).name);
    }
    for ([_]struct { raw: []const u8, cb: format.Codebook, w: format.Window, want: Refusal }{
        .{ .raw = "{}", .cb = .pair1, .w = .w16, .want = error.Exl3CodebookTableUnknown },
        .{ .raw = "{\"codebook_table\":\"pair1-w16-v9\",\"codebook_digest\":\"c089b4eb\"}", .cb = .pair1, .w = .w16, .want = error.Exl3CodebookTableUnknown },
        .{ .raw = "{\"codebook_table\":7,\"codebook_digest\":\"c089b4eb\"}", .cb = .pair1, .w = .w16, .want = error.Exl3CodebookTableUnknown },
        .{ .raw = "{\"codebook_table\":\"pair1-w16-v1\"}", .cb = .pair1, .w = .w16, .want = error.Exl3CodebookTableDigestMismatch },
        .{ .raw = "{\"codebook_table\":\"pair1-w16-v1\",\"codebook_digest\":\"deadbeef\"}", .cb = .pair1, .w = .w16, .want = error.Exl3CodebookTableDigestMismatch },
        .{ .raw = "{\"codebook_table\":\"pair1-w16-v1\",\"codebook_digest\":\"c089b4eb\"}", .cb = .spec1, .w = .w16, .want = error.Exl3CodebookTableCodebookMismatch },
        .{ .raw = "{\"codebook_table\":\"pair1-w16-v1\",\"codebook_digest\":\"c089b4eb\"}", .cb = .mcg, .w = .w16, .want = error.Exl3CodebookTableCodebookMismatch },
        .{ .raw = "{\"codebook_table\":\"pair1-w16-v1\",\"codebook_digest\":\"c089b4eb\"}", .cb = .pair1, .w = .w15, .want = error.Exl3CodebookTableWindowMismatch },
        .{ .raw = "{\"codebook_table\":\"std1\",\"codebook_digest\":\"2ac2594d\"}", .cb = .std1, .w = .w15, .want = error.Exl3CodebookTableDigestMismatch },
        .{ .raw = "{\"codebook_table\":\"std1\",\"codebook_digest\":\"c5dba171\"}", .cb = .spec1, .w = .w15, .want = error.Exl3CodebookTableCodebookMismatch },
        .{ .raw = "{}", .cb = .spec1, .w = .w15, .want = error.Exl3CodebookTableUnknown },
        // spec1-w15 before it grew to n = 16..128.
        .{ .raw = "{\"codebook_table\":\"spec1-w15\",\"codebook_digest\":\"2ac2594d\"}", .cb = .spec1, .w = .w15, .want = error.Exl3CodebookTableLegacyConfig },
    }) |c| try t.expectError(c.want, resolveText(c.raw, c.cb, c.w));
}

test "std1 is the one pair at every rate and decodes at every window" {
    const t = std.testing;
    const table = lookup("std1").?;
    try t.expect(table.codebook == .std1 and table.window == .w15 and table.fields == 2);
    for (std.enums.values(format.Window)) |w| try t.expect(table.admits(w));
    for (table.rows) |r| try t.expectEqual(Row{ 0x840C1F37, 0x8FFE8FD0, 0, 0 }, r);
    for (std.enums.values(format.Window)) |w| {
        const ok = "{\"codebook_table\":\"std1\",\"codebook_digest\":\"c5dba171\"}";
        try t.expectEqualStrings("std1", (try resolveText(ok, .std1, w)).name);
    }
}

test "std1 stops at K4 while spec1-w15 reaches K1 and K8" {
    const t = std.testing;
    const std1 = lookup("std1").?;
    const spec1 = lookup("spec1-w15").?;
    for ([_]u32{ 16, 18, 70, 120, 128 }) |n| {
        try t.expect(std1.row(.{ .n = n }) == null);
        try t.expect(n % 2 != 0 or spec1.row(.{ .n = n }) != null);
    }
    for ([_]u32{ 24, 36, 64 }) |n| try t.expect(std1.row(.{ .n = n }) != null);
}

test "spec1-w16 is the searched set at window 16" {
    const t = std.testing;
    const table = lookup("spec1-w16").?;
    try t.expect(table.codebook == .spec1 and table.window == .w16 and table.fields == 2);
    for (std.enums.values(format.Window)) |w| try t.expect(table.admits(w));
    const searched = [_]Pair{
        .{ 0x862d1c19, 0x8fee8c30 },
        .{ 0x862d1d37, 0x8fee8c30 },
        .{ 0x862d1d37, 0x8ffe8c10 },
        .{ 0x842f1f37, 0x8ffc8d10 },
        .{ 0x840f1f67, 0x8ffc8d10 },
        .{ 0x844f1f77, 0x8ffe8d40 },
        .{ 0x840d1f37, 0x8ffe8e10 },
        .{ 0x84091f33, 0x8fe88e00 },
        .{ 0x844d1f77, 0x8fe08e00 },
        .{ 0x843c1fa7, 0x8ffe8e40 },
        .{ 0x844c1f7f, 0x8ffc8f02 },
        .{ 0x841c1fb9, 0x8fe88f04 },
        .{ 0x842c1fc7, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8ffe8fc1 },
        .{ 0x83fc1fa3, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8fee8fc0 },
        .{ 0x84061f77, 0x8fd08f80 },
        .{ 0x84081f6f, 0x8ffe8fc0 },
        .{ 0x84081f25, 0x8ffe8fe0 },
        .{ 0x83f81f6f, 0x8ff48fc0 },
    };
    for (searched, 0..) |p, i| try t.expectEqual(Row{ p[0], p[1], 0, 0 }, table.rows[i]);
}

test "spec1-w15 holds the searched pairs at every rate" {
    const t = std.testing;
    const table = lookup("spec1-w15").?;
    try t.expectEqualStrings("37f72431", table.digest);
    try t.expectEqual(@as(usize, 57), table.rows.len);
    // The 21-row version it replaced, digest 2ac2594d, is its rows at n = 24..64.
    var ns: [k4_count]u32 = undefined;
    for (&ns, 0..) |*n, i| n.* = k4_lo + 2 * @as(u32, @intCast(i));
    try t.expectEqualStrings("2ac2594d", &digestOf(2, &ns, table.rows[(k4_lo - table.n_lo) / 2 ..][0..k4_count]));
    try t.expectEqual(Row{ 0x84d1cf89, 0x8fe08947, 0, 0 }, table.row(.{ .n = 16 }).?);
    try t.expectEqual(Row{ 0x84d1cfc9, 0x8ff08b67, 0, 0 }, table.row(.{ .n = 22 }).?);
    try t.expectEqual(Row{ 0x3fed4cef, 0x8fe28ff0, 0, 0 }, table.row(.{ .n = 66 }).?);
    try t.expectEqual(Row{ 0x8000387b, 0x8fe08ff7, 0, 0 }, table.row(.{ .n = 80 }).?);
    try t.expectEqual(Row{ 0xd0003751, 0x8ff98fff, 0, 0 }, table.row(.{ .n = 96 }).?);
    try t.expectEqual(Row{ 0xc00031f1, 0x8ffe8ffa, 0, 0 }, table.row(.{ .n = 112 }).?);
    try t.expectEqual(Row{ 0x80003811, 0x8ffc8ffc, 0, 0 }, table.row(.{ .n = 128 }).?);
    try t.expect(format.Rate.fromK(8).in(table.rates()) and format.Rate.fromK(1).in(table.rates()));
    try t.expect(table.window == .w15 and table.admits(.w8) and table.admits(.w16));
    const searched = [_]struct { n: u32, mult: u32, mask: u32 }{
        .{ .n = 24, .mult = 0x84c1cf49, .mask = 0x8fe08b47 },
        .{ .n = 26, .mult = 0x84c1cf99, .mask = 0x8fa08b66 },
        .{ .n = 28, .mult = 0x862c1d37, .mask = 0x8ff28c50 },
        .{ .n = 30, .mult = 0x84641fb7, .mask = 0x8fa08c40 },
        .{ .n = 32, .mult = 0x846c1fa7, .mask = 0x8fe08c80 },
        .{ .n = 34, .mult = 0x844c1fd7, .mask = 0x8ffe8c98 },
        .{ .n = 36, .mult = 0x846c1fa7, .mask = 0x8ffe8d10 },
        .{ .n = 38, .mult = 0x845c1f77, .mask = 0x8ffa8e10 },
        .{ .n = 40, .mult = 0x843c1fa7, .mask = 0x8ffe8e40 },
        .{ .n = 42, .mult = 0x843c1fa7, .mask = 0x8ffe8e40 },
        .{ .n = 44, .mult = 0x842c1fc7, .mask = 0x8fff8f00 },
        .{ .n = 46, .mult = 0x843c1fb7, .mask = 0x8ffe8f10 },
        .{ .n = 48, .mult = 0x84181fb5, .mask = 0x8fe08f80 },
        .{ .n = 50, .mult = 0x84081f75, .mask = 0x8ffa8f72 },
        .{ .n = 52, .mult = 0x84081f75, .mask = 0x8fd68fd0 },
        .{ .n = 54, .mult = 0x84241f27, .mask = 0x8ffc8fc0 },
        .{ .n = 56, .mult = 0x84081f73, .mask = 0x8fe08f80 },
        .{ .n = 58, .mult = 0x84081f75, .mask = 0x8fbe8ff0 },
    };
    for (searched) |r| try t.expectEqual(Row{ r.mult, r.mask, 0, 0 }, table.row(.{ .n = r.n }).?);
    for ([_]Row{ .{ 0x3fed4cef, 0x8fe08fe0, 0, 0 }, .{ 0x3fed4cef, 0x8fc18ff2, 0, 0 }, .{ 0x3fed4cef, 0x8ff08ff0, 0, 0 } }, [_]u32{ 60, 62, 64 }) |want, n| try t.expectEqual(want, table.row(.{ .n = n }).?);
    try t.expectEqual(table.row(.{ .n = 40 }).?, table.row(.{ .n = 42 }).?);
    try t.expectEqual(format.HashMask{ .mult = 0x84181FB5, .mask = 0x8FE08F80 }, tabledPair(index(table), .{ .n = 48 }).?);
}
