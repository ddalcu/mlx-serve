//! A served drafter's round ladder (`rounds.Ladder`), stored once per release: the first boot times it and stores it,
//! later boots read it, so every boot of one build verifies the same draft counts. The key is the build (its id covers
//! the executable or release and the MLX build), the chip, the RAM, the model and the server's `MLX_` settings, bar the
//! diagnostics and `MLX_SERVE_ROUNDS_LADDER` itself. `MLX_SERVE_ROUNDS_LADDER` names the folder; `=0` stores nothing.

const std = @import("std");
const builtin = @import("builtin");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const rounds = @import("../../../sdk/spec/rounds.zig");

const Ladder = rounds.Ladder;
const MAX_ROWS = rounds.MAX_ROWS;

/// What a ladder depends on.
pub const Parts = struct { build: []const u8, chip: []const u8, mem_bytes: u64, model_dir: []const u8, levers: [32]u8 };

/// The server's `MLX_` settings as one digest, in any order, bar the diagnostics and the folder variable.
pub fn levers() [32]u8 {
    var acc: [32]u8 = @splat(0);
    if (comptime builtin.os.tag == .windows) return acc; // no C `environ` to walk; drafters serve on Metal
    var i: usize = 0;
    while (std.c.environ[i]) |e| : (i += 1) {
        const entry = std.mem.span(e);
        if (!std.mem.startsWith(u8, entry, "MLX_")) continue;
        if (std.mem.indexOf(u8, entry, "DIAG") != null or std.mem.startsWith(u8, entry, "MLX_SERVE_ROUNDS_LADDER=")) continue;
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(entry, &d, .{});
        for (&acc, d) |*x, y| x.* ^= y;
    }
    return acc;
}

/// The key's id: 24 hex digits of the sha256 of its parts.
pub fn keyId(p: Parts) [24]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ p.build, p.chip, p.model_dir }) |part| {
        h.update(part);
        h.update("\x00");
    }
    h.update(std.mem.asBytes(&((p.mem_bytes + (1 << 29)) >> 30)));
    h.update(&p.levers);
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.bytesToHex(d[0..12].*, .lower);
}

/// `ladder1`, then `plain`, `block`, and the `copies` and `drafts` rows, ms with 4 decimals.
pub fn encode(buf: []u8, l: *const Ladder) ![]const u8 {
    var w = std.Io.Writer.fixed(buf);
    try w.print("ladder1\nplain {d:.4}\nblock {d}\n", .{ l.plain, l.block });
    for ([_][]const u8{ "copies", "drafts" }, [_]*const [MAX_ROWS]f32{ &l.copies, &l.drafts }) |name, row| {
        try w.writeAll(name);
        for (row) |ms| try w.print(" {d:.4}", .{ms});
        try w.writeByte('\n');
    }
    return w.buffered();
}

/// Null on any other version or shape, and on a cost a timed ladder cannot hold.
pub fn decode(text: []const u8) ?Ladder {
    var lines = std.mem.splitScalar(u8, text, '\n');
    if (!std.mem.eql(u8, lines.next() orelse return null, "ladder1")) return null;
    var l: Ladder = .{ .plain = 0, .copies = undefined, .drafts = undefined, .block = 0 };
    const plain = lines.next() orelse return null;
    if (!std.mem.startsWith(u8, plain, "plain ")) return null;
    l.plain = std.fmt.parseFloat(f32, plain[6..]) catch return null;
    const block = lines.next() orelse return null;
    if (!std.mem.startsWith(u8, block, "block ")) return null;
    l.block = std.fmt.parseInt(u32, block[6..], 10) catch return null;
    if (l.block == 0 or l.block > rounds.MAX_DRAFTS) return null;
    for ([_][]const u8{ "copies", "drafts" }, [_]*[MAX_ROWS]f32{ &l.copies, &l.drafts }) |name, row| {
        var f = std.mem.splitScalar(u8, lines.next() orelse return null, ' ');
        if (!std.mem.eql(u8, f.next() orelse return null, name)) return null;
        for (row) |*ms| ms.* = std.fmt.parseFloat(f32, f.next() orelse return null) catch return null;
        if (f.next() != null) return null;
    }
    if (!std.math.isFinite(l.plain) or l.plain <= 0) return null;
    for ([_]f32{l.drafts[0]} ++ l.copies) |ms| if (!std.math.isFinite(ms) or ms <= 0) return null;
    for (l.drafts[1 .. l.block + 1]) |ms| if (!std.math.isFinite(ms) or ms <= 0) return null;
    return l;
}

/// The folder: the variable's value, else `~/.mlx-serve/rounds` (made when missing; none in a test build, which stays
/// hermetic); null at `0`.
pub fn dir(io: std.Io, buf: []u8) ?[]const u8 {
    if (env.span(.MLX_SERVE_ROUNDS_LADDER)) |v| return if (v.len == 0 or v[0] == '0') null else v;
    if (builtin.is_test) return null;
    const home = std.mem.span(env.get(.HOME) orelse return null);
    const d = std.fmt.bufPrint(buf, "{s}/.mlx-serve/rounds", .{home}) catch return null;
    std.Io.Dir.cwd().createDirPath(io, d) catch return null;
    return d;
}

/// `<folder>/rounds-ladder-<id>.txt`.
pub fn path(buf: []u8, folder: []const u8, id: []const u8) ?[]const u8 {
    return std.fmt.bufPrint(buf, "{s}/rounds-ladder-{s}.txt", .{ folder, id }) catch null;
}

/// The ladder stored at `p`; null when there is none or it does not parse.
pub fn read(io: std.Io, p: []const u8) ?Ladder {
    const f = std.Io.Dir.openFileAbsolute(io, p, .{}) catch return null;
    defer f.close(io);
    var rb: [1024]u8 = undefined;
    var rs = f.reader(io, &rb);
    var text: [2048]u8 = undefined;
    const n = rs.interface.readSliceShort(&text) catch return null;
    return decode(std.mem.trimEnd(u8, text[0..n], "\n"));
}

/// Writes `l` to `p` through a temporary file, so a reader never sees half of one.
pub fn store(io: std.Io, p: []const u8, l: *const Ladder) !void {
    var body: [2048]u8 = undefined;
    const text = try encode(&body, l);
    var tmp_buf: [600]u8 = undefined;
    const tmp = try std.fmt.bufPrint(&tmp_buf, "{s}.tmp", .{p});
    {
        const f = try std.Io.Dir.createFileAbsolute(io, tmp, .{});
        defer f.close(io);
        var wb: [2048]u8 = undefined;
        var fw = f.writer(io, &wb);
        try fw.interface.writeAll(text);
        try fw.interface.flush();
    }
    try std.Io.Dir.renameAbsolute(tmp, p, io);
}

/// At load: the stored ladder of `parts` into `b.ladder` (then nothing times it), else its path kept in `b` for
/// `keep`. Logs `[rounds] ladder read` or `[rounds] ladder: none stored`.
pub fn restore(io: std.Io, b: anytype, parts: Parts) void {
    var dir_buf: [512]u8 = undefined;
    const folder = dir(io, &dir_buf) orelse return;
    const id = keyId(parts);
    const p = path(&b.ladder_file, folder, &id) orelse return;
    b.ladder_file_len = p.len;
    if (read(io, p)) |l| {
        b.ladder = l;
        b.ladder_asked = true;
        log.info("[rounds] ladder read: {s} (plain step {d:.1} ms, {d} drafts)\n", .{ p, l.plain, l.block });
    } else log.info("[rounds] ladder: none stored at {s}; the load times it\n", .{p});
}

/// After the timing request: `l` stored where `restore` named, for every later boot of this key.
pub fn keep(io: std.Io, b: anytype, l: *const Ladder) void {
    if (b.ladder_file_len == 0) return;
    const p = b.ladder_file[0..b.ladder_file_len];
    store(io, p, l) catch |e| return log.warn("[rounds] ladder not stored ({s}) at {s}\n", .{ @errorName(e), p });
    log.info("[rounds] ladder stored: {s}\n", .{p});
}

const testing = std.testing;

fn testLadder() Ladder {
    var l: Ladder = .{ .plain = 12.5, .copies = undefined, .drafts = undefined, .block = 6 };
    for (&l.copies, &l.drafts, 0..) |*c, *d, k| {
        c.* = 13.0 + 1.25 * @as(f32, @floatFromInt(k));
        d.* = 14.0 + 1.5 * @as(f32, @floatFromInt(k));
    }
    return l;
}

// Bar: a ladder is stored once and read back to the stored values; another key (a lever, the build, the model) is a
// miss, and a later store under the first key leaves the other key's file absent.
test "ladder store: a timed ladder is stored once, read back, and a key change misses" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(testing.io, &root_buf)];
    const base: Parts = .{ .build = "dev 0123456789abcdef", .chip = "Apple M5 Max", .mem_bytes = 128 << 30, .model_dir = "/models/inkling2", .levers = @splat(0) };
    const l = testLadder();
    var p_buf: [600]u8 = undefined;
    const id = keyId(base);
    const p = path(&p_buf, root, &id).?;
    try testing.expect(read(testing.io, p) == null);
    try store(testing.io, p, &l);
    const got = read(testing.io, p).?;
    try testing.expectEqual(l.plain, got.plain);
    try testing.expectEqual(l.block, got.block);
    for (l.copies, got.copies) |x, y| try testing.expectApproxEqAbs(x, y, 1e-4);
    for (l.drafts, got.drafts) |x, y| try testing.expectApproxEqAbs(x, y, 1e-4);
    var lever = base;
    lever.levers[0] = 1;
    var build = base;
    build.build = "dev fedcba9876543210";
    var model = base;
    model.model_dir = "/models/other";
    for ([_]Parts{ lever, build, model }) |other| {
        const oid = keyId(other);
        try testing.expect(!std.mem.eql(u8, &oid, &id));
        var o_buf: [600]u8 = undefined;
        try testing.expect(read(testing.io, path(&o_buf, root, &oid).?) == null);
    }
}

test "ladder store: a malformed or untimed file is a miss" {
    var buf: [2048]u8 = undefined;
    var l = testLadder();
    const text = try encode(&buf, &l);
    try testing.expect(decode(text[0 .. text.len - 1]) != null);
    try testing.expect(decode("ladder0\nplain 1\n") == null);
    l.drafts[2] = std.math.inf(f32);
    try testing.expect(decode(try encode(&buf, &l)) == null);
}
