//! A prompt's Engram rows read once per table and past the page cache.
//!
//! The pack's tables (oMLX's lossless layout, ~100 GB a layer) keep a row's e4m3 codes (256 B) and its e8m0 scales
//! (8 B) in two regions of one file. Stock, a prompt grid's rows are read span by span (256 tokens: 6,144 ids a table),
//! each span's distinct rows by two preads a row through the page cache: a 16K-token prompt's ~587K row reads cost ~522K
//! 16 KiB page-ins (~8 GiB of file cache, ~1.8 s for the first table, layer 1 waiting ~1.1 s with the pack's files
//! evicted: en1, w-pio-20261101-1419). The page cache absorbs the rows that repeat across spans (~47%) and the scale
//! pages shared by neighbouring rows; read past it one pread at a time, the same reads cost a 4 KiB device IO each
//! (en1's direct arm: 1.17M IOs, 8% slower).
//! Beside a streamed bank (`Table.readDirect`; a resident pack keeps the page cache, which serves a warm prompt's rows
//! faster), a prompt grid's table is read whole: every span's ids sorted and made distinct, the distinct rows read past the
//! cache (F_NOCACHE: the file system blocks that hold them, fstatfs `f_bsize`, APFS 4,096 B, straight into an aligned
//! buffer) on up to 64 threads, each a contiguous run of the sorted rows that keeps its last scale block for the rows
//! after it, then each span's rows copied out of them; a decode token's rows read past the cache as before, a pread a
//! half-row. A repacked table (`engram/engram-manifest.json`: a row's codes then its scales in one 264 B record) is
//! read a record a row (`readRecord`): one pread through the blocks that hold it (two blocks for the 6.4% of records
//! that straddle one), in a whole-table read and a decode token's alike. Exact: the same bytes land in the same rows;
//! only how they are read changes.

const std = @import("std");
const builtin = @import("builtin");

/// The most bytes one read through the blocks may span: a longer row reads with one pread as before.
pub const max_span: usize = 16384;

/// The tables' files read through their blocks (`register`): fd and block.
const max_files = 8;
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    files: [max_files]struct { fd: c_int, block: u64 } = @splat(.{ .fd = -1, .block = 0 }),
    n_files: usize = 0,
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_engram_direct", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

/// At a table's open: the file set F_NOCACHE and its block kept, when the block is a power of two
/// in [512, 4096] (else the file stays on the page cache).
pub fn register(fd: c_int) void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    if (ps().n_files == max_files) return;
    var st: Statfs = undefined;
    if (fstatfs(fd, &st) != 0) return;
    const b: u64 = st.f_bsize;
    if (b < 512 or b > 4096 or b & (b - 1) != 0) return;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) == -1) return;
    ps().files[ps().n_files] = .{ .fd = fd, .block = b };
    ps().n_files += 1;
}
/// Darwin's `struct statfs` (64-bit inodes, arm64's only layout): only `f_bsize` is read.
const Statfs = extern struct { f_bsize: u32, f_iosize: i32, rest: [2160]u8 };
extern "c" fn fstatfs(fd: c_int, buf: *Statfs) c_int;
comptime {
    if (@sizeOf(Statfs) != 2168) @compileError("engram_direct: struct statfs is 2,168 bytes on Darwin");
}

/// At a table's close: its file let go of (a later file may take the same descriptor).
pub fn unregister(fd: c_int) void {
    var i: usize = 0;
    while (i < ps().n_files) {
        if (ps().files[i].fd == fd) {
            ps().files[i] = ps().files[ps().n_files - 1];
            ps().n_files -= 1;
        } else i += 1;
    }
}

fn blockOf(fd: c_int) u64 {
    for (ps().files[0..ps().n_files]) |f| if (f.fd == fd) return f.block;
    return 0;
}

/// `dst.len` bytes at `off` of `fd`: through the blocks that cover them when `fd` is registered (one F_NOCACHE pread of
/// whole blocks into an aligned buffer, the bytes copied out), else one pread. False on a short read.
pub fn readRow(fd: c_int, dst: []u8, off: u64) bool {
    // Windows opens no Engram table (`engram.Table.open`).
    if (comptime builtin.os.tag == .windows) return false;
    const block = blockOf(fd);
    const lo = if (block > 0) off / block * block else off;
    const hi = if (block > 0) std.mem.alignForward(u64, off + dst.len, block) else off + dst.len;
    if (block == 0 or hi - lo > max_span) {
        return std.c.pread(fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }
    var buf: [max_span]u8 align(16384) = undefined;
    const n: usize = @intCast(hi - lo);
    const got = std.c.pread(fd, &buf, n, @intCast(lo));
    const skip: usize = @intCast(off - lo);
    if (got < 0 or @as(usize, @intCast(got)) < skip + dst.len) return false;
    @memcpy(dst, buf[skip..][0..dst.len]);
    return true;
}

/// The longest record `readRecord` reads with one call (a head_dim of 4096's codes and scales).
pub const max_record: usize = 4096 + 128;

/// A record's two parts, adjacent at `off` (`w.len` bytes, then `s.len`): one `readRow` of both, split into `w` and
/// `s` (a record past `max_record`: two). False on a short read.
pub fn readRecord(fd: c_int, w: []u8, s: []u8, off: u64) bool {
    const n = w.len + s.len;
    if (n > max_record) return readRow(fd, w, off) and readRow(fd, s, off + w.len);
    var rec: [max_record]u8 = undefined;
    if (!readRow(fd, rec[0..n], off)) return false;
    @memcpy(w, rec[0..w.len]);
    @memcpy(s, rec[w.len..n]);
    return true;
}

/// One thread's last read through a file's blocks: the next row whose bytes it covers is copied from it, no read.
pub const Block = struct {
    lo: u64 = std.math.maxInt(u64),
    len: usize = 0,
    buf: [max_span]u8 align(16384) = undefined,
    /// `dst.len` bytes at `off` of `fd`: from the kept block when it covers them, else read through the blocks that
    /// cover them (kept for the next row); one pread when `fd` is not registered or the span is past `max_span`.
    pub fn read(b: *Block, fd: c_int, dst: []u8, off: u64) bool {
        if (comptime builtin.os.tag == .windows) return false;
        const block = blockOf(fd);
        if (block == 0) return readRow(fd, dst, off);
        if (b.lo != std.math.maxInt(u64) and off >= b.lo and off + dst.len <= b.lo + b.len) {
            const at: usize = @intCast(off - b.lo);
            @memcpy(dst, b.buf[at..][0..dst.len]);
            return true;
        }
        const lo = off / block * block;
        const hi = std.mem.alignForward(u64, off + dst.len, block);
        if (hi - lo > max_span) return readRow(fd, dst, off);
        const n: usize = @intCast(hi - lo);
        b.lo = std.math.maxInt(u64);
        const got = std.c.pread(fd, &b.buf, n, @intCast(lo));
        const skip: usize = @intCast(off - lo);
        if (got < 0 or @as(usize, @intCast(got)) < skip + dst.len) return false;
        b.lo = lo;
        b.len = @intCast(got);
        @memcpy(dst, b.buf[skip..][0..dst.len]);
        return true;
    }
};

test "engram_direct readRecord: a record's codes then scales in one read, split; past the end a short read" {
    const a = std.testing.allocator;
    var data: [3 * 264]u8 = undefined;
    for (&data, 0..) |*x, i| x.* = @truncate(i *% 7 +% 3);
    const path = try std.fmt.allocPrintSentinel(a, "/tmp/dsv41-engram-records-{d}.bin", .{std.c.getpid()}, 0);
    defer a.free(path);
    const out = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(out >= 0);
    const wrote = std.c.write(out, &data, data.len);
    _ = std.c.close(out);
    defer _ = std.c.unlink(path);
    try std.testing.expectEqual(@as(isize, data.len), wrote);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    var cw: [256]u8 = undefined;
    var cs: [8]u8 = undefined;
    try std.testing.expect(readRecord(fd, &cw, &cs, 264));
    try std.testing.expectEqualSlices(u8, data[264..520], &cw);
    try std.testing.expectEqualSlices(u8, data[520..528], &cs);
    try std.testing.expect(!readRecord(fd, &cw, &cs, 2 * 264 + 100));
}
