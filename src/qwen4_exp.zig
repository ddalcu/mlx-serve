//! Qwen3.8-Flash-Next (`model_type` qwen4_exp) host-side pieces: the hashed
//! n-gram embedding (PLE) id math and the mmapped quantized n-gram table.
//!
//! The 51B-parameter table (320M rows x 160) is never uploaded to the GPU:
//! a token touches 16 rows, so the rows are dequantized from the mmap on the
//! host and only the [T, 2560] result is sent. Memory cost = page cache.
//! Format: `ngram_table.bin` is a safetensors-format file holding one merged
//! 4-bit affine table (`weight` U32 [R, dim*bits/32], `scales`/`biases` BF16
//! [R, dim/gs]) written by `tests/convert_qwen38_flash_next.py`.

const std = @import("std");

const MASK64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SPLITMIX_GAMMA: u64 = 0x9E3779B97F4A7C15;
const SPLITMIX_M1: u64 = 0xBF58476D1CE4E5B9;
const SPLITMIX_M2: u64 = 0x94D049BB133111EB;
const PRIME_1: u64 = 10007;

fn splitmix64(v0: u64) u64 {
    var v = v0 +% SPLITMIX_GAMMA;
    v = (v ^ (v >> 30)) *% SPLITMIX_M1;
    v = (v ^ (v >> 27)) *% SPLITMIX_M2;
    return v ^ (v >> 31);
}

fn isPrime(v: u64) bool {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    var d: u64 = 3;
    while (d * d <= v) : (d += 2) {
        if (v % d == 0) return false;
    }
    return true;
}

fn nthPrimeAfter(start: u64, count: u32) u64 {
    var p = start;
    for (0..count) |_| {
        p += 1;
        while (!isPrime(p)) p += 1;
    }
    return p;
}

pub const MAX_HEADS = 32;

/// Everything `Qwen4ExpTextNGramEmbedding.__init__` derives from the config.
pub const NgramHash = struct {
    ngram_size: u32,
    heads_per_ngram: u32,
    n_heads: u32,
    eos: u32,
    multipliers: [8]i64,
    vocab: [MAX_HEADS]i64,
    offsets: [MAX_HEADS]i64,
    total_rows: u64,

    pub fn init(unigram_vocab: u32, ngram_size: u32, heads_per_ngram: u32, vocab_base: u64, divisor: u64, seed: u64, ple_layer_index: u32, eos: u32) NgramHash {
        var h: NgramHash = .{
            .ngram_size = ngram_size,
            .heads_per_ngram = heads_per_ngram,
            .n_heads = (ngram_size - 1) * heads_per_ngram,
            .eos = eos,
            .multipliers = @splat(0),
            .vocab = @splat(0),
            .offsets = @splat(0),
            .total_rows = 0,
        };
        std.debug.assert(h.n_heads <= MAX_HEADS and ngram_size <= 8);
        const max_long: u64 = (1 << 63) - 1;
        const half_bound: u64 = @max(1, (max_long / @max(unigram_vocab, 1)) / 2);
        const base_seed: u64 = seed +% PRIME_1 *% ple_layer_index;
        for (0..ngram_size) |i| {
            const v = base_seed +% SPLITMIX_GAMMA *% (@as(u64, i) + 1);
            h.multipliers[i] = @intCast(2 * (splitmix64(v) % half_bound) + 1);
        }
        var total: u64 = 0;
        for (0..h.n_heads) |i| {
            const global = ple_layer_index * h.n_heads + @as(u32, @intCast(i));
            const size = nthPrimeAfter(vocab_base - 1, global + 1);
            h.vocab[i] = @intCast(size);
            h.offsets[i] = @intCast(total);
            total += size;
        }
        h.total_rows = (total + divisor - 1) / divisor * divisor;
        return h;
    }

    /// Row ids for `ids`, given the (ngram_size-1) tokens that precede them
    /// (`eos` for a fresh sequence). `out` is `[ids.len][n_heads]` row-major.
    /// Mirrors `_shift_right_ignore_eos` + the mixed-id hash: a shifted token
    /// is `eos` when the shift crosses the most recent eos before it.
    pub fn rowIds(self: *const NgramHash, prev: []const u32, ids: []const u32, out: []i64) void {
        const ctx: usize = self.ngram_size - 1;
        std.debug.assert(prev.len == ctx and out.len == ids.len * self.n_heads);
        var last_eos: i64 = -1;
        var t: usize = 0;
        while (t < ctx + ids.len) : (t += 1) {
            const tok = tokAt(prev, ids, t);
            if (t >= ctx) {
                const seg_pos: i64 = @as(i64, @intCast(t)) - (last_eos + 1);
                var mixed: i64 = @as(i64, tok) *% self.multipliers[0];
                const row = out[(t - ctx) * self.n_heads ..][0..self.n_heads];
                var n: usize = 2;
                var pos: usize = 1;
                while (n <= self.ngram_size) : (n += 1) {
                    while (pos < n) : (pos += 1) {
                        const shifted: u32 = if (seg_pos >= @as(i64, @intCast(pos)) and t >= pos) tokAt(prev, ids, t - pos) else self.eos;
                        mixed ^= @as(i64, shifted) *% self.multipliers[pos];
                    }
                    const h0 = (n - 2) * self.heads_per_ngram;
                    for (h0..h0 + self.heads_per_ngram) |h| {
                        row[h] = @mod(mixed, self.vocab[h]) + self.offsets[h];
                    }
                }
            }
            if (tok == self.eos) last_eos = @intCast(t);
        }
    }

    fn tokAt(prev: []const u32, ids: []const u32, i: usize) u32 {
        return if (i < prev.len) prev[i] else ids[i - prev.len];
    }
};

/// The merged quantized table, mmapped read-only.
pub const NgramTable = struct {
    map: []align(std.heap.page_size_min) const u8,
    rows: u64,
    dim: u32,
    bits: u32,
    group_size: u32,
    w_off: usize,
    s_off: usize,
    b_off: usize,
    wcols: u32,
    scols: u32,
    /// Kept open for the pool's `pread` gather (page faults on one mapping
    /// serialize on the VM map lock; preads run in parallel).
    fd: std.c.fd_t = -1,
    pool: ?*PrefetchPool = null,

    pub fn open(path: []const u8) !NgramTable {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        if (path.len >= pbuf.len) return error.NameTooLong;
        @memcpy(pbuf[0..path.len], path);
        pbuf[path.len] = 0;
        const fd = std.c.open(pbuf[0..path.len :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.FileNotFound;
        errdefer _ = std.c.close(fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
        const size: usize = @intCast(st.size);
        const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
        errdefer std.posix.munmap(map);
        if (size < 8) return error.NgramTableTruncated;
        const hlen: usize = @intCast(std.mem.readInt(u64, map[0..8], .little));
        if (8 + hlen > size) return error.NgramTableTruncated;
        var t = try parse(map, map[8 .. 8 + hlen], 8 + hlen);
        t.fd = fd;
        if (plePrefetchEnabled()) t.pool = PrefetchPool.create() catch null;
        return t;
    }

    fn parse(map: []align(std.heap.page_size_min) const u8, header: []const u8, data_off: usize) !NgramTable {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, header, .{});
        const obj = parsed.object;
        const meta = obj.get("__metadata__") orelse return error.NgramTableHeader;
        const bits: u32 = try std.fmt.parseInt(u32, meta.object.get("bits").?.string, 10);
        const gs: u32 = try std.fmt.parseInt(u32, meta.object.get("group_size").?.string, 10);
        const w = obj.get("weight") orelse return error.NgramTableHeader;
        const s = obj.get("scales") orelse return error.NgramTableHeader;
        const b = obj.get("biases") orelse return error.NgramTableHeader;
        const rows: u64 = @intCast(w.object.get("shape").?.array.items[0].integer);
        const wcols: u32 = @intCast(w.object.get("shape").?.array.items[1].integer);
        const scols: u32 = @intCast(s.object.get("shape").?.array.items[1].integer);
        const t: NgramTable = .{
            .map = map,
            .rows = rows,
            .dim = scols * gs,
            .bits = bits,
            .group_size = gs,
            .w_off = data_off + @as(usize, @intCast(w.object.get("data_offsets").?.array.items[0].integer)),
            .s_off = data_off + @as(usize, @intCast(s.object.get("data_offsets").?.array.items[0].integer)),
            .b_off = data_off + @as(usize, @intCast(b.object.get("data_offsets").?.array.items[0].integer)),
            .wcols = wcols,
            .scols = scols,
        };
        if (t.dim * bits != wcols * 32) return error.NgramTableHeader;
        const end = t.b_off + rows * scols * 2;
        if (end > map.len) return error.NgramTableTruncated;
        return t;
    }

    pub fn close(self: *NgramTable) void {
        if (self.pool) |p| p.destroy();
        self.pool = null;
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.fd = -1;
        std.posix.munmap(self.map);
    }

    /// Dequantize one row into `out[0..dim]` (MLX affine layout: element i
    /// sits at bits [(i % per_word) * bits ..] of word i / per_word).
    pub fn row(self: *const NgramTable, r: u64, out: []f32) void {
        std.debug.assert(r < self.rows and out.len >= self.dim);
        const words = self.map[self.w_off + r * self.wcols * 4 ..][0 .. self.wcols * 4];
        const scales = self.map[self.s_off + r * self.scols * 2 ..][0 .. self.scols * 2];
        const biases = self.map[self.b_off + r * self.scols * 2 ..][0 .. self.scols * 2];
        self.dequantRow(words, scales, biases, out);
    }

    fn dequantRow(self: *const NgramTable, words: []const u8, scales: []const u8, biases: []const u8, out: []f32) void {
        const per_word: u32 = 32 / self.bits;
        const mask: u32 = (@as(u32, 1) << @intCast(self.bits)) - 1;
        var i: u32 = 0;
        while (i < self.dim) : (i += 1) {
            const word = std.mem.readInt(u32, words[(i / per_word) * 4 ..][0..4], .little);
            const q: u32 = (word >> @intCast((i % per_word) * self.bits)) & mask;
            const g = i / self.group_size;
            const sc = bf16ToF32(std.mem.readInt(u16, scales[g * 2 ..][0..2], .little));
            const bi = bf16ToF32(std.mem.readInt(u16, biases[g * 2 ..][0..2], .little));
            out[i] = @as(f32, @floatFromInt(q)) * sc + bi;
        }
    }

    /// Gather + concatenate the `n_heads` rows of each token: `out` is
    /// `[ids.len / n_heads][n_heads * dim]` row-major.
    pub fn gather(self: *const NgramTable, row_ids: []const i64, out: []f32) void {
        const need: usize = self.wcols * 4 + self.scols * 4;
        // Decode/verify widths only: a 4096-row prefill chunk is 65k rows,
        // mostly page-cache hits, and 1024 wake rounds cost more than they save.
        if (self.pool) |p| if (self.fd >= 0 and need <= PrefetchPool.ROW_BUF and row_ids.len <= PrefetchPool.MAX_ROWS) {
            const wl: usize = self.wcols * 4;
            const sl: usize = self.scols * 2;
            var start: usize = 0;
            while (start < row_ids.len) : (start += PrefetchPool.MAX_ROWS) {
                const end = @min(start + PrefetchPool.MAX_ROWS, row_ids.len);
                if (!p.run(self, row_ids[start..end])) break;
                for (start..end) |i| {
                    const b = &p.bufs[i - start];
                    self.dequantRow(b[0..wl], b[wl .. wl + sl], b[wl + sl .. wl + 2 * sl], out[i * self.dim ..][0..self.dim]);
                }
            }
            if (start >= row_ids.len) return;
        };
        for (row_ids, 0..) |r, i| self.row(@intCast(r), out[i * self.dim ..][0..self.dim]);
    }

    /// One (row, region) pread into the pool's row buffer. False on a short read.
    fn preadSite(self: *const NgramTable, r: u64, region: usize, buf: []u8) bool {
        const wl: usize = self.wcols * 4;
        const sl: usize = self.scols * 2;
        const off: usize, const dst: []u8 = switch (region) {
            0 => .{ self.w_off + r * wl, buf[0..wl] },
            1 => .{ self.s_off + r * sl, buf[wl .. wl + sl] },
            else => .{ self.b_off + r * sl, buf[wl + sl .. wl + 2 * sl] },
        };
        return std.c.pread(self.fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }

};

/// Persistent gather workers. Every row's three regions are one SSD read on
/// the cold 32 GB table (~100 us), 48 per token: serial mmap faults were ~5 ms
/// of every decode step, 16 fault threads ~0.7 ms, and more threads got SLOWER
/// (faults on one mapping serialize on the VM map lock), so workers `pread`
/// instead and dequantize their rows in place. Workers wake on a generation
/// bump and count themselves down; the caller spins (the job is ~100 us).
const PrefetchPool = struct {
    const N = 48;
    const MAX_ROWS = 64;
    const ROW_BUF = 512;
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    gen: u64 = 0,
    quit: bool = false,
    table: ?*const NgramTable = null,
    rows: []const i64 = &.{},
    bufs: [MAX_ROWS][ROW_BUF]u8 = undefined,
    pending: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(u32) = .init(0),
    threads: [N]std.Thread = undefined,

    fn create() !*PrefetchPool {
        const a = std.heap.page_allocator;
        const p = try a.create(PrefetchPool);
        p.* = .{};
        var started: usize = 0;
        errdefer {
            p.shutdown(started);
            a.destroy(p);
        }
        for (0..N) |i| {
            p.threads[i] = try std.Thread.spawn(.{ .stack_size = 64 * 1024 }, worker, .{ p, i });
            started += 1;
        }
        return p;
    }

    fn destroy(self: *PrefetchPool) void {
        self.shutdown(N);
        std.heap.page_allocator.destroy(self);
    }

    fn shutdown(self: *PrefetchPool, started: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        for (self.threads[0..started]) |t| t.join();
    }

    /// Fan the `3 * rows.len` preads over the workers; rows land in `bufs`.
    fn run(self: *PrefetchPool, table: *const NgramTable, rows: []const i64) bool {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.table = table;
        self.rows = rows;
        self.failed.store(0, .release);
        self.pending.store(N, .release);
        self.gen += 1;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        while (self.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
        return self.failed.load(.acquire) == 0;
    }

    fn worker(self: *PrefetchPool, idx: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        var seen: u64 = 0;
        while (true) {
            self.mu.lockUncancelable(io);
            while (self.gen == seen and !self.quit) self.cv.wait(io, &self.mu) catch {};
            if (self.quit) {
                self.mu.unlock(io);
                return;
            }
            seen = self.gen;
            const table = self.table.?;
            const rows = self.rows;
            self.mu.unlock(io);
            var i = idx;
            while (i < rows.len * 3) : (i += N) {
                if (!table.preadSite(@intCast(rows[i / 3]), i % 3, &self.bufs[i / 3])) _ = self.failed.fetchAdd(1, .acq_rel);
            }
            _ = self.pending.fetchSub(1, .acq_rel);
        }
    }
};

fn plePrefetchEnabled() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v) |v| return v;
    const raw = std.c.getenv("QWEN4_PLE_PREFETCH");
    const v = raw == null or raw.?[0] != '0';
    S.v = v;
    return v;
}

pub fn bf16ToF32(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

// ── tests ──

const testing = std.testing;

test "ngram hash reproduces the reference multipliers, primes and offsets" {
    const h = NgramHash.init(248320, 3, 8, 20_000_000, 128, 1234, 0, 248044);
    try testing.expectEqual(@as(i64, 23703573157769), h.multipliers[0]);
    try testing.expectEqual(@as(i64, 20109073645365), h.multipliers[1]);
    try testing.expectEqual(@as(i64, 8052911324071), h.multipliers[2]);
    try testing.expectEqual(@as(i64, 20000003), h.vocab[0]);
    try testing.expectEqual(@as(i64, 20000171), h.vocab[15]);
    try testing.expectEqual(@as(i64, 300001275), h.offsets[15]);
    try testing.expectEqual(@as(u64, 320001536), h.total_rows);
}

test "ngram row ids match the reference on an eos-split history" {
    // Reference (modeling_qwen4_exp.py, run in python): history
    // [eos, eos | 5, 7, eos, 9, 11]; shifts reset across the eos.
    const h = NgramHash.init(248320, 3, 8, 20_000_000, 128, 1234, 0, 248044);
    const prev = [_]u32{ 248044, 248044 };
    const ids = [_]u32{ 5, 7, 248044, 9, 11 };
    var out: [5 * 16]i64 = undefined;
    h.rowIds(&prev, &ids, &out);
    const want = [5][16]i64{
        .{ 15389869, 39778609, 55713969, 62213332, 88817728, 118483999, 133731511, 155458159, 179763390, 197956758, 205378969, 220499474, 242466248, 265658744, 293662119, 315720898 },
        .{ 12441580, 26378836, 53347667, 75104214, 99467174, 114254887, 126436461, 156012011, 169119442, 187827161, 214803956, 239809754, 242938905, 266427765, 294337448, 314484167 },
        .{ 10204458, 27984170, 41283776, 68842151, 85621153, 118821647, 129504214, 158727320, 176298516, 181690702, 206665473, 238343128, 252151767, 267018740, 285543023, 319927855 },
        .{ 18043673, 37626835, 51159316, 78294604, 94015356, 106720349, 136526052, 144330141, 176817901, 186368539, 203707490, 230017629, 247662678, 266533413, 293096193, 307951937 },
        .{ 10041117, 28960672, 48420531, 71664411, 83016360, 106800418, 122476460, 150044571, 163654473, 184259024, 206781966, 224776026, 248853488, 273290488, 294849492, 303242927 },
    };
    for (want, 0..) |row, t| {
        for (row, 0..) |v, i| try testing.expectEqual(v, out[t * 16 + i]);
    }
}

test "ngram table row dequant follows the MLX affine nibble layout" {
    // Two rows, dim 32, 4-bit, one group: word k packs elements 8k..8k+7,
    // element i at nibble i % 8.
    var buf: [8 + 512 + 2 * 16 + 2 * 2 + 2 * 2]u8 = undefined;
    const header = "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[2,4],\"data_offsets\":[0,32]},\"scales\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[32,36]},\"biases\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[36,40]}}";
    var hdr: [512]u8 = @splat(' ');
    @memcpy(hdr[0..header.len], header);
    std.mem.writeInt(u64, buf[0..8], 512, .little);
    @memcpy(buf[8..520], &hdr);
    const data = buf[520..];
    // row 0: elements 0..31 = i % 16; row 1: all 3
    var i: u32 = 0;
    while (i < 4) : (i += 1) {
        var w: u32 = 0;
        var j: u32 = 0;
        while (j < 8) : (j += 1) w |= ((i * 8 + j) % 16) << @intCast(j * 4);
        std.mem.writeInt(u32, data[i * 4 ..][0..4], w, .little);
        std.mem.writeInt(u32, data[16 + i * 4 ..][0..4], 0x33333333, .little);
    }
    // scales: row0 = 0.5 (bf16 0x3F00), row1 = 2.0 (0x4000); biases: row0 = 1.0 (0x3F80), row1 = -1 (0xBF80)
    std.mem.writeInt(u16, data[32..34], 0x3F00, .little);
    std.mem.writeInt(u16, data[34..36], 0x4000, .little);
    std.mem.writeInt(u16, data[36..38], 0x3F80, .little);
    std.mem.writeInt(u16, data[38..40], 0xBF80, .little);
    const aligned = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), buf.len);
    defer std.heap.page_allocator.free(aligned);
    @memcpy(aligned, &buf);
    const t = try NgramTable.parse(aligned, aligned[8..520], 520);
    try testing.expectEqual(@as(u32, 32), t.dim);
    var out: [32]f32 = undefined;
    t.row(0, &out);
    try testing.expectEqual(@as(f32, 1.0), out[0]);
    try testing.expectEqual(@as(f32, 0.5 * 7 + 1.0), out[7]);
    try testing.expectEqual(@as(f32, 0.5 * 15 + 1.0), out[31]);
    t.row(1, &out);
    try testing.expectEqual(@as(f32, 5.0), out[13]);
}

/// Module-owned state for one loaded qwen4_exp model: the n-gram hash and
/// the mmapped table. Non-null on `Transformer.qwen4` ⇒ the arch is served
/// serially with speculation off (`ownsModuleDecodeState`), which is what
/// the per-request PLE/indexer state in `SSMCacheEntry.aux_state` needs
/// until the snapshot machinery carries it.
pub const Qwen4State = struct {
    hash: NgramHash,
    table: NgramTable,

    pub fn deinit(self: *Qwen4State) void {
        self.table.close();
    }
};
