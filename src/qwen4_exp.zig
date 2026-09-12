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
const log = @import("log.zig");

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

/// Config-driven bounds; `model.validateQwen4Config` refuses a checkpoint past them at load.
pub const MAX_HEADS = 32;
pub const MAX_NGRAM_SIZE = 8;

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

    /// `ple_layer_index` is the PLE's ordinal among the config's injection points (0 for the
    /// one we support). Fallible: every bound writes a fixed array.
    pub fn init(unigram_vocab: u32, ngram_size: u32, heads_per_ngram: u32, vocab_base: u64, divisor: u64, seed: u64, ple_layer_index: u32, eos: u32) !NgramHash {
        if (ngram_size < 2 or ngram_size > MAX_NGRAM_SIZE) return error.InvalidQwen4NgramSize;
        if (heads_per_ngram == 0 or (ngram_size - 1) * heads_per_ngram > MAX_HEADS) {
            return error.InvalidQwen4NgramHeads;
        }
        if (divisor == 0 or vocab_base < 2) return error.InvalidQwen4NgramVocab;
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
var warm_env_cached: ?bool = null;
pub var warm_override: ?bool = null;

fn warmEnabled() bool {
    if (warm_override) |v| return v;
    if (warm_env_cached) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_SERVE_NGRAM_WARM") orelse break :blk true;
        break :blk raw[0] != '0';
    };
    warm_env_cached = v;
    return v;
}

/// What the background page-cache warm has read so far, and the table's total size. Published
/// by the warm thread, read lock-free by metrics and `/props`; zero when nothing is warming.
pub var live_warm_bytes = std.atomic.Value(u64).init(0);
pub var live_warm_total = std.atomic.Value(u64).init(0);

/// A progress line at each 8 GB step or after 10 s of silence, never twice per step. Pure.
pub const WARM_LOG_BYTES: u64 = 8 << 30;
pub const WARM_LOG_NS: u64 = 10_000_000_000;

pub const WarmProgress = struct {
    next_bytes: u64 = WARM_LOG_BYTES,
    last_ns: u64 = 0,

    pub fn should(self: *WarmProgress, bytes: u64, elapsed_ns: u64) bool {
        const by_bytes = bytes >= self.next_bytes;
        const by_time = elapsed_ns -| self.last_ns >= WARM_LOG_NS;
        if (!by_bytes and !by_time) return false;
        while (self.next_bytes <= bytes) self.next_bytes += WARM_LOG_BYTES;
        self.last_ns = elapsed_ns;
        return true;
    }
};

fn asGb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1073741824.0;
}

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
    /// Boot-time page-cache warm: the weights load evicts this file, and the
    /// first long prompt then faults 48 rows/token from SSD (38k: 174 s vs
    /// 55 s warm). preads through the kept fd, never the mapping.
    warm_thread: ?std.Thread = null,
    warm_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    warm_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

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
        if (hlen > size - 8) return error.NgramTableTruncated;
        var t = try parse(map, map[8 .. 8 + hlen], 8 + hlen);
        t.fd = fd;
        if (plePrefetchEnabled()) t.pool = PrefetchPool.create() catch null;
        return t;
    }

    /// Widths `mx.quantize` packs and `dequantRow` unpacks.
    fn bitsSupported(bits: u32) bool {
        return switch (bits) {
            2, 3, 4, 5, 6, 8 => true,
            else => false,
        };
    }

    const HeaderRegion = struct {
        rows: u64,
        cols: u64,
        start: u64, // relative to the data section, as the header spells it
        end: u64,

        fn overlaps(a: HeaderRegion, b: HeaderRegion) bool {
            return a.start < b.end and b.start < a.end;
        }
    };

    /// One header entry, every access checked and the region proven to hold exactly
    /// `rows x cols x elem` bytes inside the mapping.
    fn headerRegion(
        obj: std.json.ObjectMap,
        key: []const u8,
        dtype: []const u8,
        elem: u64,
        map_len: usize,
        data_off: usize,
    ) !HeaderRegion {
        const v = obj.get(key) orelse return error.NgramTableHeader;
        if (v != .object) return error.NgramTableHeader;
        const o = v.object;
        const dt = o.get("dtype") orelse return error.NgramTableHeader;
        if (dt != .string or !std.mem.eql(u8, dt.string, dtype)) return error.NgramTableHeader;
        const shape = o.get("shape") orelse return error.NgramTableHeader;
        if (shape != .array or shape.array.items.len != 2) return error.NgramTableHeader;
        if (shape.array.items[0] != .integer or shape.array.items[1] != .integer) return error.NgramTableHeader;
        const dofs = o.get("data_offsets") orelse return error.NgramTableHeader;
        if (dofs != .array or dofs.array.items.len != 2) return error.NgramTableHeader;
        if (dofs.array.items[0] != .integer or dofs.array.items[1] != .integer) return error.NgramTableHeader;

        const rows_i = shape.array.items[0].integer;
        const cols_i = shape.array.items[1].integer;
        const start_i = dofs.array.items[0].integer;
        const end_i = dofs.array.items[1].integer;
        if (rows_i <= 0 or cols_i <= 0 or start_i < 0 or end_i < start_i) return error.NgramTableRegion;
        const r: HeaderRegion = .{
            .rows = @intCast(rows_i),
            .cols = @intCast(cols_i),
            .start = @intCast(start_i),
            .end = @intCast(end_i),
        };
        if (r.cols > std.math.maxInt(u32) or r.rows > std.math.maxInt(u32)) return error.NgramTableRegion;
        const need = std.math.mul(u64, r.rows, r.cols * elem) catch return error.NgramTableRegion;
        if (r.end - r.start != need) return error.NgramTableRegion;
        const abs_end = std.math.add(u64, data_off, r.end) catch return error.NgramTableTruncated;
        if (abs_end > map_len) return error.NgramTableTruncated;
        return r;
    }

    /// Absent `format` stamp is accepted once, loudly; a different format is a refusal.
    var stamp_warned: bool = false;

    fn parse(map: []align(std.heap.page_size_min) const u8, header: []const u8, data_off: usize) !NgramTable {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, header, .{}) catch return error.NgramTableHeader;
        if (parsed != .object) return error.NgramTableHeader;
        const obj = parsed.object;
        const meta_v = obj.get("__metadata__") orelse return error.NgramTableHeader;
        if (meta_v != .object) return error.NgramTableHeader;
        const meta = meta_v.object;
        if (meta.get("format")) |f| {
            if (f != .string or !std.mem.eql(u8, f.string, "mlx-serve-ngram")) return error.NgramTableHeader;
        } else if (!stamp_warned) {
            stamp_warned = true;
            log.info("[qwen4] ngram table has no \"format\" stamp (written before the converter added it); accepting\n", .{});
        }
        const bits_v = meta.get("bits") orelse return error.NgramTableHeader;
        const gs_v = meta.get("group_size") orelse return error.NgramTableHeader;
        if (bits_v != .string or gs_v != .string) return error.NgramTableHeader;
        const bits: u32 = std.fmt.parseInt(u32, bits_v.string, 10) catch return error.NgramTableHeader;
        const gs: u32 = std.fmt.parseInt(u32, gs_v.string, 10) catch return error.NgramTableHeader;
        if (!bitsSupported(bits)) return error.NgramTableBits;
        if (gs == 0 or gs > 1024) return error.NgramTableBits;

        const w = try headerRegion(obj, "weight", "U32", 4, map.len, data_off);
        const sc = try headerRegion(obj, "scales", "BF16", 2, map.len, data_off);
        const bi = try headerRegion(obj, "biases", "BF16", 2, map.len, data_off);
        // The three regions describe the same rows and may not overlap.
        if (sc.rows != w.rows or bi.rows != w.rows or sc.cols != bi.cols) return error.NgramTableRegion;
        if (w.overlaps(sc) or w.overlaps(bi) or sc.overlaps(bi)) return error.NgramTableRegion;

        const dim: u64 = sc.cols * gs;
        if (dim > std.math.maxInt(u32)) return error.NgramTableRegion;
        if (dim * bits != w.cols * 32) return error.NgramTableHeader;

        return .{
            .map = map,
            .rows = w.rows,
            .dim = @intCast(dim),
            .bits = bits,
            .group_size = gs,
            .w_off = data_off + @as(usize, @intCast(w.start)),
            .s_off = data_off + @as(usize, @intCast(sc.start)),
            .b_off = data_off + @as(usize, @intCast(bi.start)),
            .wcols = @intCast(w.cols),
            .scols = @intCast(sc.cols),
        };
    }

    pub fn close(self: *NgramTable) void {
        if (self.warm_thread) |th| {
            self.warm_stop.store(true, .release);
            th.join();
            self.warm_thread = null;
        }
        live_warm_bytes.store(0, .release);
        live_warm_total.store(0, .release);
        if (self.pool) |p| p.destroy();
        self.pool = null;
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.fd = -1;
        std.posix.munmap(self.map);
    }

    const WARM_CHUNK: usize = 8 << 20;

    /// Read the whole table through the fd once, in the background, so the
    /// first prompt's PLE gathers hit a warm page cache. Call only once the
    /// table sits at its final address (the thread holds `self`). Off via
    /// MLX_SERVE_NGRAM_WARM=0.
    pub fn startWarm(self: *NgramTable) void {
        if (self.fd < 0 or self.warm_thread != null) return;
        // The off arm says so: a cold first request faults rows off the SSD (38k prompt: 174 s vs 55 s).
        if (!warmEnabled()) {
            log.info("[qwen4] ngram table warm: disabled (MLX_SERVE_NGRAM_WARM=0) - the first long prompt faults the table in from SSD\n", .{});
            return;
        }
        self.warm_stop.store(false, .release);
        self.warm_bytes.store(0, .release);
        live_warm_bytes.store(0, .release);
        live_warm_total.store(self.map.len, .release);
        log.info("[qwen4] ngram table warm: started, {d:.1} GB in the background (page cache; MLX_SERVE_NGRAM_WARM=0 disables)\n", .{asGb(self.map.len)});
        self.warm_thread = std.Thread.spawn(.{}, warmMain, .{self}) catch null;
    }

    fn warmMain(self: *NgramTable) void {
        var scratch: [WARM_CHUNK]u8 align(16) = undefined;
        const wio = std.Io.Threaded.global_single_threaded.io();
        const t0 = std.Io.Timestamp.now(wio, .boot);
        var off: u64 = 0;
        const total: u64 = self.map.len;
        var prog: WarmProgress = .{};
        while (off < total) {
            if (self.warm_stop.load(.acquire)) return;
            const want: usize = @intCast(@min(total - off, WARM_CHUNK));
            const got = std.c.pread(self.fd, &scratch, want, @intCast(off));
            if (got <= 0) return;
            off += @intCast(got);
            self.warm_bytes.store(off, .release);
            live_warm_bytes.store(off, .release);
            // One clock read per 8 MB pread is free next to the read itself.
            const el: u64 = @intCast(t0.untilNow(wio, .boot).nanoseconds);
            if (prog.should(off, el)) log.info("[qwen4] ngram table warm: {d:.1}/{d:.1} GB after {d:.0} s\n", .{ asGb(off), asGb(total), @as(f64, @floatFromInt(el)) / 1e9 });
        }
        const secs: f64 = @as(f64, @floatFromInt(t0.untilNow(wio, .boot).nanoseconds)) / 1e9;
        log.info("[qwen4] ngram table warm: done, {d:.1} GB in {d:.1} s (page cache; MLX_SERVE_NGRAM_WARM=0 disables)\n", .{ asGb(total), secs });
    }

    /// Dequantize one row into `out[0..dim]` (mx.quantize packing: element i
    /// sits at bit offset i * bits of the little-endian u32 stream and may
    /// straddle a word boundary at 3/5/6 bits).
    pub fn row(self: *const NgramTable, r: u64, out: []f32) void {
        std.debug.assert(r < self.rows and out.len >= self.dim);
        const words = self.map[self.w_off + r * self.wcols * 4 ..][0 .. self.wcols * 4];
        const scales = self.map[self.s_off + r * self.scols * 2 ..][0 .. self.scols * 2];
        const biases = self.map[self.b_off + r * self.scols * 2 ..][0 .. self.scols * 2];
        self.dequantRow(words, scales, biases, out);
    }

    fn dequantRow(self: *const NgramTable, words: []const u8, scales: []const u8, biases: []const u8, out: []f32) void {
        const mask: u32 = (@as(u32, 1) << @intCast(self.bits)) - 1;
        var i: u32 = 0;
        while (i < self.dim) : (i += 1) {
            const off = i * self.bits;
            const w = off / 32;
            const shift = off % 32;
            var v: u64 = std.mem.readInt(u32, words[w * 4 ..][0..4], .little);
            if (shift + self.bits > 32) v |= @as(u64, std.mem.readInt(u32, words[w * 4 + 4 ..][0..4], .little)) << 32;
            const q: u32 = @truncate((v >> @intCast(shift)) & mask);
            const g = i / self.group_size;
            const sc = bf16ToF32(std.mem.readInt(u16, scales[g * 2 ..][0..2], .little));
            const bi = bf16ToF32(std.mem.readInt(u16, biases[g * 2 ..][0..2], .little));
            out[i] = @as(f32, @floatFromInt(q)) * sc + bi;
        }
    }

    /// Gather + concatenate the `n_heads` rows of each token: `out` is
    /// `[ids.len / n_heads][n_heads * dim]` row-major. `kv_len` is the context position this
    /// gather runs at and picks the wide arm; the output is byte-identical either way.
    pub fn gather(self: *const NgramTable, row_ids: []const i64, out: []f32, kv_len: u64) void {
        const need: usize = self.wcols * 4 + self.scols * 4;
        // Prefill-width gathers ride the pool only past `PREFILL_PREFETCH_MIN_KV`: a resident
        // table loses 2-7% to the wake rounds, an evicted one (weights pushed the 32 GB
        // mapping out) went 67.7 -> 267.9 ms per 1000 tokens on the serial walk.
        const wide = row_ids.len > PrefetchPool.MAX_ROWS;
        const wide_ok = !wide or plePrefillPrefetchEnabled(kv_len);
        // Announce the arm that actually runs, not the lever that permits it.
        const pooled = wide_ok and self.pool != null and self.fd >= 0 and need <= PrefetchPool.ROW_BUF;
        if (wide) notePrefillGatherArm(pooled, row_ids.len);
        if (self.pool) |p| if (self.fd >= 0 and need <= PrefetchPool.ROW_BUF and wide_ok) {
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
        const nthr = parThreads();
        if (nthr > 1 and row_ids.len >= 1024) {
            var pctx: ParCtx = .{ .table = self, .rows = row_ids, .out = out, .next = std.atomic.Value(usize).init(0) };
            var th: [64]std.Thread = undefined;
            const want = @min(nthr, 64);
            var started: usize = 0;
            while (started < want) : (started += 1) {
                th[started] = std.Thread.spawn(.{ .stack_size = 64 * 1024 }, parWorker, .{&pctx}) catch break;
            }
            if (started > 0) {
                parWorker(&pctx);
                for (th[0..started]) |t| t.join();
                return;
            }
        }
        for (row_ids, 0..) |r, i| self.row(@intCast(r), out[i * self.dim ..][0..self.dim]);
    }

    /// EXPERIMENT (QWEN4_PLE_PAR, default 1 = today's serial path). A prefill
    /// chunk gathers 65k rows x 3 sites; the serial loop takes every fault by
    /// itself. Spreading the same faults over threads overlaps them.
    fn parThreads() u32 {
        if (ple_par_override) |v| return v;
        const S = struct {
            var v: ?u32 = null;
        };
        if (S.v) |v| return v;
        var v: u32 = 1;
        if (std.c.getenv("QWEN4_PLE_PAR")) |raw| v = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 1;
        S.v = v;
        return v;
    }

    const ParCtx = struct {
        table: *const NgramTable,
        rows: []const i64,
        out: []f32,
        next: std.atomic.Value(usize),
    };

    fn parWorker(ctx: *ParCtx) void {
        const dim = ctx.table.dim;
        const BLK: usize = 256;
        while (true) {
            const start = ctx.next.fetchAdd(BLK, .acq_rel);
            if (start >= ctx.rows.len) return;
            const end = @min(start + BLK, ctx.rows.len);
            for (start..end) |i| ctx.table.row(@intCast(ctx.rows[i]), ctx.out[i * dim ..][0..dim]);
        }
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
    /// Fan-out rounds issued; the engagement counter the prefill test reads.
    runs: std.atomic.Value(u64) = .init(0),
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
        _ = self.runs.fetchAdd(1, .monotonic);
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

/// One-shot engagement lines per arm; both arms say something.
/// Narrowest gather that is a genuine prefill chunk rather than a warmup forward.
pub const PREFILL_SAY_MIN_ROWS: usize = 1024;

/// [arm][bucket]: arm 0/1 = serial/pooled, bucket 0/1 = warmup/prefill width.
pub var ple_prefill_arm_said: [2][2]std.atomic.Value(bool) =
    .{ .{ .init(false), .init(false) }, .{ .init(false), .init(false) } };

fn notePrefillGatherArm(pooled: bool, rows: usize) void {
    const arm: usize = if (pooled) 1 else 0;
    const bucket: usize = if (rows >= PREFILL_SAY_MIN_ROWS) 1 else 0;
    if (ple_prefill_arm_said[arm][bucket].swap(true, .monotonic)) return;
    const width: []const u8 = if (bucket == 1) "prefill width" else "warmup width";
    if (pooled) {
        const batches = (rows + PrefetchPool.MAX_ROWS - 1) / PrefetchPool.MAX_ROWS;
        log.info("[qwen4] PLE prefill gather: POOLED ({s}: {d} rows, {d} batches of {d}; past kv {d}, QWEN4_PLE_PREFETCH_PREFILL=0 forces the serial walk)\n", .{ width, rows, batches, PrefetchPool.MAX_ROWS, plePrefillPrefetchMinKv() });
    } else {
        log.info("[qwen4] PLE prefill gather: SERIAL mmap walk ({s}: {d} rows; the pool engages past kv {d}, QWEN4_PLE_PREFETCH_PREFILL_MIN_KV overrides)\n", .{ width, rows, plePrefillPrefetchMinKv() });
    }
}

/// Test seams for the prefill gate below (both envs are read once per process).
/// Test seam for `parThreads` — mirrors `ple_prefill_prefetch_override` so the
/// threaded gather arm is reachable from `zig build test` without the env var.
pub var ple_par_override: ?u32 = null;

pub var ple_prefill_prefetch_override: ?bool = null;
pub var ple_prefill_min_kv_override: ?u64 = null;

/// The kv length past which a wide prefill gather takes the pool: the top of the measured
/// cost range (the pool loses at every rung to 256k on a resident table; the only win is the
/// evicted table on the 374k ladder). `QWEN4_PLE_PREFETCH_PREFILL_MIN_KV` overrides.
pub const PREFILL_PREFETCH_MIN_KV: u64 = 262144;

pub const PrefillPrefetchMode = enum { off, kv_gated, on };

/// `QWEN4_PLE_PREFETCH_PREFILL`: absent = the kv gate, `0` = serial walk, `1` = pool.
pub fn plePrefillPrefetchModeFromEnv(raw: ?[]const u8) PrefillPrefetchMode {
    const r = raw orelse return .kv_gated;
    if (r.len == 0) return .kv_gated;
    if (r[0] == '0') return .off;
    if (r[0] == '1') return .on;
    return .kv_gated;
}

/// The threshold, or the constant when the override is absent or unparsable.
pub fn plePrefillPrefetchMinKvFromEnv(raw: ?[]const u8) u64 {
    const r = raw orelse return PREFILL_PREFETCH_MIN_KV;
    const t = std.mem.trim(u8, r, " \t");
    if (t.len == 0) return PREFILL_PREFETCH_MIN_KV;
    return std.fmt.parseInt(u64, t, 10) catch PREFILL_PREFETCH_MIN_KV;
}

pub fn plePrefillPrefetchWanted(mode: PrefillPrefetchMode, kv_len: u64, min_kv: u64) bool {
    return switch (mode) {
        .off => false,
        .on => true,
        .kv_gated => kv_len >= min_kv,
    };
}

fn plePrefillPrefetchMinKv() u64 {
    if (ple_prefill_min_kv_override) |v| return v;
    const S = struct {
        var v: ?u64 = null;
    };
    if (S.v) |v| return v;
    const raw = std.c.getenv("QWEN4_PLE_PREFETCH_PREFILL_MIN_KV");
    const v = plePrefillPrefetchMinKvFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    S.v = v;
    return v;
}

fn plePrefillPrefetchEnabled(kv_len: u64) bool {
    if (ple_prefill_prefetch_override) |v| return v;
    const S = struct {
        var v: ?PrefillPrefetchMode = null;
    };
    const mode = S.v orelse blk: {
        const raw = std.c.getenv("QWEN4_PLE_PREFETCH_PREFILL");
        const m = plePrefillPrefetchModeFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
        S.v = m;
        break :blk m;
    };
    return plePrefillPrefetchWanted(mode, kv_len, plePrefillPrefetchMinKv());
}

pub fn bf16ToF32(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

// ── tests ──

const testing = std.testing;

test "ngram hash reproduces the reference multipliers, primes and offsets" {
    const h = try NgramHash.init(248320, 3, 8, 20_000_000, 128, 1234, 0, 248044);
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
    const h = try NgramHash.init(248320, 3, 8, 20_000_000, 128, 1234, 0, 248044);
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

test "ngram table row dequant reads the dense mx.quantize packing at every width" {
    // mx.quantize packs element i at bit offset i*bits of the little-endian
    // u32 stream (wcols = dim*bits/32); at 3/5/6 bits elements straddle words.
    // dim 32, one group, q[i] = i % 4, scale 1, bias 0 ⇒ out[i] == i % 4.
    inline for ([_]u32{ 2, 3, 4, 5, 6, 8 }) |bits| {
        const wcols = 32 * bits / 32;
        var packed_words: [8]u32 = @splat(0);
        var i: u32 = 0;
        while (i < 32) : (i += 1) {
            const off = i * bits;
            const q: u64 = i % 4;
            const w = off / 32;
            packed_words[w] |= @truncate(q << @intCast(off % 32));
            if (off % 32 + bits > 32) packed_words[w + 1] |= @truncate(q >> @intCast(32 - off % 32));
        }
        var hbuf: [256]u8 = undefined;
        const header = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"bits\":\"{d}\",\"group_size\":\"32\"}},\"weight\":{{\"dtype\":\"U32\",\"shape\":[1,{d}],\"data_offsets\":[0,{d}]}},\"scales\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[{d},{d}]}},\"biases\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[{d},{d}]}}}}", .{ bits, wcols, wcols * 4, wcols * 4, wcols * 4 + 2, wcols * 4 + 2, wcols * 4 + 4 });
        const total = 8 + 256 + wcols * 4 + 4;
        const buf = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), total);
        defer std.heap.page_allocator.free(buf);
        @memset(buf, ' ');
        std.mem.writeInt(u64, buf[0..8], 256, .little);
        @memcpy(buf[8 .. 8 + header.len], header);
        const data = buf[264..];
        for (0..wcols) |w| std.mem.writeInt(u32, data[w * 4 ..][0..4], packed_words[w], .little);
        std.mem.writeInt(u16, data[wcols * 4 ..][0..2], 0x3F80, .little);
        std.mem.writeInt(u16, data[wcols * 4 + 2 ..][0..2], 0, .little);
        const t = try NgramTable.parse(buf, buf[8..264], 264);
        var out: [32]f32 = undefined;
        t.row(0, &out);
        for (out, 0..) |v, k| try testing.expectEqual(@as(f32, @floatFromInt(k % 4)), v);
    }
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

test "ngram prefill prefetch is KV-GATED: a short prompt walks, a long one pools" {
    const min = PREFILL_PREFETCH_MIN_KV;
    try testing.expect(!plePrefillPrefetchWanted(.kv_gated, 0, min));
    for ([_]u64{ 4096, 8192, 16_384, 65_536, 131_072, 262_143 }) |kv| {
        try testing.expect(!plePrefillPrefetchWanted(.kv_gated, kv, min));
    }
    try testing.expect(plePrefillPrefetchWanted(.kv_gated, min, min));
    try testing.expect(plePrefillPrefetchWanted(.kv_gated, 355_000, min));
    try testing.expect(plePrefillPrefetchWanted(.kv_gated, 8192, 4096));
    try testing.expect(!plePrefillPrefetchWanted(.kv_gated, 8192, 131_072));
    try testing.expect(!plePrefillPrefetchWanted(.off, 1_000_000, min));
    try testing.expect(plePrefillPrefetchWanted(.on, 0, min));

    try testing.expectEqual(PrefillPrefetchMode.kv_gated, plePrefillPrefetchModeFromEnv(null));
    try testing.expectEqual(PrefillPrefetchMode.kv_gated, plePrefillPrefetchModeFromEnv(""));
    try testing.expectEqual(PrefillPrefetchMode.off, plePrefillPrefetchModeFromEnv("0"));
    try testing.expectEqual(PrefillPrefetchMode.on, plePrefillPrefetchModeFromEnv("1"));

    try testing.expectEqual(min, plePrefillPrefetchMinKvFromEnv(null));
    try testing.expectEqual(@as(u64, 131_072), plePrefillPrefetchMinKvFromEnv("131072"));
    try testing.expectEqual(min, plePrefillPrefetchMinKvFromEnv("64k"));
    try testing.expectEqual(min, plePrefillPrefetchMinKvFromEnv(""));
    try testing.expectEqual(@as(u64, 0), plePrefillPrefetchMinKvFromEnv("0")); // an explicit always-on
}

test "ngram prefill gather: 4096 rows through the pool equal the direct mmap read" {
    // 4-bit, group 32, dim 32 -> wcols 4 u32, scols 1. 4096 rows = 64 pool batches.
    const ROWS: usize = 4096;
    const HDR: usize = 512;
    const W: usize = ROWS * 16;
    const SB: usize = ROWS * 2;
    const buf = try testing.allocator.alloc(u8, 8 + HDR + W + 2 * SB);
    defer testing.allocator.free(buf);
    const header = "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
        "\"weight\":{\"dtype\":\"U32\",\"shape\":[4096,4],\"data_offsets\":[0,65536]}," ++
        "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4096,1],\"data_offsets\":[65536,73728]}," ++
        "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4096,1],\"data_offsets\":[73728,81920]}}";
    std.mem.writeInt(u64, buf[0..8], HDR, .little);
    @memset(buf[8 .. 8 + HDR], ' ');
    @memcpy(buf[8..][0..header.len], header);
    for (buf[8 + HDR ..], 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);

    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});

    warm_override = false; // the warm thread would race the fd for no benefit
    defer warm_override = null;
    var t = try NgramTable.open(path);
    defer t.close();
    const pool = t.pool orelse return error.SkipZigTest;
    try testing.expectEqual(@as(u32, 32), t.dim);

    const ids = try testing.allocator.alloc(i64, ROWS);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*r, i| r.* = @intCast((i *% 1237) % ROWS);
    const ref = try testing.allocator.alloc(f32, ROWS * t.dim);
    defer testing.allocator.free(ref);
    const got = try testing.allocator.alloc(f32, ROWS * t.dim);
    defer testing.allocator.free(got);

    ple_prefill_arm_said[0][0].store(false, .monotonic);
    ple_prefill_arm_said[0][1].store(false, .monotonic);
    ple_prefill_arm_said[1][0].store(false, .monotonic);
    ple_prefill_arm_said[1][1].store(false, .monotonic);
    const said = struct {
        fn f(arm: usize, bucket: usize) bool {
            return ple_prefill_arm_said[arm][bucket].load(.monotonic);
        }
    }.f;

    // Below the kv gate: serial mmap walk, no pool round.
    ple_prefill_min_kv_override = 65_536;
    defer ple_prefill_min_kv_override = null;
    const before = pool.runs.load(.monotonic);
    t.gather(ids, ref, 8192);
    try testing.expectEqual(before, pool.runs.load(.monotonic));
    try testing.expect(said(0, 1));
    try testing.expect(!said(0, 0));
    try testing.expect(!said(1, 1) and !said(1, 0));

    // Past the threshold the same gather rides the pool.
    t.gather(ids, got, 131_072);
    try testing.expectEqual(before + ROWS / PrefetchPool.MAX_ROWS, pool.runs.load(.monotonic));
    try testing.expect(said(1, 1));
    try testing.expect(!said(1, 0));

    try testing.expectEqualSlices(f32, ref, got);

    ple_prefill_prefetch_override = false;
    defer ple_prefill_prefetch_override = null;
    const forced_off = pool.runs.load(.monotonic);
    t.gather(ids, got, 1_000_000);
    try testing.expectEqual(forced_off, pool.runs.load(.monotonic));
    try testing.expectEqualSlices(f32, ref, got);
    ple_prefill_prefetch_override = true;
    t.gather(ids, got, 0);
    try testing.expectEqual(forced_off + ROWS / PrefetchPool.MAX_ROWS, pool.runs.load(.monotonic));
    try testing.expectEqualSlices(f32, ref, got);

    // A wide-but-short gather (> MAX_ROWS, < PREFILL_SAY_MIN_ROWS) reports the warmup bucket.
    const warm = ids[0..128];
    const w_out = try testing.allocator.alloc(f32, warm.len * t.dim);
    defer testing.allocator.free(w_out);
    t.gather(warm, w_out, 0); // forced on: the warmup forward runs at kv 0
    try testing.expect(said(1, 0));
    try testing.expectEqualSlices(f32, got[0 .. warm.len * t.dim], w_out);

    const serial_warm_before = said(0, 0);
    const dec = ids[0..16];
    const d_ref = try testing.allocator.alloc(f32, dec.len * t.dim);
    defer testing.allocator.free(d_ref);
    const d_got = try testing.allocator.alloc(f32, dec.len * t.dim);
    defer testing.allocator.free(d_got);
    ple_prefill_prefetch_override = false;
    const dec_before = pool.runs.load(.monotonic);
    t.gather(dec, d_ref, 1_000_000);
    try testing.expectEqual(dec_before + 1, pool.runs.load(.monotonic)); // still pooled
    ple_prefill_prefetch_override = true;
    t.gather(dec, d_got, 0);
    try testing.expectEqualSlices(f32, d_ref, d_got);
    try testing.expectEqual(serial_warm_before, said(0, 0));
}

test "ngram prefill gather: the threaded walk is byte-identical to the serial one" {
    // The threaded arm only engages past 1024 rows, so 4096 exercises the
    // work-stealing loop over several 256-row blocks per worker. A wrong `end`
    // bound or a `dim` mix-up in parWorker corrupts rows here; the serial walk
    // is the reference.
    const ROWS: usize = 4096;
    const HDR: usize = 512;
    const W: usize = ROWS * 16;
    const SB: usize = ROWS * 2;
    const buf = try testing.allocator.alloc(u8, 8 + HDR + W + 2 * SB);
    defer testing.allocator.free(buf);
    const header = "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
        "\"weight\":{\"dtype\":\"U32\",\"shape\":[4096,4],\"data_offsets\":[0,65536]}," ++
        "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4096,1],\"data_offsets\":[65536,73728]}," ++
        "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4096,1],\"data_offsets\":[73728,81920]}}";
    std.mem.writeInt(u64, buf[0..8], HDR, .little);
    @memset(buf[8 .. 8 + HDR], ' ');
    @memcpy(buf[8..][0..header.len], header);
    for (buf[8 + HDR ..], 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);

    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});

    warm_override = false;
    defer warm_override = null;
    var t = try NgramTable.open(path);
    defer t.close();
    try testing.expectEqual(@as(u32, 32), t.dim);

    // Keep the pool out of it: this test is about the threaded mmap walk only.
    ple_prefill_prefetch_override = false;
    defer ple_prefill_prefetch_override = null;

    const ids = try testing.allocator.alloc(i64, ROWS);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*r, i| r.* = @intCast((i *% 1237) % ROWS);

    const ref = try testing.allocator.alloc(f32, ROWS * t.dim);
    defer testing.allocator.free(ref);
    const got = try testing.allocator.alloc(f32, ROWS * t.dim);
    defer testing.allocator.free(got);
    @memset(got, std.math.nan(f32));

    ple_par_override = 1;
    t.gather(ids, ref, 0);
    ple_par_override = 16;
    t.gather(ids, got, 0);
    ple_par_override = null;

    try testing.expectEqualSlices(f32, ref, got);

    // A row count under the 1024 floor must stay on the serial path and still
    // agree — the gate itself is part of the contract.
    const SHORT: usize = 512;
    @memset(got[0 .. SHORT * t.dim], std.math.nan(f32));
    ple_par_override = 16;
    t.gather(ids[0..SHORT], got[0 .. SHORT * t.dim], 0);
    ple_par_override = null;
    try testing.expectEqualSlices(f32, ref[0 .. SHORT * t.dim], got[0 .. SHORT * t.dim]);
}

test "ngram table warm: touches the whole file in the background; close() joins mid-warm" {
    // The 4-bit fixture from the nibble-layout test, written to a real file
    // so open()'s kept fd serves the warm preads.
    var buf: [8 + 512 + 2 * 16 + 2 * 2 + 2 * 2]u8 = undefined;
    const header = "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"},\"weight\":{\"dtype\":\"U32\",\"shape\":[2,4],\"data_offsets\":[0,32]},\"scales\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[32,36]},\"biases\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[36,40]}}";
    var hdr: [512]u8 = @splat(' ');
    @memcpy(hdr[0..header.len], header);
    std.mem.writeInt(u64, buf[0..8], 512, .little);
    @memcpy(buf[8..520], &hdr);
    @memset(buf[520..], 0x33);
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = &buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});

    warm_override = true;
    defer warm_override = null;
    var t = try NgramTable.open(path);
    t.startWarm();
    try testing.expect(t.warm_thread != null);
    var spins: u32 = 0;
    while (t.warm_bytes.load(.acquire) < buf.len) : (spins += 1) {
        if (spins > 10_000) return error.WarmNeverFinished;
        var ts: std.c.timespec = .{ .sec = 0, .nsec = 1_000_000 };
        _ = std.c.nanosleep(&ts, null);
    }
    try testing.expectEqual(@as(u64, buf.len), t.warm_bytes.load(.acquire));
    t.close();

    // close() during the warm joins instead of racing the fd/munmap.
    var t2 = try NgramTable.open(path);
    t2.startWarm();
    t2.close();

    // Kill switch: no thread.
    warm_override = false;
    var t3 = try NgramTable.open(path);
    t3.startWarm();
    try testing.expect(t3.warm_thread == null);
    t3.close();
}

/// A whole `ngram_table.bin` image in one page-aligned buffer. Caller frees with the page allocator.
fn ngramTestImage(header: []const u8, data_bytes: usize) ![]align(std.heap.page_size_min) u8 {
    const hlen: usize = 512;
    std.debug.assert(header.len <= hlen);
    const buf = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), 8 + hlen + data_bytes);
    @memset(buf, ' ');
    std.mem.writeInt(u64, buf[0..8], hlen, .little);
    @memcpy(buf[8 .. 8 + header.len], header);
    @memset(buf[8 + hlen ..], 0);
    return buf;
}

fn ngramTestParse(header: []const u8, data_bytes: usize) !NgramTable {
    const buf = try ngramTestImage(header, data_bytes);
    return NgramTable.parse(buf, buf[8..520], 520);
}

/// rows 4, dim 64, 4-bit, group 32 => wcols 8, scols 2; w 128 B, s/b 16 B each.
const NGRAM_GOOD_HEADER =
    "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"4\",\"group_size\":\"32\"}," ++
    "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
    "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
    "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}";
const NGRAM_GOOD_BYTES = 160;

test "ngram table header: a missing or wrong-typed field is a named error, never a trap" {
    const t = try ngramTestParse(NGRAM_GOOD_HEADER, NGRAM_GOOD_BYTES);
    try testing.expectEqual(@as(u64, 4), t.rows);
    try testing.expectEqual(@as(u32, 64), t.dim);
    try testing.expectEqual(@as(u32, 4), t.bits);
    std.heap.page_allocator.free(@constCast(t.map));

    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"bits\":4,\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"F32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableHeader, ngramTestParse(
        "{\"__metadata__\":{\"format\":\"pt\",\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
}

test "ngram table header: bits must be a width mx.quantize actually ships" {
    try testing.expectError(error.NgramTableBits, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"32\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,64],\"data_offsets\":[0,1024]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[1024,1040]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[1040,1056]}}",
        1056,
    ));
    try testing.expectError(error.NgramTableBits, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"7\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,14],\"data_offsets\":[0,224]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[224,240]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[240,256]}}",
        256,
    ));
    try testing.expectError(error.NgramTableBits, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"0\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        160,
    ));
}

test "ngram table header: every region is bounded, sized by its own shape and disjoint" {
    // Weight region too small for rows x wcols x 4.
    try testing.expectError(error.NgramTableRegion, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,64]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    // Scales overlapping the weights.
    try testing.expectError(error.NgramTableRegion, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[120,136]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableRegion, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[0,8],\"data_offsets\":[0,0]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[0,2],\"data_offsets\":[0,0]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[0,2],\"data_offsets\":[0,0]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableRegion, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[128,136]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}",
        NGRAM_GOOD_BYTES,
    ));
    try testing.expectError(error.NgramTableTruncated, ngramTestParse(
        "{\"__metadata__\":{\"bits\":\"4\",\"group_size\":\"32\"}," ++
            "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
            "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
            "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[400,416]}}",
        NGRAM_GOOD_BYTES,
    ));
}

test "NgramHash.init refuses a config past its fixed arrays instead of asserting" {
    try testing.expectError(error.InvalidQwen4NgramSize, NgramHash.init(248320, 9, 8, 20_000_000, 128, 1234, 0, 248044));
    try testing.expectError(error.InvalidQwen4NgramSize, NgramHash.init(248320, 1, 8, 20_000_000, 128, 1234, 0, 248044));
    try testing.expectError(error.InvalidQwen4NgramHeads, NgramHash.init(248320, 5, 16, 20_000_000, 128, 1234, 0, 248044));
    try testing.expectError(error.InvalidQwen4NgramHeads, NgramHash.init(248320, 3, 0, 20_000_000, 128, 1234, 0, 248044));
    try testing.expectError(error.InvalidQwen4NgramVocab, NgramHash.init(248320, 3, 8, 20_000_000, 0, 1234, 0, 248044));
    const wide = try NgramHash.init(248320, 5, 8, 20_000_000, 128, 1234, 0, 248044);
    try testing.expectEqual(@as(u32, 32), wide.n_heads);
}

test "WarmProgress emits on the byte step, on the silence timeout, and never twice for one step" {
    const GB: u64 = 1 << 30;
    const S: u64 = 1_000_000_000;
    var p: WarmProgress = .{};
    // Nothing before the first step, however long it takes... except that a
    // long silence is itself worth a line.
    try testing.expect(!p.should(1 * GB, 1 * S));
    try testing.expect(!p.should(2 * GB, 9 * S));
    try testing.expect(p.should(3 * GB, 10 * S)); // silence timeout
    try testing.expect(!p.should(4 * GB, 11 * S)); // clock restarted by that line
    // Crossing the byte step emits once, and the step advances past it.
    try testing.expect(p.should(WARM_LOG_BYTES, 12 * S));
    try testing.expect(!p.should(WARM_LOG_BYTES, 13 * S));
    try testing.expect(!p.should(WARM_LOG_BYTES + 1, 13 * S));
    // A jump of several steps still emits exactly once and does not backlog.
    try testing.expect(p.should(WARM_LOG_BYTES * 4, 14 * S));
    try testing.expect(!p.should(WARM_LOG_BYTES * 4 + 1, 15 * S));
    try testing.expect(p.should(WARM_LOG_BYTES * 5, 16 * S));
}
