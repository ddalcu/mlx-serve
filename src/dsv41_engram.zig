//! DeepSeek-V4.1 Engram: the n-gram hash of DeepSeek's `inference/engram.py`,
//! the on-disk tables it indexes, and the gated add into the residual stream.
//! The tables (~100 GB a layer) are read from disk per lookup, never loaded.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const qwen4_ple = @import("qwen4_ple.zig");

pub const MAX_LAYERS = 4;
pub const MAX_NGRAM = 8;
pub const MAX_HEADS = 16;
pub const MAX_COLS = (MAX_NGRAM - 1) * MAX_HEADS;

/// numpy's `default_rng(seed)`: SeedSequence entropy mixing feeding PCG64
/// (XSL-RR), and `integers()`'s Lemire draw. The hash multipliers come from it;
/// any other generator rehashes the whole table.
pub const NumpyRng = struct {
    state: u128,
    inc: u128,

    const MULT: u128 = (@as(u128, 2549297995355413924) << 64) | 4865540595714422341;

    pub fn init(seed: u32) NumpyRng {
        const INIT_A: u32 = 0x43b0d7e5;
        const MULT_A: u32 = 0x931e8875;
        const INIT_B: u32 = 0x8b51f9dd;
        const MULT_B: u32 = 0x58f38ded;
        const hashmix = struct {
            fn f(value: u32, hash_const: *u32) u32 {
                var v = value ^ hash_const.*;
                hash_const.* *%= MULT_A;
                v *%= hash_const.*;
                return v ^ (v >> 16);
            }
        }.f;
        const mix = struct {
            fn f(x: u32, y: u32) u32 {
                const r = 0xca01f9dd *% x -% 0x4973f715 *% y;
                return r ^ (r >> 16);
            }
        }.f;
        var pool: [4]u32 = undefined;
        var hc = INIT_A;
        for (&pool, 0..) |*p, i| p.* = hashmix(if (i == 0) seed else 0, &hc);
        for (0..4) |src| for (0..4) |dst| {
            if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], &hc));
        };
        var words: [8]u32 = undefined;
        var hb = INIT_B;
        for (&words, 0..) |*w, i| {
            var v = pool[i % 4] ^ hb;
            hb *%= MULT_B;
            v *%= hb;
            w.* = v ^ (v >> 16);
        }
        var u: [4]u64 = undefined;
        for (&u, 0..) |*x, k| x.* = @as(u64, words[2 * k]) | (@as(u64, words[2 * k + 1]) << 32);
        var r: NumpyRng = .{ .state = 0, .inc = ((@as(u128, u[2]) << 64 | u[3]) << 1) | 1 };
        r.step();
        r.state +%= @as(u128, u[0]) << 64 | u[1];
        r.step();
        return r;
    }

    fn step(self: *NumpyRng) void {
        self.state = self.state *% MULT +% self.inc;
    }

    fn next64(self: *NumpyRng) u64 {
        self.step();
        const x: u64 = @truncate((self.state >> 64) ^ self.state);
        return std.math.rotr(u64, x, @as(u32, @truncate(self.state >> 122)));
    }

    /// A draw in [0, n): numpy's 64-bit Lemire arm, the one `integers` takes
    /// for any range wider than 32 bits.
    pub fn below(self: *NumpyRng, n: u64) u64 {
        std.debug.assert(n > 1 << 32);
        var m: u128 = @as(u128, self.next64()) * n;
        var leftover: u64 = @truncate(m);
        if (leftover < n) {
            const threshold = (std.math.maxInt(u64) - (n - 1)) % n;
            while (leftover < threshold) {
                m = @as(u128, self.next64()) * n;
                leftover = @truncate(m);
            }
        }
        return @intCast(m >> 64);
    }
};

fn isPrime(n: u64) bool {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    var i: u64 = 3;
    while (i * i <= n) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/// The per-layer hash constants: one odd multiplier per lookback, one prime
/// bucket per (n-gram size, head), and each bucket's first row.
pub const Hash = struct {
    n_layers: u32,
    max_ngram: u32,
    heads: u32,
    pad: u32,
    mult: [MAX_LAYERS][MAX_NGRAM]u64 = @splat(@splat(0)),
    primes: [MAX_LAYERS][MAX_COLS]u64 = @splat(@splat(0)),
    offsets: [MAX_LAYERS][MAX_COLS]u64 = @splat(@splat(0)),

    /// `pad` is the compressed id of the config's pad token. Refuses a config
    /// whose prime buckets do not add up to its declared table rows.
    pub fn init(cfg: *const model.ModelConfig, pad: u32) !Hash {
        const n_layers = cfg.dsv41_n_engram_layers;
        var h: Hash = .{ .n_layers = n_layers, .max_ngram = cfg.dsv41_engram_max_ngram, .heads = cfg.dsv41_engram_n_heads, .pad = pad };
        if (n_layers > MAX_LAYERS or h.max_ngram < 2 or h.max_ngram > MAX_NGRAM or h.heads == 0 or h.heads > MAX_HEADS) return error.EngramGeometry;
        const bound = (@as(u64, std.math.maxInt(i64)) / cfg.dsv41_engram_compressed_vocab) / 2;
        if (bound <= 1 << 32) return error.EngramGeometry;
        var p = cfg.dsv41_engram_vocab_size - 1;
        for (0..n_layers) |l| {
            var rng = NumpyRng.init(10007 * @as(u32, cfg.dsv41_engram_layers[l]));
            for (0..h.max_ngram) |k| h.mult[l][k] = rng.below(bound) * 2 + 1;
            var off: u64 = 0;
            for (0..h.cols()) |c| {
                p += 1;
                while (!isPrime(p)) p += 1;
                h.primes[l][c] = p;
                h.offsets[l][c] = off;
                off += p;
            }
            if (off != cfg.dsv41_engram_rows[l]) return error.EngramTableRows;
        }
        return h;
    }

    pub fn cols(self: *const Hash) u32 {
        return (self.max_ngram - 1) * self.heads;
    }

    /// Row ids of the n-grams ending at each of `ids`, which follow `hist` (the
    /// request's earlier compressed ids). `out` is `[ids.len][n_layers][cols]`.
    /// Look-back stops at the sequence start, where the slots read `pad`.
    pub fn rows(self: *const Hash, hist: []const u32, ids: []const u32, out: []u32) void {
        const nc = self.cols();
        std.debug.assert(out.len == ids.len * self.n_layers * nc);
        for (ids, 0..) |_, i| {
            const pos = hist.len + i;
            var toks: [MAX_NGRAM]u64 = undefined;
            for (0..self.max_ngram) |s| {
                toks[s] = if (pos < s) self.pad else if (pos - s >= hist.len) ids[pos - s - hist.len] else hist[pos - s];
            }
            for (0..self.n_layers) |l| {
                const dst = out[(i * self.n_layers + l) * nc ..][0..nc];
                var rolling = toks[0] *% self.mult[l][0];
                for (1..self.max_ngram) |k| {
                    rolling ^= toks[k] *% self.mult[l][k];
                    for (0..self.heads) |hd| {
                        const c = (k - 1) * self.heads + hd;
                        dst[c] = @intCast(rolling % self.primes[l][c] + self.offsets[l][c]);
                    }
                }
            }
        }
    }
};

/// Token id to compressed id: raw little-endian u32s (`engram-token-map.u32`,
/// the repack), a JSON list (`engram_token_map.json`, pipenetwork packs), or,
/// for a pack that ships none (oMLX's), the one DeepSeek's `engram.py` derives
/// from V4.1's tokenizer.
pub fn loadTokenMap(gpa: std.mem.Allocator, dir_path: []const u8, vocab: u32, compressed: u32) ![]u32 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    const map = try gpa.alloc(u32, vocab);
    errdefer gpa.free(map);
    if (dir.readFileAlloc(io, "engram-token-map.u32", gpa, .limited(64 << 20))) |raw| {
        defer gpa.free(raw);
        if (raw.len != @as(usize, vocab) * 4) return error.EngramTokenMap;
        for (map, 0..) |*v, i| v.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
    } else |_| {
        if (dir.readFileAlloc(io, "engram_token_map.json", gpa, .limited(64 << 20))) |raw| {
            defer gpa.free(raw);
            const parsed = std.json.parseFromSlice([]const u32, gpa, raw, .{}) catch return error.EngramTokenMap;
            defer parsed.deinit();
            if (parsed.value.len != vocab) return error.EngramTokenMap;
            @memcpy(map, parsed.value);
        } else |_| {
            unpackTokenMap(map, v41_token_map) catch return error.EngramTokenMapMissing;
            if (std.mem.max(u32, map) + 1 != compressed) return error.EngramTokenMapMissing;
        }
    }
    var hi: u32 = 0;
    for (map) |v| hi = @max(hi, v);
    // Every multiplier derives from the compressed vocab: a map built for
    // another tokenizer would silently rehash the whole table.
    if (hi + 1 != compressed) return error.EngramTokenMap;
    return map;
}

/// V4.1's map (`tests/dump_dsv41_token_map.py`): one bit per token id, set where
/// the token opens the next compressed id, then every other token's id as a u32.
const v41_token_map = @embedFile("fixtures/dsv41_engram_token_map.bin");

fn unpackTokenMap(map: []u32, raw: []const u8) !void {
    const bits = (map.len + 7) / 8;
    if (raw.len < bits) return error.EngramTokenMap;
    var next: u32 = 0;
    var at = bits;
    for (map, 0..) |*v, t| {
        if ((raw[t / 8] >> @intCast(t % 8)) & 1 != 0) {
            v.* = next;
            next += 1;
        } else {
            if (at + 4 > raw.len) return error.EngramTokenMap;
            v.* = std.mem.readInt(u32, raw[at..][0..4], .little);
            at += 4;
        }
    }
    if (at != raw.len) return error.EngramTokenMap;
}

/// One layer's table, read per lookup from disk: the repack's flat mxfp8
/// records (code bytes, then one e8m0 scale per 32), or an MLX affine
/// table inside a safetensors shard (weight / scales / biases tensors).
pub const Table = struct {
    fd: std.c.fd_t,
    rows: u64,
    head_dim: u32,
    mode: model.QuantMode,
    bits: u32,
    group_size: u32,
    w_off: u64,
    w_stride: u64,
    wlen: usize,
    s_off: u64,
    s_stride: u64,
    slen: usize,
    b_off: u64 = 0,

    pub fn close(self: *Table) void {
        _ = std.c.close(self.fd);
    }

    /// Bytes this table occupies inside the pack's safetensors shards (read by
    /// pread, never resident); a repack table lives outside them.
    pub fn shardBytes(self: *const Table) u64 {
        return if (self.mode == .affine) self.rows * (self.w_stride + 2 * self.s_stride) else 0;
    }

    /// Layer `layer`'s table in `dir`; `index` is its position among the
    /// config's Engram layers.
    pub fn open(gpa: std.mem.Allocator, dir_path: []const u8, index: usize, layer: usize, rows: u64, head_dim: u32) !Table {
        _ = index;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
        defer dir.close(io);
        if (dir.readFileAlloc(io, "engram/engram-manifest.json", a, .limited(16 << 20))) |raw| {
            const Layer = struct { layer_id: u32, file: []const u8, rows: u64, record_bytes: u32, quant: struct { bits: u32, group_size: u32, mode: []const u8, head_dim: u32 } };
            const M = struct { layers: []const Layer };
            const man = std.json.parseFromSliceLeaky(M, a, raw, .{ .ignore_unknown_fields = true }) catch return error.EngramManifest;
            for (man.layers) |ly| {
                if (ly.layer_id != layer) continue;
                const rec: u32 = head_dim + head_dim / 32;
                if (ly.rows != rows or ly.record_bytes != rec or ly.quant.bits != 8 or ly.quant.group_size != 32 or
                    !std.mem.eql(u8, ly.quant.mode, "mxfp8") or ly.quant.head_dim != head_dim) return error.EngramManifest;
                if (std.mem.indexOfAny(u8, ly.file, "/\\") != null) return error.EngramManifest;
                const path = try std.fmt.allocPrint(a, "engram/{s}", .{ly.file});
                const fd = try openAt(a, dir_path, path);
                errdefer _ = std.c.close(fd);
                if (fileSize(fd) != rows * rec) return error.EngramTableSize;
                return .{ .fd = fd, .rows = rows, .head_dim = head_dim, .mode = .mxfp8, .bits = 8, .group_size = 32, .w_off = 0, .w_stride = rec, .wlen = head_dim, .s_off = head_dim, .s_stride = rec, .slen = head_dim / 32 };
            }
            return error.EngramManifest;
        } else |_| {}
        // An MLX pack: the three tensors in one shard named by the index.
        const index_raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20)) catch return error.EngramTableMissing;
        const idx = std.json.parseFromSliceLeaky(std.json.Value, a, index_raw, .{}) catch return error.EngramTableMissing;
        const wm = (if (idx == .object) idx.object.get("weight_map") else null) orelse return error.EngramTableMissing;
        if (wm != .object) return error.EngramTableMissing;
        var names: [3][]const u8 = undefined;
        var file: ?[]const u8 = null;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |part, i| {
            names[i] = try std.fmt.allocPrint(a, "layers.{d}.engram.embed.{s}", .{ layer, part });
            if (wm.object.get(names[i]) == null) names[i] = try std.fmt.allocPrint(a, "{s}{s}", .{ model.dsv41_text_prefix, names[i] });
            const f = wm.object.get(names[i]) orelse return error.EngramTableMissing;
            if (f != .string or std.mem.indexOfAny(u8, f.string, "/\\") != null) return error.EngramTableMissing;
            if (file) |prev| if (!std.mem.eql(u8, prev, f.string)) return error.EngramTableSplit;
            file = f.string;
        }
        const fd = try openAt(a, dir_path, file.?);
        errdefer _ = std.c.close(fd);
        var hl: [8]u8 = undefined;
        if (std.c.pread(fd, &hl, 8, 0) != 8) return error.EngramTableHeader;
        const hlen = std.mem.readInt(u64, &hl, .little);
        if (hlen > 16 << 20) return error.EngramTableHeader;
        const hbuf = try a.alloc(u8, @intCast(hlen));
        if (std.c.pread(fd, hbuf.ptr, hbuf.len, 8) != @as(isize, @intCast(hlen))) return error.EngramTableHeader;
        const hdr = std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{}) catch return error.EngramTableHeader;
        if (hdr != .object) return error.EngramTableHeader;
        const size = fileSize(fd);
        var regions: [3]qwen4_ple.HeaderRegion = undefined;
        for (names, 0..) |name, i| {
            regions[i] = qwen4_ple.headerRegion(hdr.object, name, if (i == 0) "U32" else "BF16", if (i == 0) 4 else 2, @intCast(size), @intCast(8 + hlen), .matrix) catch return error.EngramTableHeader;
            if (regions[i].rows != rows) return error.EngramTableSize;
        }
        const wcols = regions[0].cols;
        const scols = regions[1].cols;
        if (regions[2].cols != scols or scols == 0 or head_dim % scols != 0 or (wcols * 32) % head_dim != 0) return error.EngramTableGeometry;
        const base = 8 + hlen;
        return .{
            .fd = fd,
            .rows = rows,
            .head_dim = head_dim,
            .mode = .affine,
            .bits = @intCast(wcols * 32 / head_dim),
            .group_size = @intCast(head_dim / scols),
            .w_off = base + regions[0].start,
            .w_stride = wcols * 4,
            .wlen = @intCast(wcols * 4),
            .s_off = base + regions[1].start,
            .s_stride = scols * 2,
            .slen = @intCast(scols * 2),
            .b_off = base + regions[2].start,
        };
    }

    /// Rows `ids`, dequantized: f32 `[ids.len, head_dim]` (caller frees).
    pub fn gather(self: *const Table, gpa: std.mem.Allocator, ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
        const n = ids.len;
        const w = try gpa.alignedAlloc(u8, .@"4", n * self.wlen);
        defer gpa.free(w);
        const sc = try gpa.alignedAlloc(u8, .@"4", n * self.slen);
        defer gpa.free(sc);
        const bi = try gpa.alignedAlloc(u8, .@"4", if (self.mode == .affine) n * self.slen else 0);
        defer gpa.free(bi);
        const job: Job = .{ .t = self, .ids = ids, .w = w, .s = sc, .b = bi };
        if (n >= PAR_MIN_ROWS) {
            var threads: [PAR_THREADS]?std.Thread = @splat(null);
            for (&threads, 0..) |*t, k| t.* = std.Thread.spawn(.{}, Job.runVoid, .{ &job, k, PAR_THREADS }) catch null;
            var failed = false;
            for (threads, 0..) |t, k| {
                if (t) |th| th.join() else failed = failed or !job.run(k, PAR_THREADS);
            }
            if (failed or job.bad.load(.acquire)) return error.EngramRead;
        } else if (!job.run(0, 1)) return error.EngramRead;
        const wcols: c_int = @intCast(self.wlen / 4);
        const scols: c_int = @intCast(if (self.mode == .affine) self.slen / 2 else self.slen);
        const rows: c_int = @intCast(n);
        const wa = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ rows, wcols }, 2, .uint32);
        defer _ = mlx.mlx_array_free(wa);
        const sdt: mlx.mlx_dtype = if (self.mode == .affine) .bfloat16 else .uint8;
        const sa = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{ rows, scols }, 2, sdt);
        defer _ = mlx.mlx_array_free(sa);
        const ba: mlx.mlx_array = if (self.mode == .affine) mlx.mlx_array_new_data(bi.ptr, &[_]c_int{ rows, scols }, 2, .bfloat16) else .{ .ctx = null };
        defer if (ba.ctx != null) {
            _ = mlx.mlx_array_free(ba);
        };
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_dequantize(&out, wa, sa, ba, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), self.mode.cstr(), .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        return out;
    }

    const Job = struct {
        t: *const Table,
        ids: []const u32,
        w: []u8,
        s: []u8,
        b: []u8,
        bad: std.atomic.Value(bool) = .init(false),

        fn runVoid(self: *const Job, first: usize, stride: usize) void {
            _ = self.run(first, stride);
        }

        /// Every `stride`-th row from `first`: preads run in parallel where
        /// faults on one mapping would serialize.
        fn run(self: *const Job, first: usize, stride: usize) bool {
            const t = self.t;
            var i = first;
            while (i < self.ids.len) : (i += stride) {
                const r: u64 = self.ids[i];
                if (r >= t.rows or
                    !preadAll(t.fd, self.w[i * t.wlen ..][0..t.wlen], t.w_off + r * t.w_stride) or
                    !preadAll(t.fd, self.s[i * t.slen ..][0..t.slen], t.s_off + r * t.s_stride) or
                    (t.mode == .affine and !preadAll(t.fd, self.b[i * t.slen ..][0..t.slen], t.b_off + r * t.s_stride)))
                {
                    @constCast(&self.bad).store(true, .release);
                    return false;
                }
            }
            return true;
        }
    };
};

const PAR_MIN_ROWS = 512;
const PAR_THREADS = 16;

fn preadAll(fd: std.c.fd_t, dst: []u8, off: u64) bool {
    return std.c.pread(fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
}

fn fileSize(fd: std.c.fd_t) u64 {
    return @intCast(@max(std.c.lseek(fd, 0, std.c.SEEK.END), 0));
}

fn openAt(a: std.mem.Allocator, dir_path: []const u8, name: []const u8) !std.c.fd_t {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, name }, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    return fd;
}

const testing = std.testing;

fn releaseConfig() model.ModelConfig {
    var cfg: model.ModelConfig = .{};
    cfg.model_type = "deepseek_v41";
    cfg.dsv41_n_engram_layers = 2;
    cfg.dsv41_engram_layers[0] = 1;
    cfg.dsv41_engram_layers[1] = 14;
    cfg.dsv41_engram_rows[0] = 384006168;
    cfg.dsv41_engram_rows[1] = 384016682;
    cfg.dsv41_engram_max_ngram = 4;
    cfg.dsv41_engram_vocab_size = 16000000;
    cfg.dsv41_engram_n_heads = 8;
    cfg.dsv41_engram_compressed_vocab = 99092;
    return cfg;
}

test "dsv41 engram: hash multipliers are numpy default_rng(10007 * layer) draws" {
    // The integers DeepSeek's engram.py derives for the release (also shipped
    // verbatim in the OpensourceWTF repack's engram-manifest.json).
    const cfg = releaseConfig();
    const h = try Hash.init(&cfg, 2);
    const want = [2][4]u64{
        .{ 76632096046245, 4839876093313, 35959672319349, 73987337458391 },
        .{ 67716810739261, 51510806800915, 30921347202721, 82619226485591 },
    };
    for (want, 0..) |row, l| try testing.expectEqualSlices(u64, &row, h.mult[l][0..4]);
}

test "dsv41 engram: prime buckets continue across layers and fill the declared tables" {
    const cfg = releaseConfig();
    const h = try Hash.init(&cfg, 2);
    try testing.expectEqualSlices(u64, &.{ 16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183 }, h.primes[0][0..8]);
    try testing.expectEqualSlices(u64, &.{ 16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889 }, h.primes[1][16..24]);
    try testing.expectEqualSlices(u64, &.{ 0, 16000477, 32000964, 48001463, 64001970 }, h.offsets[1][0..5]);
    var bad = cfg;
    bad.dsv41_engram_rows[1] += 1;
    try testing.expectError(error.EngramTableRows, Hash.init(&bad, 2));
}

test "dsv41 engram: row ids match DeepSeek's NgramHashState across prefill and decode" {
    // inference/engram.py over compressed ids [3, 99091, 2, 77, 77, 1024], then
    // one decode step [5] at position 6 (identity token map, pad = 2).
    const cfg = releaseConfig();
    const h = try Hash.init(&cfg, 2);
    const prompt = [_]u32{ 3, 99091, 2, 77, 77, 1024 };
    var pre: [6 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &prompt, &pre);
    try testing.expectEqualSlices(u32, &.{ 7010988, 29651738, 47710115, 48176674, 72877570, 81111450, 101228567, 120693201, 143450283, 146576661, 168605081, 191902092, 196072724, 214158605, 233287660, 246388310, 261549600, 286523983, 293659617, 313227845, 320939872, 340508126, 357771274, 372619255 }, pre[0..24]);
    try testing.expectEqualSlices(u32, &.{ 11344931, 28156158, 45201617, 61942444, 78326219, 85383927, 106445668, 125242217, 129707946, 158452422, 167536917, 177080926, 202551179, 221735906, 238090062, 254314270, 259798390, 277434364, 301936998, 307898224, 335252679, 339837629, 352557585, 368186488 }, pre[(5 * 2 + 1) * 24 ..][0..24]);
    var dec: [2 * 24]u32 = undefined;
    h.rows(&prompt, &.{5}, &dec);
    try testing.expectEqualSlices(u32, &.{ 14359625, 17064925, 34766379, 48382691, 68824895, 91643820, 111054239, 121732398, 132594636, 146910113, 174982973, 181797503, 206306954, 210573457, 224988242, 243323351, 267856496, 279543941, 298892988, 316569280, 333687525, 351365567, 362014820, 382511691 }, dec[0..24]);
    // A chunk boundary anywhere hashes the same rows.
    var split: [6 * 2 * 24]u32 = undefined;
    h.rows(&.{}, prompt[0..2], split[0 .. 2 * 48]);
    h.rows(prompt[0..2], prompt[2..], split[2 * 48 ..]);
    try testing.expectEqualSlices(u32, &pre, &split);
}

/// An MLX pack's layer-1 table in one shard: 4 rows, head_dim 64, 8-bit g32
/// (weight [4, 16] U32, scales/biases [4, 2] BF16 = 288 bytes); `prefix` nests
/// its names (oMLX's packs: `language_model.`).
pub fn writeAffineFixture(io: std.Io, dir: std.Io.Dir, comptime prefix: []const u8) !void {
    const n = prefix ++ "layers.1.engram.embed.";
    const hdr = "{\"" ++ n ++ "weight\":{\"dtype\":\"U32\",\"shape\":[4,16],\"data_offsets\":[0,256]}," ++
        "\"" ++ n ++ "scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[256,272]}," ++
        "\"" ++ n ++ "biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[272,288]}}";
    var shard: [8 + hdr.len + 288]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], hdr.len, .little);
    @memcpy(shard[8..][0..hdr.len], hdr);
    try dir.writeFile(io, .{ .sub_path = "model-00001.safetensors", .data = &shard });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{" ++
        "\"" ++ n ++ "weight\":\"model-00001.safetensors\",\"" ++ n ++ "scales\":\"model-00001.safetensors\"," ++
        "\"" ++ n ++ "biases\":\"model-00001.safetensors\"}}" });
}

test "dsv41 engram: a table nested under language_model. (oMLX's packs) opens like a bare one" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeAffineFixture(io, td.dir, model.dsv41_text_prefix);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 0, 1, 4, 64);
    defer t.close();
    try testing.expectEqual(@as(u64, 288), t.shardBytes());
}

test "dsv41 engram: a pack with no token map (oMLX's) gets V4.1's own tokenizer's" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    const map = try loadTokenMap(testing.allocator, dir, 129280, 99092);
    defer testing.allocator.free(map);
    // the, The, " the", " The", THE, " THE" normalize alike (engram.py's ids for the release tokenizer).
    for ([_]u32{ 1805, 671, 270, 455, 20852, 6367 }) |id| try testing.expectEqual(@as(u32, 237), map[id]);
    try testing.expectEqual(@as(u32, 2), map[2]);
    try testing.expectEqual(@as(u32, 99091), map[129279]);
    // Another tokenizer's sizes: the pack must ship its own.
    try testing.expectError(error.EngramTokenMapMissing, loadTokenMap(testing.allocator, dir, 129280, 99000));
    try testing.expectError(error.EngramTokenMapMissing, loadTokenMap(testing.allocator, dir, 151936, 99092));
}

test "dsv41 engram: an affine table bills the shard bytes it occupies, a repack table none" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeAffineFixture(io, td.dir, "");
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    var t = try Table.open(testing.allocator, dir, 0, 1, 4, 64);
    defer t.close();
    try testing.expectEqual(@as(u64, 288), t.shardBytes());

    try td.dir.createDirPath(io, "engram");
    try td.dir.writeFile(io, .{ .sub_path = "engram/engram-manifest.json", .data = "{\"layers\":[{\"layer_id\":1,\"file\":\"engram-L1.bin\",\"rows\":4,\"record_bytes\":66," ++
        "\"quant\":{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\",\"head_dim\":64}}]}" });
    try td.dir.writeFile(io, .{ .sub_path = "engram/engram-L1.bin", .data = &@as([4 * 66]u8, @splat(0)) });
    var r = try Table.open(testing.allocator, dir, 0, 1, 4, 64);
    defer r.close();
    try testing.expectEqual(@as(u64, 0), r.shardBytes());
}
