//! DeepSeek-V4.1 Engram: the n-gram hash of DeepSeek's `inference/engram.py`,
//! the on-disk tables it indexes, and the gated add into the residual stream.
//! The tables (~100 GB a layer) are read from disk per lookup, never loaded.
const std = @import("std");
const builtin = @import("builtin");
const crt = @import("../../sdk/util/crt.zig");
const mlx = @import("../../mlx/ffi.zig");
const Config = @import("config.zig").Config;
const weights_load = @import("../../sdk/weights/load.zig");
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;
const qwen4_ple = @import("../qwen4_exp/ple_table.zig");
/// the tables read past the page cache through their blocks.
const edir = @import("prompt/engram_direct.zig");
/// the Engram LRU a decode read takes its rows from, its misses
/// read on a pool; a prompt grid's rows and the rounds' bonus rows go into it.
const edec = @import("decode/engram_decode.zig");
const gguf_claim = @import("../../sdk/quant/gguf/claim.zig");
const gguf_root = @import("../../sdk/quant/gguf/root.zig");
const ds41_gguf = @import("../../sdk/quant/gguf/arch/deepseek41.zig");

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
    /// The compressed id of a token that takes no part in an n-gram (an image span's): every n-gram
    /// reaching back to or past it hashes as all-pad, as the release's `engram_hash` blocks them.
    pub const DEAD: u32 = std.math.maxInt(u32);

    n_layers: u32,
    max_ngram: u32,
    heads: u32,
    pad: u32,
    mult: [MAX_LAYERS][MAX_NGRAM]u64 = @splat(@splat(0)),
    primes: [MAX_LAYERS][MAX_COLS]u64 = @splat(@splat(0)),
    offsets: [MAX_LAYERS][MAX_COLS]u64 = @splat(@splat(0)),

    /// `pad` is the compressed id of the config's pad token. Refuses a config
    /// whose prime buckets do not add up to its declared table rows.
    pub fn init(cfg: *const Config, pad: u32) !Hash {
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
            var blocked = false;
            for (0..self.max_ngram) |s| {
                const t: u32 = if (pos < s) self.pad else if (pos - s >= hist.len) ids[pos - s - hist.len] else hist[pos - s];
                blocked = blocked or t == DEAD;
                toks[s] = if (blocked) self.pad else t;
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
    // A GGUF picked by its file (`--mlx-gguf`) is the pack of its folder.
    const root = if (std.mem.endsWith(u8, dir_path, ".gguf")) std.fs.path.dirname(dir_path) orelse dir_path else dir_path;
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{});
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
const v41_token_map = @embedFile("../../fixtures/dsv41_engram_token_map.bin");

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
    fd: c_int,
    rows: u64,
    head_dim: u32,
    mode: QuantMode,
    bits: u32,
    group_size: u32,
    w_off: u64,
    w_stride: u64,
    wlen: usize,
    s_off: u64,
    s_stride: u64,
    slen: usize,
    b_off: u64 = 0,
    /// The table sits in a shard the preflight bills (`preflight.modelDiskBytes`), so `shardBytes` takes it back out.
    in_shards: bool = false,
    /// A prompt grid's rows read whole and past the page cache (`readDirect`); else span by span through it.
    direct: bool = false,
    /// The codes' own descriptor, which reads past the page cache (`openCodes`); -1: none.
    cfd: c_int = -1,
    /// Set while the page cache has no room for the codes (`pastCache`): they are read through `cfd`.
    past_cache: std.atomic.Value(bool) = .init(false),

    /// A second descriptor for the codes of a resident pack's table, read past the page cache while `pastCache` says
    /// so; the scales stay on `fd`. A row's codes sit in a table far larger than memory and are seldom read twice,
    /// its scales a page of a few thousand that every chunk reads again: with no memory to spare the cache evicted
    /// the scales to hold the codes, and every row cost two reads.
    pub fn openCodes(self: *Table) void {
        if (comptime !builtin.os.tag.isDarwin()) return;
        if (self.records() or self.mode == .gguf or self.cfd >= 0) return;
        var path: [std.c.PATH_MAX]u8 = undefined;
        if (std.c.fcntl(self.fd, std.c.F.GETPATH, &path) == -1) return;
        const fd = crt.openRead(@ptrCast(&path));
        if (fd < 0) return;
        _ = std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0));
        edir.register(fd);
        self.cfd = fd;
    }

    pub fn pastCache(self: *Table, on: bool) void {
        self.past_cache.store(on and self.cfd >= 0, .release);
    }

    fn wfd(self: *const Table) c_int {
        return if (self.past_cache.load(.acquire)) self.cfd else self.fd;
    }

    /// The table read past the page cache: a streamed bank's slot rows hold the memory the cache would (on a resident
    /// pack the cache serves a warm prompt's rows faster).
    pub fn readDirect(self: *Table) void {
        edir.register(self.fd);
        self.direct = true;
    }

    pub fn close(self: *Table) void {
        // the LRU let go of (its early reads in flight waited for).
        edec.forget(self.fd);
        // The direct Engram reads: the descriptor's block registration dropped with it.
        edir.unregister(self.fd);
        crt.close(self.fd);
        if (self.cfd >= 0) {
            edir.unregister(self.cfd);
            crt.close(self.cfd);
        }
    }

    /// A repacked table (`engram/engram-manifest.json`): a row's codes then its scales, one record a row, read with
    /// one call (`edir.readRecord`).
    pub fn records(self: *const Table) bool {
        return self.mode != .affine and self.w_stride == self.s_stride and self.s_off == self.w_off + self.wlen and
            self.wlen + self.slen <= edir.max_record;
    }
    /// Bytes this table occupies inside the pack's billed safetensors shards (read by
    /// pread, never resident); a repack table lives outside them.
    pub fn shardBytes(self: *const Table) u64 {
        if (!self.in_shards) return 0;
        return self.rows * (self.w_stride + self.s_stride * @as(u64, if (self.mode == .affine) 2 else 1));
    }

    /// Layer `layer`'s table in `dir`; `index` is its position among the
    /// config's Engram layers.
    pub fn open(gpa: std.mem.Allocator, dir_path: []const u8, index: usize, layer: usize, rows: u64, head_dim: u32) !Table {
        if (comptime @import("builtin").os.tag == .windows) return error.EngramUnsupportedOnWindows; // its tables open through std.c.open
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const io = std.Io.Threaded.global_single_threaded.io();
        if (try openGgufTable(a, io, dir_path, index, layer, rows, head_dim)) |t| return t;
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
                errdefer crt.close(fd);
                if (fileSize(fd) != rows * rec) return error.EngramTableSize;
                return mxfp8Records(fd, rows, head_dim, 0);
            }
            return error.EngramManifest;
        } else |_| {}
        if (try openRawTable(a, dir, dir_path, layer, rows, head_dim)) |t| return t;
        // An MLX pack: the three tensors in one shard named by the index.
        const index_raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20)) catch return error.EngramTableMissing;
        const idx = std.json.parseFromSliceLeaky(std.json.Value, a, index_raw, .{}) catch return error.EngramTableMissing;
        const wm = (if (idx == .object) idx.object.get("weight_map") else null) orelse return error.EngramTableMissing;
        if (wm != .object) return error.EngramTableMissing;
        var names: [3][]const u8 = undefined;
        var file: ?[]const u8 = null;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |part, i| {
            names[i] = try std.fmt.allocPrint(a, "layers.{d}.engram.embed.{s}", .{ layer, part });
            if (wm.object.get(names[i]) == null) names[i] = try std.fmt.allocPrint(a, "{s}{s}", .{ weights_load.dsv41_text_prefix, names[i] });
            const f = wm.object.get(names[i]) orelse return error.EngramTableMissing;
            if (f != .string or std.mem.indexOfAny(u8, f.string, "/\\") != null) return error.EngramTableMissing;
            if (file) |prev| if (!std.mem.eql(u8, prev, f.string)) return error.EngramTableSplit;
            file = f.string;
        }
        const fd = try openAt(a, dir_path, file.?);
        errdefer crt.close(fd);
        var hl: [8]u8 = undefined;
        if (crt.pread(fd, &hl, 8, 0) != 8) return error.EngramTableHeader;
        const hlen = std.mem.readInt(u64, &hl, .little);
        if (hlen > 16 << 20) return error.EngramTableHeader;
        const hbuf = try a.alloc(u8, @intCast(hlen));
        if (crt.pread(fd, hbuf.ptr, hbuf.len, 8) != @as(isize, @intCast(hlen))) return error.EngramTableHeader;
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
            .in_shards = true,
        };
    }

    /// A `--mlx-gguf` set's Engram (`ggufTable`), null when `dir_path` is not a GGUF we serve.
    fn openGgufTable(a: std.mem.Allocator, io: std.Io, dir_path: []const u8, index: usize, layer: usize, rows: u64, head_dim: u32) !?Table {
        const shard = gguf_claim.servablePath(io, a, dir_path) orelse return null;
        var set = try gguf_root.gguf.File.open(a, shard);
        defer set.deinit();
        return try ggufTable(a, &set, index, layer, rows, head_dim);
    }

    /// Engram table `index` (layer `layer`) of the GGUF `set`, read in place. IQ4_NL rows with the block scales inside
    /// each row (`slen` 0): in their own file beside the first shard (`deepseek41.engram_file_key`), or the re-split's
    /// shard 6, a run of the one `ple_table`. The DS4 layout keeps the release's fp8 records in the set as I8.
    fn ggufTable(a: std.mem.Allocator, set: *const gguf_root.gguf.File, index: usize, layer: usize, rows: u64, head_dim: u32) !Table {
        var nb: [64]u8 = undefined;
        const leaf = try std.fmt.bufPrint(&nb, "blk.{d}." ++ ds41_gguf.engram_leaf, .{layer});
        if (set.getString(ds41_gguf.engram_file_key)) |name| {
            if (std.mem.indexOfAny(u8, name, "/\\") != null) return error.EngramTableMissing;
            const dir = std.fs.path.dirname(set.paths[0]) orelse return error.EngramTableMissing;
            const path = try std.fs.path.join(a, &.{ dir, name });
            var f = gguf_root.gguf.File.open(a, path) catch return error.EngramTableMissing;
            defer f.deinit();
            const loc = f.locate(leaf) orelse return error.EngramTableMissing;
            const t = loc.tensor;
            if (t.ty != .iq4_nl or t.n_dims != 2 or t.dims[0] != head_dim or t.dims[1] != rows) return error.EngramTableGeometry;
            const row = t.data.len / rows;
            const fd = try openAt(a, dir, name);
            return .{ .fd = fd, .rows = rows, .head_dim = head_dim, .mode = .gguf, .bits = 4, .group_size = 32, .w_off = loc.offset, .w_stride = row, .wlen = row, .s_off = 0, .s_stride = 0, .slen = 0 };
        }
        if (set.locate(leaf)) |loc| {
            const t = loc.tensor;
            const rec = head_dim + head_dim / 32;
            if (t.ty != ds41_gguf.engram_row_type or t.n_dims != 2 or t.dims[0] != rec or t.dims[1] != rows) return error.EngramTableGeometry;
            const fd = try openAt(a, std.fs.path.dirname(loc.path) orelse ".", std.fs.path.basename(loc.path));
            errdefer crt.close(fd);
            if (fileSize(fd) < loc.offset + rows * rec) return error.EngramTableSize;
            return mxfp8Records(fd, rows, head_dim, loc.offset);
        }
        const loc = set.locate(ds41_gguf.ple_table) orelse return error.EngramTableMissing;
        const t = loc.tensor;
        const start = try ds41_gguf.pleRowStart(set, index);
        if (t.ty != .iq4_nl or t.n_dims != 2 or t.dims[0] != head_dim or t.dims[1] < start + rows) return error.EngramTableGeometry;
        const row = t.data.len / t.dims[1];
        const fd = try openAt(a, std.fs.path.dirname(loc.path) orelse ".", std.fs.path.basename(loc.path));
        return .{ .fd = fd, .rows = rows, .head_dim = head_dim, .mode = .gguf, .bits = 4, .group_size = 32, .w_off = loc.offset + start * row, .w_stride = row, .wlen = row, .s_off = 0, .s_stride = 0, .slen = 0 };
    }

    /// named apart from the method `records`, the plugin's.
    /// `rows` mxfp8 records from `base`: `head_dim` e4m3 codes, then one e8m0 scale per 32.
    fn mxfp8Records(fd: c_int, rows: u64, head_dim: u32, base: u64) Table {
        const rec: u64 = head_dim + head_dim / 32;
        return .{ .fd = fd, .rows = rows, .head_dim = head_dim, .mode = .mxfp8, .bits = 8, .group_size = 32, .w_off = base, .w_stride = rec, .wlen = head_dim, .s_off = base + head_dim, .s_stride = rec, .slen = head_dim / 32 };
    }

    /// oMLX's lossless layout: `omlx_deepseek_v41.engram_tables` names the
    /// release's own shard, its table e4m3 with one e8m0 scale per 32 (mxfp8).
    fn openRawTable(a: std.mem.Allocator, dir: std.Io.Dir, dir_path: []const u8, layer: usize, rows: u64, head_dim: u32) !?Table {
        const io = std.Io.Threaded.global_single_threaded.io();
        const raw = dir.readFileAlloc(io, "config.json", a, .limited(64 << 20)) catch return null;
        const Spec = struct { weight_file: []const u8, weight_key: []const u8, scale_file: ?[]const u8 = null, scale_key: ?[]const u8 = null, bits: ?u32 = null };
        const cfg = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return null;
        const fmt = (if (cfg == .object) cfg.object.get("omlx_deepseek_v41") else null) orelse return null;
        const tables = (if (fmt == .object) fmt.object.get("engram_tables") else null) orelse return null;
        if (tables != .object) return null;
        const name = try std.fmt.allocPrint(a, "{s}layers.{d}.engram.embed", .{ weights_load.dsv41_text_prefix, layer });
        const v = tables.object.get(name) orelse return null;
        const spec = std.json.parseFromValueLeaky(Spec, a, v, .{ .ignore_unknown_fields = true }) catch return error.EngramTableHeader;
        if (spec.bits != null) return null; // a requantized table rides the index arm
        const scale_key = spec.scale_key orelse return error.EngramTableHeader;
        if (spec.scale_file) |sf| if (!std.mem.eql(u8, sf, spec.weight_file)) return error.EngramTableSplit;
        if (std.mem.indexOf(u8, spec.weight_file, "..") != null or std.fs.path.isAbsolute(spec.weight_file)) return error.EngramTableMissing;
        const fd = try openAt(a, dir_path, spec.weight_file);
        errdefer crt.close(fd);
        var hl: [8]u8 = undefined;
        if (crt.pread(fd, &hl, 8, 0) != 8) return error.EngramTableHeader;
        const hlen = std.mem.readInt(u64, &hl, .little);
        if (hlen > 16 << 20) return error.EngramTableHeader;
        const hbuf = try a.alloc(u8, @intCast(hlen));
        if (crt.pread(fd, hbuf.ptr, hbuf.len, 8) != @as(isize, @intCast(hlen))) return error.EngramTableHeader;
        const hdr = std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{}) catch return error.EngramTableHeader;
        if (hdr != .object) return error.EngramTableHeader;
        const size: usize = @intCast(fileSize(fd));
        const base = 8 + hlen;
        const w = qwen4_ple.headerRegion(hdr.object, spec.weight_key, "F8_E4M3", 1, size, @intCast(base), .matrix) catch return error.EngramTableHeader;
        const s = qwen4_ple.headerRegion(hdr.object, scale_key, "F8_E8M0", 1, size, @intCast(base), .matrix) catch return error.EngramTableHeader;
        if (w.rows != rows or s.rows != rows) return error.EngramTableSize;
        if (w.cols != head_dim or s.cols * 32 != head_dim) return error.EngramTableGeometry;
        return .{ .fd = fd, .rows = rows, .head_dim = head_dim, .mode = .mxfp8, .bits = 8, .group_size = 32, .w_off = base + w.start, .w_stride = head_dim, .wlen = head_dim, .s_off = base + s.start, .s_stride = s.cols, .slen = s.cols, .in_shards = billedShard(a, io, dir, spec.weight_file) };
    }

    /// Whether the preflight bills `file` (`preflight.modelDiskBytes`): a shard the index names, or with no index a
    /// top-level one.
    fn billedShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, file: []const u8) bool {
        const top = std.mem.indexOfScalar(u8, file, '/') == null;
        const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20)) catch return top;
        const idx = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return top;
        const wm = (if (idx == .object) idx.object.get("weight_map") else null) orelse return top;
        if (wm != .object) return top;
        for (wm.object.values()) |v| if (v == .string and std.mem.eql(u8, v.string, file)) return true;
        return false;
    }

    /// Rows `ids` as they sit on disk (host only, any thread).
    pub const Raw = struct {
        w: []align(4) u8,
        s: []align(4) u8,
        b: []align(4) u8,

        pub fn deinit(self: Raw, gpa: std.mem.Allocator) void {
            gpa.free(self.w);
            gpa.free(self.s);
            gpa.free(self.b);
        }
    };

    pub fn read(self: *const Table, gpa: std.mem.Allocator, ids: []const u32) !Raw {
        // A prompt's spans repeat rows (common n-grams): each distinct row is read once.
        if (ids.len > GCD_ROWS) if (try self.readDistinct(gpa, ids)) |r| return r;
        return self.readAll(gpa, ids);
    }

    /// `read` of the distinct ids, copied out to every position; null when no id repeats.
    fn readDistinct(self: *const Table, gpa: std.mem.Allocator, ids: []const u32) !?Raw {
        const n = ids.len;
        const order = try gpa.alloc(u64, n);
        defer gpa.free(order);
        for (order, ids, 0..) |*o, id, i| o.* = (@as(u64, id) << 32) | i;
        std.mem.sort(u64, order, {}, std.sort.asc(u64));
        const uniq = try gpa.alloc(u32, n);
        defer gpa.free(uniq);
        const slot = try gpa.alloc(u32, n);
        defer gpa.free(slot);
        var nu: usize = 0;
        for (order) |o| {
            const id: u32 = @intCast(o >> 32);
            if (nu == 0 or uniq[nu - 1] != id) {
                uniq[nu] = id;
                nu += 1;
            }
            slot[@as(u32, @truncate(o))] = @intCast(nu - 1);
        }
        if (nu == n) return null;
        const u = try self.readAll(gpa, uniq[0..nu]);
        defer u.deinit(gpa);
        const w = try gpa.alignedAlloc(u8, .@"4", n * self.wlen);
        errdefer gpa.free(w);
        const sc = try gpa.alignedAlloc(u8, .@"4", n * self.slen);
        errdefer gpa.free(sc);
        const bi = try gpa.alignedAlloc(u8, .@"4", if (self.mode == .affine) n * self.slen else 0);
        for (slot, 0..) |k, i| {
            @memcpy(w[i * self.wlen ..][0..self.wlen], u.w[k * self.wlen ..][0..self.wlen]);
            @memcpy(sc[i * self.slen ..][0..self.slen], u.s[k * self.slen ..][0..self.slen]);
            if (bi.len > 0) @memcpy(bi[i * self.slen ..][0..self.slen], u.b[k * self.slen ..][0..self.slen]);
        }
        return .{ .w = w, .s = sc, .b = bi };
    }

    /// The direct Engram reads: `ids` (sorted, distinct) read past the cache on up to `PAR_THREADS_DIRECT` threads,
    /// each a contiguous run of them: a row's codes through the blocks that hold them, its scales (and biases) from the
    /// block the run read last when it holds them, so neighbouring rows' scales cost one read. Rows in `ids`'s order.
    /// `cap`: the threads the read may run on (at most `GRID_THREADS`).
    fn readSorted(self: *const Table, gpa: std.mem.Allocator, ids: []const u32, cap: usize) !Raw {
        const n = ids.len;
        const w = try gpa.alignedAlloc(u8, .@"4", n * self.wlen);
        errdefer gpa.free(w);
        const sc = try gpa.alignedAlloc(u8, .@"4", n * self.slen);
        errdefer gpa.free(sc);
        const bi = try gpa.alignedAlloc(u8, .@"4", if (self.mode == .affine) n * self.slen else 0);
        errdefer gpa.free(bi);
        const Runs = struct {
            t: *const Table,
            ids: []const u32,
            w: []u8,
            s: []u8,
            b: []u8,
            bad: std.atomic.Value(bool) = .init(false),
            fn go(r: *@This(), first: usize, last: usize) void {
                var sb: edir.Block = .{};
                var bb: edir.Block = .{};
                const tb = r.t;
                const rec = tb.records();
                for (first..last) |i| {
                    const row: u64 = r.ids[i];
                    if (rec) {
                        // a repacked table: the row's record, its codes then its scales, in one read
                        if (row >= tb.rows or !edir.readRecord(tb.fd, r.w[i * tb.wlen ..][0..tb.wlen], r.s[i * tb.slen ..][0..tb.slen], tb.w_off + row * tb.w_stride)) {
                            r.bad.store(true, .release);
                            return;
                        }
                        continue;
                    }
                    if (row >= tb.rows or
                        !edir.readRow(tb.fd, r.w[i * tb.wlen ..][0..tb.wlen], tb.w_off + row * tb.w_stride) or
                        !sb.read(tb.fd, r.s[i * tb.slen ..][0..tb.slen], tb.s_off + row * tb.s_stride) or
                        (tb.mode == .affine and !bb.read(tb.fd, r.b[i * tb.slen ..][0..tb.slen], tb.b_off + row * tb.s_stride)))
                    {
                        r.bad.store(true, .release);
                        return;
                    }
                }
            }
        };
        var runs: Runs = .{ .t = self, .ids = ids, .w = w, .s = sc, .b = bi };
        const parts = std.math.clamp(n / PAR_ROWS_PER_THREAD, 1, @min(cap, @as(usize, GRID_THREADS)));
        var threads: [GRID_THREADS]?std.Thread = @splat(null);
        for (threads[1..parts], 1..) |*slot, k| slot.* = std.Thread.spawn(.{}, Runs.go, .{ &runs, k * n / parts, (k + 1) * n / parts }) catch null;
        runs.go(0, n / parts);
        for (threads[1..parts], 1..) |th, k| {
            if (th) |thread| thread.join() else runs.go(k * n / parts, (k + 1) * n / parts);
        }
        if (runs.bad.load(.acquire)) return error.EngramRead;
        return .{ .w = w, .s = sc, .b = bi };
    }
    fn readAll(self: *const Table, gpa: std.mem.Allocator, ids: []const u32) !Raw {
        const n = ids.len;
        const w = try gpa.alignedAlloc(u8, .@"4", n * self.wlen);
        errdefer gpa.free(w);
        const sc = try gpa.alignedAlloc(u8, .@"4", n * self.slen);
        errdefer gpa.free(sc);
        const bi = try gpa.alignedAlloc(u8, .@"4", if (self.mode == .affine) n * self.slen else 0);
        errdefer gpa.free(bi);
        var job: Job = .{ .t = self, .ids = ids, .w = w, .s = sc, .b = bi };
        if (builtin.os.tag.isDarwin() and n <= GCD_ROWS) {
            // a decode token's or verify's rows wait at the first Engram layer while the GPU idles: the
            // rows the LRU holds copied out of it, the others read on its pool.
            try self.readHeld(&job);
            return .{ .w = w, .s = sc, .b = bi };
        }
        // A DSpark verify's 6 x 24 rows wait at the first Engram layer: 16 rows per thread, this one included.
        // A direct read waits on the drive, not on a page fault: twice the threads.
        const parts = std.math.clamp(n / PAR_ROWS_PER_THREAD, 1, @as(usize, if (self.direct) PAR_THREADS_DIRECT else PAR_THREADS));
        if (parts > 1) {
            var threads: [PAR_THREADS_DIRECT]?std.Thread = @splat(null);
            for (threads[1..parts], 1..) |*t, k| t.* = std.Thread.spawn(.{}, Job.runVoid, .{ &job, k, parts }) catch null;
            var failed = !job.run(0, parts);
            for (threads[1..parts], 1..) |t, k| {
                if (t) |th| th.join() else failed = !job.run(k, parts) or failed;
            }
            if (failed or job.bad.load(.acquire)) return error.EngramRead;
        } else if (!job.run(0, 1)) return error.EngramRead;
        return .{ .w = w, .s = sc, .b = bi };
    }

    /// `job`'s rows (at most `GCD_ROWS`) the LRU holds copied out of it, the others read on the LRU's
    /// pool, each read (a repacked table's record, else a row's codes, scales or biases) a work item, then put in the LRU.
    fn readHeld(self: *const Table, job: *Job) !void {
        const n = job.ids.len;
        const bl: usize = if (self.mode == .affine) self.slen else 0;
        const parts: usize = if (self.records()) 1 else if (self.mode == .affine) 3 else 2;
        var held: [GCD_ROWS]bool = undefined;
        var pos: [GCD_ROWS]u32 = undefined;
        var np: usize = 0;
        if (edec.take(self.fd, job.ids, self.wlen, self.slen, bl, job.w, job.s, job.b, held[0..n]) == n) return;
        for (held[0..n], 0..) |h, i| if (!h) {
            pos[np] = @intCast(i);
            np += 1;
        };
        const rest: Job.Parts = .{ .job = job, .pos = pos[0..np], .parts = parts };
        edec.parallel(&rest, np * parts, Job.Parts.one);
        if (job.bad.load(.acquire)) return error.EngramRead;
        edec.putMisses(self.fd, job.ids, held[0..n], self.wlen, self.slen, bl, job.w, job.s, job.b);
    }
    /// `ids`' rows read on a GCD worker into the LRU, for the verify that reads them next
    /// (engram_decode.zig `early`, the table's own slot `slot`); returns at once.
    pub fn readEarly(self: *const Table, slot: usize, ids: []const u32) void {
        edec.early(slot, self, readInto, self.fd, self.wlen, self.slen, if (self.mode == .affine) self.slen else 0, ids);
    }
    /// `ids`' rows into `w`, `s` and `b` (`Table.read`'s layout), a row a GCD work item; false on a failed read.
    fn readInto(ctx: *const anyopaque, ids: []const u32, w: []u8, s: []u8, b: []u8) bool {
        const self: *const Table = @ptrCast(@alignCast(ctx));
        var job: Job = .{ .t = self, .ids = ids, .w = w, .s = s, .b = b };
        dispatch_apply_f(ids.len, dispatch_get_global_queue(qos_user_initiated, 0), &job, Job.applyOne);
        return !job.bad.load(.acquire);
    }
    /// The first `n` rows of `raw` dequantized: f32 `[n, head_dim]` (caller frees).
    pub fn dequant(self: *const Table, raw: Raw, n: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        if (self.mode == .gguf) {
            // IQ4_NL blocks in the kernels' split row layout.
            const kernels = gguf_root.kernels;
            const rb = kernels.rowBytes(.iq4_nl, self.head_dim);
            const tmp = try std.heap.page_allocator.alloc(u8, n * rb);
            defer std.heap.page_allocator.free(tmp);
            kernels.packRows(.iq4_nl, self.head_dim, tmp, raw.w[0 .. n * self.wlen]);
            const wa = mlx.mlx_array_new_data(tmp.ptr, &[_]c_int{ @intCast(n), @intCast(rb) }, 2, .uint8);
            defer _ = mlx.mlx_array_free(wa);
            return kernels.dequantize(.iq4_nl, wa, .float32, s);
        }
        const wcols: c_int = @intCast(self.wlen / 4);
        const scols: c_int = @intCast(if (self.mode == .affine) self.slen / 2 else self.slen);
        const rows: c_int = @intCast(n);
        const wa = mlx.mlx_array_new_data(raw.w.ptr, &[_]c_int{ rows, wcols }, 2, .uint32);
        defer _ = mlx.mlx_array_free(wa);
        const sdt: mlx.mlx_dtype = if (self.mode == .affine) .bfloat16 else .uint8;
        const sa = mlx.mlx_array_new_data(raw.s.ptr, &[_]c_int{ rows, scols }, 2, sdt);
        defer _ = mlx.mlx_array_free(sa);
        const ba: mlx.mlx_array = if (self.mode == .affine) mlx.mlx_array_new_data(raw.b.ptr, &[_]c_int{ rows, scols }, 2, .bfloat16) else .{ .ctx = null };
        defer if (ba.ctx != null) {
            _ = mlx.mlx_array_free(ba);
        };
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_dequantize(&out, wa, sa, ba, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), self.mode.cstr(), .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        return out;
    }

    /// Rows `ids`, dequantized: f32 `[ids.len, head_dim]` (caller frees).
    pub fn gather(self: *const Table, gpa: std.mem.Allocator, ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
        const raw = try self.read(gpa, ids);
        defer raw.deinit(gpa);
        return self.dequant(raw, ids.len, s);
    }

    /// Asks the kernel to read every `stride`-th of `ids` from `first` into the page cache now
    /// (`F_RDADVISE`: asynchronous, no copy), so the preads behind it find the rows cached or in
    /// flight and a batch of random rows costs about one SSD latency, not one per row. An
    /// out-of-range id is left for `read` to refuse.
    fn advise(self: *const Table, ids: []const u32, first: usize, stride: usize) void {
        if (comptime !builtin.os.tag.isDarwin()) return;
        var i = first;
        while (i < ids.len) : (i += stride) {
            const r: u64 = ids[i];
            if (r >= self.rows) continue;
            rdadvise(self.fd, self.w_off + r * self.w_stride, self.wlen);
            if (self.slen > 0) rdadvise(self.fd, self.s_off + r * self.s_stride, self.slen);
            if (self.mode == .affine) rdadvise(self.fd, self.b_off + r * self.s_stride, self.slen);
        }
    }

    const Job = struct {
        t: *const Table,
        ids: []const u32,
        w: []u8,
        s: []u8,
        b: []u8,
        bad: std.atomic.Value(bool) = .init(false),

        fn applyOne(ctx: ?*anyopaque, i: usize) callconv(.c) void {
            const job: *Job = @ptrCast(@alignCast(ctx.?));
            _ = job.run(i, job.ids.len);
        }

        fn runVoid(self: *Job, first: usize, stride: usize) void {
            _ = self.run(first, stride);
        }
        /// one read of the job's rows at `pos`, a work item of the LRU's pool: item `i` is
        /// row `pos[i / parts]`'s record (a repacked table, `parts` 1), else its codes, scales or biases (`i % parts`).
        const Parts = struct {
            job: *Job,
            pos: []const u32,
            parts: usize,
            fn one(ctx: *const anyopaque, i: usize) void {
                const p: *const Parts = @ptrCast(@alignCast(ctx));
                const j = p.job;
                const t = j.t;
                const k: usize = p.pos[i / p.parts];
                const r: u64 = j.ids[k];
                const part = i % p.parts;
                if (!(r < t.rows and readPart(t, j, k, r, part))) j.bad.store(true, .release);
            }
            /// Row `r`'s read `part` into the job's row `k`: a repacked table's record (`records`), its codes then its
            /// scales, in one read; else its codes, scales or biases.
            fn readPart(t: *const Table, j: *const Job, k: usize, r: u64, part: usize) bool {
                if (t.records()) return edir.readRecord(t.fd, j.w[k * t.wlen ..][0..t.wlen], j.s[k * t.slen ..][0..t.slen], t.w_off + r * t.w_stride);
                return switch (part) {
                    0 => edir.readRow(t.wfd(), j.w[k * t.wlen ..][0..t.wlen], t.w_off + r * t.w_stride),
                    1 => edir.readRow(t.fd, j.s[k * t.slen ..][0..t.slen], t.s_off + r * t.s_stride),
                    else => edir.readRow(t.fd, j.b[k * t.slen ..][0..t.slen], t.b_off + r * t.s_stride),
                };
            }
        };
        /// Every `stride`-th row from `first`, read past the page cache: preads run in parallel where faults on one
        /// mapping would serialize.
        fn run(self: *Job, first: usize, stride: usize) bool {

            const t = self.t;
            const rec = t.records();
            var i = first;
            while (i < self.ids.len) : (i += stride) {
                const r: u64 = self.ids[i];
                if (rec) {
                    // a repacked table: the row's record, its codes then its scales, in one read
                    if (r >= t.rows or !edir.readRecord(t.fd, self.w[i * t.wlen ..][0..t.wlen], self.s[i * t.slen ..][0..t.slen], t.w_off + r * t.w_stride)) {
                        self.bad.store(true, .release);
                        return false;
                    }
                    continue;
                }
                if (r >= t.rows or
                    !edir.readRow(t.wfd(), self.w[i * t.wlen ..][0..t.wlen], t.w_off + r * t.w_stride) or
                    !edir.readRow(t.fd, self.s[i * t.slen ..][0..t.slen], t.s_off + r * t.s_stride) or
                    (t.mode == .affine and !edir.readRow(t.fd, self.b[i * t.slen ..][0..t.slen], t.b_off + r * t.s_stride)))
                {
                    self.bad.store(true, .release);
                    return false;
                }
            }
            return true;
        }
    };
};

/// Whether the page cache has no room for a prompt's Engram codes: under an eighth of RAM free or reclaimable, a
/// chunk's code pages (16 KiB a row read) push the tables' scales out of it.
pub fn cacheTight(available: u64, total: u64) bool {
    return available < total / 8;
}

/// Engram rows read ahead on a thread of its own: per grid (a run of `len` tokens from position `at`),
/// layer by layer and span by span, the order the forward asks for them, so the SSD reads hide behind
/// the GPU work of the spans, layers and chunks before.
pub const Prefetch = struct {
    gpa: std.mem.Allocator,
    tables: []const *const Table,
    grids: []Grid,
    span: usize,
    nc: usize,
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// A batch's grids (one a state, `startBatch`) read a layer at a time across all of them, as its forward asks.
    across: bool = false,
    /// What `take` spent waiting for rows the drive had not handed over yet: the forward's time at an Engram layer
    /// with nothing for the GPU to do.
    waited_ms: i64 = 0,

    /// `ids` is `[len][tables.len][nc]` row ids (`Hash.rows`).
    pub const Spec = struct { at: usize, len: usize, ids: []const u32 };

    const Grid = struct {
        at: usize,
        len: usize,
        ids: []u32,
        items: []Item,
    };
    const Item = struct {
        raw: ?Table.Raw = null,
        state: std.atomic.Value(u8) = .init(pending),
    };
    const pending = 0;
    const done = 1;
    const failed = 2;

    /// Reads `specs` in order. Copies what it is given; the returned value must not move (the thread holds it).
    pub fn start(gpa: std.mem.Allocator, tables: []const *const Table, specs: []const Spec, nc: usize, span: usize) !*Prefetch {
        return startOrdered(gpa, tables, specs, nc, span, false);
    }

    /// `start` for a batch of states, one grid each: every grid's rows of a layer before the next layer's.
    pub fn startBatch(gpa: std.mem.Allocator, tables: []const *const Table, specs: []const Spec, nc: usize, span: usize) !*Prefetch {
        return startOrdered(gpa, tables, specs, nc, span, true);
    }

    fn startOrdered(gpa: std.mem.Allocator, tables: []const *const Table, specs: []const Spec, nc: usize, span: usize, across: bool) !*Prefetch {
        const self = try gpa.create(Prefetch);
        errdefer gpa.destroy(self);
        const own_tables = try gpa.dupe(*const Table, tables);
        errdefer gpa.free(own_tables);
        const grids = try gpa.alloc(Grid, specs.len);
        var made: usize = 0;
        errdefer {
            for (grids[0..made]) |g| freeGrid(gpa, g);
            gpa.free(grids);
        }
        for (specs, grids) |sp, *g| {
            const ids = try gpa.dupe(u32, sp.ids);
            errdefer gpa.free(ids);
            const items = try gpa.alloc(Item, tables.len * ((sp.len + span - 1) / span));
            @memset(items, .{});
            g.* = .{ .at = sp.at, .len = sp.len, .ids = ids, .items = items };
            made += 1;
        }
        self.* = .{ .gpa = gpa, .tables = own_tables, .grids = grids, .span = span, .nc = nc, .across = across };
        self.thread = try std.Thread.spawn(.{}, run, .{self});
        return self;
    }

    fn freeGrid(gpa: std.mem.Allocator, g: Grid) void {
        for (g.items) |*item| if (item.raw) |r| r.deinit(gpa);
        gpa.free(g.items);
        gpa.free(g.ids);
    }

    fn run(self: *Prefetch) void {
        if (self.across) {
            for (0..self.tables.len) |layer| for (self.grids) |*g| if (!self.readGrid(g, layer)) return;
        } else for (self.grids) |*g| for (0..self.tables.len) |layer| if (!self.readGrid(g, layer)) return;
    }

    /// Every span of `layer` in `g`; false once stopped. A prompt grid's table read whole (its distinct rows once, past
    /// the cache); a decode token's span by span.
    fn readGrid(self: *Prefetch, g: *Grid, layer: usize) bool {
        const n_spans = g.items.len / self.tables.len;
        if (self.stop.load(.acquire)) return false;
        if (g.len >= grid_rows and self.tables[layer].direct) {
            self.readGridTable(g, self.tables[layer], layer, n_spans);
            return true;
        }
        for (0..n_spans) |k| {
            if (self.stop.load(.acquire)) return false;
            const item = &g.items[layer * n_spans + k];
            item.raw = self.readSpan(g, self.tables[layer], layer, k) catch null;
            item.state.store(if (item.raw != null) done else failed, .release);
        }
        return true;
    }

    /// The direct Engram reads: table `layer`'s rows of the whole grid: its ids sorted and made distinct, read once
    /// (`Table.readSorted`), then each span's rows copied out of them; every span of the table done together (the layer
    /// reads them all before its eval). A failed read fails every span of the table, as a failed span read fails that
    /// span.
    fn readGridTable(self: *Prefetch, g: *Grid, table: *const Table, layer: usize, n_spans: usize) void {
        const nc = self.nc;
        const nt = self.tables.len;
        const all = self.gpa.alloc(u32, g.len * nc) catch return failTable(g, layer, n_spans);
        defer self.gpa.free(all);
        for (0..g.len) |tok| @memcpy(all[tok * nc ..][0..nc], g.ids[(tok * nt + layer) * nc ..][0..nc]);
        const uniq = self.gpa.dupe(u32, all) catch return failTable(g, layer, n_spans);
        defer self.gpa.free(uniq);
        std.mem.sort(u32, uniq, {}, std.sort.asc(u32));
        var nu: usize = 0;
        for (0..uniq.len) |i| {
            if (nu == 0 or uniq[nu - 1] != uniq[i]) {
                uniq[nu] = uniq[i];
                nu += 1;
            }
        }
        // the table on up to `GRID_THREADS` threads.
        const u = table.readSorted(self.gpa, uniq[0..nu], GRID_THREADS) catch return failTable(g, layer, n_spans);
        defer u.deinit(self.gpa);
        for (0..n_spans) |k| {
            const lo = k * self.span;
            const hi = @min(g.len, lo + self.span);
            const item = &g.items[layer * n_spans + k];
            item.raw = spanRows(self.gpa, table, uniq[0..nu], u, all[lo * nc .. hi * nc]) catch null;
            item.state.store(if (item.raw != null) done else failed, .release);
        }
        // the grid's distinct rows into the LRU, once its spans are handed over.
        edec.keep(table.fd, table.wlen, table.slen, if (table.mode == .affine) table.slen else 0, uniq[0..nu], u.w, u.s, u.b);
    }
    fn failTable(g: *Grid, layer: usize, n_spans: usize) void {
        for (0..n_spans) |k| g.items[layer * n_spans + k].state.store(failed, .release);
    }
    /// The rows of `ids` (each one of `uniq`, sorted) copied out of `u` (`uniq`'s rows in order): a new `Raw` the
    /// caller frees, laid out as `Table.read` lays out `ids`'s.
    fn spanRows(gpa: std.mem.Allocator, table: *const Table, uniq: []const u32, u: Table.Raw, ids: []const u32) !Table.Raw {
        const n = ids.len;
        const w = try gpa.alignedAlloc(u8, .@"4", n * table.wlen);
        errdefer gpa.free(w);
        const sc = try gpa.alignedAlloc(u8, .@"4", n * table.slen);
        errdefer gpa.free(sc);
        const bi = try gpa.alignedAlloc(u8, .@"4", if (table.mode == .affine) n * table.slen else 0);
        errdefer gpa.free(bi);
        for (ids, 0..) |id, i| {
            var lo: usize = 0;
            var hi: usize = uniq.len;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (uniq[mid] < id) lo = mid + 1 else hi = mid;
            }
            if (lo == uniq.len or uniq[lo] != id) return error.EngramRead;
            @memcpy(w[i * table.wlen ..][0..table.wlen], u.w[lo * table.wlen ..][0..table.wlen]);
            @memcpy(sc[i * table.slen ..][0..table.slen], u.s[lo * table.slen ..][0..table.slen]);
            if (bi.len > 0) @memcpy(bi[i * table.slen ..][0..table.slen], u.b[lo * table.slen ..][0..table.slen]);
        }
        return .{ .w = w, .s = sc, .b = bi };
    }
    fn readSpan(self: *Prefetch, g: *const Grid, table: *const Table, layer: usize, k: usize) !Table.Raw {
        const lo = k * self.span;
        const hi = @min(g.len, lo + self.span);
        const mine = try self.gpa.alloc(u32, (hi - lo) * self.nc);
        defer self.gpa.free(mine);
        for (lo..hi) |t| @memcpy(mine[(t - lo) * self.nc ..][0..self.nc], g.ids[(t * self.tables.len + layer) * self.nc ..][0..self.nc]);
        return table.read(self.gpa, mine);
    }

    /// Does a grid start at `at` and cover `len` tokens?
    pub fn covers(self: *const Prefetch, at: usize, len: usize) bool {
        for (self.grids) |g| if (g.at == at and g.len >= len) return true;
        return false;
    }

    /// The rows of `layer`'s span `k` of the grid at `at`, once read (the caller frees them).
    pub fn take(self: *Prefetch, at: usize, layer: usize, k: usize) !Table.Raw {
        for (self.grids) |*g| if (g.at == at) {
            const item = &g.items[layer * (g.items.len / self.tables.len) + k];
            if (item.state.load(.acquire) == pending) {
                const t0 = crt.monotonicMs();
                while (item.state.load(.acquire) == pending) std.Thread.yield() catch {};
                self.waited_ms += crt.monotonicMs() - t0;
            }
            const raw = item.raw orelse return error.EngramRead;
            item.raw = null;
            return raw;
        };
        return error.EngramRead;
    }

    /// Stops the thread, frees what was never taken and the prefetch itself.
    pub fn finish(self: *Prefetch) void {
        self.stop.store(true, .release);
        if (self.thread) |t| t.join();
        for (self.grids) |g| freeGrid(self.gpa, g);
        self.gpa.free(self.grids);
        self.gpa.free(self.tables);
        self.gpa.destroy(self);
    }
};

/// The most rows read a row a GCD work item (`Table.read`).
const GCD_ROWS = 512;
const qos_user_initiated: isize = 0x19;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;
extern "c" fn dispatch_apply_f(iterations: usize, queue: ?*anyopaque, context: ?*anyopaque, work: *const fn (?*anyopaque, usize) callconv(.c) void) void;

const PAR_ROWS_PER_THREAD = 16;
const PAR_THREADS = 32;

/// The direct Engram reads: a prompt span's read threads when the reads go past the cache.
const PAR_THREADS_DIRECT = 64;
/// a prompt grid's table read's threads (the stock: 64), a deeper queue for the drive (a machine
/// parameter, not the cell's). The table's distinct rows, about 155K on a cold 16K card at about 4 KB each, one direct
/// pread a row; the first Engram layer otherwise waits on table 0 with the GPU idle. Exact: the same rows into the same
/// buffers; only how many threads read them.
const GRID_THREADS = 128;
comptime {
    if (GRID_THREADS < PAR_THREADS_DIRECT) @compileError("GRID_THREADS below the stock 64");
}
/// The direct Engram reads: a grid of at least this many tokens (a prompt's) reads each table whole.
const grid_rows = 1024;
fn preadAll(fd: c_int, dst: []u8, off: u64) bool {
    return crt.pread(fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
}

/// macOS `struct radvisory`.
const Radvisory = extern struct { ra_offset: i64, ra_count: c_int };

fn rdadvise(fd: c_int, off: u64, len: usize) void {
    var ra: Radvisory = .{ .ra_offset = @intCast(off), .ra_count = @intCast(len) };
    _ = std.c.fcntl(fd, std.c.F.RDADVISE, &ra);
}

fn fileSize(fd: c_int) u64 {
    return @intCast(@max(crt.seekEnd(fd), 0));
}

fn openAt(a: std.mem.Allocator, dir_path: []const u8, name: []const u8) !c_int {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, name }, 0);
    const fd = crt.openRead(path);
    if (fd < 0) return error.FileNotFound;
    // Rows are read at random: read-ahead would pull their neighbours in.
    if (comptime builtin.os.tag.isDarwin()) _ = std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0));
    return fd;
}

const testing = std.testing;

fn releaseConfig() Config {
    var cfg: Config = .{};
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

test "dsv41 engram: a dead token (an image span's) makes every n-gram across it the all-pad one" {
    const cfg = releaseConfig();
    const h = try Hash.init(&cfg, 2);
    // ... 3 [image] 77: the image's own row and 77's shifts past it read pad.
    const ids = [_]u32{ 3, Hash.DEAD, 77 };
    var got: [3 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &ids, &got);
    var want: [3 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &.{2}, want[48..96]); // the image's own row: every slot pad
    try testing.expectEqualSlices(u32, want[48..96], got[48..96]);
    h.rows(&.{ 2, 2 }, &.{77}, want[96..]); // 77 with only pad behind it
    try testing.expectEqualSlices(u32, want[96..], got[96..]);
    // The token before the image hashes as if nothing followed it (no look-ahead).
    h.rows(&.{}, &.{3}, want[0..48]);
    try testing.expectEqualSlices(u32, want[0..48], got[0..48]);
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
    try writeAffineFixture(io, td.dir, weights_load.dsv41_text_prefix);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 0, 1, 4, 64);
    defer t.close();
    try testing.expectEqual(@as(u64, 288), t.shardBytes());
}

/// oMLX's lossless layout: 4 rows of head_dim 64 as raw e4m3 codes (1.0) with row r's first scale 2^r, its second 0.5.
fn writeRawFp8Fixture(io: std.Io, dir: std.Io.Dir) !void {
    return writeRawFp8FixtureAt(io, dir, "engram/model-00002-of-00002.safetensors");
}

/// `writeRawFp8Fixture` with the table in `file`; a top-level `file` is a shard the index names.
fn writeRawFp8FixtureAt(io: std.Io, dir: std.Io.Dir, comptime file: []const u8) !void {
    const hdr = "{\"layers.1.engram.embed.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[4,64],\"data_offsets\":[0,256]}," ++
        "\"layers.1.engram.embed.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[4,2],\"data_offsets\":[256,264]}}";
    var shard: [8 + hdr.len + 264]u8 = undefined;
    std.mem.writeInt(u64, shard[0..8], hdr.len, .little);
    @memcpy(shard[8..][0..hdr.len], hdr);
    const data = shard[8 + hdr.len ..];
    @memset(data[0..256], 0x38); // e4m3 1.0
    for (0..4) |r| {
        data[256 + 2 * r] = @intCast(127 + r); // 2^r for the first 32 columns
        data[256 + 2 * r + 1] = 126; // 0.5 for the rest
    }
    const top = comptime std.mem.indexOfScalar(u8, file, '/') == null;
    if (!top) try dir.createDirPath(io, std.fs.path.dirname(file).?);
    try dir.writeFile(io, .{ .sub_path = file, .data = &shard });
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"omlx_deepseek_v41\":{\"version\":1,\"engram_tables\":{" ++
        "\"language_model.layers.1.engram.embed\":{\"weight_key\":\"layers.1.engram.embed.weight\",\"weight_file\":\"" ++ file ++ "\"," ++
        "\"scale_key\":\"layers.1.engram.embed.scale\",\"scale_file\":\"" ++ file ++ "\",\"bias_key\":null,\"bits\":null,\"group_size\":null}}}}" });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = if (top)
        "{\"weight_map\":{\"layers.1.engram.embed.weight\":\"" ++ file ++ "\",\"layers.1.engram.embed.scale\":\"" ++ file ++ "\"}}"
    else
        "{\"weight_map\":{}}" });
}

test "dsv41 engram: oMLX's raw fp8 table inside a shard the index names bills its bytes there" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeRawFp8FixtureAt(io, td.dir, "model-00002-of-00002.safetensors");
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 0, 1, 4, 64);
    defer t.close();
    try testing.expectEqual(@as(u64, 4 * 64 + 4 * 2), t.shardBytes());
}

test "dsv41 engram: oMLX's lossless pack reads the release's raw fp8 table from the shard its config names" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeRawFp8Fixture(io, td.dir);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 0, 1, 4, 64);
    defer t.close();
    try testing.expectEqual(@as(u64, 0), t.shardBytes());
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const rows = try t.gather(testing.allocator, &.{ 3, 0 }, s);
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_array_eval(rows));
    const v = mlx.mlx_array_data_float32(rows).?;
    try testing.expectEqual(@as(f32, 8), v[0]);
    try testing.expectEqual(@as(f32, 0.5), v[63]);
    try testing.expectEqual(@as(f32, 1), v[64]);
    try testing.expectEqual(@as(f32, 0.5), v[64 + 32]);
}

/// The DS4 layout's GGUF: an F32 tensor, then layer 1's table as I8 [66, 4] records of head_dim 64 holding
/// `writeRawFp8Fixture`'s rows (codes 1.0, row r's first scale 2^r, its second 0.5).
fn writeDs4Fixture(io: std.Io, dir: std.Io.Dir) !void {
    const a = testing.allocator;
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(a);
    try b.appendSlice(a, "GGUF" ++ std.mem.toBytes(@as(u32, 3)) ++ std.mem.toBytes(@as(u64, 2)) ++ std.mem.toBytes(@as(u64, 0)));
    const tensors = [_]struct { []const u8, []const u64, u32, u64 }{
        .{ "x", &.{8}, 0, 0 },
        .{ "blk.1." ++ ds41_gguf.engram_leaf, &.{ 66, 4 }, @intFromEnum(ds41_gguf.engram_row_type), 32 },
    };
    for (tensors) |t| {
        try b.appendSlice(a, &std.mem.toBytes(@as(u64, t[0].len)));
        try b.appendSlice(a, t[0]);
        try b.appendSlice(a, &std.mem.toBytes(@as(u32, @intCast(t[1].len))));
        for (t[1]) |d| try b.appendSlice(a, &std.mem.toBytes(d));
        try b.appendSlice(a, &std.mem.toBytes(t[2]));
        try b.appendSlice(a, &std.mem.toBytes(t[3]));
    }
    while (b.items.len % 32 != 0) try b.append(a, 0);
    try b.appendNTimes(a, 0, 32);
    for (0..4) |r| {
        try b.appendNTimes(a, 0x38, 64);
        try b.appendSlice(a, &.{ @intCast(127 + r), 126 });
    }
    try dir.writeFile(io, .{ .sub_path = "m.gguf", .data = b.items });
}

test "dsv41 engram: the DS4 layout's I8 table reads the release's fp8 records in place; another geometry is refused" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeDs4Fixture(io, td.dir);
    var buf: [512]u8 = undefined;
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/m.gguf", .{buf[0..try td.dir.realPath(io, &buf)]});
    defer testing.allocator.free(path);
    var set = try gguf_root.gguf.File.open(testing.allocator, path);
    defer set.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t = try Table.ggufTable(arena.allocator(), &set, 0, 1, 4, 64);
    defer t.close();
    try testing.expectEqual(@as(u64, 0), t.shardBytes());
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const rows = try t.gather(testing.allocator, &.{ 3, 0 }, s);
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_array_eval(rows));
    const v = mlx.mlx_array_data_float32(rows).?;
    try testing.expectEqual(@as(f32, 8), v[0]);
    try testing.expectEqual(@as(f32, 0.5), v[63]);
    try testing.expectEqual(@as(f32, 1), v[64]);
    try testing.expectEqual(@as(f32, 0.5), v[64 + 32]);
    try testing.expectError(error.EngramTableGeometry, Table.ggufTable(arena.allocator(), &set, 0, 1, 5, 64));
    try testing.expectError(error.EngramTableGeometry, Table.ggufTable(arena.allocator(), &set, 0, 1, 4, 32));
}

test "dsv41 engram: a DS4 GGUF's hash constants are the ones the plugin derives, its rows read in place (a real GGUF, env-gated)" {
    const path = std.mem.span(@import("../../env.zig").get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    var set = try gguf_root.gguf.File.open(testing.allocator, path);
    defer set.deinit();
    if (set.getString("deepseek41.engram.encoding") == null) return error.SkipZigTest;
    const hp = try ds41_gguf.Hparams.read(&set);
    var cfg = releaseConfig();
    cfg.dsv41_n_engram_layers = hp.n_engram;
    cfg.dsv41_engram_max_ngram = hp.engram_ngram;
    cfg.dsv41_engram_n_heads = hp.engram_heads;
    cfg.dsv41_engram_vocab_size = hp.engram_vocab;
    cfg.dsv41_engram_compressed_vocab = hp.engram_compressed;
    for (0..hp.n_engram) |k| {
        cfg.dsv41_engram_layers[k] = @intCast(hp.engram_layers[k]);
        cfg.dsv41_engram_rows[k] = hp.engram_rows[k];
    }
    const map = try loadTokenMap(testing.allocator, path, hp.vocab, hp.engram_compressed);
    defer testing.allocator.free(map);
    const file_map = set.getArray("deepseek41.engram.token_map").?;
    try testing.expectEqual(map.len, file_map.len);
    for (map, 0..) |v, i| try testing.expectEqual(@as(i64, v), file_map.int(i));
    const h = try Hash.init(&cfg, map[hp.engram_pad]);
    const primes = set.getArray("deepseek41.engram.primes").?;
    const mults = set.getArray("deepseek41.engram.multipliers").?;
    for (0..hp.n_engram) |l| {
        for (0..h.cols()) |c| try testing.expectEqual(primes.int(l * h.cols() + c), @as(i64, @intCast(h.primes[l][c])));
        for (0..h.max_ngram) |k| try testing.expectEqual(mults.int(l * h.max_ngram + k), @as(i64, @intCast(h.mult[l][k])));
    }

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for (0..hp.n_engram) |k| {
        var t = try Table.ggufTable(arena.allocator(), &set, k, hp.engram_layers[k], hp.engram_rows[k], hp.engram_dim);
        defer t.close();
        const rows = try t.gather(testing.allocator, &.{ 0, @intCast(hp.engram_rows[k] / 2), @intCast(hp.engram_rows[k] - 1) }, s);
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_array_eval(rows));
        var sq: f64 = 0;
        for (mlx.mlx_array_data_float32(rows).?[0 .. 3 * hp.engram_dim]) |v| {
            try testing.expect(std.math.isFinite(v));
            sq += v * v;
        }
        try testing.expect(sq > 0);
    }
}

test "dsv41 engram: rows asked of the kernel ahead read as a plain gather does; an out-of-range id is skipped there and refused by the read" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeRawFp8Fixture(io, td.dir);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 0, 1, 4, 64);
    defer t.close();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const ids = [_]u32{ 3, 1, 3, 0, 7 };
    t.advise(&ids, 0, 1);
    try testing.expectError(error.EngramRead, t.read(testing.allocator, &ids));
    const rows = try t.gather(testing.allocator, ids[0..4], s);
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_array_eval(rows));
    const v = mlx.mlx_array_data_float32(rows).?;
    for (ids[0..4], 0..) |id, i| {
        try testing.expectEqual(@as(f32, @floatFromInt(@as(u32, 1) << @intCast(id))), v[i * 64]);
        try testing.expectEqual(@as(f32, 0.5), v[i * 64 + 63]);
    }
}

test "dsv41 engram: a read split over threads lands every row where one thread puts it, and still refuses a bad id" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeRawFp8Fixture(io, td.dir);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 0, 1, 4, 64);
    defer t.close();
    // A 6-row verify's 144 rows of one table.
    var ids: [144]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 7 + i / 5) % 4);
    const raw = try t.read(testing.allocator, &ids);
    defer raw.deinit(testing.allocator);
    for (ids, 0..) |id, i| {
        try testing.expectEqual(@as(u8, @intCast(127 + id)), raw.s[2 * i]);
        try testing.expectEqual(@as(u8, 126), raw.s[2 * i + 1]);
        try testing.expect(std.mem.allEqual(u8, raw.w[64 * i ..][0..64], 0x38));
    }
    ids[100] = 7;
    try testing.expectError(error.EngramRead, t.read(testing.allocator, &ids));
}

test "dsv41 engram: a prefetch hands each span's rows as a gather of the same ids would" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeRawFp8Fixture(io, td.dir);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 0, 1, 4, 64);
    defer t.close();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // 5 tokens, 2 layers, 2 columns: ids [(token * 2 + layer) * 2 + column]; spans of 2 leave a ragged last one.
    var ids: [5 * 2 * 2]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 3 + 1) % 4);
    const tables = [_]*const Table{ &t, &t };
    // The second grid starts at token 5 and reads ids[0..8] again: a lookahead behind the first.
    var pf = try Prefetch.start(testing.allocator, &tables, &.{ .{ .at = 0, .len = 5, .ids = &ids }, .{ .at = 5, .len = 2, .ids = ids[0..8] } }, 2, 2);
    defer pf.finish();
    try testing.expect(pf.covers(5, 2) and pf.covers(0, 3) and !pf.covers(0, 6) and !pf.covers(4, 1));
    for (0..2) |layer| for (0..3) |k| {
        const lo = k * 2;
        const hi = @min(5, lo + 2);
        var mine: [4]u32 = undefined;
        for (lo..hi) |tok| @memcpy(mine[(tok - lo) * 2 ..][0..2], ids[(tok * 2 + layer) * 2 ..][0..2]);
        const want = try t.gather(testing.allocator, mine[0 .. (hi - lo) * 2], s);
        defer _ = mlx.mlx_array_free(want);
        const raw = try pf.take(0, layer, k);
        defer raw.deinit(testing.allocator);
        const got = try t.dequant(raw, (hi - lo) * 2, s);
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(want));
        try mlx.check(mlx.mlx_array_eval(got));
        const n = (hi - lo) * 2 * 64;
        try testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(want).?[0..n], mlx.mlx_array_data_float32(got).?[0..n]);
    };
    // The lookahead grid is the same rows as a gather of its ids; the grid that was not asked for is freed unread.
    const ahead = try pf.take(5, 1, 0);
    defer ahead.deinit(testing.allocator);
    var mine: [4]u32 = undefined;
    for (0..2) |tok| @memcpy(mine[tok * 2 ..][0..2], ids[(tok * 2 + 1) * 2 ..][0..2]);
    const want = try t.gather(testing.allocator, &mine, s);
    defer _ = mlx.mlx_array_free(want);
    const got = try t.dequant(ahead, 4, s);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(want).?[0 .. 4 * 64], mlx.mlx_array_data_float32(got).?[0 .. 4 * 64]);
}

test "dsv41 engram: codes read past the page cache are the rows read through it" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeRawFp8Fixture(io, td.dir);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    var plain = try Table.open(testing.allocator, dir, 0, 1, 4, 64);
    defer plain.close();
    const ids = [_]u32{ 3, 0, 2, 2, 1 };
    const through = try plain.read(testing.allocator, &ids);
    defer through.deinit(testing.allocator);
    // A table of its own: the rows the first one read sit in the LRU under that one's descriptor.
    var t = try Table.open(testing.allocator, dir, 0, 1, 4, 64);
    defer t.close();
    // Without its own descriptor the table stays on the cache whatever it is told.
    t.pastCache(true);
    try testing.expectEqual(t.fd, t.wfd());
    t.openCodes();
    try testing.expect(t.cfd >= 0 and t.cfd != t.fd);
    try testing.expectEqual(t.fd, t.wfd());
    t.pastCache(true);
    try testing.expectEqual(t.cfd, t.wfd());
    const past = try t.read(testing.allocator, &ids);
    defer past.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, through.w, past.w);
    try testing.expectEqualSlices(u8, through.s, past.s);
    t.pastCache(false);
    try testing.expectEqual(t.fd, t.wfd());
    try testing.expect(cacheTight(24 << 30, 256 << 30) and !cacheTight(130 << 30, 256 << 30));
}

test "dsv41 engram: a prompt grid read whole past the cache (a streamed bank's table) hands each span the span read's rows" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeRawFp8Fixture(io, td.dir);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    var plain = try Table.open(testing.allocator, dir, 0, 1, 4, 64);
    defer plain.close();
    var direct = try Table.open(testing.allocator, dir, 0, 1, 4, 64);
    defer direct.close();
    direct.readDirect();
    try testing.expect(!plain.direct and direct.direct);
    // One layer, two columns, a grid of `grid_rows` tokens in spans of 256.
    const ids = try testing.allocator.alloc(u32, grid_rows * 2);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 7 + 3) % 4);
    var pfs: [2]*Prefetch = undefined;
    for (&pfs, [_]*const Table{ &plain, &direct }) |*pf, t| pf.* = try Prefetch.start(testing.allocator, &.{t}, &.{.{ .at = 0, .len = grid_rows, .ids = ids }}, 2, 256);
    defer for (pfs) |pf| pf.finish();
    for (0..grid_rows / 256) |k| {
        const a = try pfs[0].take(0, 0, k);
        defer a.deinit(testing.allocator);
        const b = try pfs[1].take(0, 0, k);
        defer b.deinit(testing.allocator);
        try testing.expectEqualSlices(u8, a.w, b.w);
        try testing.expectEqualSlices(u8, a.s, b.s);
    }
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
