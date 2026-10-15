//! A Qwen3.5-family target built for the Neural Engine by ANEMLL Forge (`qwen38_coreai_build.py`),
//! run through Core AI with no MLX: the host side of Forge's `CoreAIQwenBridge`. A bundle is
//! `<dir>/model` (config, tokenizer, `embed_tokens_fp16.npy`) + `<dir>/coreai` (manifest.json,
//! 4-layer `chunk_*.aimodel` packages, `head_T8.aimodel`).
//!
//! Each package has one entry per context: `v8_<ctx>k` runs T (8) rows (decode), `p64_<ctx>k` runs
//! TP (64) prompt rows. A call runs every chunk then the head in one blocking bridge call. DeltaNet
//! states ping-pong between two buffer sets per chunk (a call reads set `par` and writes the other);
//! chunk k's `y` buffer is chunk k+1's `x`. The host owns the KV caches: accepted rows are copied out
//! of each call's `k<j>_new` / `v<j>_new` (V8: INT8 values with an FP16 scale per token and head).

const std = @import("std");
const ffi = @import("../coreai_ffi.zig");
const log = @import("../log.zig");
const dflash = @import("coreai_dflash.zig");
const prefix = @import("coreai_prefix.zig");

pub const Error = error{
    CoreAiUnavailable,
    BadManifest,
    BadConfig,
    BadEmbeddings,
    UnsupportedKvFormat,
    LoadFailed,
    Fp8OnM5,
    /// Forge builds the qwen3_5 family only; this host runtime is that graph's.
    UnsupportedArch,
    UnboundInput,
    RunFailed,
    ContextFull,
    Cancelled,
    OutOfMemory,
};

pub const KvFormat = enum { fp16, v8 };

/// True when `path` is a Forge ANE bundle (see `isBundleDir`).
pub fn isBundle(io: std.Io, path: []const u8) bool {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch return false;
    defer dir.close(io);
    return isBundleDir(io, dir);
}

/// `coreai/manifest.json` and `model/config.json`.
pub fn isBundleDir(io: std.Io, dir: std.Io.Dir) bool {
    for ([_][]const u8{ "coreai/manifest.json", "model/config.json" }) |rel| {
        const st = dir.statFile(io, rel, .{}) catch return false;
        if (st.kind != .file) return false;
    }
    return true;
}

/// What a bundle keeps resident: every package's program (the drafter's too) plus the embedding table.
pub fn bundleBytes(io: std.Io, dir: std.Io.Dir) ?u64 {
    const total = packageBytes(io, dir, "coreai") orelse return null;
    const all = total + (packageBytes(io, dir, "drafter") orelse 0);
    const emb = dir.statFile(io, "model/embed_tokens_fp16.npy", .{}) catch return if (all > 0) all else null;
    return all + emb.size;
}

fn packageBytes(io: std.Io, dir: std.Io.Dir, sub: []const u8) ?u64 {
    var total: u64 = 0;
    var cdir = dir.openDir(io, sub, .{ .iterate = true }) catch return null;
    defer cdir.close(io);
    var it = cdir.iterate();
    while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".aimodel")) continue;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const p = std.fmt.bufPrint(&buf, "{s}/main.mlirb", .{e.name}) catch continue;
        const st = cdir.statFile(io, p, .{}) catch continue;
        total += st.size;
    }
    return total;
}

// ── manifest + config ──

pub const ChunkSpec = struct {
    file: []const u8,
    entries: []const []const u8,
    gdn_j: []const u32,
    att_j: []const u32,
    /// Layers inside the chunk with their own `tap<l>` output; the chunk's last layer is `y`.
    taps: []const u32,
    last: u32,
};

pub const Manifest = struct {
    arena: std.heap.ArenaAllocator,
    T: u32,
    TP: u32,
    /// Context entries, ascending, with the KV rows each holds (`kv_len`) and its prefill entry's.
    ctxs: []const u32,
    kv_len: []const u32,
    pkv_len: []const u32,
    kv_format: KvFormat,
    /// `dim_token` key cache: (KV head, head dim, token).
    keys_t: bool,
    /// Target layers whose hidden states feed the DFlash2 drafter (empty: no drafter).
    taps: []const u32,
    /// Some chunk's attention uses FP8 (`ATT_INT8MM` sm8 / pvf8 / pvf5): an M6 build.
    fp8: bool,
    chunks: []const ChunkSpec,
    head_file: []const u8,

    pub fn deinit(self: *Manifest) void {
        self.arena.deinit();
    }

    pub fn cap(self: *const Manifest, ctx: u32) u32 {
        for (self.ctxs, self.kv_len) |c, l| if (c == ctx) return l;
        return ctx;
    }

    /// Whether `ctx` has a 64-row prefill entry whose KV history matches the decode entry's buffers.
    pub fn hasPrefill(self: *const Manifest, ctx: u32) bool {
        if (self.TP == 0) return false;
        for (self.ctxs, self.pkv_len) |c, l| if (c == ctx) return l == self.cap(ctx);
        return false;
    }
};

pub fn parseManifest(parent: std.mem.Allocator, json: []const u8) Error!Manifest {
    var arena = std.heap.ArenaAllocator.init(parent);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.BadManifest;
    if (root != .object) return error.BadManifest;
    const o = root.object;

    const kv = (o.get("kv_cache") orelse return error.BadManifest);
    if (kv != .object) return error.BadManifest;
    const fmt = try str(kv.object.get("format"));
    const kv_format: KvFormat = if (std.mem.eql(u8, fmt, "v8")) .v8 else if (std.mem.eql(u8, fmt, "fp16")) .fp16 else return error.UnsupportedKvFormat;
    const key_layout = if (kv.object.get("key_layout")) |v| try str(v) else "token_dim";

    const ctxs = try u32s(a, o.get("ctxs"));
    std.mem.sort(u32, ctxs, {}, std.sort.asc(u32));
    const kv_len = try a.alloc(u32, ctxs.len);
    const pkv_len = try a.alloc(u32, ctxs.len);
    for (ctxs, kv_len, pkv_len) |c, *l, *pl| {
        l.* = try lenFor(o.get("kv_len"), c);
        pl.* = try lenFor(o.get("pkv_len"), c);
    }

    const cv = o.get("chunks") orelse return error.BadManifest;
    if (cv != .array or cv.array.items.len == 0) return error.BadManifest;
    var fp8 = false;
    const chunks = try a.alloc(ChunkSpec, cv.array.items.len);
    for (cv.array.items, chunks) |c, *spec| {
        if (c != .object) return error.BadManifest;
        const ev = c.object.get("entries") orelse return error.BadManifest;
        if (ev != .array) return error.BadManifest;
        const entries = try a.alloc([]const u8, ev.array.items.len);
        for (ev.array.items, entries) |e, *dst| dst.* = try str(e);
        if (c.object.get("numerics")) |n| if (n == .object) if (n.object.get("ATT_INT8MM")) |f| if (f == .string) {
            fp8 = fp8 or fp8Forms(f.string);
        };
        const layers = try u32s(a, c.object.get("layers"));
        if (layers.len != 2) return error.BadManifest;
        spec.* = .{
            .file = try str(c.object.get("file")),
            .entries = entries,
            .gdn_j = try u32s(a, c.object.get("gdn_j")),
            .att_j = try u32s(a, c.object.get("att_j")),
            .taps = if (c.object.get("taps")) |t| try u32s(a, t) else &.{},
            .last = layers[1],
        };
    }
    const head = o.get("head") orelse return error.BadManifest;
    if (head != .object) return error.BadManifest;

    return .{
        .arena = arena,
        .T = try int(o.get("T")),
        .TP = if (o.get("TP")) |v| try int(v) else 0,
        .ctxs = ctxs,
        .kv_len = kv_len,
        .pkv_len = pkv_len,
        .kv_format = kv_format,
        .keys_t = std.mem.eql(u8, key_layout, "dim_token"),
        .taps = if (o.get("taps")) |t| try u32s(a, t) else &.{},
        .fp8 = fp8,
        .chunks = chunks,
        .head_file = try str(head.object.get("file")),
    };
}

/// Whether an `ATT_INT8MM` form list holds an FP8 form (softmax / PV probabilities).
pub fn fp8Forms(forms: []const u8) bool {
    var it = std.mem.tokenizeScalar(u8, forms, ',');
    while (it.next()) |f| if (std.mem.eql(u8, f, "sm8") or std.mem.startsWith(u8, f, "pvf")) return true;
    return false;
}

/// The Apple silicon generation in a CPU brand string ("Apple M5 Ultra" -> 5); null when not one.
pub fn chipGeneration(brand: []const u8) ?u32 {
    const rest = if (std.mem.startsWith(u8, brand, "Apple M")) brand["Apple M".len..] else return null;
    const end = std.mem.indexOfNone(u8, rest, "0123456789") orelse rest.len;
    return std.fmt.parseInt(u32, rest[0..end], 10) catch null;
}

fn thisChipGeneration() ?u32 {
    var buf: [128]u8 = undefined;
    var len: usize = buf.len;
    if (std.c.sysctlbyname("machdep.cpu.brand_string", &buf, &len, null, 0) != 0) return null;
    return chipGeneration(std.mem.sliceTo(&buf, 0));
}

fn str(v: ?std.json.Value) Error![]const u8 {
    const x = v orelse return error.BadManifest;
    return if (x == .string) x.string else error.BadManifest;
}

fn int(v: ?std.json.Value) Error!u32 {
    const x = v orelse return error.BadManifest;
    return if (x == .integer and x.integer >= 0) @intCast(x.integer) else error.BadManifest;
}

fn u32s(a: std.mem.Allocator, v: ?std.json.Value) Error![]u32 {
    const x = v orelse return error.BadManifest;
    if (x != .array) return error.BadManifest;
    const out = try a.alloc(u32, x.array.items.len);
    for (x.array.items, out) |e, *d| d.* = try int(e);
    return out;
}

/// `{"8192": 8192, ...}` lookup; a missing map or key means the context length itself.
fn lenFor(v: ?std.json.Value, ctx: u32) Error!u32 {
    const m = v orelse return ctx;
    if (m != .object) return error.BadManifest;
    var key: [16]u8 = undefined;
    const k = std.fmt.bufPrint(&key, "{d}", .{ctx}) catch unreachable;
    return if (m.object.get(k)) |l| try int(l) else ctx;
}

pub const TextConfig = struct {
    hidden: u32,
    vocab: u32,
    /// Rotary dims (head_dim x partial_rotary_factor) and their base.
    rot: u32,
    rope_theta: f64,
};

pub fn parseTextConfig(a: std.mem.Allocator, json: []const u8) Error!TextConfig {
    const parsed = std.json.parseFromSlice(std.json.Value, a, json, .{}) catch return error.BadConfig;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const top = parsed.value.object;
    const mt = top.get("model_type") orelse return error.BadConfig;
    if (mt != .string or !std.mem.eql(u8, mt.string, "qwen3_5")) return error.UnsupportedArch;
    const t = if (top.get("text_config")) |v| (if (v == .object) v.object else return error.BadConfig) else top;
    const rp = t.get("rope_parameters") orelse return error.BadConfig;
    if (rp != .object) return error.BadConfig;
    const head_dim = num(t.get("head_dim")) orelse return error.BadConfig;
    const partial = num(rp.object.get("partial_rotary_factor")) orelse 1.0;
    return .{
        .hidden = @intFromFloat(num(t.get("hidden_size")) orelse return error.BadConfig),
        .vocab = @intFromFloat(num(t.get("vocab_size")) orelse return error.BadConfig),
        .rot = @intFromFloat(head_dim * partial),
        .rope_theta = num(rp.object.get("rope_theta")) orelse return error.BadConfig,
    };
}

pub fn num(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

// ── embeddings (`embed_tokens_fp16.npy`, mmapped) ──

pub const NpyHeader = struct { rows: u64, cols: u64, data_offset: usize };

/// A C-order little-endian FP16 2-D `.npy` header (format 1.0 / 2.0 / 3.0).
pub fn parseNpyHeader(bytes: []const u8) Error!NpyHeader {
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadEmbeddings;
    const major = bytes[6];
    const hlen: usize, const start: usize = switch (major) {
        1 => .{ std.mem.readInt(u16, bytes[8..10], .little), 10 },
        2, 3 => blk: {
            if (bytes.len < 12) return error.BadEmbeddings;
            break :blk .{ std.mem.readInt(u32, bytes[8..12], .little), 12 };
        },
        else => return error.BadEmbeddings,
    };
    if (bytes.len < start + hlen) return error.BadEmbeddings;
    const h = bytes[start .. start + hlen];
    if (std.mem.indexOf(u8, h, "'descr': '<f2'") == null) return error.BadEmbeddings;
    if (std.mem.indexOf(u8, h, "'fortran_order': False") == null) return error.BadEmbeddings;
    const s0 = (std.mem.indexOf(u8, h, "'shape': (") orelse return error.BadEmbeddings) + 10;
    const s1 = std.mem.indexOfScalarPos(u8, h, s0, ')') orelse return error.BadEmbeddings;
    var it = std.mem.tokenizeAny(u8, h[s0..s1], ", ");
    const rows = std.fmt.parseInt(u64, it.next() orelse return error.BadEmbeddings, 10) catch return error.BadEmbeddings;
    const cols = std.fmt.parseInt(u64, it.next() orelse return error.BadEmbeddings, 10) catch return error.BadEmbeddings;
    if (it.next() != null) return error.BadEmbeddings;
    return .{ .rows = rows, .cols = cols, .data_offset = start + hlen };
}

/// The file at `path` mapped read-only; null when it cannot be.
pub fn mapFile(path: [*:0]const u8) ?[]align(std.heap.page_size_min) const u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const size: usize = @intCast(@max(std.c.lseek(fd, 0, std.c.SEEK.END), 0));
    if (size == 0) return null;
    return std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0) catch null;
}

const Embeddings = struct {
    map: []align(std.heap.page_size_min) const u8,
    rows: u64,
    cols: u64,
    data: [*]const f16,

    fn open(path: []const u8, hidden: u32) Error!Embeddings {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&pbuf, "{s}", .{path}, 0) catch return error.BadEmbeddings;
        const map = mapFile(pz) orelse return error.BadEmbeddings;
        errdefer std.posix.munmap(map);
        const h = try parseNpyHeader(map[0..@min(map.len, 4096)]);
        if (h.cols != hidden or h.data_offset + h.rows * h.cols * 2 > map.len or h.data_offset % 2 != 0) return error.BadEmbeddings;
        return .{ .map = map, .rows = h.rows, .cols = h.cols, .data = @ptrCast(@alignCast(map.ptr + h.data_offset)) };
    }

    fn row(self: *const Embeddings, id: u32) []const f16 {
        return self.data[@as(usize, id) * self.cols ..][0..self.cols];
    }

    fn close(self: *Embeddings) void {
        std.posix.munmap(self.map);
    }
};

// ── host math (Forge `_common` / `quantize_values`) ──

/// RoPE inverse frequencies: 1 / theta^(i / rot) for i = 0, 2, ..., rot - 2.
pub fn ropeInv(out: []f64, rot: u32, theta: f64) void {
    for (out, 0..) |*v, i| v.* = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rot)));
}

/// Row r's position: p0 + r, clamped to the last real token (padding rows repeat it).
pub fn rowPos(p0: u32, n: u32, r: u32) u32 {
    return @min(p0 + r, p0 + n - 1);
}

/// numpy's `rint`: round half to even.
pub fn rintEven(x: f32) f32 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return 2.0 * @round(x / 2.0);
    return r;
}

/// Symmetric INT8 for one (token, head) row: the FP16 scale that is stored is the one rounded with.
pub fn quantizeRow(values: []const f16, codes: []i8) f16 {
    var amax: f32 = 0;
    for (values) |v| amax = @max(amax, @abs(@as(f32, v)));
    const scale: f16 = @floatCast(@max(amax / 127.0, 1e-6));
    const s: f32 = scale;
    for (values, codes) |v, *c| c.* = @intFromFloat(std.math.clamp(rintEven(@as(f32, v) / s), -127, 127));
    return scale;
}

// ── sampling ──

pub const Sampling = struct {
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1,
    min_p: f32 = 0,
};

/// First index of the largest logit (numpy argmax).
pub fn argmax(logits: []const f16) u32 {
    var best: usize = 0;
    var bv: f32 = logits[0];
    for (logits[1..], 1..) |v, i| {
        if (@as(f32, v) > bv) {
            bv = v;
            best = i;
        }
    }
    return @intCast(best);
}

pub const Cand = struct { logit: f32, id: u32 };

/// Temperature / top-k / top-p / min-p sampling; temperature <= 0 is argmax. `scratch` holds vocab entries.
pub fn sample(logits: []const f16, s: Sampling, rng: std.Random, scratch: []Cand) u32 {
    return drawFrom(dist(logits, s, scratch) orelse return argmax(logits), rng);
}

/// Draw from candidates whose `logit` holds an unnormalized probability.
pub fn drawFrom(cands: []const Cand, rng: std.Random) u32 {
    var sum: f32 = 0;
    for (cands) |c| sum += c.logit;
    var u = rng.float(f32) * sum;
    for (cands) |c| {
        if (u < c.logit) return c.id;
        u -= c.logit;
    }
    return cands[cands.len - 1].id;
}

/// One row's sampling distribution (top-k, temperature, min-p, top-p), descending, probabilities in
/// `logit`; null for greedy. Plain and speculative decoding draw from exactly this.
pub fn dist(logits: []const f16, s: Sampling, scratch: []Cand) ?[]Cand {
    if (s.temperature <= 0) return null;
    const k: usize = if (s.top_k == 0) logits.len else @min(s.top_k, logits.len);
    var cands = topK(logits, k, scratch);
    const inv_t = 1.0 / s.temperature;
    const top = cands[0].logit;
    var total: f32 = 0;
    for (cands) |*c| {
        c.logit = @exp((c.logit - top) * inv_t);
        total += c.logit;
    }
    var n: usize = cands.len;
    if (s.min_p > 0) {
        const floor = s.min_p * cands[0].logit;
        n = 1;
        while (n < cands.len and cands[n].logit >= floor) n += 1;
    }
    if (s.top_p < 1) {
        var acc: f32 = 0;
        var m: usize = 0;
        while (m < n) : (m += 1) {
            acc += cands[m].logit / total;
            if (acc >= s.top_p) {
                m += 1;
                break;
            }
        }
        n = @max(m, 1);
    }
    return cands[0..n];
}

/// Null when draft `dk` stands against the target's `row`: greedy, when it is the argmax; sampled,
/// with the target's probability of it. Otherwise the target's own token, drawn without the draft.
pub fn verifyDraft(row: []const f16, dk: u32, s: Sampling, rng: std.Random, scratch: []Cand) ?u32 {
    const dd = dist(row, s, scratch) orelse {
        const best = argmax(row);
        return if (best == dk) null else best;
    };
    const hit: ?usize = for (dd, 0..) |c, j| {
        if (c.id == dk) break j;
    } else null;
    var total: f32 = 0;
    for (dd) |c| total += c.logit;
    if (hit) |h| {
        if (rng.float(f32) * total < dd[h].logit) return null;
        dd[h].logit = 0;
    }
    return drawFrom(dd, rng);
}

/// The k largest logits, descending.
pub fn topK(logits: []const f16, k: usize, scratch: []Cand) []Cand {
    const desc = struct {
        fn f(_: void, a: Cand, b: Cand) bool {
            return a.logit > b.logit;
        }
    }.f;
    if (k == logits.len) {
        for (logits, scratch[0..logits.len], 0..) |v, *c, i| c.* = .{ .logit = v, .id = @intCast(i) };
        std.sort.pdq(Cand, scratch[0..logits.len], {}, desc);
        return scratch[0..logits.len];
    }
    // Insertion into a sorted window of k: cheap for the small k sampling uses.
    var n: usize = 0;
    for (logits, 0..) |v, i| {
        const x: f32 = v;
        if (n == k and x <= scratch[k - 1].logit) continue;
        var j: usize = if (n < k) n else k - 1;
        if (n < k) n += 1;
        while (j > 0 and scratch[j - 1].logit < x) : (j -= 1) scratch[j] = scratch[j - 1];
        scratch[j] = .{ .logit = x, .id = @intCast(i) };
    }
    return scratch[0..n];
}

// ── bridge objects ──

pub const Spec = struct {
    name: [:0]const u8,
    dtype: ffi.DType,
    shape: []const i64,
    strides: []const i64,
};

/// A strided view of an IOSurface buffer (element strides, like the model's preferred layout).
pub const Tensor = struct {
    base: [*]u8,
    dtype: ffi.DType,
    shape: []const i64,
    strides: []const i64,
    nbytes: usize,

    pub fn f16s(self: Tensor) [*]f16 {
        return @ptrCast(@alignCast(self.base));
    }
    pub fn i8s(self: Tensor) [*]i8 {
        return @ptrCast(self.base);
    }
    pub fn stride(self: Tensor, axis: usize) usize {
        return @intCast(self.strides[axis]);
    }
    pub fn zero(self: Tensor) void {
        @memset(self.base[0..self.nbytes], 0);
    }
};

pub const Buffer = struct {
    handle: *anyopaque,
    t: Tensor,
};

pub const Named = struct { name: [:0]const u8, buf: Buffer };

pub fn findNamed(list: []const Named, name: []const u8) ?Buffer {
    for (list) |n| if (std.mem.eql(u8, n.name, name)) return n.buf;
    return null;
}

pub const Function = struct {
    handle: *anyopaque,
    inputs: []const Spec,
    outputs: []const Spec,
};

pub fn loadFunctionWith(api: *const ffi.Api, a: std.mem.Allocator, model: *anyopaque, name: []const u8) !Function {
    const nz = try a.dupeSentinel(u8, name, 0);
    var e: ffi.ErrBuf = .{};
    const h = api.function_load(model, nz, e.ptr(), e.cap()) orelse {
        log.err("[coreai] function {s}: {s}\n", .{ name, e.text() });
        return error.LoadFailed;
    };
    const need = api.function_describe(h, null, 0);
    const json = try a.alloc(u8, @intCast(need));
    _ = api.function_describe(h, json.ptr, need);
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, std.mem.sliceTo(json, 0), .{}) catch return error.LoadFailed;
    if (parsed != .object) return error.LoadFailed;
    return .{ .handle = h, .inputs = try specs(a, parsed.object.get("inputs")), .outputs = try specs(a, parsed.object.get("outputs")) };
}

/// A zero-filled IOSurface buffer in the spec's preferred layout.
pub fn bufferWith(api: *const ffi.Api, s: Spec) !Buffer {
    var e: ffi.ErrBuf = .{};
    const h = api.buffer_create(@backingInt(s.dtype), @intCast(s.shape.len), s.shape.ptr, s.strides.ptr, e.ptr(), e.cap()) orelse {
        log.err("[coreai] buffer {s}: {s}\n", .{ s.name, e.text() });
        return error.LoadFailed;
    };
    errdefer api.release(h);
    const base: [*]u8 = @ptrCast(api.buffer_address(h) orelse return error.LoadFailed);
    return .{ .handle = h, .t = .{ .base = base, .dtype = s.dtype, .shape = s.shape, .strides = s.strides, .nbytes = @intCast(api.buffer_nbytes(h)) } };
}

/// One buffer per spec; none left behind on failure.
pub fn buffersForWith(api: *const ffi.Api, a: std.mem.Allocator, list: []const Spec) ![]Named {
    const out = try a.alloc(Named, list.len);
    for (list, out, 0..) |sp, *n, i| {
        errdefer releaseNamed(api, out[0..i]);
        n.* = .{ .name = sp.name, .buf = try bufferWith(api, sp) };
    }
    return out;
}

pub fn namedBytes(list: []const Named) u64 {
    var n: u64 = 0;
    for (list) |x| n += x.buf.t.nbytes;
    return n;
}

pub fn releaseNamed(api: *const ffi.Api, list: []const Named) void {
    for (list) |n| api.release(n.buf.handle);
}

/// A binding of every input and output of `f` to the buffers of the same names.
pub fn bindNamed(api: *const ffi.Api, f: Function, ins: []const Named, outs: []const Named) !*anyopaque {
    var in_n: [64][*:0]const u8 = undefined;
    var in_b: [64]?*anyopaque = undefined;
    var out_n: [64][*:0]const u8 = undefined;
    var out_b: [64]?*anyopaque = undefined;
    if (f.inputs.len > 64 or f.outputs.len > 64) return error.UnboundInput;
    for (f.inputs, 0..) |sp, i| {
        in_n[i] = sp.name.ptr;
        in_b[i] = (findNamed(ins, sp.name) orelse return error.UnboundInput).handle;
    }
    for (f.outputs, 0..) |sp, i| {
        out_n[i] = sp.name.ptr;
        out_b[i] = (findNamed(outs, sp.name) orelse return error.UnboundInput).handle;
    }
    return bindWith(api, f.handle, in_n[0..f.inputs.len], in_b[0..f.inputs.len], out_n[0..f.outputs.len], out_b[0..f.outputs.len]);
}

pub fn bindWith(api: *const ffi.Api, f: *anyopaque, in_n: []const [*:0]const u8, in_b: []const ?*anyopaque, out_n: []const [*:0]const u8, out_b: []const ?*anyopaque) !*anyopaque {
    var e: ffi.ErrBuf = .{};
    return api.binding_create(f, @intCast(in_n.len), in_n.ptr, in_b.ptr, @intCast(out_n.len), out_n.ptr, out_b.ptr, 0, null, null, e.ptr(), e.cap()) orelse {
        log.err("[coreai] bind: {s}\n", .{e.text()});
        return error.LoadFailed;
    };
}

const EntryKind = enum { decode, prefill };

const Entry = struct {
    name: []const u8,
    kind: EntryKind,
    ctx: u32,
};

/// `v8_16k` → decode at 16384, `p64_8k` → prefill at 8192.
pub fn parseEntryName(name: []const u8) ?struct { prefill: bool, ctx: u32 } {
    const us = std.mem.indexOfScalar(u8, name, '_') orelse return null;
    if (!std.mem.endsWith(u8, name, "k")) return null;
    const k = std.fmt.parseInt(u32, name[us + 1 .. name.len - 1], 10) catch return null;
    const head = name[0..us];
    const prefill = std.mem.eql(u8, head, "p64");
    if (!prefill and !std.mem.eql(u8, head, "v8")) return null;
    return .{ .prefill = prefill, .ctx = k * 1024 };
}

const Chunk = struct {
    model: *anyopaque,
    /// Indexed like `Engine.entries`.
    fns: []Function,
    att_j: []const u32,
    /// DeltaNet states, two sets: a call reads `st[par]` and writes `st[1 - par]` (the `*_out` outputs).
    st: [2][]Named,
    /// Per entry: every output that is not a state.
    out: [][]Named,
    /// `k<j>`, `v<j>`, `vs<j>` for the current context entry.
    kv: []Named,
};

pub const Synced = struct { logits: []const f16, cached: usize };

/// One speculative cycle's output: the accepted drafts, then the target's next token.
pub const Cycle = struct { tokens: [8]u32, n: usize, accepted: usize };

/// The drafter's call stalls when it lands within ~2 ms of the target verify's return (Forge, M6).
const DRAFT_GAP_NS: u64 = 3 * std.time.ns_per_ms;

const TapSrc = struct { chunk: usize, name: [:0]const u8 };

pub const Engine = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    api: *const ffi.Api,
    /// The engine whose loaded functions, manifest and embeddings this one borrows (`slot`); null
    /// owns them.
    base: ?*Engine,
    man: Manifest,
    cfg: TextConfig,
    emb: Embeddings,
    inv: []f64,
    entries: []Entry,
    chunks: []Chunk,
    head_model: *anyopaque,
    head: Function,
    hout: []Named,
    logits: Tensor,
    /// Per-call inputs shared by every chunk: decode (`inp`) and prefill (`pin`, empty without TP).
    inp: []Named,
    hx: Buffer,
    pin: []Named,
    mask: Buffer,
    ctx: u32,
    par: u1 = 0,
    pos: u32 = 0,
    pending: u32 = 0,
    /// KV rows present in the cache.
    hi: u32 = 0,
    last: usize = 0,
    /// Bindings per (entry, parity): every chunk, then the head.
    plans: []?[]?*anyopaque,
    /// Tokens the model state holds, in order (prompt-prefix reuse across requests).
    fed: std.ArrayList(u32) = .empty,
    scratch: []Cand,
    io: std.Io,
    /// The bundle's DFlash2 drafter (`<bundle>/drafter`); null decodes one token per call.
    drafter: ?*dflash.Drafter,
    /// Where each tap's rows sit in a call's outputs (`tap<l>`, or the chunk's `y` for its last layer).
    tap_src: []TapSrc,
    /// The last call's tap rows (rows x taps x hidden): the drafter's context input.
    feat: []f16,
    verify_end: std.Io.Timestamp,
    /// Checkpoints of earlier sequences, shared by a pool's engines (`useCache`); null keeps only the
    /// live state.
    cache: ?*prefix.Cache,
    /// The deepest checkpoint the live state continues (pinned), and where the next one is due.
    node: ?*prefix.Node,
    stride: u32,
    next_cp: u32,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, bundle: []const u8) !*Engine {
        const api = ffi.get() catch return error.CoreAiUnavailable;
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.* = undefined;
        self.allocator = allocator;
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();
        self.api = api;
        self.base = null;
        const a = self.arena.allocator();

        const man_json = try readFile(io, a, bundle, "coreai/manifest.json", 1 << 22);
        self.man = try parseManifest(allocator, man_json);
        errdefer self.man.deinit();
        // The M5 ANE compiler rejects FP8: Core AI would run the chunks on the GPU, and that fallback
        // compile of a large build can fill the disk and abort the process.
        if (self.man.fp8 and thisChipGeneration() == 5) {
            log.err("[coreai] {s} is an M6 build (FP8 attention), which the M5 Neural Engine cannot run; rebuild it for M5 without FP8 (Forge: ATT_INT8MM=s8,s8b ATT_SOFTMAX=online)\n", .{bundle});
            return error.Fp8OnM5;
        }
        const cfg_json = try readFile(io, a, bundle, "model/config.json", 1 << 22);
        self.cfg = try parseTextConfig(a, cfg_json);
        self.inv = try a.alloc(f64, self.cfg.rot / 2);
        ropeInv(self.inv, self.cfg.rot, self.cfg.rope_theta);
        const emb_path = try std.fmt.allocPrint(a, "{s}/model/embed_tokens_fp16.npy", .{bundle});
        self.emb = try Embeddings.open(emb_path, self.cfg.hidden);
        errdefer self.emb.close();
        self.io = io;
        self.drafter = null;
        self.verify_end = std.Io.Timestamp.now(io, .awake);
        self.cache = null;
        self.node = null;
        self.stride = 0;
        self.next_cp = 0;
        if (self.man.T > 8) return error.BadManifest;

        // The canonical entries; a dual M6 / M5 build's `<entry>_m5` copies are not loaded by name.
        const spec0 = self.man.chunks[0];
        var entries: std.ArrayList(Entry) = .empty;
        for (spec0.entries) |name| if (parseEntryName(name)) |p| {
            try entries.append(a, .{ .name = name, .kind = if (p.prefill) .prefill else .decode, .ctx = p.ctx });
        };
        self.entries = entries.items;
        const entry_names = try a.alloc([]const u8, self.entries.len);
        for (self.entries, entry_names) |e, *n| n.* = e.name;

        self.chunks = try a.alloc(Chunk, self.man.chunks.len);
        const t0 = std.Io.Timestamp.now(io, .awake).nanoseconds;
        var loaded: usize = 0;
        errdefer for (self.chunks[0..loaded]) |ch| self.unload(ch);
        for (self.man.chunks, self.chunks, 0..) |spec, *ch, ci| {
            if (spec.entries.len != spec0.entries.len) return error.BadManifest;
            ch.fns = try a.alloc(Function, self.entries.len);
            ch.model = try self.loadPackage(io, bundle, spec.file, entry_names, ch.fns);
            ch.att_j = spec.att_j;
            loaded += 1;
            log.info("[coreai] loaded {s} ({d}/{d})\n", .{ spec.file, ci + 1, self.man.chunks.len });
        }
        var head: [1]Function = undefined;
        self.head_model = try self.loadPackage(io, bundle, self.man.head_file, &.{"h8"}, &head);
        self.head = head[0];
        errdefer {
            self.api.release(self.head.handle);
            self.api.release(self.head_model);
        }
        try checkLimits(self.head.outputs.len, self.kvHeadDim());
        try self.initState();
        errdefer self.releaseState();
        if (self.tap_src.len > 0) self.drafter = self.openDrafter(io, bundle);
        self.reset(false);
        self.warnGpuFallbacks(io, bundle);
        const dt = @divTrunc(std.Io.Timestamp.now(io, .awake).nanoseconds - t0, std.time.ns_per_ms);
        log.info("[coreai] ready: {d} chunks + head, contexts {any}, KV {s}{s}, loaded in {d} ms\n", .{
            self.chunks.len, self.man.ctxs, @tagName(self.man.kv_format), if (self.man.keys_t) " (keys transposed)" else "", dt,
        });
        return self;
    }

    /// Another engine on `base`'s loaded functions, with its own KV, DeltaNet states, buffers and
    /// drafter context. Core AI runs one loaded function's calls in turn but different ones at
    /// once, so requests on two engines overlap whenever they are in different chunks.
    pub fn slot(base: *Engine) !*Engine {
        const self = try base.allocator.create(Engine);
        errdefer base.allocator.destroy(self);
        self.* = base.*;
        self.base = base;
        self.arena = std.heap.ArenaAllocator.init(base.allocator);
        errdefer self.arena.deinit();
        self.chunks = try self.arena.allocator().alloc(Chunk, base.chunks.len);
        for (self.chunks, base.chunks) |*c, b| c.* = .{ .model = b.model, .fns = b.fns, .att_j = b.att_j, .st = undefined, .out = undefined, .kv = &.{} };
        self.drafter = null;
        self.cache = null;
        self.node = null;
        self.stride = 0;
        self.next_cp = 0;
        try self.initState();
        errdefer self.releaseState();
        if (base.drafter) |d| self.drafter = try d.slot();
        self.reset(false);
        return self;
    }

    /// This engine's buffers for the loaded functions: DeltaNet states, outputs, per-call inputs, KV
    /// at the smallest context; an empty state at position 0.
    fn initState(self: *Engine) !void {
        const a = self.arena.allocator();
        // Every list empty first: a failure part way releases exactly what it made.
        for (self.chunks) |*ch| {
            ch.st = .{ &.{}, &.{} };
            ch.out = &.{};
            ch.kv = &.{};
        }
        self.hout = &.{};
        self.inp = &.{};
        self.pin = &.{};
        self.plans = &.{};
        errdefer self.releaseLists();
        for (self.man.chunks, self.chunks) |spec, *ch| {
            const f0 = ch.fns[self.firstEntry(.decode) orelse return error.BadManifest];
            var names: std.ArrayList([]const u8) = .empty;
            for (spec.gdn_j) |j| for ([_][]const u8{ "conv", "rec", "pend" }) |n| try names.append(a, try std.fmt.allocPrint(a, "{s}{d}", .{ n, j }));
            for (&ch.st) |*set| set.* = try self.buffersFor(f0, names.items, .input);
            ch.out = try a.alloc([]Named, self.entries.len);
            @memset(ch.out, &.{});
            for (ch.fns, ch.out) |f, *o| {
                var outs: std.ArrayList([]const u8) = .empty;
                for (f.outputs) |s| if (!std.mem.endsWith(u8, s.name, "_out")) try outs.append(a, s.name);
                o.* = try self.buffersFor(f, outs.items, .output);
            }
        }
        var hnames: std.ArrayList([]const u8) = .empty;
        for (self.head.outputs) |s| try hnames.append(a, s.name);
        self.hout = try self.buffersFor(self.head, hnames.items, .output);
        self.logits = (findNamed(self.hout, "logits") orelse return error.BadManifest).t;

        const f_dec = self.chunks[0].fns[self.firstEntry(.decode).?];
        self.inp = try self.buffersFor(f_dec, &.{ "cos", "sin", "conv_sel", "commit", "commit_last", "x" }, .input);
        self.pin = if (self.firstEntry(.prefill)) |pi|
            try self.buffersFor(self.chunks[0].fns[pi], &.{ "cos", "sin", "conv_sel", "commit", "commit_last", "conv_sel_out", "valid", "x" }, .input)
        else
            &.{};

        self.plans = try a.alloc(?[]?*anyopaque, self.entries.len * 2);
        @memset(self.plans, null);
        self.fed = .empty;
        self.ctx = 0;
        self.par = 0;
        self.pos = 0;
        self.pending = 0;
        self.hi = 0;
        self.last = 0;
        self.tap_src = try self.tapSources();
        self.feat = try a.alloc(f16, @max(self.man.T, self.man.TP) * self.tap_src.len * self.cfg.hidden);
        self.scratch = try a.alloc(Cand, self.cfg.vocab);
        self.hx = (try self.buffersFor(self.head, &.{"x"}, .input))[0].buf;
        errdefer self.api.release(self.hx.handle);
        self.mask = undefined;
        try self.allocKv(self.man.ctxs[0], 0);
    }

    /// This engine's buffer lists and bindings (a part-built state's empty lists release nothing).
    fn releaseLists(self: *Engine) void {
        self.dropPlans();
        for (self.chunks) |ch| {
            releaseNamed(self.api, ch.kv);
            for (ch.out) |o| releaseNamed(self.api, o);
            for (ch.st) |set| releaseNamed(self.api, set);
        }
        releaseNamed(self.api, self.inp);
        releaseNamed(self.api, self.pin);
        releaseNamed(self.api, self.hout);
    }

    /// Bytes of this engine's own buffers (at the current context) and its drafter's.
    pub fn bufferBytes(self: *const Engine) u64 {
        var n: u64 = self.hx.t.nbytes + self.mask.t.nbytes;
        for (self.chunks) |ch| {
            n += namedBytes(ch.kv);
            for (ch.out) |o| n += namedBytes(o);
            for (ch.st) |set| n += namedBytes(set);
        }
        n += namedBytes(self.inp) + namedBytes(self.pin) + namedBytes(self.hout);
        if (self.drafter) |d| n += d.bufferBytes();
        return n;
    }

    fn releaseState(self: *Engine) void {
        self.releaseLists();
        self.api.release(self.hx.handle);
        self.api.release(self.mask.handle);
    }

    /// The widest `k<j>_new` output's head dim.
    fn kvHeadDim(self: *const Engine) usize {
        var hd: usize = 0;
        for (self.chunks) |ch| for (ch.fns) |f| for (f.outputs) |o| {
            if (o.name[0] == 'k' and std.mem.endsWith(u8, o.name, "_new") and o.shape.len == 3) hd = @max(hd, @as(usize, @intCast(o.shape[2])));
        };
        return hd;
    }

    fn unload(self: *Engine, ch: Chunk) void {
        for (ch.fns) |f| self.api.release(f.handle);
        self.api.release(ch.model);
    }

    /// A slot goes before its base (it borrows the base's functions).
    pub fn deinit(self: *Engine) void {
        if (self.drafter) |d| d.deinit();
        self.releaseState();
        if (self.base == null) {
            for (self.chunks) |ch| self.unload(ch);
            self.api.release(self.head.handle);
            self.api.release(self.head_model);
            self.emb.close();
            self.man.deinit();
        }
        self.fed.deinit(self.allocator);
        self.arena.deinit();
        self.allocator.destroy(self);
    }

    /// Positions the largest context entry holds.
    pub fn maxPositions(self: *const Engine) u32 {
        return self.man.cap(self.man.ctxs[self.man.ctxs.len - 1]);
    }

    fn firstEntry(self: *const Engine, kind: EntryKind) ?usize {
        for (self.entries, 0..) |e, i| if (e.kind == kind) return i;
        return null;
    }

    fn entryFor(self: *const Engine, kind: EntryKind, ctx: u32) ?usize {
        for (self.entries, 0..) |e, i| if (e.kind == kind and e.ctx == ctx) return i;
        return null;
    }

    fn loadModel(self: *Engine, bundle: []const u8, file: []const u8) !*anyopaque {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&pbuf, "{s}/coreai/{s}", .{ bundle, file }, 0) catch return error.LoadFailed;
        var e: ffi.ErrBuf = .{};
        return self.api.model_load(p, @backingInt(ffi.Compute.ane), 1, e.ptr(), e.cap()) orelse {
            log.err("[coreai] load {s}: {s}\n", .{ file, e.text() });
            return error.LoadFailed;
        };
    }

    /// The package `file` with its functions `names` into `fns`. Core AI can keep a compile cache entry
    /// whose ANE programs the system has since evicted; its functions then fail to load (`nilError`), so
    /// on a failure the entry is dropped and the package compiled again, once.
    fn loadPackage(self: *Engine, io: std.Io, bundle: []const u8, file: []const u8, names: []const []const u8, fns: []Function) !*anyopaque {
        const model = try self.loadModel(bundle, file);
        self.loadFunctions(model, names, fns) catch |err| {
            self.api.release(model);
            const a = self.arena.allocator();
            const dir = cacheEntry(io, a, try std.fs.path.join(a, &.{ bundle, "coreai", file })) orelse return err;
            log.warn("[coreai] {s}: dropping its compile cache entry and compiling it again\n", .{file});
            std.Io.Dir.cwd().deleteTree(io, dir) catch return err;
            const again = try self.loadModel(bundle, file);
            errdefer self.api.release(again);
            try self.loadFunctions(again, names, fns);
            return again;
        };
        return model;
    }

    fn loadFunctions(self: *Engine, model: *anyopaque, names: []const []const u8, fns: []Function) !void {
        for (names, fns, 0..) |n, *f, i| f.* = loadFunctionWith(self.api, self.arena.allocator(), model, n) catch |err| {
            for (fns[0..i]) |g| self.api.release(g.handle);
            return err;
        };
    }

    const Kind = enum { input, output };

    /// One zero-filled buffer per name, in the function's preferred layout.
    fn buffersFor(self: *Engine, f: Function, names: []const []const u8, kind: Kind) ![]Named {
        const a = self.arena.allocator();
        const out = try a.alloc(Named, names.len);
        for (names, out, 0..) |n, *dst, i| {
            errdefer releaseNamed(self.api, out[0..i]);
            const list = if (kind == .input) f.inputs else f.outputs;
            const s = for (list) |sp| {
                if (std.mem.eql(u8, sp.name, n)) break sp;
            } else {
                log.err("[coreai] no {s} named {s}\n", .{ @tagName(kind), n });
                return error.BadManifest;
            };
            dst.* = .{ .name = s.name, .buf = try bufferWith(self.api, s) };
        }
        return out;
    }

    // ── KV ──

    /// KV buffers for `ctx`, rows [0, keep) carried over from the current ones. Built in full before
    /// the swap, so a failure leaves the current context as it was.
    fn allocKv(self: *Engine, ctx: u32, keep: u32) !void {
        const di = self.entryFor(.decode, ctx) orelse return error.BadManifest;
        const a = self.arena.allocator();
        const fresh = try a.alloc([]Named, self.chunks.len);
        @memset(fresh, &.{});
        errdefer for (fresh) |f| releaseNamed(self.api, f);
        for (self.chunks, fresh) |ch, *f| {
            var names: std.ArrayList([]const u8) = .empty;
            for (ch.att_j) |j| {
                try names.append(a, try std.fmt.allocPrint(a, "k{d}", .{j}));
                try names.append(a, try std.fmt.allocPrint(a, "v{d}", .{j}));
                if (self.man.kv_format == .v8) try names.append(a, try std.fmt.allocPrint(a, "vs{d}", .{j}));
            }
            f.* = try self.buffersFor(ch.fns[di], names.items, .input);
            for (f.*) |n| {
                if (n.name[0] == 'v' and n.name[1] == 's') {
                    const v = n.buf.t.f16s()[0 .. n.buf.t.nbytes / 2];
                    @memset(v, 1.0); // unfilled positions are masked; any finite scale
                }
                if (keep > 0) if (findNamed(ch.kv, n.name)) |old| try copyRows(self.allocator, old.t, n.buf.t, keep, self.tokenAxis(n.name));
            }
        }
        const mask = (try self.buffersFor(self.chunks[0].fns[di], &.{"mask"}, .input))[0].buf;
        for (self.chunks, fresh) |*ch, f| {
            releaseNamed(self.api, ch.kv);
            ch.kv = f;
        }
        if (self.ctx != 0) self.api.release(self.mask.handle);
        self.mask = mask;
        self.ctx = ctx;
        self.dropPlans();
    }

    fn tokenAxis(self: *const Engine, name: []const u8) usize {
        return if (name[0] == 'k' and self.man.keys_t) 2 else 1;
    }

    fn dropPlans(self: *Engine) void {
        for (self.plans) |*p| if (p.*) |bs| {
            for (bs) |b| self.api.release(b);
            self.allocator.free(bs);
            p.* = null;
        };
    }

    /// Switch to another context entry (all are loaded): new KV buffers, rows [0, hi) moved.
    fn resize(self: *Engine, ctx: u32) !void {
        const keep = @min(self.hi, self.man.cap(ctx));
        const from = self.ctx;
        try self.allocKv(ctx, keep);
        self.hi = keep;
        log.info("[coreai] context {d} -> {d} at position {d}\n", .{ from, ctx, self.pos });
    }

    /// Grow to the smallest context that holds `need` positions; false when none does.
    fn fit(self: *Engine, need: u32) !bool {
        if (need <= self.man.cap(self.ctx)) return true;
        for (self.man.ctxs) |c| if (self.man.cap(c) >= need) {
            try self.resize(c);
            return true;
        };
        return false;
    }

    /// Fresh DeltaNet states at position 0; `shrink` returns to the smallest (fastest) context.
    pub fn reset(self: *Engine, shrink: bool) void {
        for (self.chunks) |ch| for (ch.st[self.par]) |n| n.buf.t.zero();
        self.pos = 0;
        self.pending = 0;
        self.hi = 0;
        self.fed.clearRetainingCapacity();
        if (self.drafter) |d| d.reset();
        if (self.cache) |c| {
            c.mu.lockUncancelable(self.io);
            defer c.mu.unlock(self.io);
            self.follow(null);
        }
        self.next_cp = self.stride;
        if (shrink and self.ctx != self.man.ctxs[0]) self.resize(self.man.ctxs[0]) catch |err| log.warn("[coreai] shrink failed: {s}\n", .{@errorName(err)});
    }

    // ── calls ──

    fn plan(self: *Engine, ei: usize) ![]?*anyopaque {
        const key = ei * 2 + self.par;
        if (self.plans[key]) |p| return p;
        const prefill = self.entries[ei].kind == .prefill;
        const shared = if (prefill) self.pin else self.inp;
        const bs = try self.allocator.alloc(?*anyopaque, self.chunks.len + 1);
        errdefer self.allocator.free(bs);
        var made: usize = 0;
        errdefer for (bs[0..made]) |b| self.api.release(b);
        var x = findNamed(shared, "x").?;
        for (self.chunks) |ch| {
            const f = ch.fns[ei];
            var in_n: [96][*:0]const u8 = undefined;
            var in_b: [96]?*anyopaque = undefined;
            if (f.inputs.len > in_n.len) return error.UnboundInput;
            for (f.inputs, 0..) |s, i| {
                const b = if (std.mem.eql(u8, s.name, "x")) x else if (std.mem.eql(u8, s.name, "mask")) self.mask else findNamed(shared, s.name) orelse findNamed(ch.st[self.par], s.name) orelse findNamed(ch.kv, s.name) orelse {
                    log.err("[coreai] {s}: input {s} has no buffer\n", .{ self.entries[ei].name, s.name });
                    return error.UnboundInput;
                };
                in_n[i] = s.name.ptr;
                in_b[i] = b.handle;
            }
            var out_n: [96][*:0]const u8 = undefined;
            var out_b: [96]?*anyopaque = undefined;
            if (f.outputs.len > out_n.len) return error.UnboundInput;
            for (f.outputs, 0..) |s, i| {
                const b = if (std.mem.endsWith(u8, s.name, "_out"))
                    findNamed(ch.st[1 - self.par], s.name[0 .. s.name.len - 4]).?
                else
                    findNamed(ch.out[ei], s.name).?;
                out_n[i] = s.name.ptr;
                out_b[i] = b.handle;
            }
            bs[made] = try bindWith(self.api, f.handle, in_n[0..f.inputs.len], in_b[0..f.inputs.len], out_n[0..f.outputs.len], out_b[0..f.outputs.len]);
            made += 1;
            x = findNamed(ch.out[ei], "y").?;
        }
        const hx = if (prefill) self.hx else x;
        var hn: [16][*:0]const u8 = undefined;
        var hb: [16]?*anyopaque = undefined;
        for (self.hout, 0..) |n, i| {
            hn[i] = n.name.ptr;
            hb[i] = n.buf.handle;
        }
        const x_name: [:0]const u8 = "x";
        bs[made] = try bindWith(self.api, self.head.handle, &.{x_name.ptr}, &.{hx.handle}, hn[0..self.hout.len], hb[0..self.hout.len]);
        self.plans[key] = bs;
        return bs;
    }

    fn run(self: *Engine, bindings: []const ?*anyopaque) !void {
        var e: ffi.ErrBuf = .{};
        if (self.api.run(@intCast(bindings.len), bindings.ptr, null, e.ptr(), e.cap()) != 0) {
            log.err("[coreai] run: {s}\n", .{e.text()});
            return error.RunFailed;
        }
    }

    fn flip(self: *Engine) void {
        self.par ^= 1;
    }

    /// x[0, :, 0, t] = embedding of ids[t]; other rows zero.
    fn putEmbeddings(self: *Engine, x: Tensor, ids: []const u32) void {
        x.zero();
        const dst = x.f16s();
        const sh = x.stride(1);
        const st = x.stride(3);
        for (ids, 0..) |id, t| {
            const row = self.emb.row(id);
            for (row, 0..) |v, h| dst[h * sh + t * st] = v;
        }
    }

    /// Positions, the DeltaNet commit of the last call's accepted rows, and the history mask.
    fn common(self: *Engine, d: []const Named, p0: u32, n: u32, rows: u32) void {
        const cos = findNamed(d, "cos").?.t;
        const sin = findNamed(d, "sin").?.t;
        const half = self.inv.len;
        for (0..rows) |r| {
            const p: f64 = @floatFromInt(rowPos(p0, n, @intCast(r)));
            for (0..2 * half) |c| {
                const ang = p * self.inv[c % half];
                cos.f16s()[r * cos.stride(0) + c * cos.stride(1)] = @floatCast(@cos(ang));
                sin.f16s()[r * sin.stride(0) + c * sin.stride(1)] = @floatCast(@sin(ang));
            }
        }
        const k = self.pending;
        const sel = findNamed(d, "conv_sel").?.t;
        sel.zero();
        for (0..3) |i| sel.f16s()[i * sel.stride(0) + (@as(usize, k) + i) * sel.stride(1)] = 1;
        const commit = findNamed(d, "commit").?.t;
        commit.zero();
        for (0..k) |i| commit.f16s()[i * commit.stride(1)] = 1;
        const last = findNamed(d, "commit_last").?.t;
        last.zero();
        if (k > 0) last.f16s()[(k - 1) * last.stride(1)] = 1;
        const m = self.mask.t;
        const L: usize = @intCast(m.shape[1]);
        for (0..L) |i| m.f16s()[i * m.stride(1)] = if (i < p0) 0 else -1e4;
    }

    /// Up to T tokens at `pos` through the decode entry; returns the logits view (row r = ids[r]).
    pub fn call(self: *Engine, ids: []const u32) !Tensor {
        std.debug.assert(ids.len > 0 and ids.len <= self.man.T);
        const n: u32 = @intCast(ids.len);
        if (!try self.fit(self.pos + n)) return error.ContextFull;
        self.putEmbeddings(findNamed(self.inp, "x").?.t, ids);
        self.common(self.inp, self.pos, n, self.man.T);
        const ei = self.entryFor(.decode, self.ctx).?;
        try self.run(try self.plan(ei));
        self.flip();
        self.last = ei;
        return self.logits;
    }

    /// Commit the first k rows of the last call: their K / V rows go into the caches.
    pub fn accept(self: *Engine, k: u32) void {
        self.appendRows(self.last, self.pos, k);
        self.pending = k;
        self.pos += k;
        self.hi = @max(self.hi, self.pos);
    }

    /// Up to TP prompt tokens through the prefill entry, all committed; returns the last token's logits row.
    fn prefillBlock(self: *Engine, ids: []const u32) ![]const f16 {
        const n: u32 = @intCast(ids.len);
        if (!try self.fit(self.pos + n)) return error.ContextFull;
        const ei = self.entryFor(.prefill, self.ctx) orelse return error.BadManifest;
        const p0 = self.pos;
        self.putEmbeddings(findNamed(self.pin, "x").?.t, ids);
        self.common(self.pin, p0, n, self.man.TP);
        const so = findNamed(self.pin, "conv_sel_out").?.t;
        so.zero();
        for (0..3) |i| so.f16s()[i * so.stride(0) + (@as(usize, n) + i) * so.stride(1)] = 1;
        const valid = findNamed(self.pin, "valid").?.t;
        valid.zero();
        for (0..n) |i| valid.f16s()[i * valid.stride(1)] = 1;
        const bs = try self.plan(ei);
        try self.run(bs[0..self.chunks.len]);
        self.flip();
        // The head runs on the last real row only.
        const y = findNamed(self.chunks[self.chunks.len - 1].out[ei], "y").?.t;
        const hx = self.hx.t;
        hx.zero();
        for (0..self.cfg.hidden) |h| hx.f16s()[h * hx.stride(1)] = y.f16s()[h * y.stride(1) + (n - 1) * y.stride(3)];
        try self.run(bs[self.chunks.len..]);
        self.appendRows(ei, p0, n);
        self.last = ei;
        self.pending = 0;
        self.pos = p0 + n;
        self.hi = @max(self.hi, self.pos);
        return self.logitsRow(0);
    }

    pub fn logitsRow(self: *const Engine, r: usize) []const f16 {
        return self.logits.f16s()[r * self.logits.stride(0) ..][0..self.cfg.vocab];
    }

    /// Accepted K / V rows of entry `ei`'s last outputs into the cache at [p0, p0 + count).
    fn appendRows(self: *Engine, ei: usize, p0: u32, count: u32) void {
        if (count == 0) return;
        var key: [16]u8 = undefined;
        var codes: [1024]i8 = undefined;
        for (self.chunks) |ch| for (ch.att_j) |j| {
            const kn = findNamed(ch.out[ei], std.fmt.bufPrint(&key, "k{d}_new", .{j}) catch unreachable).?.t;
            const vn = findNamed(ch.out[ei], std.fmt.bufPrint(&key, "v{d}_new", .{j}) catch unreachable).?.t;
            const kc = findNamed(ch.kv, std.fmt.bufPrint(&key, "k{d}", .{j}) catch unreachable).?.t;
            const vc = findNamed(ch.kv, std.fmt.bufPrint(&key, "v{d}", .{j}) catch unreachable).?.t;
            const vs = if (self.man.kv_format == .v8) findNamed(ch.kv, std.fmt.bufPrint(&key, "vs{d}", .{j}) catch unreachable).?.t else null;
            const heads: usize = @intCast(kn.shape[0]);
            const hd: usize = @intCast(kn.shape[2]);
            var vals: [1024]f16 = undefined;
            for (0..heads) |h| for (0..count) |r| {
                const t = @as(usize, p0) + r;
                for (0..hd) |d| {
                    const kv = kn.f16s()[h * kn.stride(0) + r * kn.stride(1) + d * kn.stride(2)];
                    const at = if (self.man.keys_t) h * kc.stride(0) + d * kc.stride(1) + t * kc.stride(2) else h * kc.stride(0) + t * kc.stride(1) + d * kc.stride(2);
                    kc.f16s()[at] = kv;
                    vals[d] = vn.f16s()[h * vn.stride(0) + r * vn.stride(1) + d * vn.stride(2)];
                }
                const vat = h * vc.stride(0) + t * vc.stride(1);
                if (vs) |s| {
                    s.f16s()[h * s.stride(0) + t * s.stride(1)] = quantizeRow(vals[0..hd], codes[0..hd]);
                    for (0..hd) |d| vc.i8s()[vat + d * vc.stride(2)] = codes[d];
                } else {
                    for (0..hd) |d| vc.f16s()[vat + d * vc.stride(2)] = vals[d];
                }
            };
        };
    }

    /// Prompt tokens from `pos`: prefill blocks while more than T remain, then decode-entry calls.
    /// Returns the last token's logits row.
    /// `ingest_from`: positions from here on also go into the drafter's context (its window only sees
    /// the last W + R rows before the prompt's end, so earlier ones would be overwritten unused).
    pub fn feed(self: *Engine, ids: []const u32, stop: ?*const std.atomic.Value(bool), ingest_from: ?u32) ![]const f16 {
        var i: usize = 0;
        var out: []const f16 = &.{};
        while (i < ids.len) {
            if (stop) |s| if (s.load(.acquire)) return error.Cancelled;
            const r = ids.len - i;
            const blk_tp = @min(r, self.man.TP);
            if (r > self.man.T and self.man.TP > 0 and try self.fit(self.pos + @as(u32, @intCast(blk_tp))) and self.man.hasPrefill(self.ctx)) {
                const p0 = self.pos;
                out = try self.prefillBlock(ids[i .. i + blk_tp]);
                try self.ingest(p0, @intCast(blk_tp), ingest_from);
                try self.fed.appendSlice(self.allocator, ids[i .. i + blk_tp]);
                i += blk_tp;
                if (self.cache != null and self.pos >= self.next_cp) self.checkpoint();
            } else {
                const n = @min(r, self.man.T);
                _ = try self.call(ids[i .. i + n]);
                out = self.logitsRow(n - 1);
                try self.ingest(self.pos, @intCast(n), ingest_from);
                self.accept(@intCast(n));
                try self.fed.appendSlice(self.allocator, ids[i .. i + n]);
                i += n;
            }
        }
        return out;
    }

    /// Feed one generated token and return its logits row.
    /// A failure resets the state (a full context leaves it as it was).
    pub fn step(self: *Engine, token: u32) ![]const f16 {
        _ = self.call(&.{token}) catch |err| return if (err == error.ContextFull) err else self.dropState(err);
        self.accept(1);
        self.fed.append(self.allocator, token) catch |err| return self.dropState(err);
        return self.logitsRow(0);
    }

    /// Bring the state to `prompt`: continue the live state when the prompt extends it, or a cached
    /// checkpoint when that reaches further, else start over.
    /// Returns the last prompt token's logits and how many prompt tokens were already in the state.
    /// A cancelled sync leaves the state reset, so the next request starts over.
    pub fn sync(self: *Engine, prompt: []const u32, stop: ?*const std.atomic.Value(bool)) !Synced {
        const from: ?u32 = if (self.drafter) |d| @intCast(prompt.len -| (d.W + d.R)) else null;
        const n = self.fed.items.len;
        var start: usize = if (n > 0 and prompt.len > n and std.mem.eql(u32, self.fed.items, prompt[0..n])) n else 0;
        if (self.cache) |c| {
            c.mu.lockUncancelable(self.io);
            var failed: ?anyerror = null;
            if (c.lookup(prompt)) |hit| if (hit.end() > start) {
                if (self.restore(hit, prompt)) {
                    log.info("[coreai] prompt: {d} of {d} tokens from a checkpoint\n", .{ hit.end(), prompt.len });
                    start = hit.end();
                } else |err| failed = err;
            };
            c.mu.unlock(self.io);
            if (failed) |err| return self.dropState(err);
        }
        if (start == 0) self.reset(true);
        const cached = start;
        // A checkpoint before the prompt's last T tokens: a chat's next turn renders this turn's
        // generation prompt differently, so the prefix it shares ends about there.
        const turn = prompt.len -| self.man.T;
        if (self.cache != null and turn > start) {
            _ = self.feed(prompt[start..turn], stop, from) catch |err| return self.dropState(err);
            self.checkpoint();
            start = turn;
        }
        return .{ .logits = self.feed(prompt[start..], stop, from) catch |err| return self.dropState(err), .cached = cached };
    }

    // ── prefix cache ──

    /// Keep checkpoints of earlier sequences in `c` (the caller's): every `stride` positions, before
    /// each prompt's tail and after each request.
    pub fn useCache(self: *Engine, c: *prefix.Cache, stride: u32) void {
        std.debug.assert(stride > 0);
        self.cache = c;
        self.stride = stride;
        self.next_cp = stride;
    }

    /// Save the live state as a checkpoint under the one it continues. Without a cache or new
    /// tokens it does nothing; a failed save costs only the reuse.
    pub fn checkpoint(self: *Engine) void {
        const c = self.cache orelse return;
        std.debug.assert(self.fed.items.len == self.pos);
        self.next_cp = (self.pos / self.stride + 1) * self.stride;
        const start: u32 = if (self.node) |n| n.end() else 0;
        if (self.pos <= start) return;
        c.mu.lockUncancelable(self.io);
        defer c.mu.unlock(self.io);
        const toks = self.fed.items[start..self.pos];
        const n = c.child(self.node, toks) orelse blk: {
            const n = c.add(self.node, toks, self.kvLen(self.pos - start), self.stateLen()) catch |err| {
                log.warn("[coreai] checkpoint at {d} not saved: {s}\n", .{ self.pos, @errorName(err) });
                return;
            };
            self.moveKv(start, self.pos, n.kv, true);
            self.saveState(n.state.?);
            break :blk n;
        };
        if (n.state == null) self.saveState(c.restate(n, self.stateLen()) catch |err| {
            log.warn("[coreai] checkpoint at {d} not saved: {s}\n", .{ self.pos, @errorName(err) });
            return;
        });
        c.touch(n);
        self.follow(n);
        c.trim();
        log.debug("[coreai] checkpoint at {d}; cache {d} states, {d} MB\n", .{ self.pos, c.states, c.bytes >> 20 });
    }

    /// The live state to checkpoint `n` of `prompt` (cache locked).
    fn restore(self: *Engine, n: *prefix.Node, prompt: []const u32) !void {
        const end = n.end();
        self.hi = 0; // every row comes from the checkpoint: a context switch moves none
        if (!try self.fit(end)) return error.ContextFull;
        var at: ?*prefix.Node = n;
        while (at) |x| : (at = x.parent) self.moveKv(x.start, x.end(), x.kv, false);
        try self.loadState(n.state.?);
        self.pos = end;
        self.hi = end;
        self.fed.clearRetainingCapacity();
        try self.fed.appendSlice(self.allocator, prompt[0..end]);
        self.next_cp = (end / self.stride + 1) * self.stride;
        self.cache.?.touch(n);
        self.follow(n);
    }

    /// The live state now continues `n` (cache locked).
    fn follow(self: *Engine, n: ?*prefix.Node) void {
        const c = self.cache.?;
        if (n) |x| c.pin(x);
        if (self.node) |old| c.unpin(old);
        self.node = n;
    }

    fn kvLen(self: *const Engine, rows: u32) usize {
        var len: usize = 0;
        for (self.chunks) |ch| for (ch.kv) |kv| {
            len += rowsLen(kv.buf.t, self.tokenAxis(kv.name), rows);
        };
        return len;
    }

    /// KV rows [a, b) of every attention layer into `buf` (`out`) or back from it.
    fn moveKv(self: *Engine, a: u32, b: u32, buf: []u8, out: bool) void {
        var o: usize = 0;
        for (self.chunks) |ch| for (ch.kv) |kv| {
            const ax = self.tokenAxis(kv.name);
            const len = rowsLen(kv.buf.t, ax, b - a);
            moveRows(kv.buf.t, ax, a, b, buf[o..][0..len], out);
            o += len;
        };
    }

    /// The pending count and the DeltaNet states, then the drafter's context when it holds any.
    fn stateLen(self: *const Engine) usize {
        var len: usize = @sizeOf(u32);
        for (self.chunks) |ch| for (ch.st[self.par]) |n| {
            len += n.buf.t.nbytes;
        };
        if (self.drafter) |d| if (d.hasContext()) {
            len += d.snapLen();
        };
        return len;
    }

    fn saveState(self: *const Engine, out: []u8) void {
        @memcpy(out[0..@sizeOf(u32)], std.mem.asBytes(&self.pending));
        var o: usize = @sizeOf(u32);
        for (self.chunks) |ch| for (ch.st[self.par]) |n| {
            @memcpy(out[o..][0..n.buf.t.nbytes], n.buf.t.base[0..n.buf.t.nbytes]);
            o += n.buf.t.nbytes;
        };
        if (self.drafter) |d| if (o < out.len) d.save(out[o..]);
    }

    fn loadState(self: *Engine, in: []const u8) !void {
        @memcpy(std.mem.asBytes(&self.pending), in[0..@sizeOf(u32)]);
        var o: usize = @sizeOf(u32);
        for (self.chunks) |ch| for (ch.st[self.par]) |n| {
            @memcpy(n.buf.t.base[0..n.buf.t.nbytes], in[o..][0..n.buf.t.nbytes]);
            o += n.buf.t.nbytes;
        };
        if (self.drafter) |d| if (o < in.len) try d.load(in[o..]) else d.reset();
    }

    fn ingest(self: *Engine, p0: u32, n: u32, from: ?u32) !void {
        const d = self.drafter orelse return;
        const f = from orelse return;
        if (p0 + n <= f) return;
        const skip = if (f > p0) f - p0 else 0;
        const rows = self.features(n);
        const per = rows.len / n;
        try d.addContext(rows[skip * per ..], p0 + skip, n - skip);
    }

    /// The last call's first n tap rows, contiguous (row, tap, hidden).
    fn features(self: *Engine, n: usize) []f16 {
        const D = self.cfg.hidden;
        const per = self.tap_src.len * D;
        for (self.tap_src, 0..) |src, ti| {
            const y = findNamed(self.chunks[src.chunk].out[self.last], src.name).?.t;
            for (0..n) |r| for (0..D) |h| {
                self.feat[r * per + ti * D + h] = y.f16s()[h * y.stride(1) + r * y.stride(3)];
            };
        }
        return self.feat[0 .. n * per];
    }

    fn tapSources(self: *Engine) ![]TapSrc {
        const a = self.arena.allocator();
        const out = try a.alloc(TapSrc, self.man.taps.len);
        for (self.man.taps, out) |l, *dst| {
            dst.* = for (self.man.chunks, 0..) |c, ci| {
                if (c.last == l) break .{ .chunk = ci, .name = "y" };
                if (std.mem.indexOfScalar(u32, c.taps, l) != null) break .{ .chunk = ci, .name = try std.fmt.allocPrintSentinel(a, "tap{d}", .{l}, 0) };
            } else return error.BadManifest;
        }
        return out;
    }

    fn openDrafter(self: *Engine, io: std.Io, bundle: []const u8) ?*dflash.Drafter {
        const dir = std.fs.path.join(self.arena.allocator(), &.{ bundle, "drafter" }) catch return null;
        var probe = std.Io.Dir.cwd().openDir(io, dir, .{}) catch return null;
        probe.close(io);
        const d = dflash.Drafter.open(self.allocator, io, self.api, dir, self.cfg.vocab) catch |err| {
            log.warn("[coreai] drafter not loaded ({s}): one token per call\n", .{@errorName(err)});
            return null;
        };
        if (d.feat_dim != self.tap_src.len * self.cfg.hidden or d.T != self.man.T) {
            log.warn("[coreai] drafter does not match the target's taps: one token per call\n", .{});
            d.deinit();
            return null;
        }
        return d;
    }

    /// Core AI falls back to the GPU silently when the ANE compiler rejects a graph (two-pass softmax on
    /// an M5 Ultra, 64-row prefill at 32K+ contexts); this executable's compile cache says which packages it did.
    fn warnGpuFallbacks(self: *Engine, io: std.Io, bundle: []const u8) void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var pkgs: std.ArrayList([]const u8) = .empty;
        for (self.man.chunks) |c| pkgs.append(a, std.fs.path.join(a, &.{ bundle, "coreai", c.file }) catch return) catch return;
        pkgs.append(a, std.fs.path.join(a, &.{ bundle, "coreai", self.man.head_file }) catch return) catch return;
        if (self.drafter) |d| pkgs.append(a, d.path) catch return;
        var n: usize = 0;
        for (pkgs.items) |pkg| {
            if (packageOnGpu(io, a, pkg)) {
                if (n == 0) log.warn("[coreai] Core AI runs part of these packages on the GPU, not the ANE:\n", .{});
                log.warn("[coreai]   {s}\n", .{std.fs.path.basename(pkg)});
                n += 1;
            }
        }
        if (n > 0) log.warn("[coreai] {d} of {d} packages fall back (some or all of their entries)\n", .{ n, pkgs.items.len });
    }

    /// One speculative cycle after `anchor` (sampled, not yet in the model): draft T - 1, verify
    /// [anchor + drafts] in one call, accept while each draft is what the target draws (greedy: its
    /// argmax; sampled: with the target's probability of it, the first rejection resampling without
    /// it), commit anchor + the accepted drafts. An accepted stop token ends the cycle uncommitted.
    /// A failure resets the state (a full context leaves it as it was).
    pub fn cycle(self: *Engine, anchor: u32, s: Sampling, rng: std.Random, eos: []const u32) !Cycle {
        return self.cycleOnce(anchor, s, rng, eos) catch |err| return if (err == error.ContextFull) err else self.dropState(err);
    }

    fn cycleOnce(self: *Engine, anchor: u32, s: Sampling, rng: std.Random, eos: []const u32) !Cycle {
        const d = self.drafter.?;
        const T = self.man.T;
        const p = self.pos;
        if (!try self.fit(p + T)) return error.ContextFull;
        const since: u64 = @intCast(@max(self.verify_end.untilNow(self.io, .awake).nanoseconds, 0));
        if (since < DRAFT_GAP_NS) std.Io.sleep(self.io, .fromNanoseconds(@intCast(DRAFT_GAP_NS - since)), .awake) catch {};
        var ids: [8]u32 = undefined;
        ids[0] = anchor;
        try d.propose(anchor, p, self.emb.row(anchor), ids[1..T]);
        _ = try self.call(ids[0..T]);
        self.verify_end = std.Io.Timestamp.now(self.io, .awake);
        var k: usize = 0;
        var nxt: ?u32 = null;
        while (k < T - 1) : (k += 1) {
            const dk = ids[k + 1];
            if (verifyDraft(self.logitsRow(k), dk, s, rng, self.scratch)) |t| {
                nxt = t;
                break;
            }
            if (std.mem.indexOfScalar(u32, eos, dk) != null) {
                nxt = dk;
                break;
            }
        }
        const next = nxt orelse sample(self.logitsRow(T - 1), s, rng, self.scratch);
        const committed: u32 = @intCast(k + 1);
        try d.addContext(self.features(committed), p, committed);
        self.accept(committed);
        try self.fed.appendSlice(self.allocator, ids[0..committed]);
        var out: Cycle = .{ .tokens = undefined, .n = k + 1, .accepted = k };
        @memcpy(out.tokens[0..k], ids[1 .. k + 1]);
        out.tokens[k] = next;
        return out;
    }

    fn dropState(self: *Engine, err: anyerror) anyerror {
        self.reset(false);
        return err;
    }

    pub fn draw(self: *Engine, logits: []const f16, s: Sampling, rng: std.Random) u32 {
        return sample(logits, s, rng, self.scratch);
    }
};

pub const CacheLimits = struct { max_bytes: u64, max_states: u32, stride: u32 };

/// `bundle` loaded once and `slots` engines on it (`Engine.slot`) behind one worker, sharing one
/// prefix cache (none without `limits`; `max_bytes` 0 is no byte cap).
pub fn openPool(allocator: std.mem.Allocator, io: std.Io, bundle: []const u8, slots: u32, limits: ?CacheLimits, ctx: *anyopaque, run: *const fn (*anyopaque, *Engine, *anyopaque) void) !*Worker {
    const engines = try allocator.alloc(*Engine, @max(slots, 1));
    var opened: usize = 0;
    errdefer {
        var i = opened;
        while (i > 0) : (i -= 1) engines[i - 1].deinit();
        allocator.free(engines);
    }
    const cache: ?*prefix.Cache = if (limits) |l| blk: {
        const c = try allocator.create(prefix.Cache);
        c.* = .init(allocator, if (l.max_bytes == 0) std.math.maxInt(u64) else l.max_bytes, l.max_states);
        break :blk c;
    } else null;
    errdefer if (cache) |c| {
        c.deinit();
        allocator.destroy(c);
    };
    for (engines, 0..) |*e, i| {
        e.* = if (i == 0) try Engine.open(allocator, io, bundle) else try engines[0].slot();
        opened += 1;
        if (cache) |c| e.*.useCache(c, limits.?.stride);
    }
    if (engines.len > 1) log.info("[coreai] {d} slots on one load\n", .{engines.len});
    return Worker.start(allocator, io, engines, ctx, run);
}

/// One thread per engine, all taking jobs from one queue, in order. Core AI never touches MLX, so
/// requests run here beside the MLX inference thread instead of taking turns with it.
pub const Worker = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    engines: []*Engine,
    threads: []std.Thread = &.{},
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, engine: *Engine, job: *anyopaque) void,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(*anyopaque) = .empty,
    stopping: bool = false,

    pub fn start(allocator: std.mem.Allocator, io: std.Io, engines: []*Engine, ctx: *anyopaque, run: *const fn (*anyopaque, *Engine, *anyopaque) void) !*Worker {
        const w = try allocator.create(Worker);
        errdefer allocator.destroy(w);
        w.* = .{ .allocator = allocator, .io = io, .engines = engines, .ctx = ctx, .run = run };
        const threads = try allocator.alloc(std.Thread, engines.len);
        errdefer allocator.free(threads);
        for (threads, 0..) |*t, i| t.* = std.Thread.spawn(.{}, loop, .{ w, i }) catch |err| {
            w.threads = threads[0..i];
            w.stop();
            return err;
        };
        w.threads = threads;
        return w;
    }

    pub fn post(self: *Worker, job: *anyopaque) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        try self.queue.append(self.allocator, job);
        self.cond.signal(self.io);
    }

    /// Finish the posted jobs, join. The engines stay the owner's to close.
    pub fn deinit(self: *Worker) void {
        self.stop();
        self.allocator.free(self.threads);
        self.queue.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// What the pool holds beside the bundle: every slot's buffers and the prefix cache's byte cap.
    pub fn stateBytes(self: *const Worker) u64 {
        var n: u64 = 0;
        for (self.engines) |e| n += e.bufferBytes();
        if (self.engines[0].cache) |c| if (c.max_bytes != std.math.maxInt(u64)) {
            n += c.max_bytes;
        };
        return n;
    }

    /// `deinit`, then the engines, their shared cache and the engine list: an `openPool` worker.
    pub fn close(self: *Worker) void {
        const a = self.allocator;
        const engines = self.engines;
        const cache = engines[0].cache;
        self.deinit();
        var i = engines.len;
        while (i > 0) : (i -= 1) engines[i - 1].deinit();
        if (cache) |c| {
            c.deinit();
            a.destroy(c);
        }
        a.free(engines);
    }

    fn stop(self: *Worker) void {
        self.mu.lockUncancelable(self.io);
        self.stopping = true;
        self.cond.broadcast(self.io);
        self.mu.unlock(self.io);
        for (self.threads) |t| t.join();
    }

    fn loop(self: *Worker, i: usize) void {
        while (true) {
            self.mu.lockUncancelable(self.io);
            while (self.queue.items.len == 0 and !self.stopping) self.cond.waitUncancelable(self.io, &self.mu);
            // Stopping still runs what was posted: a dropped job would hold its slot's `in_pass`.
            if (self.queue.items.len == 0) {
                self.mu.unlock(self.io);
                return;
            }
            const job = self.queue.orderedRemove(0);
            self.mu.unlock(self.io);
            self.run(self.ctx, self.engines[i], job);
        }
    }
};

/// This executable's Core AI compile cache entry for the package at `pkg` (its `main.hash` names it).
fn cacheEntry(io: std.Io, a: std.mem.Allocator, pkg: []const u8) ?[]const u8 {
    var hash_buf: [64]u8 = undefined;
    const hash = std.Io.Dir.cwd().readFile(io, std.fs.path.join(a, &.{ pkg, "main.hash" }) catch return null, &hash_buf) catch return null;
    var os_buf: [32]u8 = undefined;
    var os_len: usize = os_buf.len;
    if (std.c.sysctlbyname("kern.osversion", &os_buf, &os_len, null, 0) != 0) return null;
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const exe = ffi.exeCacheName(&exe_buf) orelse return null;
    const home = std.c.getenv("HOME") orelse return null;
    return std.fmt.allocPrint(a, "{s}/Library/Caches/coreai-cache/{s}/{s}/{x}", .{ std.mem.span(home), std.mem.sliceTo(&os_buf, 0), exe, hash }) catch null;
}

/// Whether the cached specialization of the package at `pkg` holds GPU regions. False when the cache
/// cannot be read.
pub fn packageOnGpu(io: std.Io, a: std.mem.Allocator, pkg: []const u8) bool {
    const dir = cacheEntry(io, a, pkg) orelse return false;
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return false;
    defer d.close(io);
    var walker = d.walk(a) catch return false;
    defer walker.deinit();
    while (walker.next(io) catch return false) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.basename, ".mpsgraph")) continue;
        const bytes = e.dir.readFileAlloc(io, e.basename, a, .limited(1 << 28)) catch continue;
        if (std.mem.indexOf(u8, bytes, "_GPU_region_") != null) return true;
    }
    return false;
}

fn specs(a: std.mem.Allocator, v: ?std.json.Value) ![]Spec {
    const arr = v orelse return error.LoadFailed;
    if (arr != .array) return error.LoadFailed;
    const out = try a.alloc(Spec, arr.array.items.len);
    for (arr.array.items, out) |e, *s| {
        if (e != .object) return error.LoadFailed;
        const o = e.object;
        const dt = try str(o.get("dtype"));
        s.* = .{
            .name = try a.dupeSentinel(u8, try str(o.get("name")), 0),
            .dtype = if (std.mem.eql(u8, dt, "float16")) .f16 else if (std.mem.eql(u8, dt, "float32")) .f32 else if (std.mem.eql(u8, dt, "int8")) .i8 else if (std.mem.eql(u8, dt, "int32")) .i32 else if (std.mem.eql(u8, dt, "uint8")) .u8 else return error.LoadFailed,
            .shape = try i64s(a, o.get("shape")),
            .strides = try i64s(a, o.get("strides")),
        };
        if (s.shape.len != s.strides.len or s.shape.len == 0) return error.LoadFailed;
    }
    return out;
}

fn i64s(a: std.mem.Allocator, v: ?std.json.Value) ![]i64 {
    const x = v orelse return error.LoadFailed;
    if (x != .array) return error.LoadFailed;
    const out = try a.alloc(i64, x.array.items.len);
    for (x.array.items, out) |e, *d| d.* = if (e == .integer) e.integer else return error.LoadFailed;
    return out;
}

/// Rows [0, keep) along `axis` of a rank-2/3 tensor into another of the same layout but another length.
fn copyRows(a: std.mem.Allocator, src: Tensor, dst: Tensor, keep: u32, axis: usize) !void {
    const buf = try a.alloc(u8, rowsLen(src, axis, keep));
    defer a.free(buf);
    moveRows(src, axis, 0, keep, buf, true);
    moveRows(dst, axis, 0, keep, buf, false);
}

/// Head outputs and KV head dims past the fixed host arrays that hold them (`plan`, `appendRows`).
pub fn checkLimits(head_outputs: usize, kv_head_dim: usize) Error!void {
    if (head_outputs > 16 or kv_head_dim > 1024) return error.BadManifest;
}

fn rowsLen(t: Tensor, axis: usize, n: u32) usize {
    var len: usize = if (t.dtype == .i8) 1 else 2;
    for (t.shape, 0..) |d, i| len *= if (i == axis) n else @as(usize, @intCast(d));
    return len;
}

/// Rows [a, b) along `axis` of a rank-2/3 tensor into `buf` (`out`) or back from it, packed in the
/// tensor's axis order; contiguous runs move whole.
fn moveRows(t: Tensor, axis: usize, a: u32, b: u32, buf: []u8, out: bool) void {
    const es: usize = if (t.dtype == .i8) 1 else 2;
    var lim = [3]usize{ 1, 1, 1 };
    var st = [3]usize{ 0, 0, 0 };
    var lo = [3]usize{ 0, 0, 0 };
    const off = 3 - t.shape.len;
    for (t.shape, 0..) |d, i| {
        lim[off + i] = @intCast(d);
        st[off + i] = t.stride(i);
    }
    lim[off + axis] = b - a;
    lo[off + axis] = a;
    var o: usize = 0;
    for (0..lim[0]) |i| for (0..lim[1]) |j| {
        const at = ((lo[0] + i) * st[0] + (lo[1] + j) * st[1] + lo[2] * st[2]) * es;
        const runs: usize = if (st[2] == 1) 1 else lim[2];
        const run = if (st[2] == 1) lim[2] * es else es;
        for (0..runs) |k| {
            const mem = t.base[at + k * st[2] * es ..][0..run];
            if (out) @memcpy(buf[o..][0..run], mem) else @memcpy(mem, buf[o..][0..run]);
            o += run;
        }
    };
}

fn readFile(io: std.Io, a: std.mem.Allocator, dir: []const u8, rel: []const u8, limit: usize) ![]u8 {
    const p = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, rel });
    return std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(limit)) catch |err| {
        log.err("[coreai] read {s}: {s}\n", .{ p, @errorName(err) });
        return error.BadManifest;
    };
}

// ── tests ──

const testing = std.testing;

const test_manifest =
    \\{"T": 8, "TP": 64, "pend": 8, "ctxs": [16384, 8192], "pctxs": [8192, 16384],
    \\ "kv_len": {"8192": 8192, "16384": 16384}, "pkv_len": {"8192": 8192, "16384": 16000},
    \\ "kv_cache": {"format": "v8", "key_layout": "dim_token"}, "head": {"file": "head_T8.aimodel"},
    \\ "taps": [3], "chunks": [{"file": "chunk_L00-03.aimodel", "layers": [0, 3], "entries": ["v8_8k", "p64_8k"], "gdn_j": [0, 1, 2], "att_j": [3]}]}
;

test "manifest: sorted contexts, per-context KV rows, prefill only where its history matches" {
    var m = try parseManifest(testing.allocator, test_manifest);
    defer m.deinit();
    try testing.expectEqualSlices(u32, &.{ 8192, 16384 }, m.ctxs);
    try testing.expectEqual(@as(u32, 16384), m.cap(16384));
    try testing.expect(m.hasPrefill(8192));
    try testing.expect(!m.hasPrefill(16384));
    try testing.expect(m.keys_t);
    try testing.expect(!m.fp8);
    try testing.expectEqual(KvFormat.v8, m.kv_format);
    try testing.expectEqualSlices(u32, &.{3}, m.chunks[0].att_j);
    try testing.expectEqualStrings("head_T8.aimodel", m.head_file);
    try testing.expectEqualSlices(u32, &.{3}, m.taps);
    try testing.expectEqual(@as(u32, 3), m.chunks[0].last);
}

test "manifest: an unsupported KV format is refused by name" {
    const kv8 = "{\"T\": 8, \"pend\": 8, \"ctxs\": [8192], \"kv_cache\": {\"format\": \"kv8\"}, \"head\": {\"file\": \"h\"}, \"chunks\": [{\"file\": \"c\", \"layers\": [0, 3], \"entries\": [], \"gdn_j\": [], \"att_j\": []}]}";
    try testing.expectError(error.UnsupportedKvFormat, parseManifest(testing.allocator, kv8));
}

test "FP8 attention forms, and the chip generation from the CPU brand" {
    try testing.expect(fp8Forms("s8,s8b,sm8,pvf8"));
    try testing.expect(fp8Forms("pvf5"));
    try testing.expect(!fp8Forms("s8,s8b"));
    try testing.expect(!fp8Forms(""));
    try testing.expectEqual(@as(?u32, 5), chipGeneration("Apple M5 Ultra"));
    try testing.expectEqual(@as(?u32, 6), chipGeneration("Apple M6"));
    try testing.expectEqual(@as(?u32, 12), chipGeneration("Apple M12 Max"));
    try testing.expectEqual(@as(?u32, null), chipGeneration("Intel(R) Core(TM) i9"));
}

test "text config: rotary dims follow partial_rotary_factor" {
    const cfg = try parseTextConfig(testing.allocator,
        \\{"model_type": "qwen3_5", "text_config": {"hidden_size": 5120, "vocab_size": 248320, "head_dim": 256,
        \\ "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000}}}
    );
    try testing.expectEqual(@as(u32, 64), cfg.rot);
    try testing.expectEqual(@as(u32, 5120), cfg.hidden);
    try testing.expectEqual(@as(f64, 1e7), cfg.rope_theta);
    // This runtime is the qwen3_5 graph's: another family is refused by name.
    try testing.expectError(error.UnsupportedArch, parseTextConfig(testing.allocator,
        \\{"model_type": "llama", "hidden_size": 4096, "vocab_size": 128256, "head_dim": 128, "rope_parameters": {"rope_theta": 500000}}
    ));
}

test "limits: head outputs and KV head dims must fit the host arrays" {
    try checkLimits(16, 1024);
    try testing.expectError(error.BadManifest, checkLimits(17, 256));
    try testing.expectError(error.BadManifest, checkLimits(4, 1025));
}

test "bundle files: manifest and config make a bundle; its bill is packages, drafter and embeddings" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "coreai/chunk_L00-03.aimodel");
    try tmp.dir.createDirPath(io, "drafter/dflash2.aimodel");
    try tmp.dir.createDirPath(io, "model");
    try tmp.dir.writeFile(io, .{ .sub_path = "coreai/manifest.json", .data = "{}" });
    try testing.expect(!isBundleDir(io, tmp.dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "model/config.json", .data = "{}" });
    try testing.expect(isBundleDir(io, tmp.dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "coreai/chunk_L00-03.aimodel/main.mlirb", .data = &@as([100]u8, @splat('x')) });
    try tmp.dir.writeFile(io, .{ .sub_path = "drafter/dflash2.aimodel/main.mlirb", .data = &@as([10]u8, @splat('x')) });
    try tmp.dir.writeFile(io, .{ .sub_path = "model/embed_tokens_fp16.npy", .data = &@as([5]u8, @splat('x')) });
    try testing.expectEqual(@as(?u64, 115), bundleBytes(io, tmp.dir));
}

test "entry names: decode and prefill entries with their context" {
    const d = parseEntryName("v8_16k").?;
    try testing.expect(!d.prefill);
    try testing.expectEqual(@as(u32, 16384), d.ctx);
    try testing.expect(parseEntryName("p64_8k").?.prefill);
    try testing.expect(parseEntryName("h8") == null);
}

test "compile cache entry: named by the package's main.hash, in hex; none without one" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const pkg = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expect(cacheEntry(io, arena.allocator(), pkg) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "main.hash", .data = "\xab\x01" });
    const dir = cacheEntry(io, arena.allocator(), pkg).?;
    try testing.expect(std.mem.indexOf(u8, dir, "/Library/Caches/coreai-cache/") != null);
    try testing.expect(std.mem.endsWith(u8, dir, "/ab01"));
}

test "npy header: FP16 C-order 2-D, data offset after the header" {
    var buf: [128]u8 = undefined;
    const dict = "{'descr': '<f2', 'fortran_order': False, 'shape': (248320, 5120), }";
    @memcpy(buf[0..6], "\x93NUMPY");
    buf[6] = 1;
    buf[7] = 0;
    std.mem.writeInt(u16, buf[8..10], dict.len, .little);
    @memcpy(buf[10 .. 10 + dict.len], dict);
    const h = try parseNpyHeader(buf[0 .. 10 + dict.len]);
    try testing.expectEqual(@as(u64, 248320), h.rows);
    try testing.expectEqual(@as(u64, 5120), h.cols);
    try testing.expectEqual(@as(usize, 10 + dict.len), h.data_offset);
    @memcpy(buf[10 .. 10 + 14], "{'descr': '<f4");
    try testing.expectError(error.BadEmbeddings, parseNpyHeader(buf[0 .. 10 + dict.len]));
}

test "padding rows repeat the last real position" {
    try testing.expectEqual(@as(u32, 10), rowPos(10, 3, 0));
    try testing.expectEqual(@as(u32, 12), rowPos(10, 3, 2));
    try testing.expectEqual(@as(u32, 12), rowPos(10, 3, 7));
}

test "rope inverse frequencies match 1 / theta^(i / rot)" {
    var inv: [32]f64 = undefined;
    ropeInv(&inv, 64, 1e7);
    try testing.expectEqual(@as(f64, 1), inv[0]);
    try testing.expectApproxEqRel(1.0 / std.math.pow(f64, 1e7, 62.0 / 64.0), inv[31], 1e-15);
}

test "INT8 values: half rounds to even, codes clip at 127, the stored scale is the one used" {
    try testing.expectEqual(@as(f32, 2), rintEven(2.5));
    try testing.expectEqual(@as(f32, 4), rintEven(3.5));
    try testing.expectEqual(@as(f32, -2), rintEven(-2.5));
    const vals = [_]f16{ 1.0, -0.5, 0.25, 0 };
    var codes: [4]i8 = undefined;
    const s = quantizeRow(&vals, &codes);
    try testing.expectEqual(@as(f16, @floatCast(1.0 / 127.0)), s);
    try testing.expectEqual(@as(i8, 127), codes[0]);
    try testing.expectEqual(@as(i8, @intFromFloat(rintEven(-0.5 / @as(f32, s)))), codes[1]);
    try testing.expectEqual(@as(i8, 0), codes[3]);
    const zeros = [_]f16{ 0, 0 };
    var zc: [2]i8 = undefined;
    try testing.expectEqual(@as(f16, @floatCast(1e-6)), quantizeRow(&zeros, &zc));
}

test "sampling: temperature 0 is the first argmax; top-k 1 always draws the top token" {
    const logits = [_]f16{ 0.5, 3.0, 3.0, -1 };
    var scratch: [4]Cand = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    try testing.expectEqual(@as(u32, 1), sample(&logits, .{}, prng.random(), &scratch));
    for (0..16) |_| try testing.expectEqual(@as(u32, 1), sample(&logits, .{ .temperature = 1, .top_k = 1 }, prng.random(), &scratch));
    const peaked = [_]f16{ 0, 20, 0, 0 };
    for (0..16) |_| try testing.expectEqual(@as(u32, 1), sample(&peaked, .{ .temperature = 0.7, .top_p = 0.9 }, prng.random(), &scratch));
}

test "dist: temperature 0 is greedy; min-p and top-p cut the tail" {
    const logits = [_]f16{ 0, 4, 3, 2 };
    var scratch: [4]Cand = undefined;
    try testing.expect(dist(&logits, .{ .temperature = 0 }, &scratch) == null);
    // Relative probabilities 1, e^-1, e^-2, e^-4: min-p 0.3 keeps the first two.
    const mp = dist(&logits, .{ .temperature = 1, .min_p = 0.3 }, &scratch).?;
    try testing.expectEqual(@as(usize, 2), mp.len);
    try testing.expectEqual(@as(u32, 1), mp[0].id);
    // The top token alone already holds 66% of the mass: top-p 0.5 keeps it alone.
    try testing.expectEqual(@as(usize, 1), dist(&logits, .{ .temperature = 1, .top_p = 0.5 }, &scratch).?.len);
    try testing.expectEqual(@as(usize, 4), dist(&logits, .{ .temperature = 1 }, &scratch).?.len);
}

test "verifyDraft: greedy keeps the argmax only; sampled keeps a likely draft, replaces an absent one" {
    const logits = [_]f16{ 0, 16, 1, 0 };
    var scratch: [4]Cand = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    try testing.expect(verifyDraft(&logits, 1, .{ .temperature = 0 }, rng, &scratch) == null);
    try testing.expectEqual(@as(?u32, 1), verifyDraft(&logits, 2, .{ .temperature = 0 }, rng, &scratch));
    // Token 1 holds all but ~1e-6 of the mass at temperature 1: drafted, it stands.
    for (0..20) |_| try testing.expect(verifyDraft(&logits, 1, .{ .temperature = 1 }, rng, &scratch) == null);
    // A draft outside the top-k is always replaced, and never by itself.
    for (0..20) |_| {
        const t = verifyDraft(&logits, 3, .{ .temperature = 1, .top_k = 2 }, rng, &scratch).?;
        try testing.expect(t == 1 or t == 2);
    }
}

test "top-k keeps the k largest, descending" {
    const logits = [_]f16{ 1, 5, 3, 4, 2 };
    var scratch: [5]Cand = undefined;
    const got = topK(&logits, 3, &scratch);
    try testing.expectEqual(@as(usize, 3), got.len);
    try testing.expectEqual(@as(u32, 1), got[0].id);
    try testing.expectEqual(@as(u32, 3), got[1].id);
    try testing.expectEqual(@as(u32, 2), got[2].id);
}

test "copyRows moves the kept prefix along the token axis only" {
    var src_v = [_]i8{ 1, 2, 3, 4, 5, 6 }; // (1, 3, 2): tokens along axis 1
    var dst_v: [8]i8 = @splat(0); // (1, 4, 2)
    const src: Tensor = .{ .base = @ptrCast(&src_v), .dtype = .i8, .shape = &.{ 1, 3, 2 }, .strides = &.{ 6, 2, 1 }, .nbytes = 6 };
    const dst: Tensor = .{ .base = @ptrCast(&dst_v), .dtype = .i8, .shape = &.{ 1, 4, 2 }, .strides = &.{ 8, 2, 1 }, .nbytes = 8 };
    try copyRows(testing.allocator, src, dst, 2, 1);
    try testing.expectEqualSlices(i8, &.{ 1, 2, 3, 4, 0, 0, 0, 0 }, &dst_v);
}

test "moveRows: rows [a, b) out along either token axis and back, padding untouched" {
    // Keys transposed (2 heads, 2 dims, 4 tokens, row pitch 5): tokens are the inner axis.
    var k = [_]i8{ 1, 2, 3, 4, 99, 5, 6, 7, 8, 99, 9, 10, 11, 12, 99, 13, 14, 15, 16, 99 };
    const kt: Tensor = .{ .base = @ptrCast(&k), .dtype = .i8, .shape = &.{ 2, 2, 4 }, .strides = &.{ 10, 5, 1 }, .nbytes = 20 };
    var buf: [8]u8 = undefined;
    try testing.expectEqual(@as(usize, 8), rowsLen(kt, 2, 2));
    moveRows(kt, 2, 1, 3, &buf, true);
    try testing.expectEqualSlices(u8, &.{ 2, 3, 6, 7, 10, 11, 14, 15 }, &buf);
    var back: [20]i8 = @splat(0);
    const bt: Tensor = .{ .base = @ptrCast(&back), .dtype = .i8, .shape = &.{ 2, 2, 4 }, .strides = &.{ 10, 5, 1 }, .nbytes = 20 };
    moveRows(bt, 2, 1, 3, &buf, false);
    try testing.expectEqualSlices(i8, &.{ 0, 2, 3, 0, 0, 0, 6, 7, 0, 0, 0, 10, 11, 0, 0, 0, 14, 15, 0, 0 }, &back);
    // Values (1 head, 3 tokens, 2 dims) along axis 1, and a 2-D scale row (1 head, 3 tokens) as f16.
    var v = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const vt: Tensor = .{ .base = @ptrCast(&v), .dtype = .i8, .shape = &.{ 1, 3, 2 }, .strides = &.{ 6, 2, 1 }, .nbytes = 6 };
    moveRows(vt, 1, 2, 3, buf[0..2], true);
    try testing.expectEqualSlices(u8, &.{ 5, 6 }, buf[0..2]);
    var s = [_]f16{ 0.5, 1.5, 2.5 };
    const sv: Tensor = .{ .base = @ptrCast(&s), .dtype = .f16, .shape = &.{ 1, 3 }, .strides = &.{ 3, 1 }, .nbytes = 6 };
    try testing.expectEqual(@as(usize, 4), rowsLen(sv, 1, 2));
    moveRows(sv, 1, 0, 2, buf[0..4], true);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(s[0..2]), buf[0..4]);
}

test "pool: one thread per engine, both busy at once on different engines" {
    const Rec = struct {
        mu: std.Io.Mutex = .init,
        started: usize = 0,
        met: usize = 0,
        engines: [2]usize = .{ 0, 0 },
        fn run(ctx: *anyopaque, engine: *Engine, _: *anyopaque) void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            r.mu.lockUncancelable(testing.io);
            r.engines[r.started] = @intFromPtr(engine);
            r.started += 1;
            r.mu.unlock(testing.io);
            // Each job waits (up to 5 s) for the other to start: only parallel threads meet.
            for (0..5000) |_| {
                r.mu.lockUncancelable(testing.io);
                const both = r.started == 2;
                if (both) r.met += 1;
                r.mu.unlock(testing.io);
                if (both) return;
                std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    var rec: Rec = .{};
    var two = [_]*Engine{ @ptrFromInt(64), @ptrFromInt(128) };
    const w = try Worker.start(testing.allocator, testing.io, &two, &rec, Rec.run);
    try w.post(@ptrFromInt(8));
    try w.post(@ptrFromInt(16));
    while (true) {
        rec.mu.lockUncancelable(testing.io);
        const done = rec.met == 2;
        rec.mu.unlock(testing.io);
        if (done) break;
        std.Thread.yield() catch {};
    }
    w.deinit();
    try testing.expect(rec.engines[0] != rec.engines[1]);
}

test "worker runs posted jobs in order, off the posting thread, all of them when stopped" {
    const Rec = struct {
        mu: std.Io.Mutex = .init,
        got: [3]usize = undefined,
        n: usize = 0,
        poster: std.Thread.Id,
        off_thread: bool = true,
        fn run(ctx: *anyopaque, _: *Engine, job: *anyopaque) void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            r.mu.lockUncancelable(testing.io);
            defer r.mu.unlock(testing.io);
            r.got[r.n] = @intFromPtr(job);
            r.n += 1;
            if (std.Thread.getCurrentId() == r.poster) r.off_thread = false;
        }
    };
    var rec: Rec = .{ .poster = std.Thread.getCurrentId() };
    var one = [_]*Engine{undefined};
    const w = try Worker.start(testing.allocator, testing.io, &one, &rec, Rec.run);
    for ([_]usize{ 8, 16, 24 }) |j| try w.post(@ptrFromInt(j));
    w.deinit(); // stopping at once still runs every posted job
    try testing.expectEqualSlices(usize, &.{ 8, 16, 24 }, &rec.got);
    try testing.expect(rec.off_thread);
}

test "bundle: greedy decode reproduces Forge's tokens (COREAI_TEST_BUNDLE + COREAI_TEST_REF)" {
    const bundle = std.c.getenv("COREAI_TEST_BUNDLE") orelse return error.SkipZigTest;
    const ref_path = std.c.getenv("COREAI_TEST_REF") orelse return error.SkipZigTest;
    const io = testing.io;
    const eng = try Engine.open(testing.allocator, io, std.mem.span(bundle));
    defer eng.deinit();
    const ref_json = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(ref_path), testing.allocator, .limited(1 << 24));
    defer testing.allocator.free(ref_json);
    const ref = try std.json.parseFromSlice(std.json.Value, testing.allocator, ref_json, .{});
    defer ref.deinit();
    var it = ref.value.object.iterator();
    while (it.next()) |case| {
        const prompt = try u32s(testing.allocator, case.value_ptr.object.get("prompt"));
        defer testing.allocator.free(prompt);
        const want = try u32s(testing.allocator, case.value_ptr.object.get("greedy"));
        defer testing.allocator.free(want);
        eng.reset(true);
        var logits = try eng.feed(prompt, null, null);
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(testing.allocator);
        for (want) |_| {
            const t = argmax(logits);
            try got.append(testing.allocator, t);
            logits = try eng.step(t);
        }
        try testing.expectEqualSlices(u32, want, got.items);

        // Greedy speculative decoding commits exactly the target's own choices.
        if (eng.drafter == null) continue;
        eng.reset(true);
        const synced = try eng.sync(prompt, null);
        var spec: std.ArrayList(u32) = .empty;
        defer spec.deinit(testing.allocator);
        try spec.append(testing.allocator, argmax(synced.logits));
        var prng = std.Random.DefaultPrng.init(1);
        var cycles: usize = 0;
        var accepted: usize = 0;
        while (spec.items.len < want.len) {
            const c = try eng.cycle(spec.items[spec.items.len - 1], .{}, prng.random(), &.{});
            try spec.appendSlice(testing.allocator, c.tokens[0..c.n]);
            cycles += 1;
            accepted += c.accepted;
        }
        try testing.expectEqualSlices(u32, want, spec.items[0..want.len]);
        std.debug.print("{s}: {d} cycles, {d} of {d} drafts accepted\n", .{ case.key_ptr.*, cycles, accepted, 7 * cycles });
    }
}

test "bundle: a restored checkpoint continues like the live state (COREAI_TEST_BUNDLE + COREAI_TEST_REF)" {
    const bundle = std.c.getenv("COREAI_TEST_BUNDLE") orelse return error.SkipZigTest;
    const ref_path = std.c.getenv("COREAI_TEST_REF") orelse return error.SkipZigTest;
    const io = testing.io;
    const a = testing.allocator;
    const eng = try Engine.open(a, io, std.mem.span(bundle));
    defer eng.deinit();
    var cache = prefix.Cache.init(a, std.math.maxInt(u64), 64);
    defer cache.deinit();
    eng.useCache(&cache, 256);
    const ref_json = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(ref_path), a, .limited(1 << 24));
    defer a.free(ref_json);
    const ref = try std.json.parseFromSlice(std.json.Value, a, ref_json, .{});
    defer ref.deinit();
    var it = ref.value.object.iterator();
    const prompt = try u32s(a, it.next().?.value_ptr.object.get("prompt"));
    defer a.free(prompt);
    // Long enough for 256-token checkpoints and 64-row prefill blocks.
    var long: std.ArrayList(u32) = .empty;
    defer long.deinit(a);
    while (long.items.len < 600) try long.appendSlice(a, prompt);
    const L = long.items.len;

    // Live: 24 greedy tokens, with a checkpoint after 16 as a finished request leaves one.
    const s1 = try eng.sync(long.items, null);
    try testing.expectEqual(@as(usize, 0), s1.cached);
    var live: [24]u32 = undefined;
    var logits = s1.logits;
    for (&live, 0..) |*t, i| {
        if (i == 16) eng.checkpoint();
        t.* = argmax(logits);
        logits = try eng.step(t.*);
    }

    // The same prompt again: from the checkpoint before its last T tokens.
    eng.reset(false);
    const s2 = try eng.sync(long.items, null);
    try testing.expectEqual(L - eng.man.T, s2.cached);
    logits = s2.logits;
    for (live) |want| {
        try testing.expectEqual(want, argmax(logits));
        logits = try eng.step(want);
    }

    // The prompt and 17 generated tokens: from the checkpoint at +16, plain and speculative.
    var p3: std.ArrayList(u32) = .empty;
    defer p3.deinit(a);
    try p3.appendSlice(a, long.items);
    try p3.appendSlice(a, live[0..17]);
    eng.reset(false);
    const s3 = try eng.sync(p3.items, null);
    try testing.expectEqual(L + 16, s3.cached);
    logits = s3.logits;
    for (live[17..]) |want| {
        try testing.expectEqual(want, argmax(logits));
        logits = try eng.step(want);
    }
    if (eng.drafter == null) return;
    eng.reset(false);
    const s4 = try eng.sync(p3.items, null);
    try testing.expectEqual(L + 16, s4.cached);
    var spec: std.ArrayList(u32) = .empty;
    defer spec.deinit(a);
    try spec.append(a, argmax(s4.logits));
    var prng = std.Random.DefaultPrng.init(1);
    while (spec.items.len < live.len - 17) {
        const c = try eng.cycle(spec.items[spec.items.len - 1], .{}, prng.random(), &.{});
        try spec.appendSlice(a, c.tokens[0..c.n]);
    }
    try testing.expectEqualSlices(u32, live[17..], spec.items[0 .. live.len - 17]);
}

test "bundle: a slot on the same load decodes like its base, and neither disturbs the other (COREAI_TEST_BUNDLE + COREAI_TEST_REF)" {
    const bundle = std.c.getenv("COREAI_TEST_BUNDLE") orelse return error.SkipZigTest;
    const ref_path = std.c.getenv("COREAI_TEST_REF") orelse return error.SkipZigTest;
    const io = testing.io;
    const a = testing.allocator;
    const eng = try Engine.open(a, io, std.mem.span(bundle));
    defer eng.deinit();
    const other = try eng.slot();
    defer other.deinit();
    const ref_json = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(ref_path), a, .limited(1 << 24));
    defer a.free(ref_json);
    const ref = try std.json.parseFromSlice(std.json.Value, a, ref_json, .{});
    defer ref.deinit();
    var it = ref.value.object.iterator();
    const first = it.next().?.value_ptr.object;
    const second = (it.next() orelse return error.SkipZigTest).value_ptr.object;
    const p1 = try u32s(a, first.get("prompt"));
    defer a.free(p1);
    const w1 = try u32s(a, first.get("greedy"));
    defer a.free(w1);
    const p2 = try u32s(a, second.get("prompt"));
    defer a.free(p2);
    const w2 = try u32s(a, second.get("greedy"));
    defer a.free(w2);
    // Interleaved token by token: each engine keeps its own state.
    var l1 = (try eng.sync(p1, null)).logits;
    var l2 = (try other.sync(p2, null)).logits;
    for (0..@min(w1.len, w2.len)) |i| {
        try testing.expectEqual(w1[i], argmax(l1));
        try testing.expectEqual(w2[i], argmax(l2));
        l1 = try eng.step(w1[i]);
        l2 = try other.step(w2[i]);
    }
}
