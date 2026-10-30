//! NVIDIA ModelOpt checkpoints (`quant_method: modelopt`), served as published: MXFP8 linears
//! reinterpreted in place, NVFP4 routed experts stacked into banks with their csf-compressed
//! scales decoded at load, per-expert `weight_scale_2` kept beside each bank.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const crt = @import("../../util/crt.zig");
const Weights = @import("../../weights/resident.zig").Weights;
const sushi_pack = @import("../exl3/sushi_pack.zig");
const status = @import("../../util/system.zig");
const LayerSpan = @import("../../weights/shard.zig").LayerSpan;
const Shard = @import("../../weights/shard.zig").Shard;
const headOnly = @import("../../weights/shard.zig").headOnly;

/// True for a config whose `quantization_config.quant_method` is `modelopt`.
pub fn declared(root: std.json.ObjectMap) bool {
    const qc = root.get("quantization_config") orelse return false;
    if (qc != .object) return false;
    const m = qc.object.get("quant_method") orelse return false;
    return m == .string and std.mem.eql(u8, m.string, "modelopt");
}

/// Tensors the MLX load skips: routed experts (`buildBanks` reads them from disk), the n-gram
/// shards (read in place by the PLE gather) and activation scales (16-bit activations here).
pub fn skipsAtLoad(key: []const u8) bool {
    return std.mem.indexOf(u8, key, ".mlp.experts.") != null or
        std.mem.indexOf(u8, key, ".ngram_embedding.") != null or
        std.mem.endsWith(u8, key, ".input_scale");
}

/// The mlx-lm name of a ModelOpt tensor (the layout our Flash Next packs use). Trunk layer
/// `mtp_layer` (GLM's NextN layer past the trunk) becomes `mtp.0`, its block under `mtp.0.block`.
pub fn mlxName(a: std.mem.Allocator, key: []const u8, mtp_layer: ?u32) ![]u8 {
    const prefixes = [_][2][]const u8{ .{ "model.language_model.", "language_model.model." }, .{ "lm_head.", "language_model.lm_head." }, .{ "mtp.", "language_model.mtp." } };
    var head: []const u8 = "";
    var rest = key;
    var lbuf: [64]u8 = undefined;
    const mtp = if (mtp_layer) |n| std.fmt.bufPrint(&lbuf, "model.language_model.layers.{d}", .{n}) catch unreachable else "";
    if (mtp.len > 0 and std.mem.startsWith(u8, key, mtp) and (key.len == mtp.len or key[mtp.len] == '.')) {
        rest = if (key.len == mtp.len) "" else key[mtp.len + 1 ..];
        head = "language_model.mtp.0.block";
        for ([_][]const u8{ "eh_proj.", "enorm.", "hnorm.", "shared_head." }) |p| if (std.mem.startsWith(u8, rest, p)) {
            head = "language_model.mtp.0.";
            if (p[0] == 's') rest = rest[p.len..];
        };
        if (head[head.len - 1] != '.' and rest.len > 0) head = "language_model.mtp.0.block.";
    } else for (prefixes) |p| if (std.mem.startsWith(u8, key, p[0])) {
        head = p[1];
        rest = key[p[0].len..];
        break;
    };
    const suffix = "weight_scale";
    if (std.mem.endsWith(u8, rest, suffix) and (rest.len == suffix.len or rest[rest.len - suffix.len - 1] == '.'))
        return std.fmt.allocPrint(a, "{s}{s}scales", .{ head, rest[0 .. rest.len - suffix.len] });
    return std.fmt.allocPrint(a, "{s}{s}", .{ head, rest });
}

/// Renames every loaded tensor to the mlx-lm layout, reinterprets fp weight bytes as the u32
/// words MLX packs, dequantizes the vision tower's linears to bf16 (its encoder runs dense),
/// then adds the routed-expert banks read from `model_dir`.
pub fn adapt(weights: *Weights, model_dir: []const u8, mtp_layer: ?u32, clamped: bool, span: ?LayerSpan, shard: ?Shard) !void {
    const a = weights.allocator;
    // The loads' own stream: a CPU-stream view of a GPU load moves its buffer to managed memory.
    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);
    var old = weights.map;
    weights.map = std.StringHashMap(mlx.mlx_array).init(a);
    defer {
        var it = old.iterator();
        while (it.next()) |kv| {
            _ = mlx.mlx_array_free(kv.value_ptr.*);
            a.free(kv.key_ptr.*);
        }
        old.deinit();
    }
    var it = old.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (std.mem.endsWith(u8, key, ".weight_scale") or std.mem.endsWith(u8, key, ".weight_scale_2")) continue;
        var buf: [512]u8 = undefined;
        const scale_key = std.fmt.bufPrint(&buf, "{s}_scale", .{key}) catch return error.NameTooLong;
        const scales = if (std.mem.endsWith(u8, key, ".weight")) old.get(scale_key) else null;
        const name = try mlxName(a, key, mtp_layer);
        errdefer a.free(name);
        if (scales == null) {
            const sh = mlx.getShape(kv.value_ptr.*);
            if (std.mem.endsWith(u8, key, "conv1d.weight") and sh.len == 3 and sh[2] != 1) {
                // HF [C, 1, K]; MLX convolves [C, K, 1] (mlx-lm's sanitize).
                var t = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(t);
                try mlx.check(mlx.mlx_swapaxes(&t, kv.value_ptr.*, 1, 2, s));
                try weights.map.put(name, t);
                continue;
            }
            try putShared(weights, name, kv.value_ptr.*);
            continue;
        }
        if (mlx.mlx_array_dtype(scales.?) == .float32) {
            try weights.map.put(name, try dequantBlockFp8(kv.value_ptr.*, scales.?, s));
            continue;
        }
        var words = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(words);
        try mlx.check(mlx.mlx_view(&words, kv.value_ptr.*, .uint32, s));
        var buf2: [512]u8 = undefined;
        const global = old.get(std.fmt.bufPrint(&buf2, "{s}_scale_2", .{key}) catch return error.NameTooLong);
        // The vision encoder runs dense, and GLM's `kv_b_proj` is split into dense halves later.
        if (std.mem.startsWith(u8, key, "model.visual.") or std.mem.endsWith(u8, key, ".kv_b_proj.weight")) {
            try weights.map.put(name, try dequantDense(words, scales.?, global, s));
            continue;
        }
        if (global != null) return error.ModelOptUnsupportedGlobalScale;
        try putShared(weights, name, words);
        const sname = try mlxName(a, scale_key, mtp_layer);
        errdefer a.free(sname);
        try putShared(weights, sname, scales.?);
    }
    try buildBanks(weights, model_dir, mtp_layer, status.memoryPressureLevel, clamped, span, shard);
}

/// FP8_PB_WO: e4m3 codes `[R, C]` with one f32 scale per 128x128 tile (`[R/128, 1, C/128, 1]`), as bf16.
fn dequantBlockFp8(codes: mlx.mlx_array, scales: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(codes);
    const ssh = mlx.getShape(scales);
    if (sh.len != 2 or ssh.len != 4 or ssh[1] != 1 or ssh[3] != 1) return error.ModelOptBlockScales;
    var grid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grid);
    try mlx.check(mlx.mlx_reshape(&grid, scales, &[_]c_int{ ssh[0], ssh[2] }, 2, s));
    const slabs = try sushi_pack.dequantSlabs(codes, grid, 1, @intCast(sh[0]), s);
    defer _ = mlx.mlx_array_free(slabs);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, slabs, sh.ptr, 2, s));
    return out;
}

/// A ModelOpt linear as bf16: NVFP4 when it carries `weight_scale_2`, else MXFP8. ModelOpt's
/// `weight_scale_2` is amax / (448 * 6); MLX's nvfp4 `global_scale` is the amax itself.
fn dequantDense(words: mlx.mlx_array, scales: mlx.mlx_array, global: ?mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const nv = global != null;
    var amax = mlx.mlx_array{ .ctx = null };
    defer if (amax.ctx != null) {
        _ = mlx.mlx_array_free(amax);
    };
    if (global) |g| {
        const k = mlx.mlx_array_new_float(448.0 * 6.0);
        defer _ = mlx.mlx_array_free(k);
        amax = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&amax, g, k, s));
    }
    var dense = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_dequantize(&dense, words, scales, .{ .ctx = null }, mlx.mlx_optional_int.some(if (nv) 16 else 32), mlx.mlx_optional_int.some(if (nv) 4 else 8), if (nv) "nvfp4" else "mxfp8", amax, .{ .value = .bfloat16, .has_value = true }, s));
    return dense;
}

fn putShared(weights: *Weights, name: []u8, v: mlx.mlx_array) !void {
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, v));
    try weights.map.put(name, h);
}

const Loc = struct { file: u32, off: u64, len: u64, rows: u32, cols: u32 };

const Span = struct { start: u64, end: u64 };

/// The pages that cover `len` bytes at `off`. `F_NOCACHE` only holds for page-aligned offsets,
/// lengths and buffers; tensors sit at arbitrary offsets, and an unaligned read goes through the
/// page cache whatever the flag says (a 165 GB load filled ~120 GB of cache).
fn alignedSpan(off: u64, len: u64) Span {
    const page: u64 = std.heap.page_size_min;
    return .{ .start = off / page * page, .end = std.mem.alignForward(u64, off + len, page) };
}

/// Byte locations of every routed-expert tensor, read from the shard headers.
const Source = struct {
    arena: std.heap.ArenaAllocator,
    fds: std.ArrayList(c_int) = .empty,
    locs: std.StringHashMapUnmanaged(Loc) = .empty,

    fn deinit(self: *Source) void {
        for (self.fds.items) |fd| crt.close(fd);
        self.arena.deinit();
    }

    fn get(self: *const Source, comptime fmt: []const u8, args: anytype) ?Loc {
        var buf: [256]u8 = undefined;
        return self.locs.get(std.fmt.bufPrint(&buf, fmt, args) catch return null);
    }
};

/// One thread's page-aligned staging buffer and the reads that go through it.
const Reader = struct {
    src: *const Source,
    stage: []align(std.heap.page_size_min) u8 = &.{},

    fn deinit(self: *Reader) void {
        pageFree(self.stage);
    }

    fn read(self: *Reader, l: Loc, dst: []u8) !void {
        if (dst.len != l.len) return error.ModelOptTensorSize;
        const span = alignedSpan(l.off, l.len);
        const size: usize = @intCast(span.end - span.start);
        if (self.stage.len < size) {
            pageFree(self.stage);
            self.stage = &.{};
            self.stage = try pageAlloc(size);
        }
        const skip: usize = @intCast(l.off - span.start);
        var done: usize = 0;
        while (done < skip + dst.len) {
            const got = crt.pread(self.src.fds.items[l.file], self.stage[done..].ptr, size - done, @intCast(span.start + done));
            if (got <= 0) return error.ModelOptRead;
            done += @intCast(got);
        }
        @memcpy(dst, self.stage[skip..][0..dst.len]);
    }
};

/// One header entry's span within the shard's `data_len` data bytes, checked before any cast.
fn tensorLoc(o: std.json.ObjectMap, data_len: u64) !Loc {
    const offs = o.get("data_offsets") orelse return error.ModelOptHeader;
    const shape = o.get("shape") orelse return error.ModelOptHeader;
    if (offs != .array or offs.array.items.len != 2 or shape != .array) return error.ModelOptHeader;
    var se: [2]u64 = undefined;
    for (offs.array.items, &se) |v, *d| {
        if (v != .integer or v.integer < 0) return error.ModelOptHeader;
        d.* = @intCast(v.integer);
    }
    if (se[1] < se[0] or se[1] > data_len) return error.ModelOptHeader;
    var dims = [2]u32{ 1, 1 };
    for (shape.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.ModelOptHeader;
        if (i < 2) dims[i] = @intCast(v.integer);
    }
    return .{ .file = 0, .off = se[0], .len = se[1] - se[0], .rows = dims[0], .cols = dims[1] };
}

fn openSource(model_dir: []const u8) !Source {
    var src: Source = .{ .arena = std.heap.ArenaAllocator.init(std.heap.page_allocator) };
    errdefer src.deinit();
    const a = src.arena.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, model_dir, .{});
    defer dir.close(io);
    const index = try dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 * 1024 * 1024));
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, index, .{});
    const wm = if (parsed == .object) parsed.object.get("weight_map") orelse return error.ModelOptIndex else return error.ModelOptIndex;
    if (wm != .object) return error.ModelOptIndex;
    var files: std.StringHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| if (v == .string) try files.put(a, v.string, {});
    var fit = files.keyIterator();
    while (fit.next()) |name| {
        if (std.mem.indexOf(u8, name.*, "..") != null or std.fs.path.isAbsolute(name.*)) return error.ModelOptIndex;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ model_dir, name.* }, 0);
        const fd = crt.openRead(path);
        if (fd < 0) return error.FileNotFound;
        try src.fds.append(a, fd);
        if (comptime builtin.os.tag.isDarwin()) _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
        var head: [8]u8 = undefined;
        if (crt.pread(fd, &head, 8, 0) != 8) return error.ModelOptHeader;
        const hlen = std.mem.readInt(u64, &head, .little);
        if (hlen > 64 * 1024 * 1024) return error.ModelOptHeader;
        const hbuf = try a.alloc(u8, @intCast(hlen));
        if (crt.pread(fd, hbuf.ptr, hbuf.len, 8) != @as(isize, @intCast(hbuf.len))) return error.ModelOptHeader;
        const fsize = crt.seekEnd(fd);
        if (fsize < 8 + @as(i64, @intCast(hlen))) return error.ModelOptHeader;
        const data_len: u64 = @as(u64, @intCast(fsize)) - 8 - hlen;
        const h = try std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{});
        if (h != .object) return error.ModelOptHeader;
        var hit = h.object.iterator();
        while (hit.next()) |e| {
            if (std.mem.indexOf(u8, e.key_ptr.*, ".mlp.experts.") == null or e.value_ptr.* != .object) continue;
            var loc = try tensorLoc(e.value_ptr.object, data_len);
            loc.file = @intCast(src.fds.items.len - 1);
            loc.off += 8 + hlen;
            try src.locs.put(a, e.key_ptr.*, loc);
        }
    }
    return src;
}

const PROJS = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// One [E, out, in] projection bank: e2m1 codes, e4m3 scales, each expert's f32 global scale.
const Bank = struct {
    base: []const u8,
    proj: usize,
    experts: u32 = 0,
    rows: u32 = 0,
    in: u32 = 0,
    w: []align(std.heap.page_size_min) u8 = &.{},
    sc: []align(std.heap.page_size_min) u8 = &.{},
    gs: []f32 = &.{},
    err: ?anyerror = null,

    fn fill(self: *Bank, src: *const Source) !void {
        const b = self.base;
        const pj = PROJS[self.proj];
        const w0 = src.get("{s}.mlp.experts.0.{s}.weight", .{ b, pj }) orelse return error.ModelOptMissingExpert;
        self.rows = w0.rows;
        self.in = w0.cols * 2;
        const cols = self.in / 16;
        const wlen: usize = @as(usize, w0.rows) * w0.cols;
        const slen: usize = @as(usize, w0.rows) * cols;
        self.w = try pageAlloc(wlen * self.experts);
        self.sc = try pageAlloc(slen * self.experts);
        self.gs = try std.heap.page_allocator.alloc(f32, self.experts);
        var scratch: std.ArrayList(u8) = .empty;
        defer scratch.deinit(std.heap.page_allocator);
        var rd: Reader = .{ .src = src };
        defer rd.deinit();
        for (0..self.experts) |e| {
            try rd.read(src.get("{s}.mlp.experts.{d}.{s}.weight", .{ b, e, pj }) orelse return error.ModelOptMissingExpert, self.w[e * wlen ..][0..wlen]);
            const dst = self.sc[e * slen ..][0..slen];
            if (src.get("{s}.mlp.experts.{d}.{s}.weight_scale.nvfp4_csf_fixed", .{ b, e, pj })) |fx| {
                const ex = src.get("{s}.mlp.experts.{d}.{s}.weight_scale.nvfp4_csf_exceptions", .{ b, e, pj }) orelse return error.ModelOptMissingExpert;
                try scratch.resize(std.heap.page_allocator, fx.len + ex.len + 4);
                const fbytes = scratch.items[0..fx.len];
                try rd.read(fx, fbytes);
                const off = std.mem.alignForward(usize, fx.len, 4);
                const ebytes: []align(4) u8 = @alignCast(scratch.items[off..][0..ex.len]);
                try rd.read(ex, ebytes);
                try decodeCsf(fbytes, std.mem.bytesAsSlice(u32, ebytes), w0.rows, cols, dst);
            } else if (src.get("{s}.mlp.experts.{d}.{s}.weight_scale.csf32_bases", .{ b, e, pj })) |bl| {
                var locs: [4]Loc = .{ bl, undefined, undefined, undefined };
                inline for (.{ "codes", "offsets", "exceptions" }, 1..) |part, i|
                    locs[i] = src.get("{s}.mlp.experts.{d}.{s}.weight_scale.csf32_" ++ part, .{ b, e, pj }) orelse return error.ModelOptMissingExpert;
                var at: [5]usize = undefined;
                at[0] = 0;
                for (locs, 0..) |l, i| at[i + 1] = std.mem.alignForward(usize, at[i] + l.len, 2);
                try scratch.resize(std.heap.page_allocator, at[4]);
                const parts: []align(2) u8 = @alignCast(scratch.items);
                for (locs, 0..) |l, i| try rd.read(l, parts[at[i]..][0..l.len]);
                const offs: []align(2) u8 = @alignCast(parts[at[2]..][0..locs[2].len]);
                const excs: []align(2) u8 = @alignCast(parts[at[3]..][0..locs[3].len]);
                try decodeCsf32(parts[0..bl.len], parts[at[1]..][0..locs[1].len], std.mem.bytesAsSlice(u16, offs), std.mem.bytesAsSlice(u16, excs), w0.rows, cols, dst);
            } else {
                try rd.read(src.get("{s}.mlp.experts.{d}.{s}.weight_scale", .{ b, e, pj }) orelse return error.ModelOptMissingExpert, dst);
            }
            var g: [4]u8 = undefined;
            try rd.read(src.get("{s}.mlp.experts.{d}.{s}.weight_scale_2", .{ b, e, pj }) orelse return error.ModelOptMissingExpert, &g);
            self.gs[e] = @bitCast(std.mem.readInt(u32, &g, .little));
        }
    }
};

fn pageAlloc(len: usize) ![]align(std.heap.page_size_min) u8 {
    return std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), len);
}

fn pageFree(m: []align(std.heap.page_size_min) u8) void {
    if (m.len > 0) std.heap.page_allocator.free(m);
}

/// `data` copied into an MLX array: MLX's allocator wires its memory, which the pageable scratch
/// it came from is not. Lands in the GPU's pool where weights live there.
fn putCopy(weights: *Weights, data: []const u8, shape: []const c_int, dtype: mlx.mlx_dtype, comptime fmt: []const u8, args: anytype) !void {
    const arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), dtype);
    if (arr.ctx == null) return error.ModelOptAlloc;
    try putArray(weights, try mlx.toGpuPool(arr), fmt, args);
}

/// Stores `arr` under a formatted key; on failure frees both.
fn putArray(weights: *Weights, arr: mlx.mlx_array, comptime fmt: []const u8, args: anytype) !void {
    errdefer _ = mlx.mlx_array_free(arr);
    const key = try std.fmt.allocPrint(weights.allocator, fmt, args);
    errdefer weights.allocator.free(key);
    try weights.map.put(key, arr);
}

fn bf16Array(vals: []const f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const f = mlx.mlx_array_new_data(vals.ptr, &[_]c_int{@intCast(vals.len)}, 1, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Banks read per round. A round's scratch is ordinary pageable memory, so it stays small; every
/// finished bank is copied into MLX's own memory before the next round starts.
const BANKS_PER_ROUND = 6;

fn fillOne(b: *Bank, src: *const Source) void {
    b.fill(src) catch |e| {
        b.err = e;
    };
}

fn fillRound(round: []Bank, src: *const Source) void {
    var threads: [BANKS_PER_ROUND]?std.Thread = @splat(null);
    for (round[1..], 1..) |*b, i| threads[i] = std.Thread.spawn(.{}, fillOne, .{ b, src }) catch null;
    fillOne(&round[0], src);
    for (round[1..], 1..) |*b, i| if (threads[i]) |t| t.join() else fillOne(b, src);
}

/// Every layer's routed experts as `switch_mlp` banks: `<proj>.weight` u32 [E, out, in/8],
/// `<proj>.scales` e4m3 [E, out, in/16]; the gate's global scales as `gate_proj.expert_scale`
/// (applied before the SiLU), up x down folded into the routing weight as `expert_scale`. A clamped
/// SwiGLU limits `up` itself, so there the up scale stays apart (`up_proj.expert_scale`, applied
/// before the clamp) and `expert_scale` is the down scale alone.
/// `pressure` is read before each round: once the OS is paging, the load stops by name rather
/// than take the machine down with it.
/// A bank's layer base under this rank's numbering.
fn spanBase(a: std.mem.Allocator, base: []const u8, span: ?LayerSpan) ![]const u8 {
    const sp = span orelse return base;
    var buf: [512]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "{s}.", .{base});
    var out: [512]u8 = undefined;
    const r = sp.remap(key, &out) orelse return error.DistLayerSpanMismatch;
    return a.dupe(u8, r[0 .. r.len - 1]);
}

fn buildBanks(weights: *Weights, model_dir: []const u8, mtp_layer: ?u32, pressure: *const fn () u32, clamped: bool, span: ?LayerSpan, shard: ?Shard) !void {
    var src = try openSource(model_dir);
    defer src.deinit();
    const a = src.arena.allocator();
    var experts: std.StringHashMapUnmanaged(u32) = .empty;
    var lit = src.locs.keyIterator();
    while (lit.next()) |k| {
        if (!std.mem.endsWith(u8, k.*, ".gate_proj.weight")) continue;
        var nb: [512]u8 = undefined;
        if (span) |sp| if (sp.remap(k.*, &nb) == null) continue;
        if (shard) |sh| if (sh.rank != 0 and headOnly(k.*)) continue;
        const at = std.mem.indexOf(u8, k.*, ".mlp.experts.").?;
        const tail = k.*[at + ".mlp.experts.".len ..];
        const e = try std.fmt.parseInt(u32, tail[0..std.mem.indexOfScalar(u8, tail, '.').?], 10);
        const gop = try experts.getOrPut(a, k.*[0..at]);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* = @max(gop.value_ptr.*, e + 1);
    }
    var banks: std.ArrayList(Bank) = .empty;
    var eit = experts.iterator();
    while (eit.next()) |kv| for (0..PROJS.len) |p| try banks.append(a, .{ .base = kv.key_ptr.*, .proj = p, .experts = kv.value_ptr.* });
    defer for (banks.items) |*b| {
        pageFree(b.w);
        pageFree(b.sc);
        if (b.gs.len > 0) std.heap.page_allocator.free(b.gs);
    };

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var bytes: u64 = 0;
    var done: usize = 0;
    while (done < banks.items.len) {
        if (status.underMemoryPressure(pressure())) {
            log.err("[modelopt] the OS is paging: stopped after {d} of {d} expert banks ({d:.1} GB). Quit other apps to free memory, then retry\n", .{ done, banks.items.len, @as(f64, @floatFromInt(bytes)) / 1073741824.0 });
            return error.InsufficientMemory;
        }
        const round = banks.items[done..@min(done + BANKS_PER_ROUND, banks.items.len)];
        fillRound(round, &src);
        for (round) |*b| {
            if (b.err) |e| return e;
            const dst = try mlxName(a, try spanBase(a, b.base, span), mtp_layer);
            const E: c_int = @intCast(b.experts);
            var rows: usize = b.rows;
            var in: usize = b.in;
            var w: []const u8 = b.w;
            var sc: []const u8 = b.sc;
            var w_part: []align(std.heap.page_size_min) u8 = &.{};
            defer pageFree(w_part);
            var sc_part: []align(std.heap.page_size_min) u8 = &.{};
            defer pageFree(sc_part);
            if (shard) |sh| if (!headOnly(b.base)) {
                w_part = try pageAlloc(b.w.len / sh.size);
                sc_part = try pageAlloc(b.sc.len / sh.size);
                if (b.proj < 2) {
                    rows /= sh.size;
                    sh.slice(w_part, b.w, b.experts, b.rows, b.in / 2);
                    sh.slice(sc_part, b.sc, b.experts, b.rows, b.in / 16);
                } else {
                    in /= sh.size;
                    sh.slice(w_part, b.w, @as(usize, b.experts) * b.rows, b.in / 2, 1);
                    sh.slice(sc_part, b.sc, @as(usize, b.experts) * b.rows, b.in / 16, 1);
                }
                w = w_part;
                sc = sc_part;
            };
            try putCopy(weights, w, &.{ E, @intCast(rows), @intCast(in / 8) }, .uint32, "{s}.mlp.switch_mlp.{s}.weight", .{ dst, PROJS[b.proj] });
            try putCopy(weights, sc, &.{ E, @intCast(rows), @intCast(in / 16) }, .uint8, "{s}.mlp.switch_mlp.{s}.scales", .{ dst, PROJS[b.proj] });
            bytes += w.len + sc.len;
            pageFree(b.w);
            pageFree(b.sc);
            b.w = &.{};
            b.sc = &.{};
        }
        done += round.len;
    }
    for (banks.items, 0..) |*b, i| {
        if (b.proj != 0) continue;
        const up = banks.items[i + 1].gs;
        const down = banks.items[i + 2].gs;
        const dst = try mlxName(a, try spanBase(a, b.base, span), mtp_layer);
        try putArray(weights, try bf16Array(b.gs, s), "{s}.mlp.switch_mlp.gate_proj.expert_scale", .{dst});
        if (clamped) {
            try putArray(weights, try bf16Array(up, s), "{s}.mlp.switch_mlp.up_proj.expert_scale", .{dst});
            try putArray(weights, try bf16Array(down, s), "{s}.mlp.switch_mlp.expert_scale", .{dst});
        } else {
            for (up, down) |*u, d| u.* *= d;
            try putArray(weights, try bf16Array(up, s), "{s}.mlp.switch_mlp.expert_scale", .{dst});
        }
    }
    log.info("[modelopt] {d} expert banks over {d} layers, {d:.1} GB (NVFP4, csf scales decoded)\n", .{ banks.items.len, experts.count(), @as(f64, @floatFromInt(bytes)) / 1073741824.0 });
}

/// Bytes the decoded banks take beyond the pack's expert tensors on disk: csf-compressed scales are
/// stored as one e4m3 byte per 16 weights. 0 when the headers cannot be read (the load names that).
pub fn bankExpansionBytes(model_dir: []const u8) u64 {
    var src = openSource(model_dir) catch return 0;
    defer src.deinit();
    var disk: u64 = 0;
    var decoded: u64 = 0;
    var it = src.locs.iterator();
    while (it.next()) |kv| {
        disk += kv.value_ptr.len;
        if (std.mem.endsWith(u8, kv.key_ptr.*, ".weight")) decoded += kv.value_ptr.len + kv.value_ptr.len / 8;
    }
    return decoded -| disk;
}

/// `csf_32x2_v1` scales of one [rows, cols] NVFP4 matrix (FlashInfer's `_csf_blocks.py`): blocks of
/// 32 rows x 2 columns, visited column pair first within each 128-row x 4-column tile; each block
/// packs its 4-bit codes row-major (scale = row base + code), then its exception records
/// `slot | byte << 8` (slot = 2 * row + column) whose count comes from per-page prefix `offsets`.
pub fn decodeCsf32(bases: []const u8, codes: []const u8, offsets: []const u16, exceptions: []const u16, rows: usize, cols: usize, out: []u8) !void {
    const rb = rows / 32;
    const cb = (cols + 1) / 2;
    const tiles = (rb + 3) / 4 * ((cb + 1) / 2);
    if (rows % 32 != 0 or cols == 0 or bases.len != rows or codes.len != (rows * cols + 1) / 2 or
        out.len != rows * cols or offsets.len != rb * cb + (tiles + 31) / 32) return error.CsfGeometry;
    var field: usize = 0;
    var rec: usize = 0;
    var k: usize = 0;
    var tile: usize = 0;
    var r0: usize = 0;
    while (r0 < rb) : (r0 += 4) {
        var c0: usize = 0;
        while (c0 < cb) : (c0 += 2) {
            for (c0..@min(c0 + 2, cb)) |c| for (r0..@min(r0 + 4, rb)) |r| {
                const width = @min(2, cols - 2 * c);
                for (0..32) |rr| for (0..width) |cc| {
                    const row = r * 32 + rr;
                    out[row * cols + 2 * c + cc] = bases[row] +% ((codes[field / 2] >> @intCast(field % 2 * 4)) & 15);
                    field += 1;
                };
                const idx = k + tile / 32;
                if (offsets[idx + 1] < offsets[idx]) return error.CsfGeometry;
                for (0..offsets[idx + 1] - offsets[idx]) |_| {
                    if (rec >= exceptions.len) return error.CsfGeometry;
                    const e = exceptions[rec];
                    const slot = e & 255;
                    if (slot >= 64 or slot % 2 >= width) return error.CsfException;
                    out[(r * 32 + slot / 2) * cols + 2 * c + slot % 2] = @intCast(e >> 8);
                    rec += 1;
                }
                k += 1;
            };
            tile += 1;
        }
    }
    if (rec != exceptions.len) return error.CsfGeometry;
}

/// csf scales of one [rows, cols] NVFP4 matrix (cols = in / 16): per 16-row slab the fixed
/// plane holds 16 row bases then each row's 4-bit codes (low nibble first), scale = base +
/// code; each exception word `(row * cols + col) | byte << 24` replaces one decoded byte.
pub fn decodeCsf(fixed: []const u8, exceptions: []const u32, rows: usize, cols: usize, out: []u8) !void {
    const half = cols / 2;
    if (rows % 16 != 0 or cols % 2 != 0 or fixed.len != rows / 16 * (16 + 16 * half) or out.len != rows * cols) return error.CsfGeometry;
    for (0..rows / 16) |slab| {
        const plane = fixed[slab * (16 + 16 * half) ..][0 .. 16 + 16 * half];
        for (0..16) |i| {
            const base = plane[i];
            const codes = plane[16 + i * half ..][0..half];
            const dst = out[(slab * 16 + i) * cols ..][0..cols];
            for (codes, 0..) |b, j| {
                dst[2 * j] = base +% (b & 15);
                dst[2 * j + 1] = base +% (b >> 4);
            }
        }
    }
    for (exceptions) |e| {
        const pos = e & 0xFFFFFF;
        if (pos >= out.len) return error.CsfException;
        out[pos] = @truncate(e >> 24);
    }
}

test "csf scales: base plus code per row, exceptions replace their byte" {
    const rows = 16;
    const cols = 4;
    var fixed: [16 + 16 * cols / 2]u8 = undefined;
    for (0..rows) |r| {
        fixed[r] = @intCast(r * 10);
        const codes = [4]u8{ @intCast(r % 16), 15, 0, 1 };
        fixed[16 + r * 2] = codes[0] | codes[1] << 4;
        fixed[16 + r * 2 + 1] = codes[2] | codes[3] << 4;
    }
    const exceptions = [_]u32{ (3 * cols + 2) | 200 << 24, (15 * cols + 3) | 7 << 24 };
    var out: [rows * cols]u8 = undefined;
    try decodeCsf(&fixed, &exceptions, rows, cols, &out);
    for (0..rows) |r| {
        const codes = [4]u8{ @intCast(r % 16), 15, 0, 1 };
        for (0..cols) |c| {
            const want: u8 = if (r == 3 and c == 2) 200 else if (r == 15 and c == 3) 7 else @intCast(r * 10 + codes[c]);
            try std.testing.expectEqual(want, out[r * cols + c]);
        }
    }
    try std.testing.expectError(error.CsfGeometry, decodeCsf(fixed[0 .. fixed.len - 1], &exceptions, rows, cols, &out));
    try std.testing.expectError(error.CsfException, decodeCsf(&fixed, &[_]u32{rows * cols}, rows, cols, &out));
}

test "csf32 block scales decode as FlashInfer's reference does" {
    const blob = @embedFile("../../../fixtures/csf32_plane.bin");
    const hdr = std.mem.bytesAsSlice(u32, blob[0..16]);
    const rows: usize = hdr[0];
    const cols: usize = hdr[1];
    var at: usize = 16;
    const bases = blob[at..][0..rows];
    at += rows;
    const codes = blob[at..][0 .. rows * cols / 2 + rows * cols % 2];
    at += codes.len;
    const offsets = try std.testing.allocator.alloc(u16, hdr[2]);
    defer std.testing.allocator.free(offsets);
    @memcpy(std.mem.sliceAsBytes(offsets), blob[at..][0 .. hdr[2] * 2]);
    at += hdr[2] * 2;
    const exceptions = try std.testing.allocator.alloc(u16, hdr[3]);
    defer std.testing.allocator.free(exceptions);
    @memcpy(std.mem.sliceAsBytes(exceptions), blob[at..][0 .. hdr[3] * 2]);
    at += hdr[3] * 2;
    const want = blob[at..][0 .. rows * cols];
    const out = try std.testing.allocator.alloc(u8, rows * cols);
    defer std.testing.allocator.free(out);
    try decodeCsf32(bases, codes, offsets, exceptions, rows, cols, out);
    try std.testing.expectEqualSlices(u8, want, out);
    try std.testing.expectError(error.CsfGeometry, decodeCsf32(bases, codes, offsets, exceptions[0 .. exceptions.len - 1], rows, cols, out));
}

test "ModelOpt header spans are checked before use" {
    const a = std.testing.allocator;
    const cases = [_]struct { json: []const u8, ok: bool }{
        .{ .json = "{\"shape\":[2,3],\"data_offsets\":[4,10]}", .ok = true },
        .{ .json = "{\"shape\":[2,3],\"data_offsets\":[10,4]}", .ok = false },
        .{ .json = "{\"shape\":[2,3],\"data_offsets\":[0,17]}", .ok = false },
        .{ .json = "{\"shape\":[2,3],\"data_offsets\":[-1,4]}", .ok = false },
        .{ .json = "{\"shape\":[-2,3],\"data_offsets\":[0,4]}", .ok = false },
        .{ .json = "{\"shape\":[2,3],\"data_offsets\":[0]}", .ok = false },
        .{ .json = "{\"shape\":\"x\",\"data_offsets\":[0,4]}", .ok = false },
    };
    for (cases) |c| {
        const v = try std.json.parseFromSlice(std.json.Value, a, c.json, .{});
        defer v.deinit();
        const r = tensorLoc(v.value.object, 16);
        if (c.ok) {
            const l = try r;
            try std.testing.expectEqual(@as(u64, 6), l.len);
            try std.testing.expectEqual(@as(u32, 3), l.cols);
        } else try std.testing.expectError(error.ModelOptHeader, r);
    }
}

test "ModelOpt names map onto the mlx-lm layout" {
    const a = std.testing.allocator;
    const cases = [_][2][]const u8{
        .{ "model.language_model.layers.3.self_attn.q_proj.weight", "language_model.model.layers.3.self_attn.q_proj.weight" },
        .{ "model.language_model.layers.3.self_attn.q_proj.weight_scale", "language_model.model.layers.3.self_attn.q_proj.scales" },
        .{ "lm_head.weight", "language_model.lm_head.weight" },
        .{ "lm_head.weight_scale", "language_model.lm_head.scales" },
        .{ "mtp.layers.0.self_attn.k_norm.weight", "language_model.mtp.layers.0.self_attn.k_norm.weight" },
        .{ "model.visual.blocks.0.attn.qkv.weight_scale", "model.visual.blocks.0.attn.qkv.scales" },
        .{ "model.language_model.layers.45.eh_proj.weight", "language_model.mtp.0.eh_proj.weight" },
        .{ "model.language_model.layers.45.shared_head.norm.weight", "language_model.mtp.0.norm.weight" },
        .{ "model.language_model.layers.45.self_attn.q_a_proj.weight_scale", "language_model.mtp.0.block.self_attn.q_a_proj.scales" },
        .{ "model.language_model.layers.45", "language_model.mtp.0.block" },
        .{ "model.language_model.layers.450.input_layernorm.weight", "language_model.model.layers.450.input_layernorm.weight" },
    };
    for (cases) |c| {
        const got = try mlxName(a, c[0], 45);
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
    try std.testing.expect(skipsAtLoad("model.language_model.layers.0.mlp.experts.12.up_proj.weight"));
    try std.testing.expect(skipsAtLoad("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight"));
    try std.testing.expect(!skipsAtLoad("model.language_model.layers.0.mlp.shared_expert.up_proj.weight"));
}

test "ModelOpt FP8_PB_WO dequant: e4m3 code x its 128x128 tile scale" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var codes_host: [256 * 128]u8 = undefined;
    @memset(&codes_host, 0x38); // e4m3 1.0
    codes_host[129 * 128 + 5] = 0x40; // 2.0
    const codes = mlx.mlx_array_new_data(&codes_host, &[_]c_int{ 256, 128 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(codes);
    const sc_host = [_]f32{ 2, 3 };
    const scales = mlx.mlx_array_new_data(&sc_host, &[_]c_int{ 2, 1, 1, 1 }, 4, .float32);
    defer _ = mlx.mlx_array_free(scales);
    const dense = try dequantBlockFp8(codes, scales, s);
    defer _ = mlx.mlx_array_free(dense);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, dense, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const v = mlx.mlx_array_data_float32(f).?[0 .. 256 * 128];
    try std.testing.expectEqual(@as(f32, 2), v[0]);
    try std.testing.expectEqual(@as(f32, 2), v[127 * 128 + 127]);
    try std.testing.expectEqual(@as(f32, 3), v[128 * 128]);
    try std.testing.expectEqual(@as(f32, 6), v[129 * 128 + 5]);
}

test "ModelOpt NVFP4 dense dequant: code x e4m3 scale x weight_scale_2" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // One row of 32 codes (0..15 twice, two per byte, low nibble first), group scales e4m3 2.0 and 1.0.
    var bytes: [16]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @intCast((2 * i % 16) | ((2 * i + 1) % 16) << 4);
    const w8 = mlx.mlx_array_new_data(&bytes, &[_]c_int{ 1, 16 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(w8);
    var words = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(words);
    try mlx.check(mlx.mlx_view(&words, w8, .uint32, s));
    const sb = [_]u8{ 0x40, 0x38 };
    const scales = mlx.mlx_array_new_data(&sb, &[_]c_int{ 1, 2 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(scales);
    const g = [_]f32{0.25};
    const global = mlx.mlx_array_new_data(&g, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(global);
    const dense = try dequantDense(words, scales, global, s);
    defer _ = mlx.mlx_array_free(dense);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, dense, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const got = mlx.mlx_array_data_float32(f).?;
    const e2m1 = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
    for (0..32) |i| try std.testing.expectEqual(e2m1[i % 16] * @as(f32, if (i < 16) 2.0 else 1.0) * 0.25, got[i]);
}

var probe_calls: usize = 0;
var probe_trips_at: usize = std.math.maxInt(usize);

/// A pressure reading that counts its calls and turns to "warning" from call number `probe_trips_at`.
fn countingPressure() u32 {
    probe_calls += 1;
    return if (probe_calls >= probe_trips_at) 2 else 1;
}

fn miniByte(l: usize, e: usize, p: usize, i: usize, salt: u8) u8 {
    return @truncate(l *% 131 +% e *% 31 +% p *% 7 +% i *% 3 +% salt);
}

/// Global scales are powers of two, so the bf16 products `buildBanks` stores are exact.
fn miniScale2(l: usize, e: usize, p: usize) f32 {
    return switch (p) {
        0 => @floatFromInt(@as(u32, 1) << @intCast(e % 3)),
        1 => @floatFromInt(@as(u32, 1) << @intCast(l % 2)),
        else => 0.5,
    };
}

/// A ModelOpt pack of `layers` MoE layers x `experts` experts (4 rows, 16 inputs, plain e4m3 scales).
const MiniPack = struct {
    layers: usize,
    experts: usize,
    /// Weight bytes per row (the layer has 4 rows).
    cols: usize = 8,
    /// Scale tensor stored beside each weight: plain e4m3 by default, a csf part when its name and size say so.
    scale_suffix: []const u8 = "weight_scale",
    scale_len: ?usize = null,
};

pub fn writeMiniPack(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, pack: MiniPack) !void {
    const layers = pack.layers;
    const experts = pack.experts;
    const wlen = 4 * pack.cols;
    const slen = pack.scale_len orelse wlen / 8;
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(a);
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(a);
    try hdr.append(a, '{');
    for (0..layers) |l| for (0..experts) |e| for (PROJS, 0..) |pj, p| {
        const stem = try std.fmt.allocPrint(a, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ l, e, pj });
        defer a.free(stem);
        const wshape = try std.fmt.allocPrint(a, "[4,{d}]", .{pack.cols});
        defer a.free(wshape);
        const sshape = try std.fmt.allocPrint(a, "[{d}]", .{slen});
        defer a.free(sshape);
        const parts = [_]struct { suffix: []const u8, dtype: []const u8, shape: []const u8, len: usize }{
            .{ .suffix = "weight", .dtype = "U8", .shape = wshape, .len = wlen },
            .{ .suffix = pack.scale_suffix, .dtype = "U8", .shape = sshape, .len = slen },
            .{ .suffix = "weight_scale_2", .dtype = "F32", .shape = "[1]", .len = 4 },
        };
        for (parts, 0..) |part, k| {
            const at = data.items.len;
            if (k == 2) {
                var g: [4]u8 = undefined;
                std.mem.writeInt(u32, &g, @bitCast(miniScale2(l, e, p)), .little);
                try data.appendSlice(a, &g);
            } else for (0..part.len) |i| try data.append(a, miniByte(l, e, p, i, @intCast(k)));
            const entry = try std.fmt.allocPrint(a, "{s}\"{s}.{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{ if (hdr.items.len > 1) "," else "", stem, part.suffix, part.dtype, part.shape, at, data.items.len });
            defer a.free(entry);
            try hdr.appendSlice(a, entry);
        }
    };
    try hdr.append(a, '}');
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, hdr.items.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, hdr.items);
    try file.appendSlice(a, data.items);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file.items });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.0.gate_proj.weight\":\"model.safetensors\"}}" });
}

fn tmpPath(a: std.mem.Allocator, tmp: *std.testing.TmpDir) ![]u8 {
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr))), tmp.sub_path });
}

fn expectMiniBanks(weights: *const Weights, layers: usize, experts: usize) !void {
    const a = std.testing.allocator;
    for (0..layers) |l| {
        for (PROJS, 0..) |pj, p| {
            for ([_][]const u8{ "weight", "scales" }, 0..) |what, k| {
                const key = try std.fmt.allocPrint(a, "language_model.model.layers.{d}.mlp.switch_mlp.{s}.{s}", .{ l, pj, what });
                defer a.free(key);
                const arr = weights.get(key) orelse return error.MissingBank;
                try mlx.check(mlx.mlx_array_eval(arr));
                const per: usize = if (k == 0) 32 else 4;
                try std.testing.expectEqual(experts * per, mlx.mlx_array_size(arr) * (if (k == 0) @as(usize, 4) else 1));
                const bytes = mlx.mlx_array_data_uint8(if (k == 0) blk: {
                    var v = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_view(&v, arr, .uint8, mlx.mlx_default_cpu_stream_new()));
                    try mlx.check(mlx.mlx_array_eval(v));
                    break :blk v;
                } else arr).?;
                for (0..experts) |e| for (0..per) |i|
                    try std.testing.expectEqual(miniByte(l, e, p, i, @intCast(k)), bytes[e * per + i]);
            }
        }
        for ([_][]const u8{ "switch_mlp.gate_proj.expert_scale", "switch_mlp.expert_scale" }, 0..) |name, k| {
            const key = try std.fmt.allocPrint(a, "language_model.model.layers.{d}.mlp.{s}", .{ l, name });
            defer a.free(key);
            const arr = weights.get(key) orelse return error.MissingBank;
            try mlx.check(mlx.mlx_array_eval(arr));
            const got = mlx.mlx_array_data_bfloat16(arr).?;
            for (0..experts) |e| {
                const want: f32 = if (k == 0) miniScale2(l, e, 0) else miniScale2(l, e, 1) * miniScale2(l, e, 2);
                try std.testing.expectEqual(@as(u16, @intCast(@as(u32, @bitCast(want)) >> 16)), got[e]);
            }
        }
    }
}

/// The device address `arr`'s data sits at, read by a kernel: asking MLX for the pointer would move
/// a GPU buffer to managed memory first.
fn deviceAddress(arr: mlx.mlx_array) !usize {
    const ins = [_][*:0]const u8{"w"};
    const in_vec = mlx.mlx_vector_string_new_data(&ins, 1);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const outs = [_][*:0]const u8{"addr"};
    const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const src = "unsigned long long p = reinterpret_cast<unsigned long long>(w); addr[0] = (uint32_t)p; addr[1] = (uint32_t)(p >> 32);";
    const k = mlx.mlx_fast_cuda_kernel_new("msv_test_device_address", in_vec, out_vec, src, "", false, 0);
    defer mlx.mlx_fast_cuda_kernel_free(k);
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{2}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, 1, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, 1, 1, 1));
    const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{arr}, 1);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, k, v, c, mlx.gpuStream()));
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    try mlx.check(mlx.mlx_array_eval(out));
    const words = mlx.mlx_array_data_uint32(out).?;
    return @as(usize, words[1]) << 32 | words[0];
}

/// Whether the CUDA driver reports `addr` as managed (host-visible) memory.
fn isManaged(addr: usize) !bool {
    const lib = std.c.dlopen("libcuda.so.1", .{ .LAZY = true }) orelse return error.NoCudaDriver;
    const Get = *const fn (*c_uint, c_int, usize) callconv(.c) c_int;
    const get: Get = @ptrCast(@alignCast(std.c.dlsym(lib, "cuPointerGetAttribute") orelse return error.NoCudaDriver));
    var v: c_uint = 0;
    if (get(&v, 8, addr) != 0) return error.PointerQueryFailed; // CU_POINTER_ATTRIBUTE_IS_MANAGED
    return v != 0;
}

test "ModelOpt: an FP8 linear loaded into the GPU's pool stays there" {
    if (!mlx.cudaAvailable() or !mlx.weightsInGpuPool()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeMiniPack(a, std.testing.io, tmp.dir, .{ .layers = 1, .experts = 1 });
    const path = try tmpPath(a, &tmp);
    defer a.free(path);
    var weights = Weights.init(a);
    defer weights.deinit();
    // As the loader leaves an MXFP8 linear: e4m3 codes and ue8m0 scales, still lazy on the weight stream.
    _ = mlx.mlx_clear_cache();
    const ls = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(ls);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    var codes = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&codes, &[_]c_int{ 256, 128 }, 2, 1, key, ls));
    try weights.map.put(try a.dupe(u8, "model.language_model.layers.0.self_attn.o_proj.weight"), codes);
    var scales = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&scales, &[_]c_int{ 256, 4 }, 2, 1, key, ls));
    try weights.map.put(try a.dupe(u8, "model.language_model.layers.0.self_attn.o_proj.weight_scale"), scales);

    try adapt(&weights, path, null, false, null, null);
    const w = weights.get("language_model.model.layers.0.self_attn.o_proj.weight") orelse return error.MissingWeight;
    try mlx.check(mlx.mlx_array_eval(w));
    try std.testing.expect(!try isManaged(try deviceAddress(w)));
}

test "ModelOpt banks: every expert's bytes and global scales land in its layer's banks" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeMiniPack(a, std.testing.io, tmp.dir, .{ .layers = 7, .experts = 3 });
    const path = try tmpPath(a, &tmp);
    defer a.free(path);
    var weights = Weights.init(a);
    defer weights.deinit();
    probe_calls = 0;
    try buildBanks(&weights, path, null, countingPressure, false, null, null);
    try expectMiniBanks(&weights, 7, 3);
    try std.testing.expectEqual((21 + BANKS_PER_ROUND - 1) / BANKS_PER_ROUND, probe_calls);
}

test "ModelOpt banks: memory pressure stops the load between rounds, by name, with the finished banks published" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeMiniPack(a, std.testing.io, tmp.dir, .{ .layers = 7, .experts = 3 });
    const path = try tmpPath(a, &tmp);
    defer a.free(path);
    var weights = Weights.init(a);
    defer weights.deinit();
    probe_trips_at = 3;
    defer probe_trips_at = std.math.maxInt(usize);
    probe_calls = 0;
    try std.testing.expectError(error.InsufficientMemory, buildBanks(&weights, path, null, countingPressure, false, null, null));
    try std.testing.expectEqual(@as(usize, 3), probe_calls);
    // Two rounds of six banks (weights + scales) were already MLX's when the third was refused.
    try std.testing.expectEqual(@as(u32, 2 * BANKS_PER_ROUND * 2), weights.map.count());
}

test "ModelOpt banks: the resident bill is the decoded banks, not the compressed scales on disk" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // 2 layers x 3 experts x 3 projections: a 256-byte weight decodes to 256 + 256/8 = 288, its disk
    // span is 256 + a 10-byte csf scale + the 4-byte global scale = 270.
    try writeMiniPack(a, std.testing.io, tmp.dir, .{ .layers = 2, .experts = 3, .cols = 64, .scale_suffix = "weight_scale.nvfp4_csf_fixed", .scale_len = 10 });
    const path = try tmpPath(a, &tmp);
    defer a.free(path);
    try std.testing.expectEqual(@as(u64, 18 * 18), bankExpansionBytes(path));
    // Plain scales are stored as decoded: nothing to add.
    var plain = std.testing.tmpDir(.{});
    defer plain.cleanup();
    try writeMiniPack(a, std.testing.io, plain.dir, .{ .layers = 2, .experts = 3 });
    const plain_path = try tmpPath(a, &plain);
    defer a.free(plain_path);
    try std.testing.expectEqual(@as(u64, 0), bankExpansionBytes(plain_path));
}

test "ModelOpt reads: a tensor is read through the pages that cover it, so F_NOCACHE holds" {
    const P: u64 = std.heap.page_size_min;
    const inside = alignedSpan(P + 5, 100);
    try std.testing.expectEqual(P, inside.start);
    try std.testing.expectEqual(2 * P, inside.end);
    const straddle = alignedSpan(P - 4, 8);
    try std.testing.expectEqual(@as(u64, 0), straddle.start);
    try std.testing.expectEqual(2 * P, straddle.end);
    const exact = alignedSpan(2 * P, P);
    try std.testing.expectEqual(2 * P, exact.start);
    try std.testing.expectEqual(3 * P, exact.end);
}

test "ModelOpt banks: a clamped SwiGLU keeps the up scale apart, to scale up before the clamp" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeMiniPack(a, std.testing.io, tmp.dir, .{ .layers = 2, .experts = 3 });
    const path = try tmpPath(a, &tmp);
    defer a.free(path);
    var weights = Weights.init(a);
    defer weights.deinit();
    probe_calls = 0;
    try buildBanks(&weights, path, null, countingPressure, true, null, null);
    for (0..2) |l| {
        for ([_][]const u8{ "switch_mlp.gate_proj.expert_scale", "switch_mlp.up_proj.expert_scale", "switch_mlp.expert_scale" }, 0..) |name, k| {
            const key = try std.fmt.allocPrint(a, "language_model.model.layers.{d}.mlp.{s}", .{ l, name });
            defer a.free(key);
            const arr = weights.get(key) orelse return error.MissingBank;
            try mlx.check(mlx.mlx_array_eval(arr));
            const got = mlx.mlx_array_data_bfloat16(arr).?;
            for (0..3) |e| {
                const want: f32 = miniScale2(l, e, k); // gate, up, and the down scale alone
                try std.testing.expectEqual(@as(u16, @intCast(@as(u32, @bitCast(want)) >> 16)), got[e]);
            }
        }
    }
}
