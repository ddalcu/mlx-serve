//! NVIDIA ModelOpt checkpoints (`quant_method: modelopt`), served as published: MXFP8 linears
//! reinterpreted in place, NVFP4 routed experts stacked into banks with their csf-compressed
//! scales decoded at load, per-expert `weight_scale_2` kept beside each bank.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const crt = @import("crt.zig");
const Weights = @import("model.zig").Weights;

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

/// The mlx-lm name of a ModelOpt tensor (the layout our Flash Next packs use).
pub fn mlxName(a: std.mem.Allocator, key: []const u8) ![]u8 {
    const prefixes = [_][2][]const u8{ .{ "model.language_model.", "language_model.model." }, .{ "lm_head.", "language_model.lm_head." }, .{ "mtp.", "language_model.mtp." } };
    var head: []const u8 = "";
    var rest = key;
    for (prefixes) |p| if (std.mem.startsWith(u8, key, p[0])) {
        head = p[1];
        rest = key[p[0].len..];
        break;
    };
    const suffix = ".weight_scale";
    if (std.mem.endsWith(u8, rest, suffix)) return std.fmt.allocPrint(a, "{s}{s}.scales", .{ head, rest[0 .. rest.len - suffix.len] });
    return std.fmt.allocPrint(a, "{s}{s}", .{ head, rest });
}

/// Renames every loaded tensor to the mlx-lm layout, reinterprets fp weight bytes as the u32
/// words MLX packs, dequantizes the vision tower's linears to bf16 (its encoder runs dense),
/// then adds the routed-expert banks read from `model_dir`.
pub fn adapt(weights: *Weights, model_dir: []const u8, s: mlx.mlx_stream) !void {
    const a = weights.allocator;
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
        const name = try mlxName(a, key);
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
        var words = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(words);
        try mlx.check(mlx.mlx_view(&words, kv.value_ptr.*, .uint32, s));
        var buf2: [512]u8 = undefined;
        const global = old.get(std.fmt.bufPrint(&buf2, "{s}_scale_2", .{key}) catch return error.NameTooLong);
        if (std.mem.startsWith(u8, key, "model.visual.")) {
            try weights.map.put(name, try dequantDense(words, scales.?, global, s));
            continue;
        }
        if (global != null) return error.ModelOptUnsupportedGlobalScale;
        try putShared(weights, name, words);
        const sname = try mlxName(a, scale_key);
        errdefer a.free(sname);
        try putShared(weights, sname, scales.?);
    }
    try buildBanks(weights, model_dir);
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

    fn read(self: *const Source, l: Loc, dst: []u8) !void {
        if (dst.len != l.len) return error.ModelOptTensorSize;
        var done: usize = 0;
        while (done < dst.len) {
            const got = crt.pread(self.fds.items[l.file], dst[done..].ptr, dst.len - done, @intCast(l.off + done));
            if (got <= 0) return error.ModelOptRead;
            done += @intCast(got);
        }
    }
};

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
        const h = try std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{});
        if (h != .object) return error.ModelOptHeader;
        var hit = h.object.iterator();
        while (hit.next()) |e| {
            if (std.mem.indexOf(u8, e.key_ptr.*, ".mlp.experts.") == null or e.value_ptr.* != .object) continue;
            const o = e.value_ptr.object;
            const offs = (o.get("data_offsets") orelse return error.ModelOptHeader).array.items;
            const shape = (o.get("shape") orelse return error.ModelOptHeader).array.items;
            const rows: u32 = if (shape.len >= 1) @intCast(shape[0].integer) else 1;
            const cols: u32 = if (shape.len >= 2) @intCast(shape[1].integer) else 1;
            const start: u64 = @intCast(offs[0].integer);
            try src.locs.put(a, e.key_ptr.*, .{ .file = @intCast(src.fds.items.len - 1), .off = 8 + hlen + start, .len = @as(u64, @intCast(offs[1].integer)) - start, .rows = rows, .cols = cols });
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
        for (0..self.experts) |e| {
            try src.read(src.get("{s}.mlp.experts.{d}.{s}.weight", .{ b, e, pj }) orelse return error.ModelOptMissingExpert, self.w[e * wlen ..][0..wlen]);
            const dst = self.sc[e * slen ..][0..slen];
            if (src.get("{s}.mlp.experts.{d}.{s}.weight_scale.nvfp4_csf_fixed", .{ b, e, pj })) |fx| {
                const ex = src.get("{s}.mlp.experts.{d}.{s}.weight_scale.nvfp4_csf_exceptions", .{ b, e, pj }) orelse return error.ModelOptMissingExpert;
                try scratch.resize(std.heap.page_allocator, fx.len + ex.len + 4);
                const fbytes = scratch.items[0..fx.len];
                try src.read(fx, fbytes);
                const off = std.mem.alignForward(usize, fx.len, 4);
                const ebytes: []align(4) u8 = @alignCast(scratch.items[off..][0..ex.len]);
                try src.read(ex, ebytes);
                try decodeCsf(fbytes, std.mem.bytesAsSlice(u32, ebytes), w0.rows, cols, dst);
            } else {
                try src.read(src.get("{s}.mlp.experts.{d}.{s}.weight_scale", .{ b, e, pj }) orelse return error.ModelOptMissingExpert, dst);
            }
            var g: [4]u8 = undefined;
            try src.read(src.get("{s}.mlp.experts.{d}.{s}.weight_scale_2", .{ b, e, pj }) orelse return error.ModelOptMissingExpert, &g);
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

const Owned = struct { mem: []align(std.heap.page_size_min) u8 };

fn onRelease(payload: ?*anyopaque) callconv(.c) void {
    const o: *Owned = @ptrCast(@alignCast(payload.?));
    pageFree(o.mem);
    std.heap.page_allocator.destroy(o);
}

/// `mem` as an MLX array without a copy where Metal accepts the buffer; MLX owns the unmap.
fn wrapOwned(mem: []align(std.heap.page_size_min) u8, shape: []const c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const o = try std.heap.page_allocator.create(Owned);
    o.* = .{ .mem = mem };
    const arr = mlx.mlx_array_new_data_managed_payload(mem.ptr, shape.ptr, @intCast(shape.len), dtype, o, onRelease);
    if (arr.ctx == null) return error.ModelOptWrap;
    return arr;
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

/// Every layer's routed experts as `switch_mlp` banks: `<proj>.weight` u32 [E, out, in/8],
/// `<proj>.scales` e4m3 [E, out, in/16]; the gate's global scales as `gate_proj.expert_scale`
/// (applied before the SiLU), up x down folded into the routing weight as `expert_scale`.
fn buildBanks(weights: *Weights, model_dir: []const u8) !void {
    var src = try openSource(model_dir);
    defer src.deinit();
    const a = src.arena.allocator();
    var experts: std.StringHashMapUnmanaged(u32) = .empty;
    var lit = src.locs.keyIterator();
    while (lit.next()) |k| {
        if (!std.mem.endsWith(u8, k.*, ".gate_proj.weight")) continue;
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

    const Pool = struct {
        next: std.atomic.Value(usize) = .init(0),
        fn work(pool: *@This(), list: []Bank, source: *const Source) void {
            while (true) {
                const i = pool.next.fetchAdd(1, .monotonic);
                if (i >= list.len) return;
                list[i].fill(source) catch |e| {
                    list[i].err = e;
                };
            }
        }
    };
    var pool: Pool = .{};
    var threads: [12]?std.Thread = @splat(null);
    for (&threads) |*t| t.* = std.Thread.spawn(.{}, Pool.work, .{ &pool, banks.items, &src }) catch null;
    pool.work(banks.items, &src);
    for (threads) |t| if (t) |th| th.join();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var bytes: u64 = 0;
    for (banks.items) |*b| {
        if (b.err) |e| return e;
        const dst = try mlxName(a, b.base);
        const E: c_int = @intCast(b.experts);
        const w_arr = try wrapOwned(b.w, &.{ E, @intCast(b.rows), @intCast(b.in / 8) }, .uint32);
        bytes += b.w.len + b.sc.len;
        b.w = &.{};
        try weights.map.put(try std.fmt.allocPrint(weights.allocator, "{s}.mlp.switch_mlp.{s}.weight", .{ dst, PROJS[b.proj] }), w_arr);
        const s_arr = try wrapOwned(b.sc, &.{ E, @intCast(b.rows), @intCast(b.in / 16) }, .uint8);
        b.sc = &.{};
        try weights.map.put(try std.fmt.allocPrint(weights.allocator, "{s}.mlp.switch_mlp.{s}.scales", .{ dst, PROJS[b.proj] }), s_arr);
    }
    for (banks.items, 0..) |*b, i| {
        if (b.proj != 0) continue;
        const up = banks.items[i + 1].gs;
        const down = banks.items[i + 2].gs;
        const dst = try mlxName(a, b.base);
        try weights.map.put(try std.fmt.allocPrint(weights.allocator, "{s}.mlp.switch_mlp.gate_proj.expert_scale", .{dst}), try bf16Array(b.gs, s));
        for (up, down) |*u, d| u.* *= d;
        try weights.map.put(try std.fmt.allocPrint(weights.allocator, "{s}.mlp.switch_mlp.expert_scale", .{dst}), try bf16Array(up, s));
    }
    log.info("[modelopt] {d} expert banks over {d} layers, {d:.1} GB (NVFP4, csf scales decoded)\n", .{ banks.items.len, experts.count(), @as(f64, @floatFromInt(bytes)) / 1073741824.0 });
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

test "ModelOpt names map onto the mlx-lm layout" {
    const a = std.testing.allocator;
    const cases = [_][2][]const u8{
        .{ "model.language_model.layers.3.self_attn.q_proj.weight", "language_model.model.layers.3.self_attn.q_proj.weight" },
        .{ "model.language_model.layers.3.self_attn.q_proj.weight_scale", "language_model.model.layers.3.self_attn.q_proj.scales" },
        .{ "lm_head.weight", "language_model.lm_head.weight" },
        .{ "mtp.layers.0.self_attn.k_norm.weight", "language_model.mtp.layers.0.self_attn.k_norm.weight" },
        .{ "model.visual.blocks.0.attn.qkv.weight_scale", "model.visual.blocks.0.attn.qkv.scales" },
    };
    for (cases) |c| {
        const got = try mlxName(a, c[0]);
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
    try std.testing.expect(skipsAtLoad("model.language_model.layers.0.mlp.experts.12.up_proj.weight"));
    try std.testing.expect(skipsAtLoad("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight"));
    try std.testing.expect(!skipsAtLoad("model.language_model.layers.0.mlp.shared_expert.up_proj.weight"));
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
