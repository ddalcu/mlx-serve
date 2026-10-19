//! W8A8 for media DiT linears on NAX GPUs (M5+), on by default only with 64 GiB of RAM
//! (`--w8a8` / `--no-w8a8` override). LOSSY: the tensor unit's int8 x int8 rate is ~1.7x its bf16
//! rate there, and a DiT forward is GEMM-bound at every resolution.
//!
//! Weights and activations are rotated along K by the same block-256
//! Hadamard with fixed signs (orthogonal, so `x W^T` is unchanged), which
//! spreads channel outliers far enough that ONE scale per row quantizes
//! both to int8 well: y[m,n] = xs[m] * ws[n] * sum_k xq[m,k] * wq[n,k].
//! One scale per row keeps the integer accumulation free of per-group
//! epilogues, which is what cost a per-group arm all of its speed.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const BLOCK: c_int = 256;

/// Smallest RAM at which the route is on without being asked: a 4-bit pack's int8 copy is ~1.8x
/// its linears and nothing falls back to the packed weights once the copy exists.
pub const DEFAULT_MIN_RAM_BYTES: u64 = 64 << 30;

/// The out-of-the-box setting: NAX hardware and room for the int8 copies.
pub fn defaultOn(nax_present: bool, ram_bytes: u64) bool {
    return nax_present and ram_bytes >= DEFAULT_MIN_RAM_BYTES;
}

/// Process switch; `main` sets it from `defaultOn` before it reads `--w8a8` / `--no-w8a8`. Off under
/// `zig build test`, so every other oracle keeps its packed numerics and the tests of this route opt in.
pub var enabled: bool = false;

/// Test seam for the hardware answer, so the header-only tests run on any machine.
pub var nax_override: ?bool = null;

/// Whether the int8 kernels can run here: the matrix unit they are written for.
pub fn nax() bool {
    return nax_override orelse @import("transformer.zig").naxAvailable();
}

/// Whether this linear can take the route: NAX present, K a multiple of the
/// rotation block, N a multiple of the store tile's width.
pub fn eligible(k: u32, n: u32) bool {
    if (!enabled) return false;
    if (k % @as(u32, @intCast(BLOCK)) != 0 or n % 64 != 0) return false;
    return nax();
}

/// Group size read for quantized weights when billing: every shipped media pack is g64
/// (another group mis-bills K by the ratio).
const BILL_GROUP: u64 = 64;

/// Bytes a DiT shard holds once resident under this route: the file's, with each
/// linear the loaders convert (`converted`, and `eligible`) re-billed at its int8
/// copy (N*K + 4N) because the originals are released. Null when the route is off
/// or the header is unreadable.
pub fn residentBytes(io: std.Io, dir: std.Io.Dir, name: []const u8) ?u64 {
    if (!enabled) return null;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = dir.openFile(io, name, .{}) catch return null;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    const len = rs.interface.takeInt(u64, .little) catch return null;
    if (len > 64 << 20) return null;
    const hdr = a.alloc(u8, @intCast(len)) catch return null;
    rs.interface.readSliceAll(hdr) catch return null;
    const h = std.json.parseFromSliceLeaky(std.json.Value, a, hdr, .{}) catch return null;
    if (h != .object) return null;
    var total: u64 = 8 + len;
    var it = h.object.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) continue;
        if (std.mem.endsWith(u8, key, ".weight")) {
            total += int8Bytes(h.object, key, a) orelse tensorBytes(e.value_ptr.*);
        } else if (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases")) {
            const base = std.fmt.allocPrint(a, "{s}.weight", .{key[0 .. key.len - 7]}) catch return null;
            if (int8Bytes(h.object, base, a) == null) total += tensorBytes(e.value_ptr.*);
        } else total += tensorBytes(e.value_ptr.*);
    }
    return total;
}

/// The block attention and MLP linears of every DiT the loaders convert (Krea `blocks.N.attn.wq`,
/// Qwen-Image and MageFlow `transformer_blocks.N.img_mlp...`, H3 `blocks.N.mlp.fc1` and its token
/// refiner, FLUX.2 `transformer_blocks.N.ff.linear_in` and `single_transformer_blocks.N.attn.to_out`,
/// LTX `transformer.transformer_blocks.N.audio_attn1.to_q`);
/// modulation, embedders and Krea's text-fusion stack stay packed.
pub fn converted(name: []const u8) bool {
    const stacks = [_][]const u8{ "blocks.", "transformer_blocks.", "single_transformer_blocks.", "transformer.transformer_blocks.", "token_refiner.blocks.", "decoder.layers." };
    const in_stack = for (stacks) |p| {
        if (std.mem.startsWith(u8, name, p)) break true;
    } else false;
    if (!in_stack) return false;
    const stems = [_][]const u8{ "attn", ".mlp.", ".img_mlp.", ".txt_mlp.", ".ff.", "_ff.", ".ff_context.", ".ff_", ".shared.", ".skip." };
    for (stems) |t| if (std.mem.indexOf(u8, name, t) != null) return true;
    return false;
}

fn tensorBytes(v: std.json.Value) u64 {
    const off = (if (v == .object) v.object.get("data_offsets") else null) orelse return 0;
    if (off != .array or off.array.items.len != 2 or off.array.items[0] != .integer or off.array.items[1] != .integer) return 0;
    return @intCast(@max(off.array.items[1].integer - off.array.items[0].integer, 0));
}

fn shape2(v: std.json.Value) ?[2]u64 {
    const sh = (if (v == .object) v.object.get("shape") else null) orelse return null;
    if (sh != .array or sh.array.items.len != 2) return null;
    for (sh.array.items) |d| if (d != .integer or d.integer <= 0) return null;
    return .{ @intCast(sh.array.items[0].integer), @intCast(sh.array.items[1].integer) };
}

/// The int8 copy's bytes for the 2-D weight `wkey` when the loaders would convert it.
fn int8Bytes(h: std.json.ObjectMap, wkey: []const u8, a: std.mem.Allocator) ?u64 {
    if (!converted(wkey)) return null;
    const w = h.get(wkey) orelse return null;
    const wsh = shape2(w) orelse return null;
    const skey = std.fmt.allocPrint(a, "{s}.scales", .{wkey[0 .. wkey.len - 7]}) catch return null;
    const k: u64 = if (h.get(skey)) |sc| (shape2(sc) orelse return null)[1] * BILL_GROUP else blk: {
        const dt = (if (w == .object) w.object.get("dtype") else null) orelse return null;
        if (dt != .string or !(std.mem.eql(u8, dt.string, "BF16") or std.mem.eql(u8, dt.string, "F16") or std.mem.eql(u8, dt.string, "F32"))) return null;
        break :blk wsh[1];
    };
    if (!eligible(@intCast(k), @intCast(wsh[0]))) return null;
    return wsh[0] * k + 4 * wsh[0];
}

pub const Weight = struct {
    wq: mlx.mlx_array, // int8 [N, K], rotated
    ws: mlx.mlx_array, // f32 [N]
    n: c_int,
    k: c_int,

    pub fn deinit(self: *Weight) void {
        _ = mlx.mlx_array_free(self.wq);
        _ = mlx.mlx_array_free(self.ws);
    }

    /// From a dense `[N, K]` weight (any float dtype). Evaluated before return.
    pub fn fromDense(w: mlx.mlx_array, s: mlx.mlx_stream) !Weight {
        const sh = mlx.getShape(w);
        if (sh.len != 2) return error.W8a8BadWeightRank;
        const n = sh[0];
        const k = sh[1];
        var wf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wf);
        try mlx.check(mlx.mlx_astype(&wf, w, .float32, s));
        const r = try rotateRows(wf, n, k, s);
        defer _ = mlx.mlx_array_free(r);
        return quantizeRowsHost(r, n, k, s);
    }

    /// Rows of `parts` stacked in order (one GEMM for sibling projections on one input).
    pub fn concat(parts: []const *const Weight, s: mlx.mlx_stream) !Weight {
        var qs: [8]mlx.mlx_array = undefined;
        var ss: [8]mlx.mlx_array = undefined;
        var n: c_int = 0;
        for (parts, 0..) |p, i| {
            qs[i] = p.wq;
            ss[i] = p.ws;
            n += p.n;
        }
        return joined(qs[0..parts.len], ss[0..parts.len], 0, n, parts[0].k, s);
    }

    /// Rows `start..start+n` as a Weight sharing this one's storage.
    pub fn rows(self: *const Weight, start: c_int, n: c_int, s: mlx.mlx_stream) !Weight {
        var wq = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(wq);
        try mlx.check(mlx.mlx_slice(&wq, self.wq, &[_]c_int{ start, 0 }, 2, &[_]c_int{ start + n, self.k }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var ws = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_slice(&ws, self.ws, &[_]c_int{start}, 1, &[_]c_int{start + n}, 1, &[_]c_int{1}, 1, s));
        const both = [_]mlx.mlx_array{ wq, ws };
        const v = mlx.mlx_vector_array_new_data(&both, both.len);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_eval(v));
        return .{ .wq = wq, .ws = ws, .n = n, .k = self.k };
    }

    /// Rows interleaved (gate j, up j), the layout `forwardSwiglu` reads.
    pub fn interleave(gate: *const Weight, up: *const Weight, s: mlx.mlx_stream) !Weight {
        return joined(&.{ gate.wq, up.wq }, &.{ gate.ws, up.ws }, 1, gate.n + up.n, gate.k, s);
    }

    /// `axis` 0 concatenates; 1 stacks then flattens, interleaving rows.
    fn joined(qs: []const mlx.mlx_array, ss: []const mlx.mlx_array, axis: c_int, n: c_int, k: c_int, s: mlx.mlx_stream) !Weight {
        const qv = mlx.mlx_vector_array_new_data(qs.ptr, qs.len);
        defer _ = mlx.mlx_vector_array_free(qv);
        const sv = mlx.mlx_vector_array_new_data(ss.ptr, ss.len);
        defer _ = mlx.mlx_vector_array_free(sv);
        var q0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q0);
        var s0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(s0);
        if (axis == 0) {
            try mlx.check(mlx.mlx_concatenate_axis(&q0, qv, 0, s));
            try mlx.check(mlx.mlx_concatenate_axis(&s0, sv, 0, s));
        } else {
            try mlx.check(mlx.mlx_stack_axis(&q0, qv, 1, s));
            try mlx.check(mlx.mlx_stack_axis(&s0, sv, 1, s));
        }
        var wq = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(wq);
        try mlx.check(mlx.mlx_reshape(&wq, q0, &[_]c_int{ n, k }, 2, s));
        var ws = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_reshape(&ws, s0, &[_]c_int{n}, 1, s));
        var cq = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cq);
        try mlx.check(mlx.mlx_contiguous(&cq, wq, false, s));
        _ = mlx.mlx_array_free(wq);
        wq = cq;
        const both = [_]mlx.mlx_array{ wq, ws };
        const v = mlx.mlx_vector_array_new_data(&both, both.len);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_eval(v));
        return .{ .wq = wq, .ws = ws, .n = n, .k = k };
    }

    /// From an affine-quantized `[N, K*bits/32]` weight: dequantized, then as `fromDense`.
    pub fn fromAffine(w: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !Weight {
        var dq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dq);
        try mlx.check(mlx.mlx_dequantize(&dq, w, scales, biases, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true }, s));
        return fromDense(dq, s);
    }
};

/// Frees the arrays the int8 copy replaced and nulls the handles (freeing a null one is a no-op).
pub fn release(arrs: []const *mlx.mlx_array) void {
    for (arrs) |a| {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    }
}

/// The int8 copy of a loader's linear, which REPLACES its arrays; null when the shape is off the route.
/// A null `scales` marks a dense weight stored `[in, out]`; a packed one is `[out, in*bits/32]`.
pub fn takeLinear(w: *mlx.mlx_array, scales: *mlx.mlx_array, biases: *mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?Weight {
    const packed_w = scales.ctx != null;
    const wsh = mlx.getShape(w.*);
    const k: c_int = if (packed_w) mlx.getShape(scales.*)[1] * @as(c_int, @intCast(group_size)) else wsh[0];
    if (!eligible(@intCast(k), @intCast(wsh[if (packed_w) 0 else 1]))) return null;
    const q = if (packed_w) try Weight.fromAffine(w.*, scales.*, biases.*, bits, group_size, s) else blk: {
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose(&t, w.*, s));
        break :blk try Weight.fromDense(t, s);
    };
    release(&.{ w, scales, biases });
    return q;
}

pub const Fuse = enum {
    /// Rows stacked in order; each part is re-pointed at its rows of the result (no second copy).
    stack,
    /// A [gate, up] pair interleaved row by row for `forwardSwiglu`; the parts are consumed.
    interleave,
};

/// One weight for sibling projections that read the same input, from their `w8`
/// slots. Null, parts untouched, when any part has none.
pub fn fuse(parts: []const *?Weight, how: Fuse, s: mlx.mlx_stream) !?Weight {
    var ws: [4]*const Weight = undefined;
    for (parts, 0..) |p, i| ws[i] = if (p.*) |*q| q else return null;
    var f = switch (how) {
        .stack => try Weight.concat(ws[0..parts.len], s),
        .interleave => try Weight.interleave(ws[0], ws[1], s),
    };
    errdefer f.deinit();
    var off: c_int = 0;
    for (parts) |p| {
        const n = p.*.?.n;
        const next: ?Weight = if (how == .stack) try f.rows(off, n, s) else null;
        p.*.?.deinit();
        p.* = next;
        off += n;
    }
    return f;
}

/// Int8 copies of a name-keyed weight map's block linears (LTX `Component`, `model.Weights`):
/// keyed by the linear's base name, they REPLACE its packed `<base>.weight/.scales/.biases`.
pub const Overlay = struct {
    map: std.StringHashMapUnmanaged(Weight) = .empty,

    pub fn deinit(self: *Overlay, a: std.mem.Allocator) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            e.value_ptr.deinit();
            a.free(e.key_ptr.*);
        }
        self.map.deinit(a);
    }

    pub fn get(self: *const Overlay, base: []const u8) ?*const Weight {
        return self.map.getPtr(base);
    }

    /// Whether the linear behind `weight_key` (`<base>.weight`) is held here.
    pub fn has(self: *const Overlay, weight_key: []const u8) bool {
        return std.mem.endsWith(u8, weight_key, ".weight") and self.map.contains(weight_key[0 .. weight_key.len - ".weight".len]);
    }

    /// A linear of the map whose input width the engine knows: it fixes the pack's (bits, group size), which
    /// the shapes alone cannot (8-bit g64 and 4-bit g128 pack alike).
    pub const Probe = struct { base: []const u8, in_features: u32 };

    /// Convert every linear of `weights` that `converted` names and `eligible` takes, freeing its arrays.
    /// A packed linear converts when it has the probe's (bits, group size); dense ones convert as they are.
    /// Returns how many converted.
    pub fn convert(self: *Overlay, a: std.mem.Allocator, weights: *std.StringHashMap(mlx.mlx_array), probe: Probe, s: mlx.mlx_stream) !u32 {
        var kb: [2][600]u8 = undefined;
        const pw = weights.get(try std.fmt.bufPrint(&kb[0], "{s}.weight", .{probe.base})) orelse return 0;
        const params: ?@import("transformer.zig").QuantParams = if (weights.get(try std.fmt.bufPrint(&kb[1], "{s}.scales", .{probe.base}))) |sc|
            @import("transformer.zig").affineParamsFromGeometry(pw, sc, probe.in_features) orelse return 0
        else
            null;
        var todo: std.ArrayList([]u8) = .empty;
        defer {
            for (todo.items) |b| a.free(b);
            todo.deinit(a);
        }
        var it = weights.iterator();
        while (it.next()) |e| {
            const k = e.key_ptr.*;
            if (!std.mem.endsWith(u8, k, ".weight") or !converted(k)) continue;
            try todo.append(a, try a.dupe(u8, k[0 .. k.len - ".weight".len]));
        }
        var n: u32 = 0;
        for (todo.items) |base| {
            if (try self.convertOne(a, weights, base, params, s)) n += 1;
        }
        return n;
    }

    fn convertOne(self: *Overlay, a: std.mem.Allocator, weights: *std.StringHashMap(mlx.mlx_array), base: []const u8, params: ?@import("transformer.zig").QuantParams, s: mlx.mlx_stream) !bool {
        var kb: [3][600]u8 = undefined;
        const keys = [3][]const u8{
            try std.fmt.bufPrint(&kb[0], "{s}.weight", .{base}),
            try std.fmt.bufPrint(&kb[1], "{s}.scales", .{base}),
            try std.fmt.bufPrint(&kb[2], "{s}.biases", .{base}),
        };
        const w = weights.get(keys[0]) orelse return false;
        const wsh = mlx.getShape(w);
        if (wsh.len != 2) return false;
        var q: Weight = undefined;
        if (weights.get(keys[1])) |sc| {
            const qp = params orelse return false;
            const bi = weights.get(keys[2]) orelse return false;
            const ssh = mlx.getShape(sc);
            if (ssh.len != 2) return false;
            const k_in: u32 = @intCast(ssh[1] * @as(c_int, @intCast(qp.group_size)));
            if (@as(u32, @intCast(wsh[1])) * 32 != k_in * qp.bits or !eligible(k_in, @intCast(wsh[0]))) return false;
            q = try Weight.fromAffine(w, sc, bi, qp.bits, qp.group_size, s);
        } else {
            switch (mlx.mlx_array_dtype(w)) {
                .bfloat16, .float16, .float32 => {},
                else => return false,
            }
            if (!eligible(@intCast(wsh[1]), @intCast(wsh[0]))) return false;
            q = try Weight.fromDense(w, s);
        }
        errdefer q.deinit();
        const owned = try a.dupe(u8, base);
        errdefer a.free(owned);
        try self.map.put(a, owned, q);
        for (keys) |k| if (weights.fetchRemove(k)) |kv| {
            _ = mlx.mlx_array_free(kv.value);
            a.free(kv.key);
        };
        return true;
    }
};

/// `H(signs * w)` per 256-block of each row, `w` f32 `[n, k]`.
fn rotateRows(w: mlx.mlx_array, n: c_int, k: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var sw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw);
    try mlx.check(mlx.mlx_multiply(&sw, w, try signsFor(k, .float32, s), s));
    var blocks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blocks);
    const bsh = [_]c_int{ n, @divExact(k, BLOCK), BLOCK };
    try mlx.check(mlx.mlx_reshape(&blocks, sw, &bsh, 3, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_hadamard_transform(&h, blocks, .{ .value = 1.0 / 16.0, .has_value = true }, s));
    var out = mlx.mlx_array_new();
    const osh = [_]c_int{ n, k };
    try mlx.check(mlx.mlx_reshape(&out, h, &osh, 2, s));
    return out;
}

/// Symmetric int8 with one scale per row, via MLX ops (load time).
fn quantizeRowsHost(r: mlx.mlx_array, n: c_int, k: c_int, s: mlx.mlx_stream) !Weight {
    _ = k;
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_abs(&a, r, s));
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_max_axis(&amax, a, 1, true, s));
    const inv127 = mlx.mlx_array_new_float(1.0 / 127.0);
    defer _ = mlx.mlx_array_free(inv127);
    const floor_ = mlx.mlx_array_new_float(1.0e-20);
    defer _ = mlx.mlx_array_free(floor_);
    var sc0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc0);
    try mlx.check(mlx.mlx_multiply(&sc0, amax, inv127, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_maximum(&sc, sc0, floor_, s));
    var q0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q0);
    try mlx.check(mlx.mlx_divide(&q0, r, sc, s));
    var q1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q1);
    try mlx.check(mlx.mlx_round(&q1, q0, 0, s));
    const lo = mlx.mlx_array_new_float(-127.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(127.0);
    defer _ = mlx.mlx_array_free(hi);
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    try mlx.check(mlx.mlx_clip(&q2, q1, lo, hi, s));
    var wq = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(wq);
    try mlx.check(mlx.mlx_astype(&wq, q2, .int8, s));
    var ws = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ws);
    const ssh = [_]c_int{n};
    try mlx.check(mlx.mlx_reshape(&ws, sc, &ssh, 1, s));
    const both = [_]mlx.mlx_array{ wq, ws };
    const v = mlx.mlx_vector_array_new_data(&both, both.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
    return .{ .wq = wq, .ws = ws, .n = n, .k = @intCast(mlx.getShape(wq)[1]) };
}

// ── rotation signs ──

const SignsKey = struct { k: c_int, dt: mlx.mlx_dtype };
var signs_cache: std.AutoHashMapUnmanaged(SignsKey, mlx.mlx_array) = .{};

/// The fixed ±1 vector of length `k` both sides of a linear are rotated with.
/// Deterministic (splitmix64 of the index) so a weight converted at load and an
/// activation rotated at run time always agree.
fn signsFor(k: c_int, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const key = SignsKey{ .k = k, .dt = dt };
    if (signs_cache.get(key)) |a| return a;
    const buf = try std.heap.c_allocator.alloc(f32, @intCast(k));
    defer std.heap.c_allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = if (splitmix64(i) & 1 == 0) 1.0 else -1.0;
    const sh = [_]c_int{k};
    const f = mlx.mlx_array_new_data(buf.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(f);
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&a, f, dt, s));
    try mlx.check(mlx.mlx_array_eval(a));
    try signs_cache.put(std.heap.c_allocator, key, a);
    return a;
}

fn splitmix64(i: usize) u64 {
    var z: u64 = @as(u64, i) +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

// ── kernels ──

const HEADER =
    \\#include <metal_tensor>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace mpp::tensor_ops;
    \\
;

/// One threadgroup (8 simdgroups) per row; lane l of a simdgroup holds elements
/// l*8 .. l*8+7 of each 256-block, so the 256-point transform is three
/// in-register butterfly stages and five shuffle stages. The row's amax reduces
/// across the threadgroup before anything is written.
const RQ_SOURCE =
    \\const int K = x_shape[1];
    \\const uint row = threadgroup_position_in_grid.y;
    \\const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int nch = K / 256;
    \\threadgroup float red[8];
    \\float v[CH][8];
    \\float amax = 0.0f;
    \\const device T* xr = x + (size_t)row * K;
    \\for (int c = 0; c < CH; c++) {
    \\  const int ch = int(sg) + 8 * c;
    \\  if (ch >= nch) break;
    \\  const int base = ch * 256 + int(lane) * 8;
    \\  for (int p = 0; p < 8; p++) v[c][p] = float(xr[base + p]) * float(signs[base + p]);
    \\  for (int h = 1; h < 8; h <<= 1)
    \\    for (int p = 0; p < 8; p++) if ((p & h) == 0) { float a = v[c][p], b = v[c][p + h]; v[c][p] = a + b; v[c][p + h] = a - b; }
    \\  for (uint m = 1; m < 32; m <<= 1) {
    \\    const float sgn = (lane & m) ? -1.0f : 1.0f;
    \\    for (int p = 0; p < 8; p++) v[c][p] = fma(sgn, v[c][p], simd_shuffle_xor(v[c][p], m));
    \\  }
    \\  for (int p = 0; p < 8; p++) { v[c][p] *= 0.0625f; amax = max(amax, abs(v[c][p])); }
    \\}
    \\amax = simd_max(amax);
    \\if (lane == 0) red[sg] = amax;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\amax = red[0];
    \\for (int i = 1; i < 8; i++) amax = max(amax, red[i]);
    \\const float sc = max(amax / 127.0f, 1.0e-20f);
    \\const float inv = 1.0f / sc;
    \\device int8_t* qr = xq + (size_t)row * K;
    \\for (int c = 0; c < CH; c++) {
    \\  const int ch = int(sg) + 8 * c;
    \\  if (ch >= nch) break;
    \\  const int base = ch * 256 + int(lane) * 8;
    \\  char4 lo, hi;
    \\  for (int p = 0; p < 4; p++) {
    \\    lo[p] = char(clamp(rint(v[c][p] * inv), -127.0f, 127.0f));
    \\    hi[p] = char(clamp(rint(v[c][p + 4] * inv), -127.0f, 127.0f));
    \\  }
    \\  *(device char4*)(qr + base) = lo;
    \\  *(device char4*)(qr + base + 4) = hi;
    \\}
    \\if (sg == 0 && lane == 0) xs[row] = sc;
;

/// A BM x BN output tile per threadgroup (four simdgroups share one tensor op),
/// K stepped by 64, tiles row-major. The epilogue follows (`EPI_*`).
const GEMM_BODY =
    \\const int K = xq_shape[1];
    \\const int M = xq_shape[0];
    \\const int N = wq_shape[0];
    \\const int m0 = int(threadgroup_position_in_grid.y) * BM;
    \\const int n0 = int(threadgroup_position_in_grid.x) * BN;
    \\constexpr auto desc = matmul2d_descriptor(BM, BN, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, execution_simdgroups<4>> op;
    \\tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, M));
    \\tensor<device int8_t, dextents<int, 2>, tensor_inline> B((device int8_t*)wq, dextents<int, 2>(K, N));
    \\auto a0 = A.template slice<64, BM>(0, m0);
    \\auto b0 = B.template slice<64, BN>(0, n0);
    \\auto cT = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
    \\for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0;
    \\for (int k0 = 0; k0 < K; k0 += 64) {
    \\  auto a = A.template slice<64, BM>(k0, m0);
    \\  auto b = B.template slice<64, BN>(k0, n0);
    \\  op.run(a, b, cT);
    \\}
    \\
;

/// Full tiles store through the destination tensor; an edge tile stores
/// element by element, since the tensor store does not clip.
const EPI_PLAIN =
    \\auto dT = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), T>();
    \\for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    \\  auto ids = cT.get_multidimensional_index(i);
    \\  const int n = min(n0 + ids[0], N - 1), m = min(m0 + ids[1], M - 1);
    \\  dT[i] = T(float(cT[i]) * xs[m] * ws[n]);
    \\}
    \\if (m0 + BM <= M && n0 + BN <= N) {
    \\  tensor<device T, dextents<int, 2>, tensor_inline> D((device T*)y, dextents<int, 2>(N, M));
    \\  auto dS = D.template slice<BN, BM>(n0, m0);
    \\  dT.store(dS);
    \\} else {
    \\  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    \\    auto ids = cT.get_multidimensional_index(i);
    \\    const int n = n0 + ids[0], m = m0 + ids[1];
    \\    if (m < M && n < N) y[(size_t)m * N + n] = dT[i];
    \\  }
    \\}
;

/// `silu(gate) * up` on a weight whose rows interleave (gate j, up j): a thread
/// holds runs of four consecutive columns of one row, i.e. two whole pairs.
const EPI_SWIGLU =
    \\const int NO = N / 2;
    \\for (uint16_t i = 0; i < cT.get_capacity(); i += 4) {
    \\  auto ids = cT.get_multidimensional_index(i);
    \\  const int n = n0 + ids[0], m = m0 + ids[1];
    \\  if (m >= M || n >= N) continue;
    \\  const float xsm = xs[m];
    \\  const float g0 = float(cT[i]) * xsm * ws[n], u0 = float(cT[i + 1]) * xsm * ws[n + 1];
    \\  const float g1 = float(cT[i + 2]) * xsm * ws[n + 2], u1 = float(cT[i + 3]) * xsm * ws[n + 3];
    \\  *(device vec<T, 2>*)(y + (size_t)m * NO + n / 2) = vec<T, 2>(T(g0 / (1.0f + exp(-g0)) * u0), T(g1 / (1.0f + exp(-g1)) * u1));
    \\}
;

var rq_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemm_kernel: ?mlx.mlx_fast_metal_kernel = null;
var swiglu_kernel: ?mlx.mlx_fast_metal_kernel = null;
const RqKey = struct { m: c_int, k: c_int, dt: mlx.mlx_dtype };
const GemmKey = struct { m: c_int, n: c_int, dt: mlx.mlx_dtype, swiglu: bool };
var rq_cfg: std.AutoHashMapUnmanaged(RqKey, mlx.mlx_fast_metal_kernel_config) = .{};
var gemm_cfg: std.AutoHashMapUnmanaged(GemmKey, mlx.mlx_fast_metal_kernel_config) = .{};
var engaged_logged = false;

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn rqConfig(key: RqKey) !mlx.mlx_fast_metal_kernel_config {
    if (rq_cfg.get(key)) |c| return c;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.m, key.k }, 2, .int8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{key.m}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256, key.m, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CH", @divTrunc(@divExact(key.k, BLOCK) + 7, 8)));
    try rq_cfg.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

fn gemmConfig(key: GemmKey) !mlx.mlx_fast_metal_kernel_config {
    if (gemm_cfg.get(key)) |c| return c;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_n = if (key.swiglu) @divExact(key.n, 2) else key.n;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.m, out_n }, 2, key.dt));
    // Measured on M5 Ultra: tall tiles win below ~8k rows, wide tiles above.
    const bm: c_int = if (key.m < 8192) 128 else 64;
    const bn: c_int = @divExact(8192, bm);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(key.n + bn - 1, bn) * 128, @divTrunc(key.m + bm - 1, bm), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BM", bm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BN", bn));
    try gemm_cfg.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

/// A rotated, quantized activation: int8 [M, K] + f32 [M].
pub const Act = struct {
    xq: mlx.mlx_array,
    xs: mlx.mlx_array,
    pub fn deinit(self: *Act) void {
        _ = mlx.mlx_array_free(self.xq);
        _ = mlx.mlx_array_free(self.xs);
    }
};

/// Rotate + quantize `x2` [M, K] (float) in one launch.
pub fn quantize(x2: mlx.mlx_array, s: mlx.mlx_stream) !Act {
    const sh = mlx.getShape(x2);
    const m = sh[0];
    const k = sh[1];
    const dt = mlx.mlx_array_dtype(x2);
    if (rq_kernel == null) rq_kernel = try newKernel("msv_w8a8_rq", &.{ "x", "signs" }, &.{ "xq", "xs" }, RQ_SOURCE, "");
    const inputs = [_]mlx.mlx_array{ x2, try signsFor(k, dt, s) };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, rq_kernel.?, iv, try rqConfig(.{ .m = m, .k = k, .dt = dt }), s));
    var xq = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(xq);
    var xs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(xs);
    try mlx.check(mlx.mlx_vector_array_get(&xq, outs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&xs, outs, 1));
    return .{ .xq = xq, .xs = xs };
}

/// `act @ w^T` in `dt`: [M, N], or [M, N/2] `silu(gate) * up` when `swiglu`
/// (`w` from `Weight.interleave`).
pub fn gemm(act: Act, w: *const Weight, dt: mlx.mlx_dtype, swiglu: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    const m = mlx.getShape(act.xq)[0];
    const kp = if (swiglu) &swiglu_kernel else &gemm_kernel;
    if (kp.* == null) kp.* = if (swiglu)
        try newKernel("msv_w8a8_swiglu", &.{ "xq", "wq", "xs", "ws" }, &.{"y"}, GEMM_BODY ++ EPI_SWIGLU, HEADER)
    else
        try newKernel("msv_w8a8_gemm", &.{ "xq", "wq", "xs", "ws" }, &.{"y"}, GEMM_BODY ++ EPI_PLAIN, HEADER);
    const inputs = [_]mlx.mlx_array{ act.xq, w.wq, act.xs, w.ws };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kp.*.?, iv, try gemmConfig(.{ .m = m, .n = w.n, .dt = dt, .swiglu = swiglu }), s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    if (!engaged_logged) {
        engaged_logged = true;
        log.info("[w8a8] engaged: NAX int8 x int8, block-256 rotation (first M={d} K={d} N={d})\n", .{ m, w.k, w.n });
    }
    return y;
}

// Sibling projections (q/k/v/gate, gate/up) read one activation, so its
// quantization is kept for the next call on the same array. The memo retains
// `memo_x`, so its address cannot be reused by another activation meanwhile.
var memo_x: mlx.mlx_array = .{ .ctx = null };
var memo_act: ?Act = null;
pub var quantize_calls: u64 = 0;

pub fn dropMemo() void {
    if (memo_act) |*a| a.deinit();
    if (memo_x.ctx != null) _ = mlx.mlx_array_free(memo_x);
    memo_act = null;
    memo_x = .{ .ctx = null };
}

fn retained(a: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, a));
    return r;
}

/// `x[..., K] @ w^T` → `[..., N]` in x's dtype.
pub fn forward(x: mlx.mlx_array, w: *const Weight, s: mlx.mlx_stream) !mlx.mlx_array {
    return forwardImpl(x, w, false, s);
}

/// `forward` for an activation stream in any float dtype, the result in `dt`: bf16 and f16 run as they
/// are (so sibling projections of one array still share a quantization), anything else goes through bf16.
pub fn forwardAs(x: mlx.mlx_array, w: *const Weight, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const xdt = mlx.mlx_array_dtype(x);
    const native = xdt == .bfloat16 or xdt == .float16;
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    if (!native) try mlx.check(mlx.mlx_astype(&xc, x, .bfloat16, s));
    const y = try forward(if (native) x else xc, w, s);
    if (mlx.mlx_array_dtype(y) == dt) return y;
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, y, dt, s));
    return out;
}

/// `silu(x @ gate^T) * (x @ up^T)` → `[..., N/2]`, `w` from `Weight.interleave`.
pub fn forwardSwiglu(x: mlx.mlx_array, w: *const Weight, s: mlx.mlx_stream) !mlx.mlx_array {
    return forwardImpl(x, w, true, s);
}

/// The two projections of an interleaved pair, apart (an attached LoRA acts on each).
pub const Pair = struct { gate: mlx.mlx_array, up: mlx.mlx_array };

pub fn forwardPair(x: mlx.mlx_array, w: *const Weight, s: mlx.mlx_stream) !Pair {
    const y = try forwardImpl(x, w, false, s);
    defer _ = mlx.mlx_array_free(y);
    const sh = mlx.getShape(y);
    const half = @divExact(sh[sh.len - 1], 2);
    const rows: c_int = @intCast(@divExact(mlx.mlx_array_size(y), @as(usize, @intCast(w.n))));
    var y3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y3);
    try mlx.check(mlx.mlx_reshape(&y3, y, &[_]c_int{ rows, half, 2 }, 3, s));
    var osh: [8]c_int = undefined;
    @memcpy(osh[0..sh.len], sh);
    osh[sh.len - 1] = half;
    var out: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&out, 0..) |*o, i| {
        const j: c_int = @intCast(i);
        var col = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col);
        try mlx.check(mlx.mlx_slice(&col, y3, &[_]c_int{ 0, 0, j }, 3, &[_]c_int{ rows, half, j + 1 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_reshape(o, col, &osh, sh.len, s));
    }
    return .{ .gate = out[0], .up = out[1] };
}

fn forwardImpl(x: mlx.mlx_array, w: *const Weight, swiglu: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    if (sh.len == 0 or sh[sh.len - 1] != w.k) return error.W8a8BadInput;
    const rows: c_int = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(w.k))));
    const dt = mlx.mlx_array_dtype(x);
    const hit = memo_act != null and mlx.mlx_array_shape(memo_x) == mlx.mlx_array_shape(x);
    if (!hit) {
        var x2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x2);
        try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, w.k }, 2, s));
        const act = try quantize(x2, s);
        quantize_calls += 1;
        dropMemo();
        memo_act = act;
        memo_x = try retained(x);
    }
    const y2 = try gemm(memo_act.?, w, dt, swiglu, s);
    defer _ = mlx.mlx_array_free(y2);
    var osh: [8]c_int = undefined;
    @memcpy(osh[0..sh.len], sh);
    osh[sh.len - 1] = if (swiglu) @divExact(w.n, 2) else w.n;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&y, y2, &osh, sh.len, s));
    return y;
}

/// For the quality arms: the bytes the bill says converting `shards` (files under `dir_path`) changes
/// a pack's residency by, resident minus file. Needs `enabled` and NAX.
pub fn billedDelta(io: std.Io, dir_path: []const u8, shards: []const []const u8) !i64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);
    var delta: i64 = 0;
    for (shards) |name| {
        const st = try dir.statFile(io, name, .{});
        const resident = residentBytes(io, dir, name) orelse return error.W8a8BillOff;
        delta += @as(i64, @intCast(resident)) - @as(i64, @intCast(st.size));
    }
    return delta;
}

/// Loading with the route on must hold what the packed load held plus what the bill says conversion
/// changes: originals kept beside the copies would add the whole copy on top, whatever the pack's width.
pub fn expectBilled(packed_bytes: usize, int8_bytes: usize, billed_delta: i64) !void {
    const got = @as(i64, @intCast(int8_bytes)) - @as(i64, @intCast(packed_bytes));
    std.debug.print("[w8a8] resident delta {d} billed {d}\n", .{ got, billed_delta });
    try testing.expect(@abs(got - billed_delta) < packed_bytes / 20);
}

/// Seeded normal noise of `dt`, for the quality arms that need an input and no fixture.
pub fn randn(shape: []const c_int, seed: u64, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, x, dt, s));
    return out;
}

/// How close `got` is to `want`, whatever the shapes: cosine and rms(got) / rms(want).
/// A lossy route is held to BOTH: cosine cannot see a scale error.
pub const Agreement = struct { cos: f64, rms_ratio: f64 };

pub fn agreement(got: mlx.mlx_array, want: mlx.mlx_array, s: mlx.mlx_stream) !Agreement {
    const n: c_int = @intCast(mlx.mlx_array_size(want));
    var flat: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (flat) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&flat, [_]mlx.mlx_array{ got, want }) |*f, src| {
        var cast = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cast);
        try mlx.check(mlx.mlx_astype(&cast, src, .float32, s));
        try mlx.check(mlx.mlx_reshape(f, cast, &[_]c_int{n}, 1, s));
    }
    var dots: [3]f64 = undefined;
    for (&dots, [_][2]usize{ .{ 0, 1 }, .{ 0, 0 }, .{ 1, 1 } }) |*d, ij| {
        var p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p);
        try mlx.check(mlx.mlx_multiply(&p, flat[ij[0]], flat[ij[1]], s));
        var tot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tot);
        try mlx.check(mlx.mlx_sum(&tot, p, false, s));
        var v: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&v, tot));
        d.* = v;
    }
    if (!(dots[1] > 0 and dots[2] > 0)) return .{ .cos = std.math.nan(f64), .rms_ratio = std.math.nan(f64) };
    return .{ .cos = dots[0] / @sqrt(dots[1] * dots[2]), .rms_ratio = @sqrt(dots[1] / dots[2]) };
}

// ── tests ──

const testing = std.testing;

pub fn relRms(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, a, .float32, s));
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, b, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, af, bf, s));
    var dd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dd);
    try mlx.check(mlx.mlx_multiply(&dd, d, d, s));
    var bb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bb);
    try mlx.check(mlx.mlx_multiply(&bb, bf, bf, s));
    var md = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(md);
    try mlx.check(mlx.mlx_mean(&md, dd, false, s));
    var mb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mb);
    try mlx.check(mlx.mlx_mean(&mb, bb, false, s));
    try mlx.check(mlx.mlx_array_eval(md));
    try mlx.check(mlx.mlx_array_eval(mb));
    var vd: f32 = 0;
    var vb: f32 = 0;
    _ = mlx.mlx_array_item_float32(&vd, md);
    _ = mlx.mlx_array_item_float32(&vb, mb);
    try testing.expect(std.math.isFinite(vd) and std.math.isFinite(vb));
    return @sqrt(vd / vb);
}

/// Normal values with a few channels 20x larger, the outlier pattern rotation is for.
fn outlierInput(rows: c_int, k: c_int, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ rows, k }, 2, .float32, 0.0, 1.0, key, s));
    const buf = try testing.allocator.alloc(f32, @intCast(k));
    defer testing.allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = if (i % 197 == 3) 20.0 else 1.0;
    const g = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{k}, 1, .float32);
    defer _ = mlx.mlx_array_free(g);
    var xg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xg);
    try mlx.check(mlx.mlx_multiply(&xg, x, g, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, xg, .bfloat16, s));
    return out;
}

test "w8a8 quantize kernel equals the rotate-then-quantize reference" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const m: c_int = 37;
    const k: c_int = 1536;
    const x = try outlierInput(m, k, 1, s);
    defer _ = mlx.mlx_array_free(x);
    var act = try quantize(x, s);
    defer act.deinit();
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    const ref = try rotateRows(xf, m, k, s);
    defer _ = mlx.mlx_array_free(ref);
    var qf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qf);
    try mlx.check(mlx.mlx_astype(&qf, act.xq, .float32, s));
    var xs2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xs2);
    try mlx.check(mlx.mlx_reshape(&xs2, act.xs, &[_]c_int{ m, 1 }, 2, s));
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_multiply(&deq, qf, xs2, s));
    // int8 per row of a Gaussianized row: ~0.9% relative error.
    try testing.expect(try relRms(deq, ref, s) < 0.015);
}

/// Relative error of the int8 route against the f32 matmul for [m, k] x [k, n].
fn linearError(m: c_int, k: c_int, n: c_int, s: mlx.mlx_stream) !f32 {
    const x = try outlierInput(m, k, 2, s);
    defer _ = mlx.mlx_array_free(x);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 3));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_random_normal(&w, &[_]c_int{ n, k }, 2, .float32, 0.0, 0.02, key, s));
    var wt = try Weight.fromDense(w, s);
    defer wt.deinit();
    dropMemo();
    defer dropMemo();
    const y = try forward(x, &wt, s);
    defer _ = mlx.mlx_array_free(y);
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var wtr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wtr);
    try mlx.check(mlx.mlx_transpose(&wtr, w, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, xf, wtr, s));
    try testing.expectEqual(@as(c_int, n), mlx.getShape(y)[1]);
    return relRms(y, ref, s);
}

test "w8a8 linear matches the dense matmul, edge tiles included" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // M and N off both tile shapes (tall below 8192 rows, wide above) so the guarded edge store runs.
    try testing.expect(try linearError(300, 1024, 320, s) < 0.02);
    try testing.expect(try linearError(8250, 256, 320, s) < 0.02);
}

test "w8a8 serves any row count: no floor, so a released original is never needed" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]c_int{ 1, 3, 19 }) |m| try testing.expect(try linearError(m, 512, 128, s) < 0.02);
}

test "w8a8 sibling projections share one quantization" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const x = try outlierInput(256, 512, 4, s);
    defer _ = mlx.mlx_array_free(x);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_random_normal(&w, &[_]c_int{ 128, 512 }, 2, .float32, 0.0, 0.02, key, s));
    var wt = try Weight.fromDense(w, s);
    defer wt.deinit();
    dropMemo();
    defer dropMemo();
    const before = quantize_calls;
    const a = try forward(x, &wt, s);
    defer _ = mlx.mlx_array_free(a);
    const b = try forward(x, &wt, s);
    defer _ = mlx.mlx_array_free(b);
    try testing.expectEqual(before + 1, quantize_calls);
}

fn randomWeight(n: c_int, k: c_int, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&w, &[_]c_int{ n, k }, 2, .float32, 0.0, 0.02, key, s));
    return w;
}

fn denseRef(x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, w, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&y, xf, wt, s));
    return y;
}

test "w8a8 interleaved gate/up GEMM equals silu(gate) * up" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const x = try outlierInput(300, 512, 6, s);
    defer _ = mlx.mlx_array_free(x);
    const wg = try randomWeight(192, 512, 7, s);
    defer _ = mlx.mlx_array_free(wg);
    const wu = try randomWeight(192, 512, 8, s);
    defer _ = mlx.mlx_array_free(wu);
    var g8 = try Weight.fromDense(wg, s);
    defer g8.deinit();
    var u8_ = try Weight.fromDense(wu, s);
    defer u8_.deinit();
    var gu = try Weight.interleave(&g8, &u8_, s);
    defer gu.deinit();
    dropMemo();
    defer dropMemo();
    const y = try forwardSwiglu(x, &gu, s);
    defer _ = mlx.mlx_array_free(y);
    const g = try denseRef(x, wg, s);
    defer _ = mlx.mlx_array_free(g);
    const u = try denseRef(x, wu, s);
    defer _ = mlx.mlx_array_free(u);
    var sg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg);
    try mlx.check(mlx.mlx_sigmoid(&sg, g, s));
    var gs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gs);
    try mlx.check(mlx.mlx_multiply(&gs, g, sg, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_multiply(&ref, gs, u, s));
    try testing.expectEqual(@as(c_int, 192), mlx.getShape(y)[1]);
    try testing.expect(try relRms(y, ref, s) < 0.03);
}

test "w8a8 concatenated projections equal each projection" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const x = try outlierInput(260, 256, 9, s);
    defer _ = mlx.mlx_array_free(x);
    const wa = try randomWeight(128, 256, 10, s);
    defer _ = mlx.mlx_array_free(wa);
    const wb = try randomWeight(64, 256, 11, s);
    defer _ = mlx.mlx_array_free(wb);
    var a8 = try Weight.fromDense(wa, s);
    defer a8.deinit();
    var b8 = try Weight.fromDense(wb, s);
    defer b8.deinit();
    var cat = try Weight.concat(&.{ &a8, &b8 }, s);
    defer cat.deinit();
    dropMemo();
    defer dropMemo();
    const y = try forward(x, &cat, s);
    defer _ = mlx.mlx_array_free(y);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_slice(&tail, y, &[_]c_int{ 0, 128 }, 2, &[_]c_int{ 260, 192 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    const ref = try denseRef(x, wb, s);
    defer _ = mlx.mlx_array_free(ref);
    try testing.expect(try relRms(tail, ref, s) < 0.02);
}

test "w8a8 stacked parts are views of the fused weight and forward like standalone ones" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const x = try outlierInput(300, 256, 12, s);
    defer _ = mlx.mlx_array_free(x);
    const wa = try randomWeight(128, 256, 13, s);
    defer _ = mlx.mlx_array_free(wa);
    const wb = try randomWeight(64, 256, 14, s);
    defer _ = mlx.mlx_array_free(wb);
    var a: ?Weight = try Weight.fromDense(wa, s);
    var b: ?Weight = try Weight.fromDense(wb, s);
    var alone = try Weight.fromDense(wb, s);
    defer alone.deinit();
    var fused = (try fuse(&.{ &a, &b }, .stack, s)).?;
    defer fused.deinit();
    defer a.?.deinit();
    defer b.?.deinit();
    dropMemo();
    defer dropMemo();
    const want = try forward(x, &alone, s);
    defer _ = mlx.mlx_array_free(want);
    const got = try forward(x, &b.?, s);
    defer _ = mlx.mlx_array_free(got);
    // Same quantized rows, same kernel: the view is bit-identical to the standalone weight.
    try testing.expect(try relRms(got, want, s) == 0.0);
}

test "w8a8 pair projections equal the two standalone projections" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const x = try outlierInput(300, 512, 15, s);
    defer _ = mlx.mlx_array_free(x);
    const wg = try randomWeight(192, 512, 16, s);
    defer _ = mlx.mlx_array_free(wg);
    const wu = try randomWeight(192, 512, 17, s);
    defer _ = mlx.mlx_array_free(wu);
    var g8 = try Weight.fromDense(wg, s);
    defer g8.deinit();
    var u8_ = try Weight.fromDense(wu, s);
    defer u8_.deinit();
    var gu = try Weight.interleave(&g8, &u8_, s);
    defer gu.deinit();
    dropMemo();
    defer dropMemo();
    const pair = try forwardPair(x, &gu, s);
    defer _ = mlx.mlx_array_free(pair.gate);
    defer _ = mlx.mlx_array_free(pair.up);
    const g = try forward(x, &g8, s);
    defer _ = mlx.mlx_array_free(g);
    const u = try forward(x, &u8_, s);
    defer _ = mlx.mlx_array_free(u);
    try testing.expect(try relRms(pair.gate, g, s) == 0.0);
    try testing.expect(try relRms(pair.up, u, s) == 0.0);
}

test "w8a8 resident bytes re-bill each convertible linear at its int8 copy" {
    nax_override = true; // only shard headers are read, so any machine can run this
    defer nax_override = null;
    enabled = true;
    defer enabled = false;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // block linears: 4-bit g64 [N=128, K=256] and dense bf16 [64, 256]; a K off the rotation
    // block; a 1-D norm; an eligible linear outside the block stacks (packed stays packed).
    const hdr =
        \\{"blocks.0.attn.wq.weight":{"dtype":"U32","shape":[128,32],"data_offsets":[0,16384]},
        \\"blocks.0.attn.wq.scales":{"dtype":"BF16","shape":[128,4],"data_offsets":[16384,17408]},
        \\"blocks.0.attn.wq.biases":{"dtype":"BF16","shape":[128,4],"data_offsets":[17408,18432]},
        \\"blocks.0.mlp.up.weight":{"dtype":"BF16","shape":[64,256],"data_offsets":[18432,51200]},
        \\"blocks.0.mlp.down.weight":{"dtype":"BF16","shape":[64,100],"data_offsets":[51200,64000]},
        \\"blocks.0.attn.norm.weight":{"dtype":"BF16","shape":[64],"data_offsets":[64000,64128]},
        \\"txtfusion.refiner_blocks.0.mlp.up.weight":{"dtype":"BF16","shape":[64,256],"data_offsets":[64128,96896]}}
    ;
    var buf: [8 + hdr.len]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], hdr.len, .little);
    @memcpy(buf[8..], hdr);
    try tmp.dir.writeFile(io, .{ .sub_path = "t.safetensors", .data = &buf });
    const got = residentBytes(io, tmp.dir, "t.safetensors").?;
    // header; wq: 128*256 + 4*128; up: 64*256 + 4*64; the K-off linear, the norm and the text-fusion linear keep their bytes.
    try testing.expectEqual(@as(u64, 8 + hdr.len + 128 * 256 + 4 * 128 + 64 * 256 + 4 * 64 + 12800 + 128 + 32768), got);
    try testing.expect(converted("token_refiner.blocks.0.attn.qkv_proj.weight") and converted("transformer_blocks.3.img_mlp.gate_layer.weight"));
    try testing.expect(!converted("blocks.0.adaln_proj.linear.weight") and !converted("txtfusion.layerwise_blocks.0.attn.wq.weight") and !converted("img_in.weight"));
    // LTX / ACE-Step / Music3 / Hunyuan3D block linears; their embedders, encoders and routed experts stay packed.
    for ([_][]const u8{
        "transformer.transformer_blocks.7.audio_attn1.to_q.weight",   "transformer.transformer_blocks.7.audio_to_video_attn.to_out.weight",
        "transformer.transformer_blocks.7.audio_ff.proj_in.weight",   "decoder.layers.3.cross_attn.k_proj.weight",
        "decoder.layers.3.mlp.down_proj.weight",                      "transformer_blocks.4.ff_in.weight",
        "transformer_blocks.4.attn.to_out.0.weight",                  "blocks.9.attn2.k.weight",
        "blocks.9.moe.shared.fc1.weight",                             "blocks.12.skip.linear.weight",
    }) |n| try testing.expect(converted(n));
    for ([_][]const u8{
        "transformer.patchify_proj.weight",                           "transformer.adaln_single.linear.weight",
        "encoder.lyric_encoder.layers.0.mlp.up_proj.weight",          "decoder.condition_embedder.weight",
        "blocks.9.moe.experts.fc1.weight",                            "blocks.9.moe.gate.weight",
        "x_embedder.weight",                                          "proj_in.weight",
    }) |n| try testing.expect(!converted(n));
}

test "w8a8 is on by default only with NAX and 64 GiB of RAM" {
    try testing.expect(defaultOn(true, 64 << 30));
    try testing.expect(defaultOn(true, 512 << 30));
    try testing.expect(!defaultOn(true, (64 << 30) - 1));
    try testing.expect(!defaultOn(true, 0)); // an unreadable size is not a big machine
    try testing.expect(!defaultOn(false, 512 << 30));
}

/// A random dense `[n, k]` bf16 weight; with `bits` > 0 also its affine packing (weight, scales, biases).
fn testLinear(n: c_int, k: c_int, bits: c_int, gs: c_int, seed: u64, s: mlx.mlx_stream) !struct { dense: mlx.mlx_array, parts: [3]mlx.mlx_array } {
    const dense = try randn(&.{ n, k }, seed, .bfloat16, s);
    var parts: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    if (bits > 0) {
        var vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_quantize(&vec, dense, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{ .ctx = null }, s));
        for (&parts, 0..) |*p, i| try mlx.check(mlx.mlx_vector_array_get(p, vec, i));
    }
    return .{ .dense = dense, .parts = parts };
}

fn putPacked(a: std.mem.Allocator, map: *std.StringHashMap(mlx.mlx_array), base: []const u8, parts: [3]mlx.mlx_array) !void {
    for (parts, [_][]const u8{ "weight", "scales", "biases" }) |arr, leaf|
        try map.put(try std.fmt.allocPrint(a, "{s}.{s}", .{ base, leaf }), arr);
}

test "w8a8 overlay converts the pack's own (bits, group), dense weights as they are, and frees what it replaced" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    enabled = true;
    defer enabled = false;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var map = std.StringHashMap(mlx.mlx_array).init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    // The probe fixes 4-bit g32 at K=512; the 8-bit g64 sibling packs at another ratio and stays packed.
    const q = try testLinear(128, 512, 4, 32, 1, s);
    defer _ = mlx.mlx_array_free(q.dense);
    try putPacked(a, &map, "blocks.0.attn.q", q.parts);
    const k = try testLinear(128, 512, 4, 32, 2, s);
    _ = mlx.mlx_array_free(k.dense);
    try putPacked(a, &map, "blocks.0.attn.k", k.parts);
    const odd = try testLinear(128, 512, 8, 64, 3, s);
    _ = mlx.mlx_array_free(odd.dense);
    try putPacked(a, &map, "blocks.0.attn.odd", odd.parts);
    const up = try testLinear(128, 256, 0, 0, 4, s);
    try map.put(try a.dupe(u8, "blocks.0.mlp.up.weight"), up.dense);
    try map.put(try a.dupe(u8, "blocks.0.attn.q_norm.weight"), try randn(&.{512}, 5, .bfloat16, s));
    try map.put(try a.dupe(u8, "txt.embed.weight"), try randn(&.{128, 512}, 6, .bfloat16, s));

    const x = try randn(&.{ 64, 512 }, 7, .bfloat16, s);
    defer _ = mlx.mlx_array_free(x);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_quantized_matmul(&ref, x, q.parts[0], q.parts[1], q.parts[2], true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine", s));

    var ov: Overlay = .{};
    defer ov.deinit(a);
    const n = try ov.convert(a, &map, .{ .base = "blocks.0.attn.q", .in_features = 512 }, s);
    try testing.expectEqual(@as(u32, 3), n); // q, k and the dense up
    try testing.expect(ov.has("blocks.0.attn.q.weight") and ov.has("blocks.0.attn.k.weight") and ov.has("blocks.0.mlp.up.weight"));
    try testing.expect(!ov.has("blocks.0.attn.odd.weight") and !ov.has("txt.embed.weight") and !ov.has("blocks.0.attn.q_norm.weight"));
    for ([_][]const u8{ "blocks.0.attn.q.weight", "blocks.0.attn.q.scales", "blocks.0.attn.q.biases", "blocks.0.mlp.up.weight" }) |key|
        try testing.expect(map.get(key) == null);
    for ([_][]const u8{ "blocks.0.attn.odd.weight", "blocks.0.attn.odd.scales", "blocks.0.attn.q_norm.weight", "txt.embed.weight" }) |key|
        try testing.expect(map.get(key) != null);
    const y = try forward(x, ov.get("blocks.0.attn.q").?, s);
    defer _ = mlx.mlx_array_free(y);
    try testing.expect(try relRms(y, ref, s) < 0.03);
}

test "w8a8 takeLinear takes a packed or a dense linear and leaves an off-route one alone" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    enabled = true;
    defer enabled = false;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const p = try testLinear(128, 512, 4, 64, 1, s);
    defer _ = mlx.mlx_array_free(p.dense);
    var parts = p.parts;
    var q = (try takeLinear(&parts[0], &parts[1], &parts[2], 4, 64, s)).?;
    defer q.deinit();
    try testing.expect(parts[0].ctx == null and parts[1].ctx == null and parts[2].ctx == null);
    try testing.expectEqual(@as(c_int, 128), q.n);
    try testing.expectEqual(@as(c_int, 512), q.k);
    // dense weights are stored [in, out]
    var dw = try randn(&.{ 256, 64 }, 2, .float16, s);
    var none: mlx.mlx_array = .{ .ctx = null };
    var nb: mlx.mlx_array = .{ .ctx = null };
    var qd = (try takeLinear(&dw, &none, &nb, 0, 0, s)).?;
    defer qd.deinit();
    try testing.expectEqual(@as(c_int, 64), qd.n);
    try testing.expectEqual(@as(c_int, 256), qd.k);
    try testing.expect(dw.ctx == null);
    // K off the rotation block: untouched
    var odd = try randn(&.{ 300, 64 }, 3, .bfloat16, s);
    defer _ = mlx.mlx_array_free(odd);
    try testing.expect((try takeLinear(&odd, &none, &nb, 0, 0, s)) == null);
    try testing.expect(odd.ctx != null);
}

test "w8a8 serves f16 activations as they are: right answer, one quantization for sibling projections" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const xb = try outlierInput(64, 512, 2, s);
    defer _ = mlx.mlx_array_free(xb);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, xb, .float16, s));
    const w = try randn(&.{ 128, 512 }, 3, .float32, s);
    defer _ = mlx.mlx_array_free(w);
    var wt = try Weight.fromDense(w, s);
    defer wt.deinit();
    dropMemo();
    defer dropMemo();
    const calls = quantize_calls;
    const y = try forwardAs(x, &wt, .float16, s);
    defer _ = mlx.mlx_array_free(y);
    const y2 = try forwardAs(x, &wt, .float16, s);
    defer _ = mlx.mlx_array_free(y2);
    try testing.expectEqual(calls + 1, quantize_calls);
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(y));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var wtr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wtr);
    try mlx.check(mlx.mlx_transpose(&wtr, w, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, xf, wtr, s));
    try testing.expect(try relRms(y, ref, s) < 0.03);
}
