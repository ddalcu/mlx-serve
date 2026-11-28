//! Hy4-flash's Engram: a hashed n-gram memory injected into the hyper-connection streams BEFORE attention, on
//! the pack layers the config marks (1/11/21/31).
//!
//! The hash is numpy-exact (`blocks/numpy_rng.zig` draws the multipliers): each token maps through the frozen
//! `lookup_table` to a compressed id, the ids of an n-gram (current, prev, prev2) multiply the layer's odd
//! multipliers, XOR together, and reduce modulo the head's own vocab size. The 16 heads' rows come from the
//! layer's streamed table (16 preads per token, dequantized off-disk), round-trip through fp8 e4m3 per head
//! vector, normalize per head, concatenate to 1536, and gate into a residual shared by the streams: the
//! value projection's signal, multiplied per lane by how well that lane's query matches the memory's key.
//!
//! Reference (`HYV4FlashEngramLayer`): `g = dot(norm_key[hc](key_proj[hc](emb)), norm_query[hc](stream[hc])) *
//! hidden^-0.5`, `gate = sigmoid(sign(g) * sqrt(max(|g|, 1e-6)))`,
//! `residual = gate * (pre_gamma * rms(value_proj(emb))) + per_hc_scales * signal`, zeroed where the current
//! token carries no n-gram (a pad or a visual row).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const proj = @import("proj.zig");
const weights = @import("../weights.zig");
const crt = @import("../../../sdk/util/crt.zig");
const log = @import("../../../sdk/util/log.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Proj = proj.Proj;
const gpa = std.heap.c_allocator;

/// The e4m3fn grid torch converts to: exponent bias 7, 3 mantissa bits, no infinities (the format `fn`).
const e4m3_max: f32 = 448.0;

pub const Engram = struct {
    tbl: weights.Engram,
    /// The frozen `lookup_table` (`I32[lookup_size]`) evaluated once at load: the hash is host arithmetic.
    lookup_host: []const i32,
    /// The layer's odd multipliers, numpy-exact (`hashMultipliers`).
    mult: [3]i64,
    /// The 16 heads' vocab sizes (order 2's 8 then order 3's 8) and their table row offsets.
    moduli: [16]u32,
    head_off: [16]u32,
    heads: u32,
    head_dim: u32,
    /// `heads * head_dim` — the concatenated memory width the projections read.
    emb_width: u32,
    key_proj: [4]Proj,
    key_bias: [4]Array,
    norm_key: [4]Array,
    norm_query: [4]Array,
    value_proj: Proj,
    value_bias: Array,
    /// The per-head norm weights stacked `[heads, head_dim]` at load.
    head_norm_w: Array,
    pre_gamma: Array,
    hc_scales: [4]Array,
    pad_id: u32,
    /// The image/video placeholder ids: a span that carries projector output owns no n-gram either.
    image_token_id: u32,
    video_token_id: u32,
    lookup_size: u32,
    fp8: bool,
    hc_mult: u32,
    hidden: u32,

    /// The memory math over the normalized `emb` `[rows, emb_width]`: `signal = pre_gamma * rms(value_proj(emb))`,
    /// a per-lane gate against the stream row, then `gate * signal + hc_scale * signal`. The caller applies the
    /// pad/visual mask.
    pub fn gateResidual(e: *Engram, g: *Graph, emb: Array, streams: Array) !Array {
        const rows: c_int = ops.shape(emb)[0];
        const hidden: c_int = @intCast(e.hidden);
        const lanes: c_int = @intCast(e.hc_mult);

        var t = g.scope();
        defer t.deinit();
        const vp = try e.value_proj.forward(&t, emb);
        const value = try t.add(vp, e.value_bias);
        const vf = try t.astype(value, .float32);
        const vms = try t.mean(try t.square(vf), -1, true);
        const vinv = try t.rsqrt(try t.addScalar(vms, defaultEps));
        // The reference multiplies the bf16-rmsnormed signal by the bf16 pre_gamma: keep the lane dtype.
        const signal = try t.astype(try t.mul(try t.mul(vf, vinv), e.pre_gamma), .bfloat16);

        var sums: [4]Array = undefined;
        for (0..@as(usize, @intCast(lanes))) |k| {
            // Every norm in the Engram layer is a plain `nn.RMSNorm`: torch's default eps (float32's, it
            // normalizes in f32), not the config's `rms_norm_eps`.
            const key = try t.rmsNorm(try t.add(try e.key_proj[k].forward(&t, emb), e.key_bias[k]), e.norm_key[k], defaultEps);
            const query = try t.slice(streams, &.{ 0, 0, @intCast(k), 0 }, &.{ 1, @intCast(rows), @as(c_int, @intCast(k)) + 1, hidden });
            const q = try t.rmsNorm(try t.reshape(query, &.{ rows, hidden }), e.norm_query[k], defaultEps);
            const dot = try t.mulScalar(try t.sum(try t.mul(key, q), -1, false), 1.0 / @sqrt(@as(f32, @floatFromInt(e.hidden))));
            const mag = try t.sqrt(try t.maximum(try t.abs(dot), try t.scalarLike(dot, 1e-6)));
            const signed = try t.where(try t.less(dot, try t.scalarLike(dot, 0)), try t.neg(mag), mag);
            sums[k] = try t.sigmoid(signed); // [rows]
        }

        // residual[k] = gate[k] * signal + hc_scale[k] * signal
        var out_parts: [4]Array = undefined;
        for (0..@as(usize, @intCast(lanes))) |k| {
            const gate = try t.expandDims(sums[k], -1); // [rows, 1]
            const branch = try t.mul(signal, e.hc_scales[k]);
            out_parts[k] = try t.add(try t.mul(gate, signal), branch);
        }
        const out = try t.concat(&.{ out_parts[0], out_parts[1], out_parts[2], out_parts[3] }, -1); // [rows, lanes*hidden]
        return g.adopt(t.escape(try t.reshape(out, &.{ rows, lanes, hidden })));
    }

    /// The residual `[rows, lanes, hidden]` the streams add, `streams` `[slots, rows, lanes, hidden]`,
    /// `ids` `[slots, rows]` int32. A single slot (the engine's text forwards) is required.
    pub fn forward(e: *Engram, g: *Graph, streams: Array, ids: Array) !Array {
        const sh = ops.shape(streams);
        const n = sh[0];
        const rows: usize = @intCast(sh[1]);
        if (n != 1) return error.EngramBatchedUnsupported;

        try ops.eval(ids);
        const tok = mlx.mlx_array_data_int32(ids) orelse return error.NoData;
        const tokens = tok[0 .. rows * @as(usize, @intCast(sh[2]))];

        // 1. The n-gram rows, hashed exactly as numpy draws the multipliers.
        var row_ids: [4096 * 16]u32 = undefined;
        if (rows * e.heads > row_ids.len) return error.TooManyRows;
        var live: [4096]bool = undefined;
        if (rows > live.len) return error.TooManyRows;
        for (0..rows) |r| {
            const cur = tokens[r];
            const prev: i32 = if (r >= 1) tokens[r - 1] else -1;
            // A pad, or a span the vision tower splices, owns no n-gram.
            live[r] = cur != @as(i32, @intCast(e.pad_id)) and cur != @as(i32, @intCast(e.image_token_id)) and
                (e.video_token_id == 0 or cur != @as(i32, @intCast(e.video_token_id)));
            const base = r * e.heads;
            if (!live[r]) {
                @memset(row_ids[base..][0..e.heads], 0);
                continue;
            }
            const ids3 = [3]i32{ cur, prev, if (r >= 2) tokens[r - 2] else -1 };
            hashRows(e, ids3, row_ids[base..][0..e.heads]);
        }

        // 2. Gather the 16 x 96 rows off the streamed table, dequantized, fp8-round-tripped, bf16-rounded.
        const vec = try gpa.alloc(f32, rows * e.emb_width);
        defer gpa.free(vec);
        try gatherRows(e, row_ids[0 .. rows * e.heads], vec);

        // 3. The memory math.
        var t = g.scope();
        defer t.deinit();
        const emb0 = try t.fromSlice(f32, vec, &.{ @intCast(rows), @intCast(e.heads), @intCast(e.head_dim) });
        const normed_heads = blk: {
            // Per-head RMSNorm with torch's default eps (float32's; `nn.RMSNorm` normalizes in f32).
            const x = try t.astype(emb0, .float32);
            const sq = try t.square(x);
            const ms = try t.mean(sq, -1, true);
            const inv = try t.rsqrt(try t.addScalar(ms, defaultEps));
            const nrm = try t.astype(try t.mul(x, inv), .bfloat16);
            break :blk try t.mul(nrm, e.head_norm_w);
        };
        const emb = try t.reshape(normed_heads, &.{ @intCast(rows), @intCast(e.emb_width) });
        const shaped = try gateResidual(e, g, emb, streams); // [rows, lanes, hidden]

        // Zero the rows whose current token owns no n-gram (pad / visual) and hand back the stream dtype.
        var pad_flags = try gpa.alloc(i32, rows);
        defer gpa.free(pad_flags);
        for (0..rows) |r| pad_flags[r] = @intFromBool(live[r]);
        const keep = try t.fromSlice(i32, pad_flags, &.{ @intCast(rows), 1, 1 });
        const masked = try t.mul(shaped, try t.astype(keep, ops.dtype(shaped)));
        return g.adopt(t.escape(try t.astype(masked, ops.dtype(streams))));
    }

    /// The 16 row ids for one token's n-gram: `mix = lookup[id_0]*m0 XOR lookup[id_1]*m1 XOR ...`, reduced
    /// modulo each head's vocab size and lifted by that head's offset into the shared table.
    fn hashRows(e: *Engram, ids3: [3]i32, out: []u32) void {
        var mixed: [3]i64 = undefined;
        for (ids3, 0..) |raw, i| {
            // A token past the lookup table (or before the sequence) reads the frozen pad row.
            var id: usize = if (raw < 0) @intCast(e.pad_id) else @intCast(raw);
            if (id >= e.lookup_size) id = e.pad_id;
            const comp: i64 = e.lookup_host[id];
            mixed[i] = comp *% e.mult[i];
        }
        // Order 2: XOR of the first two; order 3: XOR of all three.
        const mix2 = mixed[0] ^ mixed[1];
        const mix3 = mixed[0] ^ mixed[1] ^ mixed[2];
        for (0..e.heads) |h| {
            const mix = if (h < e.heads / 2) mix2 else mix3;
            const m: u64 = @intCast(e.moduli[h]);
            out[h] = e.head_off[h] + @as(u32, @intCast(@mod(@as(u64, @bitCast(mix)), m)));
        }
    }
};

/// Read each row's packed codes, scales and biases off the streamed table and dequantize to f32 (then the
/// reference's bf16 cast, the fp8 round trip and a bf16 rounding again) into `out` `[rows*heads*head_dim]`.
fn gatherRows(e: *Engram, row_ids: []const u32, out: []f32) !void {
    const dim = e.head_dim;
    const words = @divExact(dim, 8); // 4-bit codes packed eight to a u32
    const wbuf = try gpa.alloc(u8, words * 4);
    defer gpa.free(wbuf);
    const sbuf = try gpa.alloc(u8, e.tbl.s_stride);
    defer gpa.free(sbuf);
    const bbuf = try gpa.alloc(u8, e.tbl.b_stride);
    defer gpa.free(bbuf);
    for (row_ids, 0..) |row, i| {
        const woff = e.tbl.w_off + @as(u64, row) * e.tbl.w_stride;
        const soff = e.tbl.s_off + @as(u64, row) * e.tbl.s_stride;
        const boff = e.tbl.b_off + @as(u64, row) * e.tbl.b_stride;
        if (crt.pread(e.tbl.fd, wbuf.ptr, wbuf.len, @intCast(woff)) != @as(isize, @intCast(wbuf.len))) return error.TableRead;
        if (crt.pread(e.tbl.fd, sbuf[0..].ptr, e.tbl.s_stride, @intCast(soff)) != @as(isize, @intCast(e.tbl.s_stride))) return error.TableRead;
        if (crt.pread(e.tbl.fd, bbuf[0..].ptr, e.tbl.b_stride, @intCast(boff)) != @as(isize, @intCast(e.tbl.b_stride))) return error.TableRead;
        // 4-bit groups of 32: a group's scale/bias pair covers 32 codes.
        const groups = @divExact(dim, 32);
        const base = i * dim;
        for (0..dim) |d| {
            const word = std.mem.readInt(u32, wbuf[(d / 8) * 4 ..][0..4], .little);
            const nib = (word >> @intCast((d % 8) * 4)) & 0xf;
            const grp = (d / 32) * 2;
            _ = groups;
            const sc: f32 = @floatCast(readBf16(sbuf[grp..][0..2]));
            const bi: f32 = @floatCast(readBf16(bbuf[grp..][0..2]));
            // The reference casts the gathered rows to the hidden dtype first: bf16 of the dequantized value.
            out[base + d] = roundBf16(@as(f32, @floatFromInt(nib)) * sc + bi);
        }
        // Then the fp8 round trip per 96-wide head vector, and the dtype cast again.
        if (e.fp8) fp8VectorScales(out[base..][0..dim], dim);
    }
}

fn readBf16(bytes: []const u8) f16 {
    const bits = std.mem.readInt(u16, bytes[0..2], .little);
    const f32bits: u32 = @as(u32, bits) << 16;
    return @floatCast(@as(f32, @bitCast(f32bits)));
}

/// Round an f32 to the nearest bf16 (ties to even), returned as f32.
fn roundBf16(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    const lsb: u32 = (bits >> 16) & 1;
    const rounded = bits +% 0x7fff +% lsb;
    return @bitCast(rounded & 0xffff0000);
}

/// torch's `nn.RMSNorm` default eps: its kernel normalizes in float32, so `eps=None` is float32's epsilon.
const defaultEps: f32 = std.math.floatEps(f32);

/// The e4m3fn grid (positive codes 1..126), ascending: exponent bias 7, 3 mantissa bits, subnormals to 2^-9.
const e4m3_table: [126]f32 = blk: {
    var t: [126]f32 = undefined;
    for (0..126) |i| {
        const code: u32 = @intCast(i + 1);
        const e = code >> 3;
        const m = code & 7;
        const mf: f32 = @as(f32, @floatFromInt(m)) / 8.0;
        if (e == 0) {
            // Subnormal: m * 2^-9, exact in f32.
            t[i] = @as(f32, @floatFromInt(m)) / 512.0;
        } else {
            const exp2: f32 = if (e >= 7)
                @as(f32, @floatFromInt(@as(u32, 1) << @intCast(e - 7)))
            else
                1.0 / @as(f32, @floatFromInt(@as(u32, 1) << @intCast(7 - e)));
            t[i] = (1.0 + mf) * exp2;
        }
    }
    break :blk t;
};

/// Round a non-negative magnitude onto the e4m3fn grid, ties to the even mantissa (torch's cast).
fn e4m3Nearest(a: f32) f32 {
    if (a <= 0) return 0;
    if (a >= e4m3_max) return e4m3_max;
    var best: f32 = e4m3_table[0];
    var best_d = @abs(a - best);
    for (e4m3_table, 0..) |v, i| {
        const d = @abs(a - v);
        // Nearest wins; a tie goes to the candidate with the even mantissa (torch's ties-to-even).
        if (d < best_d or (d == best_d and (i % 2) == 0)) {
            best = v;
            best_d = d;
        }
    }
    return best;
}

/// The per-head 96-vector fp8 round trip in place over `out` chunks of `dim`.
fn fp8VectorScales(out: []f32, dim: usize) void {
    var i: usize = 0;
    while (i < out.len) : (i += dim) {
        var mx: f32 = 0;
        for (out[i..][0..dim]) |x| mx = @max(mx, @abs(x));
        const scale = @max(mx, 1e-12) / e4m3_max;
        for (out[i..][0..dim]) |*x| {
            const q = std.math.clamp(x.* / scale, -e4m3_max, e4m3_max);
            const mag = e4m3Nearest(@abs(q));
            const signed = if (q < 0) -mag else mag;
            x.* = roundBf16(signed * scale);
        }
    }
}

// ── tests ──

const testing = std.testing;

test "hy_v4 engram: the fp8 e4m3 round trip matches torch's cast on golden magnitudes" {
    // torch: `(x/scale).clamp(+-448).to(float8_e4m3fn).to(bf16) * scale`, scale = max|x|/448.
    var vec = [_]f32{ 1.0, -0.5, 0.3333, 2.0, -3.0, 0.0 };
    fp8VectorScales(&vec, vec.len);
    // max = 3.0 -> scale = 3/448. The grid is coarse: values snap to 448-grid multiples of scale.
    for (vec) |v| try testing.expect(@abs(v) <= 3.0 + 1e-6);
    try testing.expectEqual(@as(f32, 0), vec[5]);
    // A lone value round-trips exactly to itself when it lands on the grid (1.0 * scale with max=1).
    var one = [_]f32{1.0};
    fp8VectorScales(&one, 1);
    try testing.expectApproxEqAbs(@as(f32, 1.0), one[0], 1e-4);
}

test "hy_v4 engram: the hash mixes lookup ids, multipliers and per-head moduli" {
    // A tiny synthetic Engram with a lookup table and multipliers; the oracle recomputes the mix in f64-free
    // integer arithmetic, exactly as numpy + the reference do.
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();
    var e = Engram{
        .tbl = undefined,
        .lookup_host = &[_]i32{ 7, 11, 13, 17 },
        .mult = .{ 3, 5, 7 },
        .moduli = @splat(0),
        .head_off = @splat(0),
        .heads = 16,
        .head_dim = 96,
        .emb_width = 16 * 96,
        .key_proj = @splat(.{}),
        .key_bias = @splat(.{}),
        .norm_key = @splat(.{}),
        .norm_query = @splat(.{}),
        .value_proj = .{},
        .value_bias = .{},
        .head_norm_w = .{},
        .pre_gamma = .{},
        .hc_scales = @splat(.{}),
        .pad_id = 3,
        .image_token_id = 120120,
        .video_token_id = 0,
        .lookup_size = 4,
        .fp8 = true,
        .hc_mult = 4,
        .hidden = 8,
    };
    // Order 2's heads 0..7 use moduli 5..12, order 3's heads 8..15 use 13..20.
    for (0..8) |h| e.moduli[h] = @intCast(5 + h);
    for (8..16) |h| e.moduli[h] = @intCast(13 + h - 8);
    var got: [16]u32 = undefined;
    e.hashRows(.{ 1, 0, 2 }, &got);
    // compressed: lookup[1]=11, lookup[0]=7, lookup[2]=13.
    const mix2: i64 = (11 * 3) ^ (7 * 5);
    const mix3: i64 = mix2 ^ (13 * 7);
    for (0..16) |h| {
        const mix = if (h < 8) mix2 else mix3;
        const want: u32 = @intCast(@mod(@as(u64, @bitCast(mix)), @as(u64, e.moduli[h])));
        try testing.expectEqual(want, got[h]);
    }
    // A current token that IS the pad id owns no n-gram (the caller zeroes it, not the hash).
    try testing.expectEqual(@as(i32, 3), @as(i32, @intCast(e.pad_id)));
}

test "hy_v4 engram: torch's nn.RMSNorm default eps is float32's, not bf16's" {
    try testing.expectEqual(std.math.floatEps(f32), defaultEps);
    try testing.expectEqual(@as(f32, 1.1920929e-7), defaultEps);
}

test "hy_v4 engram probe: concat sixteen bf16 head norms stacks to [1536]" {
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();
    var parts: [16]Array = undefined;
    for (0..16) |i| {
        const buf = try testing.allocator.alloc(f32, 96);
        defer testing.allocator.free(buf);
        for (buf, 0..) |*v, j| v.* = @floatFromInt(i * 97 + j);
        parts[i] = try g.astype(try g.fromSlice(f32, buf, &.{96}), .bfloat16);
    }
    defer for (parts) |p| g.drop(p);
    const cat = try g.concat(&parts, -1);
    defer g.drop(cat);
    try ops.eval(cat);
    try testing.expectEqual(@as(usize, 1536), mlx.mlx_array_size(cat));
}
