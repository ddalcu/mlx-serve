//! The rotary angle rows the fused q/k norm+RoPE kernels read, cached per model so every layer of one forward shares
//! one table, and the hd-256 kernel's call as the dense decoder makes it.
//! copied_from: sdk/blocks/attention/proj.zig (`qkAngleRowsFor`), engines/mlx/forward/dense.zig (the hd-256
//! `fusedQkNormRope256` decode arm).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const qk_norm_rope = @import("../../blocks/rope/qk_norm_rope.zig");

const block = @import("../block.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;

/// One table per rope family (0 = base theta, 1 = the scaled spectrum: YaRN, a global layer), rebuilt when its key
/// moves. Owned by the model; free with `deinit`.
pub const AngleCache = struct {
    slots: [2]Slot = .{ .{}, .{} },

    const Slot = struct { arr: Array = .{ .ctx = null }, offset: c_int = 0, rd: c_int = 0, rows: c_int = 0, mscale: f32 = 0, scale: f32 = 0 };

    pub fn deinit(self: *AngleCache) void {
        for (&self.slots) |*s| if (s.arr.ctx != null) {
            _ = mlx.mlx_array_free(s.arr);
            s.* = .{};
        };
    }

    /// `[rows * rd]` cos|sin rows for positions `offset..offset+rows`, borrowed from the cache.
    pub fn rowsFor(self: *AngleCache, s: ops.Stream, family: usize, rd: c_int, base: mlx.mlx_optional_float, freqs: Array, offset: c_int, rows: c_int, mscale: f32, scale: f32) !Array {
        const slot = &self.slots[family];
        if (slot.arr.ctx != null and slot.offset == offset and slot.rd == rd and slot.rows == rows and slot.mscale == mscale and slot.scale == scale) return slot.arr;
        if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
        slot.arr = .{ .ctx = null };
        slot.arr = try qk_norm_rope.ropeAngleRows(s, rd, base, freqs, offset, rows, std.heap.c_allocator, mscale, scale);
        slot.* = .{ .arr = slot.arr, .offset = offset, .rd = rd, .rows = rows, .mscale = mscale, .scale = scale };
        return slot.arr;
    }
};

/// The angle rows from `cache`, or built once per forward when the model keeps none.
pub fn angles(c: *const Ctx, cache: ?*AngleCache, family: usize, rd: c_int, base: mlx.mlx_optional_float, freqs: Array, offset: c_int, rows: c_int, mscale: f32, scale: f32) !Array {
    const g = c.g;
    if (cache) |ac| return ac.rowsFor(g.s, family, rd, base, freqs, offset, rows, mscale, scale);
    const theta: i64 = if (base.has_value) @as(u32, @bitCast(base.value)) else -1;
    const key: block.Memo.Key = .{ .what = .angles, .n = .{ rd, theta, @intCast(@intFromPtr(freqs.ctx)), offset, rows, @as(u32, @bitCast(mscale)), @as(u32, @bitCast(scale)) } };
    if (c.recall(key)) |m| return m;
    return c.remember(key, try g.adopt(try qk_norm_rope.ropeAngleRows(g.s, rd, base, freqs, offset, rows, std.heap.c_allocator, mscale, scale)));
}

/// q, k (and a normed v) normed, transposed to `[1, H, S, 256]` and rotated in one dispatch; null when the kernel
/// declines (the caller keeps the composed chain). The outputs are owned by `g`; the third is null without `v`.
pub fn fused256(g: *Graph, q: Array, k: Array, v: Array, q_norm: Array, k_norm: Array, v_norm: Array, ang: Array, eps_arr: Array, hq: c_int, hk: c_int, seq: c_int, rd: c_int) !?[3]Array {
    const r = (qk_norm_rope.fusedQkNormRope256(g.s, q, k, v, q_norm, k_norm, v_norm, ang, eps_arr, hq, hk, seq, rd) catch null) orelse return null;
    var out: [3]Array = undefined;
    for (r, 0..) |a, i| out[i] = if (a.ctx == null) a else g.adopt(a) catch |e| {
        for (r[i + 1 ..]) |b| if (b.ctx != null) {
            _ = mlx.mlx_array_free(b);
        };
        return e;
    };
    return out;
}
