//! Video Sparse Attention for MiniMax-H3 students trained with it (FastH3 Omni), as FastVideo's
//! MLX runtime runs it (`fastvideo/mlx_runtime/minimax_h3_vsa.py`, Apache-2.0).
//!
//! The packed sequence is cut into 128-row tiles: each dense segment (text, an image or audio
//! reference, the target audio) in its own chunks, each video region (a reference video, the
//! target video) in 4x4x8 (t, h, w) blocks. Dense rows attend to every key. A video query tile
//! attends to every dense tile plus its own top-k tiles of EACH video region, ranked by mean-pooled
//! q.k. Every row then adds the trained `to_gate_compress` branch: softmax over all pooled tile
//! scores times the pooled values, scaled per row by the gate projection.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const h3_glue = @import("h3_glue.zig");
const Scope = @import("mlx_scope.zig").Scope;

const A = mlx.mlx_array;
const S = mlx.mlx_stream;

pub const TILE: u32 = 128;
const TILE_SHAPE = [3]u32{ 4, 4, 8 };

/// `MINIMAX_H3_VSA=0` runs a VSA pack with dense attention and no gate branch (the A/B arm).
pub fn enabled() bool {
    const raw = std.c.getenv("MINIMAX_H3_VSA") orelse return true;
    return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
}

/// The trained settings from the release's `fastvideo_inference.json`, copied into `config.json`.
pub const Contract = struct {
    sparsity: f64,
    /// Share of each reference video's tiles a query keeps (`p2_multi_region`); null keeps
    /// reference videos dense.
    ref_keep_rate: ?f64,

    /// Null for a pack trained without VSA; a VSA contract this engine cannot run is an error.
    pub fn fromConfig(allocator: std.mem.Allocator, config_json: []const u8) !?Contract {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, config_json, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        const o = parsed.value.object;
        const backend = o.get("attention_backend") orelse return null;
        if (backend != .string or !std.mem.eql(u8, backend.string, "VIDEO_SPARSE_ATTN_H3")) return null;
        const tile = o.get("vsa_tile_size") orelse return null;
        if (tile != .integer or tile.integer != TILE) return error.VsaTileUnsupported;
        const sparsity = number(o.get("vsa_sparsity")) orelse 0.9;
        if (!(sparsity >= 0 and sparsity < 1)) return error.BadVsaContract;
        var keep: ?f64 = null;
        if (o.get("vsa_ref_policy")) |p| if (p != .null) {
            if (p != .string or !std.mem.eql(u8, p.string, "p2_multi_region")) return error.BadVsaContract;
            const k = number(o.get("vsa_ref_keep_rate")) orelse return error.BadVsaContract;
            if (!(k > 0 and k <= 1)) return error.BadVsaContract;
            keep = k;
        };
        return .{ .sparsity = sparsity, .ref_keep_rate = keep };
    }

    fn number(v: ?std.json.Value) ?f64 {
        return switch (v orelse return null) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
    }
};

/// One packed segment, in packed order: a dense row count, or a video token grid (t, h, w).
pub const Seg = union(enum) { dense: u32, grid: [3]u32 };

/// Tiles each query keeps of a region of `n` tiles at `sparsity` (FastVideo `compute_topk`).
fn topk(sparsity: f64, n: u32) u32 {
    const k: u32 = @intFromFloat(@ceil((1.0 - sparsity) * @as(f64, @floatFromInt(n))));
    return @max(1, @min(k, n));
}

/// The tile map of one packed layout (FastVideo `build_h3_segment_geometry`): dense tiles first,
/// then each video region's tiles, the target video last.
pub const Geometry = struct {
    seq: u32,
    n_tiles: u32,
    n_prefix: u32,
    /// Valid rows per tile.
    vbs: []i32,
    /// [n_tiles * TILE]: the packed row in each tile slot, `seq` in a padding slot.
    tile_rows: []i32,
    /// [seq]: each packed row's tile slot.
    row_slot: []i32,
    /// The rows of the dense tiles, in tile order.
    dense_rows: []i32,
    /// Each video region's tile range, the target last.
    spans: [][2]u32,
    /// Tiles a video query keeps in each region.
    keep: []u32,

    /// The last segment must be the target video's grid.
    pub fn init(a: std.mem.Allocator, segs: []const Seg, c: Contract) !Geometry {
        if (segs.len == 0 or segs[segs.len - 1] != .grid) return error.BadVsaLayout;
        var seq: u32 = 0;
        var n_dense_tiles: u32 = 0;
        var n_dense_rows: u32 = 0;
        var n_video_tiles: u32 = 0;
        var n_regions: u32 = 0;
        for (segs, 0..) |sg, i| {
            const g = region(segs, i, c) orelse {
                const n = rows(sg);
                n_dense_tiles += std.math.divCeil(u32, n, TILE) catch unreachable;
                n_dense_rows += n;
                seq += n;
                continue;
            };
            for (g) |ax| if (ax == 0) return error.BadVsaLayout;
            n_video_tiles += tileCount(g);
            n_regions += 1;
            seq += g[0] * g[1] * g[2];
        }
        const n_tiles = n_dense_tiles + n_video_tiles;
        var self: Geometry = .{
            .seq = seq,
            .n_tiles = n_tiles,
            .n_prefix = n_dense_tiles,
            .vbs = try a.alloc(i32, n_tiles),
            .tile_rows = try a.alloc(i32, @as(usize, n_tiles) * TILE),
            .row_slot = try a.alloc(i32, seq),
            .dense_rows = try a.alloc(i32, n_dense_rows),
            .spans = try a.alloc([2]u32, n_regions),
            .keep = try a.alloc(u32, n_regions),
        };
        errdefer self.deinit(a);
        @memset(self.tile_rows, @intCast(seq));

        // Dense segments: segment-pure chunks of TILE rows.
        var row: u32 = 0;
        var tile: u32 = 0;
        var d: u32 = 0;
        for (segs, 0..) |sg, i| {
            if (region(segs, i, c)) |g| {
                row += g[0] * g[1] * g[2];
                continue;
            }
            const n = rows(sg);
            for (0..n) |j| {
                const r: u32 = row + @as(u32, @intCast(j));
                if (j % TILE == 0) {
                    self.vbs[tile] = @intCast(@min(TILE, n - @as(u32, @intCast(j))));
                    tile += 1;
                }
                self.tile_rows[(tile - 1) * TILE + j % TILE] = @intCast(r);
                self.dense_rows[d] = @intCast(r);
                d += 1;
            }
            row += n;
        }

        // Video regions: 4x4x8 blocks in (t, h, w) raster order, rows raster inside each block.
        row = 0;
        var ri: u32 = 0;
        for (segs, 0..) |sg, i| {
            const g = region(segs, i, c) orelse {
                row += rows(sg);
                continue;
            };
            const first = tile;
            var bt: u32 = 0;
            while (bt < g[0]) : (bt += TILE_SHAPE[0]) {
                var bh: u32 = 0;
                while (bh < g[1]) : (bh += TILE_SHAPE[1]) {
                    var bw: u32 = 0;
                    while (bw < g[2]) : (bw += TILE_SHAPE[2]) {
                        var slot: u32 = tile * TILE;
                        for (bt..@min(bt + TILE_SHAPE[0], g[0])) |t| for (bh..@min(bh + TILE_SHAPE[1], g[1])) |h| for (bw..@min(bw + TILE_SHAPE[2], g[2])) |w| {
                            self.tile_rows[slot] = @intCast(row + (t * g[1] + h) * g[2] + w);
                            slot += 1;
                        };
                        self.vbs[tile] = @intCast(slot - tile * TILE);
                        tile += 1;
                    }
                }
            }
            self.spans[ri] = .{ first, tile };
            // Reference videos keep `ref_keep_rate`, the target follows the trained sparsity; a
            // dense step (sparsity 0) keeps every region whole.
            const sp = if (c.sparsity == 0) 0 else if (ri + 1 < n_regions) 1.0 - c.ref_keep_rate.? else c.sparsity;
            self.keep[ri] = topk(sp, tile - first);
            ri += 1;
            row += g[0] * g[1] * g[2];
        }
        for (self.tile_rows, 0..) |r, slot| if (r < seq) {
            self.row_slot[@intCast(r)] = @intCast(slot);
        };
        return self;
    }

    pub fn deinit(self: *Geometry, a: std.mem.Allocator) void {
        a.free(self.vbs);
        a.free(self.tile_rows);
        a.free(self.row_slot);
        a.free(self.dense_rows);
        a.free(self.spans);
        a.free(self.keep);
    }

    /// Key tiles every video query reads: all dense tiles plus each region's keep.
    pub fn keyTiles(self: *const Geometry) u32 {
        var k = self.n_prefix;
        for (self.keep) |x| k += x;
        return k;
    }

    /// A reference video stays dense without a keep rate; the target is always a region.
    fn region(segs: []const Seg, i: usize, c: Contract) ?[3]u32 {
        return switch (segs[i]) {
            .dense => null,
            .grid => |g| if (i + 1 == segs.len or c.ref_keep_rate != null) g else null,
        };
    }

    fn rows(sg: Seg) u32 {
        return switch (sg) {
            .dense => |n| n,
            .grid => |g| g[0] * g[1] * g[2],
        };
    }

    fn tileCount(g: [3]u32) u32 {
        var n: u32 = 1;
        for (g, TILE_SHAPE) |ax, t| n *= std.math.divCeil(u32, ax, t) catch unreachable;
        return n;
    }
};

/// How video query tiles attend: tensor units, FastVideo's simdgroup kernel, or gather + SDPA
/// (any stream).
pub const Arm = enum { nax, simd, gather };

fn nax() bool {
    return @import("transformer.zig").verifyQmmNaxAvailable();
}

/// A geometry's index maps on the device, built once per generation.
pub const Plan = struct {
    allocator: std.mem.Allocator,
    geo: Geometry,
    tile_rows: A,
    dense_rows: A,
    /// [seq]: each row's source in concat(dense-row outputs, video-tile outputs).
    out_src: A,
    /// [seq]: each row's tile.
    row_tile: A,
    vbs: A,
    vbs_f: A,
    dense_sel: A,
    /// n_prefix, n_video, key tiles per video query, n_tiles: the kernel's shape.
    meta: A,
    arm: Arm,

    pub fn init(a: std.mem.Allocator, segs: []const Seg, c: Contract, s: S) !Plan {
        var geo = try Geometry.init(a, segs, c);
        errdefer geo.deinit(a);
        const g = &geo;
        const n_dense: u32 = @intCast(g.dense_rows.len);
        const out_src = try a.alloc(i32, g.seq);
        defer a.free(out_src);
        const row_tile = try a.alloc(i32, g.seq);
        defer a.free(row_tile);
        for (g.dense_rows, 0..) |r, p| out_src[@intCast(r)] = @intCast(p);
        for (g.tile_rows[g.n_prefix * TILE ..], 0..) |r, j| if (r < g.seq) {
            out_src[@intCast(r)] = @intCast(n_dense + j);
        };
        for (g.row_slot, row_tile) |slot, *t| t.* = @divTrunc(slot, @as(i32, TILE));
        const vbs_f = try a.alloc(f32, g.n_tiles);
        defer a.free(vbs_f);
        for (g.vbs, vbs_f) |n, *f| f.* = @floatFromInt(n);
        const dense_sel = try a.alloc(i32, g.n_prefix);
        defer a.free(dense_sel);
        for (dense_sel, 0..) |*x, i| x.* = @intCast(i);
        const n_video = g.n_tiles - g.n_prefix;
        const meta = [_]i32{ @intCast(g.n_prefix), @intCast(n_video), @intCast(g.keyTiles()), @intCast(g.n_tiles) };

        const arm: Arm = if (!mlx.streamIsGpu(s)) .gather else if (nax()) .nax else .simd;
        log.info("[minimax-h3] VSA: {d} dense + {d} video tiles in {d} region(s); a video tile reads {d} of {d} tiles, {s} arm\n", .{
            g.n_prefix, n_video, g.spans.len, g.keyTiles(), g.n_tiles, @tagName(arm),
        });
        return .{
            .allocator = a,
            .geo = geo,
            .tile_rows = arr(i32, g.tile_rows, &.{@intCast(g.tile_rows.len)}),
            .dense_rows = arr(i32, g.dense_rows, &.{@intCast(n_dense)}),
            .out_src = arr(i32, out_src, &.{@intCast(g.seq)}),
            .row_tile = arr(i32, row_tile, &.{@intCast(g.seq)}),
            .vbs = arr(i32, g.vbs, &.{@intCast(g.n_tiles)}),
            .vbs_f = arr(f32, vbs_f, &.{ 1, @intCast(g.n_tiles), 1 }),
            .dense_sel = arr(i32, dense_sel, &.{ 1, 1, @intCast(g.n_prefix) }),
            .meta = arr(i32, &meta, &.{4}),
            .arm = arm,
        };
    }

    pub fn deinit(self: *Plan) void {
        for ([_]A{ self.tile_rows, self.dense_rows, self.out_src, self.row_tile, self.vbs, self.vbs_f, self.dense_sel, self.meta }) |x| _ = mlx.mlx_array_free(x);
        self.geo.deinit(self.allocator);
    }

    fn arr(comptime T: type, data: []const T, shape: []const c_int) A {
        return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), if (T == i32) .int32 else .float32);
    }
};

/// q, k, v [1, H, S, D] after norm and rope, gate [S, H*D]; returns token-major [S, H*D].
pub fn attend(p: *const Plan, q: A, k: A, v: A, gate: A, scale: f32, s: S) !A {
    const a = p.allocator;
    var sc = Scope.init(a, s);
    defer sc.deinit();
    const g = &p.geo;
    const sh = mlx.getShape(q);
    const h = sh[1];
    const d = sh[3];
    const seq: c_int = @intCast(g.seq);
    const nt: c_int = @intCast(g.n_tiles);
    const np: c_int = @intCast(g.n_prefix);
    const nv: c_int = nt - np;
    const dt = mlx.mlx_array_dtype(q);
    if (sh[2] != seq) return error.VsaRowMismatch;
    // The Metal kernels read 128-wide heads.
    const arm: Arm = if (d != 128) .gather else p.arm;

    // Tiled copies [1, H, n_tiles*TILE, D], zero in padding slots.
    const zero = try sc.zeros(&.{ 1, h, 1, d }, dt);
    var tiled: [3]A = undefined;
    for (&tiled, [_]A{ q, k, v }) |*t, x| t.* = try sc.takeAxis(try sc.concat(&.{ x, zero }, 2), p.tile_rows, 2);

    // Per-tile means in f32: [H, n_tiles, D]. A Metal arm pools without materializing f32 copies:
    // multi-GB temporaries per block are fresh allocations under the server's buffer-pool cap.
    const fused = arm != .gather;
    var pooled: [3]A = undefined;
    for (&pooled, tiled) |*m, t| m.* = if (fused) try poolKernel(&sc, p, t) else blk: {
        const sum = try sc.sumAxis(try sc.astype(try sc.reshape(t, &.{ h, nt, TILE, d }), .float32), 2);
        break :blk try sc.div(try sc.reshape(sum, &.{ h, nt, d }), p.vbs_f);
    };
    const scores = try sc.div(try sc.matmul(pooled[0], try sc.transpose(pooled[1], &.{ 0, 2, 1 })), try sc.scalar(@sqrt(@as(f32, @floatFromInt(d)))));

    // Every dense tile, then each region's top-k by pooled score: [H, n_video, key tiles].
    const vs = try sc.slice(scores, 1, np, nt);
    var parts: std.ArrayList(A) = .empty;
    defer parts.deinit(a);
    try parts.append(a, try sc.broadcast(p.dense_sel, &.{ h, nv, np }));
    for (g.spans, g.keep) |span, kk| {
        const cols = try sc.slice(vs, 2, @intCast(span[0]), @intCast(span[1]));
        const top = try sc.slice(try sc.argpartition(try sc.neg(cols), @intCast(kk - 1), 2), 2, 0, @intCast(kk));
        try parts.append(a, try sc.add(try sc.astype(top, .int32), try sc.astype(try sc.scalar(@floatFromInt(span[0])), .int32)));
    }
    const sel = try sc.concat(parts.items, 2);

    const video = switch (arm) {
        .nax, .simd => try kernelAttend(&sc, p, tiled, sel, scale),
        .gather => try gatherAttend(&sc, p, tiled, sel, scale),
    };

    const dense = try sc.sdpa(try sc.takeAxis(q, p.dense_rows, 2), k, v, scale);

    // The gate branch: pooled attention over all tiles, read by each row's tile.
    const comp = try sc.matmul(try sc.softmax(scores, 2), pooled[2]);
    if (fused) return sc.out(try assembleKernel(&sc, p, dense, video, comp, gate));
    const out = try sc.takeAxis(try sc.concat(&.{ dense, video }, 2), p.out_src, 2);
    const gh = try sc.transpose(try sc.reshape(gate, &.{ seq, h, d }), &.{ 1, 0, 2 });
    const gated = try sc.astype(try sc.mul(try sc.takeAxis(comp, p.row_tile, 1), gh), dt);
    const res = try sc.add(out, try sc.reshape(gated, &.{ 1, h, seq, d }));
    return sc.out(try sc.reshape(try sc.transpose(res, &.{ 0, 2, 1, 3 }), &.{ seq, h * d }));
}

/// Per-tile f32 mean of tiled x [1, H, n_tiles * TILE, 128] over its valid rows: [H, n_tiles, 128].
const POOL_SOURCE =
    \\const uint d = thread_position_in_grid.x;
    \\const uint t = thread_position_in_grid.y;
    \\const uint h = thread_position_in_grid.z;
    \\const int n_tiles = meta[3];
    \\const int n = vbs[t];
    \\const device T* src = x + (size_t(h) * n_tiles + t) * TILE * 128 + d;
    \\float acc = 0.0f;
    \\for (int j = 0; j < n; j++) acc += float(src[size_t(j) * 128]);
    \\out[(size_t(h) * n_tiles + t) * 128 + d] = acc / float(n);
;

/// Each row's output, token-major [S, H*128]: its dense or video attention row plus the gate term
/// rounded to T first, as the op chain rounds it (`out + T(comp * gate)`).
const ASSEMBLE_SOURCE =
    \\const uint c = thread_position_in_grid.x * 4;
    \\const uint r = thread_position_in_grid.y;
    \\const int n_dense = shape[0];
    \\const int n_tiles = shape[1];
    \\const int n_video_rows = shape[2];
    \\const int hd = shape[3];
    \\const uint h = c / 128;
    \\const uint dd = c % 128;
    \\const int src = out_src[r];
    \\const device T* a = src < n_dense ? dense + (size_t(h) * n_dense + src) * 128 + dd
    \\    : video + (size_t(h) * n_video_rows + (src - n_dense)) * 128 + dd;
    \\const device float* m = comp + (size_t(h) * n_tiles + row_tile[r]) * 128 + dd;
    \\const device T* g = gate + size_t(r) * hd + c;
    \\for (int i = 0; i < 4; i++) {
    \\    const T gv = static_cast<T>(m[i] * float(g[i]));
    \\    out[size_t(r) * hd + c + i] = static_cast<T>(float(a[i]) + float(gv));
    \\}
;

var pool_kernel: ?mlx.mlx_fast_metal_kernel = null;
var assemble_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn runKernel(sc: *Scope, k: mlx.mlx_fast_metal_kernel, ins: []const A, out_shape: []const c_int, out_dt: mlx.mlx_dtype, dt: mlx.mlx_dtype, grid: [3]c_int, tg: c_int) !A {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, out_shape.ptr, out_shape.len, out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TILE", TILE));
    var outs: [1]A = undefined;
    try h3_glue.apply(k, c, ins, &outs, sc.s);
    return sc.keep(outs[0]);
}

fn poolKernel(sc: *Scope, p: *const Plan, x: A) !A {
    if (pool_kernel == null) pool_kernel = try h3_glue.makeKernel("msv_h3_vsa_pool", &.{ "x", "vbs", "meta" }, &.{"out"}, POOL_SOURCE, "");
    const h = mlx.getShape(x)[1];
    const nt: c_int = @intCast(p.geo.n_tiles);
    return runKernel(sc, pool_kernel.?, &.{ x, p.vbs, p.meta }, &.{ h, nt, 128 }, .float32, mlx.mlx_array_dtype(x), .{ 128, nt, h }, 128);
}

fn assembleKernel(sc: *Scope, p: *const Plan, dense: A, video: A, comp: A, gate: A) !A {
    if (assemble_kernel == null) assemble_kernel = try h3_glue.makeKernel("msv_h3_vsa_assemble", &.{ "dense", "video", "comp", "gate", "out_src", "row_tile", "shape" }, &.{"out"}, ASSEMBLE_SOURCE, "");
    const h = mlx.getShape(dense)[1];
    const seq: c_int = @intCast(p.geo.seq);
    const shape = try sc.fromI32(&.{ @intCast(p.geo.dense_rows.len), @intCast(p.geo.n_tiles), mlx.getShape(video)[2], h * 128 }, &.{4});
    return runKernel(sc, assemble_kernel.?, &.{ dense, video, comp, gate, p.out_src, p.row_tile, shape }, &.{ seq, h * 128 }, mlx.mlx_array_dtype(dense), mlx.mlx_array_dtype(dense), .{ h * 32, seq, 1 }, 256);
}

/// Video query tiles over their selected key tiles by gather + masked SDPA, bounded per chunk.
fn gatherAttend(sc: *Scope, p: *const Plan, tiled: [3]A, sel: A, scale: f32) !A {
    const g = &p.geo;
    const sh = mlx.getShape(tiled[0]);
    const h = sh[1];
    const d = sh[3];
    const nt: c_int = @intCast(g.n_tiles);
    const np: c_int = @intCast(g.n_prefix);
    const nv: c_int = nt - np;
    const ks: c_int = @intCast(g.keyTiles());
    const t: c_int = TILE;

    const flat = try sc.add(sel, try sc.reshape(try sc.arange(0, h * nt, nt), &.{ h, 1, 1 }));
    const k_rows = try sc.reshape(tiled[1], &.{ h * nt, t * d });
    const v_rows = try sc.reshape(tiled[2], &.{ h * nt, t * d });
    const qv = try sc.reshape(try sc.slice(tiled[0], 2, np * t, nt * t), &.{ h, nv, t, d });
    const sizes = try sc.reshape(try sc.takeAxis(p.vbs, sel, 0), &.{ h, nv, ks, 1 });
    const valid = try sc.less(try sc.reshape(try sc.arange(0, t, 1), &.{ 1, 1, 1, t }), sizes);

    const bytes_per_tile: usize = 2 * @as(usize, @intCast(h * ks * t * d)) * (mlx.dtypeBytes(mlx.mlx_array_dtype(tiled[1])) orelse 4);
    const chunk: c_int = @intCast(@max(1, @min(@as(usize, @intCast(nv)), (512 << 20) / bytes_per_tile)));
    var outs: std.ArrayList(A) = .empty;
    defer outs.deinit(sc.a);
    var lo: c_int = 0;
    while (lo < nv) : (lo += chunk) {
        // Each chunk's gathered K/V die with its own scope once its output is evaluated.
        var cs = Scope.init(sc.a, sc.s);
        defer cs.deinit();
        const hi = @min(lo + chunk, nv);
        const n = hi - lo;
        const ids = try cs.reshape(try cs.slice(flat, 1, lo, hi), &.{-1});
        const gk = try cs.reshape(try cs.take(k_rows, ids), &.{ 1, h * n, ks * t, d });
        const gv = try cs.reshape(try cs.take(v_rows, ids), &.{ 1, h * n, ks * t, d });
        const qc = try cs.reshape(try cs.slice(qv, 1, lo, hi), &.{ 1, h * n, t, d });
        const m = try cs.reshape(try cs.slice(valid, 1, lo, hi), &.{ 1, h * n, 1, ks * t });
        const o = try sc.keep(cs.out(try cs.reshape(try cs.sdpaMasked(qc, gk, gv, scale, m), &.{ h, n * t, d })));
        if (nv > chunk) try mlx.check(mlx.mlx_array_eval(o));
        try outs.append(sc.a, o);
    }
    return sc.reshape(try sc.concat(outs.items, 1), &.{ 1, h, nv * t, d });
}

// FastVideo's SIMD-group block-sparse kernel (`minimax_h3_vsa_simd.py`), reading our head-major
// [1, H, n_tiles*TILE, D] tiles. One threadgroup = (32-row slice of a video query tile, tile,
// head); each of its 4 simdgroups owns 8 query rows. Per selected key tile, 16-row K/V chunks are
// staged in threadgroup memory and consumed with 8x8 simdgroup MMAs and an f32 online softmax.
// Only a ragged key tile's valid rows are scored; rows past a query tile's valid count are never
// written (no reader looks at them).
const KERNEL_SOURCE =
    \\constexpr int D = 128;
    \\constexpr int NSG = 4;
    \\constexpr int NT = NSG * 32;
    \\constexpr int BK = 16;
    \\constexpr int NKB = BK / 8;
    \\constexpr int LD = D + 8;
    \\constexpr float NEG = -3.0e38f;
    \\threadgroup T ksm[BK * LD];
    \\threadgroup T vsm[BK * LD];
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint sid = simdgroup_index_in_threadgroup;
    \\const uint lane = thread_index_in_simdgroup;
    \\const int qsub = int(threadgroup_position_in_grid.x);
    \\const int q_tile = int(threadgroup_position_in_grid.y);
    \\const int head = int(threadgroup_position_in_grid.z);
    \\const int n_prefix = meta[0];
    \\const int n_video = meta[1];
    \\const int k_sel = meta[2];
    \\const int n_tiles = meta[3];
    \\const float scale_log2 = scale * 1.44269504088896341f;
    \\const int qt = n_prefix + q_tile;
    \\const int q_valid = vbs[qt];
    \\const int row_base = qsub * NSG * 8;
    \\if (row_base >= q_valid) return;
    \\const long plane = long(head) * n_tiles * TILE;
    \\const int srow0 = row_base + int(sid) * 8;
    \\const int fm = int(((lane / 4) & 4) + ((lane / 2) % 4));
    \\const int fn = int(((lane / 4) & 2) * 2 + (lane % 2) * 2);
    \\simdgroup_matrix<T, 8, 8> qf[16];
    \\simdgroup_matrix<float, 8, 8> acc[16];
    \\const device T* qp = q + (plane + long(qt) * TILE + srow0) * D;
    \\for (int kk = 0; kk < 16; kk++) {
    \\    simdgroup_load(qf[kk], qp + kk * 8, D);
    \\    acc[kk] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\}
    \\float row_m = NEG;
    \\float row_l = 0.0f;
    \\const device int* picks = sel + (long(head) * n_video + q_tile) * k_sel;
    \\for (int s = 0; s < k_sel; s++) {
    \\    const int kt = picks[s];
    \\    const int k_valid = vbs[kt];
    \\    const long kv0 = plane + long(kt) * TILE;
    \\    for (int j0 = 0; j0 < k_valid; j0 += BK) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        for (int i = int(tid); i < BK * (D / 4); i += NT) {
    \\            const int r = i / (D / 4);
    \\            const int c4 = i - r * (D / 4);
    \\            const long src = (kv0 + j0 + r) * D + c4 * 4;
    \\            *(threadgroup vec<T, 4>*)(ksm + r * LD + c4 * 4) = *(const device vec<T, 4>*)(k + src);
    \\            *(threadgroup vec<T, 4>*)(vsm + r * LD + c4 * 4) = *(const device vec<T, 4>*)(v + src);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        simdgroup_matrix<float, 8, 8> sf[NKB];
    \\        for (int c = 0; c < NKB; c++) sf[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\        for (int kk = 0; kk < 16; kk++) {
    \\            for (int c = 0; c < NKB; c++) {
    \\                simdgroup_matrix<T, 8, 8> km;
    \\                simdgroup_load(km, (const threadgroup T*)(ksm + c * 8 * LD + kk * 8), LD, ulong2(0, 0), true);
    \\                simdgroup_multiply_accumulate(sf[c], qf[kk], km, sf[c]);
    \\            }
    \\        }
    \\        simdgroup_matrix<T, 8, 8> pf[NKB];
    \\        float cmax = NEG;
    \\        for (int c = 0; c < NKB; c++) {
    \\            for (int e = 0; e < 2; e++) {
    \\                const int col = j0 + c * 8 + fn + e;
    \\                const float sv = col < k_valid ? sf[c].thread_elements()[e] * scale_log2 : NEG;
    \\                sf[c].thread_elements()[e] = sv;
    \\                cmax = metal::max(cmax, sv);
    \\            }
    \\        }
    \\        cmax = metal::max(cmax, simd_shuffle_xor(cmax, ushort(1)));
    \\        cmax = metal::max(cmax, simd_shuffle_xor(cmax, ushort(8)));
    \\        const float m_new = metal::max(row_m, cmax);
    \\        const float alpha = metal::fast::exp2(row_m - m_new);
    \\        row_m = m_new;
    \\        row_l *= alpha;
    \\        for (int c = 0; c < NKB; c++) {
    \\            for (int e = 0; e < 2; e++) {
    \\                const float pv = metal::fast::exp2(sf[c].thread_elements()[e] - m_new);
    \\                row_l += pv;
    \\                pf[c].thread_elements()[e] = static_cast<T>(pv);
    \\            }
    \\        }
    \\        for (int kk = 0; kk < 16; kk++) {
    \\            acc[kk].thread_elements()[0] *= alpha;
    \\            acc[kk].thread_elements()[1] *= alpha;
    \\        }
    \\        for (int kk = 0; kk < 16; kk++) {
    \\            for (int c = 0; c < NKB; c++) {
    \\                simdgroup_matrix<T, 8, 8> vm;
    \\                simdgroup_load(vm, (const threadgroup T*)(vsm + c * 8 * LD + kk * 8), LD);
    \\                simdgroup_multiply_accumulate(acc[kk], pf[c], vm, acc[kk]);
    \\            }
    \\        }
    \\    }
    \\}
    \\float l = row_l;
    \\l += simd_shuffle_xor(l, ushort(1));
    \\l += simd_shuffle_xor(l, ushort(8));
    \\const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    \\const int orow = srow0 + fm;
    \\if (orow < q_valid) {
    \\    device T* op = out + ((long(head) * n_video + q_tile) * TILE + orow) * D + fn;
    \\    for (int kk = 0; kk < 16; kk++) {
    \\        op[kk * 8] = static_cast<T>(acc[kk].thread_elements()[0] * inv);
    \\        op[kk * 8 + 1] = static_cast<T>(acc[kk].thread_elements()[1] * inv);
    \\    }
    \\}
;
const KERNEL_HEADER =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
;

const NAX_SOURCE =
    \\omlx_nax::h3_vsa_nax<T, 4, TILE>(q, k, v, out, sel, vbs, meta, scale,
    \\    simdgroup_index_in_threadgroup, thread_index_in_simdgroup, threadgroup_position_in_grid);
;

/// One compiled kernel and its launch config, cached per shape.
const Launch = struct {
    handle: ?mlx.mlx_fast_metal_kernel = null,
    key: ?struct { h: c_int, nv: c_int, dt: mlx.mlx_dtype } = null,
    cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null },
};
var launches: [2]Launch = .{ .{}, .{} };

/// Both kernels write [1, H, n_video * TILE, 128] from 128-thread threadgroups: the simdgroup kernel
/// one per (32-row slice, tile, head), the NAX kernel one per (64-row block, head).
fn kernelAttend(sc: *Scope, p: *const Plan, tiled: [3]A, sel: A, scale: f32) !A {
    const sh = mlx.getShape(tiled[0]);
    const dt = mlx.mlx_array_dtype(tiled[0]);
    const nv: c_int = @intCast(p.geo.n_tiles - p.geo.n_prefix);
    const is_nax = p.arm == .nax;
    const l = &launches[@intFromBool(is_nax)];
    const ins = [_][*:0]const u8{ "q", "k", "v", "sel", "vbs", "meta", "scale" };
    if (l.handle == null) l.handle = if (is_nax)
        try h3_glue.makeKernel("msv_h3_vsa_nax", &ins, &.{"out"}, NAX_SOURCE, @embedFile("kernels/nax_tiles.metal") ++ @embedFile("kernels/nax_attention.metal") ++ @embedFile("kernels/h3_vsa_nax.metal"))
    else
        try h3_glue.makeKernel("msv_h3_vsa", &ins, &.{"out"}, KERNEL_SOURCE, KERNEL_HEADER);
    if (l.key == null or l.key.?.h != sh[1] or l.key.?.nv != nv or l.key.?.dt != dt) {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ 1, sh[1], nv * TILE, 128 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 4, dt));
        if (is_nax) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 128 * nv * (TILE / 64), sh[1], 1));
        } else {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 128 * (TILE / 32), nv, sh[1]));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TILE", TILE));
        if (l.cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(l.cfg);
        l.cfg = c;
        l.key = .{ .h = sh[1], .nv = nv, .dt = dt };
    }
    // A 0-dim f32 binds `constant float&`.
    const scale_a = try sc.scalar(scale);
    var outs: [1]A = undefined;
    try h3_glue.apply(l.handle.?, l.cfg, &.{ tiled[0], tiled[1], tiled[2], sel, p.vbs, p.meta, scale_a }, &outs, sc.s);
    return sc.keep(outs[0]);
}

// ── tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;

/// The fixture's layout: text | image ref | ref soundtrack | ref video | target audio | target video.
const FIXTURE_SEGS = [_]Seg{ .{ .dense = 37 }, .{ .dense = 20 }, .{ .dense = 10 }, .{ .grid = .{ 4, 4, 12 } }, .{ .dense = 6 }, .{ .grid = .{ 6, 6, 10 } } };
const FIXTURE_CONTRACT = Contract{ .sparsity = 0.75, .ref_keep_rate = 0.5 };

test "h3 vsa: the contract is read from the pack's config, refused when it cannot run" {
    const a = testing.allocator;
    const omni = "{\"attention_backend\":\"VIDEO_SPARSE_ATTN_H3\",\"vsa_sparsity\":0.9,\"vsa_tile_size\":128," ++
        "\"vsa_ref_policy\":\"p2_multi_region\",\"vsa_ref_keep_rate\":0.1}";
    const c = (try Contract.fromConfig(a, omni)).?;
    try testing.expectEqual(@as(f64, 0.9), c.sparsity);
    try testing.expectEqual(@as(?f64, 0.1), c.ref_keep_rate);
    try testing.expect((try Contract.fromConfig(a, "{\"model_type\":\"minimax_h3\"}")) == null);
    try testing.expect((try Contract.fromConfig(a, "{\"attention_backend\":\"SDPA\",\"vsa_tile_size\":128}")) == null);
    const no_policy = (try Contract.fromConfig(a, "{\"attention_backend\":\"VIDEO_SPARSE_ATTN_H3\",\"vsa_tile_size\":128}")).?;
    try testing.expectEqual(@as(?f64, null), no_policy.ref_keep_rate);
    try testing.expectError(error.VsaTileUnsupported, Contract.fromConfig(a, "{\"attention_backend\":\"VIDEO_SPARSE_ATTN_H3\",\"vsa_tile_size\":64}"));
    try testing.expectError(error.BadVsaContract, Contract.fromConfig(a, "{\"attention_backend\":\"VIDEO_SPARSE_ATTN_H3\",\"vsa_tile_size\":128,\"vsa_ref_policy\":\"p2_multi_region\"}"));
    try testing.expectError(error.BadVsaContract, Contract.fromConfig(a, "{\"attention_backend\":\"VIDEO_SPARSE_ATTN_H3\",\"vsa_tile_size\":128,\"vsa_sparsity\":1.0}"));
}

test "h3 vsa: the tile map matches FastVideo's build_h3_segment_geometry" {
    // Expected values from FastVideo's own builder (tests/dump_h3_vsa_fixtures.py; row slots from its untile map).
    const a = testing.allocator;
    var g = try Geometry.init(a, &FIXTURE_SEGS, FIXTURE_CONTRACT);
    defer g.deinit(a);
    try testing.expectEqual(@as(u32, 625), g.seq);
    try testing.expectEqual(@as(u32, 4), g.n_prefix);
    try testing.expectEqualSlices(i32, &.{ 37, 20, 10, 6, 128, 64, 128, 32, 64, 16, 64, 16, 32, 8 }, g.vbs);
    try testing.expectEqualSlices([2]u32, &.{ .{ 4, 6 }, .{ 6, 14 } }, g.spans);
    const rows = [_]usize{ 0, 36, 37, 57, 67, 75, 259, 265, 275, 624 };
    const slots = [_]i32{ 0, 36, 128, 256, 512, 640, 384, 768, 776, 1671 };
    for (rows, slots) |r, want| try testing.expectEqual(want, g.row_slot[r]);
    // ceil(0.5 * 2) of the reference video, ceil(0.25 * 8) of the target.
    try testing.expectEqualSlices(u32, &.{ 1, 2 }, g.keep);
    try testing.expectEqual(@as(u32, 7), g.keyTiles());
    // The trained Omni sparsity on a 47-tile target keeps ceil(0.1 * 47) = 5, never fewer than one.
    try testing.expectEqual(@as(u32, 5), topk(0.9, 47));
    try testing.expectEqual(@as(u32, 1), topk(0.9, 3));
}

test "h3 vsa: without a keep rate a reference video stays dense" {
    const a = testing.allocator;
    var g = try Geometry.init(a, &FIXTURE_SEGS, .{ .sparsity = 0.75, .ref_keep_rate = null });
    defer g.deinit(a);
    // The reference video's 192 rows chunk into two dense tiles; only the target is a region.
    try testing.expectEqual(@as(u32, 6), g.n_prefix);
    try testing.expectEqual(@as(usize, 1), g.spans.len);
    try testing.expectError(error.BadVsaLayout, Geometry.init(a, &.{.{ .dense = 4 }}, FIXTURE_CONTRACT));
}

test "h3 vsa: a dense step keeps every region whole, whatever the reference keep rate" {
    const a = testing.allocator;
    var g = try Geometry.init(a, &FIXTURE_SEGS, .{ .sparsity = 0, .ref_keep_rate = 0.5 });
    defer g.deinit(a);
    try testing.expectEqualSlices(u32, &.{ 2, 8 }, g.keep);
}

test "h3 vsa: rows that do not match the layout are refused" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var plan = try Plan.init(a, &FIXTURE_SEGS, FIXTURE_CONTRACT, s);
    defer plan.deinit();
    var sc = Scope.init(a, s);
    defer sc.deinit();
    const x = try sc.zeros(&.{ 1, 2, 624, 128 }, .float32);
    try testing.expectError(error.VsaRowMismatch, attend(&plan, x, x, x, try sc.zeros(&.{ 624, 256 }, .float32), 1.0, s));
}

test "h3 vsa: the gather arm equals FastVideo's reference on its own fixture" {
    // Bar: FastVideo's `h3_vsa_attention(impl="reference")` in f32 on the CPU, gate branch included.
    const path = std.c.getenv("H3_VSA_FIXTURE") orelse return error.SkipZigTest;
    log.enableStderr();
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tensors = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensors);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_load_safetensors(&tensors, &meta, path, s));
    var t: [5]A = undefined;
    for (&t, [_][*:0]const u8{ "q", "k", "v", "gate", "out" }) |*x, name| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_map_string_to_array_get(x, tensors, name));
    }
    defer for (t) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var plan = try Plan.init(a, &FIXTURE_SEGS, FIXTURE_CONTRACT, s);
    defer plan.deinit();

    var sc = Scope.init(a, s);
    defer sc.deinit();
    // [S, H, D] -> [1, H, S, D]
    var hm: [3]A = undefined;
    for (&hm, t[0..3]) |*x, src| x.* = try sc.reshape(try sc.transpose(src, &.{ 1, 0, 2 }), &.{ 1, 2, 625, 128 });
    const got = try sc.keep(try attend(&plan, hm[0], hm[1], hm[2], try sc.reshape(t[3], &.{ 625, 256 }), 1.0 / @sqrt(128.0), s));
    const want = try sc.reshape(t[4], &.{ 625, 256 });
    const dot = try sc.sumAxis(try sc.reshape(try sc.mul(got, want), &.{-1}), 0);
    const gg = try sc.sumAxis(try sc.reshape(try sc.mul(got, got), &.{-1}), 0);
    const ww = try sc.sumAxis(try sc.reshape(try sc.mul(want, want), &.{-1}), 0);
    for ([_]A{ dot, gg, ww }) |x| try mlx.check(mlx.mlx_array_eval(x));
    const fd = mlx.mlx_array_data_float32(dot).?[0];
    const fg = mlx.mlx_array_data_float32(gg).?[0];
    const fw = mlx.mlx_array_data_float32(ww).?[0];
    const cos = fd / @sqrt(fg * fw);
    const rms_ratio = @sqrt(fg / fw);
    log.info("[h3-vsa] gather arm vs FastVideo: cos {d:.7}, rms ratio {d:.6}\n", .{ cos, rms_ratio });
    try testing.expect(cos > 0.99999);
    try testing.expectApproxEqAbs(@as(f32, 1.0), rms_ratio, 1e-4);
}

/// Random [1, H, seq, 128] q/k/v and a [seq, H*128] gate, bf16.
fn randomInputs(sc: *Scope, h: c_int, seq: c_int) ![4]A {
    var out: [4]A = undefined;
    for (&out, 0..) |*x, i| {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, 11 + i));
        const shape: []const c_int = if (i == 3) &.{ seq, h * 128 } else &.{ 1, h, seq, 128 };
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&r, shape.ptr, shape.len, .bfloat16, 0, 1, key, sc.s));
        x.* = try sc.keep(r);
    }
    return out;
}

fn rmsDiff(sc: *Scope, x: A, y: A) !f32 {
    const d = try sc.sub(try sc.astype(x, .float32), try sc.astype(y, .float32));
    const m = try sc.sumAxis(try sc.reshape(try sc.mul(d, d), &.{-1}), 0);
    try mlx.check(mlx.mlx_array_eval(m));
    return @sqrt(mlx.mlx_array_data_float32(m).?[0] / @as(f32, @floatFromInt(mlx.mlx_array_size(x))));
}

test "h3 vsa kernel: each Metal arm is no further from f32 truth than the bf16 gather arm" {
    // Truth = the gather arm on the same values in f32 (pooling is f32 either way, so the
    // selection is identical); ragged tiles in both a reference region and the target.
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const segs = [_]Seg{ .{ .dense = 300 }, .{ .dense = 256 }, .{ .grid = .{ 6, 6, 12 } }, .{ .dense = 64 }, .{ .grid = .{ 9, 10, 20 } } };
    var plan = try Plan.init(a, &segs, .{ .sparsity = 0.9, .ref_keep_rate = 0.1 }, s);
    defer plan.deinit();
    var gather_plan = plan;
    gather_plan.arm = .gather;

    var sc = Scope.init(a, s);
    defer sc.deinit();
    const h = 4;
    const seq: c_int = @intCast(plan.geo.seq);
    const x = try randomInputs(&sc, h, seq);
    var xf: [4]A = undefined;
    for (&xf, x) |*f, b| f.* = try sc.astype(b, .float32);
    const scale: f32 = 1.0 / @sqrt(128.0);
    const gath = try sc.keep(try attend(&gather_plan, x[0], x[1], x[2], x[3], scale, s));
    const truth = try sc.keep(try attend(&gather_plan, xf[0], xf[1], xf[2], xf[3], scale, s));
    const e_gath = try rmsDiff(&sc, gath, truth);
    log.enableStderr();
    for ([_]Arm{ .simd, .nax }) |arm| {
        if (arm == .nax and !nax()) continue;
        var kp = plan;
        kp.arm = arm;
        const e = try rmsDiff(&sc, try sc.keep(try attend(&kp, x[0], x[1], x[2], x[3], scale, s)), truth);
        log.info("[h3-vsa] rms vs f32 truth: {s} {d:.6}, gather {d:.6}\n", .{ @tagName(arm), e, e_gath });
        try testing.expect(std.math.isFinite(e));
        try testing.expect(e <= 1.5 * e_gath + 1e-4);
    }
}

test "h3 vsa kernel: a head width the kernels cannot read runs the gather arm end to end" {
    // Bar: a Metal plan on 64-wide heads equals the gather plan on the same values.
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const segs = [_]Seg{ .{ .dense = 200 }, .{ .grid = .{ 6, 6, 12 } } };
    var plan = try Plan.init(a, &segs, .{ .sparsity = 0.9, .ref_keep_rate = null }, s);
    defer plan.deinit();
    var gather_plan = plan;
    gather_plan.arm = .gather;
    var sc = Scope.init(a, s);
    defer sc.deinit();
    const seq: c_int = @intCast(plan.geo.seq);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 3));
    var x: [4]A = undefined;
    for (&x, 0..) |*t, i| {
        const shape: []const c_int = if (i == 3) &.{ seq, 2 * 64 } else &.{ 1, 2, seq, 64 };
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&r, shape.ptr, shape.len, .float32, 0, 1, key, s));
        t.* = try sc.keep(r);
    }
    const want = try sc.keep(try attend(&gather_plan, x[0], x[1], x[2], x[3], 0.125, s));
    const got = try sc.keep(try attend(&plan, x[0], x[1], x[2], x[3], 0.125, s));
    try testing.expect(try rmsDiff(&sc, got, want) < 1e-6);
}

test "h3 vsa kernel: ubench against dense attention at a 720p clip's packed shape" {
    // Under the server's 8 GB buffer-pool cap: without it the multi-GB temporaries price low.
    if (!@import("transformer.zig").diagEnvValueOn(std.c.getenv("H3_VSA_UBENCH"))) return error.SkipZigTest;
    var prev_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&prev_limit, 8 << 30));
    defer _ = mlx.mlx_set_cache_limit(&prev_limit, prev_limit);
    log.enableStderr();
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    // FastH3 Omni at 1280x704 x 124 frames with one image reference: text (incl. its vision tokens),
    // the reference, the target audio, 37 x 22 x 40 video rows.
    const segs = [_]Seg{ .{ .dense = 960 }, .{ .dense = 900 }, .{ .dense = 414 }, .{ .grid = .{ 37, 22, 40 } } };
    var plan = try Plan.init(a, &segs, .{ .sparsity = 0.9, .ref_keep_rate = 0.1 }, s);
    defer plan.deinit();
    var sc = Scope.init(a, s);
    defer sc.deinit();
    const seq: c_int = @intCast(plan.geo.seq);
    const x = try randomInputs(&sc, 56, seq);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    var hid = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&hid, &[_]c_int{ seq, 5376 }, 2, .bfloat16, 0, 1, key, s));
    const xin = try sc.keep(hid);
    var gw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&gw, &[_]c_int{ 5376, 7168 }, 2, .bfloat16, 0, 0.01, key, s));
    const gate_w = try sc.keep(gw);
    for (x ++ [_]A{ xin, gate_w }) |t| try mlx.check(mlx.mlx_array_eval(t));
    const scale: f32 = 1.0 / @sqrt(128.0);
    const names = [_][]const u8{ "dense sdpa bf16", "dense int8 (w8a8 default)", "vsa", "gate projection bf16" };
    for (names, 0..) |name, arm| {
        var best: u64 = std.math.maxInt(u64);
        for (0..4) |_| {
            const t0 = std.Io.Timestamp.now(io, .boot);
            const o = switch (arm) {
                0 => try sc.sdpa(x[0], x[1], x[2], scale),
                1 => try sc.keep((try @import("int8_attn.zig").attention(x[0], x[1], x[2], scale, s)) orelse return error.SkipZigTest),
                2 => try sc.keep(try attend(&plan, x[0], x[1], x[2], x[3], scale, s)),
                else => try sc.matmul(xin, gate_w),
            };
            try mlx.check(mlx.mlx_array_eval(o));
            best = @min(best, @as(u64, @intCast(t0.untilNow(io, .boot).nanoseconds)) / 1_000_000);
        }
        log.info("[h3-vsa ubench] {s}: {d} ms per layer at {d} rows\n", .{ name, best, plan.geo.seq });
    }
}
