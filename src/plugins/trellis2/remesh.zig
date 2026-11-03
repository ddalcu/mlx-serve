//! Remesh of an arbitrary triangle soup into a clean closed manifold surface (pure CPU, no MLX).
//!
//! TRELLIS.2's dual grid is arbitrary-topology by design: open sheets, double layers and edges shared by
//! three or four faces (a fifth of a car's edges). The reference exports through a narrow-band dual
//! contouring remesh; this is the sign-free version of the same idea. It samples the UNSIGNED distance to the
//! soup on a voxel grid and marching-cubes the iso-surface at `OFFSET` cells: a watertight tube around every
//! sheet. A closed hull becomes an outer skin (facing out) and an inner skin (facing the hollow), both oriented
//! away from the original surface, so no inside/outside decision is ever made on a leaky shell.
//!
//! The grid is cut into slabs along x; each slab owns a dense window of the field, runs independently on a
//! worker, and shares only edge keys with its neighbours, so memory stays a few tens of MB at any resolution.

const std = @import("std");
const mc = @import("../../sdk/mesh/marching_cubes.zig");
const tmesh = @import("mesh.zig");
const sparse = @import("sparse.zig");

const TriMesh = tmesh.TriMesh;

/// Tube radius in cells. Below ~0.5 a sheet that sits midway between two grid planes would vanish.
pub const OFFSET: f32 = 0.75;
/// Distances past this are never needed: a corner next to an inside one is closer than `OFFSET + 1`.
const CAP: f32 = OFFSET + 1.0;
/// Empty cells kept around the box so tubes are never clipped.
const PAD: usize = 2;
const SLAB: usize = 16;
const MAX_THREADS: usize = 16;

/// The point of triangle `abc` nearest to `p` (Ericson, Real-Time Collision Detection 5.1.5).
fn closestPoint(p: [3]f32, a: [3]f32, b: [3]f32, c: [3]f32) [3]f32 {
    const ab = sub3(b, a);
    const ac = sub3(c, a);
    const ap = sub3(p, a);
    const d1 = dot3(ab, ap);
    const d2 = dot3(ac, ap);
    if (d1 <= 0 and d2 <= 0) return a;
    const bp = sub3(p, b);
    const d3 = dot3(ab, bp);
    const d4 = dot3(ac, bp);
    if (d3 >= 0 and d4 <= d3) return b;
    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 and d1 >= 0 and d3 <= 0) {
        const v = d1 / (d1 - d3);
        return .{ a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2] };
    }
    const cp = sub3(p, c);
    const d5 = dot3(ab, cp);
    const d6 = dot3(ac, cp);
    if (d6 >= 0 and d5 <= d6) return c;
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 and d2 >= 0 and d6 <= 0) {
        const w = d2 / (d2 - d6);
        return .{ a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2] };
    }
    const va = d3 * d6 - d5 * d4;
    if (va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0) {
        const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return .{ b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2]) };
    }
    const sum = va + vb + vc;
    if (sum == 0) return a; // a sliver with no area
    const v = vb / sum;
    const w = vc / sum;
    return .{ a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w };
}

fn pointTriDist2(p: [3]f32, a: [3]f32, b: [3]f32, c: [3]f32) f32 {
    return dist2(p, closestPoint(p, a, b, c));
}

fn sub3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn dot3(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn dist2(a: [3]f32, b: [3]f32) f32 {
    const d = sub3(a, b);
    return dot3(d, d);
}

const SlabOut = struct {
    arena: std.heap.ArenaAllocator,
    keys: []const u64 = &.{},
    pos: []const f32 = &.{},
    /// Per vertex: the grid axis (1..3) along which the field grows, signed.
    out_dir: []const i8 = &.{},
    tris: []const u32 = &.{},
};

const Ctx = struct {
    n: usize, // cells per axis
    origin: f32,
    h: f32,
    cells: []const f32, // triangle corners in cell units, 9 per triangle
    slab_start: []const u32, // nslabs + 1 offsets into slab_tris
    slab_tris: []const u32,
    results: []SlabOut,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
};

/// The iso-surface of the distance field at `OFFSET` cells over `res` cells across [-0.5, 0.5]^3, as a closed
/// triangle mesh wound so its normals point away from the soup. New mesh; `m` is untouched.
pub fn remesh(gpa: std.mem.Allocator, m: TriMesh, res: u32) !TriMesh {
    const n: usize = @as(usize, res) + 2 * PAD;
    const h: f32 = 1.0 / @as(f32, @floatFromInt(res));
    const origin: f32 = -0.5 - @as(f32, @floatFromInt(PAD)) * h;
    const nt = m.tris.len / 3;
    const nslabs = (n + SLAB - 1) / SLAB;

    const cells = try gpa.alloc(f32, nt * 9);
    defer gpa.free(cells);
    for (0..nt) |t| for (0..3) |k| for (0..3) |a| {
        cells[t * 9 + k * 3 + a] = (m.verts[m.tris[t * 3 + k] * 3 + a] - origin) / h;
    };

    // Triangles per slab: those whose x extent, grown by CAP, reaches the slab's corner planes.
    const start = try gpa.alloc(u32, nslabs + 1);
    defer gpa.free(start);
    var counts = try gpa.alloc(u32, nslabs);
    defer gpa.free(counts);
    @memset(counts, 0);
    for (0..nt) |t| {
        const r = slabRange(cells[t * 9 ..][0..9], nslabs);
        for (r[0]..r[1] + 1) |s| counts[s] += 1;
    }
    var total: u32 = 0;
    for (0..nslabs) |s| {
        start[s] = total;
        total += counts[s];
    }
    start[nslabs] = total;
    const slab_tris = try gpa.alloc(u32, total);
    defer gpa.free(slab_tris);
    @memset(counts, 0);
    for (0..nt) |t| {
        const r = slabRange(cells[t * 9 ..][0..9], nslabs);
        for (r[0]..r[1] + 1) |s| {
            slab_tris[start[s] + counts[s]] = @intCast(t);
            counts[s] += 1;
        }
    }

    const results = try gpa.alloc(SlabOut, nslabs);
    defer gpa.free(results);
    for (results) |*r| r.* = .{ .arena = std.heap.ArenaAllocator.init(std.heap.page_allocator) };
    defer for (results) |*r| r.arena.deinit();

    var ctx = Ctx{ .n = n, .origin = origin, .h = h, .cells = cells, .slab_start = start, .slab_tris = slab_tris, .results = results };
    const workers = @min(@min(MAX_THREADS, std.Thread.getCpuCount() catch 1), nslabs);
    var threads: [MAX_THREADS]?std.Thread = @splat(null);
    for (1..workers) |k| threads[k] = std.Thread.spawn(.{}, worker, .{&ctx}) catch null;
    worker(&ctx);
    for (threads[1..workers]) |t| if (t) |th| th.join();
    if (ctx.failed.load(.acquire)) return error.OutOfMemory;

    return merge(gpa, results);
}

/// Slabs `[lo, hi]` a triangle (nine cell-space coordinates) can influence.
fn slabRange(tri: *const [9]f32, nslabs: usize) [2]usize {
    var lo = tri[0];
    var hi = tri[0];
    for (0..3) |k| {
        lo = @min(lo, tri[k * 3]);
        hi = @max(hi, tri[k * 3]);
    }
    const s_lo: i64 = @max(0, @as(i64, @intFromFloat(@ceil((lo - CAP) / @as(f32, SLAB)))) - 1);
    const s_hi: i64 = @min(@as(i64, @intCast(nslabs)) - 1, @as(i64, @intFromFloat(@floor((hi + CAP) / @as(f32, SLAB)))));
    if (s_hi < s_lo) return .{ 0, 0 };
    return .{ @intCast(s_lo), @intCast(s_hi) };
}

fn worker(ctx: *Ctx) void {
    const side = ctx.n + 1;
    const buf = std.heap.page_allocator.alloc(f32, (SLAB + 1) * side * side) catch {
        ctx.failed.store(true, .release);
        return;
    };
    defer std.heap.page_allocator.free(buf);
    while (true) {
        const s = ctx.next.fetchAdd(1, .monotonic);
        if (s >= ctx.results.len or ctx.failed.load(.acquire)) return;
        runSlab(ctx, s, buf) catch {
            ctx.failed.store(true, .release);
            return;
        };
    }
}

fn runSlab(ctx: *Ctx, s: usize, d2: []f32) !void {
    const n = ctx.n;
    const side = n + 1;
    const xa = s * SLAB;
    const xb = @min(n, (s + 1) * SLAB);
    const planes = xb - xa + 1;
    const field = d2[0 .. planes * side * side];
    @memset(field, CAP * CAP);

    const lo = ctx.slab_start[s];
    const hi = ctx.slab_start[s + 1];
    for (ctx.slab_tris[lo..hi]) |t| {
        const c = ctx.cells[@as(usize, t) * 9 ..][0..9];
        const a = [3]f32{ c[0], c[1], c[2] };
        const b = [3]f32{ c[3], c[4], c[5] };
        const cc = [3]f32{ c[6], c[7], c[8] };
        var r_lo: [3]usize = undefined;
        var r_hi: [3]usize = undefined;
        for (0..3) |ax| {
            const mn = @min(a[ax], @min(b[ax], cc[ax])) - CAP;
            const mx = @max(a[ax], @max(b[ax], cc[ax])) + CAP;
            const lim_lo: f32 = if (ax == 0) @floatFromInt(xa) else 0;
            const lim_hi: f32 = if (ax == 0) @floatFromInt(xb) else @floatFromInt(n);
            const l = @max(lim_lo, @ceil(mn));
            const u = @min(lim_hi, @floor(mx));
            if (u < l) {
                r_lo[ax] = 1;
                r_hi[ax] = 0;
            } else {
                r_lo[ax] = @intFromFloat(l);
                r_hi[ax] = @intFromFloat(u);
            }
        }
        if (r_hi[0] < r_lo[0] or r_hi[1] < r_lo[1] or r_hi[2] < r_lo[2]) continue;
        for (r_lo[0]..r_hi[0] + 1) |i| for (r_lo[1]..r_hi[1] + 1) |j| {
            const row = ((i - xa) * side + j) * side;
            for (r_lo[2]..r_hi[2] + 1) |k| {
                const d = pointTriDist2(.{ @floatFromInt(i), @floatFromInt(j), @floatFromInt(k) }, a, b, cc);
                if (d < field[row + k]) field[row + k] = d;
            }
        };
    }

    const out = &ctx.results[s];
    const al = out.arena.allocator();
    var keys: std.ArrayList(u64) = .empty;
    var pos: std.ArrayList(f32) = .empty;
    var dirs: std.ArrayList(i8) = .empty;
    var tris: std.ArrayList(u32) = .empty;
    var local = std.AutoHashMap(u64, u32).init(al);
    const level2 = OFFSET * OFFSET;
    for (0..planes - 1) |pi| for (0..n) |j| for (0..n) |k| {
        var v2: [8]f32 = undefined;
        var cube: u8 = 0;
        for (0..8) |c| {
            const o = mc.CORNER[c];
            v2[c] = field[((pi + o[0]) * side + j + o[1]) * side + k + o[2]];
            if (v2[c] < level2) cube |= @as(u8, 1) << @intCast(c);
        }
        const edges = mc.EDGE_TABLE[cube];
        if (edges == 0) continue;
        var vidx: [12]u32 = undefined;
        for (0..12) |e| {
            if (edges & (@as(u12, 1) << @intCast(e)) == 0) continue;
            const eo = mc.EDGE_ORIGIN[e];
            const oi = xa + pi + eo[0];
            const oj = j + eo[1];
            const ok = k + eo[2];
            const key = (@as(u64, (oi * side + oj) * side + ok)) * 3 + mc.EDGE_AXIS[e];
            const g = try local.getOrPut(key);
            if (g.found_existing) {
                vidx[e] = g.value_ptr.*;
                continue;
            }
            const ca = mc.EDGE_CORNERS[e][0];
            const cb = mc.EDGE_CORNERS[e][1];
            const da = @sqrt(v2[ca]);
            const db = @sqrt(v2[cb]);
            const t: f32 = if (@abs(db - da) < 1e-12) 0.5 else (OFFSET - da) / (db - da);
            const pa = mc.CORNER[ca];
            const pb = mc.CORNER[cb];
            const base = [3]usize{ xa + pi, j, k };
            const vid: u32 = @intCast(keys.items.len);
            for (0..3) |ax| {
                const x0: f32 = @floatFromInt(base[ax] + pa[ax]);
                const x1: f32 = @floatFromInt(base[ax] + pb[ax]);
                try pos.append(al, ctx.origin + (x0 + t * (x1 - x0)) * ctx.h);
            }
            const ax: usize = @intCast(mc.EDGE_AXIS[e]);
            const along: i8 = if (pb[ax] > pa[ax]) 1 else -1;
            try dirs.append(al, @as(i8, @intCast(ax + 1)) * (if (db > da) along else -along));
            try keys.append(al, key);
            g.value_ptr.* = vid;
            vidx[e] = vid;
        }
        const row = mc.TRI_TABLE[cube];
        var q: usize = 0;
        while (q < 16 and row[q] >= 0) : (q += 3) {
            const a = vidx[@intCast(row[q])];
            const b = vidx[@intCast(row[q + 1])];
            const c = vidx[@intCast(row[q + 2])];
            if (a == b or b == c or a == c) continue;
            try tris.appendSlice(al, &.{ a, b, c });
        }
    };
    out.keys = keys.items;
    out.pos = pos.items;
    out.out_dir = dirs.items;
    out.tris = tris.items;
}

/// Welds the slabs' vertices by edge key, then winds every face so its normal agrees with the direction in which
/// the field grows (one vote: the tables wind all cells alike).
fn merge(gpa: std.mem.Allocator, slabs: []const SlabOut) !TriMesh {
    var total_v: usize = 0;
    var total_t: usize = 0;
    for (slabs) |s| {
        total_v += s.keys.len;
        total_t += s.tris.len;
    }
    var global = std.AutoHashMap(u64, u32).init(gpa);
    defer global.deinit();
    try global.ensureTotalCapacity(@intCast(total_v));
    var verts: std.ArrayList(f32) = .empty;
    errdefer verts.deinit(gpa);
    var dirs: std.ArrayList(i8) = .empty;
    defer dirs.deinit(gpa);
    var tris: std.ArrayList(u32) = .empty;
    errdefer tris.deinit(gpa);
    try tris.ensureTotalCapacity(gpa, total_t);
    var remap: std.ArrayList(u32) = .empty;
    defer remap.deinit(gpa);
    for (slabs) |s| {
        remap.clearRetainingCapacity();
        for (s.keys, 0..) |k, i| {
            const g = global.getOrPutAssumeCapacity(k);
            if (!g.found_existing) {
                g.value_ptr.* = @intCast(verts.items.len / 3);
                try verts.appendSlice(gpa, s.pos[i * 3 ..][0..3]);
                try dirs.append(gpa, s.out_dir[i]);
            }
            try remap.append(gpa, g.value_ptr.*);
        }
        for (s.tris) |t| tris.appendAssumeCapacity(remap.items[t]);
    }

    var agree: f64 = 0;
    var f: usize = 0;
    while (f < tris.items.len) : (f += 3) {
        const a = vec3(verts.items, tris.items[f]);
        const b = vec3(verts.items, tris.items[f + 1]);
        const c = vec3(verts.items, tris.items[f + 2]);
        const e1 = sub3(b, a);
        const e2 = sub3(c, a);
        const nrm = [3]f32{ e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
        var want = [3]f32{ 0, 0, 0 };
        for (0..3) |k| {
            const d = dirs.items[tris.items[f + k]];
            const ax: usize = @intCast(@abs(d) - 1);
            want[ax] += if (d > 0) 1.0 else -1.0;
        }
        agree += @as(f64, dot3(nrm, want));
    }
    if (agree < 0) {
        f = 0;
        while (f < tris.items.len) : (f += 3) std.mem.swap(u32, &tris.items[f + 1], &tris.items[f + 2]);
    }
    return .{ .verts = try verts.toOwnedSlice(gpa), .tris = try tris.toOwnedSlice(gpa) };
}

fn vec3(v: []const f32, i: u32) [3]f32 {
    return .{ v[i * 3], v[i * 3 + 1], v[i * 3 + 2] };
}

// ── Snapping to the original surface ─────────────────────────────────────────

/// Cell size of the triangle index `snapToSurface` builds: a few voxels, so a point within a voxel or two of
/// the soup finds its nearest triangle among the 27 cells around it.
const SNAP_CELLS: usize = 256;
const SNAP_PAD: usize = 2;
const SNAP_MAX_RING: usize = 4;

const SnapCtx = struct {
    g: usize, // cells per axis
    origin: f32,
    cell: f32,
    start: []const u32,
    tris: []const u32,
    soup: TriMesh,
    points: []f32,
    mask: []const u8,
};

/// Moves every masked point (`points` `[n, 3]`, in place) to the nearest point of the triangle soup `soup`: the
/// reference's BVH projection of each texel onto the mesh the volume was decoded for. Unmasked points and points
/// with no triangle nearby are left alone.
pub fn snapToSurface(gpa: std.mem.Allocator, soup: TriMesh, points: []f32, mask: []const u8) !void {
    const g = SNAP_CELLS + 2 * SNAP_PAD;
    const cell: f32 = 1.0 / @as(f32, SNAP_CELLS);
    const origin: f32 = -0.5 - @as(f32, SNAP_PAD) * cell;
    const nt = soup.tris.len / 3;
    const start = try gpa.alloc(u32, g * g * g + 1);
    defer gpa.free(start);
    @memset(start, 0);
    // Counting sort of triangles into the cells their boxes touch.
    const counts = try gpa.alloc(u32, g * g * g);
    defer gpa.free(counts);
    @memset(counts, 0);
    for (0..nt) |t| {
        const r = triCells(soup, t, g, origin, cell);
        for (r.lo[0]..r.hi[0] + 1) |x| for (r.lo[1]..r.hi[1] + 1) |y| for (r.lo[2]..r.hi[2] + 1) |z| {
            counts[(x * g + y) * g + z] += 1;
        };
    }
    var acc: u32 = 0;
    for (counts, 0..) |c, i| {
        start[i] = acc;
        acc += c;
    }
    start[g * g * g] = acc;
    const tris = try gpa.alloc(u32, acc);
    defer gpa.free(tris);
    @memset(counts, 0); // reused as each cell's fill count
    for (0..nt) |t| {
        const r = triCells(soup, t, g, origin, cell);
        for (r.lo[0]..r.hi[0] + 1) |x| for (r.lo[1]..r.hi[1] + 1) |y| for (r.lo[2]..r.hi[2] + 1) |z| {
            const ci = (x * g + y) * g + z;
            tris[start[ci] + counts[ci]] = @intCast(t);
            counts[ci] += 1;
        };
    }
    const ctx = SnapCtx{ .g = g, .origin = origin, .cell = cell, .start = start, .tris = tris, .soup = soup, .points = points, .mask = mask };
    sparse.parallelRange(SnapCtx, ctx, mask.len, snapRange);
}

const CellBox = struct { lo: [3]usize, hi: [3]usize };

fn cellOf(v: f32, g: usize, origin: f32, cell: f32) usize {
    const c = @floor((v - origin) / cell);
    return @intFromFloat(std.math.clamp(c, 0, @as(f32, @floatFromInt(g - 1))));
}

fn triCells(soup: TriMesh, t: usize, g: usize, origin: f32, cell: f32) CellBox {
    var lo: [3]usize = undefined;
    var hi: [3]usize = undefined;
    for (0..3) |ax| {
        const a = soup.verts[soup.tris[t * 3] * 3 + ax];
        const b = soup.verts[soup.tris[t * 3 + 1] * 3 + ax];
        const c = soup.verts[soup.tris[t * 3 + 2] * 3 + ax];
        lo[ax] = cellOf(@min(a, @min(b, c)), g, origin, cell);
        hi[ax] = cellOf(@max(a, @max(b, c)), g, origin, cell);
    }
    return .{ .lo = lo, .hi = hi };
}

fn snapRange(c: SnapCtx, lo: usize, hi: usize) void {
    for (lo..hi) |i| {
        if (c.mask[i] == 0) continue;
        const p = vec3(c.points, @intCast(i));
        const cx = cellOf(p[0], c.g, c.origin, c.cell);
        const cy = cellOf(p[1], c.g, c.origin, c.cell);
        const cz = cellOf(p[2], c.g, c.origin, c.cell);
        var best: f32 = std.math.inf(f32);
        var at = p;
        var ring: usize = 1;
        while (ring <= SNAP_MAX_RING) : (ring += 1) {
            const x0 = cx -| ring;
            const y0 = cy -| ring;
            const z0 = cz -| ring;
            for (x0..@min(c.g, cx + ring + 1)) |x| for (y0..@min(c.g, cy + ring + 1)) |y| for (z0..@min(c.g, cz + ring + 1)) |z| {
                const ci = (x * c.g + y) * c.g + z;
                for (c.tris[c.start[ci]..c.start[ci + 1]]) |t| {
                    const q = closestPoint(p, vec3(c.soup.verts, c.soup.tris[t * 3]), vec3(c.soup.verts, c.soup.tris[t * 3 + 1]), vec3(c.soup.verts, c.soup.tris[t * 3 + 2]));
                    const d = dist2(p, q);
                    if (d < best) {
                        best = d;
                        at = q;
                    }
                }
            };
            // Within `ring` cells of the point's own, every nearer triangle has been seen.
            const reach = @as(f32, @floatFromInt(ring)) * c.cell;
            if (best <= reach * reach) break;
        }
        @memcpy(c.points[i * 3 ..][0..3], &at);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "trellis2 remesh: the point-triangle distance covers the face, edge and vertex regions" {
    const a = [3]f32{ 0, 0, 0 };
    const b = [3]f32{ 2, 0, 0 };
    const c = [3]f32{ 0, 2, 0 };
    try testing.expectApproxEqAbs(@as(f32, 1), pointTriDist2(.{ 0.5, 0.5, 1 }, a, b, c), 1e-6); // over the face
    try testing.expectApproxEqAbs(@as(f32, 0), pointTriDist2(.{ 1, 1, 0 }, a, b, c), 1e-6); // on the hypotenuse
    try testing.expectApproxEqAbs(@as(f32, 0.5), pointTriDist2(.{ 2, 1, 0 }, a, b, c), 1e-6); // beside the hypotenuse
    try testing.expectApproxEqAbs(@as(f32, 2), pointTriDist2(.{ -1, -1, 0 }, a, b, c), 1e-6); // nearest vertex a
    try testing.expectApproxEqAbs(@as(f32, 1), pointTriDist2(.{ 1, -1, 0 }, a, b, c), 1e-6); // nearest edge ab
    try testing.expectApproxEqAbs(@as(f32, 0.75), pointTriDist2(.{ 0.5, 0.5, 0.5 }, .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 }), 1e-6); // no area: |p|^2
}

fn expectClosedAndConsistent(gpa: std.mem.Allocator, tris: []const u32) !void {
    var dir = std.AutoHashMap(u64, u32).init(gpa);
    defer dir.deinit();
    var f: usize = 0;
    while (f < tris.len) : (f += 3) {
        for (0..3) |k| {
            const key = (@as(u64, tris[f + k]) << 32) | tris[f + (k + 1) % 3];
            const e = try dir.getOrPut(key);
            e.value_ptr.* = if (e.found_existing) e.value_ptr.* + 1 else 1;
        }
    }
    var it = dir.iterator();
    while (it.next()) |e| {
        try testing.expectEqual(@as(u32, 1), e.value_ptr.*);
        const rev = (e.key_ptr.* << 32) | (e.key_ptr.* >> 32);
        try testing.expect(dir.contains(rev));
    }
}

test "trellis2 remesh: a flat sheet becomes a closed tube whose skins face away from it" {
    const gpa = testing.allocator;
    const sheet = TriMesh{
        .verts = @constCast(&[_]f32{ -0.2, -0.2, 0, 0.2, -0.2, 0, 0.2, 0.2, 0, -0.2, 0.2, 0 }),
        .tris = @constCast(&[_]u32{ 0, 1, 2, 0, 2, 3 }),
    };
    const res: u32 = 64;
    var out = try remesh(gpa, sheet, res);
    defer out.deinit(gpa);
    try testing.expect(out.tris.len > 0);
    try expectClosedAndConsistent(gpa, out.tris);
    const h = 1.0 / @as(f32, @floatFromInt(res));
    var up: usize = 0;
    var down: usize = 0;
    for (0..out.verts.len / 3) |i| {
        const x = out.verts[i * 3];
        const y = out.verts[i * 3 + 1];
        const z = out.verts[i * 3 + 2];
        if (@abs(x) < 0.15 and @abs(y) < 0.15) { // away from the rim, a skin sits OFFSET cells off the plane
            try testing.expectApproxEqAbs(OFFSET * h, @abs(z), 0.05 * h);
            if (z > 0) up += 1 else down += 1;
        }
    }
    try testing.expect(up > 100 and down > 100);
    // Outward = away from the sheet: the upper skin faces +z, the lower skin -z.
    var f: usize = 0;
    while (f < out.tris.len) : (f += 3) {
        const a = vec3(out.verts, out.tris[f]);
        const b = vec3(out.verts, out.tris[f + 1]);
        const c = vec3(out.verts, out.tris[f + 2]);
        const e1 = sub3(b, a);
        const e2 = sub3(c, a);
        const nz = e1[0] * e2[1] - e1[1] * e2[0];
        const cz = (a[2] + b[2] + c[2]) / 3;
        if (@abs(cz) > 0.5 * h and @abs((a[0] + b[0] + c[0]) / 3) < 0.15 and @abs((a[1] + b[1] + c[1]) / 3) < 0.15)
            try testing.expect(nz * cz > 0);
    }
}

/// A unit icosphere of the given radius, `levels` subdivisions.
fn icosphere(gpa: std.mem.Allocator, r: f32, levels: u32) !TriMesh {
    const t = (1.0 + @sqrt(5.0)) / 2.0;
    var verts: std.ArrayList(f32) = .empty;
    errdefer verts.deinit(gpa);
    const base = [12][3]f32{ .{ -1, t, 0 }, .{ 1, t, 0 }, .{ -1, -t, 0 }, .{ 1, -t, 0 }, .{ 0, -1, t }, .{ 0, 1, t }, .{ 0, -1, -t }, .{ 0, 1, -t }, .{ t, 0, -1 }, .{ t, 0, 1 }, .{ -t, 0, -1 }, .{ -t, 0, 1 } };
    for (base) |v| {
        const l = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        try verts.appendSlice(gpa, &.{ v[0] / l * r, v[1] / l * r, v[2] / l * r });
    }
    var tris: std.ArrayList(u32) = .empty;
    errdefer tris.deinit(gpa);
    try tris.appendSlice(gpa, &.{ 0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1 });
    for (0..levels) |_| {
        var next: std.ArrayList(u32) = .empty;
        errdefer next.deinit(gpa);
        var mid = std.AutoHashMap(u64, u32).init(gpa);
        defer mid.deinit();
        var f: usize = 0;
        while (f < tris.items.len) : (f += 3) {
            var m3: [3]u32 = undefined;
            for (0..3) |k| {
                const a = tris.items[f + k];
                const b = tris.items[f + (k + 1) % 3];
                const key = (@as(u64, @min(a, b)) << 32) | @max(a, b);
                const g = try mid.getOrPut(key);
                if (!g.found_existing) {
                    var p: [3]f32 = undefined;
                    for (0..3) |ax| p[ax] = (verts.items[a * 3 + ax] + verts.items[b * 3 + ax]) / 2;
                    const l = @sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
                    g.value_ptr.* = @intCast(verts.items.len / 3);
                    try verts.appendSlice(gpa, &.{ p[0] / l * r, p[1] / l * r, p[2] / l * r });
                }
                m3[k] = g.value_ptr.*;
            }
            const t0 = tris.items[f];
            const t1 = tris.items[f + 1];
            const t2 = tris.items[f + 2];
            try next.appendSlice(gpa, &.{ t0, m3[0], m3[2], t1, m3[1], m3[0], t2, m3[2], m3[1], m3[0], m3[1], m3[2] });
        }
        tris.deinit(gpa);
        tris = next;
    }
    return .{ .verts = try verts.toOwnedSlice(gpa), .tris = try tris.toOwnedSlice(gpa) };
}

test "trellis2 remesh: a spherical shell gets an outer skin facing out and an inner skin facing the hollow" {
    const gpa = testing.allocator;
    var sphere = try icosphere(gpa, 0.3, 3);
    defer sphere.deinit(gpa);
    const res: u32 = 64;
    var out = try remesh(gpa, sphere, res);
    defer out.deinit(gpa);
    try expectClosedAndConsistent(gpa, out.tris);
    const h = 1.0 / @as(f32, @floatFromInt(res));
    var outer: usize = 0;
    var inner: usize = 0;
    var f: usize = 0;
    while (f < out.tris.len) : (f += 3) {
        const a = vec3(out.verts, out.tris[f]);
        const b = vec3(out.verts, out.tris[f + 1]);
        const c = vec3(out.verts, out.tris[f + 2]);
        const e1 = sub3(b, a);
        const e2 = sub3(c, a);
        const nrm = [3]f32{ e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
        const cen = [3]f32{ (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3 };
        const r = @sqrt(dot3(cen, cen));
        const radial = dot3(nrm, cen);
        if (r > 0.3) {
            try testing.expectApproxEqAbs(0.3 + OFFSET * h, r, 0.5 * h);
            try testing.expect(radial > 0);
            outer += 1;
        } else {
            try testing.expectApproxEqAbs(0.3 - OFFSET * h, r, 0.5 * h);
            try testing.expect(radial < 0);
            inner += 1;
        }
    }
    try testing.expect(outer > 1000 and inner > 1000);
}

test "trellis2 remesh: three sheets sharing one edge (non-manifold) come out as a closed manifold" {
    const gpa = testing.allocator;
    // A book: the shared edge runs along y; the pages fan out in x/z.
    const verts = [_]f32{ 0, -0.2, 0, 0, 0.2, 0, 0.2, -0.2, 0, 0.2, 0.2, 0, -0.1, -0.2, 0.17, -0.1, 0.2, 0.17, -0.1, -0.2, -0.17, -0.1, 0.2, -0.17 };
    const tris = [_]u32{ 0, 1, 2, 1, 3, 2, 0, 5, 1, 0, 4, 5, 0, 7, 6, 0, 1, 7 };
    var out = try remesh(gpa, .{ .verts = @constCast(&verts), .tris = @constCast(&tris) }, 64);
    defer out.deinit(gpa);
    try testing.expect(out.tris.len > 0);
    try expectClosedAndConsistent(gpa, out.tris);
}

test "trellis2 remesh: the result does not depend on how many slabs or threads cut the grid" {
    const gpa = testing.allocator;
    var sphere = try icosphere(gpa, 0.25, 2);
    defer sphere.deinit(gpa);
    // 40 cells + padding spans three slabs: a vertex on a slab boundary must be shared, never duplicated.
    var out = try remesh(gpa, sphere, 40);
    defer out.deinit(gpa);
    try expectClosedAndConsistent(gpa, out.tris);
    var seen = std.AutoHashMap([3]u32, void).init(gpa);
    defer seen.deinit();
    for (0..out.verts.len / 3) |i| {
        const key = [3]u32{ @bitCast(out.verts[i * 3]), @bitCast(out.verts[i * 3 + 1]), @bitCast(out.verts[i * 3 + 2]) };
        const g = try seen.getOrPut(key);
        try testing.expect(!g.found_existing);
    }
}

test "trellis2 remesh: snapToSurface moves masked points onto the nearest triangle and leaves the rest" {
    const gpa = testing.allocator;
    const sheet = TriMesh{
        .verts = @constCast(&[_]f32{ -0.2, -0.2, 0, 0.2, -0.2, 0, 0.2, 0.2, 0, -0.2, 0.2, 0 }),
        .tris = @constCast(&[_]u32{ 0, 1, 2, 0, 2, 3 }),
    };
    var pts = [_]f32{ 0.05, 0.1, 0.004, -0.1, -0.05, -0.003, 0.205, 0.0, 0.002, 0.0, 0.0, 0.5 };
    const mask = [_]u8{ 1, 1, 1, 0 };
    try snapToSurface(gpa, sheet, &pts, &mask);
    for ([_]f32{ 0.05, 0.1, 0, -0.1, -0.05, 0, 0.2, 0, 0 }, pts[0..9]) |want, got| try testing.expectApproxEqAbs(want, got, 1e-6); // onto the faces, then the sheet's edge
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0.5 }, pts[9..12]); // unmasked: untouched
}
