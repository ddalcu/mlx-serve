//! TRELLIS.2 geometry on the host (pure CPU, no MLX): the O-Voxel dual grid to a triangle mesh,
//! cleanup (floaters, small holes, face orientation), and the bake of the decoded PBR voxel volume
//! into UV-space textures. Mirrors `o_voxel.convert.flexible_dual_grid_to_mesh`, CuMesh's
//! `fill_holes` / `unify_face_orientations` and `o_voxel.postprocess.to_glb` (microsoft/TRELLIS.2, MIT).

const std = @import("std");
const sparse = @import("sparse.zig");
const mc = @import("../../sdk/mesh/marching_cubes.zig");
const rasterize = @import("../../sdk/mesh/rasterize.zig");
const mesh_simplify = @import("../../sdk/mesh/mesh_simplify.zig");

/// World box the volume spans (`aabb`): `[-0.5, 0.5]^3`.
pub const AABB_MIN: f32 = -0.5;
/// Dual vertices may leave their voxel by this margin (`voxel_margin`).
const VOXEL_MARGIN: f32 = 0.5;
/// Channels of the decoder head: 3 vertex offsets, 3 edge flags, 1 split weight.
pub const HEAD_CHANNELS: usize = 7;

pub const TriMesh = struct {
    verts: []f32, // xyz
    tris: []u32,

    pub fn deinit(self: *TriMesh, gpa: std.mem.Allocator) void {
        gpa.free(self.verts);
        gpa.free(self.tris);
        self.* = undefined;
    }
};

// ── Dual grid ────────────────────────────────────────────────────────────────

/// The four voxels around a +axis edge of voxel `i`, in quad order.
const EDGE_VOXELS = [3][4][3]i32{
    .{ .{ 0, 0, 0 }, .{ 0, 0, 1 }, .{ 0, 1, 1 }, .{ 0, 1, 0 } },
    .{ .{ 0, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 1 }, .{ 0, 0, 1 } },
    .{ .{ 0, 0, 0 }, .{ 0, 1, 0 }, .{ 1, 1, 0 }, .{ 1, 0, 0 } },
};

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn softplus(x: f32) f32 {
    return if (x > 20) x else std.math.log1p(@exp(x));
}

/// One vertex per active voxel (`coords` `[M, 4]`, decoder head `raw` `[M, 7]`: offset logits, edge flags,
/// split weight) and a quad, split into two triangles, wherever an intersected edge has all four
/// surrounding voxels. Vertex i is voxel i; quads come in voxel order, then axis x, y, z.
pub fn dualGridMesh(gpa: std.mem.Allocator, coords: []const i32, raw: []const f32, res: u32) !TriMesh {
    const m = coords.len / 4;
    std.debug.assert(raw.len == m * HEAD_CHANNELS);
    const voxel: f32 = 1.0 / @as(f32, @floatFromInt(res));
    const verts = try gpa.alloc(f32, m * 3);
    errdefer gpa.free(verts);
    const weight = try gpa.alloc(f32, m);
    defer gpa.free(weight);
    for (0..m) |i| {
        for (0..3) |k| {
            const dual = (1 + 2 * VOXEL_MARGIN) * sigmoid(raw[i * HEAD_CHANNELS + k]) - VOXEL_MARGIN;
            verts[i * 3 + k] = (@as(f32, @floatFromInt(coords[i * 4 + 1 + k])) + dual) * voxel + AABB_MIN;
        }
        weight[i] = softplus(raw[i * HEAD_CHANNELS + 6]);
    }
    const table = try sparse.Table.init(gpa, coords);
    defer table.deinit(gpa);
    var tris: std.ArrayList(u32) = .empty;
    errdefer tris.deinit(gpa);
    for (0..m) |i| {
        const c = coords[i * 4 ..][0..4];
        for (0..3) |axis| {
            if (raw[i * HEAD_CHANNELS + 3 + axis] <= 0) continue;
            var q: [4]u32 = undefined;
            const found = for (EDGE_VOXELS[axis], 0..) |o, k| {
                const x = c[1] + o[0];
                const y = c[2] + o[1];
                const z = c[3] + o[2];
                q[k] = (if (x < 0 or y < 0 or z < 0) null else table.find(sparse.voxelKey(x, y, z))) orelse break false;
            } else true;
            if (!found) continue;
            const w02 = weight[q[0]] * weight[q[2]];
            const w13 = weight[q[1]] * weight[q[3]];
            const t = if (w02 > w13) [6]u32{ q[0], q[1], q[2], q[0], q[2], q[3] } else [6]u32{ q[0], q[1], q[3], q[3], q[1], q[2] };
            try tris.appendSlice(gpa, &t);
        }
    }
    return .{ .verts = verts, .tris = try tris.toOwnedSlice(gpa) };
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

/// Drops vertices no triangle uses and renumbers (new mesh; `m` is left to the caller).
pub fn compact(gpa: std.mem.Allocator, m: TriMesh) !TriMesh {
    const nv = m.verts.len / 3;
    const remap = try gpa.alloc(u32, nv);
    defer gpa.free(remap);
    @memset(remap, std.math.maxInt(u32));
    var used: u32 = 0;
    for (m.tris) |t| {
        if (remap[t] == std.math.maxInt(u32)) {
            remap[t] = used;
            used += 1;
        }
    }
    const verts = try gpa.alloc(f32, used * 3);
    errdefer gpa.free(verts);
    for (0..nv) |i| {
        if (remap[i] != std.math.maxInt(u32)) @memcpy(verts[remap[i] * 3 ..][0..3], m.verts[i * 3 ..][0..3]);
    }
    const tris = try gpa.alloc(u32, m.tris.len);
    for (tris, m.tris) |*o, t| o.* = remap[t];
    return .{ .verts = verts, .tris = tris };
}

fn find(parent: []u32, x: u32) u32 {
    var r = x;
    while (parent[r] != r) r = parent[r];
    var c = x;
    while (parent[c] != r) {
        const next = parent[c];
        parent[c] = r;
        c = next;
    }
    return r;
}

/// Removes connected pieces (by shared vertices) whose total surface area is below `min_area`; returns the
/// new triangle list (caller frees).
pub fn dropSmallComponents(gpa: std.mem.Allocator, verts: []const f32, tris: []const u32, min_area: f64) ![]u32 {
    const nverts = verts.len / 3;
    const parent = try gpa.alloc(u32, nverts);
    defer gpa.free(parent);
    for (parent, 0..) |*p, i| p.* = @intCast(i);
    var f: usize = 0;
    while (f < tris.len) : (f += 3) {
        const a = find(parent, tris[f]);
        for (1..3) |k| {
            const b = find(parent, tris[f + k]);
            if (a != b) parent[b] = a;
        }
    }
    const area = try gpa.alloc(f64, nverts);
    defer gpa.free(area);
    @memset(area, 0);
    f = 0;
    while (f < tris.len) : (f += 3) area[find(parent, tris[f])] += triArea(verts, tris[f..][0..3]);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    f = 0;
    while (f < tris.len) : (f += 3) {
        if (area[find(parent, tris[f])] >= min_area) try out.appendSlice(gpa, tris[f..][0..3]);
    }
    return out.toOwnedSlice(gpa);
}

fn triArea(verts: []const f32, t: []const u32) f64 {
    const a = verts[t[0] * 3 ..][0..3];
    const b = verts[t[1] * 3 ..][0..3];
    const c = verts[t[2] * 3 ..][0..3];
    const e1 = [3]f64{ b[0] - a[0], b[1] - a[1], b[2] - a[2] };
    const e2 = [3]f64{ c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    const x = e1[1] * e2[2] - e1[2] * e2[1];
    const y = e1[2] * e2[0] - e1[0] * e2[2];
    const z = e1[0] * e2[1] - e1[1] * e2[0];
    return 0.5 * @sqrt(x * x + y * y + z * z);
}

fn edgeKey(a: u32, b: u32) u64 {
    return (@as(u64, @min(a, b)) << 32) | @max(a, b);
}

const BoundaryEdge = struct { a: u32, b: u32 };

/// Fills every boundary loop whose perimeter is at most `max_perimeter` with a fan around its centroid,
/// wound to continue the surface it closes (CuMesh `fill_holes`). Returns a new mesh and the loops filled.
pub fn fillHoles(gpa: std.mem.Allocator, m: TriMesh, max_perimeter: f32) !struct { mesh: TriMesh, filled: usize } {
    const Rec = struct { count: u32, a: u32, b: u32 };
    var edges = std.AutoHashMap(u64, Rec).init(gpa);
    defer edges.deinit();
    var f: usize = 0;
    while (f < m.tris.len) : (f += 3) {
        for (0..3) |k| {
            const a = m.tris[f + k];
            const b = m.tris[f + (k + 1) % 3];
            const e = try edges.getOrPut(edgeKey(a, b));
            if (e.found_existing) e.value_ptr.count += 1 else e.value_ptr.* = .{ .count = 1, .a = a, .b = b };
        }
    }
    var boundary: std.ArrayList(BoundaryEdge) = .empty;
    defer boundary.deinit(gpa);
    var it = edges.valueIterator();
    while (it.next()) |r| {
        if (r.count == 1) try boundary.append(gpa, .{ .a = r.a, .b = r.b });
    }
    // Deterministic order regardless of hash iteration.
    std.mem.sortUnstable(BoundaryEdge, boundary.items, {}, struct {
        fn lt(_: void, x: BoundaryEdge, y: BoundaryEdge) bool {
            return if (x.a != y.a) x.a < y.a else x.b < y.b;
        }
    }.lt);

    // vertex -> its (at most two) boundary edges; a vertex with more is a pinch, and its loops are left alone.
    const Inc = struct { e: [2]u32 = .{ 0, 0 }, n: u32 = 0 };
    var inc = std.AutoHashMap(u32, Inc).init(gpa);
    defer inc.deinit();
    for (boundary.items, 0..) |e, i| {
        for ([_]u32{ e.a, e.b }) |v| {
            const g = try inc.getOrPut(v);
            if (!g.found_existing) g.value_ptr.* = .{};
            if (g.value_ptr.n < 2) g.value_ptr.e[g.value_ptr.n] = @intCast(i);
            g.value_ptr.n += 1;
        }
    }
    const used = try gpa.alloc(bool, boundary.items.len);
    defer gpa.free(used);
    @memset(used, false);

    var verts: std.ArrayList(f32) = .empty;
    errdefer verts.deinit(gpa);
    try verts.appendSlice(gpa, m.verts);
    var tris: std.ArrayList(u32) = .empty;
    errdefer tris.deinit(gpa);
    try tris.appendSlice(gpa, m.tris);
    var filled: usize = 0;
    var loop: std.ArrayList(u32) = .empty; // boundary edge indices around the loop
    defer loop.deinit(gpa);
    for (boundary.items, 0..) |start, si| {
        if (used[si]) continue;
        loop.clearRetainingCapacity();
        var ok = true;
        var cur: u32 = @intCast(si);
        var at: u32 = start.b; // walk away from `start.a`
        var perim: f32 = 0;
        while (true) {
            used[cur] = true;
            try loop.append(gpa, cur);
            const e = boundary.items[cur];
            perim += dist(m.verts, e.a, e.b);
            const g = inc.get(at).?;
            if (g.n != 2) {
                ok = false;
                break;
            }
            const next = if (g.e[0] == cur) g.e[1] else g.e[0];
            const e2 = boundary.items[next];
            at = if (e2.a == at) e2.b else e2.a;
            if (next == si) break;
            if (used[next]) {
                ok = false;
                break;
            }
            cur = next;
        }
        if (!ok or loop.items.len < 3 or perim > max_perimeter) continue;
        var cx: f32 = 0;
        var cy: f32 = 0;
        var cz: f32 = 0;
        for (loop.items) |ei| {
            const e = boundary.items[ei];
            cx += m.verts[e.a * 3];
            cy += m.verts[e.a * 3 + 1];
            cz += m.verts[e.a * 3 + 2];
        }
        const n: f32 = @floatFromInt(loop.items.len);
        const c: u32 = @intCast(verts.items.len / 3);
        try verts.appendSlice(gpa, &.{ cx / n, cy / n, cz / n });
        for (loop.items) |ei| {
            const e = boundary.items[ei];
            try tris.appendSlice(gpa, &.{ e.b, e.a, c });
        }
        filled += 1;
    }
    return .{ .mesh = .{ .verts = try verts.toOwnedSlice(gpa), .tris = try tris.toOwnedSlice(gpa) }, .filled = filled };
}

fn dist(verts: []const f32, a: u32, b: u32) f32 {
    const dx = verts[a * 3] - verts[b * 3];
    const dy = verts[a * 3 + 1] - verts[b * 3 + 1];
    const dz = verts[a * 3 + 2] - verts[b * 3 + 2];
    return @sqrt(dx * dx + dy * dy + dz * dz);
}

fn signedVolume(verts: []const f32, t: []const u32) f64 {
    const a = verts[t[0] * 3 ..][0..3];
    const b = verts[t[1] * 3 ..][0..3];
    const c = verts[t[2] * 3 ..][0..3];
    return (@as(f64, a[0]) * (@as(f64, b[1]) * c[2] - @as(f64, b[2]) * c[1]) +
        @as(f64, a[1]) * (@as(f64, b[2]) * c[0] - @as(f64, b[0]) * c[2]) +
        @as(f64, a[2]) * (@as(f64, b[0]) * c[1] - @as(f64, b[1]) * c[0])) / 6.0;
}

/// Area-weighted vertex normals (unit; zero for an unused vertex).
pub fn vertexNormals(gpa: std.mem.Allocator, verts: []const f32, tris: []const u32) ![]f32 {
    const n = try gpa.alloc(f32, verts.len);
    @memset(n, 0);
    var f: usize = 0;
    while (f < tris.len) : (f += 3) {
        const a = verts[tris[f] * 3 ..][0..3];
        const b = verts[tris[f + 1] * 3 ..][0..3];
        const c = verts[tris[f + 2] * 3 ..][0..3];
        const e1 = [3]f32{ b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        const e2 = [3]f32{ c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        const fn_ = [3]f32{ e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
        for (0..3) |k| for (0..3) |j| {
            n[tris[f + k] * 3 + j] += fn_[j];
        };
    }
    var i: usize = 0;
    while (i < n.len) : (i += 3) {
        const len = @sqrt(n[i] * n[i] + n[i + 1] * n[i + 1] + n[i + 2] * n[i + 2]);
        if (len > 0) for (0..3) |k| {
            n[i + k] /= len;
        };
    }
    return n;
}

// ── Volume sampling and bake ─────────────────────────────────────────────────

/// A sparse attribute volume (`coords` `[M, 4]`, `attrs` `[M, C]`) over `res`^3 voxels of the world box.
pub const Volume = struct {
    table: sparse.Table,
    attrs: []const f32,
    channels: usize,
    res: u32,

    pub fn init(gpa: std.mem.Allocator, coords: []const i32, attrs: []const f32, channels: usize, res: u32) !Volume {
        std.debug.assert(attrs.len == coords.len / 4 * channels);
        return .{ .table = try sparse.Table.init(gpa, coords), .attrs = attrs, .channels = channels, .res = res };
    }

    pub fn deinit(self: *Volume, gpa: std.mem.Allocator) void {
        self.table.deinit(gpa);
    }

    /// Trilinear sample at world point `p` over the eight cells around it, missing voxels dropped and the
    /// weights renormalized (FlexGEMM's `grid_sample_3d`, whose neighbour indices truncate toward zero).
    pub fn sample(self: *const Volume, p: [3]f32, out: []f32) void {
        const vs: f32 = @floatFromInt(self.res);
        const q = [3]f32{ (p[0] - AABB_MIN) * vs, (p[1] - AABB_MIN) * vs, (p[2] - AABB_MIN) * vs };
        @memset(out, 0);
        var wsum: f32 = 0;
        for (0..8) |o| {
            var n: [3]i32 = undefined;
            var w: f32 = 1;
            for (0..3) |k| {
                const off: f32 = if ((o >> @intCast(2 - k)) & 1 == 1) 0.5 else -0.5;
                n[k] = @intFromFloat(q[k] + off);
                w *= 1 - @abs(@as(f32, @floatFromInt(n[k])) + 0.5 - q[k]);
            }
            if (n[0] < 0 or n[1] < 0 or n[2] < 0) continue;
            const row = self.table.find(sparse.voxelKey(n[0], n[1], n[2])) orelse continue;
            wsum += w;
            for (out, self.attrs[row * self.channels ..][0..self.channels]) |*d, a| d.* += w * a;
        }
        const inv = 1.0 / @max(wsum, 1e-12);
        for (out) |*d| d.* *= inv;
    }
};

pub const Atlas = struct {
    /// `[tex * tex, channels]`, row 0 on top; texels no triangle covers are 0.
    attrs: []f32,
    /// 1 where a triangle covers the texel.
    mask: []u8,

    pub fn deinit(self: *Atlas, gpa: std.mem.Allocator) void {
        gpa.free(self.attrs);
        gpa.free(self.mask);
    }
};

pub const Texels = struct {
    /// `[tex * tex, 3]` surface point under each texel (row 0 on top); 0 where no triangle covers it.
    pos: []f32,
    /// 1 where a triangle covers the texel.
    mask: []u8,

    pub fn deinit(self: *Texels, gpa: std.mem.Allocator) void {
        gpa.free(self.pos);
        gpa.free(self.mask);
    }
};

/// Rasterizes the UV layout (`uvs` xatlas-style, v up) into a `tex` x `tex` image: row r, column c holds the 3D
/// point at u = c / (tex - 1), v = 1 - r / (tex - 1), the rasterizer's own texel-centre convention.
pub fn texelPositions(gpa: std.mem.Allocator, positions: []const f32, uvs: []const f32, indices: []const u32, tex: u32) !Texels {
    const nv = positions.len / 3;
    const clip = try gpa.alloc(f32, nv * 4);
    defer gpa.free(clip);
    for (0..nv) |i| {
        clip[i * 4] = uvs[i * 2] * 2 - 1;
        clip[i * 4 + 1] = 1 - uvs[i * 2 + 1] * 2; // the rasterizer's row 0 is clip y = -1
        clip[i * 4 + 2] = 0;
        clip[i * 4 + 3] = 1;
    }
    var raster = try rasterize.rasterize(gpa, clip, indices, tex, tex);
    defer raster.deinit();
    const pos = try rasterize.interpolate(gpa, positions, 3, &raster, indices, 0);
    errdefer gpa.free(pos);
    const mask = try gpa.alloc(u8, @as(usize, tex) * tex);
    for (mask, raster.face_id) |*m, id| m.* = @intFromBool(id != 0);
    return .{ .pos = pos, .mask = mask };
}

/// The volume sampled at every covered texel (`attrs` `[tex * tex, channels]`, 0 elsewhere).
pub fn sampleAtlas(gpa: std.mem.Allocator, t: *const Texels, vol: *const Volume) !Atlas {
    const attrs = try gpa.alloc(f32, t.mask.len * vol.channels);
    errdefer gpa.free(attrs);
    @memset(attrs, 0);
    const mask = try gpa.dupe(u8, t.mask);
    const Ctx = struct { t: *const Texels, vol: *const Volume, attrs: []f32 };
    const body = struct {
        fn run(c: Ctx, lo: usize, hi: usize) void {
            for (lo..hi) |i| {
                if (c.t.mask[i] == 0) continue;
                c.vol.sample(.{ c.t.pos[i * 3], c.t.pos[i * 3 + 1], c.t.pos[i * 3 + 2] }, c.attrs[i * c.vol.channels ..][0..c.vol.channels]);
            }
        }
    }.run;
    sparse.parallelRange(Ctx, .{ .t = t, .vol = vol, .attrs = attrs }, t.mask.len, body);
    return .{ .attrs = attrs, .mask = mask };
}

// ── The reference's cleanup chain ────────────────────────────────────────────

const HOLE_PERIMETER: f32 = 3e-2;
const FLOATER_AREA: f64 = 1e-5;

/// One round of `to_glb`'s cleanup: floaters out, unused vertices out, small holes closed. Consumes `m`.
fn clean(gpa: std.mem.Allocator, m: TriMesh) !TriMesh {
    var cur = m;
    errdefer cur.deinit(gpa);
    const kept = try dropSmallComponents(gpa, cur.verts, cur.tris, FLOATER_AREA);
    gpa.free(cur.tris);
    cur.tris = kept;
    const packed_mesh = try compact(gpa, cur);
    cur.deinit(gpa);
    cur = packed_mesh;
    const filled = try fillHoles(gpa, cur, HOLE_PERIMETER);
    cur.deinit(gpa);
    return filled.mesh;
}

fn simplified(gpa: std.mem.Allocator, m: TriMesh, target_faces: u32) !TriMesh {
    const in = mc.Mesh{ .vertices = m.verts, .normals = &.{}, .indices = m.tris };
    const out = try mesh_simplify.decimate(gpa, &in, target_faces);
    gpa.free(out.normals);
    return .{ .verts = out.vertices, .tris = out.indices };
}

/// `to_glb`'s cleanup chain: close small holes, simplify to 3x the target, clean, simplify to the target, clean.
/// Consumes `m`. Winding is left as it came: a remeshed hull's inner skin faces the hollow.
pub fn simplifyAndClean(gpa: std.mem.Allocator, m: TriMesh, target_faces: u32) !TriMesh {
    var cur = try clean(gpa, m);
    errdefer cur.deinit(gpa);
    if (cur.tris.len / 3 > target_faces) {
        const coarse = try simplified(gpa, cur, target_faces *| 3);
        cur.deinit(gpa);
        cur = try clean(gpa, coarse);
        const fine = try simplified(gpa, cur, target_faces);
        cur.deinit(gpa);
        cur = try clean(gpa, fine);
    }
    return cur;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

/// A closed unit cube centred at the origin, 12 outward triangles.
fn cube(gpa: std.mem.Allocator) !TriMesh {
    const v = [_]f32{ -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1 };
    const t = [_]u32{
        0, 3, 2, 0, 2, 1, // -z
        4, 5, 6, 4, 6, 7, // +z
        0, 1, 5, 0, 5, 4, // -y
        3, 7, 6, 3, 6, 2, // +y
        0, 4, 7, 0, 7, 3, // -x
        1, 2, 6, 1, 6, 5, // +x
    };
    return .{ .verts = try gpa.dupe(f32, &v), .tris = try gpa.dupe(u32, &t) };
}

fn volumeOf(verts: []const f32, tris: []const u32) f64 {
    var vol: f64 = 0;
    var f: usize = 0;
    while (f < tris.len) : (f += 3) vol += signedVolume(verts, tris[f..][0..3]);
    return vol;
}

/// Every undirected edge is used by exactly one triangle in each direction: a closed, consistently wound surface.
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

test "trellis2 mesh: a flat sheet of voxels with +z edges flagged becomes a quad grid, two triangles each" {
    const gpa = testing.allocator;
    var coords: [9 * 4]i32 = undefined;
    var raw: [9 * HEAD_CHANNELS]f32 = undefined;
    for (0..9) |i| {
        coords[i * 4 .. i * 4 + 4][0] = 0;
        coords[i * 4 + 1] = @intCast(i / 3);
        coords[i * 4 + 2] = @intCast(i % 3);
        coords[i * 4 + 3] = 5;
        @memcpy(raw[i * HEAD_CHANNELS ..][0..HEAD_CHANNELS], &[_]f32{ 0, 0, 0, -1, -1, 1, 0 });
    }
    var m = try dualGridMesh(gpa, &coords, &raw, 8);
    defer m.deinit(gpa);
    try testing.expectEqual(@as(usize, 4 * 6), m.tris.len); // 2 x 2 quads
    // sigmoid(0) = 0.5 -> the dual vertex sits at its voxel centre: (c + 0.5) / 8 - 0.5.
    try testing.expectApproxEqAbs(@as(f32, (0 + 0.5) / 8.0 - 0.5), m.verts[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, (5 + 0.5) / 8.0 - 0.5), m.verts[2], 1e-6);
}

test "trellis2 mesh: a quad is dropped when a surrounding voxel is missing, and the heavier diagonal wins the split" {
    const gpa = testing.allocator;
    // Four voxels around one +z edge; weight logits make w0*w2 > w1*w3.
    const coords = [_]i32{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0 };
    var raw: [4 * HEAD_CHANNELS]f32 = @splat(0);
    raw[5] = 1; // voxel 0: +z edge flagged
    raw[6] = 5; // voxel 0 weight softplus(5)
    raw[2 * HEAD_CHANNELS + 6] = 5; // voxel 2 (opposite corner)
    var m = try dualGridMesh(gpa, &coords, &raw, 4);
    defer m.deinit(gpa);
    try testing.expectEqual(@as(usize, 6), m.tris.len);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 0, 2, 3 }, m.tris); // w02 > w13: the 0-2 diagonal
    var gone = try dualGridMesh(gpa, coords[0 .. 3 * 4], raw[0 .. 3 * HEAD_CHANNELS], 4);
    defer gone.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), gone.tris.len);
}

test "trellis2 mesh: compact drops unused vertices and renumbers" {
    const gpa = testing.allocator;
    var m = TriMesh{ .verts = try gpa.dupe(f32, &.{ 9, 9, 9, 0, 0, 0, 1, 0, 0, 0, 1, 0 }), .tris = try gpa.dupe(u32, &.{ 1, 2, 3 }) };
    defer m.deinit(gpa);
    var c = try compact(gpa, m);
    defer c.deinit(gpa);
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 1, 0, 0, 0, 1, 0 }, c.verts);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, c.tris);
}

test "trellis2 mesh: floaters below the area threshold go, the main surface stays" {
    const gpa = testing.allocator;
    var c = try cube(gpa);
    defer c.deinit(gpa);
    // The cube (area 24) on vertices 0-7, plus one loose triangle (area 0.5) on vertices 8-10.
    var verts: std.ArrayList(f32) = .empty;
    defer verts.deinit(gpa);
    try verts.appendSlice(gpa, c.verts);
    try verts.appendSlice(gpa, &.{ 5, 5, 5, 6, 5, 5, 5, 6, 5 });
    var tris: std.ArrayList(u32) = .empty;
    defer tris.deinit(gpa);
    try tris.appendSlice(gpa, c.tris);
    try tris.appendSlice(gpa, &.{ 8, 9, 10 });
    const kept = try dropSmallComponents(gpa, verts.items, tris.items, 1.0);
    defer gpa.free(kept);
    try testing.expectEqual(@as(usize, 36), kept.len);
    const all = try dropSmallComponents(gpa, verts.items, tris.items, 0.1);
    defer gpa.free(all);
    try testing.expectEqual(@as(usize, 39), all.len);
}

test "trellis2 mesh: a small hole is closed with a fan that continues the surface; a large one is left" {
    const gpa = testing.allocator;
    var c = try cube(gpa);
    defer c.deinit(gpa);
    // Remove the -z face (two triangles): a hole of perimeter 8 on a cube of side 2.
    const open_tris = c.tris[6..];
    const open = TriMesh{ .verts = c.verts, .tris = open_tris };
    const kept = try fillHoles(gpa, open, 1.0);
    defer {
        var mm = kept.mesh;
        mm.deinit(gpa);
    }
    try testing.expectEqual(@as(usize, 0), kept.filled);
    try testing.expectEqual(open_tris.len, kept.mesh.tris.len);
    const done = try fillHoles(gpa, open, 8.5);
    defer {
        var mm = done.mesh;
        mm.deinit(gpa);
    }
    try testing.expectEqual(@as(usize, 1), done.filled);
    try testing.expectEqual(open_tris.len + 4 * 3, done.mesh.tris.len); // one fan triangle per boundary edge
    try expectClosedAndConsistent(gpa, done.mesh.tris);
}

test "trellis2 mesh: the volume sample blends the eight cells around a point and ignores missing voxels" {
    const gpa = testing.allocator;
    // Two voxels side by side along x in a 4^3 volume: attribute 2 and 4.
    const coords = [_]i32{ 0, 1, 1, 1, 0, 2, 1, 1 };
    const attrs = [_]f32{ 2, 4 };
    var vol = try Volume.init(gpa, &coords, &attrs, 1, 4);
    defer vol.deinit(gpa);
    var out: [1]f32 = undefined;
    // Cell centres sit at (i + 0.5) / 4 - 0.5.
    const c1 = (1 + 0.5) / 4.0 + AABB_MIN;
    const c2 = (2 + 0.5) / 4.0 + AABB_MIN;
    vol.sample(.{ c1, c1, c1 }, &out);
    try testing.expectApproxEqAbs(@as(f32, 2), out[0], 1e-4);
    vol.sample(.{ (c1 + c2) / 2, c1, c1 }, &out);
    try testing.expectApproxEqAbs(@as(f32, 3), out[0], 1e-4);
    vol.sample(.{ 0.4, 0.4, 0.4 }, &out); // nothing there
    try testing.expectEqual(@as(f32, 0), out[0]);
}

test "trellis2 mesh: each texel holds the surface point at u = col/(tex-1), v = 1 - row/(tex-1)" {
    const gpa = testing.allocator;
    // A quad covering uv [0.1, 0.9]^2 whose 3D position is (u, v, 0): the atlas then reads its own uv back.
    const uvs = [_]f32{ 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9 };
    var positions: [12]f32 = undefined;
    for (0..4) |i| {
        positions[i * 3] = uvs[i * 2];
        positions[i * 3 + 1] = uvs[i * 2 + 1];
        positions[i * 3 + 2] = 0;
    }
    const indices = [_]u32{ 0, 1, 2, 0, 2, 3 };
    const tex: u32 = 64;
    var t = try texelPositions(gpa, &positions, &uvs, &indices, tex);
    defer t.deinit(gpa);
    const denom: f32 = @floatFromInt(tex - 1);
    for ([_][2]u32{ .{ 10, 10 }, .{ 20, 50 }, .{ 50, 20 }, .{ 32, 32 } }) |rc| {
        const i = rc[0] * tex + rc[1];
        try testing.expectEqual(@as(u8, 1), t.mask[i]);
        try testing.expectApproxEqAbs(@as(f32, @floatFromInt(rc[1])) / denom, t.pos[i * 3], 0.02);
        try testing.expectApproxEqAbs(1 - @as(f32, @floatFromInt(rc[0])) / denom, t.pos[i * 3 + 1], 0.02);
    }
    try testing.expectEqual(@as(u8, 0), t.mask[0]); // outside the quad
    try testing.expectEqual(@as(u8, 0), t.mask[tex * tex - 1]);
}

test "trellis2 mesh: simplifyAndClean closes the cube's hole, drops a floater and leaves a closed outward surface" {
    const gpa = testing.allocator;
    var c = try cube(gpa);
    defer c.deinit(gpa);
    var verts: std.ArrayList(f32) = .empty;
    for (c.verts) |v| try verts.append(gpa, v * 0.002); // side 0.004: its hole is under the 0.03 perimeter limit
    try verts.appendSlice(gpa, &.{ 0.001, 0.001, 5, 0.002, 0.001, 5, 0.001, 0.002, 5 }); // a speck far from the cube
    var tris: std.ArrayList(u32) = .empty;
    try tris.appendSlice(gpa, c.tris[6..]); // the -z face removed: a hole
    try tris.appendSlice(gpa, &.{ 8, 9, 10 });
    const m = TriMesh{ .verts = try verts.toOwnedSlice(gpa), .tris = try tris.toOwnedSlice(gpa) };
    var out = try simplifyAndClean(gpa, m, 1000);
    defer out.deinit(gpa);
    try expectClosedAndConsistent(gpa, out.tris);
    try testing.expect(volumeOf(out.verts, out.tris) > 0);
    try testing.expectEqual(@as(usize, 9), out.verts.len / 3); // 8 corners + the hole's centre
}

test "trellis2 oracle: the dual grid mesh equals the reference's vertices and triangles (TRELLIS2_FIXTURES)" {
    const gpa = testing.allocator;
    const dir = std.mem.span(std.c.getenv("TRELLIS2_FIXTURES") orelse return error.SkipZigTest);
    const model_mod = @import("../../sdk/weights/load.zig");
    const mlx = @import("../../mlx/ffi.zig");
    var fx = try model_mod.loadWeightsFile(gpa, dir, "shape_dec.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const host = struct {
        fn f32s(a: mlx.mlx_array, st: mlx.mlx_stream) ![]const f32 {
            var c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c);
            try mlx.check(mlx.mlx_contiguous(&c, a, false, st));
            try mlx.check(mlx.mlx_array_eval(c));
            return (mlx.mlx_array_data_float32(c) orelse return error.MlxError)[0..mlx.mlx_array_size(c)];
        }
        fn i32s(a: mlx.mlx_array, st: mlx.mlx_stream) ![]const i32 {
            var c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c);
            try mlx.check(mlx.mlx_astype(&c, a, .int32, st));
            var d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(d);
            try mlx.check(mlx.mlx_contiguous(&d, c, false, st));
            try mlx.check(mlx.mlx_array_eval(d));
            return (mlx.mlx_array_data_int32(d) orelse return error.MlxError)[0..mlx.mlx_array_size(d)];
        }
    };
    const coords = try gpa.dupe(i32, try host.i32s(fx.get("out_coords").?, s));
    defer gpa.free(coords);
    const raw = try gpa.dupe(f32, try host.f32s(fx.get("raw").?, s));
    defer gpa.free(raw);
    const want_v = try gpa.dupe(f32, try host.f32s(fx.get("verts").?, s));
    defer gpa.free(want_v);
    const want_t = try gpa.dupe(i32, try host.i32s(fx.get("tris").?, s));
    defer gpa.free(want_t);
    var m = try dualGridMesh(gpa, coords, raw, 512);
    defer m.deinit(gpa);
    try testing.expectEqual(want_v.len, m.verts.len);
    var worst: f32 = 0;
    for (want_v, m.verts) |a, b| worst = @max(worst, @abs(a - b));
    std.debug.print("dual grid: {d} vertices, {d} faces, worst vertex error {e:.2}\n", .{ m.verts.len / 3, m.tris.len / 3, worst });
    try testing.expect(worst < 1e-5);
    try testing.expectEqual(want_t.len, m.tris.len);
    for (want_t, m.tris) |a, b| try testing.expectEqual(@as(u32, @intCast(a)), b);
}
