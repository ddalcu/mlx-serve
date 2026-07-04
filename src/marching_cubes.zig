//! Marching cubes surface extraction (classic Lorensen–Cline tables), pure Zig,
//! zero MLX deps. Turns a scalar field sampled on a regular grid into an indexed,
//! watertight triangle mesh with per-vertex normals. Used by the Hunyuan3D-2.1
//! image→mesh engine to polygonise the decoded SDF grid, but the module knows
//! nothing about that pipeline.
//!
//! Sign convention: INSIDE = positive field values, surface at `level`. Triangles
//! are wound so a CCW winding faces along −∇field (outward for an inside-positive
//! field), matching glTF's front-face convention.

const std = @import("std");

pub const Mesh = struct {
    vertices: []f32, // xyz interleaved, world space
    normals: []f32, // xyz interleaved, unit length
    indices: []u32, // CCW = outward (glTF front face)

    pub fn deinit(self: *Mesh, alloc: std.mem.Allocator) void {
        alloc.free(self.vertices);
        alloc.free(self.normals);
        alloc.free(self.indices);
        self.* = undefined;
    }
};

/// Extract the `level` isosurface of `grid`. `grid` is `n[0]*n[1]*n[2]` samples in
/// x-major `ij` order: idx = (ix*n[1] + iy)*n[2] + iz. World coords of a corner are
/// `corner*scale + offset` per axis. Caller owns the returned mesh.
pub fn extract(
    alloc: std.mem.Allocator,
    grid: []const f32,
    n: [3]usize,
    level: f32,
    scale: [3]f32,
    offset: [3]f32,
) !Mesh {
    std.debug.assert(grid.len == n[0] * n[1] * n[2]);

    var verts: std.ArrayList(f32) = .empty;
    errdefer verts.deinit(alloc);
    var norms: std.ArrayList(f32) = .empty;
    errdefer norms.deinit(alloc);
    var indices: std.ArrayList(u32) = .empty;
    errdefer indices.deinit(alloc);
    var deg: std.ArrayList(u32) = .empty; // vertices with a degenerate gradient
    defer deg.deinit(alloc);

    var edge_map = std.AutoHashMap(u64, u32).init(alloc);
    defer edge_map.deinit();

    var b = Builder{
        .alloc = alloc,
        .grid = grid,
        .n = n,
        .level = level,
        .scale = scale,
        .offset = offset,
        .verts = &verts,
        .norms = &norms,
        .edge_map = &edge_map,
        .deg = &deg,
    };

    if (n[0] >= 2 and n[1] >= 2 and n[2] >= 2) {
        var ix: usize = 0;
        while (ix + 1 < n[0]) : (ix += 1) {
            var iy: usize = 0;
            while (iy + 1 < n[1]) : (iy += 1) {
                var iz: usize = 0;
                while (iz + 1 < n[2]) : (iz += 1) {
                    var vals: [8]f32 = undefined;
                    for (0..8) |c| {
                        const ci = ix + CORNER[c][0];
                        const cj = iy + CORNER[c][1];
                        const ck = iz + CORNER[c][2];
                        vals[c] = grid[idx3(n, ci, cj, ck)];
                    }
                    var cube: u8 = 0;
                    for (0..8) |c| {
                        if (vals[c] < level) cube |= @as(u8, 1) << @intCast(c);
                    }
                    const edges = EDGE_TABLE[cube];
                    if (edges == 0) continue;

                    var vidx: [12]u32 = undefined;
                    for (0..12) |e| {
                        if (edges & (@as(u12, 1) << @intCast(e)) != 0) {
                            vidx[e] = try b.vertexForEdge(ix, iy, iz, e, vals);
                        }
                    }

                    const tris = TRI_TABLE[cube];
                    var t: usize = 0;
                    while (t < 16 and tris[t] >= 0) : (t += 3) {
                        try indices.append(alloc, vidx[@intCast(tris[t])]);
                        try indices.append(alloc, vidx[@intCast(tris[t + 1])]);
                        try indices.append(alloc, vidx[@intCast(tris[t + 2])]);
                    }
                }
            }
        }
    }

    // Global winding: orient so a CCW face's geometric normal agrees with the
    // −∇field vertex normals (outward). The standard tables produce a consistent
    // orientation, so a single vote flips the whole mesh if the convention is
    // inverted for our inside-positive sign.
    orientWinding(verts.items, norms.items, indices.items);

    // Degenerate-gradient vertices: fall back to accumulated face normals.
    if (deg.items.len > 0) {
        try fixDegenerateNormals(alloc, verts.items, norms.items, indices.items, deg.items);
    }

    return Mesh{
        .vertices = try verts.toOwnedSlice(alloc),
        .normals = try norms.toOwnedSlice(alloc),
        .indices = try indices.toOwnedSlice(alloc),
    };
}

const Builder = struct {
    alloc: std.mem.Allocator,
    grid: []const f32,
    n: [3]usize,
    level: f32,
    scale: [3]f32,
    offset: [3]f32,
    verts: *std.ArrayList(f32),
    norms: *std.ArrayList(f32),
    edge_map: *std.AutoHashMap(u64, u32),
    deg: *std.ArrayList(u32),

    fn vertexForEdge(self: *Builder, ix: usize, iy: usize, iz: usize, e: usize, vals: [8]f32) !u32 {
        // Canonical grid-edge key (shared by adjacent cells → watertight dedup).
        const eo = EDGE_ORIGIN[e];
        const oi = ix + eo[0];
        const oj = iy + eo[1];
        const ok = iz + eo[2];
        const corner_lin = (oi * self.n[1] + oj) * self.n[2] + ok;
        const key = @as(u64, corner_lin) * 3 + EDGE_AXIS[e];
        if (self.edge_map.get(key)) |v| return v;

        const a = EDGE_CORNERS[e][0];
        const bb = EDGE_CORNERS[e][1];
        const ai = ix + CORNER[a][0];
        const aj = iy + CORNER[a][1];
        const ak = iz + CORNER[a][2];
        const bi = ix + CORNER[bb][0];
        const bj = iy + CORNER[bb][1];
        const bk = iz + CORNER[bb][2];

        const va = vals[a];
        const vb = vals[bb];
        const d = vb - va;
        const t: f32 = if (@abs(d) < 1e-12) 0.5 else (self.level - va) / d;

        const pa = self.worldOf(ai, aj, ak);
        const pb = self.worldOf(bi, bj, bk);
        var pos: [3]f32 = undefined;
        for (0..3) |k| pos[k] = pa[k] + t * (pb[k] - pa[k]);

        const ga = self.gradAt(ai, aj, ak);
        const gb = self.gradAt(bi, bj, bk);
        var nrm: [3]f32 = undefined;
        for (0..3) |k| nrm[k] = -(ga[k] + t * (gb[k] - ga[k]));
        const len = @sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);

        const vidx: u32 = @intCast(self.verts.items.len / 3);
        if (len < 1e-9) {
            try self.norms.appendSlice(self.alloc, &[_]f32{ 0, 0, 0 });
            try self.deg.append(self.alloc, vidx);
        } else {
            try self.norms.appendSlice(self.alloc, &[_]f32{ nrm[0] / len, nrm[1] / len, nrm[2] / len });
        }
        try self.verts.appendSlice(self.alloc, &pos);
        try self.edge_map.put(key, vidx);
        return vidx;
    }

    fn worldOf(self: *Builder, i: usize, j: usize, k: usize) [3]f32 {
        return .{
            @as(f32, @floatFromInt(i)) * self.scale[0] + self.offset[0],
            @as(f32, @floatFromInt(j)) * self.scale[1] + self.offset[1],
            @as(f32, @floatFromInt(k)) * self.scale[2] + self.offset[2],
        };
    }

    /// Central-difference world-space gradient at a grid corner (one-sided on faces).
    fn gradAt(self: *Builder, i: usize, j: usize, k: usize) [3]f32 {
        return .{
            self.diff(0, i, j, k),
            self.diff(1, i, j, k),
            self.diff(2, i, j, k),
        };
    }

    fn diff(self: *Builder, axis: usize, i: usize, j: usize, k: usize) f32 {
        const c: [3]usize = .{ i, j, k };
        const dim = self.n[axis];
        var lo = c;
        var hi = c;
        var span: f32 = 2;
        if (c[axis] == 0) {
            hi[axis] += 1;
            span = 1;
        } else if (c[axis] + 1 >= dim) {
            lo[axis] -= 1;
            span = 1;
        } else {
            lo[axis] -= 1;
            hi[axis] += 1;
        }
        const fhi = self.grid[idx3(self.n, hi[0], hi[1], hi[2])];
        const flo = self.grid[idx3(self.n, lo[0], lo[1], lo[2])];
        return (fhi - flo) / (span * self.scale[axis]);
    }
};

inline fn idx3(n: [3]usize, i: usize, j: usize, k: usize) usize {
    return (i * n[1] + j) * n[2] + k;
}

fn orientWinding(verts: []const f32, norms: []const f32, indices: []u32) void {
    if (indices.len == 0) return;
    var agree: f64 = 0;
    var t: usize = 0;
    while (t < indices.len) : (t += 3) {
        const gn = faceNormal(verts, indices[t], indices[t + 1], indices[t + 2]);
        var vn: [3]f32 = .{ 0, 0, 0 };
        for ([_]u32{ indices[t], indices[t + 1], indices[t + 2] }) |vi| {
            for (0..3) |c| vn[c] += norms[vi * 3 + c];
        }
        agree += @as(f64, gn[0] * vn[0] + gn[1] * vn[1] + gn[2] * vn[2]);
    }
    if (agree < 0) {
        var i: usize = 0;
        while (i < indices.len) : (i += 3) {
            const tmp = indices[i + 1];
            indices[i + 1] = indices[i + 2];
            indices[i + 2] = tmp;
        }
    }
}

fn faceNormal(verts: []const f32, ia: u32, ib: u32, ic: u32) [3]f32 {
    var e1: [3]f32 = undefined;
    var e2: [3]f32 = undefined;
    for (0..3) |c| {
        e1[c] = verts[ib * 3 + c] - verts[ia * 3 + c];
        e2[c] = verts[ic * 3 + c] - verts[ia * 3 + c];
    }
    return .{
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    };
}

fn fixDegenerateNormals(
    alloc: std.mem.Allocator,
    verts: []const f32,
    norms: []f32,
    indices: []const u32,
    deg: []const u32,
) !void {
    const nverts = verts.len / 3;
    const acc = try alloc.alloc(f32, nverts * 3);
    defer alloc.free(acc);
    @memset(acc, 0);
    var t: usize = 0;
    while (t < indices.len) : (t += 3) {
        const gn = faceNormal(verts, indices[t], indices[t + 1], indices[t + 2]);
        for ([_]u32{ indices[t], indices[t + 1], indices[t + 2] }) |vi| {
            for (0..3) |c| acc[vi * 3 + c] += gn[c];
        }
    }
    for (deg) |vi| {
        const x = acc[vi * 3 + 0];
        const y = acc[vi * 3 + 1];
        const z = acc[vi * 3 + 2];
        const len = @sqrt(x * x + y * y + z * z);
        if (len < 1e-12) {
            norms[vi * 3 + 0] = 0;
            norms[vi * 3 + 1] = 1;
            norms[vi * 3 + 2] = 0;
        } else {
            norms[vi * 3 + 0] = x / len;
            norms[vi * 3 + 1] = y / len;
            norms[vi * 3 + 2] = z / len;
        }
    }
}

// ---------------------------------------------------------------------------
// Classic Lorensen–Cline tables (public domain, Paul Bourke's canonical form).
// Cube corners, edges 0..11, and the offset of each edge's lower corner.
// ---------------------------------------------------------------------------

const CORNER = [8][3]usize{
    .{ 0, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 0, 1 }, .{ 1, 0, 1 }, .{ 1, 1, 1 }, .{ 0, 1, 1 },
};

const EDGE_CORNERS = [12][2]u8{
    .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 0 },
    .{ 4, 5 }, .{ 5, 6 }, .{ 6, 7 }, .{ 7, 4 },
    .{ 0, 4 }, .{ 1, 5 }, .{ 2, 6 }, .{ 3, 7 },
};

const EDGE_AXIS = [12]u64{ 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2 };

const EDGE_ORIGIN = [12][3]usize{
    .{ 0, 0, 0 }, .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 0 },
    .{ 0, 0, 1 }, .{ 1, 0, 1 }, .{ 0, 1, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 1, 0 }, .{ 0, 1, 0 },
};

const EDGE_TABLE = [256]u12{
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99,  0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66,  0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55,  0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc,  0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55,  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,  0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
};

const TRI_TABLE = [256][16]i8{
    .{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
    .{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
    .{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
    .{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
    .{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
    .{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
    .{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
    .{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
    .{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
    .{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
    .{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
    .{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
    .{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
    .{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
    .{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
    .{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
    .{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
    .{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
    .{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
    .{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
    .{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
    .{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
    .{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
    .{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
    .{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
    .{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
    .{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
    .{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
    .{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
    .{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
    .{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
    .{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
    .{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
    .{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
    .{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
    .{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
    .{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
    .{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
    .{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
    .{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
    .{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
    .{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
    .{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
    .{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
    .{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
    .{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
    .{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
    .{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
    .{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
    .{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
    .{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
    .{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
    .{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
    .{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
    .{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
    .{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
    .{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
    .{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
    .{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
    .{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
    .{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
    .{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
    .{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
    .{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
    .{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
    .{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
    .{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
    .{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
    .{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
    .{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
    .{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
    .{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
    .{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
    .{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
    .{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
    .{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
    .{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
    .{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
    .{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
    .{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
    .{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
    .{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
    .{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
    .{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
    .{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
    .{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
    .{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
    .{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
    .{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
    .{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
    .{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
    .{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
    .{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
    .{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
    .{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
    .{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
    .{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
    .{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
    .{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
    .{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
    .{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
    .{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
    .{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
    .{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
    .{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
    .{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
    .{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
    .{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
    .{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
    .{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
    .{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
    .{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
    .{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
    .{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
    .{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
    .{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
    .{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
    .{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
    .{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
    .{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
    .{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
    .{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
    .{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
    .{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    .{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
};

// ---------------------------------------------------------------------------
// Tests (hermetic — analytic fields, no weights, no env vars).
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Inside-positive sphere field f = r0 − |p − center| on an [n]³ grid over [−1,1]³.
fn buildSphere(a: std.mem.Allocator, np: usize, center: [3]f32, r0: f32) ![]f32 {
    const grid = try a.alloc(f32, np * np * np);
    const step: f32 = 2.0 / @as(f32, @floatFromInt(np - 1));
    for (0..np) |i| {
        for (0..np) |j| {
            for (0..np) |k| {
                const x = @as(f32, @floatFromInt(i)) * step - 1.0 - center[0];
                const y = @as(f32, @floatFromInt(j)) * step - 1.0 - center[1];
                const z = @as(f32, @floatFromInt(k)) * step - 1.0 - center[2];
                grid[idx3(.{ np, np, np }, i, j, k)] = r0 - @sqrt(x * x + y * y + z * z);
            }
        }
    }
    return grid;
}

const EdgeStats = struct { edges: usize, all_shared_twice: bool };

/// Every undirected edge must be referenced by exactly two triangles (closed 2-manifold).
fn edgeStats(a: std.mem.Allocator, indices: []const u32) !EdgeStats {
    var counts = std.AutoHashMap(u64, u32).init(a);
    defer counts.deinit();
    var t: usize = 0;
    while (t < indices.len) : (t += 3) {
        const tri = [3]u32{ indices[t], indices[t + 1], indices[t + 2] };
        for (0..3) |e| {
            var lo = tri[e];
            var hi = tri[(e + 1) % 3];
            if (lo > hi) std.mem.swap(u32, &lo, &hi);
            const key = (@as(u64, lo) << 32) | hi;
            const gop = try counts.getOrPut(key);
            if (gop.found_existing) gop.value_ptr.* += 1 else gop.value_ptr.* = 1;
        }
    }
    var all_twice = true;
    var it = counts.valueIterator();
    while (it.next()) |v| {
        if (v.* != 2) all_twice = false;
    }
    return .{ .edges = counts.count(), .all_shared_twice = all_twice };
}

test "marching cubes: sphere is a closed 2-manifold with correct Euler characteristic" {
    const a = testing.allocator;
    const np: usize = 64;
    const grid = try buildSphere(a, np, .{ 0, 0, 0 }, 0.35);
    defer a.free(grid);

    const step: f32 = 2.0 / @as(f32, @floatFromInt(np - 1));
    var mesh = try extract(a, grid, .{ np, np, np }, 0.0, .{ step, step, step }, .{ -1, -1, -1 });
    defer mesh.deinit(a);

    const v = mesh.vertices.len / 3;
    const f = mesh.indices.len / 3;
    try testing.expect(v > 0 and f > 0);

    const stats = try edgeStats(a, mesh.indices);
    try testing.expect(stats.all_shared_twice); // closed 2-manifold
    // Euler characteristic V − E + F == 2 for a genus-0 closed surface.
    const chi = @as(i64, @intCast(v)) - @as(i64, @intCast(stats.edges)) + @as(i64, @intCast(f));
    try testing.expectEqual(@as(i64, 2), chi);
}

test "marching cubes: sphere radius and outward normals" {
    const a = testing.allocator;
    const np: usize = 64;
    const r0: f32 = 0.35;
    const grid = try buildSphere(a, np, .{ 0, 0, 0 }, r0);
    defer a.free(grid);

    const step: f32 = 2.0 / @as(f32, @floatFromInt(np - 1));
    var mesh = try extract(a, grid, .{ np, np, np }, 0.0, .{ step, step, step }, .{ -1, -1, -1 });
    defer mesh.deinit(a);

    var max_err: f32 = 0;
    var i: usize = 0;
    while (i < mesh.vertices.len) : (i += 3) {
        const x = mesh.vertices[i];
        const y = mesh.vertices[i + 1];
        const z = mesh.vertices[i + 2];
        const r = @sqrt(x * x + y * y + z * z);
        max_err = @max(max_err, @abs(r - r0));

        // Normal must point outward (radially).
        const nx = mesh.normals[i];
        const ny = mesh.normals[i + 1];
        const nz = mesh.normals[i + 2];
        const nl = @sqrt(nx * nx + ny * ny + nz * nz);
        try testing.expect(@abs(nl - 1.0) < 1e-3); // unit length
        const dot = (x * nx + y * ny + z * nz) / r;
        try testing.expect(dot > 0.9);
    }
    try testing.expect(max_err < step); // within one cell
}

test "marching cubes: winding is CCW-outward" {
    const a = testing.allocator;
    const np: usize = 48;
    const grid = try buildSphere(a, np, .{ 0, 0, 0 }, 0.35);
    defer a.free(grid);

    const step: f32 = 2.0 / @as(f32, @floatFromInt(np - 1));
    var mesh = try extract(a, grid, .{ np, np, np }, 0.0, .{ step, step, step }, .{ -1, -1, -1 });
    defer mesh.deinit(a);

    // For a sphere at the origin, the geometric normal of every CCW triangle must
    // point away from the center (i.e. agree with the face centroid direction).
    var bad: usize = 0;
    var t: usize = 0;
    while (t < mesh.indices.len) : (t += 3) {
        const gn = faceNormal(mesh.vertices, mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]);
        var c: [3]f32 = .{ 0, 0, 0 };
        for ([_]u32{ mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2] }) |vi| {
            for (0..3) |d| c[d] += mesh.vertices[vi * 3 + d] / 3.0;
        }
        if (gn[0] * c[0] + gn[1] * c[1] + gn[2] * c[2] <= 0) bad += 1;
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "marching cubes: two disjoint spheres give Euler characteristic 4" {
    const a = testing.allocator;
    const np: usize = 80;
    // Two spheres centered at ±0.45 on x, each r=0.2, well separated.
    const grid = try a.alloc(f32, np * np * np);
    defer a.free(grid);
    const step: f32 = 2.0 / @as(f32, @floatFromInt(np - 1));
    for (0..np) |i| {
        for (0..np) |j| {
            for (0..np) |k| {
                const x = @as(f32, @floatFromInt(i)) * step - 1.0;
                const y = @as(f32, @floatFromInt(j)) * step - 1.0;
                const z = @as(f32, @floatFromInt(k)) * step - 1.0;
                const d0 = @sqrt((x + 0.45) * (x + 0.45) + y * y + z * z);
                const d1 = @sqrt((x - 0.45) * (x - 0.45) + y * y + z * z);
                grid[idx3(.{ np, np, np }, i, j, k)] = @max(0.2 - d0, 0.2 - d1);
            }
        }
    }
    var mesh = try extract(a, grid, .{ np, np, np }, 0.0, .{ step, step, step }, .{ -1, -1, -1 });
    defer mesh.deinit(a);

    const v = mesh.vertices.len / 3;
    const f = mesh.indices.len / 3;
    const stats = try edgeStats(a, mesh.indices);
    try testing.expect(stats.all_shared_twice);
    const chi = @as(i64, @intCast(v)) - @as(i64, @intCast(stats.edges)) + @as(i64, @intCast(f));
    try testing.expectEqual(@as(i64, 4), chi);
}

test "marching cubes: non-zero level shifts the radius" {
    const a = testing.allocator;
    const np: usize = 64;
    const r0: f32 = 0.35;
    const level: f32 = 0.1;
    const grid = try buildSphere(a, np, .{ 0, 0, 0 }, r0);
    defer a.free(grid);

    const step: f32 = 2.0 / @as(f32, @floatFromInt(np - 1));
    var mesh = try extract(a, grid, .{ np, np, np }, level, .{ step, step, step }, .{ -1, -1, -1 });
    defer mesh.deinit(a);

    // Surface at f = level ⇒ radius r0 − level.
    const expect_r = r0 - level;
    var max_err: f32 = 0;
    var i: usize = 0;
    while (i < mesh.vertices.len) : (i += 3) {
        const r = @sqrt(mesh.vertices[i] * mesh.vertices[i] +
            mesh.vertices[i + 1] * mesh.vertices[i + 1] +
            mesh.vertices[i + 2] * mesh.vertices[i + 2]);
        max_err = @max(max_err, @abs(r - expect_r));
    }
    try testing.expect(max_err < step);
}

test "marching cubes: empty (all-negative) grid yields no geometry" {
    const a = testing.allocator;
    const np: usize = 16;
    const grid = try a.alloc(f32, np * np * np);
    defer a.free(grid);
    @memset(grid, -1.0);
    var mesh = try extract(a, grid, .{ np, np, np }, 0.0, .{ 1, 1, 1 }, .{ 0, 0, 0 });
    defer mesh.deinit(a);
    try testing.expectEqual(@as(usize, 0), mesh.vertices.len);
    try testing.expectEqual(@as(usize, 0), mesh.indices.len);
}
