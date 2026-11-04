//! UV unwrapping via vendored xatlas (lib/xatlas, MIT) for the Hunyuan3D-2.1
//! texture paint stage. Mirrors the Python `xatlas.parametrize(positions,
//! faces)` call the reference makes with ALL defaults (hy3dpaint
//! utils/uvwrap_utils.py): default ChartOptions/PackOptions, positions + u32
//! triangle indices only. UVs come back normalized to [0,1] by atlas
//! width/height; `vmapping[i]` names the ORIGINAL vertex each (possibly
//! seam-duplicated) output vertex came from.

const std = @import("std");

extern fn uvw_parametrize(positions: [*]const f32, vertex_count: u32, indices: [*]const u32, index_count: u32, face_material: ?[*]const u32) ?*anyopaque;
extern fn uvw_vertex_count(r: *const anyopaque) u32;
extern fn uvw_index_count(r: *const anyopaque) u32;
extern fn uvw_chart_count(r: *const anyopaque) u32;
extern fn uvw_atlas_width(r: *const anyopaque) u32;
extern fn uvw_atlas_height(r: *const anyopaque) u32;
extern fn uvw_vmapping(r: *const anyopaque) [*]const u32;
extern fn uvw_indices(r: *const anyopaque) [*]const u32;
extern fn uvw_uvs(r: *const anyopaque) [*]const f32;
extern fn uvw_free(r: *anyopaque) void;

pub const Unwrap = struct {
    /// New vertex → original vertex index (length = new vertex count).
    vmapping: []u32,
    /// Re-indexed triangles into the NEW vertex list (length = input index count).
    indices: []u32,
    /// Normalized [0,1] UVs, packed u,v (length = new vertex count * 2).
    uvs: []f32,
    chart_count: u32,
    atlas_width: u32,
    atlas_height: u32,

    pub fn deinit(self: *Unwrap, allocator: std.mem.Allocator) void {
        allocator.free(self.vmapping);
        allocator.free(self.indices);
        allocator.free(self.uvs);
        self.* = undefined;
    }
};

pub const UvwrapError = error{ UnwrapFailed, OutOfMemory };

/// Unwrap a triangle mesh: `positions` packed xyz (len % 3 == 0), `indices`
/// triangles (len % 3 == 0). Copies the shim result into caller-owned slices.
pub fn parametrize(allocator: std.mem.Allocator, positions: []const f32, indices: []const u32) UvwrapError!Unwrap {
    return run(allocator, positions, indices, null);
}

/// Triangles one region holds at least, and the most regions a mesh is cut into: both depend on the mesh
/// alone, so the unwrap is the same on every machine.
const REGION_FACES = 8192;
const MAX_REGIONS = 128;

/// `parametrize` for a big connected surface. xatlas segments a connected surface on ONE thread and only
/// parameterizes the charts in parallel, so the mesh is cut into spatial regions first and each region
/// segments on its own core. The cuts add a seam along each region border; the texture is baked from a
/// volume and dilated, so a seam there costs nothing visible. Below one region it is `parametrize`.
pub fn parametrizeParallel(allocator: std.mem.Allocator, positions: []const f32, indices: []const u32) UvwrapError!Unwrap {
    if (!validMesh(positions, indices)) return error.UnwrapFailed;
    const parts: u32 = @intCast(@min(indices.len / 3 / REGION_FACES, MAX_REGIONS));
    if (parts < 2) return run(allocator, positions, indices, null);
    const labels = try regionLabels(allocator, positions, indices, parts);
    defer allocator.free(labels);
    return run(allocator, positions, indices, labels);
}

fn validMesh(positions: []const f32, indices: []const u32) bool {
    return positions.len != 0 and positions.len % 3 == 0 and indices.len >= 3 and indices.len % 3 == 0;
}

/// One label in `[0, parts)` per triangle: a recursive median split of the triangle centroids along their
/// widest axis, so the blocks are equal in triangles (to one) and compact.
pub fn regionLabels(allocator: std.mem.Allocator, positions: []const f32, indices: []const u32, parts: u32) ![]u32 {
    const tris = indices.len / 3;
    const cent = try allocator.alloc([3]f32, tris);
    defer allocator.free(cent);
    for (cent, 0..) |*c, f| {
        for (0..3) |ax| {
            var sum: f32 = 0;
            for (0..3) |k| sum += positions[indices[f * 3 + k] * 3 + ax];
            c[ax] = sum / 3;
        }
    }
    const order = try allocator.alloc(u32, tris);
    defer allocator.free(order);
    for (order, 0..) |*o, f| o.* = @intCast(f);
    const labels = try allocator.alloc(u32, tris);
    splitRegion(order, cent, labels, 0, parts);
    return labels;
}

fn splitRegion(order: []u32, cent: []const [3]f32, labels: []u32, first: u32, parts: u32) void {
    if (parts <= 1) {
        for (order) |f| labels[f] = first;
        return;
    }
    var lo = cent[order[0]];
    var hi = lo;
    for (order) |f| for (0..3) |ax| {
        lo[ax] = @min(lo[ax], cent[f][ax]);
        hi[ax] = @max(hi[ax], cent[f][ax]);
    };
    var axis: usize = 0;
    for (1..3) |ax| {
        if (hi[ax] - lo[ax] > hi[axis] - lo[axis]) axis = ax;
    }
    const Along = struct {
        cent: []const [3]f32,
        axis: usize,
        fn less(self: @This(), a: u32, b: u32) bool {
            return self.cent[a][self.axis] < self.cent[b][self.axis];
        }
    };
    std.sort.pdq(u32, order, Along{ .cent = cent, .axis = axis }, Along.less);
    const left = parts / 2;
    const cut = order.len * left / parts;
    splitRegion(order[0..cut], cent, labels, first, left);
    splitRegion(order[cut..], cent, labels, first + left, parts - left);
}

fn run(allocator: std.mem.Allocator, positions: []const f32, indices: []const u32, face_material: ?[]const u32) UvwrapError!Unwrap {
    if (!validMesh(positions, indices)) return error.UnwrapFailed;
    const r = uvw_parametrize(positions.ptr, @intCast(positions.len / 3), indices.ptr, @intCast(indices.len), if (face_material) |m| m.ptr else null) orelse
        return error.UnwrapFailed;
    defer uvw_free(r);

    const nv: usize = uvw_vertex_count(r);
    const ni: usize = uvw_index_count(r);
    var out = Unwrap{
        .vmapping = try allocator.dupe(u32, uvw_vmapping(r)[0..nv]),
        .indices = undefined,
        .uvs = undefined,
        .chart_count = uvw_chart_count(r),
        .atlas_width = uvw_atlas_width(r),
        .atlas_height = uvw_atlas_height(r),
    };
    errdefer allocator.free(out.vmapping);
    out.indices = try allocator.dupe(u32, uvw_indices(r)[0..ni]);
    errdefer allocator.free(out.indices);
    out.uvs = try allocator.dupe(f32, uvw_uvs(r)[0 .. nv * 2]);
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic (the vendored lib is deterministic single-threaded).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

const cube_positions = [_]f32{
    0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, // z=0 face corners
    0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, // z=1 face corners
};
// 12 triangles, CCW-outward.
const cube_indices = [_]u32{
    0, 2, 1, 0, 3, 2, // -z
    4, 5, 6, 4, 6, 7, // +z
    0, 1, 5, 0, 5, 4, // -y
    3, 6, 2, 3, 7, 6, // +y
    0, 4, 7, 0, 7, 3, // -x
    1, 2, 6, 1, 6, 5, // +x
};

test "uvwrap: cube unwraps with valid mapping, normalized non-degenerate UVs" {
    const a = testing.allocator;
    var uw = try parametrize(a, &cube_positions, &cube_indices);
    defer uw.deinit(a);

    try testing.expect(uw.vmapping.len >= 8); // seams duplicate vertices
    try testing.expectEqual(cube_indices.len, uw.indices.len);
    try testing.expect(uw.chart_count > 0);
    try testing.expect(uw.atlas_width > 0 and uw.atlas_height > 0);
    for (uw.vmapping) |orig| try testing.expect(orig < 8);
    for (uw.indices) |idx| try testing.expect(idx < uw.vmapping.len);
    for (uw.uvs) |v| try testing.expect(v >= 0.0 and v <= 1.0);

    // Every output triangle must be non-degenerate in UV space.
    var t: usize = 0;
    while (t < uw.indices.len) : (t += 3) {
        const ia = uw.indices[t + 0];
        const ib = uw.indices[t + 1];
        const ic = uw.indices[t + 2];
        const ax = uw.uvs[ia * 2 + 0];
        const ay = uw.uvs[ia * 2 + 1];
        const bx = uw.uvs[ib * 2 + 0];
        const by = uw.uvs[ib * 2 + 1];
        const cx = uw.uvs[ic * 2 + 0];
        const cy = uw.uvs[ic * 2 + 1];
        const area2 = @abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
        try testing.expect(area2 > 0.0);
    }

    // Positions survive the mapping: each new vertex's original xyz is one of
    // the cube's corners (identity check through vmapping).
    for (uw.vmapping) |orig| {
        const x = cube_positions[orig * 3];
        try testing.expect(x == 0.0 or x == 1.0);
    }
}

/// A flat `n` x `n` grid of quads (2 n^2 triangles): one connected surface xatlas unwraps as a single chart.
fn sheet(a: std.mem.Allocator, n: usize) !struct { pos: []f32, idx: []u32 } {
    const pos = try a.alloc(f32, (n + 1) * (n + 1) * 3);
    errdefer a.free(pos);
    for (0..n + 1) |y| for (0..n + 1) |x| {
        const v = (y * (n + 1) + x) * 3;
        pos[v] = @floatFromInt(x);
        pos[v + 1] = @floatFromInt(y);
        pos[v + 2] = 0;
    };
    const idx = try a.alloc(u32, n * n * 6);
    var k: usize = 0;
    for (0..n) |y| for (0..n) |x| {
        const v: u32 = @intCast(y * (n + 1) + x);
        const w: u32 = @intCast(n + 1);
        for ([6]u32{ v, v + 1, v + w + 1, v, v + w + 1, v + w }) |i| {
            idx[k] = i;
            k += 1;
        }
    };
    return .{ .pos = pos, .idx = idx };
}

test "uvwrap: region labels split a surface into equal, compact blocks" {
    const a = testing.allocator;
    const m = try sheet(a, 100); // 20000 triangles
    defer a.free(m.pos);
    defer a.free(m.idx);
    const labels = try regionLabels(a, m.pos, m.idx, 4);
    defer a.free(labels);
    try testing.expectEqual(m.idx.len / 3, labels.len);
    var count = std.mem.zeroes([4]usize);
    var lo: [4][2]f32 = @splat(.{ 1e9, 1e9 });
    var hi: [4][2]f32 = @splat(.{ -1e9, -1e9 });
    for (labels, 0..) |l, f| {
        try testing.expect(l < 4);
        count[l] += 1;
        for (0..3) |c| for (0..2) |ax| {
            const v = m.pos[m.idx[f * 3 + c] * 3 + ax];
            lo[l][ax] = @min(lo[l][ax], v);
            hi[l][ax] = @max(hi[l][ax], v);
        };
    }
    for (0..4) |r| {
        try testing.expectEqual(@as(usize, 5000), count[r]);
        // A quadrant of the 100 x 100 sheet, not a strip across it.
        try testing.expect((hi[r][0] - lo[r][0]) * (hi[r][1] - lo[r][1]) < 0.35 * 100 * 100);
    }
}

test "uvwrap: a connected surface unwraps as one chart group per region, so every core has work" {
    const a = testing.allocator;
    const m = try sheet(a, 100);
    defer a.free(m.pos);
    defer a.free(m.idx);
    var plain = try parametrize(a, m.pos, m.idx);
    defer plain.deinit(a);
    var split = try parametrizeParallel(a, m.pos, m.idx);
    defer split.deinit(a);
    try testing.expectEqual(@as(u32, 1), plain.chart_count);
    // 20000 triangles at REGION_FACES each: two regions, which no chart may span.
    try testing.expect(split.chart_count >= 2);
    try testing.expectEqual(m.idx.len, split.indices.len);
    for (split.vmapping) |orig| try testing.expect(orig < m.pos.len / 3);
    for (split.indices) |idx| try testing.expect(idx < split.vmapping.len);
    for (split.uvs) |v| try testing.expect(v >= 0.0 and v <= 1.0);
}

test "uvwrap: a mesh below one region's worth of faces is unwrapped exactly as the default call does" {
    const a = testing.allocator;
    var plain = try parametrize(a, &cube_positions, &cube_indices);
    defer plain.deinit(a);
    var split = try parametrizeParallel(a, &cube_positions, &cube_indices);
    defer split.deinit(a);
    try testing.expectEqualSlices(u32, plain.vmapping, split.vmapping);
    try testing.expectEqualSlices(f32, plain.uvs, split.uvs);
}

test "uvwrap: deterministic across runs" {
    const a = testing.allocator;
    var ua = try parametrize(a, &cube_positions, &cube_indices);
    defer ua.deinit(a);
    var ub = try parametrize(a, &cube_positions, &cube_indices);
    defer ub.deinit(a);
    try testing.expectEqualSlices(u32, ua.vmapping, ub.vmapping);
    try testing.expectEqualSlices(u32, ua.indices, ub.indices);
    try testing.expectEqualSlices(f32, ua.uvs, ub.uvs);
}

test "uvwrap: empty/invalid input errors instead of crashing" {
    const a = testing.allocator;
    try testing.expectError(error.UnwrapFailed, parametrize(a, &[_]f32{}, &[_]u32{}));
    try testing.expectError(error.UnwrapFailed, parametrize(a, &[_]f32{ 0, 0, 0 }, &[_]u32{ 0, 0 }));
}
