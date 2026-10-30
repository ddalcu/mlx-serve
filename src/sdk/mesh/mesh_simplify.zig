//! Quadric mesh decimation via vendored Fast-Quadric-Mesh-Simplification
//! (lib/fqms, MIT), the code trimesh's `simplify_quadric_decimation` runs
//! through fast_simplification. The Hunyuan3D-2.1 paint reference decimates
//! the shape mesh to 40k faces before its xatlas unwrap (`remesh_mesh`).
//! Pure CPU, no MLX.

const std = @import("std");
const mc = @import("marching_cubes.zig");

extern fn fqms_simplify(positions: [*]const f32, vertex_count: u32, indices: [*]const u32, index_count: u32, target_faces: u32) ?*anyopaque;
extern fn fqms_vertex_count(r: *const anyopaque) u32;
extern fn fqms_index_count(r: *const anyopaque) u32;
extern fn fqms_positions(r: *const anyopaque) [*]const f32;
extern fn fqms_indices(r: *const anyopaque) [*]const u32;
extern fn fqms_free(r: *anyopaque) void;

/// Decimate `mesh` to at most `target_faces` triangles; a copy when it is
/// already that small. Normals of a decimated mesh are area-weighted face
/// normals (CCW = outward is preserved: the simplifier rejects flips).
pub fn decimate(alloc: std.mem.Allocator, mesh: *const mc.Mesh, target_faces: u32) !mc.Mesh {
    if (mesh.indices.len / 3 <= target_faces) {
        const vertices = try alloc.dupe(f32, mesh.vertices);
        errdefer alloc.free(vertices);
        const normals = try alloc.dupe(f32, mesh.normals);
        errdefer alloc.free(normals);
        return .{ .vertices = vertices, .normals = normals, .indices = try alloc.dupe(u32, mesh.indices) };
    }
    const r = fqms_simplify(mesh.vertices.ptr, @intCast(mesh.vertices.len / 3), mesh.indices.ptr, @intCast(mesh.indices.len), target_faces) orelse
        return error.DecimateFailed;
    defer fqms_free(r);
    const nv: usize = fqms_vertex_count(r);
    const vertices = try alloc.dupe(f32, fqms_positions(r)[0 .. nv * 3]);
    errdefer alloc.free(vertices);
    const indices = try alloc.dupe(u32, fqms_indices(r)[0..fqms_index_count(r)]);
    errdefer alloc.free(indices);
    return .{ .vertices = vertices, .normals = try faceWeightedNormals(alloc, vertices, indices), .indices = indices };
}

fn faceWeightedNormals(alloc: std.mem.Allocator, verts: []const f32, indices: []const u32) ![]f32 {
    const n = try alloc.alloc(f32, verts.len);
    @memset(n, 0);
    var t: usize = 0;
    while (t < indices.len) : (t += 3) {
        const a = indices[t] * 3;
        const b = indices[t + 1] * 3;
        const c = indices[t + 2] * 3;
        const e1 = [3]f32{ verts[b] - verts[a], verts[b + 1] - verts[a + 1], verts[b + 2] - verts[a + 2] };
        const e2 = [3]f32{ verts[c] - verts[a], verts[c + 1] - verts[a + 1], verts[c + 2] - verts[a + 2] };
        const fnrm = [3]f32{ e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
        for ([_]u32{ a, b, c }) |v| for (0..3) |k| {
            n[v + k] += fnrm[k];
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

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic (the vendored lib is deterministic single-threaded).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

fn sphereMesh(a: std.mem.Allocator, np: usize, radius: f32) !mc.Mesh {
    const grid = try a.alloc(f32, np * np * np);
    defer a.free(grid);
    const c: f32 = @as(f32, @floatFromInt(np - 1)) / 2.0;
    for (0..np) |x| for (0..np) |y| for (0..np) |z| {
        const dx = @as(f32, @floatFromInt(x)) - c;
        const dy = @as(f32, @floatFromInt(y)) - c;
        const dz = @as(f32, @floatFromInt(z)) - c;
        grid[(x * np + y) * np + z] = radius - @sqrt(dx * dx + dy * dy + dz * dz);
    };
    return mc.extract(a, grid, .{ np, np, np }, 0.0, .{ 1, 1, 1 }, .{ -c, -c, -c });
}

fn bounds(v: []const f32) [6]f32 {
    var b = [6]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    var i: usize = 0;
    while (i < v.len) : (i += 3) for (0..3) |k| {
        b[k] = @min(b[k], v[i + k]);
        b[3 + k] = @max(b[3 + k], v[i + k]);
    };
    return b;
}

test "decimate: a dense sphere comes down to the target with its shape and outward normals" {
    const a = testing.allocator;
    var src = try sphereMesh(a, 128, 50);
    defer src.deinit(a);
    try testing.expect(src.indices.len / 3 > 80_000);

    var out = try decimate(a, &src, 40_000);
    defer out.deinit(a);
    const faces = out.indices.len / 3;
    try testing.expect(faces <= 40_000 and faces > 30_000);
    try testing.expectEqual(out.vertices.len, out.normals.len);

    const bs = bounds(src.vertices);
    const bo = bounds(out.vertices);
    for (0..6) |k| try testing.expect(@abs(bo[k] - bs[k]) <= 0.01 * (bs[3 + k % 3] - bs[k % 3]));

    var t: usize = 0;
    while (t < out.indices.len) : (t += 3) {
        const ia = out.indices[t];
        const ib = out.indices[t + 1];
        const ic = out.indices[t + 2];
        try testing.expect(ia != ib and ib != ic and ia != ic);
        try testing.expect(ia * 3 < out.vertices.len and ib * 3 < out.vertices.len and ic * 3 < out.vertices.len);
        // CCW = outward on a sphere at the origin: the vertex normal points away from it.
        const p = out.vertices[ia * 3 ..][0..3];
        const n = out.normals[ia * 3 ..][0..3];
        try testing.expect(n[0] * p[0] + n[1] * p[1] + n[2] * p[2] > 0);
    }
}

test "decimate: a mesh at or under the target comes back byte-identical" {
    const a = testing.allocator;
    var src = try sphereMesh(a, 24, 8);
    defer src.deinit(a);
    var out = try decimate(a, &src, @intCast(src.indices.len / 3));
    defer out.deinit(a);
    try testing.expectEqualSlices(f32, src.vertices, out.vertices);
    try testing.expectEqualSlices(f32, src.normals, out.normals);
    try testing.expectEqualSlices(u32, src.indices, out.indices);
}
