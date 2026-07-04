//! UV unwrapping via vendored xatlas (lib/xatlas, MIT) for the Hunyuan3D-2.1
//! texture paint stage. Mirrors the Python `xatlas.parametrize(positions,
//! faces)` call the reference makes with ALL defaults (hy3dpaint
//! utils/uvwrap_utils.py): default ChartOptions/PackOptions, positions + u32
//! triangle indices only. UVs come back normalized to [0,1] by atlas
//! width/height; `vmapping[i]` names the ORIGINAL vertex each (possibly
//! seam-duplicated) output vertex came from.

const std = @import("std");

extern fn uvw_parametrize(positions: [*]const f32, vertex_count: u32, indices: [*]const u32, index_count: u32) ?*anyopaque;
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
    if (positions.len == 0 or positions.len % 3 != 0 or indices.len < 3 or indices.len % 3 != 0)
        return error.UnwrapFailed;
    const r = uvw_parametrize(positions.ptr, @intCast(positions.len / 3), indices.ptr, @intCast(indices.len)) orelse
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
