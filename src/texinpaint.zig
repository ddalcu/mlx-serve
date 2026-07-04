//! Texture-atlas inpainting for the Hunyuan3D-2.1 paint (bake) stage — pure Zig, std only, zero MLX.
//!
//! Port of hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp (the "smooth" method,
//! `meshVerticeInpaint_smooth`) plus the hole-fill that follows it in MeshRender.py::uv_inpaint.
//!
//! Reference call site (MeshRender.py::uv_inpaint, method="NS"):
//!     texture_np, mask = meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)   # method="smooth"
//!     texture_np = cv2.inpaint((texture_np*255).astype(uint8), 255 - mask, 3, cv2.INPAINT_NS)
//! meshVerticeInpaint defaults to method="smooth" and that is the only path uv_inpaint exercises, so
//! the "forward" and "meshVerticeColor" variants in the cpp are intentionally NOT ported.
//!
//! Two stages:
//!   1. Vertex-graph colour propagation (`meshVerticeInpaintSmooth`): seed each mesh vertex from the
//!      texel its UV lands on (mask>0), then diffuse colour along MESH EDGES to un-seeded vertices as an
//!      inverse-squared-distance weighted average of already-coloured neighbours, using the reference's
//!      exact `smooth_count` pass discipline. Colours are written back only at the exact rounded texel
//!      of each coloured vertex's UV (a sparse fill — triangle interiors are left to stage 2).
//!   2. Hole fill for texels still unfilled after stage 1. The reference uses cv2.INPAINT_NS
//!      (Navier–Stokes). We replace it with ITERATIVE DIFFUSION FILL — Jacobi relaxation from the valid
//!      texels (Dirichlet boundary) until max per-texel delta < 1e-4 or 512 iterations. This is the
//!      SINGLE DELIBERATE, VISUAL-ONLY DEVIATION from the reference; the harmonic (Laplace) solution it
//!      converges to obeys the maximum principle, so it introduces no new colour extrema. Everything
//!      else matches the cpp exactly.
//!
//! UV → texel convention (pinned from calculateUVCoordinates in the cpp):
//!     col = round(u        * (width  - 1))      // "uv_v" in the cpp
//!     row = round((1 - v)  * (height - 1))      // "uv_u" in the cpp (note the internal v-flip)
//!     texel_linear_index = row * width + col    // row-major; no texel-centre half-pixel offset
//! Coordinates are clamped to [0, dim-1] before the integer cast; for in-range UVs this is a no-op and
//! identical to the reference (the cpp does not clamp — degenerate out-of-range UVs would index OOB there).
//!
//! Public bake API: `inpaintAtlas`. It assumes ONE shared vertex/uv index buffer (pos_idx == uv_idx),
//! i.e. the standard "one UV per vertex" unwrapped-atlas case the Hunyuan3D bake path produces; the
//! reference supports distinct pos_idx/uv_idx and `meshVerticeInpaintSmooth` keeps them as separate
//! parameters, so distinct-index meshes are one call away if ever needed.

const std = @import("std");

/// Max texture channels handled by the fixed-size accumulation buffers. The bake stage is RGB (3).
const MAX_CHANNEL: usize = 4;

fn vertexPos(positions: []const f32, idx: usize) [3]f32 {
    return .{ positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2] };
}

/// Inverse-squared-distance edge weight: (1 / max(||a-b||, 1e-4))^2  (calculateDistanceWeight).
fn distWeight(a: [3]f32, b: [3]f32) f32 {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    const d = @sqrt(dx * dx + dy * dy + dz * dz);
    const w = 1.0 / @max(d, 1e-4);
    return w * w;
}

/// Map a normalized coordinate to a texel index in [0, dim-1] via round() (ties away from zero, as C++).
fn coordToIndex(coord: f32, dim: usize) usize {
    const maxf: f32 = @floatFromInt(dim - 1);
    const r = @round(coord * maxf);
    const clamped = std.math.clamp(r, 0.0, maxf);
    return @intFromFloat(clamped);
}

/// UV index -> linear texel index (row-major). See the header for the exact convention.
fn uvToTexel(uvs: []const f32, uv_idx: usize, width: usize, height: usize) usize {
    const u = uvs[uv_idx * 2];
    const v = uvs[uv_idx * 2 + 1];
    const col = coordToIndex(u, width);
    const row = coordToIndex(1.0 - v, height);
    return row * width + col;
}

/// Stage 1: vertex-graph colour propagation, "smooth" mode. In-place on `texture` and `mask`.
/// `texture` is width*height*channel f32; `mask` is width*height u8 (>0 = valid), updated to 255 at
/// every coloured vertex's texel. `pos_idx`/`uv_idx` are faces*3 (usually the same slice).
pub fn meshVerticeInpaintSmooth(
    allocator: std.mem.Allocator,
    texture: []f32,
    mask: []u8,
    width: usize,
    height: usize,
    channel: usize,
    positions: []const f32,
    uvs: []const f32,
    pos_idx: []const u32,
    uv_idx: []const u32,
) !void {
    std.debug.assert(channel <= MAX_CHANNEL);
    const vtx_num = positions.len / 3;
    const num_faces = pos_idx.len / 3;

    // --- buildGraph: directed edges k -> (k+1)%3 within each triangle, over position-vertex indices.
    //     Duplicate edges are kept (the reference does not dedup; they intentionally re-weight neighbours).
    var graph = try allocator.alloc(std.ArrayList(u32), vtx_num);
    for (graph) |*g| g.* = .empty;
    defer {
        for (graph) |*g| g.deinit(allocator);
        allocator.free(graph);
    }
    for (0..num_faces) |i| {
        for (0..3) |k| {
            const from: usize = @intCast(pos_idx[i * 3 + k]);
            const to = pos_idx[i * 3 + (k + 1) % 3];
            try graph[from].append(allocator, to);
        }
    }

    // --- initializeVertexDataGeneric (smooth: float mask, value 1.0).
    const vtx_mask = try allocator.alloc(f32, vtx_num);
    defer allocator.free(vtx_mask);
    @memset(vtx_mask, 0);
    const vtx_color = try allocator.alloc(f32, vtx_num * channel);
    defer allocator.free(vtx_color);
    @memset(vtx_color, 0);

    var uncolored: std.ArrayList(u32) = .empty;
    defer uncolored.deinit(allocator);

    for (0..num_faces) |i| {
        for (0..3) |k| {
            const vuv: usize = @intCast(uv_idx[i * 3 + k]);
            const vidx: usize = @intCast(pos_idx[i * 3 + k]);
            const texel = uvToTexel(uvs, vuv, width, height);
            if (mask[texel] > 0) {
                vtx_mask[vidx] = 1.0;
                for (0..channel) |c| vtx_color[vidx * channel + c] = texture[texel * channel + c];
            } else {
                try uncolored.append(allocator, @intCast(vidx));
            }
        }
    }

    // --- performSmoothingAlgorithm<float>. is_colored = mask>0; set_colored = mask=1.0.
    var smooth_count: i32 = 2;
    var last_uncolored_count: usize = 0;
    while (smooth_count > 0) {
        var uncolored_count: usize = 0;
        for (uncolored.items) |raw| {
            const vidx: usize = @intCast(raw);
            var sum = [_]f32{0} ** MAX_CHANNEL;
            var total_weight: f32 = 0;
            const p0 = vertexPos(positions, vidx);
            for (graph[vidx].items) |craw| {
                const cidx: usize = @intCast(craw);
                if (vtx_mask[cidx] > 0) {
                    const p1 = vertexPos(positions, cidx);
                    const w = distWeight(p0, p1);
                    for (0..channel) |c| sum[c] += vtx_color[cidx * channel + c] * w;
                    total_weight += w;
                }
            }
            if (total_weight > 0.0) {
                for (0..channel) |c| vtx_color[vidx * channel + c] = sum[c] / total_weight;
                vtx_mask[vidx] = 1.0;
            } else {
                uncolored_count += 1;
            }
        }
        if (last_uncolored_count == uncolored_count) {
            smooth_count -= 1;
        } else {
            smooth_count += 1;
        }
        last_uncolored_count = uncolored_count;
    }

    // --- createOutputArrays: write coloured vertices' colours to their texels + mark mask 255.
    for (0..num_faces) |i| {
        for (0..3) |k| {
            const vidx: usize = @intCast(pos_idx[i * 3 + k]);
            if (vtx_mask[vidx] == 1.0) {
                const vuv: usize = @intCast(uv_idx[i * 3 + k]);
                const texel = uvToTexel(uvs, vuv, width, height);
                for (0..channel) |c| texture[texel * channel + c] = vtx_color[vidx * channel + c];
                mask[texel] = 255;
            }
        }
    }
}

/// Stage 2: iterative diffusion (Jacobi) hole fill. In-place on `texture` (width*height*channel f32).
/// Fills every texel where `valid[p]==0` from the valid ones (fixed Dirichlet boundary). Converges to
/// the harmonic extension → no new colour extrema (maximum principle). Deliberate cv2.INPAINT_NS
/// replacement (see header). No valid texels → returns unchanged.
pub fn diffuseFill(
    allocator: std.mem.Allocator,
    texture: []f32,
    valid: []const u8,
    width: usize,
    height: usize,
    channel: usize,
) !void {
    std.debug.assert(channel <= MAX_CHANNEL);
    const n = width * height;

    // Mean of the valid texels per channel — the initial guess for every unknown texel. It is within
    // [min,max] of the valid field, so seeding with it can never create an extremum, and for a constant
    // field it is already the exact answer.
    var mean = [_]f64{0} ** MAX_CHANNEL;
    var valid_count: usize = 0;
    for (0..n) |p| {
        if (valid[p] > 0) {
            valid_count += 1;
            for (0..channel) |c| mean[c] += texture[p * channel + c];
        }
    }
    if (valid_count == 0) return;
    const inv: f64 = 1.0 / @as(f64, @floatFromInt(valid_count));
    for (0..channel) |c| mean[c] *= inv;
    for (0..n) |p| {
        if (valid[p] == 0) {
            for (0..channel) |c| texture[p * channel + c] = @floatCast(mean[c]);
        }
    }

    const scratch = try allocator.dupe(f32, texture);
    defer allocator.free(scratch);

    var iter: usize = 0;
    while (iter < 512) : (iter += 1) {
        var max_delta: f32 = 0;
        var y: usize = 0;
        while (y < height) : (y += 1) {
            var x: usize = 0;
            while (x < width) : (x += 1) {
                const p = y * width + x;
                if (valid[p] != 0) continue;
                for (0..channel) |c| {
                    var s: f32 = 0;
                    var cnt: f32 = 0;
                    if (x > 0) {
                        s += texture[(p - 1) * channel + c];
                        cnt += 1;
                    }
                    if (x + 1 < width) {
                        s += texture[(p + 1) * channel + c];
                        cnt += 1;
                    }
                    if (y > 0) {
                        s += texture[(p - width) * channel + c];
                        cnt += 1;
                    }
                    if (y + 1 < height) {
                        s += texture[(p + width) * channel + c];
                        cnt += 1;
                    }
                    const cur = texture[p * channel + c];
                    const nv = if (cnt > 0) s / cnt else cur;
                    const d = @abs(nv - cur);
                    if (d > max_delta) max_delta = d;
                    scratch[p * channel + c] = nv;
                }
            }
        }
        for (0..n) |p| {
            if (valid[p] == 0) {
                for (0..channel) |c| texture[p * channel + c] = scratch[p * channel + c];
            }
        }
        if (max_delta < 1e-4) break;
    }
}

/// Public bake-stage entry: fill the RGB atlas in place. `mask` (>0 = valid) is read only; the internal
/// working mask (valid set after stage 1) drives the diffusion. `indices` is used as both pos_idx and
/// uv_idx (one UV per vertex — see header).
pub fn inpaintAtlas(
    allocator: std.mem.Allocator,
    texture: []f32,
    mask: []const u8,
    width: usize,
    height: usize,
    positions: []const f32,
    uvs: []const f32,
    indices: []const u32,
) !void {
    const channel: usize = 3;
    std.debug.assert(texture.len == width * height * channel);
    std.debug.assert(mask.len == width * height);

    const work_mask = try allocator.dupe(u8, mask);
    defer allocator.free(work_mask);

    try meshVerticeInpaintSmooth(allocator, texture, work_mask, width, height, channel, positions, uvs, indices, indices);
    try diffuseFill(allocator, texture, work_mask, width, height, channel);
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

/// Build an m×m grid mesh in the z=0 plane, UVs spanning [0,1]², triangulated into 2(m-1)² faces.
/// One UV per vertex (shared index buffer). Vertex (i,j) has linear index j*m + i.
const GridMesh = struct {
    positions: []f32,
    uvs: []f32,
    indices: []u32,
    m: usize,

    fn init(allocator: std.mem.Allocator, m: usize) !GridMesh {
        const vtx = m * m;
        const positions = try allocator.alloc(f32, vtx * 3);
        const uvs = try allocator.alloc(f32, vtx * 2);
        const denom: f32 = @floatFromInt(m - 1);
        for (0..m) |j| {
            for (0..m) |i| {
                const vi = j * m + i;
                const fi: f32 = @floatFromInt(i);
                const fj: f32 = @floatFromInt(j);
                positions[vi * 3 + 0] = fi; // planar, unit spacing -> uniform edge lengths
                positions[vi * 3 + 1] = fj;
                positions[vi * 3 + 2] = 0;
                uvs[vi * 2 + 0] = fi / denom; // u
                uvs[vi * 2 + 1] = fj / denom; // v
            }
        }
        var idx: std.ArrayList(u32) = .empty;
        for (0..m - 1) |j| {
            for (0..m - 1) |i| {
                const a: u32 = @intCast(j * m + i);
                const b: u32 = @intCast(j * m + i + 1);
                const c: u32 = @intCast((j + 1) * m + i + 1);
                const d: u32 = @intCast((j + 1) * m + i);
                try idx.appendSlice(allocator, &.{ a, b, c });
                try idx.appendSlice(allocator, &.{ a, c, d });
            }
        }
        return .{ .positions = positions, .uvs = uvs, .indices = try idx.toOwnedSlice(allocator), .m = m };
    }

    fn deinit(self: *GridMesh, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.uvs);
        allocator.free(self.indices);
    }

    /// The texel a vertex's UV lands on (matches the module's convention).
    fn vertexTexel(self: GridMesh, vi: usize, width: usize, height: usize) usize {
        return uvToTexel(self.uvs, vi, width, height);
    }
};

test "vertex propagation: checkerboard seed fills all mesh texels within seed [min,max] (maximum principle)" {
    const a = testing.allocator;
    const m = 5;
    var mesh = try GridMesh.init(a, m);
    defer mesh.deinit(a);

    const w = 16;
    const h = 16;
    const ch = 3;
    const texture = try a.alloc(f32, w * h * ch);
    defer a.free(texture);
    @memset(texture, 0);
    const mask = try a.alloc(u8, w * h);
    defer a.free(mask);
    @memset(mask, 0);

    // Checkerboard seeds: (i+j) even -> valid with a spatially-varying colour in [0.2, 0.8].
    var min_seed: f32 = 1e9;
    var max_seed: f32 = -1e9;
    for (0..m) |j| {
        for (0..m) |i| {
            if ((i + j) % 2 != 0) continue;
            const vi = j * m + i;
            const texel = mesh.vertexTexel(vi, w, h);
            const col = 0.2 + 0.6 * @as(f32, @floatFromInt(i + j)) / @as(f32, @floatFromInt(2 * (m - 1)));
            texture[texel * ch + 0] = col;
            texture[texel * ch + 1] = col * 0.5;
            texture[texel * ch + 2] = 1.0 - col;
            mask[texel] = 255;
            min_seed = @min(min_seed, @min(col, @min(col * 0.5, 1.0 - col)));
            max_seed = @max(max_seed, @max(col, @max(col * 0.5, 1.0 - col)));
        }
    }

    try meshVerticeInpaintSmooth(a, texture, mask, w, h, ch, mesh.positions, mesh.uvs, mesh.indices, mesh.indices);

    // Every mesh-covered texel is now valid and its value lies within the seed range (weighted averages).
    for (0..m * m) |vi| {
        const texel = mesh.vertexTexel(vi, w, h);
        try testing.expect(mask[texel] == 255);
        for (0..ch) |c| {
            const val = texture[texel * ch + c];
            try testing.expect(val >= min_seed - 1e-4);
            try testing.expect(val <= max_seed + 1e-4);
        }
    }
}

test "vertex propagation: a uniform seed colour propagates exactly (no drift)" {
    const a = testing.allocator;
    const m = 6;
    var mesh = try GridMesh.init(a, m);
    defer mesh.deinit(a);

    const w = 24;
    const h = 24;
    const ch = 3;
    const texture = try a.alloc(f32, w * h * ch);
    defer a.free(texture);
    @memset(texture, 0);
    const mask = try a.alloc(u8, w * h);
    defer a.free(mask);
    @memset(mask, 0);

    const C = [3]f32{ 0.37, 0.62, 0.11 };
    for (0..m) |j| {
        for (0..m) |i| {
            if ((i + j) % 2 != 0) continue;
            const vi = j * m + i;
            const texel = mesh.vertexTexel(vi, w, h);
            for (0..ch) |c| texture[texel * ch + c] = C[c];
            mask[texel] = 255;
        }
    }

    try meshVerticeInpaintSmooth(a, texture, mask, w, h, ch, mesh.positions, mesh.uvs, mesh.indices, mesh.indices);

    for (0..m * m) |vi| {
        const texel = mesh.vertexTexel(vi, w, h);
        try testing.expect(mask[texel] == 255);
        for (0..ch) |c| try testing.expectApproxEqAbs(C[c], texture[texel * ch + c], 1e-5);
    }
}

test "diffusion hole fill: isolated hole in a constant field fills to that constant" {
    const a = testing.allocator;
    const w = 10;
    const h = 10;
    const ch = 3;
    const K = [3]f32{ 0.5, 0.25, 0.9 };
    const texture = try a.alloc(f32, w * h * ch);
    defer a.free(texture);
    const valid = try a.alloc(u8, w * h);
    defer a.free(valid);
    for (0..w * h) |p| {
        valid[p] = 1;
        for (0..ch) |c| texture[p * ch + c] = K[c];
    }
    // Punch a 3×3 hole in the interior (mask 0, and scribble garbage so the fill must recompute it).
    for (3..6) |y| {
        for (3..6) |x| {
            const p = y * w + x;
            valid[p] = 0;
            for (0..ch) |c| texture[p * ch + c] = -7.0;
        }
    }

    try diffuseFill(a, texture, valid, w, h, ch);

    for (3..6) |y| {
        for (3..6) |x| {
            const p = y * w + x;
            for (0..ch) |c| try testing.expectApproxEqAbs(K[c], texture[p * ch + c], 1e-3);
        }
    }
}

test "diffusion hole fill: linear gradient fills smoothly with no new extrema" {
    const a = testing.allocator;
    const w = 8;
    const h = 8;
    const ch = 1;
    const texture = try a.alloc(f32, w * h * ch);
    defer a.free(texture);
    const valid = try a.alloc(u8, w * h);
    defer a.free(valid);
    const denom: f32 = @floatFromInt(w - 1);
    for (0..h) |y| {
        for (0..w) |x| {
            const p = y * w + x;
            valid[p] = 1;
            texture[p] = @as(f32, @floatFromInt(x)) / denom; // horizontal gradient in [0,1]
        }
    }
    // Full-height 2-wide vertical hole at columns 3,4 (garbage-fill first).
    for (0..h) |y| {
        for (3..5) |x| {
            const p = y * w + x;
            valid[p] = 0;
            texture[p] = 42.0;
        }
    }

    try diffuseFill(a, texture, valid, w, h, ch);

    for (0..h) |y| {
        for (3..5) |x| {
            const p = y * w + x;
            const v = texture[p];
            // No new extrema: stays within the valid field's [0,1].
            try testing.expect(v >= -1e-4);
            try testing.expect(v <= 1.0 + 1e-4);
            // Harmonic extension of a linear field is that field: value ~= x/(w-1).
            const expected = @as(f32, @floatFromInt(x)) / denom;
            try testing.expectApproxEqAbs(expected, v, 2e-2);
        }
    }
}

test "empty mask (nothing valid): inpaintAtlas leaves the texture unchanged" {
    const a = testing.allocator;
    const m = 4;
    var mesh = try GridMesh.init(a, m);
    defer mesh.deinit(a);

    const w = 12;
    const h = 12;
    const ch = 3;
    const texture = try a.alloc(f32, w * h * ch);
    defer a.free(texture);
    for (texture, 0..) |*t, k| t.* = @as(f32, @floatFromInt(k % 7)) / 7.0;
    const before = try a.dupe(f32, texture);
    defer a.free(before);

    const mask = try a.alloc(u8, w * h);
    defer a.free(mask);
    @memset(mask, 0);

    try inpaintAtlas(a, texture, mask, w, h, mesh.positions, mesh.uvs, mesh.indices);

    try testing.expectEqualSlices(f32, before, texture);
}

test "determinism: two inpaintAtlas runs are byte-identical" {
    const a = testing.allocator;
    const m = 5;
    var mesh = try GridMesh.init(a, m);
    defer mesh.deinit(a);

    const w = 16;
    const h = 16;
    const ch = 3;

    const base = try a.alloc(f32, w * h * ch);
    defer a.free(base);
    @memset(base, 0);
    const mask = try a.alloc(u8, w * h);
    defer a.free(mask);
    @memset(mask, 0);
    for (0..m) |j| {
        for (0..m) |i| {
            if ((i + j) % 3 != 0) continue;
            const vi = j * m + i;
            const texel = mesh.vertexTexel(vi, w, h);
            const col = 0.1 + 0.8 * @as(f32, @floatFromInt(i * m + j)) / @as(f32, @floatFromInt(m * m));
            for (0..ch) |c| base[texel * ch + c] = col;
            mask[texel] = 255;
        }
    }

    const run1 = try a.dupe(f32, base);
    defer a.free(run1);
    const run2 = try a.dupe(f32, base);
    defer a.free(run2);

    try inpaintAtlas(a, run1, mask, w, h, mesh.positions, mesh.uvs, mesh.indices);
    try inpaintAtlas(a, run2, mask, w, h, mesh.positions, mesh.uvs, mesh.indices);

    try testing.expectEqualSlices(f32, run1, run2);
}
