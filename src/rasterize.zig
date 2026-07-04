//! Pure-Zig port of the Hunyuan3D-2.1 reference CPU rasterizer.
//!
//! Ported VERBATIM from the reference custom_rasterizer kernel:
//!   reference/hy3dpaint/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp
//!     - calculateSignedArea2            (rasterizer.h:12-14)   -> signedArea2
//!     - calculateBarycentricCoordinate  (rasterizer.h:16-35)   -> barycentric
//!     - isBarycentricCoordInBounds      (rasterizer.h:37-41)   -> inBounds
//!     - rasterizeTriangleCPU            (rasterizer.cpp:3-39)  -> rasterizeTriangle
//!     - rasterizeImagecoordsKernelCPU   (rasterizer.cpp:81-92) -> per-face screen projection in rasterize()
//!     - barycentricFromImgcoordCPU      (rasterizer.cpp:41-79) -> barycentricFromImgcoord
//!     - rasterize_image_cpu             (rasterizer.cpp:94-123)-> rasterize (zbuffer build + bary decode)
//! and the Python callers that pin the input convention:
//!   custom_rasterizer/custom_rasterizer/render.py  (rasterize / interpolate)
//!   DifferentiableRenderer/MeshRender.py           (raster_rasterize / raster_interpolate)
//!
//! Input convention (pinned from render.py `rasterize(pos, tri, resolution, ...)`):
//!   - V ("pos") is CLIP space [N,4]: the kernel does the perspective divide itself
//!     (every access divides x/y/z by the 4th component w). Not NDC.
//!   - F ("tri") is [M,3] int vertex indices.
//!   - resolution is [height, width]; the kernel is called with width=resolution[1],
//!     height=resolution[0].
//!   - The wrapper always passes clamp_depth=zeros(0), use_depth_prior=0, so the CPU
//!     depth-prior branch is dead (d == null): depth_thres is always 0.
//!   - Screen map: sx = (x/w*0.5 + 0.5)*(width-1) + 0.5 ; sy = (0.5 + 0.5*y/w)*(height-1) + 0.5.
//!     No y-flip. Pixel centers at +0.5. sz = z/w*0.49999 + 0.5.
//!   - Face ids are stored +1; 0 means background.
//!
//! Float/double discipline is load-bearing (matches the C++ exactly): signed areas and
//! most products are float (f32); the reciprocal `1.0 / area` and `1.0 - beta - gamma`
//! use a DOUBLE 1.0 literal (f64 intermediate, rounded back to f32), while the
//! perspective renormalizer `1.0f / (b0+b1+b2)` uses a FLOAT literal (pure f32).

const std = @import("std");

/// #define MAXINT 2147483647  (INT_MAX). The z-buffer token uses it as the radix that
/// separates the quantized depth (high part) from the face id+1 (low part), and its
/// square appears in the empty-buffer sentinel.
const MAXINT: u64 = 2147483647;

/// Empty z-buffer value: (INT64)MAXINT*MAXINT + (MAXINT-1)   (rasterizer.cpp:102-103).
/// Decodes to face id sentinel (MAXINT-1) => background.
const Z_EMPTY: u64 = MAXINT * MAXINT + (MAXINT - 1);

pub const Raster = struct {
    /// H*W row-major. 0 = background, else face index + 1.
    face_id: []u32,
    /// H*W*3 row-major, perspective-corrected barycentrics (sum to 1 on covered pixels).
    bary: []f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Raster) void {
        self.allocator.free(self.face_id);
        self.allocator.free(self.bary);
    }
};

// calculateSignedArea2 (rasterizer.h:12-14) — all f32.
fn signedArea2(a: [2]f32, b: [2]f32, c: [2]f32) f32 {
    return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1]);
}

// calculateBarycentricCoordinate (rasterizer.h:16-35). Returns {alpha, beta, gamma}
// paired with {a, b, c}. Degenerate (area==0) -> {-1,-1,-1}.
fn barycentric(a: [2]f32, b: [2]f32, c: [2]f32, p: [2]f32) [3]f32 {
    const beta_tri = signedArea2(a, p, c);
    const gamma_tri = signedArea2(a, b, p);
    const area = signedArea2(a, b, c);
    if (area == 0) return .{ -1.0, -1.0, -1.0 };
    // float tri_inv = 1.0 / area;  -> DOUBLE 1.0 literal: f64 divide, round to f32.
    const tri_inv: f32 = @floatCast(1.0 / @as(f64, area));
    const beta = beta_tri * tri_inv;
    const gamma = gamma_tri * tri_inv;
    // float alpha = 1.0 - beta - gamma; -> DOUBLE 1.0: f64 intermediate, round to f32.
    const alpha: f32 = @floatCast(1.0 - @as(f64, beta) - @as(f64, gamma));
    return .{ alpha, beta, gamma };
}

// isBarycentricCoordInBounds (rasterizer.h:37-41).
fn inBounds(bc: [3]f32) bool {
    return bc[0] >= 0.0 and bc[0] <= 1.0 and
        bc[1] >= 0.0 and bc[1] <= 1.0 and
        bc[2] >= 0.0 and bc[2] <= 1.0;
}

// Screen-space projection of one clip-space component (rasterizer.cpp:61-63,87-89).
fn screenX(vx: f32, vw: f32, width: u32) f32 {
    const wm1: f32 = @floatFromInt(@as(i64, width) - 1);
    return (vx / vw * 0.5 + 0.5) * wm1 + 0.5;
}
fn screenY(vy: f32, vw: f32, height: u32) f32 {
    const hm1: f32 = @floatFromInt(@as(i64, height) - 1);
    return (0.5 + 0.5 * vy / vw) * hm1 + 0.5;
}
fn screenZ(vz: f32, vw: f32) f32 {
    return vz / vw * 0.49999 + 0.5;
}

fn vert4(v: []const f32, i: u32) [4]f32 {
    const base = @as(usize, i) * 4;
    return .{ v[base], v[base + 1], v[base + 2], v[base + 3] };
}

// C `int x = (float)f;` truncation toward zero. Guarded against NaN / far-out-of-range
// (degenerate projections) so we never hit Zig's @intFromFloat safety trap; the real
// values here (pixel bounds, z_quantize in [0,262144]) are always tiny.
fn truncToI32(x: f32) i32 {
    if (std.math.isNan(x)) return 0;
    if (x >= 2_000_000_000.0) return std.math.maxInt(i32);
    if (x <= -2_000_000_000.0) return std.math.minInt(i32);
    return @intFromFloat(x);
}

// rasterizeTriangleCPU (rasterizer.cpp:3-39), depth-prior branch removed (d == null =>
// depth_thres == 0). vt* are screen-space {sx, sy, sz}. Writes min-depth tokens into
// zbuffer. The bounding-box iteration is clamped to the viewport — this is
// output-identical to the reference (it `continue`s off-viewport pixels) but prevents an
// unbounded spin if a projection produced a huge finite bbox.
fn rasterizeTriangle(idx: usize, vt0: [3]f32, vt1: [3]f32, vt2: [3]f32, width: u32, height: u32, zbuffer: []u64) void {
    const w_i: i32 = @intCast(width);
    const h_i: i32 = @intCast(height);

    const x_min = @min(vt0[0], @min(vt1[0], vt2[0]));
    const x_max = @max(vt0[0], @max(vt1[0], vt2[0]));
    const y_min = @min(vt0[1], @min(vt1[1], vt2[1]));
    const y_max = @max(vt0[1], @max(vt1[1], vt2[1]));

    var px: i32 = @max(0, truncToI32(x_min));
    while (@as(f32, @floatFromInt(px)) < x_max + 1.0 and px < w_i) : (px += 1) {
        if (px < 0 or px >= w_i) continue;
        var py: i32 = @max(0, truncToI32(y_min));
        while (@as(f32, @floatFromInt(py)) < y_max + 1.0 and py < h_i) : (py += 1) {
            if (py < 0 or py >= h_i) continue;

            const vt = [2]f32{ @as(f32, @floatFromInt(px)) + 0.5, @as(f32, @floatFromInt(py)) + 0.5 };
            const bc = barycentric(.{ vt0[0], vt0[1] }, .{ vt1[0], vt1[1] }, .{ vt2[0], vt2[1] }, vt);
            if (!inBounds(bc)) continue;

            const pixel: usize = @intCast(py * w_i + px);
            const depth = bc[0] * vt0[2] + bc[1] * vt1[2] + bc[2] * vt2[2];
            const depth_thres: f32 = 0; // d == null

            // int z_quantize = depth * (2<<17);  (2<<17 == 262144)
            const z_quantize = truncToI32(depth * 262144.0);
            // INT64 token = (INT64)z_quantize * MAXINT + (INT64)(idx+1);  (unsigned wrap)
            const z_u: u64 = @bitCast(@as(i64, z_quantize));
            const token: u64 = z_u *% MAXINT +% (@as(u64, @intCast(idx)) + 1);

            if (depth < depth_thres) continue;
            zbuffer[pixel] = @min(zbuffer[pixel], token);
        }
    }
}

// barycentricFromImgcoordCPU (rasterizer.cpp:41-79). Decodes one pixel of the z-buffer
// into a face id (+1) and perspective-corrected barycentrics.
fn barycentricFromImgcoord(
    v_clip: []const f32,
    indices: []const u32,
    findices: []u32,
    zbuffer: []const u64,
    width: u32,
    height: u32,
    bary_map: []f32,
    pix: usize,
) void {
    const f_raw = zbuffer[pix] % MAXINT;
    if (f_raw == MAXINT - 1) {
        findices[pix] = 0;
        bary_map[pix * 3] = 0;
        bary_map[pix * 3 + 1] = 0;
        bary_map[pix * 3 + 2] = 0;
        return;
    }
    findices[pix] = @intCast(f_raw);
    var bc = [3]f32{ 0, 0, 0 };
    // Reference does `f -= 1; if (f >= 0)` on an UNSIGNED f (always true here since the
    // background sentinel MAXINT-1 was already handled and any real token has f_raw>=1).
    if (f_raw >= 1) {
        const f: usize = @intCast(f_raw - 1);
        const vt = [2]f32{
            @as(f32, @floatFromInt(pix % width)) + 0.5,
            @as(f32, @floatFromInt(pix / width)) + 0.5,
        };
        const v0 = vert4(v_clip, indices[f * 3]);
        const v1 = vert4(v_clip, indices[f * 3 + 1]);
        const v2 = vert4(v_clip, indices[f * 3 + 2]);

        const s0 = [2]f32{ screenX(v0[0], v0[3], width), screenY(v0[1], v0[3], height) };
        const s1 = [2]f32{ screenX(v1[0], v1[3], width), screenY(v1[1], v1[3], height) };
        const s2 = [2]f32{ screenX(v2[0], v2[3], width), screenY(v2[1], v2[3], height) };

        bc = barycentric(s0, s1, s2, vt);

        // Perspective correction: divide by w, renormalize to sum 1.
        bc[0] = bc[0] / v0[3];
        bc[1] = bc[1] / v1[3];
        bc[2] = bc[2] / v2[3];
        // float w = 1.0f / (...)  -> FLOAT literal: pure f32.
        const wn: f32 = 1.0 / (bc[0] + bc[1] + bc[2]);
        bc[0] *= wn;
        bc[1] *= wn;
        bc[2] *= wn;
    }
    bary_map[pix * 3] = bc[0];
    bary_map[pix * 3 + 1] = bc[1];
    bary_map[pix * 3 + 2] = bc[2];
}

/// Rasterize a triangle mesh. `v_clip` is N*4 clip-space vertices (perspective divide is
/// done internally); `indices` is M*3 vertex indices. Returns per-pixel face id (+1,
/// 0=background) and perspective-corrected barycentrics.
pub fn rasterize(
    allocator: std.mem.Allocator,
    v_clip: []const f32,
    indices: []const u32,
    width: u32,
    height: u32,
) !Raster {
    const num_pixels = @as(usize, width) * @as(usize, height);
    const num_faces = indices.len / 3;

    const zbuffer = try allocator.alloc(u64, num_pixels);
    defer allocator.free(zbuffer);
    @memset(zbuffer, Z_EMPTY);

    // rasterizeImagecoordsKernelCPU over every face (use_depth_prior == 0 branch).
    var f: usize = 0;
    while (f < num_faces) : (f += 1) {
        const v0 = vert4(v_clip, indices[f * 3]);
        const v1 = vert4(v_clip, indices[f * 3 + 1]);
        const v2 = vert4(v_clip, indices[f * 3 + 2]);
        const vt0 = [3]f32{ screenX(v0[0], v0[3], width), screenY(v0[1], v0[3], height), screenZ(v0[2], v0[3]) };
        const vt1 = [3]f32{ screenX(v1[0], v1[3], width), screenY(v1[1], v1[3], height), screenZ(v1[2], v1[3]) };
        const vt2 = [3]f32{ screenX(v2[0], v2[3], width), screenY(v2[1], v2[3], height), screenZ(v2[2], v2[3]) };
        rasterizeTriangle(f, vt0, vt1, vt2, width, height, zbuffer);
    }

    const face_id = try allocator.alloc(u32, num_pixels);
    errdefer allocator.free(face_id);
    const bary = try allocator.alloc(f32, num_pixels * 3);
    errdefer allocator.free(bary);

    var pix: usize = 0;
    while (pix < num_pixels) : (pix += 1) {
        barycentricFromImgcoord(v_clip, indices, face_id, zbuffer, width, height, bary, pix);
    }

    return .{ .face_id = face_id, .bary = bary, .allocator = allocator };
}

/// Per-pixel barycentric interpolation of vertex attributes (H*W*C row-major).
/// `attrs` is N*C vertex attributes. For covered pixels: sum_i bary[i]*attr[vert_i].
/// For background pixels (face_id == 0): `background`. The reference `interpolate`
/// (render.py) always yields 0 at background (its barycentrics are 0 there), so
/// background=0 reproduces it exactly; the parameter generalizes that.
pub fn interpolate(
    allocator: std.mem.Allocator,
    attrs: []const f32,
    channels: usize,
    raster: *const Raster,
    indices: []const u32,
    background: f32,
) ![]f32 {
    const num_pixels = raster.face_id.len;
    const out = try allocator.alloc(f32, num_pixels * channels);
    errdefer allocator.free(out);

    var pix: usize = 0;
    while (pix < num_pixels) : (pix += 1) {
        const fid = raster.face_id[pix];
        if (fid == 0) {
            var c: usize = 0;
            while (c < channels) : (c += 1) out[pix * channels + c] = background;
            continue;
        }
        const f = @as(usize, fid - 1);
        const vi0 = @as(usize, indices[f * 3]);
        const vi1 = @as(usize, indices[f * 3 + 1]);
        const vi2 = @as(usize, indices[f * 3 + 2]);
        const b0 = raster.bary[pix * 3];
        const b1 = raster.bary[pix * 3 + 1];
        const b2 = raster.bary[pix * 3 + 2];
        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const a0 = attrs[vi0 * channels + c];
            const a1 = attrs[vi1 * channels + c];
            const a2 = attrs[vi2 * channels + c];
            out[pix * channels + c] = b0 * a0 + b1 * a1 + b2 * a2;
        }
    }
    return out;
}

// =============================================================================
// Camera math — verbatim port of DifferentiableRenderer/camera_utils.py.
//
//   get_mv_matrix                        (camera_utils.py:34-70)  -> getMvMatrix
//   get_orthographic_projection_matrix   (camera_utils.py:73-95)  -> getOrthographicProjection
//   get_perspective_projection_matrix    (camera_utils.py:98-107) -> getPerspectiveProjection
//
// Transform chain (MeshRender.py::_create_view_state):
//   pos_camera = mv  @ [x,y,z,1]      (transform_pos(r_mv, vtx, keepdim=True))
//   pos_clip   = proj @ pos_camera    (transform_pos(proj, pos_camera))
// i.e. clip = proj @ mv @ [x,y,z,1]; the vertex matmuls run in float32 (torch), and
// pos_clip is exactly the CLIP-space [N,4] input that `rasterize` expects.
//
// PRECISION: the reference builds the matrices in numpy float64 (math.cos/sin, np.cross,
// np.linalg.norm) and only casts the FINAL 4x4 to float32 (`.astype(np.float32)`). We
// mirror that: all matrix math is f64, entries rounded to f32 at store time. The
// per-vertex transforms (transformVec4 / projectToClip) run in f32, matching torch.
// =============================================================================

/// Row-major 4x4 matrix: m[row][col].
pub const Mat4 = [4][4]f32;

fn dot3(a: [3]f64, b: [3]f64) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
fn cross3(a: [3]f64, b: [3]f64) [3]f64 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}
fn normalize3(v: [3]f64) [3]f64 {
    const n = @sqrt(dot3(v, v));
    return .{ v[0] / n, v[1] / n, v[2] / n };
}

/// World-to-camera (view) matrix. `center = {0,0,0}` reproduces the reference's `None`.
/// elev/azim in degrees. (camera_utils.py:34-70.)
pub fn getMvMatrix(elev_in: f64, azim_in: f64, camera_distance: f64, center: [3]f64) Mat4 {
    const elev = -elev_in;
    const azim = azim_in + 90.0;
    const elev_rad = elev * std.math.pi / 180.0;
    const azim_rad = azim * std.math.pi / 180.0;

    const cam = [3]f64{
        camera_distance * @cos(elev_rad) * @cos(azim_rad),
        camera_distance * @cos(elev_rad) * @sin(azim_rad),
        camera_distance * @sin(elev_rad),
    };

    const lookat = normalize3(.{ center[0] - cam[0], center[1] - cam[1], center[2] - cam[2] });
    const up0 = [3]f64{ 0, 0, 1.0 };
    const right = normalize3(cross3(lookat, up0));
    const up = normalize3(cross3(right, lookat));
    const neg_lookat = [3]f64{ -lookat[0], -lookat[1], -lookat[2] };

    // w2c rotation rows are [right; up; -lookat]; translation = -R^T @ cam.
    const t = [3]f64{ -dot3(right, cam), -dot3(up, cam), -dot3(neg_lookat, cam) };

    return .{
        .{ toF32(right[0]), toF32(right[1]), toF32(right[2]), toF32(t[0]) },
        .{ toF32(up[0]), toF32(up[1]), toF32(up[2]), toF32(t[1]) },
        .{ toF32(neg_lookat[0]), toF32(neg_lookat[1]), toF32(neg_lookat[2]), toF32(t[2]) },
        .{ 0, 0, 0, 1 },
    };
}

/// Orthographic projection (camera_utils.py:73-95).
pub fn getOrthographicProjection(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) Mat4 {
    var m: Mat4 = .{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
    m[0][0] = toF32(2.0 / (right - left));
    m[1][1] = toF32(2.0 / (top - bottom));
    m[2][2] = toF32(-2.0 / (far - near));
    m[0][3] = toF32(-(right + left) / (right - left));
    m[1][3] = toF32(-(top + bottom) / (top - bottom));
    m[2][3] = toF32(-(far + near) / (far - near));
    m[3][3] = 1.0; // from np.eye(4), not overwritten
    return m;
}

/// Perspective projection (camera_utils.py:98-107). fovy in degrees, aspect = W/H.
pub fn getPerspectiveProjection(fovy_deg: f64, aspect_wh: f64, near: f64, far: f64) Mat4 {
    const fovy_rad = fovy_deg * std.math.pi / 180.0;
    const th = @tan(fovy_rad / 2.0);
    return .{
        .{ toF32(1.0 / (th * aspect_wh)), 0, 0, 0 },
        .{ 0, toF32(1.0 / th), 0, 0 },
        .{ 0, 0, toF32(-(far + near) / (far - near)), toF32(-2.0 * far * near / (far - near)) },
        .{ 0, 0, -1, 0 },
    };
}

/// m @ v (f32), v a 4-vector. Mirrors torch's float32 matmul used for pos_camera/pos_clip.
pub fn transformVec4(m: Mat4, v: [4]f32) [4]f32 {
    var out: [4]f32 = undefined;
    for (0..4) |r| {
        out[r] = m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + m[r][3] * v[3];
    }
    return out;
}

/// Full model-view-projection of a 3D point to CLIP space, sequential mv-then-proj to
/// match the reference (pos_camera then pos_clip, each a float32 matmul). Output is the
/// [x,y,z,w] clip vector `rasterize` consumes (perspective divide happens there).
pub fn projectToClip(mv: Mat4, proj: Mat4, p: [3]f32) [4]f32 {
    const cam = transformVec4(mv, .{ p[0], p[1], p[2], 1.0 });
    return transformVec4(proj, cam);
}

// f64 -> f32 store (matches numpy `.astype(np.float32)` at matrix-build time).
inline fn toF32(x: f64) f32 {
    return @floatCast(x);
}

// =============================================================================
// Tests (hermetic, no MLX). Run: `zig test src/rasterize.zig`
//
// The helpers below (ref*) are INDEPENDENT transcriptions of the reference formulas,
// so the tests validate the port's loop/index/bbox/z-buffer plumbing against a
// freshly-written copy of the arithmetic rather than against the port itself.
// Camera tests use GOLDEN matrices captured from the reference camera_utils.py functions
// (numpy float32), so they validate against ground truth, not a re-transcription.
// =============================================================================

fn refScreenX(vx: f32, vw: f32, width: u32) f32 {
    const wm1: f32 = @floatFromInt(@as(i64, width) - 1);
    return (vx / vw * 0.5 + 0.5) * wm1 + 0.5;
}
fn refScreenY(vy: f32, vw: f32, height: u32) f32 {
    const hm1: f32 = @floatFromInt(@as(i64, height) - 1);
    return (0.5 + 0.5 * vy / vw) * hm1 + 0.5;
}
fn refBary(s0: [2]f32, s1: [2]f32, s2: [2]f32, p: [2]f32) [3]f32 {
    const area = (s2[0] - s0[0]) * (s1[1] - s0[1]) - (s1[0] - s0[0]) * (s2[1] - s0[1]);
    if (area == 0) return .{ -1, -1, -1 };
    const beta_tri = (s2[0] - s0[0]) * (p[1] - s0[1]) - (p[0] - s0[0]) * (s2[1] - s0[1]);
    const gamma_tri = (p[0] - s0[0]) * (s1[1] - s0[1]) - (s1[0] - s0[0]) * (p[1] - s0[1]);
    const tri_inv: f32 = @floatCast(1.0 / @as(f64, area));
    const beta = beta_tri * tri_inv;
    const gamma = gamma_tri * tri_inv;
    const alpha: f32 = @floatCast(1.0 - @as(f64, beta) - @as(f64, gamma));
    return .{ alpha, beta, gamma };
}
fn refInside(bc: [3]f32) bool {
    return bc[0] >= 0 and bc[0] <= 1 and bc[1] >= 0 and bc[1] <= 1 and bc[2] >= 0 and bc[2] <= 1;
}
fn refCovers(v: []const f32, vi0: u32, vi1: u32, vi2: u32, width: u32, height: u32, px: u32, py: u32) bool {
    const a = vert4(v, vi0);
    const b = vert4(v, vi1);
    const c = vert4(v, vi2);
    const s0 = [2]f32{ refScreenX(a[0], a[3], width), refScreenY(a[1], a[3], height) };
    const s1 = [2]f32{ refScreenX(b[0], b[3], width), refScreenY(b[1], b[3], height) };
    const s2 = [2]f32{ refScreenX(c[0], c[3], width), refScreenY(c[1], c[3], height) };
    const p = [2]f32{ @as(f32, @floatFromInt(px)) + 0.5, @as(f32, @floatFromInt(py)) + 0.5 };
    return refInside(refBary(s0, s1, s2, p));
}

test "1: single triangle coverage matches analytic half-plane at every pixel" {
    const a = std.testing.allocator;
    const W: u32 = 24;
    const H: u32 = 24;
    // Clip (w=1): (-1,-1),(1,-1),(-1,1) -> screen right-triangle over the lower-left half.
    const v = [_]f32{
        -1, -1, 0, 1,
        1,  -1, 0, 1,
        -1, 1,  0, 1,
    };
    const idx = [_]u32{ 0, 1, 2 };
    var r = try rasterize(a, &v, &idx, W, H);
    defer r.deinit();

    var covered: usize = 0;
    var background: usize = 0;
    var py: u32 = 0;
    while (py < H) : (py += 1) {
        var px: u32 = 0;
        while (px < W) : (px += 1) {
            const inside = refCovers(&v, 0, 1, 2, W, H, px, py);
            const is_cov = r.face_id[py * W + px] != 0;
            try std.testing.expectEqual(inside, is_cov);
            if (is_cov) {
                covered += 1;
                try std.testing.expectEqual(@as(u32, 1), r.face_id[py * W + px]); // face 0 => id 1
            } else background += 1;
        }
    }
    // The triangle must partially cover (both inside and outside pixels present).
    try std.testing.expect(covered > 0);
    try std.testing.expect(background > 0);
}

test "2: overlapping triangles at different depths — nearer wins, correct face ids" {
    const a = std.testing.allocator;
    const W: u32 = 24;
    const H: u32 = 24;
    // Face 0 = small triangle, NEARER (z=-0.5 -> depth ~0.25).
    // Face 1 = big triangle, FARTHER (z=+0.5 -> depth ~0.75), and it contains face 0's region.
    const v = [_]f32{
        // face 0 (near)
        -0.5, -0.5, -0.5, 1,
        0.5,  -0.5, -0.5, 1,
        -0.5, 0.5,  -0.5, 1,
        // face 1 (far)
        -1, -1, 0.5, 1,
        1,  -1, 0.5, 1,
        -1, 1,  0.5, 1,
    };
    const idx = [_]u32{ 0, 1, 2, 3, 4, 5 };
    var r = try rasterize(a, &v, &idx, W, H);
    defer r.deinit();

    var n1: usize = 0;
    var n2: usize = 0;
    var py: u32 = 0;
    while (py < H) : (py += 1) {
        var px: u32 = 0;
        while (px < W) : (px += 1) {
            const cov_a = refCovers(&v, 0, 1, 2, W, H, px, py);
            const cov_b = refCovers(&v, 3, 4, 5, W, H, px, py);
            // Nearer (face 0) wins where it covers; else face 1 where it covers; else bg.
            const expected: u32 = if (cov_a) 1 else if (cov_b) 2 else 0;
            try std.testing.expectEqual(expected, r.face_id[py * W + px]);
            if (expected == 1) n1 += 1;
            if (expected == 2) n2 += 1;
        }
    }
    // Both faces must be visible somewhere (overlap resolved to 1, B-only resolved to 2).
    try std.testing.expect(n1 > 0);
    try std.testing.expect(n2 > 0);
}

test "3: perspective-corrected barycentric matches analytic perspective-correct value" {
    const a = std.testing.allocator;
    const W: u32 = 16;
    const H: u32 = 16;
    // Triangle with per-vertex w varying (1, 2, 4) so perspective correction is non-trivial.
    // x/w, y/w chosen to cover a central region; z arbitrary (unused by bary).
    const v = [_]f32{
        -0.6, -0.6, 0, 1,
        1.2,  -1.2, 0, 2, //  x/w=0.6, y/w=-0.6
        0.0,  2.8,  0, 4, //  x/w=0.0, y/w= 0.7
    };
    const idx = [_]u32{ 0, 1, 2 };
    const attr = [_]f32{ 0.0, 1.0, 2.0 }; // 1 channel, one value per vertex

    var r = try rasterize(a, &v, &idx, W, H);
    defer r.deinit();
    const interp = try interpolate(a, &attr, 1, &r, &idx, 0.0);
    defer a.free(interp);

    // Probe the covered pixel nearest the screen-space centroid (well-conditioned).
    const c0 = [2]f32{ refScreenX(-0.6, 1, W), refScreenY(-0.6, 1, H) };
    const c1 = [2]f32{ refScreenX(1.2, 2, W), refScreenY(-1.2, 2, H) };
    const c2 = [2]f32{ refScreenX(0.0, 4, W), refScreenY(2.8, 4, H) };
    const cx = (c0[0] + c1[0] + c2[0]) / 3.0;
    const cy = (c0[1] + c1[1] + c2[1]) / 3.0;

    var best_px: u32 = 0;
    var best_py: u32 = 0;
    var best_d: f32 = std.math.floatMax(f32);
    var found = false;
    var py: u32 = 0;
    while (py < H) : (py += 1) {
        var px: u32 = 0;
        while (px < W) : (px += 1) {
            if (r.face_id[py * W + px] == 0) continue;
            const dx = (@as(f32, @floatFromInt(px)) + 0.5) - cx;
            const dy = (@as(f32, @floatFromInt(py)) + 0.5) - cy;
            const d = dx * dx + dy * dy;
            if (d < best_d) {
                best_d = d;
                best_px = px;
                best_py = py;
                found = true;
            }
        }
    }
    try std.testing.expect(found);

    // Independent perspective-correct interpolation in f64 (screen bary -> /w -> renorm).
    const p = [2]f64{ @as(f64, @floatFromInt(best_px)) + 0.5, @as(f64, @floatFromInt(best_py)) + 0.5 };
    const S0 = [2]f64{ c0[0], c0[1] };
    const S1 = [2]f64{ c1[0], c1[1] };
    const S2 = [2]f64{ c2[0], c2[1] };
    const area = (S2[0] - S0[0]) * (S1[1] - S0[1]) - (S1[0] - S0[0]) * (S2[1] - S0[1]);
    const beta_tri = (S2[0] - S0[0]) * (p[1] - S0[1]) - (p[0] - S0[0]) * (S2[1] - S0[1]);
    const gamma_tri = (p[0] - S0[0]) * (S1[1] - S0[1]) - (S1[0] - S0[0]) * (p[1] - S0[1]);
    const beta = beta_tri / area;
    const gamma = gamma_tri / area;
    const alpha = 1.0 - beta - gamma;
    const pc0 = alpha / 1.0;
    const pc1 = beta / 2.0;
    const pc2 = gamma / 4.0;
    const norm = pc0 + pc1 + pc2;
    const w0 = pc0 / norm;
    const w1 = pc1 / norm;
    const w2 = pc2 / norm;
    const expected: f64 = w0 * 0.0 + w1 * 1.0 + w2 * 2.0;

    const got = interp[best_py * W + best_px];
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), got, 1e-5);
}

test "4: degenerate and behind-camera triangles produce no coverage and no crash" {
    const a = std.testing.allocator;
    const W: u32 = 16;
    const H: u32 = 16;

    // (a) Collinear vertices -> area 0 -> {-1,-1,-1} -> never in bounds.
    {
        const v = [_]f32{
            -1, -1, 0, 1,
            0,  0,  0, 1,
            1,  1,  0, 1,
        };
        const idx = [_]u32{ 0, 1, 2 };
        var r = try rasterize(a, &v, &idx, W, H);
        defer r.deinit();
        for (r.face_id) |fid| try std.testing.expectEqual(@as(u32, 0), fid);
    }

    // (b) Behind camera: z/w = -3 -> depth = -3*0.49999+0.5 < 0 -> depth<depth_thres skip.
    {
        const v = [_]f32{
            -1, -1, -3, 1,
            1,  -1, -3, 1,
            -1, 1,  -3, 1,
        };
        const idx = [_]u32{ 0, 1, 2 };
        var r = try rasterize(a, &v, &idx, W, H);
        defer r.deinit();
        for (r.face_id) |fid| try std.testing.expectEqual(@as(u32, 0), fid);
    }

    // (c) Entirely off-screen triangle -> no covered pixel.
    {
        const v = [_]f32{
            5, 5, 0, 1,
            6, 5, 0, 1,
            5, 6, 0, 1,
        };
        const idx = [_]u32{ 0, 1, 2 };
        var r = try rasterize(a, &v, &idx, W, H);
        defer r.deinit();
        for (r.face_id) |fid| try std.testing.expectEqual(@as(u32, 0), fid);
    }
}

test "5: barycentrics sum to 1 on every covered pixel" {
    const a = std.testing.allocator;
    const W: u32 = 20;
    const H: u32 = 20;
    // Varying-w triangle (exercises perspective renormalization).
    const v = [_]f32{
        -0.7, -0.7, 0.1, 1,
        1.6,  -0.8, 0.2, 2,
        -0.4, 2.4,  0.3, 3,
    };
    const idx = [_]u32{ 0, 1, 2 };
    var r = try rasterize(a, &v, &idx, W, H);
    defer r.deinit();

    var covered: usize = 0;
    var pix: usize = 0;
    while (pix < r.face_id.len) : (pix += 1) {
        if (r.face_id[pix] == 0) continue;
        covered += 1;
        const s = r.bary[pix * 3] + r.bary[pix * 3 + 1] + r.bary[pix * 3 + 2];
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), s, 1e-5);
    }
    try std.testing.expect(covered > 0);
}

// Golden values captured from the reference camera_utils.py (numpy float32) — see the
// generator in the task log. Row-major flattened 4x4.
fn expectMat(golden: [16]f32, m: Mat4, tol: f32) !void {
    for (0..4) |r| {
        for (0..4) |c| {
            try std.testing.expectApproxEqAbs(golden[r * 4 + c], m[r][c], tol);
        }
    }
}

test "6: getMvMatrix matches reference golden matrices" {
    // elev=0, azim=0, dist=2, center=None
    try expectMat(.{
        -1.0,               6.123234262925839e-17, 0.0,  -0.0,
        0.0,                0.0,                    1.0,  -0.0,
        6.123234262925839e-17, 1.0,                 -0.0, -2.0,
        0.0,                0.0,                    0.0,  1.0,
    }, getMvMatrix(0.0, 0.0, 2.0, .{ 0, 0, 0 }), 1e-6);

    // elev=30, azim=45, dist=1.8, center=None
    try expectMat(.{
        -0.7071067690849304, -0.7071067690849304, 0.0,                -8.597808592849266e-17,
        -0.3535533845424652, 0.3535533845424652,  0.8660253882408142, -1.1022519813703505e-16,
        -0.6123724579811096, 0.6123724579811096,  -0.5,               -1.7999999523162842,
        0.0,                 0.0,                  0.0,                1.0,
    }, getMvMatrix(30.0, 45.0, 1.8, .{ 0, 0, 0 }), 1e-6);

    // elev=-20, azim=90, dist=3, center=None
    try expectMat(.{
        -1.2246468525851679e-16, -1.0,                    0.0,                -0.0,
        0.3420201539993286,      -4.1885388812476027e-17, 0.9396926164627075, -4.1009159698321055e-17,
        -0.9396926164627075,     1.1507915352850032e-16,  0.3420201539993286, -3.0,
        0.0,                     0.0,                     0.0,                1.0,
    }, getMvMatrix(-20.0, 90.0, 3.0, .{ 0, 0, 0 }), 1e-6);

    // elev=15, azim=120, dist=2.5, center=[0.1,-0.2,0.3]
    try expectMat(.{
        0.41770491003990173,  -0.9085827469825745,  0.0,                  -0.2234870344400406,
        -0.3320940136909485,  -0.1526743620634079,  0.9308083057403564,   -0.2765679657459259,
        -0.8457163572311401,  -0.3888031840324402,  -0.36550772190093994, -2.474583625793457,
        0.0,                  0.0,                   0.0,                  1.0,
    }, getMvMatrix(15.0, 120.0, 2.5, .{ 0.1, -0.2, 0.3 }), 1e-6);
}

test "7: projection matrices match reference golden" {
    // perspective 49.13deg, aspect 1.0, near 0.01, far 100 (square resolution)
    try expectMat(.{
        2.1877193450927734, 0.0,                0.0,                 0.0,
        0.0,                2.1877193450927734, 0.0,                 0.0,
        0.0,                0.0,                -1.0002000331878662, -0.020002000033855438,
        0.0,                0.0,                -1.0,                0.0,
    }, getPerspectiveProjection(49.13, 1.0, 0.01, 100.0), 1e-6);

    // perspective aspect 4/3
    try expectMat(.{
        1.6407893896102905, 0.0,                0.0,                 0.0,
        0.0,                2.1877193450927734, 0.0,                 0.0,
        0.0,                0.0,                -1.0002000331878662, -0.020002000033855438,
        0.0,                0.0,                -1.0,                0.0,
    }, getPerspectiveProjection(49.13, 4.0 / 3.0, 0.01, 100.0), 1e-6);

    // orthographic from set_orth_scale(1.2): half=0.6, near=0.1, far=100
    try expectMat(.{
        1.6666666269302368, 0.0,                0.0,                 -0.0,
        0.0,                1.6666666269302368, 0.0,                 -0.0,
        0.0,                0.0,                -0.0200200192630291, -1.0020020008087158,
        0.0,                0.0,                0.0,                 1.0,
    }, getOrthographicProjection(-0.6, 0.6, -0.6, 0.6, 0.1, 100.0), 1e-6);
}

test "8: projectToClip end-to-end matches reference clip coords" {
    const verts = [_][3]f32{ .{ 0.3, -0.4, 0.5 }, .{ 0.0, 0.0, 0.0 }, .{ -0.25, 0.1, -0.15 } };
    const mv = getMvMatrix(30.0, 45.0, 1.8, .{ 0, 0, 0 });

    const proj_p = getPerspectiveProjection(49.13, 1.0, 0.01, 100.0);
    const golden_p = [_][4]f32{
        .{ 0.15469510853290558, 0.40587732195854187, 2.4591546058654785, 2.4786605834960938 },
        .{ 0.0, 0.0, 1.7803579568862915, 1.7999999523162842 }, // ~0 x,y (1e-16)
        .{ 0.23204267024993896, -0.01347663626074791, 1.4909697771072388, 1.5106695890426636 },
    };
    for (verts, golden_p) |v, g| {
        const clip = projectToClip(mv, proj_p, v);
        for (0..4) |k| try std.testing.expectApproxEqAbs(g[k], clip[k], 1e-4);
    }

    const proj_o = getOrthographicProjection(-0.6, 0.6, -0.6, 0.6, 0.1, 100.0);
    const golden_o = [_][4]f32{
        .{ 0.117851123213768, 0.30920884013175964, -0.9523791670799255, 1.0 },
        .{ 0.0, 0.0, -0.965965986251831, 1.0 },
        .{ 0.1767766773700714, -0.010266883298754692, -0.9717583656311035, 1.0 },
    };
    for (verts, golden_o) |v, g| {
        const clip = projectToClip(mv, proj_o, v);
        for (0..4) |k| try std.testing.expectApproxEqAbs(g[k], clip[k], 1e-4);
    }
}

test "9: mv rotation rows are orthonormal (right-handed view basis)" {
    const m = getMvMatrix(37.0, 66.0, 2.3, .{ 0, 0, 0 });
    // rows 0,1,2 = right, up, -lookat; each unit length, mutually orthogonal.
    const rows = [_][3]f32{
        .{ m[0][0], m[0][1], m[0][2] },
        .{ m[1][0], m[1][1], m[1][2] },
        .{ m[2][0], m[2][1], m[2][2] },
    };
    for (rows) |row| {
        const n = row[0] * row[0] + row[1] * row[1] + row[2] * row[2];
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), n, 1e-5);
    }
    inline for (.{ .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 } }) |pair| {
        const a = rows[pair[0]];
        const b = rows[pair[1]];
        const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), d, 1e-5);
    }
    // bottom row is [0,0,0,1].
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 1 }, m[3]);
}
