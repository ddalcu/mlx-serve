//! Pure-Zig (CPU, std only — no MLX) port of the Hunyuan3D-2.1 texture-bake GEOMETRY
//! stack. This is the deterministic scaffolding around the multiview diffusion model:
//! set the mesh into render space, render its normal/position geometry maps, extract the
//! per-texel UV-space geometry, and back-project + cos⁴-blend the decoded view images
//! into a UV texture atlas.
//!
//! Ported from the reference (paths relative to reference/hy3dpaint):
//!   DifferentiableRenderer/camera_utils.py
//!     - get_mv_matrix / get_orthographic_projection_matrix  -> reused from rasterize.zig
//!   DifferentiableRenderer/MeshRender.py
//!     - set_mesh                (:665-724)  -> prepareMesh / buildMeshGeom
//!     - render_normal path      (:459-474, _get_normals_for_shading :417-446) -> renderGeometryMaps.normal
//!     - render_position path    (:476-490)  -> renderGeometryMaps.position
//!     - extract_textiles        (:923-979)  -> extractTextiles
//!     - render_sketch_from_depth(:1095-1111)-> detectDepthEdges (Sobel stand-in for cv2.Canny)
//!     - back_project back_sample(:1113-1315)-> backProject
//!     - fast_bake_texture       (:1352-1378)-> bakeBlend
//!   utils/pipeline_utils.py
//!     - bake_view_selection @ max_num_view=6 (:40-109) collapses to the fixed 6-view
//!       candidate table (textureGenPipeline.py:57-59) -> defaultViews
//!
//! MATRIX CONVENTION (pinned): matrices returned by mvMatrix/orthoMatrix are ROW-MAJOR
//! flattened `[16]f32`, element (row r, col c) at index r*4 + c. This mirrors the numpy
//! w2c/ortho arrays exactly (numpy is row-major and the reference indexes them [row][col]).
//! A point is transformed by `clip = proj @ mv @ [x,y,z,1]` (proj/mv applied on the left of
//! a column vector) — the sequential mv-then-proj matmul the reference uses. Internally we
//! reuse rasterize.zig's `Mat4 = [4][4]f32` (row-major m[row][col]) plus its projectToClip /
//! transformVec4; the flat `[16]f32` accessors are the public convenience the paint engine
//! consumes.
//!
//! FLOAT DISCIPLINE: matrix BUILD is f64 rounded to f32 at store time (numpy .astype, handled
//! in rasterize.zig). Everything else here — per-vertex transforms, face normals, cos maps,
//! cos⁴ weights, the 3e-3 depth compare, and the bake accumulators — is f32, matching torch's
//! float32 throughout the reference. No fp16 anywhere.

const std = @import("std");
const rast = @import("rasterize.zig");

// =============================================================================
// Views (utils/pipeline_utils.py bake_view_selection + textureGenPipeline.py table)
// =============================================================================

pub const View = struct { azim: f32, elev: f32, weight: f32 };

/// The fixed 6-view set. With `max_num_view = 6`, bake_view_selection's greedy loop runs
/// `max_selected_view_num - len(selected) = 0` iterations, so the selection is EXACTLY the
/// first six candidates from Hunyuan3DPaintConfig:
///   azims   = [0, 90, 180, 270, 0,  180]
///   elevs   = [0, 0,  0,   0,   90, -90]
///   weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
pub fn defaultViews() [6]View {
    return .{
        .{ .azim = 0, .elev = 0, .weight = 1.0 },
        .{ .azim = 90, .elev = 0, .weight = 0.1 },
        .{ .azim = 180, .elev = 0, .weight = 0.5 },
        .{ .azim = 270, .elev = 0, .weight = 0.1 },
        .{ .azim = 0, .elev = 90, .weight = 0.05 },
        .{ .azim = 180, .elev = -90, .weight = 0.05 },
    };
}

// =============================================================================
// Camera (public [16]f32 accessors over rasterize.zig's ported camera math)
// =============================================================================

fn flatten(m: rast.Mat4) [16]f32 {
    var out: [16]f32 = undefined;
    for (0..4) |r| {
        for (0..4) |c| out[r * 4 + c] = m[r][c];
    }
    return out;
}

/// World-to-camera (view) matrix, row-major flattened. Wraps get_mv_matrix (elev=-elev,
/// azim+=90, z-up look-at); `center = origin` reproduces the reference `center=None`.
/// NOTE the argument order (azim, elev) matches the task API; get_mv_matrix takes (elev, azim).
pub fn mvMatrix(azim_deg: f32, elev_deg: f32, distance: f32) [16]f32 {
    return flatten(rast.getMvMatrix(elev_deg, azim_deg, distance, .{ 0, 0, 0 }));
}

/// Orthographic projection, row-major flattened. `scale` = ortho_scale (the reference's
/// set_orth_scale uses left/right/bottom/top = ∓scale/2, so this maps [-scale/2, scale/2]²).
pub fn orthoMatrix(scale: f32, near: f32, far: f32) [16]f32 {
    const s = @as(f64, scale) * 0.5;
    return flatten(rast.getOrthographicProjection(-s, s, -s, s, near, far));
}

/// Rendering / baking configuration. Defaults mirror the reference MeshRender orth camera
/// (camera_distance 1.45, ortho_scale 1.2, near 0.1, far 100), bake_angle_thres 75°, and
/// set_boundary_unreliable_scale(2).
pub const RenderConfig = struct {
    camera_distance: f32 = 1.45,
    ortho_scale: f32 = 1.2,
    near: f32 = 0.1,
    far: f32 = 100.0,
    bake_angle_thres: f32 = 75.0,
    /// The `scale` in set_boundary_unreliable_scale; the boundary kernel half-size is
    /// int(boundary_scale/512 * render_size), i.e. 8 (→17×17) at render_size 2048.
    boundary_scale: f32 = 2.0,
};

fn mvMat(view: View, cfg: RenderConfig) rast.Mat4 {
    return rast.getMvMatrix(view.elev, view.azim, cfg.camera_distance, .{ 0, 0, 0 });
}
fn projMat(cfg: RenderConfig) rast.Mat4 {
    const s = @as(f64, cfg.ortho_scale) * 0.5;
    return rast.getOrthographicProjection(-s, s, -s, s, cfg.near, cfg.far);
}

// =============================================================================
// Small f32 vector helpers (torch-parity face-normal math)
// =============================================================================

fn sub3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}
fn cross3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}
fn normalize3(v: [3]f32) [3]f32 {
    const n = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n <= 0) return .{ 0, 0, 0 };
    return .{ v[0] / n, v[1] / n, v[2] / n };
}
fn vec3(a: []const f32, i: usize) [3]f32 {
    return .{ a[i * 3], a[i * 3 + 1], a[i * 3 + 2] };
}

/// F.normalize(cross(v1-v0, v2-v0)) per triangle, from the given N*3 positions & F*3 indices.
fn computeFaceNormals(allocator: std.mem.Allocator, positions: []const f32, indices: []const u32) ![]f32 {
    const num_faces = indices.len / 3;
    const out = try allocator.alloc(f32, num_faces * 3);
    errdefer allocator.free(out);
    var f: usize = 0;
    while (f < num_faces) : (f += 1) {
        const v0 = vec3(positions, indices[f * 3]);
        const v1 = vec3(positions, indices[f * 3 + 1]);
        const v2 = vec3(positions, indices[f * 3 + 2]);
        const n = normalize3(cross3(sub3(v1, v0), sub3(v2, v0)));
        out[f * 3] = n[0];
        out[f * 3 + 1] = n[1];
        out[f * 3 + 2] = n[2];
    }
    return out;
}

// =============================================================================
// MeshGeom (set_mesh product)
// =============================================================================

pub const MeshGeom = struct {
    /// N*3 remapped (+normalized) vertex positions in render space.
    positions: []f32,
    /// N*2 UV coordinates (v-flipped).
    uvs: []f32,
    /// F*3 position (triangle) indices.
    indices: []u32,
    /// F*3 UV indices (== `indices` when the mesh shares one index set).
    uv_indices: []u32,
    /// F*3 world-space per-face normals (from `positions`).
    face_normals: []f32,
    /// scale_factor used by render_position (`0.5 - pos/scale_factor`).
    scale_factor: f32,
    num_verts: usize,
    num_faces: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MeshGeom) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.uvs);
        self.allocator.free(self.indices);
        self.allocator.free(self.uv_indices);
        self.allocator.free(self.face_normals);
    }
};

/// Assemble a MeshGeom from ALREADY-render-space positions/uvs (no remap/normalize).
/// Duplicates the inputs (caller keeps ownership of its slices) and computes face normals.
/// prepareMesh calls this after remap+v-flip+auto-center; tests use it directly to pin
/// exact geometry.
pub fn buildMeshGeom(
    allocator: std.mem.Allocator,
    positions: []const f32,
    uvs: []const f32,
    indices: []const u32,
    uv_indices: []const u32,
    scale_factor: f32,
) !MeshGeom {
    const pos = try allocator.dupe(f32, positions);
    errdefer allocator.free(pos);
    const uv = try allocator.dupe(f32, uvs);
    errdefer allocator.free(uv);
    const idx = try allocator.dupe(u32, indices);
    errdefer allocator.free(idx);
    const uvidx = try allocator.dupe(u32, uv_indices);
    errdefer allocator.free(uvidx);
    const fn_ = try computeFaceNormals(allocator, pos, idx);
    errdefer allocator.free(fn_);
    return .{
        .positions = pos,
        .uvs = uv,
        .indices = idx,
        .uv_indices = uvidx,
        .face_normals = fn_,
        .scale_factor = scale_factor,
        .num_verts = positions.len / 3,
        .num_faces = indices.len / 3,
        .allocator = allocator,
    };
}

/// set_mesh port. Applies (in order, matching the reference's sequential torch ops):
///   1. axis remap: negate x,y then swap y,z  →  (x,y,z) -> (-x, z, -y)
///   2. UV v-flip: v -> 1 - v
///   3. auto-center + normalize to a bounding sphere of radius scale_factor/2:
///        center = (max_bb + min_bb)/2 ;  scale = max‖p-center‖ * 2
///        p := (p - center) * (scale_factor / scale)
/// then computes world-space face normals. `uv_indices == indices` for a shared index set.
pub fn prepareMesh(
    allocator: std.mem.Allocator,
    positions_in: []const f32,
    uvs_in: []const f32,
    indices: []const u32,
    uv_indices: []const u32,
    scale_factor: f32,
) !MeshGeom {
    const n = positions_in.len / 3;

    // 1. axis remap.
    var pos = try allocator.alloc(f32, positions_in.len);
    defer allocator.free(pos);
    var v: usize = 0;
    while (v < n) : (v += 1) {
        const x = positions_in[v * 3];
        const y = positions_in[v * 3 + 1];
        const z = positions_in[v * 3 + 2];
        pos[v * 3] = -x;
        pos[v * 3 + 1] = z;
        pos[v * 3 + 2] = -y;
    }

    // 2. UV v-flip.
    var uv = try allocator.alloc(f32, uvs_in.len);
    defer allocator.free(uv);
    var u: usize = 0;
    while (u < uvs_in.len / 2) : (u += 1) {
        uv[u * 2] = uvs_in[u * 2];
        uv[u * 2 + 1] = 1.0 - uvs_in[u * 2 + 1];
    }

    // 3. auto-center + normalize.
    var min_bb = [3]f32{ std.math.floatMax(f32), std.math.floatMax(f32), std.math.floatMax(f32) };
    var max_bb = [3]f32{ -std.math.floatMax(f32), -std.math.floatMax(f32), -std.math.floatMax(f32) };
    v = 0;
    while (v < n) : (v += 1) {
        for (0..3) |c| {
            const p = pos[v * 3 + c];
            min_bb[c] = @min(min_bb[c], p);
            max_bb[c] = @max(max_bb[c], p);
        }
    }
    const center = [3]f32{
        (max_bb[0] + min_bb[0]) * 0.5,
        (max_bb[1] + min_bb[1]) * 0.5,
        (max_bb[2] + min_bb[2]) * 0.5,
    };
    var max_r: f32 = 0;
    v = 0;
    while (v < n) : (v += 1) {
        const dx = pos[v * 3] - center[0];
        const dy = pos[v * 3 + 1] - center[1];
        const dz = pos[v * 3 + 2] - center[2];
        const r = @sqrt(dx * dx + dy * dy + dz * dz);
        max_r = @max(max_r, r);
    }
    const scale = max_r * 2.0;
    const factor: f32 = if (scale > 0) scale_factor / scale else 1.0;
    v = 0;
    while (v < n) : (v += 1) {
        for (0..3) |c| pos[v * 3 + c] = (pos[v * 3 + c] - center[c]) * factor;
    }

    return buildMeshGeom(allocator, pos, uv, indices, uv_indices, scale_factor);
}

// =============================================================================
// Per-face pixel fill (shader_type == "face": covered pixels get their face's attribute)
// =============================================================================

fn fillFaceAttr(
    allocator: std.mem.Allocator,
    raster: *const rast.Raster,
    face_attr: []const f32,
    channels: usize,
    background: f32,
) ![]f32 {
    const num_pixels = raster.face_id.len;
    const out = try allocator.alloc(f32, num_pixels * channels);
    errdefer allocator.free(out);
    var pix: usize = 0;
    while (pix < num_pixels) : (pix += 1) {
        const fid = raster.face_id[pix];
        if (fid == 0) {
            for (0..channels) |c| out[pix * channels + c] = background;
            continue;
        }
        const f = @as(usize, fid - 1);
        for (0..channels) |c| out[pix * channels + c] = face_attr[f * channels + c];
    }
    return out;
}

// =============================================================================
// renderGeometryMaps (render_normal + render_position + mask)
// =============================================================================

pub const GeomMaps = struct {
    /// size*size*3 world-space face normal, mapped (n+1)/2, background EXACTLY 1.0.
    normal: []f32,
    /// size*size*3 `0.5 - pos/scale_factor` (interpolated), background EXACTLY 1.0.
    position: []f32,
    /// size*size visibility mask: 1.0 covered, 0.0 background.
    mask: []f32,
    size: u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GeomMaps) void {
        self.allocator.free(self.normal);
        self.allocator.free(self.position);
        self.allocator.free(self.mask);
    }
};

/// Render the world-space normal map and the position map for one view at `render_size²`.
/// Background is EXACTLY 1.0 in both (bg_color [1,1,1] → the UNet detects background via
/// position != 1). The normal is the world (use_abs_coor) FACE normal, (n+1)/2; the position
/// is the per-vertex `0.5 - pos/scale_factor` interpolated across the face.
pub fn renderGeometryMaps(
    allocator: std.mem.Allocator,
    geom: *const MeshGeom,
    view: View,
    render_size: u32,
    cfg: RenderConfig,
) !GeomMaps {
    const mv = mvMat(view, cfg);
    const proj = projMat(cfg);

    // Clip-space vertices (proj @ mv @ [p,1]).
    const clip = try allocator.alloc(f32, geom.num_verts * 4);
    defer allocator.free(clip);
    var v: usize = 0;
    while (v < geom.num_verts) : (v += 1) {
        const c = rast.projectToClip(mv, proj, vec3(geom.positions, v));
        clip[v * 4] = c[0];
        clip[v * 4 + 1] = c[1];
        clip[v * 4 + 2] = c[2];
        clip[v * 4 + 3] = c[3];
    }

    var raster = try rast.rasterize(allocator, clip, geom.indices, render_size, render_size);
    defer raster.deinit();

    // Normal map: per-face world normal, (n+1)/2, bg 1.0. Because bg_color is [1,1,1], the
    // reference's `(bg_masked + 1)*0.5` yields exactly 1.0 on background, so a direct fill of
    // (face_normal+1)/2 on covered / 1.0 on bg is identical.
    const half_norm = try allocator.alloc(f32, geom.num_faces * 3);
    defer allocator.free(half_norm);
    for (0..geom.num_faces * 3) |i| half_norm[i] = (geom.face_normals[i] + 1.0) * 0.5;
    const normal = try fillFaceAttr(allocator, &raster, half_norm, 3, 1.0);
    errdefer allocator.free(normal);

    // Position map: per-vertex tex_position = 0.5 - pos/scale_factor, interpolated, bg 1.0.
    const tex_pos = try allocator.alloc(f32, geom.num_verts * 3);
    defer allocator.free(tex_pos);
    for (0..geom.num_verts * 3) |i| tex_pos[i] = 0.5 - geom.positions[i] / geom.scale_factor;
    const position = try rast.interpolate(allocator, tex_pos, 3, &raster, geom.indices, 1.0);
    errdefer allocator.free(position);

    // Visibility mask.
    const mask = try allocator.alloc(f32, raster.face_id.len);
    errdefer allocator.free(mask);
    for (raster.face_id, 0..) |fid, i| mask[i] = if (fid > 0) 1.0 else 0.0;

    return .{ .normal = normal, .position = position, .mask = mask, .size = render_size, .allocator = allocator };
}

// =============================================================================
// extractTextiles (UV-space rasterization -> per-texel world position/normal)
// =============================================================================

pub const Textiles = struct {
    /// count*3 world-space position of each covered texel.
    positions: []f32,
    /// count*3 world-space (per-face) normal of each covered texel.
    normals: []f32,
    /// count*2 texel coordinate (row i, col j) of each covered texel.
    grid: []u32,
    count: usize,
    tex_size: u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Textiles) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.normals);
        self.allocator.free(self.grid);
    }
};

/// Rasterize the mesh in UV space (`clip = [2u-1, 2v-1, -1, 1]`, uv_indices) at texture_size²
/// and record, for every covered texel, the interpolated world POSITION (via pos indices) and
/// the per-face world NORMAL, plus its (row, col) grid coordinate. This is the compact texel
/// list back_project projects through each view.
pub fn extractTextiles(
    allocator: std.mem.Allocator,
    geom: *const MeshGeom,
    texture_size: u32,
) !Textiles {
    // UV-space clip vertices: [2u-1, 2v-1, 2*0-1, 2*1-1] = [2u-1, 2v-1, -1, 1].
    const clip = try allocator.alloc(f32, geom.num_verts * 4);
    defer allocator.free(clip);
    var v: usize = 0;
    while (v < geom.num_verts) : (v += 1) {
        clip[v * 4] = geom.uvs[v * 2] * 2.0 - 1.0;
        clip[v * 4 + 1] = geom.uvs[v * 2 + 1] * 2.0 - 1.0;
        clip[v * 4 + 2] = -1.0;
        clip[v * 4 + 3] = 1.0;
    }

    var raster = try rast.rasterize(allocator, clip, geom.uv_indices, texture_size, texture_size);
    defer raster.deinit();

    // Interpolated world positions use the POSITION indices (face f in uv_indices ↔ face f in
    // indices — the shared face ordering of an OBJ). Background (0) is masked out below.
    const position = try rast.interpolate(allocator, geom.positions, 3, &raster, geom.indices, 0.0);
    defer allocator.free(position);

    var positions: std.ArrayList(f32) = .empty;
    errdefer positions.deinit(allocator);
    var normals: std.ArrayList(f32) = .empty;
    errdefer normals.deinit(allocator);
    var grid: std.ArrayList(u32) = .empty;
    errdefer grid.deinit(allocator);

    var count: usize = 0;
    var pix: usize = 0;
    while (pix < raster.face_id.len) : (pix += 1) {
        const fid = raster.face_id[pix];
        if (fid == 0) continue;
        const f = @as(usize, fid - 1);
        try positions.appendSlice(allocator, position[pix * 3 .. pix * 3 + 3]);
        try normals.appendSlice(allocator, geom.face_normals[f * 3 .. f * 3 + 3]);
        try grid.append(allocator, @intCast(pix / texture_size)); // row i
        try grid.append(allocator, @intCast(pix % texture_size)); // col j
        count += 1;
    }

    return .{
        .positions = try positions.toOwnedSlice(allocator),
        .normals = try normals.toOwnedSlice(allocator),
        .grid = try grid.toOwnedSlice(allocator),
        .count = count,
        .tex_size = texture_size,
        .allocator = allocator,
    };
}

// =============================================================================
// Depth-edge detection (Sobel-gradient-threshold stand-in for cv2.Canny(30,80))
// =============================================================================

/// DEVIATION (documented): the reference marks unreliable depth boundaries with
/// `cv2.Canny(depth*255, 30, 80)`. Canny = Sobel gradient → non-max suppression → hysteresis,
/// producing thin edges. We port the EFFECT with a Sobel-3×3 gradient-magnitude threshold:
/// edges are pixels where ‖∇(depth*255)‖ exceeds `high` (80, Canny's strong threshold). This
/// flags the same obvious depth discontinuities (occlusion silhouettes are large steps that
/// overwhelm 80×), and since the sketch is only DILATED and subtracted from the bake mask, a
/// slightly thicker edge is conservative (marks marginally more texels unreliable). It is NOT
/// non-max-suppressed, so edges are 1–2 px wider than Canny's — acceptable per the porting
/// note. `depth_img` is in [0,1]; borders (no full 3×3 neighborhood) are never edges.
fn detectDepthEdges(allocator: std.mem.Allocator, depth_img: []const f32, w: u32, h: u32) ![]f32 {
    const out = try allocator.alloc(f32, depth_img.len);
    errdefer allocator.free(out);
    @memset(out, 0);
    const high: f32 = 80.0;
    if (w < 3 or h < 3) return out;
    const wi: usize = w;
    var y: usize = 1;
    while (y < h - 1) : (y += 1) {
        var x: usize = 1;
        while (x < w - 1) : (x += 1) {
            // 0-255 scaled neighborhood (matches the reference's uint8 cast before Canny).
            const p00 = depth_img[(y - 1) * wi + (x - 1)] * 255.0;
            const p01 = depth_img[(y - 1) * wi + x] * 255.0;
            const p02 = depth_img[(y - 1) * wi + (x + 1)] * 255.0;
            const p10 = depth_img[y * wi + (x - 1)] * 255.0;
            const p12 = depth_img[y * wi + (x + 1)] * 255.0;
            const p20 = depth_img[(y + 1) * wi + (x - 1)] * 255.0;
            const p21 = depth_img[(y + 1) * wi + x] * 255.0;
            const p22 = depth_img[(y + 1) * wi + (x + 1)] * 255.0;
            const gx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
            const gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);
            const mag = @sqrt(gx * gx + gy * gy);
            if (mag > high) out[y * wi + x] = 1.0;
        }
    }
    return out;
}

// =============================================================================
// backProject (back_sample) + bakeBlend (fast_bake_texture)
// =============================================================================

pub const ViewProjection = struct {
    /// tex_size*tex_size*channels — sampled color per atlas texel (0 where this view didn't paint).
    texture: []f32,
    /// tex_size*tex_size — RAW cos weight per atlas texel (before the view weight / **exp).
    cos: []f32,
    /// tex_size*tex_size — boundary (dilated depth-edge) flag sampled per atlas texel.
    boundary: []f32,
    /// View weight (from `View.weight`), consumed by bakeBlend as `weight * cos^exp`.
    weight: f32,
    tex_size: u32,
    channels: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ViewProjection) void {
        self.allocator.free(self.texture);
        self.allocator.free(self.cos);
        self.allocator.free(self.boundary);
    }
};

fn boundaryKernelHalf(boundary_scale: f32, resolution: u32) usize {
    const k = boundary_scale / 512.0 * @as(f32, @floatFromInt(resolution));
    if (k <= 0) return 0;
    return @intFromFloat(k); // int() truncates toward zero (k > 0 here)
}

/// Back-project one decoded view image (`image` = image_h*image_w*channels, row-major) onto
/// the UV atlas via the "back_sample" method: rasterize the mesh from the view, build the
/// camera-space depth + cos maps + (eroded) reliability mask, then for each texel project its
/// world position into the image, gate on frustum + depth (|Δz| < 3e-3) + cos-threshold + the
/// reliability mask, and bilinearly sample the color. Returns per-texel color + RAW cos + a
/// boundary flag laid out over the full atlas. `bakeBlend` applies the view weight and cos⁴.
pub fn backProject(
    allocator: std.mem.Allocator,
    geom: *const MeshGeom,
    textiles: *const Textiles,
    view: View,
    image: []const f32,
    image_w: u32,
    image_h: u32,
    channels: usize,
    cfg: RenderConfig,
) !ViewProjection {
    const mv = mvMat(view, cfg);
    const proj = projMat(cfg);
    const num_pixels = @as(usize, image_w) * @as(usize, image_h);

    // Camera-space vertex positions (w=1, affine w2c) + clip positions.
    const cam = try allocator.alloc(f32, geom.num_verts * 3);
    defer allocator.free(cam);
    const clip = try allocator.alloc(f32, geom.num_verts * 4);
    defer allocator.free(clip);
    var v: usize = 0;
    while (v < geom.num_verts) : (v += 1) {
        const p = vec3(geom.positions, v);
        const cc = rast.transformVec4(mv, .{ p[0], p[1], p[2], 1.0 });
        cam[v * 3] = cc[0];
        cam[v * 3 + 1] = cc[1];
        cam[v * 3 + 2] = cc[2];
        const cl = rast.transformVec4(proj, cc);
        clip[v * 4] = cl[0];
        clip[v * 4 + 1] = cl[1];
        clip[v * 4 + 2] = cl[2];
        clip[v * 4 + 3] = cl[3];
    }

    // Camera-space per-face normals (cos map is computed from these, not the world normals).
    const cam_fn = try computeFaceNormals(allocator, cam, geom.indices);
    defer allocator.free(cam_fn);

    var raster = try rast.rasterize(allocator, clip, geom.indices, image_w, image_h);
    defer raster.deinit();

    // visible_mask.
    var visible = try allocator.alloc(f32, num_pixels);
    defer allocator.free(visible);
    for (raster.face_id, 0..) |fid, i| visible[i] = if (fid > 0) 1.0 else 0.0;

    // Per-pixel camera-space depth (interpolated vertex cam_z).
    const vert_z = try allocator.alloc(f32, geom.num_verts);
    defer allocator.free(vert_z);
    for (0..geom.num_verts) |i| vert_z[i] = cam[i * 3 + 2];
    const depth = try rast.interpolate(allocator, vert_z, 1, &raster, geom.indices, 0.0);
    defer allocator.free(depth);

    // cos map: cosine_similarity([0,0,-1], camera-space face normal) = -n_z (unit normals);
    // below cos(bake_angle_thres) -> 0. Per-face fill, background stays 0.
    const cos_thres = @cos(cfg.bake_angle_thres / 180.0 * std.math.pi);
    var cos_img = try allocator.alloc(f32, num_pixels);
    defer allocator.free(cos_img);
    for (raster.face_id, 0..) |fid, i| {
        if (fid == 0) {
            cos_img[i] = 0;
            continue;
        }
        const f = @as(usize, fid - 1);
        var cval = -cam_fn[f * 3 + 2];
        if (cval < cos_thres) cval = 0;
        cos_img[i] = cval;
    }

    // depth normalization over covered pixels -> depth_image = depth_norm * visible.
    var dmin: f32 = std.math.floatMax(f32);
    var dmax: f32 = -std.math.floatMax(f32);
    var any_cov = false;
    for (0..num_pixels) |i| {
        if (visible[i] > 0) {
            dmin = @min(dmin, depth[i]);
            dmax = @max(dmax, depth[i]);
            any_cov = true;
        }
    }
    const depth_image = try allocator.alloc(f32, num_pixels);
    defer allocator.free(depth_image);
    const drange = dmax - dmin;
    for (0..num_pixels) |i| {
        if (any_cov and visible[i] > 0 and drange > 0) {
            depth_image[i] = (depth[i] - dmin) / drange;
        } else {
            depth_image[i] = 0;
        }
    }

    const sketch = try detectDepthEdges(allocator, depth_image, image_w, image_h);
    defer allocator.free(sketch);

    // Boundary erosion / dilation (skipped at small render sizes where the kernel rounds to 0).
    const khalf = boundaryKernelHalf(cfg.boundary_scale, @max(image_w, image_h));
    if (khalf > 0) {
        try applyBoundaryErosion(allocator, visible, sketch, image_w, image_h, khalf);
    }

    // cos_image[visible==0] = 0.
    for (0..num_pixels) |i| {
        if (visible[i] == 0) cos_img[i] = 0;
    }

    // Allocate atlas-sized outputs (zeros where this view doesn't paint).
    const ts = textiles.tex_size;
    const num_texels = @as(usize, ts) * @as(usize, ts);
    const texture = try allocator.alloc(f32, num_texels * channels);
    errdefer allocator.free(texture);
    @memset(texture, 0);
    const cos_out = try allocator.alloc(f32, num_texels);
    errdefer allocator.free(cos_out);
    @memset(cos_out, 0);
    const boundary_out = try allocator.alloc(f32, num_texels);
    errdefer allocator.free(boundary_out);
    @memset(boundary_out, 0);

    // img_proj = diag(proj00, proj11, 1, 1): NDC x,y scaling only (z stays raw camera z).
    const proj00 = proj[0][0];
    const proj11 = proj[1][1];
    const fw: f32 = @floatFromInt(image_w);
    const fh: f32 = @floatFromInt(image_h);
    const depth_thres: f32 = 3e-3;

    var t: usize = 0;
    while (t < textiles.count) : (t += 1) {
        const p = vec3(textiles.positions, t);
        const c = rast.transformVec4(mv, .{ p[0], p[1], p[2], 1.0 });
        // v_proj = camera-space point @ img_proj.
        const vpx = c[0] * proj00;
        const vpy = c[1] * proj11;
        const vpz = c[2]; // v_z

        const inner = @abs(vpx) <= 1.0 and @abs(vpy) <= 1.0;

        const cx = clampf(vpx, -1.0, 1.0);
        const cy = clampf(vpy, -1.0, 1.0);
        const img_x = clampIdx((cx * 0.5 + 0.5) * fw, image_w);
        const img_y = clampIdx((cy * 0.5 + 0.5) * fh, image_h);
        const idx = img_y * image_w + img_x;

        const sampled_z = depth[idx];
        const sampled_m = visible[idx];
        const sampled_w = cos_img[idx];

        const valid = inner and (@abs(vpz - sampled_z) < depth_thres) and (sampled_m * sampled_w > 0);
        if (!valid) continue;

        // Bilinear sample (raw v_proj fractional part; inner guarantees the in-range window).
        const wx = (vpx * 0.5 + 0.5) * fw - @as(f32, @floatFromInt(img_x));
        const wy = (vpy * 0.5 + 0.5) * fh - @as(f32, @floatFromInt(img_y));
        const img_x_r = @min(img_x + 1, image_w - 1);
        const img_y_r = @min(img_y + 1, image_h - 1);
        const idx_lr = img_y * image_w + img_x_r;
        const idx_rl = img_y_r * image_w + img_x;
        const idx_rr = img_y_r * image_w + img_x_r;

        const atlas = @as(usize, textiles.grid[t * 2]) * ts + @as(usize, textiles.grid[t * 2 + 1]);
        for (0..channels) |ch| {
            const a00 = image[idx * channels + ch];
            const a01 = image[idx_lr * channels + ch];
            const a10 = image[idx_rl * channels + ch];
            const a11 = image[idx_rr * channels + ch];
            const top = a00 * (1 - wx) + a01 * wx;
            const bot = a10 * (1 - wx) + a11 * wx;
            texture[atlas * channels + ch] = top * (1 - wy) + bot * wy;
        }
        cos_out[atlas] = sampled_w;
        boundary_out[atlas] = sketch[idx];
    }

    return .{
        .texture = texture,
        .cos = cos_out,
        .boundary = boundary_out,
        .weight = view.weight,
        .tex_size = ts,
        .channels = channels,
        .allocator = allocator,
    };
}

fn clampf(x: f32, lo: f32, hi: f32) f32 {
    return @max(lo, @min(hi, x));
}
/// floor(x) then clamp into [0, res-1] as a usize index (mirrors the reference `.long()` +
/// clamp; the float here is already produced from a clamped NDC so it is finite and small).
fn clampIdx(x: f32, res: u32) usize {
    const fl = @floor(x);
    if (fl <= 0) return 0;
    const i: i64 = @intFromFloat(fl);
    const hi: i64 = @as(i64, res) - 1;
    if (i > hi) return @intCast(hi);
    return @intCast(i);
}

/// Reference boundary handling: erode visible_mask (a pixel survives only if its full
/// (2·khalf+1)² neighborhood is foreground) and dilate the depth-edge sketch (any edge in the
/// neighborhood → 1), then `visible *= (sketch < 0.5)`. Mutates `visible` and `sketch` in place.
fn applyBoundaryErosion(
    allocator: std.mem.Allocator,
    visible: []f32,
    sketch: []f32,
    w: u32,
    h: u32,
    khalf: usize,
) !void {
    const wi: usize = w;
    const hi: usize = h;
    const eroded = try allocator.dupe(f32, visible);
    defer allocator.free(eroded);
    const dilated = try allocator.dupe(f32, sketch);
    defer allocator.free(dilated);

    var y: usize = 0;
    while (y < hi) : (y += 1) {
        var x: usize = 0;
        while (x < wi) : (x += 1) {
            const y0 = if (y >= khalf) y - khalf else 0;
            const y1 = @min(hi - 1, y + khalf);
            const x0 = if (x >= khalf) x - khalf else 0;
            const x1 = @min(wi - 1, x + khalf);
            var any_bg = false;
            var any_edge = false;
            var yy = y0;
            while (yy <= y1) : (yy += 1) {
                var xx = x0;
                while (xx <= x1) : (xx += 1) {
                    if (visible[yy * wi + xx] == 0) any_bg = true;
                    if (sketch[yy * wi + xx] > 0) any_edge = true;
                }
            }
            eroded[y * wi + x] = if (any_bg) 0 else 1;
            dilated[y * wi + x] = if (any_edge) 1 else 0;
        }
    }
    for (0..visible.len) |i| {
        visible[i] = eroded[i] * (if (dilated[i] < 0.5) @as(f32, 1) else 0);
        sketch[i] = dilated[i];
    }
}

pub const BakeResult = struct {
    /// tex_size*tex_size*channels merged atlas.
    atlas: []f32,
    /// tex_size*tex_size — true where any view painted (trust > 1e-8).
    mask: []bool,
    tex_size: u32,
    channels: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *BakeResult) void {
        self.allocator.free(self.atlas);
        self.allocator.free(self.mask);
    }
};

/// fast_bake_texture port. Blends each view's texture with weight `weight * cos^exp`,
/// accumulating `Σ tex·w / Σ w` into the atlas. A view whose painted texels are >99% already
/// covered contributes nothing (skipped). `exp` is the reference bake_exp (4 → cos⁴).
pub fn bakeBlend(
    allocator: std.mem.Allocator,
    views: []const ViewProjection,
    exp: f32,
    tex_size: u32,
    channels: usize,
) !BakeResult {
    const num_texels = @as(usize, tex_size) * @as(usize, tex_size);
    const merge = try allocator.alloc(f32, num_texels * channels);
    defer allocator.free(merge);
    @memset(merge, 0);
    const trust = try allocator.alloc(f32, num_texels);
    defer allocator.free(trust);
    @memset(trust, 0);

    for (views) |vp| {
        // view_sum / painted_sum use the WEIGHTED cos>0 set, which equals cos>0 (weight>0,
        // cos^exp>0 iff cos>0).
        var view_sum: usize = 0;
        var painted_sum: usize = 0;
        for (0..num_texels) |i| {
            if (vp.cos[i] > 0) {
                view_sum += 1;
                if (trust[i] > 0) painted_sum += 1;
            }
        }
        if (view_sum > 0) {
            const ratio = @as(f32, @floatFromInt(painted_sum)) / @as(f32, @floatFromInt(view_sum));
            if (ratio > 0.99) continue;
        }
        for (0..num_texels) |i| {
            if (vp.cos[i] <= 0) continue;
            const wcos = vp.weight * std.math.pow(f32, vp.cos[i], exp);
            for (0..channels) |ch| merge[i * channels + ch] += vp.texture[i * channels + ch] * wcos;
            trust[i] += wcos;
        }
    }

    const atlas = try allocator.alloc(f32, num_texels * channels);
    errdefer allocator.free(atlas);
    const mask = try allocator.alloc(bool, num_texels);
    errdefer allocator.free(mask);
    for (0..num_texels) |i| {
        const denom = @max(trust[i], 1e-8);
        for (0..channels) |ch| atlas[i * channels + ch] = merge[i * channels + ch] / denom;
        mask[i] = trust[i] > 1e-8;
    }

    return .{ .atlas = atlas, .mask = mask, .tex_size = tex_size, .channels = channels, .allocator = allocator };
}

// =============================================================================
// Tests (hermetic, no MLX). Run: `zig test src/bake.zig`
// =============================================================================

const testing = std.testing;

test "1: camera golden matrices (get_mv_matrix / get_orthographic_projection_matrix)" {
    // --- mv(azim=0, elev=0, d=1.45), derived by hand from camera_utils.py ---
    // elev=-0=0, azim=0+90=90. cam=[1.45·cos0·cos90, 1.45·cos0·sin90, 1.45·sin0]=[0,1.45,0].
    // lookat=normalize(origin-cam)=[0,-1,0]; up0=[0,0,1];
    // right=normalize(cross(lookat,up0))=cross([0,-1,0],[0,0,1])=[-1,0,0];
    // up=normalize(cross(right,lookat))=cross([-1,0,0],[0,-1,0])=[0,0,1];
    // w2c rows=[right;up;-lookat]=[[-1,0,0],[0,0,1],[0,1,0]];
    // t=[-right·cam,-up·cam,-(-lookat)·cam]=[0,0,-1.45].
    const m0 = mvMatrix(0, 0, 1.45);
    const exp0 = [16]f32{
        -1, 0, 0, 0,
        0,  0, 1, 0,
        0,  1, 0, -1.45,
        0,  0, 0, 1,
    };
    for (0..16) |i| try testing.expectApproxEqAbs(exp0[i], m0[i], 1e-5);

    // --- mv(azim=90, elev=45, d=1.45), derived by hand ---
    // elev=-45, azim=180. cos(-45)=√2/2, sin(-45)=-√2/2, cos180=-1, sin180=0.
    // cam=[1.45·(√2/2)·(-1), 0, 1.45·(-√2/2)]=[-1.0253048,0,-1.0253048].
    // lookat=normalize([1.0253048,0,1.0253048])=[√2/2,0,√2/2];
    // right=normalize(cross(lookat,[0,0,1]))=[0,-1,0];
    // up=normalize(cross([0,-1,0],lookat))=[-√2/2,0,√2/2];
    // rows=[right;up;-lookat]; t=[0,0,-1.45].
    const s = @sqrt(2.0) / 2.0;
    const m1 = mvMatrix(90, 45, 1.45);
    const exp1 = [16]f32{
        0,  -1, 0,  0,
        -s, 0,  s,  0,
        -s, 0,  -s, -1.45,
        0,  0,  0,  1,
    };
    for (0..16) |i| try testing.expectApproxEqAbs(exp1[i], m1[i], 1e-5);

    // --- ortho(scale=1.2, near=0.1, far=100) ---
    // m00=m11=2/1.2=1.6666667; m22=-2/99.9=-0.02002002; m23=-100.1/99.9=-1.002002; m33=1.
    const o = orthoMatrix(1.2, 0.1, 100.0);
    const expo = [16]f32{
        1.6666667, 0,         0,           0,
        0,         1.6666667, 0,           0,
        0,         0,         -0.02002002, -1.002002,
        0,         0,         0,           1,
    };
    for (0..16) |i| try testing.expectApproxEqAbs(expo[i], o[i], 1e-5);
}

test "2: defaultViews matches the fixed 6-view table exactly" {
    const vs = defaultViews();
    const azims = [6]f32{ 0, 90, 180, 270, 0, 180 };
    const elevs = [6]f32{ 0, 0, 0, 0, 90, -90 };
    const weights = [6]f32{ 1, 0.1, 0.5, 0.1, 0.05, 0.05 };
    for (0..6) |i| {
        try testing.expectEqual(azims[i], vs[i].azim);
        try testing.expectEqual(elevs[i], vs[i].elev);
        try testing.expectEqual(weights[i], vs[i].weight);
    }
}

test "3: prepareMesh — axis remap, v-flip, auto-center + 1.15 normalization" {
    const a = testing.allocator;
    // Unit-cube-of-side-2 with the corner at the origin: x,y,z ∈ {0, 2}.
    var pos: [24]f32 = undefined;
    var uv: [16]f32 = undefined;
    var k: usize = 0;
    for ([_]f32{ 0, 2 }) |x| {
        for ([_]f32{ 0, 2 }) |y| {
            for ([_]f32{ 0, 2 }) |z| {
                pos[k * 3] = x;
                pos[k * 3 + 1] = y;
                pos[k * 3 + 2] = z;
                uv[k * 2] = 0.3;
                uv[k * 2 + 1] = 0.7;
                k += 1;
            }
        }
    }
    const idx = [_]u32{ 0, 1, 2 }; // one dummy face; only positions/uv matter for this test
    var geom = try prepareMesh(a, &pos, &uv, &idx, &idx, 1.15);
    defer geom.deinit();

    // remap (x,y,z)->(-x,z,-y): the cube spans x'∈[-2,0], y'∈[0,2], z'∈[-2,0]; center=(-1,1,-1);
    // scale = √3·2 = 3.4641016; factor = 1.15/3.4641016 = 0.331992.
    // Input corner (0,0,0) -> remapped (0,0,0) -> (0-(-1),0-1,0-(-1))·factor = (1,-1,1)·0.331992.
    const factor: f32 = 1.15 / (@sqrt(3.0) * 2.0);
    // find the vertex whose original coords were (0,0,0): it is index 0 (x=0,y=0,z=0).
    try testing.expectApproxEqAbs(@as(f32, 1) * factor, geom.positions[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1) * factor, geom.positions[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1) * factor, geom.positions[2], 1e-5);

    // Every corner sits at distance √3·factor = 1.15/2 = 0.575 from the (new) origin.
    var max_r: f32 = 0;
    for (0..8) |vtx| {
        const p = vec3(geom.positions, vtx);
        max_r = @max(max_r, @sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]));
    }
    try testing.expectApproxEqAbs(@as(f32, 0.575), max_r, 1e-5);
    try testing.expectEqual(@as(f32, 1.15), geom.scale_factor);

    // v-flip: 0.7 -> 0.3.
    try testing.expectApproxEqAbs(@as(f32, 0.3), geom.uvs[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.3), geom.uvs[1], 1e-6);
}

test "4: renderGeometryMaps — bg exactly 1.0; covered = analytic (n+1)/2 and 0.5-pos/sf" {
    const a = testing.allocator;
    // A quad in the world xz-plane at y=0, centered at origin, half-size 0.3. Facing the
    // azim=0/elev=0 camera (at world +y). Wind it so its world normal is (0,1,0) toward the
    // camera; (n+1)/2 = (0.5, 1.0, 0.5).
    const h: f32 = 0.3;
    const positions = [_]f32{
        -h, 0, -h,
        h,  0, -h,
        h,  0, h,
        -h, 0, h,
    };
    // Triangles wound for +y normal: (0,3,2) and (0,2,1).
    const indices = [_]u32{ 0, 3, 2, 0, 2, 1 };
    const uvs = [_]f32{ 0, 0, 1, 0, 1, 1, 0, 1 };
    var geom = try buildMeshGeom(a, &positions, &uvs, &indices, &indices, 1.15);
    defer geom.deinit();
    // Sanity: face normal is +y.
    try testing.expectApproxEqAbs(@as(f32, 0), geom.face_normals[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), geom.face_normals[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), geom.face_normals[2], 1e-6);

    const size: u32 = 48;
    var maps = try renderGeometryMaps(a, &geom, .{ .azim = 0, .elev = 0, .weight = 1 }, size, .{});
    defer maps.deinit();

    var covered: usize = 0;
    var background: usize = 0;
    // center pixel (world origin projects to image center) — probe position there.
    var best_pix: usize = 0;
    var best_d: f32 = std.math.floatMax(f32);
    const cxf: f32 = @as(f32, @floatFromInt(size)) * 0.5;

    for (0..size * size) |pix| {
        if (maps.mask[pix] > 0) {
            covered += 1;
            // covered normal == (0.5, 1.0, 0.5).
            try testing.expectApproxEqAbs(@as(f32, 0.5), maps.normal[pix * 3], 1e-5);
            try testing.expectApproxEqAbs(@as(f32, 1.0), maps.normal[pix * 3 + 1], 1e-5);
            try testing.expectApproxEqAbs(@as(f32, 0.5), maps.normal[pix * 3 + 2], 1e-5);
            const px: f32 = @floatFromInt(pix % size);
            const py: f32 = @floatFromInt(pix / size);
            const dd = (px - cxf) * (px - cxf) + (py - cxf) * (py - cxf);
            if (dd < best_d) {
                best_d = dd;
                best_pix = pix;
            }
        } else {
            background += 1;
            // background EXACTLY 1.0 in all channels of both maps.
            for (0..3) |c| {
                try testing.expectEqual(@as(f32, 1.0), maps.normal[pix * 3 + c]);
                try testing.expectEqual(@as(f32, 1.0), maps.position[pix * 3 + c]);
            }
        }
    }
    try testing.expect(covered > 0);
    try testing.expect(background > 0);

    // At the quad center the interpolated world pos ≈ (0,0,0) -> tex_position = 0.5 - 0 = 0.5.
    try testing.expectApproxEqAbs(@as(f32, 0.5), maps.position[best_pix * 3], 2e-2);
    try testing.expectApproxEqAbs(@as(f32, 0.5), maps.position[best_pix * 3 + 1], 2e-2);
    try testing.expectApproxEqAbs(@as(f32, 0.5), maps.position[best_pix * 3 + 2], 2e-2);
}

// ---- Cube builder shared by test 5 (helper kept local to the test section) ----

/// Build a cube (side 2·r, centered at origin, 24 verts / 12 tris) with INWARD-facing
/// normals and a distinct UV island per face. Inward winding is required for the reference
/// bake logic: the axis remap in a real pipeline flips triangle orientation (det -1), so the
/// camera-facing surface's camera-space normal is -z (cos>0). We build directly in render
/// space, so we wind inward here to reproduce that. Faces are ordered
/// [+x,-x,+y,-y,+z,-z]; each gets a UV square in a 3×2 grid.
fn buildTestCube(a: std.mem.Allocator, r: f32) !MeshGeom {
    // 6 faces × 4 corners. For a face at +axis (outward = +axis), inward normal = -axis.
    // Corner order chosen so cross(v1-v0, v2-v0) points INWARD (toward origin).
    var positions: [72]f32 = undefined; // 24 * 3
    var uvs: [48]f32 = undefined; // 24 * 2
    var indices: [36]u32 = undefined; // 12 * 3

    // Each face defined by 4 corners a,b,c,d wound so the first triangle (a,b,c) has an
    // inward normal. Faces: +x,-x,+y,-y,+z,-z.
    const faces = [6][4][3]f32{
        // +x face (x=+r): inward normal -x. Corners around x=+r.
        .{ .{ r, -r, -r }, .{ r, -r, r }, .{ r, r, r }, .{ r, r, -r } },
        // -x face (x=-r): inward normal +x.
        .{ .{ -r, -r, -r }, .{ -r, r, -r }, .{ -r, r, r }, .{ -r, -r, r } },
        // +y face (y=+r): inward normal -y.
        .{ .{ -r, r, -r }, .{ r, r, -r }, .{ r, r, r }, .{ -r, r, r } },
        // -y face (y=-r): inward normal +y.
        .{ .{ -r, -r, -r }, .{ -r, -r, r }, .{ r, -r, r }, .{ r, -r, -r } },
        // +z face (z=+r): inward normal -z.
        .{ .{ -r, -r, r }, .{ -r, r, r }, .{ r, r, r }, .{ r, -r, r } },
        // -z face (z=-r): inward normal +z.
        .{ .{ -r, -r, -r }, .{ r, -r, -r }, .{ r, r, -r }, .{ -r, r, -r } },
    };

    var vbase: usize = 0;
    for (faces, 0..) |corners, fi| {
        const col: f32 = @floatFromInt(fi % 3);
        const row: f32 = @floatFromInt(fi / 3);
        const m: f32 = 0.04; // island margin
        const us0 = col / 3.0 + m;
        const us1 = (col + 1.0) / 3.0 - m;
        const vs0 = row / 2.0 + m;
        const vs1 = (row + 1.0) / 2.0 - m;
        const uv_corner = [4][2]f32{ .{ us0, vs0 }, .{ us1, vs0 }, .{ us1, vs1 }, .{ us0, vs1 } };
        for (0..4) |ci| {
            positions[(vbase + ci) * 3] = corners[ci][0];
            positions[(vbase + ci) * 3 + 1] = corners[ci][1];
            positions[(vbase + ci) * 3 + 2] = corners[ci][2];
            uvs[(vbase + ci) * 2] = uv_corner[ci][0];
            uvs[(vbase + ci) * 2 + 1] = uv_corner[ci][1];
        }
        // two triangles (a,b,c) (a,c,d)
        indices[fi * 6] = @intCast(vbase);
        indices[fi * 6 + 1] = @intCast(vbase + 1);
        indices[fi * 6 + 2] = @intCast(vbase + 2);
        indices[fi * 6 + 3] = @intCast(vbase);
        indices[fi * 6 + 4] = @intCast(vbase + 2);
        indices[fi * 6 + 5] = @intCast(vbase + 3);
        vbase += 4;
    }
    return buildMeshGeom(a, &positions, &uvs, &indices, &indices, 1.15);
}

test "5: axis-colored cube mini-bake — each face's island gets its color, no bleeding" {
    const a = testing.allocator;
    const r: f32 = 0.35;
    var geom = try buildTestCube(a, r);
    defer geom.deinit();

    // Confirm inward winding: the +x face (face 0, triangle 0) must have normal -x = (-1,0,0).
    try testing.expectApproxEqAbs(@as(f32, -1), geom.face_normals[0], 1e-5);
    // -y face (face 3) has inward normal +y. Two triangles per face -> face 3 = triangle 6;
    // its normal.y is face_normals[6*3 + 1] = face_normals[19].
    try testing.expectApproxEqAbs(@as(f32, 1), geom.face_normals[19], 1e-5);

    const tex: u32 = 128;
    var tiles = try extractTextiles(a, &geom, tex);
    defer tiles.deinit();
    try testing.expect(tiles.count > 0);

    // Face colors (index by face 0..5 = +x,-x,+y,-y,+z,-z).
    const colors = [6][3]f32{
        .{ 1, 0, 0 }, // +x red
        .{ 0, 1, 1 }, // -x cyan
        .{ 0, 1, 0 }, // +y green
        .{ 1, 0, 1 }, // -y magenta
        .{ 0, 0, 1 }, // +z blue
        .{ 1, 1, 0 }, // -z yellow
    };
    // View i (defaultViews order) looks from a direction and bakes exactly one face:
    //   view0 azim0   -> camera +y -> +y face (index 2)
    //   view1 azim90  -> camera -x -> -x face (index 1)
    //   view2 azim180 -> camera -y -> -y face (index 3)
    //   view3 azim270 -> camera +x -> +x face (index 0)
    //   view4 elev90  -> camera -z -> -z face (index 5)
    //   view5 elev-90 -> camera +z -> +z face (index 4)
    const view_face = [6]usize{ 2, 1, 3, 0, 5, 4 };
    const views = defaultViews();

    const render: u32 = 64;
    var projections: [6]ViewProjection = undefined;
    var made: usize = 0;
    defer for (0..made) |i| projections[i].deinit();

    for (0..6) |i| {
        // Decoded image for this view = the flat color of the face it sees.
        const col = colors[view_face[i]];
        const img = try a.alloc(f32, render * render * 3);
        defer a.free(img);
        var pxi: usize = 0;
        while (pxi < render * render) : (pxi += 1) {
            img[pxi * 3] = col[0];
            img[pxi * 3 + 1] = col[1];
            img[pxi * 3 + 2] = col[2];
        }
        projections[i] = try backProject(a, &geom, &tiles, views[i], img, render, render, 3, .{});
        made += 1;
    }

    var baked = try bakeBlend(a, projections[0..], 4.0, tex, 3);
    defer baked.deinit();

    // For each face, probe the atlas texel at its UV-island center. It must hold that face's
    // color; and NO texel of one face's island may carry a different face's color (no bleed).
    for (0..6) |fi| {
        const col: f32 = @floatFromInt(fi % 3);
        const row: f32 = @floatFromInt(fi / 3);
        const uc = (col + 0.5) / 3.0;
        const vc = (row + 0.5) / 2.0;
        // UV (uc,vc) -> texel: rasterize maps clip x=2u-1 to column; the atlas index used in
        // extract/back_project is (row_i * tex + col_j) with row_i = texel row = pix/tex.
        // The UV-space raster's screen y grows with v (no flip), so row_i ≈ vc*(tex-1).
        const cj: usize = @intFromFloat(uc * @as(f32, @floatFromInt(tex - 1)));
        const ri: usize = @intFromFloat(vc * @as(f32, @floatFromInt(tex - 1)));
        const ti = ri * tex + cj;
        try testing.expect(baked.mask[ti]);
        try testing.expectApproxEqAbs(colors[fi][0], baked.atlas[ti * 3], 1e-3);
        try testing.expectApproxEqAbs(colors[fi][1], baked.atlas[ti * 3 + 1], 1e-3);
        try testing.expectApproxEqAbs(colors[fi][2], baked.atlas[ti * 3 + 2], 1e-3);
    }
}

test "6: depth occlusion — the hidden quad receives no color from a view where it is occluded" {
    const a = testing.allocator;
    // Two quads facing the azim=0/elev=0 camera (at +y), both spanning the same x,z extent so
    // the near one fully occludes the far one. Inward normal (-y) so they bake in view0.
    const h: f32 = 0.3;
    // near quad at y = +0.3 (nearer to camera), far quad at y = -0.1.
    const yn: f32 = 0.3;
    const yf: f32 = -0.1;
    // 8 verts (4 per quad). Wind for inward normal (-y) — same corner order as test 5's +y
    // face: (-h,y,-h),(h,y,-h),(h,y,h),(-h,y,h) gives cross(v1-v0,v2-v0) = (0,-1,0).
    const positions = [_]f32{
        // near quad (verts 0..3)
        -h, yn, -h,
        h,  yn, -h,
        h,  yn, h,
        -h, yn, h,
        // far quad (verts 4..7)
        -h, yf, -h,
        h,  yf, -h,
        h,  yf, h,
        -h, yf, h,
    };
    const indices = [_]u32{ 0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7 };
    // Near quad island: u∈[0.05,0.45]; far quad island: u∈[0.55,0.95]; both v∈[0.05,0.95].
    const uvs = [_]f32{
        0.05, 0.05,
        0.05, 0.95,
        0.45, 0.95,
        0.45, 0.05,
        0.55, 0.05,
        0.55, 0.95,
        0.95, 0.95,
        0.95, 0.05,
    };
    var geom = try buildMeshGeom(a, &positions, &uvs, &indices, &indices, 1.15);
    defer geom.deinit();
    // Both quads must have inward normal (0,-1,0). near quad = triangles 0,1; far quad = 2,3.
    try testing.expectApproxEqAbs(@as(f32, -1), geom.face_normals[1], 1e-5); // near triangle 0 normal.y
    try testing.expectApproxEqAbs(@as(f32, -1), geom.face_normals[2 * 3 + 1], 1e-5); // far triangle 2 normal.y

    const tex: u32 = 128;
    var tiles = try extractTextiles(a, &geom, tex);
    defer tiles.deinit();

    const render: u32 = 64;
    const img = try a.alloc(f32, render * render * 3);
    defer a.free(img);
    for (0..render * render) |i| {
        img[i * 3] = 1.0; // red
        img[i * 3 + 1] = 0.0;
        img[i * 3 + 2] = 0.0;
    }

    var vp = try backProject(a, &geom, &tiles, .{ .azim = 0, .elev = 0, .weight = 1 }, img, render, render, 3, .{});
    defer vp.deinit();
    var projections = [_]ViewProjection{vp};
    var baked = try bakeBlend(a, projections[0..], 4.0, tex, 3);
    defer baked.deinit();

    // Count painted texels in each island.
    var near_painted: usize = 0;
    var far_painted: usize = 0;
    for (0..tex) |ri| {
        for (0..tex) |cj| {
            const u = @as(f32, @floatFromInt(cj)) / @as(f32, @floatFromInt(tex - 1));
            const ti = ri * tex + cj;
            if (!baked.mask[ti]) continue;
            if (u >= 0.05 and u <= 0.45) {
                near_painted += 1;
                // near quad painted red.
                try testing.expectApproxEqAbs(@as(f32, 1.0), baked.atlas[ti * 3], 1e-3);
            } else if (u >= 0.55 and u <= 0.95) {
                far_painted += 1;
            }
        }
    }
    // The near (occluding) quad is painted; the far (occluded) quad gets NOTHING.
    try testing.expect(near_painted > 0);
    try testing.expectEqual(@as(usize, 0), far_painted);
}

test "7: depth-edge detector flags an obvious discontinuity, ignores flat regions" {
    const a = testing.allocator;
    const w: u32 = 16;
    const h: u32 = 16;
    // Left half depth 0.0, right half depth 1.0 -> a vertical step at x=8.
    const depth = try a.alloc(f32, w * h);
    defer a.free(depth);
    for (0..h) |y| {
        for (0..w) |x| depth[y * w + x] = if (x >= 8) @as(f32, 1.0) else 0.0;
    }
    const edges = try detectDepthEdges(a, depth, w, h);
    defer a.free(edges);

    // The step column (x=7 or 8, interior rows) must be flagged; a flat interior column must not.
    var step_flags: usize = 0;
    for (1..h - 1) |y| {
        if (edges[y * w + 7] > 0 or edges[y * w + 8] > 0) step_flags += 1;
    }
    try testing.expect(step_flags > 0);
    // A column deep in the flat left region is never an edge.
    for (1..h - 1) |y| try testing.expectEqual(@as(f32, 0), edges[y * w + 3]);

    // A perfectly flat depth map has no edges anywhere.
    const flat = try a.alloc(f32, w * h);
    defer a.free(flat);
    @memset(flat, 0.5);
    const flat_edges = try detectDepthEdges(a, flat, w, h);
    defer a.free(flat_edges);
    for (flat_edges) |e| try testing.expectEqual(@as(f32, 0), e);
}

test "8: boundary erosion shrinks the mask and dilates edges" {
    const a = testing.allocator;
    const w: u32 = 9;
    const h: u32 = 9;
    // Fully-foreground mask; a single edge pixel at the center.
    const visible = try a.alloc(f32, w * h);
    defer a.free(visible);
    @memset(visible, 1.0);
    const sketch = try a.alloc(f32, w * h);
    defer a.free(sketch);
    @memset(sketch, 0.0);
    sketch[4 * w + 4] = 1.0;

    try applyBoundaryErosion(a, visible, sketch, w, h, 1); // khalf=1 -> 3×3

    // Erosion: interior pixels whose 3×3 window is all-foreground survive; the true border
    // ring touches nothing off-image here (all foreground), so a fully-foreground mask stays 1
    // in the interior. The dilated single edge zeroes the mask in its 3×3 neighborhood.
    for (1..h - 1) |y| {
        for (1..w - 1) |x| {
            const near_edge = (x >= 3 and x <= 5 and y >= 3 and y <= 5);
            if (near_edge) {
                try testing.expectEqual(@as(f32, 0), visible[y * w + x]);
            } else {
                try testing.expectEqual(@as(f32, 1), visible[y * w + x]);
            }
        }
    }
    // Dilated sketch is 1 over the 3×3 around the center.
    try testing.expectEqual(@as(f32, 1), sketch[3 * w + 3]);
    try testing.expectEqual(@as(f32, 1), sketch[5 * w + 5]);
    try testing.expectEqual(@as(f32, 0), sketch[0]);
}
