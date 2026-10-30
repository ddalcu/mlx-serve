//! Pixal3D (TencentARC, MIT; `TencentARC/Pixal3D`): TRELLIS.2's cascade (`trellis2_pipeline.zig`) with every flow
//! also reading the photo pixel by pixel. Each voxel token is projected through a pinhole camera onto the DINOv3
//! patch map (and, past the sparse structure, onto its NAF upsampling, `naf.zig`), and every DiT block adds a
//! projection of what it lands on to its cross-attention output (`trellis2.Proj`); the cross-attention itself reads
//! only the CLS and register tokens. Multi-view: posed photos through the `_mv` checkpoints of `pipeline_mv.json`,
//! each projected through its own camera, the features averaged over views.
//!
//! The camera is a horizontal FOV from the request (the reference estimates it with MoGe-2, not ported here).
//! Pack: the published layout plus `dinov3/` (DINOv3 ViT-L/16, transformers layout) and `naf/model.safetensors`
//! (valeoai's `naf_release.pth` as safetensors). Parity: `pixal3d oracle` tests (tests/dump_pixal3d_fixtures.py).

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const scope_mod = @import("../../mlx/scope.zig");
const sse = @import("../../sdk/media/progress.zig");
const qvis = @import("../../sdk/vision/qwen_vision.zig");
const t2 = @import("../trellis2/trellis2.zig");
const tp = @import("../trellis2/pipeline.zig");
const dino = @import("../../sdk/vision/dinov3.zig");
const naf_mod = @import("naf.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const Scope = scope_mod.Scope;

pub const Mat4 = [4][4]f32;

/// `get_proj_cond_*`'s horizontal FOV (radians): the single view's when the request names none.
pub const DEFAULT_FOV: f32 = 0.8575560450553894;
/// The shipped four-view rig (`assets/mv_images/example/transforms.json`): the front view, then every 90 degrees
/// about z at eye level, with this FOV.
pub const RIG_FOV: f32 = 0.3490658503988659;
const RIG_D: f32 = 3.1192049980163574;
pub const RIG = [4]Mat4{
    .{ .{ 1, 0, 0, 0 }, .{ 0, 0, -1, -RIG_D }, .{ 0, 1, 0, 0 }, .{ 0, 0, 0, 1 } },
    .{ .{ 0, 0, 1, RIG_D }, .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 0, 1 } },
    .{ .{ -1, 0, 0, 0 }, .{ 0, 0, 1, RIG_D }, .{ 0, 1, 0, 0 }, .{ 0, 0, 0, 1 } },
    .{ .{ 0, 0, -1, -RIG_D }, .{ -1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 0, 1 } },
};
pub const MAX_VIEWS = 8;
/// Every stage's DINOv3 input sizes: the sparse structure and the 512 shape flow read 512 px, the rest 1024.
const SIZES = [2]u32{ 512, 1024 };

// ── cameras ─────────────────────────────────────────────────────────────────

/// `ProjGrid.front_view_transform_matrix` at distance `d`: the camera at (0, -d, 0) looking along +y, z up.
pub fn frontView(d: f32) Mat4 {
    return .{ .{ 1, 0, 0, 0 }, .{ 0, 0, -1, -d }, .{ 0, 1, 0, 0 }, .{ 0, 0, 0, 1 } };
}

/// inference.py's `distance_from_fov` for a 512 px view: the distance at which the unit cube's x extent spans the
/// frame, so the crop around the subject and the camera agree.
pub fn distanceFromFov(fov: f32) f32 {
    const f_pixels = 16.0 / @tan(fov / 2.0) * 512.0 / 32.0;
    return @floatCast(@as(f64, f_pixels) * -0.5 / -256.0);
}

fn matMul(a: Mat4, b: Mat4) Mat4 {
    var o: Mat4 = undefined;
    for (0..4) |i| for (0..4) |j| {
        var acc: f32 = 0;
        for (0..4) |k| acc += a[i][k] * b[k][j];
        o[i][j] = acc;
    };
    return o;
}

/// General 4x4 inverse (partial pivoting, f64), null when singular.
pub fn invert(m: Mat4) ?Mat4 {
    var a: [4][8]f64 = undefined;
    for (0..4) |i| for (0..8) |j| {
        a[i][j] = if (j < 4) m[i][j] else @floatFromInt(@intFromBool(j - 4 == i));
    };
    for (0..4) |c| {
        var p = c;
        for (c + 1..4) |r| if (@abs(a[r][c]) > @abs(a[p][c])) {
            p = r;
        };
        if (@abs(a[p][c]) < 1e-9) return null;
        std.mem.swap([8]f64, &a[c], &a[p]);
        const inv = 1.0 / a[c][c];
        for (&a[c]) |*v| v.* *= inv;
        for (0..4) |r| {
            if (r == c) continue;
            const f = a[r][c];
            for (0..8) |j| a[r][j] -= f * a[c][j];
        }
    }
    var o: Mat4 = undefined;
    for (0..4) |i| for (0..4) |j| {
        o[i][j] = @floatCast(a[i][4 + j]);
    };
    return o;
}

/// `compute_relative_calc_mat`: each camera-to-world `c2w[i]` re-expressed so the first view sits at the canonical
/// front pose at its own distance, `F @ inv(C_0) @ C_i`. Null when the first camera is singular.
pub fn viewMatrices(out: []Mat4, c2w: []const Mat4) ?void {
    const t = c2w[0];
    const d: f32 = @sqrt(t[0][3] * t[0][3] + t[1][3] * t[1][3] + t[2][3] * t[2][3]);
    const inv0 = invert(t) orelse return null;
    for (out, c2w) |*o, c| o.* = matMul(frontView(d), matMul(inv0, c));
}

/// `torch.linspace(-1, 1, n)[i]` in f32, the reference's own two-sided formula.
fn linspace(i: usize, n: usize) f32 {
    const step: f32 = 2.0 / @as(f32, @floatFromInt(n - 1));
    return if (i < n / 2) -1.0 + step * @as(f32, @floatFromInt(i)) else 1.0 - step * @as(f32, @floatFromInt(n - 1 - i));
}

/// The world point (Blender axes, the unit cube) of lattice cell `c` (x, y, z) of a `grid`^3 `ProjGrid`: its
/// linspace lattice rotated by (x, y, z) -> (x, -z, y) and halved.
pub fn cellPoint(c: [3]i32, grid: usize) [3]f32 {
    const x = linspace(@intCast(c[0]), grid);
    const y = linspace(@intCast(c[1]), grid);
    const z = linspace(@intCast(c[2]), grid);
    return .{ x / 2, -z / 2, y / 2 };
}

/// A view as the projection reads it: world-to-camera and the focal length in pixels at a given image size.
pub const Camera = struct {
    w2c: Mat4,
    fov: f32,

    pub fn focal(self: Camera, res: u32) f32 {
        return 16.0 / @tan(self.fov / 2.0) * @as(f32, @floatFromInt(res)) / 32.0;
    }

    /// `project_points_to_image_batch`: pixel position (x right, y down) of world point `p` on a `res` px image
    /// (the camera looks down its -z with y up, Blender's convention).
    pub fn project(self: Camera, res: u32, p: [3]f32) [2]f32 {
        const m = self.w2c;
        var cam: [3]f32 = undefined;
        for (0..3) |i| cam[i] = p[0] * m[i][0] + p[1] * m[i][1] + p[2] * m[i][2] + m[i][3];
        const f = self.focal(res);
        const zd = -cam[2] + 1e-8;
        const half = @as(f32, @floatFromInt(res)) / 2.0;
        return .{ f * cam[0] / zd + half, -(f * cam[1] / zd) + half };
    }
};

/// The cameras of `n` views: the single view's front pose at the distance its FOV implies, or each multi-view
/// camera-to-world relative to the first (`c2w` empty = the shipped rig's first `n` poses). Null when singular.
pub fn camerasFor(out: []Camera, fov: f32, c2w: []const Mat4) ?void {
    if (out.len == 1 and c2w.len == 0) {
        out[0] = .{ .w2c = invert(frontView(distanceFromFov(fov))).?, .fov = fov };
        return;
    }
    var calc: [MAX_VIEWS]Mat4 = undefined;
    viewMatrices(calc[0..out.len], if (c2w.len == 0) RIG[0..out.len] else c2w) orelse return null;
    for (out, calc[0..out.len]) |*o, c| o.* = .{ .w2c = invert(c) orelse return null, .fov = fov };
}

/// The 400 for a view count the request cannot have, or null.
pub fn viewsRefusal(views: usize, cameras: usize, has_mv: bool) ?[]const u8 {
    if (views > MAX_VIEWS) return std.fmt.comptimePrint("'images' takes at most {d} views", .{MAX_VIEWS});
    if (views > 1 and !has_mv) return "this Pixal3D pack has no multi-view models (pipeline_mv.json); send one 'image'";
    if (cameras != 0 and cameras != views) return "'cameras' needs one 4x4 matrix per entry of 'images'";
    if (cameras == 0 and views > RIG.len) return std.fmt.comptimePrint("more than {d} views need 'cameras' (the default rig has {d} poses)", .{ RIG.len, RIG.len });
    return null;
}

pub const CamerasParse = union(enum) { ok: []Mat4, bad: []const u8 };

/// `cameras`: an array of 4x4 camera-to-world matrices (row-major nested arrays, Blender axes). Caller frees `ok`.
pub fn parseCameras(alloc: std.mem.Allocator, v: std.json.Value) !CamerasParse {
    if (v != .array) return .{ .bad = BAD_CAMERAS };
    const out = try alloc.alloc(Mat4, v.array.items.len);
    for (v.array.items, out) |m, *o| {
        o.* = matrixOf(m) orelse {
            alloc.free(out);
            return .{ .bad = BAD_CAMERAS };
        };
        if (invert(o.*) == null) {
            alloc.free(out);
            return .{ .bad = "'cameras' holds a singular matrix" };
        }
    }
    return .{ .ok = out };
}

pub const BAD_CAMERAS = "'cameras' must be an array of 4x4 camera-to-world matrices";

fn matrixOf(m: std.json.Value) ?Mat4 {
    if (m != .array or m.array.items.len != 4) return null;
    var o: Mat4 = undefined;
    for (m.array.items, 0..) |row, i| {
        if (row != .array or row.array.items.len != 4) return null;
        for (row.array.items, 0..) |e, j| o[i][j] = switch (e) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return null,
        };
    }
    return o;
}

// ── view images ─────────────────────────────────────────────────────────────

/// The single view at `size`: the subject's crop (`prepareImageMargin` at 1.1) Lanczos-resized, `[size^2 * 3]`
/// f32 in [0, 1]. Caller frees.
fn singleViewImage(alloc: std.mem.Allocator, prep: tp.Prepared, size: u32) ![]f32 {
    const px = try qvis.resizeInterleavedPil(alloc, prep.rgb, prep.side, prep.side, size, size, 3, .lanczos);
    defer alloc.free(px);
    const out = try alloc.alloc(f32, px.len);
    for (out, px) |*o, v| o.* = @as(f32, @floatFromInt(v)) / 255.0;
    return out;
}

/// inference_mv.py's `to_cond_tensor`: the RGBA view resized to `size` square (Pillow resamples premultiplied RGBa
/// and converts back, truncating; at the source's own size it copies), then its colour times its alpha. Never
/// cropped: the framing is the camera's.
pub fn multiViewImage(alloc: std.mem.Allocator, rgba: []const u8, w: u32, h: u32, size: u32) ![]f32 {
    const same = w == size and h == size;
    var owned: ?[]u8 = null;
    defer if (owned) |o| alloc.free(o);
    const px = if (same) rgba else blk: {
        const pre = try alloc.alloc(u8, rgba.len);
        defer alloc.free(pre);
        for (0..rgba.len / 4) |i| {
            const a: u32 = rgba[i * 4 + 3];
            for (0..3) |c| {
                const t = @as(u32, rgba[i * 4 + c]) * a + 128;
                pre[i * 4 + c] = @intCast(((t >> 8) + t) >> 8);
            }
            pre[i * 4 + 3] = rgba[i * 4 + 3];
        }
        owned = try qvis.resizeInterleavedPil(alloc, pre, h, w, size, size, 4, .lanczos);
        break :blk owned.?;
    };
    const out = try alloc.alloc(f32, @as(usize, size) * size * 3);
    for (0..@as(usize, size) * size) |i| {
        const a: u32 = px[i * 4 + 3];
        const af = @as(f32, @floatFromInt(a)) / 255.0;
        for (0..3) |c| {
            const v: u32 = px[i * 4 + c];
            const straight: u32 = if (same or a == 0 or a == 255) v else @min(255, 255 * v / a);
            out[i * 3 + c] = @as(f32, @floatFromInt(straight)) / 255.0 * af;
        }
    }
    return out;
}

// ── conditioning ────────────────────────────────────────────────────────────

pub const Stage = enum {
    ss,
    shape_512,
    shape_1024,
    tex_1024,

    /// Index into `SIZES` of the stage's DINOv3 input.
    fn size(self: Stage) usize {
        return switch (self) {
            .ss, .shape_512 => 0,
            .shape_1024, .tex_1024 => 1,
        };
    }

    /// `naf_target_size`, or 0 for the sparse structure, which reads the patch map alone.
    fn nafSide(self: Stage) u32 {
        return switch (self) {
            .ss => 0,
            .shape_512, .shape_1024 => 512,
            .tex_1024 => 1024,
        };
    }
};

const ViewFeats = struct {
    /// `[1, size, size, 3]` f32 in [0, 1] per input size: the NAF guide.
    image: [2]A,
    /// `[1, 1 + registers, hidden]`: what the cross-attention reads.
    global: [2]A,
    /// `[(size/16)^2, hidden]` row-major patch tokens.
    patches: [2]A,
    cam: Camera,

    fn deinit(self: *ViewFeats) void {
        inline for (.{ &self.image, &self.global, &self.patches }) |arr| for (arr) |a| {
            _ = mlx.mlx_array_free(a);
        };
    }
};

/// What every stage's conditioning reads: each view's DINOv3 tokens at both sizes, its colour and its camera.
pub const Conditioner = struct {
    alloc: std.mem.Allocator,
    s: S,
    naf: naf_mod.Naf,
    views: []ViewFeats,

    /// `images[v][k]`: view v at `SIZES[k]`, `[size^2 * 3]` f32 in [0, 1].
    pub fn init(io: std.Io, alloc: std.mem.Allocator, s: S, dino_dir: []const u8, naf_dir: []const u8, images: []const [2][]const f32, cams: []const Camera) !Conditioner {
        var naf = try naf_mod.Naf.open(alloc, naf_dir);
        errdefer naf.deinit();
        const views = try alloc.alloc(ViewFeats, images.len);
        var done: usize = 0;
        errdefer {
            for (views[0..done]) |*v| v.deinit();
            alloc.free(views);
        }
        var enc = try dino.Encoder.open(io, alloc, dino_dir);
        defer enc.deinit();
        const prefix: c_int = @intCast(1 + enc.cfg.registers);
        for (images, cams, views) |img, cam, *v| {
            // Counted as started with empty handles, so a failure inside the view frees what it already set.
            v.* = .{ .image = .{ mlx.mlx_array_new(), mlx.mlx_array_new() }, .global = .{ mlx.mlx_array_new(), mlx.mlx_array_new() }, .patches = .{ mlx.mlx_array_new(), mlx.mlx_array_new() }, .cam = cam };
            done += 1;
            for (SIZES, 0..) |size, k| {
                var sc = Scope.init(alloc, s);
                defer sc.deinit();
                const sz: c_int = @intCast(size);
                const x = try sc.fromF32(img[k], &.{ 1, sz, sz, 3 });
                const mean = try sc.fromF32(&dino.IMAGENET_MEAN, &.{3});
                const sd = try sc.fromF32(&dino.IMAGENET_STD, &.{3});
                const tokens = try enc.forward(&sc, try sc.div(try sc.sub(x, mean), sd));
                const l: c_int = mlx.getShape(tokens)[1];
                const hid: c_int = mlx.getShape(tokens)[2];
                const g = try sc.slice(tokens, 1, 0, prefix);
                const p = try sc.reshape(try sc.slice(tokens, 1, prefix, l), &.{ l - prefix, hid });
                inline for (.{ x, g, p }) |a| try mlx.check(mlx.mlx_array_eval(a));
                _ = mlx.mlx_array_free(v.image[k]);
                _ = mlx.mlx_array_free(v.global[k]);
                _ = mlx.mlx_array_free(v.patches[k]);
                v.image[k] = sc.out(x);
                v.global[k] = sc.out(g);
                v.patches[k] = sc.out(p);
            }
        }
        return .{ .alloc = alloc, .s = s, .naf = naf, .views = views };
    }

    pub fn deinit(self: *Conditioner) void {
        for (self.views) |*v| v.deinit();
        self.alloc.free(self.views);
        self.naf.deinit();
    }

    /// One stage's conditioning for the tokens at `coords` (`[n, 4]`) of a `grid`^3 lattice: the views' global
    /// tokens `[1, 5, hidden]` and their projected features `[1, n, C']` (C' doubled where NAF upsamples), each
    /// averaged over the views. Caller frees both.
    pub fn stage(self: *Conditioner, which: Stage, coords: []const i32, grid: u32) !tp.StageCond {
        const n = coords.len / 4;
        const k = which.size();
        const size = SIZES[k];
        const lr = size / 16;
        const world = try self.alloc.alloc([3]f32, n);
        defer self.alloc.free(world);
        for (world, 0..) |*p, t| p.* = cellPoint(coords[t * 4 + 1 ..][0..3].*, grid);
        const pts = try self.alloc.alloc([2]f32, n);
        defer self.alloc.free(pts);
        var sc = Scope.init(self.alloc, self.s);
        defer sc.deinit();
        var proj: ?A = null;
        var global: ?A = null;
        for (self.views) |*v| {
            for (pts, world) |*q, p| q.* = v.cam.project(size, p);
            var guide: ?naf_mod.Guide = if (which.nafSide() != 0) try self.naf.guide(self.alloc, self.s, v.image[k], which.nafSide(), lr) else null;
            defer if (guide) |*g| g.deinit();
            const f = try sc.keep(try naf_mod.samplePoints(self.alloc, self.s, pts, size, v.patches[k], lr, guide));
            proj = if (proj) |acc| try sc.add(acc, f) else f;
            global = if (global) |acc| try sc.add(acc, v.global[k]) else v.global[k];
            try mlx.check(mlx.mlx_array_eval(proj.?));
        }
        const vn: f32 = @floatFromInt(self.views.len);
        const c: c_int = mlx.getShape(proj.?)[1];
        const p = try sc.reshape(try sc.mulS(proj.?, 1.0 / vn), &.{ 1, @intCast(n), c });
        const g = try sc.mulS(global.?, 1.0 / vn);
        try mlx.check(mlx.mlx_array_eval(p));
        try mlx.check(mlx.mlx_array_eval(g));
        return .{ .global = sc.out(g), .proj = sc.out(p) };
    }
};

fn freeCond(c: tp.StageCond) void {
    _ = mlx.mlx_array_free(c.global);
    if (c.proj) |p| _ = mlx.mlx_array_free(p);
}

// ── engine ──────────────────────────────────────────────────────────────────

/// One input photo: straight-alpha RGBA8.
pub const Photo = struct { rgba: []const u8, w: u32, h: u32 };

pub const Opts = struct {
    base: tp.Opts = .{ .pixal3d_axes = true },
    /// Horizontal field of view in radians for every view; null = the reference default of the mode (the
    /// single view's `DEFAULT_FOV`, the rig's `RIG_FOV`).
    fov: ?f32 = null,
    /// Camera-to-world per view, multi-view only (Blender axes, the first the front view); empty = the rig.
    cameras: []const Mat4 = &.{},
};

const Flows = struct { ss: []const u8, shape_512: []const u8, shape_1024: []const u8, tex_1024: []const u8 };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// `pipeline.json`: the single-view flows, the decoders and latent statistics, the DINOv3 directory.
    base: *tp.Engine,
    naf_dir: []u8,
    /// `pipeline_mv.json`'s flows, where the pack ships them.
    mv: ?Flows,
    arena: std.heap.ArenaAllocator,

    /// On the caller's stream (borrowed).
    pub fn loadOn(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, s: S) !*Engine {
        const self = try gpa.create(Engine);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .base = undefined, .naf_dir = undefined, .mv = null, .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer self.arena.deinit();
        self.base = try tp.Engine.loadOn(io, gpa, dir, s);
        errdefer self.base.deinit();
        self.naf_dir = try std.fmt.allocPrint(gpa, "{s}/naf", .{dir});
        errdefer gpa.free(self.naf_dir);
        try requireFile(io, gpa, self.naf_dir, "model.safetensors");
        self.mv = try mvInstalled(io, gpa, dir, try readMvFlows(io, self.arena.allocator(), dir));
        log.info("[mesh] Pixal3D engine ready ({s}, multi-view {s})\n", .{ dir, if (self.mv != null) "yes" else "no" });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.base.deinit();
        self.gpa.free(self.naf_dir);
        self.arena.deinit();
        self.gpa.destroy(self);
    }

    pub fn supportsResolution(self: *const Engine, res: u32) bool {
        return self.base.supportsResolution(res);
    }

    pub fn generateGlb(self: *Engine, alloc: std.mem.Allocator, photos: []const Photo, opts: Opts, prog: ?sse.Progress) ![]u8 {
        var vox = try self.generateVoxels(alloc, photos, opts, prog);
        defer vox.deinit(alloc);
        return tp.meshToGlb(alloc, &vox, opts.base, prog);
    }

    /// The GPU half, as TRELLIS.2's, every flow conditioned on the projected features of its own tokens.
    pub fn generateVoxels(self: *Engine, alloc: std.mem.Allocator, photos: []const Photo, opts: Opts, prog: ?sse.Progress) !tp.Voxels {
        if (!self.supportsResolution(opts.base.resolution)) return error.UnsupportedResolution;
        if (photos.len == 0) return error.NoViews;
        if (viewsRefusal(photos.len, opts.cameras.len, self.mv != null) != null) return error.BadViews;
        const multi = photos.len > 1;
        const m = &self.base.models;
        const flows = if (multi) self.mv.? else Flows{ .ss = m.sparse_structure_flow_model, .shape_512 = m.shape_slat_flow_model_512, .shape_1024 = m.shape_slat_flow_model_1024, .tex_1024 = m.tex_slat_flow_model_1024 };

        var cams: [MAX_VIEWS]Camera = undefined;
        camerasFor(cams[0..photos.len], opts.fov orelse (if (multi) RIG_FOV else DEFAULT_FOV), opts.cameras) orelse return error.SingularCamera;
        if (prog) |p| p.emit("pixal3d-cond", 0, 5);
        var cond = blk: {
            const images = try viewImages(alloc, photos);
            defer {
                for (images) |pair| for (pair) |img| alloc.free(img);
                alloc.free(images);
            }
            break :blk try Conditioner.init(self.io, alloc, self.base.s, self.base.dino_dir, self.naf_dir, images, cams[0..photos.len]);
        };
        defer cond.deinit();
        if (prog) |p| p.emit("pixal3d-cond", 1, 5);

        const ss_coords = blk: {
            const grid = try t2.gridCoords(alloc, 16);
            defer alloc.free(grid);
            const c = try cond.stage(.ss, grid, 16);
            defer freeCond(c);
            if (prog) |p| p.emit("pixal3d-cond", 2, 5);
            break :blk try self.base.sparseStructure(alloc, flows.ss, c, opts.base, "pixal3d-structure", prog);
        };
        defer alloc.free(ss_coords);

        var shape = try self.shapeLatent(alloc, &cond, flows, ss_coords, opts.base, prog);
        defer shape.deinit(alloc);
        var geo = try self.base.decodeShape(alloc, &shape, prog);
        defer geo.deinit(alloc);
        var cfg = t2.TEX_SAMPLER;
        if (opts.base.steps != 0) cfg.steps = opts.base.steps;
        const c = try cond.stage(.tex_1024, shape.coords, shape.res / 16);
        defer freeCond(c);
        if (prog) |p| p.emit("pixal3d-cond", 5, 5);
        const tex = try self.base.flowLatent(alloc, flows.tex_1024, shape.coords, c, shape.norm, cfg, tp.stageSeed(opts.base.seed, 3), "pixal3d-texture", prog);
        defer _ = mlx.mlx_array_free(tex);
        return self.base.texturedVoxels(alloc, &shape, &geo, tex, prog);
    }

    /// The 512 shape pass, the voxels its decoder upsamples to quantized onto the finer lattice (lowered by 128 until
    /// the tokens fit, never below 1024), and the fine pass over them.
    fn shapeLatent(self: *Engine, alloc: std.mem.Allocator, cond: *Conditioner, flows: Flows, ss_coords: []const i32, opts: tp.Opts, prog: ?sse.Progress) !tp.Engine.Shape {
        var cfg = t2.SHAPE_SAMPLER;
        if (opts.steps != 0) cfg.steps = opts.steps;
        const lr_norm = blk: {
            const c = try cond.stage(.shape_512, ss_coords, 32);
            defer freeCond(c);
            if (prog) |p| p.emit("pixal3d-cond", 3, 5);
            break :blk try self.base.flowLatent(alloc, flows.shape_512, ss_coords, c, null, cfg, tp.stageSeed(opts.seed, 1), "pixal3d-shape-512", prog);
        };
        defer _ = mlx.mlx_array_free(lr_norm);
        const hr = try self.base.upsampleCoords(alloc, ss_coords, lr_norm, prog);
        defer alloc.free(hr);
        var res = opts.resolution;
        var coords: []i32 = undefined;
        while (true) {
            coords = try tp.quantizeUniqueWith(alloc, hr, tp.LR_RES, res / 16, roundCell);
            if (coords.len / 4 < tp.MAX_TOKENS or res <= 1024) break;
            alloc.free(coords);
            res -= 128;
        }
        errdefer alloc.free(coords);
        log.info("[pixal3d] cascade: {d} tokens at {d}\n", .{ coords.len / 4, res });
        const c = try cond.stage(.shape_1024, coords, res / 16);
        defer freeCond(c);
        if (prog) |p| p.emit("pixal3d-cond", 4, 5);
        const norm = try self.base.flowLatent(alloc, flows.shape_1024, coords, c, null, cfg, tp.stageSeed(opts.seed, 2), "pixal3d-shape-1024", prog);
        return .{ .coords = coords, .norm = norm, .res = res };
    }
};

/// Every photo at both input sizes (`[v][k]`, caller frees): the single view cropped around its subject, the
/// multi-view ones as framed.
fn viewImages(alloc: std.mem.Allocator, photos: []const Photo) ![][2][]f32 {
    const out = try alloc.alloc([2][]f32, photos.len);
    var done: usize = 0;
    errdefer {
        for (out[0..done]) |pair| for (pair) |img| alloc.free(img);
        alloc.free(out);
    }
    if (photos.len == 1) {
        const prep = try tp.prepareImageMargin(alloc, photos[0].rgba, photos[0].w, photos[0].h, 1.1);
        defer alloc.free(prep.rgb);
        out[0][0] = try singleViewImage(alloc, prep, SIZES[0]);
        errdefer alloc.free(out[0][0]);
        out[0][1] = try singleViewImage(alloc, prep, SIZES[1]);
        return out;
    }
    for (photos, out) |ph, *pair| {
        pair[0] = try multiViewImage(alloc, ph.rgba, ph.w, ph.h, SIZES[0]);
        errdefer alloc.free(pair[0]);
        pair[1] = try multiViewImage(alloc, ph.rgba, ph.w, ph.h, SIZES[1]);
        done += 1;
    }
    return out;
}

/// The cascade's cell: `round((c + 0.5) / lr_res * (grid - 1))` in f32, ties to even as torch rounds.
fn roundCell(c: f32, lr_res: f32, grid: f32) i32 {
    return @intFromFloat(roundHalfEven((c + 0.5) / lr_res * (grid - 1)));
}

fn roundHalfEven(v: f32) f32 {
    const r = @round(v);
    return if (@abs(v - @trunc(v)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(v) else r;
}

fn requireFile(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, file: []const u8) !void {
    const p = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, file });
    defer gpa.free(p);
    std.Io.Dir.accessAbsolute(io, p, .{}) catch {
        log.err("[pixal3d] missing weights: {s}\n", .{p});
        return error.MissingPixal3dWeights;
    };
}

/// The four flows `pipeline_mv.json` names, or null when the pack has no such file.
/// `pipeline_mv.json` names four flows, but the pack the app downloads for one photo carries only the JSON:
/// multi-view is then absent, and `viewsRefusal` turns a several-image request away by name.
fn mvInstalled(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, flows: ?Flows) !?Flows {
    const f = flows orelse return null;
    for ([_][]const u8{ f.ss, f.shape_512, f.shape_1024, f.tex_1024 }) |rel| {
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}.safetensors", .{ dir, rel });
        defer gpa.free(path);
        std.Io.Dir.accessAbsolute(io, path, .{}) catch return null;
    }
    return f;
}

fn readMvFlows(io: std.Io, a: std.mem.Allocator, dir: []const u8) !?Flows {
    const path = try std.fmt.allocPrint(a, "{s}/pipeline_mv.json", .{dir});
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(1 << 20));
    const Raw = struct { args: struct { models: struct {
        sparse_structure_flow_model: []const u8,
        shape_slat_flow_model_512: []const u8,
        shape_slat_flow_model_1024: []const u8,
        tex_slat_flow_model_1024: []const u8,
    } } };
    const r = try std.json.parseFromSliceLeaky(Raw, a, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    const md = r.args.models;
    return .{ .ss = md.sparse_structure_flow_model, .shape_512 = md.shape_slat_flow_model_512, .shape_1024 = md.shape_slat_flow_model_1024, .tex_1024 = md.tex_slat_flow_model_1024 };
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "pixal3d camera: the FOV's distance puts the cube's left edge on the frame's first column" {
    try testing.expectApproxEqAbs(@as(f32, 4.983322), distanceFromFov(0.2), 1e-5); // the oracle's own print
    const cams = blk: {
        var c: [1]Camera = undefined;
        camerasFor(&c, 0.2, &.{}).?;
        break :blk c;
    };
    // Cell (0, 1, 1) of a 3^3 lattice is world (-0.5, 0, 0): column 0, the row through the centre.
    const px = cams[0].project(512, cellPoint(.{ 0, 1, 1 }, 3));
    try testing.expectApproxEqAbs(@as(f32, 0), px[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 256), px[1], 1e-4);
    // +z (lattice y, up) lands above the centre row, +x right of the centre column.
    const up = cams[0].project(512, cellPoint(.{ 1, 2, 1 }, 3));
    try testing.expectApproxEqAbs(@as(f32, 256), up[0], 1e-4);
    try testing.expect(up[1] < 256);
}

test "pixal3d camera: the first view snaps to the front pose and the others keep their pose relative to it" {
    var calc: [4]Mat4 = undefined;
    viewMatrices(&calc, &RIG).?;
    const f = frontView(RIG_D);
    for (0..4) |i| for (0..4) |j| try testing.expectApproxEqAbs(f[i][j], calc[0][i][j], 1e-6);
    // The rig's first pose IS the front pose, so every view is its own pose.
    for (calc, RIG) |c, r| for (0..4) |i| for (0..4) |j| try testing.expectApproxEqAbs(r[i][j], c[i][j], 1e-5);
    // A rig turned about z by 90 degrees as a whole lands on the same matrices.
    const turn: Mat4 = .{ .{ 0, -1, 0, 0 }, .{ 1, 0, 0, 0 }, .{ 0, 0, 1, 0 }, .{ 0, 0, 0, 1 } };
    var turned: [4]Mat4 = undefined;
    for (&turned, RIG) |*t, r| t.* = matMul(turn, r);
    var calc2: [4]Mat4 = undefined;
    viewMatrices(&calc2, &turned).?;
    for (calc, calc2) |a, b| for (0..4) |i| for (0..4) |j| try testing.expectApproxEqAbs(a[i][j], b[i][j], 1e-5);
}

test "pixal3d camera: invert undoes a pose and refuses a singular matrix" {
    const m = matMul(RIG[1], frontView(2.5));
    const id = matMul(m, invert(m).?);
    for (0..4) |i| for (0..4) |j| try testing.expectApproxEqAbs(@as(f32, if (i == j) 1 else 0), id[i][j], 1e-6);
    var z: Mat4 = RIG[0];
    z[2] = .{ 0, 0, 0, 0 };
    try testing.expect(invert(z) == null);
}

test "pixal3d lattice: linspace matches torch's two-sided formula and the rotation sends lattice y to world z" {
    try testing.expectEqual(@as(f32, -1), linspace(0, 16));
    try testing.expectEqual(@as(f32, 1), linspace(15, 16));
    const step: f32 = 2.0 / 15.0;
    try testing.expectEqual(1.0 - step * 7.0, linspace(8, 16)); // f32 throughout, as torch
    const p = cellPoint(.{ 0, 31, 0 }, 32);
    try testing.expectEqualSlices(f32, &.{ -0.5, 0.5, 0.5 }, &p);
}

test "pixal3d cascade cell: (c + 0.5) / 512 * (grid - 1), ties to even" {
    try testing.expectEqual(@as(i32, 0), roundCell(0, 512, 64));
    try testing.expectEqual(@as(i32, 63), roundCell(511, 512, 64));
    try testing.expectEqual(@as(f32, 2), roundHalfEven(2.5));
    try testing.expectEqual(@as(f32, 4), roundHalfEven(3.5));
    try testing.expectEqual(@as(f32, -2), roundHalfEven(-2.5));
    try testing.expectEqual(@as(f32, 3), roundHalfEven(2.6));
}

test "pixal3d views: counts, cameras and the rig are refused by name" {
    try testing.expect(viewsRefusal(1, 0, false) == null);
    try testing.expect(viewsRefusal(4, 0, true) == null);
    try testing.expect(viewsRefusal(6, 6, true) == null);
    try testing.expect(std.mem.indexOf(u8, viewsRefusal(3, 0, false).?, "multi-view") != null);
    try testing.expect(std.mem.indexOf(u8, viewsRefusal(9, 9, true).?, "at most 8") != null);
    try testing.expect(std.mem.indexOf(u8, viewsRefusal(3, 2, true).?, "one 4x4") != null);
    try testing.expect(std.mem.indexOf(u8, viewsRefusal(5, 0, true).?, "'cameras'") != null);
}

test "pixal3d cameras: nested 4x4 arrays parse; any other shape or a singular matrix is refused" {
    const gpa = testing.allocator;
    const ok_json = "[[[1,0,0,0],[0,0,-1,-3],[0,1,0,0],[0,0,0,1]],[[0,0,1,3.5],[1,0,0,0],[0,1,0,0],[0,0,0,1]]]";
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, ok_json, .{});
    defer parsed.deinit();
    const r = try parseCameras(gpa, parsed.value);
    defer if (r == .ok) gpa.free(r.ok);
    try testing.expectEqual(@as(usize, 2), r.ok.len);
    try testing.expectEqual(@as(f32, 3.5), r.ok[1][0][3]);
    for ([_][]const u8{ "[[1,0,0,0]]", "[[[1,0,0],[0,1,0],[0,0,1]]]", "{\"a\":1}", "[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,\"x\"]]]" }) |bad| {
        var p = try std.json.parseFromSlice(std.json.Value, gpa, bad, .{});
        defer p.deinit();
        try testing.expect((try parseCameras(gpa, p.value)) == .bad);
    }
    var sing = try std.json.parseFromSlice(std.json.Value, gpa, "[[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]]", .{});
    defer sing.deinit();
    const sr = try parseCameras(gpa, sing.value);
    try testing.expect(std.mem.indexOf(u8, sr.bad, "singular") != null);
}

test "pixal3d multi-view image: a resize goes through Pillow's premultiplied round trip, the source size does not" {
    const gpa = testing.allocator;
    // Uniform (200, 10, 0, a=100): premultiplied 78, unchanged by the resize, back to 198, then times alpha.
    var flat: [2 * 2 * 4]u8 = undefined;
    for (0..4) |i| @memcpy(flat[i * 4 ..][0..4], &[_]u8{ 200, 10, 0, 100 });
    const down = try multiViewImage(gpa, &flat, 2, 2, 1);
    defer gpa.free(down);
    try testing.expectApproxEqAbs(@as(f32, 198.0 / 255.0 * 100.0 / 255.0), down[0], 1e-7);
    const px = [_]u8{ 200, 10, 0, 100, 255, 255, 255, 255, 9, 9, 9, 0, 1, 2, 3, 254 };
    const same = try multiViewImage(gpa, &px, 2, 2, 2);
    defer gpa.free(same);
    try testing.expectApproxEqAbs(@as(f32, 200.0 / 255.0 * 100.0 / 255.0), same[0], 1e-7);
    try testing.expectEqual(@as(f32, 1), same[3]);
    try testing.expectEqual(@as(f32, 0), same[6]); // alpha 0 is black
}

test "pixal3d multi-view: a pack with pipeline_mv.json but no _mv weights serves one view and refuses several by name" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try tmp.dir.createDirPath(io, "ckpts");
    const flows = Flows{ .ss = "ckpts/ss_mv", .shape_512 = "ckpts/s512_mv", .shape_1024 = "ckpts/s1024_mv", .tex_1024 = "ckpts/t1024_mv" };
    try testing.expect((try mvInstalled(io, gpa, root, flows)) == null);
    for ([_][]const u8{ "ss_mv", "s512_mv", "s1024_mv" }) |n| {
        const f = try std.fmt.allocPrint(gpa, "ckpts/{s}.safetensors", .{n});
        defer gpa.free(f);
        try tmp.dir.writeFile(io, .{ .sub_path = f, .data = "x" });
    }
    try testing.expect((try mvInstalled(io, gpa, root, flows)) == null); // one flow still missing
    try tmp.dir.writeFile(io, .{ .sub_path = "ckpts/t1024_mv.safetensors", .data = "x" });
    try testing.expect((try mvInstalled(io, gpa, root, flows)) != null);
    try testing.expect((try mvInstalled(io, gpa, root, null)) == null);
    try testing.expect(viewsRefusal(2, 0, false) != null);
}

// ── oracles (tests/dump_pixal3d_fixtures.py) ────────────────────────────────

fn envDir(comptime name: []const u8) ![]const u8 {
    return std.mem.span(env.named(name) orelse return error.SkipZigTest);
}

fn loadPng(io: std.Io, gpa: std.mem.Allocator, path: []const u8) !struct { px: []u8, w: u32, h: u32 } {
    const bytes = try tp.readFileAny(io, gpa, path);
    defer gpa.free(bytes);
    const stb = @import("stb");
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const p = stb.stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &w, &h, &ch, 4) orelse return error.BadImage;
    defer stb.stbi_image_free(p);
    return .{ .px = try gpa.dupe(u8, p[0..@intCast(w * h * 4)]), .w = @intCast(w), .h = @intCast(h) };
}

/// Fixture `[3, H, W]` (CHW) as interleaved host f32.
fn chwToHwc(gpa: std.mem.Allocator, sc: *Scope, a: A) ![]f32 {
    const sh = mlx.getShape(a);
    const host = try t2.hostF32(sc, a);
    const hw: usize = @intCast(sh[1] * sh[2]);
    const out = try gpa.alloc(f32, hw * 3);
    for (0..hw) |i| for (0..3) |c| {
        out[i * 3 + c] = host[c * hw + i];
    };
    return out;
}

fn expectImageClose(label: []const u8, got: []const f32, want: []const f32, tol: f32) !void {
    try testing.expectEqual(want.len, got.len);
    var worst: f32 = 0;
    for (got, want) |g, w| worst = @max(worst, @abs(g - w));
    std.debug.print("{s}: max abs diff {e:.2}\n", .{ label, worst });
    try testing.expect(worst <= tol);
}

test "pixal3d oracle: crop, DINOv3 tokens, NAF and the projection reproduce every stage's conditioning (PIXAL3D_FIXTURES, PIXAL3D_DINOV3, PIXAL3D_NAF, PIXAL3D_IMAGE)" {
    const gpa = testing.allocator;
    const fixtures = try envDir("PIXAL3D_FIXTURES");
    const dino_dir = try envDir("PIXAL3D_DINOV3");
    const naf_dir = try envDir("PIXAL3D_NAF");
    const image_path = try envDir("PIXAL3D_IMAGE");
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);

    const img = try loadPng(io, gpa, image_path);
    defer gpa.free(img.px);
    const prep = try tp.prepareImageMargin(gpa, img.px, img.w, img.h, 1.1);
    defer gpa.free(prep.rgb);
    var pre = try model_mod.loadWeightsFile(gpa, fixtures, "preprocess.safetensors");
    defer pre.deinit();
    {
        const want = try t2.hostI32(gpa, pre.get("rgb").?);
        defer gpa.free(want);
        try testing.expectEqual(want.len, prep.rgb.len);
        var diff: usize = 0;
        for (want, prep.rgb) |a, b| diff += @intFromBool(a != b);
        std.debug.print("pixal3d crop: {d}x{d}, {d} bytes differ\n", .{ prep.side, prep.side, diff });
        try testing.expectEqual(@as(usize, 0), diff);
    }

    var cam: [1]Camera = undefined;
    camerasFor(&cam, 0.2, &.{}).?;
    var imgs: [1][2][]f32 = undefined;
    imgs[0][0] = try singleViewImage(gpa, prep, 512);
    defer gpa.free(imgs[0][0]);
    imgs[0][1] = try singleViewImage(gpa, prep, 1024);
    defer gpa.free(imgs[0][1]);
    var images: [1][2][]const f32 = .{.{ imgs[0][0], imgs[0][1] }};
    var cond = try Conditioner.init(io, gpa, s, dino_dir, naf_dir, &images, &cam);
    defer cond.deinit();

    const cases = [_]struct { file: []const u8, stage: Stage, grid: u32 }{
        .{ .file = "proj_ss.safetensors", .stage = .ss, .grid = 16 },
        .{ .file = "proj_shape512.safetensors", .stage = .shape_512, .grid = 32 },
        .{ .file = "proj_shape1024.safetensors", .stage = .shape_1024, .grid = 64 },
        .{ .file = "proj_tex1024.safetensors", .stage = .tex_1024, .grid = 64 },
    };
    for (cases) |cs| {
        var fx = try model_mod.loadWeightsFile(gpa, fixtures, cs.file);
        defer fx.deinit();
        var sc = Scope.init(gpa, s);
        defer sc.deinit();
        const coords = if (fx.get("coords")) |c| try t2.hostI32(gpa, c) else try t2.gridCoords(gpa, 16);
        defer gpa.free(coords);
        const got = try cond.stage(cs.stage, coords, cs.grid);
        defer freeCond(got);
        std.debug.print("{s}:\n", .{cs.file});
        try t2.expectClose("  global", &sc, got.global, fx.get("global").?, 0.9999, 1e-3);
        const want = fx.get("proj").?;
        try t2.expectClose("  proj", &sc, try sc.reshape(got.proj.?, mlx.getShape(want)), want, 0.9999, 1e-3);
    }
}

test "pixal3d oracle: two posed views average into the multi-view conditioning (PIXAL3D_FIXTURES, PIXAL3D_DINOV3, PIXAL3D_NAF, PIXAL3D_IMAGE)" {
    const gpa = testing.allocator;
    const fixtures = try envDir("PIXAL3D_FIXTURES");
    const dino_dir = try envDir("PIXAL3D_DINOV3");
    const naf_dir = try envDir("PIXAL3D_NAF");
    const image_path = try envDir("PIXAL3D_IMAGE");
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);

    const img = try loadPng(io, gpa, image_path);
    defer gpa.free(img.px);
    // The second view is the cutout mirrored left to right.
    const flip = try gpa.alloc(u8, img.px.len);
    defer gpa.free(flip);
    for (0..img.h) |y| for (0..img.w) |x| {
        @memcpy(flip[(y * img.w + x) * 4 ..][0..4], img.px[(y * img.w + (img.w - 1 - x)) * 4 ..][0..4]);
    };
    var fx = try model_mod.loadWeightsFile(gpa, fixtures, "proj_mv.safetensors");
    defer fx.deinit();
    var pre = try model_mod.loadWeightsFile(gpa, fixtures, "preprocess.safetensors");
    defer pre.deinit();
    var sc0 = Scope.init(gpa, s);
    defer sc0.deinit();

    var imgs: [2][2][]f32 = undefined;
    for ([_][]const u8{ img.px, flip }, 0..) |px, v| for (SIZES, 0..) |size, k| {
        imgs[v][k] = try multiViewImage(gpa, px, img.w, img.h, size);
    };
    defer for (imgs) |pair| for (pair) |a| gpa.free(a);
    {
        const want512 = try chwToHwc(gpa, &sc0, pre.get("mv512").?);
        defer gpa.free(want512);
        try expectImageClose("mv view 0 at 512", imgs[0][0], want512, 1.5 / 255.0);
        const want1024 = try chwToHwc(gpa, &sc0, fx.get("mv1024").?);
        defer gpa.free(want1024);
        try expectImageClose("mv view 1 at 1024", imgs[1][1], want1024, 1.5 / 255.0);
    }

    const tm = try t2.hostF32(&sc0, fx.get("transform_matrix").?);
    var c2w: [2]Mat4 = undefined;
    for (0..2) |v| for (0..4) |i| for (0..4) |j| {
        c2w[v][i][j] = tm[v * 16 + i * 4 + j];
    };
    const fov = (try t2.hostF32(&sc0, fx.get("fov").?))[0];
    var cams: [2]Camera = undefined;
    camerasFor(&cams, fov, &c2w).?;
    var images: [2][2][]const f32 = .{ .{ imgs[0][0], imgs[0][1] }, .{ imgs[1][0], imgs[1][1] } };
    var cond = try Conditioner.init(io, gpa, s, dino_dir, naf_dir, &images, &cams);
    defer cond.deinit();

    const cases = [_]struct { key: []const u8, stage: Stage, grid: u32 }{
        .{ .key = "ss", .stage = .ss, .grid = 16 },
        .{ .key = "shape512", .stage = .shape_512, .grid = 32 },
        .{ .key = "shape1024", .stage = .shape_1024, .grid = 64 },
    };
    for (cases) |cs| {
        var sc = Scope.init(gpa, s);
        defer sc.deinit();
        var kb: [32]u8 = undefined;
        const coords = if (fx.get(try std.fmt.bufPrint(&kb, "{s}_coords", .{cs.key}))) |c| try t2.hostI32(gpa, c) else try t2.gridCoords(gpa, 16);
        defer gpa.free(coords);
        const got = try cond.stage(cs.stage, coords, cs.grid);
        defer freeCond(got);
        std.debug.print("mv {s}:\n", .{cs.key});
        try t2.expectClose("  global", &sc, got.global, fx.get(try std.fmt.bufPrint(&kb, "{s}_global", .{cs.key})).?, 0.9999, 1e-3);
        const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_proj", .{cs.key})).?;
        try t2.expectClose("  proj", &sc, try sc.reshape(got.proj.?, mlx.getShape(want)), want, 0.9999, 1e-3);
    }
}

/// One Pixal3D flow fixture through its published checkpoint at `act`.
fn ditOracle(stem: []const u8, fixture: []const u8, dense: bool, act: mlx.mlx_dtype, min_cos: f64, tol: f64) !void {
    const gpa = testing.allocator;
    const weights = try envDir("PIXAL3D_WEIGHTS");
    const fixtures = try envDir("PIXAL3D_FIXTURES");
    const io = std.Io.Threaded.global_single_threaded.io();
    const ckpts = try std.fmt.allocPrint(gpa, "{s}/ckpts", .{weights});
    defer gpa.free(ckpts);
    var fx = try model_mod.loadWeightsFile(gpa, fixtures, fixture);
    defer fx.deinit();
    var od = try t2.openDit(io, gpa, ckpts, stem, act);
    defer od.deinit();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const x_fx = fx.get("x").?;
    var coords: []i32 = undefined;
    var x: A = undefined;
    if (dense) {
        coords = try t2.gridCoords(gpa, 16);
        x = try sc.transpose(try sc.reshape(x_fx, &.{ 1, mlx.getShape(x_fx)[1], 16 * 16 * 16 }), &.{ 0, 2, 1 });
    } else {
        coords = try t2.hostI32(gpa, fx.get("coords").?);
        const sh = mlx.getShape(x_fx);
        x = try sc.reshape(x_fx, &.{ 1, sh[0], sh[1] });
        if (fx.get("concat")) |cc| x = try sc.concat(&.{ x, try sc.reshape(cc, &.{ 1, sh[0], mlx.getShape(cc)[1] }) }, 2);
    }
    defer gpa.free(coords);
    const proj_fx = fx.get("proj").?;
    const proj = try sc.reshape(proj_fx, &.{ 1, mlx.getShape(proj_fx)[0], mlx.getShape(proj_fx)[1] });
    const rope = try t2.Rope.init(&sc, coords, od.dit.head_dim);
    var tt: f32 = undefined;
    try mlx.check(mlx.mlx_array_item_float32(&tt, fx.get("t").?));
    var taps = t2.Taps{};
    var out = try od.dit.forward(&sc, x, tt, fx.get("global").?, .{ .feats = proj }, rope, &taps);
    const want_out = fx.get("out").?;
    out = if (dense) try sc.reshape(try sc.transpose(out, &.{ 0, 2, 1 }), mlx.getShape(want_out)) else try sc.reshape(out, mlx.getShape(want_out));
    std.debug.print("{s} {s}:\n", .{ stem, if (act == .float32) "f32" else "bf16" });
    inline for (.{ .{ "h_in", &taps.h_in }, .{ "blk0", &taps.blk0 }, .{ "blk_last", &taps.blk_last } }) |tap| {
        if (fx.get(tap[0])) |want| try t2.expectClose("  " ++ tap[0], &sc, try sc.reshape(tap[1].*, mlx.getShape(want)), want, min_cos, tol);
    }
    try t2.expectClose("  out", &sc, out, want_out, min_cos, tol);
}

test "pixal3d oracle: the flow DiTs add each block's projection of the pixel features (PIXAL3D_WEIGHTS, PIXAL3D_FIXTURES)" {
    try ditOracle("ss_flow_img_dit_1_3B_64_bf16", "ss_flow.safetensors", true, .float32, 0.9999, 1e-3);
    try ditOracle("slat_flow_img2shape_dit_1_3B_512_bf16", "slat_shape512.safetensors", false, .float32, 0.9999, 1e-3);
    try ditOracle("slat_flow_imgshape2tex_dit_1_3B_1024_bf16", "slat_tex1024.safetensors", false, .float32, 0.9999, 1e-3);
}

test "pixal3d oracle: the DiTs in bf16, as the reference runs them, track the fp32 reference (PIXAL3D_WEIGHTS, PIXAL3D_FIXTURES)" {
    // The structure flow's output is a small difference of large terms: bf16 alone lands at cos 0.990, scale 0.950 (8-bit: 0.988, 0.938).
    try ditOracle("ss_flow_img_dit_1_3B_64_bf16", "ss_flow.safetensors", true, .bfloat16, 0.98, 0.08);
    try ditOracle("slat_flow_img2shape_dit_1_3B_512_bf16", "slat_shape512.safetensors", false, .bfloat16, 0.99, 0.05);
    try ditOracle("slat_flow_imgshape2tex_dit_1_3B_1024_bf16", "slat_tex1024.safetensors", false, .bfloat16, 0.99, 0.05);
}

/// `transforms.json` in a views directory: each frame's image and camera-to-world, the shared FOV.
fn loadViewsDir(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, photos: *std.ArrayList(Photo), cams: *std.ArrayList(Mat4)) !f32 {
    const tj = try std.fmt.allocPrint(gpa, "{s}/transforms.json", .{dir});
    defer gpa.free(tj);
    const bytes = try tp.readFileAny(io, gpa, tj);
    defer gpa.free(bytes);
    const Raw = struct { camera_angle_x: f32, frames: []const struct { file_path: []const u8, transform_matrix: Mat4 } };
    var parsed = try std.json.parseFromSlice(Raw, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (parsed.value.frames) |fr| {
        const p = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, fr.file_path });
        defer gpa.free(p);
        const im = try loadPng(io, gpa, p);
        try photos.append(gpa, .{ .rgba = im.px, .w = im.w, .h = im.h });
        try cams.append(gpa, fr.transform_matrix);
    }
    return parsed.value.camera_angle_x;
}

test "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB (PIXAL3D_PACK, PIXAL3D_OUT)" {
    const gpa = testing.allocator;
    const out_path = try envDir("PIXAL3D_OUT");
    const pack = try envDir("PIXAL3D_PACK");
    const io = std.Io.Threaded.global_single_threaded.io();
    var opts = Opts{};
    if (std.c.getenv("PIXAL3D_RES")) |v| opts.base.resolution = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("PIXAL3D_STEPS")) |v| opts.base.steps = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("PIXAL3D_TEX")) |v| opts.base.texture_size = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("PIXAL3D_FACES")) |v| opts.base.decimation_target = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("PIXAL3D_FOV_DEG")) |v| opts.fov = std.math.degreesToRadians(try std.fmt.parseFloat(f32, std.mem.span(v)));
    var clock = tp.TestClock{ .sw = @import("../../sdk/util/io_util.zig").Stopwatch.init(io) };
    const prog = sse.Progress{ .ctx = &clock, .cb = tp.TestClock.tick };

    var photos: std.ArrayList(Photo) = .empty;
    defer {
        for (photos.items) |p| gpa.free(p.rgba);
        photos.deinit(gpa);
    }
    var cams: std.ArrayList(Mat4) = .empty;
    defer cams.deinit(gpa);
    if (std.c.getenv("PIXAL3D_VIEWS")) |d| {
        const fov = try loadViewsDir(io, gpa, std.mem.span(d), &photos, &cams);
        if (opts.fov == null) opts.fov = fov;
        opts.cameras = cams.items;
    } else {
        const im = try loadPng(io, gpa, try envDir("PIXAL3D_IMAGE"));
        try photos.append(gpa, .{ .rgba = im.px, .w = im.w, .h = im.h });
    }
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var eng = try Engine.loadOn(io, gpa, pack, s);
    defer eng.deinit();
    _ = mlx.mlx_reset_peak_memory();
    var vox = try eng.generateVoxels(gpa, photos.items, opts, prog);
    defer vox.deinit(gpa);
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("  mlx peak {d} MiB, {d} voxels at {d}\n", .{ peak >> 20, vox.coords.len / 4, vox.res });
    const bytes = try tp.meshToGlb(gpa, &vox, opts.base, prog);
    defer gpa.free(bytes);
    const f = try std.Io.Dir.createFileAbsolute(io, out_path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, bytes);
    std.debug.print("wrote {s} ({d} bytes)\n", .{ out_path, bytes.len });
}
