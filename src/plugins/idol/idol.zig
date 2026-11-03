//! Idol (speridlabs/idol, Apache-2.0): one image to a 3D Gaussian splat, end to end.
//!
//!   image -> background matte -> crop -> 1024 x 1024 on black
//!   -> DINOv3 ViT-H+/16 tokens (feature1, 1280) + FLUX.2 VAE tokens (feature2, 128), 4101 each
//!   -> flow denoiser over an 8192 x 16 latent and a 5-value camera, Euler steps with classifier-free guidance
//!   -> octree decoder places the anchor points, Gaussian decoder turns each into 32 Gaussians.
//!
//! Pack layout: `config.json`, `denoiser|decoder|decoder_gs|birefnet.safetensors` as released (or as
//! `tests/convert_idol_weights.py` leaves them), `dinov3/{config.json,model.safetensors}` and `vae/` (the FLUX.2 VAE).
//! Parity: env-gated `IDOL_*` oracles fed by tests/dump_idol_fixtures.py.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const scope_mod = @import("../../mlx/scope.zig");
const sse = @import("../../sdk/media/progress.zig");
const dino = @import("../../sdk/vision/dinov3.zig");
const w8a8 = @import("../../sdk/blocks/dit/w8a8.zig");
const flux = @import("../flux2/flux.zig");
const L = @import("layers.zig");
const den = @import("denoiser.zig");
const dec = @import("decoders.zig");
const img = @import("image.zig");
const birefnet = @import("birefnet.zig");
const config = @import("config.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const Scope = scope_mod.Scope;
const Weights = model_mod.Weights;


pub const Opts = struct {
    seed: u64 = 42,
    steps: u32 = 50,
    cfg_strength: f32 = 3.0,
    rescale_t: f32 = 3.0,
    num_gaussians: u32 = config.DEFAULT_GAUSSIANS,
    margin: f64 = img.MARGIN_SYNTHETIC,
    /// The initial noise, `[8192 * 16]` and `[5]`: parity tests feed the reference's.
    noise_latent: ?[]const f32 = null,
    noise_camera: ?[]const f32 = null,
};

/// `decode_camera`: the predicted input camera.
pub const Camera = struct { yaw: f32, pitch: f32, radius: f32, fov: f32 };

pub fn decodeCamera(c: [5]f32) Camera {
    const n = @sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    var d = [3]f32{ 0, 0, 1 };
    if (n > 1e-6) d = .{ c[0] / n, c[1] / n, c[2] / n };
    const inv = c[3];
    const half = c[4];
    var radius: f32 = 1000;
    var fov: f32 = undefined;
    if (inv < 1e-3) {
        fov = std.math.radiansToDegrees(2 * std.math.atan(half / radius));
    } else {
        radius = 1.0 / inv;
        fov = std.math.radiansToDegrees(2 * std.math.atan(half * inv));
    }
    return .{ .yaw = std.math.atan2(d[0], d[1]), .pitch = std.math.asin(std.math.clamp(d[2], -1, 1)), .radius = radius, .fov = fov };
}

/// The flow time grid: `linspace(1, 0, steps + 1)` shifted towards 1 by `rescale_t`.
pub fn timeAt(i: u32, steps: u32, rescale_t: f32) f32 {
    const u: f64 = 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
    const r: f64 = rescale_t;
    return @floatCast(r * u / (1 + (r - 1) * u));
}

const Raw = struct {
    denoiser: struct { args: den.Cfg = .{} } = .{},
    decoder: struct { args: dec.Cfg = .{} } = .{},
    image_size: u32 = img.IMAGE_SIZE,
    decoder_gs: struct { args: struct {
        num_blocks: u32 = 16,
        model_channels: u32 = 1024,
        cond_channels: u32 = 16,
        representation_config: struct {
            num_gaussians: u32 = 32,
            lr: struct { _xyz: f32 = 1, _features_dc: f32 = 1, _opacity: f32 = 1, _scaling: f32 = 1, _rotation: f32 = 0.1 } = .{},
            perturbe_size: f32 = 1.5,
            @"3d_filter_kernel_size": f32 = 0.0009,
            scaling_bias: f32 = 0.004,
            opacity_bias: f32 = 0.1,
        } = .{},
    } = .{} } = .{},
    max_voxel_level: u32 = 8,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    s: S,
    size: u32,
    level: u32,
    rep: dec.Rep,
    w_den: Weights,
    w_oct: Weights,
    w_gs: Weights,
    denoiser: den.Denoiser,
    octree: dec.Octree,
    gaussians: dec.GaussianDecoder,
    dino: dino.Encoder,
    vae: flux.VaeEncoder,
    /// The background remover, when the pack ships it: without one, an image with no alpha is all subject.
    matte: ?birefnet.Net,

    pub fn loadOn(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, s: S) !*Engine {
        const cfg_path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
        defer gpa.free(cfg_path);
        const bytes = try readFile(io, gpa, cfg_path);
        defer gpa.free(bytes);
        var parsed = try std.json.parseFromSlice(Raw, gpa, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const raw = parsed.value;
        const rc = raw.decoder_gs.args.representation_config;
        if (rc.num_gaussians != dec.PER_ANCHOR or raw.image_size % 64 != 0) return error.UnsupportedIdolConfig;

        const self = try gpa.create(Engine);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.s = s;
        self.level = raw.max_voxel_level;
        self.size = raw.image_size;
        self.rep = .{ .lr_xyz = rc.lr._xyz, .lr_dc = rc.lr._features_dc, .lr_scaling = rc.lr._scaling, .lr_rotation = rc.lr._rotation, .lr_opacity = rc.lr._opacity, .perturbe_size = rc.perturbe_size, .kernel = rc.@"3d_filter_kernel_size", .scaling_bias = rc.scaling_bias, .opacity_bias = rc.opacity_bias };

        const act: mlx.mlx_dtype = .float16;
        self.w_den = try model_mod.loadWeightsFile(gpa, dir, "denoiser.safetensors");
        errdefer self.w_den.deinit();
        try L.castLinears(gpa, &self.w_den, &.{ "blocks.", "noise_refiner.", "context_refiner." }, act);
        try scope_mod.evalAll(&self.w_den);
        if (w8a8.w8a8State().enabled) {
            const n = try self.w_den.w8.convert(gpa, &self.w_den.map, .{ .base = "blocks.0.attn.qkv", .in_features = raw.denoiser.args.model_channels }, s);
            log.info("[idol] int8: {d} denoiser linears\n", .{n});
        }
        try L.transposeLinears(gpa, &self.w_den, &.{ "blocks.", "noise_refiner.", "context_refiner." });
        self.denoiser = try den.Denoiser.load(gpa, &self.w_den, raw.denoiser.args, act);
        errdefer self.denoiser.deinit();

        self.w_oct = try model_mod.loadWeightsFile(gpa, dir, "decoder.safetensors");
        errdefer self.w_oct.deinit();
        try L.castLinears(gpa, &self.w_oct, &.{"blocks."}, act);
        try L.transposeLinears(gpa, &self.w_oct, &.{"blocks."});
        try scope_mod.evalAll(&self.w_oct);
        self.octree = try dec.Octree.load(gpa, &self.w_oct, act, raw.decoder.args);
        errdefer self.octree.deinit();

        self.w_gs = try model_mod.loadWeightsFile(gpa, dir, "decoder_gs.safetensors");
        errdefer self.w_gs.deinit();
        try L.castLinears(gpa, &self.w_gs, &.{"blocks."}, act);
        try L.transposeLinears(gpa, &self.w_gs, &.{"blocks."});
        try scope_mod.evalAll(&self.w_gs);
        self.gaussians = try dec.GaussianDecoder.load(gpa, &self.w_gs, act, .{ .model_channels = raw.decoder_gs.args.model_channels, .cond_channels = raw.decoder_gs.args.cond_channels, .num_blocks = raw.decoder_gs.args.num_blocks });
        errdefer self.gaussians.deinit();

        const dino_dir = try std.fmt.allocPrint(gpa, "{s}/dinov3", .{dir});
        defer gpa.free(dino_dir);
        self.dino = try dino.Encoder.open(io, gpa, dino_dir, .bfloat16);
        errdefer self.dino.deinit();
        self.dino.affine_final = true;
        self.vae = try flux.loadVaeEncoder(io, gpa, s, dir);
        errdefer self.vae.deinit();
        self.matte = null;
        if (packHas(io, dir, "birefnet.safetensors")) {
            self.matte = try birefnet.Net.load(gpa, s, try model_mod.loadWeightsFile(gpa, dir, "birefnet.safetensors"));
        }
        return self;
    }

    pub fn deinit(self: *Engine) void {
        if (self.matte) |*m| m.deinit();
        self.vae.deinit();
        self.dino.deinit();
        self.gaussians.deinit();
        self.octree.deinit();
        self.denoiser.deinit();
        self.w_gs.deinit();
        self.w_oct.deinit();
        self.w_den.deinit();
        self.gpa.destroy(self);
    }

    pub const Cond = struct {
        f1: A,
        f2: A,

        pub fn deinit(self: Cond) void {
            _ = mlx.mlx_array_free(self.f1);
            _ = mlx.mlx_array_free(self.f2);
        }
    };

    /// DINOv3 and FLUX.2 VAE tokens of the square RGB8 image (1024 as released): `[1, 4101, 1280]` and
    /// `[1, 4101, 128]` f32, the CLS and register rows first (zeros, for the VAE's). `noise_seed` samples the VAE's
    /// posterior; null takes its mean.
    pub fn condition(self: *Engine, rgb: []const u8, noise_seed: ?u64) !Cond {
        const n: usize = self.size;
        var sc = Scope.init(self.gpa, self.s);
        defer sc.deinit();
        const px = try self.gpa.alloc(f32, n * n * 3);
        defer self.gpa.free(px);
        for (px, rgb, 0..) |*o, v, i| o.* = (@as(f32, @floatFromInt(v)) / 255.0 - dino.IMAGENET_MEAN[i % 3]) / dino.IMAGENET_STD[i % 3];
        const nhwc = try sc.fromF32(px, &.{ 1, @intCast(n), @intCast(n), 3 });
        const f1 = try self.dino.forward(&sc, nhwc);
        try mlx.check(mlx.mlx_array_eval(f1));

        // The VAE reads pixels in [0, 1], planar.
        const planar = try self.gpa.alloc(f32, n * n * 3);
        defer self.gpa.free(planar);
        for (0..n * n) |i| for (0..3) |c| {
            planar[c * n * n + i] = @as(f32, @floatFromInt(rgb[i * 3 + c])) / 255.0;
        };
        var g = ops.Graph.init(self.gpa, self.s);
        defer g.deinit();
        const nchw = try g.fromSlice(f32, planar, &.{ 1, 3, @intCast(n), @intCast(n) });
        // The reference conditions on a SAMPLE of the VAE's posterior, not its mean (`latent_dist.sample()`).
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        const lat = if (noise_seed) |seed| blk: {
            try mlx.check(mlx.mlx_random_key(&key, seed));
            break :blk try self.vae.encodeSample(&g, nchw, key);
        } else try self.vae.encode(&g, nchw); // [1, 128, 64, 64] bn-normalized
        const grid: c_int = @intCast(n / 16);
        const tok = try g.reshape(try g.transpose(lat, &.{ 0, 2, 3, 1 }), &.{ 1, grid * grid, 128 });
        const pad = try g.zeros(&.{ 1, @intCast(1 + self.dino.cfg.registers), 128 }, .float32);
        const f2 = try g.concat(&.{ pad, tok }, 1);
        try mlx.check(mlx.mlx_array_eval(f2));
        return .{ .f1 = sc.out(f1), .f2 = g.escape(f2) };
    }

    pub const Generated = struct {
        splats: dec.Splats,
        camera: Camera,
        pub fn deinit(self: *Generated, gpa: std.mem.Allocator) void {
            self.splats.deinit(gpa);
        }
    };

    /// The sampled latent `[8192 * 16]` and camera `[5]` from the image tokens (host copies, caller frees).
    pub fn sample(self: *Engine, cond: Cond, o: Opts, prog: ?sse.Progress) !struct { latent: []f32, camera: [5]f32 } {
        const gpa = self.gpa;
        const nz = self.denoiser.cfg.q_token_length;
        const nc = self.denoiser.cfg.cam_channels;
        var top = Scope.init(gpa, self.s);
        defer top.deinit();
        // Both context branches are refined once: they depend on neither the latent nor the time.
        const zero1 = try top.zeros(mlx.getShape(cond.f1), .float32);
        const zero2 = try top.zeros(mlx.getShape(cond.f2), .float32);
        const ctx_c = try self.denoiser.context(&top, cond.f1, cond.f2);
        const ctx_u = try self.denoiser.context(&top, zero1, zero2);

        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, o.seed));
        var k1 = mlx.mlx_array_new();
        var k2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k1);
        defer _ = mlx.mlx_array_free(k2);
        try mlx.check(mlx.mlx_random_split(&k1, &k2, key, self.s));
        var z = mlx.mlx_array_new();
        var cam = mlx.mlx_array_new();
        if (o.noise_latent) |nl| z = mlx.mlx_array_new_data(nl.ptr, &[_]c_int{ 1, @intCast(nz), @intCast(self.denoiser.cfg.in_channels) }, 3, .float32) else try mlx.check(mlx.mlx_random_normal(&z, &[_]c_int{ 1, @intCast(nz), @intCast(self.denoiser.cfg.in_channels) }, 3, .float32, 0, 1, k1, self.s));
        if (o.noise_camera) |nc_| cam = mlx.mlx_array_new_data(nc_.ptr, &[_]c_int{ 1, 1, @intCast(nc) }, 3, .float32) else try mlx.check(mlx.mlx_random_normal(&cam, &[_]c_int{ 1, 1, @intCast(nc) }, 3, .float32, 0, 1, k2, self.s));
        defer _ = mlx.mlx_array_free(z);
        defer _ = mlx.mlx_array_free(cam);

        for (0..o.steps) |i| {
            if (prog) |p| if (p.boundary()) return error.Cancelled;
            const t = timeAt(@intCast(i), o.steps, o.rescale_t);
            const tp = timeAt(@intCast(i + 1), o.steps, o.rescale_t);
            var sc = Scope.init(gpa, self.s);
            defer sc.deinit();
            const st = try self.denoiser.stream(&sc, z, cam, 1000 * t);
            const vc = try self.denoiser.joint(&sc, st, ctx_c);
            var vl = vc.latent;
            var vm = vc.camera;
            if (o.cfg_strength > 0) {
                const vu = try self.denoiser.joint(&sc, st, ctx_u);
                const w = o.cfg_strength;
                vl = try sc.sub(try sc.mulS(vc.latent, 1 + w), try sc.mulS(vu.latent, w));
                vm = try sc.sub(try sc.mulS(vc.camera, 1 + w), try sc.mulS(vu.camera, w));
            }
            const dt = t - tp;
            const zn = try sc.sub(z, try sc.mulS(vl, dt));
            const cn = try sc.sub(cam, try sc.mulS(vm, dt));
            try mlx.check(mlx.mlx_array_eval(zn));
            try mlx.check(mlx.mlx_array_eval(cn));
            _ = mlx.mlx_array_free(z);
            _ = mlx.mlx_array_free(cam);
            z = sc.out(zn);
            cam = sc.out(cn);
            if (prog) |p| p.emit("idol-denoise", @intCast(i + 1), o.steps);
        }
        const out = try gpa.alloc(f32, nz * self.denoiser.cfg.in_channels);
        errdefer gpa.free(out);
        try mlx.check(mlx.mlx_array_eval(z));
        @memcpy(out, (mlx.mlx_array_data_float32(z) orelse return error.NoData)[0..out.len]);
        try mlx.check(mlx.mlx_array_eval(cam));
        var cm: [5]f32 = undefined;
        @memcpy(&cm, (mlx.mlx_array_data_float32(cam) orelse return error.NoData)[0..5]);
        return .{ .latent = out, .camera = cm };
    }

    /// Latent -> Gaussians: the octree decoder's anchors, then the Gaussian decoder over them.
    pub fn decode(self: *Engine, latent: []const f32, num_gaussians: u32, seed: u64, prog: ?sse.Progress) !dec.Splats {
        const gpa = self.gpa;
        const lat = mlx.mlx_array_new_data(latent.ptr, &[_]c_int{ 1, @intCast(self.denoiser.cfg.q_token_length), @intCast(self.denoiser.cfg.in_channels) }, 3, .float32);
        defer _ = mlx.mlx_array_free(lat);
        const anchors_n: u32 = @intCast(@max(1, num_gaussians / dec.PER_ANCHOR));
        const Hook = struct {
            p: ?sse.Progress,
            fn report(ctx: ?*anyopaque, l: u32, total: u32) void {
                const h: *@This() = @ptrCast(@alignCast(ctx.?));
                if (h.p) |p| p.emit("idol-octree", l, total);
            }
        };
        var hook: Hook = .{ .p = prog };
        const pts = try dec.sampleAnchors(gpa, self.s, &self.octree, lat, anchors_n, self.level, seed +% 0x9E3779B97F4A7C15, .{ .ctx = &hook, .report = Hook.report });
        defer gpa.free(pts);
        if (prog) |p| if (p.boundary()) return error.Cancelled;
        var sc = Scope.init(gpa, self.s);
        defer sc.deinit();
        const feats = try self.gaussians.forward(&sc, pts, lat);
        try mlx.check(mlx.mlx_array_eval(feats));
        const host = (mlx.mlx_array_data_float32(feats) orelse return error.NoData)[0 .. @as(usize, anchors_n) * dec.OUT];
        if (prog) |p| p.emit("idol-gaussians", 1, 1);
        return dec.toSplats(gpa, &self.gaussians, self.rep, pts, host);
    }

    /// An image to Gaussians. `rgba` is straight-alpha RGBA8; its own alpha is the subject when it has any.
    pub fn generate(self: *Engine, rgba: []const u8, w: u32, h: u32, o: Opts, prog: ?sse.Progress) !Generated {
        const gpa = self.gpa;
        if (prog) |p| p.emit("idol-preprocess", 0, 1);
        const resized = try img.toShortSide(gpa, rgba, w, h, self.size);
        defer resized.deinit(gpa);
        // A photo with no alpha gets BiRefNet's matte; a cutout keeps its own.
        var matte: ?[]u8 = null;
        defer if (matte) |m| gpa.free(m);
        if (!img.hasRealAlpha(resized.pix)) if (self.matte) |*net| {
            if (prog) |p| p.emit("idol-matte", 0, 1);
            const px = @as(usize, resized.w) * resized.h;
            const plain = try gpa.alloc(u8, px * 3);
            defer gpa.free(plain);
            for (0..px) |i| plain[i * 3 ..][0..3].* = resized.pix[i * 4 ..][0..3].*;
            matte = try net.matte(plain, resized.w, resized.h);
        };
        const rgb = try img.prepare(gpa, resized, matte, o.margin, self.size);
        defer gpa.free(rgb);
        return self.generatePrepared(rgb, o, prog);
    }

    /// The same from the prepared 1024 x 1024 RGB8 image.
    pub fn generatePrepared(self: *Engine, rgb: []const u8, o: Opts, prog: ?sse.Progress) !Generated {
        const gpa = self.gpa;
        if (prog) |p| p.emit("idol-condition", 0, 1);
        const cond = try self.condition(rgb, o.seed +% 0x51ED270B);
        defer cond.deinit();
        if (prog) |p| p.emit("idol-condition", 1, 1);
        const s = try self.sample(cond, o, prog);
        defer gpa.free(s.latent);
        const camera = decodeCamera(s.camera);
        log.info("[idol] camera yaw={d:.1} pitch={d:.1} radius={d:.2} fov={d:.1}\n", .{ std.math.radiansToDegrees(camera.yaw), std.math.radiansToDegrees(camera.pitch), camera.radius, camera.fov });
        const splats = try self.decode(s.latent, o.num_gaussians, o.seed, prog);
        return .{ .splats = splats, .camera = camera };
    }
};

fn packHas(io: std.Io, dir: []const u8, name: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name }) catch return false;
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

fn readFile(io: std.Io, gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(gpa, .limited(1 << 20));
}

// ── tests ──

const testing = std.testing;

test "idol: the flow time grid runs 1 to 0, shifted towards noise by rescale_t" {
    try testing.expectApproxEqAbs(@as(f32, 1), timeAt(0, 50, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), timeAt(50, 50, 3), 1e-6);
    // u = 0.5, shift 3: 3 * 0.5 / (1 + 2 * 0.5) = 0.75
    try testing.expectApproxEqAbs(@as(f32, 0.75), timeAt(25, 50, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), timeAt(1, 2, 1), 1e-6);
}

test "idol: decode_camera reads a direction, an inverse distance and a half width" {
    const c = decodeCamera(.{ 0, 1, 0, 0.5, 0.25 });
    try testing.expectApproxEqAbs(@as(f32, 0), c.yaw, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2), c.radius, 1e-6);
    try testing.expectApproxEqAbs(std.math.radiansToDegrees(2 * std.math.atan(@as(f32, 0.125))), c.fov, 1e-4);
    const ortho = decodeCamera(.{ 1, 0, 0, 0, 0.5 });
    try testing.expectEqual(@as(f32, 1000), ortho.radius);
    try testing.expectApproxEqAbs(std.math.pi / 2.0, ortho.yaw, 1e-6);
}

// ── oracles (env-gated: tests/dump_idol_fixtures.py, tests/convert_idol_weights.py) ──

const env = @import("../../env.zig");
const t2 = @import("../trellis2/trellis2.zig");
const splat = @import("splat.zig");
const glue = @import("glue.zig");

fn envDir(comptime name: []const u8) ![]const u8 {
    return std.mem.span(env.named(name) orelse return error.SkipZigTest);
}

fn openEngine(s: S) !*Engine {
    const io = std.Io.Threaded.global_single_threaded.io();
    return Engine.loadOn(io, testing.allocator, try envDir("IDOL_PACK"), s);
}

fn hostRgb(gpa: std.mem.Allocator, w: Weights, key: []const u8) ![]u8 {
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer {
        _ = mlx.mlx_stream_free(sc.s);
        sc.deinit();
    }
    const f = try t2.hostF32(&sc, w.get(key).?);
    const out = try gpa.alloc(u8, f.len);
    for (out, f) |*o, v| o.* = @intFromFloat(v);
    return out;
}

fn loadFixtures() !Weights {
    return model_mod.loadWeightsFile(testing.allocator, try envDir("IDOL_FIXTURES"), "fixtures.safetensors");
}

test "idol oracle: the crop of the reference's matte reproduces its prepared image (IDOL_FIXTURES)" {
    const gpa = testing.allocator;
    var fx = try loadFixtures();
    defer fx.deinit();
    const rgb_in = try hostRgb(gpa, fx, "rgb_in");
    defer gpa.free(rgb_in);
    const prepared = try hostRgb(gpa, fx, "prepared");
    defer gpa.free(prepared);
    const alpha = try hostRgb(gpa, fx, "birefnet_alpha");
    defer gpa.free(alpha);
    const sh = mlx.getShape(fx.get("rgb_in").?);
    const w: u32 = @intCast(sh[1]);
    const h: u32 = @intCast(sh[0]);
    const rgba = try gpa.alloc(u8, w * h * 4);
    defer gpa.free(rgba);
    for (0..w * h) |i| {
        rgba[i * 4 ..][0..3].* = rgb_in[i * 3 ..][0..3].*;
        rgba[i * 4 + 3] = 255;
    }
    const resized = try img.toShortSide(gpa, rgba, w, h, img.IMAGE_SIZE);
    defer resized.deinit(gpa);
    try testing.expectEqual(@as(usize, resized.w) * resized.h, alpha.len);
    const got = try img.prepare(gpa, resized, alpha, img.MARGIN_SYNTHETIC, img.IMAGE_SIZE);
    defer gpa.free(got);
    var diff: usize = 0;
    var worst: u8 = 0;
    for (got, prepared) |a, b| {
        diff += @intFromBool(a != b);
        worst = @max(worst, if (a > b) a - b else b - a);
    }
    std.debug.print("idol crop: {d} of {d} bytes differ, worst {d}\n", .{ diff, got.len, worst });
    try testing.expect(worst <= 2);
}

test "idol oracle: DINOv3 and FLUX.2 VAE tokens match the reference (IDOL_FIXTURES, IDOL_PACK)" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var fx = try loadFixtures();
    defer fx.deinit();
    const prepared = try hostRgb(gpa, fx, "prepared");
    defer gpa.free(prepared);
    const e = try openEngine(s);
    defer e.deinit();
    const c = try e.condition(prepared, null);
    defer c.deinit();
    const sampled = try e.condition(prepared, 7);
    defer sampled.deinit();
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    try t2.expectClose("feature1 (DINOv3)", &sc, c.f1, fx.get("feature1").?, 0.999, 0.02);
    try t2.expectClose("feature2 mean (VAE)", &sc, c.f2, fx.get("feature2_mean").?, 0.998, 0.02);
    // Two independent samples share only the mean: cos ~0.84^2 = 0.70, equal energy.
    try t2.expectClose("feature2 sample (VAE)", &sc, sampled.f2, fx.get("feature2").?, 0.65, 0.05);
}

test "idol oracle: one velocity, guided and not, matches the reference (IDOL_FIXTURES, IDOL_PACK)" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var fx = try loadFixtures();
    defer fx.deinit();
    const e = try openEngine(s);
    defer e.deinit();
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const f1 = fx.get("feature1").?;
    const f2 = fx.get("feature2").?;
    const ctx = try e.denoiser.context(&sc, f1, f2);
    const ctx_u = try e.denoiser.context(&sc, try sc.zeros(mlx.getShape(f1), .float32), try sc.zeros(mlx.getShape(f2), .float32));
    const t = try t2.hostF32(&sc, fx.get("t").?);
    const v = try e.denoiser.velocity(&sc, fx.get("noise_latent").?, fx.get("noise_camera").?, t[0], ctx);
    try t2.expectClose("v_latent", &sc, v.latent, fx.get("v_latent").?, 0.995, 0.05);
    try t2.expectClose("v_camera", &sc, v.camera, fx.get("v_camera").?, 0.995, 0.05);
    const vu = try e.denoiser.velocity(&sc, fx.get("noise_latent").?, fx.get("noise_camera").?, t[0], ctx_u);
    try t2.expectClose("v_latent_uncond", &sc, vu.latent, fx.get("v_latent_uncond").?, 0.995, 0.05);
}

fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        aa += @as(f64, x) * x;
        bb += @as(f64, y) * y;
    }
    return dot / @sqrt(aa * bb);
}

test "idol oracle: the fused glue keeps the velocity as close to the reference as the chain it replaces (IDOL_FIXTURES, IDOL_PACK)" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var fx = try loadFixtures();
    defer fx.deinit();
    const e = try openEngine(s);
    defer e.deinit();
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const t = try t2.hostF32(&sc, fx.get("t").?);
    const want = try t2.hostF32(&sc, fx.get("v_latent").?);
    var got: [2][]const f32 = undefined;
    for (&got, 0..) |*out, arm| {
        glue.disabled_for_test = arm == 0;
        defer glue.disabled_for_test = false;
        const ctx = try e.denoiser.context(&sc, fx.get("feature1").?, fx.get("feature2").?);
        out.* = try t2.hostF32(&sc, (try e.denoiser.velocity(&sc, fx.get("noise_latent").?, fx.get("noise_camera").?, t[0], ctx)).latent);
    }
    const chain = cosine(got[0], want);
    const fused = cosine(got[1], want);
    std.debug.print("idol velocity vs the reference: chain cos {d:.7}, fused cos {d:.7}; fused vs chain {d:.7}\n", .{ chain, fused, cosine(got[0], got[1]) });
    try testing.expect(cosine(got[0], got[1]) > 0.9999);
    try testing.expect(fused > chain - 1e-4);
}

test "idol oracle: the w8a8 route stays within its bar of the packed velocity (IDOL_FIXTURES, IDOL_PACK)" {
    if (!@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var fx = try loadFixtures();
    defer fx.deinit();
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const prev = w8a8.w8a8State().enabled;
    defer w8a8.w8a8State().enabled = prev;
    const t = try t2.hostF32(&sc, fx.get("t").?);
    var v: [2]A = undefined;
    for (&v, 0..) |*out, arm| {
        w8a8.w8a8State().enabled = arm == 1;
        const e = try openEngine(s);
        defer e.deinit();
        const ctx = try e.denoiser.context(&sc, fx.get("feature1").?, fx.get("feature2").?);
        out.* = (try e.denoiser.velocity(&sc, fx.get("noise_latent").?, fx.get("noise_camera").?, t[0], ctx)).latent;
        try mlx.check(mlx.mlx_array_eval(out.*));
    }
    w8a8.dropMemo();
    const g = try w8a8.agreement(v[1], v[0], s);
    const want = try t2.hostF32(&sc, fx.get("v_latent").?);
    std.debug.print("idol w8a8 velocity: int8 vs packed cos {d:.6} rms ratio {d:.5}; to the reference packed {d:.6} int8 {d:.6}\n", .{ g.cos, g.rms_ratio, cosine(try t2.hostF32(&sc, v[0]), want), cosine(try t2.hostF32(&sc, v[1]), want) });
    try testing.expect(g.cos < 0.99999); // the route engaged: packed numerics would be identical
    try testing.expect(g.cos > 0.998);
    try testing.expectApproxEqAbs(@as(f64, 1.0), g.rms_ratio, 0.02);
}

test "idol oracle: the octree logits and the Gaussians match the reference (IDOL_FIXTURES, IDOL_PACK)" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var fx = try loadFixtures();
    defer fx.deinit();
    const e = try openEngine(s);
    defer e.deinit();
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const lat = fx.get("oct_latent").?;
    const centres = try t2.hostF32(&sc, fx.get("oct_coords").?);
    const level = (try t2.hostF32(&sc, fx.get("oct_level").?))[0];
    const logits = try e.octree.logits(&sc, centres, level, lat);
    try t2.expectClose("octree logits", &sc, logits, fx.get("oct_logits").?, 0.995, 0.05);

    const pts = try t2.hostF32(&sc, fx.get("gs_points").?);
    const raw = try e.gaussians.forward(&sc, pts, lat);
    try t2.expectClose("gaussian raw", &sc, raw, fx.get("gs_raw").?, 0.995, 0.05);
    const host = try t2.hostF32(&sc, fx.get("gs_raw").?); // activate the REFERENCE's raw outputs: isolates the activation maths
    var splats = try dec.toSplats(gpa, &e.gaussians, e.rep, pts, host);
    defer splats.deinit(gpa);
    const wrap = struct {
        fn arr(sc_: *Scope, data: []const f32, shape: []const c_int) !A {
            return sc_.fromF32(data, shape);
        }
    };
    const n: c_int = @intCast(splats.n);
    try t2.expectClose("xyz", &sc, try wrap.arr(&sc, splats.xyz, &.{ n, 3 }), fx.get("gs_xyz").?, 0.99999, 1e-4);
    try t2.expectClose("colour", &sc, try wrap.arr(&sc, splats.dc, &.{ n, 3 }), fx.get("gs_color").?, 0.99999, 1e-4);
    try t2.expectClose("scaling", &sc, try wrap.arr(&sc, splats.scale, &.{ n, 3 }), fx.get("gs_scaling").?, 0.99999, 1e-4);
    try t2.expectClose("rotation", &sc, try wrap.arr(&sc, splats.rot, &.{ n, 4 }), fx.get("gs_rotation").?, 0.99999, 1e-4);
    try t2.expectClose("opacity", &sc, try wrap.arr(&sc, splats.opacity, &.{ n, 1 }), fx.get("gs_opacity").?, 0.99999, 1e-4);
}

test "idol oracle: four guided steps from the reference's noise land on its latent (IDOL_FIXTURES, IDOL_PACK)" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var fx = try loadFixtures();
    defer fx.deinit();
    const e = try openEngine(s);
    defer e.deinit();
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const noise = try t2.hostF32(&sc, fx.get("noise_latent").?);
    const cam = try t2.hostF32(&sc, fx.get("noise_camera").?);
    const cond: Engine.Cond = .{ .f1 = fx.get("feature1").?, .f2 = fx.get("feature2").? };
    const r = try e.sample(cond, .{ .steps = 4, .noise_latent = noise, .noise_camera = cam }, null);
    defer gpa.free(r.latent);
    const lat = try sc.fromF32(r.latent, &.{ 1, 8192, 16 });
    try t2.expectClose("sample latent", &sc, lat, fx.get("sample_latent").?, 0.99, 0.05);
    try t2.expectClose("sample camera", &sc, try sc.fromF32(&r.camera, &.{ 1, 1, 5 }), fx.get("sample_camera").?, 0.99, 0.1);
}

test "idol end to end: an image becomes a splat, written as PLY and GLB (IDOL_PACK, IDOL_IMAGE, IDOL_OUT)" {
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const out_dir = try envDir("IDOL_OUT");
    const bytes = try @import("../trellis2/pipeline.zig").readFileAny(io, gpa, try envDir("IDOL_IMAGE"));
    defer gpa.free(bytes);
    const photo = @import("../../sdk/media/image_pixels.zig").decodeRgba(gpa, bytes) orelse return error.BadImage;
    defer photo.deinit(gpa);
    const e = try openEngine(s);
    defer e.deinit();
    const steps: u32 = if (env.named("IDOL_STEPS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 50;
    var gen = try e.generate(photo.pix, photo.w, photo.h, .{ .steps = steps }, null);
    defer gen.deinit(gpa);
    const ply = try splat.writePly(gpa, &gen.splats);
    defer gpa.free(ply);
    const glb = try splat.writeGlb(gpa, &gen.splats);
    defer gpa.free(glb);
    for ([_]struct { name: []const u8, bytes: []const u8 }{ .{ .name = "splat.ply", .bytes = ply }, .{ .name = "splat.glb", .bytes = glb } }) |f| {
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ out_dir, f.name });
        defer gpa.free(path);
        const file = try std.Io.Dir.createFileAbsolute(io, path, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, f.bytes);
    }
    std.debug.print("idol: {d} Gaussians, camera fov {d:.1}\n", .{ gen.splats.n, gen.camera.fov });
    try testing.expect(gen.splats.n >= config.MIN_GAUSSIANS);
}

test "idol oracle: BiRefNet's matte of the resized image matches the reference's (IDOL_FIXTURES, IDOL_PACK)" {
    const gpa = testing.allocator;
    var fx = try loadFixtures();
    defer fx.deinit();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var net = try birefnet.Net.load(gpa, s, try model_mod.loadWeightsFile(gpa, try envDir("IDOL_PACK"), "birefnet.safetensors"));
    defer net.deinit();
    const rgb_in = try hostRgb(gpa, fx, "rgb_in");
    defer gpa.free(rgb_in);
    const want = try hostRgb(gpa, fx, "birefnet_alpha");
    defer gpa.free(want);
    const sh = mlx.getShape(fx.get("rgb_in").?);
    const w: u32 = @intCast(sh[1]);
    const h: u32 = @intCast(sh[0]);
    const rgba = try gpa.alloc(u8, w * h * 4);
    defer gpa.free(rgba);
    for (0..w * h) |i| {
        rgba[i * 4 ..][0..3].* = rgb_in[i * 3 ..][0..3].*;
        rgba[i * 4 + 3] = 255;
    }
    const resized = try img.toShortSide(gpa, rgba, w, h, img.IMAGE_SIZE);
    defer resized.deinit(gpa);
    const px = @as(usize, resized.w) * resized.h;
    const plain = try gpa.alloc(u8, px * 3);
    defer gpa.free(plain);
    for (0..px) |i| plain[i * 3 ..][0..3].* = resized.pix[i * 4 ..][0..3].*;
    const got = try net.matte(plain, resized.w, resized.h);
    defer gpa.free(got);
    try testing.expectEqual(want.len, got.len);
    var sum: u64 = 0;
    var far: usize = 0;
    var inter: usize = 0;
    var uni: usize = 0;
    for (got, want) |a, b| {
        const d: u8 = if (a > b) a - b else b - a;
        sum += d;
        far += @intFromBool(d > 16);
        inter += @intFromBool(a > 127 and b > 127);
        uni += @intFromBool(a > 127 or b > 127);
    }
    std.debug.print("idol matte: mean abs diff {d:.3}, {d} of {d} pixels differ by more than 16, IoU {d:.5}\n", .{ @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(got.len)), far, got.len, @as(f64, @floatFromInt(inter)) / @as(f64, @floatFromInt(@max(uni, 1))) });
    try testing.expect(@as(f64, @floatFromInt(inter)) / @as(f64, @floatFromInt(@max(uni, 1))) > 0.995);
    try testing.expect(far * 200 < got.len);
}
