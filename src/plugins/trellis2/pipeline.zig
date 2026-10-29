//! TRELLIS.2 image-to-3D (`microsoft/TRELLIS.2-4B`) end to end: DINOv3 conditioning, the sparse-structure
//! flow and decoder, the shape flow (a 512 pass, then a 1024 cascade over the voxels its decoder upsamples
//! to), the shape decoder and O-Voxel dual grid, the texture flow and decoder, then cleanup, UV unwrap and a
//! bake of the decoded PBR voxels into a textured GLB. Mirrors `Trellis2ImageTo3DPipeline.run` +
//! `o_voxel.postprocess.to_glb` (microsoft/TRELLIS.2, MIT) without the remesh branch and the background remover.
//!
//! Pack layout (the published one, as the MLX packs keep it): `pipeline.json`, `ckpts/`, the SS decoder under
//! `microsoft/TRELLIS-image-large/ckpts/`, and `dinov3/{config.json,model.safetensors}`. Every model loads for
//! its own stage and is freed before the next, so the peak is one DiT plus its activations.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const scope_mod = @import("../../mlx/scope.zig");
const sse = @import("../../sdk/media/progress.zig");
const glb = @import("../../sdk/mesh/glb.zig");
const png = @import("../../sdk/media/png.zig");
const uvwrap = @import("../../sdk/mesh/uvwrap.zig");
const texinpaint = @import("../../sdk/mesh/texinpaint.zig");
const qvis = @import("../../sdk/vision/qwen_vision.zig");
const t2 = @import("trellis2.zig");
const sp = @import("sparse.zig");
const dino = @import("../../sdk/vision/dinov3.zig");
const tmesh = @import("mesh.zig");
const remesh = @import("remesh.zig");
const io_util = @import("../../sdk/util/io_util.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const Scope = scope_mod.Scope;

const LATENT: usize = 32;
const SS_GRID: usize = 16;
const SS_RES: usize = 64;
const SS_POOLED: usize = 32;
/// The cascade lowers its resolution until the 1024-flow token count is under this (`max_num_tokens`).
pub const MAX_TOKENS: usize = 49152;
pub const LR_RES: u32 = 512;
/// `alpha > 0.8 * 255` decides the subject's box.
const ALPHA_CUT: u8 = 204;
const MAX_INPUT_SIDE: u32 = 1024;

pub const Opts = struct {
    seed: u64 = 42,
    /// Finest shape resolution: 512, or 1024 / 1536 through the cascade.
    resolution: u32 = 1024,
    /// Sampler steps for all three flows; 0 keeps the published 12.
    steps: u32 = 0,
    texture_size: u32 = 2048,
    decimation_target: u32 = 300_000,
    /// Pixal3D's export axes (`exportAxes`).
    pixal3d_axes: bool = false,
    /// The GLB material's finish (`glb.TexturedMesh`): 1 and 0 keep what the model predicted.
    roughness_scale: f32 = 1.0,
    clearcoat: f32 = 0,
};

/// One flow's conditioning: the image tokens its cross-attention reads and, for Pixal3D, the pixel-aligned
/// features of each of its tokens.
pub const StageCond = struct { global: A, proj: ?A = null };

const Norm = struct { mean: [LATENT]f32, std: [LATENT]f32 };

const Pipeline = struct {
    args: struct {
        models: struct {
            sparse_structure_decoder: []const u8,
            sparse_structure_flow_model: []const u8,
            shape_slat_decoder: []const u8,
            shape_slat_flow_model_512: []const u8,
            shape_slat_flow_model_1024: []const u8,
            tex_slat_decoder: []const u8,
            tex_slat_flow_model_512: []const u8 = "",
            tex_slat_flow_model_1024: []const u8,
        },
        shape_slat_normalization: Norm,
        tex_slat_normalization: Norm,
    },
};

/// What the GPU stages decode: the finest voxel set (`[M, 4]`), the dual-grid head over it (`[M, 7]`), its PBR
/// attributes (`[M, 6]`: base colour, metallic, roughness, alpha in 0..1) and the grid resolution.
pub const Voxels = struct {
    coords: []i32,
    raw: []f32,
    attrs: []f32,
    res: u32,

    pub fn deinit(self: *Voxels, alloc: std.mem.Allocator) void {
        alloc.free(self.coords);
        alloc.free(self.raw);
        alloc.free(self.attrs);
    }
};

/// Which finest shape resolutions exist: the cascade's 1024 and 1536, and 512 where the pack has its texture flow.
pub fn resolutionServed(res: u32, has_512: bool) bool {
    return switch (res) {
        512 => has_512,
        1024, 1536 => true,
        else => false,
    };
}

/// Each flow stage draws its own noise: the request's seed offset by the stage, wrapping at the top of u64.
pub fn stageSeed(seed: u64, stage: u64) u64 {
    return seed +% stage;
}

/// `rel` when `<dir>/<rel>.safetensors` is on disk, else "": the pack the app downloads leaves the optional 512
/// texture flow out, though the published `pipeline.json` still names it.
fn onDiskOrEmpty(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, rel: []const u8) ![]const u8 {
    if (rel.len == 0) return rel;
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.safetensors", .{ dir, rel });
    defer gpa.free(path);
    std.Io.Dir.accessAbsolute(io, path, .{}) catch return "";
    return rel;
}

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// The caller's stream (borrowed).
    s: S,
    root: []u8,
    dino_dir: []u8,
    arena: std.heap.ArenaAllocator,
    models: @FieldType(@FieldType(Pipeline, "args"), "models"),
    shape_norm: Norm,
    tex_norm: Norm,

    pub fn loadOn(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, s: S) !*Engine {
        const self = try gpa.create(Engine);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
        self.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.root = try gpa.dupe(u8, dir);
        errdefer gpa.free(self.root);
        const pj = try std.fmt.allocPrint(gpa, "{s}/pipeline.json", .{dir});
        defer gpa.free(pj);
        const bytes = try readFile(io, gpa, pj);
        defer gpa.free(bytes);
        const parsed = try std.json.parseFromSliceLeaky(Pipeline, a, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        self.models = parsed.args.models;
        self.shape_norm = parsed.args.shape_slat_normalization;
        self.tex_norm = parsed.args.tex_slat_normalization;
        self.dino_dir = try std.fmt.allocPrint(gpa, "{s}/dinov3", .{dir});
        errdefer gpa.free(self.dino_dir);
        self.s = s;
        self.models.tex_slat_flow_model_512 = try onDiskOrEmpty(io, gpa, dir, self.models.tex_slat_flow_model_512);
        for ([_][]const u8{ self.models.sparse_structure_decoder, self.models.sparse_structure_flow_model, self.models.shape_slat_decoder, self.models.shape_slat_flow_model_512, self.models.shape_slat_flow_model_1024, self.models.tex_slat_decoder, self.models.tex_slat_flow_model_1024 }) |rel| {
            const p = try std.fmt.allocPrint(gpa, "{s}/{s}.safetensors", .{ dir, rel });
            defer gpa.free(p);
            std.Io.Dir.accessAbsolute(io, p, .{}) catch {
                log.err("[trellis2] missing weights: {s}\n", .{p});
                return error.MissingTrellis2Weights;
            };
        }
        const dp = try std.fmt.allocPrint(gpa, "{s}/model.safetensors", .{self.dino_dir});
        defer gpa.free(dp);
        std.Io.Dir.accessAbsolute(io, dp, .{}) catch {
            log.err("[trellis2] missing DINOv3 conditioner: {s} (the gated facebook/dinov3-vitl16-pretrain-lvd1689m weights go in <pack>/dinov3/)\n", .{dp});
            return error.MissingTrellis2Weights;
        };
        log.info("[mesh] TRELLIS.2 engine ready ({s})\n", .{dir});
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.gpa.free(self.root);
        self.gpa.free(self.dino_dir);
        self.arena.deinit();
        self.gpa.destroy(self);
    }

    /// `rel` ("ckpts/<stem>") split into its directory under the pack and the stem.
    fn split(self: *const Engine, rel: []const u8) !struct { dir: []u8, stem: []const u8 } {
        const slash = std.mem.lastIndexOfScalar(u8, rel, '/') orelse return error.BadModelPath;
        return .{ .dir = try std.fmt.allocPrint(self.gpa, "{s}/{s}", .{ self.root, rel[0..slash] }), .stem = rel[slash + 1 ..] };
    }

    /// The 512 texture flow is optional in a pack; the 512 pipeline needs it.
    pub fn hasResolution512(self: *const Engine) bool {
        return self.models.tex_slat_flow_model_512.len > 0;
    }

    pub fn supportsResolution(self: *const Engine, res: u32) bool {
        return resolutionServed(res, self.hasResolution512());
    }

    pub fn generateGlb(self: *Engine, alloc: std.mem.Allocator, rgba: []const u8, w: u32, h: u32, opts: Opts, prog: ?sse.Progress) ![]u8 {
        var vox = try self.generateVoxels(alloc, rgba, w, h, opts, prog);
        defer vox.deinit(alloc);
        return meshToGlb(alloc, &vox, opts, prog);
    }

    /// The GPU half: the decoded voxel set with its dual-grid head and PBR attributes.
    pub fn generateVoxels(self: *Engine, alloc: std.mem.Allocator, rgba: []const u8, w: u32, h: u32, opts: Opts, prog: ?sse.Progress) !Voxels {
        if (!self.supportsResolution(opts.resolution)) return error.UnsupportedResolution;
        const img = try prepareImage(alloc, rgba, w, h);
        defer alloc.free(img.rgb);

        var cond = try self.conditioning(alloc, img, opts.resolution > LR_RES, prog);
        defer cond.deinit();

        const ss_coords = try self.sparseStructure(alloc, self.models.sparse_structure_flow_model, .{ .global = cond.c512 }, opts, "trellis-structure", prog);
        defer alloc.free(ss_coords);

        var shape = try self.shapeLatent(alloc, ss_coords, &cond, opts, prog);
        defer shape.deinit(alloc);
        var geo = try self.decodeShape(alloc, &shape, prog);
        defer geo.deinit(alloc);
        var cfg = t2.TEX_SAMPLER;
        if (opts.steps != 0) cfg.steps = opts.steps;
        const lr = opts.resolution == LR_RES;
        const tex_rel = if (lr) self.models.tex_slat_flow_model_512 else self.models.tex_slat_flow_model_1024;
        const norm = try self.flowLatent(alloc, tex_rel, shape.coords, .{ .global = if (lr) cond.c512 else cond.c1024.? }, shape.norm, cfg, stageSeed(opts.seed, 3), "trellis-texture", prog);
        defer _ = mlx.mlx_array_free(norm);
        return self.texturedVoxels(alloc, &shape, &geo, norm, prog);
    }

    /// The voxels a decoded shape and its texture latent (normalized, `[1, n, 32]`) describe.
    pub fn texturedVoxels(self: *Engine, alloc: std.mem.Allocator, shape: *const Shape, geo: *Geometry, tex_norm: A, prog: ?sse.Progress) !Voxels {
        const attrs = try self.decodeTexture(alloc, tex_norm, &geo.plan, prog);
        errdefer alloc.free(attrs);
        const finest = geo.plan.levels.items[geo.plan.levels.items.len - 1].coords;
        const coords = try alloc.dupe(i32, finest);
        errdefer alloc.free(coords);
        return .{ .coords = coords, .raw = try alloc.dupe(f32, geo.raw), .attrs = attrs, .res = shape.res };
    }

    // ── conditioning ────────────────────────────────────────────────────────

    const Cond = struct {
        c512: A,
        c1024: ?A,

        fn deinit(self: *Cond) void {
            _ = mlx.mlx_array_free(self.c512);
            if (self.c1024) |c| _ = mlx.mlx_array_free(c);
        }
    };

    fn conditioning(self: *Engine, alloc: std.mem.Allocator, img: Prepared, want_1024: bool, prog: ?sse.Progress) !Cond {
        if (prog) |p| p.emit("trellis-cond", 0, 1);
        var enc = try dino.Encoder.open(self.io, alloc, self.dino_dir);
        defer enc.deinit();
        var out = Cond{ .c512 = try encodeAt(alloc, &enc, self.s, img, 512), .c1024 = null };
        errdefer out.deinit();
        if (want_1024) out.c1024 = try encodeAt(alloc, &enc, self.s, img, 1024);
        if (prog) |p| p.emit("trellis-cond", 1, 1);
        return out;
    }

    // ── sparse structure ────────────────────────────────────────────────────

    /// `[n, 4]` (batch 0, x, y, z) occupied cells at 32^3, from the flow at `flow_rel` (its proj features, if
    /// any, one row per cell of the 16^3 grid in `gridCoords` order), reporting progress as `stage`.
    pub fn sparseStructure(self: *Engine, alloc: std.mem.Allocator, flow_rel: []const u8, cond: StageCond, opts: Opts, stage: []const u8, prog: ?sse.Progress) ![]i32 {
        const grid = try t2.gridCoords(alloc, SS_GRID);
        defer alloc.free(grid);
        var z: A = undefined;
        {
            const m = try self.split(flow_rel);
            defer self.gpa.free(m.dir);
            var od = try t2.openDit(self.io, alloc, m.dir, m.stem, .bfloat16);
            defer od.deinit();
            var sc = Scope.init(alloc, self.s);
            defer sc.deinit();
            const noise = try normalNoise(&sc, &.{ 1, SS_GRID * SS_GRID * SS_GRID, 8 }, opts.seed);
            const rope = try t2.Rope.init(&sc, grid, od.dit.head_dim);
            const flow = t2.Flow{ .dit = &od.dit, .rope = rope, .cond = cond.global, .neg_cond = try sc.mulS(cond.global, 0), .proj = cond.proj };
            var cfg = t2.SS_SAMPLER;
            if (opts.steps != 0) cfg.steps = opts.steps;
            const ctx = StepCtx{ .prog = prog, .stage = stage };
            z = try flow.sample(alloc, self.s, noise, cfg, &ctx, onStep);
        }
        defer _ = mlx.mlx_array_free(z);

        const m = try self.split(self.models.sparse_structure_decoder);
        defer self.gpa.free(m.dir);
        const file = try std.fmt.allocPrint(alloc, "{s}.safetensors", .{m.stem});
        defer alloc.free(file);
        var w = try model_mod.loadWeightsFile(alloc, m.dir, file);
        defer w.deinit();
        var dec = try t2.SsDecoder.load(alloc, &w, .float32);
        defer dec.deinit();
        var sc = Scope.init(alloc, self.s);
        defer sc.deinit();
        const lat = try sc.reshape(try sc.transpose(z, &.{ 0, 2, 1 }), &.{ 1, 8, SS_GRID, SS_GRID, SS_GRID });
        const logits = try hostF32(&sc, try dec.forward(&sc, lat));
        const cells = try t2.occupiedCells(alloc, logits, SS_RES, SS_POOLED);
        if (cells.len == 0) {
            alloc.free(cells);
            return error.EmptyStructure;
        }
        log.info("[trellis2] sparse structure: {d} voxels at {d}^3\n", .{ cells.len / 4, SS_POOLED });
        return cells;
    }

    // ── shape ───────────────────────────────────────────────────────────────

    pub const Shape = struct {
        /// `[n, 4]` voxel coordinates of the latent and its `[n, 32]` NORMALIZED values (the texture flow's concat).
        coords: []i32,
        norm: A,
        /// The shape resolution the latent's voxels decode to.
        res: u32,

        pub fn deinit(self: *Shape, alloc: std.mem.Allocator) void {
            alloc.free(self.coords);
            _ = mlx.mlx_array_free(self.norm);
        }
    };

    fn shapeLatent(self: *Engine, alloc: std.mem.Allocator, ss_coords: []const i32, cond: *const Cond, opts: Opts, prog: ?sse.Progress) !Shape {
        var cfg = t2.SHAPE_SAMPLER;
        if (opts.steps != 0) cfg.steps = opts.steps;
        if (opts.resolution == LR_RES) {
            const coords = try alloc.dupe(i32, ss_coords);
            errdefer alloc.free(coords);
            const norm = try self.flowLatent(alloc, self.models.shape_slat_flow_model_512, coords, .{ .global = cond.c512 }, null, cfg, stageSeed(opts.seed, 1), "trellis-shape", prog);
            return .{ .coords = coords, .norm = norm, .res = LR_RES };
        }
        // 512 pass, then the voxels its decoder upsamples to, quantized onto the finer grid.
        const lr_norm = try self.flowLatent(alloc, self.models.shape_slat_flow_model_512, ss_coords, .{ .global = cond.c512 }, null, cfg, stageSeed(opts.seed, 1), "trellis-shape-512", prog);
        defer _ = mlx.mlx_array_free(lr_norm);
        const hr = try self.upsampleCoords(alloc, ss_coords, lr_norm, prog);
        defer alloc.free(hr);
        var res = opts.resolution;
        var coords: []i32 = undefined;
        while (true) {
            coords = try quantizeUnique(alloc, hr, LR_RES, res / 16);
            if (coords.len / 4 < MAX_TOKENS or res <= 1024) break;
            alloc.free(coords);
            res -= 128;
        }
        errdefer alloc.free(coords);
        if (res != opts.resolution) log.info("[trellis2] token cap: resolution lowered to {d}\n", .{res});
        log.info("[trellis2] cascade: {d} tokens at {d}\n", .{ coords.len / 4, res });
        const norm = try self.flowLatent(alloc, self.models.shape_slat_flow_model_1024, coords, .{ .global = cond.c1024.? }, null, cfg, stageSeed(opts.seed, 2), "trellis-shape-1024", prog);
        return .{ .coords = coords, .norm = norm, .res = res };
    }

    /// The 512 voxels (`[m, 4]`, caller frees) the shape decoder subdivides a 512 latent's `coords` into.
    pub fn upsampleCoords(self: *Engine, alloc: std.mem.Allocator, coords: []const i32, lr_norm: A, prog: ?sse.Progress) ![]i32 {
        var plan = try sp.Plan.init(alloc, coords);
        defer plan.deinit();
        const m = try self.split(self.models.shape_slat_decoder);
        defer self.gpa.free(m.dir);
        var dec = try sp.openDecoder(self.io, alloc, m.dir, m.stem, .float16);
        defer dec.deinit();
        var sc = Scope.init(alloc, self.s);
        defer sc.deinit();
        const lr = try self.denormalize(&sc, lr_norm, self.shape_norm);
        if (prog) |p| p.emit("trellis-upsample", 0, 1);
        try dec.dec.upsample(&sc, lr, &plan);
        if (prog) |p| p.emit("trellis-upsample", 1, 1);
        return alloc.dupe(i32, plan.levels.items[plan.levels.items.len - 1].coords);
    }

    /// One flow over `coords`: noise (plus `concat` channels) to a NORMALIZED latent `[1, n, 32]` (caller frees).
    pub fn flowLatent(self: *Engine, alloc: std.mem.Allocator, rel: []const u8, coords: []const i32, cond: StageCond, concat: ?A, cfg: t2.SamplerCfg, seed: u64, stage: []const u8, prog: ?sse.Progress) !A {
        const m = try self.split(rel);
        defer self.gpa.free(m.dir);
        var od = try t2.openDit(self.io, alloc, m.dir, m.stem, .bfloat16);
        defer od.deinit();
        var sc = Scope.init(alloc, self.s);
        defer sc.deinit();
        const n: c_int = @intCast(coords.len / 4);
        const noise = try normalNoise(&sc, &.{ 1, n, LATENT }, seed);
        const rope = try t2.Rope.init(&sc, coords, od.dit.head_dim);
        const flow = t2.Flow{ .dit = &od.dit, .rope = rope, .cond = cond.global, .neg_cond = try sc.mulS(cond.global, 0), .concat = concat, .proj = cond.proj };
        const ctx = StepCtx{ .prog = prog, .stage = stage };
        return flow.sample(alloc, self.s, noise, cfg, &ctx, onStep);
    }

    /// `x * std + mean` over `[1, n, 32]`, returned as `[n, 32]`.
    fn denormalize(_: *Engine, sc: *Scope, x: A, norm: Norm) !A {
        const mean = try sc.fromF32(&norm.mean, &.{LATENT});
        const sd = try sc.fromF32(&norm.std, &.{LATENT});
        const n = mlx.getShape(x)[1];
        return sc.reshape(try sc.add(try sc.mul(x, sd), mean), &.{ n, LATENT });
    }

    pub const Geometry = struct {
        plan: sp.Plan,
        /// Decoder head `[M, 7]` over the finest level.
        raw: []f32,

        pub fn deinit(self: *Geometry, alloc: std.mem.Allocator) void {
            self.plan.deinit();
            alloc.free(self.raw);
        }
    };

    pub fn decodeShape(self: *Engine, alloc: std.mem.Allocator, shape: *const Shape, prog: ?sse.Progress) !Geometry {
        if (prog) |p| p.emit("trellis-shape-decode", 0, 1);
        const m = try self.split(self.models.shape_slat_decoder);
        defer self.gpa.free(m.dir);
        var dec = try sp.openDecoder(self.io, alloc, m.dir, m.stem, .float16);
        defer dec.deinit();
        var plan = try sp.Plan.init(alloc, shape.coords);
        errdefer plan.deinit();
        var sc = Scope.init(alloc, self.s);
        defer sc.deinit();
        const z = try self.denormalize(&sc, shape.norm, self.shape_norm);
        const raw = try hostF32(&sc, try dec.dec.decode(&sc, z, &plan));
        if (prog) |p| p.emit("trellis-shape-decode", 1, 1);
        return .{ .plan = plan, .raw = try alloc.dupe(f32, raw) };
    }

    // ── texture ─────────────────────────────────────────────────────────────

    /// PBR attributes `[M, 6]` (base colour, metallic, roughness, alpha in 0..1) over the finest level, from the
    /// texture latent (normalized, `[1, n, 32]`).
    fn decodeTexture(self: *Engine, alloc: std.mem.Allocator, norm: A, plan: *sp.Plan, prog: ?sse.Progress) ![]f32 {
        if (prog) |p| p.emit("trellis-texture-decode", 0, 1);
        const m = try self.split(self.models.tex_slat_decoder);
        defer self.gpa.free(m.dir);
        var dec = try sp.openDecoder(self.io, alloc, m.dir, m.stem, .float16);
        defer dec.deinit();
        var sc = Scope.init(alloc, self.s);
        defer sc.deinit();
        const z = try self.denormalize(&sc, norm, self.tex_norm);
        const raw = try hostF32(&sc, try dec.dec.decodeGuided(&sc, z, plan));
        const out = try alloc.dupe(f32, raw);
        for (out) |*v| v.* = v.* * 0.5 + 0.5;
        if (prog) |p| p.emit("trellis-texture-decode", 1, 1);
        return out;
    }

};


/// Cell count the remesh runs at: finer than the decimated mesh it feeds, so detail is cut by the simplifier, not here.
const REMESH_RES: u32 = 512;

/// The CPU half: dual grid, remesh, cleanup, UV unwrap, texture bake, GLB.
pub fn meshToGlb(alloc: std.mem.Allocator, vox: *const Voxels, opts: Opts, prog: ?sse.Progress) ![]u8 {
    // Every float-to-int cast downstream (slabs, cells, samples) takes positions built from this.
    for (vox.raw) |v| if (!std.math.isFinite(v)) return error.NonFiniteDecoderOutput;
    const res = vox.res;
    if (prog) |p| p.emit("trellis-mesh", 0, 1);
    var raw_mesh = try sse.offload(tmesh.dualGridMesh, .{ alloc, vox.coords, vox.raw, res });
    defer raw_mesh.deinit(alloc);
    log.info("[trellis2] dual grid: {d} vertices, {d} faces\n", .{ raw_mesh.verts.len / 3, raw_mesh.tris.len / 3 });
    const tube = try sse.offload(remesh.remesh, .{ alloc, raw_mesh, @min(res, REMESH_RES) });
    var mesh = try sse.offload(tmesh.simplifyAndClean, .{ alloc, tube, opts.decimation_target });
    defer mesh.deinit(alloc);
    log.info("[trellis2] cleaned mesh: {d} vertices, {d} faces\n", .{ mesh.verts.len / 3, mesh.tris.len / 3 });
    if (prog) |p| {
        if (p.boundary()) return error.Cancelled;
        p.emit("trellis-mesh", 1, 1);
        p.emit("trellis-uv", 0, 1);
    }
    var uw = try sse.offload(uvwrap.parametrizeParallel, .{ alloc, mesh.verts, mesh.tris });
    defer uw.deinit(alloc);
    log.info("[trellis2] unwrapped: {d} vertices, {d} charts\n", .{ uw.vmapping.len, uw.chart_count });
    const positions = try alloc.alloc(f32, uw.vmapping.len * 3);
    defer alloc.free(positions);
    const base_normals = try tmesh.vertexNormals(alloc, mesh.verts, mesh.tris);
    defer alloc.free(base_normals);
    const normals = try alloc.alloc(f32, positions.len);
    defer alloc.free(normals);
    for (uw.vmapping, 0..) |orig, i| {
        @memcpy(positions[i * 3 ..][0..3], mesh.verts[orig * 3 ..][0..3]);
        @memcpy(normals[i * 3 ..][0..3], base_normals[orig * 3 ..][0..3]);
    }
    if (prog) |p| {
        if (p.boundary()) return error.Cancelled;
        p.emit("trellis-uv", 1, 1);
        p.emit("trellis-bake", 0, 1);
    }

    const tex = opts.texture_size;
    var vol = try tmesh.Volume.init(alloc, vox.coords, vox.attrs, 6, res);
    defer vol.deinit(alloc);
    // The remesh sits a voxel or two off the decoded surface: each texel's point goes back onto the dual-grid
    // mesh before the volume is read, so the trilinear neighbours are the voxels that carry the attributes.
    var texels = try sse.offload(tmesh.texelPositions, .{ alloc, positions, uw.uvs, uw.indices, tex });
    defer texels.deinit(alloc);
    try sse.offload(remesh.snapToSurface, .{ alloc, raw_mesh, texels.pos, texels.mask });
    var atlas = try sse.offload(tmesh.sampleAtlas, .{ alloc, &texels, &vol });
    defer atlas.deinit(alloc);

    const n_texels = @as(usize, tex) * tex;
    const albedo = try alloc.alloc(f32, n_texels * 3);
    defer alloc.free(albedo);
    const mr = try alloc.alloc(f32, n_texels * 3);
    defer alloc.free(mr);
    for (0..n_texels) |i| {
        const a = atlas.attrs[i * 6 ..][0..6];
        @memcpy(albedo[i * 3 ..][0..3], a[0..3]);
        mr[i * 3] = 0;
        mr[i * 3 + 1] = a[4]; // roughness -> G
        mr[i * 3 + 2] = a[3]; // metallic -> B
    }
    for ([_][]f32{ albedo, mr }) |img| try sse.offload(texinpaint.inpaintAtlas, .{ alloc, img, atlas.mask, tex, tex, positions, uw.uvs, uw.indices });
    if (prog) |p| {
        if (p.boundary()) return error.Cancelled;
        p.emit("trellis-bake", 1, 1);
    }

    const albedo_png = try encodePng(alloc, albedo, tex);
    defer alloc.free(albedo_png);
    const mr_png = try encodePng(alloc, mr, tex);
    defer alloc.free(mr_png);
    exportAxes(positions, opts.pixal3d_axes);
    exportAxes(normals, opts.pixal3d_axes);
    const gltf_uvs = try alloc.alloc(f32, uw.uvs.len);
    defer alloc.free(gltf_uvs);
    for (0..uw.uvs.len / 2) |i| {
        gltf_uvs[i * 2] = uw.uvs[i * 2];
        gltf_uvs[i * 2 + 1] = 1 - uw.uvs[i * 2 + 1];
    }
    const tm = glb.TexturedMesh{ .positions = positions, .normals = normals, .uvs = gltf_uvs, .indices = uw.indices, .albedo_png = albedo_png, .mr_png = mr_png, .roughness_factor = opts.roughness_scale, .clearcoat = opts.clearcoat };
    return sse.offload(glb.writeGlbTextured, .{ alloc, &tm });
}


// ── helpers ──────────────────────────────────────────────────────────────────

const StepCtx = struct { prog: ?sse.Progress, stage: []const u8 };

fn onStep(ctx: *const StepCtx, i: u32, n: u32) anyerror!void {
    const p = ctx.prog orelse return;
    if (p.boundary()) return error.Cancelled;
    p.emit(ctx.stage, i, n);
}

fn readFile(io: std.Io, gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(gpa, .limited(1 << 20));
}

/// Row-major f32 values of `a`, valid for `sc`'s lifetime.
fn hostF32(sc: *Scope, a: A) ![]const f32 {
    var c = mlx.mlx_array_new();
    const flat = try sc.res(mlx.mlx_contiguous(&c, try sc.astype(a, .float32), false, sc.s), &c);
    try mlx.check(mlx.mlx_array_eval(flat));
    return (mlx.mlx_array_data_float32(flat) orelse return error.MlxError)[0..mlx.mlx_array_size(flat)];
}

fn normalNoise(sc: *Scope, shape: []const c_int, seed: u64) !A {
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, seed));
    defer _ = mlx.mlx_array_free(key);
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0.0, 1.0, key, sc.s), &o);
}

fn encodeAt(alloc: std.mem.Allocator, enc: *const dino.Encoder, s: S, img: Prepared, size: u32) !A {
    const pix = try dino.preprocess(alloc, img.rgb, img.side, img.side, size);
    defer alloc.free(pix);
    var sc = Scope.init(alloc, s);
    defer sc.deinit();
    const x = try sc.fromF32(pix, &.{ 1, @intCast(size), @intCast(size), 3 });
    const tokens = try enc.forward(&sc, x);
    try mlx.check(mlx.mlx_array_eval(tokens));
    return sc.out(tokens);
}

/// The distinct cells of `grid` (cells per axis) the finer voxels `hr` (`[n, 4]` at `lr_res`) fall in, sorted
/// by (x, y, z): `int((c + 0.5) / lr_res * grid)` in f32, the reference's own arithmetic.
fn quantizeUnique(alloc: std.mem.Allocator, hr: []const i32, lr_res: u32, grid: u32) ![]i32 {
    return quantizeUniqueWith(alloc, hr, lr_res, grid, truncCell);
}

fn truncCell(c: f32, lr_res: f32, grid: f32) i32 {
    return @intFromFloat((c + 0.5) / lr_res * grid);
}

/// `quantizeUnique` with the reference's own cell arithmetic `cell(c, lr_res, grid)`.
pub fn quantizeUniqueWith(alloc: std.mem.Allocator, hr: []const i32, lr_res: u32, grid: u32, comptime cell: fn (f32, f32, f32) i32) ![]i32 {
    const keys = try alloc.alloc(u64, hr.len / 4);
    defer alloc.free(keys);
    const res_f: f32 = @floatFromInt(lr_res);
    const grid_f: f32 = @floatFromInt(grid);
    for (keys, 0..) |*k, i| {
        var q: [3]i32 = undefined;
        for (0..3) |a| q[a] = cell(@floatFromInt(hr[i * 4 + 1 + a]), res_f, grid_f);
        k.* = sp.voxelKey(q[0], q[1], q[2]);
    }
    std.mem.sortUnstable(u64, keys, {}, std.sort.asc(u64));
    var out: std.ArrayList(i32) = .empty;
    errdefer out.deinit(alloc);
    var prev: ?u64 = null;
    const mask: u64 = (1 << sp.AXIS_BITS) - 1;
    for (keys) |k| {
        if (prev != null and prev.? == k) continue;
        prev = k;
        try out.appendSlice(alloc, &.{ 0, @intCast(k >> (2 * sp.AXIS_BITS)), @intCast((k >> sp.AXIS_BITS) & mask), @intCast(k & mask) });
    }
    return out.toOwnedSlice(alloc);
}

pub const Prepared = struct {
    /// Square RGB8, the subject over black.
    rgb: []u8,
    side: u32,
};

/// `preprocess_image` minus the background remover: shrink to at most 1024 px, take the square around the
/// subject's alpha box (alpha above 80%; the whole frame when there is none), multiply the colour by the alpha.
pub fn prepareImage(alloc: std.mem.Allocator, rgba_in: []const u8, w_in: u32, h_in: u32) !Prepared {
    return prepareImageMargin(alloc, rgba_in, w_in, h_in, 1.0);
}

/// `prepareImage` with the box side scaled by `margin` before the crop (Pixal3D's 1.1).
pub fn prepareImageMargin(alloc: std.mem.Allocator, rgba_in: []const u8, w_in: u32, h_in: u32, margin: f64) !Prepared {
    var w = w_in;
    var h = h_in;
    var owned: ?[]u8 = null;
    defer if (owned) |o| alloc.free(o);
    var rgba = rgba_in;
    const big = @max(w, h);
    if (big > MAX_INPUT_SIDE) {
        const scale = @as(f64, MAX_INPUT_SIDE) / @as(f64, @floatFromInt(big));
        const nw: u32 = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(w)) * scale)));
        const nh: u32 = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(h)) * scale)));
        owned = try qvis.resizeInterleavedPil(alloc, rgba, h, w, nh, nw, 4, .lanczos);
        rgba = owned.?;
        w = nw;
        h = nh;
    }
    var x0: u32 = w;
    var y0: u32 = h;
    var x1: u32 = 0;
    var y1: u32 = 0;
    for (0..h) |y| for (0..w) |x| {
        if (rgba[(y * w + x) * 4 + 3] > ALPHA_CUT) {
            x0 = @min(x0, @as(u32, @intCast(x)));
            x1 = @max(x1, @as(u32, @intCast(x)));
            y0 = @min(y0, @as(u32, @intCast(y)));
            y1 = @max(y1, @as(u32, @intCast(y)));
        }
    };
    if (x1 < x0) {
        x0 = 0;
        y0 = 0;
        x1 = w - 1;
        y1 = h - 1;
    }
    const size: i64 = @intFromFloat(@as(f64, @floatFromInt(@max(x1 - x0, y1 - y0))) * margin);
    const half = @divFloor(size, 2);
    const cx2: i64 = @as(i64, x0) + x1; // twice the centre, to stay in integers
    const cy2: i64 = @as(i64, y0) + y1;
    const bx0 = roundHalfEven2(cx2 - 2 * half);
    const by0 = roundHalfEven2(cy2 - 2 * half);
    const side: u32 = @intCast(2 * half);
    if (side == 0) return error.EmptySubject;
    const rgb = try alloc.alloc(u8, @as(usize, side) * side * 3);
    errdefer alloc.free(rgb);
    for (0..side) |y| for (0..side) |x| {
        const sx = bx0 + @as(i64, @intCast(x));
        const sy = by0 + @as(i64, @intCast(y));
        const o = (y * side + x) * 3;
        if (sx < 0 or sy < 0 or sx >= w or sy >= h) {
            @memset(rgb[o..][0..3], 0);
            continue;
        }
        const p = rgba[(@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sx))) * 4 ..][0..4];
        const alpha = @as(f32, @floatFromInt(p[3])) / 255.0;
        for (0..3) |c| rgb[o + c] = @intFromFloat(@as(f32, @floatFromInt(p[c])) / 255.0 * alpha * 255.0);
    };
    return .{ .rgb = rgb, .side = side };
}

/// `round(c2 / 2)` with ties to even, as Python rounds each corner of the crop box.
fn roundHalfEven2(c2: i64) i64 {
    const q = @divFloor(c2, 2);
    if (@mod(c2, 2) == 0) return q;
    return if (@mod(q, 2) == 0) q else q + 1;
}

/// The reference's export axes, rotations both, so the winding stays: `to_glb`'s z up -> y up, (x, y, z) ->
/// (x, z, -y); with `pixal3d`, then inference.py's `rot`, (x, y, z) -> (-x, -z, -y): its view-aligned latents
/// keep y up, so the decoded (x, y, z) ends on (-x, y, -z).
fn exportAxes(v: []f32, pixal3d: bool) void {
    var i: usize = 0;
    while (i < v.len) : (i += 3) {
        const x = v[i];
        const y = v[i + 1];
        const z = v[i + 2];
        v[i..][0..3].* = if (pixal3d) .{ -x, y, -z } else .{ x, z, -y };
    }
}

fn encodePng(alloc: std.mem.Allocator, img: []const f32, tex: u32) ![]u8 {
    const bytes = try alloc.alloc(u8, img.len);
    defer alloc.free(bytes);
    for (img, bytes) |v, *b| b.* = @intFromFloat(@round(std.math.clamp(v, 0.0, 1.0) * 255.0));
    return png.encodeRgb(alloc, bytes, tex, tex);
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "trellis2 pipeline: the 512 texture flow counts only when its weights are on disk" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try tmp.dir.createDirPath(io, "ckpts");
    try testing.expectEqualStrings("", try onDiskOrEmpty(io, gpa, root, "ckpts/tex512"));
    try testing.expectEqualStrings("", try onDiskOrEmpty(io, gpa, root, ""));
    try tmp.dir.writeFile(io, .{ .sub_path = "ckpts/tex512.safetensors", .data = "x" });
    try testing.expectEqualStrings("ckpts/tex512", try onDiskOrEmpty(io, gpa, root, "ckpts/tex512"));
    // A request for 512 is then refused by name instead of failing after the structure and shape stages.
    try testing.expect(!resolutionServed(512, (try onDiskOrEmpty(io, gpa, root, "ckpts/gone")).len > 0));
}

test "trellis2 pipeline: the subject box is the square around alpha above 80%, black where it leaves the frame" {
    const gpa = testing.allocator;
    // 10 x 8: a red band (rows 0..1, all columns) plus a faint pixel (alpha 100) that must not extend the box.
    var img = try gpa.alloc(u8, 10 * 8 * 4);
    defer gpa.free(img);
    @memset(img, 0);
    for (0..2) |y| for (0..10) |x| {
        @memcpy(img[(y * 10 + x) * 4 ..][0..4], &[_]u8{ 255, 0, 0, 255 });
    };
    @memcpy(img[(7 * 10 + 9) * 4 ..][0..4], &[_]u8{ 0, 255, 0, 100 });
    const p = try prepareImage(gpa, img, 10, 8);
    defer gpa.free(p.rgb);
    // size = 9, half = 4, centre (4.5, 0.5): corners round to x 0..8 and y -4..4 -> an 8 x 8 crop.
    try testing.expectEqual(@as(u32, 8), p.side);
    const row = p.side * 3;
    try testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, p.rgb[0..3]); // above the frame
    try testing.expectEqualSlices(u8, &.{ 255, 0, 0 }, p.rgb[4 * row ..][0..3]); // the band's first row
    try testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, p.rgb[6 * row ..][0..3]); // transparent below it
}

test "trellis2 pipeline: a non-finite decoder output is a named error before any float-to-int cast" {
    const gpa = testing.allocator;
    var coords = [_]i32{ 0, 1, 1, 1 };
    var raw = [_]f32{ 0, std.math.nan(f32), 0, 0, 0, 0 };
    var attrs = std.mem.zeroes([6]f32);
    const vox = Voxels{ .coords = &coords, .raw = &raw, .attrs = &attrs, .res = 1024 };
    try testing.expectError(error.NonFiniteDecoderOutput, meshToGlb(gpa, &vox, .{}, null));
}

test "trellis2 pipeline: a seed at the top of u64 wraps into the stage seeds instead of overflowing" {
    try testing.expectEqual(@as(u64, 3), stageSeed(0, 3));
    try testing.expectEqual(@as(u64, 1), stageSeed(std.math.maxInt(u64), 2));
    try testing.expect(stageSeed(std.math.maxInt(u64), 1) != stageSeed(std.math.maxInt(u64), 2));
}

test "trellis2 pipeline: a one-pixel-high strip is a crop or a named refusal, never an underflow" {
    const gpa = testing.allocator;
    const w: u32 = 2 * MAX_INPUT_SIDE;
    const strip = try gpa.alloc(u8, @as(usize, w) * 4);
    defer gpa.free(strip);
    @memset(strip, 255);
    if (prepareImageMargin(gpa, strip, w, 1, 1.2)) |p| gpa.free(p.rgb) else |err| try testing.expect(err == error.EmptySubject);
}

test "trellis2 pipeline: colour is multiplied by alpha with the reference's float32 truncation" {
    const gpa = testing.allocator;
    var img: [4 * 4 * 4]u8 = undefined;
    for (0..16) |i| @memcpy(img[i * 4 ..][0..4], &[_]u8{ 255, 0, 0, 255 });
    img[3] = 205; // alpha of the first pixel, still above the cut
    const p = try prepareImage(gpa, &img, 4, 4);
    defer gpa.free(p.rgb);
    try testing.expectEqual(@as(u32, 2), p.side); // size 3, half 1: the reference's own off-by-one crop
    const want: u8 = @intFromFloat(@as(f32, 255) / 255.0 * (@as(f32, 205) / 255.0) * 255.0);
    try testing.expectEqualSlices(u8, &.{ want, 0, 0, 255, 0, 0 }, p.rgb[0..6]);
}

test "trellis2 pipeline: quantizeUnique lands on the coarser grid, deduplicated and sorted" {
    const gpa = testing.allocator;
    // 512 -> 64 cells: cell = c >> 3.
    const hr = [_]i32{ 0, 15, 0, 0, 0, 8, 0, 0, 0, 9, 7, 7, 0, 0, 0, 0 };
    const q = try quantizeUnique(gpa, &hr, 512, 64);
    defer gpa.free(q);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0, 0, 1, 0, 0 }, q);
}

test "trellis2 pipeline: ties in the crop box round to even like Python" {
    try testing.expectEqual(@as(i64, 4), roundHalfEven2(9)); // 4.5
    try testing.expectEqual(@as(i64, 6), roundHalfEven2(11)); // 5.5
    try testing.expectEqual(@as(i64, 3), roundHalfEven2(6));
}

test "trellis2 pipeline: the export axes are a rotation (x kept, y <- z, z <- -y)" {
    var v = [_]f32{ 1, 2, 3 };
    exportAxes(&v, false);
    try testing.expectEqualSlices(f32, &.{ 1, 3, -2 }, &v);
}

test "trellis2 pipeline: Pixal3D's export lands the decoded (x, y, z) on (-x, y, -z), its lattice y up" {
    var v = [_]f32{ 1, 2, 3 };
    exportAxes(&v, true);
    try testing.expectEqualSlices(f32, &.{ -1, 2, -3 }, &v);
}

test "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE, TRELLIS2_OUT; TRELLIS2_LOAD replays dumped voxels)" {
    const gpa = testing.allocator;
    const out_path = std.mem.span(std.c.getenv("TRELLIS2_OUT") orelse return error.SkipZigTest);
    const io = std.Io.Threaded.global_single_threaded.io();
    var opts = Opts{};
    if (std.c.getenv("TRELLIS2_RES")) |v| opts.resolution = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("TRELLIS2_STEPS")) |v| opts.steps = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("TRELLIS2_TEX")) |v| opts.texture_size = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("TRELLIS2_FACES")) |v| opts.decimation_target = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    var clock = TestClock{ .sw = io_util.Stopwatch.init(io) };
    const prog = sse.Progress{ .ctx = &clock, .cb = TestClock.tick };

    var vox: Voxels = undefined;
    if (std.c.getenv("TRELLIS2_LOAD")) |path| {
        vox = try loadVoxels(io, gpa, std.mem.span(path));
    } else {
        const pack = std.mem.span(std.c.getenv("TRELLIS2_PACK") orelse return error.SkipZigTest);
        const image = std.mem.span(std.c.getenv("TRELLIS2_IMAGE") orelse return error.SkipZigTest);
        const encoded = try readFileAny(io, gpa, image);
        defer gpa.free(encoded);
        const stb = @import("stb");
        var w: c_int = 0;
        var h: c_int = 0;
        var ch: c_int = 0;
        const px = stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &ch, 4) orelse return error.BadImage;
        defer stb.stbi_image_free(px);
        const s = mlx.mlx_default_gpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var eng = try Engine.loadOn(io, gpa, pack, s);
        defer eng.deinit();
        _ = mlx.mlx_reset_peak_memory();
        vox = try eng.generateVoxels(gpa, px[0..@intCast(w * h * 4)], @intCast(w), @intCast(h), opts, prog);
        var peak: usize = 0;
        _ = mlx.mlx_get_peak_memory(&peak);
        std.debug.print("  mlx peak {d} MiB, {d} voxels\n", .{ peak >> 20, vox.coords.len / 4 });
        if (std.c.getenv("TRELLIS2_DUMP")) |d| try saveVoxels(io, std.mem.span(d), &vox);
    }
    defer vox.deinit(gpa);
    const bytes = try meshToGlb(gpa, &vox, opts, prog);
    defer gpa.free(bytes);
    const f = try std.Io.Dir.createFileAbsolute(io, out_path, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, bytes);
    std.debug.print("wrote {s} ({d} bytes)\n", .{ out_path, bytes.len });
}

fn saveVoxels(io: std.Io, path: []const u8, v: *const Voxels) !void {
    const f = try std.Io.Dir.createFileAbsolute(io, path, .{});
    defer f.close(io);
    const head = [3]u32{ 0x58544c32, v.res, @intCast(v.coords.len / 4) };
    try f.writeStreamingAll(io, std.mem.sliceAsBytes(&head));
    try f.writeStreamingAll(io, std.mem.sliceAsBytes(v.coords));
    try f.writeStreamingAll(io, std.mem.sliceAsBytes(v.raw));
    try f.writeStreamingAll(io, std.mem.sliceAsBytes(v.attrs));
}

fn loadVoxels(io: std.Io, gpa: std.mem.Allocator, path: []const u8) !Voxels {
    const bytes = try readFileAny(io, gpa, path);
    defer gpa.free(bytes);
    const head = std.mem.bytesAsSlice(u32, bytes[0..12]);
    if (head[0] != 0x58544c32) return error.BadVoxelDump;
    const m: usize = head[2];
    var out = Voxels{ .coords = try gpa.alloc(i32, m * 4), .raw = try gpa.alloc(f32, m * 7), .attrs = try gpa.alloc(f32, m * 6), .res = head[1] };
    errdefer out.deinit(gpa);
    var off: usize = 12;
    inline for (.{ &out.coords, &out.raw, &out.attrs }) |field| {
        const n = std.mem.sliceAsBytes(field.*).len;
        @memcpy(std.mem.sliceAsBytes(field.*), bytes[off..][0..n]);
        off += n;
    }
    return out;
}

/// Prints each stage's first and last tick with the elapsed time.
pub const TestClock = struct {
    sw: io_util.Stopwatch,
    last: [48]u8 = undefined,
    last_len: usize = 0,

    pub fn tick(ctx: *anyopaque, stage: []const u8, step: u32, total: u32) void {
        const self: *TestClock = @ptrCast(@alignCast(ctx));
        const same = std.mem.eql(u8, self.last[0..self.last_len], stage);
        if (!same or step == total) std.debug.print("  [{d:>7.1}s] {s} {d}/{d}\n", .{ @as(f64, @floatFromInt(self.sw.read())) / 1e9, stage, step, total });
        const l = @min(stage.len, self.last.len);
        @memcpy(self.last[0..l], stage[0..l]);
        self.last_len = l;
    }
};

pub fn readFileAny(io: std.Io, gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(gpa, .limited(1 << 30));
}
