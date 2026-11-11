//! MiniMax-H3-Image's generation, staged like H3's: the H3 text encoder (with its vision tower when references
//! ride along) is loaded, run and freed, then the DiT samples its six PDD steps, then the 2D image VAE decodes.
//! The sequence is H3's REF2VA layout at one latent frame with no audio: `[text | each reference | target]`
//! (diffusers `MiniMaxH3ImageModularPipeline`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const log = @import("../../sdk/util/log.zig");
const crt = @import("../../sdk/util/crt.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const gen_sse = @import("../../sdk/media/progress.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const mage_flow = @import("../mage_flow/mage_flow.zig");
const h3 = @import("../minimax_h3/h3.zig");
const vae = @import("../minimax_h3/vae.zig");
const config = @import("config.zig");

const S = mlx.mlx_stream;
const A = mlx.mlx_array;

/// Tokens the prompt (and each reference label) keeps.
pub const MAX_PROMPT_TOKENS = 4096;
/// Reference rows are held at this clock for every step, noised just short of clean.
pub const REFERENCE_T: f64 = h3.VISUAL_COND_TIMESTEP;

/// One edit reference, decoded by the caller at the sizes `config.boundShortEdge` chose.
pub const Ref = struct {
    /// [1, 3, 1, H, W] in [-1, 1] at the VAE size, cropped top-left to whole VAE cells (multiples of 16).
    pixels: A,
    /// [1, 3, 1, h, w] in [-1, 1] at the vision tower's canvas (`h3v.fitCanvas`).
    vision: A,
};

pub const Request = struct {
    prompt: []const u8,
    /// The canvas (multiples of 32).
    width: u32,
    height: u32,
    seed: u64,
    refs: []const Ref = &.{},
};

pub const Paths = struct {
    dir: []const u8,

    fn file(self: Paths, a: std.mem.Allocator, name: []const u8) ![]u8 {
        return std.fs.path.join(a, &.{ self.dir, name });
    }
};

/// RGB [1, 3, height, width] in [0, 1], f32. Caller frees.
pub fn generate(a: std.mem.Allocator, io: std.Io, paths: Paths, g: *const config.Geometry, req: Request, progress: ?gen_sse.Progress, s: S) !A {
    const dt = mlx.mlx_dtype.bfloat16;
    const lat_h = req.height / vae.VAE_RATIO;
    const lat_w = req.width / vae.VAE_RATIO;
    const pdd = try h3.Pdd.fromSigmas(g.grid());
    const steps = pdd.steps();
    log.info("[minimax-h3-image] {d}x{d}, {d} reference(s), {d} steps\n", .{ req.width, req.height, req.refs.len, steps });
    var clock: Lap = .{ .t = crt.monotonicMs() };

    // The DiT runs every forward at prefill width, where the dequant+GEMM route wins (as H3's video run).
    const dq_prev = mage_flow.mf_dq_gemm_override;
    mage_flow.mf_dq_gemm_override = @as(?usize, 2048);
    defer mage_flow.mf_dq_gemm_override = dq_prev;

    // ── 1. references -> condition rows (the VAE encoder, freed before the text encoder loads) ──
    var cond_rows: ?A = null;
    defer if (cond_rows) |c| {
        _ = mlx.mlx_array_free(c);
    };
    const blocks = try a.alloc(h3.RefBlock, req.refs.len);
    defer a.free(blocks);
    if (req.refs.len > 0) {
        if (progress) |p| p.emit("Encoding references", 0, steps);
        cond_rows = try encodeReferences(a, paths, req, blocks, s);
        log.info("[minimax-h3-image] {d} reference(s) -> {d} condition rows ({d} ms)\n", .{ req.refs.len, mlx.getShape(cond_rows.?)[0], clock.lap() });
    }

    // ── 2. the presentation: `Picture i: ` + a vision block per reference, then the prompt, raw ids ──
    if (progress) |p| p.emit("Encoding prompt", 0, steps);
    const tok_dir = try a.dupe(u8, paths.dir);
    defer a.free(tok_dir);
    var tok = try tok_mod.loadTokenizerAny(io, a, tok_dir);
    defer tok.deinit();
    var owned: std.ArrayList([]i32) = .empty;
    var pairs: std.ArrayList(A) = .empty;
    var present: std.ArrayList(h3.PresentItem) = .empty;
    defer {
        for (owned.items) |x| a.free(x);
        owned.deinit(a);
        for (pairs.items) |x| _ = mlx.mlx_array_free(x);
        pairs.deinit(a);
        present.deinit(a);
    }
    var label: [32]u8 = undefined;
    for (req.refs, 0..) |r, i| {
        // The label carries no angle brackets, unlike the `<Picture i>` the prompt refers to it by.
        try owned.append(a, try encodeIds(a, &tok, try std.fmt.bufPrint(&label, "Picture {d}: ", .{i + 1})));
        try present.append(a, .{ .text = owned.items[owned.items.len - 1] });
        const shp = mlx.getShape(r.vision);
        const vh: u32 = @intCast(shp[shp.len - 2]);
        const vw: u32 = @intCast(shp[shp.len - 1]);
        try pairs.append(a, try h3.stillAsFramePair(r.vision, s));
        try present.append(a, .{ .vision = .{ .frames = pairs.items[pairs.items.len - 1], .grid = h3.h3v.gridFor(vh, vw) } });
    }
    try owned.append(a, try encodeIds(a, &tok, req.prompt));
    try present.append(a, .{ .text = owned.items[owned.items.len - 1] });

    // ── 3. the text encoder, freed before the DiT loads ──
    var enc = blk: {
        const te_path = try paths.file(a, "text_encoder.safetensors");
        defer a.free(te_path);
        var tw = try model_mod.loadWeightsSingleFile(a, te_path);
        defer tw.deinit();
        var te = try h3.TextEncoder.load(a, &tw, g.h3.te, dt, s);
        defer te.deinit();
        if (req.refs.len > 0) try te.loadVision(&tw, s);
        var e = try te.encodeItems(a, present.items, s);
        errdefer e.deinit();
        try mlx.check(mlx.mlx_array_eval(e.hidden));
        break :blk e;
    };
    defer enc.deinit();
    _ = mlx.mlx_clear_cache();
    const text_len: u32 = @intCast(mlx.getShape(enc.hidden)[0]);
    log.info("[minimax-h3-image] prompt encoded: {d} rows, encoder released ({d} ms)\n", .{ text_len, clock.lap() });

    // ── 4. the DiT: six PDD steps, one head each ──
    var layout = try h3.PackedLayout.initFull(a, text_len, 1, lat_h, lat_w, 0, &.{}, 1, blocks);
    defer layout.deinit();
    layout.text_tags = enc.tags;
    const sigmas = try pdd.sigmas(a, 1.0);
    defer a.free(sigmas);
    const clock_cfg: h3.StepClock = .{ .visual = REFERENCE_T, .max_t = pdd.max_t };

    const n_rows: c_int = @intCast((lat_h / h3.PATCH_H) * (lat_w / h3.PATCH_W));
    var x = try h3.randomNormal(&[_]c_int{ n_rows, @intCast(g.h3.dit.videoPatchDim()) }, req.seed, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(x);
    {
        const dit_path = try paths.file(a, "transformer.safetensors");
        defer a.free(dit_path);
        var model = try h3.loadDit(a, dit_path, g.h3.dit, dt, s);
        defer model.deinit();
        if (model.hasAudio()) return error.NotAnImageDit;
        try h3.checkPddHeads(@intCast(mlx.getShape(model.video_out_w)[0]), 0, model.cfg, &pdd);
        const ts = try h3.collectScheduleTs(a, &layout, sigmas[0..steps], 1.0, 1.0, clock_cfg);
        defer a.free(ts);
        try model.precomputeAdaln(ts, s);
        try model.evalWeights();
        const refined = try model.refineText(enc.hidden, s);
        defer _ = mlx.mlx_array_free(refined);
        var rope = try h3.buildRope(a, &layout, model.inv_freq, dt, s);
        defer rope.deinit();
        try mlx.check(mlx.mlx_array_eval(refined));
        log.info("[minimax-h3-image] dit loaded, {d} rows ({d} ms)\n", .{ layout.seq_len, clock.lap() });

        for (0..steps) |i| {
            var plan = try h3.buildTimestepPlanGlobal(a, &layout, sigmas[i], 1.0, 1.0, clock_cfg, model.adaln_tables.?.ts);
            defer plan.deinit();
            var heads = try model.pddHeads(&pdd, @intCast(i), 1.0, 1.0, s);
            defer heads.deinit();
            var sg = ops.Graph.init(a, s);
            defer sg.deinit();
            const rows_in = if (cond_rows) |c| try sg.concat(&.{ c, x }, 0) else x;
            var out = try model.forward(&layout, &plan, refined, rows_in, .{ .ctx = null }, rope, heads, 1.0, null, true, s);
            defer out.deinit();
            // `forward` hands back the denoise direction (-v): x += -v * (sigma_next - sigma), the velocity form.
            const nx = sg.escape(try sg.add(x, try sg.mul(out.video, try sg.scalar(@floatCast(sigmas[i + 1] - sigmas[i]), .float32))));
            _ = mlx.mlx_array_free(x);
            x = nx;
            try mlx.check(mlx.mlx_array_eval(x));
            _ = mlx.mlx_clear_cache();
            if (progress) |p| {
                if (p.boundary()) return error.Cancelled;
                p.emit("Generating", @intCast(i + 1), steps);
            }
            log.info("[minimax-h3-image] step {d}/{d} sigma {d:.4} ({d} ms)\n", .{ i + 1, steps, sigmas[i], clock.lap() });
        }
    }
    _ = mlx.mlx_clear_cache();

    // ── 5. the image VAE ──
    if (progress) |p| p.emit("Decoding image", steps, steps);
    const z = try h3.unpatchifyVideoRows(x, 1, lat_h, lat_w, s);
    defer _ = mlx.mlx_array_free(z);
    const vae_path = try paths.file(a, "image_vae.safetensors");
    defer a.free(vae_path);
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    // The released VAE is f32 and its decoder normalizes in f32: it is small, so it runs as shipped.
    var dec = try vae.Decoder.load(a, &vw, g.h3.vae, mlx.mlx_dtype.float32, s);
    defer dec.deinit();
    const rgb = try vae.decodeImage(&dec, z, a, s);
    errdefer _ = mlx.mlx_array_free(rgb);
    try mlx.check(mlx.mlx_array_eval(rgb));
    _ = mlx.mlx_clear_cache();
    log.info("[minimax-h3-image] decoded ({d} ms)\n", .{clock.lap()});
    return rgb;
}

/// Milliseconds since the previous lap, for the stage log.
const Lap = struct {
    t: i64,

    fn lap(self: *Lap) i64 {
        const now = crt.monotonicMs();
        defer self.t = now;
        return now - self.t;
    }
};

/// Raw ids, no special tokens and no chat template, kept to `MAX_PROMPT_TOKENS`. Caller owns.
fn encodeIds(a: std.mem.Allocator, tok: *tok_mod.Tokenizer, text: []const u8) ![]i32 {
    const u = try tok.encode(a, text);
    defer a.free(u);
    const n = @min(u.len, MAX_PROMPT_TOKENS);
    const out = try a.alloc(i32, n);
    for (u[0..n], out) |v, *o| o.* = @intCast(v);
    return out;
}

/// Each reference's posterior SAMPLED (the released recipe), rounded to f16, normalized, patchified and noised to
/// `REFERENCE_T`, in reference order: the rows the layout reserved ahead of the target. Fills `blocks`.
fn encodeReferences(a: std.mem.Allocator, paths: Paths, req: Request, blocks: []h3.RefBlock, s: S) !A {
    const vae_path = try paths.file(a, "image_vae.safetensors");
    defer a.free(vae_path);
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    var enc = try vae.Encoder.load(a, &vw, s);
    defer enc.deinit();
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var rows: [config.edit_ref_cap]A = undefined;
    for (req.refs, blocks, 0..) |r, *b, i| {
        // A scope per reference: its moments and sample are freed before the next one encodes.
        var rg = g.scope();
        defer rg.deinit();
        const full = try rg.adopt(try enc.imageMoments(r.pixels, a, s));
        // Cropped top-left to whole 2x2 patches, as the release crops its sample.
        const fs = ops.shape(full);
        const lh = fs[3] - @mod(fs[3], 2);
        const lw = fs[4] - @mod(fs[4], 2);
        const m = try rg.slice(full, &.{ 0, 0, 0, 0, 0 }, &.{ fs[0], fs[1], fs[2], lh, lw });
        b.* = .{ .kind = .image, .latent_h = @intCast(lh), .latent_w = @intCast(lw) };
        const lat = try samplePosterior(&rg, m, req.seed +% 2 +% i);
        const norm = try rg.adopt(try enc.normalize(lat, s));
        rows[i] = try g.keep(try rg.adopt(try h3.patchifyVideo(norm, s)));
    }
    const cat = try g.concat(rows[0..req.refs.len], 0);
    // x_t = t * x0 + (1 - t) * noise at t = 0.999, constant for the whole run.
    const noise = try g.adopt(try h3.randomNormal(ops.shape(cat), req.seed +% 1, mlx.mlx_dtype.float32, s));
    const out = try g.add(try g.mul(cat, try g.scalar(@floatCast(REFERENCE_T), .float32)), try g.mul(noise, try g.scalar(@floatCast(1.0 - REFERENCE_T), .float32)));
    try ops.eval(out);
    return g.escape(out);
}

/// `mean + exp(0.5 * clamp(logvar, -30, 20)) * eps` over [1, 48, ...] moments, rounded through f16 like the
/// release: f32 [1, 24, ...], owned by `g`.
fn samplePosterior(g: *ops.Graph, moments: A, seed: u64) !A {
    var sg = g.scope();
    defer sg.deinit();
    const half = @divExact(ops.shape(moments)[1], 2);
    const mean = try sg.sliceAxis(moments, 1, 0, half);
    const logvar = try sg.sliceAxis(moments, 1, half, 2 * half);
    const lv = try sg.call(mlx.mlx_clip, .{ logvar, try sg.scalar(-30, .float32), try sg.scalar(20, .float32) });
    const sd = try sg.exp(try sg.mul(lv, try sg.scalar(0.5, .float32)));
    const eps = try sg.adopt(try h3.randomNormal(ops.shape(mean), seed, mlx.mlx_dtype.float32, sg.s));
    const z = try sg.add(mean, try sg.mul(sd, eps));
    return g.keep(try sg.astype(try sg.astype(z, .float16), .float32));
}

test "minimax_h3_image: the image VAE reconstructs a smooth picture through its posterior mean (MINIMAX_H3_IMAGE_MODEL)" {
    const dir = std.mem.span(std.c.getenv("MINIMAX_H3_IMAGE_MODEL") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cfg_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(cfg_path);
    const cfg_json = try std.Io.Dir.cwd().readFileAlloc(std.Io.Threaded.global_single_threaded.io(), cfg_path, a, .limited(1 << 20));
    defer a.free(cfg_json);
    const g = try config.geometry(a, cfg_json);
    const H = 256;
    const W = 256;
    var px: [3 * H * W]f32 = undefined;
    for (0..H) |y| for (0..W) |x| {
        const fy: f32 = @floatFromInt(y);
        const fx: f32 = @floatFromInt(x);
        const disc: f32 = if ((fy - 128) * (fy - 128) + (fx - 100) * (fx - 100) < 60 * 60) 1 else 0;
        px[0 * H * W + y * W + x] = 0.8 * disc + 0.1;
        px[1 * H * W + y * W + x] = fy / H;
        px[2 * H * W + y * W + x] = fx / W;
    };
    var og = ops.Graph.init(a, s);
    defer og.deinit();
    const unit = try og.fromSlice(f32, &px, &.{ 1, 3, 1, H, W });
    const pixels = try og.addScalar(try og.mulScalar(unit, 2), -1);

    const vae_path = try std.fs.path.join(a, &.{ dir, "image_vae.safetensors" });
    defer a.free(vae_path);
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    var enc = try vae.Encoder.load(a, &vw, s);
    defer enc.deinit();
    const m = try og.adopt(try enc.imageMoments(pixels, a, s));
    const z = try og.adopt(try enc.normalize(try og.sliceAxis(m, 1, 0, 24), s));
    var dec = try vae.Decoder.load(a, &vw, g.h3.vae, mlx.mlx_dtype.float32, s);
    defer dec.deinit();
    const c = try og.contiguous(try og.adopt(try vae.decodeImage(&dec, z, a, s)));
    try ops.eval(c);
    const out = mlx.mlx_array_data_float32(c).?[0 .. 3 * H * W];
    var err: f64 = 0;
    for (out, px) |o, p| err += @abs(o - p);
    err /= px.len;
    try std.testing.expect(err < 0.05);
}

