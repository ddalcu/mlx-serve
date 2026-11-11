//! MiniMax-H3-Image's generation, staged like H3's: the H3 text encoder (with its vision tower when references
//! ride along) is loaded, run and freed, then the DiT samples its six PDD steps, then the 2D image VAE decodes.
//! The sequence is H3's REF2VA layout at one latent frame with no audio: `[text | each reference | target]`
//! (diffusers `MiniMaxH3ImageModularPipeline`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
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
            const rows_in = if (cond_rows) |c| try concat2(c, x, s) else x;
            defer if (cond_rows != null) {
                _ = mlx.mlx_array_free(rows_in);
            };
            var out = try model.forward(&layout, &plan, refined, rows_in, .{ .ctx = null }, rope, heads, 1.0, null, true, s);
            defer out.deinit();
            // `forward` hands back the denoise direction (-v): x += -v * (sigma_next - sigma), the velocity form.
            const nx = try axpy(x, out.video, @floatCast(sigmas[i + 1] - sigmas[i]), s);
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
    var rows: std.ArrayList(A) = .empty;
    defer {
        for (rows.items) |r| _ = mlx.mlx_array_free(r);
        rows.deinit(a);
    }
    for (req.refs, blocks, 0..) |r, *b, i| {
        const full = try enc.imageMoments(r.pixels, a, s);
        defer _ = mlx.mlx_array_free(full);
        // Cropped top-left to whole 2x2 patches, as the release crops its sample.
        const fs = mlx.getShape(full);
        const lh = fs[3] - @mod(fs[3], 2);
        const lw = fs[4] - @mod(fs[4], 2);
        var m = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(m);
        try mlx.check(mlx.mlx_slice(&m, full, &[_]c_int{ 0, 0, 0, 0, 0 }, 5, &[_]c_int{ fs[0], fs[1], fs[2], lh, lw }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
        b.* = .{ .kind = .image, .latent_h = @intCast(lh), .latent_w = @intCast(lw) };
        const lat = try samplePosterior(m, req.seed +% 2 +% i, s);
        defer _ = mlx.mlx_array_free(lat);
        const norm = try enc.normalize(lat, s);
        defer _ = mlx.mlx_array_free(norm);
        try rows.append(a, try h3.patchifyVideo(norm, s));
    }
    const cat = if (rows.items.len == 1) try copy(rows.items[0], s) else try concatN(rows.items, s);
    defer _ = mlx.mlx_array_free(cat);
    // x_t = t * x0 + (1 - t) * noise at t = 0.999, constant for the whole run.
    const noise = try h3.randomNormal(mlx.getShape(cat), req.seed +% 1, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(noise);
    const kept = try scale(cat, @floatCast(REFERENCE_T), s);
    defer _ = mlx.mlx_array_free(kept);
    const out = try axpy(kept, noise, @floatCast(1.0 - REFERENCE_T), s);
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// `mean + exp(0.5 * clamp(logvar, -30, 20)) * eps` over [1, 48, ...] moments, rounded through f16 like the
/// release, returned f32 [1, 24, ...].
fn samplePosterior(moments: A, seed: u64, s: S) !A {
    var halves = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(halves);
    try mlx.check(mlx.mlx_split(&halves, moments, 2, 1, s));
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    var logvar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logvar);
    try mlx.check(mlx.mlx_vector_array_get(&mean, halves, 0));
    try mlx.check(mlx.mlx_vector_array_get(&logvar, halves, 1));
    const lo = mlx.mlx_array_new_float(-30.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(20.0);
    defer _ = mlx.mlx_array_free(hi);
    var lv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lv);
    try mlx.check(mlx.mlx_clip(&lv, logvar, lo, hi, s));
    const half_lv = try scale(lv, 0.5, s);
    defer _ = mlx.mlx_array_free(half_lv);
    var std_ = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(std_);
    try mlx.check(mlx.mlx_exp(&std_, half_lv, s));
    const eps = try h3.randomNormal(mlx.getShape(mean), seed, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(eps);
    var noise = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(noise);
    try mlx.check(mlx.mlx_multiply(&noise, std_, eps, s));
    var z = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(z);
    try mlx.check(mlx.mlx_add(&z, mean, noise, s));
    var z16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(z16);
    try mlx.check(mlx.mlx_astype(&z16, z, mlx.mlx_dtype.float16, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, z16, mlx.mlx_dtype.float32, s));
    return out;
}

// ── tiny op helpers (owned results) ──

fn scale(x: A, k: f32, s: S) !A {
    const kk = mlx.mlx_array_new_float(k);
    defer _ = mlx.mlx_array_free(kk);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, x, kk, s));
    return o;
}

/// `x + y * k`.
fn axpy(x: A, y: A, k: f32, s: S) !A {
    const yk = try scale(y, k, s);
    defer _ = mlx.mlx_array_free(yk);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, x, yk, s));
    return o;
}

fn copy(x: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}

fn concat2(x: A, y: A, s: S) !A {
    return concatN(&.{ x, y }, s);
}

fn concatN(xs: []const A, s: S) !A {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (xs) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 0, s));
    return o;
}

// ── output resize ──

/// [1, 3, H, W] f32 -> [1, 3, oh, ow] by area averaging (each output pixel the coverage-weighted mean of the source
/// pixels under it): the canvas is only ever scaled DOWN to the requested size. Caller frees.
pub fn resizeArea(a: std.mem.Allocator, img: A, oh: u32, ow: u32, s: S) !A {
    const c = try copy(img, s);
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_array_eval(c));
    const shp = mlx.getShape(c);
    const ih: usize = @intCast(shp[2]);
    const iw: usize = @intCast(shp[3]);
    const src = (mlx.mlx_array_data_float32(c) orelse return error.NoData)[0 .. 3 * ih * iw];
    const tmp = try a.alloc(f32, 3 * ih * ow);
    defer a.free(tmp);
    const out = try a.alloc(f32, 3 * @as(usize, oh) * ow);
    defer a.free(out);
    for (0..3 * ih) |row| areaLine(src[row * iw ..][0..iw], 1, tmp[row * ow ..][0..ow], 1);
    for (0..3) |ch| for (0..ow) |x| areaLine(tmp[ch * ih * ow + x ..][0 .. (ih - 1) * ow + 1], ow, out[ch * oh * ow + x ..][0 .. (oh - 1) * ow + 1], ow);
    const arr = mlx.mlx_array_new_data(out.ptr, &[_]c_int{ 1, 3, @intCast(oh), @intCast(ow) }, 4, .float32);
    return arr;
}

/// One strided line resampled by area: `src` has (len-1)/ss+1 samples, `dst` (len-1)/ds+1.
fn areaLine(src: []const f32, ss: usize, dst: []f32, ds: usize) void {
    const n_in = (src.len - 1) / ss + 1;
    const n_out = (dst.len - 1) / ds + 1;
    const k = @as(f64, @floatFromInt(n_in)) / @as(f64, @floatFromInt(n_out));
    for (0..n_out) |o| {
        const lo = @as(f64, @floatFromInt(o)) * k;
        const hi = lo + k;
        var acc: f64 = 0;
        var i: usize = @intFromFloat(@floor(lo));
        while (i < n_in and @as(f64, @floatFromInt(i)) < hi) : (i += 1) {
            const fi: f64 = @floatFromInt(i);
            const cover = @min(hi, fi + 1) - @max(lo, fi);
            if (cover > 0) acc += cover * src[i * ss];
        }
        dst[o * ds] = @floatCast(acc / k);
    }
}

test "minimax_h3_image: area resize averages the pixels under each output pixel" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // 2x4 per channel -> 1x2: each output is the mean of a 2x2 block.
    var px: [24]f32 = undefined;
    for (&px, 0..) |*v, i| v.* = @floatFromInt(i);
    const img = mlx.mlx_array_new_data(&px, &[_]c_int{ 1, 3, 2, 4 }, 4, .float32);
    defer _ = mlx.mlx_array_free(img);
    const r = try resizeArea(a, img, 1, 2, s);
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_array_eval(r));
    const d = mlx.mlx_array_data_float32(r).?;
    // channel 0: rows [0 1 2 3], [4 5 6 7] -> (0+1+4+5)/4, (2+3+6+7)/4
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), d[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), d[1], 1e-6);
    // channel 2 starts at 16.
    try std.testing.expectApproxEqAbs(@as(f32, 18.5), d[4], 1e-6);
    // A non-integer ratio still preserves a flat field.
    var flat: [3 * 3 * 5]f32 = undefined;
    @memset(&flat, 0.25);
    const f = mlx.mlx_array_new_data(&flat, &[_]c_int{ 1, 3, 3, 5 }, 4, .float32);
    defer _ = mlx.mlx_array_free(f);
    const rf = try resizeArea(a, f, 2, 3, s);
    defer _ = mlx.mlx_array_free(rf);
    try mlx.check(mlx.mlx_array_eval(rf));
    for (mlx.mlx_array_data_float32(rf).?[0..18]) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.25), v, 1e-6);
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
    const unit = mlx.mlx_array_new_data(&px, &[_]c_int{ 1, 3, 1, H, W }, 5, .float32);
    defer _ = mlx.mlx_array_free(unit);
    const pm1 = try scale(unit, 2, s);
    defer _ = mlx.mlx_array_free(pm1);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var pixels = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pixels);
    try mlx.check(mlx.mlx_subtract(&pixels, pm1, one, s));

    const vae_path = try std.fs.path.join(a, &.{ dir, "image_vae.safetensors" });
    defer a.free(vae_path);
    var vw = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer vw.deinit();
    var enc = try vae.Encoder.load(a, &vw, s);
    defer enc.deinit();
    const m = try enc.imageMoments(pixels, a, s);
    defer _ = mlx.mlx_array_free(m);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_slice(&mean, m, &[_]c_int{ 0, 0, 0, 0, 0 }, 5, &[_]c_int{ 1, 24, 1, H / 16, W / 16 }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
    const z = try enc.normalize(mean, s);
    defer _ = mlx.mlx_array_free(z);
    var dec = try vae.Decoder.load(a, &vw, g.h3.vae, mlx.mlx_dtype.float32, s);
    defer dec.deinit();
    const rgb = try vae.decodeImage(&dec, z, a, s);
    defer _ = mlx.mlx_array_free(rgb);
    const c = try copy(rgb, s);
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_array_eval(c));
    const out = mlx.mlx_array_data_float32(c).?[0 .. 3 * H * W];
    var err: f64 = 0;
    for (out, px) |o, p| err += @abs(o - p);
    err /= px.len;
    try std.testing.expect(err < 0.05);
}

