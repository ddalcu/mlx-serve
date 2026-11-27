//! Ming-Image's pipeline (the release's `ImageGenerationPipeline`): the prompt's two caption streams (`llm.zig`),
//! flow-match Euler over the DiT (`dit.zig`) on the release's schedule, the Qwen-Image VAE decoding four channels
//! (RGBA). Everything runs in bf16 but the latents and the scheduler step (f32), as the release does.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const log = @import("../../sdk/util/log.zig");
const sse = @import("../../sdk/media/progress.zig");
const flow_match = @import("../../sdk/blocks/schedulers/flow_match.zig");
const qwen_vae = @import("../../sdk/blocks/vae/qwen_image_vae.zig");
const llm = @import("llm.zig");
const dit_mod = @import("dit.zig");
const config = @import("config.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const S = mlx.mlx_stream;

pub const VAE_DOWNSAMPLE = config.VAE_DOWNSAMPLE;
const TRAIN_STEPS: f32 = 1000;

/// The release's schedule, `n` sigmas then the terminal 0: the scheduler's grid from 1 to `sigma_min`, which the
/// pipeline overrides to 0 (`linspace(1, 0, n)`), shifted by the checkpoint's static `shift` (the pipeline's
/// `use_dynamic_shifting = True` writes a FrozenDict item the scheduler never reads). The last sigma is 0, so the
/// release's final step is a zero-length one that `stepCount` leaves out.
pub fn scheduleSigmas(out: []f32, n: usize, shift: f64) void {
    std.debug.assert(out.len == n + 1);
    for (0..n) |i| {
        const lin: f64 = if (n == 1) 1.0 else 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
        out[i] = @floatCast(flow_match.timeShift(f64, shift, lin));
    }
    out[n] = 0;
}

/// The model forwards `n` steps run: one per non-zero sigma interval.
pub fn stepCount(n: u32) u32 {
    return if (n <= 1) n else n - 1;
}

/// The release's model time for sigma `σ`: the f32 chain `(1000 − σ·1000)/1000·t_scale`.
pub fn modelTime(sigma: f32, t_scale: f32) f32 {
    const ts = sigma * TRAIN_STEPS;
    return (TRAIN_STEPS - ts) / TRAIN_STEPS * t_scale;
}

fn readNum(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, rel: []const u8, key: []const u8, default: f64) !f64 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ model_dir, rel });
    defer a.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch return default;
    defer a.free(bytes);
    var p = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer p.deinit();
    return switch (p.value.object.get(key) orelse return default) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => default,
    };
}

fn stageMs(io: std.Io, t0: *std.Io.Timestamp) u64 {
    const ns = t0.untilNow(io, .awake).nanoseconds;
    t0.* = std.Io.Timestamp.now(io, .awake);
    return @intCast(@divTrunc(ns, 1_000_000));
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,
    lcfg: llm.Config,
    dcfg: dit_mod.Config,
    shift: f64,
    enc: llm.Encoder,
    dit: dit_mod.Dit,
    vae: qwen_vae.Decoder,

    /// Every array of the engine lives and runs on `s`.
    pub fn loadOn(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, s: S) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        self.allocator = a;
        self.io = io;
        self.s = s;
        self.lcfg = try llm.Config.parse(io, a, model_dir);
        self.dcfg = try dit_mod.Config.parse(io, a, model_dir);
        if (self.lcfg.cap_dim != self.dcfg.cap_dim or self.lcfg.inner_dim != self.dcfg.dim) return error.MingImageBadConfig;
        self.shift = try readNum(io, a, model_dir, "scheduler/scheduler_config.json", "shift", 1.0);
        const factor = try readNum(io, a, model_dir, "vae/config.json", "scaling_factor", 1.0);
        const shift_f = try readNum(io, a, model_dir, "vae/config.json", "shift_factor", 0.0);
        if (try readNum(io, a, model_dir, "vae/config.json", "z_dim", 0) != @as(f64, @floatFromInt(self.dcfg.in_ch))) return error.MingImageBadConfig;
        self.dit = try dit_mod.Dit.load(io, a, s, model_dir, self.dcfg);
        errdefer self.dit.deinit();
        self.vae = try qwen_vae.loadDecoder(io, a, s, model_dir, llm.DT, .{ .scaled = .{ .factor = @floatCast(factor), .shift = @floatCast(shift_f) } });
        errdefer self.vae.deinit();
        try self.enc.load(io, a, s, model_dir, self.lcfg);
        log.info("[image] Ming-Image ready (MLLM {d} layers x {d} experts, DiT {d}x{d}, shift {d:.1})\n", .{
            self.lcfg.layers, self.lcfg.experts, self.dcfg.layers, self.dcfg.dim, self.shift,
        });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.enc.deinit();
        self.vae.deinit();
        self.dit.deinit();
        self.allocator.destroy(self);
    }

    pub const Opts = struct { guidance_scale: f32 = config.DEFAULT_GUIDANCE };

    /// The image [1, 4, H, W] f32 in [0, 1] (RGBA, owned) at a bucket size (`config.bucket`).
    pub fn generateImage(self: *Engine, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: Opts, progress: ?sse.Progress) !Array {
        var g = Graph.init(self.allocator, self.s);
        defer g.deinit();
        const lh: c_int = @intCast(height / VAE_DOWNSAMPLE);
        const lw: c_int = @intCast(width / VAE_DOWNSAMPLE);
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const noise = try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ 1, @intCast(self.dcfg.in_ch), lh, lw }, @as(usize, 4), ops.Dtype.float32, @as(f32, 0), @as(f32, 1), key });
        return self.generateFrom(prompt, noise, steps, opts, progress);
    }

    /// The pipeline from given f32 latents [1, C, H/8, W/8] (borrowed).
    pub fn generateFrom(self: *Engine, prompt: []const u8, init: Array, steps: u32, opts: Opts, progress: ?sse.Progress) !Array {
        const a = self.allocator;
        var g = Graph.init(a, self.s);
        defer g.deinit();
        const n: u32 = if (steps == 0) config.DEFAULT_STEPS else steps;
        const sigmas = try a.alloc(f32, n + 1);
        defer a.free(sigmas);
        scheduleSigmas(sigmas, n, self.shift);
        const n_run = stepCount(n);
        const cfg_on = opts.guidance_scale > 1.0;
        const sh = ops.shape(init);
        const ht = @divExact(sh[2], 2);
        const wt = @divExact(sh[3], 2);
        const n_img = ht * wt;
        log.info("[ming-image] {d}x{d} steps={d} guidance={d:.2} ({s})\n", .{ sh[3] * 8, sh[2] * 8, n, opts.guidance_scale, if (cfg_on) "two forwards per step" else "one forward per step" });

        if (progress) |p| p.emit("Encoding prompt", 0, n_run);
        var t0 = std.Io.Timestamp.now(self.io, .awake);
        const ids = try self.enc.promptIds(a, prompt);
        defer a.free(ids);
        const cond = try self.enc.encode(&g, ids);
        const cap_len: usize = @intCast(ops.shape(cond.learn)[1] + ops.shape(cond.direct)[1]);
        const pos_ids = try dit_mod.positions(a, @intCast(ht), @intCast(wt), cap_len);
        defer a.free(pos_ids);
        const rope = try dit_mod.ropeTable(&g, self.dcfg, pos_ids);
        const cap_rope: dit_mod.Rope = .{ .cos = try g.sliceAxis(rope.cos, 1, n_img, @intCast(pos_ids.len)), .sin = try g.sliceAxis(rope.sin, 1, n_img, @intCast(pos_ids.len)) };
        const cap = try self.dit.caption(&g, cond.learn, cond.direct, cap_rope);
        // The release's negative branch: both caption streams zeroed.
        const neg: ?Array = if (cfg_on) try self.dit.caption(&g, try g.mulScalar(cond.learn, 0), try g.mulScalar(cond.direct, 0), cap_rope) else null;
        try mlx.check(mlx.mlx_array_eval(cap));
        if (neg) |x| try mlx.check(mlx.mlx_array_eval(x));
        const prompt_ms = stageMs(self.io, &t0);

        const mods = try a.alloc(dit_mod.Mod, self.dit.noise.len + self.dit.blocks.len);
        defer a.free(mods);
        var lat = try g.keep(init);
        for (0..n_run) |i| {
            if (progress) |p| if (p.boundary()) return error.Cancelled;
            var st = g.scope();
            defer st.deinit();
            const temb = try self.dit.timeEmbed(&st, modelTime(sigmas[i], self.dcfg.t_scale));
            try self.dit.mods(&st, temb, mods);
            const patches = try dit_mod.patchify(&st, lat);
            const step: dit_mod.Dit.Step = .{ .patches = patches, .caption = cap, .mods = mods, .temb = temb, .rope = rope, .n_img = n_img };
            var pred = try self.dit.forward(&st, step);
            if (neg) |nc| {
                var ns = step;
                ns.caption = nc;
                const pn = try self.dit.forward(&st, ns);
                pred = try st.add(pred, try st.mul(try st.sub(pred, pn), try st.scalar(opts.guidance_scale, .float32)));
            }
            // The model predicts the negated velocity.
            const v = try dit_mod.unpatchify(&st, try st.neg(pred), @intCast(self.dcfg.in_ch), ht, wt);
            const next = try g.keep(try flow_match.eulerStep(&st, lat, v, sigmas[i + 1] - sigmas[i], .float32));
            try mlx.check(mlx.mlx_array_eval(next));
            g.drop(lat);
            lat = next;
            if (progress) |p| p.emit("Generating", @intCast(i + 1), n_run);
        }
        const denoise_ms = stageMs(self.io, &t0);

        if (progress) |p| p.emit("Decoding image", n_run, n_run);
        const px = try self.vae.decode(&g, lat);
        // The VAE's clamp to [-1, 1], then the image processor's x/2 + 1/2.
        const img = try g.minimum(try g.maximum(try g.addScalar(try g.mulScalar(px, 0.5), 0.5), try g.scalar(0, .float32)), try g.scalar(1, .float32));
        const out = g.escape(img);
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        log.info("[ming-image] stages: prompt {d} ms ({d} tokens), denoise {d} ms ({d} forwards), vae decode {d} ms\n", .{ prompt_ms, ids.len, denoise_ms, n_run, stageMs(self.io, &t0) });
        return out;
    }
};

// ── tests ──

const testing = std.testing;

test "ming_image schedule: linspace(1, 0) shifted by 6, the terminal 0, and the zero-length last step left out" {
    var sig: [13]f32 = undefined;
    scheduleSigmas(&sig, 12, 6.0);
    try testing.expectEqual(@as(f32, 1.0), sig[0]);
    try testing.expectEqual(@as(f32, 0.0), sig[11]);
    try testing.expectEqual(@as(f32, 0.0), sig[12]);
    // σ₁ = 6·(10/11) / (1 + 5·(10/11))
    try testing.expectApproxEqAbs(@as(f32, 60.0 / 61.0), sig[1], 1e-6);
    try testing.expectEqual(@as(u32, 11), stepCount(12));
    var one: [2]f32 = undefined;
    scheduleSigmas(&one, 1, 6.0);
    try testing.expectEqual([2]f32{ 1, 0 }, one);
    try testing.expectEqual(@as(u32, 1), stepCount(1));
    try testing.expectEqual(@as(f32, 0), modelTime(1.0, 1000));
    try testing.expectEqual(@as(f32, 1000), modelTime(0.0, 1000));
}

// Reference oracle on a tiny random pack (`tests/dump_ming_image_fixtures.py`, f32 on CPU): the prompt ids, both
// caption streams, one DiT forward on the reference's own streams at a canvas both sequences pad, one VAE decode,
// and 3-step pipelines without and with CFG. Env-gated: MING_IMAGE_TEST_MODEL + MING_IMAGE_FIXTURE.
fn parity(g: *Graph, got: Array, want: Array) !struct { cos: f32, rms: f32 } {
    var c = g.scope();
    defer c.deinit();
    const x = try c.reshape(try c.astype(got, .float32), &.{-1});
    const y = try c.reshape(try c.astype(want, .float32), &.{-1});
    const all = try c.concat(&.{ try c.sum(try c.mul(x, y), 0, true), try c.sum(try c.mul(x, x), 0, true), try c.sum(try c.mul(y, y), 0, true) }, 0);
    try mlx.check(mlx.mlx_array_eval(all));
    const d = mlx.mlx_array_data_float32(all).?;
    return .{ .cos = d[0] / @sqrt(d[1] * d[2]), .rms = @sqrt(d[1] / d[2]) };
}

fn expectParity(g: *Graph, name: []const u8, got: Array, want: Array, min_cos: f32) !void {
    try testing.expectEqual(mlx.mlx_array_size(want), mlx.mlx_array_size(got));
    const p = try parity(g, got, want);
    log.info("[ming-image oracle] {s}: cos {d:.6} rms ratio {d:.4}\n", .{ name, p.cos, p.rms });
    try testing.expect(p.cos >= min_cos);
    try testing.expect(@abs(p.rms - 1.0) < 0.03);
}

test "ming_image oracle: conditioning, DiT, VAE and the pipeline match the release on a tiny pack" {
    const model = std.mem.span(std.c.getenv("MING_IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
    const path = std.mem.span(std.c.getenv("MING_IMAGE_FIXTURE") orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try @import("../../sdk/weights/load.zig").loadWeightsSingleFile(a, path);
    defer fx.deinit();
    const eng = try Engine.loadOn(io, a, model, mlx.mlx_default_gpu_stream_new());
    defer eng.deinit();
    var g = Graph.init(a, eng.s);
    defer g.deinit();
    const get = struct {
        fn f(w: anytype, k: []const u8) !Array {
            const x = w.get(k) orelse return error.MissingFixture;
            try mlx.check(mlx.mlx_array_eval(x));
            return x;
        }
    }.f;
    const prompt = "A minimal poster that says \"Ming\" over a red sun.";

    const ids = try eng.enc.promptIds(a, prompt);
    defer a.free(ids);
    const want_ids = try get(&fx, "ids");
    try testing.expectEqual(@as(usize, @intCast(ops.shape(want_ids)[0])), ids.len);
    for (ids, mlx.mlx_array_data_int32(want_ids).?[0..ids.len]) |t, w| try testing.expectEqual(@as(i32, @intCast(t)), w);

    const cond = try eng.enc.encode(&g, ids);
    try expectParity(&g, "learnable stream", cond.learn, try get(&fx, "learn"), 0.999);
    try expectParity(&g, "direct stream", cond.direct, try get(&fx, "direct"), 0.999);

    // One DiT forward on the reference's own caption streams.
    {
        const lat = try get(&fx, "dit_latent");
        const sh = ops.shape(lat);
        const ht = @divExact(sh[2], 2);
        const wt = @divExact(sh[3], 2);
        const learn = try g.expandDims(try get(&fx, "learn"), 0);
        const direct = try g.expandDims(try get(&fx, "direct"), 0);
        const cap_len: usize = @intCast(ops.shape(learn)[1] + ops.shape(direct)[1]);
        const pos = try dit_mod.positions(a, @intCast(ht), @intCast(wt), cap_len);
        defer a.free(pos);
        const rope = try dit_mod.ropeTable(&g, eng.dcfg, pos);
        const n_img = ht * wt;
        const cap_rope: dit_mod.Rope = .{ .cos = try g.sliceAxis(rope.cos, 1, n_img, @intCast(pos.len)), .sin = try g.sliceAxis(rope.sin, 1, n_img, @intCast(pos.len)) };
        const cap = try eng.dit.caption(&g, try g.astype(learn, llm.DT), try g.astype(direct, llm.DT), cap_rope);
        const temb = try eng.dit.timeEmbed(&g, 0.37 * eng.dcfg.t_scale);
        const mods = try a.alloc(dit_mod.Mod, eng.dit.noise.len + eng.dit.blocks.len);
        defer a.free(mods);
        try eng.dit.mods(&g, temb, mods);
        const out = try eng.dit.forward(&g, .{ .patches = try dit_mod.patchify(&g, lat), .caption = cap, .mods = mods, .temb = temb, .rope = rope, .n_img = n_img });
        try expectParity(&g, "dit", try dit_mod.unpatchify(&g, out, sh[1], ht, wt), try get(&fx, "dit_out"), 0.999);
    }

    // The reference decoded `vae_in` as is; ours divides by the scaling factor first, as the pipeline does.
    const px = try eng.vae.decode(&g, try g.mulScalar(try get(&fx, "vae_in"), eng.vae.latents.scaled.factor));
    try expectParity(&g, "vae", try g.minimum(try g.maximum(px, try g.scalar(-1, .float32)), try g.scalar(1, .float32)), try get(&fx, "vae_out"), 0.999);

    for ([_]struct { name: []const u8, cfg: f32 }{ .{ .name = "e2e", .cfg = 1.0 }, .{ .name = "e2e_cfg", .cfg = 2.5 } }) |c| {
        const img = try g.adopt(try eng.generateFrom(prompt, try get(&fx, "e2e_noise"), 3, .{ .guidance_scale = c.cfg }, null));
        try expectParity(&g, c.name, img, try get(&fx, c.name), 0.998);
    }
}
