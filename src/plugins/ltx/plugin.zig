//! LTX-Video 2.3 / 2.5 on the media table (`/v1/video/generations`): text-to-video with sound, first/last-frame
//! keyframes, audio-to-video (two-stage), the two-stage and HQ pipelines, runtime LoRAs and the DiffVAE decoder.
//! Anything the pack cannot honor is a named refusal, never a silent downgrade (a missing first-frame encoder is the
//! one graceful case, as it always was). The result is the response's JSON members (24 fps RGB, PCM16 audio).

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const lora = @import("../../sdk/weights/lora.zig");
const frame_in = @import("../../sdk/media/frame_in.zig");
const video_out = @import("../../sdk/media/video_out.zig");
const wav = @import("../../sdk/media/wav.zig");
const ltx = @import("video.zig");
const ltx_audio = @import("audio.zig");
const diffvae_fwd = @import("diffvae_forward.zig");
const engine_mod = @import("engine.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .video,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .set_loras = &setLoras,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;
const A = mlx.mlx_array;
const FPS: f32 = 24.0;

const Plugin = struct {
    e: *engine_mod.Engine,
    result: []u8 = &.{},
};

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const stream: ?mlx.mlx_stream = if (ctx.stream.ctx == null) null else .{ .ctx = ctx.stream.ctx };
    const e = engine_mod.Engine.load(io(), gpa, ctx.model_dir.slice(), stream) catch |err| {
        out.* = Verdict.named(if (err == error.OutOfMemory) .busy else .failed, switch (err) {
            error.NoGemmaDir => "LTX's text encoder is not installed: a 2.3 pack reads mlx-community/gemma-3-12b-it-4bit (or LTX_GEMMA_DIR); a 2.5 pack ships it in gemma4-12b-ltx-v1/",
            else => @errorName(err),
        });
        return null;
    };
    const p = gpa.create(Plugin) catch {
        e.deinit();
        out.* = Verdict.named(.busy, "out of memory");
        return null;
    };
    p.* = .{ .e = e };
    return p;
}

fn unload(pp: *anyopaque) callconv(.c) void {
    const p: *Plugin = @ptrCast(@alignCast(pp));
    p.e.deinit();
    gpa.free(p.result);
    gpa.destroy(p);
}

/// Empty `loras` detaches; the same paths and scales reuse what is attached.
fn setLoras(pp: *anyopaque, loras: abi.Span(abi.Lora)) callconv(.c) Verdict {
    const p: *Plugin = @ptrCast(@alignCast(pp));
    var paths: [lora.MAX_LORAS][]const u8 = undefined;
    var scales: [lora.MAX_LORAS]f32 = undefined;
    const ls = loras.slice();
    if (ls.len > lora.MAX_LORAS) return bad("too many LoRA adapters requested");
    for (ls, 0..) |l, i| {
        paths[i] = l.path.slice();
        scales[i] = l.scale;
    }
    const matched = p.e.setLoras(paths[0..ls.len], scales[0..ls.len]) catch |err| return switch (err) {
        error.LoraNoMatch => bad("LoRA(s) have no modules matching this model's DiT — wrong LoRA for this architecture?"),
        error.BadLoraPath => bad("'lora_path'/'lora_paths' must be absolute path(s) to .safetensors file(s)"),
        error.TooManyLoras => bad("too many LoRA adapters requested"),
        error.OutOfMemory => Verdict.refuse(.busy, "out of memory"),
        else => bad("failed to load a LoRA file"),
    };
    if (ls.len > 0) log.info("[video] lora: matched {d} module-attachment(s) across {d} adapter(s)\n", .{ matched, ls.len });
    return .ok;
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

/// `default` when omitted, else held to [lo, hi].
fn stepsIn(b: *const job.Body, key: []const u8, default: u32, lo: u32, hi: u32) ?u32 {
    const n = b.uint(key) orelse return default;
    return if (n < lo or n > hi) null else @intCast(n);
}

fn u32Field(b: *const job.Body, key: []const u8, default: u32) u32 {
    return @intCast(@min(b.uint(key) orelse default, std.math.maxInt(u32)));
}

fn f32Field(b: *const job.Body, key: []const u8) ?f32 {
    return if (b.float(key)) |v| @floatCast(v) else null;
}

/// The frame at the full canvas (cropped to the encoder's /32 grid) and, for the two-stage pipelines, at stage 1's
/// half grid.
const Frames = struct {
    full: ?A = null,
    half: ?A = null,

    fn deinit(f: *Frames) void {
        if (f.full) |x| _ = mlx.mlx_array_free(x);
        if (f.half) |x| _ = mlx.mlx_array_free(x);
    }
};

fn encGrid(n: u32) u32 {
    return (n / 32) * 32;
}

/// The request's soundtrack (a2vid): the ORIGINAL decode (muxed into the result) and the conditioning latent.
const Soundtrack = struct {
    pcm: ?wav.Decoded = null,
    cond: ?A = null,

    fn deinit(t: *Soundtrack) void {
        if (t.pcm) |d| gpa.free(d.pcm);
        if (t.cond) |c| _ = mlx.mlx_array_free(c);
    }
};

fn soundtrack(e: *engine_mod.Engine, b64: []const u8, pipeline: engine_mod.Pipeline, num_frames: u32, out: *Soundtrack) !?Verdict {
    if (engine_mod.a2vidPipelineError(pipeline)) |_| return bad("audio-to-video requires a two-stage pipeline — set \"pipeline\":\"two_stage\" or \"two_stage_hq\"");
    const acomp = if (e.audio) |*a| a else return bad("audio-to-video requires audio_vae.safetensors (encoder) — download it into the model dir");
    const bytes = job.base64Decode(gpa, b64) catch return bad("audio: invalid base64");
    defer gpa.free(bytes);
    const dec = wav.decode(gpa, bytes) catch return bad("audio: expected a PCM16/PCM24/float32 WAV");
    out.pcm = dec;
    // Stereo at 16 kHz → mel → VAE encode, truncated to the video's token budget (the reference never pads).
    const stereo = try wav.toStereoInterleaved(gpa, dec.pcm, dec.channels);
    defer gpa.free(stereo);
    const cond_pcm = try wav.resampleLinear(gpa, stereo, 2, dec.sample_rate, ltx_audio.COND_SAMPLE_RATE);
    defer gpa.free(cond_pcm);
    const max_tokens = ltx.computeAudioTokenCount(num_frames, FPS);
    out.cond = ltx_audio.encodeAudioCond(gpa, acomp, cond_pcm, max_tokens, e.s) catch |err| {
        if (err == error.AudioTooShort) return bad("audio: clip too short (needs at least ~50 ms)");
        return job.failure(err, "audio: conditioning encode failed");
    };
    log.info("[video] audio-to-video: {d} tokens (budget {d}) from {d} Hz {d}ch clip\n", .{ mlx.getShape(out.cond.?)[1], max_tokens, dec.sample_rate, dec.channels });
    return null;
}

fn step(pp: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const p: *Plugin = @ptrCast(@alignCast(pp));
    return run(p, req, progress) catch |err| job.failure(err, "generation failed");
}

fn run(p: *Plugin, req: *const abi.Request, progress: *const abi.Progress) !Verdict {
    const e = p.e;
    const dir = req.dir orelse return Verdict.refuse(.failed, "ltx: the host gave no pack dir");
    var b = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer b.deinit();
    const prompt = b.str("prompt") orelse return bad("missing 'prompt'");
    if (prompt.len == 0) return bad("empty 'prompt'");

    const num_frames = u32Field(&b, "num_frames", 9);
    const height = u32Field(&b, "height", 256);
    const width = u32Field(&b, "width", 384);
    const rgb = video_out.rgbBytes(num_frames, width, height);
    if (rgb > video_out.MAX_RGB_BYTES) return req.refuseFmt(.unsupported, video_out.TOO_LARGE, video_out.TOO_LARGE_FMT, .{ num_frames, width, height, rgb / (1024 * 1024), video_out.MAX_RGB_BYTES / (1024 * 1024) });
    const seed = b.uint("seed") orelse 42;

    const pipeline = engine_mod.Pipeline.fromName(b.str("pipeline"));
    const guiders = engine_mod.guiderDefaults(pipeline, f32Field(&b, "cfg_scale"), f32Field(&b, "cfg_audio_scale"), f32Field(&b, "stg_scale"));
    const steps = stepsIn(&b, "steps", guiders.stage1_steps_default, 1, 100) orelse return bad("'steps' must be in [1,100]");
    const stage2_steps = stepsIn(&b, "stage2_steps", 0, 0, 100) orelse return bad("'stage2_steps' must be in [0,100] (0 = the pipeline's default)");

    // Two-stage prerequisites: never a silent one-stage downgrade.
    if (pipeline != .one_stage) {
        if (height % 64 != 0 or width % 64 != 0) return bad("two-stage pipelines need width/height divisible by 64 (half-resolution stage)");
        if (e.vae_encoder == null) return bad("two-stage pipelines require vae_encoder.safetensors (latent statistics) — download it into the model dir");
        if (dir.fileBytes(engine_mod.TransformerVariant.dev.fileName()) == 0) return bad("two-stage pipelines require transformer-dev.safetensors — download it into the model dir");
        if (dir.fileBytes(engine_mod.TransformerVariant.distilled.fileName()) == 0) return bad("two-stage pipelines require transformer-distilled.safetensors (stage-2 refine) — download it into the model dir");
        if (dir.fileBytes(ltx.UPSAMPLER_PREFIX ++ ".safetensors") == 0) return bad("two-stage pipelines require spatial_upscaler_x2_v1_1.safetensors — download it into the model dir");
    }

    // The user asked for THIS soundtrack: every failure is named.
    var track: Soundtrack = .{};
    defer track.deinit();
    if (b.str("audio")) |a64| if (a64.len > 0) if (try soundtrack(e, a64, pipeline, num_frames, &track)) |v| return v;

    const pos_ids = try engine_mod.tokenizePadded(gpa, &e.tok, prompt);
    defer gpa.free(pos_ids);
    const neg_ids = try engine_mod.tokenizePadded(gpa, &e.tok, engine_mod.NEGATIVE_PROMPT);
    defer gpa.free(neg_ids);

    // First frame: graceful (no encoder or an undecodable image is text-to-video, as it always was).
    var first: Frames = .{};
    defer first.deinit();
    var enc: ?*const ltx.Component = null;
    if (b.str("first_frame_image")) |b64| if (b64.len > 0) {
        if (e.vae_encoder) |*ve| {
            if (job.imageUpload(gpa, b64)) |bytes| {
                defer gpa.free(bytes);
                first.full = frame_in.pm1(gpa, bytes, encGrid(height), encGrid(width), .bfloat16, e.s);
                if (first.full != null) enc = ve else log.warn("[video] first_frame_image decode failed — text-to-video\n", .{});
                if (pipeline != .one_stage and first.full != null) first.half = frame_in.pm1(gpa, bytes, encGrid(height / 2), encGrid(width / 2), .bfloat16, e.s);
            } else |err| log.warn("[video] first_frame_image base64 decode failed: {} — text-to-video\n", .{err});
        } else log.warn("[video] vae_encoder not loaded — ignoring first_frame_image (text-to-video)\n", .{});
    };

    // Last frame: the user asked for this ending, so every failure is named.
    var last: Frames = .{};
    defer last.deinit();
    if (b.str("last_frame_image")) |b64| if (b64.len > 0) {
        const ve = if (e.vae_encoder) |*x| x else return bad("last frame conditioning needs vae_encoder.safetensors — download it into the model dir");
        const bytes = job.imageUpload(gpa, b64) catch return bad("last_frame_image: invalid base64");
        defer gpa.free(bytes);
        if (num_frames < 9) return bad("last_frame_image needs at least 9 frames (one latent frame cannot hold an anchor and still generate)");
        last.full = frame_in.pm1(gpa, bytes, encGrid(height), encGrid(width), .bfloat16, e.s) orelse return bad("last_frame_image: expected a PNG/JPEG image");
        if (pipeline != .one_stage) last.half = frame_in.pm1(gpa, bytes, encGrid(height / 2), encGrid(width / 2), .bfloat16, e.s) orelse return bad("last_frame_image: expected a PNG/JPEG image");
        enc = ve;
    };

    // The DiffVAE is LTX's own decoder; its tile budget follows the memory free when this request began.
    const machine = req.machine orelse &abi.Machine{};
    var vae_choice = ltx.VaeChoice{ .conv = &e.vae, .seed = seed +% 0x5DEC0DE, .tile_tokens = diffvae_fwd.tileTokensForMemory(machine.ram_available) };
    if (b.str("decoder")) |dec| {
        if (std.mem.eql(u8, dec, "diffusion")) {
            if (dir.fileBytes(diffvae_fwd.FILE_NAME) == 0) return bad("'decoder':\"diffusion\" requires vae_diffusion_decoder.safetensors — the 8-bit LTX-2.5 pack ships it, the 4-bit pack does not");
            vae_choice.diffusion = try e.ensureDiffusionDecoder(io());
            log.info("[video] decoder: diffusion (DiffVAE)\n", .{});
        } else if (!std.mem.eql(u8, dec, "conv") and dec.len > 0) return bad("'decoder' must be \"conv\" or \"diffusion\"");
    }

    const prog = job.progressOf(progress);
    var frames = generate(e, pipeline, vae_choice, enc, &first, &last, &track, pos_ids, neg_ids, num_frames, height, width, steps, stage2_steps, seed, guiders, prog) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        error.KeyframeCanvasTooShort => bad("keyframes need at least 9 frames (one latent frame cannot hold an anchor and still generate)"),
        else => job.failure(err, "generation failed"),
    };
    defer frames.deinit(gpa);

    var pcm: ?[]u8 = null;
    defer if (pcm) |x| gpa.free(x);
    var audio: ?video_out.Audio = null;
    if (track.pcm) |dec| {
        // a2vid muxes the ORIGINAL clip (higher fidelity than a VAE round trip), trimmed to the video; >2 channels
        // downmix to stereo, the layouts the client builds.
        const mux: []const f32 = if (dec.channels > 2) try wav.toStereoInterleaved(gpa, dec.pcm, dec.channels) else dec.pcm;
        defer if (dec.channels > 2) gpa.free(mux);
        const ch: u32 = @min(dec.channels, 2);
        pcm = try video_out.pcm16le(gpa, mux[0..engine_mod.a2vidMuxSampleCount(mux.len, ch, dec.sample_rate, frames.frames, FPS)]);
        audio = .{ .sample_rate = dec.sample_rate, .channels = ch, .pcm16 = pcm.? };
    } else if (e.audio) |*acomp| if (frames.audio_latent) |al| {
        if (ltx_audio.decodeAudio(gpa, acomp, al, e.s)) |wv| {
            var w = wv;
            defer w.deinit(gpa);
            pcm = try video_out.pcm16le(gpa, w.pcm);
            audio = .{ .sample_rate = w.sample_rate, .channels = w.channels, .pcm16 = pcm.? };
        } else |err| log.warn("[video] audio decode failed: {} — video stays silent\n", .{err});
    };

    const members = try video_out.members(gpa, frames.frames, frames.height, frames.width, @intFromFloat(FPS), frames.rgb, audio);
    gpa.free(p.result);
    p.result = members;
    return .ok;
}

fn generate(
    e: *engine_mod.Engine,
    pipeline: engine_mod.Pipeline,
    vae_choice: ltx.VaeChoice,
    enc: ?*const ltx.Component,
    first: *const Frames,
    last: *const Frames,
    track: *const Soundtrack,
    pos_ids: []const i32,
    neg_ids: []const i32,
    num_frames: u32,
    height: u32,
    width: u32,
    steps: u32,
    stage2_steps: u32,
    seed: u64,
    guiders: engine_mod.Guiders,
    prog: ltx.Progress,
) !ltx.VideoFrames {
    switch (pipeline) {
        // The schedule the loaded variant was trained for; the request never forces a swap here.
        .one_stage => return ltx.generateVideoFrames(io(), gpa, e.ltx_cfg, &e.transformer, &e.connector, vae_choice, enc, first.full, last.full, e.gemma_dir, pos_ids, neg_ids, engine_mod.PAD_ID, num_frames, height, width, FPS, steps, e.transformer_variant == .distilled, seed, guiders.vp, guiders.ap, prog, e.s),
        .two_stage, .two_stage_hq => {
            try e.ensureTransformer(.dev);
            var swapper = engine_mod.Stage2Swap{ .engine = e };
            const opts = ltx.TwoStageOpts{
                .hq = pipeline == .two_stage_hq,
                .stage1_steps = steps,
                .stage2_steps = stage2_steps,
                .upsampler = try e.ensureUpsampler(io()),
                .swap_ctx = @ptrCast(&swapper),
                .swap = engine_mod.Stage2Swap.swap,
            };
            return ltx.generateVideoFramesTwoStage(io(), gpa, e.ltx_cfg, &e.transformer, &e.connector, vae_choice, &e.vae_encoder.?, first.half, first.full, last.half, last.full, track.cond, e.gemma_dir, pos_ids, neg_ids, engine_mod.PAD_ID, num_frames, height, width, FPS, opts, seed, guiders.vp, guiders.ap, prog, e.s);
        },
    }
}

fn encode(pp: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const p: *Plugin = @ptrCast(@alignCast(pp));
    if (p.result.len == 0) return Verdict.refuse(.failed, "ltx: no finished job to encode");
    out.* = .of(p.result);
    return .ok;
}

// ── the tiny pack ──

const Tensors = sdk.tensors.Tensors;
/// Fixture widths: video 2 heads x 32, audio 2 heads x 16, the text encoder 64 wide over 2 layers.
const V = 64;
const AU = 32;
const LAT = 128;
const TE = 64;
const TE_LAYERS = 2;

/// A 4-bit, group-32 affine linear with a dense bias, as the DiT's projections ship.
fn qlin(t: *Tensors, base: []const u8, out: u64, in: u64) !void {
    try t.words("{s}.weight", .{base}, &.{ out, in / 8 });
    try t.fill("{s}.scales", .{base}, &.{ out, in / 32 }, .BF16, 0.01);
    try t.fill("{s}.biases", .{base}, &.{ out, in / 32 }, .BF16, -0.075);
    try t.rand("{s}.bias", .{base}, &.{out}, .BF16, 0.02);
}

fn dense(t: *Tensors, base: []const u8, out: u64, in: u64) !void {
    try t.rand("{s}.weight", .{base}, &.{ out, in }, .BF16, 0.05);
    try t.rand("{s}.bias", .{base}, &.{out}, .BF16, 0.02);
}

fn ditAttn(t: *Tensors, a: std.mem.Allocator, base: []const u8, q_in: u64, kv_in: u64, inner: u64, heads: u64) !void {
    const p = struct {
        fn k(aa: std.mem.Allocator, b: []const u8, comptime n: []const u8) ![]const u8 {
            return std.fmt.allocPrint(aa, "{s}." ++ n, .{b});
        }
    }.k;
    try qlin(t, try p(a, base, "to_q"), inner, q_in);
    try qlin(t, try p(a, base, "to_k"), inner, kv_in);
    try qlin(t, try p(a, base, "to_v"), inner, kv_in);
    try qlin(t, try p(a, base, "to_out"), q_in, inner);
    try qlin(t, try p(a, base, "to_gate_logits"), heads, q_in);
    try t.fill("{s}.q_norm.weight", .{base}, &.{inner}, .BF16, 1);
    try t.fill("{s}.k_norm.weight", .{base}, &.{inner}, .BF16, 1);
}

fn adaln(t: *Tensors, base: []const u8, dim: u64, params: u64) !void {
    try t.rand("{s}.emb.timestep_embedder.linear1.weight", .{base}, &.{ dim, 256 }, .BF16, 0.05);
    try t.rand("{s}.emb.timestep_embedder.linear1.bias", .{base}, &.{dim}, .BF16, 0.02);
    try t.rand("{s}.emb.timestep_embedder.linear2.weight", .{base}, &.{ dim, dim }, .BF16, 0.05);
    try t.rand("{s}.emb.timestep_embedder.linear2.bias", .{base}, &.{dim}, .BF16, 0.02);
    try t.rand("{s}.linear.weight", .{base}, &.{ params * dim, dim }, .BF16, 0.05);
    try t.rand("{s}.linear.bias", .{base}, &.{params * dim}, .BF16, 0.02);
}

fn tinyDit(a: std.mem.Allocator) ![]u8 {
    var t = Tensors.init(a, 201);
    const b = "transformer.transformer_blocks.0";
    inline for (.{ .{ "scale_shift_table", 9, V }, .{ "audio_scale_shift_table", 9, AU }, .{ "scale_shift_table_a2v_ca_video", 5, V }, .{ "scale_shift_table_a2v_ca_audio", 5, AU }, .{ "prompt_scale_shift_table", 2, V }, .{ "audio_prompt_scale_shift_table", 2, AU } }) |x|
        try t.rand(b ++ "." ++ x[0], .{}, &.{ x[1], x[2] }, .F32, 0.05);
    try ditAttn(&t, a, b ++ ".attn1", V, V, V, 2);
    try ditAttn(&t, a, b ++ ".attn2", V, V, V, 2);
    try ditAttn(&t, a, b ++ ".audio_attn1", AU, AU, AU, 2);
    try ditAttn(&t, a, b ++ ".audio_attn2", AU, AU, AU, 2);
    try ditAttn(&t, a, b ++ ".audio_to_video_attn", V, AU, AU, 2);
    try ditAttn(&t, a, b ++ ".video_to_audio_attn", AU, V, AU, 2);
    try qlin(&t, b ++ ".ff.proj_in", 4 * V, V);
    try qlin(&t, b ++ ".ff.proj_out", V, 4 * V);
    try qlin(&t, b ++ ".audio_ff.proj_in", 4 * AU, AU);
    try qlin(&t, b ++ ".audio_ff.proj_out", AU, 4 * AU);
    try dense(&t, "transformer.patchify_proj", V, LAT);
    try dense(&t, "transformer.audio_patchify_proj", AU, LAT);
    try adaln(&t, "transformer.adaln_single", V, 9);
    try adaln(&t, "transformer.av_ca_video_scale_shift_adaln_single", V, 4);
    try adaln(&t, "transformer.audio_adaln_single", AU, 9);
    try adaln(&t, "transformer.av_ca_audio_scale_shift_adaln_single", AU, 4);
    try adaln(&t, "transformer.av_ca_a2v_gate_adaln_single", V, 1);
    try adaln(&t, "transformer.av_ca_v2a_gate_adaln_single", AU, 1);
    try adaln(&t, "transformer.prompt_adaln_single", V, 2);
    try adaln(&t, "transformer.audio_prompt_adaln_single", AU, 2);
    try t.rand("transformer.scale_shift_table", .{}, &.{ 2, V }, .F32, 0.05);
    try t.rand("transformer.audio_scale_shift_table", .{}, &.{ 2, AU }, .F32, 0.05);
    try dense(&t, "transformer.proj_out", LAT, V);
    try dense(&t, "transformer.audio_proj_out", LAT, AU);
    return t.bytes();
}

fn tinyConnector(a: std.mem.Allocator) ![]u8 {
    var t = Tensors.init(a, 202);
    const agg = TE * (TE_LAYERS + 1);
    try dense(&t, "connector.text_embedding_projection.video_aggregate_embed", V, agg);
    try dense(&t, "connector.text_embedding_projection.audio_aggregate_embed", AU, agg);
    inline for (.{ .{ "connector.video_embeddings_connector", V }, .{ "connector.audio_embeddings_connector", AU } }) |c| {
        const dim = c[1];
        const p = c[0] ++ ".transformer_1d_blocks.0";
        try t.rand(c[0] ++ ".learnable_registers", .{}, &.{ 128, dim }, .BF16, 0.5);
        inline for (.{ "to_q", "to_k", "to_v", "to_out.0" }) |n| try dense(&t, p ++ ".attn1." ++ n, dim, dim);
        try dense(&t, p ++ ".attn1.to_gate_logits", 2, dim);
        try t.fill(p ++ ".attn1.q_norm.weight", .{}, &.{dim}, .BF16, 1);
        try t.fill(p ++ ".attn1.k_norm.weight", .{}, &.{dim}, .BF16, 1);
        try dense(&t, p ++ ".ff.net.0.proj", 4 * dim, dim);
        try dense(&t, p ++ ".ff.net.2", dim, 4 * dim);
    }
    return t.bytes();
}

/// The conv decoder's structure at 16 channels: 2/2/4/6/4 residual blocks between four depth-to-space convs.
fn tinyVae(a: std.mem.Allocator) ![]u8 {
    const C = 16;
    var t = Tensors.init(a, 203);
    try t.fill("vae_decoder.per_channel_statistics.mean", .{}, &.{LAT}, .F32, 0);
    try t.fill("vae_decoder.per_channel_statistics.std", .{}, &.{LAT}, .F32, 1);
    try t.rand("vae_decoder.conv_in.conv.weight", .{}, &.{ C, 3, 3, 3, LAT }, .BF16, 0.02);
    try t.rand("vae_decoder.conv_in.conv.bias", .{}, &.{C}, .BF16, 0.02);
    inline for (.{ .{ 0, 2 }, .{ 2, 2 }, .{ 4, 4 }, .{ 6, 6 }, .{ 8, 4 } }) |st| for (0..st[1]) |blk| inline for (.{ "conv1", "conv2" }) |cv| {
        try t.rand("vae_decoder.up_blocks.{d}.res_blocks.{d}." ++ cv ++ ".conv.weight", .{ st[0], blk }, &.{ C, 3, 3, 3, C }, .BF16, 0.02);
        try t.rand("vae_decoder.up_blocks.{d}.res_blocks.{d}." ++ cv ++ ".conv.bias", .{ st[0], blk }, &.{C}, .BF16, 0.02);
    };
    inline for (.{ .{ 1, 8 }, .{ 3, 8 }, .{ 5, 2 }, .{ 7, 4 } }) |up| {
        try t.rand("vae_decoder.up_blocks.{d}.conv.conv.weight", .{up[0]}, &.{ C * up[1], 3, 3, 3, C }, .BF16, 0.05);
        try t.rand("vae_decoder.up_blocks.{d}.conv.conv.bias", .{up[0]}, &.{C * up[1]}, .BF16, 0.02);
    }
    try t.rand("vae_decoder.conv_out.conv.weight", .{}, &.{ 48, 3, 3, 3, C }, .BF16, 0.05);
    try t.rand("vae_decoder.conv_out.conv.bias", .{}, &.{48}, .BF16, 0.02);
    return t.bytes();
}

/// A two-layer Gemma-3 text model (the 2.5 pack's in-pack encoder slot; the capture reads every layer's state).
fn tinyGemma(a: std.mem.Allocator) ![]u8 {
    var t = Tensors.init(a, 204);
    try t.rand("model.embed_tokens.weight", .{}, &.{ 260, TE }, .BF16, 0.5);
    for (0..TE_LAYERS) |l| {
        inline for (.{ "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm" }) |n|
            try t.fill("model.layers.{d}." ++ n ++ ".weight", .{l}, &.{TE}, .BF16, 0);
        inline for (.{ .{ "self_attn.q_proj", 64, TE }, .{ "self_attn.k_proj", 32, TE }, .{ "self_attn.v_proj", 32, TE }, .{ "self_attn.o_proj", TE, 64 }, .{ "mlp.gate_proj", 128, TE }, .{ "mlp.up_proj", 128, TE }, .{ "mlp.down_proj", TE, 128 } }) |w|
            try t.rand("model.layers.{d}." ++ w[0] ++ ".weight", .{l}, &.{ w[1], w[2] }, .BF16, 0.1);
        try t.fill("model.layers.{d}.self_attn.q_norm.weight", .{l}, &.{32}, .BF16, 0);
        try t.fill("model.layers.{d}.self_attn.k_norm.weight", .{l}, &.{32}, .BF16, 0);
    }
    try t.fill("model.norm.weight", .{}, &.{TE}, .BF16, 0);
    return t.bytes();
}

pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const tok = try sdk.byte_tokenizer.json(a, &.{});
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = 
            \\{"model_type":"AudioVideo","model_version":"2.5.0",
            \\ "transformer":{"num_layers":1,"num_attention_heads":2,"attention_head_dim":32,
            \\  "audio_num_attention_heads":2,"audio_attention_head_dim":16},
            \\ "connector":{"num_blocks":1}}
            },
            .{ .name = "transformer-dev.safetensors", .bytes = try tinyDit(a) },
            .{ .name = "connector.safetensors", .bytes = try tinyConnector(a) },
            .{ .name = "vae_decoder.safetensors", .bytes = try tinyVae(a) },
            .{ .name = "gemma4-12b-ltx-v1/config.json", .bytes = 
            \\{"model_type":"gemma3_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":2,
            \\ "num_key_value_heads":1,"head_dim":32,"intermediate_size":128,"vocab_size":260,"rms_norm_eps":1e-6,
            \\ "query_pre_attn_scalar":32,"rope_theta":1000000.0,"rope_local_base_freq":10000.0,
            \\ "sliding_window":512,"sliding_window_pattern":6,"max_position_embeddings":2048}
            },
            .{ .name = "gemma4-12b-ltx-v1/model.safetensors", .bytes = try tinyGemma(a) },
            .{ .name = "gemma4-12b-ltx-v1/tokenizer.json", .bytes = tok },
        }),
        .body = "{\"prompt\":\"a red ball rolls\",\"num_frames\":9,\"width\":64,\"height\":64,\"steps\":2,\"seed\":3}",
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());

test "ltx plugin: a seeded clip on the tiny 2.5 pack is deterministic JSON members; bad fields are refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expect(std.mem.startsWith(u8, first, "\"frames\":9,\"height\":64,\"width\":64,\"fps\":24,\"format\":\"rgb8\",\"data\":\""));
    try testing.expectEqualStrings(first, (try pack.run(pack.fx.body)).ok);
    inline for (.{
        .{ "{}", "missing 'prompt'" },
        .{ "{\"prompt\":\"x\",\"steps\":0}", "'steps' must be in [1,100]" },
        .{ "{\"prompt\":\"x\",\"pipeline\":\"two_stage\",\"width\":96}", "two-stage pipelines need width/height divisible by 64 (half-resolution stage)" },
        .{ "{\"prompt\":\"x\",\"pipeline\":\"two_stage\"}", "two-stage pipelines require vae_encoder.safetensors (latent statistics) — download it into the model dir" },
        .{ "{\"prompt\":\"x\",\"audio\":\"AAAA\"}", "audio-to-video requires a two-stage pipeline — set \"pipeline\":\"two_stage\" or \"two_stage_hq\"" },
        .{ "{\"prompt\":\"x\",\"last_frame_image\":\"AAAA\"}", "last frame conditioning needs vae_encoder.safetensors — download it into the model dir" },
        .{ "{\"prompt\":\"x\",\"decoder\":\"diffusion\"}", "'decoder':\"diffusion\" requires vae_diffusion_decoder.safetensors — the 8-bit LTX-2.5 pack ships it, the 4-bit pack does not" },
        .{ "{\"prompt\":\"x\",\"decoder\":\"other\"}", "'decoder' must be \"conv\" or \"diffusion\"" },
    }) |c| try testing.expectEqualStrings(c[1], (try pack.run(c[0])).refused.why.slice());
}
