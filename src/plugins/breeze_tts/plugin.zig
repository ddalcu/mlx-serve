//! Breeze-TTS-2 on the media table: `/v1/audio/speech` from text, with instruction-based voice design
//! (`instruction`), speaker slots (`voice`: "S0".."S9"), and zero-shot cloning from a base64 reference WAV
//! (`ref_audio`) plus its transcript (`ref_text`). The result is a 24 kHz mono WAV.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const log = @import("../../sdk/util/log.zig");
const progress_mod = @import("../../sdk/media/progress.zig");
const wav = @import("../../sdk/media/wav.zig");
const mlx = @import("../../mlx/ffi.zig");
const engine = @import("engine.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .audio,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

const Engine = struct {
    eng: *engine.Engine,
    out: []const u8 = &.{},

    fn setOut(self: *Engine, bytes: []const u8) void {
        gpa.free(self.out);
        self.out = bytes;
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    // Compute on the stream the host handed us; the engine never makes its own GPU stream.
    const s = mlx.mlx_stream{ .ctx = ctx.stream.ctx };
    const eng = engine.Engine.load(io, gpa, s, ctx.model_dir.slice()) catch |err| {
        out.* = job.failure(err, "breeze_tts load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        eng.deinit();
        out.* = job.failure(error.OutOfMemory, "breeze_tts load");
        return null;
    };
    e.* = .{ .eng = eng };
    return e;
}

fn unload(e: *anyopaque) callconv(.c) void {
    const eng: *Engine = @ptrCast(@alignCast(e));
    eng.eng.deinit();
    gpa.free(eng.out);
    gpa.destroy(eng);
}

/// Collects the engine's f32 chunks and folds host progress and the cancel boundary into the frame loop.
const Sink = struct {
    buf: std.ArrayList(f32) = .empty,
    p: progress_mod.Progress,
    spp: usize, // codec upsample ratio (1920 = 12.5 fps @ 24 kHz), from the loaded engine
    done_frames: u32 = 0,
    total_frames: u32 = 1,
    failed: bool = false,

    fn emit(self: *Sink, pcm: []const f32) void {
        self.buf.appendSlice(gpa, pcm) catch {
            self.failed = true;
            return;
        };
        self.done_frames += @intCast(pcm.len / self.spp);
        self.p.emit("breeze", self.done_frames, self.total_frames);
    }

    /// Called before every frame: the boundary is where a cancel lands and chat gets the GPU.
    fn atBoundary(self: *Sink) bool {
        return self.p.boundary();
    }
};

fn sinkEmit(user: *anyopaque, pcm: []const f32) void {
    const sink: *Sink = @ptrCast(@alignCast(user));
    sink.emit(pcm);
}

fn sinkCancel(user: *anyopaque) bool {
    const sink: *Sink = @ptrCast(@alignCast(user));
    return sink.atBoundary();
}

/// Keys the host consumes for any speech route (routing, SSE, output format, rate) and so must not be
/// refused here, plus the keys this plugin reads itself. Anything else is refused by name in `step`.
const consumed_keys = [_][]const u8{
    "model",           "stream",  "response_format", "speed", // host
    "input",           "text",    "voice",           "instruction", "instructions", // plugin
    "ref_audio",       "ref_text", "seed",           "max_frames",
};

fn step(e: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const eng: *Engine = @ptrCast(@alignCast(e));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();

    // Every field the plugin cannot honour is refused by name, never ignored.
    var it = body.parsed.value.object.iterator();
    while (it.next()) |kv| {
        const k = kv.key_ptr.*;
        var known = false;
        for (consumed_keys) |c| known = known or std.mem.eql(u8, k, c);
        if (!known) return req.refuseFmt(
            .unsupported,
            "breeze_tts does not know this field",
            "'{s}' is not supported by breeze_tts",
            .{k},
        );
    }

    const text = body.str("input") orelse body.str("text") orelse
        return bad("this route needs text in 'input'");
    if (text.len == 0) return bad("'input' is empty");

    const voice = body.str("voice");
    if (voice) |v| {
        if (!engine.voiceSlotValid(v)) return req.refuseFmt(
            .unsupported,
            "'voice' must be a speaker slot",
            "'voice' must be \"S0\"..\"S9\", not '{s}'",
            .{v},
        );
    }
    // OpenAI's speech field is "instructions"; breeze's own is "instruction". Either names the same thing.
    const instruction = body.str("instruction") orelse body.str("instructions");

    const ref_audio = body.str("ref_audio");
    var ref_bytes: []u8 = &.{};
    defer gpa.free(ref_bytes);
    if (ref_audio) |b64| {
        ref_bytes = job.base64Decode(gpa, b64) catch
            return bad("'ref_audio' is not valid base64");
        const decoded = wav.decode(gpa, ref_bytes) catch {
            gpa.free(ref_bytes);
            ref_bytes = &.{};
            return bad("'ref_audio' is not a decodable WAV");
        };
        gpa.free(decoded.pcm);
        const ref_text = body.str("ref_text") orelse
            return bad("'ref_audio' needs its transcript in 'ref_text'");
        if (ref_text.len == 0) return bad("'ref_text' is empty");
    }

    const seed = body.whole("seed") catch
        return bad("'seed' must be a whole number");
    const max_frames = body.whole("max_frames") catch
        return bad("'max_frames' must be a whole number");

    const total = engine.estimateFrames(engine.speakUnits(text), if (max_frames) |m| @intCast(@min(m, @as(u64, std.math.maxInt(u32)))) else null);

    var sink = Sink{
        .p = job.progressOf(progress),
        .spp = eng.eng.samples_per_frame,
        .total_frames = total,
    };
    defer sink.buf.deinit(gpa);

    const er = engine.Request{
        .text = text,
        .voice = voice,
        .instruction = instruction,
        .ref_audio = if (ref_audio != null) ref_bytes else null,
        .ref_text = if (ref_audio != null) body.str("ref_text") else null,
        .max_frames = if (max_frames) |m| @intCast(@min(m, @as(u64, std.math.maxInt(u32)))) else null,
        .seed = seed,
    };
    const st = engine.StreamSpec{
        .emit = &sinkEmit,
        .emit_user = &sink,
        .chunk_frames = 12,
        .cancel = &sinkCancel,
        .cancel_user = &sink,
    };
    const syn = eng.eng.synthesizeStream(gpa, er, st) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        error.EmptyText => bad("'input' has nothing speakable"),
        error.CloneMissingRefText => bad("'ref_audio' needs its transcript in 'ref_text'"),
        error.UnknownVoice => bad("'voice' must be \"S0\"..\"S9\""),
        error.UnsupportedRefRate => bad("'ref_audio' must be mono PCM at 16 or 24 kHz"),
        error.PromptTooLongForFrameBudget => bad("the prompt needs more frames than the budget allows; raise 'max_frames'"),
        error.TokenBeyondTeVocab => bad("the prompt holds a token the text encoder cannot represent"),
        else => {
            log.err("[breeze] synthesis failed: {s}\n", .{@errorName(err)});
            return job.failure(err, "breeze_tts synthesis");
        },
    };
    // Stream mode: Synthesis.pcm is empty and must not be freed; the frame table is ours.
    defer if (syn.frames.len > 0) gpa.free(@constCast(syn.frames));

    if (sink.failed) return job.failure(error.OutOfMemory, "breeze_tts output buffer");
    if (sink.buf.items.len == 0) return job.failure(error.NoFramesEmitted, "breeze_tts synthesis");

    const bytes = wav.encodePcm16Mono(gpa, sink.buf.items, eng.eng.sampleRate()) catch
        return job.failure(error.OutOfMemory, "breeze_tts wav");
    eng.setOut(bytes);
    sink.p.emit("breeze", sink.total_frames, sink.total_frames);
    return .ok;
}

fn encode(e: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const eng: *Engine = @ptrCast(@alignCast(e));
    if (eng.out.len == 0) return Verdict.refuse(.failed, "no result: run a job first");
    out.* = abi.Str.of(eng.out);
    return .ok;
}

/// The engine pins its widths (hidden 2048/1152/1024, the 262158-row text-encoder table, the 16×2051
/// audio table, the real Mimi decoder), so a loadable fixture is ~2.5 GB and a CPU-stream job peaks
/// near 14 GiB; the suite skips on smaller machines. Layer counts are pinned too (28/26/12 — they
/// index KV arrays), but the MLP intermediate widths are tensor-driven, so the fixture ships 512-wide
/// MLPs. Config-driven dims are a possible later polish.
pub const test_min_ram_gib = 24;

/// A generated Breeze-TTS-2 at the published widths with random weights: the full 262158-row text
/// encoder embedding table, the pinned 28/26/12 transformer layers with 512-wide MLPs, a byte-level
/// tokenizer, and the full-size Mimi codec decoder (the encoder loads lazily for clone requests,
/// which the conformance body never sends). "hi" budgets 9 frames; lm_head's tall negative EOS
/// column on dim 0 keeps frame 0 from sampling EOS (that column's sign is deterministic for this
/// fixture), so the job always synthesizes at least one frame. Everything a seeded run produces is
/// reproducible, which is all the suite asks of it.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    const bh: u64 = 2048; // backbone hidden
    const th: u64 = 1152; // text encoder hidden
    const dh: u64 = 1024; // depth decoder hidden
    const inter: u64 = 512; // fixture MLP intermediate (the engine reads it from the tensors)

    var t = T.init(a, 7);
    try t.rand("text_encoder.embed_tokens.weight", .{}, &.{ engine.TE_VOCAB, th }, .BF16, 0.02);
    try t.fill("text_encoder.norm.weight", .{}, &.{th}, .BF16, 1);
    inline for (.{
        .{ "self_attn.q_proj.weight", engine.TE_HEADS * engine.TE_HEAD_DIM, th, @as(f32, 0.02) },
        .{ "self_attn.k_proj.weight", engine.TE_KV * engine.TE_HEAD_DIM, th, @as(f32, 0.02) },
        .{ "self_attn.v_proj.weight", engine.TE_KV * engine.TE_HEAD_DIM, th, @as(f32, 0.02) },
        .{ "self_attn.o_proj.weight", th, engine.TE_HEADS * engine.TE_HEAD_DIM, @as(f32, 0.02) },
        .{ "mlp.gate_proj.weight", inter, th, @as(f32, 0.05) },
        .{ "mlp.up_proj.weight", inter, th, @as(f32, 0.05) },
        .{ "mlp.down_proj.weight", th, inter, @as(f32, 0.05) },
    }) |w| for (0..engine.TE_LAYERS) |i|
        try t.rand("{s}" ++ w[0], .{te_layer(a, i)}, &.{ w[1], w[2] }, .BF16, w[3]);
    for (0..engine.TE_LAYERS) |i| {
        const p = te_layer(a, i);
        try t.fill("{s}self_attn.q_norm.weight", .{p}, &.{engine.TE_HEAD_DIM}, .BF16, 0); // loaded as +1
        try t.fill("{s}self_attn.k_norm.weight", .{p}, &.{engine.TE_HEAD_DIM}, .BF16, 0);
        inline for (.{ "pre_self_attn", "post_self_attn", "pre_feedforward", "post_feedforward" }) |ln|
            try t.fill("{s}" ++ ln ++ "_layernorm.weight", .{p}, &.{th}, .BF16, 1);
    }
    try t.rand("text_encoder_proj.weight", .{}, &.{ bh, th }, .BF16, 0.05);
    inline for (.{
        .{ "self_attn.q_proj.weight", bh, bh, @as(f32, 0.02) },
        .{ "self_attn.k_proj.weight", engine.B_KV_DIM, bh, @as(f32, 0.02) },
        .{ "self_attn.v_proj.weight", engine.B_KV_DIM, bh, @as(f32, 0.02) },
        .{ "self_attn.o_proj.weight", bh, bh, @as(f32, 0.02) },
        .{ "mlp.gate_proj.weight", inter, bh, @as(f32, 0.05) },
        .{ "mlp.up_proj.weight", inter, bh, @as(f32, 0.05) },
        .{ "mlp.down_proj.weight", bh, inter, @as(f32, 0.05) },
    }) |w| for (0..engine.B_LAYERS) |i|
        try t.rand("{s}" ++ w[0], .{bb_layer(a, i)}, &.{ w[1], w[2] }, .BF16, w[3]);
    for (0..engine.B_LAYERS) |i| {
        const p = bb_layer(a, i);
        try t.fill("{s}self_attn.q_norm.weight", .{p}, &.{128}, .BF16, 0); // loaded as +1
        try t.fill("{s}self_attn.k_norm.weight", .{p}, &.{128}, .BF16, 0);
        try t.fill("{s}input_layernorm.weight", .{p}, &.{bh}, .BF16, 1);
        try t.fill("{s}post_attention_layernorm.weight", .{p}, &.{bh}, .BF16, 1);
    }
    try t.fill("backbone_model.norm.weight", .{}, &.{bh}, .BF16, 1);
    try t.rand("depth_decoder.model.embed_tokens.weight", .{}, &.{ engine.NUM_CODEBOOKS * engine.CB_STRIDE, bh }, .BF16, 0.3);
    try t.rand("depth_decoder.model.inputs_embeds_projector.weight", .{}, &.{ dh, bh }, .BF16, 0.05);
    try t.fill("depth_decoder.model.norm.weight", .{}, &.{dh}, .BF16, 1);
    inline for (.{
        .{ "self_attn.q_proj.weight", dh, dh, @as(f32, 0.02) },
        .{ "self_attn.k_proj.weight", 256, dh, @as(f32, 0.02) },
        .{ "self_attn.v_proj.weight", 256, dh, @as(f32, 0.02) },
        .{ "self_attn.o_proj.weight", dh, dh, @as(f32, 0.02) },
        .{ "mlp.gate_proj.weight", inter, dh, @as(f32, 0.05) },
        .{ "mlp.up_proj.weight", inter, dh, @as(f32, 0.05) },
        .{ "mlp.down_proj.weight", dh, inter, @as(f32, 0.05) },
    }) |w| for (0..engine.D_LAYERS) |i|
        try t.rand("{s}" ++ w[0], .{dd_layer(a, i)}, &.{ w[1], w[2] }, .BF16, w[3]);
    for (0..engine.D_LAYERS) |i| {
        const p = dd_layer(a, i);
        try t.fill("{s}input_layernorm.weight", .{p}, &.{dh}, .BF16, 1);
        try t.fill("{s}post_attention_layernorm.weight", .{p}, &.{dh}, .BF16, 1);
    }
    // The depth head reads rows [o][v] of a flat [15, 1024, 2051] table.
    try t.rand("depth_decoder.codebooks_head.weight", .{}, &.{ engine.NUM_CODEBOOKS - 1, dh, engine.CB_VOCAB }, .BF16, 0.02);
    // Noise everywhere, plus a tall negative column on the EOS row: frame 0 cannot sample EOS
    // (every run synthesizes at least one frame); a later EOS is as deterministic as anything else.
    {
        const v = try a.alloc(f32, engine.CB_LM * bh);
        var prng = std.Random.DefaultPrng.init(99);
        for (v) |*x| x.* = prng.random().floatNorm(f32) * 0.02;
        // EOS must lose frame 0 for THIS fixture's hidden state (whose sign at dim 0 is what the
        // run deterministically produces): one tall negative column, so logit_eos = -1e6·h[0].
        @memset(v[engine.CB_EOS * bh ..], 0);
        v[engine.CB_EOS * bh] = -1e6;
        try t.values("lm_head.weight", .{}, &.{ engine.CB_LM, bh }, .BF16, v);
    }

    // The full Mimi codec decoder at published dims (latent 1024, codebook 512/RVQ 256, decoder
    // 1536→96 over four upsample stages, an 8-layer pre-transformer at hidden 512).
    var c = T.init(a, 8);
    const lat: u64 = 1024;
    const cbd: u64 = 512;
    const rd: u64 = 256; // RVQ codebook dim
    const ph: u64 = 512; // pre-transformer hidden
    const dd: u64 = 1536;
    inline for (.{ "rvq_first", "rvq_rest" }, .{ 1, engine.NUM_CODEBOOKS - 1 }) |q, n| {
        for (0..n) |i| {
            try c.rand("decoder.quantizer." ++ q ++ ".vq.layers.{d}._codebook.embedding_sum", .{i}, &.{ 2048, rd }, .F32, 1);
            try c.fill("decoder.quantizer." ++ q ++ ".vq.layers.{d}._codebook.cluster_usage", .{i}, &.{2048}, .F32, 1);
        }
        try c.rand("decoder.quantizer." ++ q ++ ".output_proj.weight", .{}, &.{ cbd, rd, 1 }, .F32, 0.3);
    }
    try c.rand("decoder.pre_conv.conv.weight", .{}, &.{ lat, cbd, 3 }, .F32, 0.2);
    try c.rand("decoder.pre_conv.conv.bias", .{}, &.{lat}, .F32, 0.1);
    try c.rand("decoder.pre_transformer.input_proj.weight", .{}, &.{ ph, lat }, .F32, 0.2);
    try c.rand("decoder.pre_transformer.input_proj.bias", .{}, &.{ph}, .F32, 0.1);
    try c.rand("decoder.pre_transformer.output_proj.weight", .{}, &.{ lat, ph }, .F32, 0.2);
    try c.rand("decoder.pre_transformer.output_proj.bias", .{}, &.{lat}, .F32, 0.1);
    try c.fill("decoder.pre_transformer.norm.weight", .{}, &.{ph}, .F32, 1);
    for (0..8) |i| {
        const p = "decoder.pre_transformer.layers.{d}.";
        try c.fill(p ++ "input_layernorm.weight", .{i}, &.{ph}, .F32, 1);
        try c.fill(p ++ "post_attention_layernorm.weight", .{i}, &.{ph}, .F32, 1);
        inline for (.{ "q", "k", "v" }) |n| try c.rand(p ++ "self_attn." ++ n ++ "_proj.weight", .{i}, &.{ 1024, ph }, .F32, 0.2);
        try c.rand(p ++ "self_attn.o_proj.weight", .{i}, &.{ ph, 1024 }, .F32, 0.2);
        try c.fill(p ++ "self_attn_layer_scale.scale", .{i}, &.{ph}, .F32, 0.01);
        try c.fill(p ++ "mlp_layer_scale.scale", .{i}, &.{ph}, .F32, 0.01);
        try c.rand(p ++ "mlp.gate_proj.weight", .{i}, &.{ 1024, ph }, .F32, 0.2);
        try c.rand(p ++ "mlp.up_proj.weight", .{i}, &.{ 1024, ph }, .F32, 0.2);
        try c.rand(p ++ "mlp.down_proj.weight", .{i}, &.{ ph, 1024 }, .F32, 0.2);
    }
    for (0..2) |i| {
        try c.rand("decoder.upsample.{d}.0.conv.weight", .{i}, &.{ lat, lat, 2 }, .F32, 0.2);
        try c.rand("decoder.upsample.{d}.0.conv.bias", .{i}, &.{lat}, .F32, 0.1);
        try c.rand("decoder.upsample.{d}.1.dwconv.conv.weight", .{i}, &.{ lat, 1, 7 }, .F32, 0.2);
        try c.rand("decoder.upsample.{d}.1.dwconv.conv.bias", .{i}, &.{lat}, .F32, 0.1);
        try c.fill("decoder.upsample.{d}.1.norm.weight", .{i}, &.{lat}, .F32, 1);
        try c.fill("decoder.upsample.{d}.1.norm.bias", .{i}, &.{lat}, .F32, 0);
        try c.rand("decoder.upsample.{d}.1.pwconv1.weight", .{i}, &.{ 4 * lat, lat }, .F32, 0.2);
        try c.rand("decoder.upsample.{d}.1.pwconv1.bias", .{i}, &.{ 4 * lat }, .F32, 0.1);
        try c.rand("decoder.upsample.{d}.1.pwconv2.weight", .{i}, &.{ lat, 4 * lat }, .F32, 0.2);
        try c.rand("decoder.upsample.{d}.1.pwconv2.bias", .{i}, &.{lat}, .F32, 0.1);
        try c.fill("decoder.upsample.{d}.1.gamma", .{i}, &.{lat}, .F32, 0.1);
    }
    try c.rand("decoder.decoder.0.conv.weight", .{}, &.{ dd, lat, 7 }, .F32, 0.2);
    try c.rand("decoder.decoder.0.conv.bias", .{}, &.{dd}, .F32, 0.1);
    const rates = [_]u64{ 8, 5, 4, 3 };
    for (rates, 1..) |r, bi| {
        const in: u64 = dd >> @intCast(bi - 1);
        const out: u64 = dd >> @intCast(bi);
        try snake(&c, "decoder.decoder.{d}.block.0", .{bi}, in);
        try c.rand("decoder.decoder.{d}.block.1.conv.weight", .{bi}, &.{ in, out, 2 * r }, .F32, 0.2);
        try c.rand("decoder.decoder.{d}.block.1.conv.bias", .{bi}, &.{out}, .F32, 0.1);
        for (2..5) |u| {
            try snake(&c, "decoder.decoder.{d}.block.{d}.act1", .{ bi, u }, out);
            try c.rand("decoder.decoder.{d}.block.{d}.conv1.conv.weight", .{ bi, u }, &.{ out, out, 7 }, .F32, 0.2);
            try c.rand("decoder.decoder.{d}.block.{d}.conv1.conv.bias", .{ bi, u }, &.{out}, .F32, 0.1);
            try snake(&c, "decoder.decoder.{d}.block.{d}.act2", .{ bi, u }, out);
            try c.rand("decoder.decoder.{d}.block.{d}.conv2.conv.weight", .{ bi, u }, &.{ out, out, 1 }, .F32, 0.2);
            try c.rand("decoder.decoder.{d}.block.{d}.conv2.conv.bias", .{ bi, u }, &.{out}, .F32, 0.1);
        }
    }
    try snake(&c, "decoder.decoder.5", .{}, dd >> 4);
    try c.rand("decoder.decoder.6.conv.weight", .{}, &.{ 1, dd >> 4, 7 }, .F32, 0.2);
    try c.rand("decoder.decoder.6.conv.bias", .{}, &.{1}, .F32, 0.1);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"breeze_tts"}
            },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{}) },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
            .{ .name = "audio_tokenizer/model.safetensors", .bytes = try c.bytes() },
        }),
        // The conformance body: two units budget ~5 frames, which is the whole job.
        .body = "{\"input\":\"hi\"}",
    };
}

/// Layer-path prefixes for the three pinned stacks (the fixture prints keys per layer index).
fn te_layer(a: std.mem.Allocator, i: usize) []const u8 {
    return std.fmt.allocPrint(a, "text_encoder.layers.{d}.", .{i}) catch @panic("oom");
}
fn bb_layer(a: std.mem.Allocator, i: usize) []const u8 {
    return std.fmt.allocPrint(a, "backbone_model.layers.{d}.", .{i}) catch @panic("oom");
}
fn dd_layer(a: std.mem.Allocator, i: usize) []const u8 {
    return std.fmt.allocPrint(a, "depth_decoder.model.layers.{d}.", .{i}) catch @panic("oom");
}

/// Snake1 activation tables (alpha/beta of size `ch`) for a decoder residual block.
fn snake(c: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, ch: u64) !void {
    try c.rand(fmt ++ ".alpha", args, &.{ch}, .F32, 0.1);
    try c.rand(fmt ++ ".beta", args, &.{ch}, .F32, 0.1);
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());

test "breeze_tts plugin: a fixture job is a deterministic WAV; misuse is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    log.enableStderr(); // the suite's gate is off in tests; we want the failure line
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const run1 = try pack.run(pack.fx.body);
    if (run1 == .refused) std.debug.print("breeze fixture job refused: {s}\n", .{run1.refused.why.slice()});
    const first = try arena.allocator().dupe(u8, run1.ok);
    try testing.expectEqualStrings("RIFF", first[0..4]);
    try testing.expect(first.len > 44);
    try testing.expectEqualSlices(u8, first, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("this route needs text in 'input'", (try pack.run("{}")).refused.why.slice());
    // Without the host's message buffer, refuseFmt falls back to its short why; the per-field
    // text ("'max_new_tokens' is not supported by breeze_tts") is what the HTTP path delivers.
    try testing.expectEqualStrings("breeze_tts does not know this field", (try pack.run("{\"input\":\"hi\",\"max_new_tokens\":5}")).refused.why.slice());
}
