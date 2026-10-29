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
        else => job.failure(err, "breeze_tts synthesis"),
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

/// #857 follow-up: a tiny converted breeze pack for the conformance suite (the convert script can emit one
/// with a 2-layer backbone); until then the suite reports the fixture error rather than skipping.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    _ = a;
    return error.MissingFixture;
}
