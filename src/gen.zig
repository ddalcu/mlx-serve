//! Unified native media-generation engines (image / audio / video), hosted by
//! the ONE main `mlx-serve` server instead of three standalone serve loops.
//!
//! Design: the slots on `LoadedModel` are named by MODALITY — `image_engine`,
//! `audio_engine`, `video_engine` — not by the current implementation (FLUX /
//! Qwen3-TTS / LTX-Video). The wrapper structs here own whatever sub-models the
//! current backend needs; swapping FLUX for another image model later touches
//! only `ImageEngine` internals, never the registry/server plumbing.
//!
//! Threading: every method here that touches mlx (load + generate) runs on the
//! scheduler's INFERENCE thread (the sole mlx caller — even array frees go
//! there). The HTTP handler bodies (`handleImage`/`handleAudio`/`handleVideo`)
//! also run on that thread, posted as a job via `Scheduler.runGeneration`, so
//! SSE writes to the parked connection are single-writer-safe.

const std = @import("std");
const mlx = @import("mlx.zig");
const flux = @import("flux.zig");
const krea = @import("krea.zig");
const tts = @import("tts.zig");
const ltx = @import("ltx_video.zig");
const tok_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const chat_mod = @import("chat.zig");
const log = @import("log.zig");
const sse = @import("gen_sse.zig");
const server_mod = @import("server.zig");

const Conn = server_mod.Conn;

/// The three media-generation modalities. Detected from `config.json`'s
/// `model_type` and carried on the load path so the registry installs the
/// right engine slot and the server dispatches the right endpoint.
pub const Modality = enum {
    image,
    audio,
    video,

    pub fn capability(self: Modality) []const u8 {
        return switch (self) {
            .image => "image",
            .audio => "audio",
            .video => "video",
        };
    }

    /// Static, borrowed-static `ModelConfig.model_type` marker for each
    /// modality. Stable string literals (never freed) — `ModelConfig`
    /// treats `model_type` as borrowed-static, so a heap dupe is wrong here.
    pub fn modelType(self: Modality) []const u8 {
        return switch (self) {
            .image => "flux2",
            .audio => "qwen3_tts",
            .video => "AudioVideo",
        };
    }
};

/// Classify a `model_type` string into a media modality, or null for a
/// regular LM/embedding arch. Pure — the load arms dispatch on this off the
/// (stub) config's `model_type`, so it must accept the markers from
/// `Modality.modelType` AND the raw config strings discovery peeks
/// ("flux2-klein-4b", "qwen3_tts", "AudioVideo").
pub fn modalityFromType(model_type: []const u8) ?Modality {
    if (std.mem.startsWith(u8, model_type, "flux2")) return .image;
    if (std.mem.startsWith(u8, model_type, "krea")) return .image;
    if (std.mem.eql(u8, model_type, "qwen3_tts")) return .audio;
    if (std.mem.eql(u8, model_type, "AudioVideo")) return .video;
    return null;
}

/// Peek `model_dir/config.json` for its `model_type` string (owned dupe, caller
/// frees) or null on any read/parse error. Cheap — used both to route to a media
/// modality and to pick the image backend (FLUX vs Krea).
pub fn peekModelType(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    const path = std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir}) catch return null;
    defer allocator.free(path);
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return null;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const mt = parsed.value.object.get("model_type") orelse return null;
    if (mt != .string) return null;
    return allocator.dupe(u8, mt.string) catch null;
}

/// Classify a model dir into a media modality (reads its `model_type`), or null
/// for a regular LM/embedding arch. The video (LTX "AudioVideo") branch
/// additionally requires `connector.safetensors` so a generic "AudioVideo"
/// config without the LTX bundle isn't misrouted.
pub fn detectModality(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?Modality {
    const mt = peekModelType(io, allocator, model_dir) orelse return null;
    defer allocator.free(mt);
    const modality = modalityFromType(mt) orelse return null;
    if (modality == .video) {
        // Require the connector — distinguishes the LTX bundle from any other
        // "AudioVideo" config and ensures the text path can load.
        const conn_path = std.fmt.allocPrint(allocator, "{s}/connector.safetensors", .{model_dir}) catch return null;
        defer allocator.free(conn_path);
        const cf = std.Io.Dir.openFileAbsolute(io, conn_path, .{}) catch return null;
        cf.close(io);
    }
    return modality;
}

// ════════════════════════════════════════════════════════════════════════
// Engine wrappers — own the backend sub-models. Allocated on the heap so the
// `?*Engine` slot on `LoadedModel` is a stable pointer (mirrors `ds4_engine`).
// load() + every generate() run on the inference thread.
// ════════════════════════════════════════════════════════════════════════

const PAD_TOKEN_FLUX: i32 = 151643; // Qwen2/3 pad token
const FLUX_SEQ_LEN: usize = 512; // mflux Qwen3 tokenizer max_length

/// FLUX.2 image backend internals (the original `ImageEngine` body verbatim).
/// Holds the three sub-models + tokenizer; owned by the `ImageBackend` union.
const FluxImpl = struct {
    s: mlx.mlx_stream,
    te: flux.TextEncoder,
    dit: flux.Dit,
    vae: flux.Vae,
    tok: tok_mod.Tokenizer,

    fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !FluxImpl {
        var self: FluxImpl = undefined;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.te = try flux.loadTextEncoder(io, allocator, self.s, model_dir);
        errdefer self.te.deinit();
        self.dit = try flux.loadDit(io, allocator, self.s, model_dir);
        errdefer self.dit.deinit();
        self.vae = try flux.loadVae(io, allocator, self.s, model_dir);
        errdefer self.vae.deinit();
        // Tokenizer lives in the `tokenizer/` subdir for FLUX.2.
        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        log.info("[image] FLUX models + tokenizer ready\n", .{});
        return self;
    }

    fn deinit(self: *FluxImpl) void {
        self.te.deinit();
        self.dit.deinit();
        self.vae.deinit();
        self.tok.deinit();
    }

    /// Tokenize the prompt (Qwen3 chat template) and run the FLUX pipeline →
    /// PNG bytes (caller frees).
    fn generatePng(self: *FluxImpl, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, progress: ?sse.Progress) ![]u8 {
        // mflux Qwen3 chat template (enable_thinking=False adds an empty <think> block).
        const templated = try std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", .{prompt});
        defer allocator.free(templated);

        const enc = try self.tok.encode(allocator, templated);
        defer allocator.free(enc);

        var ids = try allocator.alloc(i32, FLUX_SEQ_LEN);
        defer allocator.free(ids);
        var mask = try allocator.alloc(i32, FLUX_SEQ_LEN);
        defer allocator.free(mask);
        const real = @min(enc.len, FLUX_SEQ_LEN);
        for (0..FLUX_SEQ_LEN) |i| {
            if (i < real) {
                ids[i] = @intCast(enc[i]);
                mask[i] = 1;
            } else {
                ids[i] = PAD_TOKEN_FLUX;
                mask[i] = 0;
            }
        }

        const img = try flux.generate(&self.te, &self.dit, &self.vae, ids, mask, seed, steps, height, width, progress);
        defer _ = mlx.mlx_array_free(img);
        return krea.imageToPng(allocator, img, self.s);
    }
};

/// The image modality dispatches to one backend architecture. FLUX today, Krea
/// now; SD3/Qwen-Image later = one more arm + one impl file. This is the
/// established convention — audio/video keep a single backend until they gain a
/// second arch, at which point the same union pattern applies.
const ImageBackend = union(enum) {
    flux: FluxImpl,
    krea: *krea.Engine,
};

/// Image modality engine. The slot on `LoadedModel` stays modality-named; the
/// internals are swappable per architecture (`ImageBackend`).
pub const ImageEngine = struct {
    allocator: std.mem.Allocator,
    backend: ImageBackend,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*ImageEngine {
        const self = try allocator.create(ImageEngine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        // Re-peek the arch to pick the backend (detectModality already proved
        // config.json parses). `krea*` → Krea; everything else → FLUX.
        const is_krea = blk: {
            const mt = peekModelType(io, allocator, model_dir) orelse break :blk false;
            defer allocator.free(mt);
            break :blk std.mem.startsWith(u8, mt, "krea");
        };
        if (is_krea) {
            self.backend = .{ .krea = try krea.Engine.load(io, allocator, model_dir) };
        } else {
            self.backend = .{ .flux = try FluxImpl.load(io, allocator, model_dir) };
        }
        return self;
    }

    pub fn deinit(self: *ImageEngine) void {
        switch (self.backend) {
            .flux => |*f| f.deinit(),
            .krea => |k| k.deinit(),
        }
        self.allocator.destroy(self);
    }

    pub fn generatePng(self: *ImageEngine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, progress: ?sse.Progress) ![]u8 {
        return switch (self.backend) {
            .flux => |*f| f.generatePng(allocator, prompt, width, height, seed, steps, progress),
            .krea => |k| k.generatePng(allocator, prompt, width, height, seed, steps, progress),
        };
    }

    /// Resolve a requested WxH per backend. FLUX has a fixed 1024² latent grid;
    /// Krea accepts any multiple of 16 in [256, 2048].
    pub fn normalizeSize(self: *const ImageEngine, req_w: u32, req_h: u32) struct { w: u32, h: u32 } {
        return switch (self.backend) {
            .flux => .{ .w = 1024, .h = 1024 },
            .krea => .{ .w = clampKreaDim(req_w), .h = clampKreaDim(req_h) },
        };
    }
};

/// Round a requested dimension to a multiple of 16 in [256, 2048] (Krea's
/// VAE ×8 + DiT patch ×2 alignment).
fn clampKreaDim(v: u32) u32 {
    const rounded = ((v + 15) / 16) * 16;
    return std.math.clamp(rounded, 256, 2048);
}

/// Audio backend (currently Qwen3-TTS). The `tts.Synthesizer` already bundles
/// talker + codec + tokenizer, so this is a thin owner.
pub const AudioEngine = struct {
    allocator: std.mem.Allocator,
    synth: tts.Synthesizer,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*AudioEngine {
        const self = try allocator.create(AudioEngine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        const s = mlx.mlx_default_gpu_stream_new();
        self.synth = try tts.Synthesizer.load(io, allocator, s, model_dir);
        log.info("[audio] TTS synthesizer ready (sample_rate={d})\n", .{self.synth.model.cfg.sample_rate});
        return self;
    }

    pub fn deinit(self: *AudioEngine) void {
        self.synth.deinit();
        self.allocator.destroy(self);
    }
};

const LTX_PAD_LEN: usize = 256; // gemma left-pad length
const LTX_PAD_ID: i32 = 0; // gemma <pad>
const LTX_GEMMA_BOS: i32 = 2; // <bos>

// Trimmed DEFAULT_NEGATIVE_PROMPT — left-pad truncates from the left anyway.
const LTX_NEGATIVE_PROMPT =
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, " ++
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted " ++
    "proportions, unnatural skin tones, deformed facial features, extra limbs, disfigured " ++
    "hands, inconsistent perspective, camera shake, color banding, cartoonish rendering, " ++
    "3D CGI look, unrealistic materials, uncanny valley effect, exaggerated expressions";

/// Video backend (currently LTX-Video 2.3). Holds the three components + the
/// resolved Gemma text-encoder dir + its tokenizer. Components load on the CPU
/// stream; the forward graph runs on the GPU stream.
pub const VideoEngine = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    transformer: ltx.Component,
    connector: ltx.Component,
    vae: ltx.Component,
    tok: tok_mod.Tokenizer,
    gemma_dir: []u8,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*VideoEngine {
        const self = try allocator.create(VideoEngine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;

        self.gemma_dir = try resolveGemmaDir(io, allocator);
        errdefer allocator.free(self.gemma_dir);
        log.info("[video] gemma text encoder: {s}\n", .{self.gemma_dir});

        const cpu_s = mlx.mlx_default_cpu_stream_new();
        self.s = mlx.mlx_default_gpu_stream_new();

        const tp = try std.fmt.allocPrintSentinel(allocator, "{s}/transformer-dev.safetensors", .{model_dir}, 0);
        defer allocator.free(tp);
        self.transformer = try ltx.loadComponent(allocator, tp, cpu_s);
        errdefer self.transformer.deinit();
        const cp = try std.fmt.allocPrintSentinel(allocator, "{s}/connector.safetensors", .{model_dir}, 0);
        defer allocator.free(cp);
        self.connector = try ltx.loadComponent(allocator, cp, cpu_s);
        errdefer self.connector.deinit();
        const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.safetensors", .{model_dir}, 0);
        defer allocator.free(vp);
        self.vae = try ltx.loadComponent(allocator, vp, cpu_s);
        errdefer self.vae.deinit();
        var it = self.vae.map.iterator();
        while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*); // VAE conv graph wants materialized weights

        self.tok = try tok_mod.loadTokenizerAny(io, allocator, self.gemma_dir);
        log.info("[video] LTX components + tokenizer ready\n", .{});
        return self;
    }

    pub fn deinit(self: *VideoEngine) void {
        self.transformer.deinit();
        self.connector.deinit();
        self.vae.deinit();
        self.tok.deinit();
        self.allocator.free(self.gemma_dir);
        self.allocator.destroy(self);
    }
};

/// LTX's text encoder is Gemma-3-12B (4-bit). It's a normal downloadable model
/// the app pulls into `~/.mlx-serve/models` (as the LTX bundle dependency, and
/// selectable as a chat model). The repo id maps to a `<author>/<name>` dir.
const LTX_GEMMA_REPO_DIR = "mlx-community/gemma-3-12b-it-4bit";

/// Locate the Gemma-3-12B text encoder ONLY under `~/.mlx-serve/models` — the
/// single source of truth for downloaded models. No HF-cache magic: the app
/// owns downloads. `$LTX_GEMMA_DIR` stays as an explicit override (tests /
/// custom installs). A candidate is accepted only if it has a `config.json`,
/// so a partial download never gets handed back.
fn resolveGemmaDir(io: std.Io, allocator: std.mem.Allocator) ![]u8 {
    if (std.c.getenv("LTX_GEMMA_DIR")) |env| {
        const e = std.mem.span(env);
        if (e.len > 0) return allocator.dupe(u8, e);
    }
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoGemmaDir);
    // 2-level `<author>/<name>` layout (what DownloadManager writes), then a
    // flat `<name>` layout (legacy / manual placement).
    const candidates = [_][]const u8{ LTX_GEMMA_REPO_DIR, "gemma-3-12b-it-4bit" };
    for (candidates) |rel| {
        const dir = std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models/{s}", .{ home, rel }) catch continue;
        var ok = false;
        {
            const cfg = std.fmt.allocPrint(allocator, "{s}/config.json", .{dir}) catch {
                allocator.free(dir);
                continue;
            };
            defer allocator.free(cfg);
            if (std.Io.Dir.openFileAbsolute(io, cfg, .{})) |f| {
                f.close(io);
                ok = true;
            } else |_| {}
        }
        if (ok) return dir; // caller owns
        allocator.free(dir);
    }
    return error.NoGemmaDir;
}

/// Tokenize like the reference LTX gemma tokenizer: `[<bos>] + encode(text)`,
/// then LEFT-pad/truncate to LTX_PAD_LEN with LTX_PAD_ID.
fn ltxTokenizePadded(allocator: std.mem.Allocator, tokenizer: *tok_mod.Tokenizer, text: []const u8) ![]i32 {
    const enc = try tokenizer.encode(allocator, text);
    defer allocator.free(enc);
    return ltxPadWithBos(allocator, enc, LTX_GEMMA_BOS, LTX_PAD_LEN, LTX_PAD_ID);
}

/// Pure BOS-prepend + left-pad (testable without a live tokenizer).
fn ltxPadWithBos(allocator: std.mem.Allocator, enc: []const u32, bos: i32, pad_len: usize, pad_id: i32) ![]i32 {
    const has_bos = enc.len > 0 and enc[0] == @as(u32, @intCast(bos));
    const total = if (has_bos) enc.len else enc.len + 1;
    const ids = try allocator.alloc(i32, pad_len);
    const real = @min(total, pad_len);
    const pad = pad_len - real;
    for (0..pad) |i| ids[i] = pad_id;
    for (0..real) |i| {
        const idx = total - real + i;
        if (has_bos) {
            ids[pad + i] = @intCast(enc[idx]);
        } else {
            ids[pad + i] = if (idx == 0) bos else @intCast(enc[idx - 1]);
        }
    }
    return ids;
}

// ════════════════════════════════════════════════════════════════════════
// HTTP handler bodies. Called on the INFERENCE thread (via the gen job). The
// connection is parked (single-writer), so SSE writes here are safe. `lm` is
// already resolved + refcounted by the connection thread.
// ════════════════════════════════════════════════════════════════════════

/// POST /v1/images/generations — base64 PNG (or SSE progress + complete).
pub fn handleImage(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *ImageEngine) !void {
    const prompt_raw = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt'");
    const prompt = try jsonUnescape(allocator, prompt_raw);
    defer allocator.free(prompt);
    if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

    // Requested size (default 1024²); the backend resolves it (FLUX is fixed
    // 1024², Krea accepts any multiple of 16 in [256,2048]).
    var req_w: u32 = 1024;
    var req_h: u32 = 1024;
    if (extractJsonString(body, "size")) |size| {
        if (parseSize(size)) |wh| {
            req_w = wh.w;
            req_h = wh.h;
        }
    }
    const sz = engine.normalizeSize(req_w, req_h);
    const width = sz.w;
    const height = sz.h;
    if (req_w != width or req_h != height) {
        log.warn("[image] requested {d}x{d} resolved to {d}x{d} for this backend\n", .{ req_w, req_h, width, height });
    }
    const seed: u64 = extractJsonInt(body, "seed") orelse 42;
    const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse 4);

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[image] generating {d}x{d} steps={d} stream={}: {d} chars\n", .{ width, height, steps, want_stream, prompt.len });
    var sctx = sse.StreamCtx{ .conn = conn };
    const prog: ?sse.Progress = if (want_stream) sctx.progress() else null;
    if (want_stream) try conn.writeAll(sse.headers);

    const png_bytes = engine.generatePng(allocator, prompt, width, height, seed, steps, prog) catch |err| {
        log.err("[image] generation failed: {}\n", .{err});
        if (want_stream) {
            sse.sendError(conn, "generation failed");
            return;
        }
        return sendError(conn, 500, "generation failed");
    };
    defer allocator.free(png_bytes);

    const b64_len = std.base64.standard.Encoder.calcSize(png_bytes.len);
    const b64 = try allocator.alloc(u8, b64_len);
    defer allocator.free(b64);
    _ = std.base64.standard.Encoder.encode(b64, png_bytes);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, if (want_stream) "data: {\"type\":\"complete\",\"data\":[{\"b64_json\":\"" else "{\"created\":0,\"data\":[{\"b64_json\":\"");
    try out.appendSlice(allocator, b64);
    try out.appendSlice(allocator, if (want_stream) "\"}]}\n\n" else "\"}]}");
    log.info("[image] -> {d} PNG bytes ({d} b64)\n", .{ png_bytes.len, b64.len });
    if (want_stream) {
        try conn.writeAll(out.items);
        return;
    }
    return sendBytesJson(conn, allocator, out.items);
}

/// POST /v1/audio/speech — WAV bytes (or SSE progress + base64-WAV complete).
pub fn handleAudio(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *AudioEngine) !void {
    const input = extractJsonString(body, "input") orelse extractJsonString(body, "text") orelse return sendError(conn, 400, "missing 'input'");
    const text = try jsonUnescape(allocator, input);
    defer allocator.free(text);
    if (text.len == 0) return sendError(conn, 400, "empty 'input'");

    // Optional reference voice for zero-shot cloning: `ref_audio` is a base64
    // WAV (24 kHz mono, the app normalizes it). Decode → f32 samples. Ignored
    // (plain voice) when the model has no speaker encoder or the WAV is bad.
    var ref_samples: ?[]f32 = null;
    defer if (ref_samples) |r| allocator.free(r);
    if (extractJsonString(body, "ref_audio")) |raw_ref| {
        const b64 = try jsonUnescape(allocator, raw_ref); // handles \/ from Swift JSONSerialization
        defer allocator.free(b64);
        if (b64.len > 0) {
            if (base64DecodeAlloc(allocator, b64)) |wav_bytes| {
                defer allocator.free(wav_bytes);
                if (decodeWavToF32(allocator, wav_bytes)) |samples| {
                    if (engine.synth.supportsCloning()) {
                        ref_samples = samples;
                        log.info("[audio] reference voice: {d} samples → cloning\n", .{samples.len});
                    } else {
                        allocator.free(samples);
                        log.warn("[audio] model has no speaker encoder — ignoring ref_audio\n", .{});
                    }
                } else |e| log.warn("[audio] ref_audio WAV decode failed: {} — plain voice\n", .{e});
            } else |e| log.warn("[audio] ref_audio base64 decode failed: {} — plain voice\n", .{e});
        }
    }

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[audio] synthesizing {d} chars stream={} clone={}\n", .{ text.len, want_stream, ref_samples != null });
    var sctx = sse.StreamCtx{ .conn = conn };
    const prog: ?sse.Progress = if (want_stream) sctx.progress() else null;
    if (want_stream) try conn.writeAll(sse.headers);

    const wav = engine.synth.synthesizeWav(text, 2048, prog, ref_samples) catch |err| {
        log.err("[audio] synthesis failed: {}\n", .{err});
        if (want_stream) {
            sse.sendError(conn, "synthesis failed");
            return;
        }
        return sendError(conn, 500, "synthesis failed");
    };
    defer allocator.free(wav);
    log.info("[audio] -> {d} WAV bytes\n", .{wav.len});
    if (want_stream) {
        const b64_len = std.base64.standard.Encoder.calcSize(wav.len);
        const b64 = try allocator.alloc(u8, b64_len);
        defer allocator.free(b64);
        _ = std.base64.standard.Encoder.encode(b64, wav);
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(allocator);
        try out.appendSlice(allocator, "data: {\"type\":\"complete\",\"format\":\"wav\",\"data\":\"");
        try out.appendSlice(allocator, b64);
        try out.appendSlice(allocator, "\"}\n\n");
        try conn.writeAll(out.items);
        return;
    }
    return sendBytes(conn, allocator, "audio/wav", wav);
}

/// POST /v1/video/generations — base64 RGB8 frames (or SSE progress + complete).
pub fn handleVideo(io: std.Io, allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *VideoEngine) !void {
    const prompt_raw = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt'");
    const prompt = try jsonUnescape(allocator, prompt_raw);
    defer allocator.free(prompt);
    if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

    const num_frames: u32 = @intCast(extractJsonInt(body, "num_frames") orelse 9);
    const height: u32 = @intCast(extractJsonInt(body, "height") orelse 256);
    const width: u32 = @intCast(extractJsonInt(body, "width") orelse 384);
    const seed: u64 = extractJsonInt(body, "seed") orelse 42;
    const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse 30);
    const frame_rate: f32 = 24.0;

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[video] generating {d}f {d}x{d} steps={d} stream={}: {d} chars\n", .{ num_frames, height, width, steps, want_stream, prompt.len });

    const pos_ids = try ltxTokenizePadded(allocator, &engine.tok, prompt);
    defer allocator.free(pos_ids);
    const neg_ids = try ltxTokenizePadded(allocator, &engine.tok, LTX_NEGATIVE_PROMPT);
    defer allocator.free(neg_ids);

    var sctx = sse.StreamCtx{ .conn = conn };
    const prog: ?ltx.Progress = if (want_stream) sctx.progress() else null;
    if (want_stream) try conn.writeAll(sse.headers);

    var frames = ltx.generateVideoFrames(io, allocator, .{}, &engine.transformer, &engine.connector, &engine.vae, engine.gemma_dir, pos_ids, neg_ids, LTX_PAD_ID, num_frames, height, width, frame_rate, steps, seed, 3.0, 7.0, 0.7, prog, engine.s) catch |err| {
        log.err("[video] generation failed: {}\n", .{err});
        if (want_stream) {
            conn.writeAll("data: {\"type\":\"error\",\"message\":\"generation failed\"}\n\n") catch {};
            return;
        }
        return sendError(conn, 500, "generation failed");
    };
    defer frames.deinit(allocator);
    log.info("[video] -> {d}f {d}x{d} ({d} rgb bytes)\n", .{ frames.frames, frames.height, frames.width, frames.rgb.len });

    const b64_len = std.base64.standard.Encoder.calcSize(frames.rgb.len);
    const b64 = try allocator.alloc(u8, b64_len);
    defer allocator.free(b64);
    _ = std.base64.standard.Encoder.encode(b64, frames.rgb);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    const prefix = if (want_stream) "data: {\"type\":\"complete\"," else "{\"created\":0,";
    const head = try std.fmt.allocPrint(allocator, "{s}\"frames\":{d},\"height\":{d},\"width\":{d},\"fps\":{d},\"format\":\"rgb8\",\"data\":\"", .{ prefix, frames.frames, frames.height, frames.width, @as(u32, @intFromFloat(frame_rate)) });
    defer allocator.free(head);
    try out.appendSlice(allocator, head);
    try out.appendSlice(allocator, b64);
    try out.appendSlice(allocator, if (want_stream) "\"}\n\n" else "\"}");
    if (want_stream) {
        try conn.writeAll(out.items);
        return;
    }
    return sendBytesJson(conn, allocator, out.items);
}

// ════════════════════════════════════════════════════════════════════════
// Stub CPU state for a media model. The gen path bypasses the transformer, so
// `config`/`tokenizer`/`chat_config` on the LoadedModel are minimal stubs that
// only keep server-side reads of `lm.config.?` / `lm.chat_config.?` from
// crashing. Mirrors `runDs4Serve`'s stub construction. Used by BOTH the
// startup gen-primary path and the cold-load (`/v1/load-model`) path.
// ════════════════════════════════════════════════════════════════════════

pub const StubCpuState = struct {
    config: *model_mod.ModelConfig,
    tok: *tok_mod.Tokenizer,
    chat_config: *chat_mod.ChatConfig,
};

/// Build heap-allocated stub config/tokenizer/chat_config for `modality`.
/// Ownership transfers to the LoadedModel on a successful load (mirrors the
/// ds4/llama stubs). `freeStubCpuState` frees them on the failure path.
pub fn buildStubCpuState(allocator: std.mem.Allocator, modality: Modality) !StubCpuState {
    const config = try allocator.create(model_mod.ModelConfig);
    errdefer allocator.destroy(config);
    config.* = model_mod.ModelConfig{
        .model_type = modality.modelType(),
        .weight_prefix = "model",
        .num_hidden_layers = 1,
        .hidden_size = 1,
        .head_dim = 1,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .max_position_embeddings = 4096,
        .is_encoder_only = false,
    };

    const tok = try allocator.create(tok_mod.Tokenizer);
    errdefer allocator.destroy(tok);
    var byte_map: [256]u21 = undefined;
    var b: usize = 0;
    while (b < 256) : (b += 1) byte_map[b] = @intCast(b);
    tok.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .tok_type = .byte_level_bpe,
        .byte_to_unicode = byte_map,
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };
    errdefer tok.deinit();

    const cc = try allocator.create(chat_mod.ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

pub fn freeStubCpuState(allocator: std.mem.Allocator, s: *StubCpuState) void {
    allocator.destroy(s.config);
    s.tok.deinit();
    allocator.destroy(s.tok);
    s.chat_config.deinit();
    allocator.destroy(s.chat_config);
}

/// Sum the safetensors footprint of a media model dir for the eviction gate.
/// Walks the top level + one level of subdirs (FLUX keeps weights in
/// transformer/, vae/, text_encoder/; LTX keeps them top-level). Returns 0 on
/// any read failure (treated as "unknown" → the registry skips the byte cap).
pub fn estimateResidentBytes(io: std.Io, model_dir: []const u8) u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    var total: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".safetensors")) {
            const st = dir.statFile(io, entry.name, .{}) catch continue;
            total += @intCast(st.size);
        } else if (entry.kind == .directory) {
            var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
            defer sub.close(io);
            var sit = sub.iterate();
            while (sit.next(io) catch null) |se| {
                if (se.kind != .file or !std.mem.endsWith(u8, se.name, ".safetensors")) continue;
                const st = sub.statFile(io, se.name, .{}) catch continue;
                total += @intCast(st.size);
            }
        }
    }
    return total;
}

// ── HTTP response helpers (self-contained; mirror the old *_server.zig) ──

fn sendBytesJson(conn: *Conn, allocator: std.mem.Allocator, json: []const u8) !void {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(allocator);
    try hdr.appendSlice(allocator, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
    var num: [20]u8 = undefined;
    const ns = std.fmt.bufPrint(&num, "{d}", .{json.len}) catch unreachable;
    try hdr.appendSlice(allocator, ns);
    try hdr.appendSlice(allocator, "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n");
    try conn.writeAllNoFlush(hdr.items);
    try conn.writeAll(json);
}

fn sendBytes(conn: *Conn, allocator: std.mem.Allocator, content_type: []const u8, payload: []const u8) !void {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(allocator);
    try hdr.appendSlice(allocator, "HTTP/1.1 200 OK\r\nContent-Type: ");
    try hdr.appendSlice(allocator, content_type);
    try hdr.appendSlice(allocator, "\r\nContent-Length: ");
    var num: [20]u8 = undefined;
    const ns = std.fmt.bufPrint(&num, "{d}", .{payload.len}) catch unreachable;
    try hdr.appendSlice(allocator, ns);
    try hdr.appendSlice(allocator, "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n");
    try conn.writeAllNoFlush(hdr.items);
    try conn.writeAll(payload);
}

fn sendError(conn: *Conn, code: u16, msg: []const u8) !void {
    var body_buf: [256]u8 = undefined;
    const body = std.fmt.bufPrint(&body_buf, "{{\"error\":{{\"message\":\"{s}\"}}}}", .{msg}) catch return;
    var hdr: [256]u8 = undefined;
    const head = std.fmt.bufPrint(&hdr, "HTTP/1.1 {d} Error\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ code, body.len }) catch return;
    try conn.writeAllNoFlush(head);
    try conn.writeAll(body);
}

// ── Minimal JSON parsing helpers (top-level keys only) ──

fn extractJsonString(body: []const u8, key: []const u8) ?[]const u8 {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    if (i >= body.len or body[i] != '"') return null;
    i += 1;
    const start = i;
    while (i < body.len) : (i += 1) {
        if (body[i] == '\\') {
            i += 1;
            continue;
        }
        if (body[i] == '"') return body[start..i];
    }
    return null;
}

/// Parse a "WxH" size string (e.g. "1024x1024", "512x768") → {w,h}, or null.
fn parseSize(size: []const u8) ?struct { w: u32, h: u32 } {
    const xi = std.mem.indexOfScalar(u8, size, 'x') orelse std.mem.indexOfScalar(u8, size, 'X') orelse return null;
    const w = std.fmt.parseInt(u32, size[0..xi], 10) catch return null;
    const h = std.fmt.parseInt(u32, size[xi + 1 ..], 10) catch return null;
    if (w == 0 or h == 0) return null;
    return .{ .w = w, .h = h };
}

fn extractJsonInt(body: []const u8, key: []const u8) ?u64 {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    const start = i;
    while (i < body.len and (std.ascii.isDigit(body[i]))) i += 1;
    if (i == start) return null;
    return std.fmt.parseInt(u64, body[start..i], 10) catch null;
}

/// Base64-decode (standard alphabet) into an owned buffer.
fn base64DecodeAlloc(allocator: std.mem.Allocator, b64: []const u8) ![]u8 {
    const dec = std.base64.standard.Decoder;
    const n = try dec.calcSizeForSlice(b64);
    const out = try allocator.alloc(u8, n);
    errdefer allocator.free(out);
    try dec.decode(out, b64);
    return out;
}

/// Decode a 16-bit PCM mono WAV → f32 samples in [-1, 1]. Scans the RIFF
/// chunks for `data` (so a non-canonical header with extra chunks still works);
/// assumes mono (the app normalizes reference audio to 24 kHz mono int16).
fn decodeWavToF32(allocator: std.mem.Allocator, wav: []const u8) ![]f32 {
    if (wav.len < 44 or !std.mem.eql(u8, wav[0..4], "RIFF") or !std.mem.eql(u8, wav[8..12], "WAVE")) return error.BadWav;
    var pos: usize = 12;
    while (pos + 8 <= wav.len) {
        const cid = wav[pos .. pos + 4];
        const csize: usize = std.mem.readInt(u32, wav[pos + 4 ..][0..4], .little);
        if (std.mem.eql(u8, cid, "data")) {
            const start = pos + 8;
            const end = @min(start + csize, wav.len);
            const n = (end - start) / 2;
            const out = try allocator.alloc(f32, n);
            for (0..n) |i| {
                const v = std.mem.readInt(i16, wav[start + i * 2 ..][0..2], .little);
                out[i] = @as(f32, @floatFromInt(v)) / 32768.0;
            }
            return out;
        }
        pos += 8 + csize + (csize & 1); // chunks are word-aligned
    }
    return error.NoDataChunk;
}

fn jsonUnescape(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < raw.len) : (i += 1) {
        if (raw[i] != '\\') {
            try out.append(allocator, raw[i]);
            continue;
        }
        i += 1;
        if (i >= raw.len) break;
        switch (raw[i]) {
            'n' => try out.append(allocator, '\n'),
            't' => try out.append(allocator, '\t'),
            'r' => try out.append(allocator, '\r'),
            '"' => try out.append(allocator, '"'),
            '\\' => try out.append(allocator, '\\'),
            '/' => try out.append(allocator, '/'),
            'u' => {
                if (i + 4 < raw.len) {
                    const cp = std.fmt.parseInt(u21, raw[i + 1 .. i + 5], 16) catch 0;
                    var bb: [4]u8 = undefined;
                    const len = std.unicode.utf8Encode(cp, &bb) catch 0;
                    try out.appendSlice(allocator, bb[0..len]);
                    i += 4;
                }
            },
            else => try out.append(allocator, raw[i]),
        }
    }
    return out.toOwnedSlice(allocator);
}

// ── Tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;

test "modalityFromType classifies the media archs + markers (incl. krea)" {
    try testing.expectEqual(Modality.image, modalityFromType("flux2-klein-4b").?);
    try testing.expectEqual(Modality.image, modalityFromType("flux2").?);
    try testing.expectEqual(Modality.image, modalityFromType("krea2_turbo").?);
    try testing.expectEqual(Modality.image, modalityFromType("krea").?);
    try testing.expectEqual(Modality.audio, modalityFromType("qwen3_tts").?);
    try testing.expectEqual(Modality.video, modalityFromType("AudioVideo").?);
    try testing.expectEqual(@as(?Modality, null), modalityFromType("gemma4"));
    try testing.expectEqual(@as(?Modality, null), modalityFromType("qwen3_5_moe"));
}

test "parseSize parses WxH and rejects garbage" {
    const a = parseSize("1024x1024").?;
    try testing.expectEqual(@as(u32, 1024), a.w);
    try testing.expectEqual(@as(u32, 1024), a.h);
    const b = parseSize("512x768").?;
    try testing.expectEqual(@as(u32, 512), b.w);
    try testing.expectEqual(@as(u32, 768), b.h);
    try testing.expectEqual(@as(?@TypeOf(a), null), parseSize("auto"));
    try testing.expectEqual(@as(?@TypeOf(a), null), parseSize("1024"));
    try testing.expectEqual(@as(?@TypeOf(a), null), parseSize("0x512"));
}

test "clampKreaDim rounds to multiples of 16 in [256,2048]" {
    try testing.expectEqual(@as(u32, 1024), clampKreaDim(1024));
    try testing.expectEqual(@as(u32, 512), clampKreaDim(500)); // 500 → 512
    try testing.expectEqual(@as(u32, 256), clampKreaDim(16)); // clamp up
    try testing.expectEqual(@as(u32, 2048), clampKreaDim(5000)); // clamp down
    try testing.expectEqual(@as(u32, 768), clampKreaDim(768));
}

// Characterization guard for the FLUX `generatePng` path through the
// `ImageEngine` backend union (covers the Part-A extraction). Env-gated on a
// FLUX model dir; in CI it skips. Asserts a non-empty PNG comes back so a broken
// delegation or backend dispatch fails loudly.
//   IMAGE_TEST_MODEL=<flux dir>  (optional IMAGE_TEST_STEPS, default 1)
test "ImageEngine FLUX generatePng produces a PNG (characterization)" {
    const model_dir = std.mem.span(std.c.getenv("IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const steps: u32 = if (std.c.getenv("IMAGE_TEST_STEPS")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1) else 1;
    var eng = try ImageEngine.load(io, a, model_dir);
    defer eng.deinit();
    const sz = eng.normalizeSize(1024, 1024);
    const pngb = try eng.generatePng(a, "a red fox in the snow", sz.w, sz.h, 42, steps, null);
    defer a.free(pngb);
    try testing.expect(pngb.len > 8);
    // PNG magic
    try testing.expectEqualSlices(u8, &[_]u8{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, pngb[0..8]);
}

test "Modality.modelType round-trips through modalityFromType" {
    for ([_]Modality{ .image, .audio, .video }) |m| {
        try testing.expectEqual(m, modalityFromType(m.modelType()).?);
    }
}

test "extractJsonInt parses seed/steps" {
    try testing.expectEqual(@as(?u64, 7), extractJsonInt("{\"seed\": 7}", "seed"));
    try testing.expectEqual(@as(?u64, 20), extractJsonInt("{\"steps\":20,\"x\":1}", "steps"));
    try testing.expectEqual(@as(?u64, null), extractJsonInt("{\"prompt\":\"hi\"}", "seed"));
}

test "extractJsonString + jsonUnescape" {
    const body = "{\"model\":\"x\",\"input\":\"Hello\\nworld\"}";
    const raw = extractJsonString(body, "input").?;
    try testing.expectEqualStrings("Hello\\nworld", raw);
    const un = try jsonUnescape(testing.allocator, raw);
    defer testing.allocator.free(un);
    try testing.expectEqualStrings("Hello\nworld", un);
}

test "ltxPadWithBos prepends gemma <bos> (off-prompt regression)" {
    const a = testing.allocator;
    const enc = [_]u32{ 236746, 2604, 37423 };
    const ids = try ltxPadWithBos(a, &enc, 2, 8, 0);
    defer a.free(ids);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0, 0, 2, 236746, 2604, 37423 }, ids);
}

test "ltxPadWithBos does not double an existing <bos>" {
    const a = testing.allocator;
    const enc = [_]u32{ 2, 236746, 2604 };
    const ids = try ltxPadWithBos(a, &enc, 2, 6, 0);
    defer a.free(ids);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0, 2, 236746, 2604 }, ids);
}

test "buildStubCpuState builds a media stub keyed by modality" {
    const a = testing.allocator;
    var stub = try buildStubCpuState(a, .image);
    defer freeStubCpuState(a, &stub);
    try testing.expectEqualStrings("flux2", stub.config.model_type);
    try testing.expect(!stub.config.is_encoder_only);
    try testing.expectEqual(modalityFromType(stub.config.model_type).?, Modality.image);
}
