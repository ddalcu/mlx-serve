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
const lora_mod = @import("lora.zig");
const nsfw = @import("nsfw.zig");
const tts = @import("tts.zig");
const ltx = @import("ltx_video.zig");
const ltx_audio = @import("ltx_audio.zig");
const wav_mod = @import("wav.zig");
const png_mod = @import("png.zig");
const tok_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const chat_mod = @import("chat.zig");
const log = @import("log.zig");
const sse = @import("gen_sse.zig");
const server_mod = @import("server.zig");
const stb = @cImport({
    @cInclude("stb_image.h");
});

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
    // Guard the openFileAbsolute assert (ReleaseFast UB on relative/empty paths).
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null;
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
        const conn_path = std.fmt.allocPrintSentinel(allocator, "{s}/connector.safetensors", .{model_dir}, 0) catch return null;
        defer allocator.free(conn_path);
        if (!fileExists(io, conn_path)) return null;
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
    vae_enc: ?flux.VaeEncoder,
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
        self.vae_enc = flux.loadVaeEncoder(io, allocator, self.s, model_dir) catch |e| blk: {
            log.warn("[image] FLUX VAE encoder load failed ({}) — image-to-image disabled\n", .{e});
            break :blk null;
        };
        errdefer if (self.vae_enc) |*e| e.deinit();
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
        if (self.vae_enc) |*e| e.deinit();
        self.tok.deinit();
    }

    /// Tokenize the prompt (Qwen3 chat template) and run the FLUX pipeline →
    /// image [1,3,H,W] f32 in [0,1] (owned mlx array; caller frees).
    fn generateImage(self: *FluxImpl, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: ImageGenOpts, progress: ?sse.Progress) !mlx.mlx_array {
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
        var fopts = flux.GenOpts{ .cond_gain = opts.cond_gain, .cond_weights = opts.cond_weights };
        var init_lat: ?mlx.mlx_array = null;
        defer if (init_lat) |l| {
            _ = mlx.mlx_array_free(l);
        };
        if (opts.edit_image) |pix| {
            // Instruction edit: clean in-context reference, full noise start.
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            init_lat = try ve.encode(pix);
            fopts.ref_latents = init_lat;
        } else if (opts.init_image) |pix| {
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            init_lat = try ve.encode(pix);
            fopts.init_latents = init_lat;
            fopts.start_step = img2imgStartStep(steps, opts.strength);
        }
        return flux.generateWithOpts(&self.te, &self.dit, &self.vae, ids, mask, seed, steps, height, width, fopts, progress);
    }

    fn generatePng(self: *FluxImpl, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: ImageGenOpts, progress: ?sse.Progress) ![]u8 {
        const img = try self.generateImage(allocator, prompt, width, height, seed, steps, opts, progress);
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

/// Per-request image-generation options shared by both backends.
pub const ImageGenOpts = struct {
    /// img2img source pixels [1,3,H,W] f32 [0,1], pre-resized to the target
    /// size (VAE-encoded by the backend).
    init_image: ?mlx.mlx_array = null,
    /// How far to renoise the source (diffusers convention: 1 = ignore it,
    /// low = small change). Only meaningful with `init_image`.
    strength: f32 = 0.6,
    /// Instruction editing (FLUX.2 only): source pixels [1,3,H,W] f32 [0,1]
    /// conditioned as CLEAN in-context reference tokens — generation starts
    /// from pure noise and attends to them (`strength` does not apply).
    edit_image: ?mlx.mlx_array = null,
    /// Conditioning rebalance: global gain × per-tapped-layer weights
    /// (FLUX: 3 taps, Krea: 12 taps).
    cond_gain: f32 = 1.0,
    cond_weights: ?[]const f32 = null,
};

/// Image modality engine. The slot on `LoadedModel` stays modality-named; the
/// internals are swappable per architecture (`ImageBackend`).
pub const ImageEngine = struct {
    allocator: std.mem.Allocator,
    backend: ImageBackend,
    // Runtime LoRA state: the File owns the adapter arrays the attached Refs
    // point at, so it must live until the next detach (clearLora).
    lora_file: ?lora_mod.File = null,
    lora_path: ?[]u8 = null,
    lora_scale: f32 = 1.0,
    lora_matched: u32 = 0,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*ImageEngine {
        const self = try allocator.create(ImageEngine);
        errdefer allocator.destroy(self);
        self.* = .{ .allocator = allocator, .backend = undefined };
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
        self.clearLora();
        switch (self.backend) {
            .flux => |*f| f.deinit(),
            .krea => |k| k.deinit(),
        }
        self.allocator.destroy(self);
    }

    fn stream(self: *ImageEngine) mlx.mlx_stream {
        return switch (self.backend) {
            .flux => |*f| f.s,
            .krea => |k| k.s,
        };
    }

    /// Number of tapped text-encoder layers `cond_weights` must cover.
    pub fn condWeightCount(self: *const ImageEngine) usize {
        return switch (self.backend) {
            .flux => 3,
            .krea => 12,
        };
    }

    /// True when the VAE encoder loaded, i.e. img2img is available.
    pub fn supportsImg2Img(self: *const ImageEngine) bool {
        return switch (self.backend) {
            .flux => |*f| f.vae_enc != null,
            .krea => |k| k.vae_enc != null,
        };
    }

    /// True when instruction editing (in-context reference conditioning) is
    /// available — a trained FLUX.2 capability; Krea has no edit training.
    pub fn supportsEdit(self: *const ImageEngine) bool {
        return switch (self.backend) {
            .flux => |*f| f.vae_enc != null,
            .krea => false,
        };
    }

    /// Reconcile the engine's attached LoRA with the request: `path == null`
    /// detaches; a new path (or scale) loads + attaches; the same path+scale is
    /// a no-op reuse. Returns the number of matched DiT modules.
    pub fn setLora(self: *ImageEngine, path: ?[]const u8, scale: f32) !u32 {
        if (path) |p| {
            if (self.lora_path) |cur| {
                if (std.mem.eql(u8, cur, p) and scale == self.lora_scale) return self.lora_matched;
            }
            self.clearLora();
            var lf = try lora_mod.loadFile(self.allocator, p);
            const matched = switch (self.backend) {
                .flux => |*f| flux.attachLora(&f.dit, &lf, scale),
                .krea => |k| krea.attachLora(&k.dit, &lf, scale),
            };
            if (matched == 0) {
                lf.deinit();
                return error.LoraNoMatch;
            }
            self.lora_file = lf;
            self.lora_path = try self.allocator.dupe(u8, p);
            self.lora_scale = scale;
            self.lora_matched = matched;
            return matched;
        }
        self.clearLora();
        return 0;
    }

    fn clearLora(self: *ImageEngine) void {
        switch (self.backend) {
            .flux => |*f| flux.detachLora(&f.dit),
            .krea => |k| krea.detachLora(&k.dit),
        }
        if (self.lora_file) |*lf| lf.deinit();
        self.lora_file = null;
        if (self.lora_path) |p| self.allocator.free(p);
        self.lora_path = null;
        self.lora_matched = 0;
    }

    pub fn generatePng(self: *ImageEngine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, progress: ?sse.Progress) ![]u8 {
        const img = try self.generateImage(allocator, prompt, width, height, seed, steps, .{}, progress);
        defer _ = mlx.mlx_array_free(img);
        return krea.imageToPng(allocator, img, self.stream());
    }

    /// Generate the raw image [1,3,H,W] f32 [0,1] (owned mlx array). Lets the
    /// caller run the content filter on the pixels before PNG-encoding.
    pub fn generateImage(self: *ImageEngine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: ImageGenOpts, progress: ?sse.Progress) !mlx.mlx_array {
        return switch (self.backend) {
            .flux => |*f| f.generateImage(allocator, prompt, width, height, seed, steps, opts, progress),
            .krea => |k| blk: {
                if (opts.edit_image != null) break :blk error.EditUnsupported;
                const kopts = krea.GenOpts{
                    .init_image = opts.init_image,
                    .start_step = if (opts.init_image != null) img2imgStartStep(steps, opts.strength) else 0,
                    .cond_gain = opts.cond_gain,
                    .cond_weights = opts.cond_weights,
                };
                break :blk k.generateImageOpts(allocator, prompt, width, height, seed, steps, kopts, progress);
            },
        };
    }

    /// Encode an image [1,3,H,W] f32 [0,1] → PNG bytes (caller frees).
    pub fn toPng(self: *ImageEngine, allocator: std.mem.Allocator, img: mlx.mlx_array) ![]u8 {
        return krea.imageToPng(allocator, img, self.stream());
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

// ════════════════════════════════════════════════════════════════════════
// NSFW content filter (Krea 2 Community License §4.2). A single shared classifier
// (Falconsai ViT, src/nsfw.zig) is lazy-loaded once from ~/.mlx-serve/models and
// applied to EVERY generated image (FLUX + Krea). On by default; `--no-safety`
// or per-request `"safety": false` disables it. FAILS OPEN: if the classifier
// isn't downloaded/loadable, image gen proceeds unfiltered (with a warning).
// Loaded + run on the inference thread (the sole mlx caller) — gen is serial
// there, so the lazy-init singleton needs no lock.
// ════════════════════════════════════════════════════════════════════════

const NSFW_REPO_DIR = "Falconsai/nsfw_image_detection";
var g_nsfw: ?nsfw.Classifier = null;
var g_nsfw_tried: bool = false;

/// Locate the auto-downloaded classifier dir under ~/.mlx-serve/models (must
/// contain model.safetensors), or null (→ fail open).
fn resolveNsfwDir(allocator: std.mem.Allocator, io: std.Io) ?[]u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse return null);
    const dir = std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models/{s}", .{ home, NSFW_REPO_DIR }) catch return null;
    const marker = std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{dir}) catch {
        allocator.free(dir);
        return null;
    };
    defer allocator.free(marker);
    if (std.Io.Dir.openFileAbsolute(io, marker, .{})) |f| {
        f.close(io);
        return dir; // caller owns
    } else |_| {
        allocator.free(dir);
        return null;
    }
}

/// Lazy-load the shared classifier (once). Returns null on the fail-open path
/// (model missing or load error) — logged once.
fn ensureNsfwClassifier(io: std.Io, allocator: std.mem.Allocator) ?*nsfw.Classifier {
    if (g_nsfw_tried) return if (g_nsfw) |*c| c else null;
    g_nsfw_tried = true;
    const dir = resolveNsfwDir(allocator, io) orelse {
        log.warn("[image] content filter ON but classifier not found at ~/.mlx-serve/models/{s} — failing OPEN (images NOT filtered). Download it to enable.\n", .{NSFW_REPO_DIR});
        return null;
    };
    defer allocator.free(dir);
    g_nsfw = nsfw.load(io, allocator, dir) catch |err| {
        log.warn("[image] NSFW classifier load failed ({s}) — failing OPEN (images NOT filtered)\n", .{@errorName(err)});
        g_nsfw = null;
        return null;
    };
    log.info("[image] NSFW content filter ready (Falconsai ViT)\n", .{});
    return if (g_nsfw) |*c| c else null;
}

/// P(nsfw) threshold above which a generated image is blocked. Default 0.5;
/// operators can tune via `MLX_SERVE_NSFW_THRESHOLD` (stricter = lower).
fn nsfwThreshold() f32 {
    if (std.c.getenv("MLX_SERVE_NSFW_THRESHOLD")) |v| {
        return std.fmt.parseFloat(f32, std.mem.span(v)) catch nsfw.NSFW_THRESHOLD;
    }
    return nsfw.NSFW_THRESHOLD;
}

/// True if the request explicitly opts out of the content filter via
/// `"safety": false`.
fn bodyDisablesSafety(body: []const u8) bool {
    const pat = "\"safety\"";
    const ki = std.mem.indexOf(u8, body, pat) orelse return false;
    var i = ki + pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    return std.mem.startsWith(u8, body[i..], "false");
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

// Reference DEFAULT_NEGATIVE_PROMPT, plus a subtitle/caption block after
// "artifacts around text": quoted dialogue in the prompt makes the model burn
// scrambled subtitle-like captions into the frame; these terms steer CFG away
// from that. The audio tail (lip sync, muted/distorted voice, background
// noise, dialogue terms) is load-bearing for speech when audio CFG runs; if
// the whole thing ever exceeds LTX_PAD_LEN (~229 tokens today, 256 budget),
// ltxPadWithBos left-truncates and keeps that tail.
const LTX_NEGATIVE_PROMPT =
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, " ++
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted " ++
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, " ++
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts " ++
    "around text, subtitles, closed captions, burned-in captions, on-screen text, " ++
    "text overlay, lower thirds, karaoke-style lyrics, watermark, " ++
    "inconsistent perspective, camera shake, incorrect depth of field, " ++
    "background too sharp, background clutter, distracting reflections, harsh shadows, " ++
    "inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, " ++
    "unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, " ++
    "exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted " ++
    "audio, distorted voice, robotic voice, echo, background noise, off-sync audio, " ++
    "incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward " ++
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, " ++
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

/// LTX transformer variants: DEV (non-distilled, needs CFG — two-stage stage 1)
/// vs DISTILLED (guidance baked in — one-stage + two-stage stage 2).
pub const TransformerVariant = enum {
    dev,
    distilled,

    pub fn fileName(self: TransformerVariant) []const u8 {
        return switch (self) {
            .dev => "transformer-dev.safetensors",
            .distilled => "transformer-distilled.safetensors",
        };
    }
};

/// Video backend (currently LTX-Video 2.3). Holds the components + the
/// resolved Gemma text-encoder dir + its tokenizer. Components load on the CPU
/// stream; the forward graph runs on the GPU stream. The 11 GB transformer slot
/// holds ONE variant at a time; `ensureTransformer` swaps it (deinit + reload)
/// so dev + distilled are never resident together.
pub const VideoEngine = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    transformer: ltx.Component,
    transformer_variant: TransformerVariant,
    connector: ltx.Component,
    vae: ltx.Component,
    audio: ?ltx.Component = null, // audio VAE + vocoder; null → video has no sound
    vae_encoder: ?ltx.Component = null, // image VAE encoder; null → image-to-video + two-stage disabled
    upsampler: ?ltx.Component = null, // spatial x2 latent upsampler; lazy-loaded for two-stage
    tok: tok_mod.Tokenizer,
    gemma_dir: []u8,
    model_dir: []u8,
    // Runtime LoRA state (mirrors ImageEngine): the File owns the adapter
    // arrays the transformer Component's `lora` pointer reads through, so it
    // must live until the next detach (clearLora).
    lora_file: ?lora_mod.File = null,
    lora_path: ?[]u8 = null,
    lora_scale: f32 = 1.0,
    lora_matched: u32 = 0,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*VideoEngine {
        const self = try allocator.create(VideoEngine);
        errdefer allocator.destroy(self);
        self.* = undefined;
        self.allocator = allocator;
        self.audio = null;
        self.vae_encoder = null;
        self.upsampler = null;
        self.lora_file = null;
        self.lora_path = null;
        self.lora_scale = 1.0;
        self.lora_matched = 0;

        self.gemma_dir = try resolveGemmaDir(io, allocator);
        errdefer allocator.free(self.gemma_dir);
        log.info("[video] gemma text encoder: {s}\n", .{self.gemma_dir});
        self.model_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(self.model_dir);

        const cpu_s = mlx.mlx_default_cpu_stream_new();
        self.s = mlx.mlx_default_gpu_stream_new();

        // Initial transformer: prefer DISTILLED (the correct one-stage default —
        // the dev model without CFG produces visibly worse output); fall back to
        // dev for bundles downloaded before transformer-distilled shipped.
        const initial: TransformerVariant = if (self.hasVariant(io, .distilled)) .distilled else .dev;
        self.transformer = try loadTransformerVariant(allocator, model_dir, initial, cpu_s);
        self.transformer_variant = initial;
        errdefer self.transformer.deinit();
        if (initial == .dev)
            log.warn("[video] transformer-distilled.safetensors not found — one-stage falls back to the dev transformer (download the distilled variant for reference-quality fast generations)\n", .{});

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

        // Optional audio VAE + vocoder → the generated video gets a sound track.
        // Absent (video-only checkpoints, or not yet downloaded) is graceful.
        self.audio = loadAudioVae(io, allocator, model_dir, cpu_s);
        errdefer if (self.audio) |*a| a.deinit();

        // Optional VAE encoder → image-to-video (first-frame conditioning) and
        // the two-stage latent (de)normalization. Absent is graceful → t2v only.
        self.vae_encoder = loadVaeEncoder(io, allocator, model_dir, cpu_s);
        errdefer if (self.vae_encoder) |*e| e.deinit();

        self.tok = try tok_mod.loadTokenizerAny(io, allocator, self.gemma_dir);
        log.info("[video] LTX components + tokenizer ready (transformer={s})\n", .{@tagName(initial)});
        return self;
    }

    fn hasVariant(self: *VideoEngine, io: std.Io, variant: TransformerVariant) bool {
        var buf: [1024]u8 = undefined;
        const p = std.fmt.bufPrintZ(&buf, "{s}/{s}", .{ self.model_dir, variant.fileName() }) catch return false;
        return fileExists(io, p);
    }

    /// Swap the transformer slot to `want` (no-op when already loaded). The old
    /// component is freed BEFORE the new one loads so dev + distilled (11 GB
    /// each) never coexist.
    pub fn ensureTransformer(self: *VideoEngine, want: TransformerVariant) !void {
        if (self.transformer_variant == want) return;
        log.info("[video] swapping transformer: {s} -> {s}\n", .{ @tagName(self.transformer_variant), @tagName(want) });
        self.transformer.deinit();
        const cpu_s = mlx.mlx_default_cpu_stream_new();
        self.transformer = try loadTransformerVariant(self.allocator, self.model_dir, want, cpu_s);
        self.transformer_variant = want;
        // The fresh Component boots with `lora = null` — re-install the
        // attached adapter so a mid-pipeline swap (Stage2Swap) keeps it.
        self.applyLora();
    }

    /// Reconcile the attached LoRA with the request (mirrors ImageEngine):
    /// `path == null` detaches; the same path+scale is a no-op reuse; a new
    /// path/scale loads + installs on the transformer Component. Returns the
    /// number of adapter modules present in the DiT.
    pub fn setLora(self: *VideoEngine, path: ?[]const u8, scale: f32) !u32 {
        if (path) |p| {
            if (self.lora_path) |cur| {
                if (std.mem.eql(u8, cur, p) and scale == self.lora_scale) return self.lora_matched;
            }
            self.clearLora();
            var lf = try lora_mod.loadFile(self.allocator, p);
            const matched = ltx.countLoraMatches(&self.transformer, &lf);
            if (matched == 0) {
                lf.deinit();
                return error.LoraNoMatch;
            }
            self.lora_file = lf;
            self.lora_path = try self.allocator.dupe(u8, p);
            self.lora_scale = scale;
            self.lora_matched = matched;
            self.applyLora();
            return matched;
        }
        self.clearLora();
        return 0;
    }

    fn clearLora(self: *VideoEngine) void {
        self.transformer.lora = null;
        if (self.lora_file) |*lf| lf.deinit();
        self.lora_file = null;
        if (self.lora_path) |p| self.allocator.free(p);
        self.lora_path = null;
        self.lora_matched = 0;
    }

    fn applyLora(self: *VideoEngine) void {
        self.transformer.lora = if (self.lora_file) |*lf| lf else null;
        self.transformer.lora_scale = self.lora_scale;
    }

    /// Lazily load the spatial-x2 upsampler for the two-stage boundary.
    pub fn ensureUpsampler(self: *VideoEngine, io: std.Io) !*const ltx.Component {
        if (self.upsampler) |*u| return u;
        var buf: [1024]u8 = undefined;
        const p = std.fmt.bufPrintZ(&buf, "{s}/{s}.safetensors", .{ self.model_dir, ltx.UPSAMPLER_PREFIX }) catch return error.MissingUpsampler;
        if (!fileExists(io, p)) return error.MissingUpsampler;
        const cpu_s = mlx.mlx_default_cpu_stream_new();
        var comp = try ltx.loadComponent(self.allocator, p, cpu_s);
        var it = comp.map.iterator();
        while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*); // conv graph wants materialized weights
        self.upsampler = comp;
        log.info("[video] latent upsampler ready ({d} tensors)\n", .{comp.count()});
        return &self.upsampler.?;
    }

    pub fn deinit(self: *VideoEngine) void {
        self.clearLora();
        self.transformer.deinit();
        self.connector.deinit();
        self.vae.deinit();
        if (self.audio) |*a| a.deinit();
        if (self.vae_encoder) |*e| e.deinit();
        if (self.upsampler) |*u| u.deinit();
        self.tok.deinit();
        self.allocator.free(self.gemma_dir);
        self.allocator.free(self.model_dir);
        self.allocator.destroy(self);
    }
};

fn loadTransformerVariant(allocator: std.mem.Allocator, model_dir: []const u8, variant: TransformerVariant, cpu_s: mlx.mlx_stream) !ltx.Component {
    const tp = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, variant.fileName() }, 0);
    defer allocator.free(tp);
    return ltx.loadComponent(allocator, tp, cpu_s);
}

/// Load the LTX VAE encoder (`vae_encoder.safetensors`, ~0.6 GB, MLX-layout
/// bf16) from the model dir for image-to-video. Absent → null (I2V disabled,
/// text-to-video unaffected). Mirrors `loadAudioVae`.
fn loadVaeEncoder(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, cpu_s: mlx.mlx_stream) ?ltx.Component {
    var p: [1024]u8 = undefined;
    const path = std.fmt.bufPrintZ(&p, "{s}/vae_encoder.safetensors", .{model_dir}) catch return null;
    if (!fileExists(io, path)) {
        log.info("[video] no vae_encoder.safetensors in {s} — image-to-video disabled (text-to-video only)\n", .{model_dir});
        return null;
    }
    var comp = ltx.loadComponent(allocator, path, cpu_s) catch |e| {
        log.warn("[video] vae_encoder load failed ({}) — image-to-video disabled\n", .{e});
        return null;
    };
    var it = comp.map.iterator();
    while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*); // conv graph wants materialized weights
    log.info("[video] VAE encoder ready ({d} tensors) — image-to-video enabled\n", .{comp.count()});
    return comp;
}

/// Decode a PNG/JPEG image (raw file bytes) → BCFHW `[1,3,1,target_h,target_w]`
/// bf16 in [-1,1], bilinear-resized (matches the reference `x/127.5 - 1`
/// normalization; the resize is bilinear, not LANCZOS — close enough for the
/// first-frame anchor and not parity-tested). Returns null on decode failure.
fn decodeImageToBCFHW(allocator: std.mem.Allocator, encoded: []const u8, target_h: u32, target_w: u32, s: mlx.mlx_stream) ?mlx.mlx_array {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const src_ptr = stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &ch, 3) orelse return null;
    defer stb.stbi_image_free(src_ptr);
    const sw: usize = @intCast(w);
    const sh: usize = @intCast(h);
    if (sw == 0 or sh == 0) return null;
    const src = src_ptr[0 .. sw * sh * 3];

    const th: usize = target_h;
    const tw: usize = target_w;
    const out = allocator.alloc(f32, 3 * th * tw) catch return null;
    defer allocator.free(out);

    const clampIdx = struct {
        fn f(v: isize, n: usize) usize {
            if (v < 0) return 0;
            const uv: usize = @intCast(v);
            return if (uv >= n) n - 1 else uv;
        }
    }.f;

    var oy: usize = 0;
    while (oy < th) : (oy += 1) {
        const fy = (@as(f32, @floatFromInt(oy)) + 0.5) * @as(f32, @floatFromInt(sh)) / @as(f32, @floatFromInt(th)) - 0.5;
        const fy0 = @floor(fy);
        const wy = fy - fy0;
        const y0 = clampIdx(@intFromFloat(fy0), sh);
        const y1 = clampIdx(@as(isize, @intFromFloat(fy0)) + 1, sh);
        var ox: usize = 0;
        while (ox < tw) : (ox += 1) {
            const fx = (@as(f32, @floatFromInt(ox)) + 0.5) * @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(tw)) - 0.5;
            const fx0 = @floor(fx);
            const wx = fx - fx0;
            const x0 = clampIdx(@intFromFloat(fx0), sw);
            const x1 = clampIdx(@as(isize, @intFromFloat(fx0)) + 1, sw);
            var c: usize = 0;
            while (c < 3) : (c += 1) {
                const p00: f32 = @floatFromInt(src[(y0 * sw + x0) * 3 + c]);
                const p10: f32 = @floatFromInt(src[(y0 * sw + x1) * 3 + c]);
                const p01: f32 = @floatFromInt(src[(y1 * sw + x0) * 3 + c]);
                const p11: f32 = @floatFromInt(src[(y1 * sw + x1) * 3 + c]);
                const top = p00 * (1.0 - wx) + p10 * wx;
                const bot = p01 * (1.0 - wx) + p11 * wx;
                const v = top * (1.0 - wy) + bot * wy;
                out[c * th * tw + oy * tw + ox] = v / 127.5 - 1.0;
            }
        }
    }

    const arr = mlx.mlx_array_new_data(out.ptr, &[_]c_int{ 1, 3, 1, @intCast(th), @intCast(tw) }, 5, .float32);
    defer _ = mlx.mlx_array_free(arr);
    var bf = mlx.mlx_array_new();
    if (mlx.mlx_astype(&bf, arr, .bfloat16, s) != 0) {
        _ = mlx.mlx_array_free(bf);
        return null;
    }
    _ = mlx.mlx_array_eval(bf);
    return bf;
}

/// Reference dims for edit mode: keep the source's aspect ratio, cap the area
/// at ~1MP (klein's trained scale), round each side down to a multiple of 32
/// (the official prep's crop granularity; also satisfies the VAE /8 + latent
/// patchify /2). Never upscales; floors at 32.
fn fitRefDims(w: u32, h: u32) struct { w: u32, h: u32 } {
    const cap: f64 = 1024.0 * 1024.0;
    const area: f64 = @as(f64, @floatFromInt(w)) * @as(f64, @floatFromInt(h));
    const scale: f64 = @min(1.0, std.math.sqrt(cap / @max(area, 1.0)));
    const sw: u32 = @intFromFloat(@as(f64, @floatFromInt(w)) * scale);
    const sh: u32 = @intFromFloat(@as(f64, @floatFromInt(h)) * scale);
    return .{
        .w = @max(32, (sw / 32) * 32),
        .h = @max(32, (sh / 32) * 32),
    };
}

/// Native pixel dims of an encoded PNG/JPEG, without decoding the pixels.
fn imageNativeSize(encoded: []const u8) ?struct { w: u32, h: u32 } {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    if (stb.stbi_info_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &ch) == 0) return null;
    if (w <= 0 or h <= 0) return null;
    return .{ .w = @intCast(w), .h = @intCast(h) };
}

/// Decode a PNG/JPEG image (raw file bytes) → `[1,3,target_h,target_w]` f32
/// in [0,1]. COVER semantics: bilinear-sampled from the largest centered
/// source window matching the target's aspect ratio — the image is never
/// stretched; mismatched aspects lose edges to a center crop instead of
/// distorting the subject. Returns null on decode failure.
fn decodeImageToBCHW(allocator: std.mem.Allocator, encoded: []const u8, target_h: u32, target_w: u32) ?mlx.mlx_array {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const src_ptr = stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &ch, 3) orelse return null;
    defer stb.stbi_image_free(src_ptr);
    const sw: usize = @intCast(w);
    const sh: usize = @intCast(h);
    if (sw == 0 or sh == 0) return null;
    const src = src_ptr[0 .. sw * sh * 3];

    const th: usize = target_h;
    const tw: usize = target_w;
    const out = allocator.alloc(f32, 3 * th * tw) catch return null;
    defer allocator.free(out);

    // Centered source window with the target's aspect ratio.
    const src_ar = @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(sh));
    const tgt_ar = @as(f32, @floatFromInt(tw)) / @as(f32, @floatFromInt(th));
    var win_w: f32 = @floatFromInt(sw);
    var win_h: f32 = @floatFromInt(sh);
    if (src_ar > tgt_ar) {
        win_w = win_h * tgt_ar; // too wide → crop the sides
    } else {
        win_h = win_w / tgt_ar; // too tall → crop top/bottom
    }
    const x_off = (@as(f32, @floatFromInt(sw)) - win_w) * 0.5;
    const y_off = (@as(f32, @floatFromInt(sh)) - win_h) * 0.5;

    const clampIdx = struct {
        fn f(v: isize, n: usize) usize {
            if (v < 0) return 0;
            const uv: usize = @intCast(v);
            return if (uv >= n) n - 1 else uv;
        }
    }.f;

    var oy: usize = 0;
    while (oy < th) : (oy += 1) {
        const fy = y_off + (@as(f32, @floatFromInt(oy)) + 0.5) * win_h / @as(f32, @floatFromInt(th)) - 0.5;
        const fy0 = @floor(fy);
        const wy = fy - fy0;
        const y0 = clampIdx(@intFromFloat(fy0), sh);
        const y1 = clampIdx(@as(isize, @intFromFloat(fy0)) + 1, sh);
        var ox: usize = 0;
        while (ox < tw) : (ox += 1) {
            const fx = x_off + (@as(f32, @floatFromInt(ox)) + 0.5) * win_w / @as(f32, @floatFromInt(tw)) - 0.5;
            const fx0 = @floor(fx);
            const wx = fx - fx0;
            const x0 = clampIdx(@intFromFloat(fx0), sw);
            const x1 = clampIdx(@as(isize, @intFromFloat(fx0)) + 1, sw);
            var c: usize = 0;
            while (c < 3) : (c += 1) {
                const p00: f32 = @floatFromInt(src[(y0 * sw + x0) * 3 + c]);
                const p10: f32 = @floatFromInt(src[(y0 * sw + x1) * 3 + c]);
                const p01: f32 = @floatFromInt(src[(y1 * sw + x0) * 3 + c]);
                const p11: f32 = @floatFromInt(src[(y1 * sw + x1) * 3 + c]);
                const top = p00 * (1.0 - wx) + p10 * wx;
                const bot = p01 * (1.0 - wx) + p11 * wx;
                const v = top * (1.0 - wy) + bot * wy;
                out[c * th * tw + oy * tw + ox] = v / 255.0;
            }
        }
    }

    const arr = mlx.mlx_array_new_data(out.ptr, &[_]c_int{ 1, 3, @intCast(th), @intCast(tw) }, 4, .float32);
    _ = mlx.mlx_array_eval(arr);
    return arr;
}

/// Load the LTX audio VAE + vocoder (`audio_vae.safetensors` + `vocoder.safetensors`,
/// the q4 MLX-layout files from `dgrauet/ltx-2.3-mlx-q4`) from the model dir, or
/// from `$LTX_AUDIO_DIR`. Both files absent → null (the video stays silent).
fn loadAudioVae(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, cpu_s: mlx.mlx_stream) ?ltx.Component {
    // The directory holding the two audio files: model dir, or an override.
    const dir: []const u8 = if (std.c.getenv("LTX_AUDIO_DIR")) |env| blk: {
        const e = std.mem.span(env);
        break :blk if (e.len > 0) e else model_dir;
    } else model_dir;
    var ap: [1024]u8 = undefined;
    var vp: [1024]u8 = undefined;
    const audio_path = std.fmt.bufPrintZ(&ap, "{s}/audio_vae.safetensors", .{dir}) catch return null;
    const voc_path = std.fmt.bufPrintZ(&vp, "{s}/vocoder.safetensors", .{dir}) catch return null;
    if (!fileExists(io, audio_path) or !fileExists(io, voc_path)) {
        log.info("[video] no audio VAE/vocoder in {s} — generated video will be silent\n", .{dir});
        return null;
    }
    var comp = ltx_audio.loadAudioComponents(allocator, audio_path, voc_path, cpu_s) catch |e| {
        log.warn("[video] audio VAE load failed ({}) — video will be silent\n", .{e});
        return null;
    };
    log.info("[video] audio VAE + vocoder ready ({d} tensors) — video will have sound\n", .{comp.count()});
    return comp;
}

fn fileExists(io: std.Io, path: [:0]const u8) bool {
    // openFileAbsolute ASSERTS the path is absolute — a failed assert is
    // `unreachable`, i.e. ReleaseFast UB that can miscompile the CALLER (see
    // the openDirAbsolute gotcha in CLAUDE.md). Paths here come from --model /
    // $LTX_AUDIO_DIR / $LTX_GEMMA_DIR, all user-controlled, so guard first.
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return false;
    if (std.Io.Dir.openFileAbsolute(io, path, .{})) |f| {
        f.close(io);
        return true;
    } else |_| return false;
}

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
        // A relative override would feed openFileAbsolute's assert downstream
        // (ReleaseFast UB) — ignore it loudly instead.
        if (e.len > 0 and std.fs.path.isAbsolute(e)) return allocator.dupe(u8, e);
        if (e.len > 0) log.warn("[video] ignoring non-absolute LTX_GEMMA_DIR: {s}\n", .{e});
    }
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoGemmaDir);
    if (!std.fs.path.isAbsolute(home)) return error.NoGemmaDir;
    // 2-level `<author>/<name>` layout (what DownloadManager writes), then a
    // flat `<name>` layout (legacy / manual placement).
    const candidates = [_][]const u8{ LTX_GEMMA_REPO_DIR, "gemma-3-12b-it-4bit" };
    for (candidates) |rel| {
        const dir = std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models/{s}", .{ home, rel }) catch continue;
        var ok = false;
        {
            const cfg = std.fmt.allocPrintSentinel(allocator, "{s}/config.json", .{dir}, 0) catch {
                allocator.free(dir);
                continue;
            };
            defer allocator.free(cfg);
            ok = fileExists(io, cfg);
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
pub fn handleImage(io: std.Io, allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *ImageEngine) !void {
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

    // Source image: `image` (base64 PNG/JPEG) + `mode` ("variation" default /
    // "edit"). Variation = SDEdit renoise at `strength` (both backends);
    // edit = FLUX.2 in-context reference conditioning (instruction edits —
    // "make the hair blue" — with the source attended to clean; no strength).
    var init_img: ?mlx.mlx_array = null;
    defer if (init_img) |ii| {
        _ = mlx.mlx_array_free(ii);
    };
    var strength: f32 = 0.6;
    var edit_mode = false;
    if (extractJsonString(body, "mode")) |m| {
        if (std.mem.eql(u8, m, "edit")) {
            edit_mode = true;
        } else if (!std.mem.eql(u8, m, "variation")) {
            return sendError(conn, 400, "'mode' must be \"edit\" or \"variation\"");
        }
    }
    if (extractJsonString(body, "image")) |raw_img| {
        if (edit_mode and !engine.supportsEdit())
            return sendError(conn, 400, "instruction editing (mode:\"edit\") requires a FLUX.2 model — this model only supports mode:\"variation\"");
        if (!engine.supportsImg2Img())
            return sendError(conn, 400, "image-to-image needs the VAE encoder weights, which failed to load for this model");
        if (extractJsonFloat(body, "strength")) |sv| {
            if (!(sv > 0.0 and sv <= 1.0)) return sendError(conn, 400, "'strength' must be in (0,1]");
            strength = @floatCast(sv);
        }
        const img_bytes = base64DecodeAlloc(allocator, raw_img) catch
            return sendError(conn, 400, "invalid base64 in 'image'");
        defer allocator.free(img_bytes);
        if (edit_mode) {
            // The reference keeps its OWN aspect ratio (fit to ~1MP, /32 dims —
            // official prep behavior); its latent grid is independent of the
            // output grid, so nothing gets squished or cropped away.
            const nat = imageNativeSize(img_bytes) orelse
                return sendError(conn, 400, "could not decode 'image' (PNG/JPEG supported)");
            const rd = fitRefDims(nat.w, nat.h);
            init_img = decodeImageToBCHW(allocator, img_bytes, rd.h, rd.w) orelse
                return sendError(conn, 400, "could not decode 'image' (PNG/JPEG supported)");
            log.info("[image] edit: reference {d}x{d} -> {d}x{d} (in-context conditioning)\n", .{ nat.w, nat.h, rd.w, rd.h });
        } else {
            // Variation shares the output's latent grid — cover + center-crop
            // to the output dims (never stretched).
            init_img = decodeImageToBCHW(allocator, img_bytes, height, width) orelse
                return sendError(conn, 400, "could not decode 'image' (PNG/JPEG supported)");
            log.info("[image] img2img: source {d} bytes, strength={d:.2}\n", .{ img_bytes.len, strength });
        }
    } else if (edit_mode) {
        return sendError(conn, 400, "mode:\"edit\" needs an 'image' to edit");
    }

    // Conditioning rebalance: global gain + per-tapped-layer weights.
    var cond_gain: f32 = 1.0;
    if (extractJsonFloat(body, "cond_gain")) |g| {
        if (!(g >= 0.0 and g <= 10.0)) return sendError(conn, 400, "'cond_gain' must be in [0,10]");
        cond_gain = @floatCast(g);
    }
    var wbuf: [16]f32 = undefined;
    var cond_weights: ?[]const f32 = null;
    if (std.mem.indexOf(u8, body, "\"cond_weights\"") != null) {
        const wl = extractCondWeights(body, &wbuf) orelse
            return sendError(conn, 400, "invalid 'cond_weights' (numbers, comma/space separated, or a JSON array)");
        if (wl.len != engine.condWeightCount()) {
            var msg_buf: [128]u8 = undefined;
            const msg = std.fmt.bufPrint(&msg_buf, "'cond_weights' needs exactly {d} values for this model (got {d})", .{ engine.condWeightCount(), wl.len }) catch "wrong 'cond_weights' count";
            return sendError(conn, 400, msg);
        }
        cond_weights = wl;
        log.info("[image] rebalance: gain={d:.2} weights={d}\n", .{ cond_gain, wl.len });
    }

    // Style LoRA: absolute path to a .safetensors adapter (+ optional scale).
    // No `lora_path` in the request detaches whatever was attached before.
    if (extractJsonString(body, "lora_path")) |lp_raw| {
        const lp = try jsonUnescape(allocator, lp_raw);
        defer allocator.free(lp);
        const lscale: f32 = @floatCast(extractJsonFloat(body, "lora_scale") orelse 1.0);
        const matched = engine.setLora(lp, lscale) catch |err| switch (err) {
            error.LoraNoMatch => return sendError(conn, 400, "LoRA has no modules matching this model's DiT — wrong LoRA for this architecture?"),
            error.BadLoraPath => return sendError(conn, 400, "'lora_path' must be an absolute path to a .safetensors file"),
            error.OutOfMemory => return err,
            else => return sendError(conn, 400, "failed to load the LoRA file"),
        };
        log.info("[image] lora: matched {d} modules from {s} (scale {d:.2})\n", .{ matched, lp, lscale });
    } else {
        _ = engine.setLora(null, 1.0) catch {};
    }

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[image] generating {d}x{d} steps={d} stream={}: {d} chars\n", .{ width, height, steps, want_stream, prompt.len });
    var sctx = sse.StreamCtx{ .conn = conn };
    const prog: ?sse.Progress = if (want_stream) sctx.progress() else null;
    if (want_stream) try conn.writeAll(sse.headers);

    const gen_opts = ImageGenOpts{
        .init_image = if (edit_mode) null else init_img,
        .strength = strength,
        .edit_image = if (edit_mode) init_img else null,
        .cond_gain = cond_gain,
        .cond_weights = cond_weights,
    };
    const img = engine.generateImage(allocator, prompt, width, height, seed, steps, gen_opts, prog) catch |err| {
        log.err("[image] generation failed: {}\n", .{err});
        if (want_stream) {
            sse.sendError(conn, "generation failed");
            return;
        }
        return sendError(conn, 500, "generation failed");
    };
    defer _ = mlx.mlx_array_free(img);

    // Content filter (Krea license §4.2; on by default, `--no-safety` /
    // `"safety":false` to disable). Run the NSFW classifier on the generated
    // pixels; if flagged, refuse. Fail OPEN if the classifier is unavailable.
    if (server_mod.image_safety_filter and !bodyDisablesSafety(body)) {
        if (ensureNsfwClassifier(io, allocator)) |clf| {
            const p_nsfw = clf.classify(img) catch |err| blk: {
                log.warn("[image] classifier error ({s}) — failing OPEN\n", .{@errorName(err)});
                break :blk @as(f32, 0);
            };
            if (p_nsfw > nsfwThreshold()) {
                log.warn("[image] output blocked by content filter (P(nsfw)={d:.3})\n", .{p_nsfw});
                if (want_stream) {
                    sse.sendError(conn, "generated image blocked by the content filter");
                    return;
                }
                return sendError(conn, 400, "generated image blocked by the content filter (set \"safety\":false or run with --no-safety to override)");
            }
        }
    }

    const png_bytes = try engine.toPng(allocator, img);
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
/// Convert interleaved f32 PCM in [-1,1] to little-endian signed 16-bit bytes.
fn f32ToPcm16leBytes(allocator: std.mem.Allocator, pcm: []const f32) ![]u8 {
    const out = try allocator.alloc(u8, pcm.len * 2);
    for (pcm, 0..) |v, i| {
        const clamped = @max(@as(f32, -1.0), @min(@as(f32, 1.0), v));
        const iv: i16 = @intFromFloat(@round(clamped * 32767.0));
        const u: u16 = @bitCast(iv);
        out[i * 2] = @intCast(u & 0xff);
        out[i * 2 + 1] = @intCast((u >> 8) & 0xff);
    }
    return out;
}

/// The three LTX pipelines. `one_stage` = distilled fast path (reference
/// TextToVideoPipeline); the two-stage modes generate at half resolution with
/// the dev model + full guidance, upscale latents, then refine with the
/// distilled model (reference TwoStagePipeline / TwoStageHQPipeline).
pub const VideoPipeline = enum {
    one_stage,
    two_stage,
    two_stage_hq,

    pub fn fromBody(body: []const u8) VideoPipeline {
        const raw = extractJsonString(body, "pipeline") orelse return .one_stage;
        if (std.mem.eql(u8, raw, "two_stage")) return .two_stage;
        if (std.mem.eql(u8, raw, "two_stage_hq")) return .two_stage_hq;
        return .one_stage;
    }
};

/// STG perturbs block 28 by default (reference MultiModalGuiderParams for the
/// Euler two-stage pipeline; HQ uses no STG blocks).
const STG_BLOCKS_DEFAULT = [_]u32{28};

pub const VideoGuiders = struct {
    vp: ltx.GuiderParams,
    ap: ltx.GuiderParams,
    stage1_steps_default: u32,
};

/// Reference per-pipeline guidance defaults, with per-request overrides:
/// Audio-to-video pipeline gate: the reference a2vid pipelines are two-stage
/// only (stage 1 needs real CFG against the frozen soundtrack; the distilled
/// one-stage schedule was never trained with audio conditioning). Returns the
/// 400 message, or null when the pipeline is allowed.
pub fn a2vidPipelineError(pipeline: VideoPipeline) ?[]const u8 {
    return switch (pipeline) {
        .one_stage => "audio-to-video requires a two-stage pipeline — set \"pipeline\":\"two_stage\" or \"two_stage_hq\"",
        .two_stage, .two_stage_hq => null,
    };
}

/// Sample count (interleaved, all channels) to mux for a2vid: the ORIGINAL
/// clip trimmed to the video duration — never longer than the clip itself.
pub fn a2vidMuxSampleCount(pcm_len: usize, channels: u32, sample_rate: u32, video_frames: u32, fps: f32) usize {
    if (channels == 0 or fps <= 0) return 0;
    const dur_s = @as(f64, @floatFromInt(video_frames)) / @as(f64, fps);
    const max_frames: usize = @intFromFloat(dur_s * @as(f64, @floatFromInt(sample_rate)));
    return @min(pcm_len - pcm_len % channels, max_frames * channels);
}

/// `cfg_scale` (video), `cfg_audio_scale` (audio), `stg_scale`.
pub fn videoGuiderDefaults(pipeline: VideoPipeline, cfg_video: ?f32, cfg_audio: ?f32, stg: ?f32) VideoGuiders {
    switch (pipeline) {
        // one-stage (distilled) is designed for cfg 1.0 — no guidance, one DiT
        // forward/step. Overridable; rescale only engages when guided.
        .one_stage => return .{
            .vp = .{ .cfg = cfg_video orelse 1.0, .rescale = 0.7 },
            .ap = .{ .cfg = cfg_audio orelse (cfg_video orelse 1.0), .rescale = 0.7 },
            .stage1_steps_default = 30,
        },
        .two_stage => return .{
            .vp = .{ .cfg = cfg_video orelse 3.0, .stg = stg orelse 0.0, .rescale = 0.7, .modality = 3.0, .stg_blocks = &STG_BLOCKS_DEFAULT },
            .ap = .{ .cfg = cfg_audio orelse 7.0, .stg = stg orelse 0.0, .rescale = 0.7, .modality = 3.0, .stg_blocks = &STG_BLOCKS_DEFAULT },
            .stage1_steps_default = 30,
        },
        // HQ: res_2s sampler, no STG blocks, softer video rescale (0.45), full
        // audio rescale (1.0).
        .two_stage_hq => return .{
            .vp = .{ .cfg = cfg_video orelse 3.0, .stg = stg orelse 0.0, .rescale = 0.45, .modality = 3.0, .stg_blocks = &.{} },
            .ap = .{ .cfg = cfg_audio orelse 7.0, .stg = stg orelse 0.0, .rescale = 1.0, .modality = 3.0, .stg_blocks = &.{} },
            .stage1_steps_default = 15,
        },
    }
}

/// Stage-2 transformer provider for the two-stage boundary: swaps the engine's
/// transformer slot from dev to distilled (freeing dev first).
const Stage2Swap = struct {
    engine: *VideoEngine,

    fn swap(ctx: *anyopaque) anyerror!*const ltx.Component {
        const self: *Stage2Swap = @ptrCast(@alignCast(ctx));
        try self.engine.ensureTransformer(.distilled);
        return &self.engine.transformer;
    }
};

pub fn handleVideo(io: std.Io, allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *VideoEngine) !void {
    const prompt_raw = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt'");
    const prompt = try jsonUnescape(allocator, prompt_raw);
    defer allocator.free(prompt);
    if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

    const num_frames: u32 = @intCast(extractJsonInt(body, "num_frames") orelse 9);
    const height: u32 = @intCast(extractJsonInt(body, "height") orelse 256);
    const width: u32 = @intCast(extractJsonInt(body, "width") orelse 384);
    const seed: u64 = extractJsonInt(body, "seed") orelse 42;
    const frame_rate: f32 = 24.0;

    const pipeline = VideoPipeline.fromBody(body);
    const cfg_video: ?f32 = if (extractJsonFloat(body, "cfg_scale")) |v| @floatCast(v) else null;
    const cfg_audio: ?f32 = if (extractJsonFloat(body, "cfg_audio_scale")) |v| @floatCast(v) else null;
    const stg: ?f32 = if (extractJsonFloat(body, "stg_scale")) |v| @floatCast(v) else null;
    const guiders = videoGuiderDefaults(pipeline, cfg_video, cfg_audio, stg);
    const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse guiders.stage1_steps_default);
    const stage2_steps: u32 = @intCast(extractJsonInt(body, "stage2_steps") orelse 0);

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[video] generating {s} {d}f {d}x{d} steps={d} cfg={d:.1}/{d:.1} stg={d:.1} stream={}: {d} chars\n", .{ @tagName(pipeline), num_frames, height, width, steps, guiders.vp.cfg, guiders.ap.cfg, guiders.vp.stg, want_stream, prompt.len });

    // Two-stage prerequisites: even half-res grid, the VAE encoder (latent
    // statistics), the upsampler, and BOTH transformer variants on disk.
    // Missing pieces are an explicit 400 — never a silent one-stage downgrade.
    if (pipeline != .one_stage) {
        if (height % 64 != 0 or width % 64 != 0)
            return sendError(conn, 400, "two-stage pipelines need width/height divisible by 64 (half-resolution stage)");
        if (engine.vae_encoder == null)
            return sendError(conn, 400, "two-stage pipelines require vae_encoder.safetensors (latent statistics) — download it into the model dir");
        if (!engine.hasVariant(io, .dev))
            return sendError(conn, 400, "two-stage pipelines require transformer-dev.safetensors — download it into the model dir");
        if (!engine.hasVariant(io, .distilled))
            return sendError(conn, 400, "two-stage pipelines require transformer-distilled.safetensors (stage-2 refine) — download it into the model dir");
        _ = engine.ensureUpsampler(io) catch
            return sendError(conn, 400, "two-stage pipelines require spatial_upscaler_x2_v1_1.safetensors — download it into the model dir");
    }

    // Style LoRA: absolute path to a .safetensors adapter (+ optional scale),
    // applied to the DiT at runtime. No `lora_path` in the request detaches
    // whatever was attached before (same contract as handleImage).
    if (extractJsonString(body, "lora_path")) |lp_raw| {
        const lp = try jsonUnescape(allocator, lp_raw);
        defer allocator.free(lp);
        const lscale: f32 = @floatCast(extractJsonFloat(body, "lora_scale") orelse 1.0);
        const matched = engine.setLora(lp, lscale) catch |err| switch (err) {
            error.LoraNoMatch => return sendError(conn, 400, "LoRA has no modules matching this model's DiT — wrong LoRA for this architecture?"),
            error.BadLoraPath => return sendError(conn, 400, "'lora_path' must be an absolute path to a .safetensors file"),
            error.OutOfMemory => return err,
            else => return sendError(conn, 400, "failed to load the LoRA file"),
        };
        log.info("[video] lora: matched {d} modules from {s} (scale {d:.2})\n", .{ matched, lp, lscale });
    } else {
        _ = engine.setLora(null, 1.0) catch {};
    }

    // ── audio-to-video: `audio` is a base64 WAV (PCM16/24/f32, any rate,
    // mono/stereo). Unlike `first_frame_image` this is NOT graceful — the user
    // asked for THIS soundtrack, so a silent downgrade to generated audio
    // would be a wrong result. Explicit 400s instead.
    var audio_cond: ?mlx.mlx_array = null;
    defer if (audio_cond) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var a2v_pcm: ?wav_mod.Decoded = null; // original decode — muxed into the mp4
    defer if (a2v_pcm) |d| allocator.free(d.pcm);
    if (extractJsonString(body, "audio")) |raw_audio| {
        const b64 = try jsonUnescape(allocator, raw_audio); // handles \/ from Swift
        defer allocator.free(b64);
        if (b64.len > 0) {
            if (a2vidPipelineError(pipeline)) |msg| return sendError(conn, 400, msg);
            if (engine.audio == null)
                return sendError(conn, 400, "audio-to-video requires audio_vae.safetensors (encoder) — download it into the model dir");
            const wav_bytes = base64DecodeAlloc(allocator, b64) catch
                return sendError(conn, 400, "audio: invalid base64");
            defer allocator.free(wav_bytes);
            const dec = wav_mod.decode(allocator, wav_bytes) catch
                return sendError(conn, 400, "audio: expected a PCM16/PCM24/float32 WAV");
            a2v_pcm = dec;
            // Conditioning path: stereo @ 16 kHz → mel → VAE encode → [1,Na,128],
            // truncated to the video's token budget (the reference never pads).
            const stereo = try wav_mod.toStereoInterleaved(allocator, dec.pcm, dec.channels);
            defer allocator.free(stereo);
            const cond_pcm = try wav_mod.resampleLinear(allocator, stereo, 2, dec.sample_rate, ltx_audio.COND_SAMPLE_RATE);
            defer allocator.free(cond_pcm);
            const max_tokens = ltx.computeAudioTokenCount(num_frames, frame_rate);
            audio_cond = ltx_audio.encodeAudioCond(allocator, &engine.audio.?, cond_pcm, max_tokens, engine.s) catch |err| {
                if (err == error.AudioTooShort)
                    return sendError(conn, 400, "audio: clip too short (needs at least ~50 ms)");
                log.err("[video] audio conditioning encode failed: {}\n", .{err});
                return sendError(conn, 500, "audio: conditioning encode failed");
            };
            log.info("[video] audio-to-video: {d} tokens (budget {d}) from {d} Hz {d}ch clip\n", .{ mlx.getShape(audio_cond.?)[1], max_tokens, dec.sample_rate, dec.channels });
        }
    }

    const pos_ids = try ltxTokenizePadded(allocator, &engine.tok, prompt);
    defer allocator.free(pos_ids);
    const neg_ids = try ltxTokenizePadded(allocator, &engine.tok, LTX_NEGATIVE_PROMPT);
    defer allocator.free(neg_ids);

    // Optional image-to-video: `first_frame_image` is a base64 PNG/JPEG (the app
    // sends the picked file). Decode + preprocess to the encoder's pixel grid
    // ((H/32)*32 × (W/32)*32). Graceful: missing encoder, bad image, or no field
    // → text-to-video (mirrors `ref_audio` in handleAudio).
    var cond_img: ?mlx.mlx_array = null;
    defer if (cond_img) |c| {
        _ = mlx.mlx_array_free(c);
    };
    var cond_img_half: ?mlx.mlx_array = null; // two-stage stage-1 grid
    defer if (cond_img_half) |c| {
        _ = mlx.mlx_array_free(c);
    };
    var enc_ptr: ?*const ltx.Component = null;
    if (extractJsonString(body, "first_frame_image")) |raw_img| {
        if (engine.vae_encoder) |*ve| {
            const b64 = try jsonUnescape(allocator, raw_img); // handles \/ from Swift JSONSerialization
            defer allocator.free(b64);
            if (b64.len > 0) {
                if (base64DecodeAlloc(allocator, b64)) |img_bytes| {
                    defer allocator.free(img_bytes);
                    const enc_h = (height / 32) * 32;
                    const enc_w = (width / 32) * 32;
                    if (decodeImageToBCFHW(allocator, img_bytes, enc_h, enc_w, engine.s)) |arr| {
                        cond_img = arr;
                        enc_ptr = ve;
                        log.info("[video] image-to-video: first frame {d}x{d}\n", .{ enc_h, enc_w });
                    } else log.warn("[video] first_frame_image decode failed — text-to-video\n", .{});
                    // Two-stage conditions stage 1 at the half-resolution grid
                    // (the reference re-prepares the image per stage).
                    if (pipeline != .one_stage and cond_img != null) {
                        const half_h = ((height / 2) / 32) * 32;
                        const half_w = ((width / 2) / 32) * 32;
                        cond_img_half = decodeImageToBCFHW(allocator, img_bytes, half_h, half_w, engine.s);
                        if (cond_img_half == null) log.warn("[video] half-res first frame decode failed — stage 1 unconditioned\n", .{});
                    }
                } else |e| log.warn("[video] first_frame_image base64 decode failed: {} — text-to-video\n", .{e});
            }
        } else {
            log.warn("[video] vae_encoder not loaded — ignoring first_frame_image (text-to-video)\n", .{});
        }
    }

    var sctx = sse.StreamCtx{ .conn = conn };
    const prog: ?ltx.Progress = if (want_stream) sctx.progress() else null;
    if (want_stream) try conn.writeAll(sse.headers);

    var frames = switch (pipeline) {
        .one_stage => blk: {
            // Run the schedule the loaded variant was trained for; the request
            // never forces a swap here (dev-only bundles keep working).
            const distilled = engine.transformer_variant == .distilled;
            break :blk ltx.generateVideoFrames(io, allocator, .{}, &engine.transformer, &engine.connector, &engine.vae, enc_ptr, cond_img, engine.gemma_dir, pos_ids, neg_ids, LTX_PAD_ID, num_frames, height, width, frame_rate, steps, distilled, seed, guiders.vp, guiders.ap, prog, engine.s);
        },
        .two_stage, .two_stage_hq => blk: {
            engine.ensureTransformer(.dev) catch |err| break :blk err;
            var swapper = Stage2Swap{ .engine = engine };
            const opts = ltx.TwoStageOpts{
                .hq = pipeline == .two_stage_hq,
                .stage1_steps = steps,
                .stage2_steps = stage2_steps,
                .upsampler = engine.ensureUpsampler(io) catch |err| break :blk err,
                .swap_ctx = @ptrCast(&swapper),
                .swap = Stage2Swap.swap,
            };
            break :blk ltx.generateVideoFramesTwoStage(io, allocator, .{}, &engine.transformer, &engine.connector, &engine.vae, &engine.vae_encoder.?, cond_img_half, cond_img, audio_cond, engine.gemma_dir, pos_ids, neg_ids, LTX_PAD_ID, num_frames, height, width, frame_rate, opts, seed, guiders.vp, guiders.ap, prog, engine.s);
        },
    } catch |err| {
        if (err == error.Cancelled) {
            // Client hung up mid-generation (progress write failed) — the
            // denoise loop aborted; nothing to write, the socket is dead.
            log.info("[video] generation cancelled — client disconnected\n", .{});
            return;
        }
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

    // ── optional audio: decode the DiT audio latent → 16-bit PCM, base64.
    // a2vid: the ORIGINAL input clip is muxed instead (reference behavior —
    // higher fidelity than a VAE round-trip), trimmed to the video duration.
    var audio_b64: ?[]u8 = null;
    defer if (audio_b64) |a| allocator.free(a);
    var audio_sr: u32 = 0;
    var audio_ch: u32 = 0;
    if (a2v_pcm) |dec| {
        // >2-channel sources downmix to stereo (the client's mux path only
        // builds mono/stereo layouts).
        const mux_pcm: []const f32 = if (dec.channels > 2)
            try wav_mod.toStereoInterleaved(allocator, dec.pcm, dec.channels)
        else
            dec.pcm;
        defer if (dec.channels > 2) allocator.free(@constCast(mux_pcm));
        const mux_ch: u32 = @min(dec.channels, 2);
        const n = a2vidMuxSampleCount(mux_pcm.len, mux_ch, dec.sample_rate, frames.frames, frame_rate);
        const pcm = try f32ToPcm16leBytes(allocator, mux_pcm[0..n]);
        defer allocator.free(pcm);
        const al_len = std.base64.standard.Encoder.calcSize(pcm.len);
        const ab = try allocator.alloc(u8, al_len);
        _ = std.base64.standard.Encoder.encode(ab, pcm);
        audio_b64 = ab;
        audio_sr = dec.sample_rate;
        audio_ch = mux_ch;
        log.info("[video] -> original audio passthrough {d} samples {d}ch {d}Hz\n", .{ n / mux_ch, mux_ch, dec.sample_rate });
    } else if (engine.audio) |*acomp| {
        if (frames.audio_latent) |al| {
            if (ltx_audio.decodeAudio(allocator, acomp, al, engine.s)) |wav_v| {
                var wav = wav_v;
                defer wav.deinit(allocator);
                const pcm = try f32ToPcm16leBytes(allocator, wav.pcm);
                defer allocator.free(pcm);
                const al_len = std.base64.standard.Encoder.calcSize(pcm.len);
                const ab = try allocator.alloc(u8, al_len);
                _ = std.base64.standard.Encoder.encode(ab, pcm);
                audio_b64 = ab;
                audio_sr = wav.sample_rate;
                audio_ch = wav.channels;
                log.info("[video] -> audio {d} samples {d}ch {d}Hz ({d} pcm bytes)\n", .{ wav.frames, wav.channels, wav.sample_rate, pcm.len });
            } else |err| {
                log.warn("[video] audio decode failed: {} — video stays silent\n", .{err});
            }
        }
    }

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    const prefix = if (want_stream) "data: {\"type\":\"complete\"," else "{\"created\":0,";
    const head = try std.fmt.allocPrint(allocator, "{s}\"frames\":{d},\"height\":{d},\"width\":{d},\"fps\":{d},\"format\":\"rgb8\",\"data\":\"", .{ prefix, frames.frames, frames.height, frames.width, @as(u32, @intFromFloat(frame_rate)) });
    defer allocator.free(head);
    try out.appendSlice(allocator, head);
    try out.appendSlice(allocator, b64);
    try out.appendSlice(allocator, "\"");
    if (audio_b64) |ab| {
        const ah = try std.fmt.allocPrint(allocator, ",\"audio_sample_rate\":{d},\"audio_channels\":{d},\"audio_format\":\"pcm_s16le\",\"audio_data\":\"", .{ audio_sr, audio_ch });
        defer allocator.free(ah);
        try out.appendSlice(allocator, ah);
        try out.appendSlice(allocator, ab);
        try out.appendSlice(allocator, "\"");
    }
    try out.appendSlice(allocator, if (want_stream) "}\n\n" else "}");
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

/// Parse a JSON number (int or float) for `key`. Accepts a leading sign, digits,
/// and a decimal point (no exponent — gen params don't need it).
fn extractJsonFloat(body: []const u8, key: []const u8) ?f64 {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    const start = i;
    if (i < body.len and (body[i] == '-' or body[i] == '+')) i += 1;
    while (i < body.len and (std.ascii.isDigit(body[i]) or body[i] == '.')) i += 1;
    if (i == start) return null;
    return std.fmt.parseFloat(f64, body[start..i]) catch null;
}

/// Parse a comma/whitespace-separated float list ("1,2,3" / "0.5 1 -2") into
/// `buf`. Empty tokens are skipped; any unparseable token or more values than
/// `buf` holds → null. Returns the filled slice of `buf`.
fn parseFloatList(text: []const u8, buf: []f32) ?[]f32 {
    var n: usize = 0;
    var it = std.mem.tokenizeAny(u8, text, ", \t\r\n");
    while (it.next()) |tok| {
        if (n >= buf.len) return null;
        buf[n] = std.fmt.parseFloat(f32, tok) catch return null;
        if (!std.math.isFinite(buf[n])) return null;
        n += 1;
    }
    if (n == 0) return null;
    return buf[0..n];
}

/// Map an img2img strength onto the denoise schedule: skip the first
/// `steps - round(steps·strength)` steps (diffusers convention: strength 1 →
/// full schedule from pure noise; low strength → few steps, small change).
/// Clamped so at least one step always runs.
fn img2imgStartStep(steps: u32, strength: f32) u32 {
    const fsteps: f32 = @floatFromInt(steps);
    const run: u32 = @intFromFloat(@round(fsteps * std.math.clamp(strength, 0.0, 1.0)));
    const start = steps -| run;
    return @min(start, steps -| 1);
}

/// Extract the `cond_weights` request field: either a JSON number array
/// (`[1, 0.5, …]`) or a comma/space-separated string (`"1 0.5 …"`).
fn extractCondWeights(body: []const u8, buf: []f32) ?[]f32 {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{"cond_weights"}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    if (i >= body.len) return null;
    if (body[i] == '[') {
        const end = std.mem.indexOfScalarPos(u8, body, i + 1, ']') orelse return null;
        return parseFloatList(body[i + 1 .. end], buf);
    }
    if (body[i] == '"') {
        const end = std.mem.indexOfScalarPos(u8, body, i + 1, '"') orelse return null;
        return parseFloatList(body[i + 1 .. end], buf);
    }
    return null;
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

test "bodyDisablesSafety detects per-request opt-out" {
    try testing.expect(bodyDisablesSafety("{\"prompt\":\"x\",\"safety\":false}"));
    try testing.expect(bodyDisablesSafety("{\"safety\": false }"));
    try testing.expect(!bodyDisablesSafety("{\"prompt\":\"x\",\"safety\":true}"));
    try testing.expect(!bodyDisablesSafety("{\"prompt\":\"x\"}"));
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

test "f32ToPcm16leBytes converts, clamps, and is little-endian" {
    const alloc = testing.allocator;
    // 0.0 → 0; 1.0 → 32767; -1.0 → -32767; out-of-range clamps; midscale rounds.
    const pcm = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 0.5 };
    const bytes = try f32ToPcm16leBytes(alloc, &pcm);
    defer alloc.free(bytes);
    try testing.expectEqual(@as(usize, 12), bytes.len);
    const read = struct {
        fn le(b: []const u8, i: usize) i16 {
            return @bitCast(@as(u16, b[i * 2]) | (@as(u16, b[i * 2 + 1]) << 8));
        }
    }.le;
    try testing.expectEqual(@as(i16, 0), read(bytes, 0));
    try testing.expectEqual(@as(i16, 32767), read(bytes, 1));
    try testing.expectEqual(@as(i16, -32767), read(bytes, 2));
    try testing.expectEqual(@as(i16, 32767), read(bytes, 3)); // 2.0 clamps to 1.0
    try testing.expectEqual(@as(i16, -32767), read(bytes, 4)); // -2.0 clamps to -1.0
    try testing.expectEqual(@as(i16, @intFromFloat(@round(0.5 * 32767.0))), read(bytes, 5));
}

test "extractJsonFloat parses cfg scales (int + float + sign)" {
    try testing.expectEqual(@as(?f64, 1.0), extractJsonFloat("{\"cfg_scale\": 1.0}", "cfg_scale"));
    try testing.expectEqual(@as(?f64, 3.5), extractJsonFloat("{\"cfg_scale\":3.5,\"x\":1}", "cfg_scale"));
    try testing.expectEqual(@as(?f64, 7), extractJsonFloat("{\"cfg_audio_scale\": 7}", "cfg_audio_scale"));
    try testing.expectEqual(@as(?f64, null), extractJsonFloat("{\"prompt\":\"hi\"}", "cfg_scale"));
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

test "VideoPipeline.fromBody parses the pipeline field" {
    try testing.expectEqual(VideoPipeline.one_stage, VideoPipeline.fromBody("{\"prompt\":\"x\"}"));
    try testing.expectEqual(VideoPipeline.one_stage, VideoPipeline.fromBody("{\"pipeline\":\"one_stage\"}"));
    try testing.expectEqual(VideoPipeline.two_stage, VideoPipeline.fromBody("{\"pipeline\":\"two_stage\"}"));
    try testing.expectEqual(VideoPipeline.two_stage_hq, VideoPipeline.fromBody("{\"pipeline\":\"two_stage_hq\"}"));
    try testing.expectEqual(VideoPipeline.one_stage, VideoPipeline.fromBody("{\"pipeline\":\"garbage\"}"));
}

test "videoGuiderDefaults mirrors the reference per-pipeline guidance" {
    // one-stage: no guidance by default (single forward), override respected
    const one = videoGuiderDefaults(.one_stage, null, null, null);
    try testing.expect(!one.vp.needsGuidance());
    try testing.expect(!one.ap.needsGuidance());
    try testing.expectEqual(@as(u32, 30), one.stage1_steps_default);
    const one_ovr = videoGuiderDefaults(.one_stage, 3.0, null, null);
    try testing.expectApproxEqAbs(@as(f32, 3.0), one_ovr.vp.cfg, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 3.0), one_ovr.ap.cfg, 0.0); // audio follows video override

    // two-stage: cfg 3/7, rescale 0.7, modality 3.0, STG block 28 available
    const two = videoGuiderDefaults(.two_stage, null, null, null);
    try testing.expectApproxEqAbs(@as(f32, 3.0), two.vp.cfg, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 7.0), two.ap.cfg, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.7), two.vp.rescale, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 3.0), two.vp.modality, 0.0);
    try testing.expectEqual(@as(usize, 1), two.vp.stg_blocks.len);
    try testing.expectEqual(@as(u32, 28), two.vp.stg_blocks[0]);
    try testing.expect(!two.vp.needsPerturbed()); // stg defaults 0.0
    const two_stg = videoGuiderDefaults(.two_stage, null, null, 1.0);
    try testing.expect(two_stg.vp.needsPerturbed());

    // HQ: rescale 0.45 video / 1.0 audio, no STG blocks, 15 default steps
    const hq = videoGuiderDefaults(.two_stage_hq, null, null, null);
    try testing.expectApproxEqAbs(@as(f32, 0.45), hq.vp.rescale, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), hq.ap.rescale, 0.0);
    try testing.expectEqual(@as(usize, 0), hq.vp.stg_blocks.len);
    try testing.expectEqual(@as(u32, 15), hq.stage1_steps_default);
    const hq_stg = videoGuiderDefaults(.two_stage_hq, null, null, 1.0);
    try testing.expect(!hq_stg.vp.needsPerturbed()); // no blocks → no perturbed forward
}

test "a2vid pipeline gate: one-stage rejected, two-stage variants allowed" {
    try testing.expect(a2vidPipelineError(.one_stage) != null);
    try testing.expect(a2vidPipelineError(.two_stage) == null);
    try testing.expect(a2vidPipelineError(.two_stage_hq) == null);
}

test "a2vidMuxSampleCount trims to video duration and never exceeds the clip" {
    // 10 s stereo 48 kHz clip, 4 s of video (97 frames @ 24 fps ≈ 4.0417 s).
    const clip: usize = 10 * 48000 * 2;
    const n = a2vidMuxSampleCount(clip, 2, 48000, 97, 24.0);
    try testing.expectEqual(@as(usize, 194000 * 2), n); // floor(97/24*48000)*2
    // Clip shorter than the video → the whole clip, channel-aligned.
    try testing.expectEqual(@as(usize, 8000), a2vidMuxSampleCount(8000, 2, 48000, 97, 24.0));
    try testing.expectEqual(@as(usize, 8000), a2vidMuxSampleCount(8001, 2, 48000, 97, 24.0));
    // Degenerate inputs
    try testing.expectEqual(@as(usize, 0), a2vidMuxSampleCount(100, 0, 48000, 97, 24.0));
    try testing.expectEqual(@as(usize, 0), a2vidMuxSampleCount(100, 2, 48000, 97, 0.0));
}

test "LTX negative prompt keeps the reference audio negatives (speech guidance)" {
    // The audio tail of the reference DEFAULT_NEGATIVE_PROMPT does real work
    // for dialogue: with audio CFG active (two-stage, ap.cfg=7.0) these terms
    // push the soundtrack away from ambient noise toward clean speech. A
    // trimmed copy that keeps only the visual head silently weakens speech.
    // Overflow is safe: ltxPadWithBos left-truncates, keeping this tail.
    const audio_negatives = [_][]const u8{
        "mismatched lip sync",
        "silent or muted audio",
        "distorted voice",
        "robotic voice",
        "background noise",
        "off-sync audio",
        "incorrect dialogue",
        "added dialogue",
        "repetitive speech",
    };
    for (audio_negatives) |term| {
        try testing.expect(std.mem.indexOf(u8, LTX_NEGATIVE_PROMPT, term) != null);
    }
}

test "LTX negative prompt suppresses burned-in subtitles/captions (quoted-dialogue class)" {
    // Quoted dialogue in the prompt is LTX's speech trigger, but the model
    // also reads quotes as ON-SCREEN TEXT and burns scrambled subtitle-like
    // captions into the frame. These terms steer CFG away from that failure
    // (they only act when a guider runs the negative forward — two-stage, or
    // cfg > 1). They must sit BEFORE the audio tail so an overflow
    // left-truncation sheds them ahead of the load-bearing speech negatives;
    // the full prompt encodes to ~229 gemma tokens, under the 256 pad.
    const subtitle_negatives = [_][]const u8{
        "subtitles",
        "closed captions",
        "burned-in captions",
        "on-screen text",
        "text overlay",
        "lower thirds",
        "karaoke-style lyrics",
        "watermark",
    };
    for (subtitle_negatives) |term| {
        try testing.expect(std.mem.indexOf(u8, LTX_NEGATIVE_PROMPT, term) != null);
    }
    const first_audio = std.mem.indexOf(u8, LTX_NEGATIVE_PROMPT, "mismatched lip sync").?;
    for (subtitle_negatives) |term| {
        try testing.expect(std.mem.indexOf(u8, LTX_NEGATIVE_PROMPT, term).? < first_audio);
    }
}

test "fileExists guards non-absolute paths (openFileAbsolute UB class)" {
    const io = std.Io.Threaded.global_single_threaded.io();
    // Relative and empty paths must return false, not hit the stdlib assert.
    try testing.expect(!fileExists(io, "relative/path.safetensors"));
    try testing.expect(!fileExists(io, ""));
}

test "parseFloatList splits on commas/spaces and rejects garbage" {
    var buf: [16]f32 = undefined;
    const a = parseFloatList("1,2,3", &buf).?;
    try testing.expectEqual(@as(usize, 3), a.len);
    try testing.expectEqual(@as(f32, 2.0), a[1]);
    const b = parseFloatList("  0.5 1.25\t-2 ", &buf).?;
    try testing.expectEqual(@as(usize, 3), b.len);
    try testing.expectEqual(@as(f32, -2.0), b[2]);
    const c = parseFloatList("1, 2,, 3", &buf).?; // empty tokens skipped
    try testing.expectEqual(@as(usize, 3), c.len);
    try testing.expect(parseFloatList("1,x,3", &buf) == null);
    try testing.expect(parseFloatList("", &buf) == null);
    try testing.expect(parseFloatList("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17", &buf) == null); // > buf.len
}

test "img2imgStartStep maps strength onto the schedule (diffusers convention)" {
    // strength 1.0 → start at 0 (≈ full noise); 0.5 on 8 steps → skip 4;
    // tiny strength still runs at least 1 step.
    try testing.expectEqual(@as(u32, 0), img2imgStartStep(8, 1.0));
    try testing.expectEqual(@as(u32, 4), img2imgStartStep(8, 0.5));
    try testing.expectEqual(@as(u32, 6), img2imgStartStep(8, 0.25));
    try testing.expectEqual(@as(u32, 7), img2imgStartStep(8, 0.01));
    try testing.expectEqual(@as(u32, 2), img2imgStartStep(4, 0.6));
    try testing.expectEqual(@as(u32, 0), img2imgStartStep(1, 0.3));
}

test "fitRefDims preserves aspect, caps at ~1MP, rounds to multiples of 32, never upscales" {
    // 2:1 landscape above the cap → scaled down, aspect kept (±32-rounding).
    const a = fitRefDims(2000, 1000);
    try testing.expect(a.w * a.h <= 1024 * 1024);
    try testing.expectEqual(@as(u32, 0), a.w % 32);
    try testing.expectEqual(@as(u32, 0), a.h % 32);
    const ar: f64 = @as(f64, @floatFromInt(a.w)) / @as(f64, @floatFromInt(a.h));
    try testing.expect(ar > 1.85 and ar < 2.15);
    // Small image: no upscale, just 32-rounding down.
    const b = fitRefDims(300, 500);
    try testing.expectEqual(@as(u32, 288), b.w);
    try testing.expectEqual(@as(u32, 480), b.h);
    // Already-conforming square passes through.
    const c = fitRefDims(1024, 1024);
    try testing.expectEqual(@as(u32, 1024), c.w);
    try testing.expectEqual(@as(u32, 1024), c.h);
    // Degenerate tiny inputs stay valid (≥32).
    const d = fitRefDims(10, 3000);
    try testing.expect(d.w >= 32 and d.h >= 32);
}

test "decodeImageToBCHW covers with a center crop instead of stretching" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // 100x50 source: black | white(center 50 cols) | black. Covering a 50x50
    // target must sample ONLY the centered square window → all white.
    // (The old stretch mapped the full width → black bands at the sides.)
    const W = 100;
    const H = 50;
    var rgb: [W * H * 3]u8 = undefined;
    for (0..H) |y| for (0..W) |x| {
        const v: u8 = if (x >= 25 and x < 75) 255 else 0;
        const o = (y * W + x) * 3;
        rgb[o] = v;
        rgb[o + 1] = v;
        rgb[o + 2] = v;
    };
    const png_bytes = try png_mod.encodeRgb(a, &rgb, W, H);
    defer a.free(png_bytes);
    const arr = decodeImageToBCHW(a, png_bytes, 50, 50) orelse return error.DecodeFailed;
    defer _ = mlx.mlx_array_free(arr);
    _ = mlx.mlx_array_eval(arr);
    const d = mlx.mlx_array_data_float32(arr) orelse return error.NoData;
    const n: usize = @intCast(mlx.mlx_array_size(arr));
    var mean: f64 = 0;
    for (0..n) |i| mean += d[i];
    mean /= @floatFromInt(n);
    try testing.expect(mean > 0.9); // stretch gives ~0.5 here
}

test "extractCondWeights accepts a JSON array or a separated string" {
    var buf: [16]f32 = undefined;
    const a = extractCondWeights("{\"cond_weights\":[1, 2.5, -3]}", &buf).?;
    try testing.expectEqual(@as(usize, 3), a.len);
    try testing.expectEqual(@as(f32, 2.5), a[1]);
    const b = extractCondWeights("{\"cond_weights\":\"1 1 1 1\"}", &buf).?;
    try testing.expectEqual(@as(usize, 4), b.len);
    try testing.expect(extractCondWeights("{}", &buf) == null);
    try testing.expect(extractCondWeights("{\"cond_weights\":[1,bad]}", &buf) == null);
}
