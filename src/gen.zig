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
const mage_flow_mod = @import("mage_flow.zig");
const lora_mod = @import("lora.zig");
const tts = @import("tts.zig");
const acestep = @import("acestep.zig");
const music3 = @import("music3.zig");
const kokoro = @import("kokoro.zig");
const ltx = @import("ltx_video.zig");
const diffvae_fwd = @import("ltx_diffvae_forward.zig");
const ltx_audio = @import("ltx_audio.zig");
const minimax_h3 = @import("minimax_h3.zig");
const hy3d = @import("hunyuan3d.zig");
const hy3d_paint = @import("hunyuan3d_paint.zig");
const glb_mod = @import("glb.zig");
const wav_mod = @import("wav.zig");
const png_mod = @import("png.zig");
const tok_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const chat_mod = @import("chat.zig");
const log = @import("log.zig");
const metrics = @import("status.zig");
const sse = @import("gen_sse.zig");
const server_mod = @import("server.zig");
const multipart = @import("multipart.zig");
const discovery = @import("model_discovery.zig");
const stb = @import("stb");

const Conn = server_mod.Conn;

/// The three media-generation modalities. Detected from `config.json`'s
/// `model_type` and carried on the load path so the registry installs the
/// right engine slot and the server dispatches the right endpoint.
pub const Modality = enum {
    image,
    audio,
    video,
    mesh,

    pub fn capability(self: Modality) []const u8 {
        return switch (self) {
            .image => "image",
            .audio => "audio",
            .video => "video",
            .mesh => "3d",
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
            .mesh => "hunyuan3d_2_1",
        };
    }
};

/// Classify a `model_type` string into a media modality, or null for a
/// regular LM/embedding arch. Pure — the load arms dispatch on this off the
/// (stub) config's `model_type`, so it must accept the markers from
/// `Modality.modelType` AND the raw config strings discovery peeks
/// ("flux2-klein-4b", "qwen3_tts", "AudioVideo").
/// Every media `model_type` this server serves. The ONE list the two
/// duplicated predicates below are checked against.
///
/// `model_discovery.isMediaModelType` cannot call `modalityFromType` — that
/// module stays filesystem-only so it never pulls in mlx — so the duplication
/// is deliberate and documented. What was missing was a guard: `minimax_h3`
/// was registered here and NOT there, so discovery rejected the model with
/// "unsupported model_type" while the engine that serves it was ready and
/// waiting. The test at the bottom of this file pins them together.
pub const media_model_types = [_][]const u8{
    "flux2",     "krea",       "mage_flow",      "mageflow",
    "qwen3_tts", "acestep",    "kokoro",         "AudioVideo",
    "hunyuan3d", "minimax_h3", "minimax_music3",
};

pub fn modalityFromType(model_type: []const u8) ?Modality {
    if (std.mem.startsWith(u8, model_type, "flux2")) return .image;
    if (std.mem.startsWith(u8, model_type, "krea")) return .image;
    if (std.mem.startsWith(u8, model_type, "mage_flow") or std.mem.eql(u8, model_type, "mageflow")) return .image;
    if (std.mem.eql(u8, model_type, "qwen3_tts")) return .audio;
    if (std.mem.eql(u8, model_type, "acestep")) return .audio;
    if (std.mem.eql(u8, model_type, "minimax_music3")) return .audio;
    if (std.mem.eql(u8, model_type, "kokoro")) return .audio;
    if (std.mem.eql(u8, model_type, "AudioVideo")) return .video;
    if (std.mem.eql(u8, model_type, "minimax_h3")) return .video;
    if (std.mem.startsWith(u8, model_type, "hunyuan3d")) return .mesh;
    return null;
}

/// Endpoint-level media route. `.speech` and `.music` share the `.audio`
/// modality/engine slot — the loaded `AudioBackend` arm decides which endpoint
/// is valid (wrong pairing → explicit 400, never a silent misinterpretation).
pub const GenRoute = enum {
    image,
    speech,
    music,
    video,
    mesh,

    pub fn modality(self: GenRoute) Modality {
        return switch (self) {
            .image => .image,
            .speech, .music => .audio,
            .video => .video,
            .mesh => .mesh,
        };
    }
};

/// Which audio backend a `model_type` selects (pure; pins the dispatch the
/// `AudioEngine.load` re-peek performs).
pub fn audioBackendKindForType(model_type: []const u8) AudioBackendKind {
    if (std.mem.eql(u8, model_type, "acestep")) return .music;
    if (std.mem.eql(u8, model_type, "minimax_music3")) return .music3;
    if (std.mem.eql(u8, model_type, "kokoro")) return .kokoro;
    return .tts;
}

/// Which arm of `AudioBackend` a checkpoint loads into. `.tts` (Qwen3-TTS) and
/// `.kokoro` both serve `/v1/audio/speech` but have DISJOINT controls: Qwen3-TTS
/// clones from `ref_audio` and has no voice list, Kokoro has 54 named blendable
/// voices and no cloning. The handler refuses the wrong control rather than
/// ignoring it.
pub const AudioBackendKind = enum {
    tts,
    music,
    music3,
    kokoro,

    /// Music-generation backends serve /v1/audio/music-generations and
    /// advertise "music" beside "audio"; the TTS arms never do.
    pub fn servesMusic(self: AudioBackendKind) bool {
        return self == .music or self == .music3;
    }
};

/// Peek `model_dir/config.json` for its `model_type` string (owned dupe, caller
/// frees) or null on any read/parse error. Cheap — used both to route to a media
/// modality and to pick the image backend (FLUX vs Krea).
pub fn peekModelType(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    // Guard the openFileAbsolute assert (ReleaseFast UB on relative/empty paths).
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null;
    if (readConfigModelType(io, allocator, model_dir)) |mt| return mt;
    // Diffusers-style repos (Mage-Flow) have no root config.json / model_type —
    // the pipeline identity lives in model_index.json's `_class_name`. Synthesize
    // the "mage_flow" marker so routing + the backend dispatch light up.
    if (isMageFlowRepo(io, allocator, model_dir)) return allocator.dupe(u8, "mage_flow") catch null;
    // Same for an mflux FLUX.2 conversion with no config.json at all (the only
    // MLX build of klein 9B). Identified by the DiT's own weight names, through
    // the SAME predicate discovery uses — a private copy here is how `list` and
    // the loader end up disagreeing about whether a dir is a model.
    if (isMfluxFlux2Repo(io, allocator, model_dir)) return allocator.dupe(u8, "flux2-klein") catch null;
    return null;
}

/// True when `model_dir` holds FLUX.2 DiT weights but no config.json to say so.
/// Thin path→Dir wrapper over `model_discovery.peekMfluxFlux2`.
fn isMfluxFlux2Repo(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return discovery.peekMfluxFlux2(io, allocator, dir);
}

/// Read `model_dir/config.json`'s `model_type` (owned dupe) or null on any
/// read/parse error or when the field is absent.
fn readConfigModelType(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
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

/// True when `model_dir/model_index.json` marks a MageFlow pipeline (its
/// `_class_name` is "MageFlowPipeline", or the `_mage_flow_version` tag exists).
fn isMageFlowRepo(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const path = std.fmt.allocPrint(allocator, "{s}/model_index.json", .{model_dir}) catch return false;
    defer allocator.free(path);
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1024 * 1024)) catch return false;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    if (parsed.value.object.get("_mage_flow_version") != null) return true;
    const cn = parsed.value.object.get("_class_name") orelse return false;
    return cn == .string and std.mem.eql(u8, cn.string, "MageFlowPipeline");
}

/// Classify a model dir into a media modality (reads its `model_type`), or null
/// for a regular LM/embedding arch. The video (LTX "AudioVideo") branch
/// additionally requires `connector.safetensors` so a generic "AudioVideo"
/// config without the LTX bundle isn't misrouted.
/// A file that must be present for `model_type` to be accepted as that media
/// backend, or null when the `model_type` alone is sufficient.
///
/// This is keyed on the TYPE, not the modality. It used to be
/// `if (modality == .video) require connector.safetensors` — a marker that
/// belongs to LTX only. The moment a second video backend existed, that guard
/// rejected it, `detectModality` returned null, and the loader fell through to
/// the MLX TEXT path: it globbed all four of MiniMax-H3's safetensors into one
/// weight map and died on `model.norm.weight`. A per-MODALITY guard cannot
/// survive a modality growing a second backend.
pub fn requiredMarkerFor(model_type: []const u8) ?[]const u8 {
    // The table lives in model_discovery (fs-only, so discovery and
    // register-by-path apply the SAME completeness rule) — this module can
    // import that one, just not the other way around.
    return discovery.requiredMediaMarker(model_type);
}

pub fn detectModality(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?Modality {
    const mt = peekModelType(io, allocator, model_dir) orelse return null;
    defer allocator.free(mt);
    const modality = modalityFromType(mt) orelse return null;
    if (requiredMarkerFor(mt)) |marker| {
        const p = std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, marker }, 0) catch return null;
        defer allocator.free(p);
        if (!fileExists(io, p)) {
            log.warn("[gen] {s} at {s} is missing {s}; not treating it as a media model\n", .{ mt, model_dir, marker });
            return null;
        }
    }
    return modality;
}

/// True when `model_dir` declares a media `model_type` whose required marker
/// is missing — an incomplete pack (an in-flight or interrupted download, or
/// a stray fragment). The load path refuses these BY NAME: falling through to
/// the text loader globs whatever safetensors ARE present and dies on the
/// first missing weight (`unreachable` in ReleaseFast — live 2026-08-08, a
/// turbo-lora fragment killed the server on a plain Generate).
pub fn incompleteMediaDir(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const mt = peekModelType(io, allocator, model_dir) orelse return false;
    defer allocator.free(mt);
    if (modalityFromType(mt) == null) return false;
    const marker = requiredMarkerFor(mt) orelse return false;
    const p = std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, marker }, 0) catch return false;
    defer allocator.free(p);
    return !fileExists(io, p);
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
    /// Text encoder — nullable because LOW-MEM mode (iPhone) loads it lazily
    /// per request and frees it right after the prompt encode: it's ~half the
    /// pipeline's resident bytes but runs exactly one forward per generation.
    te: ?flux.TextEncoder,
    dit: flux.Dit,
    vae: flux.Vae,
    vae_enc: ?flux.VaeEncoder,
    tok: tok_mod.Tokenizer,
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []u8,
    low_mem: bool,

    /// Low-mem policy, pure for testing: iOS always (jetsam ceilings);
    /// MLXSERVE_LOWMEM=1/0 forces either way; otherwise AUTO on machines with
    /// ≤ 16 GB of RAM — measured cost is ~0.1–0.3 s per image (the encoder
    /// mmap-reloads from page cache) vs ~1.8 GB lower peak, a clear win when
    /// the Metal working-set ceiling is ~12 GB (16 GB mini class).
    fn lowMemFromInputs(is_ios: bool, env: ?[]const u8, total_ram_bytes: u64) bool {
        if (is_ios) return true;
        if (env) |e| {
            if (std.mem.eql(u8, e, "1")) return true;
            if (std.mem.eql(u8, e, "0")) return false;
        }
        return total_ram_bytes > 0 and total_ram_bytes <= 17 * 1024 * 1024 * 1024;
    }

    fn lowMemDefault() bool {
        const env: ?[]const u8 = if (std.c.getenv("MLXSERVE_LOWMEM")) |v| std.mem.sliceTo(v, 0) else null;
        return lowMemFromInputs(@import("build_options").ios, env, metrics.getTotalMemBytes());
    }

    fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !FluxImpl {
        var self: FluxImpl = undefined;
        self.io = io;
        self.allocator = allocator;
        self.low_mem = lowMemDefault();
        self.model_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(self.model_dir);
        self.s = mlx.mlx_default_gpu_stream_new();
        if (self.low_mem) {
            self.te = null;
            log.info("[image] FLUX low-mem mode: text encoder loads per request\n", .{});
        } else {
            self.te = try flux.loadTextEncoder(io, allocator, self.s, model_dir);
        }
        errdefer if (self.te) |*t| t.deinit();
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
        if (self.te) |*t| t.deinit();
        self.dit.deinit();
        self.vae.deinit();
        if (self.vae_enc) |*e| e.deinit();
        self.tok.deinit();
        self.allocator.free(self.model_dir);
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
        var ref_lats: [MAX_EDIT_IMAGES]mlx.mlx_array = undefined;
        var ref_lat_n: usize = 0;
        defer for (ref_lats[0..ref_lat_n]) |l| {
            _ = mlx.mlx_array_free(l);
        };
        if (opts.edit_images.len > 0) {
            // Instruction edit: clean in-context references, full noise start.
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            if (opts.edit_images.len > MAX_EDIT_IMAGES) return error.TooManyEditImages;
            for (opts.edit_images) |pix| {
                ref_lats[ref_lat_n] = try ve.encode(pix);
                ref_lat_n += 1;
            }
            fopts.ref_latents = ref_lats[0..ref_lat_n];
        } else if (opts.init_image) |pix| {
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            init_lat = try ve.encode(pix);
            fopts.init_latents = init_lat;
            fopts.start_step = img2imgStartStep(steps, opts.strength);
        }
        // Phased text encoder: encode the prompt (materialized inside
        // encodePrompt — mlx laziness would otherwise pin the weights), then
        // in low-mem mode free the encoder + its cache before the denoise
        // loop. Same math either way: the conditioning tensor is already
        // computed, so outputs are byte-identical to the resident-TE path
        // (pinned by tests/test_flux_lowmem.sh).
        if (self.te == null) {
            self.te = try flux.loadTextEncoder(self.io, self.allocator, self.s, self.model_dir);
        }
        const cond = try flux.encodePrompt(&self.te.?, ids, mask, fopts);
        if (self.low_mem) {
            self.te.?.deinit();
            self.te = null;
            _ = mlx.mlx_clear_cache();
            log.info("[image] low-mem: text encoder freed after encode\n", .{});
        }
        return flux.generateFromCondWithOpts(&self.dit, &self.vae, cond, ids.len, seed, steps, height, width, fopts, progress);
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
    mage_flow: *mage_flow_mod.Engine,
};

/// Most reference images an edit request may carry (the primary 'image' plus
/// extra 'ref_images'). Each ~1MP reference adds ~4096 DiT tokens, so the cap
/// bounds attention memory; the official sampler tops out around 10.
pub const MAX_EDIT_IMAGES = 4;

/// Ceiling on one video response's raw RGB volume. The whole `frames.rgb`
/// buffer is base64'd into ONE JSON body and the app decodes it in memory;
/// past this the client drops the socket and a finished render dies as
/// `WriteFailed` (#283: 5 chained windows = 701f at 1056x864 = 1.9 GB).
/// Twin of the app's `VideoModelPreset.maxFramePayloadBytes`; refuse at
/// admission, by name, before any denoise step.
pub const MAX_VIDEO_RGB_BYTES: u64 = 768 * 1024 * 1024;

pub fn videoRgbTransportReason(delivered_frames: u32, width: u32, height: u32) ?[]const u8 {
    const bytes: u64 = @as(u64, delivered_frames) * @as(u64, width) * @as(u64, height) * 3;
    if (bytes <= MAX_VIDEO_RGB_BYTES) return null;
    const S = struct {
        var buf: [192]u8 = undefined;
    };
    return std.fmt.bufPrint(&S.buf, "{d} frames at {d}x{d} is {d} MB of raw RGB; one response carries at most {d} MB — fewer frames or windows, or a smaller canvas", .{ delivered_frames, width, height, bytes / (1024 * 1024), MAX_VIDEO_RGB_BYTES / (1024 * 1024) }) catch "video too large for one response";
}


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
    /// Multiple entries (the edited source first, then extra references) each
    /// ride at their own t offset: "replace the face in image 1 with the face
    /// from image 2". Empty = not an edit.
    edit_images: []const mlx.mlx_array = &.{},
    /// Raw reference bytes (PNG/JPEG) for backends that do their OWN edit
    /// preprocessing rather than consuming pre-decoded `edit_images` — MageFlow
    /// needs a target-size VAE resize AND a separate 384/smart-resize for its
    /// vision tower, so it resizes from the source bytes. Primary first. Empty =
    /// not a MageFlow edit.
    edit_image_bytes: []const []const u8 = &.{},
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
    // Runtime LoRA state: the Stack owns the adapter arrays the attached
    // Refs point at, so it must live until the next detach (clearLora).
    lora_stack: ?lora_mod.Stack = null,
    lora_matched: u32 = 0,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*ImageEngine {
        const self = try allocator.create(ImageEngine);
        errdefer allocator.destroy(self);
        self.* = .{ .allocator = allocator, .backend = undefined };
        // Re-peek the arch to pick the backend (detectModality already proved the
        // config parses). `mage_flow*` → MageFlow; `krea*` → Krea; else FLUX.
        if (peekModelType(io, allocator, model_dir)) |mt| {
            defer allocator.free(mt);
            if (std.mem.startsWith(u8, mt, "mage_flow") or std.mem.eql(u8, mt, "mageflow")) {
                self.backend = .{ .mage_flow = try mage_flow_mod.Engine.load(io, allocator, model_dir) };
                return self;
            }
            if (std.mem.startsWith(u8, mt, "krea")) {
                self.backend = .{ .krea = try krea.Engine.load(io, allocator, model_dir) };
                return self;
            }
        }
        self.backend = .{ .flux = try FluxImpl.load(io, allocator, model_dir) };
        return self;
    }

    pub fn deinit(self: *ImageEngine) void {
        self.clearLora();
        switch (self.backend) {
            .flux => |*f| f.deinit(),
            .krea => |k| k.deinit(),
            .mage_flow => |m| m.deinit(),
        }
        self.allocator.destroy(self);
    }

    fn stream(self: *ImageEngine) mlx.mlx_stream {
        return switch (self.backend) {
            .flux => |*f| f.s,
            .krea => |k| k.s,
            .mage_flow => |m| m.s,
        };
    }

    /// Number of tapped text-encoder layers `cond_weights` must cover.
    pub fn condWeightCount(self: *const ImageEngine) usize {
        return switch (self.backend) {
            .flux => 3,
            .krea => 12,
            .mage_flow => 0, // conditioning-rebalance not wired for MageFlow yet
        };
    }

    /// True when the VAE encoder loaded, i.e. img2img is available.
    pub fn supportsImg2Img(self: *const ImageEngine) bool {
        return switch (self.backend) {
            .flux => |*f| f.vae_enc != null,
            .krea => |k| k.vae_enc != null,
            .mage_flow => false, // img2img lands with the MageFlow VAE encoder
        };
    }

    /// True when instruction editing (in-context reference conditioning) is
    /// available — a trained FLUX.2 capability; Krea has no edit training.
    pub fn supportsEdit(self: *const ImageEngine) bool {
        return switch (self.backend) {
            .flux => |*f| f.vae_enc != null,
            .krea => false,
            .mage_flow => |m| m.supportsEdit(), // Mage-Flow-Edit-Turbo checkpoint
        };
    }

    /// True when the edit backend consumes RAW reference bytes (`edit_image_bytes`)
    /// and does its own resizing, rather than pre-decoded `edit_images`. MageFlow
    /// needs a target-size VAE resize AND a separate VLM resize per reference.
    pub fn editUsesRawBytes(self: *const ImageEngine) bool {
        return self.backend == .mage_flow;
    }

    /// Reconcile the engine's attached LoRA stack with the request: an empty
    /// `paths` detaches; the same paths+scales (same order) is a no-op
    /// reuse; anything else clears and re-attaches every adapter. Mirrors
    /// mflux's `lora_paths`/`lora_scales`: adapters are summed at forward
    /// time (`lora.deltaSum`), not merged into the base weight. Returns the
    /// total number of (module, adapter) attachments across the stack.
    pub fn setLoras(self: *ImageEngine, paths: []const []const u8, scales: []const f32) !u32 {
        if (paths.len == 0) {
            self.clearLora();
            return 0;
        }
        if (paths.len > lora_mod.MAX_LORAS) return error.TooManyLoras;
        if (self.lora_stack) |*st| {
            if (st.matches(paths, scales)) return self.lora_matched;
        }
        self.clearLora();
        var stack: lora_mod.Stack = .{ .allocator = self.allocator };
        errdefer stack.deinit();
        for (paths, scales) |p, sc| {
            const dup_p = try self.allocator.dupe(u8, p);
            errdefer self.allocator.free(dup_p);
            const arch: lora_mod.Arch = switch (self.backend) {
                .flux => .flux2,
                .krea => .krea2,
                .mage_flow => .generic,
            };
            const lf = try lora_mod.loadFile(self.allocator, p, arch);
            stack.files[stack.count] = lf;
            stack.paths[stack.count] = dup_p;
            stack.scales[stack.count] = sc;
            stack.count += 1;
        }
        const matched = switch (self.backend) {
            .flux => |*f| flux.attachLora(&f.dit, &stack),
            .krea => |k| krea.attachLora(&k.dit, &stack),
            .mage_flow => 0, // MageFlow does not support LoRA (matches mflux)
        };
        if (matched == 0) {
            stack.deinit();
            return error.LoraNoMatch;
        }
        self.lora_stack = stack;
        self.lora_matched = matched;
        return matched;
    }

    fn clearLora(self: *ImageEngine) void {
        switch (self.backend) {
            .flux => |*f| flux.detachLora(&f.dit),
            .krea => |k| krea.detachLora(&k.dit),
            .mage_flow => {}, // no LoRA attached
        }
        if (self.lora_stack) |*st| st.deinit();
        self.lora_stack = null;
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
                if (opts.edit_images.len != 0) break :blk error.EditUnsupported;
                const kopts = krea.GenOpts{
                    .init_image = opts.init_image,
                    .start_step = if (opts.init_image != null) img2imgStartStep(steps, opts.strength) else 0,
                    .cond_gain = opts.cond_gain,
                    .cond_weights = opts.cond_weights,
                };
                break :blk k.generateImageOpts(allocator, prompt, width, height, seed, steps, kopts, progress);
            },
            // Turbo txt2img (guidance 1.0, no CFG), or the multi-reference edit
            // when the request carries reference images (Edit checkpoint only).
            .mage_flow => |m| if (opts.edit_image_bytes.len != 0)
                m.editImage(allocator, prompt, opts.edit_image_bytes, width, height, seed, steps, progress)
            else
                m.generateImage(allocator, prompt, width, height, seed, steps, progress),
        };
    }

    /// Encode an image [1,3,H,W] f32 [0,1] → PNG bytes (caller frees).
    pub fn toPng(self: *ImageEngine, allocator: std.mem.Allocator, img: mlx.mlx_array) ![]u8 {
        return krea.imageToPng(allocator, img, self.stream());
    }

    /// Resolve a requested WxH per backend. FLUX (klein) honors any multiple
    /// of 32 in [256, 1536] — its patchify/VAE are shape-derived (pinned by
    /// the non-square edit round-trip test), and smaller grids are the
    /// activation-memory lever that lets 8 GB iPhones generate at all.
    /// Krea accepts any multiple of 16 in [256, 2048].
    pub fn normalizeSize(self: *const ImageEngine, req_w: u32, req_h: u32) struct { w: u32, h: u32 } {
        return switch (self.backend) {
            .flux => .{ .w = clampFluxDim(req_w), .h = clampFluxDim(req_h) },
            .krea => .{ .w = clampKreaDim(req_w), .h = clampKreaDim(req_h) },
            // MageFlow is native-resolution, VAE downsample 16 → multiples of 16.
            .mage_flow => .{ .w = clampKreaDim(req_w), .h = clampKreaDim(req_h) },
        };
    }

    /// The ceiling `normalizeSize` clamps a dimension to. The edit path needs it
    /// as a NUMBER, not as a clamp: clamping width and height INDEPENDENTLY
    /// discards the aspect ratio, which is how a 4032x3024 phone photo came back
    /// 2048x2048. Pinned to the clamps themselves by a drift test.
    pub fn maxDimFor(kind: std.meta.Tag(ImageBackend)) u32 {
        return switch (kind) {
            .flux => 1536,
            .krea, .mage_flow => 2048,
        };
    }

    pub fn maxDim(self: *const ImageEngine) u32 {
        return maxDimFor(std.meta.activeTag(self.backend));
    }
};

/// Round a requested dimension to a multiple of 32 in [256, 1536] (klein's
/// crop granularity — the same /32 rule fitRefDims uses; ~1MP trained scale,
/// 1536 covers the widest preset edge). 0/omitted → the 1024 default.
pub fn clampFluxDim(v: u32) u32 {
    if (v == 0) return 1024;
    const rounded = ((v + 31) / 32) * 32;
    return std.math.clamp(rounded, 256, 1536);
}

/// Round a requested dimension to a multiple of 16 in [256, 2048] (Krea's
/// VAE ×8 + DiT patch ×2 alignment).
fn clampKreaDim(v: u32) u32 {
    const rounded = ((v + 15) / 16) * 16;
    return std.math.clamp(rounded, 256, 2048);
}

/// The audio modality hosts MULTIPLE architectures (the `ImageBackend`
/// convention): Qwen3-TTS speech synthesis and ACE-Step music generation.
pub const AudioBackend = union(enum) {
    tts: tts.Synthesizer,
    music: *acestep.Engine,
    music3: *music3.Engine,
    kokoro: *kokoro.Engine,
};

/// Audio engine — a tagged-union owner, dispatched on `config.json`'s
/// `model_type` at load (`qwen3_tts` → TTS, `acestep` → music). The
/// `LoadedModel.audio_engine` slot stays single + modality-named.
pub const AudioEngine = struct {
    allocator: std.mem.Allocator,
    backend: AudioBackend,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*AudioEngine {
        const self = try allocator.create(AudioEngine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        const mt = peekModelType(io, allocator, model_dir);
        defer if (mt) |m| allocator.free(m);
        if (mt != null and audioBackendKindForType(mt.?) == .music) {
            self.backend = .{ .music = try acestep.Engine.load(io, allocator, model_dir, FluxImpl.lowMemDefault()) };
            log.info("[audio] ACE-Step music engine ready\n", .{});
            return self;
        }
        if (mt != null and audioBackendKindForType(mt.?) == .music3) {
            self.backend = .{ .music3 = try music3.Engine.load(io, allocator, model_dir) };
            log.info("[audio] MiniMax Music 3 engine ready\n", .{});
            return self;
        }
        if (mt != null and audioBackendKindForType(mt.?) == .kokoro) {
            const ks = mlx.mlx_default_gpu_stream_new();
            self.backend = .{ .kokoro = try kokoro.Engine.load(io, allocator, model_dir, ks) };
            log.info("[audio] Kokoro TTS ready (sample_rate={d})\n", .{self.backend.kokoro.sampleRate()});
            return self;
        }
        const s = mlx.mlx_default_gpu_stream_new();
        self.backend = .{ .tts = try tts.Synthesizer.load(io, allocator, s, model_dir) };
        log.info("[audio] TTS synthesizer ready (sample_rate={d})\n", .{self.backend.tts.model.cfg.sample_rate});
        return self;
    }

    pub fn deinit(self: *AudioEngine) void {
        switch (self.backend) {
            .tts => |*synth| synth.deinit(),
            .music => |e| e.deinit(),
            .music3 => |e| e.deinit(),
            .kokoro => |e| e.deinit(),
        }
        self.allocator.destroy(self);
    }
};

/// Mesh backend (currently Hunyuan3D-2.1 shape). Thin owner of the hunyuan3d
/// engine — the DINO conditioner, DiT, and ShapeVAE decoder live in
/// `src/hunyuan3d.zig` (mirrors `AudioEngine` over `tts.Synthesizer`). When a
/// second 3D arch arrives this becomes an `ImageBackend`-style tagged union.
pub const MeshEngine = struct {
    allocator: std.mem.Allocator,
    engine: *hy3d.Engine,
    /// P2 paint (texture) stage dir, discovered lazily beside the shape model
    /// (SIBLING dir `<models root>/local/hunyuan3d-2-1-paint-8bit` or the
    /// `HY3D_PAINT_DIR` override). Null → `"texture": true` requests get a 400.
    /// The paint engine itself loads per-request and frees after (memory
    /// staging: shape 3.5 GB + paint ~4.6 GB never both need residency —
    /// the shape stage completes before the paint stage starts).
    paint_dir: ?[]u8 = null,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*MeshEngine {
        const self = try allocator.create(MeshEngine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.paint_dir = null;
        self.engine = try hy3d.Engine.load(io, allocator, model_dir);
        self.paint_dir = findPaintDir(allocator, model_dir);
        if (self.paint_dir) |p| log.info("[mesh] paint (texture) weights available: {s}\n", .{p});
        log.info("[mesh] Hunyuan3D shape engine ready\n", .{});
        return self;
    }

    pub fn deinit(self: *MeshEngine) void {
        if (self.paint_dir) |p| self.allocator.free(p);
        self.engine.deinit();
        self.allocator.destroy(self);
    }
};

/// Locate the paint-stage model dir: `HY3D_PAINT_DIR` env override, else the
/// combined single-HF-repo layout `<shape_dir>/paint`, else the converted
/// sibling `<parent-of-shape-dir>/hunyuan3d-2-1-paint-8bit` (the local
/// convert script writes next to the shape dir). Returns null (graceful)
/// when absent.
fn findPaintDir(allocator: std.mem.Allocator, shape_dir: []const u8) ?[]u8 {
    return findStageModelDir(allocator, shape_dir, "paint", "hunyuan3d-2-1-paint-8bit", "HY3D_PAINT_DIR");
}

/// Shared stage-model discovery, in priority order:
///   1. `env_var` override (absolute + has a config.json) — debugging seam;
///      when set, it is the ONLY candidate (no silent fallback).
///   2. `<shape_dir>/<subdir_name>` — the combined single-HF-repo layout
///      (shape at the root, stage weights in subdirs; ONE download).
///   3. `<parent-of-shape-dir>/<sibling_name>` — the local convert-script
///      layout (three sibling dirs under `.../local/`).
fn findStageModelDir(allocator: std.mem.Allocator, shape_dir: []const u8, subdir_name: []const u8, sibling_name: []const u8, env_var: [*:0]const u8) ?[]u8 {
    if (std.c.getenv(env_var)) |v| {
        const p = std.mem.span(v);
        if (p.len > 0 and std.fs.path.isAbsolute(p) and dirHasConfig(p)) {
            return allocator.dupe(u8, p) catch null;
        }
        return null;
    }
    if (std.fs.path.join(allocator, &.{ shape_dir, subdir_name })) |sub| {
        if (dirHasConfig(sub)) return sub;
        allocator.free(sub);
    } else |_| {}
    const parent = std.fs.path.dirname(shape_dir) orelse return null;
    const sib = std.fs.path.join(allocator, &.{ parent, sibling_name }) catch return null;
    if (dirHasConfig(sib)) return sib;
    allocator.free(sib);
    return null;
}

fn dirHasConfig(dir: []const u8) bool {
    if (dir.len == 0 or !std.fs.path.isAbsolute(dir)) return false; // openDirAbsolute UB guard
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const cfg = std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}) catch return false;
    const io = std.Io.Threaded.global_single_threaded.io();
    const f = std.Io.Dir.openFileAbsolute(io, cfg, .{}) catch return false;
    f.close(io);
    return true;
}

// The reference's `TOKENIZER_MAX_LENGTH` (ltx_core/text_encoders/gemma/
// gemma_assets.py), for 2.3 and 2.5 alike. `LTXGemmaTokenizer` DEFAULTS to 256,
// but no caller uses that default — `build_gemma_tokenizer` passes 1024, and the
// encoder's own docstring says "the tokenizer pads every prompt to max_length
// (1024)". We shipped 256 from the 2.3 port onwards, which left the connector
// tiling its 128 learnable registers over 2 slots instead of 8 and gave the DiT
// a quarter of the text rows it was trained to cross-attend to.
const LTX_PAD_LEN: usize = 1024; // gemma left-pad length
const LTX_PAD_ID: i32 = 0; // gemma <pad>
const LTX_GEMMA_BOS: i32 = 2; // <bos>

// Reference DEFAULT_NEGATIVE_PROMPT, plus a subtitle/caption block after
// "artifacts around text": quoted dialogue in the prompt makes the model burn
// scrambled subtitle-like captions into the frame; these terms steer CFG away
// from that. The audio tail (lip sync, muted/distorted voice, background
// noise, dialogue terms) is load-bearing for speech when audio CFG runs; if
// the whole thing ever exceeds LTX_PAD_LEN (~229 tokens today, 1024 budget),
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
/// The `.video` modality slot, one arm per backend — the same shape
/// `ImageEngine` uses for flux|krea|mage_flow. Adding a backend is one arm plus
/// an impl file; every call site holds `*VideoEngine` and dispatches here.
pub const VideoEngine = struct {
    allocator: std.mem.Allocator,
    backend: union(enum) {
        ltx: *LtxVideoEngine,
        h3: *H3VideoEngine,
    },

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*VideoEngine {
        const self = try allocator.create(VideoEngine);
        errdefer allocator.destroy(self);
        self.* = .{ .allocator = allocator, .backend = undefined };
        if (peekModelType(io, allocator, model_dir)) |mt| {
            defer allocator.free(mt);
            if (std.mem.eql(u8, mt, "minimax_h3")) {
                self.backend = .{ .h3 = try H3VideoEngine.load(io, allocator, model_dir) };
                return self;
            }
        }
        self.backend = .{ .ltx = try LtxVideoEngine.load(io, allocator, model_dir) };
        return self;
    }

    pub fn deinit(self: *VideoEngine) void {
        switch (self.backend) {
            .ltx => |e| e.deinit(),
            .h3 => |e| e.deinit(),
        }
        self.allocator.destroy(self);
    }

    /// LoRA is an LTX-only capability; H3 ships no adapter format, so this is a
    /// NAMED refusal rather than a silent no-op that reports a match count of 0.
    pub fn setLora(self: *VideoEngine, path: ?[]const u8, scale: f32) !u32 {
        return switch (self.backend) {
            .ltx => |e| e.setLora(path, scale),
            .h3 => if (path == null) 0 else error.LoraUnsupported,
        };
    }
};

/// `h3ConfigDeclaresRef2va` over a model dir's own `config.json`.
fn h3DirDeclaresRef2va(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return false;
    const path = std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir}) catch return false;
    defer allocator.free(path);
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1024 * 1024)) catch return false;
    defer allocator.free(content);
    return h3ConfigDeclaresRef2va(allocator, content);
}

/// True when a MiniMax-H3 `config.json` declares the `ref2va` task. The two
/// partitions share the text encoder, both VAEs and every geometry number, so
/// nothing about the files themselves tells them apart — only the converter's
/// declared task list does, and an FL2VA pack handed references would generate
/// while silently ignoring them.
fn h3ConfigDeclaresRef2va(allocator: std.mem.Allocator, config_json: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, config_json, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const tasks = parsed.value.object.get("tasks") orelse return false;
    if (tasks != .array) return false;
    for (tasks.array.items) |t| {
        if (t == .string and std.mem.eql(u8, t.string, "ref2va")) return true;
    }
    return false;
}

/// MiniMax-H3 video+audio. Holds only paths: `minimax_h3.generate` stages the
/// text encoder and the DiT sequentially because they cannot both be resident,
/// so there is nothing useful to keep loaded between requests.
pub const H3VideoEngine = struct {
    allocator: std.mem.Allocator,
    model_dir: []u8,
    /// Whether this pack is the REF2VA partition (read from its config's task
    /// list at load — the file layout is identical either way).
    supports_refs: bool = false,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*H3VideoEngine {
        const self = try allocator.create(H3VideoEngine);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .model_dir = try allocator.dupe(u8, model_dir),
            .supports_refs = h3DirDeclaresRef2va(io, allocator, model_dir),
        };
        return self;
    }

    pub fn deinit(self: *H3VideoEngine) void {
        self.allocator.free(self.model_dir);
        self.allocator.destroy(self);
    }

    pub fn paths(self: *const H3VideoEngine, a: std.mem.Allocator) !minimax_h3.GenPaths {
        return .{
            .tokenizer_dir = self.model_dir,
            .text_encoder = try std.fmt.allocPrint(a, "{s}/text_encoder.safetensors", .{self.model_dir}),
            .dit = try std.fmt.allocPrint(a, "{s}/transformer.safetensors", .{self.model_dir}),
            .vae = try std.fmt.allocPrint(a, "{s}/video_vae.safetensors", .{self.model_dir}),
            .audio_vae = try std.fmt.allocPrint(a, "{s}/audio_vae.safetensors", .{self.model_dir}),
            .turbo_lora = try std.fmt.allocPrint(a, "{s}/turbo_lora.safetensors", .{self.model_dir}),
        };
    }

    /// Whether the pack ships the Turbo LoRA. Probed per request, not cached
    /// at engine load: the 744 MB file can land in the folder while the
    /// server is up, and a stale "no" would 400 a capability that exists.
    pub fn hasTurboLora(self: *const H3VideoEngine, io: std.Io, a: std.mem.Allocator) bool {
        const p = std.fs.path.join(a, &.{ self.model_dir, "turbo_lora.safetensors" }) catch return false;
        defer a.free(p);
        const st = std.Io.Dir.cwd().statFile(io, p, .{}) catch return false;
        return st.size > 0;
    }
};

pub const LtxVideoEngine = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    transformer: ltx.Component,
    transformer_variant: TransformerVariant,
    connector: ltx.Component,
    vae: ltx.Component,
    audio: ?ltx.Component = null, // audio VAE + vocoder; null → video has no sound
    vae_encoder: ?ltx.Component = null, // image VAE encoder; null → image-to-video + two-stage disabled
    upsampler: ?ltx.Component = null, // spatial x2 latent upsampler; lazy-loaded for two-stage
    // LTX's own DiffVAE decoder — what their published clips are decoded with.
    // Lazy like the upsampler: the 4-bit pack does not ship it, and a request
    // that never asks for it must not pay 0.83 GB.
    diffusion_decoder: ?ltx.Component = null,
    tok: tok_mod.Tokenizer,
    gemma_dir: []u8,
    model_dir: []u8,
    /// Which LTX release this pack is, read from its own config.json at load.
    /// It decides the text encoder (2.5 ships its own, in-pack) and the two
    /// DiT-side differences; the geometry is shared.
    ltx_cfg: ltx.LtxConfig = .{},
    // Runtime LoRA state (mirrors ImageEngine): the Stack owns the adapter
    // arrays the transformer Component's `lora` pointer reads through, so it
    // must live until the next detach (clearLora).
    lora_stack: ?lora_mod.Stack = null,
    lora_matched: u32 = 0,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*LtxVideoEngine {
        const self = try allocator.create(LtxVideoEngine);
        errdefer allocator.destroy(self);
        self.* = undefined;
        self.allocator = allocator;
        self.audio = null;
        self.vae_encoder = null;
        self.upsampler = null;
        self.diffusion_decoder = null;
        self.lora_stack = null;
        self.lora_matched = 0;

        self.ltx_cfg = ltx.loadLtxConfig(io, allocator, model_dir);
        self.gemma_dir = try resolveGemmaDir(io, allocator, model_dir, self.ltx_cfg.version);
        errdefer allocator.free(self.gemma_dir);
        log.info("[video] LTX {s} — gemma text encoder: {s}\n", .{ @tagName(self.ltx_cfg.version), self.gemma_dir });
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

    fn hasVariant(self: *LtxVideoEngine, io: std.Io, variant: TransformerVariant) bool {
        var buf: [1024]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ self.model_dir, variant.fileName() }, 0) catch return false;
        return fileExists(io, p);
    }

    /// Swap the transformer slot to `want` (no-op when already loaded). The old
    /// component is freed BEFORE the new one loads so dev + distilled (11 GB
    /// each) never coexist.
    pub fn ensureTransformer(self: *LtxVideoEngine, want: TransformerVariant) !void {
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

    /// Reconcile the attached LoRA stack with the request (mirrors
    /// `ImageEngine.setLoras`): empty `paths` detaches; the same
    /// paths+scales is a no-op reuse; anything else loads + installs every
    /// adapter on the transformer Component. Returns the total number of
    /// (module, adapter) attachments present in the DiT.
    pub fn setLoras(self: *LtxVideoEngine, paths: []const []const u8, scales: []const f32) !u32 {
        if (paths.len == 0) {
            self.clearLora();
            return 0;
        }
        if (paths.len > lora_mod.MAX_LORAS) return error.TooManyLoras;
        if (self.lora_stack) |*st| {
            if (st.matches(paths, scales)) return self.lora_matched;
        }
        self.clearLora();
        var stack: lora_mod.Stack = .{ .allocator = self.allocator };
        errdefer stack.deinit();
        for (paths, scales) |p, sc| {
            const dup_p = try self.allocator.dupe(u8, p);
            errdefer self.allocator.free(dup_p);
            const lf = try lora_mod.loadFile(self.allocator, p, .flux2);
            stack.files[stack.count] = lf;
            stack.paths[stack.count] = dup_p;
            stack.scales[stack.count] = sc;
            stack.count += 1;
        }
        const matched = ltx.countLoraMatches(&self.transformer, &stack);
        if (matched == 0) {
            stack.deinit();
            return error.LoraNoMatch;
        }
        self.lora_stack = stack;
        self.lora_matched = matched;
        self.applyLora();
        return matched;
    }

    fn clearLora(self: *LtxVideoEngine) void {
        self.transformer.lora = null;
        if (self.lora_stack) |*st| st.deinit();
        self.lora_stack = null;
        self.lora_matched = 0;
    }

    fn applyLora(self: *LtxVideoEngine) void {
        self.transformer.lora = if (self.lora_stack) |*st| st else null;
    }

    /// Lazily load the spatial-x2 upsampler for the two-stage boundary.
    pub fn ensureUpsampler(self: *LtxVideoEngine, io: std.Io) !*const ltx.Component {
        if (self.upsampler) |*u| return u;
        var buf: [1024]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}.safetensors", .{ self.model_dir, ltx.UPSAMPLER_PREFIX }, 0) catch return error.MissingUpsampler;
        if (!fileExists(io, p)) return error.MissingUpsampler;
        const cpu_s = mlx.mlx_default_cpu_stream_new();
        var comp = try ltx.loadComponent(self.allocator, p, cpu_s);
        var it = comp.map.iterator();
        while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*); // conv graph wants materialized weights
        self.upsampler = comp;
        log.info("[video] latent upsampler ready ({d} tensors)\n", .{comp.count()});
        return &self.upsampler.?;
    }

    /// Lazily load the DiffVAE decoder. Absent → `error.MissingDiffusionDecoder`,
    /// which the handler turns into a NAMED 400; a silent fall back to the conv
    /// decoder would answer a different question than the one asked.
    pub fn ensureDiffusionDecoder(self: *LtxVideoEngine, io: std.Io) !*const ltx.Component {
        if (self.diffusion_decoder) |*d| return d;
        var buf: [1024]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ self.model_dir, diffvae_fwd.FILE_NAME }, 0) catch return error.MissingDiffusionDecoder;
        if (!fileExists(io, p)) return error.MissingDiffusionDecoder;
        const cpu_s = mlx.mlx_default_cpu_stream_new();
        self.diffusion_decoder = try diffvae_fwd.load(self.allocator, p, cpu_s);
        return &self.diffusion_decoder.?;
    }

    pub fn deinit(self: *LtxVideoEngine) void {
        self.clearLora();
        self.transformer.deinit();
        self.connector.deinit();
        self.vae.deinit();
        if (self.audio) |*a| a.deinit();
        if (self.vae_encoder) |*e| e.deinit();
        if (self.upsampler) |*u| u.deinit();
        if (self.diffusion_decoder) |*d| d.deinit();
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
    const path = std.fmt.bufPrintSentinel(&p, "{s}/vae_encoder.safetensors", .{model_dir}, 0) catch return null;
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

// ── OpenAI `POST /v1/images/edits` (multipart) → our JSON edit request ──
// Pure translation, the ollama.zig principle: ONE request schema and ONE engine
// path, with the vendor surface rewritten into it. Everything downstream
// (`handleImage`) is untouched, so every guard, 400 and capability check
// applies identically whether the client speaks our JSON or OpenAI multipart.

pub const EditFormError = error{
    NotMultipart,
    MissingPrompt,
    MissingImage,
    TooManyImages,
    MaskUnsupported,
    MultipleChoicesUnsupported,
    UrlResponseUnsupported,
    OutputFormatUnsupported,
    StreamUnsupported,
    OutOfMemory,
};

/// The 400 text for each rejection — every one names what we can't do and why,
/// never a silent no-op (the llmprobe-flagged class).
pub fn editFormErrorMessage(err: EditFormError) []const u8 {
    return switch (err) {
        error.NotMultipart => "/v1/images/edits expects multipart/form-data with a 'boundary' parameter",
        error.MissingPrompt => "missing required form field 'prompt'",
        error.MissingImage => "missing required form field 'image' (the picture to edit)",
        error.TooManyImages => "too many 'image' parts (at most 4: the edited source plus 3 references)",
        error.MaskUnsupported => "'mask' is not supported: this server's editors are maskless in-context models (describe the change in the prompt instead)",
        error.MultipleChoicesUnsupported => "'n' must be 1 — this engine generates a single image per request",
        error.UrlResponseUnsupported => "'response_format' must be 'b64_json' — this server does not host generated files",
        error.OutputFormatUnsupported => "'output_format' must be 'png' — this server always returns PNG",
        error.StreamUnsupported => "'stream' is not supported on /v1/images/edits (use /v1/images/generations for SSE progress)",
        error.OutOfMemory => "out of memory",
    };
}

/// Rewrite an OpenAI images-edit form into the `/v1/images/generations` body.
/// Returns owned JSON (caller frees). Files are re-encoded as base64 (the
/// internal schema's transport); everything OpenAI accepts but we can't honor
/// is an explicit error rather than a silently ignored field.
pub fn openaiEditFormToJson(allocator: std.mem.Allocator, body: []const u8, content_type: []const u8) EditFormError![]u8 {
    const boundary = multipart.boundaryFromContentType(content_type) orelse return error.NotMultipart;
    var it = multipart.Iterator.init(body, boundary) catch return error.NotMultipart;

    var images: [MAX_EDIT_IMAGES][]const u8 = undefined;
    var images_n: usize = 0;
    var prompt: ?[]const u8 = null;
    var model: ?[]const u8 = null;
    var size: ?[]const u8 = null;
    var lora_paths: ?[]const u8 = null;
    var lora_scales: ?[]const u8 = null;
    var lora_path: ?[]const u8 = null;
    var lora_scale: ?[]const u8 = null;

    while (it.next()) |part| {
        // `image`, `image[]` and `image[0]` are all in the wild.
        if (std.mem.eql(u8, part.name, "image") or std.mem.startsWith(u8, part.name, "image[")) {
            if (part.data.len == 0) continue;
            if (images_n >= MAX_EDIT_IMAGES) return error.TooManyImages;
            images[images_n] = part.data;
            images_n += 1;
        } else if (std.mem.eql(u8, part.name, "prompt")) {
            prompt = part.data;
        } else if (std.mem.eql(u8, part.name, "model")) {
            model = part.data;
        } else if (std.mem.eql(u8, part.name, "size")) {
            // "auto" means "you decide" — leave it out and let the edit path
            // resolve from the reference.
            if (!std.mem.eql(u8, part.data, "auto") and part.data.len != 0) size = part.data;
        } else if (std.mem.eql(u8, part.name, "mask")) {
            if (part.data.len != 0) return error.MaskUnsupported;
        } else if (std.mem.eql(u8, part.name, "n")) {
            if (part.data.len != 0 and !std.mem.eql(u8, part.data, "1")) return error.MultipleChoicesUnsupported;
        } else if (std.mem.eql(u8, part.name, "response_format")) {
            if (part.data.len != 0 and !std.mem.eql(u8, part.data, "b64_json")) return error.UrlResponseUnsupported;
        } else if (std.mem.eql(u8, part.name, "output_format")) {
            if (part.data.len != 0 and !std.mem.eql(u8, part.data, "png")) return error.OutputFormatUnsupported;
        } else if (std.mem.eql(u8, part.name, "stream")) {
            if (std.mem.eql(u8, part.data, "true")) return error.StreamUnsupported;
        } else if (std.mem.eql(u8, part.name, "lora_paths")) {
            if (part.data.len != 0) lora_paths = part.data;
        } else if (std.mem.eql(u8, part.name, "lora_scales")) {
            if (part.data.len != 0) lora_scales = part.data;
        } else if (std.mem.eql(u8, part.name, "lora_path")) {
            if (part.data.len != 0) lora_path = part.data;
        } else if (std.mem.eql(u8, part.name, "lora_scale")) {
            if (part.data.len != 0) lora_scale = part.data;
        }
        // background / quality / input_fidelity / output_compression / user /
        // partial_images: accepted and ignored — they don't change what we'd
        // produce, so rejecting them would break working clients for nothing.
    }

    const p = prompt orelse return error.MissingPrompt;
    if (p.len == 0) return error.MissingPrompt;
    if (images_n == 0) return error.MissingImage;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"mode\":\"edit\",\"prompt\":");
    try chat_mod.appendJsonString(allocator, &out, p);
    if (model) |m| {
        if (m.len != 0) {
            try out.appendSlice(allocator, ",\"model\":");
            try chat_mod.appendJsonString(allocator, &out, m);
        }
    }
    if (size) |sz| {
        try out.appendSlice(allocator, ",\"size\":");
        try chat_mod.appendJsonString(allocator, &out, sz);
    }
    // LoRA fields ride through to `parseLoraFields` (issue #268: they were
    // silently dropped). The array forms are JSON text and pass as-is — a
    // malformed array is that parser's named 400, not ours.
    if (lora_paths) |v| {
        try out.appendSlice(allocator, ",\"lora_paths\":");
        try out.appendSlice(allocator, v);
    }
    if (lora_scales) |v| {
        try out.appendSlice(allocator, ",\"lora_scales\":");
        try out.appendSlice(allocator, v);
    }
    if (lora_path) |v| {
        try out.appendSlice(allocator, ",\"lora_path\":");
        try chat_mod.appendJsonString(allocator, &out, v);
    }
    if (lora_scale) |v| {
        try out.appendSlice(allocator, ",\"lora_scale\":");
        try out.appendSlice(allocator, v);
    }
    try out.appendSlice(allocator, ",\"image\":\"");
    try appendBase64(allocator, &out, images[0]);
    try out.appendSlice(allocator, "\"");
    if (images_n > 1) {
        try out.appendSlice(allocator, ",\"ref_images\":[");
        for (images[1..images_n], 0..) |img, i| {
            if (i != 0) try out.appendSlice(allocator, ",");
            try out.appendSlice(allocator, "\"");
            try appendBase64(allocator, &out, img);
            try out.appendSlice(allocator, "\"");
        }
        try out.appendSlice(allocator, "]");
    }
    try out.appendSlice(allocator, "}");
    return out.toOwnedSlice(allocator);
}

fn appendBase64(allocator: std.mem.Allocator, out: *std.ArrayList(u8), bytes: []const u8) !void {
    const n = std.base64.standard.Encoder.calcSize(bytes.len);
    const start = out.items.len;
    try out.resize(allocator, start + n);
    _ = std.base64.standard.Encoder.encode(out.items[start..], bytes);
}

/// Reshape a target size to `src`'s aspect ratio at the SAME pixel budget
/// (`budget_w × budget_h`), rounded to a multiple of 16. Used by the byte-based
/// edit path so the output geometry matches the reference image the user handed
/// us while still honoring the resolution they picked as an area.
fn fitAspect(src_w: u32, src_h: u32, budget_w: u32, budget_h: u32) struct { w: u32, h: u32 } {
    if (src_w == 0 or src_h == 0) return .{ .w = budget_w, .h = budget_h };
    const budget: f64 = @as(f64, @floatFromInt(budget_w)) * @as(f64, @floatFromInt(budget_h));
    const aspect: f64 = @as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(src_h));
    const h = std.math.sqrt(budget / aspect);
    const w = h * aspect;
    return .{
        .w = @max(16, (@as(u32, @intFromFloat(@round(w))) / 16) * 16),
        .h = @max(16, (@as(u32, @intFromFloat(@round(h))) / 16) * 16),
    };
}

/// Scale (w,h) down proportionally until neither exceeds `cap`, floored to /16.
/// Preserves the aspect ratio `fitAspect` just established. Extreme aspects can
/// still meet the 256 floor inside `normalizeSize` — nothing can honor a 2048
/// ceiling and a 256 floor past ~8:1 — but that is a far smaller distortion than
/// squaring the image off.
/// Named so the two geometry helpers below can hand results to each other —
/// Zig treats each anonymous `struct { w, h }` as a distinct type.
const Dims = struct { w: u32, h: u32 };

fn fitWithinCap(w: u32, h: u32, cap: u32) Dims {
    const longest = @max(w, h);
    if (longest <= cap or longest == 0 or cap == 0) return .{ .w = w, .h = h };
    const scale = @as(f64, @floatFromInt(cap)) / @as(f64, @floatFromInt(longest));
    const sw = @round(@as(f64, @floatFromInt(w)) * scale);
    const sh = @round(@as(f64, @floatFromInt(h)) * scale);
    return .{
        .w = @max(16, (@as(u32, @intFromFloat(sw)) / 16) * 16),
        .h = @max(16, (@as(u32, @intFromFloat(sh)) / 16) * 16),
    };
}

/// Output geometry for a byte-based edit: keep the primary reference's aspect at
/// the requested pixel budget, THEN scale into the backend's dimension cap.
/// Both steps are load-bearing — `fitAspect` alone left `normalizeSize`'s
/// per-dimension clamp free to square the result off again, which it did for any
/// reference bigger than the cap (i.e. every modern phone photo).
fn resolveEditTargetSize(src_w: u32, src_h: u32, budget_w: u32, budget_h: u32, cap: u32) Dims {
    const fit = fitAspect(src_w, src_h, budget_w, budget_h);
    return fitWithinCap(fit.w, fit.h, cap);
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
/// [1,3,H,W] in [0,1] (decodeImageToBCHW's cover output) -> [1,3,1,H,W] in
/// [-1,1] f32 — the shape/range MiniMax-H3's keyframe encoder consumes.
fn unitToPm1BCFHW(bchw: mlx.mlx_array, target_h: u32, target_w: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const two = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, bchw, two, s));
    var pm1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pm1);
    try mlx.check(mlx.mlx_subtract(&pm1, scaled, one, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const shape5 = [_]c_int{ 1, 3, 1, @intCast(target_h), @intCast(target_w) };
    try mlx.check(mlx.mlx_reshape(&out, pm1, &shape5, 5, s));
    return out;
}

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
    const audio_path = std.fmt.bufPrintSentinel(&ap, "{s}/audio_vae.safetensors", .{dir}, 0) catch return null;
    const voc_path = std.fmt.bufPrintSentinel(&vp, "{s}/vocoder.safetensors", .{dir}, 0) catch return null;
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
fn resolveGemmaDir(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, version: ltx.LtxVersion) ![]u8 {
    if (std.c.getenv("LTX_GEMMA_DIR")) |env| {
        const e = std.mem.span(env);
        // A relative override would feed openFileAbsolute's assert downstream
        // (ReleaseFast UB) — ignore it loudly instead.
        if (e.len > 0 and std.fs.path.isAbsolute(e)) return allocator.dupe(u8, e);
        if (e.len > 0) log.warn("[video] ignoring non-absolute LTX_GEMMA_DIR: {s}\n", .{e});
    }
    // 2.5 ships a FINE-TUNED encoder inside the pack, so the pack is
    // self-contained and there is no shared repo to fall back to: a 2.5 pack
    // whose subdir is missing is incomplete, not a 2.3 pack.
    if (version.textEncoderSubdir()) |sub| {
        const dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, sub });
        errdefer allocator.free(dir);
        const cfg = try std.fmt.allocPrintSentinel(allocator, "{s}/config.json", .{dir}, 0);
        defer allocator.free(cfg);
        if (fileExists(io, cfg)) return dir;
        allocator.free(dir);
        log.err("[video] LTX 2.5 pack is missing its text encoder ({s}/{s})\n", .{ model_dir, sub });
        return error.NoGemmaDir;
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
pub fn handleImage(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *ImageEngine) !void {
    const prompt_raw = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt'");
    const prompt = try jsonUnescape(allocator, prompt_raw);
    defer allocator.free(prompt);
    if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

    // Requested size (default 1024²); the backend resolves it (FLUX is fixed
    // 1024², Krea accepts any multiple of 16 in [256,2048]).
    var req_w: u32 = 1024;
    var req_h: u32 = 1024;
    // Whether the CLIENT chose a size. An edit with no size means "keep the
    // source's resolution" (the reference pipeline's `max_size = source size`
    // default) — indistinguishable from an explicit 1024² unless we track it.
    var size_given = false;
    if (extractJsonString(body, "size")) |size| {
        if (parseSize(size)) |wh| {
            req_w = wh.w;
            req_h = wh.h;
            size_given = true;
        }
    }
    const sz = engine.normalizeSize(req_w, req_h);
    var width = sz.w;
    var height = sz.h;
    if (req_w != width or req_h != height) {
        log.warn("[image] requested {d}x{d} resolved to {d}x{d} for this backend\n", .{ req_w, req_h, width, height });
    }
    const seed: u64 = extractJsonInt(body, "seed") orelse 42;
    const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse 4);

    // Source image: `image` (base64 PNG/JPEG) + `mode` ("variation" default /
    // "edit"). Variation = SDEdit renoise at `strength` (both backends);
    // edit = FLUX.2 in-context reference conditioning (instruction edits —
    // "make the hair blue" — with the source attended to clean; no strength).
    // Edit mode also takes `ref_images` (a JSON array of base64 PNG/JPEG):
    // extra in-context references beside the edited source — "replace the
    // face in image 1 with the face from image 2".
    var init_img: ?mlx.mlx_array = null;
    defer if (init_img) |ii| {
        _ = mlx.mlx_array_free(ii);
    };
    var edit_imgs: [MAX_EDIT_IMAGES]mlx.mlx_array = undefined;
    var edit_imgs_n: usize = 0;
    defer for (edit_imgs[0..edit_imgs_n]) |ei| {
        _ = mlx.mlx_array_free(ei);
    };
    // Raw reference bytes for byte-based edit backends (MageFlow); owned copies
    // that must outlive `generateImage`.
    var edit_byte_bufs: [MAX_EDIT_IMAGES][]u8 = undefined;
    var edit_byte_n: usize = 0;
    defer for (edit_byte_bufs[0..edit_byte_n]) |b| allocator.free(b);
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
            return sendError(conn, 400, "instruction editing (mode:\"edit\") requires a FLUX.2 or Mage-Flow-Edit model");
        if (!edit_mode and !engine.supportsImg2Img())
            return sendError(conn, 400, "image-to-image (mode:\"variation\") isn't available for this model — either its VAE encoder failed to load, or this backend has no variation path");
        if (extractJsonFloat(body, "strength")) |sv| {
            if (!(sv > 0.0 and sv <= 1.0)) return sendError(conn, 400, "'strength' must be in (0,1]");
            strength = @floatCast(sv);
        }
        const img_bytes = base64DecodeAlloc(allocator, raw_img) catch
            return sendError(conn, 400, "invalid base64 in 'image'");
        defer allocator.free(img_bytes);
        if (edit_mode and engine.editUsesRawBytes()) {
            // Byte-based edit backend (MageFlow): keep the source bytes; the
            // engine does its own target-size VAE resize + VLM resize.
            edit_byte_bufs[0] = try allocator.dupe(u8, img_bytes);
            edit_byte_n = 1;
            // MageFlow edits AT the target grid: every reference is resized to
            // (W,H), so a square target squashes a 3:2 photo — and an editor
            // that changes the geometry of its input is wrong by construction.
            // NO size in the request = the reference pipeline's default
            // (`max_size = source size`): hand back the source's own resolution.
            // An explicit size keeps the PRIMARY reference's aspect ratio and
            // treats the request as the pixel budget to fit it into.
            if (imageNativeSize(img_bytes)) |nat| {
                // No size given => the reference IS the budget, which `fitAspect`
                // resolves to the source's own dimensions (/16).
                const fit = if (size_given)
                    resolveEditTargetSize(nat.w, nat.h, width, height, engine.maxDim())
                else
                    resolveEditTargetSize(nat.w, nat.h, nat.w, nat.h, engine.maxDim());
                const nz = engine.normalizeSize(fit.w, fit.h);
                if (nz.w != width or nz.h != height)
                    log.info("[image] edit: target {d}x{d} -> {d}x{d} (primary reference is {d}x{d}, size {s})\n", .{ width, height, nz.w, nz.h, nat.w, nat.h, if (size_given) "requested" else "matched to source" });
                width = nz.w;
                height = nz.h;
            }
            log.info("[image] edit: reference {d} bytes (byte-based backend)\n", .{img_bytes.len});
        } else if (edit_mode) {
            // The reference keeps its OWN aspect ratio (fit to ~1MP, /32 dims —
            // official prep behavior); its latent grid is independent of the
            // output grid, so nothing gets squished or cropped away.
            const nat = imageNativeSize(img_bytes) orelse
                return sendError(conn, 400, "could not decode 'image' (PNG/JPEG supported)");
            const rd = fitRefDims(nat.w, nat.h);
            edit_imgs[0] = decodeImageToBCHW(allocator, img_bytes, rd.h, rd.w) orelse
                return sendError(conn, 400, "could not decode 'image' (PNG/JPEG supported)");
            edit_imgs_n = 1;
            log.info("[image] edit: reference {d}x{d} -> {d}x{d} (in-context conditioning)\n", .{ nat.w, nat.h, rd.w, rd.h });
            // Output grid (independent of the reference's own conditioning grid
            // above — the reference rides at its own `rd` grid in every attention
            // step and is never resized onto the output canvas, unlike the
            // byte-based backend above). NO size in the request = "Match
            // source": the reference's own resolution IS the output target,
            // same contract the byte-based backend already honors above. An
            // EXPLICIT size is honored LITERALLY — the reference's aspect ratio
            // has no architectural claim on the output grid here, so reshaping
            // a requested 512x512 into the reference's aspect (as the
            // byte-based backend must) just produced the wrong resolution.
            // Without the no-size branch, `width`/`height` stay at the
            // 1024x1024 default from earlier and every edit comes back square
            // regardless of what the client asked to match.
            const fit = if (size_given)
                fitWithinCap(width, height, engine.maxDim())
            else
                resolveEditTargetSize(nat.w, nat.h, nat.w, nat.h, engine.maxDim());
            const nz = engine.normalizeSize(fit.w, fit.h);
            if (nz.w != width or nz.h != height)
                log.info("[image] edit: target {d}x{d} -> {d}x{d} (primary reference is {d}x{d}, size {s})\n", .{ width, height, nz.w, nz.h, nat.w, nat.h, if (size_given) "requested" else "matched to source" });
            width = nz.w;
            height = nz.h;
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

    // Extra in-context references (edit mode only): each keeps its own aspect
    // ratio like the primary and rides at its own t offset in the DiT.
    if (std.mem.indexOf(u8, body, "\"ref_images\"") != null) {
        if (!edit_mode) return sendError(conn, 400, "'ref_images' requires mode:\"edit\"");
        var it = iterJsonStringArray(body, "ref_images") orelse
            return sendError(conn, 400, "invalid 'ref_images' (must be a JSON array of base64 strings)");
        while (it.next()) |raw_ref| {
            const cur_n = if (engine.editUsesRawBytes()) edit_byte_n else edit_imgs_n;
            if (cur_n >= MAX_EDIT_IMAGES)
                return sendError(conn, 400, "too many reference images ('ref_images' takes at most 3 beside 'image')");
            const ref_bytes = base64DecodeAlloc(allocator, raw_ref) catch
                return sendError(conn, 400, "invalid base64 in 'ref_images'");
            defer allocator.free(ref_bytes);
            if (engine.editUsesRawBytes()) {
                edit_byte_bufs[edit_byte_n] = try allocator.dupe(u8, ref_bytes);
                edit_byte_n += 1;
                log.info("[image] edit ref {d}: {d} bytes (byte-based backend)\n", .{ edit_byte_n, ref_bytes.len });
                continue;
            }
            const rnat = imageNativeSize(ref_bytes) orelse
                return sendError(conn, 400, "could not decode a 'ref_images' entry (PNG/JPEG supported)");
            const rrd = fitRefDims(rnat.w, rnat.h);
            edit_imgs[edit_imgs_n] = decodeImageToBCHW(allocator, ref_bytes, rrd.h, rrd.w) orelse
                return sendError(conn, 400, "could not decode a 'ref_images' entry (PNG/JPEG supported)");
            edit_imgs_n += 1;
            log.info("[image] edit ref {d}: {d}x{d} -> {d}x{d}\n", .{ edit_imgs_n, rnat.w, rnat.h, rrd.w, rrd.h });
        }
        if (it.bad) return sendError(conn, 400, "invalid 'ref_images' (must be a JSON array of base64 strings)");
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

    // Style LoRA(s): one or more absolute paths to .safetensors adapters,
    // each with an optional scale — mirrors mflux's `--lora-paths`/
    // `--lora-scales`. Accepts the array form (`lora_paths`/`lora_scales`)
    // or the original single-adapter form (`lora_path`/`lora_scale`) for
    // backward compatibility. No LoRA fields in the request detaches
    // whatever was attached before.
    {
        var lora_path_bufs: [lora_mod.MAX_LORAS][]u8 = undefined;
        var lora_scales: [lora_mod.MAX_LORAS]f32 = undefined;
        const lora_n = parseLoraFields(allocator, body, &lora_path_bufs, &lora_scales) catch |err| switch (err) {
            error.TooManyLoraPaths => return sendError(conn, 400, "too many 'lora_paths' (max 8)"),
            error.BadLoraPathsJson => return sendError(conn, 400, "invalid 'lora_paths' (must be a JSON array of strings)"),
            error.BadLoraScalesJson => return sendError(conn, 400, "invalid 'lora_scales' (numbers, comma/space separated, or a JSON array)"),
            error.OutOfMemory => return err,
        };
        defer for (lora_path_bufs[0..lora_n]) |p| allocator.free(p);

        var lora_paths: [lora_mod.MAX_LORAS][]const u8 = undefined;
        for (lora_path_bufs[0..lora_n], 0..) |p, i| lora_paths[i] = p;
        const matched = engine.setLoras(lora_paths[0..lora_n], lora_scales[0..lora_n]) catch |err| switch (err) {
            error.LoraNoMatch => return sendError(conn, 400, "LoRA(s) have no modules matching this model's DiT — wrong LoRA for this architecture?"),
            error.BadLoraPath => return sendError(conn, 400, "'lora_path'/'lora_paths' must be absolute path(s) to .safetensors file(s)"),
            error.TooManyLoras => return sendError(conn, 400, "too many LoRA adapters requested"),
            error.OutOfMemory => return err,
            else => return sendError(conn, 400, "failed to load a LoRA file"),
        };
        if (lora_n > 0)
            log.info("[image] lora: matched {d} module-attachment(s) across {d} adapter(s)\n", .{ matched, lora_n });
    }

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[image] generating {d}x{d} steps={d} stream={}: {d} chars\n", .{ width, height, steps, want_stream, prompt.len });
    var sctx = sse.StreamCtx{ .conn = conn, .stream = want_stream };
    const prog: ?sse.Progress = sctx.progress();
    if (want_stream) try conn.writeAll(sse.headers);

    const gen_opts = ImageGenOpts{
        .init_image = init_img, // null in edit mode
        .strength = strength,
        .edit_images = edit_imgs[0..edit_imgs_n],
        .edit_image_bytes = edit_byte_bufs[0..edit_byte_n],
        .cond_gain = cond_gain,
        .cond_weights = cond_weights,
    };
    const img = engine.generateImage(allocator, prompt, width, height, seed, steps, gen_opts, prog) catch |err| {
        // Client hung up mid-generation — there is nobody to answer, and
        // saying "generation failed" would be a lie about a job we stopped.
        if (err == error.Cancelled) {
            log.info("[image] generation cancelled — client disconnected\n", .{});
            return;
        }
        log.err("[image] generation failed: {}\n", .{err});
        if (want_stream) {
            sse.sendError(conn, "generation failed");
            return;
        }
        return sendError(conn, 500, "generation failed");
    };
    defer _ = mlx.mlx_array_free(img);

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
    const synth = switch (engine.backend) {
        .tts => |*t| t,
        .music, .music3 => return sendError(conn, 400, "loaded audio model is a music generator; POST /v1/audio/music-generations"),
        .kokoro => |k| return handleKokoroSpeech(allocator, conn, body, k),
    };
    // Pre-warm (docs/qwentts-cache.md): `{"warm_only":true,"ref_audio":...}`
    // embeds + caches the clone voice WITHOUT synthesizing, so the FIRST
    // sentence of a voice session skips the cold ECAPA forward. Misuse is a
    // named 400, never a silent no-op (the media-gen rule).
    if (sse.bodyWantsTrue(body, "warm_only")) {
        if (!synth.supportsCloning()) return sendError(conn, 400, "this model has no speaker encoder; nothing to warm");
        const raw_ref = extractJsonString(body, "ref_audio") orelse return sendError(conn, 400, "warm_only requires ref_audio");
        const b64 = try jsonUnescape(allocator, raw_ref);
        defer allocator.free(b64);
        if (b64.len == 0) return sendError(conn, 400, "warm_only requires ref_audio");
        const wav_bytes = base64DecodeAlloc(allocator, b64) catch return sendError(conn, 400, "ref_audio is not valid base64");
        defer allocator.free(wav_bytes);
        const samples = decodeWavToF32(allocator, wav_bytes) catch return sendError(conn, 400, "ref_audio is not a decodable WAV");
        defer allocator.free(samples);
        const was_cached = synth.warmSpeaker(samples) catch |err| {
            log.err("[audio] warm_only failed: {}\n", .{err});
            return sendError(conn, 500, "speaker embedding failed");
        };
        log.info("[audio] warm_only: speaker embedding {s}\n", .{if (was_cached) "already cached" else "cached"});
        return sendBytesJson(conn, allocator, if (was_cached)
            "{\"warmed\":true,\"cache\":\"hit\"}"
        else
            "{\"warmed\":true,\"cache\":\"miss\"}");
    }

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
        const io_t = std.Io.Threaded.global_single_threaded.io();
        const t0 = std.Io.Timestamp.now(io_t, .boot);
        const b64 = try jsonUnescape(allocator, raw_ref); // handles \/ from Swift JSONSerialization
        defer allocator.free(b64);
        if (b64.len > 0) {
            if (base64DecodeAlloc(allocator, b64)) |wav_bytes| {
                defer allocator.free(wav_bytes);
                if (decodeWavToF32(allocator, wav_bytes)) |samples| {
                    if (synth.supportsCloning()) {
                        ref_samples = samples;
                        const dec_ns: u64 = @intCast(t0.untilNow(io_t, .boot).nanoseconds);
                        log.info("[audio] ref decode {d:.2} ms; reference voice: {d} samples → cloning\n", .{ @as(f64, @floatFromInt(dec_ns)) / 1e6, samples.len });
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
    var sctx = sse.StreamCtx{ .conn = conn, .stream = want_stream };
    const prog: ?sse.Progress = sctx.progress();
    if (want_stream) try conn.writeAll(sse.headers);

    const wav = synth.synthesizeWav(text, 2048, prog, ref_samples) catch |err| {
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

/// `POST /v1/audio/speech` on a Kokoro checkpoint.
///
/// Shares the endpoint with Qwen3-TTS but NOT its controls, and the difference
/// is refused rather than ignored (the named-400 rule): `ref_audio` is a
/// Qwen3-TTS control and Kokoro cannot clone, so asking for it here is an
/// error, not a silently plain-voiced answer. Conversely `voice` selects one of
/// the 54 packs, and a comma-separated list BLENDS them
/// (`"af_bella,af_sky"`) — the reference's own convention.
fn handleKokoroSpeech(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *kokoro.Engine) !void {
    if (sse.bodyWantsTrue(body, "warm_only")) {
        return sendError(conn, 400, "this model does not support voice cloning; nothing to warm");
    }
    const input = extractJsonString(body, "input") orelse extractJsonString(body, "text") orelse
        return sendError(conn, 400, "missing 'input'");
    const text = try jsonUnescape(allocator, input);
    defer allocator.free(text);
    if (text.len == 0) return sendError(conn, 400, "empty 'input'");

    if (extractJsonString(body, "ref_audio")) |raw| {
        if (raw.len > 0) return sendError(conn, 400, "this model does not support voice cloning; use 'voice' to pick or blend a built-in voice");
    }

    var voice_buf: ?[]u8 = null;
    defer if (voice_buf) |v| allocator.free(v);
    var voice: []const u8 = kokoro.DEFAULT_VOICE;
    if (extractJsonString(body, "voice")) |raw| {
        const unescaped = try jsonUnescape(allocator, raw);
        if (unescaped.len == 0) {
            allocator.free(unescaped);
        } else {
            voice_buf = unescaped;
            voice = unescaped;
        }
    }
    if (!engine.hasVoice(voice)) {
        var msg: [256]u8 = undefined;
        const m = std.fmt.bufPrint(&msg, "unknown voice '{s}'; see /v1/models for the available voices", .{voice}) catch "unknown voice";
        return sendError(conn, 400, m);
    }

    const speed: f32 = @floatCast(extractJsonFloat(body, "speed") orelse 1.0);
    if (!(speed > 0.0) or speed > 5.0) return sendError(conn, 400, "'speed' must be in (0, 5]");

    const seed: u64 = if (extractJsonFloat(body, "seed")) |v| @intFromFloat(@max(0, v)) else 0;

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[kokoro] {d} chars voice={s} speed={d:.2} stream={}\n", .{ text.len, voice, speed, want_stream });
    if (want_stream) try conn.writeAll(sse.headers);

    const out = engine.synthesizeWav(text, voice, speed, seed) catch |err| {
        log.err("[kokoro] synthesis failed: {}\n", .{err});
        if (want_stream) {
            sse.sendError(conn, "synthesis failed");
            return;
        }
        return sendError(conn, 500, "synthesis failed");
    };
    defer allocator.free(out);
    log.info("[kokoro] -> {d} WAV bytes\n", .{out.len});

    if (want_stream) {
        const b64_len = std.base64.standard.Encoder.calcSize(out.len);
        const b64 = try allocator.alloc(u8, b64_len);
        defer allocator.free(b64);
        _ = std.base64.standard.Encoder.encode(b64, out);
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(allocator);
        try buf.appendSlice(allocator, "data: {\"type\":\"complete\",\"format\":\"wav\",\"data\":\"");
        try buf.appendSlice(allocator, b64);
        try buf.appendSlice(allocator, "\"}\n\n");
        try conn.writeAll(buf.items);
        return;
    }
    return sendBytes(conn, allocator, "audio/wav", out);
}

/// `instrumental: true` beside words to sing is contradictory. Letting either
/// side quietly win is the failure mode — a sticky checkbox silently discards a
/// verse the user typed, or the checkbox does nothing — so the pair is a NAMED
/// 400 (the `/v1/images/edits` rule: everything we cannot honor is named).
/// Whitespace-only lyrics count as ABSENT so an app that keeps a blank lyrics
/// editor mounted beside the checkbox is fine. Both backends read this ONE
/// predicate, so the rule cannot drift between them.
pub fn instrumentalConflicts(instrumental: bool, lyrics: []const u8) bool {
    return instrumental and std.mem.trim(u8, lyrics, " \t\r\n").len != 0;
}

/// `POST /v1/audio/music-generations` — ACE-Step text2music / cover / complete.
/// `{"model", "prompt" (style/genre/mood, REQUIRED), "lyrics" ("" →
/// "[Instrumental]"), "instrumental" (bool, the explicit spelling of empty
/// lyrics; a 400 if real lyrics ride along), "vocal_language" ("en"), "bpm", "keyscale",
/// "timesignature", "duration_seconds" (default 60, valid 10–600), "seed",
/// "ref_audio" (base64 WAV, style/timbre), "task" ("text2music" | "cover" |
/// "complete"), "src_audio" (base64 WAV 10–600 s, the cover/complete source;
/// its length becomes the track length), "cover_strength" (0–1, default 1),
/// "cover_noise_strength" (0–1, default 0), "track_classes" (complete: array
/// from the TRACK_NAMES vocabulary), "stream"}`. Response mirrors `/v1/audio/speech`: raw `audio/wav` bytes
/// non-stream; SSE `progress` per stage/step + a base64 `complete` event when
/// streaming. Targeting a TTS voice model here is an explicit 400.
pub fn handleMusic(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *AudioEngine) !void {
    switch (engine.backend) {
        .music => |m| return handleMusicAcestep(allocator, conn, body, m),
        .music3 => |m| return handleMusic3(allocator, conn, body, m),
        .tts, .kokoro => return sendError(conn, 400, "loaded audio model is a TTS voice; POST /v1/audio/speech"),
    }
}

fn handleMusicAcestep(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, music: *acestep.Engine) !void {
    const raw_prompt = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt' (style/genre/mood description)");
    const prompt = try jsonUnescape(allocator, raw_prompt);
    defer allocator.free(prompt);
    if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

    var lyrics: []u8 = try allocator.dupe(u8, "");
    defer allocator.free(lyrics);
    if (extractJsonString(body, "lyrics")) |raw| {
        allocator.free(lyrics);
        lyrics = try jsonUnescape(allocator, raw);
    }
    var language: []u8 = try allocator.dupe(u8, "en");
    defer allocator.free(language);
    if (extractJsonString(body, "vocal_language")) |raw| {
        allocator.free(language);
        language = try jsonUnescape(allocator, raw);
    }
    const instrumental = sse.bodyWantsTrue(body, "instrumental");
    if (instrumentalConflicts(instrumental, lyrics))
        return sendError(conn, 400, "'instrumental' is true but 'lyrics' is non-empty — send one or the other");
    const cond_lyrics = acestep.resolveLyrics(instrumental, lyrics);
    const keyscale = extractJsonString(body, "keyscale") orelse "";
    const timesignature = extractJsonString(body, "timesignature") orelse "";
    var bpm: ?u32 = null;
    if (extractJsonInt(body, "bpm")) |b| {
        if (b < 30 or b > 300) return sendError(conn, 400, "'bpm' must be in [30,300]");
        bpm = @intCast(b);
    }
    const duration: u32 = @intCast(extractJsonInt(body, "duration_seconds") orelse 60);
    if (duration < acestep.MIN_DURATION_S or duration > acestep.MAX_DURATION_S)
        return sendError(conn, 400, "'duration_seconds' must be in [10,600]");
    const seed: u64 = extractJsonInt(body, "seed") orelse 42;

    // `task`: text2music (default) | cover | complete (vocal-to-BGM). The task
    // is the DiT's context stream + the instruction line; cover/complete read
    // a full-length `src_audio`, whose latent decides the track length.
    var task: acestep.Task = .text2music;
    if (extractJsonString(body, "task")) |raw| {
        task = std.meta.stringToEnum(acestep.Task, raw) orelse return sendError(conn, 400, "'task' must be one of text2music, cover, complete");
    }
    const cover_strength: f32 = @floatCast(extractJsonFloat(body, "cover_strength") orelse 1.0);
    const cover_noise_strength: f32 = @floatCast(extractJsonFloat(body, "cover_noise_strength") orelse 0.0);
    if (cover_strength < 0.0 or cover_strength > 1.0 or cover_noise_strength < 0.0 or cover_noise_strength > 1.0)
        return sendError(conn, 400, "'cover_strength' and 'cover_noise_strength' must be in [0,1]");
    if (task != .cover and (extractJsonFloat(body, "cover_strength") != null or extractJsonFloat(body, "cover_noise_strength") != null))
        return sendError(conn, 400, "'cover_strength' / 'cover_noise_strength' only apply to task \"cover\"");
    var track_classes: []u8 = try allocator.dupe(u8, "");
    defer allocator.free(track_classes);
    if (iterJsonStringArray(body, "track_classes")) |it0| {
        if (task != .complete) return sendError(conn, 400, "'track_classes' only applies to task \"complete\"");
        var it = it0;
        var names: std.ArrayList([]const u8) = .empty;
        defer names.deinit(allocator);
        while (it.next()) |n| try names.append(allocator, n);
        if (it.bad) return sendError(conn, 400, "'track_classes' must be an array of strings");
        allocator.free(track_classes);
        track_classes = acestep.joinTrackClasses(allocator, names.items) catch
            return sendError(conn, 400, "'track_classes' entries must be from: woodwinds, brass, fx, synth, strings, percussion, keyboard, guitar, bass, drums, backing_vocals, vocals");
    }
    if (task == .cover and !music.fsqAvailable())
        return sendError(conn, 400, "task \"cover\" needs fsq.safetensors beside model.safetensors (this pack predates cover mode — re-download it, or fetch fsq.safetensors from the HF mirror into the model folder)");

    // `src_audio`: the cover / complete SOURCE as a base64 WAV, full length
    // (10–600 s, NOT windowed like ref_audio). Named 400s, never a silent
    // text2music downgrade.
    var src_audio: ?mlx.mlx_array = null;
    defer if (src_audio) |r| {
        _ = mlx.mlx_array_free(r);
    };
    if (extractJsonString(body, "src_audio")) |raw_src| {
        const b64 = try jsonUnescape(allocator, raw_src);
        defer allocator.free(b64);
        if (b64.len > 0) {
            if (task == .text2music) return sendError(conn, 400, "'src_audio' needs task \"cover\" or \"complete\" (text2music takes 'ref_audio' for style)");
            const wav_bytes = base64DecodeAlloc(allocator, b64) catch return sendError(conn, 400, "src_audio: invalid base64");
            defer allocator.free(wav_bytes);
            const dec = wav_mod.decode(allocator, wav_bytes) catch return sendError(conn, 400, "src_audio: expected a PCM16/PCM24/float32 WAV");
            defer allocator.free(dec.pcm);
            const secs = (dec.pcm.len / dec.channels) / dec.sample_rate;
            if (secs < acestep.MIN_DURATION_S) return sendError(conn, 400, "src_audio: clip too short (needs at least 10 s)");
            if (secs > acestep.MAX_DURATION_S) return sendError(conn, 400, "src_audio: clip longer than 600 s");
            const stereo = try wav_mod.toStereoInterleaved(allocator, dec.pcm, dec.channels);
            defer allocator.free(stereo);
            const at48k = try wav_mod.resampleLinear(allocator, stereo, 2, dec.sample_rate, music.cfg.sample_rate);
            defer allocator.free(at48k);
            const n: c_int = @intCast(at48k.len / 2);
            src_audio = mlx.mlx_array_new_data(at48k.ptr, &[_]c_int{ 1, n, 2 }, 3, .float32);
            log.info("[music] source audio: {d}s {d} Hz {d}ch clip for task {s}\n", .{ secs, dec.sample_rate, dec.channels, @tagName(task) });
        }
    }
    if (task != .text2music and src_audio == null)
        return sendError(conn, 400, "task \"cover\" / \"complete\" needs 'src_audio' (base64 WAV of the source track)");

    // `ref_audio` (#259): a base64 WAV whose style/timbre the track should
    // follow. Fills the condition encoder's timbre slot (VAE latent mean of a
    // 30 s window, `acestep.referenceWindow`) instead of the silence latent.
    // NOT graceful (the a2vid rule): the user asked for THIS clip, so a silent
    // text2music downgrade is a wrong result — named 400s instead.
    var ref_audio: ?mlx.mlx_array = null;
    defer if (ref_audio) |r| {
        _ = mlx.mlx_array_free(r);
    };
    if (extractJsonString(body, "ref_audio")) |raw_ref| {
        const b64 = try jsonUnescape(allocator, raw_ref);
        defer allocator.free(b64);
        if (b64.len > 0) {
            const wav_bytes = base64DecodeAlloc(allocator, b64) catch return sendError(conn, 400, "ref_audio: invalid base64");
            defer allocator.free(wav_bytes);
            const dec = wav_mod.decode(allocator, wav_bytes) catch return sendError(conn, 400, "ref_audio: expected a PCM16/PCM24/float32 WAV");
            defer allocator.free(dec.pcm);
            if (dec.pcm.len / dec.channels < dec.sample_rate) return sendError(conn, 400, "ref_audio: clip too short (needs at least 1 s)");
            const stereo = try wav_mod.toStereoInterleaved(allocator, dec.pcm, dec.channels);
            defer allocator.free(stereo);
            const at48k = try wav_mod.resampleLinear(allocator, stereo, 2, dec.sample_rate, music.cfg.sample_rate);
            defer allocator.free(at48k);
            const window = try acestep.referenceWindow(allocator, at48k, music.cfg.sample_rate);
            defer allocator.free(window);
            const n: c_int = @intCast(window.len / 2);
            ref_audio = mlx.mlx_array_new_data(window.ptr, &[_]c_int{ 1, n, 2 }, 3, .float32);
            log.info("[music] reference audio: {d:.1}s {d} Hz {d}ch clip -> 30 s timbre window\n", .{ @as(f32, @floatFromInt(dec.pcm.len / dec.channels)) / @as(f32, @floatFromInt(dec.sample_rate)), dec.sample_rate, dec.channels });
        }
    }

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[music] generating {d}s task={s} seed={d} lyrics={d}ch instrumental={} ref_audio={} src_audio={} stream={}\n", .{ duration, @tagName(task), seed, cond_lyrics.len, instrumental, ref_audio != null, src_audio != null, want_stream });
    var sctx = sse.StreamCtx{ .conn = conn, .stream = want_stream };
    const prog: ?sse.Progress = sctx.progress();
    if (want_stream) try conn.writeAll(sse.headers);

    const req = acestep.MusicRequest{
        .caption = prompt,
        .lyrics = cond_lyrics,
        .language = language,
        .bpm = bpm,
        .keyscale = keyscale,
        .timesignature = timesignature,
        .duration_s = duration,
        .seed = seed,
        .ref_audio = ref_audio,
        .task = task,
        .src_audio = src_audio,
        .cover_strength = cover_strength,
        .cover_noise_strength = cover_noise_strength,
        .track_classes = track_classes,
    };
    const wav = music.generateWav(allocator, req, prog) catch |err| {
        log.err("[music] generation failed: {}\n", .{err});
        if (want_stream) {
            sse.sendError(conn, "music generation failed");
            return;
        }
        return sendError(conn, 500, "music generation failed");
    };
    defer allocator.free(wav);
    log.info("[music] -> {d} WAV bytes\n", .{wav.len});
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

/// `POST /v1/audio/music-generations` — MiniMax Music 3 text2music.
/// `{"model", "prompt" (style/genre/mood caption, REQUIRED), "lyrics"
/// (REQUIRED unless `instrumental` — the model is lyric-conditioned; structure
/// tags like `[verse]` each on their own line), "instrumental" (bool; sends the
/// `[Instrumental]` section tag from MiniMax's own model card as the whole
/// lyric block — the open weights have no `is_instrumental` parameter, so text
/// is the only lever, and the tag is the same one ACE-Step uses),
/// "bpm" (30-300) and "keyscale" — no request field exists for these on this
/// engine, so they are folded into the caption as MiniMax's own
/// `BPM: 96. Key: C major.` (`music3.captionWithFacts`), skipped when the
/// prompt already says them; "duration_seconds" (default 60, valid 1-360, an
/// UPPER bound — the model may stop earlier), "steps" (flow-match steps,
/// default 30, valid 4-100), "seed", "stream"}`. ACE-Step's
/// bpm/keyscale/timesignature/vocal_language have NO equivalent here and are
/// named 400s rather than silent ignores. Response mirrors the ACE-Step
/// handler: raw `audio/wav` non-stream, SSE progress + base64 complete when
/// streaming.
fn handleMusic3(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, m3: *music3.Engine) !void {
    // `timesignature` and `vocal_language` stay named 400s: MiniMax's card
    // documents neither, and inventing caption text for an undocumented knob is
    // worse than saying we cannot honor it. `bpm` and `keyscale` USED to be
    // refused here too — wrongly. Global Metadata on that same card lists BPM,
    // key and scale, and its example caption reads
    // "Genre: acoustic pop. BPM: 96. Key: C major.", so they are supported;
    // they are just caption TEXT here rather than conditioning fields.
    for ([_][]const u8{ "timesignature", "vocal_language" }) |field| {
        const present = extractJsonString(body, field) != null or extractJsonInt(body, field) != null;
        if (present) {
            var msg: [160]u8 = undefined;
            const m = std.fmt.bufPrint(&msg, "'{s}' is an ACE-Step field; MiniMax Music 3 has no documented equivalent (put it in 'prompt' yourself)", .{field}) catch "unsupported field";
            return sendError(conn, 400, m);
        }
    }
    const raw_prompt = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt' (style/genre/mood description)");
    const prompt = try jsonUnescape(allocator, raw_prompt);
    defer allocator.free(prompt);
    if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");
    const instrumental = sse.bodyWantsTrue(body, "instrumental");
    var lyrics: []u8 = try allocator.dupe(u8, "");
    defer allocator.free(lyrics);
    if (extractJsonString(body, "lyrics")) |raw| {
        allocator.free(lyrics);
        lyrics = try jsonUnescape(allocator, raw);
    }
    if (instrumentalConflicts(instrumental, lyrics))
        return sendError(conn, 400, "'instrumental' is true but 'lyrics' is non-empty — send one or the other");
    if (extractJsonString(body, "ref_audio")) |raw| if (raw.len > 0)
        return sendError(conn, 400, "'ref_audio' is not supported by this model (MiniMax Music 3 takes no reference audio) — it is an ACE-Step field");
    if (extractJsonString(body, "src_audio")) |raw| if (raw.len > 0)
        return sendError(conn, 400, "'src_audio' is not supported by this model (MiniMax Music 3 has no cover / complete mode) — it is an ACE-Step field");
    if (extractJsonString(body, "task")) |raw| if (!std.mem.eql(u8, raw, "text2music"))
        return sendError(conn, 400, "'task' is not supported by this model (MiniMax Music 3 is text2music only) — cover / complete are ACE-Step tasks");
    if (!instrumental and std.mem.trim(u8, lyrics, " \t\r\n").len == 0)
        return sendError(conn, 400, "missing 'lyrics' (MiniMax Music 3 is lyric-conditioned; structure tags like [verse] go on their own lines, or send \"instrumental\": true)");
    const cond_lyrics = music3.resolveLyrics(instrumental, lyrics);

    // Tempo and key ride the CAPTION on this engine (see captionWithFacts), as
    // does the no-vocals intent — the lyric tag alone leaves vocal TEXTURE in
    // (measured 2026-08-18), and MiniMax's api marks `prompt` required for an
    // instrumental track while making `lyrics` optional. Built BEFORE the
    // 5000-token pre-check below so every added clause is COUNTED, never
    // smuggled past the cap.
    var bpm: ?u32 = null;
    if (extractJsonInt(body, "bpm")) |b| {
        if (b < 30 or b > 300) return sendError(conn, 400, "'bpm' must be in [30,300]");
        bpm = @intCast(b);
    }
    const keyscale = extractJsonString(body, "keyscale") orelse "";
    const caption = try music3.captionWithFacts(
        allocator,
        prompt,
        bpm,
        keyscale,
        instrumental and music3.instrumentalCaptionEnabled(),
    );
    defer allocator.free(caption);
    const caption_grew = caption.len != prompt.len;

    const duration: u32 = @intCast(extractJsonInt(body, "duration_seconds") orelse 60);
    if (duration < music3.MIN_DURATION_S or duration > music3.MAX_DURATION_S)
        return sendError(conn, 400, "'duration_seconds' must be in [1,360]");
    const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse music3.DEFAULT_STEPS);
    if (steps < 4 or steps > 100) return sendError(conn, 400, "'steps' must be in [4,100]");
    const seed: u64 = extractJsonInt(body, "seed") orelse 42;

    // Pre-validate the prompt budget BEFORE any SSE bytes go out, so the cap
    // is a clean named 400 instead of a mid-stream error.
    {
        const toks = m3.tokenizePrompt(allocator, caption, cond_lyrics) catch |err| switch (err) {
            error.PromptTooLong => return sendError(conn, 400, "assembled prompt exceeds 5000 tokens"),
            else => return err,
        };
        allocator.free(toks.ids);
        allocator.free(toks.uncond);
    }

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[music3] generating {d}s steps={d} seed={d} lyrics={d}ch instrumental={} caption_facts={} stream={}\n", .{ duration, steps, seed, cond_lyrics.len, instrumental, caption_grew, want_stream });
    var sctx = sse.StreamCtx{ .conn = conn, .stream = want_stream };
    const prog: ?sse.Progress = sctx.progress();
    if (want_stream) try conn.writeAll(sse.headers);

    const req = music3.MusicRequest{
        .caption = caption,
        .lyrics = cond_lyrics,
        .duration_s = duration,
        .seed = seed,
        .steps = steps,
    };
    const wav = m3.generateWav(allocator, req, prog) catch |err| {
        if (err == error.Cancelled) {
            log.info("[music3] generation cancelled by client\n", .{});
            return;
        }
        log.err("[music3] generation failed: {}\n", .{err});
        if (want_stream) {
            sse.sendError(conn, "music generation failed");
            return;
        }
        return sendError(conn, 500, "music generation failed");
    };
    defer allocator.free(wav);
    log.info("[music3] -> {d} WAV bytes\n", .{wav.len});
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

/// H3 result -> the SAME wire shape the LTX path emits (base64 rgb8 frames plus
/// interleaved pcm_s16le), so the Swift client's existing decode and
/// AVAssetWriter mux need no new branch.
fn sendH3Video(allocator: std.mem.Allocator, conn: *Conn, res: *const minimax_h3.GenResult, want_stream: bool) !void {
    const s = mlx.gpuStream();
    const audio_mod = @import("minimax_h3_audio.zig");

    // [1,3,F,H,W] in [-1,1] -> [F,H,W,3] u8
    const rgb = try minimax_h3.pixelsToRgb8(allocator, res, s);
    defer allocator.free(rgb);
    log.info("[video] -> {d}f {d}x{d} ({d} rgb bytes)\n", .{ res.frame_count, res.height, res.width, rgb.len });

    const b64_len = std.base64.standard.Encoder.calcSize(rgb.len);
    const b64 = try allocator.alloc(u8, b64_len);
    defer allocator.free(b64);
    _ = std.base64.standard.Encoder.encode(b64, rgb);

    var audio_b64: ?[]u8 = null;
    defer if (audio_b64) |a| allocator.free(a);
    if (res.audio) |wave| {
        const pcm = try minimax_h3.audioToPcm16(allocator, wave, s);
        defer allocator.free(pcm);
        const al = std.base64.standard.Encoder.calcSize(pcm.len);
        const ab = try allocator.alloc(u8, al);
        _ = std.base64.standard.Encoder.encode(ab, pcm);
        audio_b64 = ab;
    }

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    const prefix = if (want_stream) "data: {\"type\":\"complete\"," else "{\"created\":0,";
    const head = try std.fmt.allocPrint(allocator, "{s}\"frames\":{d},\"height\":{d},\"width\":{d},\"fps\":24,\"format\":\"rgb8\",\"data\":\"", .{ prefix, res.frame_count, res.height, res.width });
    defer allocator.free(head);
    try out.appendSlice(allocator, head);
    try out.appendSlice(allocator, b64);
    try out.appendSlice(allocator, "\"");
    if (audio_b64) |ab| {
        const ah = try std.fmt.allocPrint(allocator, ",\"audio_sample_rate\":{d},\"audio_channels\":2,\"audio_format\":\"pcm_s16le\",\"audio_data\":\"", .{audio_mod.SAMPLE_RATE});
        defer allocator.free(ah);
        try out.appendSlice(allocator, ah);
        try out.appendSlice(allocator, ab);
        try out.appendSlice(allocator, "\"");
    }
    try out.appendSlice(allocator, if (want_stream) "}\n\n" else "}");
    if (want_stream) return conn.writeAll(out.items);
    return sendBytesJson(conn, allocator, out.items);
}

/// Stage-2 transformer provider for the two-stage boundary: swaps the engine's
/// transformer slot from dev to distilled (freeing dev first).
const Stage2Swap = struct {
    engine: *LtxVideoEngine,

    fn swap(ctx: *anyopaque) anyerror!*const ltx.Component {
        const self: *Stage2Swap = @ptrCast(@alignCast(ctx));
        try self.engine.ensureTransformer(.distilled);
        return &self.engine.transformer;
    }
};

pub fn handleVideo(io: std.Io, allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *VideoEngine) !void {
    return switch (engine.backend) {
        .ltx => |e| handleVideoLtx(io, allocator, conn, body, e),
        .h3 => |e| handleVideoH3(io, allocator, conn, body, e),
    };
}

/// SSE sink for a video request. Preview JPEGs are opt-in and require stream.
fn videoStreamCtx(conn: *Conn, allocator: std.mem.Allocator, body: []const u8, want_stream: bool) sse.StreamCtx {
    const pr = sse.parsePreview(body);
    return .{
        .conn = conn,
        .stream = want_stream,
        .allocator = allocator,
        .preview = want_stream and pr.enabled,
        .preview_frames = pr.frames,
        .preview_max_side = pr.max_side,
    };
}

/// Base64-decode a JSON string value (unescaping `\/` first — Swift clients
/// escape every slash) into an owned buffer.
fn jsonB64Alloc(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    const unescaped = try jsonUnescape(allocator, raw);
    defer allocator.free(unescaped);
    return base64DecodeAlloc(allocator, unescaped);
}

/// One reference's bytes, before `resolveRefs` has decided its canvas. Images
/// and video frames stay ENCODED — the canvas is not known yet and the server
/// resizes by decoding AT a size, so there is nothing useful to decode into.
const PendingRefs = struct {
    allocator: std.mem.Allocator,
    images: std.ArrayList([]u8) = .empty,
    /// Per reference video: its encoded frames, then its 32 kHz stereo
    /// soundtrack (interleaved) if it carries one.
    video_frames: std.ArrayList(std.ArrayList([]u8)) = .empty,
    video_audio: std.ArrayList(?[]f32) = .empty,
    audios: std.ArrayList([]f32) = .empty,

    fn deinit(self: *PendingRefs) void {
        const a = self.allocator;
        for (self.images.items) |b| a.free(b);
        self.images.deinit(a);
        for (self.video_frames.items) |*fr| {
            for (fr.items) |b| a.free(b);
            fr.deinit(a);
        }
        self.video_frames.deinit(a);
        for (self.video_audio.items) |p| if (p) |x| a.free(x);
        self.video_audio.deinit(a);
        for (self.audios.items) |p| a.free(p);
        self.audios.deinit(a);
    }
};

/// Decode a base64 WAV to 32 kHz STEREO interleaved f32 — the audio VAE's rate.
/// The VAE is not sensitive to sub-sample resample accuracy, so the linear
/// resampler the a2vid conditioning path already uses is enough here too.
fn refWavTo32kStereo(allocator: std.mem.Allocator, raw_b64: []const u8) ![]f32 {
    const wav_bytes = try jsonB64Alloc(allocator, raw_b64);
    defer allocator.free(wav_bytes);
    const dec = try wav_mod.decode(allocator, wav_bytes);
    defer allocator.free(dec.pcm);
    const stereo = try wav_mod.toStereoInterleaved(allocator, dec.pcm, dec.channels);
    defer allocator.free(stereo);
    return wav_mod.resampleLinear(allocator, stereo, 2, dec.sample_rate, minimax_h3.audio_mod.SAMPLE_RATE);
}

/// `[2, L]` f32 in [-1, 1] from interleaved stereo — the audio encoder's shape
/// (the stereo channels ride the BATCH axis; the encoder itself is mono).
fn stereoInterleavedToArray(allocator: std.mem.Allocator, pcm: []const f32) !mlx.mlx_array {
    const frames = pcm.len / 2;
    const planar = try allocator.alloc(f32, frames * 2);
    defer allocator.free(planar);
    for (0..frames) |i| {
        planar[i] = pcm[i * 2];
        planar[frames + i] = pcm[i * 2 + 1];
    }
    const shp = [_]c_int{ 2, @intCast(frames) };
    return mlx.mlx_array_new_data(planar.ptr, &shp, 2, mlx.mlx_dtype.float32);
}

/// Parse and decode the ref2va reference fields into `out` (caller owns every
/// entry). Returns null on success, or the NAMED 400 message to send — a
/// reference the server cannot use must never be silently dropped, because the
/// generation then succeeds while ignoring what the user asked it to follow.
///
/// `err_buf` backs the messages that name an index; the rest are literals.
fn parseH3Refs(
    allocator: std.mem.Allocator,
    body: []const u8,
    gen_w: u32,
    gen_h: u32,
    gen_frames: u32,
    out: *std.ArrayList(minimax_h3.RefMedia),
    err_buf: []u8,
) !?[]const u8 {
    const mode: minimax_h3.RefImageSizing = blk: {
        const raw = extractJsonString(body, "ref_image_size") orelse break :blk .match;
        if (std.mem.eql(u8, raw, "match")) break :blk .match;
        if (std.mem.eql(u8, raw, "max")) break :blk .max;
        return "'ref_image_size' must be \"match\" or \"max\"";
    };

    var pend = PendingRefs{ .allocator = allocator };
    defer pend.deinit();

    if (iterJsonStringArray(body, "ref_images")) |it0| {
        var it = it0;
        while (it.next()) |b64| {
            const bytes = jsonB64Alloc(allocator, b64) catch
                return std.fmt.bufPrint(err_buf, "'ref_images'[{d}] is not valid base64", .{pend.images.items.len}) catch
                    "a 'ref_images' entry is not valid base64";
            try pend.images.append(allocator, bytes);
        }
        if (it.bad) return "'ref_images' must be an array of base64 PNG/JPEG strings";
    }

    if (iterJsonObjectArray(body, "ref_videos")) |it0| {
        var it = it0;
        while (it.next()) |obj| {
            const vi = pend.video_frames.items.len;
            // Handed to `pend` EMPTY and filled through its own slot: every
            // rejection below is a named 400, i.e. a NORMAL return, so an
            // errdefer would not fire and the frames decoded so far would leak.
            try pend.video_frames.append(allocator, .empty);
            const frames = &pend.video_frames.items[vi];
            var fit = iterJsonStringArray(obj, "frames") orelse
                return std.fmt.bufPrint(err_buf, "'ref_videos'[{d}] needs a 'frames' array of base64 PNG/JPEG strings", .{vi}) catch
                    "a 'ref_videos' entry needs a 'frames' array";
            while (fit.next()) |b64| {
                const bytes = jsonB64Alloc(allocator, b64) catch
                    return std.fmt.bufPrint(err_buf, "'ref_videos'[{d}].frames[{d}] is not valid base64", .{ vi, frames.items.len }) catch
                        "a 'ref_videos' frame is not valid base64";
                try frames.append(allocator, bytes);
            }
            if (fit.bad)
                return std.fmt.bufPrint(err_buf, "'ref_videos'[{d}].frames must be an array of base64 PNG/JPEG strings", .{vi}) catch
                    "a 'ref_videos' frames array is malformed";
            // The soundtrack is a FIELD on its own video object, so a missing
            // one cannot shift the pairing the way a parallel array's hole did.
            var track: ?[]f32 = null;
            if (extractJsonString(obj, "audio")) |raw| {
                if (raw.len > 0) {
                    track = refWavTo32kStereo(allocator, raw) catch
                        return std.fmt.bufPrint(err_buf, "'ref_videos'[{d}].audio must be a PCM16/PCM24/float32 WAV", .{vi}) catch
                            "a 'ref_videos' soundtrack could not be decoded";
                }
            }
            try pend.video_audio.append(allocator, track);
        }
        if (it.bad) return "'ref_videos' must be an array of {\"frames\":[…],\"audio\":\"…\"} objects";
    }

    if (iterJsonStringArray(body, "ref_audios")) |it0| {
        var it = it0;
        while (it.next()) |b64| {
            const pcm = refWavTo32kStereo(allocator, b64) catch
                return std.fmt.bufPrint(err_buf, "'ref_audios'[{d}] must be a PCM16/PCM24/float32 WAV", .{pend.audios.items.len}) catch
                    "a 'ref_audios' entry could not be decoded";
            try pend.audios.append(allocator, pcm);
        }
        if (it.bad) return "'ref_audios' must be an array of base64 WAV strings";
    }

    const n_total = pend.images.items.len + pend.video_frames.items.len + pend.audios.items.len;
    if (n_total == 0) return null;

    // Source dimensions, which is all `resolveRefs` sizes from. A reference
    // whose pixels cannot even be measured is a 400 here rather than a decode
    // failure three steps later with no field name attached to it.
    var inputs_i = try allocator.alloc(minimax_h3.RefInput, pend.images.items.len);
    defer allocator.free(inputs_i);
    for (pend.images.items, 0..) |bytes, i| {
        const sz = imageNativeSize(bytes) orelse
            return std.fmt.bufPrint(err_buf, "could not decode 'ref_images'[{d}] (PNG/JPEG expected)", .{i}) catch
                "a 'ref_images' entry could not be decoded (PNG/JPEG expected)";
        inputs_i[i] = .{ .kind = .image, .w = sz.w, .h = sz.h };
    }
    var inputs_v = try allocator.alloc(minimax_h3.RefInput, pend.video_frames.items.len);
    defer allocator.free(inputs_v);
    for (pend.video_frames.items, 0..) |fr, i| {
        if (fr.items.len == 0)
            return std.fmt.bufPrint(err_buf, "'ref_videos'[{d}] has no frames", .{i}) catch "a 'ref_videos' entry has no frames";
        const sz = imageNativeSize(fr.items[0]) orelse
            return std.fmt.bufPrint(err_buf, "could not decode 'ref_videos'[{d}].frames[0] (PNG/JPEG expected)", .{i}) catch
                "a 'ref_videos' frame could not be decoded (PNG/JPEG expected)";
        // The reference node truncates a reference longer than the generation
        // before snapping it to the ladder; a clip the output cannot span
        // costs sampling rows for footage the model can never reach.
        const supplied: u32 = @intCast(fr.items.len);
        inputs_v[i] = .{
            .kind = .video,
            .w = sz.w,
            .h = sz.h,
            .frames = @min(supplied, gen_frames),
            .audio_samples = 0,
            .soundtrack_samples = if (pend.video_audio.items[i]) |p| @intCast(p.len / 2) else null,
        };
    }
    var inputs_a = try allocator.alloc(minimax_h3.RefInput, pend.audios.items.len);
    defer allocator.free(inputs_a);
    for (pend.audios.items, 0..) |pcm, i| {
        inputs_a[i] = .{ .kind = .audio, .audio_samples = @intCast(pcm.len / 2) };
    }

    var res = switch (try minimax_h3.resolveRefs(allocator, inputs_i, inputs_v, inputs_a, gen_w, gen_h, mode)) {
        .ok => |r| r,
        .reject => |why| return why.message(),
    };
    defer res.deinit();

    // Decode at the canvases the resolver picked — the VAE canvas for the DiT
    // payload, and the Qwen canvas for the vision tower. Two decodes rather
    // than one resize, because the server has no image resampler.
    for (res.refs) |r| {
        // Appended EMPTY first and filled through the caller's own slot: a
        // named-400 return is a NORMAL return, so an errdefer would not fire
        // and a second decode failing would strand the first array. The
        // caller's deinit loop owns every partially-filled entry.
        try out.append(allocator, .{ .ref = r });
        const media = &out.items[out.items.len - 1];
        switch (r.kind) {
            .image => {
                const bytes = pend.images.items[r.src_index];
                const vfit = minimax_h3.h3v.fitCanvas(r.canvas.h, r.canvas.w);
                const bad = std.fmt.bufPrint(err_buf, "could not decode 'ref_images'[{d}] (PNG/JPEG expected)", .{r.src_index}) catch
                    "a 'ref_images' entry could not be decoded (PNG/JPEG expected)";
                media.pixels = decodeImageToBCFHW(allocator, bytes, r.canvas.h, r.canvas.w, mlx.gpuStream()) orelse return bad;
                media.vision = decodeImageToBCFHW(allocator, bytes, vfit.h, vfit.w, mlx.gpuStream()) orelse return bad;
            },
            .video => {
                const fr = pend.video_frames.items[r.src_index].items;
                const vfit = minimax_h3.h3v.fitCanvas(r.canvas.h, r.canvas.w);
                const bad = std.fmt.bufPrint(err_buf, "could not decode a 'ref_videos'[{d}] frame (PNG/JPEG expected)", .{r.src_index}) catch
                    "a 'ref_videos' frame could not be decoded (PNG/JPEG expected)";
                media.pixels = try decodeRefVideoFrames(allocator, fr[0..r.frames], r.canvas.h, r.canvas.w, 1, 2) orelse return bad;
                media.vision = try decodeRefVideoFrames(allocator, fr[0..r.frames], vfit.h, vfit.w, minimax_h3.qwenFrameStride(), 0) orelse return bad;
                if (pend.video_audio.items[r.src_index]) |pcm|
                    media.waveform = try stereoInterleavedToArray(allocator, pcm);
            },
            .audio => media.waveform = try stereoInterleavedToArray(allocator, pend.audios.items[r.src_index]),
        }
    }
    return null;
}

/// Decode every `stride`-th encoded frame at `th`x`tw` and stack them.
/// `cat_axis` 2 gives the VAE's `[1,3,T,H,W]`; 0 gives the vision tower's
/// `[n,3,H,W]`. Null when any frame fails to decode.
fn decodeRefVideoFrames(
    allocator: std.mem.Allocator,
    frames: []const []u8,
    th: u32,
    tw: u32,
    stride: u32,
    cat_axis: c_int,
) !?mlx.mlx_array {
    var parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (parts.items) |p| _ = mlx.mlx_array_free(p);
        parts.deinit(allocator);
    }
    var i: usize = 0;
    while (i < frames.len) : (i += stride) {
        const bcfhw = decodeImageToBCFHW(allocator, frames[i], th, tw, mlx.gpuStream()) orelse return null;
        if (cat_axis == 2) {
            try parts.append(allocator, bcfhw);
        } else {
            // [1,3,1,H,W] -> [1,3,H,W] so the stack lands on the frame axis.
            defer _ = mlx.mlx_array_free(bcfhw);
            var four = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(four);
            const shp4 = [_]c_int{ 1, 3, @intCast(th), @intCast(tw) };
            try mlx.check(mlx.mlx_reshape(&four, bcfhw, &shp4, 4, mlx.gpuStream()));
            try parts.append(allocator, four);
        }
    }
    if (parts.items.len == 0) return null;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (parts.items) |p| _ = mlx.mlx_vector_array_append_value(vec, p);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, cat_axis, mlx.gpuStream()));
    return o;
}

/// MiniMax-H3 text-to-audio-video.
///
/// The request surface is deliberately NARROWER than LTX's: H3 has no CFG
/// scale and no pipeline mode, and its frame counts live on a 17k+5 ladder
/// rather than 8N+1. Anything the client sends that this backend cannot honor
/// is a NAMED 400 — a silently ignored field is the class the app's preset
/// rules exist to prevent. LoRAs it DOES take: `lora_paths`/`lora_scales`
/// (or the legacy singular pair), stacking with the engine-owned Turbo
/// distillation adapter that `"turbo": true` attaches.
fn handleVideoH3(io: std.Io, allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *H3VideoEngine) !void {
    const prompt_raw = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt'");
    const prompt = try jsonUnescape(allocator, prompt_raw);
    defer allocator.free(prompt);
    if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

    // `cfg_scale` / `stg_scale` / `pipeline` are NOT rejected: the app sends
    // them unconditionally for every video backend, so 400-ing on their
    // PRESENCE made every H3 request fail. Hiding a control is not the same
    // as not sending the field. They are ignored, which is honest here — H3
    // is CFG-distilled and single-pipeline, so there is no setting they could
    // have selected that we silently dropped.

    // Style LoRA(s), through the SAME parser the image and LTX handlers use —
    // one grammar for `lora_paths`/`lora_scales` (and the legacy singular
    // pair) across every backend that takes adapters. They stack with Turbo:
    // the engine sums every attached delta on each linear.
    var lora_path_bufs: [lora_mod.MAX_LORAS][]u8 = undefined;
    var lora_scales: [lora_mod.MAX_LORAS]f32 = undefined;
    const lora_n = parseLoraFields(allocator, body, &lora_path_bufs, &lora_scales) catch |err| switch (err) {
        error.TooManyLoraPaths => return sendError(conn, 400, "too many 'lora_paths' (max 8)"),
        error.BadLoraPathsJson => return sendError(conn, 400, "invalid 'lora_paths' (must be a JSON array of strings)"),
        error.BadLoraScalesJson => return sendError(conn, 400, "invalid 'lora_scales' (numbers, comma/space separated, or a JSON array)"),
        error.OutOfMemory => return err,
    };
    defer for (lora_path_bufs[0..lora_n]) |p| allocator.free(p);
    var lora_paths: [lora_mod.MAX_LORAS][]const u8 = undefined;
    for (lora_path_bufs[0..lora_n], 0..) |p, i| lora_paths[i] = p;
    // Paths are proven HERE, not inside the engine: H3 loads its DiT per
    // request, so `loadFile`'s own check is reached minutes into a generation
    // — long after the handler could answer 400. Same rule either way
    // (`lora.validatePath`), so the two cannot disagree.
    for (lora_paths[0..lora_n]) |p| {
        lora_mod.validatePath(p) catch
            return sendError(conn, 400, "'lora_paths' must be absolute paths to readable .safetensors files");
    }

    // Turbo: the 4-step distillation LoRA (larryvrh/MiniMax-H3-Turbo-Lora).
    // Steps default to 4, which is also the floor: on the ema_ckpt850 weights
    // the mirrors now ship, 4 steps is already sharp — the 6-8 default came
    // from ckpt500, which needed the extra steps to firm up. 30 stays the
    // non-turbo default.
    const turbo = sse.bodyWantsTrue(body, "turbo");
    if (turbo and !engine.hasTurboLora(io, allocator))
        return sendError(conn, 400, "this pack has no turbo_lora.safetensors — download minimax_h3_turbo_4step_ema_ckpt850.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora (Apache-2.0) into the model folder as turbo_lora.safetensors");

    const width: u32 = @intCast(extractJsonInt(body, "width") orelse 256);
    const height: u32 = @intCast(extractJsonInt(body, "height") orelse 256);
    const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse (if (turbo) @as(u64, 4) else 30));
    const seed: u64 = @intCast(extractJsonInt(body, "seed") orelse 0);
    const requested_frames: u32 = @intCast(extractJsonInt(body, "num_frames") orelse 56);

    if (width % 32 != 0 or height % 32 != 0)
        return sendError(conn, 400, "width and height must be multiples of 32");
    if (turbo and steps < 4)
        return sendError(conn, 400, "turbo needs at least 4 steps (the distillation's own floor, and its default)");

    // Chained windows: N back-to-back `num_frames`-frame windows, each
    // conditioned on its predecessor's last decoded frame through the fl2va
    // path. `num_frames` is PER WINDOW; the response reports the delivered
    // (joined) count. Needs the FL2VA pack — a reference has no keyframe row
    // to chain through — so the REF2VA partition is refused by name.
    const chain_windows: u32 = @intCast(extractJsonInt(body, "chain_windows") orelse 1);
    if (chain_windows < 1 or chain_windows > 6)
        return sendError(conn, 400, "chain_windows must be 1-6");
    if (chain_windows > 1 and engine.supports_refs)
        return sendError(conn, 400, "chained windows ride FL2VA keyframe conditioning — the REF2VA pack cannot serve them; load an FL2VA checkpoint");

    // Snap to the model's own ladder and SAY SO: silently generating a
    // different length than asked is how a client's audio mux drifts.
    const shape = minimax_h3.temporalShape(requested_frames);
    if (videoRgbTransportReason(minimax_h3.chainDeliveredFrames(chain_windows, shape.frame_count), width, height)) |reason|
        return sendError(conn, 400, reason);
    const want_stream = sse.bodyWantsTrue(body, "stream");
    const preview_req = sse.parsePreview(body);
    log.info("[video] minimax-h3 {d}x{d} {d}f/window (requested {d}, snapped to the 17k+5 ladder) steps={d} turbo={} loras={d} chain={d} stream={} preview={}\n", .{ width, height, shape.frame_count, requested_frames, steps, turbo, lora_n, chain_windows, want_stream, preview_req.enabled });

    // fl2va keyframes. NOT graceful (the a2vid rule: the user asked for THIS
    // frame): an undecodable image is a named 400, never a silent t2va. The
    // reference's resize policy per anchor: first = plain STRETCH to the
    // canvas (the geometry anchor), last = aspect-preserving center-COVER.
    var keyframes_buf: [2]minimax_h3.Keyframe = undefined;
    var n_kf: usize = 0;
    defer for (keyframes_buf[0..n_kf]) |kf| {
        _ = mlx.mlx_array_free(kf.pixels);
        if (kf.vision) |v| _ = mlx.mlx_array_free(v);
    };
    // The keyframe also enters the Qwen conditioning as a `<Picture i>` block,
    // which has its OWN canvas rule (multiple of 32 above a 3136-pixel floor).
    // On every servable target the two agree and the VAE copy is reused; below
    // the floor they diverge, so decode a second copy at the vision canvas
    // rather than patchify pixels whose grid says something else.
    const vfit = minimax_h3.h3v.fitCanvas(height, width);
    const vision_differs = vfit.h != height or vfit.w != width;
    inline for (.{ .{ "first_frame_image", minimax_h3.KeyframeAnchor.first }, .{ "last_frame_image", minimax_h3.KeyframeAnchor.last } }) |spec| {
        if (extractJsonString(body, spec[0])) |raw_img| {
            const b64 = try jsonUnescape(allocator, raw_img);
            defer allocator.free(b64);
            if (b64.len > 0) {
                const img_bytes = base64DecodeAlloc(allocator, b64) catch
                    return sendError(conn, 400, "keyframe image is not valid base64");
                defer allocator.free(img_bytes);
                const decodeAt = struct {
                    fn f(a: std.mem.Allocator, bytes: []const u8, anchor: minimax_h3.KeyframeAnchor, th: u32, tw: u32) ?mlx.mlx_array {
                        return switch (anchor) {
                            // first = geometry anchor, plain STRETCH;
                            // last = follower, aspect-preserving center-COVER.
                            .first => decodeImageToBCFHW(a, bytes, th, tw, mlx.gpuStream()),
                            .last => blk: {
                                const bchw = decodeImageToBCHW(a, bytes, th, tw) orelse break :blk null;
                                defer _ = mlx.mlx_array_free(bchw);
                                break :blk unitToPm1BCFHW(bchw, th, tw, mlx.gpuStream()) catch null;
                            },
                        };
                    }
                }.f;
                const arr = decodeAt(allocator, img_bytes, spec[1], height, width);
                if (arr == null) return sendError(conn, 400, "keyframe image could not be decoded (PNG/JPEG expected)");
                const vis: ?mlx.mlx_array = if (!vision_differs) null else blk: {
                    const v = decodeAt(allocator, img_bytes, spec[1], vfit.h, vfit.w);
                    if (v == null) {
                        _ = mlx.mlx_array_free(arr.?);
                        return sendError(conn, 400, "keyframe image could not be decoded (PNG/JPEG expected)");
                    }
                    break :blk v;
                };
                keyframes_buf[n_kf] = .{ .anchor = spec[1], .pixels = arr.?, .vision = vis };
                n_kf += 1;
                log.info("[video] minimax-h3 {s} keyframe conditioning engaged ({d}x{d}, vision block {d}x{d})\n", .{ @tagName(spec[1]), width, height, vfit.w, vfit.h });
            }
        }
    }

    // ── ref2va references ──
    // Refused on an FL2VA pack rather than ignored: both partitions ship the
    // same files and the same geometry, so nothing downstream would notice, and
    // the generation would come back looking like the model ignored the user.
    var refs: std.ArrayList(minimax_h3.RefMedia) = .empty;
    defer {
        for (refs.items) |*m| m.deinit();
        refs.deinit(allocator);
    }
    const has_ref_fields = std.mem.indexOf(u8, body, "\"ref_images\"") != null or
        std.mem.indexOf(u8, body, "\"ref_videos\"") != null or
        std.mem.indexOf(u8, body, "\"ref_audios\"") != null;
    if (has_ref_fields and !engine.supports_refs)
        return sendError(conn, 400, "this MiniMax-H3 pack does not support references (it declares no 'ref2va' task) — load a REF2VA checkpoint");
    if (has_ref_fields) {
        var err_buf: [256]u8 = undefined;
        if (try parseH3Refs(allocator, body, width, height, shape.frame_count, &refs, &err_buf)) |msg|
            return sendError(conn, 400, msg);
        for (refs.items) |m| {
            log.info("[video] minimax-h3 reference {s} #{d}: {d}x{d} {d}f, latent {d}x{d}x{d}, audio_t {d}\n", .{
                @tagName(m.ref.kind), m.ref.ordinal,  m.ref.canvas.w, m.ref.canvas.h,
                m.ref.frames,         m.ref.latent_t, m.ref.latent_h, m.ref.latent_w,
                m.ref.audio_t,
            });
        }
    }

    // Generations here run for MINUTES, so a silent socket is indistinguishable
    // from a wedged server; the client drives its meter off these events. The
    // progress sink is handed over on BOTH paths — a non-streaming request
    // emits nothing but still gets the disconnect probe, or a client that gives
    // up (or times out: a 1344x768 clip outlasts most default timeouts) leaves
    // the GPU running to the end with every queued request behind it.
    var sctx = videoStreamCtx(conn, allocator, body, want_stream);
    const prog: ?sse.Progress = sctx.progress();
    if (want_stream) try conn.writeAll(sse.headers);

    const paths = try engine.paths(allocator);
    defer {
        allocator.free(paths.text_encoder);
        allocator.free(paths.dit);
        allocator.free(paths.vae);
        if (paths.audio_vae) |p| allocator.free(p);
        if (paths.turbo_lora) |p| allocator.free(p);
    }

    var res = minimax_h3.generate(allocator, io, paths, .{
        .prompt = prompt,
        .width = width,
        .height = height,
        .frames = requested_frames,
        .steps = steps,
        .seed = seed,
        .fast = sse.bodyBool(body, "fast"),
        .turbo = turbo,
        .lora_paths = lora_paths[0..lora_n],
        .lora_scales = lora_scales[0..lora_n],
        .chain_windows = chain_windows,
        .keyframes = keyframes_buf[0..n_kf],
        .refs = refs.items,
    }, prog, mlx.gpuStream()) catch |e| {
        // Client hung up mid-generation — there is nobody to answer, and
        // saying "generation failed" would be a lie about a job we stopped.
        if (e == error.Cancelled) {
            log.info("[video] generation cancelled — client disconnected\n", .{});
            return;
        }
        // The LoRA failures are the user's to fix and each has a distinct
        // remedy, so they are named 400s rather than one opaque 500 — the
        // whole reason a wrong-architecture adapter is worth detecting at all.
        const named: ?[]const u8 = switch (e) {
            error.LoraNoMatch => "a LoRA has no modules matching MiniMax-H3's DiT — wrong architecture for this adapter?",
            error.BadLoraPath => "'lora_paths' must be absolute paths to .safetensors files",
            error.TooManyLoras => "too many LoRA adapters (max 8, and turbo takes one of the slots)",
            error.TurboLoraIncomplete => "turbo_lora.safetensors is incomplete — re-download minimax_h3_turbo_4step_ckpt500.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora",
            else => null,
        };
        if (named) |msg| {
            log.err("[video] minimax-h3 lora: {any}\n", .{e});
            if (want_stream) return sse.sendError(conn, msg);
            return sendError(conn, 400, msg);
        }
        log.err("[video] minimax-h3 generation failed: {any}\n", .{e});
        // Mid-stream the headers are already out, so an error must be an SSE
        // event, not a status line the client will never parse.
        if (want_stream) return sse.sendError(conn, "MiniMax-H3 generation failed");
        return sendError(conn, 500, "MiniMax-H3 generation failed");
    };
    defer res.deinit();

    try sendH3Video(allocator, conn, &res, want_stream);
}

fn handleVideoLtx(io: std.Io, allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *LtxVideoEngine) !void {
    const prompt_raw = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt'");
    const prompt = try jsonUnescape(allocator, prompt_raw);
    defer allocator.free(prompt);
    if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

    const num_frames: u32 = @intCast(extractJsonInt(body, "num_frames") orelse 9);
    const height: u32 = @intCast(extractJsonInt(body, "height") orelse 256);
    const width: u32 = @intCast(extractJsonInt(body, "width") orelse 384);
    if (videoRgbTransportReason(num_frames, width, height)) |reason|
        return sendError(conn, 400, reason);
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
    const preview_req = sse.parsePreview(body);
    log.info("[video] generating {s} {d}f {d}x{d} steps={d} cfg={d:.1}/{d:.1} stg={d:.1} stream={} preview={}: {d} chars\n", .{ @tagName(pipeline), num_frames, height, width, steps, guiders.vp.cfg, guiders.ap.cfg, guiders.vp.stg, want_stream, preview_req.enabled, prompt.len });

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

    // Style LoRA(s): one or more absolute paths to .safetensors adapters,
    // each with an optional scale, applied to the DiT at runtime. Accepts
    // the array form (`lora_paths`/`lora_scales`) or the original
    // single-adapter form (`lora_path`/`lora_scale`) — same contract as
    // handleImage. No LoRA fields in the request detaches whatever was
    // attached before.
    {
        var lora_path_bufs: [lora_mod.MAX_LORAS][]u8 = undefined;
        var lora_scales: [lora_mod.MAX_LORAS]f32 = undefined;
        const lora_n = parseLoraFields(allocator, body, &lora_path_bufs, &lora_scales) catch |err| switch (err) {
            error.TooManyLoraPaths => return sendError(conn, 400, "too many 'lora_paths' (max 8)"),
            error.BadLoraPathsJson => return sendError(conn, 400, "invalid 'lora_paths' (must be a JSON array of strings)"),
            error.BadLoraScalesJson => return sendError(conn, 400, "invalid 'lora_scales' (numbers, comma/space separated, or a JSON array)"),
            error.OutOfMemory => return err,
        };
        defer for (lora_path_bufs[0..lora_n]) |p| allocator.free(p);

        var lora_paths: [lora_mod.MAX_LORAS][]const u8 = undefined;
        for (lora_path_bufs[0..lora_n], 0..) |p, i| lora_paths[i] = p;
        const matched = engine.setLoras(lora_paths[0..lora_n], lora_scales[0..lora_n]) catch |err| switch (err) {
            error.LoraNoMatch => return sendError(conn, 400, "LoRA(s) have no modules matching this model's DiT — wrong LoRA for this architecture?"),
            error.BadLoraPath => return sendError(conn, 400, "'lora_path'/'lora_paths' must be absolute path(s) to .safetensors file(s)"),
            error.TooManyLoras => return sendError(conn, 400, "too many LoRA adapters requested"),
            error.OutOfMemory => return err,
            else => return sendError(conn, 400, "failed to load a LoRA file"),
        };
        if (lora_n > 0)
            log.info("[video] lora: matched {d} module-attachment(s) across {d} adapter(s)\n", .{ matched, lora_n });
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

    // Last-frame anchor (#260): `last_frame_image` pins the LAST latent frame
    // the same way. NOT graceful — the user asked for this ending, so a silent
    // text-to-video downgrade is a wrong result: named 400s instead. The
    // two-stage half-grid pair mirrors the first-frame one.
    var last_img: ?mlx.mlx_array = null;
    defer if (last_img) |c| {
        _ = mlx.mlx_array_free(c);
    };
    var last_img_half: ?mlx.mlx_array = null;
    defer if (last_img_half) |c| {
        _ = mlx.mlx_array_free(c);
    };
    if (extractJsonString(body, "last_frame_image")) |raw_img| {
        const b64 = try jsonUnescape(allocator, raw_img);
        defer allocator.free(b64);
        if (b64.len > 0) {
            const ve = if (engine.vae_encoder) |*e| e else return sendError(conn, 400, "last frame conditioning needs vae_encoder.safetensors — download it into the model dir");
            const img_bytes = base64DecodeAlloc(allocator, b64) catch return sendError(conn, 400, "last_frame_image: invalid base64");
            defer allocator.free(img_bytes);
            if (num_frames < 9) return sendError(conn, 400, "last_frame_image needs at least 9 frames (one latent frame cannot hold an anchor and still generate)");
            const enc_h = (height / 32) * 32;
            const enc_w = (width / 32) * 32;
            last_img = decodeImageToBCFHW(allocator, img_bytes, enc_h, enc_w, engine.s) orelse return sendError(conn, 400, "last_frame_image: expected a PNG/JPEG image");
            if (pipeline != .one_stage) {
                const half_h = ((height / 2) / 32) * 32;
                const half_w = ((width / 2) / 32) * 32;
                last_img_half = decodeImageToBCFHW(allocator, img_bytes, half_h, half_w, engine.s) orelse return sendError(conn, 400, "last_frame_image: expected a PNG/JPEG image");
            }
            enc_ptr = ve;
            log.info("[video] last frame anchor {d}x{d}\n", .{ enc_h, enc_w });
        }
    }

    // `decoder`: which VAE turns the final latent into pixels. Default is the
    // conv `vae_decoder` we have always shipped; `diffusion` is LTX's own
    // DiffVAE, which their published clips are decoded with and which only the
    // 8-bit pack ships. An absent file is a NAMED 400, never a silent downgrade.
    var vae_choice = ltx.VaeChoice{ .conv = &engine.vae, .seed = seed +% 0x5DEC0DE };
    if (extractJsonString(body, "decoder")) |raw_dec| {
        const dec = try jsonUnescape(allocator, raw_dec);
        defer allocator.free(dec);
        if (std.mem.eql(u8, dec, "diffusion")) {
            vae_choice.diffusion = engine.ensureDiffusionDecoder(io) catch
                return sendError(conn, 400, "'decoder':\"diffusion\" requires vae_diffusion_decoder.safetensors — the 8-bit LTX-2.5 pack ships it, the 4-bit pack does not");
            log.info("[video] decoder: diffusion (DiffVAE)\n", .{});
        } else if (!std.mem.eql(u8, dec, "conv") and dec.len > 0) {
            return sendError(conn, 400, "'decoder' must be \"conv\" or \"diffusion\"");
        }
    }

    var sctx = videoStreamCtx(conn, allocator, body, want_stream);
    const prog: ?ltx.Progress = sctx.progress();
    if (want_stream) try conn.writeAll(sse.headers);

    var frames = switch (pipeline) {
        .one_stage => blk: {
            // Run the schedule the loaded variant was trained for; the request
            // never forces a swap here (dev-only bundles keep working).
            const distilled = engine.transformer_variant == .distilled;
            break :blk ltx.generateVideoFrames(io, allocator, engine.ltx_cfg, &engine.transformer, &engine.connector, vae_choice, enc_ptr, cond_img, last_img, engine.gemma_dir, pos_ids, neg_ids, LTX_PAD_ID, num_frames, height, width, frame_rate, steps, distilled, seed, guiders.vp, guiders.ap, prog, engine.s);
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
            break :blk ltx.generateVideoFramesTwoStage(io, allocator, engine.ltx_cfg, &engine.transformer, &engine.connector, vae_choice, &engine.vae_encoder.?, cond_img_half, cond_img, last_img_half, last_img, audio_cond, engine.gemma_dir, pos_ids, neg_ids, LTX_PAD_ID, num_frames, height, width, frame_rate, opts, seed, guiders.vp, guiders.ap, prog, engine.s);
        },
    } catch |err| {
        if (err == error.Cancelled) {
            // Client hung up mid-generation (progress write failed) — the
            // denoise loop aborted; nothing to write, the socket is dead.
            log.info("[video] generation cancelled — client disconnected\n", .{});
            return;
        }
        if (err == error.KeyframeCanvasTooShort) {
            if (want_stream) {
                conn.writeAll("data: {\"type\":\"error\",\"message\":\"keyframes need at least 9 frames\"}\n\n") catch {};
                return;
            }
            return sendError(conn, 400, "keyframes need at least 9 frames (one latent frame cannot hold an anchor and still generate)");
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

/// Shape → raw mesh → paint (texture) stage → textured GLB. The paint engine
/// loads lazily per request and frees before returning.
fn paintedGlb(allocator: std.mem.Allocator, engine: *MeshEngine, rgba: []const u8, w: u32, h: u32, shape_opts: hy3d.MeshOpts, body: []const u8, prog: ?sse.Progress) ![]u8 {
    const paint_dir = engine.paint_dir orelse return error.PaintUnavailable; // guarded at parse
    var mesh = try engine.engine.generateMeshRaw(allocator, rgba, w, h, shape_opts, prog);
    defer mesh.deinit(allocator);

    const io = std.Io.Threaded.global_single_threaded.io();
    const paint = try hy3d_paint.PaintEngine.load(io, allocator, paint_dir);
    defer paint.deinit();

    var popts = hy3d_paint.PaintOpts{ .seed = shape_opts.seed };
    if (extractJsonInt(body, "texture_steps")) |ts| {
        if (ts >= 1 and ts <= 100) popts.steps = @intCast(ts);
    }
    return paint.paintMeshToGlb(allocator, &mesh, rgba, w, h, popts, prog);
}

/// POST /v1/3d/generations — base64 GLB (or SSE progress + complete).
/// The engine takes straight-alpha RGBA8 (its preprocess recenters the subject
/// via the alpha bbox and composites on white — so an app-side cutout with real
/// alpha conditions best, and an opaque photo still works as a fallback).
pub fn handleMesh(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, engine: *MeshEngine) !void {
    const raw_img = extractJsonString(body, "image") orelse return sendError(conn, 400, "missing 'image' (base64 PNG/JPEG of the subject)");
    const b64 = try jsonUnescape(allocator, raw_img); // handles \/ from Swift JSONSerialization
    defer allocator.free(b64);
    if (b64.len == 0) return sendError(conn, 400, "empty 'image'");
    const img_bytes = base64DecodeAlloc(allocator, b64) catch
        return sendError(conn, 400, "invalid base64 in 'image'");
    defer allocator.free(img_bytes);
    const img = decodeImageRgba(allocator, img_bytes) orelse
        return sendError(conn, 400, "could not decode 'image' (PNG/JPEG supported)");
    defer allocator.free(img.pix);

    const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse 30);
    const res: u32 = @intCast(extractJsonInt(body, "octree_resolution") orelse 256);
    if (res < 64 or res > 512) return sendError(conn, 400, "'octree_resolution' must be in [64,512]");
    const seed: u64 = extractJsonInt(body, "seed") orelse 42;
    var guidance: f32 = 5.0;
    if (extractJsonFloat(body, "guidance_scale")) |g| {
        if (!(g >= 0.0 and g <= 20.0)) return sendError(conn, 400, "'guidance_scale' must be in [0,20]");
        guidance = @floatCast(g);
    }

    // P2 texture stage (opt-in): requires the converted paint weights. The
    // 400 here is explicit — never a silent untextured downgrade (the a2vid
    // precedent: the user asked for THIS output).
    const want_texture = sse.bodyWantsTrue(body, "texture");
    if (want_texture and engine.paint_dir == null)
        return sendError(conn, 400, "texture requested but the paint weights are not installed (run tests/convert_hunyuan3d_paint_weights.py, or set HY3D_PAINT_DIR)");

    const want_stream = sse.bodyWantsTrue(body, "stream");
    log.info("[mesh] generating steps={d} res={d} guidance={d:.1} seed={d} texture={} stream={} from {d}x{d} image\n", .{ steps, res, guidance, seed, want_texture, want_stream, img.w, img.h });
    var sctx = sse.StreamCtx{ .conn = conn, .stream = want_stream };
    const prog: ?sse.Progress = sctx.progress();
    if (want_stream) try conn.writeAll(sse.headers);

    const opts = hy3d.MeshOpts{ .steps = steps, .guidance = guidance, .seed = seed, .octree_resolution = res };
    const glb_bytes = blk: {
        if (!want_texture) {
            break :blk engine.engine.generateGlb(allocator, img.pix, img.w, img.h, opts, prog);
        }
        // Texture path: shape → raw mesh → paint stage (loaded per request,
        // freed after — the paint UNet+DINO+VAE ride ~4.6 GB beside the
        // 3.5 GB shape engine only for the duration of this request).
        break :blk paintedGlb(allocator, engine, img.pix, img.w, img.h, opts, body, prog);
    } catch |err| {
        if (err == error.Cancelled) {
            // Client hung up mid-generation (progress write failed) — nothing
            // to write, the socket is dead.
            log.info("[mesh] generation cancelled — client disconnected\n", .{});
            return;
        }
        log.err("[mesh] generation failed: {}\n", .{err});
        if (want_stream) {
            sse.sendError(conn, "generation failed");
            return;
        }
        return sendError(conn, 500, "generation failed");
    };
    defer allocator.free(glb_bytes);
    log.info("[mesh] -> {d} GLB bytes\n", .{glb_bytes.len});

    const b64_len = std.base64.standard.Encoder.calcSize(glb_bytes.len);
    const ob64 = try allocator.alloc(u8, b64_len);
    defer allocator.free(ob64);
    _ = std.base64.standard.Encoder.encode(ob64, glb_bytes);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, if (want_stream) "data: {\"type\":\"complete\",\"format\":\"glb\",\"data\":\"" else "{\"created\":0,\"format\":\"glb\",\"data\":\"");
    try out.appendSlice(allocator, ob64);
    try out.appendSlice(allocator, if (want_stream) "\"}\n\n" else "\"}");
    if (want_stream) {
        try conn.writeAll(out.items);
        return;
    }
    return sendBytesJson(conn, allocator, out.items);
}

/// Decode a PNG/JPEG image (raw file bytes) → owned straight-alpha RGBA8
/// pixels + dims (stb forces 4 channels; 3-channel sources get opaque alpha).
fn decodeImageRgba(allocator: std.mem.Allocator, encoded: []const u8) ?struct { pix: []u8, w: u32, h: u32 } {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const src_ptr = stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &ch, 4) orelse return null;
    defer stb.stbi_image_free(src_ptr);
    if (w <= 0 or h <= 0) return null;
    const n: usize = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 4;
    const out = allocator.alloc(u8, n) catch return null;
    @memcpy(out, src_ptr[0..n]);
    return .{ .pix = out, .w = @intCast(w), .h = @intCast(h) };
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
    if (model_dir.len == 0 or model_dir[0] != '/') return 0; // openDirAbsolute UB class
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    return sumSafetensorsIn(io, dir);
}

fn sumSafetensorsIn(io: std.Io, dir: std.Io.Dir) u64 {
    // Symlinked weights count (statFile follows) — an HF hub-cache snapshot
    // is ALL symlinks into ../../blobs; skipping them billed a pack at 0.
    var total: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if ((entry.kind == .file or entry.kind == .sym_link) and std.mem.endsWith(u8, entry.name, ".safetensors")) {
            const st = dir.statFile(io, entry.name, .{}) catch continue;
            if (st.kind != .file) continue;
            total += @intCast(st.size);
        } else if (entry.kind == .directory) {
            var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
            defer sub.close(io);
            var sit = sub.iterate();
            while (sit.next(io) catch null) |se| {
                if (se.kind != .file and se.kind != .sym_link) continue;
                if (!std.mem.endsWith(u8, se.name, ".safetensors")) continue;
                const st = sub.statFile(io, se.name, .{}) catch continue;
                if (st.kind != .file) continue;
                total += @intCast(st.size);
            }
        }
    }
    return total;
}

/// Peak resident bytes for a backend whose parts do NOT all coexist.
/// `resident` is what the engine holds for its whole lifetime; `stages` are
/// DISJOINT — each is loaded, used and released before the next runs, so only
/// the biggest is ever on top of `resident`. A stage that GENERATES carries its
/// own transients inside its number, because they are not uniform across stages
/// (a text-encoder pass over a few hundred rows allocates nothing like a
/// 124-frame denoise).
///
/// Sum-of-directory is what a backend gets when it declares no plan, and it is
/// the RIGHT answer for the resident-engine backends (krea, mage_flow,
/// hunyuan3d, acestep, tts all hold text-encoder + DiT + VAE on one struct for
/// the engine's lifetime). It goes wrong in BOTH directions the moment a
/// backend loads something it later frees, or reads weights from outside its
/// own directory — see `ltxPeakBytes` for one backend doing each.
pub fn stagedPeakBytes(resident: u64, stages: []const u64) u64 {
    var biggest: u64 = 0;
    for (stages) |st| biggest = @max(biggest, st);
    return resident + biggest;
}

/// LTX's plan. Its engine is RESIDENT (transformer + connector + VAEs + audio
/// stay on `LtxVideoEngine` for its lifetime), with two corrections the
/// directory sum cannot make:
///
///   `spare_transformer` — packs ship `transformer-dev` AND
///   `transformer-distilled` (~10.5 GiB each) and `ensureTransformer` frees one
///   BEFORE loading the other, precisely so they never coexist. The sum bills a
///   phantom.
///
///   `text_encoder` — the Gemma encoder is a SEPARATE shared repo, loaded by
///   `ltx_video.gemmaCapture` per generation and freed when it returns. Being
///   outside the model dir, the sum bills 7.5 GiB at zero, and it is resident
///   on top of the whole engine while it runs.
///
/// No activation term: nothing has been measured for this backend, and
/// inventing one would newly refuse loads that work today.
pub fn ltxPeakBytes(dir_sum: u64, spare_transformer: u64, text_encoder: u64) u64 {
    return stagedPeakBytes(dir_sum -| spare_transformer, &.{text_encoder});
}

/// Percent of `transformer.safetensors` still resident once `precomputeAdaln`
/// has tabled and FREED the 13B modulation weights (~39% of the DiT's
/// parameters, so the share barely moves with quant width). Measured 0.615 on
/// the 8-bit pack (32.83 → 20.19 GiB) and 0.623 on the 4-bit (17.41 → 10.84);
/// billed at 0.65 so a pack whose AdaLN share is smaller than ours still fails
/// safe.
pub const H3_DIT_RESIDENT_PCT: u64 = 65;

/// Transients the two GENERATING stages carry on top of their weights: the
/// packed [text|cond|audio|video] sequence's activations while sampling, and
/// the VAE decode's frame buffers. Measured 4.0-5.0 GiB at 768x448 / 124f
/// (process peak minus self-reported DiT residency, both packs); billed at 6.
/// It scales with pixels x frames, which a per-MODEL load gate cannot see —
/// bounding a specific request is not something this estimator can do, and
/// the old formula's incidental margin was the same order.
///
/// The TEXT-ENCODER stage gets none of it: that is one forward over a few
/// hundred prompt rows, so a shared "+ activations" on the max of all three
/// stages bills the biggest stage for transients it never allocates — which
/// is what refused the 8-bit pack on every Mac under ~96 GB.
pub const H3_ACTIVATION_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// MiniMax Music 3's non-weight working set at the request caps: batch-2 KV
/// cache for 36 layers at 9000 frames + 5000 prompt tokens (~4.1 GB), the
/// bf16 frame-hidden buffer (~0.6 GB), and DiT/vocoder window transients.
pub const MUSIC3_GEN_BUFFER_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// The DiT term of the H3 bill. `precompute` mirrors
/// MINIMAX_H3_ADALN_PRECOMPUTE: with it off the modulation weights are never
/// freed and the whole file stays resident, so the shed size would under-bill
/// by ~12 GiB into an uncatchable Metal OOM.
pub fn h3DitResidentBytes(dit_file: u64, precompute: bool) u64 {
    if (!precompute) return dit_file;
    return dit_file * H3_DIT_RESIDENT_PCT / 100;
}

/// MiniMax-H3's staged residency plan, as a bill. `minimax_h3.generate` runs
/// three DISJOINT stages: the text encoder is loaded, run and FREED before the
/// DiT loads (`Model.load` is scoped), and the DiT is released before the VAEs
/// load — so the peak is the BIGGEST stage, never a sum. The two VAEs are one
/// stage: the video decoder is still resident when the audio one loads.
/// `dit_resident` is post-AdaLN-precompute (`h3DitResidentBytes`), which the
/// file size overstates by ~39%.
pub fn h3PeakBytes(te: u64, dit_resident: u64, video_vae: u64, audio_vae: u64) u64 {
    const vaes = video_vae + audio_vae;
    const generating = @max(dit_resident, vaes);
    if (te == 0 and generating == 0) return 0; // unknown dir → never block
    return stagedPeakBytes(0, &.{ te, generating + H3_ACTIVATION_BYTES });
}

/// Per-backend generation-peak estimate for the media load preflight. A
/// backend with a STAGED residency plan declares it here; every other type
/// keeps the sum-of-safetensors default — over-billing fails safe (a refused
/// load names its numbers), under-billing kills the process mid-request.
pub fn estimatePeakResidentBytesIn(io: std.Io, dir: std.Io.Dir, model_type: []const u8) u64 {
    const sz = struct {
        fn f(io_: std.Io, d: std.Io.Dir, name: []const u8) u64 {
            const st = d.statFile(io_, name, .{}) catch return 0;
            return @intCast(st.size);
        }
    }.f;
    if (std.mem.eql(u8, model_type, "minimax_h3")) {
        // The Turbo LoRA (when the pack ships one) is resident ALONGSIDE the
        // DiT and precompute does not free it, so it rides the DiT term at
        // full size — billed whenever present, since the gate estimate is
        // per-model, not per-request.
        const dit = h3DitResidentBytes(
            sz(io, dir, "transformer.safetensors"),
            minimax_h3.adalnPrecomputeOn(),
        ) + sz(io, dir, "turbo_lora.safetensors");
        return h3PeakBytes(
            sz(io, dir, "text_encoder.safetensors"),
            dit,
            sz(io, dir, "video_vae.safetensors"),
            sz(io, dir, "audio_vae.safetensors"),
        );
    }
    if (std.mem.eql(u8, model_type, "minimax_music3")) {
        // The whole engine is resident for its lifetime (no staging), so the
        // sum is the right weight bill — plus the AR stage's working set the
        // directory cannot see: the batch-2 KV cache (~4.1 GB at the 9000-frame
        // + 5000-token caps), the frame-hidden buffer (~0.6 GB bf16), and the
        // DiT/vocoder window transients.
        const sum = sumSafetensorsIn(io, dir);
        if (sum == 0) return 0; // unknown dir -> never block
        return sum + MUSIC3_GEN_BUFFER_BYTES;
    }
    if (std.mem.eql(u8, model_type, "AudioVideo")) {
        // Both variants ship; only one is ever loaded. Subtract the smaller so
        // an asymmetric future pack still bills its larger one.
        const spare = @min(
            sz(io, dir, "transformer-dev.safetensors"),
            sz(io, dir, "transformer-distilled.safetensors"),
        );
        return ltxPeakBytes(sumSafetensorsIn(io, dir), spare, 0);
    }
    return sumSafetensorsIn(io, dir);
}

/// Sum of the `.safetensors` under an absolute path, or 0 if it is not
/// readable — 0 means "unknown", which every caller treats as "do not block".
fn sumSafetensorsAt(io: std.Io, path: []const u8) u64 {
    if (path.len == 0 or path[0] != '/') return 0; // openDirAbsolute UB class
    var d = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    return sumSafetensorsIn(io, d);
}

/// LTX's text encoder, resolved the way `resolveGemmaDir` does but without an
/// allocator (this runs inside the load gate). Absent → 0.
fn ltxTextEncoderBytes(io: std.Io) u64 {
    var buf: [1024]u8 = undefined;
    if (std.c.getenv("LTX_GEMMA_DIR")) |env| {
        const e = std.mem.span(env);
        if (std.fs.path.isAbsolute(e)) return sumSafetensorsAt(io, e);
    }
    const home = std.mem.span(std.c.getenv("HOME") orelse return 0);
    for ([_][]const u8{ LTX_GEMMA_REPO_DIR, "gemma-3-12b-it-4bit" }) |rel| {
        const p = std.fmt.bufPrint(&buf, "{s}/.mlx-serve/models/{s}", .{ home, rel }) catch continue;
        const n = sumSafetensorsAt(io, p);
        if (n > 0) return n;
    }
    return 0;
}

pub fn estimatePeakResidentBytes(io: std.Io, model_dir: []const u8, model_type: []const u8) u64 {
    if (model_dir.len == 0 or model_dir[0] != '/') return 0; // openDirAbsolute UB class
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    const in_dir = estimatePeakResidentBytesIn(io, dir, model_type);
    // LTX reads its text encoder from a DIFFERENT repo, so it is a stage the
    // model dir cannot see. Every other backend's weights are all in its own
    // directory; if that stops being true, it belongs here beside this one.
    if (in_dir > 0 and std.mem.eql(u8, model_type, "AudioVideo"))
        return stagedPeakBytes(in_dir, &.{ltxTextEncoderBytes(io)});
    return in_dir;
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
    // A refusal that logs nothing is invisible the moment a client drops the
    // body (live 2026-08-06: the app's stream path showed a bare "HTTP 400"
    // while the log showed a clean load→unload and nothing else).
    log.warn("[gen] {d}: {s}\n", .{ code, msg });
    // Escape at the SINK: several of these messages quote a field value
    // (`mode:"edit"`), which went onto the wire as raw quotes inside a JSON
    // string — an unparseable body. See `sse.jsonEscapeMessage`.
    var esc_buf: [512]u8 = undefined;
    const esc = sse.jsonEscapeMessage(&esc_buf, msg);
    var body_buf: [640]u8 = undefined;
    const body = std.fmt.bufPrint(&body_buf, "{{\"error\":{{\"message\":\"{s}\"}}}}", .{esc}) catch return;
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
/// Parse a JSON key's value as either a bracketed number array
/// (`"key": [0.8, 1.0]`) or a quoted comma/space-separated string of numbers
/// (`"key": "0.8 1.0"`). Shared by `cond_weights` and `lora_scales`.
fn extractFloatArrayField(body: []const u8, key: []const u8, buf: []f32) ?[]f32 {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
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

fn extractCondWeights(body: []const u8, buf: []f32) ?[]f32 {
    return extractFloatArrayField(body, "cond_weights", buf);
}

/// Per-adapter scales for `lora_scales` (multi-LoRA counterpart of the
/// single `lora_scale` float field).
fn extractLoraScales(body: []const u8, buf: []f32) ?[]f32 {
    return extractFloatArrayField(body, "lora_scales", buf);
}

const LoraFieldsError = error{ TooManyLoraPaths, BadLoraPathsJson, BadLoraScalesJson, OutOfMemory };

/// Parse the LoRA fields common to image and video requests: the array form
/// (`lora_paths` + optional `lora_scales`) or the original single-adapter
/// form (`lora_path` + optional `lora_scale`), which is kept exactly
/// backward-compatible. Writes up to `lora_mod.MAX_LORAS` unescaped,
/// allocator-owned path strings into `path_bufs` (caller frees them) and
/// their resolved scales into `scale_buf`. Returns 0 with both buffers
/// untouched when neither field is present — the "detach whatever was
/// attached" case. Missing `lora_scales` entries default to 1.0, matching
/// mflux's `resolve_scales`.
fn parseLoraFields(
    allocator: std.mem.Allocator,
    body: []const u8,
    path_bufs: *[lora_mod.MAX_LORAS][]u8,
    scale_buf: *[lora_mod.MAX_LORAS]f32,
) LoraFieldsError!usize {
    var n: usize = 0;
    errdefer for (path_bufs[0..n]) |p| allocator.free(p);

    if (std.mem.indexOf(u8, body, "\"lora_paths\"") != null) {
        var it = iterJsonStringArray(body, "lora_paths") orelse return error.BadLoraPathsJson;
        while (it.next()) |raw| {
            if (n >= lora_mod.MAX_LORAS) return error.TooManyLoraPaths;
            path_bufs[n] = try jsonUnescape(allocator, raw);
            n += 1;
        }
        if (it.bad) return error.BadLoraPathsJson;
    } else if (extractJsonString(body, "lora_path")) |lp_raw| {
        path_bufs[0] = try jsonUnescape(allocator, lp_raw);
        n = 1;
    }
    if (n == 0) return 0;

    if (std.mem.indexOf(u8, body, "\"lora_scales\"") != null) {
        var sbuf: [lora_mod.MAX_LORAS]f32 = undefined;
        const sl = extractLoraScales(body, &sbuf) orelse return error.BadLoraScalesJson;
        const m = @min(sl.len, n);
        @memcpy(scale_buf[0..m], sl[0..m]);
        for (scale_buf[m..n]) |*sc| sc.* = 1.0;
    } else if (n == 1) {
        // Legacy single-adapter form: honor 'lora_scale' exactly as before.
        scale_buf[0] = @floatCast(extractJsonFloat(body, "lora_scale") orelse 1.0);
    } else {
        for (scale_buf[0..n]) |*sc| sc.* = 1.0;
    }
    return n;
}

/// Iterate the string elements of a JSON array field (`"key": ["a", "b"]`).
/// Scanner-grade like the other extract helpers — values must not contain
/// escaped quotes, which holds for base64 payloads. A non-string element sets
/// `bad` so the caller can 400 instead of silently ignoring it.
const JsonStringArrayIter = struct {
    rest: []const u8,
    bad: bool = false,

    fn next(self: *JsonStringArrayIter) ?[]const u8 {
        var i: usize = 0;
        while (i < self.rest.len) : (i += 1) {
            switch (self.rest[i]) {
                '"' => break,
                ']' => return null,
                ',', ' ', '\t', '\n', '\r' => continue,
                else => {
                    self.bad = true;
                    return null;
                },
            }
        }
        if (i >= self.rest.len) {
            self.bad = true; // ran out before the closing ']'
            return null;
        }
        i += 1;
        const start = i;
        while (i < self.rest.len) : (i += 1) {
            if (self.rest[i] == '\\') {
                i += 1;
                continue;
            }
            if (self.rest[i] == '"') {
                const v = self.rest[start..i];
                self.rest = self.rest[i + 1 ..];
                return v;
            }
        }
        self.bad = true; // unterminated string
        return null;
    }
};

/// Walks an array of JSON OBJECTS, handing back each element's raw `{…}` slice
/// so the caller can read its fields with the same scanners it uses on a whole
/// body. Brace-balanced and string-aware, which is what keeps one element's
/// fields from leaking into the next one's.
const JsonObjectArrayIter = struct {
    rest: []const u8,
    bad: bool = false,

    fn next(self: *JsonObjectArrayIter) ?[]const u8 {
        var i: usize = 0;
        while (i < self.rest.len) : (i += 1) {
            switch (self.rest[i]) {
                '{' => break,
                ']' => return null,
                ',', ' ', '\t', '\n', '\r' => continue,
                else => {
                    self.bad = true;
                    return null;
                },
            }
        }
        if (i >= self.rest.len) {
            self.bad = true; // ran out before the closing ']'
            return null;
        }
        const start = i;
        var depth: usize = 0;
        var in_str = false;
        while (i < self.rest.len) : (i += 1) {
            const c = self.rest[i];
            if (in_str) {
                if (c == '\\') {
                    i += 1;
                } else if (c == '"') {
                    in_str = false;
                }
                continue;
            }
            switch (c) {
                '"' => in_str = true,
                '{', '[' => depth += 1,
                '}', ']' => {
                    depth -= 1;
                    if (depth == 0) {
                        const v = self.rest[start .. i + 1];
                        self.rest = self.rest[i + 1 ..];
                        return v;
                    }
                },
                else => {},
            }
        }
        self.bad = true; // unterminated object
        return null;
    }
};

/// Position an iterator at the first element of the `key` JSON object array.
/// Null when the key is absent or its value is not an array.
fn iterJsonObjectArray(body: []const u8, key: []const u8) ?JsonObjectArrayIter {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    if (i >= body.len or body[i] != '[') return null;
    return .{ .rest = body[i + 1 ..] };
}

/// Position an iterator at the first element of the `key` JSON string array.
/// Null when the key is absent or its value is not an array.
fn iterJsonStringArray(body: []const u8, key: []const u8) ?JsonStringArrayIter {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    if (i >= body.len or body[i] != '[') return null;
    return .{ .rest = body[i + 1 ..] };
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

test "modalityFromType classifies the media archs + markers (incl. krea + hunyuan3d)" {
    try testing.expectEqual(Modality.image, modalityFromType("flux2-klein-4b").?);
    try testing.expectEqual(Modality.image, modalityFromType("flux2").?);
    try testing.expectEqual(Modality.image, modalityFromType("krea2_turbo").?);
    try testing.expectEqual(Modality.image, modalityFromType("krea").?);
    try testing.expectEqual(Modality.audio, modalityFromType("qwen3_tts").?);
    try testing.expectEqual(Modality.audio, modalityFromType("acestep").?);
    try testing.expectEqual(Modality.audio, modalityFromType("minimax_music3").?);
    try testing.expectEqual(Modality.video, modalityFromType("AudioVideo").?);
    try testing.expectEqual(Modality.mesh, modalityFromType("hunyuan3d_2_1").?);
    try testing.expectEqual(Modality.mesh, modalityFromType("hunyuan3d").?);
    try testing.expectEqual(Modality.image, modalityFromType("mage_flow").?);
    try testing.expectEqual(Modality.image, modalityFromType("mageflow").?);
    try testing.expectEqual(@as(?Modality, null), modalityFromType("gemma4"));
    try testing.expectEqual(@as(?Modality, null), modalityFromType("qwen3_5_moe"));
}

test "mage_flow detects from the official diffusers layout (model_index.json, no root config.json)" {
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const root = root_buf[0..root_len];

    // Official Mage-Flow repos have no root config.json/model_type — the
    // pipeline identity lives in model_index.json (_class_name).
    try tmp.dir.createDirPath(io, "mf");
    try tmp.dir.writeFile(io, .{
        .sub_path = "mf/model_index.json",
        .data = "{\"_class_name\":\"MageFlowPipeline\",\"_mage_flow_version\":\"0.1.0\"}",
    });
    const model_dir = try std.fs.path.join(allocator, &.{ root, "mf" });
    defer allocator.free(model_dir);

    const mt = peekModelType(io, allocator, model_dir) orelse return error.TestExpectedResult;
    defer allocator.free(mt);
    try testing.expectEqualStrings("mage_flow", mt);
    try testing.expectEqual(Modality.image, detectModality(io, allocator, model_dir).?);
}

test "flux2 detects from an mflux conversion with no root config.json" {
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const root = root_buf[0..root_len];

    // `mlx-community/flux2-klein-9b-4bit` — the only MLX build of klein 9B —
    // ships no config.json, so routing has to recognize the DiT itself or the
    // dir is unloadable. Delegates to the ONE predicate discovery uses, so the
    // picker and the loader can never disagree about what this dir is.
    try tmp.dir.createDirPath(io, "k9/transformer");
    try tmp.dir.writeFile(io, .{
        .sub_path = "k9/transformer/model.safetensors.index.json",
        .data = "{\"weight_map\":{\"double_stream_modulation_img.linear.weight\":\"0.safetensors\"}}",
    });
    const model_dir = try std.fs.path.join(allocator, &.{ root, "k9" });
    defer allocator.free(model_dir);

    const mt = peekModelType(io, allocator, model_dir) orelse return error.TestExpectedResult;
    defer allocator.free(mt);
    try testing.expectEqualStrings("flux2-klein", mt);
    try testing.expectEqual(Modality.image, detectModality(io, allocator, model_dir).?);
}

test "openaiEditFormToJson: OpenAI multipart becomes our edit request" {
    const a = testing.allocator;
    const CT = "multipart/form-data; boundary=X";
    const form =
        "--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nmake it \"winter\"\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\nAAA\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nmage-flow-edit\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"n\"\r\n\r\n1\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n1024x768\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\nhigh\r\n" ++
        "--X--\r\n";
    const json = try openaiEditFormToJson(a, form, CT);
    defer a.free(json);
    // Must be real JSON (the prompt's quotes are the trap) and carry the fields
    // handleImage reads.
    var parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("edit", o.get("mode").?.string);
    try testing.expectEqualStrings("make it \"winter\"", o.get("prompt").?.string);
    try testing.expectEqualStrings("mage-flow-edit", o.get("model").?.string);
    try testing.expectEqualStrings("1024x768", o.get("size").?.string);
    try testing.expectEqualStrings("QUFB", o.get("image").?.string); // base64("AAA")
    try testing.expect(o.get("ref_images") == null);

    // Extra images become in-context references, in order.
    const multi =
        "--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\ncompose\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\n\r\nAAA\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"b.png\"\r\n\r\nBBB\r\n" ++
        "--X--\r\n";
    const j2 = try openaiEditFormToJson(a, multi, CT);
    defer a.free(j2);
    var p2 = try std.json.parseFromSlice(std.json.Value, a, j2, .{});
    defer p2.deinit();
    try testing.expectEqualStrings("QUFB", p2.value.object.get("image").?.string);
    const refs = p2.value.object.get("ref_images").?.array;
    try testing.expectEqual(@as(usize, 1), refs.items.len);
    try testing.expectEqualStrings("QkJC", refs.items[0].string); // base64("BBB")

    // `size=auto` is "you decide" — omitted so the edit path resolves from the
    // reference instead of being pinned to a default square.
    const auto = "--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\nauto\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nAAA\r\n--X--\r\n";
    const j3 = try openaiEditFormToJson(a, auto, CT);
    defer a.free(j3);
    try testing.expect(std.mem.indexOf(u8, j3, "\"size\"") == null);

    // LoRA fields reach the edit body (issue #268: they were rebuilt away).
    const lora = "--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"lora_paths\"\r\n\r\n[\"/l/a.safetensors\"]\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"lora_scales\"\r\n\r\n[0.8]\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nAAA\r\n--X--\r\n";
    const j4 = try openaiEditFormToJson(a, lora, CT);
    defer a.free(j4);
    var p4 = try std.json.parseFromSlice(std.json.Value, a, j4, .{});
    defer p4.deinit();
    try testing.expectEqualStrings("/l/a.safetensors", p4.value.object.get("lora_paths").?.array.items[0].string);
    try testing.expectEqual(@as(f64, 0.8), p4.value.object.get("lora_scales").?.array.items[0].float);
}

test "openaiEditFormToJson: everything we can't honor is an explicit error" {
    const a = testing.allocator;
    const CT = "multipart/form-data; boundary=X";
    const base = "--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nAAA\r\n";
    const cases = .{
        .{ "Content-Disposition: form-data; name=\"mask\"; filename=\"m.png\"\r\n\r\nMMM", EditFormError.MaskUnsupported },
        .{ "Content-Disposition: form-data; name=\"n\"\r\n\r\n4", EditFormError.MultipleChoicesUnsupported },
        .{ "Content-Disposition: form-data; name=\"response_format\"\r\n\r\nurl", EditFormError.UrlResponseUnsupported },
        .{ "Content-Disposition: form-data; name=\"output_format\"\r\n\r\njpeg", EditFormError.OutputFormatUnsupported },
        .{ "Content-Disposition: form-data; name=\"stream\"\r\n\r\ntrue", EditFormError.StreamUnsupported },
    };
    inline for (cases) |c| {
        const form = base ++ "--X\r\n" ++ c[0] ++ "\r\n--X--\r\n";
        try testing.expectError(c[1], openaiEditFormToJson(a, form, CT));
    }
    // Missing required fields, and a body that isn't multipart at all.
    try testing.expectError(EditFormError.MissingImage, openaiEditFormToJson(a, "--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n--X--\r\n", CT));
    try testing.expectError(EditFormError.MissingPrompt, openaiEditFormToJson(a, "--X\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nAAA\r\n--X--\r\n", CT));
    try testing.expectError(EditFormError.NotMultipart, openaiEditFormToJson(a, "{}", "application/json"));
    // Over the reference cap → a named 400, not a truncated silent edit.
    var many: std.ArrayList(u8) = .empty;
    defer many.deinit(a);
    try many.appendSlice(a, base);
    for (0..MAX_EDIT_IMAGES) |_| try many.appendSlice(a, "--X\r\nContent-Disposition: form-data; name=\"image[]\"\r\n\r\nZZZ\r\n");
    try many.appendSlice(a, "--X--\r\n");
    try testing.expectError(EditFormError.TooManyImages, openaiEditFormToJson(a, many.items, CT));
    // Every error has human text (no bare error names reaching a client).
    inline for (@typeInfo(EditFormError).error_set.error_names.?) |name| {
        try testing.expect(editFormErrorMessage(@field(EditFormError, name)).len > 10);
    }
}

test "fitAspect keeps the reference's aspect at the requested pixel budget" {
    // Square budget, 3:2 landscape reference → a 3:2 output of the same area,
    // /16. Without this the edit backend resizes the reference INTO 1024² and
    // hands back a squashed picture.
    const l = fitAspect(3000, 2000, 1024, 1024);
    try testing.expectEqual(@as(u32, 1248), l.w);
    try testing.expectEqual(@as(u32, 832), l.h);
    try testing.expect(@abs(@as(i64, l.w) * @as(i64, l.h) - 1024 * 1024) < 90_000); // ~same budget
    // Portrait mirrors it.
    const p = fitAspect(2000, 3000, 1024, 1024);
    try testing.expectEqual(l.h, p.w);
    try testing.expectEqual(l.w, p.h);
    // A square reference is already the budget — untouched.
    const s = fitAspect(512, 512, 1024, 1024);
    try testing.expectEqual(@as(u32, 1024), s.w);
    try testing.expectEqual(@as(u32, 1024), s.h);
    // A non-square budget is an AREA, not a shape: a square ref squares it.
    const nb = fitAspect(600, 600, 1536, 1024);
    try testing.expectEqual(nb.w, nb.h);
    // Degenerate dims fall back to the budget rather than dividing by zero.
    const z = fitAspect(0, 0, 768, 512);
    try testing.expectEqual(@as(u32, 768), z.w);
    try testing.expectEqual(@as(u32, 512), z.h);
    // "Match source" = the reference IS its own budget, so it round-trips to
    // its own dimensions (this is how an edit with no `size` keeps the source's
    // resolution — the reference pipeline's `max_size = source size` default).
    for ([_][2]u32{ .{ 1152, 768 }, .{ 2048, 512 }, .{ 640, 1600 }, .{ 512, 512 } }) |wh| {
        const same = fitAspect(wh[0], wh[1], wh[0], wh[1]);
        try testing.expectEqual(wh[0], same.w);
        try testing.expectEqual(wh[1], same.h);
    }
    // An off-grid source floors to /16 rather than erroring.
    const odd = fitAspect(1153, 769, 1153, 769);
    try testing.expectEqual(@as(u32, 0), odd.w % 16);
    try testing.expectEqual(@as(u32, 0), odd.h % 16);
}

test "resolveEditTargetSize keeps the reference's aspect ABOVE the backend cap" {
    // The bug this pins: `fitAspect` preserved the aspect and then the
    // per-dimension clamp in `normalizeSize` threw it away again. A 12 MP 4:3
    // phone photo on a SIZELESS edit (the plain
    // `client.images.edit(image=…, prompt=…)` call) came back 2048x2048 —
    // squared off, which is the one thing an editor must never do to its input.
    const cap: u32 = 2048;
    const cases = [_][2]u32{
        .{ 4032, 3024 }, // 12 MP 4:3 landscape
        .{ 3024, 4032 }, // and portrait
        .{ 6000, 4000 }, // 24 MP 3:2 DSLR
        .{ 8192, 8192 }, // huge square: scales, stays square
    };
    for (cases) |wh| {
        const got = resolveEditTargetSize(wh[0], wh[1], wh[0], wh[1], cap);
        try testing.expect(got.w <= cap and got.h <= cap);
        try testing.expect(got.w % 16 == 0 and got.h % 16 == 0);
        // And it must survive `normalizeSize` — that clamp is the step that
        // actually reintroduced the squash.
        const nz_w = clampKreaDim(got.w);
        const nz_h = clampKreaDim(got.h);
        const src_aspect = @as(f64, @floatFromInt(wh[0])) / @as(f64, @floatFromInt(wh[1]));
        const out_aspect = @as(f64, @floatFromInt(nz_w)) / @as(f64, @floatFromInt(nz_h));
        try testing.expect(@abs(src_aspect - out_aspect) / src_aspect < 0.02);
    }
    // Below the cap nothing moves: the existing sizeless round-trip is intact.
    for ([_][2]u32{ .{ 1152, 768 }, .{ 2048, 512 }, .{ 512, 512 } }) |wh| {
        const same = resolveEditTargetSize(wh[0], wh[1], wh[0], wh[1], cap);
        try testing.expectEqual(wh[0], same.w);
        try testing.expectEqual(wh[1], same.h);
    }
}

test "maxDim matches what normalizeSize actually clamps to (drift guard)" {
    // Two places encode the same ceiling. If someone widens a clamp and forgets
    // maxDim, the edit path silently starts squashing again.
    try testing.expectEqual(clampFluxDim(99999), ImageEngine.maxDimFor(.flux));
    try testing.expectEqual(clampKreaDim(99999), ImageEngine.maxDimFor(.krea));
    try testing.expectEqual(clampKreaDim(99999), ImageEngine.maxDimFor(.mage_flow));
}

test "Modality.mesh advertises the 3d capability" {
    try testing.expectEqualStrings("3d", Modality.mesh.capability());
}

test "GenRoute: speech + music share the audio modality slot" {
    try testing.expectEqual(Modality.audio, GenRoute.speech.modality());
    try testing.expectEqual(Modality.audio, GenRoute.music.modality());
    try testing.expectEqual(Modality.image, GenRoute.image.modality());
    try testing.expectEqual(Modality.mesh, GenRoute.mesh.modality());
}

test "audioBackendKindForType routes acestep to music, everything else to tts" {
    try testing.expect(audioBackendKindForType("acestep") == .music);
    try testing.expect(audioBackendKindForType("minimax_music3") == .music3);
    try testing.expect(audioBackendKindForType("qwen3_tts") == .tts);
    try testing.expect(audioBackendKindForType("gemma4") == .tts);
    // Both music engines serve /v1/audio/music-generations and advertise the
    // "music" capability; TTS backends never do.
    try testing.expect(AudioBackendKind.music.servesMusic());
    try testing.expect(AudioBackendKind.music3.servesMusic());
    try testing.expect(!AudioBackendKind.tts.servesMusic());
    try testing.expect(!AudioBackendKind.kokoro.servesMusic());
}

test "estimatePeakResidentBytes: minimax_music3 bills the sum plus its AR working set" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const b100: [100]u8 = @splat('x');
    const b40: [40]u8 = @splat('x');
    for ([_][]const u8{ "language_model.safetensors", "rvq_depth_decoder.safetensors", "transformer.safetensors", "condition_encoder.safetensors" }) |name|
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = &b100 });
    try tmp.dir.writeFile(io, .{ .sub_path = "vocoder.safetensors", .data = &b40 });
    // The whole engine is resident for its lifetime (no staging), so the sum
    // is right — but the AR stage's KV cache + frame-hidden buffer are real
    // resident bytes no directory sum can see.
    try std.testing.expectEqual(@as(u64, 440) + MUSIC3_GEN_BUFFER_BYTES, estimatePeakResidentBytesIn(io, tmp.dir, "minimax_music3"));
    // An unreadable/empty dir stays 0 = "unknown, never block".
    var empty = std.testing.tmpDir(.{ .iterate = true });
    defer empty.cleanup();
    try std.testing.expectEqual(@as(u64, 0), estimatePeakResidentBytesIn(io, empty.dir, "minimax_music3"));
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
    for ([_]Modality{ .image, .audio, .video, .mesh }) |m| {
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

// The connector fills the padded region by TILING its 128 learnable registers
// (`ltx_video.connectorTransform`: `num_tiles = T / num_reg`, integer division).
// A pad length that is not a whole number of tiles silently drops the remainder
// — the text embeddings come out short and nothing errors. 1024 is also the
// reference's own `TOKENIZER_MAX_LENGTH`; 256 was ours, and it gave the DiT a
// quarter of the text rows the connector was trained against.
test "LTX pad length is the reference's 1024 and a whole number of register tiles" {
    try std.testing.expectEqual(@as(usize, 1024), LTX_PAD_LEN);
    const registers: usize = 128; // connector.*.learnable_registers rows
    try std.testing.expectEqual(@as(usize, 0), LTX_PAD_LEN % registers);
    try std.testing.expectEqual(@as(usize, 8), LTX_PAD_LEN / registers);
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

test "flux low-mem policy: iOS always, env forces both ways, small-RAM Macs auto-enable" {
    const GB = 1024 * 1024 * 1024;
    const f = FluxImpl.lowMemFromInputs;
    try testing.expect(f(true, null, 128 * GB)); // iOS: always, RAM irrelevant
    try testing.expect(f(false, "1", 128 * GB)); // env force-on
    try testing.expect(!f(false, "0", 8 * GB)); // env force-off beats auto
    try testing.expect(f(false, null, 16 * GB)); // 16 GB mini: auto ON
    try testing.expect(f(false, null, 8 * GB)); // 8 GB: auto ON
    try testing.expect(!f(false, null, 24 * GB)); // 24 GB+: off (reload is pure loss)
    try testing.expect(!f(false, null, 0)); // unknown RAM: don't guess
}

test "clampFluxDim honors requested sizes on the /32 grid (512/768 are the 8GB-iPhone levers)" {
    try testing.expectEqual(@as(u32, 512), clampFluxDim(512));
    try testing.expectEqual(@as(u32, 768), clampFluxDim(768));
    try testing.expectEqual(@as(u32, 1024), clampFluxDim(1024));
    try testing.expectEqual(@as(u32, 1024), clampFluxDim(0)); // omitted → default
    try testing.expectEqual(@as(u32, 512), clampFluxDim(500)); // round up to /32
    try testing.expectEqual(@as(u32, 256), clampFluxDim(100)); // floor
    try testing.expectEqual(@as(u32, 1536), clampFluxDim(4096)); // cap
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

test "iterJsonStringArray walks ref_images entries" {
    // Two entries, whitespace tolerated, trailing fields ignored.
    var it = iterJsonStringArray("{\"ref_images\": [ \"QUJD\", \"REVG\" ], \"seed\":1}", "ref_images").?;
    try testing.expectEqualStrings("QUJD", it.next().?);
    try testing.expectEqualStrings("REVG", it.next().?);
    try testing.expect(it.next() == null);
    try testing.expect(!it.bad);
    // Empty array: no entries, not malformed.
    var e = iterJsonStringArray("{\"ref_images\":[]}", "ref_images").?;
    try testing.expect(e.next() == null);
    try testing.expect(!e.bad);
    // Absent key / non-array value → null (feature off vs 400 at the caller).
    try testing.expect(iterJsonStringArray("{\"seed\":1}", "ref_images") == null);
    try testing.expect(iterJsonStringArray("{\"ref_images\":\"QUJD\"}", "ref_images") == null);
    // Non-string element flags bad so the handler can 400 instead of ignoring.
    var b = iterJsonStringArray("{\"ref_images\":[1,2]}", "ref_images").?;
    try testing.expect(b.next() == null);
    try testing.expect(b.bad);
}

test "a named-400 on a reference set frees everything decoded before it" {
    // Every rejection here is a NORMAL return carrying a message, not a Zig
    // error, so `errdefer` does NOT fire — anything already decoded has to be
    // owned by something the caller frees. The test allocator is the assertion:
    // pre-fix each of these stranded the entries decoded before the bad one.
    const a = testing.allocator;
    var buf: [256]u8 = undefined;
    const cases = [_]struct { body: []const u8, needle: []const u8 }{
        // A valid entry, then an un-decodable base64 one.
        .{ .body = "{\"ref_images\":[\"QUJD\",\"!!!!\"]}", .needle = "'ref_images'[1]" },
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\",\"QUJD\",\"!!!!\"]}]}", .needle = "frames[2]" },
        .{ .body = "{\"ref_audios\":[\"!!!!\"]}", .needle = "'ref_audios'[0]" },
        // A whole video's frames decoded, then the SOUNDTRACK is unusable.
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\"],\"audio\":\"QUJD\"}]}", .needle = "'ref_videos'[0].audio" },
        // Malformed containers, and a sizing mode that is not one of the two.
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\",1]}]}", .needle = "'ref_videos'[0].frames" },
        .{ .body = "{\"ref_images\":[\"QUJD\"],\"ref_image_size\":\"huge\"}", .needle = "ref_image_size" },
        // Decodable base64 that is not an image: the entry is NAMED, and the
        // bytes already staged for it are freed.
        .{ .body = "{\"ref_images\":[\"QUJDRA==\"]}", .needle = "could not decode 'ref_images'[0]" },
    };
    for (cases) |c| {
        var out: std.ArrayList(minimax_h3.RefMedia) = .empty;
        defer {
            for (out.items) |*m| m.deinit();
            out.deinit(a);
        }
        const msg = (try parseH3Refs(a, c.body, 864, 480, 124, &out, &buf)) orelse {
            std.debug.print("expected a rejection for {s}\n", .{c.body});
            return error.ExpectedRejection;
        };
        try testing.expect(std.mem.indexOf(u8, msg, c.needle) != null);
    }
    // No reference fields at all is not a rejection — the feature is simply off.
    var none: std.ArrayList(minimax_h3.RefMedia) = .empty;
    defer none.deinit(a);
    try testing.expect((try parseH3Refs(a, "{\"prompt\":\"x\"}", 864, 480, 124, &none, &buf)) == null);
    try testing.expectEqual(@as(usize, 0), none.items.len);
}

test "reference audio arrives as 32 kHz stereo in the encoder's planar shape" {
    const a = testing.allocator;
    // The audio VAE is MONO and takes the stereo channels on the BATCH axis, so
    // interleaved -> [2, L] is a de-interleave, not a reshape. A reshape here
    // runs, produces the right shape, and encodes a channel-swapped chirp.
    var inter = [_]f32{ 0.0, 0.5, 0.1, 0.6, 0.2, 0.7 };
    const arr = try stereoInterleavedToArray(a, &inter);
    defer _ = mlx.mlx_array_free(arr);
    const shp = mlx.getShape(arr);
    try testing.expectEqual(@as(c_int, 2), shp[0]);
    try testing.expectEqual(@as(c_int, 3), shp[1]);
    try mlx.check(mlx.mlx_array_eval(arr));
    const d = mlx.mlx_array_data_float32(arr).?;
    const want = [_]f32{ 0.0, 0.1, 0.2, 0.5, 0.6, 0.7 };
    for (want, 0..) |v, i| try testing.expectApproxEqAbs(v, d[i], 1e-6);

    // A 16 kHz mono clip must come back at the VAE's 32 kHz, in stereo — the
    // rate is what the latent-frame count is computed from, so a clip left at
    // its own rate silently halves the reference's length.
    var mono: [1600]f32 = undefined;
    for (&mono, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    const wav_bytes = try wav_mod.encodePcm16(a, &mono, 16000, 1);
    defer a.free(wav_bytes);
    const b64 = try a.alloc(u8, std.base64.standard.Encoder.calcSize(wav_bytes.len));
    defer a.free(b64);
    _ = std.base64.standard.Encoder.encode(b64, wav_bytes);
    const pcm = try refWavTo32kStereo(a, b64);
    defer a.free(pcm);
    try testing.expectEqual(@as(usize, 3200 * 2), pcm.len);
    // Duplicated, not summed: both channels carry the mono signal.
    try testing.expectApproxEqAbs(pcm[0], pcm[1], 1e-6);
    try testing.expectApproxEqAbs(pcm[200], pcm[201], 1e-6);
}

test "h3ConfigDeclaresRef2va reads the pack's own task list" {
    const a = testing.allocator;
    // The converter writes the partition's task list; ref2va and fl2va share
    // every geometry number, so the DiT file is the ONLY thing that differs and
    // this is the only way to tell an FL2VA pack from a REF2VA one.
    try testing.expect(h3ConfigDeclaresRef2va(a, "{\"model_type\":\"minimax_h3\",\"tasks\":[\"t2va\",\"ref2va\"]}"));
    try testing.expect(!h3ConfigDeclaresRef2va(a, "{\"model_type\":\"minimax_h3\",\"tasks\":[\"t2va\",\"fl2va\"]}"));
    // Absent / malformed / wrong-typed → NOT ref2va. A pack that cannot say it
    // supports references must not be handed them: it would generate happily
    // and ignore every one of them.
    try testing.expect(!h3ConfigDeclaresRef2va(a, "{\"model_type\":\"minimax_h3\"}"));
    try testing.expect(!h3ConfigDeclaresRef2va(a, "{\"tasks\":\"ref2va\"}"));
    try testing.expect(!h3ConfigDeclaresRef2va(a, "not json"));
    // A substring match on the whole file would pass on the README-ish text a
    // config can legally carry; only the task LIST counts.
    try testing.expect(!h3ConfigDeclaresRef2va(a, "{\"note\":\"ref2va\",\"tasks\":[\"t2va\",\"fl2va\"]}"));
}

test "iterJsonObjectArray walks ref_videos entries" {
    // A reference video is an OBJECT — `{"frames":[…],"audio":"…"}` — so the
    // soundtrack is a field on the video it belongs to. The original shape was
    // a parallel `ref_video_audios` array, where a null hole silently
    // mis-pairs a soundtrack with the wrong clip.
    const body =
        "{\"ref_videos\":[ {\"frames\":[\"QQ==\",\"Qg==\"],\"audio\":\"Ug==\"} , {\"frames\":[\"Qw==\"]} ],\"seed\":3}";
    var it = iterJsonObjectArray(body, "ref_videos").?;
    const o1 = it.next().?;
    var f1 = iterJsonStringArray(o1, "frames").?;
    try testing.expectEqualStrings("QQ==", f1.next().?);
    try testing.expectEqualStrings("Qg==", f1.next().?);
    try testing.expect(f1.next() == null);
    try testing.expectEqualStrings("Ug==", extractJsonString(o1, "audio").?);
    const o2 = it.next().?;
    // The second object must NOT see the first one's audio — an unbalanced
    // scan that overruns is exactly how a soundtrack lands on the wrong clip.
    try testing.expect(extractJsonString(o2, "audio") == null);
    var f2 = iterJsonStringArray(o2, "frames").?;
    try testing.expectEqualStrings("Qw==", f2.next().?);
    try testing.expect(it.next() == null);
    try testing.expect(!it.bad);

    // A brace inside a quoted string does not close the object.
    var q = iterJsonObjectArray("{\"ref_videos\":[{\"audio\":\"a}b\",\"frames\":[\"QQ==\"]}]}", "ref_videos").?;
    const qo = q.next().?;
    try testing.expectEqualStrings("a}b", extractJsonString(qo, "audio").?);
    try testing.expect(q.next() == null);
    try testing.expect(!q.bad);

    // Empty array: no entries, not malformed.
    var e = iterJsonObjectArray("{\"ref_videos\":[]}", "ref_videos").?;
    try testing.expect(e.next() == null);
    try testing.expect(!e.bad);

    // Absent key / non-array value → null (feature off, not a 400).
    try testing.expect(iterJsonObjectArray("{\"seed\":1}", "ref_videos") == null);
    try testing.expect(iterJsonObjectArray("{\"ref_videos\":\"x\"}", "ref_videos") == null);

    // A non-object element and an unterminated array flag bad, so the handler
    // 400s by name instead of generating while ignoring what was asked for.
    var b = iterJsonObjectArray("{\"ref_videos\":[\"QQ==\"]}", "ref_videos").?;
    try testing.expect(b.next() == null);
    try testing.expect(b.bad);
    var u = iterJsonObjectArray("{\"ref_videos\":[{\"frames\":[", "ref_videos").?;
    try testing.expect(u.next() == null);
    try testing.expect(u.bad);
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

test "paint stage dir resolves from the combined single-repo layout (subdir first, sibling fallback)" {
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io_tmp = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io_tmp, &root_buf);
    const root = root_buf[0..root_len];

    const mkModelDir = struct {
        fn call(a: std.mem.Allocator, tmp_dir: std.Io.Dir, base: []const u8, rel: []const u8) ![]u8 {
            const io = std.Io.Threaded.global_single_threaded.io();
            try tmp_dir.createDirPath(io, rel);
            const cfg_rel = try std.fs.path.join(a, &.{ rel, "config.json" });
            defer a.free(cfg_rel);
            const f = try tmp_dir.createFile(io, cfg_rel, .{});
            f.close(io);
            return std.fs.path.join(a, &.{ base, rel });
        }
    }.call;

    // Combined single-HF-repo layout: shape at the root, paint/ inside.
    const shape = try mkModelDir(allocator, tmp.dir, root, "combined");
    defer allocator.free(shape);
    const paint_sub = try mkModelDir(allocator, tmp.dir, root, "combined/paint");
    defer allocator.free(paint_sub);

    const paint = findPaintDir(allocator, shape) orelse return error.TestExpectedResult;
    defer allocator.free(paint);
    try testing.expectEqualStrings(paint_sub, paint);

    // Legacy local-convert layout (sibling dirs) still resolves.
    const shape2 = try mkModelDir(allocator, tmp.dir, root, "local/hunyuan3d-2-1-8bit");
    defer allocator.free(shape2);
    const paint_sib = try mkModelDir(allocator, tmp.dir, root, "local/hunyuan3d-2-1-paint-8bit");
    defer allocator.free(paint_sib);

    const paint2 = findPaintDir(allocator, shape2) orelse return error.TestExpectedResult;
    defer allocator.free(paint2);
    try testing.expectEqualStrings(paint_sib, paint2);

    // Nothing anywhere -> graceful null (texture requests 400).
    const bare = try mkModelDir(allocator, tmp.dir, root, "bare/shape-only");
    defer allocator.free(bare);
    try testing.expect(findPaintDir(allocator, bare) == null);
}

test "media model types: discovery and modality dispatch agree" {
    // CLASS GUARD. `model_discovery.isMediaModelType` and `modalityFromType`
    // are documented duplication (discovery must not import mlx), and they
    // silently drifted: `minimax_h3` was added to the dispatcher but not to
    // discovery, so `/v1/load-model` answered
    //   400 "Model at that path has an unsupported model_type"
    // for a model the server could actually serve. Neither side is wrong on
    // its own — only their DISAGREEMENT is — so the check is bidirectional.
    for (media_model_types) |mt| {
        try std.testing.expect(discovery.isMediaModelType(mt));
        try std.testing.expect(modalityFromType(mt) != null);
    }
    // And a non-media type must be rejected by BOTH, or a chat model would be
    // routed to a media engine.
    for ([_][]const u8{ "gemma4", "qwen3", "llama", "deepseek_v4", "bert" }) |mt| {
        try std.testing.expect(!discovery.isMediaModelType(mt));
        try std.testing.expect(modalityFromType(mt) == null);
    }
}

test "stagedPeakBytes: disjoint stages never sum, and resident always carries" {
    const GB: u64 = 1024 * 1024 * 1024;
    // The whole point: stages are loaded and freed in turn, so only the
    // biggest is ever on top of what the engine holds for its lifetime.
    try std.testing.expectEqual(12 * GB, stagedPeakBytes(4 * GB, &.{ 8 * GB, 3 * GB, 1 * GB }));
    try std.testing.expectEqual(4 * GB, stagedPeakBytes(4 * GB, &.{}));
    try std.testing.expectEqual(8 * GB, stagedPeakBytes(0, &.{ 8 * GB, 3 * GB }));
    // Nothing known → 0, which the preflight reads as "unknown, never block".
    try std.testing.expectEqual(@as(u64, 0), stagedPeakBytes(0, &.{ 0, 0 }));
}

test "LTX bills ONE transformer variant, plus the text encoder its dir cannot see" {
    const MB: u64 = 1024 * 1024;
    // Real dgrauet/ltx-2.3-mlx-q4 sizes. Both transformer variants ship at
    // 10.54 GiB and `ensureTransformer` frees one BEFORE loading the other, so
    // the sum bills a phantom. The Gemma text encoder is a SEPARATE repo
    // (mlx-community/gemma-3-12b-it-4bit, 7.5 GiB) loaded per generation on top
    // of the resident engine, so the dir sum bills it at zero.
    const dir_sum: u64 = 30_318 * MB; // all eight files
    const variant: u64 = 10_793 * MB;
    const gemma: u64 = 7_680 * MB;

    const peak = ltxPeakBytes(dir_sum, variant, gemma);
    try std.testing.expectEqual(dir_sum - variant + gemma, peak);
    // Strictly below the sum-of-dir bill it replaces on a two-variant pack…
    try std.testing.expect(peak < dir_sum + dir_sum / 10);
    // …but a one-variant pack must go UP, not down: the text encoder is real
    // and was billed at nothing. Under-billing is the uncatchable-OOM side.
    try std.testing.expect(ltxPeakBytes(dir_sum - variant, 0, gemma) > dir_sum - variant);
    // A missing/unfound encoder dir contributes nothing rather than guessing.
    try std.testing.expectEqual(dir_sum - variant, ltxPeakBytes(dir_sum, variant, 0));

    // LTX 2.5 ships its text encoder INSIDE the pack, and `sumSafetensorsIn`
    // recurses one level — so `dir_sum` already carries it. It is resident
    // only while the engine is (loaded per generation, freed on return), so
    // the peak is still sum-minus-spare and the encoder must NOT also ride in
    // as a stage: that bills 6.8 GiB twice and refuses loads that fit.
    const te_in_pack: u64 = 6_800 * MB;
    const sum_25: u64 = dir_sum + te_in_pack;
    try std.testing.expectEqual(sum_25 - variant, ltxPeakBytes(sum_25, variant, 0));
    try std.testing.expect(ltxPeakBytes(sum_25, variant, te_in_pack) > sum_25 - variant);
}

test "h3 staged-residency peak bills the BIGGEST stage, never a sum of disjoint ones" {
    const GB: u64 = 1024 * 1024 * 1024;
    const act = H3_ACTIVATION_BYTES;
    // Three disjoint stages: the TE is freed before the DiT loads, the DiT is
    // freed before the VAEs load. Billing any two together refuses a Mac that
    // would work — the VAEs used to be added to the DiT term.
    try std.testing.expectEqual(35 * GB + act, h3PeakBytes(28 * GB, 35 * GB, 5 * GB, 1 * GB));
    // The two VAEs DO coexist (the video decoder is still resident when the
    // audio one loads), so they are one stage.
    try std.testing.expectEqual(9 * GB + act, h3PeakBytes(4 * GB, 3 * GB, 5 * GB, 4 * GB));
    // The TE stage carries no generation transients — one forward over a few
    // hundred prompt rows. Adding them to it is what refused the 8-bit pack.
    try std.testing.expectEqual(48 * GB, h3PeakBytes(48 * GB, 35 * GB, 5 * GB, 1 * GB));
    // All-unknown must stay 0: the preflight treats 0 as "unknown, never block".
    try std.testing.expectEqual(@as(u64, 0), h3PeakBytes(0, 0, 0, 0));

    // The real 8-bit pack on a 48 GB Mac (auto cap 29.95 GiB): 26.28 TE,
    // 32.83 DiT + 0.73 turbo, 4.85 + 0.56 VAEs. Measured process peak 26 GB.
    const MB: u64 = 1024 * 1024;
    const real = h3PeakBytes(
        26_910 * MB,
        h3DitResidentBytes(33_618 * MB, true) + 747 * MB,
        4_966 * MB,
        573 * MB,
    );
    try std.testing.expect(real < 29 * GB); // fits the 48 GB Mac's auto cap
    try std.testing.expect(real > 24 * GB); // and stays above the measured peak
}

test "h3 DiT term sheds the AdaLN weights precompute frees — unless it is off" {
    const GB: u64 = 1024 * 1024 * 1024;
    // Measured: the 8-bit pack's 32.83 GiB transformer.safetensors settles at
    // 20.19 GiB resident once precomputeAdaln frees the 13B modulation
    // weights, so the file size over-bills the DiT by ~12 GiB.
    const eight_bit: u64 = 32 * GB;
    try std.testing.expect(h3DitResidentBytes(eight_bit, true) < eight_bit);
    try std.testing.expect(h3DitResidentBytes(eight_bit, true) >= 20 * GB);
    // MINIMAX_H3_ADALN_PRECOMPUTE=0 never frees them, so the whole file stays
    // resident and billing the shed size would be an uncatchable OOM.
    try std.testing.expectEqual(eight_bit, h3DitResidentBytes(eight_bit, false));
}

test "estimatePeakResidentBytes: minimax_h3 stages, other types keep the sum" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const b300: [300]u8 = @splat('x');
    const b500: [500]u8 = @splat('x');
    const b120: [120]u8 = @splat('x');
    const b30: [30]u8 = @splat('x');
    const b1000: [1000]u8 = @splat('x');
    try tmp.dir.writeFile(io, .{ .sub_path = "text_encoder.safetensors", .data = &b300 });
    try tmp.dir.writeFile(io, .{ .sub_path = "transformer.safetensors", .data = &b500 });
    try tmp.dir.writeFile(io, .{ .sub_path = "video_vae.safetensors", .data = &b120 });
    try tmp.dir.writeFile(io, .{ .sub_path = "audio_vae.safetensors", .data = &b30 });
    // A file H3's residency plan never loads: billed by the default sum,
    // never by the staged estimate.
    try tmp.dir.writeFile(io, .{ .sub_path = "extra.safetensors", .data = &b1000 });

    // LTX (`AudioVideo`): the dir sum MINUS the transformer variant that never
    // coexists with the other. Its text encoder lives in a different repo, so
    // it is added by the outer `estimatePeakResidentBytes`, not here.
    try tmp.dir.writeFile(io, .{ .sub_path = "transformer-dev.safetensors", .data = &b500 });
    try tmp.dir.writeFile(io, .{ .sub_path = "transformer-distilled.safetensors", .data = &b500 });
    try std.testing.expectEqual(@as(u64, 2450), estimatePeakResidentBytesIn(io, tmp.dir, "AudioVideo"));
    tmp.dir.deleteFile(io, "transformer-dev.safetensors") catch {};
    tmp.dir.deleteFile(io, "transformer-distilled.safetensors") catch {};

    // H3: max(TE 300, DiT 500*65% + activations, VAEs 120+30 + activations).
    try std.testing.expectEqual(325 + H3_ACTIVATION_BYTES, estimatePeakResidentBytesIn(io, tmp.dir, "minimax_h3"));

    // A pack shipping the Turbo LoRA bills it on the DiT term (it is resident
    // ALONGSIDE the DiT and precompute does not free it), whenever present —
    // the gate estimate is per-model, not per-request.
    const b80: [80]u8 = @splat('x');
    try tmp.dir.writeFile(io, .{ .sub_path = "turbo_lora.safetensors", .data = &b80 });
    try std.testing.expectEqual(405 + H3_ACTIVATION_BYTES, estimatePeakResidentBytesIn(io, tmp.dir, "minimax_h3"));
    tmp.dir.deleteFile(io, "turbo_lora.safetensors") catch {};
    // Any other media type: the plain sum over the dir (the safe default —
    // a backend without a declared residency plan must not under-bill).
    try std.testing.expectEqual(@as(u64, 1950), estimatePeakResidentBytesIn(io, tmp.dir, "flux2"));
}

test "media markers are per-TYPE, not per-modality" {
    // REGRESSION. `detectModality` guarded the whole `.video` modality on
    // LTX's `connector.safetensors`. MiniMax-H3 has no such file, so detection
    // returned null and the loader fell through to the MLX TEXT path — it
    // globbed all four H3 safetensors into one weight map and failed on
    // `model.norm.weight`, a Qwen tensor H3 does not have.
    //
    // The invariant: a marker belongs to a BACKEND. Requiring one backend's
    // file from every model in its modality breaks the next backend added.
    try std.testing.expectEqualStrings("connector.safetensors", requiredMarkerFor("AudioVideo").?);
    try std.testing.expectEqualStrings("transformer.safetensors", requiredMarkerFor("minimax_h3").?);
    // Music3: the converter writes the vocoder LAST, so its presence is the
    // completeness marker for the whole five-file pack.
    try std.testing.expectEqualStrings("vocoder.safetensors", requiredMarkerFor("minimax_music3").?);
    // H3 must NOT be gated on LTX's file.
    try std.testing.expect(!std.mem.eql(u8, requiredMarkerFor("minimax_h3").?, "connector.safetensors"));

    // Every media type either declares its own marker or needs none; none may
    // inherit another backend's.
    for (media_model_types) |mt| {
        if (requiredMarkerFor(mt)) |m| {
            try std.testing.expect(m.len > 0);
            if (!std.mem.eql(u8, mt, "AudioVideo"))
                try std.testing.expect(!std.mem.eql(u8, m, "connector.safetensors"));
        }
    }
    // A non-media type never carries one.
    try std.testing.expect(requiredMarkerFor("gemma4") == null);
}

test "incompleteMediaDir: marker-missing media dir is refused, complete and non-media are not" {
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const root = root_buf[0..root_len];

    try tmp.dir.createDirPath(io, "fragment");
    try tmp.dir.writeFile(io, .{ .sub_path = "fragment/config.json", .data = "{\"model_type\":\"minimax_h3\"}" });
    try tmp.dir.createDirPath(io, "complete");
    try tmp.dir.writeFile(io, .{ .sub_path = "complete/config.json", .data = "{\"model_type\":\"minimax_h3\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "complete/transformer.safetensors", .data = "x" });
    try tmp.dir.createDirPath(io, "chat");
    try tmp.dir.writeFile(io, .{ .sub_path = "chat/config.json", .data = "{\"model_type\":\"gemma4\"}" });

    const frag = try std.fs.path.join(allocator, &.{ root, "fragment" });
    defer allocator.free(frag);
    const comp = try std.fs.path.join(allocator, &.{ root, "complete" });
    defer allocator.free(comp);
    const chat_dir = try std.fs.path.join(allocator, &.{ root, "chat" });
    defer allocator.free(chat_dir);

    try testing.expect(incompleteMediaDir(io, allocator, frag));
    try testing.expect(!incompleteMediaDir(io, allocator, comp));
    try testing.expect(!incompleteMediaDir(io, allocator, chat_dir));
    // The refused dir is exactly the one detectModality declines.
    try testing.expect(detectModality(io, allocator, frag) == null);
    try testing.expectEqual(Modality.video, detectModality(io, allocator, comp).?);
}

test "instrumental is a request-level rule: the flag and real lyrics are a conflict, not a race" {
    // `instrumental: true` beside words to sing is contradictory, and letting
    // either side quietly win is the failure mode — a user who typed a verse
    // and left a sticky checkbox set gets a wordless track with no explanation,
    // or a checkbox that does nothing. Both backends read the ONE predicate, so
    // the rule cannot drift between them.
    try testing.expect(instrumentalConflicts(true, "[verse]\nhello"));
    try testing.expect(instrumentalConflicts(true, "la la la"));
    // Whitespace-only is ABSENT, not a conflict: an app that keeps a blank
    // lyrics editor mounted beside the checkbox must not 400.
    try testing.expect(!instrumentalConflicts(true, ""));
    try testing.expect(!instrumentalConflicts(true, "  \n\t\r "));
    // The flag off never conflicts, whatever the lyrics say.
    try testing.expect(!instrumentalConflicts(false, "[verse]\nhello"));
    try testing.expect(!instrumentalConflicts(false, ""));
}

test "instrumental is parsed off the body only when spelled true" {
    try testing.expect(sse.bodyWantsTrue("{\"instrumental\":true}", "instrumental"));
    try testing.expect(sse.bodyWantsTrue("{\"instrumental\": true}", "instrumental"));
    try testing.expect(!sse.bodyWantsTrue("{\"instrumental\":false}", "instrumental"));
    try testing.expect(!sse.bodyWantsTrue("{\"prompt\":\"x\"}", "instrumental"));
}

test "videoRgbTransportReason: chained windows are billed into the response cap (#283)" {
    // 141f/window x 5 windows at 1056x864 = 701 frames = 1.9 GB raw: refused by name.
    try std.testing.expect(videoRgbTransportReason(minimax_h3.chainDeliveredFrames(5, 141), 1056, 864) != null);
    // One window of the same shape fits.
    try std.testing.expect(videoRgbTransportReason(141, 1056, 864) == null);
}
