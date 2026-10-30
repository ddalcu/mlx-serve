//! The LTX engine: its components (one transformer variant at a time, connector, VAEs, the optional audio VAE,
//! upsampler and DiffVAE decoder), the Gemma text encoder's dir and tokenizer, runtime LoRAs, and the request
//! arithmetic the pipelines share (prompt padding, guider defaults, the a2vid gates).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const lora_mod = @import("../../sdk/weights/lora.zig");
const ltx = @import("video.zig");
const ltx_audio = @import("audio.zig");
const diffvae_fwd = @import("diffvae_forward.zig");

// The reference's `TOKENIZER_MAX_LENGTH` (ltx_core/text_encoders/gemma/
// gemma_assets.py), for 2.3 and 2.5 alike. `LTXGemmaTokenizer` DEFAULTS to 256,
// but no caller uses that default — `build_gemma_tokenizer` passes 1024, and the
// encoder's own docstring says "the tokenizer pads every prompt to max_length
// (1024)". We shipped 256 from the 2.3 port onwards, which left the connector
// tiling its 128 learnable registers over 2 slots instead of 8 and gave the DiT
// a quarter of the text rows it was trained to cross-attend to.
pub const PAD_LEN: usize = 1024; // gemma left-pad length
pub const PAD_ID: i32 = 0; // gemma <pad>
pub const GEMMA_BOS: i32 = 2; // <bos>

// Reference DEFAULT_NEGATIVE_PROMPT, plus a subtitle/caption block after
// "artifacts around text": quoted dialogue in the prompt makes the model burn
// scrambled subtitle-like captions into the frame; these terms steer CFG away
// from that. The audio tail (lip sync, muted/distorted voice, background
// noise, dialogue terms) is load-bearing for speech when audio CFG runs; if
// the whole thing ever exceeds PAD_LEN (~229 tokens today, 1024 budget),
// padWithBos left-truncates and keeps that tail.
pub const NEGATIVE_PROMPT =
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

/// The loaded pack: one transformer variant at a time (`ensureTransformer` frees before it loads), the connector and
/// VAEs, the text encoder's dir and tokenizer, the attached LoRA stack.
pub const Engine = struct {
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

    /// `stream` runs the forward; null = a fresh GPU stream.
    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, stream: ?mlx.mlx_stream) !*Engine {
        const self = try allocator.create(Engine);
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

        const load_s = mlx.weightLoadStream();
        self.s = stream orelse mlx.mlx_default_gpu_stream_new();

        // Initial transformer: prefer DISTILLED (the correct one-stage default —
        // the dev model without CFG produces visibly worse output); fall back to
        // dev for bundles downloaded before transformer-distilled shipped.
        const initial: TransformerVariant = if (self.hasVariant(io, .distilled)) .distilled else .dev;
        self.transformer = try loadTransformerVariant(allocator, model_dir, initial, self.ltx_cfg, load_s);
        self.transformer_variant = initial;
        errdefer self.transformer.deinit();
        if (initial == .dev)
            log.warn("[video] transformer-distilled.safetensors not found — one-stage falls back to the dev transformer (download the distilled variant for reference-quality fast generations)\n", .{});

        const cp = try std.fmt.allocPrintSentinel(allocator, "{s}/connector.safetensors", .{model_dir}, 0);
        defer allocator.free(cp);
        self.connector = try ltx.loadComponent(allocator, cp, load_s);
        errdefer self.connector.deinit();
        const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.safetensors", .{model_dir}, 0);
        defer allocator.free(vp);
        self.vae = try ltx.loadComponent(allocator, vp, load_s);
        errdefer self.vae.deinit();
        var it = self.vae.map.iterator();
        while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*); // VAE conv graph wants materialized weights

        // Optional audio VAE + vocoder → the generated video gets a sound track.
        // Absent (video-only checkpoints, or not yet downloaded) is graceful.
        self.audio = loadAudioVae(io, allocator, model_dir, load_s);
        errdefer if (self.audio) |*a| a.deinit();

        // Optional VAE encoder → image-to-video (first-frame conditioning) and
        // the two-stage latent (de)normalization. Absent is graceful → t2v only.
        self.vae_encoder = loadVaeEncoder(io, allocator, model_dir, load_s);
        errdefer if (self.vae_encoder) |*e| e.deinit();

        self.tok = try tok_mod.loadTokenizerAny(io, allocator, self.gemma_dir);
        log.info("[video] LTX components + tokenizer ready (transformer={s})\n", .{@tagName(initial)});
        return self;
    }

    pub fn hasVariant(self: *Engine, io: std.Io, variant: TransformerVariant) bool {
        var buf: [1024]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ self.model_dir, variant.fileName() }, 0) catch return false;
        return fileExists(io, p);
    }

    /// Swap the transformer slot to `want` (no-op when already loaded). The old
    /// component is freed BEFORE the new one loads so dev + distilled (11 GB
    /// each) never coexist.
    pub fn ensureTransformer(self: *Engine, want: TransformerVariant) !void {
        if (self.transformer_variant == want) return;
        log.info("[video] swapping transformer: {s} -> {s}\n", .{ @tagName(self.transformer_variant), @tagName(want) });
        self.transformer.deinit();
        const load_s = mlx.weightLoadStream();
        self.transformer = try loadTransformerVariant(self.allocator, self.model_dir, want, self.ltx_cfg, load_s);
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
    pub fn setLoras(self: *Engine, paths: []const []const u8, scales: []const f32) !u32 {
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

    fn clearLora(self: *Engine) void {
        self.transformer.lora = null;
        if (self.lora_stack) |*st| st.deinit();
        self.lora_stack = null;
        self.lora_matched = 0;
    }

    fn applyLora(self: *Engine) void {
        self.transformer.lora = if (self.lora_stack) |*st| st else null;
    }

    /// Lazily load the spatial-x2 upsampler for the two-stage boundary.
    pub fn ensureUpsampler(self: *Engine, io: std.Io) !*const ltx.Component {
        if (self.upsampler) |*u| return u;
        var buf: [1024]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}.safetensors", .{ self.model_dir, ltx.UPSAMPLER_PREFIX }, 0) catch return error.MissingUpsampler;
        if (!fileExists(io, p)) return error.MissingUpsampler;
        const load_s = mlx.weightLoadStream();
        var comp = try ltx.loadComponent(self.allocator, p, load_s);
        var it = comp.map.iterator();
        while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*); // conv graph wants materialized weights
        self.upsampler = comp;
        log.info("[video] latent upsampler ready ({d} tensors)\n", .{comp.count()});
        return &self.upsampler.?;
    }

    /// Lazily load the DiffVAE decoder. Absent → `error.MissingDiffusionDecoder`,
    /// which the handler turns into a NAMED 400; a silent fall back to the conv
    /// decoder would answer a different question than the one asked.
    pub fn ensureDiffusionDecoder(self: *Engine, io: std.Io) !*const ltx.Component {
        if (self.diffusion_decoder) |*d| return d;
        var buf: [1024]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ self.model_dir, diffvae_fwd.FILE_NAME }, 0) catch return error.MissingDiffusionDecoder;
        if (!fileExists(io, p)) return error.MissingDiffusionDecoder;
        const load_s = mlx.weightLoadStream();
        self.diffusion_decoder = try diffvae_fwd.load(self.allocator, p, load_s);
        return &self.diffusion_decoder.?;
    }

    pub fn deinit(self: *Engine) void {
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

fn loadTransformerVariant(allocator: std.mem.Allocator, model_dir: []const u8, variant: TransformerVariant, cfg: ltx.LtxConfig, load_s: mlx.mlx_stream) !ltx.Component {
    const tp = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, variant.fileName() }, 0);
    defer allocator.free(tp);
    return ltx.loadTransformer(allocator, tp, cfg, load_s, mlx.gpuStream());
}

/// Load the LTX VAE encoder (`vae_encoder.safetensors`, ~0.6 GB, MLX-layout
/// bf16) from the model dir for image-to-video. Absent → null (I2V disabled,
/// text-to-video unaffected). Mirrors `loadAudioVae`.
fn loadVaeEncoder(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, load_s: mlx.mlx_stream) ?ltx.Component {
    var p: [1024]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&p, "{s}/vae_encoder.safetensors", .{model_dir}, 0) catch return null;
    if (!fileExists(io, path)) {
        log.info("[video] no vae_encoder.safetensors in {s} — image-to-video disabled (text-to-video only)\n", .{model_dir});
        return null;
    }
    var comp = ltx.loadComponent(allocator, path, load_s) catch |e| {
        log.warn("[video] vae_encoder load failed ({}) — image-to-video disabled\n", .{e});
        return null;
    };
    var it = comp.map.iterator();
    while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*); // conv graph wants materialized weights
    log.info("[video] VAE encoder ready ({d} tensors) — image-to-video enabled\n", .{comp.count()});
    return comp;
}



fn loadAudioVae(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, load_s: mlx.mlx_stream) ?ltx.Component {
    // The directory holding the two audio files: model dir, or an override.
    const dir: []const u8 = if (env.span(.LTX_AUDIO_DIR)) |e| blk: {
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
    var comp = ltx_audio.loadAudioComponents(allocator, audio_path, voc_path, load_s) catch |e| {
        log.warn("[video] audio VAE load failed ({}) — video will be silent\n", .{e});
        return null;
    };
    log.info("[video] audio VAE + vocoder ready ({d} tensors) — video will have sound\n", .{comp.count()});
    return comp;
}


/// LTX's text encoder is Gemma-3-12B (4-bit). It's a normal downloadable model
/// the app pulls into `~/.mlx-serve/models` (as the LTX bundle dependency, and
/// selectable as a chat model). The repo id maps to a `<author>/<name>` dir.
pub const GEMMA_REPO_DIR = "mlx-community/gemma-3-12b-it-4bit";

/// Locate the Gemma-3-12B text encoder ONLY under `~/.mlx-serve/models` — the
/// single source of truth for downloaded models. No HF-cache magic: the app
/// owns downloads. `$LTX_GEMMA_DIR` stays as an explicit override (tests /
/// custom installs). A candidate is accepted only if it has a `config.json`,
/// so a partial download never gets handed back.
fn resolveGemmaDir(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, version: ltx.LtxVersion) ![]u8 {
    if (env.span(.LTX_GEMMA_DIR)) |e| {
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
    const home = std.mem.span(env.get(.HOME) orelse return error.NoGemmaDir);
    if (!std.fs.path.isAbsolute(home)) return error.NoGemmaDir;
    // 2-level `<author>/<name>` layout (what DownloadManager writes), then a
    // flat `<name>` layout (legacy / manual placement).
    const candidates = [_][]const u8{ GEMMA_REPO_DIR, "gemma-3-12b-it-4bit" };
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
/// then LEFT-pad/truncate to PAD_LEN with PAD_ID.
pub fn tokenizePadded(allocator: std.mem.Allocator, tokenizer: *tok_mod.Tokenizer, text: []const u8) ![]i32 {
    const enc = try tokenizer.encode(allocator, text);
    defer allocator.free(enc);
    return padWithBos(allocator, enc, GEMMA_BOS, PAD_LEN, PAD_ID);
}

/// Pure BOS-prepend + left-pad (testable without a live tokenizer).
pub fn padWithBos(allocator: std.mem.Allocator, enc: []const u32, bos: i32, pad_len: usize, pad_id: i32) ![]i32 {
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


pub const Pipeline = enum {
    one_stage,
    two_stage,
    two_stage_hq,

    pub fn fromName(name: ?[]const u8) Pipeline {
        const raw = name orelse return .one_stage;
        if (std.mem.eql(u8, raw, "two_stage")) return .two_stage;
        if (std.mem.eql(u8, raw, "two_stage_hq")) return .two_stage_hq;
        return .one_stage;
    }
};

/// STG perturbs block 28 by default (reference MultiModalGuiderParams for the
/// Euler two-stage pipeline; HQ uses no STG blocks).
const STG_BLOCKS_DEFAULT = [_]u32{28};

pub const Guiders = struct {
    vp: ltx.GuiderParams,
    ap: ltx.GuiderParams,
    stage1_steps_default: u32,
};

/// Reference per-pipeline guidance defaults, with per-request overrides:
/// Audio-to-video pipeline gate: the reference a2vid pipelines are two-stage
/// only (stage 1 needs real CFG against the frozen soundtrack; the distilled
/// one-stage schedule was never trained with audio conditioning). Returns the
/// 400 message, or null when the pipeline is allowed.
pub fn a2vidPipelineError(pipeline: Pipeline) ?[]const u8 {
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
pub fn guiderDefaults(pipeline: Pipeline, cfg_video: ?f32, cfg_audio: ?f32, stg: ?f32) Guiders {
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
pub const Stage2Swap = struct {
    engine: *Engine,

    pub fn swap(ctx: *anyopaque) anyerror!*const ltx.Component {
        const self: *Stage2Swap = @ptrCast(@alignCast(ctx));
        try self.engine.ensureTransformer(.distilled);
        return &self.engine.transformer;
    }
};

fn fileExists(io: std.Io, path: [:0]const u8) bool {
    // openFileAbsolute asserts an absolute path (ReleaseFast UB otherwise); these come from the pack dir and env.
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return false;
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

// ── tests ──

const testing = std.testing;

test "ltx Pipeline.fromName: absent or unknown is one-stage" {
    try testing.expectEqual(Pipeline.one_stage, Pipeline.fromName(null));
    try testing.expectEqual(Pipeline.one_stage, Pipeline.fromName("one_stage"));
    try testing.expectEqual(Pipeline.two_stage, Pipeline.fromName("two_stage"));
    try testing.expectEqual(Pipeline.two_stage_hq, Pipeline.fromName("two_stage_hq"));
    try testing.expectEqual(Pipeline.one_stage, Pipeline.fromName("garbage"));
}

test "padWithBos prepends gemma <bos> (off-prompt regression)" {
    const a = testing.allocator;
    const enc = [_]u32{ 236746, 2604, 37423 };
    const ids = try padWithBos(a, &enc, 2, 8, 0);
    defer a.free(ids);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0, 0, 2, 236746, 2604, 37423 }, ids);
}

test "padWithBos does not double an existing <bos>" {
    const a = testing.allocator;
    const enc = [_]u32{ 2, 236746, 2604 };
    const ids = try padWithBos(a, &enc, 2, 6, 0);
    defer a.free(ids);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0, 2, 236746, 2604 }, ids);
}

test "guiderDefaults mirrors the reference per-pipeline guidance" {
    // one-stage: no guidance by default (single forward), override respected
    const one = guiderDefaults(.one_stage, null, null, null);
    try testing.expect(!one.vp.needsGuidance());
    try testing.expect(!one.ap.needsGuidance());
    try testing.expectEqual(@as(u32, 30), one.stage1_steps_default);
    const one_ovr = guiderDefaults(.one_stage, 3.0, null, null);
    try testing.expectApproxEqAbs(@as(f32, 3.0), one_ovr.vp.cfg, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 3.0), one_ovr.ap.cfg, 0.0); // audio follows video override

    // two-stage: cfg 3/7, rescale 0.7, modality 3.0, STG block 28 available
    const two = guiderDefaults(.two_stage, null, null, null);
    try testing.expectApproxEqAbs(@as(f32, 3.0), two.vp.cfg, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 7.0), two.ap.cfg, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.7), two.vp.rescale, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 3.0), two.vp.modality, 0.0);
    try testing.expectEqual(@as(usize, 1), two.vp.stg_blocks.len);
    try testing.expectEqual(@as(u32, 28), two.vp.stg_blocks[0]);
    try testing.expect(!two.vp.needsPerturbed()); // stg defaults 0.0
    const two_stg = guiderDefaults(.two_stage, null, null, 1.0);
    try testing.expect(two_stg.vp.needsPerturbed());

    // HQ: rescale 0.45 video / 1.0 audio, no STG blocks, 15 default steps
    const hq = guiderDefaults(.two_stage_hq, null, null, null);
    try testing.expectApproxEqAbs(@as(f32, 0.45), hq.vp.rescale, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), hq.ap.rescale, 0.0);
    try testing.expectEqual(@as(usize, 0), hq.vp.stg_blocks.len);
    try testing.expectEqual(@as(u32, 15), hq.stage1_steps_default);
    const hq_stg = guiderDefaults(.two_stage_hq, null, null, 1.0);
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
    try std.testing.expectEqual(@as(usize, 1024), PAD_LEN);
    const registers: usize = 128; // connector.*.learnable_registers rows
    try std.testing.expectEqual(@as(usize, 0), PAD_LEN % registers);
    try std.testing.expectEqual(@as(usize, 8), PAD_LEN / registers);
}

test "LTX negative prompt keeps the reference audio negatives (speech guidance)" {
    // The audio tail of the reference DEFAULT_NEGATIVE_PROMPT does real work
    // for dialogue: with audio CFG active (two-stage, ap.cfg=7.0) these terms
    // push the soundtrack away from ambient noise toward clean speech. A
    // trimmed copy that keeps only the visual head silently weakens speech.
    // Overflow is safe: padWithBos left-truncates, keeping this tail.
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
        try testing.expect(std.mem.indexOf(u8, NEGATIVE_PROMPT, term) != null);
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
        try testing.expect(std.mem.indexOf(u8, NEGATIVE_PROMPT, term) != null);
    }
    const first_audio = std.mem.indexOf(u8, NEGATIVE_PROMPT, "mismatched lip sync").?;
    for (subtitle_negatives) |term| {
        try testing.expect(std.mem.indexOf(u8, NEGATIVE_PROMPT, term).? < first_audio);
    }
}
