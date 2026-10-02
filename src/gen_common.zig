const log = @import("log.zig");
const std = @import("std");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const chat_mod = @import("chat.zig");
const discovery = @import("model_discovery.zig");
const multipart = @import("multipart.zig");

pub const Modality = enum {
    image,
    audio,
    video,
    mesh,
    /// Typed decisions (`/v1/decisions`, Laya or Kev): scorers, not generators,
    /// but they ride the media plumbing (engine slot, inference-thread job).
    decision,

    pub fn capability(self: Modality) []const u8 {
        return switch (self) {
            .image => "image",
            .audio => "audio",
            .video => "video",
            .mesh => "3d",
            .decision => "decisions",
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
            .decision => "laya",
        };
    }
};

pub const media_model_types = [_][]const u8{
    "flux2",     "krea",       "mage_flow",      "mageflow",
    "qwen3_tts", "acestep",    "kokoro",         "AudioVideo",
    "hunyuan3d", "minimax_h3", "minimax_music3", "qwen_image",
    "laya",      "kev",
};

pub fn modalityFromType(model_type: []const u8) ?Modality {
    if (std.mem.startsWith(u8, model_type, "flux2")) return .image;
    if (std.mem.startsWith(u8, model_type, "krea")) return .image;
    if (std.mem.startsWith(u8, model_type, "mage_flow") or std.mem.eql(u8, model_type, "mageflow")) return .image;
    if (std.mem.startsWith(u8, model_type, "qwen_image")) return .image;
    if (std.mem.eql(u8, model_type, "qwen3_tts")) return .audio;
    if (std.mem.eql(u8, model_type, "acestep")) return .audio;
    if (std.mem.eql(u8, model_type, "minimax_music3")) return .audio;
    if (std.mem.eql(u8, model_type, "kokoro")) return .audio;
    if (std.mem.eql(u8, model_type, "AudioVideo")) return .video;
    if (std.mem.eql(u8, model_type, "minimax_h3")) return .video;
    if (std.mem.startsWith(u8, model_type, "hunyuan3d")) return .mesh;
    if (std.mem.eql(u8, model_type, "laya") or std.mem.eql(u8, model_type, "kev")) return .decision;
    return null;
}

pub const GenRoute = enum {
    image,
    speech,
    music,
    video,
    mesh,
    decisions,

    pub fn modality(self: GenRoute) Modality {
        return switch (self) {
            .image => .image,
            .speech, .music => .audio,
            .video => .video,
            .mesh => .mesh,
            .decisions => .decision,
        };
    }
};

pub fn audioBackendKindForType(model_type: []const u8) AudioBackendKind {
    if (std.mem.eql(u8, model_type, "acestep")) return .music;
    if (std.mem.eql(u8, model_type, "minimax_music3")) return .music3;
    if (std.mem.eql(u8, model_type, "kokoro")) return .kokoro;
    return .tts;
}

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

pub fn peekModelType(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    // Guard the openFileAbsolute assert (ReleaseFast UB on relative/empty paths).
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null;
    // A Kev pack's root config.json is its qwen3_5 base: the marker is checked first (discovery agrees).
    if (isKevPack(io, model_dir)) return allocator.dupe(u8, "kev") catch null;
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
    // Laya decision checkpoints carry no root config.json either.
    if (isLayaRepo(io, model_dir)) return allocator.dupe(u8, "laya") catch null;
    // An mlx-community-style Qwen-Image-2.1 repo likewise: no root
    // config.json, model_index.json's `_class_name` the only marker.
    if (isQwenImage21Repo(io, allocator, model_dir)) return allocator.dupe(u8, "qwen_image21") catch null;
    return null;
}

pub fn isKevPack(io: std.Io, model_dir: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return discovery.peekKevPack(io, dir);
}

pub fn isLayaRepo(io: std.Io, model_dir: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return discovery.peekLayaCheckpoint(io, dir);
}

pub fn isQwenImage21Repo(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return discovery.peekQwenImage21Index(io, allocator, dir);
}

pub fn isMfluxFlux2Repo(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return discovery.peekMfluxFlux2(io, allocator, dir);
}

pub fn readConfigModelType(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
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

pub fn isMageFlowRepo(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
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

pub fn incompleteMediaDir(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const mt = peekModelType(io, allocator, model_dir) orelse return false;
    defer allocator.free(mt);
    if (modalityFromType(mt) == null) return false;
    const marker = requiredMarkerFor(mt) orelse return false;
    const p = std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, marker }, 0) catch return false;
    defer allocator.free(p);
    return !fileExists(io, p);
}

pub const StubCpuState = struct {
    config: *model_mod.ModelConfig,
    tok: *tok_mod.Tokenizer,
    chat_config: *chat_mod.ChatConfig,
};

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

pub fn estimateResidentBytes(io: std.Io, model_dir: []const u8) u64 {
    if (model_dir.len == 0 or model_dir[0] != '/') return 0; // openDirAbsolute UB class
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    return sumSafetensorsIn(io, dir);
}

pub fn sumSafetensorsIn(io: std.Io, dir: std.Io.Dir) u64 {
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

pub fn stagedPeakBytes(resident: u64, stages: []const u64) u64 {
    var biggest: u64 = 0;
    for (stages) |st| biggest = @max(biggest, st);
    return resident + biggest;
}

pub const QWEN_IMAGE_GEN_TRANSIENT_BYTES: u64 = 4 << 30;

pub fn qwenImageStagesTextEncoder(weights: u64, working_set: u64) bool {
    const low_memory: ?[]const u8 = if (std.c.getenv("MLX_SERVE_QWEN_IMAGE_LOW_MEMORY")) |v| std.mem.span(v) else null;
    return qwenImageStagesTextEncoderFromInputs(weights, working_set, low_memory);
}

pub fn qwenImageStagesTextEncoderFromInputs(weights: u64, working_set: u64, low_memory: ?[]const u8) bool {
    if (low_memory) |value| {
        if (std.mem.eql(u8, value, "1")) return true;
    }
    if (working_set == 0) return false;
    return (weights + QWEN_IMAGE_GEN_TRANSIENT_BYTES) / 3 > working_set / 4;
}

pub fn qwenImagePeakBytes(text_encoder: u64, rest: u64, working_set: u64) u64 {
    return qwenImagePeakBytesForStaging(text_encoder, rest, qwenImageStagesTextEncoder(text_encoder + rest, working_set));
}

pub fn qwenImagePeakBytesForStaging(text_encoder: u64, rest: u64, staged: bool) u64 {
    // The DiT/VAE remain resident while encoding: staging only removes the
    // encoder before denoising, so the full weight set still counts at peak.
    if (staged)
        return stagedPeakBytes(rest, &.{ text_encoder, QWEN_IMAGE_GEN_TRANSIENT_BYTES });
    return text_encoder + rest + QWEN_IMAGE_GEN_TRANSIENT_BYTES;
}

pub const QWEN_IMAGE_EDIT_TRANSIENT_BYTES: u64 = 6 << 30;

pub const QWEN_IMAGE_EDIT_ENCODE_WS_BYTES: u64 = 2 << 30;

pub const QWEN_IMAGE_EDIT_TEXT_TOKENS: u64 = 2600; // ti2i prompt budget upper bound

pub const QWEN_IMAGE_EDIT_HEADS: u64 = 32; // the checkpoint's num_attention_heads

pub fn qwenImageEditTransientBytes(refs: u32, ref_resolution: u32, out_w: u32, out_h: u32) u64 {
    const rt: u64 = @as(u64, ref_resolution / 16) * (ref_resolution / 16);
    const ot: u64 = @as(u64, out_w / 16) * (out_h / 16);
    const joint: u64 = @as(u64, refs) * rt + ot + QWEN_IMAGE_EDIT_TEXT_TOKENS;
    const widest_q: u64 = @max(ot, rt);
    const scores: u64 = QWEN_IMAGE_EDIT_HEADS * widest_q * joint * 4;
    const persistent: u64 = @as(u64, refs) * rt * (65 << 20) / 4096;
    return scores + persistent;
}

pub fn qwenImageEditPeakBytes(text_encoder: u64, rest: u64, working_set: u64, has_tower: bool) u64 {
    const transient: u64 = if (has_tower) QWEN_IMAGE_EDIT_TRANSIENT_BYTES else QWEN_IMAGE_GEN_TRANSIENT_BYTES;
    if (qwenImageStagesTextEncoder(text_encoder + rest, working_set)) {
        const te_stage: u64 = if (has_tower) text_encoder +| QWEN_IMAGE_EDIT_ENCODE_WS_BYTES else text_encoder;
        return stagedPeakBytes(rest, &.{ te_stage, transient });
    }
    return text_encoder + rest + transient;
}

pub fn ltxPeakBytes(dir_sum: u64, spare_transformer: u64, text_encoder: u64) u64 {
    return stagedPeakBytes(dir_sum -| spare_transformer, &.{text_encoder});
}

pub const H3_DIT_RESIDENT_PCT: u64 = 65;

pub const H3_ACTIVATION_BYTES: u64 = 6 * 1024 * 1024 * 1024;

pub const MUSIC3_GEN_BUFFER_BYTES: u64 = 6 * 1024 * 1024 * 1024;

pub fn h3DitResidentBytes(dit_file: u64, precompute: bool) u64 {
    if (!precompute) return dit_file;
    return dit_file * H3_DIT_RESIDENT_PCT / 100;
}

pub fn h3PeakBytes(te: u64, dit_resident: u64, video_vae: u64, audio_vae: u64) u64 {
    const vaes = video_vae + audio_vae;
    const generating = @max(dit_resident, vaes);
    if (te == 0 and generating == 0) return 0; // unknown dir → never block
    return stagedPeakBytes(0, &.{ te, generating + H3_ACTIVATION_BYTES });
}

pub fn estimatePeakResidentBytesIn(io: std.Io, dir: std.Io.Dir, model_type: []const u8) u64 {
    return estimatePeakResidentBytesInDir(io, dir, model_type, null, .{});
}

pub fn estimatePeakResidentBytesInDir(io: std.Io, dir: std.Io.Dir, model_type: []const u8, model_dir: ?[]const u8, backend: PeakBackend) u64 {
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
            backend.adaln_precompute(),
        ) + sz(io, dir, "turbo_lora.safetensors");
        return h3PeakBytes(
            sz(io, dir, "text_encoder.safetensors"),
            dit,
            sz(io, dir, "video_vae.safetensors"),
            sz(io, dir, "audio_vae.safetensors"),
        );
    }
    if (std.mem.startsWith(u8, model_type, "qwen_image")) {
        var te_dir = dir.openDir(io, "text_encoder", .{ .iterate = true }) catch return sumSafetensorsIn(io, dir);
        defer te_dir.close(io);
        const te = sumSafetensorsIn(io, te_dir);
        const rest = sumSafetensorsIn(io, dir) -| te;
        // A tower in text_encoder/ means the pack edits — bill the heavier
        // edit transient (the tower's own bytes already ride `te`). Whatever
        // `towerPresentIn` answers, the engine's `has_tower` reads it too.
        const tower = if (model_dir) |md| backend.tower_present(io, std.heap.page_allocator, md) else false;
        return if (tower) qwenImageEditPeakBytes(te, rest, backend.working_set(), true) else qwenImagePeakBytes(te, rest, backend.working_set());
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

pub fn sumSafetensorsAt(io: std.Io, path: []const u8) u64 {
    if (path.len == 0 or path[0] != '/') return 0; // openDirAbsolute UB class
    var d = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    return sumSafetensorsIn(io, d);
}

pub fn ltxTextEncoderBytes(io: std.Io) u64 {
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
    return estimatePeakResidentBytesWithBackend(io, model_dir, model_type, .{});
}
pub fn estimatePeakResidentBytesWithBackend(io: std.Io, model_dir: []const u8, model_type: []const u8, backend: PeakBackend) u64 {
    if (model_dir.len == 0 or model_dir[0] != '/') return 0; // openDirAbsolute UB class
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    const in_dir = estimatePeakResidentBytesInDir(io, dir, model_type, model_dir, backend);
    // LTX reads its text encoder from a DIFFERENT repo, so it is a stage the
    // model dir cannot see. Every other backend's weights are all in its own
    // directory; if that stops being true, it belongs here beside this one.
    if (in_dir > 0 and std.mem.eql(u8, model_type, "AudioVideo"))
        return stagedPeakBytes(in_dir, &.{ltxTextEncoderBytes(io)});
    return in_dir;
}

pub const EditFormError = error{
    NotMultipart,
    MissingPrompt,
    MissingImage,
    TooManyImages,
    MaskUnsupported,
    MultipleChoicesUnsupported,
    UrlResponseUnsupported,
    OutputFormatUnsupported,
    MalformedNumber,
    StreamUnsupported,
    OutOfMemory,
};

pub fn editFormErrorMessage(err: EditFormError) []const u8 {
    return switch (err) {
        error.NotMultipart => "/v1/images/edits expects multipart/form-data with a 'boundary' parameter",
        error.MissingPrompt => "missing required form field 'prompt'",
        error.MissingImage => "missing required form field 'image' (the picture to edit)",
        error.TooManyImages => "too many 'image' parts (at most 10: the edited source plus 9 references)",
        error.MaskUnsupported => "'mask' is not supported: this server's editors are maskless in-context models (describe the change in the prompt instead)",
        error.MultipleChoicesUnsupported => "'n' must be 1 — this engine generates a single image per request",
        error.UrlResponseUnsupported => "'response_format' must be 'b64_json' — this server does not host generated files",
        error.OutputFormatUnsupported => "'output_format' must be 'png' — this server always returns PNG",
        error.StreamUnsupported => "'stream' is not supported on /v1/images/edits (use /v1/images/generations for SSE progress)",
        error.MalformedNumber => "'steps'/'seed'/'guidance_scale'/'ref_resolution' must be a JSON number (they are spliced into the request body verbatim)",
        error.OutOfMemory => "out of memory",
    };
}

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
    var ref_resolution: ?[]const u8 = null;
    var steps: ?[]const u8 = null;
    var seed: ?[]const u8 = null;
    var guidance_scale: ?[]const u8 = null;
    var negative_prompt: ?[]const u8 = null;

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
        } else if (std.mem.eql(u8, part.name, "ref_resolution")) {
            if (part.data.len != 0) ref_resolution = part.data;
        } else if (std.mem.eql(u8, part.name, "steps")) {
            if (part.data.len != 0) steps = part.data;
        } else if (std.mem.eql(u8, part.name, "seed")) {
            if (part.data.len != 0) seed = part.data;
        } else if (std.mem.eql(u8, part.name, "guidance_scale")) {
            if (part.data.len != 0) guidance_scale = part.data;
        } else if (std.mem.eql(u8, part.name, "negative_prompt")) {
            if (part.data.len != 0) negative_prompt = part.data;
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
    if (ref_resolution) |rr| {
        if (!chat_mod.isJsonNumber(rr)) return error.MalformedNumber;
        try out.appendSlice(allocator, ",\"ref_resolution\":");
        try out.appendSlice(allocator, rr);
    }
    // Sampling knobs: each numeric value is validated as JSON grammar before
    // splicing — a value like `8,"stream":true` must die HERE as a named 400,
    // not smuggle extra fields into the body the JSON handler would honor.
    if (steps) |v| {
        if (!chat_mod.isJsonNumber(v)) return error.MalformedNumber;
        try out.appendSlice(allocator, ",\"steps\":");
        try out.appendSlice(allocator, v);
    }
    if (seed) |v| {
        if (!chat_mod.isJsonNumber(v)) return error.MalformedNumber;
        try out.appendSlice(allocator, ",\"seed\":");
        try out.appendSlice(allocator, v);
    }
    if (guidance_scale) |v| {
        if (!chat_mod.isJsonNumber(v)) return error.MalformedNumber;
        try out.appendSlice(allocator, ",\"guidance_scale\":");
        try out.appendSlice(allocator, v);
    }
    if (negative_prompt) |v| {
        try out.appendSlice(allocator, ",\"negative_prompt\":");
        try chat_mod.appendJsonString(allocator, &out, v);
    }
    try out.appendSlice(allocator, "}");
    return out.toOwnedSlice(allocator);
}

pub fn appendBase64(allocator: std.mem.Allocator, out: *std.ArrayList(u8), bytes: []const u8) !void {
    const n = std.base64.standard.Encoder.calcSize(bytes.len);
    const start = out.items.len;
    try out.resize(allocator, start + n);
    _ = std.base64.standard.Encoder.encode(out.items[start..], bytes);
}

pub const MAX_EDIT_IMAGES = 10;

pub fn fileExists(io: std.Io, path: [:0]const u8) bool {
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

pub const LTX_GEMMA_REPO_DIR = "mlx-community/gemma-3-12b-it-4bit";

pub const PeakBackend = struct {
    working_set: *const fn () u64 = noWorkingSet,
    tower_present: *const fn (std.Io, std.mem.Allocator, []const u8) bool = noTower,
    adaln_precompute: *const fn () bool = noAdalnPrecompute,

    fn noWorkingSet() u64 {
        return 0;
    }
    fn noTower(_: std.Io, _: std.mem.Allocator, _: []const u8) bool {
        return false;
    }
    fn noAdalnPrecompute() bool {
        return false;
    }
};
