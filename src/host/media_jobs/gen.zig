//! Host side of media generation: modality detection (`peekModelType`, `detectModality`), the media load bills,
//! the stub CPU state of a media entry, and the response writers and request parsers the media plugin driver
//! (`plugin/`) shares. Every media backend is a plugin (`LoadedModel.media_plugin`); its jobs run on the
//! scheduler's INFERENCE thread (the sole mlx caller), so SSE writes to the parked connection are single-writer-safe.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const w8a8 = @import("../../sdk/blocks/dit/w8a8.zig");
const low_mem = @import("low_mem.zig");
const lora_mod = @import("../../sdk/weights/lora.zig");
const ltx_bill = @import("../../plugins/ltx/config.zig");
const minimax_h3 = @import("../../plugins/minimax_h3/h3.zig");
const ane = @import("../../sdk/ane/offload.zig");
const plugin_job = @import("../../sdk/media/job.zig");
const png_mod = @import("../../sdk/media/png.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../sdk/config/model.zig");
const chat_mod = @import("../chat/chat.zig");
const log = @import("../../sdk/util/log.zig");
const metrics = @import("../status.zig");
const sse = @import("sse.zig");
const server_mod = @import("../http/server.zig");
const multipart = @import("../http/multipart.zig");
const discovery = @import("../registry/discovery.zig");
const plugin_served = @import("plugin/served.zig");
const plugin_engine = @import("plugin/engine.zig");
const stb = @import("stb");
const webp = @import("webp");

const Conn = server_mod.Conn;

const modality_mod = @import("../registry/modality.zig");
pub const Modality = modality_mod.Modality;
pub const modalityFromType = modality_mod.fromType;

/// Endpoint-level media route. `.speech`, `.music` and `.sound` share the `.audio`
/// modality slot — the loaded plugin's caps decide which endpoint is valid
/// (wrong pairing → explicit 400, never a silent misinterpretation).
pub const GenRoute = enum {
    image,
    speech,
    music,
    sound,
    video,
    mesh,
    decisions,

    pub fn modality(self: GenRoute) Modality {
        return switch (self) {
            .image => .image,
            .speech, .music, .sound => .audio,
            .video => .video,
            .mesh => .mesh,
            .decisions => .decision,
        };
    }
};

/// Whether the audio pack at `model_dir` (of `model_type`) serves the "music" or "sound" route: its plugin's caps.
pub fn servesAudioRoute(io: std.Io, model_dir: []const u8, model_type: []const u8, comptime route: enum { music, sound }) bool {
    const t = plugin_served.byName(model_type) orelse return false;
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return false; // openDirAbsolute UB class
    var d = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer d.close(io);
    return @field(plugin_engine.caps(io, d, t).routes, @tagName(route));
}

/// The pack's `model_type` (owned dupe, caller frees), or null on any read/parse error: the media plugin's that claims
/// the folder, else config.json's. Cheap — used to route to a media modality and to pick a legacy backend.
pub fn peekModelType(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    // Guard the openFileAbsolute assert (ReleaseFast UB on relative/empty paths).
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null;
    if (std.Io.Dir.openDirAbsolute(io, model_dir, .{})) |d_| {
        var d = d_;
        defer d.close(io);
        if (plugin_served.claimTypeIn(io, d, allocator)) |mt| return mt;
    } else |_| {}
    return readConfigModelType(io, allocator, model_dir);
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

/// Classify a model dir into a media modality (its media plugin's), or null for a regular LM/embedding arch or an
/// incomplete media pack (its plugin's required file missing).
pub fn detectModality(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?Modality {
    const mt = peekModelType(io, allocator, model_dir) orelse return null;
    defer allocator.free(mt);
    const modality = modalityFromType(mt) orelse return null;
    if (discovery.requiredMediaMarker(mt)) |marker| {
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
/// first missing weight (`unreachable` in ReleaseFast — live, a
/// turbo-lora fragment killed the server on a plain Generate).
pub fn incompleteMediaDir(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const mt = peekModelType(io, allocator, model_dir) orelse return false;
    defer allocator.free(mt);
    if (modalityFromType(mt) == null) return false;
    const marker = discovery.requiredMediaMarker(mt) orelse return false;
    const p = std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, marker }, 0) catch return false;
    defer allocator.free(p);
    return !fileExists(io, p);
}

// ════════════════════════════════════════════════════════════════════════
// Engine wrappers — own the backend sub-models. Allocated on the heap so the
// `?*Engine` slot on `LoadedModel` is a stable pointer (mirrors `engine`).
// load() + every generate() run on the inference thread.
// ════════════════════════════════════════════════════════════════════════

/// Per-request bounds for every decision backend, checked before any forward: one request
/// holds the inference thread until it is answered.
pub const DecisionLimits = struct {
    max_questions: usize = 64,
    max_input_tokens: usize = 32 * 1024,

    fn fromEnv() DecisionLimits {
        const d: DecisionLimits = .{};
        return .{
            .max_questions = envLimit(.MLX_SERVE_LAYA_MAX_QUESTIONS, d.max_questions),
            .max_input_tokens = envLimit(.MLX_SERVE_LAYA_MAX_INPUT_TOKENS, d.max_input_tokens),
        };
    }

    fn envLimit(comptime name: @EnumLiteral(), default: usize) usize {
        const raw = env.get(name) orelse return default;
        const v = std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        if (v == 0) {
            log.warn("[decision] ignoring {s}={s} (want a positive integer)\n", .{ @tagName(name), raw });
            return default;
        }
        return v;
    }
};

/// How long the inference thread waits for more decision requests to answer in one pass
/// (`MLX_SERVE_LAYA_BATCH_WINDOW_US`, default 0: only requests already queued are merged).
pub fn decisionBatchWindowUs() u32 {
    const raw = env.get(.MLX_SERVE_LAYA_BATCH_WINDOW_US) orelse return 0;
    return std.fmt.parseInt(u32, std.mem.sliceTo(raw, 0), 10) catch {
        log.warn("[decision] ignoring MLX_SERVE_LAYA_BATCH_WINDOW_US={s} (want microseconds)\n", .{raw});
        return 0;
    };
}

/// A decision plugin's `LoadCtx.settings_json`: the request limits in force and the merge window.
pub fn decisionSettingsJson(buf: []u8, batch_window_us: u32) []const u8 {
    const l = DecisionLimits.fromEnv();
    return std.fmt.bufPrint(buf, "{{\"max_questions\":{d},\"max_input_tokens\":{d},\"batch_window_us\":{d}}}", .{ l.max_questions, l.max_input_tokens, batch_window_us }) catch "";
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
    MalformedNumber,
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

/// Rewrite an OpenAI images-edit form into the `/v1/images/generations` body.
/// Returns owned JSON (caller frees). Files are re-encoded as base64 (the
/// internal schema's transport); everything OpenAI accepts but we can't honor
/// is an explicit error rather than a silently ignored field.
/// Most images one edit form carries (Qwen-Image-2.1's reference cap, the largest; each plugin checks its own).
pub const MAX_EDIT_IMAGES = 10;

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

// ════════════════════════════════════════════════════════════════════════
// HTTP handler bodies. Called on the INFERENCE thread (via the gen job). The
// connection is parked (single-writer), so SSE writes here are safe. `lm` is
// already resolved + refcounted by the connection thread.
// ════════════════════════════════════════════════════════════════════════

/// A generated PNG as OpenAI's `b64_json` body, or the SSE `complete` event.
pub fn sendPngResult(allocator: std.mem.Allocator, conn: *Conn, png_bytes: []const u8, want_stream: bool) !void {
    return sendB64Result(allocator, conn, png_bytes, want_stream, if (want_stream) "data: {\"type\":\"complete\",\"data\":[{\"b64_json\":\"" else "{\"created\":0,\"data\":[{\"b64_json\":\"", if (want_stream) "\"}]}\n\n" else "\"}]}");
}

/// A generated GLB as base64 JSON, or the SSE `complete` event.
pub fn sendGlbResult(allocator: std.mem.Allocator, conn: *Conn, glb_bytes: []const u8, want_stream: bool) !void {
    return sendB64Result(allocator, conn, glb_bytes, want_stream, if (want_stream) "data: {\"type\":\"complete\",\"format\":\"glb\",\"data\":\"" else "{\"created\":0,\"format\":\"glb\",\"data\":\"", if (want_stream) "\"}\n\n" else "\"}");
}

/// `bytes` base64-encoded between `head` and `tail`: an SSE event as written, else a JSON response.
fn sendB64Result(allocator: std.mem.Allocator, conn: *Conn, bytes: []const u8, want_stream: bool, head: []const u8, tail: []const u8) !void {
    const b64 = try allocator.alloc(u8, std.base64.standard.Encoder.calcSize(bytes.len));
    defer allocator.free(b64);
    _ = std.base64.standard.Encoder.encode(b64, bytes);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, head);
    try out.appendSlice(allocator, b64);
    try out.appendSlice(allocator, tail);
    if (want_stream) return conn.writeAll(out.items);
    return sendBytesJson(conn, allocator, out.items);
}

pub const MUSIC_ROUTE_HINT = "loaded audio model is a music generator; POST /v1/audio/music-generations";
pub const TTS_ROUTE_HINT = "loaded audio model is a TTS voice; POST /v1/audio/speech";
pub const SOUND_ROUTE_HINT = "loaded audio model is a text-to-audio generator; POST /v1/audio/sound-generations";

/// `extra` is a JSON fragment (`,"key":value`) spliced into the streaming `complete` event.
pub fn sendWavWith(allocator: std.mem.Allocator, conn: *Conn, wav: []const u8, want_stream: bool, extra: []const u8) !void {
    if (want_stream) {
        const b64_len = std.base64.standard.Encoder.calcSize(wav.len);
        const b64 = try allocator.alloc(u8, b64_len);
        defer allocator.free(b64);
        _ = std.base64.standard.Encoder.encode(b64, wav);
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(allocator);
        try out.appendSlice(allocator, "data: {\"type\":\"complete\",\"format\":\"wav\"");
        try out.appendSlice(allocator, extra);
        try out.appendSlice(allocator, ",\"data\":\"");
        try out.appendSlice(allocator, b64);
        try out.appendSlice(allocator, "\"}\n\n");
        try conn.writeAll(out.items);
        return;
    }
    return sendBytes(conn, allocator, "audio/wav", wav);
}

/// A result given as JSON members (`"frames":…`): the body `{"created":0,…}`, or the SSE `complete` event.
pub fn sendJsonMembers(allocator: std.mem.Allocator, conn: *Conn, members: []const u8, want_stream: bool) !void {
    if (want_stream) {
        try conn.writeAllNoFlush("data: {\"type\":\"complete\",");
        try conn.writeAllNoFlush(members);
        return conn.writeAll("}\n\n");
    }
    return sendJsonPieces(conn, allocator, &.{ "{\"created\":0,", members, "}" });
}

/// SSE sink for a video request. Preview JPEGs are opt-in and require stream.
pub fn videoStreamCtx(conn: *Conn, allocator: std.mem.Allocator, body: []const u8, want_stream: bool) sse.StreamCtx {
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

// ════════════════════════════════════════════════════════════════════════
// Stub CPU state for a media model. The gen path bypasses the transformer, so
// `config`/`tokenizer`/`chat_config` on the LoadedModel are minimal stubs that
// only keep server-side reads of `lm.config.?` / `lm.chat_config.?` from
// crashing. Mirrors the GGUF stub (`cpu_state.buildGgufStubCpuState`). Used by BOTH the
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
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return 0; // openDirAbsolute UB class
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

const h3_bill = @import("../../plugins/minimax_h3/config.zig");

/// Frees every H3 engine's resident cache; returns the bytes released. For a load that would
/// otherwise be refused for memory.
pub fn releaseMediaResidency() u64 {
    return minimax_h3.releaseAllResidents();
}

/// The generation peak a media load preflight bills: the media plugin's load bill, else the pack's safetensors —
/// over-billing fails safe (a refused load names its numbers), under-billing kills the process mid-request.
pub fn estimatePeakResidentBytesIn(io: std.Io, dir: std.Io.Dir, model_type: []const u8) u64 {
    if (plugin_served.byName(model_type)) |t| return plugin_engine.billLoad(io, dir, t);
    return sumSafetensorsIn(io, dir);
}

/// Sum of the `.safetensors` under an absolute path, or 0 if it is not
/// readable — 0 means "unknown", which every caller treats as "do not block".
pub fn sumSafetensorsAt(io: std.Io, path: []const u8) u64 {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return 0; // openDirAbsolute UB class
    var d = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    return sumSafetensorsIn(io, d);
}

pub fn estimatePeakResidentBytes(io: std.Io, model_dir: []const u8, model_type: []const u8) u64 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return 0; // openDirAbsolute UB class
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    return estimatePeakResidentBytesIn(io, dir, model_type);
}

// ── HTTP response helpers (self-contained; mirror the old *_server.zig) ──

pub fn sendBytesJson(conn: *Conn, allocator: std.mem.Allocator, json: []const u8) !void {
    return sendJsonPieces(conn, allocator, &.{json});
}

/// One JSON response whose body is `pieces` back to back (a large payload is never copied into one buffer).
fn sendJsonPieces(conn: *Conn, allocator: std.mem.Allocator, pieces: []const []const u8) !void {
    var len: usize = 0;
    for (pieces) |p| len += p.len;
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(allocator);
    try hdr.appendSlice(allocator, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
    var num: [20]u8 = undefined;
    const ns = std.fmt.bufPrint(&num, "{d}", .{len}) catch unreachable;
    try hdr.appendSlice(allocator, ns);
    try hdr.appendSlice(allocator, "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n");
    try conn.writeAllNoFlush(hdr.items);
    conn.length_framed = true;
    for (pieces[0 .. pieces.len - 1]) |p| try conn.writeAllNoFlush(p);
    try conn.writeAll(pieces[pieces.len - 1]);
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
    conn.length_framed = true;
    try conn.writeAll(payload);
}

pub fn sendError(conn: *Conn, code: u16, msg: []const u8) !void {
    // A refusal that logs nothing is invisible the moment a client drops the
    // body (the app's stream path showed a bare "HTTP 400"
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
    const head = std.fmt.bufPrint(&hdr, "HTTP/1.1 {d} Error\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", .{ code, body.len }) catch return;
    try conn.writeAllNoFlush(head);
    conn.length_framed = true;
    try conn.writeAll(body);
}

/// A media or decision job that ran out of memory hit the machine's capacity, not a server fault: 503, like chat.
pub fn jobFailureStatus(err: anyerror) u16 {
    if (err == error.OutOfMemory or mlx.gpuContextLost()) return 503;
    const latched = mlx.peekErrorName() orelse return 500;
    return if (std.mem.eql(u8, latched, "OutOfMemory")) 503 else 500;
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

/// A request's step count: `default` when omitted, else held to [lo, hi] BEFORE any
/// cast, so a client integer never reaches a narrowing cast or a schedule unchecked.
fn stepsRequest(requested: ?u64, default: u32, lo: u32, hi: u32) error{StepsOutOfRange}!u32 {
    const n = requested orelse return default;
    if (n < lo or n > hi) return error.StepsOutOfRange;
    return @intCast(n);
}

test "stepsRequest: omitted takes the default, the range is checked before the cast" {
    try std.testing.expectEqual(@as(u32, 30), try stepsRequest(null, 30, 1, 100));
    try std.testing.expectEqual(@as(u32, 1), try stepsRequest(1, 30, 1, 100));
    try std.testing.expectEqual(@as(u32, 100), try stepsRequest(100, 30, 1, 100));
    try std.testing.expectError(error.StepsOutOfRange, stepsRequest(0, 30, 1, 100));
    try std.testing.expectError(error.StepsOutOfRange, stepsRequest(101, 30, 1, 100));
    try std.testing.expectError(error.StepsOutOfRange, stepsRequest(5_000_000_000, 30, 1, 100));
    // A PDD student's range is its trained count alone.
    try std.testing.expectEqual(@as(u32, 8), try stepsRequest(null, 8, 8, 8));
    try std.testing.expectEqual(@as(u32, 8), try stepsRequest(8, 8, 8, 8));
    try std.testing.expectError(error.StepsOutOfRange, stepsRequest(4, 8, 8, 8));
}

/// `key` is in the body with a value other than `null` (clients send null for an unset field).
fn jsonFieldSet(body: []const u8, key: []const u8) bool {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return false;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return false;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t' or body[i] == '\n' or body[i] == '\r')) i += 1;
    return !std.mem.startsWith(u8, body[i..], "null");
}

test "jsonFieldSet: a null value is an unset field" {
    try std.testing.expect(jsonFieldSet("{\"ref_images\":[\"a\"]}", "ref_images"));
    try std.testing.expect(!jsonFieldSet("{\"ref_images\": null}", "ref_images"));
    try std.testing.expect(!jsonFieldSet("{\"prompt\":\"x\"}", "ref_images"));
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

/// `extractJsonInt` saturated to u32: a value past the range reaches the caller's bound check
/// as the largest u32 (a named 400) instead of an out-of-range cast.
fn extractJsonU32(body: []const u8, key: []const u8) ?u32 {
    const v = extractJsonInt(body, key) orelse return null;
    return std.math.cast(u32, v) orelse std.math.maxInt(u32);
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

/// Per-adapter scales for `lora_scales` (multi-LoRA counterpart of the
/// single `lora_scale` float field).
fn extractLoraScales(body: []const u8, buf: []f32) ?[]f32 {
    return extractFloatArrayField(body, "lora_scales", buf);
}

const LoraFieldsError = error{ TooManyLoraPaths, BadLoraPathsJson, BadLoraScalesJson, LoraPathsLocalOnly, OutOfMemory };

/// A LoRA path names a file on the server's disk, so only a client on this
/// machine may send one: from anyone else, "missing" vs "not a LoRA" answers
/// whether a host path exists (#540).
pub const LORA_LOCAL_ONLY_MSG = "'lora_paths' name files on the server's disk and are accepted only from the server's own machine";

/// Parse the LoRA fields common to image and video requests: the array form
/// (`lora_paths` + optional `lora_scales`) or the original single-adapter
/// form (`lora_path` + optional `lora_scale`), which is kept exactly
/// backward-compatible. Writes up to `lora_mod.MAX_LORAS` unescaped,
/// allocator-owned path strings into `path_bufs` (caller frees them) and
/// their resolved scales into `scale_buf`. Returns 0 with both buffers
/// untouched when neither field is present — the "detach whatever was
/// attached" case. Missing `lora_scales` entries default to 1.0, matching
/// mflux's `resolve_scales`. `local`: the client is on this machine.
pub fn parseLoraFields(
    allocator: std.mem.Allocator,
    body: []const u8,
    path_bufs: *[lora_mod.MAX_LORAS][]u8,
    scale_buf: *[lora_mod.MAX_LORAS]f32,
    local: bool,
) LoraFieldsError!usize {
    var n: usize = 0;
    errdefer for (path_bufs[0..n]) |p| allocator.free(p);

    if (jsonFieldSet(body, "lora_paths")) {
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
    if (!local) return error.LoraPathsLocalOnly;

    if (jsonFieldSet(body, "lora_scales")) {
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

/// Base64-decode an image for the media backends, which all read through
/// stb_image: a WebP comes back re-encoded as a lossless PNG (alpha kept),
/// anything else exactly as sent.
pub const imageB64Alloc = plugin_job.imageUpload;

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

test "d1 is a decision modality, routed by the model_type discovery gives it" {
    try testing.expectEqual(Modality.decision, modalityFromType("d1").?);
}

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
    try testing.expectEqual(Modality.image, modalityFromType("qwen_image21").?);
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

    // ref_resolution (the qwen speed knob) rides through the form too.
    const rr = "--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"ref_resolution\"\r\n\r\n512\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nAAA\r\n--X--\r\n";
    const j5 = try openaiEditFormToJson(a, rr, CT);
    defer a.free(j5);
    var p5 = try std.json.parseFromSlice(std.json.Value, a, j5, .{});
    defer p5.deinit();
    try testing.expectEqual(@as(i64, 512), p5.value.object.get("ref_resolution").?.integer);

    // Sampling knobs ride through too (dropping them silently ran every
    // multipart edit at the 40-step default).
    const knobs = "--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"steps\"\r\n\r\n15\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n7\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"guidance_scale\"\r\n\r\n2.5\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"negative_prompt\"\r\n\r\nblurry\r\n" ++
        "--X\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\nAAA\r\n--X--\r\n";
    const j6 = try openaiEditFormToJson(a, knobs, CT);
    defer a.free(j6);
    var p6 = try std.json.parseFromSlice(std.json.Value, a, j6, .{});
    defer p6.deinit();
    try testing.expectEqual(@as(i64, 15), p6.value.object.get("steps").?.integer);
    try testing.expectEqual(@as(i64, 7), p6.value.object.get("seed").?.integer);
    try testing.expectEqual(@as(f64, 2.5), p6.value.object.get("guidance_scale").?.float);
    try testing.expectEqualStrings("blurry", p6.value.object.get("negative_prompt").?.string);
}

test "parseLoraFields: a LoRA path is accepted only from this machine (#540)" {
    const a = testing.allocator;
    var paths: [lora_mod.MAX_LORAS][]u8 = undefined;
    var scales: [lora_mod.MAX_LORAS]f32 = undefined;
    const bodies = [_][]const u8{
        "{\"lora_paths\":[\"/etc/hosts\",\"/x.safetensors\"],\"lora_scales\":[0.5]}",
        "{\"lora_path\":\"/etc/hosts\"}",
    };
    for (bodies) |body| {
        try testing.expectError(error.LoraPathsLocalOnly, parseLoraFields(a, body, &paths, &scales, false));
        const n = try parseLoraFields(a, body, &paths, &scales, true);
        defer for (paths[0..n]) |p| a.free(p);
        try testing.expectEqualStrings("/etc/hosts", paths[0]);
    }
    // No LoRA fields detaches, from anyone.
    try testing.expectEqual(@as(usize, 0), try parseLoraFields(a, "{\"prompt\":\"p\"}", &paths, &scales, false));
}

test "parseLoraFields: scales may be negative or far beyond 1, in every request shape" {
    const a = testing.allocator;
    var paths: [lora_mod.MAX_LORAS][]u8 = undefined;
    var scales: [lora_mod.MAX_LORAS]f32 = undefined;
    const Case = struct { body: []const u8, want: []const f32 };
    const cases = [_]Case{
        .{ .body = "{\"lora_paths\":[\"/a\",\"/b\"],\"lora_scales\":[-1, 20]}", .want = &.{ -1, 20 } },
        .{ .body = "{\"lora_paths\":[\"/a\",\"/b\"],\"lora_scales\":\"-20 12.5\"}", .want = &.{ -20, 12.5 } },
        .{ .body = "{\"lora_path\":\"/a\",\"lora_scale\":-12.5}", .want = &.{-12.5} },
    };
    for (cases) |c| {
        const n = try parseLoraFields(a, c.body, &paths, &scales, true);
        defer for (paths[0..n]) |p| a.free(p);
        try testing.expectEqualSlices(f32, c.want, scales[0..n]);
    }
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
        // Numeric fields are spliced into the rebuilt JSON body: a value like
        // `8,"stream":true` would otherwise sail past this surface's own
        // named 400s inside the handler (the review's exact injection).
        .{ "Content-Disposition: form-data; name=\"steps\"\r\n\r\n8,\"stream\":true", EditFormError.MalformedNumber },
        .{ "Content-Disposition: form-data; name=\"ref_resolution\"\r\n\r\nNaN", EditFormError.MalformedNumber },
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

test "peekModelType classifies a model_index-only Qwen-Image-2.1 repo" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];

    try tmp.dir.createDirPath(io, "q");
    try tmp.dir.writeFile(io, .{ .sub_path = "q/model_index.json", .data = "{\"_class_name\":\"QwenImage21Pipeline\"}" });
    const dir = try std.fmt.allocPrint(a, "{s}/q", .{root});
    defer a.free(dir);
    const mt = peekModelType(io, a, dir) orelse return error.TestUnexpectedResult;
    defer a.free(mt);
    try testing.expectEqualStrings("qwen_image21", mt);

    // The 2.0 family is a different architecture — no classification.
    try tmp.dir.createDirPath(io, "q20");
    try tmp.dir.writeFile(io, .{ .sub_path = "q20/model_index.json", .data = "{\"_class_name\":\"QwenImagePipeline\"}" });
    const dir20 = try std.fmt.allocPrint(a, "{s}/q20", .{root});
    defer a.free(dir20);
    try testing.expect(peekModelType(io, a, dir20) == null);
}

test "TRELLIS.2: a pipeline.json-only pack routes to the mesh slot" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    try tmp.dir.createDirPath(io, "m");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/pipeline.json", .data = "{\"name\":\"Trellis2ImageTo3DPipeline\",\"args\":{}}" });
    const dir = try std.fmt.allocPrint(a, "{s}/m", .{root});
    defer a.free(dir);
    const mt = peekModelType(io, a, dir) orelse return error.TestUnexpectedResult;
    defer a.free(mt);
    try testing.expectEqualStrings("trellis2", mt);
    try testing.expectEqual(Modality.mesh, modalityFromType(mt).?);
}

test "Pixal3D: a TRELLIS.2-layout pack whose flows read proj routes to the mesh slot as pixal3d" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    try tmp.dir.createDirPath(io, "m/ckpts");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/pipeline.json", .data = "{\"name\":\"Trellis2ImageTo3DPipeline\",\"args\":{\"models\":{\"sparse_structure_flow_model\":\"ckpts/ss\"}}}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/ckpts/ss.json", .data = "{\"args\":{\"image_attn_mode\":\"proj\"}}" });
    const dir = try std.fmt.allocPrint(a, "{s}/m", .{root});
    defer a.free(dir);
    const mt = peekModelType(io, a, dir) orelse return error.TestUnexpectedResult;
    defer a.free(mt);
    try testing.expectEqualStrings("pixal3d", mt);
    try testing.expectEqual(Modality.mesh, modalityFromType(mt).?);
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

test "estimatePeakResidentBytes: yue2 bills the sum plus its generation working set" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const b100: [100]u8 = @splat('x');
    const b40: [40]u8 = @splat('x');
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &b100 });
    try tmp.dir.writeFile(io, .{ .sub_path = "vae.safetensors", .data = &b40 });
    try std.testing.expectEqual(@as(u64, 140) + @import("../../plugins/yue2/config.zig").gen_buffer_bytes, estimatePeakResidentBytesIn(io, tmp.dir, "yue2"));
    var empty = std.testing.tmpDir(.{ .iterate = true });
    defer empty.cleanup();
    try std.testing.expectEqual(@as(u64, 0), estimatePeakResidentBytesIn(io, empty.dir, "yue2"));
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
    try std.testing.expectEqual(@as(u64, 440) + @import("../../plugins/music3/config.zig").gen_buffer_bytes, estimatePeakResidentBytesIn(io, tmp.dir, "minimax_music3"));
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

test "Modality.modelType round-trips through modalityFromType" {
    for ([_]Modality{ .image, .audio, .video, .mesh, .decision, .diarization }) |m| {
        try testing.expectEqual(m, modalityFromType(m.modelType()).?);
    }
}

test "extractJsonFloat parses cfg scales (int + float + sign)" {
    try testing.expectEqual(@as(?f64, 1.0), extractJsonFloat("{\"cfg_scale\": 1.0}", "cfg_scale"));
    try testing.expectEqual(@as(?f64, 3.5), extractJsonFloat("{\"cfg_scale\":3.5,\"x\":1}", "cfg_scale"));
    try testing.expectEqual(@as(?f64, 7), extractJsonFloat("{\"cfg_audio_scale\": 7}", "cfg_audio_scale"));
    try testing.expectEqual(@as(?f64, null), extractJsonFloat("{\"prompt\":\"hi\"}", "cfg_scale"));
}

test "extractJsonU32 saturates a value past u32 so the caller's bound check refuses it" {
    try testing.expectEqual(@as(?u32, 256), extractJsonU32("{\"width\": 256}", "width"));
    try testing.expectEqual(@as(?u32, std.math.maxInt(u32)), extractJsonU32("{\"width\": 99999999999}", "width"));
    try testing.expectEqual(@as(?u32, null), extractJsonU32("{\"width\": -5}", "width"));
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

test "buildStubCpuState builds a media stub keyed by modality" {
    const a = testing.allocator;
    var stub = try buildStubCpuState(a, .image);
    defer freeStubCpuState(a, &stub);
    try testing.expectEqualStrings("flux2", stub.config.model_type);
    try testing.expect(!stub.config.is_encoder_only);
    try testing.expectEqual(modalityFromType(stub.config.model_type).?, Modality.image);
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
    const f = low_mem.fromInputs;
    try testing.expect(f(true, null, 128 * GB)); // iOS: always, RAM irrelevant
    try testing.expect(f(false, "1", 128 * GB)); // env force-on
    try testing.expect(!f(false, "0", 8 * GB)); // env force-off beats auto
    try testing.expect(f(false, null, 16 * GB)); // 16 GB mini: auto ON
    try testing.expect(f(false, null, 8 * GB)); // 8 GB: auto ON
    try testing.expect(!f(false, null, 24 * GB)); // 24 GB+: off (reload is pure loss)
    try testing.expect(!f(false, null, 0)); // unknown RAM: don't guess
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
    // coexists with the other, plus its text encoder from the other repo.
    try tmp.dir.writeFile(io, .{ .sub_path = "transformer-dev.safetensors", .data = &b500 });
    try tmp.dir.writeFile(io, .{ .sub_path = "transformer-distilled.safetensors", .data = &b500 });
    try std.testing.expectEqual(@as(u64, 2450) + ltx_bill.externalTextEncoderBytes(), estimatePeakResidentBytesIn(io, tmp.dir, "AudioVideo"));
    tmp.dir.deleteFile(io, "transformer-dev.safetensors") catch {};
    tmp.dir.deleteFile(io, "transformer-distilled.safetensors") catch {};

    // H3: max(TE 300, DiT 500*65% + activations, VAEs 120+30 + activations).
    try std.testing.expectEqual(325 + h3_bill.REQUEST_BASE_BYTES, estimatePeakResidentBytesIn(io, tmp.dir, "minimax_h3"));

    // A pack shipping the Turbo LoRA bills it on the DiT term (it is resident
    // ALONGSIDE the DiT and precompute does not free it), whenever present —
    // the gate estimate is per-model, not per-request.
    const b80: [80]u8 = @splat('x');
    try tmp.dir.writeFile(io, .{ .sub_path = "turbo_lora.safetensors", .data = &b80 });
    try std.testing.expectEqual(405 + h3_bill.REQUEST_BASE_BYTES, estimatePeakResidentBytesIn(io, tmp.dir, "minimax_h3"));
    tmp.dir.deleteFile(io, "turbo_lora.safetensors") catch {};
    // Any other media type: the plain sum over the dir (the safe default —
    // a backend without a declared residency plan must not under-bill).
    try std.testing.expectEqual(@as(u64, 1950), estimatePeakResidentBytesIn(io, tmp.dir, "flux2"));
}

test "residency bill: a DiT that --w8a8 converts is billed at its int8 copy, not the pack's bytes" {
    w8a8.nax_override = true; // only shard headers are read, so any machine can run this
    defer w8a8.nax_override = null;
    w8a8.w8a8State().enabled = true;
    defer w8a8.w8a8State().enabled = false;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // One 4-bit g64 linear [N=128, K=256] (18432 B in the pack, 33280 B as int8) and one bf16
    // linear [64, 256] (32768 B in the pack, 16640 B as int8).
    const hdr =
        \\{"blocks.0.attn.wq.weight":{"dtype":"U32","shape":[128,32],"data_offsets":[0,16384]},
        \\"blocks.0.attn.wq.scales":{"dtype":"BF16","shape":[128,4],"data_offsets":[16384,17408]},
        \\"blocks.0.attn.wq.biases":{"dtype":"BF16","shape":[128,4],"data_offsets":[17408,18432]},
        \\"blocks.0.mlp.up.weight":{"dtype":"BF16","shape":[64,256],"data_offsets":[18432,51200]}}
    ;
    var buf: [8 + hdr.len + 51200]u8 = @splat(0);
    std.mem.writeInt(u64, buf[0..8], hdr.len, .little);
    @memcpy(buf[8..][0..hdr.len], hdr);
    try tmp.dir.writeFile(io, .{ .sub_path = "transformer_mixed.safetensors", .data = &buf });
    const other: [1000]u8 = @splat('x');
    try tmp.dir.writeFile(io, .{ .sub_path = "vae.safetensors", .data = &other });
    const file: u64 = buf.len;
    // bill = vae + (file - pack bytes of the two linears) + their int8 copies
    try std.testing.expectEqual(file + 1000 - 51200 + 33280 + 16640, estimatePeakResidentBytesIn(io, tmp.dir, "krea2_turbo"));
    // a type the route does not convert keeps the plain sum
    try std.testing.expectEqual(file + 1000, estimatePeakResidentBytesIn(io, tmp.dir, "stable_audio3"));
    w8a8.w8a8State().enabled = false;
    try std.testing.expectEqual(file + 1000, estimatePeakResidentBytesIn(io, tmp.dir, "krea2_turbo"));
}

test "residency bill: ACE-Step, Music3, Hunyuan3D and LTX DiT shards are billed at their int8 copy" {
    w8a8.nax_override = true; // only shard headers are read, so any machine can run this
    defer w8a8.nax_override = null;
    w8a8.w8a8State().enabled = true;
    defer w8a8.w8a8State().enabled = false;
    const io = std.testing.io;
    // One 4-bit g64 block linear [N=128, K=256]: 18432 B in the pack, 33280 B as int8.
    const Pack = struct {
        fn shard(dir: std.Io.Dir, name: []const u8, key: []const u8) !u64 {
            var hb: [512]u8 = undefined;
            const hdr = try std.fmt.bufPrint(&hb,
                \\{{"{s}.weight":{{"dtype":"U32","shape":[128,32],"data_offsets":[0,16384]}},
                \\"{s}.scales":{{"dtype":"BF16","shape":[128,4],"data_offsets":[16384,17408]}},
                \\"{s}.biases":{{"dtype":"BF16","shape":[128,4],"data_offsets":[17408,18432]}}}}
            , .{ key, key, key });
            var buf: [8 + 512 + 18432]u8 = @splat(0);
            std.mem.writeInt(u64, buf[0..8], hdr.len, .little);
            @memcpy(buf[8..][0..hdr.len], hdr);
            const n = 8 + hdr.len + 18432;
            try dir.writeFile(io, .{ .sub_path = name, .data = buf[0..n] });
            return n;
        }
    };
    const cases = [_]struct { model_type: []const u8, shard: []const u8, key: []const u8 }{
        .{ .model_type = "acestep", .shard = "model.safetensors", .key = "decoder.layers.0.self_attn.q_proj" },
        .{ .model_type = "minimax_music3", .shard = "transformer.safetensors", .key = "transformer_blocks.0.ff_in" },
        .{ .model_type = "hunyuan3d_2_1", .shard = "dit.safetensors", .key = "blocks.0.attn1.q" },
    };
    for (cases) |c| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        const file = try Pack.shard(tmp.dir, c.shard, c.key);
        const extra: u64 = if (std.mem.eql(u8, c.model_type, "minimax_music3")) @import("../../plugins/music3/config.zig").gen_buffer_bytes else 0;
        try std.testing.expectEqual(file - 18432 + 33280 + extra, estimatePeakResidentBytesIn(io, tmp.dir, c.model_type));
    }
    // The ANE seam reads the packed weights, so under `--ane-audio` ACE-Step is billed as the pack.
    ane.aneState().media_offload.audio = true;
    defer ane.aneState().media_offload.audio = false;
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        const file = try Pack.shard(tmp.dir, "model.safetensors", "decoder.layers.0.self_attn.q_proj");
        try std.testing.expectEqual(file, estimatePeakResidentBytesIn(io, tmp.dir, "acestep"));
    }
    ane.aneState().media_offload.audio = false;
    // LTX bills both variants' resident bytes, one of them spare (they never coexist).
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const f1 = try Pack.shard(tmp.dir, "transformer-dev.safetensors", "transformer.transformer_blocks.0.attn1.to_q");
    _ = try Pack.shard(tmp.dir, "transformer-distilled.safetensors", "transformer.transformer_blocks.0.attn1.to_q");
    try std.testing.expectEqual(f1 - 18432 + 33280 + ltx_bill.externalTextEncoderBytes(), estimatePeakResidentBytesIn(io, tmp.dir, "AudioVideo"));
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
    try std.testing.expectEqualStrings("connector.safetensors", discovery.requiredMediaMarker("AudioVideo").?);
    try std.testing.expectEqualStrings("transformer.safetensors", discovery.requiredMediaMarker("minimax_h3").?);
    // Music3: the converter writes the vocoder LAST, so its presence is the
    // completeness marker for the whole five-file pack.
    try std.testing.expectEqualStrings("vocoder.safetensors", discovery.requiredMediaMarker("minimax_music3").?);
    // H3 must NOT be gated on LTX's file.
    try std.testing.expect(!std.mem.eql(u8, discovery.requiredMediaMarker("minimax_h3").?, "connector.safetensors"));

    // Every media type either declares its own marker or needs none; none may
    // inherit another backend's.
    for (plugin_served.types) |mt| {
        if (discovery.requiredMediaMarker(mt)) |m| {
            try std.testing.expect(m.len > 0);
            if (!std.mem.eql(u8, mt, "AudioVideo"))
                try std.testing.expect(!std.mem.eql(u8, m, "connector.safetensors"));
        }
    }
    // A non-media type never carries one.
    try std.testing.expect(discovery.requiredMediaMarker("gemma4") == null);
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

test "instrumental is parsed off the body only when spelled true" {
    try testing.expect(sse.bodyWantsTrue("{\"instrumental\":true}", "instrumental"));
    try testing.expect(sse.bodyWantsTrue("{\"instrumental\": true}", "instrumental"));
    try testing.expect(!sse.bodyWantsTrue("{\"instrumental\":false}", "instrumental"));
    try testing.expect(!sse.bodyWantsTrue("{\"prompt\":\"x\"}", "instrumental"));
}

test "a base64 WebP reference reaches the media backends as a PNG with the same pixels" {
    const a = std.testing.allocator;
    // cwebp -lossless -exact of a 3x2 RGBA PNG (one pixel fully transparent).
    const webp_bytes = [_]u8{
        0x52, 0x49, 0x46, 0x46, 0x4e, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
        0x56, 0x50, 0x38, 0x4c, 0x41, 0x00, 0x00, 0x00, 0x2f, 0x02, 0x40, 0x00,
        0x10, 0x2f, 0x40, 0x90, 0x6d, 0xb3, 0xcc, 0x61, 0x5e, 0xf7, 0x77, 0xb9,
        0x07, 0x99, 0xb4, 0x4d, 0xe9, 0x39, 0xff, 0x7a, 0x66, 0xad, 0x8a, 0x1a,
        0x10, 0x80, 0x1c, 0x15, 0xbc, 0xf5, 0x2f, 0x69, 0x14, 0xb6, 0x6d, 0x83,
        0x94, 0xf6, 0x35, 0x9d, 0xbe, 0x80, 0x83, 0xf8, 0xea, 0x7a, 0xeb, 0xd6,
        0xdb, 0x03, 0x08, 0x48, 0xc8, 0x94, 0x3c, 0xf2, 0x45, 0x44, 0xff, 0x23,
        0x16, 0x00,
    };
    const want = [_]u8{ 255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 10, 20, 30, 255, 200, 100, 50, 255, 1, 2, 3, 4 };
    var b64: [std.base64.standard.Encoder.calcSize(webp_bytes.len)]u8 = undefined;
    const out = try imageB64Alloc(a, std.base64.standard.Encoder.encode(&b64, &webp_bytes));
    defer a.free(out);
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const px = stb.stbi_load_from_memory(out.ptr, @intCast(out.len), &w, &h, &ch, 4) orelse return error.TestUnexpectedResult;
    defer stb.stbi_image_free(px);
    try std.testing.expectEqual(@as(c_int, 3), w);
    try std.testing.expectEqual(@as(c_int, 2), h);
    try std.testing.expectEqualSlices(u8, &want, px[0..want.len]);

    // Anything stb already reads passes through byte for byte.
    const png_bytes = try png_mod.encodeRgba(a, &want, 3, 2);
    defer a.free(png_bytes);
    var b64_png: [256]u8 = undefined;
    const same = try imageB64Alloc(a, std.base64.standard.Encoder.encode(&b64_png, png_bytes));
    defer a.free(same);
    try std.testing.expectEqualSlices(u8, png_bytes, same);
}

test "jobFailureStatus: after a lost GPU context every failure is a 503" {
    var buf: [512]u8 = undefined;
    _ = mlx.takeError(&buf);
    defer mlx.resetGpuContextLostForTest();
    mlx.latchErrorForTest("cuLaunchKernelEx(&config, func, params, nullptr) failed: an illegal memory access was encountered");
    _ = mlx.takeError(&buf);
    try std.testing.expectEqual(@as(u16, 503), jobFailureStatus(error.MlxError));
}

test "jobFailureStatus: a memory failure is a 503, any other failure a 500" {
    var stale: [512]u8 = undefined;
    _ = mlx.takeError(&stale); // an earlier test's latch would shadow this one's
    try std.testing.expectEqual(@as(u16, 503), jobFailureStatus(error.OutOfMemory));
    try std.testing.expectEqual(@as(u16, 500), jobFailureStatus(error.MlxError));
    var buf: [256]u8 = undefined;
    mlx.latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try std.testing.expectEqual(@as(u16, 503), jobFailureStatus(error.MlxError));
    _ = mlx.takeError(&buf);
    mlx.latchErrorForTest("[gather] index out of range");
    try std.testing.expectEqual(@as(u16, 500), jobFailureStatus(error.MlxError));
    _ = mlx.takeError(&buf);
}
