//! embedding length limits; handleEmbeddings, EG2 messages, sendEmbeddings; embeddingPrompt

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const scheduler_mod = @import("../gpu_thread/requests.zig");
const model_registry_mod = @import("../registry/registry.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;
const http_server = @import("../http/server.zig");

const Conn = @import("../http/conn.zig").Conn;
const imageUrlOfPart = @import("media_gate.zig").imageUrlOfPart;
const echoModelName = @import("request_text.zig").echoModelName;
const sendResponse = @import("../http/respond.zig").sendResponse;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const jsonEscape = @import("format.zig").jsonEscape;
const MediaItem = @import("../media/placeholders.zig").MediaItem;
const mediaRunTokens = @import("../media/placeholders.zig").mediaRunTokens;
const prepareMultimodalItems = @import("../media/encode.zig").prepareMultimodalItems;
const multimodalFailure = @import("../media/encode.zig").multimodalFailure;
const visionPreprocFromConfig = @import("../media/prepare.zig").visionPreprocFromConfig;
const videoPreprocFromConfig = @import("../media/prepare.zig").videoPreprocFromConfig;
const RequestMedia = @import("../media/request_media.zig").RequestMedia;
const appendImageUrlContent = @import("../media/request_media.zig").appendImageUrlContent;
const IMAGE_DECODE_REJECT = @import("../media/request_media.zig").IMAGE_DECODE_REJECT;
const appendVideoUrlContent = @import("../media/decode.zig").appendVideoUrlContent;
const ByteTokenizer = @import("../media/placeholders.zig").ByteTokenizer;

/// Effective per-input embedding token ceiling: the tighter of the operator
/// flag and the model's declared window (0 on either side = no bound from
/// that side; 0 result = unbounded). A flag above the model's window clamps
/// to the window — a 512-position BERT never accepts a 4096 override.
pub fn embedEffectiveLimit(flag: u32, model_max: u32) u32 {
    if (flag == 0) return model_max;
    if (model_max == 0) return flag;
    return @min(flag, model_max);
}

/// Overflow 400 body text: names the offending input INDEX and both counts,
/// because the client can only fix what it can see (the contextOverflowMessage
/// principle). bufPrint failure falls back to the bare sentence rather than
/// sending no body (the media-gen fixed-buffer class).
pub fn embedOverflowMessage(buf: []u8, index: usize, tokens: usize, limit: u32) []const u8 {
    return std.fmt.bufPrint(
        buf,
        "Input at index {d} exceeds the maximum embedding input length: {d} tokens given, {d} allowed",
        .{ index, tokens, limit },
    ) catch "Input exceeds the maximum embedding input length";
}

// Plan 05: vision encoder and model id moved to `LoadedModel.vision_encoder`

// and `LoadedModel.id`. Handlers read them off `lm`. `global_vision_encoder`

// and `global_model_id` singletons were removed. The `discovered_models`

// slice was also removed — `/v1/models` iterates `registry.entries` directly.

/// `<bos> ids <eos>` for bidirectional embedding models. Either special is
/// skipped when unknown; already-wrapped input is not double-wrapped.
fn wrapEncoderIds(allocator: std.mem.Allocator, ids: []const u32, bos: ?u32, eos: ?u32) ![]u32 {
    const need_bos = bos != null and (ids.len == 0 or ids[0] != bos.?);
    const need_eos = eos != null and (ids.len == 0 or ids[ids.len - 1] != eos.?);
    const out = try allocator.alloc(u32, ids.len + @intFromBool(need_bos) + @intFromBool(need_eos));
    var i: usize = 0;
    if (need_bos) {
        out[i] = bos.?;
        i += 1;
    }
    @memcpy(out[i .. i + ids.len], ids);
    i += ids.len;
    if (need_eos) out[i] = eos.?;
    return out;
}

pub fn handleEmbeddings(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    // Engine-backed (GGUF/ds4) models have no MLX transformer and both embed
    // paths forward through it: refuse by name before anything is queued.
    if (lm.transformer == null) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "This model runs on an embedded engine (GGUF); embeddings are not supported. Load an MLX embedding model instead.", null);
        return;
    }
    const tok = lm.tokenizer.?;
    const config = lm.config.?;
    // A model whose plugin owns its decode state (DeepSeek-V4/V4.1, GLM-5.3) runs no stateless hidden-state forward.
    if (config.caps().owns_decode_state) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "This model does not serve embeddings. Load an MLX embedding model instead.", null);
        return;
    }
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", null);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", null);
        return;
    }
    const root = parsed.value.object;

    const model_name = echoModelName(root, config.model_type);

    // OpenAI `dimensions` (text-embedding-3 semantics): keep the first N
    // components, L2-renormalize. Anything not a positive integer is a 400 —
    // accepting the field and ignoring it would hand callers wrong-width
    // vectors they only discover at retrieval time. Over-native is checked
    // after the forward, when the model's width is known.
    const req_dims: ?usize = if (root.get("dimensions")) |v| blk: {
        if (v != .integer or v.integer < 1) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'dimensions' must be a positive integer", null);
            return;
        }
        break :blk @intCast(v.integer);
    } else null;

    // `messages` (vLLM's chat embeddings) is the form that carries images and video: one prompt, one embedding.
    if (root.get("messages")) |messages_val| {
        if (root.get("input") != null) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Send either 'input' or 'messages', not both", null);
            return;
        }
        return handleMessagesEmbedding(allocator, stream, lm, messages_val, req_dims, model_name);
    }

    // Parse input — can be a string or array of strings
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing 'input' field", null);
        return;
    };

    // Collect input texts
    var texts = std.ArrayList([]const u8).empty;
    defer texts.deinit(allocator);

    switch (input_val) {
        .string => |s| try texts.append(allocator, s),
        .array => |arr| {
            for (arr.items) |item| {
                if (item == .string) {
                    try texts.append(allocator, item.string);
                }
            }
        },
        else => {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' must be a string or array of strings", null);
            return;
        },
    }

    if (texts.items.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' must not be empty", null);
        return;
    }

    log.info("POST /v1/embeddings ({d} inputs)\n", .{texts.items.len});

    // Tokenize every input up front so the whole request rides ONE scheduler
    // round-trip; the inference thread embeds the sequences in padded,
    // key-masked GPU batches (generate.EMBED_MAX_BATCH per forward) instead
    // of one forward per text.
    var total_tokens: usize = 0;
    var seqs = std.ArrayList([]const u32).empty;
    defer {
        for (seqs.items) |ids| allocator.free(ids);
        seqs.deinit(allocator);
    }
    for (texts.items) |text| {
        if (std.mem.trim(u8, text, " \t\r\n").len == 0) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' entries must be non-empty strings", null);
            return;
        }
        const raw_ids = try tok.encode(allocator, text);
        const ids = try wrapEmbeddingIds(allocator, config, tok.appended_special, raw_ids);
        total_tokens += ids.len;
        try seqs.append(allocator, ids);
    }

    var msg_buf: [160]u8 = undefined;
    for (seqs.items, 0..) |ids, idx| {
        if (embeddingLimitReject(&msg_buf, config, idx, ids.len)) |msg| {
            log.warn("  embedding input {d} over limit: {d} tokens\n", .{ idx, ids.len });
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, null);
            return;
        }
    }

    const embeddings = (try embeddingRows(allocator, stream, lm, seqs.items, null)) orelse return;
    defer {
        for (embeddings) |e| allocator.free(e);
        allocator.free(embeddings);
    }
    try sendEmbeddings(allocator, stream, embeddings, req_dims, model_name, total_tokens);
}

/// A tokenized input as the model's pipeline feeds it: bidirectional embedding models (EmbeddingGemma) declare
/// add_bos_token + add_eos_token and the SentencePiece encode path adds neither, so wrap here (BERT's
/// [CLS]/[SEP] come from WordPiece itself); a last-token pooler (Qwen3-Embedding) pools an APPENDED terminator,
/// which its tokenizer's TemplateProcessing adds to every encode and the reference pools THAT position — without
/// it we'd pool the final text token and quietly diverge from the model card. Frees `raw_ids`.
/// `appended`: the tokenizer's post-processor terminator (`Tokenizer.appended_special`), which can differ from
/// config.json's eos (Qwen3-Embedding-4B: `<|im_end|>` there, `<|endoftext|>` pooled).
fn wrapEmbeddingIds(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig, appended: ?u32, raw_ids: []u32) ![]u32 {
    const eos: ?u32 = if (config.num_eos_tokens > 0) config.eos_token_ids[0] else null;
    if (config.use_bidirectional_attention) {
        defer allocator.free(raw_ids);
        return wrapEncoderIds(allocator, raw_ids, config.bos_token_id, eos);
    }
    if (config.effectivePooling() == .last_token) {
        defer allocator.free(raw_ids);
        return wrapEncoderIds(allocator, raw_ids, null, appended orelse eos);
    }
    return raw_ids;
}

/// Issue #117: the effective per-input token ceiling is enforced BEFORE the forward pass — the server owns the
/// exact tokenizer, so the counts are authoritative. The over-limit 400's text (an explicit refusal, never
/// truncation), or null when `tokens` fits.
fn embeddingLimitReject(buf: []u8, config: *const model_mod.ModelConfig, index: usize, tokens: usize) ?[]const u8 {
    const limit = embedEffectiveLimit(http_server.embedding_max_length, config.max_position_embeddings);
    if (limit == 0 or tokens <= limit) return null;
    return embedOverflowMessage(buf, index, tokens, limit);
}

/// The encoder forward runs on the GPU thread (the one MLX caller); every sub-batch resets the KV cache. `vision` (one
/// prompt's soft tokens) passes to the request, whose GPU-thread side frees it. Null after the 500.
fn embeddingRows(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    seqs: []const []const u32,
    vision: ?mlx.mlx_array,
) !?[][]f32 {
    var req = scheduler_mod.EmbedRequest{
        .model = lm,
        .token_seqs = seqs,
        .vision_embeddings = vision,
        .allocator = allocator,
    };
    return http_server.launch().global_scheduler.?.computeEmbeddings(&req) catch |err| {
        if (req.error_name) |e| {
            log.err("  embedding error: {s}\n", .{e});
            allocator.free(e);
        } else {
            log.err("  embedding error: {}\n", .{err});
        }
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "Failed to compute embedding", null);
        return null;
    };
}

fn sendEmbeddings(
    allocator: std.mem.Allocator,
    stream: *Conn,
    embeddings: []const []f32,
    req_dims: ?usize,
    model_name: []const u8,
    total_tokens: usize,
) !void {
    if (req_dims) |d| {
        const native = if (embeddings.len > 0) embeddings[0].len else 0;
        if (d > native) {
            const msg = try std.fmt.allocPrint(allocator, "'dimensions' must be between 1 and {d} for this model", .{native});
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, null);
            return;
        }
    }

    // Build response JSON
    var resp_buf = std.ArrayList(u8).empty;
    defer resp_buf.deinit(allocator);

    try resp_buf.appendSlice(allocator, "{\"object\":\"list\",\"data\":[");

    for (embeddings, 0..) |full_embedding, idx| {
        const embedding = if (req_dims) |d| truncateEmbeddingDims(full_embedding, d) else full_embedding;
        if (idx > 0) try resp_buf.appendSlice(allocator, ",");

        // Format: {"object":"embedding","embedding":[...floats...],"index":N}
        const idx_str = try std.fmt.allocPrint(allocator, "{d}", .{idx});
        defer allocator.free(idx_str);
        try resp_buf.appendSlice(allocator, "{\"object\":\"embedding\",\"embedding\":[");

        for (embedding, 0..) |val, i| {
            if (i > 0) try resp_buf.appendSlice(allocator, ",");
            var buf: [32]u8 = undefined;
            const float_str = std.fmt.bufPrint(&buf, "{d:.8}", .{val}) catch "0";
            try resp_buf.appendSlice(allocator, float_str);
        }

        try resp_buf.appendSlice(allocator, "],\"index\":");
        try resp_buf.appendSlice(allocator, idx_str);
        try resp_buf.appendSlice(allocator, "}");
    }

    const total_str = try std.fmt.allocPrint(allocator, "{d}", .{total_tokens});
    defer allocator.free(total_str);
    const model_escaped = try jsonEscape(allocator, model_name);
    defer allocator.free(model_escaped);

    try resp_buf.appendSlice(allocator, "],\"model\":");
    try resp_buf.appendSlice(allocator, model_escaped);
    try resp_buf.appendSlice(allocator, ",\"usage\":{\"prompt_tokens\":");
    try resp_buf.appendSlice(allocator, total_str);
    try resp_buf.appendSlice(allocator, ",\"total_tokens\":");
    try resp_buf.appendSlice(allocator, total_str);
    try resp_buf.appendSlice(allocator, "}}");

    try sendResponse(stream, "200 OK", "application/json", resp_buf.items);
    log.info("  <- {d} embeddings ({d} tokens)\n", .{ embeddings.len, total_tokens });
}

/// `POST /v1/embeddings` with `messages`: images, video and text as ONE prompt, ONE embedding. The soft tokens
/// of every image and video enter the sequence at their placeholders, exactly where the reference's processor
/// puts them (`<bos>` … `<eos>` around the lot).
fn handleMessagesEmbedding(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    messages_val: std.json.Value,
    req_dims: ?usize,
    model_name: []const u8,
) !void {
    const config = lm.config.?;
    if (messages_val != .array or messages_val.array.items.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' must be a non-empty array", null);
        return;
    }
    if (!config.gv_budgeted) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' is served by multimodal embedding models (EmbeddingGemma 2); this model takes 'input'", null);
        return;
    }

    var media = RequestMedia.init(allocator);
    defer media.deinit();
    var built = try embeddingPrompt(allocator, lm.tokenizer.?, config, messages_val.array.items, &media, lm.vision_encoder != null);
    defer built.deinit(allocator);
    const prompt = switch (built) {
        .ok => |p| p,
        .reject => |why| {
            log.warn("POST /v1/embeddings -> 400 ({s})\n", .{why});
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, null);
            return;
        },
    };

    // Wrapped, with the placeholders still one id each; the expansion below replaces them with the soft-token runs.
    var ids = try wrapEmbeddingIds(allocator, config, lm.tokenizer.?.appended_special, try allocator.dupe(u32, prompt.ids));
    defer allocator.free(ids);
    log.info("POST /v1/embeddings (messages: {d} media items)\n", .{prompt.items.len});

    // The expanded length is known from the preprocessed sizes alone: refuse an over-limit prompt before the
    // tower runs rather than after (the tower is the expensive half).
    const expected = ids.len - prompt.items.len + mediaRunTokens(prompt.items, config);
    var msg_buf: [160]u8 = undefined;
    if (embeddingLimitReject(&msg_buf, config, 0, expected)) |msg| {
        log.warn("  embedding prompt over limit: {d} tokens\n", .{expected});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, null);
        return;
    }

    var mm = prepareMultimodalItems(allocator, lm, config, prompt.items, &ids) catch |err| {
        const failure = multimodalFailure(err);
        log.warn("POST /v1/embeddings -> {d} ({s})\n", .{ failure.code, @errorName(err) });
        try sendErrorResponse(allocator, stream, failure.status, failure.kind, failure.message, failure.code);
        return;
    };
    defer mm.deinit(allocator);

    const seqs = [_][]const u32{ids};
    // The request owns the soft tokens from here on: its inference-thread side frees them.
    const vision = mm.embeddings;
    mm.embeddings = null;
    const embeddings = (try embeddingRows(allocator, stream, lm, &seqs, vision)) orelse return;
    defer {
        for (embeddings) |e| allocator.free(e);
        allocator.free(embeddings);
    }
    try sendEmbeddings(allocator, stream, embeddings, req_dims, model_name, ids.len);
}

/// OpenAI `dimensions` semantics (text-embedding-3 class): keep the first
/// `dims` components and L2-renormalize, in place. `dims >= len` returns the
/// vector untouched so the default (no `dimensions`) path stays byte-identical.
/// A zero prefix is returned unnormalized rather than divided by zero.
fn truncateEmbeddingDims(embedding: []f32, dims: usize) []f32 {
    if (dims == 0 or dims >= embedding.len) return embedding;
    const out = embedding[0..dims];
    var sumsq: f64 = 0;
    for (out) |v| sumsq += @as(f64, v) * @as(f64, v);
    if (sumsq <= 0) return out;
    const inv = 1.0 / @sqrt(sumsq);
    for (out) |*v| v.* = @floatCast(@as(f64, v.*) * inv);
    return out;
}

/// What a `messages` embedding request renders to: the text's ids with ONE placeholder id where each image or
/// video sits, and the media in the same order (`placeholdersMatch` / `expandMediaPlaceholders` read exactly this).
const EmbeddingPrompt = struct {
    ids: []u32,
    items: []MediaItem,
};

const EmbeddingPromptResult = union(enum) {
    ok: EmbeddingPrompt,
    reject: []const u8,

    pub fn deinit(self: *EmbeddingPromptResult, allocator: std.mem.Allocator) void {
        if (self.* == .ok) {
            allocator.free(self.ok.ids);
            allocator.free(self.ok.items);
        }
    }
};

const EMBED_PART_REJECT = "content parts must be text, image_url or video_url objects";

const EMBED_NO_VISION_REJECT = "image and video parts need a model with a vision encoder: this one has none loaded (a text-only pack, or --no-vision)";

const EMBED_AUDIO_REJECT = "audio parts are not served: this model's audio tower is not loaded";

const EMBED_EMPTY_REJECT = "'messages' carry nothing to embed: add a text part, an image or a video";

const VIDEO_DECODE_REJECT = "video could not be decoded: send 'frames', a non-empty array of base64 image data URLs (data:image/jpeg|png|webp|gif|bmp;base64,...) with readable payloads; remote URLs are not fetched";

/// `messages` the way the model's own chat template reads them for an embedding (EmbeddingGemma 2's is a bare
/// concatenation): system turns first, then the rest in order; text parts verbatim, no separator invented; an
/// image or video part stands as its placeholder id. Roles are otherwise ignored. A reject carries the named 400.
fn embeddingPrompt(
    allocator: std.mem.Allocator,
    tok: anytype,
    config: *const model_mod.ModelConfig,
    messages: []const std.json.Value,
    media: *RequestMedia,
    can_see: bool,
) !EmbeddingPromptResult {
    var ids = std.ArrayList(u32).empty;
    var items = std.ArrayList(MediaItem).empty;
    var ok = false;
    defer if (!ok) {
        ids.deinit(allocator);
        items.deinit(allocator);
    };
    const image_vp = visionPreprocFromConfig(config);
    const video_vp = videoPreprocFromConfig(config);

    for ([_]bool{ true, false }) |system_pass| for (messages) |message| {
        if (message != .object) return .{ .reject = "'messages' entries must be objects" };
        const role = message.object.get("role");
        const is_system = role != null and role.? == .string and std.mem.eql(u8, role.?.string, "system");
        if (is_system != system_pass) continue;
        const content = message.object.get("content") orelse continue;
        const parts: []const std.json.Value = switch (content) {
            .null => &.{},
            .string => |text| {
                try appendEmbeddingText(allocator, tok, &ids, text);
                continue;
            },
            .array => |arr| arr.items,
            else => return .{ .reject = "'content' must be a string or an array of parts" },
        };
        for (parts) |part| {
            const type_val = if (part == .object) part.object.get("type") else null;
            if (type_val == null or type_val.? != .string) return .{ .reject = EMBED_PART_REJECT };
            const kind = type_val.?.string;
            if (std.mem.eql(u8, kind, "text")) {
                const text = part.object.get("text") orelse return .{ .reject = EMBED_PART_REJECT };
                if (text != .string) return .{ .reject = EMBED_PART_REJECT };
                try appendEmbeddingText(allocator, tok, &ids, text.string);
            } else if (std.mem.eql(u8, kind, "image_url")) {
                if (!can_see) return .{ .reject = EMBED_NO_VISION_REJECT };
                const url = imageUrlOfPart(part.object) orelse return .{ .reject = IMAGE_DECODE_REJECT };
                const slot = try media.openImages();
                if (!appendImageUrlContent(allocator, media.images(slot), url, image_vp)) return .{ .reject = IMAGE_DECODE_REJECT };
                try items.append(allocator, .{ .image = media.imagesSlice(slot).? });
                try ids.append(allocator, config.image_token_id);
            } else if (std.mem.eql(u8, kind, "video_url")) {
                if (!can_see) return .{ .reject = EMBED_NO_VISION_REJECT };
                const holder = part.object.get("video_url") orelse return .{ .reject = VIDEO_DECODE_REJECT };
                const frames = if (holder == .object) holder.object.get("frames") else null;
                if (frames == null or frames.? != .array or frames.?.array.items.len == 0) return .{ .reject = VIDEO_DECODE_REJECT };
                var urls = std.ArrayList([]const u8).empty;
                defer urls.deinit(allocator);
                for (frames.?.array.items) |frame| {
                    if (frame != .string) return .{ .reject = VIDEO_DECODE_REJECT };
                    try urls.append(allocator, frame.string);
                }
                const slot = try media.openVideos();
                appendVideoUrlContent(allocator, media.videos(slot), urls.items, video_vp);
                const clips = media.videosSlice(slot) orelse return .{ .reject = VIDEO_DECODE_REJECT };
                try items.append(allocator, .{ .video = &clips[0] });
                try ids.append(allocator, config.video_token_id);
            } else if (std.mem.eql(u8, kind, "input_audio") or std.mem.eql(u8, kind, "audio_url")) {
                return .{ .reject = EMBED_AUDIO_REJECT };
            } else return .{ .reject = EMBED_PART_REJECT };
        }
    };
    if (ids.items.len == 0) return .{ .reject = EMBED_EMPTY_REJECT };

    const out_ids = try ids.toOwnedSlice(allocator);
    errdefer allocator.free(out_ids);
    const out_items = try items.toOwnedSlice(allocator);
    ok = true;
    return .{ .ok = .{ .ids = out_ids, .items = out_items } };
}

fn appendEmbeddingText(allocator: std.mem.Allocator, tok: anytype, ids: *std.ArrayList(u32), text: []const u8) !void {
    if (text.len == 0) return;
    const encoded = try tok.encode(allocator, text);
    defer allocator.free(encoded);
    try ids.appendSlice(allocator, encoded);
}

// ── tests ──

test "wrapEncoderIds: <bos> … <eos> wrapping for embedding models" {
    const a = testing.allocator;
    // EmbeddingGemma shape: bos 2, eos 1.
    const wrapped = try wrapEncoderIds(a, &.{ 10, 11, 12 }, 2, 1);
    defer a.free(wrapped);
    try testing.expectEqualSlices(u32, &.{ 2, 10, 11, 12, 1 }, wrapped);
    // Qwen3-Embedding shape (issue #116): eos-only append — the reference
    // pools the appended <|endoftext|> position, so the id must be there.
    const eos_only = try wrapEncoderIds(a, &.{ 10, 11 }, null, 151643);
    defer a.free(eos_only);
    try testing.expectEqualSlices(u32, &.{ 10, 11, 151643 }, eos_only);

    // Missing specials are skipped, never invented.
    const no_bos = try wrapEncoderIds(a, &.{ 10, 11 }, null, 1);
    defer a.free(no_bos);
    try testing.expectEqualSlices(u32, &.{ 10, 11, 1 }, no_bos);

    // Already-wrapped input is left alone (idempotent).
    const already = try wrapEncoderIds(a, &.{ 2, 10, 1 }, 2, 1);
    defer a.free(already);
    try testing.expectEqualSlices(u32, &.{ 2, 10, 1 }, already);

    // Empty input still gets both specials.
    const empty = try wrapEncoderIds(a, &.{}, 2, 1);
    defer a.free(empty);
    try testing.expectEqualSlices(u32, &.{ 2, 1 }, empty);
}

test "wrapEmbeddingIds: a last-token pooler appends its tokenizer's terminator, not config.json's eos (#909)" {
    const a = testing.allocator;
    // Qwen3-Embedding-4B: config eos <|im_end|> 151645, post-processor appends <|endoftext|> 151643.
    var cfg = model_mod.ModelConfig{};
    cfg.pooling_mode = .last_token;
    cfg.addEosToken(151645);
    const ids = try wrapEmbeddingIds(a, &cfg, 151643, try a.dupe(u32, &.{ 10, 11 }));
    defer a.free(ids);
    try testing.expectEqualSlices(u32, &.{ 10, 11, 151643 }, ids);
    // No post-processor terminator: config eos.
    const fallback = try wrapEmbeddingIds(a, &cfg, null, try a.dupe(u32, &.{ 10, 11 }));
    defer a.free(fallback);
    try testing.expectEqualSlices(u32, &.{ 10, 11, 151645 }, fallback);
}

pub fn promptFor(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig, body: []const u8, media: *RequestMedia, can_see: bool) !EmbeddingPromptResult {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    return embeddingPrompt(allocator, ByteTokenizer{}, config, parsed.value.array.items, media, can_see);
}

test "embedEffectiveLimit: tighter of flag and model window; zeros mean unbounded" {
    // Auto (no flag): the model's window rules; a windowless model is unbounded.
    try std.testing.expectEqual(@as(u32, 512), embedEffectiveLimit(0, 512));
    try std.testing.expectEqual(@as(u32, 0), embedEffectiveLimit(0, 0));
    // Operator override: enforced when tighter…
    try std.testing.expectEqual(@as(u32, 1024), embedEffectiveLimit(1024, 32768));
    // …and clamped to the model's declared window when looser — a 512-position
    // BERT never accepts a 4096 override (issue #117 acceptance case).
    try std.testing.expectEqual(@as(u32, 512), embedEffectiveLimit(4096, 512));
    // Flag against a windowless model still binds.
    try std.testing.expectEqual(@as(u32, 2048), embedEffectiveLimit(2048, 0));
}

test "embedOverflowMessage: names the input index and both counts" {
    var buf: [160]u8 = undefined;
    const msg = embedOverflowMessage(&buf, 3, 1301, 1024);
    try std.testing.expectEqualStrings(
        "Input at index 3 exceeds the maximum embedding input length: 1301 tokens given, 1024 allowed",
        msg,
    );
}

test "truncateEmbeddingDims: OpenAI dimensions semantics (truncate + L2-renormalize)" {
    const t = std.testing;

    // Truncate to 2 of 4: keeps the first components, renormalizes to unit L2.
    var v = [_]f32{ 3.0, 4.0, 100.0, 100.0 };
    const out = truncateEmbeddingDims(&v, 2);
    try t.expectEqual(@as(usize, 2), out.len);
    try t.expectApproxEqAbs(@as(f32, 0.6), out[0], 1e-6); // 3/5
    try t.expectApproxEqAbs(@as(f32, 0.8), out[1], 1e-6); // 4/5
    var norm: f64 = 0;
    for (out) |x| norm += @as(f64, x) * @as(f64, x);
    try t.expectApproxEqAbs(@as(f64, 1.0), norm, 1e-6);

    // dims == native width: byte-identical pass-through, no renorm — the
    // default (no `dimensions`) path must never change.
    var w = [_]f32{ 0.5, -0.25, 0.125 };
    const same = truncateEmbeddingDims(&w, 3);
    try t.expectEqual(@as(usize, 3), same.len);
    try t.expectEqual(@as(f32, 0.5), same[0]);
    try t.expectEqual(@as(f32, -0.25), same[1]);
    try t.expectEqual(@as(f32, 0.125), same[2]);

    // Zero prefix: truncation must not divide by zero or emit NaN.
    var z = [_]f32{ 0.0, 0.0, 1.0 };
    const zt = truncateEmbeddingDims(&z, 2);
    try t.expectEqual(@as(usize, 2), zt.len);
    for (zt) |x| try t.expect(!std.math.isNan(x));
}
