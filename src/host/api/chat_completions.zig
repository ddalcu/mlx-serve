//! handleChatCompletions; constrained emit, handleNonStreamingGeneration

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kv_quant = @import("../../sdk/blocks/attention/kv_quant.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const request_mod = @import("../../sdk/request/request.zig");
const chat_mod = @import("../chat/chat.zig");
const rp_mod = @import("../../sdk/grammar/protocol.zig");
const log = @import("../../sdk/util/log.zig");
const responses_mod = @import("responses/model.zig");
const pld_index = @import("../../sdk/spec/pld_index.zig");
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const model_registry_mod = @import("../registry/registry.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const io_util = @import("../../sdk/util/io_util.zig");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const Stopwatch = io_util.Stopwatch;
const http_server = @import("../http/server.zig");

const Conn = @import("../http/conn.zig").Conn;
const spec_gate_threshold = @import("request_options.zig").spec_gate_threshold;
const parseKvAttnExplicit = @import("request_options.zig").parseKvAttnExplicit;
const resolveSampling = @import("sampling.zig").resolveSampling;
const generationMaxTokens = @import("sampling.zig").generationMaxTokens;
const requestProfile = @import("sampling.zig").requestProfile;
const defaultEnableMtp = @import("request_options.zig").defaultEnableMtp;
const dsv4DraftStages = @import("request_options.zig").dsv4DraftStages;
const mtpCapable = @import("request_options.zig").mtpCapable;
const admitMtpForCtx = @import("request_options.zig").admitMtpForCtx;
const getTimeoutNs = @import("../http/server.zig").getTimeoutNs;
const messageReasoningFromObj = @import("../chat/reasoning/request.zig").messageReasoningFromObj;
const promptOpensThink = @import("../chat/reasoning/request.zig").promptOpensThink;
const armThinkBound = @import("../chat/reasoning/request.zig").armThinkBound;
const armToolForce = @import("../chat/reasoning/request.zig").armToolForce;
const resolveReasoningProtocol = @import("../chat/reasoning/request.zig").resolveReasoningProtocol;
const omittedMaxTokensDefault = @import("sampling.zig").omittedMaxTokensDefault;
const launchMaxTokensDefault = @import("sampling.zig").launchMaxTokensDefault;
const parseRequestSeed = @import("sampling.zig").parseRequestSeed;
const clampJsonI32 = @import("sampling.zig").clampJsonI32;
const maxTokensOrigin = @import("sampling.zig").maxTokensOrigin;
const MAX_TOKENS_DESC_LEN = @import("sampling.zig").MAX_TOKENS_DESC_LEN;
const describeMaxTokens = @import("sampling.zig").describeMaxTokens;
const requestEosSlice = @import("sampling.zig").requestEosSlice;
const toolCallFinishReason = @import("../chat/tools/pipeline.zig").toolCallFinishReason;
const shouldParseToolCalls = @import("../chat/tools/pipeline.zig").shouldParseToolCalls;
const finishDetailsField = @import("../chat/tools/pipeline.zig").finishDetailsField;
const parseToolCallsForRequest = @import("../chat/tools/pipeline.zig").parseToolCallsForRequest;
const getEffectiveContextLength = @import("admission.zig").getEffectiveContextLength;
const logStreamError = @import("../http/errors.zig").logStreamError;
const sendGenerationError = @import("../http/errors.zig").sendGenerationError;
const checkAttentionMemory = @import("admission.zig").checkAttentionMemory;
const clampMaxTokens = @import("sampling.zig").clampMaxTokens;
const mediaRejectReason = @import("media_gate.zig").mediaRejectReason;
const dropMedia = @import("media_gate.zig").dropMedia;
const imageUrlOfPart = @import("media_gate.zig").imageUrlOfPart;
const encodeText = @import("tokenize.zig").encodeText;
const jsonEscapeOrEmpty = @import("tokenize.zig").jsonEscapeOrEmpty;
const effortWordOnly = @import("../chat/reasoning/request.zig").effortWordOnly;
const implicitEffortBudgetFor = @import("../chat/reasoning/request.zig").implicitEffortBudgetFor;
const defaultReasoningBudget = @import("../chat/reasoning/request.zig").defaultReasoningBudget;
const forcedReasoningBudget = @import("../chat/reasoning/request.zig").forcedReasoningBudget;
const resolveChatThinking = @import("../chat/reasoning/request.zig").resolveChatThinking;
const schemaMasksThinking = @import("../chat/reasoning/request.zig").schemaMasksThinking;
const continueFinalMessageRequested = @import("request_text.zig").continueFinalMessageRequested;
const continuationRejectReason = @import("request_text.zig").continuationRejectReason;
const joinedTextParts = @import("request_text.zig").joinedTextParts;
const requestCacheKey = @import("request_text.zig").requestCacheKey;
const echoModelName = @import("request_text.zig").echoModelName;
const nChoicesRejectReason = @import("completions.zig").nChoicesRejectReason;
const requestPld = @import("request_options.zig").requestPld;
const answerStopCut = @import("../chat/stops.zig").answerStopCut;
const visibleToolPreamble = @import("../chat/stops.zig").visibleToolPreamble;
const nonStreamingViaScheduler = @import("submit.zig").nonStreamingViaScheduler;
const requestEnableDrafter = @import("request_options.zig").requestEnableDrafter;
const requestHasCompany = @import("request_options.zig").requestHasCompany;
const requestSpecModes = @import("request_options.zig").requestSpecModes;
const handleStreamingGeneration = @import("chat_stream.zig").handleStreamingGeneration;
const StreamLogprobs = @import("logprobs.zig").StreamLogprobs;
const sendSSEChunk = @import("../http/sse.zig").sendSSEChunk;
const sendResponse = @import("../http/respond.zig").sendResponse;
const extractJsonField = @import("../http/errors.zig").extractJsonField;
const countRejected = @import("../http/errors.zig").countRejected;
const contextOverflowMessage = @import("../http/errors.zig").contextOverflowMessage;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const cachedFormatChat = @import("format.zig").cachedFormatChat;
const speech_stream = @import("speech_stream.zig");
const speech_api = @import("speech.zig");
const formatChatUsage = @import("format.zig").formatChatUsage;
const formatTimingsObject = @import("format.zig").formatTimingsObject;
const formatPerfBracket = @import("format.zig").formatPerfBracket;
const jsonEscape = @import("format.zig").jsonEscape;
const contentTokenRange = @import("logprobs.zig").contentTokenRange;
const formatLogprobsObject = @import("logprobs.zig").formatLogprobsObject;
const parseKvQuantOverride = @import("sampling.zig").parseKvQuantOverride;
const MropeData = @import("../media/encode.zig").MropeData;
const MultimodalPrompt = @import("../media/encode.zig").MultimodalPrompt;
const prepareMultimodalPrompt = @import("../media/encode.zig").prepareMultimodalPrompt;
const multimodalFailure = @import("../media/encode.zig").multimodalFailure;
const parseAudioInput = @import("../media/encode.zig").parseAudioInput;
const AUDIO_DECODE_REJECT = @import("../media/request_media.zig").AUDIO_DECODE_REJECT;
const visionPreprocFromConfig = @import("../media/prepare.zig").visionPreprocFromConfig;
const RequestMedia = @import("../media/request_media.zig").RequestMedia;
const appendImageUrlContent = @import("../media/request_media.zig").appendImageUrlContent;
const IMAGE_DECODE_REJECT = @import("../media/request_media.zig").IMAGE_DECODE_REJECT;
const appendVideoUrlContent = @import("../media/decode.zig").appendVideoUrlContent;
const sendAnthropicEvent = @import("anthropic.zig").sendAnthropicEvent;
const emitAnthropicTextDelta = @import("anthropic_stream.zig").emitAnthropicTextDelta;
const emitAnthropicThinkingDelta = @import("anthropic_stream.zig").emitAnthropicThinkingDelta;
const closeAnthropicThinkingBlock = @import("anthropic_stream.zig").closeAnthropicThinkingBlock;
const emitResponsesReasoningStart = @import("responses/envelope.zig").emitResponsesReasoningStart;
const emitResponsesReasoningDelta = @import("responses/envelope.zig").emitResponsesReasoningDelta;
const emitResponsesMessageStart = @import("responses/envelope.zig").emitResponsesMessageStart;
const emitResponsesMessageDelta = @import("responses/envelope.zig").emitResponsesMessageDelta;

pub fn handleChatCompletions(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    // NOTE: no `lm.transformer.?` here — this handler also serves engine-backed
    // models (GGUF/llama, ds4) whose `transformer` is null. The only MLX-specific
    // gate below reads `config.has_hybrid_layers` (valid for every model incl. the
    // GGUF stub), so the transformer is never needed at this level.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    // Parse JSON body
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/chat/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    // A valid-JSON body that isn't an object (e.g. `42` or `[1,2]`) would panic
    // on the `.object` field access below and take the whole server down, so
    // reject it as a 400 like any other malformed request.
    if (parsed.value != .object) {
        log.warn("POST /v1/chat/completions -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    if (nChoicesRejectReason(root)) |reason| {
        log.warn("POST /v1/chat/completions -> 400 (unsupported n)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }

    // Extract messages
    const messages_val = root.get("messages") orelse {
        log.warn("POST /v1/chat/completions -> 400 (missing messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' is a required field", 400);
        return;
    };

    // `messages` present but not an array (e.g. a string) would panic on the
    // `.array` access below. Reject rather than crash.
    if (messages_val != .array) {
        log.warn("POST /v1/chat/completions -> 400 (messages is not an array)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' must be an array", 400);
        return;
    }

    var messages = std.ArrayList(chat_mod.Message).empty;
    var image_decode_failed = false;
    var audio_decode_failed = false;
    defer messages.deinit(allocator);

    // Decoded image/video/audio buffers for every message in this request.
    // `Message` borrows them; this is the only thing that frees them.
    var media = RequestMedia.init(allocator);
    defer media.deinit();

    // Parse tool call structs for assistant messages (stored temporarily)
    var tool_call_lists = std.ArrayList([]const chat_mod.ToolCall).empty;
    defer {
        for (tool_call_lists.items) |tcs| allocator.free(tcs);
        tool_call_lists.deinit(allocator);
    }

    // Joined multi-part text content (allocated only when a message carries
    // 2+ text parts).
    var content_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (content_allocs.items) |s| allocator.free(s);
        content_allocs.deinit(allocator);
    }

    for (messages_val.array.items) |msg_val| {
        // A non-object array element (e.g. `messages:[1,2,3]`) would panic on
        // `.object`. Skip it rather than crash — consistent with how malformed
        // inner fields are already tolerated below.
        if (msg_val != .object) continue;
        const obj = msg_val.object;
        const role_val = obj.get("role") orelse continue;
        if (role_val != .string) continue;

        // Content can be null for assistant messages with tool_calls
        const content_val = obj.get("content");
        var msg_images: ?[]const chat_mod.ImageData = null;
        var msg_videos: ?[]const chat_mod.VideoData = null;
        var msg_audio: ?[]const chat_mod.AudioData = null;
        const content: []const u8 = if (content_val) |cv| switch (cv) {
            .string => |s| s,
            .array => |arr| blk: {
                const img_slot = try media.openImages();
                const vid_slot = try media.openVideos();
                const aud_slot = try media.openAudio();
                for (arr.items) |part| {
                    if (part != .object) continue;
                    const ptype = part.object.get("type") orelse continue;
                    if (ptype != .string) continue;
                    if (std.mem.eql(u8, ptype.string, "image_url")) {
                        const url = imageUrlOfPart(part.object) orelse {
                            image_decode_failed = true;
                            continue;
                        };
                        if (!appendImageUrlContent(allocator, media.images(img_slot), url, visionPreprocFromConfig(config))) image_decode_failed = true;
                    } else if (std.mem.eql(u8, ptype.string, "video_url")) {
                        // A video is, on the wire, an ordered array of already-
                        // decoded frame images — no video codec exists anywhere
                        // in this codebase, so frame extraction is the CLIENT's
                        // job. Each frame string is an ordinary image data URL.
                        const vid_obj = part.object.get("video_url") orelse continue;
                        if (vid_obj != .object) continue;
                        const frames_val = vid_obj.object.get("frames") orelse continue;
                        if (frames_val != .array) continue;
                        var frame_urls = std.ArrayList([]const u8).empty;
                        defer frame_urls.deinit(allocator);
                        for (frames_val.array.items) |f| {
                            if (f == .string) frame_urls.append(allocator, f.string) catch continue;
                        }
                        const before = media.videos(vid_slot).items.len;
                        appendVideoUrlContent(allocator, media.videos(vid_slot), frame_urls.items, visionPreprocFromConfig(config));
                        // The rate the client sampled the frames at (a plugin tower's timestamps read it).
                        if (vid_obj.object.get("fps")) |f| if (media.videos(vid_slot).items.len > before) {
                            const rate: f64 = switch (f) {
                                .float => |x| x,
                                .integer => |x| @floatFromInt(x),
                                else => 0,
                            };
                            if (rate > 0 and rate <= 1000) media.videos(vid_slot).items[before].fps = @floatCast(rate);
                        };
                    } else if (std.mem.eql(u8, ptype.string, "input_audio")) {
                        // OpenAI's audio part (`data` + `format`), decoded to 16 kHz mono f32 (`parseAudioInput`).
                        const a_obj = part.object.get("input_audio") orelse continue;
                        if (a_obj != .object) continue;
                        const data_val = a_obj.object.get("data") orelse continue;
                        if (data_val != .string) continue;
                        const format: ?[]const u8 = if (a_obj.object.get("format")) |f| (if (f == .string) f.string else null) else null;
                        if (parseAudioInput(allocator, data_val.string, format)) |aud| {
                            media.audio(aud_slot).append(allocator, aud) catch {
                                allocator.free(aud.samples);
                                continue;
                            };
                        } else audio_decode_failed = true;
                    }
                }
                msg_images = media.imagesSlice(img_slot);
                msg_videos = media.videosSlice(vid_slot);
                msg_audio = media.audioSlice(aud_slot);
                const joined = try joinedTextParts(allocator, arr.items);
                if (joined.owned) try content_allocs.append(allocator, joined.text);
                break :blk joined.text;
            },
            .null => "",
            else => "",
        } else "";

        // Parse tool_calls from assistant messages
        var msg_tool_calls: ?[]const chat_mod.ToolCall = null;
        if (std.mem.eql(u8, role_val.string, "assistant")) {
            if (obj.get("tool_calls")) |tc_val| {
                if (tc_val == .array) {
                    var tcs = std.ArrayList(chat_mod.ToolCall).empty;
                    for (tc_val.array.items) |tc_item| {
                        if (tc_item != .object) continue;
                        const tc_id = if (tc_item.object.get("id")) |v| (if (v == .string) v.string else "") else "";
                        const func = tc_item.object.get("function") orelse continue;
                        if (func != .object) continue;
                        const fn_name = if (func.object.get("name")) |v| (if (v == .string) v.string else "") else "";
                        const fn_args = if (func.object.get("arguments")) |v| (if (v == .string) v.string else "{}") else "{}";
                        try tcs.append(allocator, .{ .id = tc_id, .name = fn_name, .arguments = fn_args });
                    }
                    if (tcs.items.len > 0) {
                        const owned = try tcs.toOwnedSlice(allocator);
                        try tool_call_lists.append(allocator, owned);
                        msg_tool_calls = owned;
                    } else {
                        tcs.deinit(allocator);
                    }
                }
            }
        }

        // Parse tool_call_id from tool messages
        const tool_call_id: ?[]const u8 = if (std.mem.eql(u8, role_val.string, "tool"))
            (if (obj.get("tool_call_id")) |v| (if (v == .string) v.string else null) else null)
        else
            null;

        // Reasoning the client round-trips on assistant history. Dropping it
        // starves templates that persist reasoning across turns (laguna): every
        // prior turn renders the empty <think></think> nothink signature and
        // the model stops thinking from turn 2 of a session.
        const msg_reasoning: ?[]const u8 = if (std.mem.eql(u8, role_val.string, "assistant"))
            messageReasoningFromObj(obj)
        else
            null;

        // A spoken reply comes back as its id (OpenAI's history shape): it replays as its transcript.
        var spoken = content;
        if (content.len == 0 and std.mem.eql(u8, role_val.string, "assistant")) if (obj.get("audio")) |au| if (au == .object) if (au.object.get("id")) |aid| {
            const text = if (aid == .string) try speech_api.store.get(stream.io, allocator, aid.string, nowSecs(stream.io)) else null;
            if (text) |t| {
                try content_allocs.append(allocator, t);
                spoken = t;
            } else {
                log.warn("POST /v1/chat/completions -> 400 (unknown audio id)\n", .{});
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", speech_api.UNKNOWN_AUDIO_ID, 400);
                return;
            }
        };
        const content_out = spoken;

        // Skip messages with no content, no tool_calls, and no images/videos/audio
        if (content_out.len == 0 and msg_tool_calls == null and msg_images == null and msg_videos == null and msg_audio == null and msg_reasoning == null and !std.mem.eql(u8, role_val.string, "tool")) continue;

        try messages.append(allocator, .{
            .role = chat_mod.canonicalRole(role_val.string),
            .content = content_out,
            .tool_calls = msg_tool_calls,
            .tool_call_id = tool_call_id,
            .images = msg_images,
            .videos = msg_videos,
            .audio = msg_audio,
            .reasoning_content = msg_reasoning,
        });
    }

    if (image_decode_failed) {
        log.warn("POST /v1/chat/completions -> 400 (undecodable image)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", IMAGE_DECODE_REJECT, 400);
        return;
    }
    if (audio_decode_failed) {
        log.warn("POST /v1/chat/completions -> 400 (undecodable audio)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", AUDIO_DECODE_REJECT, 400);
        return;
    }
    if (messages.items.len == 0) {
        log.warn("POST /v1/chat/completions -> 400 (no valid messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "No valid messages found in request", 400);
        return;
    }

    // Support both max_tokens and max_completion_tokens (OpenAI alias). Absent
    // or <= 0 → auto (peg to remaining context).
    const requested_max_tokens = root.get("max_tokens") orelse root.get("max_completion_tokens");
    const gen = requestProfile(stream.io, lm);
    const max_tokens: u32 = generationMaxTokens(
        gen,
        requested_max_tokens,
        omittedMaxTokensDefault(getEffectiveContextLength(config)),
    );
    // For the request log line only: which of the three sources this came from.
    const max_tokens_origin = maxTokensOrigin(requested_max_tokens, launchMaxTokensDefault());

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;
    const speech_ask: ?speech_api.Ask = switch (speech_api.parse(root, config.speech() != null, is_stream)) {
        .none => null,
        .ask => |ask| ask,
        .reject => |why| {
            log.warn("POST /v1/chat/completions -> 400 ({s})\n", .{why});
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, 400);
            return;
        },
    };

    const sampled = resolveSampling(gen, root, config);
    const temperature = sampled.temperature;
    const top_p = sampled.top_p;
    const top_k = sampled.top_k;
    const min_p = sampled.min_p;

    const repeat_penalty = sampled.repeat_penalty;

    const presence_penalty = sampled.presence_penalty;

    const seed: ?u64 = parseRequestSeed(root.get("seed"));

    // Parse logprobs: "logprobs": true, "top_logprobs": N (0..MAX_TOP_LOGPROBS)
    const logprobs_n: u32 = blk: {
        const lp = root.get("logprobs") orelse break :blk 0;
        if (lp != .bool or !lp.bool) break :blk 0;
        // logprobs=true without top_logprobs defaults to 0 (just the chosen token's logprob)
        const tlp = root.get("top_logprobs") orelse break :blk 1;
        break :blk switch (tlp) {
            .integer => |i| @intCast(@min(@max(i, 0), request_mod.MAX_TOP_LOGPROBS)),
            else => 1,
        };
    };

    // Extract tools JSON from request body for chat template injection
    var tools_json: ?[]const u8 = null;
    var has_tools = root.get("tools") != null;
    var forced_tool: ?chat_mod.ForcedTool = null;

    // OpenAI parallel_tool_calls: only an explicit false clamps to one call
    // per response (the SDK sets false in strict structured-output mode).
    const allow_parallel_tools = if (root.get("parallel_tool_calls")) |v|
        !(v == .bool and !v.bool)
    else
        true;

    if (has_tools) {
        // Parse tool_choice: "none" | "auto" | "required" | {"type":"function","function":{"name":"..."}}
        if (root.get("tool_choice")) |tc| {
            if (tc == .string) {
                if (std.mem.eql(u8, tc.string, "none")) {
                    has_tools = false; // Don't inject tools at all
                } else if (std.mem.eql(u8, tc.string, "required")) {
                    forced_tool = .any;
                }
                // "auto" is the default behavior
            } else if (tc == .object) {
                // Specific function: {"type":"function","function":{"name":"fn_name"}}
                if (tc.object.get("function")) |func| {
                    if (func == .object) {
                        if (func.object.get("name")) |name_val| {
                            if (name_val == .string) forced_tool = .{ .name = name_val.string };
                        }
                    }
                }
            }
        }

        if (has_tools) {
            // Find the tools array in the raw JSON body and extract it
            if (extractJsonField(body, "tools")) |tools_str| {
                tools_json = tools_str;
            }
        }
    }
    // Refused before the prompt or any media is built: only a card that speaks tool turns takes both.
    if (speech_ask != null and has_tools and !config.speech().?.tools) {
        log.warn("POST /v1/chat/completions -> 400 ({s})\n", .{speech_api.TEXT_TOOLS});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", speech_api.TEXT_TOOLS, 400);
        return;
    }
    if (forced_tool) |ft| if (ft == .name and !chat_mod.toolsDeclare(root.get("tools"), ft.name)) {
        const msg = try std.fmt.allocPrint(allocator, "tool_choice names function \"{s}\", which is not in tools", .{ft.name});
        defer allocator.free(msg);
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        return;
    };
    const tool_choice_instruction: ?[]const u8 = if (forced_tool) |ft| try chat_mod.toolChoiceInstruction(allocator, ft) else null;
    defer if (tool_choice_instruction) |tci| allocator.free(tci);

    // Parse stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |stop_val| {
        switch (stop_val) {
            .string => |s| if (s.len > 0) try stop_sequences.append(allocator, s),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .string and item.string.len > 0) try stop_sequences.append(allocator, item.string);
                }
            },
            else => {},
        }
    }

    // Parse model name from request (use for response, fallback to config)
    const model_name = echoModelName(root, config.model_type);

    // Track allocations from response_format injection so we can free them
    var rf_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (rf_allocs.items) |a| allocator.free(a);
        rf_allocs.deinit(allocator);
    }

    // Parse response_format — inject JSON schema constraint into system message,
    // and capture the schema value for grammar-constrained sampling below.
    var grammar_schema_val: ?std.json.Value = null;
    // Backing storage for the synthetic `{"type":"object"}` schema used for
    // `response_format: {type: "json_object"}`. Lives for the request and is
    // freed at the bottom; the schema parser arena copies what it needs.
    var json_object_schema_holder: ?std.json.Parsed(std.json.Value) = null;
    defer if (json_object_schema_holder) |*p| p.deinit();
    if (root.get("response_format")) |rf| {
        if (rf == .object) {
            const rf_type = if (rf.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, rf_type, "json_schema")) {
                const declared: ?std.json.Value = if (rf.object.get("json_schema")) |js| (if (js == .object) js.object.get("schema") else null) else null;
                if (declared == null or declared.? != .object) {
                    try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "response_format.json_schema.schema must be a JSON object (use {\"type\":\"json_object\"} for unconstrained JSON)", 400);
                    return;
                }
                // Extract the schema JSON string from the raw body
                var schema_instruction = std.ArrayList(u8).empty;
                defer schema_instruction.deinit(allocator);
                try schema_instruction.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");

                if (rf.object.get("json_schema")) |js| {
                    if (js == .object) {
                        if (js.object.get("schema")) |schema_val| {
                            grammar_schema_val = schema_val;
                            var out: std.Io.Writer.Allocating = .init(allocator);
                            defer out.deinit();
                            var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
                            schema_val.jsonStringify(&jws) catch {};
                            try schema_instruction.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
                            try schema_instruction.appendSlice(allocator, out.written());
                        }
                    }
                }

                const instruction = try allocator.dupe(u8, schema_instruction.items);
                try rf_allocs.append(allocator, instruction);
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
                }
            } else if (std.mem.eql(u8, rf_type, "json_object")) {
                const instruction = "Respond with valid JSON only. No other text, no markdown fences (no ``` or ```json), no explanation. Begin your response with `{` or `[`.";
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
                }
                // Belt + braces: also constrain decoding with a permissive
                // object schema so the very first token cannot be a leading
                // backtick (Gemma 4 ignores the prompt-side instruction
                // otherwise). `additionalProperties: true` allows any keys
                // and values — `json_object` only constrains JSON-ness, not
                // a specific shape.
                if (json_object_schema_holder == null) {
                    json_object_schema_holder = std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"object\",\"additionalProperties\":true}", .{}) catch null;
                }
                if (json_object_schema_holder) |*p| {
                    grammar_schema_val = p.value;
                }
            }
        }
    }

    // Parse stream_options
    const include_usage = if (root.get("stream_options")) |so| blk: {
        if (so != .object) break :blk false;
        if (so.object.get("include_usage")) |iu| {
            break :blk iu == .bool and iu.bool;
        }
        break :blk false;
    } else false;

    // Thinking opt-ins: the OpenAI-standard `reasoning_effort` string and the
    // vendor `enable_thinking` bool (Qwen/vLLM chat_template_kwargs family).
    // Either switch turns thinking on; effort "none" alone never does.
    // A request naming NEITHER takes the arch default (off for every arch but
    // the ones whose vendor documents thinking-on).
    const default_budget = defaultReasoningBudget(gen);
    const thinking = resolveChatThinking(root, gen, config.defaultEnableThinking(tools_json != null), default_budget, effortWordOnly(allocator, lm, tok));
    const effort_cfg = thinking.effort;
    // The card's spoken recipes open no think block (`handleSpeech` likewise): the reply speaks from its first slot.
    var enable_thinking = thinking.enable and speech_ask == null;

    // Reasoning budget (max tokens in <think> block, -1 = unlimited):
    // explicit reasoning_budget_tokens > effort-mapped budget > --reasoning-budget flag
    const effort_budget: i32 = if (effort_cfg) |e| e.budget else implicitEffortBudgetFor(allocator, lm, tok, default_budget);
    const reasoning_budget: i32 = forcedReasoningBudget(gen, if (root.get("reasoning_budget_tokens")) |v| switch (v) {
        .integer => |i| clampJsonI32(i),
        else => effort_budget,
    } else effort_budget);

    // Wave 1.A: per-request KV-quant override. When unset, falls back to the
    // process-level --kv-quant default carried on the scheduler. Cross-scheme
    // hot-prefix-cache hits never happen — entries record their scheme and
    // findBestMatch filters on it.
    const kv_quant_override = parseKvQuantOverride(root);
    if (kv_quant_override) |kq| {
        switch (kq.scheme) {
            .off => log.info("  kv-quant override: off (per-request)\n", .{}),
            .affine => log.info("  kv-quant override: affine {d}-bit (per-request)\n", .{kq.bits}),
        }
    }
    const kv_attn_explicit = parseKvAttnExplicit(root);

    // Parse enable_pld: per-request override of the --pld default.
    //
    // `pld_explicit_in_json` records whether the request body had `enable_pld`
    // as an explicit field (vs falling back to the server default). The
    // adaptive gate (later, after prompt is tokenized) only disables flags
    // that came from the default — explicit user overrides are honored even
    // on novel content.
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld: bool = requestPld(root, lm);
    // Tools do NOT disable PLD: tool-pattern detection operates on emitted
    // text and is agnostic to how many tokens a decode step yields, and
    // agent traffic (tool results echoed into edits) is PLD's best workload.
    // The original blanket gate predated streaming PLD + scheduler slots;
    // equivalence with tools is pinned by tests/test_pld_tools.sh.
    //
    // PLD on hybrid SSM models (LFM2.5, Nemotron-H) works once the SSM
    // snapshot/restore paths handle null ssm_state correctly — see
    // `ssmSnapshot`/`ssmRestore` in transformer.zig.
    if (enable_pld) log.info("  pld=enabled (draft_len={d}, key_len={d})\n", .{ http_server.serverState().server_config.default_pld_draft_len, http_server.serverState().server_config.default_pld_key_len });

    // Tools do not disable the drafter; logprobs require the target's plain path.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter = requestEnableDrafter(root.get("enable_drafter"), lm.drafter != null, lm.dflash != null, config.isMoe(), config.has_hybrid_layers);
    if (enable_drafter and logprobs_n > 0) {
        log.info("  drafter=disabled (logprobs requested)\n", .{});
        enable_drafter = false;
    }
    // Qwen native MTP head: defaults ON whenever the model loaded one (the
    // sidecar only loads when it binds to this trunk; MoE mirrors the
    // drafter caution). Priority DFlash > MTP > gemma drafter > PLD at
    // dispatch (`requestSpecModes`). NOT
    // subject to the n-gram spec gate below — the trained head holds ~73%
    // per-draft acceptance even on fully novel content.
    const allow_batch_mtp = if (root.get("enable_batch_mtp")) |v| v != .bool or v.bool else true;
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        defaultEnableMtp(lm.mtp != null, dsv4DraftStages(lm));
    if (enable_mtp and lm.mtp == null and !dsv4DraftStages(lm)) enable_mtp = false;
    if (enable_mtp and logprobs_n > 0) {
        log.info("  mtp=disabled (logprobs requested)\n", .{});
        enable_mtp = false;
    }
    if (enable_drafter) {
        if (enable_pld) {
            log.info("  pld=disabled (drafter takes priority for this request)\n", .{});
            enable_pld = false;
        }
        log.info("  drafter=enabled (block_size={d})\n", .{lm.drafter_block_size});
    }

    // Log the request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);

    // Compute sizes for debug info
    var system_chars: usize = 0;
    var user_chars: usize = 0;
    var tool_msg_count: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            system_chars += msg.content.len;
        } else if (std.mem.eql(u8, msg.role, "user")) {
            user_chars += msg.content.len;
        } else if (std.mem.eql(u8, msg.role, "tool")) {
            tool_msg_count += 1;
        }
    }
    const tools_len = if (tools_json) |tj| tj.len else 0;

    // Format chat template. ds4-backed models render through the engine's
    // built-in template/tokenizer; the MLX path renders via Jinja and
    // tokenizes through the loaded BPE tokenizer. Both paths now thread
    // `tools_json` + `tool_choice_instruction` through — the ds4 helper
    // synthesizes a system-message fallback when the GGUF chat template
    // doesn't model `tools` natively (which is the DSV4 case).
    //
    // Phase 4 instrumentation + Iteration 2 cache: time the render+tokenize
    // step. The cache is engine-agnostic — same hit even when the
    // underlying call is ds4 / llama / MLX formatChat.
    // Continuation is EXPLICIT on this surface (`continue_final_message`, the
    // spelling vLLM uses). It cannot be implied by a trailing assistant
    // message the way /v1/messages implies it: agent frameworks legitimately
    // POST assistant-last history here and expect a fresh turn, and turning
    // that into a prefill would change what every one of them gets back.
    // Asked for by name, so a model that cannot serve it is refused by name.
    // A spoken reply: the card's voice-chat recipe closes the system prompt (the tool list follows it).
    if (speech_ask != null) {
        const line = speech_api.recipe(has_tools);
        if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
            const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, line });
            try rf_allocs.append(allocator, combined);
            messages.items[0].content = combined;
        } else try messages.insert(allocator, 0, .{ .role = "system", .content = line, .tool_calls = null, .tool_call_id = null });
    }
    const continue_final = continueFinalMessageRequested(root, messages.items);
    if (continue_final) if (continuationRejectReason(lm.engineOwnsTemplate())) |reason| {
        log.warn("  -> 400 (continuation unsupported by the embedded engine)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    };
    if (lm.vision_encoder == null) {
        if (mediaRejectReason(messages.items, continue_final)) |reason| {
            log.warn("POST /v1/chat/completions -> 400 ({s})\n", .{reason});
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        try dropMedia(allocator, messages.items, &content_allocs);
    }
    // The request's `chat_template_kwargs` merge over the model's for this render only.
    var render_config = chat_config.*;
    const request_kwargs: ?[]const u8 = if (root.get("chat_template_kwargs")) |kw|
        (if (kw == .object) try chat_mod.mergeTemplateKwargs(allocator, chat_config.chat_template_kwargs, kw.object) else null)
    else
        null;
    defer if (request_kwargs) |k| allocator.free(k);
    if (request_kwargs) |k| render_config.chat_template_kwargs = k;
    var tokenize_sw = Stopwatch.init(stream.io);
    var prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, tools_json, tool_choice_instruction, enable_thinking, if (effort_cfg) |e| e.effort else null, continue_final);
    var schema_proto: rp_mod.Protocol = undefined;
    var schema_proto_active = false;
    if (grammar_schema_val != null and !has_tools and enable_thinking and reasoning_budget < 0) {
        schema_proto_active = resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, true);
    }
    switch (schemaMasksThinking(
        grammar_schema_val != null,
        has_tools,
        enable_thinking,
        reasoning_budget >= 0,
        schema_proto_active,
    )) {
        .deferred => {},
        .fallback_thinking_off => {
            schema_proto_active = false;
            allocator.free(prompt_ids_raw);
            enable_thinking = false;
            prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, tools_json, tool_choice_instruction, false, if (effort_cfg) |e| e.effort else null, continue_final);
            log.info("[grammar] {s}; rerendered with thinking off\n", .{if (reasoning_budget >= 0) "finite reasoning budget" else "reasoning protocol unsupported for this model/prompt"});
        },
        .no_mask, .token_zero => schema_proto_active = false,
    }
    if (grammar_schema_val != null and !has_tools and !enable_thinking and lm.transformer != null) {
        if (!resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, false)) {
            schema_proto = rp_mod.Protocol.jsonOnly();
        }
        schema_proto_active = true;
    }
    // The card prefills a voice-chat turn with `<|mixed_start|>`; with tools the model picks a call or speech itself.
    if (speech_ask != null and !has_tools) {
        prompt_ids_raw = try allocator.realloc(prompt_ids_raw, prompt_ids_raw.len + 1);
        prompt_ids_raw[prompt_ids_raw.len - 1] = config.speech().?.mixed_start;
    }
    const tokenize_ns = tokenize_sw.read();

    var max_tokens_desc_buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    log.info("POST /v1/chat/completions ({d} msgs, max_tokens={s}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}, thinking={}, sys={d}b, user={d}b, tools={d}b, tool_msgs={d}) \n", .{ messages.items.len, describeMaxTokens(&max_tokens_desc_buf, max_tokens, max_tokens_origin), temperature, top_p, top_k, is_stream, enable_thinking, system_chars, user_chars, tools_len, tool_msg_count });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    const cache_key = requestCacheKey(root);
    var mm: MultimodalPrompt = .{};
    defer mm.deinit(allocator);
    if (lm.vision_encoder != null) {
        mm = prepareMultimodalPrompt(allocator, lm, config, messages.items, &prompt_ids_raw) catch |err| {
            allocator.free(prompt_ids_raw);
            const f = multimodalFailure(err);
            log.warn("POST /v1/chat/completions -> {d} ({s})\n", .{ f.code, @errorName(err) });
            try sendErrorResponse(allocator, stream, f.status, f.kind, f.message, f.code);
            return;
        };
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);
    enable_mtp = admitMtpForCtx(enable_mtp, prompt_ids.len);

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/chat/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }

    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    // Check if attention computation would exceed GPU memory.
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, false, kv_quant_override, lm, enable_mtp, mm.bytes(config))) return;

    log.info("  prompt={d} tokens, max_gen={d}, ctx={d}\n", .{ prompt_ids.len, effective_max_tokens, effective_ctx });

    // ── Adaptive spec-decode gating ──
    //
    // PLD and drafter both pay a per-step overhead (lookup + verify forward
    // for PLD; drafter forwards + verify forward for drafter) that only pays
    // off on echo-heavy content. On novel content, the verify forward is wasted
    // work — the bench shows drafter at 0.55× on creative content and PLD at
    // 0.91× on LFM2.5-350M heavy-echo. To make the default-on flip safe we
    // gate per-request on the prompt's n-gram repetition score.
    //
    // Rule: if the score (= ratio of distinct 3-grams that recur in the prompt)
    // is below `spec_gate_threshold` AND the user did not explicitly set the
    // flag in the JSON, disable PLD/drafter for this request. Explicit user
    // overrides are honored — if you `enable_pld:true` on novel content, you
    // get PLD even though it's likely a perf loss; user knows best.
    if ((enable_pld and !pld_explicit_in_json) or (enable_drafter and !drafter_explicit_in_json)) {
        const score = pld_index.ngramRepeatScore(allocator, prompt_ids, 3) catch 1.0; // on error, don't gate
        log.info("  spec-gate: ngram-score={d:.3} (threshold={d:.3})\n", .{ score, spec_gate_threshold });
        if (score < spec_gate_threshold) {
            if (enable_pld and !pld_explicit_in_json) {
                log.info("  pld=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_pld = false;
            }
            // A DFlash drafter is exempt: its runtime yield gate disables on
            // REALIZED acceptance after `DFLASH_GATE_WARMUP` rounds (8 wasted
            // verifies at worst), strictly better evidence than a prompt-time heuristic
            // that cannot see output echo — and llmprobe/bench bodies cannot
            // carry enable_drafter:true. The gemma cross-attention drafter
            // (0.55x measured on novel) keeps the gate.
            if (enable_drafter and !drafter_explicit_in_json and lm.dflash == null) {
                log.info("  drafter=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_drafter = false;
            }
        }
        // NOTE: the old heavy-echo MTP->PLD routing (score >=
        // 0.13 disabled MTP) was RETIRED with the verify-qmm kernels + EV
        // depth. The prompt-time n-gram score cannot separate "output will
        // echo verbatim" (PLD excels: 83 vs 75 tok/s live) from
        // "repetitive-looking agent context" (PLD flaps at ~45% per-draft,
        // runtime-disables, and lands on plain AR at 28 tok/s with the MTP
        // head idle) — live captures scored 0.334 vs 0.365, inseparable.
        // MTP now wins whenever loaded (generator priority); force PLD with
        // enable_pld:true + enable_mtp:false.
    }

    // Prompt caching: reuse KV cache for shared prefix.
    // Force invalidation when images are present — image tokens have identical IDs
    // but different vision embeddings, so prefix matching would reuse stale features.
    const eos_slice = requestEosSlice(config, root);

    var sampling = request_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .min_p = min_p,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty,
        .frequency_penalty = sampled.frequency_penalty,
        .seed = seed,
    };

    // Build grammar-constrained sampling state if a JSON schema was supplied.
    // Lifetime is scoped to this request — the SchemaConstraint must NOT be
    // moved (the embedded Constraint holds pointers into it).
    var sc: request_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();

    if (grammar_schema_val) |sv| {
        if (has_tools) {
            log.info("[grammar] skipped JSON schema mask while tools are available (tool calls must remain reachable)\n", .{});
        } else {
            const tb = try lm.grammarTokenBytes(allocator, stream.io);
            if (sc.initFromValue(allocator, sv, tb)) {
                sc_init = true;
                var deferred = false;
                if (schema_proto_active) {
                    sc.deferWithProtocol(&schema_proto);
                    deferred = schema_proto.startState().phase != .json_body;
                }
                sampling.constraint = &sc.constraint;
                if (deferred) {
                    log.info("[grammar] deferring JSON schema until the reasoning boundary (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                } else {
                    log.info("[grammar] enforcing JSON schema from token 0 (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                }
            } else |err| {
                log.warn("[grammar] schema parse failed ({s}); falling back to prompt-only enforcement\n", .{@errorName(err)});
            }
        }
    }

    // A decode-time bound owns the budget; the surfaces then deliver the
    // whole (closed) thought instead of trimming it.
    var think_bound = armThinkBound(allocator, lm, tok, prompt_ids, enable_thinking, reasoning_budget);
    defer if (think_bound) |tb| allocator.free(tb.forced);
    const surface_budget: i32 = if (think_bound != null) -1 else reasoning_budget;
    if (think_bound) |*tb| sampling.think_bound = tb;

    var tool_force = armToolForce(allocator, lm, tok, prompt_ids, forced_tool);
    defer if (tool_force) |tf| allocator.free(tf.forced);
    if (tool_force) |*tf| sampling.tool_force = tf;
    // One token per tick: no draft round can carry the turn past the forced opener.
    if (tool_force != null) {
        enable_pld = false;
        enable_drafter = false;
        enable_mtp = false;
    }

    // Hand vision ownership off to the sub-handler, which transfers it to
    // the slot at submit time.
    const sub_ve = mm.embeddings;
    mm.embeddings = null;
    const sub_mrope = mm.mrope;
    mm.mrope = .{}; // ownership transferred to the sub-handler → slot
    if (is_stream and speech_ask != null) {
        speech_stream.handle(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, model_name, include_usage, if (has_tools) tools_json else null, allow_parallel_tools, sub_ve, mm.media, cache_key, sub_mrope, tokenize_ns) catch |err| {
            logStreamError(err);
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    } else if (is_stream) {
        handleStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, has_tools, tools_json, allow_parallel_tools, logprobs_n, enable_thinking, surface_budget, enable_pld, enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, mm.media, cache_key, sub_mrope, kv_quant_override, kv_attn_explicit, tokenize_ns) catch |err| {
            logStreamError(err);
            // One mapping with the non-streaming arm: an HTTP status before the SSE head, an SSE `error` event after.
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    } else {
        handleNonStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, allow_parallel_tools, logprobs_n, enable_thinking, surface_budget, enable_pld, enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, mm.media, cache_key, sub_mrope, kv_quant_override, kv_attn_explicit, tokenize_ns, speech_ask) catch |err| {
            log.err("  -> {s}\n", .{@errorName(err)});
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    }
}

/// Generation is authoritative for constrained responses: when the generator
/// published a payload boundary, the split is a byte cut at that offset —
/// reasoning is everything before it (delimiter markup stripped, whitespace
/// policy matched to the streaming suppression), content is everything after,
/// UNREParsed. A standalone re-scan of the payload would read marker text
/// inside JSON string data ("the close delimiter") as protocol structure and
/// truncate valid JSON. Without a boundary the existing parser applies.
pub fn emitConstrainedChat(a: std.mem.Allocator, stream: *Conn, id: i64, model: []const u8, lps: *StreamLogprobs, d: *rp_mod.Delivery) !void {
    if (d.reasoning.items.len > 0) {
        try sendSSEChunk(a, stream, id, model, .{ .role = null, .content = null, .reasoning_content = d.reasoning.items }, null, null, null, .{});
    }
    if (d.content.items.len > 0) {
        try sendSSEChunk(a, stream, id, model, .{ .role = null, .content = d.content.items }, null, null, null, .{ .logprobs_json = try lps.take() });
    } else lps.dropPending();
}

pub fn emitConstrainedAnthropic(a: std.mem.Allocator, stream: *Conn, index: *u32, thinking_open: *bool, text_open: *bool, d: *rp_mod.Delivery) !void {
    if (d.reasoning.items.len > 0) {
        if (!thinking_open.*) {
            const data = try std.fmt.allocPrint(a,
                \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
            , .{index.*});
            defer a.free(data);
            try sendAnthropicEvent(stream, "content_block_start", data);
            thinking_open.* = true;
        }
        try emitAnthropicThinkingDelta(a, stream, index.*, d.reasoning.items);
    }
    if (d.content.items.len > 0) {
        if (thinking_open.*) {
            try closeAnthropicThinkingBlock(a, stream, index.*);
            thinking_open.* = false;
            index.* += 1;
        }
        if (!text_open.*) {
            const data = try std.fmt.allocPrint(a,
                \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
            , .{index.*});
            defer a.free(data);
            try sendAnthropicEvent(stream, "content_block_start", data);
            text_open.* = true;
        }
        try emitAnthropicTextDelta(a, stream, index.*, d.content.items);
    }
}

pub fn emitConstrainedResponses(a: std.mem.Allocator, stream: *Conn, seq: *u64, reasoning_id: *?[]u8, reasoning_index: *u32, reasoning_started: *bool, message_id: *?[]u8, message_index: *u32, message_started: *bool, d: *rp_mod.Delivery) !void {
    if (d.reasoning.items.len > 0) {
        if (!reasoning_started.*) {
            reasoning_id.* = try responses_mod.makeId(stream.io, a, "rs");
            reasoning_index.* = 0;
            try emitResponsesReasoningStart(a, stream, seq, reasoning_index.*, reasoning_id.*.?);
            reasoning_started.* = true;
        }
        try emitResponsesReasoningDelta(a, stream, seq, reasoning_index.*, reasoning_id.*.?, d.reasoning.items);
    }
    if (d.content.items.len > 0) {
        if (!message_started.*) {
            message_id.* = try responses_mod.makeId(stream.io, a, "msg");
            message_index.* = if (reasoning_started.*) 1 else 0;
            try emitResponsesMessageStart(a, stream, seq, message_index.*, message_id.*.?);
            message_started.* = true;
        }
        try emitResponsesMessageDelta(a, stream, seq, message_index.*, message_id.*.?, d.content.items);
    }
}

pub fn splitConstrainedResponse(allocator: std.mem.Allocator, routed: *?rp_mod.Delivery, sampling: request_mod.SamplingParams, text: []const u8, payload_byte: ?usize, keep_markup: bool, opened: bool) !chat_mod.ThinkSplit {
    if (sampling.constraint) |c| {
        if (c.proto) |p| {
            routed.* = rp_mod.Delivery.init(p);
            const d = &routed.*.?;
            d.payload_byte = payload_byte;
            try d.feed(allocator, text);
            try d.finish(allocator);
            const reasoning = std.mem.trim(u8, d.reasoning.items, "\n ");
            return .{ .reasoning_content = if (reasoning.len == 0) null else reasoning, .content = d.content.items };
        }
    }
    return if (keep_markup) chat_mod.splitThinkBlockKeepingMarkup(text, true, opened) else chat_mod.splitThinkBlock(text, true, opened);
}

fn handleNonStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: request_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    /// false = client set parallel_tool_calls:false (Anthropic:
    /// tool_choice.disable_parallel_tool_use) — at most one call per response.
    allow_parallel_tools: bool,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    vision_embeddings: ?mlx.mlx_array,
    media: []const MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?kv_quant.KVQuantConfig,
    kv_attn_explicit: ?bool,
    /// Iteration 1: tokenize_ns from the parent handleChatCompletions, so
    /// the non-streaming chat response carries `timings.tokenize_ms`.
    tokenize_ns: u64,
    /// A spoken reply (`modalities` asked for audio).
    speech: ?speech_api.Ask,
) !void {
    // Phase A8: this handler owns the vision array on entry; ownership
    // transfers to the slot on submit (the scheduler's `Slot.deinit` frees
    // it). Nulled before transfer so the early-return defer is a no-op.
    var ve_local = vision_embeddings;
    defer {
        if (ve_local) |arr| _ = mlx.mlx_array_free(arr);
    }

    var timer = Stopwatch.init(stream.io);

    // Speculative-decoding dispatch (priority: drafter > PLD > regular).
    //   1. Drafter wins if loaded AND requested AND no logprobs (drafter
    //      cannot expose realized log-probs from the draft side).
    //   2. PLD next if requested AND no logprobs AND no grammar constraint
    //      (constrained decode requires per-token state advancement).
    //   3. Otherwise the regular pipeline.
    const spec = requestSpecModes(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.dflash != null, mtpCapable(lm), sampling.shapesLogits(), logprobs_n, requestHasCompany());
    const use_mtp = spec.use_mtp;
    const use_drafter = spec.use_drafter;
    const use_pld = spec.use_pld;

    // Transfer vision ownership into the slot via the scheduler.
    const slot_ve: ?mlx.mlx_array = blk: {
        const v = ve_local;
        ve_local = null;
        break :blk v;
    };
    // Propagates to `handleChatCompletions`' one error arm, shared with the streaming twin.
    const result = try nonStreamingViaScheduler(allocator, http_server.launch().global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, has_tools, enable_thinking, use_pld, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), slot_ve, media, cache_key, mrope, logprobs_n, false, kv_quant_override, kv_attn_explicit, if (speech != null) .{} else null, stream);
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    defer if (result.logprobs) |lps| {
        for (lps) |*lp| allocator.free(lp.top_logprobs);
        allocator.free(lps);
    };
    defer if (result.audio_samples) |x| allocator.free(x);
    defer if (result.audio_codes) |x| allocator.free(x);

    // Template-opened think block (Qwen 3.5/3.6): unclosed output is reasoning.
    // Not ANDed with enable_thinking: generated reasoning is always DELIVERED,
    // never paid-for-and-dropped — thinking-off is enforced on the PROMPT side
    // (chat.noThinkTailSuffix commits the channel), so any reasoning that
    // still shows up here is real work the client gets to see.
    const opens_think = promptOpensThink(allocator, lm, tok, prompt_ids);
    const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;

    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    if (answerStopCut(result.text, result.text.len, stop_sequences, opens_think and !constrained_proto)) |cut| {
        final_text = result.text[0..cut.index];
        finish_reason = "stop";
    }

    // Merge re-opened mid-text thought channels into the leading block so the
    // split/parse below never leaks raw tags (Gemma 12B re-opens channels
    // mid-turn). A span-routed constrained response skips this: its payload
    // is authoritative and marker text inside it is DATA.
    var normalized_text: ?[]u8 = null;
    defer if (normalized_text) |n| allocator.free(n);
    if (sampling.constraint == null or sampling.constraint.?.proto == null) {
        normalized_text = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, final_text);
        if (normalized_text) |n| final_text = n;
    }

    // Apply reasoning budget: truncate reasoning by token count
    // For non-streaming, we truncate after generation since we can't interrupt mid-generation
    var budget_truncated_reasoning: ?[]const u8 = null;
    var budget_reasoning_allocated = false;
    defer if (budget_reasoning_allocated) allocator.free(budget_truncated_reasoning.?);

    if (enable_thinking and reasoning_budget >= 0) {
        const think_split = if (has_tools) chat_mod.splitThinkBlock(final_text, true, opens_think) else chat_mod.splitThinkBlockKeepingMarkup(final_text, true, opens_think);
        if (think_split.reasoning_content) |reasoning| {
            // Count tokens in reasoning by encoding it
            const reasoning_ids = try tok.encode(allocator, reasoning);
            defer allocator.free(reasoning_ids);
            if (reasoning_ids.len > @as(usize, @intCast(reasoning_budget))) {
                // Truncate: decode only budget-many tokens
                const budget_usize: usize = @intCast(reasoning_budget);
                const truncated_ids = reasoning_ids[0..budget_usize];
                const truncated_text = try tok.decode(allocator, truncated_ids, false);
                budget_truncated_reasoning = truncated_text;
                budget_reasoning_allocated = true;
                log.info("  reasoning budget truncated ({d}/{d} tokens)\n", .{ budget_usize, reasoning_ids.len });
            }
        }
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    // Check for tool calls in the output
    if (has_tools and shouldParseToolCalls(result.finish_details)) {
        log.debug("  checking {d} bytes of generated text for tool calls\n", .{final_text.len});
        const found_calls = try parseToolCallsForRequest(allocator, final_text, tools_json, allow_parallel_tools);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            var tc_perf_buf: [160]u8 = undefined;
            const tc_perf = formatPerfBracket(&tc_perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
            log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [tool_calls: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, tc_perf, tool_calls.len,
            });

            // Build tool_calls JSON array
            var tc_buf = std.ArrayList(u8).empty;
            defer tc_buf.deinit(allocator);
            try tc_buf.appendSlice(allocator, "[");
            for (tool_calls, 0..) |tc, i| {
                if (i > 0) try tc_buf.appendSlice(allocator, ",");
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const escaped_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(escaped_name);
                const escaped_args = try jsonEscape(allocator, tc.arguments);
                defer allocator.free(escaped_args);
                const tc_json = try std.fmt.allocPrint(allocator,
                    \\{{"id":"{s}","type":"function","function":{{"name":{s},"arguments":{s}}}}}
                , .{ tc_id, escaped_name, escaped_args });
                defer allocator.free(tc_json);
                try tc_buf.appendSlice(allocator, tc_json);
            }
            try tc_buf.appendSlice(allocator, "]");

            // Reasoning is delivered whenever the model produced it — the
            // request's thinking flag shaped the prompt, not the delivery.
            const tc_think_split = chat_mod.splitThinkBlock(final_text, true, opens_think);
            var tc_reasoning_json: []const u8 = "";
            var tc_reasoning_allocated = false;
            {
                if (tc_think_split.reasoning_content) |reasoning| {
                    const escaped_reasoning = try jsonEscape(allocator, reasoning);
                    tc_reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
                    allocator.free(escaped_reasoning);
                    tc_reasoning_allocated = true;
                }
            }
            defer if (tc_reasoning_allocated) allocator.free(tc_reasoning_json);

            // The same pre-markup text the stream flushes as a text delta.
            const tc_preamble = visibleToolPreamble(tc_think_split.content);
            const tc_content_esc = jsonEscapeOrEmpty(allocator, tc_preamble);
            defer if (tc_content_esc.owned) allocator.free(tc_content_esc.slice);
            const tc_content_json: []const u8 = if (tc_preamble.len > 0) tc_content_esc.slice else "null";

            const tc_timings = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
            defer allocator.free(tc_timings);
            const tc_timings_field = if (tc_timings.len > 0)
                try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{tc_timings})
            else
                try allocator.alloc(u8, 0);
            defer allocator.free(tc_timings_field);

            const tc_usage_obj = try formatChatUsage(allocator, result.prompt_tokens, result.completion_tokens, result.cached_tokens, "");
            defer allocator.free(tc_usage_obj);

            const response = try std.fmt.allocPrint(allocator,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s},"tool_calls":{s}}},"finish_reason":"{s}"{s}}}],"usage":{s}{s}}}
            , .{
                nowMs(stream.io),
                nowSecs(stream.io),
                model_name,
                tc_content_json,
                tc_reasoning_json,
                tc_buf.items,
                toolCallFinishReason(finish_reason),
                finishDetailsField(toolCallFinishReason(finish_reason), result.finish_details),
                tc_usage_obj,
                tc_timings_field,
            });
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
            return;
        }
    }

    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, finish_reason,
    });

    // Split thinking content from response. Always the think-capable split:
    // whatever reasoning was generated ships as reasoning_content, never
    // stripped (tokens we discarded still counted against tok/s).
    var routed: ?rp_mod.Delivery = null;
    defer if (routed) |*d| d.deinit(allocator);
    const think_split = try splitConstrainedResponse(allocator, &routed, sampling, final_text, result.constraint_payload_byte, !has_tools, opens_think);
    const content_text = think_split.content;

    const escaped = jsonEscapeOrEmpty(allocator, content_text);
    const escaped_text = escaped.slice;
    defer if (escaped.owned) allocator.free(escaped.slice);

    // Build logprobs JSON if requested. The array describes `message.content`,
    // so it stops at the tokens that survived the think split — see
    // `contentTokenRange`.
    var logprobs_json: []const u8 = "null";
    var logprobs_allocated = false;
    if (result.logprobs) |lps| {
        const n = @min(result.token_ids.len, lps.len);
        const r = contentTokenRange(allocator, tok, result.token_ids[0..n], final_text, content_text);
        const a = @min(r.start, n);
        const b = @min(@max(r.end, a), n);
        logprobs_json = try formatLogprobsObject(allocator, tok, result.token_ids[a..b], lps[a..b]);
        logprobs_allocated = true;
    }
    defer if (logprobs_allocated) allocator.free(logprobs_json);

    // Build reasoning_content whenever reasoning exists — delivery does not
    // key on the request's thinking flag.
    var reasoning_json: []const u8 = "";
    var reasoning_allocated = false;
    var usage_details_json: []const u8 = "";
    var usage_details_allocated = false;
    {
        // Use budget-truncated reasoning if available, otherwise use full reasoning
        const reasoning_text = if (budget_truncated_reasoning) |tr| tr else think_split.reasoning_content;
        if (reasoning_text) |reasoning| {
            const escaped_reasoning = try jsonEscape(allocator, reasoning);
            reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
            allocator.free(escaped_reasoning);
            reasoning_allocated = true;
            // usage.completion_tokens_details.reasoning_tokens (OpenAI/LM Studio
            // parity) so clients can budget visible content separately.
            if (encodeText(allocator, lm, tok, reasoning, false)) |rids| {
                defer allocator.free(rids);
                usage_details_json = try std.fmt.allocPrint(allocator, ",\"completion_tokens_details\":{{\"reasoning_tokens\":{d}}}", .{rids.len});
                usage_details_allocated = true;
            } else |_| {}
        }
    }
    defer if (reasoning_allocated) allocator.free(reasoning_json);
    defer if (usage_details_allocated) allocator.free(usage_details_json);

    const timings_obj = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    const timings_field = if (timings_obj.len > 0)
        try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{timings_obj})
    else
        try allocator.alloc(u8, 0);
    defer allocator.free(timings_field);

    // A spoken reply is OpenAI's `message.audio`: the speech, its id, and the text as its transcript.
    var audio_field: []const u8 = "";
    defer if (audio_field.len > 0) allocator.free(audio_field);
    if (speech) |ask| {
        const data = try speech_api.encode(allocator, result.audio_samples orelse &.{}, result.sample_rate, ask.format);
        defer allocator.free(data);
        const frames = if (result.audio_codes) |c| c.len / lm.config.?.speech().?.codebooks else 0;
        if (!usage_details_allocated) {
            usage_details_json = try std.fmt.allocPrint(allocator, ",\"completion_tokens_details\":{{\"audio_tokens\":{d}}}", .{frames});
            usage_details_allocated = true;
        }
        const audio_id = speech_api.Store.newId(stream.io);
        try speech_api.store.put(stream.io, audio_id, nowSecs(stream.io), content_text);
        audio_field = try std.fmt.allocPrint(allocator, ",\"audio\":{{\"id\":\"{s}\",\"data\":\"{s}\",\"expires_at\":{d},\"transcript\":{s}}}", .{ &audio_id, data, nowSecs(stream.io) + speech_api.Store.TTL_SECS, escaped_text });
    }

    const usage_obj = try formatChatUsage(allocator, result.prompt_tokens, result.completion_tokens, result.cached_tokens, usage_details_json);
    defer allocator.free(usage_obj);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s}{s}}},"logprobs":{s},"finish_reason":"{s}"{s}}}],"usage":{s}{s}}}
    , .{
        nowMs(stream.io),
        nowSecs(stream.io),
        model_name,
        if (speech != null) "null" else escaped_text,
        reasoning_json,
        audio_field,
        logprobs_json,
        finish_reason,
        finishDetailsField(finish_reason, result.finish_details),
        usage_obj,
        timings_field,
    });
    defer allocator.free(response);

    try sendResponse(stream, "200 OK", "application/json", response);
}
