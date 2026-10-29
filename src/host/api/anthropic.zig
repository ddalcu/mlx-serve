//! sendAnthropicError; sendAnthropicEvent; stop reasons, tools json, handleAnthropicMessages, non-streaming

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kv_quant = @import("../../sdk/blocks/attention/kv_quant.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const request_mod = @import("../../sdk/request/request.zig");
const chat_mod = @import("../chat/chat.zig");
const rp_mod = @import("../../sdk/grammar/protocol.zig");
const log = @import("../../sdk/util/log.zig");
const pld_index = @import("../../sdk/spec/pld_index.zig");
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const model_registry_mod = @import("../registry/registry.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const io_util = @import("../../sdk/util/io_util.zig");
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
const promptOpensThink = @import("../chat/reasoning/request.zig").promptOpensThink;
const armThinkBound = @import("../chat/reasoning/request.zig").armThinkBound;
const armToolForce = @import("../chat/reasoning/request.zig").armToolForce;
const resolveReasoningProtocol = @import("../chat/reasoning/request.zig").resolveReasoningProtocol;
const launchMaxTokensDefault = @import("sampling.zig").launchMaxTokensDefault;
const parseRequestSeed = @import("sampling.zig").parseRequestSeed;
const clampJsonI32 = @import("sampling.zig").clampJsonI32;
const maxTokensOrigin = @import("sampling.zig").maxTokensOrigin;
const MAX_TOKENS_DESC_LEN = @import("sampling.zig").MAX_TOKENS_DESC_LEN;
const describeMaxTokens = @import("sampling.zig").describeMaxTokens;
const toolCallFinishReason = @import("../chat/tools/pipeline.zig").toolCallFinishReason;
const shouldParseToolCalls = @import("../chat/tools/pipeline.zig").shouldParseToolCalls;
const parseToolCallsForRequest = @import("../chat/tools/pipeline.zig").parseToolCallsForRequest;
const getEffectiveContextLength = @import("admission.zig").getEffectiveContextLength;
const logStreamError = @import("../http/errors.zig").logStreamError;
const sendGenerationError = @import("../http/errors.zig").sendGenerationError;
const checkAttentionMemory = @import("admission.zig").checkAttentionMemory;
const clampMaxTokens = @import("sampling.zig").clampMaxTokens;
const mediaRejectReason = @import("media_gate.zig").mediaRejectReason;
const dropMedia = @import("media_gate.zig").dropMedia;
const effortWordOnly = @import("../chat/reasoning/request.zig").effortWordOnly;
const implicitEffortBudgetFor = @import("../chat/reasoning/request.zig").implicitEffortBudgetFor;
const defaultReasoningBudget = @import("../chat/reasoning/request.zig").defaultReasoningBudget;
const forcedReasoningBudget = @import("../chat/reasoning/request.zig").forcedReasoningBudget;
const reasoningEffortFromWord = @import("../chat/reasoning/request.zig").reasoningEffortFromWord;
const parseAnthropicOutputConfig = @import("../chat/reasoning/request.zig").parseAnthropicOutputConfig;
const resolveThinking = @import("../chat/reasoning/request.zig").resolveThinking;
const schemaMasksThinking = @import("../chat/reasoning/request.zig").schemaMasksThinking;
const continuationRejectReason = @import("request_text.zig").continuationRejectReason;
const joinedTextParts = @import("request_text.zig").joinedTextParts;
const requestCacheKey = @import("request_text.zig").requestCacheKey;
const echoModelName = @import("request_text.zig").echoModelName;
const requestPld = @import("request_options.zig").requestPld;
const answerStopCut = @import("../chat/stops.zig").answerStopCut;
const visibleToolPreamble = @import("../chat/stops.zig").visibleToolPreamble;
const nonStreamingViaScheduler = @import("submit.zig").nonStreamingViaScheduler;
const splitConstrainedResponse = @import("chat_completions.zig").splitConstrainedResponse;
const requestEnableDrafter = @import("request_options.zig").requestEnableDrafter;
const requestHasCompany = @import("request_options.zig").requestHasCompany;
const requestSpecModes = @import("request_options.zig").requestSpecModes;
const sendResponse = @import("../http/respond.zig").sendResponse;
const logHttpSseEvent = @import("../http/respond.zig").logHttpSseEvent;
const countRejected = @import("../http/errors.zig").countRejected;
const contextOverflowMessage = @import("../http/errors.zig").contextOverflowMessage;
const cachedFormatChat = @import("format.zig").cachedFormatChat;
const formatAnthropicUsage = @import("format.zig").formatAnthropicUsage;
const formatTimingsObject = @import("format.zig").formatTimingsObject;
const formatPerfBracket = @import("format.zig").formatPerfBracket;
const jsonEscape = @import("format.zig").jsonEscape;
const parseKvQuantOverride = @import("sampling.zig").parseKvQuantOverride;
const MropeData = @import("../media/encode.zig").MropeData;
const MultimodalPrompt = @import("../media/encode.zig").MultimodalPrompt;
const prepareMultimodalPrompt = @import("../media/encode.zig").prepareMultimodalPrompt;
const multimodalFailure = @import("../media/encode.zig").multimodalFailure;
const visionPreprocFromConfig = @import("../media/prepare.zig").visionPreprocFromConfig;
const RequestMedia = @import("../media/request_media.zig").RequestMedia;
const appendAnthropicImageBlock = @import("../media/request_media.zig").appendAnthropicImageBlock;
const IMAGE_DECODE_REJECT = @import("../media/request_media.zig").IMAGE_DECODE_REJECT;
const handleAnthropicStreaming = @import("anthropic_stream.zig").handleAnthropicStreaming;

pub fn sendAnthropicError(allocator: std.mem.Allocator, stream: *Conn, err_type: []const u8, message: []const u8, status_code: u32) !void {
    const escaped_msg = try jsonEscape(allocator, message);
    defer allocator.free(escaped_msg);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"type":"error","error":{{"type":"{s}","message":{s}}}}}
    , .{ err_type, escaped_msg });
    defer allocator.free(body);
    var status_buf: [32]u8 = undefined;
    const status = std.fmt.bufPrint(&status_buf, "{d} Error", .{status_code}) catch "500 Error";
    try sendResponse(stream, status, "application/json", body);
}

pub fn sendAnthropicEvent(stream: *Conn, event_name: []const u8, data: []const u8) !void {
    if (stream.ws_mode) |bridge| {
        // WS transport: emit only the JSON payload as a text frame; the
        // event name lives inside the JSON as `"type": "..."`. (Anthropic
        // events are not currently WS-bridged — only Responses.)
        try bridge.sendText(data);
        return;
    }
    logHttpSseEvent(event_name, data);
    try stream.writeAllNoFlush("event: ");
    try stream.writeAllNoFlush(event_name);
    try stream.writeAllNoFlush("\ndata: ");
    try stream.writeAllNoFlush(data);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

fn mapFinishToStopReason(finish_reason: []const u8) []const u8 {
    if (std.mem.eql(u8, finish_reason, "stop")) return "end_turn";
    if (std.mem.eql(u8, finish_reason, "length")) return "max_tokens";
    if (std.mem.eql(u8, finish_reason, "tool_calls")) return "tool_use";
    return "end_turn";
}

/// Anthropic stop_reason with the client-stop-sequence case: a matched stop
/// sequence reports "stop_sequence" (callers learn WHICH stop fired via the
/// echoed `stop_sequence` field) — but only over a plain "stop"; it never
/// masks a max_tokens truncation or a parsed tool call.
pub fn anthropicStopReason(finish_reason: []const u8, matched_stop_seq: ?[]const u8) []const u8 {
    if (matched_stop_seq != null and std.mem.eql(u8, finish_reason, "stop")) return "stop_sequence";
    return mapFinishToStopReason(finish_reason);
}

/// Serialize a std.json.Value to JSON text, appending to buf.
fn serializeJsonValue(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), value: std.json.Value) !void {
    switch (value) {
        .null => try buf.appendSlice(allocator, "null"),
        .bool => |b| try buf.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| {
            var num_buf: [24]u8 = undefined;
            const s = std.fmt.bufPrint(&num_buf, "{d}", .{i}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .float => |f| {
            var num_buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&num_buf, "{d}", .{f}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .string => |s| {
            const escaped = try jsonEscape(allocator, s);
            defer allocator.free(escaped);
            try buf.appendSlice(allocator, escaped);
        },
        .array => |arr| {
            try buf.append(allocator, '[');
            for (arr.items, 0..) |item, idx| {
                if (idx > 0) try buf.append(allocator, ',');
                try serializeJsonValue(allocator, buf, item);
            }
            try buf.append(allocator, ']');
        },
        .object => |obj| {
            try buf.append(allocator, '{');
            var iter = obj.iterator();
            var first = true;
            while (iter.next()) |entry| {
                if (!first) try buf.append(allocator, ',');
                first = false;
                const ek = try jsonEscape(allocator, entry.key_ptr.*);
                defer allocator.free(ek);
                try buf.appendSlice(allocator, ek);
                try buf.append(allocator, ':');
                try serializeJsonValue(allocator, buf, entry.value_ptr.*);
            }
            try buf.append(allocator, '}');
        },
        .number_string => |s| try buf.appendSlice(allocator, s),
    }
}

/// Convert Anthropic tools format to OpenAI tools format for chat template compatibility.
fn buildOpenAIToolsJson(allocator: std.mem.Allocator, tools_array: std.json.Array) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    for (tools_array.items, 0..) |tool_val, i| {
        if (i > 0) try buf.append(allocator, ',');
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        const name = if (tool.get("name")) |v| (if (v == .string) v.string else "") else "";
        const desc = if (tool.get("description")) |v| (if (v == .string) v.string else "") else "";
        const esc_n = try jsonEscape(allocator, name);
        defer allocator.free(esc_n);
        const esc_d = try jsonEscape(allocator, desc);
        defer allocator.free(esc_d);
        try buf.appendSlice(allocator, "{\"type\":\"function\",\"function\":{\"name\":");
        try buf.appendSlice(allocator, esc_n);
        try buf.appendSlice(allocator, ",\"description\":");
        try buf.appendSlice(allocator, esc_d);
        try buf.appendSlice(allocator, ",\"parameters\":");
        if (tool.get("input_schema")) |schema_val| {
            try serializeJsonValue(allocator, &buf, schema_val);
        } else {
            try buf.appendSlice(allocator, "{}");
        }
        try buf.appendSlice(allocator, "}}");
    }
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

pub fn handleAnthropicMessages(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    // No `lm.transformer.?` — engine-backed (GGUF/ds4) models have a null
    // transformer; the only gate below uses `config.has_hybrid_layers`.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/messages -> 400 (invalid JSON)\n", .{});
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        log.warn("POST /v1/messages -> 400 (body is not a JSON object)\n", .{});
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    // max_tokens is required in the Anthropic API, but through the one parse helper so a negative
    // value is an omission and the `--max-tokens` launch default reaches this surface too.
    const gen = requestProfile(stream.io, lm);
    const max_tokens: u32 = generationMaxTokens(gen, root.get("max_tokens"), launchMaxTokensDefault());
    // For the request log line only. This surface 400s on a missing budget, so
    // the origin here is never `.auto`.
    const max_tokens_origin = maxTokensOrigin(root.get("max_tokens"), launchMaxTokensDefault());
    if (max_tokens == 0) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'max_tokens' is required and must be > 0", 400);
        return;
    }

    // Parse messages array (required)
    const messages_val = root.get("messages") orelse {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'messages' is required", 400);
        return;
    };
    if (messages_val != .array) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'messages' must be an array", 400);
        return;
    }

    var messages = std.ArrayList(chat_mod.Message).empty;
    var image_decode_failed = false;
    defer messages.deinit(allocator);

    // Decoded image buffers for every message in this request. `Message`
    // borrows them; this is the only thing that frees them.
    var media = RequestMedia.init(allocator);
    defer media.deinit();

    // Track allocations for serialized tool arguments (need to outlive generation)
    var arg_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (arg_allocs.items) |a| allocator.free(a);
        arg_allocs.deinit(allocator);
    }
    var tool_call_lists = std.ArrayList([]const chat_mod.ToolCall).empty;
    defer {
        for (tool_call_lists.items) |tcs| allocator.free(tcs);
        tool_call_lists.deinit(allocator);
    }
    // Track allocated content strings (concatenated text)
    var content_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (content_allocs.items) |s| allocator.free(s);
        content_allocs.deinit(allocator);
    }

    // System prompt (Anthropic puts it at top level). Array form JOINS every
    // text block — Claude Code sends 2+ (identity + the instructions block),
    // and first-wins dropped everything after the first.
    if (root.get("system")) |sys_val| {
        const sys_text: []const u8 = switch (sys_val) {
            .string => |s| s,
            .array => |arr| blk: {
                const joined = try joinedTextParts(allocator, arr.items);
                if (joined.owned) try content_allocs.append(allocator, joined.text);
                break :blk joined.text;
            },
            else => "",
        };
        if (sys_text.len > 0) {
            try messages.append(allocator, .{ .role = "system", .content = sys_text, .tool_calls = null, .tool_call_id = null });
        }
    }

    // Convert Anthropic messages to internal format
    for (messages_val.array.items) |msg_val| {
        if (msg_val != .object) continue;
        const msg_obj = msg_val.object;
        const role_val = msg_obj.get("role") orelse continue;
        if (role_val != .string) continue;
        const role = role_val.string;
        const content_val = msg_obj.get("content");

        if (std.mem.eql(u8, role, "user")) {
            if (content_val) |cv| switch (cv) {
                .string => |s| {
                    try messages.append(allocator, .{ .role = "user", .content = s, .tool_calls = null, .tool_call_id = null });
                },
                .array => |arr| {
                    // Process tool_result blocks first, then text+image blocks.
                    for (arr.items) |block| {
                        if (block != .object) continue;
                        const btype = if (block.object.get("type")) |t| (if (t == .string) t.string else "") else "";
                        if (!std.mem.eql(u8, btype, "tool_result")) continue;

                        const tool_use_id = if (block.object.get("tool_use_id")) |v| (if (v == .string) v.string else "") else "";
                        var result_text: []const u8 = "";
                        const result_imgs = try media.openImages();
                        if (block.object.get("content")) |rc| switch (rc) {
                            .string => |s| result_text = s,
                            .array => |result_arr| {
                                const joined = try joinedTextParts(allocator, result_arr.items);
                                if (joined.owned) try content_allocs.append(allocator, joined.text);
                                result_text = joined.text;
                                // Claude Code's Read tool returns an image as a block here.
                                for (result_arr.items) |part| {
                                    if (!try appendAnthropicImageBlock(allocator, media.images(result_imgs), part, visionPreprocFromConfig(config))) image_decode_failed = true;
                                }
                            },
                            else => {},
                        };
                        try messages.append(allocator, .{ .role = "tool", .content = result_text, .tool_calls = null, .tool_call_id = tool_use_id, .images = media.imagesSlice(result_imgs) });
                    }

                    // Collect text + image blocks into a single user message so
                    // the vision encoder sees them attached to the right turn.
                    var msg_text = std.ArrayList(u8).empty;
                    defer msg_text.deinit(allocator);
                    const img_slot = try media.openImages();
                    for (arr.items) |block| {
                        if (block != .object) continue;
                        const btype = if (block.object.get("type")) |t| (if (t == .string) t.string else "") else "";
                        if (std.mem.eql(u8, btype, "text")) {
                            const text = if (block.object.get("text")) |t| (if (t == .string) t.string else "") else "";
                            if (text.len > 0) {
                                if (msg_text.items.len > 0) try msg_text.append(allocator, '\n');
                                try msg_text.appendSlice(allocator, text);
                            }
                        } else if (!try appendAnthropicImageBlock(allocator, media.images(img_slot), block, visionPreprocFromConfig(config))) {
                            image_decode_failed = true;
                        }
                    }
                    if (msg_text.items.len > 0 or media.images(img_slot).items.len > 0) {
                        const owned_text = if (msg_text.items.len > 0) blk: {
                            const s = try allocator.dupe(u8, msg_text.items);
                            try content_allocs.append(allocator, s);
                            break :blk s;
                        } else "";
                        try messages.append(allocator, .{
                            .role = "user",
                            .content = owned_text,
                            .tool_calls = null,
                            .tool_call_id = null,
                            .images = media.imagesSlice(img_slot),
                        });
                    }
                },
                else => {},
            };
        } else if (std.mem.eql(u8, role, "assistant")) {
            if (content_val) |cv| switch (cv) {
                .string => |s| {
                    try messages.append(allocator, .{ .role = "assistant", .content = s, .tool_calls = null, .tool_call_id = null });
                },
                .array => |arr| {
                    // Extract text, tool_use and thinking blocks
                    var text_content = std.ArrayList(u8).empty;
                    defer text_content.deinit(allocator);
                    var think_content = std.ArrayList(u8).empty;
                    defer think_content.deinit(allocator);
                    var tcs = std.ArrayList(chat_mod.ToolCall).empty;

                    for (arr.items) |block| {
                        if (block != .object) continue;
                        const btype = if (block.object.get("type")) |t| (if (t == .string) t.string else "") else "";

                        if (std.mem.eql(u8, btype, "text")) {
                            const text = if (block.object.get("text")) |t| (if (t == .string) t.string else "") else "";
                            try text_content.appendSlice(allocator, text);
                        } else if (std.mem.eql(u8, btype, "tool_use")) {
                            const tc_id = if (block.object.get("id")) |v| (if (v == .string) v.string else "") else "";
                            const tc_name = if (block.object.get("name")) |v| (if (v == .string) v.string else "") else "";
                            // Serialize input object to JSON string
                            var args_buf = std.ArrayList(u8).empty;
                            if (block.object.get("input")) |input_val| {
                                try serializeJsonValue(allocator, &args_buf, input_val);
                            } else {
                                try args_buf.appendSlice(allocator, "{}");
                            }
                            const args_str = try args_buf.toOwnedSlice(allocator);
                            try arg_allocs.append(allocator, args_str);
                            try tcs.append(allocator, .{ .id = tc_id, .name = tc_name, .arguments = args_str });
                        } else if (std.mem.eql(u8, btype, "thinking")) {
                            // History reasoning → Message.reasoning_content, same
                            // round-trip as chat completions' reasoning_content:
                            // templates that persist reasoning across turns
                            // (laguna) starve into nothink without it.
                            // "redacted_thinking" stays skipped (opaque payload).
                            const t = if (block.object.get("thinking")) |v| (if (v == .string) v.string else "") else "";
                            if (t.len > 0) {
                                if (think_content.items.len > 0) try think_content.append(allocator, '\n');
                                try think_content.appendSlice(allocator, t);
                            }
                        }
                    }

                    var msg_tool_calls: ?[]const chat_mod.ToolCall = null;
                    if (tcs.items.len > 0) {
                        const owned = try tcs.toOwnedSlice(allocator);
                        try tool_call_lists.append(allocator, owned);
                        msg_tool_calls = owned;
                    } else {
                        tcs.deinit(allocator);
                    }

                    const content: []const u8 = if (text_content.items.len > 0) blk: {
                        const duped = try allocator.dupe(u8, text_content.items);
                        try content_allocs.append(allocator, duped);
                        break :blk duped;
                    } else "";
                    const msg_reasoning: ?[]const u8 = if (think_content.items.len > 0) blk: {
                        const duped = try allocator.dupe(u8, think_content.items);
                        try content_allocs.append(allocator, duped);
                        break :blk duped;
                    } else null;
                    try messages.append(allocator, .{ .role = "assistant", .content = content, .tool_calls = msg_tool_calls, .tool_call_id = null, .reasoning_content = msg_reasoning });
                },
                else => {},
            };
        } else if (std.mem.eql(u8, role, "system")) {
            // Claude Code puts SessionStart hook output here; dropping it is silent context loss.
            const sys_text: []const u8 = if (content_val) |cv| switch (cv) {
                .string => |s| s,
                .array => |arr| blk: {
                    const joined = try joinedTextParts(allocator, arr.items);
                    if (joined.owned) try content_allocs.append(allocator, joined.text);
                    break :blk joined.text;
                },
                else => "",
            } else "";
            if (sys_text.len > 0) {
                try messages.append(allocator, .{ .role = "system", .content = sys_text, .tool_calls = null, .tool_call_id = null });
            }
        }
    }

    if (try chat_mod.foldSystemMessages(allocator, &messages, true)) |joined| try content_allocs.append(allocator, joined);

    if (image_decode_failed) {
        log.warn("POST /v1/messages -> 400 (undecodable image)\n", .{});
        try sendAnthropicError(allocator, stream, "invalid_request_error", IMAGE_DECODE_REJECT, 400);
        return;
    }
    if (messages.items.len == 0) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "No valid messages found in request", 400);
        return;
    }

    // Sampling parameters. Omitted fields resolve through CLI flags and the
    // model's generation_config.json — Claude Code omits ALL of them, and the
    // bare temp=1.0/top_p=1.0/no-top_k fallback sampled far outside Qwen's
    // intended envelope (model card wants top_k=20, top_p=0.95).
    const sampled = resolveSampling(gen, root, config);
    const temperature = sampled.temperature;
    const top_p = sampled.top_p;
    const top_k = sampled.top_k;
    const min_p = sampled.min_p;
    const seed: ?u64 = parseRequestSeed(root.get("seed"));

    // Tools
    var tools_json: ?[]const u8 = null;
    var tools_json_allocated = false;
    defer if (tools_json_allocated) allocator.free(tools_json.?);
    var has_tools = false;
    var allow_parallel_tools = true;
    var forced_tool: ?chat_mod.ForcedTool = null;

    if (root.get("tools")) |tools_val| {
        if (tools_val == .array and tools_val.array.items.len > 0) {
            has_tools = true;
            // Convert Anthropic tools to OpenAI format for chat template
            tools_json = try buildOpenAIToolsJson(allocator, tools_val.array);
            tools_json_allocated = true;

            // Parse tool_choice
            if (root.get("tool_choice")) |tc| {
                if (tc == .object) {
                    // Anthropic spelling of the parallel clamp.
                    if (tc.object.get("disable_parallel_tool_use")) |d| {
                        if (d == .bool and d.bool) allow_parallel_tools = false;
                    }
                    const tc_type = if (tc.object.get("type")) |t| (if (t == .string) t.string else "auto") else "auto";
                    if (std.mem.eql(u8, tc_type, "none")) {
                        has_tools = false;
                    } else if (std.mem.eql(u8, tc_type, "any")) {
                        forced_tool = .any;
                    } else if (std.mem.eql(u8, tc_type, "tool")) {
                        if (tc.object.get("name")) |name_val| {
                            if (name_val == .string) forced_tool = .{ .name = name_val.string };
                        }
                    }
                }
            }
        }
    }
    if (forced_tool) |ft| if (ft == .name and !chat_mod.toolsDeclare(root.get("tools"), ft.name)) {
        const msg = try std.fmt.allocPrint(allocator, "tool_choice names tool \"{s}\", which is not in tools", .{ft.name});
        defer allocator.free(msg);
        try sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
        return;
    };
    const tool_choice_instruction: ?[]const u8 = if (forced_tool) |ft| try chat_mod.toolChoiceInstruction(allocator, ft) else null;
    defer if (tool_choice_instruction) |tci| allocator.free(tci);

    // Stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop_sequences")) |stop_val| {
        if (stop_val == .array) {
            for (stop_val.array.items) |item| {
                if (item == .string and item.string.len > 0) try stop_sequences.append(allocator, item.string);
            }
        }
    }

    // Thinking config. An absent `thinking` object consults the arch default
    // (muse+tools delivers its always-on reasoning as thinking blocks rather
    // than paying for it and discarding it); a PRESENT object decides
    // explicitly, exactly as before.
    const word_only = effortWordOnly(allocator, lm, tok);
    const default_budget = defaultReasoningBudget(gen);
    var thinking_on: ?bool = null;
    var budget_tokens: ?i32 = null;
    if (root.get("thinking")) |think_val| {
        thinking_on = false;
        if (think_val == .object) {
            const think_type = if (think_val.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            thinking_on = std.mem.eql(u8, think_type, "enabled") or std.mem.eql(u8, think_type, "adaptive");
            if (think_val.object.get("budget_tokens")) |bt| {
                if (bt == .integer) budget_tokens = clampJsonI32(bt.integer);
            }
        }
    }

    // `output_config` (Claude Code's spelling; see parseAnthropicOutputConfig).
    // The effort word is an explicit thinking signal, so it displaces the arch
    // default and OR's with a present `thinking` object — the OpenAI surface's
    // resolveThinking rule. An explicit `budget_tokens` outranks the
    // budget derived from the word; the word itself rides through to templates
    // that read it (qwen3.8's preamble, dsv4's).
    const output_cfg = parseAnthropicOutputConfig(root);
    if (output_cfg.schema_invalid) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "output_config.format.schema must be a JSON object when format.type is json_schema", 400);
        return;
    }
    const request_effort = if (output_cfg.effort) |word| reasoningEffortFromWord(word, default_budget, word_only) else null;
    const thinking = resolveThinking(gen, thinking_on, request_effort, config.defaultEnableThinking(root.get("tools") != null), default_budget, word_only);
    var enable_thinking = thinking.enable;
    const effort_word: ?[]const u8 = if (thinking.effort) |cfg| cfg.effort else null;
    const reasoning_budget = forcedReasoningBudget(gen, budget_tokens orelse
        if (thinking.effort) |cfg| cfg.budget else implicitEffortBudgetFor(allocator, lm, tok, default_budget));
    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;
    const model_name = echoModelName(root, config.model_type);

    // Wave 1.A: per-request KV-quant override (Anthropic mirror).
    const kv_quant_override = parseKvQuantOverride(root);
    const kv_attn_explicit = parseKvAttnExplicit(root);

    // Per-request PLD override (mirror chat-completions behavior: tools and
    // hybrid SSM do not disable PLD; the adaptive ngram gate below and the
    // runtime acceptance gate handle the rest).
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld: bool = requestPld(root, lm);

    // Drafter: same disable rules as chat-completions parse site.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter = requestEnableDrafter(root.get("enable_drafter"), lm.drafter != null, lm.dflash != null, config.isMoe(), config.has_hybrid_layers);
    const allow_batch_mtp = if (root.get("enable_batch_mtp")) |v| v != .bool or v.bool else true;
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        defaultEnableMtp(lm.mtp != null, dsv4DraftStages(lm));
    if (enable_mtp and lm.mtp == null and !dsv4DraftStages(lm)) enable_mtp = false;

    // `output_config.format` json_schema — the same two-layer enforcement as
    // chat-completions' `response_format`: a schema instruction in the system
    // prompt (so the model aims at the shape) plus the grammar mask below (so
    // the shape is guaranteed). Before the render, or the instruction never
    // reaches the prompt.
    if (output_cfg.schema != null) {
        var schema_instruction = std.ArrayList(u8).empty;
        defer schema_instruction.deinit(allocator);
        try schema_instruction.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");
        {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
            output_cfg.schema.?.jsonStringify(&jws) catch {};
            try schema_instruction.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
            try schema_instruction.appendSlice(allocator, out.written());
        }
        if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
            const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, schema_instruction.items });
            try content_allocs.append(allocator, combined);
            messages.items[0].content = combined;
        } else {
            const instruction = try allocator.dupe(u8, schema_instruction.items);
            try content_allocs.append(allocator, instruction);
            try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
        }
    }

    // Log request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);
    var tool_msg_count: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "tool")) tool_msg_count += 1;
    }
    const tools_len = if (tools_json) |tj| tj.len else 0;
    // Format chat template. Iteration 1 timing + Iteration 2 cache. The
    // `effective_tools_json` swap (`null` when has_tools is false) keeps
    // the cache key consistent with what the encoder actually sees.
    const effective_tools_json: ?[]const u8 = if (has_tools) tools_json else null;
    // Anthropic's own contract: a conversation ending in an assistant message
    // means "continue this", with no flag. Implicit here and explicit on the
    // OpenAI surface is not an inconsistency — it is each surface's documented
    // behaviour, so an Anthropic SDK client gets prefill for free.
    //
    // And because it is INFERRED, a model that cannot serve one gets an
    // ordinary turn rather than a 400: the client never asked for a
    // continuation, so refusing its request would take away an answer this
    // endpoint has always given.
    const continue_final = chat_mod.continuationRequested(messages.items) and
        continuationRejectReason(lm.engineOwnsTemplate()) == null;
    if (lm.vision_encoder == null) {
        if (mediaRejectReason(messages.items, continue_final)) |reason| {
            log.warn("POST /v1/messages -> 400 ({s})\n", .{reason});
            try sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
            return;
        }
        try dropMedia(allocator, messages.items, &content_allocs);
    }
    var tokenize_sw = Stopwatch.init(stream.io);
    // The `thinking` budget object carries no effort string, but
    // `output_config.effort` does — templates that read the word (dsv4,
    // qwen3.8) get it; requests without one still render their default.
    var prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, chat_config, messages.items, effective_tools_json, tool_choice_instruction, enable_thinking, effort_word, continue_final);
    var schema_proto: rp_mod.Protocol = undefined;
    var schema_proto_active = false;
    if (output_cfg.schema != null and !has_tools and enable_thinking and reasoning_budget < 0) {
        schema_proto_active = resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, true);
    }
    switch (schemaMasksThinking(
        output_cfg.schema != null,
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
            prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, chat_config, messages.items, effective_tools_json, tool_choice_instruction, false, effort_word, continue_final);
            log.info("[grammar] {s}; rerendered with thinking off\n", .{if (reasoning_budget >= 0) "finite reasoning budget" else "reasoning protocol unsupported for this model/prompt"});
        },
        .no_mask, .token_zero => schema_proto_active = false,
    }
    if (output_cfg.schema != null and !has_tools and !enable_thinking and lm.transformer != null) {
        if (!resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, false)) {
            schema_proto = rp_mod.Protocol.jsonOnly();
        }
        schema_proto_active = true;
    }
    const tokenize_ns = tokenize_sw.read();

    var max_tokens_desc_buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    var system_chars: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) system_chars += msg.content.len;
    }
    log.info("POST /v1/messages ({d} msgs, max_tokens={s}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}, thinking={}, sys={d}b, tools={d}b, tool_msgs={d})\n", .{
        messages.items.len, describeMaxTokens(&max_tokens_desc_buf, max_tokens, max_tokens_origin), temperature, top_p, top_k, is_stream, enable_thinking, system_chars, tools_len, tool_msg_count,
    });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    const cache_key = requestCacheKey(root);
    var mm: MultimodalPrompt = .{};
    defer mm.deinit(allocator);
    if (lm.vision_encoder != null) {
        mm = prepareMultimodalPrompt(allocator, lm, config, messages.items, &prompt_ids_raw) catch |err| {
            allocator.free(prompt_ids_raw);
            const f = multimodalFailure(err);
            log.warn("POST /v1/messages -> {d} ({s})\n", .{ f.code, @errorName(err) });
            try sendAnthropicError(allocator, stream, f.kind, f.message, f.code);
            return;
        };
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);
    enable_mtp = admitMtpForCtx(enable_mtp, prompt_ids.len);

    // Adaptive spec-decode gate (Anthropic path; mirrors chat-completions).
    if ((enable_pld and !pld_explicit_in_json) or (enable_drafter and !drafter_explicit_in_json)) {
        const score = pld_index.ngramRepeatScore(allocator, prompt_ids, 3) catch 1.0;
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
        // Heavy-echo MTP->PLD routing retired (see the NOTE at the
        // chat-completions site): MTP wins whenever loaded.
    }

    // Context size enforcement
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/messages -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendAnthropicError(allocator, stream, "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }

    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    // Check if attention computation would exceed GPU memory.
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, true, kv_quant_override, lm, enable_mtp, mm.bytes(config))) return;

    log.info("  prompt={d} tokens, max_gen={d}, ctx={d}\n", .{ prompt_ids.len, effective_max_tokens, effective_ctx });

    const eos_slice = config.eosTokenSlice();
    var sampling = request_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .min_p = min_p,
        .repeat_penalty = sampled.repeat_penalty,
        .presence_penalty = sampled.presence_penalty,
        .frequency_penalty = sampled.frequency_penalty,
        .seed = seed,
    };

    // Grammar-constrained sampling for `output_config.format` json_schema —
    // the chat-completions block, same lifetime rule: the SchemaConstraint
    // must NOT be moved (the embedded Constraint holds pointers into it).
    var sc: request_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();
    if (output_cfg.schema) |sv| {
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

    // Hand vision ownership to the sub-handler (slot takes it on submit).
    const sub_ve = mm.embeddings;
    mm.embeddings = null;
    const sub_mrope = mm.mrope;
    mm.mrope = .{}; // ownership transferred to the sub-handler → slot
    if (is_stream) {
        handleAnthropicStreaming(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, allow_parallel_tools, enable_thinking, surface_budget, @intCast(prompt_ids.len), enable_pld, enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, mm.media, sub_mrope, cache_key, kv_quant_override, kv_attn_explicit, tokenize_ns) catch |err| {
            logStreamError(err);
            sendGenerationError(allocator, stream, err, .anthropic) catch {};
        };
    } else {
        handleAnthropicNonStreaming(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, allow_parallel_tools, enable_thinking, surface_budget, @intCast(prompt_ids.len), enable_pld, enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, mm.media, sub_mrope, cache_key, kv_quant_override, kv_attn_explicit, tokenize_ns) catch |err| {
            log.err("  -> {s}\n", .{@errorName(err)});
            sendGenerationError(allocator, stream, err, .anthropic) catch {};
        };
    }
}

fn handleAnthropicNonStreaming(
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
    enable_thinking: bool,
    reasoning_budget: i32,
    prompt_token_count: u32,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    vision_embeddings: ?mlx.mlx_array,
    media: []const MediaSpan,
    mrope: MropeData,
    cache_key: u64,
    /// Wave 1.A: per-request KV-quant override.
    kv_quant_override: ?kv_quant.KVQuantConfig,
    kv_attn_explicit: ?bool,
    /// Iteration 1 instrumentation: nanoseconds of render+tokenize measured
    /// by the parent handleAnthropicMessages. Threaded through so the
    /// non-streaming response carries `timings.tokenize_ms`.
    tokenize_ns: u64,
) !void {
    // Vision-array ownership: nulled below before scheduler.submit so the
    // early-return defer doesn't double-free.
    var ve_local = vision_embeddings;
    defer {
        if (ve_local) |arr| _ = mlx.mlx_array_free(arr);
    }

    var timer = Stopwatch.init(stream.io);

    // Speculative decoding dispatch — same `requestSpecModes` as
    // chat-completions (DFlash > MTP > drafter > PLD).
    const spec = requestSpecModes(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.dflash != null, mtpCapable(lm), sampling.shapesLogits(), 0, requestHasCompany());
    const use_mtp = spec.use_mtp;
    const use_drafter = spec.use_drafter;
    const use_pld = spec.use_pld;

    // Anthropic responses never carry logprobs (the API doesn't expose
    // them). Vision-bearing requests transfer ownership of `ve_local` into
    // the slot.
    const slot_ve: ?mlx.mlx_array = blk: {
        const v = ve_local;
        ve_local = null;
        break :blk v;
    };
    // M-RoPE: Anthropic path uses scalar-RoPE fallback for now (faithful M-RoPE
    // wired for /v1/chat/completions; see computeQwenMrope). Qwen image requests
    // still decode correctly — M-RoPE refines spatial grounding only.
    // Propagates to `handleAnthropicMessages`' one error arm, shared with the streaming twin.
    const result = try nonStreamingViaScheduler(allocator, http_server.launch().global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, has_tools, enable_thinking, use_pld, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), slot_ve, media, cache_key, mrope, 0, false, kv_quant_override, kv_attn_explicit, null, stream);
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    // Apply stop sequences; remember WHICH one matched (Anthropic reports it
    // as stop_reason "stop_sequence" + the echoed `stop_sequence` field).
    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    var matched_stop_seq: ?[]const u8 = null;
    const opens_think = promptOpensThink(allocator, lm, tok, prompt_ids);
    const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;
    if (answerStopCut(result.text, result.text.len, stop_sequences, opens_think and !constrained_proto)) |cut| {
        final_text = result.text[0..cut.index];
        finish_reason = "stop";
        matched_stop_seq = cut.matched;
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

    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    // Build content blocks array
    var content = std.ArrayList(u8).empty;
    defer content.deinit(allocator);
    try content.append(allocator, '[');

    var block_count: u32 = 0;

    // Thinking block.
    //
    // The split/strip here KEEP unparsed tool markup: unlike every other
    // surface, this path reassigns `final_text` from the result and hands it
    // to parseToolCallsForRequest below. Cutting the markup first would make a
    // real call unparseable. The visible text block is cut at emission instead
    // (`chat_mod.trimLeakedToolMarkup`), which is where the leak matters.
    // Reasoning the model generated is always delivered as a thinking block —
    // the request's thinking flag shaped the prompt, never the delivery.
    var routed: ?rp_mod.Delivery = null;
    defer if (routed) |*d| d.deinit(allocator);
    {
        const think_split = try splitConstrainedResponse(allocator, &routed, sampling, final_text, result.constraint_payload_byte, true, opens_think);
        // Reasoning is never fed back to the parser, so it is cut here.
        const split_reasoning: ?[]const u8 = if (think_split.reasoning_content) |r| blk: {
            const t = if (has_tools) chat_mod.trimLeakedToolMarkup(r) else r;
            break :blk if (t.len > 0) t else null;
        } else null;
        if (split_reasoning) |reasoning| {
            // Apply budget truncation
            var truncated_reasoning = reasoning;
            var trunc_allocated = false;
            defer if (trunc_allocated) allocator.free(truncated_reasoning);
            if (reasoning_budget >= 0) {
                const r_ids = try tok.encode(allocator, reasoning);
                defer allocator.free(r_ids);
                const budget_usize: usize = @intCast(reasoning_budget);
                if (r_ids.len > budget_usize) {
                    truncated_reasoning = try tok.decode(allocator, r_ids[0..budget_usize], false);
                    trunc_allocated = true;
                }
            }
            const esc_r = try jsonEscape(allocator, truncated_reasoning);
            defer allocator.free(esc_r);
            const thinking_block = try std.fmt.allocPrint(allocator,
                \\{{"type":"thinking","thinking":{s},"signature":"mlx-serve-local"}}
            , .{esc_r});
            defer allocator.free(thinking_block);
            try content.appendSlice(allocator, thinking_block);
            block_count += 1;
        }
        final_text = think_split.content;
    }

    // Check for tool calls
    if (has_tools and shouldParseToolCalls(result.finish_details)) {
        const found_calls = try parseToolCallsForRequest(allocator, final_text, tools_json, allow_parallel_tools);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            var tu_perf_buf: [160]u8 = undefined;
            const tu_perf = formatPerfBracket(&tu_perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
            log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [tool_use: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, tu_perf, tool_calls.len,
            });

            // The same pre-markup text the stream flushes as a text delta.
            const tu_preamble = visibleToolPreamble(chat_mod.trimLeakedToolMarkup(final_text));
            if (tu_preamble.len > 0) {
                if (block_count > 0) try content.append(allocator, ',');
                const esc_pre = try jsonEscape(allocator, tu_preamble);
                defer allocator.free(esc_pre);
                const pre_block = try std.fmt.allocPrint(allocator,
                    \\{{"type":"text","text":{s}}}
                , .{esc_pre});
                defer allocator.free(pre_block);
                try content.appendSlice(allocator, pre_block);
                block_count += 1;
            }

            // Emit tool_use content blocks
            for (tool_calls, 0..) |tc, i| {
                if (block_count > 0) try content.append(allocator, ',');
                const tc_id = try std.fmt.allocPrint(allocator, "toolu_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const esc_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(esc_name);
                const tc_block = try std.fmt.allocPrint(allocator,
                    \\{{"type":"tool_use","id":"{s}","name":{s},"input":{s}}}
                , .{ tc_id, esc_name, tc.arguments });
                defer allocator.free(tc_block);
                try content.appendSlice(allocator, tc_block);
                block_count += 1;
            }
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls — emit text block. Nothing parsed, so any tool
            // markup still in here is unparsed wreckage: cut it at emission.
            if (block_count > 0) try content.append(allocator, ',');
            const esc_text = try jsonEscape(allocator, if (routed != null) final_text else chat_mod.trimLeakedToolMarkup(final_text));
            defer allocator.free(esc_text);
            const text_block = try std.fmt.allocPrint(allocator,
                \\{{"type":"text","text":{s}}}
            , .{esc_text});
            defer allocator.free(text_block);
            try content.appendSlice(allocator, text_block);
        }
    } else {
        // No tools — emit text block
        if (block_count > 0) try content.append(allocator, ',');
        const esc_text = try jsonEscape(allocator, if (routed != null or !has_tools) final_text else chat_mod.trimLeakedToolMarkup(final_text));
        defer allocator.free(esc_text);
        const text_block = try std.fmt.allocPrint(allocator,
            \\{{"type":"text","text":{s}}}
        , .{esc_text});
        defer allocator.free(text_block);
        try content.appendSlice(allocator, text_block);
    }

    try content.append(allocator, ']');

    const stop_reason = anthropicStopReason(finish_reason, matched_stop_seq);
    // Echo the matched sequence exactly when the reason is "stop_sequence".
    const echo_stop_seq = std.mem.eql(u8, stop_reason, "stop_sequence");
    const stop_seq_json: []const u8 = if (echo_stop_seq) try jsonEscape(allocator, matched_stop_seq.?) else "null";
    defer if (echo_stop_seq) allocator.free(stop_seq_json);
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, stop_reason,
    });

    // Anthropic spec doesn't standardize a `timings` field; we attach one as
    // an extension (mirrors what /v1/chat/completions already does) so bench
    // tooling can read tokenize_ms / prompt_ms / predicted_ms from either
    // surface without re-implementing the SSE accumulator.
    const timings_obj = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    const timings_field = if (timings_obj.len > 0)
        try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{timings_obj})
    else
        try allocator.alloc(u8, 0);
    defer allocator.free(timings_field);

    const usage = try formatAnthropicUsage(allocator, prompt_token_count, result.completion_tokens, result.cached_tokens);
    defer allocator.free(usage);
    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"msg_{d}","type":"message","role":"assistant","content":{s},"model":"{s}","stop_reason":"{s}","stop_sequence":{s},"usage":{s}{s}}}
    , .{
        nowMs(stream.io),
        content.items,
        model_name,
        stop_reason,
        stop_seq_json,
        usage,
        timings_field,
    });
    defer allocator.free(response);
    try sendResponse(stream, "200 OK", "application/json", response);
}

// ── tests ──

test "anthropicStopReason: a matched client stop sequence reports stop_sequence, truncation stays honest" {
    // Anthropic spec: when a CLIENT stop sequence cut the text, stop_reason is
    // "stop_sequence" (with the matched string echoed) — that's how callers
    // learn WHICH stop fired. But it never masks the other reasons: a
    // max_tokens cut stays "max_tokens" (clients key truncation recovery on
    // it), a parsed tool call stays "tool_use".
    try std.testing.expectEqualStrings("stop_sequence", anthropicStopReason("stop", "beta"));
    try std.testing.expectEqualStrings("end_turn", anthropicStopReason("stop", null));
    try std.testing.expectEqualStrings("max_tokens", anthropicStopReason("length", "beta"));
    try std.testing.expectEqualStrings("tool_use", anthropicStopReason("tool_calls", "beta"));
}
