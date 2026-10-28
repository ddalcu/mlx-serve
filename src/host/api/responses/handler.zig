//! extractResponseId, extractCompletedStatus; get/delete/compact, handleResponsesInner

const std = @import("std");
const request_mod = @import("../../../sdk/request/request.zig");
const chat_mod = @import("../../chat/chat.zig");
const rp_mod = @import("../../../sdk/grammar/protocol.zig");
const log = @import("../../../sdk/util/log.zig");
const responses_mod = @import("model.zig");
const pld_index = @import("../../../sdk/spec/pld_index.zig");
const mlx_engine = @import("../../engines/mlx.zig");
const model_registry_mod = @import("../../registry/registry.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const io_util = @import("../../../sdk/util/io_util.zig");
const nowSecs = io_util.nowSecs;
const Stopwatch = io_util.Stopwatch;
const testing = std.testing;
const http_server = @import("../../http/server.zig");

const Conn = @import("../../http/conn.zig").Conn;
const spec_gate_threshold = @import("../request_options.zig").spec_gate_threshold;
const resolveKvAttnFused = @import("../request_options.zig").resolveKvAttnFused;
const parseKvAttnExplicit = @import("../request_options.zig").parseKvAttnExplicit;
const resolveSampling = @import("../sampling.zig").resolveSampling;
const generationMaxTokens = @import("../sampling.zig").generationMaxTokens;
const requestProfile = @import("../sampling.zig").requestProfile;
const defaultEnableMtp = @import("../request_options.zig").defaultEnableMtp;
const dsv4DraftStages = @import("../request_options.zig").dsv4DraftStages;
const mtpCapable = @import("../request_options.zig").mtpCapable;
const admitMtpForCtx = @import("../request_options.zig").admitMtpForCtx;
const getTimeoutNs = @import("../../http/server.zig").getTimeoutNs;
const decodeTokens = @import("../tokenize.zig").decodeTokens;
const promptOpensThink = @import("../../chat/reasoning/request.zig").promptOpensThink;
const armThinkBound = @import("../../chat/reasoning/request.zig").armThinkBound;
const armToolForce = @import("../../chat/reasoning/request.zig").armToolForce;
const resolveReasoningProtocol = @import("../../chat/reasoning/request.zig").resolveReasoningProtocol;
const DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS = @import("store.zig").DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS;
const omittedMaxTokensDefault = @import("../sampling.zig").omittedMaxTokensDefault;
const resolveRequestMaxTokens = @import("../sampling.zig").resolveRequestMaxTokens;
const parseRequestSeed = @import("../sampling.zig").parseRequestSeed;
const clampJsonI32 = @import("../sampling.zig").clampJsonI32;
const shouldParseToolCalls = @import("../../chat/tools/pipeline.zig").shouldParseToolCalls;
const parseToolCallsForRequest = @import("../../chat/tools/pipeline.zig").parseToolCallsForRequest;
const getOrInitResponseStore = @import("store.zig").getOrInitResponseStore;
const getEffectiveContextLength = @import("../admission.zig").getEffectiveContextLength;
const sendGenerationError = @import("../../http/errors.zig").sendGenerationError;
const checkAttentionMemory = @import("../admission.zig").checkAttentionMemory;
const clampMaxTokens = @import("../sampling.zig").clampMaxTokens;
const mediaRejectReason = @import("../media_gate.zig").mediaRejectReason;
const dropMedia = @import("../media_gate.zig").dropMedia;
const encodeText = @import("../tokenize.zig").encodeText;
const effortWordOnly = @import("../../chat/reasoning/request.zig").effortWordOnly;
const implicitEffortBudgetFor = @import("../../chat/reasoning/request.zig").implicitEffortBudgetFor;
const defaultReasoningBudget = @import("../../chat/reasoning/request.zig").defaultReasoningBudget;
const forcedReasoningBudget = @import("../../chat/reasoning/request.zig").forcedReasoningBudget;
const reasoningEffortFromWord = @import("../../chat/reasoning/request.zig").reasoningEffortFromWord;
const resolveThinking = @import("../../chat/reasoning/request.zig").resolveThinking;
const schemaMasksThinking = @import("../../chat/reasoning/request.zig").schemaMasksThinking;
const requestCacheKey = @import("../request_text.zig").requestCacheKey;
const echoModelName = @import("../request_text.zig").echoModelName;
const requestPld = @import("../request_options.zig").requestPld;
const StopStream = @import("../../chat/stops.zig").StopStream;
const answerStopCut = @import("../../chat/stops.zig").answerStopCut;
const nonStreamingViaScheduler = @import("../submit.zig").nonStreamingViaScheduler;
const emitConstrainedResponses = @import("../chat_completions.zig").emitConstrainedResponses;
const splitConstrainedResponse = @import("../chat_completions.zig").splitConstrainedResponse;
const StreamingTokenStream = @import("../chat_stream.zig").StreamingTokenStream;
const requestEnableDrafter = @import("../request_options.zig").requestEnableDrafter;
const requestHasCompany = @import("../request_options.zig").requestHasCompany;
const requestSpecModes = @import("../request_options.zig").requestSpecModes;
const pickStreamMode = @import("../request_options.zig").pickStreamMode;
const sendResponse = @import("../../http/respond.zig").sendResponse;
const SSE_ALLOW_HEADERS_DEFAULT = @import("../../http/respond.zig").SSE_ALLOW_HEADERS_DEFAULT;
const sendSseHeaders = @import("../../http/respond.zig").sendSseHeaders;
const logHttpSseData = @import("../../http/respond.zig").logHttpSseData;
const countRejected = @import("../../http/errors.zig").countRejected;
const contextOverflowMessage = @import("../../http/errors.zig").contextOverflowMessage;
const sendErrorResponse = @import("../../http/errors.zig").sendErrorResponse;
const utf8TrailingIncomplete = @import("../format.zig").utf8TrailingIncomplete;
const thinkHoldBackLen = @import("../format.zig").thinkHoldBackLen;
const cachedFormatChat = @import("../format.zig").cachedFormatChat;
const jsonEscape = @import("../format.zig").jsonEscape;
const parseKvQuantOverride = @import("../sampling.zig").parseKvQuantOverride;
const MultimodalPrompt = @import("../../media/encode.zig").MultimodalPrompt;
const prepareMultimodalPrompt = @import("../../media/encode.zig").prepareMultimodalPrompt;
const multimodalFailure = @import("../../media/encode.zig").multimodalFailure;
const visionPreprocFromConfig = @import("../../media/prepare.zig").visionPreprocFromConfig;
const appendImageUrlContent = @import("../../media/request_media.zig").appendImageUrlContent;
const IMAGE_DECODE_REJECT = @import("../../media/request_media.zig").IMAGE_DECODE_REJECT;
const sendStreamKeepalive = @import("../../http/sse.zig").sendStreamKeepalive;
const beatStreamKeepalive = @import("../../http/sse.zig").beatStreamKeepalive;
const sendResponsesEvent = @import("events.zig").sendResponsesEvent;
const renderResponsesToolsEcho = @import("envelope.zig").renderResponsesToolsEcho;
const renderResponsesToolChoiceEcho = @import("envelope.zig").renderResponsesToolChoiceEcho;
const renderResponsesTextEcho = @import("envelope.zig").renderResponsesTextEcho;
const renderResponsesReasoningEcho = @import("envelope.zig").renderResponsesReasoningEcho;
const renderResponsesMetadataEcho = @import("envelope.zig").renderResponsesMetadataEcho;
const ResponseEcho = @import("envelope.zig").ResponseEcho;
const buildResponsesEnvelope = @import("envelope.zig").buildResponsesEnvelope;
const emitResponsesReasoningStart = @import("envelope.zig").emitResponsesReasoningStart;
const emitResponsesReasoningDelta = @import("envelope.zig").emitResponsesReasoningDelta;
const emitResponsesReasoningEnd = @import("envelope.zig").emitResponsesReasoningEnd;
const emitResponsesReasoningEvents = @import("envelope.zig").emitResponsesReasoningEvents;
const emitResponsesFunctionCallEvents = @import("envelope.zig").emitResponsesFunctionCallEvents;
const emitResponsesMessageStart = @import("envelope.zig").emitResponsesMessageStart;
const emitResponsesMessageDelta = @import("envelope.zig").emitResponsesMessageDelta;
const emitResponsesMessageEnd = @import("envelope.zig").emitResponsesMessageEnd;
const emitResponsesMessageEvents = @import("envelope.zig").emitResponsesMessageEvents;
const storeResponse = @import("envelope.zig").storeResponse;

/// Pull the value of the first `"id":"resp_..."` field out of a JSON
/// payload. Returns an owned copy or null. Used by `WsBridge.sendText` to
/// learn the response id without parsing the whole envelope.
pub fn extractResponseId(allocator: std.mem.Allocator, data: []const u8) ?[]u8 {
    const needle = "\"id\":\"resp_";
    const start = std.mem.indexOf(u8, data, needle) orelse return null;
    const v_start = start + "\"id\":\"".len;
    const v_end = std.mem.indexOfScalarPos(u8, data, v_start, '"') orelse return null;
    return allocator.dupe(u8, data[v_start..v_end]) catch null;
}

/// Pull the value of `"status":"..."` (response-level, not item-level)
/// from a `response.completed`/`.failed`/`.incomplete` payload.
pub fn extractCompletedStatus(allocator: std.mem.Allocator, data: []const u8) ?[]u8 {
    // Only look in completed/failed/incomplete events.
    const is_terminal = std.mem.indexOf(u8, data, "\"type\":\"response.completed\"") != null or
        std.mem.indexOf(u8, data, "\"type\":\"response.failed\"") != null or
        std.mem.indexOf(u8, data, "\"type\":\"response.incomplete\"") != null;
    if (!is_terminal) return null;
    // Status field appears after the response object; first match is fine.
    const needle = "\"status\":\"";
    const start = std.mem.indexOf(u8, data, needle) orelse return null;
    const v_start = start + needle.len;
    const v_end = std.mem.indexOfScalarPos(u8, data, v_start, '"') orelse return null;
    return allocator.dupe(u8, data[v_start..v_end]) catch null;
}

pub fn handleResponsesGet(allocator: std.mem.Allocator, stream: *Conn, id: []const u8) !void {
    const store = getOrInitResponseStore(stream.io, allocator);
    if (store.get(id)) |sr| {
        defer sr.release();
        try sendResponse(stream, "200 OK", "application/json", sr.body_json);
    } else {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Response not found", 404);
    }
}

pub fn handleResponsesDelete(allocator: std.mem.Allocator, stream: *Conn, id: []const u8) !void {
    const store = getOrInitResponseStore(stream.io, allocator);
    if (store.delete(id)) {
        const esc_id = try jsonEscape(allocator, id);
        defer allocator.free(esc_id);
        const body = try std.fmt.allocPrint(allocator,
            \\{{"id":{s},"object":"response","deleted":true}}
        , .{esc_id});
        defer allocator.free(body);
        try sendResponse(stream, "200 OK", "application/json", body);
    } else {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Response not found", 404);
    }
}

fn responsesToolExists(
    tools_val: ?std.json.Value,
    name: []const u8,
    aliases: ?*const responses_mod.NamespaceAliases,
) bool {
    // A wire name we synthesized by expanding a namespace group is declared.
    if (responses_mod.splitNamespaceToolName(aliases, name)) |_| return true;
    const v = tools_val orelse return false;
    if (v != .array) return false;
    for (v.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        const t = if (tool.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
        if (!std.mem.eql(u8, t, "function")) continue;
        const tool_name = if (tool.get("name")) |nv| (if (nv == .string) nv.string else "") else "";
        if (std.mem.eql(u8, tool_name, name)) return true;
    }
    return false;
}

fn buildResponsesJsonInstruction(allocator: std.mem.Allocator, schema_val: ?std.json.Value, tools_active: bool) ![]const u8 {
    var instruction_buf = std.ArrayList(u8).empty;
    defer instruction_buf.deinit(allocator);

    if (tools_active) {
        try instruction_buf.appendSlice(allocator, "If you answer without calling a function, respond with valid JSON only. Do not include markdown or explanation outside the JSON. If a function call is needed, call the function instead; this JSON format applies only to final assistant messages. ");
    } else {
        try instruction_buf.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");
    }

    if (schema_val) |sv| {
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();
        var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
        sv.jsonStringify(&jws) catch {};
        try instruction_buf.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
        try instruction_buf.appendSlice(allocator, out.written());
    }

    return try allocator.dupe(u8, instruction_buf.items);
}

fn shouldInjectResponsesJsonInstruction(wants_json: bool, tools_active: bool, tool_choice_instruction: ?[]const u8) bool {
    if (!wants_json) return false;
    if (!tools_active) return true;

    // Required/forced tool calls must keep the prompt focused on tool syntax.
    // Structured-output instructions apply after the tool result is supplied.
    return tool_choice_instruction == null;
}

fn isJsonObjectString(allocator: std.mem.Allocator, text: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object;
}

/// Compact a conversation into a single opaque, round-trippable item.
///
/// The OpenAI Responses spec treats `encrypted_content` as provider-defined.
/// We synthesize a base64-encoded JSON envelope of the resolved messages so
/// the returned `compaction` item can be fed back into `response.create` as
/// an `input` item — exercising the full round-trip without an LLM call.
pub fn handleResponsesCompact(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/responses/compact -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    // ── model (required) ──
    const model_val = root.get("model");
    const has_model = model_val != null and model_val.? == .string and model_val.?.string.len > 0;
    if (!has_model) {
        try sendErrorResponse(allocator, stream, "422 Unprocessable Entity", "invalid_request_error", "model is required", 422);
        return;
    }

    // ── input (required) ──
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' is a required field", 400);
        return;
    };

    // ── instructions (optional) ──
    const instructions: ?[]const u8 = if (root.get("instructions")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── previous_response_id (optional) ──
    const prev_id: ?[]const u8 = if (root.get("previous_response_id")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    var prev_messages: ?[]const chat_mod.Message = null;
    var prev_sr: ?*responses_mod.StoredResponse = null;
    defer if (prev_sr) |p| p.release();
    if (prev_id) |pid| {
        const store = getOrInitResponseStore(stream.io, allocator);
        if (store.get(pid)) |sr| {
            prev_sr = sr;
            prev_messages = sr.history;
        } else {
            try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "previous_response_id not found", 404);
            return;
        }
    }

    // ── parse → resolved message history ──
    // Compaction drops images, so the preprocessing selector is irrelevant here.
    // No tool aliases are built: an echoed call keeps the name it was declared with.
    var pi = responses_mod.parseInput(allocator, input_val, instructions, prev_messages, null, appendImageUrlContent, .{}) catch |err| {
        log.warn("POST /v1/responses/compact -> 400 (input parse: {s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Failed to parse input", 400);
        return;
    };
    defer pi.deinit();

    // ── synthesize the opaque blob ──
    const blob = try responses_mod.encodeCompactionBlob(allocator, pi.messages.items);
    defer allocator.free(blob);

    // ── mint ids ──
    const resp_id = try responses_mod.makeId(stream.io, allocator, "comp");
    defer allocator.free(resp_id);
    const item_id = try responses_mod.makeId(stream.io, allocator, "cmp");
    defer allocator.free(item_id);

    const esc_resp_id = try jsonEscape(allocator, resp_id);
    defer allocator.free(esc_resp_id);
    const esc_item_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_item_id);
    const esc_blob = try jsonEscape(allocator, blob);
    defer allocator.free(esc_blob);

    const created_ts = nowSecs(stream.io);

    const out = try std.fmt.allocPrint(allocator,
        \\{{"id":{s},"object":"response.compaction","created_at":{d},"output":[{{"type":"compaction","id":{s},"encrypted_content":{s}}}],"usage":{{"input_tokens":0,"output_tokens":0,"total_tokens":0,"input_tokens_details":{{"cached_tokens":0}},"output_tokens_details":{{"reasoning_tokens":0}}}}}}
    , .{ esc_resp_id, created_ts, esc_item_id, esc_blob });
    defer allocator.free(out);

    log.info("POST /v1/responses/compact ({d} msgs -> {d}b blob)\n", .{ pi.messages.items.len, blob.len });
    try sendResponse(stream, "200 OK", "application/json", out);
}

/// Route entry for `/v1/responses`: gives the handler an error arm (it was the one generative
/// surface with no `catch` on its path) and owns the SSE sequence counter the terminal `error`
/// event must carry.
pub fn handleResponses(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    var seq_num: u64 = 0;
    handleResponsesInner(allocator, stream, body, lm, &seq_num) catch |err| {
        // The WS transport keeps its error: it has its own terminal frame and cleanup.
        if (stream.ws_mode != null) return err;
        log.err("  -> responses error: {s}\n", .{@errorName(err)});
        sendGenerationError(allocator, stream, err, .{ .responses = &seq_num }) catch {};
    };
}

fn handleResponsesInner(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
    /// Owned by `handleResponses` so its error arm can number the terminal `error` event.
    seq_num: *u64,
) !void {
    // No `lm.transformer.?` — engine-backed (GGUF/ds4) models have a null
    // transformer; the only gates below use `config.has_hybrid_layers`.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/responses -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        log.warn("POST /v1/responses -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    // ── input (required) ──
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' is a required field", 400);
        return;
    };

    // ── instructions ──
    const instructions: ?[]const u8 = if (root.get("instructions")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── previous_response_id ──
    const prev_id: ?[]const u8 = if (root.get("previous_response_id")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── store (default true) ──
    const should_store = if (root.get("store")) |v| (if (v == .bool) v.bool else true) else true;

    // ── streaming ──
    const is_stream = if (root.get("stream")) |v| (v == .bool and v.bool) else false;

    // ── text.format (or chat-style response_format alias) ──
    var text_format = responses_mod.parseTextFormat(root.get("text"));
    if (text_format.schema_value == null and !std.mem.eql(u8, text_format.kind, "json_schema") and !std.mem.eql(u8, text_format.kind, "json_object")) {
        // Fall back to top-level `response_format` (some clients reuse their
        // chat-completions adapter for /v1/responses).
        const alias = responses_mod.parseResponseFormatAlias(root.get("response_format"));
        if (alias.schema_value != null or std.mem.eql(u8, alias.kind, "json_schema") or std.mem.eql(u8, alias.kind, "json_object")) {
            text_format = alias;
            log.debug("[responses] using top-level response_format as text.format alias\n", .{});
        }
    }
    if (std.mem.eql(u8, text_format.kind, "json_schema") and (text_format.schema_value == null or text_format.schema_value.? != .object)) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "text.format.schema must be a JSON object when format.type is json_schema (use {\"type\":\"json_object\"} for unconstrained JSON)", 400);
        return;
    }
    const wants_json = std.mem.eql(u8, text_format.kind, "json_schema") or std.mem.eql(u8, text_format.kind, "json_object");
    // Belt + braces, mirroring the chat-completions path: bare `json_object`
    // carries no schema, so without a synthesized permissive one there is no
    // grammar constraint at all and prompt-ignoring models (Gemma 3 wraps the
    // object in a ```json fence — caught live by llmprobe
    // structured-json-mode-valid) emit invalid JSON.
    var json_object_schema_holder: ?std.json.Parsed(std.json.Value) = null;
    defer if (json_object_schema_holder) |*p| p.deinit();
    if (text_format.schema_value == null and std.mem.eql(u8, text_format.kind, "json_object")) {
        json_object_schema_holder = std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"object\",\"additionalProperties\":true}", .{}) catch null;
    }
    const grammar_schema_val: ?std.json.Value = if (json_object_schema_holder) |*p|
        p.value
    else
        text_format.schema_value;
    const has_current_tool_output = responses_mod.inputContainsFunctionCallOutput(input_val);

    // ── sampling params ──
    // Track whether the user explicitly supplied max_output_tokens so we can
    // echo `null` (vs. our internal default) in the response envelope.
    const req_max_output_tokens: ?u32 = blk: {
        const v = root.get("max_output_tokens") orelse root.get("max_tokens");
        // <= 0 is "auto" (the omitted-field default, the `--max-tokens` launch value when given).
        const r = resolveRequestMaxTokens(v, 0);
        break :blk if (r > 0) r else null;
    };
    const gen = requestProfile(stream.io, lm);
    const max_tokens: u32 = generationMaxTokens(gen, root.get("max_output_tokens") orelse root.get("max_tokens"), if (wants_json) DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS else omittedMaxTokensDefault(getEffectiveContextLength(config)));
    const sampled = resolveSampling(gen, root, config);
    const temperature = sampled.temperature;
    const top_p = sampled.top_p;
    const top_k = sampled.top_k;
    const min_p = sampled.min_p;
    const frequency_penalty = sampled.frequency_penalty;
    const repeat_penalty = sampled.repeat_penalty;
    const presence_penalty = sampled.presence_penalty;

    // ── echo fields (parsed but not consumed by generation; round-tripped
    // back into the response envelope to satisfy the OpenAI Responses schema) ──
    const top_logprobs_echo: u32 = if (root.get("top_logprobs")) |v| switch (v) {
        .integer => |i| if (i >= 0 and i <= 20) @intCast(i) else 0,
        else => 0,
    } else 0;
    const max_tool_calls_echo: ?u32 = if (root.get("max_tool_calls")) |v| switch (v) {
        .integer => |i| if (i >= 0) @as(?u32, @intCast(@min(i, std.math.maxInt(u32)))) else null,
        else => null,
    } else null;
    const truncation_echo: []const u8 = if (root.get("truncation")) |v|
        (if (v == .string and (std.mem.eql(u8, v.string, "auto") or std.mem.eql(u8, v.string, "disabled"))) v.string else "disabled")
    else
        "disabled";
    const parallel_tool_calls_echo: bool = if (root.get("parallel_tool_calls")) |v|
        (if (v == .bool) v.bool else true)
    else
        true;
    const background_echo: bool = if (root.get("background")) |v|
        (if (v == .bool) v.bool else false)
    else
        false;
    // Honest rejection: there is no queue/poll machinery behind `background`.
    // Accepting it and running synchronously returns status "completed" on a
    // request the caller expects to come back "queued" — a silent lie strict
    // clients flag. The WS transport already rejects it at body-build time.
    if (background_echo) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "background mode is not supported; remove \"background\": true and run the request synchronously", 400);
        return;
    }
    const service_tier_echo: []const u8 = if (root.get("service_tier")) |v|
        (if (v == .string) v.string else "default")
    else
        "default";
    const safety_identifier_echo: ?[]const u8 = if (root.get("safety_identifier")) |v|
        (if (v == .string) v.string else null)
    else
        null;
    const prompt_cache_key_echo: ?[]const u8 = if (root.get("prompt_cache_key")) |v|
        (if (v == .string) v.string else null)
    else
        null;
    const seed: ?u64 = parseRequestSeed(root.get("seed"));

    // ── stop sequences ──
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |sv| switch (sv) {
        .string => |s| if (s.len > 0) try stop_sequences.append(allocator, s),
        .array => |arr| for (arr.items) |it| {
            if (it == .string and it.string.len > 0) try stop_sequences.append(allocator, it.string);
        },
        else => {},
    };

    // ── reasoning ──
    // Budget precedence as on chat: explicit reasoning_budget_tokens > the
    // effort word's budget (`reasoningEffortFromWord`) > --reasoning-budget,
    // with the Qwen3.8 implicit-low budget when thinking names no effort.
    const default_budget = defaultReasoningBudget(gen);
    const word_only = effortWordOnly(allocator, lm, tok);
    const reasoning_cfg = responses_mod.parseReasoning(root.get("reasoning"), default_budget);
    const reasoning_named = if (root.get("reasoning")) |v| v == .object else false;
    const request_effort = if (reasoning_cfg.effort) |word| reasoningEffortFromWord(word, default_budget, word_only) else null;
    const thinking = resolveThinking(gen, if (reasoning_named) reasoning_cfg.enable else null, request_effort, false, default_budget, word_only);
    var enable_thinking = thinking.enable;
    const effort_word: ?[]const u8 = if (thinking.effort) |cfg| cfg.effort else null;
    const effort_budget: i32 = if (thinking.effort) |cfg| cfg.budget else implicitEffortBudgetFor(allocator, lm, tok, default_budget);
    const reasoning_budget: i32 = forcedReasoningBudget(gen, if (root.get("reasoning_budget_tokens")) |v| switch (v) {
        .integer => |i| clampJsonI32(i),
        else => effort_budget,
    } else effort_budget);

    // ── tools ──
    var tools_json: ?[]const u8 = null;
    var tools_json_owned = false;
    defer if (tools_json_owned) {
        if (tools_json) |tj| allocator.free(tj);
    };
    var has_tools = root.get("tools") != null;

    const tool_choice = try responses_mod.parseToolChoice(allocator, root.get("tool_choice"));
    defer if (tool_choice.instruction) |ins| allocator.free(ins);
    if (!tool_choice.include_tools) has_tools = false;
    if (tool_choice.forced) |ft| if (ft == .name and !chat_mod.toolsDeclare(root.get("tools"), ft.name)) {
        const msg = try std.fmt.allocPrint(allocator, "tool_choice names function \"{s}\", which is not in tools", .{ft.name});
        defer allocator.free(msg);
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        return;
    };

    // Maps every name the model may spell a namespaced call with (expanded
    // wire names, unique bare child names) back to (namespace, name), so a
    // resulting call returns with its namespace intact.
    var namespace_aliases = responses_mod.NamespaceAliases.init(allocator);
    defer responses_mod.freeNamespaceAliases(allocator, &namespace_aliases);

    if (has_tools) {
        if (root.get("tools")) |tools_val| if (tools_val == .array) {
            const reshaped = try responses_mod.buildToolsJson(allocator, tools_val.array, &namespace_aliases);
            tools_json = reshaped;
            tools_json_owned = true;
            if (reshaped.len <= 2) has_tools = false; // "[]" — no function tools
        };
    }

    // Once tool outputs are supplied for a structured-output request, this turn
    // must produce the final JSON answer. Keeping tools available lets local
    // models reinterpret schema fields as fake tool calls and loop forever.
    const final_answer_mode = wants_json and has_current_tool_output;
    const active_has_tools = has_tools and !final_answer_mode;
    const active_tools_json: ?[]const u8 = if (active_has_tools) tools_json else null;
    const active_tool_choice_instruction: ?[]const u8 = if (active_has_tools) tool_choice.instruction else null;
    if (final_answer_mode and has_tools) {
        log.info("[responses] final-answer mode - tools disabled after function_call_output\n", .{});
    }
    // ── model name ──
    const model_name = echoModelName(root, config.model_type);

    // ── previous response — fetch stored history ──
    var prev_messages: ?[]const chat_mod.Message = null;
    var prev_sr: ?*responses_mod.StoredResponse = null;
    defer if (prev_sr) |p| p.release();
    if (prev_id) |pid| {
        const store = getOrInitResponseStore(stream.io, allocator);
        if (store.get(pid)) |sr| {
            prev_sr = sr;
            prev_messages = sr.history;
        } else {
            try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "previous_response_id not found", 404);
            return;
        }
    }

    // ── parse input → messages ──
    var pi = responses_mod.parseInput(allocator, input_val, instructions, prev_messages, &namespace_aliases, appendImageUrlContent, visionPreprocFromConfig(config)) catch |err| {
        log.warn("POST /v1/responses -> 400 (input parse: {s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Failed to parse input", 400);
        return;
    };
    defer pi.deinit();

    if (pi.image_decode_failed) {
        log.warn("POST /v1/responses -> 400 (undecodable image)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", IMAGE_DECODE_REJECT, 400);
        return;
    }
    if (pi.messages.items.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "No valid messages found in 'input'", 400);
        return;
    }

    // ── inject json_schema instruction into system msg (prompt-side belt) ──
    var rf_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (rf_allocs.items) |a| allocator.free(a);
        rf_allocs.deinit(allocator);
    }
    const inject_json_instruction = shouldInjectResponsesJsonInstruction(wants_json, active_has_tools, active_tool_choice_instruction);
    if (inject_json_instruction) {
        const owned = try buildResponsesJsonInstruction(allocator, grammar_schema_val, active_has_tools);
        try rf_allocs.append(allocator, owned);
        if (pi.messages.items.len > 0 and std.mem.eql(u8, pi.messages.items[0].role, "system")) {
            const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ pi.messages.items[0].content, owned });
            try rf_allocs.append(allocator, combined);
            pi.messages.items[0].content = combined;
        } else {
            try pi.messages.insert(allocator, 0, .{ .role = "system", .content = owned });
        }
        if (active_has_tools) {
            log.info("[grammar] injected JSON schema prompt while tools are available (mask deferred for tool calls)\n", .{});
        }
    }

    if (lm.vision_encoder == null) {
        if (mediaRejectReason(pi.messages.items, false)) |reason| {
            log.warn("POST /v1/responses -> 400 ({s})\n", .{reason});
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        try dropMedia(allocator, pi.messages.items, &rf_allocs);
    }

    // ── format chat template ──
    // Iteration 1 timing + Iteration 2 cache. Responses sees the same
    // cache as chat-completions / messages because they all hash the
    // same canonical (messages, tools, flags) tuple.
    var tokenize_sw = Stopwatch.init(stream.io);
    var prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, chat_config, pi.messages.items, active_tools_json, active_tool_choice_instruction, enable_thinking, effort_word, false);
    var schema_proto: rp_mod.Protocol = undefined;
    var schema_proto_active = false;
    if (grammar_schema_val != null and !active_has_tools and enable_thinking) {
        schema_proto_active = resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, true);
    }
    switch (schemaMasksThinking(
        grammar_schema_val != null,
        active_has_tools,
        enable_thinking,
        reasoning_budget >= 0,
        schema_proto_active,
    )) {
        .deferred => {},
        .fallback_thinking_off => {
            schema_proto_active = false;
            allocator.free(prompt_ids_raw);
            enable_thinking = false;
            prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, chat_config, pi.messages.items, active_tools_json, active_tool_choice_instruction, false, effort_word, false);
            log.info("[grammar] reasoning protocol unsupported for this model/prompt; rerendered with thinking off\n", .{});
        },
        .no_mask, .token_zero => schema_proto_active = false,
    }
    if (grammar_schema_val != null and !active_has_tools and !enable_thinking and lm.transformer != null) {
        if (!resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, false)) {
            schema_proto = rp_mod.Protocol.jsonOnly();
        }
        schema_proto_active = true;
    }
    const tokenize_ns = tokenize_sw.read();
    log.info("POST /v1/responses ({d} msgs, max_out={d}, temp={d:.2}, stream={}, thinking={}, prev={?s})\n", .{
        pi.messages.items.len, max_tokens, temperature, is_stream, enable_thinking, prev_id,
    });

    // ── vision encoder ──
    const cache_key = requestCacheKey(root);
    var mm: MultimodalPrompt = .{};
    defer mm.deinit(allocator);
    if (lm.vision_encoder != null) {
        mm = prepareMultimodalPrompt(allocator, lm, config, pi.messages.items, &prompt_ids_raw) catch |err| {
            allocator.free(prompt_ids_raw);
            const f = multimodalFailure(err);
            log.warn("POST /v1/responses -> {d} ({s})\n", .{ f.code, @errorName(err) });
            try sendErrorResponse(allocator, stream, f.status, f.kind, f.message, f.code);
            return;
        };
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);

    // ── context limit ──
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }
    const kv_quant_override = parseKvQuantOverride(root);
    const kv_attn_explicit = parseKvAttnExplicit(root);
    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    const allow_batch_mtp = if (root.get("enable_batch_mtp")) |v| v != .bool or v.bool else true;
    var enable_mtp_resp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        defaultEnableMtp(lm.mtp != null, dsv4DraftStages(lm));
    if (enable_mtp_resp and lm.mtp == null and !dsv4DraftStages(lm)) enable_mtp_resp = false;
    enable_mtp_resp = admitMtpForCtx(enable_mtp_resp, prompt_ids.len);

    // Check if attention computation would exceed GPU memory.
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, false, kv_quant_override, lm, enable_mtp_resp, mm.bytes(config))) return;

    // ── sampling ──
    var sampling = request_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .min_p = min_p,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty,
        .frequency_penalty = frequency_penalty,
        .seed = seed,
    };
    var sc: request_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();
    if (grammar_schema_val) |sv| {
        if (active_has_tools) {
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
                log.warn("[grammar] schema parse failed ({s})\n", .{@errorName(err)});
            }
        }
    }

    // The budget is enforced at decode: the server closes the thought, so this
    // surface parses a closed block like any other.
    var think_bound = armThinkBound(allocator, lm, tok, prompt_ids, enable_thinking, reasoning_budget);
    defer if (think_bound) |tb| allocator.free(tb.forced);
    if (think_bound) |*tb| sampling.think_bound = tb;
    var tool_force = armToolForce(allocator, lm, tok, prompt_ids, if (active_has_tools) tool_choice.forced else null);
    defer if (tool_force) |tf| allocator.free(tf.forced);
    if (tool_force) |*tf| sampling.tool_force = tf;

    // ── pre-allocate response id (used in streaming envelopes too) ──
    const resp_id = try responses_mod.makeId(stream.io, allocator, "resp");
    defer allocator.free(resp_id);
    const esc_resp_id = try jsonEscape(allocator, resp_id);
    defer allocator.free(esc_resp_id);
    const esc_model = try jsonEscape(allocator, model_name);
    defer allocator.free(esc_model);

    // ── pre-render request echoes (live for both the streaming skeleton
    // and the final completed envelope) ──
    const echo_tools_json = try renderResponsesToolsEcho(allocator, root.get("tools"));
    defer allocator.free(echo_tools_json);
    const echo_tool_choice_json = try renderResponsesToolChoiceEcho(allocator, root.get("tool_choice"));
    defer allocator.free(echo_tool_choice_json);
    const echo_text_json = try renderResponsesTextEcho(allocator, root);
    defer allocator.free(echo_text_json);
    const echo_reasoning_json = try renderResponsesReasoningEcho(allocator, root);
    defer allocator.free(echo_reasoning_json);
    const echo_metadata_json = try renderResponsesMetadataEcho(allocator, root);
    defer allocator.free(echo_metadata_json);

    const response_echo = ResponseEcho{
        .tools_json = echo_tools_json,
        .tool_choice_json = echo_tool_choice_json,
        .text_json = echo_text_json,
        .reasoning_json = echo_reasoning_json,
        .metadata_json = echo_metadata_json,
        .instructions = instructions,
        .truncation = truncation_echo,
        .service_tier = service_tier_echo,
        .safety_identifier = safety_identifier_echo,
        .prompt_cache_key = prompt_cache_key_echo,
        .temperature = temperature,
        .top_p = top_p,
        .presence_penalty = presence_penalty,
        .frequency_penalty = frequency_penalty,
        .top_logprobs = top_logprobs_echo,
        .parallel_tool_calls = parallel_tool_calls_echo,
        .background = background_echo,
        .max_output_tokens = req_max_output_tokens,
        .max_tool_calls = max_tool_calls_echo,
    };

    // SSE event sequence counter, owned by the wrapper (see `handleResponses`).

    // ── streaming: send SSE headers + response.created + response.in_progress ──
    if (is_stream) {
        if (stream.ws_mode == null) {
            try sendSseHeaders(stream, "responses", SSE_ALLOW_HEADERS_DEFAULT);
        }

        // Skeleton envelope (status:in_progress, output:[])
        // Timings are all zero on the skeleton — the real numbers appear on
        // the final `response.completed` envelope after generation.
        const skel = try buildResponsesEnvelope(
            stream.io,
            allocator,
            esc_resp_id,
            esc_model,
            "in_progress",
            "[]",
            0,
            0,
            0,
            0,
            should_store,
            prev_id,
            false,
            false,
            response_echo,
            0,
            0,
            tokenize_ns,
            0,
        );
        defer allocator.free(skel);
        const created_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.created\",\"response\":{s}}}", .{skel});
        defer allocator.free(created_payload);
        try sendResponsesEvent(allocator, stream, seq_num, "response.created", created_payload);
        const ip_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.in_progress\",\"response\":{s}}}", .{skel});
        defer allocator.free(ip_payload);
        try sendResponsesEvent(allocator, stream, seq_num, "response.in_progress", ip_payload);
    }

    // ── generate (streaming path: emit deltas live; non-streaming: existing) ──
    const eos_slice = config.eosTokenSlice();

    // Streaming bookkeeping. When we live-stream a reasoning or message item,
    // we record its id + index so the post-loop block emits just the END
    // events instead of full start+delta+end via emit*Events.
    var streamed_reasoning_id: ?[]u8 = null;
    var streamed_reasoning_index: u32 = 0;
    var streamed_reasoning_started = false;
    // Bytes of the thought a tool-active stream already sent; the end sends the rest.
    var tool_reasoning_streamed: usize = 0;
    var streamed_message_id: ?[]u8 = null;
    var streamed_message_index: u32 = 0;
    var streamed_message_started = false;
    defer if (streamed_reasoning_id) |id| allocator.free(id);
    defer if (streamed_message_id) |id| allocator.free(id);

    // Wave 1.A: per-request KV-quant override (Responses path mirror).

    // Per-request PLD override for the Responses path. Mirror the
    // chat-completions auto-disable logic exactly so the same prompt picks
    // the same path on /v1/chat/completions and /v1/responses.
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld_resp: bool = requestPld(root, lm);

    // Drafter (Responses-side parsing). Same disable rules as the chat
    // and Anthropic parse sites.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter_resp = requestEnableDrafter(root.get("enable_drafter"), lm.drafter != null, lm.dflash != null, config.isMoe(), config.has_hybrid_layers);

    // Adaptive spec-decode gate (Responses path; mirrors chat-completions and
    // Anthropic). Score the full prompt's 3-gram repetition; novel content
    // (low score) skips PLD/drafter unless the user explicitly opted in.
    if ((enable_pld_resp and !pld_explicit_in_json) or (enable_drafter_resp and !drafter_explicit_in_json)) {
        const score = pld_index.ngramRepeatScore(allocator, prompt_ids, 3) catch 1.0;
        if (score < spec_gate_threshold) {
            if (enable_pld_resp and !pld_explicit_in_json) {
                log.info("  pld=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_pld_resp = false;
            }
            // DFlash exemption — same rule as the other three surfaces.
            if (enable_drafter_resp and !drafter_explicit_in_json and lm.dflash == null) {
                log.info("  drafter=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_drafter_resp = false;
            }
        }
        // Heavy-echo MTP->PLD routing retired (see the NOTE at the
        // chat-completions site): MTP wins whenever loaded.
    }
    // One token per tick: no draft round can carry the turn past the forced opener.
    if (tool_force != null) {
        enable_pld_resp = false;
        enable_drafter_resp = false;
        enable_mtp_resp = false;
    }

    var result: request_mod.GenerationResult = undefined;
    if (is_stream) {
        // Pick speculative-decoding mode for the streaming Responses path.
        const stream_mode = pickStreamMode(enable_pld_resp, enable_drafter_resp, enable_mtp_resp, lm.drafter != null, lm.dflash != null, mtpCapable(lm), sampling.shapesLogits(), 0, requestHasCompany());
        if (stream_mode == .pld) log.info("  pld=enabled (streaming responses, draft_len={d}, key_len={d})\n", .{ http_server.serverState().server_config.default_pld_draft_len, http_server.serverState().server_config.default_pld_key_len });
        if (stream_mode == .drafter) log.info("  drafter=enabled (streaming responses, block_size={d})\n", .{lm.drafter_block_size});
        if (stream_mode == .mtp) log.info("  mtp=enabled (streaming responses, depth={d})\n", .{lm.mtp_depth});

        var running: ?mlx_engine.Running = null;
        defer if (running) |r| r.done();

        // Transfer vision ownership into the slot.
        const slot_ve_resp = mm.embeddings;
        mm.embeddings = null;
        const slot_mrope = mm.mrope;
        mm.mrope = .{};
        const sch = http_server.launch().global_scheduler.?;
        running = try mlx_engine.Running.submit(sch, .{
            .model = lm,
            .prompt_ids = prompt_ids,
            .full_prompt = prompt_ids,
            .cached_tokens = 0,
            .has_tools = active_has_tools,
            .enable_thinking = enable_thinking,
            .sampling = sampling,
            .eos_token_ids = eos_slice,
            .max_tokens = effective_max_tokens,
            .timeout_ns = getTimeoutNs(),
            .enable_pld = stream_mode == .pld,
            .enable_drafter = stream_mode == .drafter,
            .drafter = if (stream_mode == .drafter) lm.drafter else null,
            .dflash = if (stream_mode == .drafter) lm.dflash else null,
            .drafter_block_size = lm.drafter_block_size,
            .allow_batch_mtp = allow_batch_mtp,
            .enable_mtp = stream_mode == .mtp,
            .mtp = if (stream_mode == .mtp) lm.mtp else null,
            .mtp_depth = lm.mtp_depth,
            .vision_embeddings = slot_ve_resp,
            .media = mm.media,
            .cache_key = cache_key,
            .client = stream.client,
            .mrope_pos = slot_mrope.pos,
            .mrope_total = slot_mrope.total,
            .mrope_delta = slot_mrope.delta,
            .pld_draft_len = http_server.serverState().server_config.default_pld_draft_len,
            .pld_key_len = http_server.serverState().server_config.default_pld_key_len,
            .kv_attn_fused = resolveKvAttnFused(config, kv_attn_explicit, prompt_ids.len, kv_quant_override),
            .logprobs_n = 0,
            .kv_quant_config = kv_quant_override,
        });
        var ts = StreamingTokenStream.initFromStream(running.?.stream, stream_mode, eos_slice);
        defer ts.deinit(allocator);

        var raw_buf = std.ArrayList(u8).empty;
        var stop_stream: StopStream = .{};
        defer stop_stream.deinit(allocator);
        defer raw_buf.deinit(allocator);
        var token_ids_buf = std.ArrayList(u32).empty;
        defer token_ids_buf.deinit(allocator);

        var utf8_carry: [3]u8 = undefined;
        var utf8_carry_len: u8 = 0;
        var stopped = false;
        var client_gone = false;
        // Prompt-opened thinking streams as reasoning even when the request
        // asked for thinking off — generated reasoning is delivered, never
        // dropped (thinking-off is enforced prompt-side, noThinkTailSuffix).
        // Prompt-derived, never flag-derived — see the chat streaming arm.
        const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;
        const opens_think = promptOpensThink(allocator, lm, tok, prompt_ids);
        var in_think_block = opens_think and !constrained_proto;
        var delivery: ?rp_mod.Delivery = if (constrained_proto) rp_mod.Delivery.init(sampling.constraint.?.proto.?) else null;
        defer if (delivery) |*d| d.deinit(allocator);
        var channel_armed = false; // the `<|channel>` marker that may open a Gemma 4 thought
        var think_buf = std.ArrayList(u8).empty;
        defer think_buf.deinit(allocator);
        var skipped_think_open = false;
        // Inkling thinking message seen — its close is <|end_message|>.
        var inkling_think = false;
        var live_output_index: u32 = 0;
        var tool_thought_open = active_has_tools;
        var tool_think_scan: chat_mod.ThinkScan = .{};

        while (true) {
            // A stop cut resolved on the previous token ends the turn here.
            if (stopped or stop_stream.flushed) break;
            const next_id: ?u32 = switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
                .token => |t| t,
                .done => if (stop_stream.holding()) null else break,
                .idle => {
                    // No tokens yet (long prefill). Probe the peer: an abandoned
                    // request must cancel instead of grinding a ghost prefill
                    // (Claude Code retries pile up serially otherwise), and the
                    // keepalive stops clients timing out on stream silence.
                    if (stream.peerClosed()) {
                        log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                        running.?.stream.cancel();
                        client_gone = true;
                        break;
                    }
                    sendStreamKeepalive(stream) catch {
                        log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                        running.?.stream.cancel();
                        client_gone = true;
                        break;
                    };
                    continue;
                },
            };
            if (stream.peerClosed()) {
                running.?.stream.cancel();
                client_gone = true;
                break;
            }
            // UTF-8 carry across BPE-token boundaries (matches chat-completion).
            var token_text = if (next_id) |token_id| blk: {
                try token_ids_buf.append(allocator, token_id);
                const raw_decoded = try decodeTokens(allocator, lm, tok, &[_]u32{token_id}, false);
                if (delivery) |*d| d.noteToken(raw_decoded.len, running.?.stream.constraintPayloadStart());
                const with_carry = if (utf8_carry_len > 0) cc: {
                    const combined = try allocator.alloc(u8, utf8_carry_len + raw_decoded.len);
                    @memcpy(combined[0..utf8_carry_len], utf8_carry[0..utf8_carry_len]);
                    @memcpy(combined[utf8_carry_len..], raw_decoded);
                    allocator.free(raw_decoded);
                    utf8_carry_len = 0;
                    break :cc combined;
                } else raw_decoded;
                const tail = utf8TrailingIncomplete(with_carry);
                if (tail > 0) {
                    @memcpy(utf8_carry[0..tail], with_carry[with_carry.len - tail ..]);
                    utf8_carry_len = @intCast(tail);
                }
                if (with_carry.len == tail) {
                    allocator.free(with_carry);
                    continue;
                }
                if (tail > 0) {
                    const trimmed = try allocator.dupe(u8, with_carry[0 .. with_carry.len - tail]);
                    allocator.free(with_carry);
                    break :blk trimmed;
                }
                break :blk with_carry;
            } else try stop_stream.takeHeld(allocator);
            defer allocator.free(token_text);

            if (stop_sequences.items.len > 0) {
                const fed = try stop_stream.feed(allocator, &raw_buf, token_text, stop_sequences.items, opens_think and !constrained_proto);
                allocator.free(token_text);
                token_text = fed.send;
                if (fed.stop != null) stopped = true;
                if (token_text.len == 0) continue;
            } else try raw_buf.appendSlice(allocator, token_text);

            // Beat BEFORE the tool early-continue below: a tool-active request
            // emits nothing past its thought until generation ends, and the
            // thinking branch holds until its close tag. Both look identical to
            // a dead server from the client's socket.
            beatStreamKeepalive(stream, .sse_comment) catch {
                log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                running.?.stream.cancel();
                client_gone = true;
                break;
            };

            // Tool-active requests hold the answer for the tool-call parse. The
            // leading thought streams as it arrives, in the chat stream's order:
            // tool hold first, then the think gate.
            if (active_has_tools) {
                if (tool_thought_open and !chat_mod.streamShouldBufferForTools(raw_buf.items)) {
                    const gate = chat_mod.streamThinkGateScan(raw_buf.items, enable_thinking, false, opens_think, &tool_think_scan);
                    tool_thought_open = gate == .hold_thinking;
                    if (gate != .flush_text) if (chat_mod.splitThinkBlock(raw_buf.items, true, opens_think).reasoning_content) |rc| {
                        const ready = if (tool_thought_open) chat_mod.streamableReasoning(rc) else rc;
                        if (chat_mod.unstreamedReasoning(ready, tool_reasoning_streamed)) |fresh| {
                            if (!streamed_reasoning_started) {
                                streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                                streamed_reasoning_index = live_output_index;
                                try emitResponsesReasoningStart(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                                streamed_reasoning_started = true;
                            }
                            try emitResponsesReasoningDelta(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, fresh);
                            tool_reasoning_streamed = ready.len;
                        }
                    };
                }
                continue;
            }

            if (delivery) |*d| {
                try d.feed(allocator, token_text);
                try emitConstrainedResponses(allocator, stream, seq_num, &streamed_reasoning_id, &streamed_reasoning_index, &streamed_reasoning_started, &streamed_message_id, &streamed_message_index, &streamed_message_started, d);
                continue;
            }

            if (in_think_block) {
                try think_buf.appendSlice(allocator, token_text);

                // Skip a literal think opener if the template did not pre-inject one.
                if (!skipped_think_open and (think_buf.items.len >= 7 or chat_mod.cannotOpenThink(think_buf.items))) {
                    if (std.mem.startsWith(u8, think_buf.items, "<think>")) {
                        var skip: usize = 7;
                        while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                        const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                        think_buf.clearAndFree(allocator);
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                        skipped_think_open = true;
                    } else if (std.mem.startsWith(u8, think_buf.items, "<|content_thinking|>")) {
                        // Inkling thinking message — closes at <|end_message|>.
                        var skip: usize = "<|content_thinking|>".len;
                        while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                        const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                        think_buf.clearAndFree(allocator);
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                        skipped_think_open = true;
                        inkling_think = true;
                    } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                        var skip: usize = 17;
                        while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                        const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                        think_buf.clearAndFree(allocator);
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                        skipped_think_open = true;
                    } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                        // partial prefix of channel-thought tag; wait for more
                    } else {
                        skipped_think_open = true;
                    }
                }

                // Detect close tag (</think>, <channel|>, or Inkling's
                // <|end_message|>; a leading <|content_text|> is a DIRECT
                // Inkling answer — close immediately with empty reasoning).
                const tp = std.mem.indexOf(u8, think_buf.items, "</think>");
                const cp = std.mem.indexOf(u8, think_buf.items, "<channel|>");
                const close_match: ?struct { pos: usize, tag: []const u8 } = blk: {
                    if (std.mem.startsWith(u8, think_buf.items, "<|content_text|>")) break :blk .{ .pos = 0, .tag = "<|content_text|>" };
                    if (inkling_think) {
                        if (std.mem.indexOf(u8, think_buf.items, "<|end_message|>")) |p| break :blk .{ .pos = p, .tag = "<|end_message|>" };
                    }
                    if (tp == null and cp == null) break :blk null;
                    if (tp == null) break :blk .{ .pos = cp.?, .tag = "<channel|>" };
                    if (cp == null) break :blk .{ .pos = tp.?, .tag = "</think>" };
                    if (tp.? <= cp.?) break :blk .{ .pos = tp.?, .tag = "</think>" };
                    break :blk .{ .pos = cp.?, .tag = "<channel|>" };
                };

                if (close_match) |m| {
                    const before = think_buf.items[0..m.pos];
                    if (before.len > 0) {
                        if (!streamed_reasoning_started) {
                            streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                            streamed_reasoning_index = live_output_index;
                            try emitResponsesReasoningStart(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                            streamed_reasoning_started = true;
                        }
                        try emitResponsesReasoningDelta(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, before);
                    }
                    if (streamed_reasoning_started) live_output_index += 1;

                    const after = m.pos + m.tag.len;
                    var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                    if (std.mem.startsWith(u8, content_after, "<|channel>\n")) {
                        content_after = content_after[11..];
                    } else if (std.mem.startsWith(u8, content_after, "<|channel>")) {
                        content_after = content_after[10..];
                    }
                    if (std.mem.startsWith(u8, content_after, "<|message_model|>")) content_after = content_after["<|message_model|>".len..];
                    if (std.mem.startsWith(u8, content_after, "<|content_text|>")) content_after = content_after["<|content_text|>".len..];
                    if (std.mem.indexOf(u8, content_after, "<|end_message|>")) |em| content_after = content_after[0..em];
                    content_after = std.mem.trimStart(u8, content_after, "\n ");
                    if (content_after.len > 0) {
                        if (!streamed_message_started) {
                            streamed_message_id = try responses_mod.makeId(stream.io, allocator, "msg");
                            streamed_message_index = live_output_index;
                            try emitResponsesMessageStart(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?);
                            streamed_message_started = true;
                        }
                        try emitResponsesMessageDelta(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?, content_after);
                    }
                    think_buf.clearRetainingCapacity();
                    in_think_block = false;
                } else if (skipped_think_open) {
                    const safe_len = thinkHoldBackLen(think_buf.items);
                    if (safe_len > 0) {
                        if (!streamed_reasoning_started) {
                            streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                            streamed_reasoning_index = live_output_index;
                            try emitResponsesReasoningStart(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                            streamed_reasoning_started = true;
                        }
                        try emitResponsesReasoningDelta(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, think_buf.items[0..safe_len]);
                        const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                        think_buf.clearRetainingCapacity();
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                    }
                }
            } else {
                // Skip Gemma 4 channel tags that may leak after the thinking block.
                // A think block the MODEL opens: seeded like a template-injected one.
                // Checked before the marker skip, which would swallow a bare `<think>`.
                if (!streamed_message_started) if (chat_mod.modelThinkOpener(channel_armed, token_text)) |opener| {
                    channel_armed = false;
                    in_think_block = true;
                    try think_buf.appendSlice(allocator, opener);
                    continue;
                };
                if (chat_mod.isChannelMarkerToken(token_text)) {
                    channel_armed = !streamed_message_started and std.mem.eql(u8, token_text, "<|channel>");
                    continue;
                }
                channel_armed = false;
                if (!streamed_message_started) {
                    streamed_message_id = try responses_mod.makeId(stream.io, allocator, "msg");
                    streamed_message_index = live_output_index;
                    try emitResponsesMessageStart(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?);
                    streamed_message_started = true;
                }
                try emitResponsesMessageDelta(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?, token_text);
            }
        }

        if (!client_gone) {
            if (delivery) |*d| {
                d.clearOutput();
                try d.finish(allocator);
                try emitConstrainedResponses(allocator, stream, seq_num, &streamed_reasoning_id, &streamed_reasoning_index, &streamed_reasoning_started, &streamed_message_id, &streamed_message_index, &streamed_message_started, d);
            }
        }

        // Flush any remaining think buffer (no close tag found) as reasoning.
        if (!client_gone and in_think_block and think_buf.items.len > 0 and !active_has_tools) {
            if (!streamed_reasoning_started) {
                streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                streamed_reasoning_index = live_output_index;
                try emitResponsesReasoningStart(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                streamed_reasoning_started = true;
            }
            try emitResponsesReasoningDelta(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, think_buf.items);
        }

        ts.finalize();

        if (client_gone) {
            // Peer disconnected mid-decode. We've already cancelled the slot;
            // tear down without spending more I/O on terminal envelope events
            // (which would just fail or pile into a dead socket buffer).
            log.info("  <- {d}+{d} tokens streamed [client_disconnect]\n", .{ ts.prompt_tokens, ts.completion_tokens });
            return;
        }

        result = ts.generationResult(
            try raw_buf.toOwnedSlice(allocator),
            try token_ids_buf.toOwnedSlice(allocator),
            if (stopped) "stop" else ts.finish_reason,
            if (delivery) |d| d.payload_byte else null,
        );
    } else {
        // Non-streaming Responses: `requestSpecModes` (DFlash > MTP > drafter
        // > PLD) so /v1/responses gets the same speedup as /v1/chat/completions.
        const spec = requestSpecModes(enable_pld_resp, enable_drafter_resp, enable_mtp_resp, lm.drafter != null, lm.dflash != null, mtpCapable(lm), sampling.shapesLogits(), 0, requestHasCompany());
        const use_mtp = spec.use_mtp;
        const use_drafter = spec.use_drafter;
        const use_pld = spec.use_pld;
        // Transfer vision ownership into the slot.
        const slot_ve_ns = mm.embeddings;
        mm.embeddings = null;
        const slot_mrope = mm.mrope;
        mm.mrope = .{};
        // Propagates to `handleResponses`' one error arm, shared with the streaming half.
        result = try nonStreamingViaScheduler(allocator, http_server.launch().global_scheduler.?, lm, tok, prompt_ids, prompt_ids, effective_max_tokens, sampling, eos_slice, 0, active_has_tools, enable_thinking, use_pld, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), slot_ve_ns, mm.media, cache_key, slot_mrope, 0, false, kv_quant_override, kv_attn_explicit, null, stream);
    }
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    // ── apply stop sequences ──
    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    const opens_think = promptOpensThink(allocator, lm, tok, prompt_ids);
    const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;
    if (answerStopCut(result.text, result.text.len, stop_sequences.items, opens_think and !constrained_proto)) |cut| {
        final_text = result.text[0..cut.index];
        finish_reason = "stop";
    }

    const status_str: []const u8 = if (std.mem.eql(u8, finish_reason, "length")) "incomplete" else "completed";

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

    // ── split thinking & tool calls ──
    // Reasoning the model actually generated is always delivered — the
    // request's thinking flag shaped the PROMPT (chat.noThinkTailSuffix),
    // never the delivery.
    var routed: ?rp_mod.Delivery = null;
    defer if (routed) |*d| d.deinit(allocator);
    const think_split = try splitConstrainedResponse(allocator, &routed, sampling, final_text, result.constraint_payload_byte, !has_tools, opens_think);
    const reasoning_text: ?[]const u8 = think_split.reasoning_content;
    const visible_text: []const u8 = think_split.content;

    var tool_calls: ?[]chat_mod.ParsedToolCall = null;
    if (active_has_tools and shouldParseToolCalls(result.finish_details)) {
        tool_calls = try parseToolCallsForRequest(allocator, final_text, active_tools_json, parallel_tool_calls_echo);
    }
    defer if (tool_calls) |tcs| {
        for (tcs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(tcs);
    };

    // ── build output[] array (shared between streaming + non-streaming) ──
    var out_buf = std.ArrayList(u8).empty;
    defer out_buf.deinit(allocator);
    try out_buf.append(allocator, '[');
    var emitted: usize = 0;
    var output_index: u32 = 0;
    var emitted_tool_calls = std.ArrayList(chat_mod.ToolCall).empty;
    defer {
        for (emitted_tool_calls.items) |tc| allocator.free(tc.id);
        emitted_tool_calls.deinit(allocator);
    }

    if (reasoning_text) |rt| if (rt.len > 0) {
        if (emitted > 0) try out_buf.append(allocator, ',');
        if (is_stream and streamed_reasoning_started) {
            // Live deltas already streamed; emit just the closing events with
            // the canonical reasoning text from splitThinkBlock.
            try responses_mod.appendReasoningItem(allocator, &out_buf, streamed_reasoning_id.?, rt);
            if (active_has_tools) if (chat_mod.unstreamedReasoning(rt, tool_reasoning_streamed)) |rest| {
                try emitResponsesReasoningDelta(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, rest);
            };
            try emitResponsesReasoningEnd(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, rt);
        } else {
            const rid = try responses_mod.makeId(stream.io, allocator, "rs");
            defer allocator.free(rid);
            try responses_mod.appendReasoningItem(allocator, &out_buf, rid, rt);
            if (is_stream) {
                try emitResponsesReasoningEvents(allocator, stream, seq_num, output_index, rid, rt);
            }
        }
        emitted += 1;
        output_index += 1;
    };

    if (tool_calls) |tcs| if (tcs.len > 0) {
        for (tcs) |tc| {
            if (!responsesToolExists(root.get("tools"), tc.name, &namespace_aliases)) {
                log.warn("[responses] dropping undeclared tool call: {s}\n", .{tc.name});
                continue;
            }
            if (!isJsonObjectString(allocator, tc.arguments)) {
                log.warn("[responses] dropping tool call with non-object arguments: {s}\n", .{tc.name});
                continue;
            }
            // Split an expanded namespace wire name back into (namespace, name)
            // so the client resolves the call by its declared namespace.
            const split = responses_mod.splitNamespaceToolName(&namespace_aliases, tc.name);
            const emit_name = if (split) |s| s.name else tc.name;
            const emit_ns: ?[]const u8 = if (split) |s| s.namespace else null;
            const fc_id = try responses_mod.makeId(stream.io, allocator, "fc");
            defer allocator.free(fc_id);
            const call_id = try responses_mod.makeId(stream.io, allocator, "call");
            defer allocator.free(call_id);
            const stored_call_id = try allocator.dupe(u8, call_id);
            emitted_tool_calls.append(allocator, .{ .id = stored_call_id, .name = tc.name, .arguments = tc.arguments }) catch |err| {
                allocator.free(stored_call_id);
                return err;
            };
            if (emitted > 0) try out_buf.append(allocator, ',');
            try responses_mod.appendFunctionCallItem(allocator, &out_buf, fc_id, call_id, emit_name, tc.arguments, emit_ns);
            emitted += 1;
            if (is_stream) {
                try emitResponsesFunctionCallEvents(allocator, stream, seq_num, output_index, fc_id, call_id, emit_name, tc.arguments, emit_ns);
            }
            output_index += 1;
        }
    };

    const has_tool_calls = emitted_tool_calls.items.len > 0;
    if (!has_tool_calls) {
        if (emitted > 0) try out_buf.append(allocator, ',');
        if (is_stream and streamed_message_started) {
            // Live deltas already streamed; emit just the closing events.
            try responses_mod.appendOutputTextMessage(allocator, &out_buf, streamed_message_id.?, visible_text);
            try emitResponsesMessageEnd(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?, visible_text);
        } else {
            const mid = try responses_mod.makeId(stream.io, allocator, "msg");
            defer allocator.free(mid);
            try responses_mod.appendOutputTextMessage(allocator, &out_buf, mid, visible_text);
            if (is_stream) {
                try emitResponsesMessageEvents(allocator, stream, seq_num, output_index, mid, visible_text);
            }
        }
        emitted += 1;
        output_index += 1;
    }

    try out_buf.append(allocator, ']');

    // ── envelope ──
    const is_incomplete = std.mem.eql(u8, status_str, "incomplete");
    const is_completed_status = std.mem.eql(u8, status_str, "completed") or is_incomplete;
    // reasoning token count for usage.output_tokens_details (re-encode of the
    // split reasoning text — exact modulo merge boundaries).
    const reasoning_tok_count: u32 = blk: {
        const rt = reasoning_text orelse break :blk 0;
        const rids = encodeText(allocator, lm, tok, rt, false) catch break :blk 0;
        defer allocator.free(rids);
        break :blk @intCast(rids.len);
    };
    const envelope = try buildResponsesEnvelope(
        stream.io,
        allocator,
        esc_resp_id,
        esc_model,
        status_str,
        out_buf.items,
        result.prompt_tokens,
        result.completion_tokens,
        result.cached_tokens,
        reasoning_tok_count,
        should_store,
        prev_id,
        is_incomplete,
        is_completed_status,
        response_echo,
        result.prefill_ns,
        result.decode_ns,
        tokenize_ns,
        result.completion_tokens,
    );
    defer allocator.free(envelope);

    // ── store response ──
    if (should_store) {
        const stored_tool_calls: ?[]const chat_mod.ToolCall = if (emitted_tool_calls.items.len > 0) emitted_tool_calls.items else null;
        storeResponse(stream.io, allocator, resp_id, model_name, status_str, envelope, pi.messages.items, visible_text, reasoning_text, stored_tool_calls, result.finish_details) catch |err| {
            log.warn("[responses] store failed: {s}\n", .{@errorName(err)});
        };
    }

    if (is_stream) {
        const completed_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.completed\",\"response\":{s}}}", .{envelope});
        defer allocator.free(completed_payload);
        try sendResponsesEvent(allocator, stream, seq_num, "response.completed", completed_payload);
        // OpenAI terminates the Responses HTTP SSE stream with the same
        // `data: [DONE]` sentinel as chat completions; generic SSE middleware
        // keys stream end off it. The WS transport must NOT get one — its
        // per-response terminator is the `response.completed` event, and a
        // trailing [DONE] frame would be misread as the NEXT turn's marker
        // on a chained session (see the WS turn loop below).
        if (stream.ws_mode == null) {
            logHttpSseData("[DONE]");
            try stream.writeAll("data: [DONE]\n\n");
        }
    } else {
        try sendResponse(stream, "200 OK", "application/json", envelope);
    }
}

// ─── WebSocket transport for /v1/responses ────────────────────────────────

// ── tests ──

test "responsesToolExists validates Responses function tool names" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\[{"type":"function","name":"smartSearch","parameters":{}},{"type":"web_search"}]
    , .{});
    defer parsed.deinit();

    try testing.expect(responsesToolExists(parsed.value, "smartSearch", null));
    try testing.expect(!responsesToolExists(parsed.value, "cruise_cards", null));
}

test "responsesToolExists accepts an expanded namespace wire name" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\[{"type":"namespace","name":"mcp__demo__","tools":[{"type":"function","name":"get_weather"}]}]
    , .{});
    defer parsed.deinit();
    var aliases = responses_mod.NamespaceAliases.init(testing.allocator);
    defer responses_mod.freeNamespaceAliases(testing.allocator, &aliases);
    const reshaped = try responses_mod.buildToolsJson(testing.allocator, parsed.value.array, &aliases);
    defer testing.allocator.free(reshaped);
    try testing.expect(responsesToolExists(parsed.value, "mcp__demo__get_weather", &aliases));
    // A unique bare member name resolves too, and splits back to its namespace.
    try testing.expect(responsesToolExists(parsed.value, "get_weather", &aliases));
    try testing.expectEqualStrings("mcp__demo__", responses_mod.splitNamespaceToolName(&aliases, "get_weather").?.namespace);
}

test "buildResponsesJsonInstruction scopes schema prompt when tools are active" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"object","properties":{"blocks":{"type":"array"}},"required":["blocks"]}
    , .{});
    defer parsed.deinit();

    const with_tools = try buildResponsesJsonInstruction(testing.allocator, parsed.value, true);
    defer testing.allocator.free(with_tools);
    try testing.expect(std.mem.indexOf(u8, with_tools, "If you answer without calling a function") != null);
    try testing.expect(std.mem.indexOf(u8, with_tools, "function call is needed") != null);
    try testing.expect(std.mem.indexOf(u8, with_tools, "\"blocks\"") != null);

    const without_tools = try buildResponsesJsonInstruction(testing.allocator, parsed.value, false);
    defer testing.allocator.free(without_tools);
    try testing.expect(std.mem.indexOf(u8, without_tools, "Respond with valid JSON only") != null);
    try testing.expect(std.mem.indexOf(u8, without_tools, "without calling a function") == null);
}

test "shouldInjectResponsesJsonInstruction skips required tool turns" {
    try testing.expect(!shouldInjectResponsesJsonInstruction(false, false, null));
    try testing.expect(shouldInjectResponsesJsonInstruction(true, false, null));
    try testing.expect(shouldInjectResponsesJsonInstruction(true, true, null));
    try testing.expect(!shouldInjectResponsesJsonInstruction(true, true, "required"));
}

test "isJsonObjectString only accepts JSON objects" {
    try testing.expect(isJsonObjectString(testing.allocator, "{\"destination\":\"CARIBBEAN\"}"));
    try testing.expect(!isJsonObjectString(testing.allocator, "[]"));
    try testing.expect(!isJsonObjectString(testing.allocator, "not-json"));
}
