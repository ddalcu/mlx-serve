//! completion prompt parsing, echo, n; handleCompletions, non/streaming completion

const std = @import("std");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const request_mod = @import("../../sdk/request/request.zig");
const log = @import("../../sdk/util/log.zig");
const pld_index = @import("../../sdk/spec/pld_index.zig");
const mlx_engine = @import("../engines/mlx.zig");
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
const resolveKvAttnFused = @import("request_options.zig").resolveKvAttnFused;
const resolveSampling = @import("sampling.zig").resolveSampling;
const generationMaxTokens = @import("sampling.zig").generationMaxTokens;
const requestProfile = @import("sampling.zig").requestProfile;
const defaultEnableMtp = @import("request_options.zig").defaultEnableMtp;
const dsv4DraftStages = @import("request_options.zig").dsv4DraftStages;
const mtpCapable = @import("request_options.zig").mtpCapable;
const admitMtpForCtx = @import("request_options.zig").admitMtpForCtx;
const getTimeoutNs = @import("../http/server.zig").getTimeoutNs;
const decodeTokens = @import("tokenize.zig").decodeTokens;
const omittedMaxTokensDefault = @import("sampling.zig").omittedMaxTokensDefault;
const launchMaxTokensDefault = @import("sampling.zig").launchMaxTokensDefault;
const parseRequestSeed = @import("sampling.zig").parseRequestSeed;
const maxTokensOrigin = @import("sampling.zig").maxTokensOrigin;
const MAX_TOKENS_DESC_LEN = @import("sampling.zig").MAX_TOKENS_DESC_LEN;
const describeMaxTokens = @import("sampling.zig").describeMaxTokens;
const requestEosSlice = @import("sampling.zig").requestEosSlice;
const finishDetailsField = @import("../chat/tools/pipeline.zig").finishDetailsField;
const getEffectiveContextLength = @import("admission.zig").getEffectiveContextLength;
const logStreamError = @import("../http/errors.zig").logStreamError;
const sendGenerationError = @import("../http/errors.zig").sendGenerationError;
const checkAttentionMemory = @import("admission.zig").checkAttentionMemory;
const clampMaxTokens = @import("sampling.zig").clampMaxTokens;
const encodeText = @import("tokenize.zig").encodeText;
const jsonEscapeOrEmpty = @import("tokenize.zig").jsonEscapeOrEmpty;
const requestCacheKey = @import("request_text.zig").requestCacheKey;
const echoModelName = @import("request_text.zig").echoModelName;
const requestPld = @import("request_options.zig").requestPld;
const StopStream = @import("../chat/stops.zig").StopStream;
const nonStreamingViaScheduler = @import("submit.zig").nonStreamingViaScheduler;
const StreamingTokenStream = @import("chat_stream.zig").StreamingTokenStream;
const requestEnableDrafter = @import("request_options.zig").requestEnableDrafter;
const requestHasCompany = @import("request_options.zig").requestHasCompany;
const requestSpecModes = @import("request_options.zig").requestSpecModes;
const pickStreamMode = @import("request_options.zig").pickStreamMode;
const StreamLogprobs = @import("logprobs.zig").StreamLogprobs;
const sendResponse = @import("../http/respond.zig").sendResponse;
const SSE_ALLOW_HEADERS_DEFAULT = @import("../http/respond.zig").SSE_ALLOW_HEADERS_DEFAULT;
const sendSseHeaders = @import("../http/respond.zig").sendSseHeaders;
const logHttpSseData = @import("../http/respond.zig").logHttpSseData;
const countRejected = @import("../http/errors.zig").countRejected;
const contextOverflowMessage = @import("../http/errors.zig").contextOverflowMessage;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const utf8TrailingIncomplete = @import("format.zig").utf8TrailingIncomplete;
const formatChatUsage = @import("format.zig").formatChatUsage;
const formatPerfBracket = @import("format.zig").formatPerfBracket;
const jsonEscape = @import("format.zig").jsonEscape;
const formatCompletionsLogprobs = @import("logprobs.zig").formatCompletionsLogprobs;
const sendStreamKeepalive = @import("../http/sse.zig").sendStreamKeepalive;

/// The vocabulary a request's token ids index. An embedded engine's stub config keeps the
/// default vocab; its ids are the engine's, and an id past its vocab aborts inside llama.cpp.
pub fn tokenVocabSize(lm: *const LoadedModel) u32 {
    if (lm.engine) |e| if (e.caps().owns_tokenizer) return e.vocabSize();
    return if (lm.config) |c| c.vocab_size else 0;
}

/// Client token ids into `out` (same length); false unless every item is an integer in [0, vocab_size).
pub fn parseTokenIds(items: []const std.json.Value, vocab_size: u32, out: []u32) bool {
    for (items, out) |item, *id| {
        if (item != .integer or item.integer < 0 or item.integer >= vocab_size) return false;
        id.* = @intCast(item.integer);
    }
    return true;
}

const CompletionPrompt = union(enum) {
    text: []const u8,
    /// Caller frees.
    ids: []u32,
    invalid: []const u8,
    missing,
};

/// `/v1/completions` `prompt`: a string or token ids (OpenAI's `int[]`, what lm-eval
/// sends), either also as a one-item list. A real batch is a named 400, like `n`.
fn parseCompletionPrompt(allocator: std.mem.Allocator, v: ?std.json.Value, vocab_size: u32) !CompletionPrompt {
    const val = v orelse return .missing;
    switch (val) {
        .null => return .missing,
        .string => |s| return .{ .text = s },
        .array => |arr| {
            if (arr.items.len == 0) return .{ .invalid = "'prompt' is empty" };
            if (arr.items[0] == .string or arr.items[0] == .array) {
                if (arr.items.len > 1) return .{ .invalid = "batched prompts are not supported: send one prompt per request" };
                return parseCompletionPrompt(allocator, arr.items[0], vocab_size);
            }
            const ids = try allocator.alloc(u32, arr.items.len);
            if (!parseTokenIds(arr.items, vocab_size, ids)) {
                allocator.free(ids);
                return .{ .invalid = "'prompt' token ids must be integers inside the model's vocabulary" };
            }
            return .{ .ids = ids };
        },
        else => return .{ .invalid = "'prompt' must be a string or a list of token ids" },
    }
}

pub const PROMPT_LOGPROBS_UNAVAILABLE_MSG = "'echo' with 'logprobs' is not supported on this model: its prefill does not return every prompt position's logits";

/// Prompt logprobs read every row of the MLX prefill's logits. An embedded engine never
/// forwards the prompt row by row; the module-owned DeepSeek forwards return the last row only.
const PromptLogprobsSupport = struct { engine_backed: bool = false, module_owned: bool = false };

fn promptLogprobsRejectReason(model: PromptLogprobsSupport) ?[]const u8 {
    return if (model.engine_backed or model.module_owned) PROMPT_LOGPROBS_UNAVAILABLE_MSG else null;
}

fn echoRequested(root: std.json.ObjectMap) bool {
    const v = root.get("echo") orelse return false;
    return v == .bool and v.bool;
}

/// Echo is served non-streaming only (what lm-eval sends); a stream is refused by
/// name rather than silently dropping the echoed prompt.
fn echoRejectReason(root: std.json.ObjectMap) ?[]const u8 {
    if (!echoRequested(root)) return null;
    const st = root.get("stream") orelse return null;
    if (st != .bool or !st.bool) return null;
    return "'echo' is not supported with 'stream': send the request with \"stream\": false";
}

pub fn nChoicesRejectReason(root: std.json.ObjectMap) ?[]const u8 {
    const v = root.get("n") orelse return null;
    switch (v) {
        .null => return null,
        .integer => |i| if (i == 1) return null,
        .float => |f| if (f == 1.0) return null,
        else => {},
    }
    return "'n' > 1 is not supported: this server returns a single choice per request; omit 'n' or set it to 1";
}

pub fn handleCompletions(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const config = lm.config.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    if (parsed.value != .object) {
        log.warn("POST /v1/completions -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;
    const cache_key = requestCacheKey(root);

    if (nChoicesRejectReason(root)) |reason| {
        log.warn("POST /v1/completions -> 400 (unsupported n)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }
    if (echoRejectReason(root)) |reason| {
        log.warn("POST /v1/completions -> 400 (unsupported echo)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }

    // Legacy-completions `logprobs` is an INTEGER (how many alternatives per
    // token), not chat's bool + `top_logprobs`. It was parsed nowhere and the
    // handler passed a hardcoded 0 while still emitting a `logprobs` key — a
    // silently ignored field, which reads to a client as "this model has no
    // opinion" rather than "this server never asked".
    const logprobs_n: u32 = if (root.get("logprobs")) |v| switch (v) {
        .integer => |i| @intCast(@min(@max(i, 0), request_mod.MAX_TOP_LOGPROBS)),
        else => 0,
    } else 0;

    const prompt_logprobs = echoRequested(root) and logprobs_n > 0;
    if (prompt_logprobs) if (promptLogprobsRejectReason(.{
        .engine_backed = lm.engineBacked(),
        .module_owned = mlx_engine.facts(lm).module_owned,
    })) |reason| {
        log.warn("POST /v1/completions -> 400 (echo logprobs: prefill returns no per-row logits)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    };

    const prompt = try parseCompletionPrompt(allocator, root.get("prompt"), tokenVocabSize(lm));
    defer if (prompt == .ids) allocator.free(prompt.ids);
    switch (prompt) {
        .missing, .invalid => {
            const reason = if (prompt == .invalid) prompt.invalid else "'prompt' is a required field";
            log.warn("POST /v1/completions -> 400 ({s})\n", .{reason});
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        },
        .text, .ids => {},
    }

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

    const sampled = resolveSampling(gen, root, config);
    const temperature = sampled.temperature;
    const top_p = sampled.top_p;
    const top_k = sampled.top_k;
    const min_p = sampled.min_p;

    const repeat_penalty = sampled.repeat_penalty;
    const presence_penalty_c = sampled.presence_penalty;

    const seed: ?u64 = parseRequestSeed(root.get("seed"));

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

    const model_name = echoModelName(root, config.model_type);

    const include_usage = if (root.get("stream_options")) |so| blk: {
        if (so != .object) break :blk false;
        if (so.object.get("include_usage")) |iu| {
            break :blk iu == .bool and iu.bool;
        }
        break :blk false;
    } else false;

    // Spec-decode flags (mirror chat-completions). FIM / code-completion
    // prompts are echo-heavy, so the old hardcoded enable_pld/enable_drafter
    // = false at submit left real speedups unused on this endpoint
    // (tests/test_completions_spec.sh). Embedded engines (ds4/llama) ignore
    // these flags at dispatch.
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld: bool = requestPld(root, lm);
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter = requestEnableDrafter(root.get("enable_drafter"), lm.drafter != null, lm.dflash != null, config.isMoe(), config.has_hybrid_layers);
    if (enable_drafter and enable_pld) enable_pld = false;
    const allow_batch_mtp = if (root.get("enable_batch_mtp")) |v| v != .bool or v.bool else true;
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        defaultEnableMtp(lm.mtp != null, dsv4DraftStages(lm));
    if (enable_mtp and lm.mtp == null and !dsv4DraftStages(lm)) enable_mtp = false;

    // Log the request
    var max_tokens_desc_buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    log.info("POST /v1/completions (max_tokens={s}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}) \n", .{ describeMaxTokens(&max_tokens_desc_buf, max_tokens, max_tokens_origin), temperature, top_p, top_k, is_stream });
    switch (prompt) {
        .text => |t| log.info("  > \"{s}{s}\"\n", .{ t[0..@min(t.len, 80)], if (t.len > 80) "..." else "" }),
        .ids => |ids| log.info("  > [{d} token ids]\n", .{ids.len}),
        else => unreachable,
    }

    // Tokenize prompt directly (no chat template). ds4-backed models
    // tokenize through the engine's GGUF vocab; MLX models go through
    // the loaded BPE tokenizer. Token ids are taken as sent.
    const prompt_ids = switch (prompt) {
        .text => |t| try encodeText(allocator, lm, tok, t, true),
        .ids => |ids| try allocator.dupe(u32, ids),
        else => unreachable,
    };
    defer allocator.free(prompt_ids);
    enable_mtp = admitMtpForCtx(enable_mtp, prompt_ids.len);

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }

    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    // Check if attention computation would exceed GPU memory.
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, false, null, lm, enable_mtp, 0)) return;

    // Adaptive spec-decode gate (mirrors chat-completions): novel prompts
    // (low 3-gram repetition) skip PLD/drafter unless explicitly requested.
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
        // Heavy-echo MTP->PLD routing retired (see the NOTE at the
        // chat-completions site): MTP wins whenever loaded.
    }

    const eos_slice = requestEosSlice(config, root);
    const sampling = request_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .min_p = min_p,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty_c,
        .frequency_penalty = sampled.frequency_penalty,
        .seed = seed,
    };

    if (is_stream) {
        handleStreamingCompletion(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, enable_pld, enable_drafter, enable_mtp, allow_batch_mtp, logprobs_n, cache_key) catch |err| {
            logStreamError(err);
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    } else {
        // The echoed text: the prompt as sent, or its ids decoded.
        const echo_text: ?[]const u8 = if (!echoRequested(root)) null else switch (prompt) {
            .text => |t| t,
            .ids => try decodeTokens(allocator, lm, tok, prompt_ids, false),
            else => unreachable,
        };
        defer if (prompt == .ids) if (echo_text) |t| allocator.free(t);
        handleNonStreamingCompletion(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, enable_pld, enable_drafter, enable_mtp, allow_batch_mtp, logprobs_n, echo_text, prompt_logprobs, cache_key) catch |err| {
            log.err("  -> {s}\n", .{@errorName(err)});
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    }
}

fn handleNonStreamingCompletion(
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
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    logprobs_n: u32,
    /// `echo`: the prompt text to prepend; `prompt_logprobs` also scores its tokens.
    echo_text: ?[]const u8,
    prompt_logprobs: bool,
    cache_key: u64,
) !void {
    var timer = Stopwatch.init(stream.io);

    // Spec dispatch: `requestSpecModes` (DFlash > MTP > drafter > PLD).
    // logprobs needs every step's own distribution, so it disables speculation
    // here exactly as it does on chat.
    const spec = requestSpecModes(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.dflash != null, mtpCapable(lm), sampling.shapesLogits(), logprobs_n, requestHasCompany());
    const use_mtp = spec.use_mtp;
    const use_drafter = spec.use_drafter;
    const use_pld = spec.use_pld;

    // Every failure class propagates to the surface's one error arm (`sendGenerationError`), shared with the streaming twin.
    var result = try nonStreamingViaScheduler(allocator, http_server.launch().global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, false, false, use_pld, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), null, &.{}, cache_key, .{}, logprobs_n, prompt_logprobs, null, null, null, stream);
    _ = &result;
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    defer for ([_]?[]request_mod.LogprobResult{ result.logprobs, result.prompt_logprobs }) |maybe| if (maybe) |lps| {
        for (lps) |*lp| allocator.free(lp.top_logprobs);
        allocator.free(lps);
    };

    // Text completion is a raw continuation: keep the first token's leading
    // space (SentencePiece `▁`) instead of the chat-style strip the scheduler
    // applies — FIM clients rely on exact indentation, and the streaming
    // handler never stripped it, so this also restores stream/non-stream
    // parity (tests/test_completions_spec.sh).
    const raw_text = try decodeTokens(allocator, lm, tok, result.token_ids, false);
    defer allocator.free(raw_text);

    var final_text: []const u8 = raw_text;
    var finish_reason = result.finish_reason;
    for (stop_sequences) |stop_seq| {
        if (std.mem.indexOf(u8, final_text, stop_seq)) |idx| {
            final_text = final_text[0..idx];
            finish_reason = "stop";
            break;
        }
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, finish_reason,
    });

    const shown_text = if (echo_text) |e| try std.mem.concat(allocator, u8, &.{ e, final_text }) else final_text;
    defer if (echo_text != null) allocator.free(shown_text);
    const escaped = jsonEscapeOrEmpty(allocator, shown_text);
    const escaped_text = escaped.slice;
    defer if (escaped.owned) allocator.free(escaped.slice);

    // `null` unless the request asked: OpenAI omits the field's content rather
    // than shipping an empty object that reads as "no alternatives exist".
    var lp_offset_base: usize = 0;
    const lp_json: []const u8 = if (logprobs_n == 0 or result.logprobs == null)
        "null"
    else if (prompt_logprobs) blk: {
        // Echo: the prompt's tokens lead, its first unscored (OpenAI's `null`).
        const ids = try std.mem.concat(allocator, u32, &.{ prompt_ids, result.token_ids });
        defer allocator.free(ids);
        const lps = try std.mem.concat(allocator, request_mod.LogprobResult, &.{ result.prompt_logprobs orelse &.{}, result.logprobs.? });
        defer allocator.free(lps);
        break :blk try formatCompletionsLogprobs(allocator, tok, ids, lps, true, &lp_offset_base);
    } else try formatCompletionsLogprobs(allocator, tok, result.token_ids, result.logprobs.?, false, &lp_offset_base);
    defer if (!std.mem.eql(u8, lp_json, "null")) allocator.free(lp_json);

    // The usage object is chat completions' (`formatChatUsage`): prompt_tokens_details.cached_tokens always present.
    const usage_obj = try formatChatUsage(allocator, result.prompt_tokens, result.completion_tokens, result.cached_tokens, "");
    defer allocator.free(usage_obj);
    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"text":{s},"logprobs":{s},"finish_reason":"{s}"{s}}}],"usage":{s}}}
    , .{
        nowMs(stream.io),
        nowSecs(stream.io),
        model_name,
        escaped_text,
        lp_json,
        finish_reason,
        finishDetailsField(finish_reason, result.finish_details),
        usage_obj,
    });
    defer allocator.free(response);

    try sendResponse(stream, "200 OK", "application/json", response);
}

fn handleStreamingCompletion(
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
    include_usage: bool,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    logprobs_n: u32,
    cache_key: u64,
) !void {
    const cmpl_id = nowMs(stream.io);
    const created_ts = nowSecs(stream.io);
    var timer = Stopwatch.init(stream.io);

    const stream_mode = pickStreamMode(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.dflash != null, mtpCapable(lm), sampling.shapesLogits(), logprobs_n, requestHasCompany());
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ http_server.serverState().server_config.default_pld_draft_len, http_server.serverState().server_config.default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});

    var running: ?mlx_engine.Running = null;
    defer if (running) |r| r.done();

    const sch = http_server.launch().global_scheduler.?;
    running = try mlx_engine.Running.submit(sch, .{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = false,
        .enable_thinking = false,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
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
        .pld_draft_len = http_server.serverState().server_config.default_pld_draft_len,
        .pld_key_len = http_server.serverState().server_config.default_pld_key_len,
        .kv_attn_fused = resolveKvAttnFused(lm.config.?, null, prompt_ids.len, null),
        .logprobs_n = logprobs_n,
        .cache_key = cache_key,
        .client = stream.client,
    });
    var ts = StreamingTokenStream.initFromStream(running.?.stream, stream_mode, eos_token_ids);

    // Legacy completions carries logprobs as four parallel arrays whose
    // `text_offset` indexes the whole completion, so the collector renders
    // the legacy shape and keeps the running offset across chunks.
    var lps = StreamLogprobs{
        .allocator = allocator,
        .tok = tok,
        .stream = running.?.stream,
        .enabled = logprobs_n > 0,
        .legacy = true,
    };
    defer lps.deinit();
    defer ts.deinit(allocator);

    // SSE headers
    try sendSseHeaders(stream, "completions", SSE_ALLOW_HEADERS_DEFAULT);

    var text_buf = std.ArrayList(u8).empty;
    var stop_stream: StopStream = .{};
    defer stop_stream.deinit(allocator);
    defer text_buf.deinit(allocator);
    var stopped = false;
    var utf8_carry_c: [3]u8 = undefined;
    var utf8_carry_c_len: u8 = 0;
    var client_gone = false;

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
        // Handle incomplete UTF-8 sequences across token boundaries
        var token_text = if (next_id) |token_id| blk: {
            try lps.note(token_id);
            const strip = tok.tok_type == .sentencepiece_bpe;
            const raw_decoded_c = try decodeTokens(allocator, lm, tok, &[_]u32{token_id}, strip and false);
            const with_carry = if (utf8_carry_c_len > 0) cc: {
                const combined = try allocator.alloc(u8, utf8_carry_c_len + raw_decoded_c.len);
                @memcpy(combined[0..utf8_carry_c_len], utf8_carry_c[0..utf8_carry_c_len]);
                @memcpy(combined[utf8_carry_c_len..], raw_decoded_c);
                allocator.free(raw_decoded_c);
                utf8_carry_c_len = 0;
                break :cc combined;
            } else raw_decoded_c;

            const tail = utf8TrailingIncomplete(with_carry);
            if (tail > 0) {
                @memcpy(utf8_carry_c[0..tail], with_carry[with_carry.len - tail ..]);
                utf8_carry_c_len = @intCast(tail);
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

        if (stop_sequences.len > 0) {
            const fed = try stop_stream.feed(allocator, &text_buf, token_text, stop_sequences, null);
            allocator.free(token_text);
            token_text = fed.send;
            if (fed.stop != null) stopped = true;
            if (token_text.len == 0) continue;
        }
        lps.sent(token_text.len, utf8_carry_c_len);

        const escaped = try jsonEscape(allocator, token_text);
        defer allocator.free(escaped);
        // Legacy `logprobs` is a sibling of `text` on the choice. OpenAI sends
        // `null` on a chunk that has none, never an absent key.
        const lp_field = (try lps.take()) orelse "null";
        const chunk = try std.fmt.allocPrint(allocator,
            \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"text":{s},"logprobs":{s},"finish_reason":null}}]}}
        , .{ cmpl_id, created_ts, model_name, escaped, lp_field });
        defer allocator.free(chunk);

        logHttpSseData(chunk);
        try stream.writeAllNoFlush("data: ");
        try stream.writeAllNoFlush(chunk);
        try stream.writeAllNoFlush("\n\n");
        try stream.flush();
    }

    // Final chunk with finish_reason
    ts.finalize();
    const finish_reason = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;
    const total_prompt = ts.prompt_tokens;

    if (!client_gone) {
        const final_lp = (try lps.take()) orelse "null";
        const final_chunk = try std.fmt.allocPrint(allocator,
            \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"text":"","logprobs":{s},"finish_reason":"{s}"{s}}}]}}
        , .{ cmpl_id, created_ts, model_name, final_lp, finish_reason, finishDetailsField(finish_reason, ts.finish_details) });
        defer allocator.free(final_chunk);

        logHttpSseData(final_chunk);
        try stream.writeAllNoFlush("data: ");
        try stream.writeAllNoFlush(final_chunk);
        try stream.writeAllNoFlush("\n\n");

        // Usage rides its own empty-choices chunk (OpenAI's stream_options
        // shape) — never the finish chunk, so the ending is stated once.
        if (include_usage) {
            const usage_obj = try formatChatUsage(allocator, total_prompt, ts.completion_tokens, ts.cached_tokens, "");
            defer allocator.free(usage_obj);
            const usage_chunk = try std.fmt.allocPrint(allocator,
                \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[],"usage":{s}}}
            , .{ cmpl_id, created_ts, model_name, usage_obj });
            defer allocator.free(usage_chunk);
            logHttpSseData(usage_chunk);
            try stream.writeAllNoFlush("data: ");
            try stream.writeAllNoFlush(usage_chunk);
            try stream.writeAllNoFlush("\n\n");
        }

        logHttpSseData("[DONE]");
        try stream.writeAll("data: [DONE]\n\n");
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed ({d}ms) [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, elapsed_ms, perf, finish_reason,
    });
}

// ── tests ──

test "parseCompletionPrompt: a token-id prompt is a prompt, a batch is a named 400 (#659)" {
    const allocator = std.testing.allocator;
    const Want = union(enum) { text: []const u8, ids: []const u32, invalid, missing };
    const cases = [_]struct { body: []const u8, want: Want }{
        .{ .body = "{\"prompt\":\"hi\"}", .want = .{ .text = "hi" } },
        .{ .body = "{\"prompt\":[\"hi\"]}", .want = .{ .text = "hi" } },
        .{ .body = "{\"prompt\":[1,2,3]}", .want = .{ .ids = &.{ 1, 2, 3 } } },
        .{ .body = "{\"prompt\":[[5,6]]}", .want = .{ .ids = &.{ 5, 6 } } },
        .{ .body = "{\"prompt\":[[1],[2]]}", .want = .invalid },
        .{ .body = "{\"prompt\":[\"a\",\"b\"]}", .want = .invalid },
        .{ .body = "{\"prompt\":[]}", .want = .invalid },
        .{ .body = "{\"prompt\":[1,-2]}", .want = .invalid },
        .{ .body = "{\"prompt\":[1,100]}", .want = .invalid },
        .{ .body = "{}", .want = .missing },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = try parseCompletionPrompt(allocator, parsed.value.object.get("prompt"), 100);
        defer if (got == .ids) allocator.free(got.ids);
        switch (case.want) {
            .text => |t| try std.testing.expectEqualStrings(t, got.text),
            .ids => |ids| try std.testing.expectEqualSlices(u32, ids, got.ids),
            .invalid => try std.testing.expect(got == .invalid),
            .missing => try std.testing.expect(got == .missing),
        }
    }
}

test "parseTokenIds: only integers inside the vocabulary reach the decoder" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, ok: bool }{
        .{ .body = "[]", .ok = true },
        .{ .body = "[0,99]", .ok = true },
        .{ .body = "[-1]", .ok = false },
        .{ .body = "[100]", .ok = false },
        .{ .body = "[1099511627776]", .ok = false },
        .{ .body = "[1.5]", .ok = false },
        .{ .body = "[\"a\",null]", .ok = false },
    };
    var out: [4]u32 = undefined;
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const items = parsed.value.array.items;
        try std.testing.expectEqual(case.ok, parseTokenIds(items, 100, out[0..items.len]));
    }
}

test "promptLogprobsRejectReason: only a prefill that returns every row can score the prompt" {
    try std.testing.expect(promptLogprobsRejectReason(.{}) == null);
    try std.testing.expect(promptLogprobsRejectReason(.{ .engine_backed = true }) != null);
    try std.testing.expect(promptLogprobsRejectReason(.{ .module_owned = true }) != null);
}

test "echoRejectReason: echo is served without stream, refused by name with it" {
    const allocator = std.testing.allocator;
    for ([_]struct { body: []const u8, rejected: bool }{
        .{ .body = "{}", .rejected = false },
        .{ .body = "{\"echo\":false,\"stream\":true}", .rejected = false },
        .{ .body = "{\"echo\":true}", .rejected = false },
        .{ .body = "{\"echo\":true,\"stream\":false}", .rejected = false },
        .{ .body = "{\"echo\":true,\"stream\":true}", .rejected = true },
    }) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(case.rejected, echoRejectReason(parsed.value.object) != null);
    }
}

test "nChoicesRejectReason: n>1 earns an honest 400, single-choice spellings pass" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, rejected: bool }{
        // Absent, explicit null, and the two spellings of "one choice" pass.
        .{ .body = "{}", .rejected = false },
        .{ .body = "{\"n\":null}", .rejected = false },
        .{ .body = "{\"n\":1}", .rejected = false },
        .{ .body = "{\"n\":1.0}", .rejected = false },
        // Anything else is a request for multi-choice (or garbage) — reject
        // instead of the silent one-choice no-op.
        .{ .body = "{\"n\":2}", .rejected = true },
        .{ .body = "{\"n\":0}", .rejected = true },
        .{ .body = "{\"n\":\"2\"}", .rejected = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const reason = nChoicesRejectReason(parsed.value.object);
        try std.testing.expectEqual(case.rejected, reason != null);
    }
}
