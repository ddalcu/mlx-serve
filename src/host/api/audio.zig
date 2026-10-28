//! transcriptions, aligner, diarizer companions

const speech_api = @import("speech.zig");
const wav_mod = @import("../../sdk/media/wav.zig");
const cachedFormatChat = @import("format.zig").cachedFormatChat;
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const request_mod = @import("../../sdk/request/request.zig");
const chat_mod = @import("../chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const scheduler_mod = @import("../gpu_thread/requests.zig");
const model_registry_mod = @import("../registry/registry.zig");
const transcription_mod = @import("audio/transcription.zig");
const forced_align_mod = @import("../../plugins/qwen3_asr/forced_align.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const PluginEngine = @import("../media_jobs/plugin/engine.zig").Engine;
const plugin_http = @import("../media_jobs/plugin/http.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const Stopwatch = io_util.Stopwatch;
const http_server = @import("../http/server.zig");
const mlx_engine = @import("../engines/mlx.zig");

const Conn = @import("../http/conn.zig").Conn;
const getTimeoutNs = @import("../http/server.zig").getTimeoutNs;
const decodeTokens = @import("tokenize.zig").decodeTokens;
const getEffectiveContextLength = @import("admission.zig").getEffectiveContextLength;
const sendGenerationError = @import("../http/errors.zig").sendGenerationError;
const checkAttentionMemory = @import("admission.zig").checkAttentionMemory;
const clampMaxTokens = @import("sampling.zig").clampMaxTokens;
const encodeText = @import("tokenize.zig").encodeText;
const jsonEscapeOrEmpty = @import("tokenize.zig").jsonEscapeOrEmpty;
const nonStreamingViaScheduler = @import("submit.zig").nonStreamingViaScheduler;
const respond = @import("../http/respond.zig");
const sendResponse = respond.sendResponse;
const gen = @import("../media_jobs/gen.zig");
const sendEnsureLoadedError = @import("../http/errors.zig").sendEnsureLoadedError;
const countRejected = @import("../http/errors.zig").countRejected;
const contextOverflowMessage = @import("../http/errors.zig").contextOverflowMessage;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const MediaItem = @import("../media/placeholders.zig").MediaItem;
const prepareMultimodalItems = @import("../media/encode.zig").prepareMultimodalItems;
const multimodalFailure = @import("../media/encode.zig").multimodalFailure;

/// Longest piece one pass of the ASR transcribes: mlx-audio's chunk length,
/// past which the reference splits the audio.
const TRANSCRIPTION_MAX_SECONDS = 1200;

/// The reference's `max_tokens` for one chunk.
const TRANSCRIPTION_MAX_TOKENS = 8192;

/// The aligner the app downloads beside a Qwen3-ASR model.
const ALIGNER_REPO = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit";

/// The diarizer the app downloads beside it.
const DIARIZER_REPO = "mlx-community/Nemotron-3-Diarization-8bit";

/// `POST /v1/audio/transcriptions`: OpenAI's form over Qwen3-ASR. The reference
/// prompt's one audio placeholder expands to the tower's rows like any chat
/// media, then the trunk decodes the transcript. verbose_json, srt, vtt and
/// diarized_json time it with Qwen3-ForcedAligner when one is on disk, cutting
/// the clip into pieces the aligner can time as the official toolkit does; a
/// piece in a language the aligner lacks is one segment. diarized_json labels
/// each word with the speaker Nemotron 3 Diarization hears over the whole clip.
pub fn handleTranscriptions(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, content_type: []const u8, lm: *LoadedModel) !void {
    const hears = mlx_engine.facts(lm).hears;
    if (!hears) {
        log.warn("POST /v1/audio/transcriptions -> 400 (no audio tower)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "This model does not transcribe audio: target a Qwen3-ASR model (and do not run it with --no-vision, which drops its audio tower)", 400);
        return;
    }
    const form = transcription_mod.parseForm(body, content_type) catch |err| {
        log.warn("POST /v1/audio/transcriptions -> 400 ({s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", transcription_mod.formErrorMessage(err), 400);
        return;
    };
    // The card's recipe names the language and nothing else: a context prompt has no place in it.
    if (form.prompt != null and lm.config.?.speech() != null) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "this model's transcription takes no 'prompt': send the language alone", 400);
        return;
    }
    const pcm = transcription_mod.decodeWav(allocator, form.file) catch |err| {
        log.warn("POST /v1/audio/transcriptions -> 400 (audio: {s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "`file` must be a WAV file (16- or 24-bit PCM, or 32-bit float)", 400);
        return;
    };
    defer allocator.free(pcm);
    const seconds = @as(f64, @floatFromInt(pcm.len)) / transcription_mod.SAMPLE_RATE;
    log.info("POST /v1/audio/transcriptions ({d:.1} s, language={s}, format={s}, prompt={d}b)\n", .{ seconds, form.language orelse "auto", @tagName(form.format), if (form.prompt) |p| p.len else 0 });

    var aligner: ?Aligner = if (transcription_mod.timed(form.format)) try findAligner(allocator) else null;
    if (aligner == null and (form.format == .srt or form.format == .vtt)) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "srt and vtt need word times from Qwen3-ForcedAligner, which is not on disk or did not load: download " ++ ALIGNER_REPO ++ " beside the ASR model", 400);
        return;
    }
    // Speakers are heard over the whole clip, so a voice keeps its letter across pieces.
    var speaker_probs: ?[]f32 = null;
    defer if (speaker_probs) |p| allocator.free(p);
    var diarizer: ?*LoadedModel = null;
    defer if (diarizer) |m| http_server.launch().global_scheduler.?.release(m);
    var speakers: u32 = 0;
    if (form.format == .diarized_json) {
        const id = (findCompanion("nemotron_diarization") orelse {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "diarized_json needs Nemotron 3 Diarization, which is not on disk or did not load: download " ++ DIARIZER_REPO ++ " beside the ASR model", 400);
            return;
        }).id;
        diarizer = http_server.launch().global_scheduler.?.ensureLoaded(id) catch |err| blk: {
            log.warn("  diarizer {s} did not load ({s}): one speaker\n", .{ id, @errorName(err) });
            break :blk null;
        };
        if (diarizer) |m| speaker_probs = (try diarizeClip(allocator, stream, m, pcm, &speakers)) orelse return;
    }
    const who: ?transcription_mod.Activity = if (speaker_probs) |p| .{ .probs = p, .speakers = speakers } else null;

    const piece_seconds: usize = if (aligner != null) transcription_mod.ALIGN_MAX_SECONDS else TRANSCRIPTION_MAX_SECONDS;
    const ends = try transcription_mod.splitPoints(allocator, pcm, piece_seconds * transcription_mod.SAMPLE_RATE);
    defer allocator.free(ends);

    var tl = transcription_mod.Timeline{};
    defer tl.deinit(allocator);
    defer if (aligner) |al| if (al.lm) |m| http_server.launch().global_scheduler.?.release(m);
    var start: usize = 0;
    for (ends) |end| {
        const heard = (try transcribeClip(allocator, stream, lm, form, pcm[start..end])) orelse return;
        defer allocator.free(heard.raw);
        const t = heard.t;
        var timed: ?Aligned = null;
        defer if (timed) |*x| x.deinit(allocator);
        if (aligner) |*al| {
            if (t.text.len > 0 and al.languages & transcription_mod.languageBit(t.language orelse "") != 0) {
                if (al.lm == null) al.lm = http_server.launch().global_scheduler.?.ensureLoaded(al.id) catch |err| blk: {
                    // verbose_json still answers without times; srt and vtt cannot.
                    if (form.format != .verbose_json) {
                        try sendEnsureLoadedError(allocator, stream, http_server.launch().global_scheduler.?, al.id, err);
                        return;
                    }
                    log.warn("  aligner {s} did not load ({s}): segments without word times\n", .{ al.id, @errorName(err) });
                    al.languages = 0;
                    break :blk null;
                };
                if (al.lm) |alm| timed = (try alignPiece(allocator, stream, alm, pcm[start..end], t.text, t.language.?, who, start * 1000 / transcription_mod.SAMPLE_RATE)) orelse return;
            }
        }
        const rate: f64 = transcription_mod.SAMPLE_RATE;
        const speaker = if (who) |w| w.speakerOver(start * 1000 / transcription_mod.SAMPLE_RATE, end * 1000 / transcription_mod.SAMPLE_RATE, 0) else 0;
        try tl.add(allocator, t.text, t.language, @as(f64, @floatFromInt(start)) / rate, @as(f64, @floatFromInt(end - start)) / rate, if (timed) |x| x.words else null, if (timed) |x| x.segments else null, speaker);
        start = end;
    }

    switch (form.format) {
        .text => try sendResponse(stream, "200 OK", "text/plain; charset=utf-8", tl.text.items),
        .json => {
            const text = jsonEscapeOrEmpty(allocator, tl.text.items);
            defer if (text.owned) allocator.free(text.slice);
            const response = try std.fmt.allocPrint(allocator, "{{\"text\":{s}}}", .{text.slice});
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
        },
        .verbose_json => {
            const response = try tl.verboseJson(allocator, seconds, form.words);
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
        },
        .srt, .vtt => {
            const response = try tl.subtitles(allocator, form.format);
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "text/plain; charset=utf-8", response);
        },
        .diarized_json => {
            const response = try tl.diarizedJson(allocator, seconds);
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
        },
    }
}

/// The transcript of one clip and the bytes it points into.
const Heard = struct { raw: []u8, t: transcription_mod.Transcript };

/// One pass of the ASR over `pcm`; null when it already sent the error.
fn transcribeClip(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, form: transcription_mod.Form, pcm: []const f32) !?Heard {
    const config = lm.config.?;
    const tok = lm.tokenizer.?;
    const audio = chat_mod.AudioData{ .samples = std.mem.sliceAsBytes(pcm) };
    var prompt_ids = if (config.speech() != null) try speechModelPrompt(allocator, stream, lm, form, &audio) else blk: {
        const prompt_text = try transcription_mod.buildPrompt(allocator, form.prompt, form.language);
        defer allocator.free(prompt_text);
        break :blk try encodeText(allocator, lm, tok, prompt_text, false);
    };
    var mm = prepareMultimodalItems(allocator, lm, config, &[_]MediaItem{.{ .audio = &audio }}, &prompt_ids) catch |err| {
        allocator.free(prompt_ids);
        const f = multimodalFailure(err);
        log.warn("POST /v1/audio/transcriptions -> {d} ({s})\n", .{ f.code, @errorName(err) });
        try sendErrorResponse(allocator, stream, f.status, f.kind, f.message, f.code);
        return null;
    };
    defer mm.deinit(allocator);
    defer allocator.free(prompt_ids);

    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return null;
    }
    const max_tokens = clampMaxTokens(TRANSCRIPTION_MAX_TOKENS, prompt_ids.len, effective_ctx);
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, max_tokens, config, false, null, lm, false, mm.bytes(config))) return null;

    // Ownership of the soft tokens moves to the slot. The reference decodes plainly: no thinking, no drafts.
    const slot_ve = mm.embeddings;
    mm.embeddings = null;
    const enable_thinking = false;
    const spec = false;
    var timer = Stopwatch.init(stream.io);
    const result = nonStreamingViaScheduler(allocator, http_server.launch().global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, .{ .temperature = form.temperature }, config.eosTokenSlice(), 0, false, enable_thinking, spec, spec, spec, spec, getTimeoutNs(), slot_ve, mm.media, 0, .{}, 0, false, null, null, null, stream) catch |err| {
        log.err("  -> {s}\n", .{@errorName(err)});
        sendGenerationError(allocator, stream, err, .openai) catch {};
        return null;
    };
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    const raw = try decodeTokens(allocator, lm, tok, result.token_ids, false);
    const t = transcription_mod.splitOutput(raw, form.language);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] language={s}\n", .{ result.prompt_tokens, result.completion_tokens, timer.read() / std.time.ns_per_ms, result.finish_reason, t.language orelse "?" });
    return .{ .raw = raw, .t = t };
}

/// The most speech a `/v1/audio/speech` request may run to, in frames (164 s at 12.5 frames a second).
const SPEECH_MAX_FRAMES = 2048;

/// `/v1/audio/speech` on a model that speaks (`Parsed.speech`): the card's TTS recipe (`Perform TTS in <language>
/// using a <gender> voice.`, the turn prefilled with `<|audio_start|>`) over `input`, as WAV or raw 16-bit PCM.
pub fn handleSpeech(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *LoadedModel) !void {
    const config = lm.config.?;
    const f = config.speech().?;
    const tok = lm.tokenizer.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return refuse(allocator, stream, "Invalid JSON in request body");
    defer parsed.deinit();
    if (parsed.value != .object) return refuse(allocator, stream, "Request body must be a JSON object");
    const root = parsed.value.object;
    const input = if (root.get("input")) |v| if (v == .string and v.string.len > 0) v.string else null else null;
    if (input == null) return refuse(allocator, stream, "'input' is a required non-empty string");
    const voice_id = if (root.get("voice")) |v| if (v == .string) v.string else "" else speech_api.DEFAULT_VOICE;
    const v = speech_api.voice(voice_id) orelse return refuse(allocator, stream, "'voice' must be one of this model's voices (`<language>-<female|male>`, listed on /v1/models)");
    const format: speech_api.Format = if (root.get("response_format")) |rf| blk: {
        if (rf == .string and std.mem.eql(u8, rf.string, "wav")) break :blk .wav;
        if (rf == .string and std.mem.eql(u8, rf.string, "pcm")) break :blk .pcm16;
        return refuse(allocator, stream, "'response_format' must be \"wav\" or \"pcm\"");
    } else .wav;
    if (root.get("speed")) |sp| if (!(sp == .integer and sp.integer == 1) and !(sp == .float and sp.float == 1.0))
        return refuse(allocator, stream, "this model speaks at one speed: 'speed' must be 1");
    if (root.get("instructions") != null) return refuse(allocator, stream, "this model takes no 'instructions': its voice is the 'voice' field");
    // `stream: true` answers like every TTS backend: one SSE `complete` event carrying the WAV (`gen.sendWavWith`).
    const want_stream = if (root.get("stream")) |sv| sv == .bool and sv.bool else false;
    if (want_stream and format != .wav) return refuse(allocator, stream, "a streamed answer is WAV: 'response_format' must be \"wav\"");

    const system = try std.fmt.allocPrint(allocator, "Perform TTS in {s} using a {s} voice.", .{ v.language, v.gender });
    defer allocator.free(system);
    const msgs = [_]chat_mod.Message{ .{ .role = "system", .content = system }, .{ .role = "user", .content = input.? } };
    var prompt_ids = try cachedFormatChat(allocator, stream.io, lm, tok, lm.chat_config.?, &msgs, null, null, false, null, false);
    prompt_ids = try allocator.realloc(prompt_ids, prompt_ids.len + 1);
    prompt_ids[prompt_ids.len - 1] = f.audio_start;
    defer allocator.free(prompt_ids);
    log.info("POST /v1/audio/speech ({d}b, voice={s}, format={s})\n", .{ input.?.len, voice_id, @tagName(format) });

    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
    }
    const max_tokens = clampMaxTokens(SPEECH_MAX_FRAMES, prompt_ids.len, effective_ctx);
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, max_tokens, config, false, null, lm, false, 0)) return;
    var timer = Stopwatch.init(stream.io);
    const result = nonStreamingViaScheduler(allocator, http_server.launch().global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, .{ .temperature = 0 }, config.eosTokenSlice(), 0, false, false, false, false, false, false, getTimeoutNs(), null, &.{}, 0, .{}, 0, false, null, null, .{}, stream) catch |err| {
        log.err("  -> {s}\n", .{@errorName(err)});
        sendGenerationError(allocator, stream, err, .openai) catch {};
        return;
    };
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    defer if (result.audio_samples) |x| allocator.free(x);
    defer if (result.audio_codes) |x| allocator.free(x);
    const samples = result.audio_samples orelse &.{};
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] {d:.2} s of speech\n", .{ result.prompt_tokens, result.completion_tokens, timer.read() / std.time.ns_per_ms, result.finish_reason, @as(f64, @floatFromInt(samples.len)) / @as(f64, @floatFromInt(f.sample_rate)) });
    const bytes = switch (format) {
        .wav => try wav_mod.encodePcm16Mono(allocator, samples, f.sample_rate),
        .pcm16 => try speech_api.pcm16(allocator, samples),
    };
    defer allocator.free(bytes);
    if (want_stream) {
        try respond.sendSseHeaders(stream, "audio.speech", respond.SSE_ALLOW_HEADERS_DEFAULT);
        return gen.sendWavWith(allocator, stream, bytes, true, "");
    }
    try sendResponse(stream, "200 OK", if (format == .wav) "audio/wav" else "audio/pcm", bytes);
}

fn refuse(allocator: std.mem.Allocator, stream: *Conn, why: []const u8) !void {
    log.warn("POST /v1/audio/speech -> 400 ({s})\n", .{why});
    try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, 400);
}

/// LFM3-Audio's card recipe through its own template: `Perform ASR in <Language>.`, or `Perform ASR.` to let the model
/// name it, over the clip.
fn speechModelPrompt(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, form: transcription_mod.Form, audio: *const chat_mod.AudioData) ![]u32 {
    const language = if (form.language) |l| transcription_mod.languageName(l) orelse l else null;
    const system = if (language) |l| try std.fmt.allocPrint(allocator, "Perform ASR in {s}.", .{l}) else try allocator.dupe(u8, "Perform ASR.");
    defer allocator.free(system);
    const msgs = [_]chat_mod.Message{
        .{ .role = "system", .content = system },
        .{ .role = "user", .content = "", .audio = (audio)[0..1] },
    };
    return cachedFormatChat(allocator, stream.io, lm, lm.tokenizer.?, lm.chat_config.?, &msgs, null, null, false, null, false);
}

/// The forced aligner a timed transcription uses: its registry id, the
/// languages its config lists, and the model once loaded (refcounted).
const Aligner = struct { id: []const u8, languages: u32, lm: ?*LoadedModel = null };

/// The model of one `arch_hint` a transcription leans on (the aligner, the
/// diarizer): the resident one first, one that failed to load never; null
/// when none is on disk.
fn findCompanion(arch_hint: []const u8) ?*LoadedModel {
    const registry = http_server.global_registry orelse return null;
    registry.mutex.lockUncancelable(registry.io);
    defer registry.mutex.unlock(registry.io);
    var pick: ?*LoadedModel = null;
    var it = registry.entries.valueIterator();
    while (it.next()) |e| {
        if (!std.mem.eql(u8, e.*.arch_hint, arch_hint) or e.*.state == .error_state) continue;
        const p = pick orelse {
            pick = e.*;
            continue;
        };
        const ready = e.*.state == .ready;
        if (ready != (p.state == .ready)) {
            if (ready) pick = e.*;
        } else if (std.mem.lessThan(u8, e.*.id, p.id)) pick = e.*;
    }
    return pick;
}

fn findAligner(allocator: std.mem.Allocator) !?Aligner {
    const entry = findCompanion("qwen3_forced_aligner") orelse return null;
    var config = model_mod.parseConfig(http_server.global_registry.?.io, allocator, entry.path) catch return null;
    defer config.deinit(allocator);
    const fa = config.forced_aligner orelse return null;
    return .{ .id = entry.id, .languages = fa.languages };
}

/// A piece's words and segments, spans of its transcript.
const Aligned = struct {
    words: []const forced_align_mod.Timed,
    segments: []const forced_align_mod.Segment,
    fn deinit(self: *Aligned, allocator: std.mem.Allocator) void {
        allocator.free(self.words);
        allocator.free(self.segments);
    }
};

/// Time the words of `text` (the transcript of `pcm`, `offset_ms` into the
/// clip) with the aligner `lm`, a segment closing where `who` hears the
/// speaker change. Null when it already sent the error.
fn alignPiece(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, pcm: []const f32, text: []const u8, language: []const u8, who: ?transcription_mod.Activity, offset_ms: u64) !?Aligned {
    const config = lm.config.?;
    const fa = config.forced_aligner.?;
    const script = forced_align_mod.scriptFor(language);
    const units = try forced_align_mod.units(allocator, text, script);
    defer forced_align_mod.freeUnits(allocator, units);
    if (units.len == 0) return .{ .words = &.{}, .segments = &.{} };
    const prompt_text = try forced_align_mod.promptText(allocator, units);
    defer allocator.free(prompt_text);
    var ids = try encodeText(allocator, lm, lm.tokenizer.?, prompt_text, false);
    const audio = chat_mod.AudioData{ .samples = std.mem.sliceAsBytes(pcm) };
    var mm = prepareMultimodalItems(allocator, lm, config, &[_]MediaItem{.{ .audio = &audio }}, &ids) catch |err| {
        allocator.free(ids);
        const f = multimodalFailure(err);
        log.warn("  aligner -> {d} ({s})\n", .{ f.code, @errorName(err) });
        try sendErrorResponse(allocator, stream, f.status, f.kind, f.message, f.code);
        return null;
    };
    defer mm.deinit(allocator);
    defer allocator.free(ids);

    // The job frees the audio rows on the inference thread.
    var job = AlignJob{ .allocator = allocator, .lm = lm, .ids = ids, .rows = mm.embeddings.?, .timestamp_token = fa.timestamp_token_id };
    mm.embeddings = null;
    var timer = Stopwatch.init(stream.io);
    var req = scheduler_mod.GenRequest{ .ctx = &job, .run = alignJobRun, .model = lm };
    http_server.launch().global_scheduler.?.runGeneration(&req) catch |err| {
        _ = mlx.mlx_array_free(job.rows);
        if (err != error.Shutdown) return err;
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
        return null;
    };
    const classes = job.classes orelse {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "word alignment failed", 500);
        return null;
    };
    defer allocator.free(classes);
    const ms = try forced_align_mod.fixTimestamps(allocator, classes, fa.step_ms);
    defer allocator.free(ms);
    if (ms.len != 2 * units.len) {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "word alignment returned a time for each <timestamp> it read, not two per word", 500);
        return null;
    }
    const words = try forced_align_mod.timedWords(allocator, units, ms, script);
    errdefer allocator.free(words);
    const said_by = if (who) |w| try w.wordSpeakers(allocator, words, offset_ms) else null;
    defer if (said_by) |x| allocator.free(x);
    const segments = try forced_align_mod.segments(allocator, text, words, said_by);
    log.info("  aligned {d} words into {d} segments ({d} tokens, {d}ms)\n", .{ words.len, segments.len, ids.len, timer.read() / std.time.ns_per_ms });
    return .{ .words = words, .segments = segments };
}

/// One aligner pass, run on the inference thread (the sole mlx caller).
const AlignJob = struct {
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    ids: []const u32,
    rows: mlx.mlx_array,
    timestamp_token: u32,
    classes: ?[]u32 = null,
};

fn alignJobRun(ctx: *anyopaque) void {
    const job: *AlignJob = @ptrCast(@alignCast(ctx));
    defer _ = mlx.mlx_array_free(job.rows);
    job.classes = mlx_engine.timestampClasses(job.allocator, job.lm, job.ids, job.rows, job.timestamp_token) catch |err| {
        log.err("  alignment failed: {s}\n", .{@errorName(err)});
        return;
    };
}

/// Speaker activity over `pcm` from the diarizer `lm`, on the inference thread.
/// Null when it already sent the error.
fn diarizeClip(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, pcm: []const f32, speakers: *u32) !?[]f32 {
    var job = DiarizeJob{ .allocator = allocator, .plugin = lm.media_plugin, .pcm = pcm };
    var timer = Stopwatch.init(stream.io);
    var req = scheduler_mod.GenRequest{ .ctx = &job, .run = diarizeJobRun, .model = lm };
    http_server.launch().global_scheduler.?.runGeneration(&req) catch |err| {
        if (err != error.Shutdown) return err;
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
        return null;
    };
    const probs = job.probs orelse {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "speaker diarization failed", 500);
        return null;
    };
    speakers.* = job.speakers;
    log.info("  diarized {d:.1} s ({d}ms)\n", .{ @as(f64, @floatFromInt(pcm.len)) / transcription_mod.SAMPLE_RATE, timer.read() / std.time.ns_per_ms });
    return probs;
}

const DiarizeJob = struct {
    allocator: std.mem.Allocator,
    /// The diarizer's plugin (`media_plugin`).
    plugin: ?*PluginEngine,
    pcm: []const f32,
    probs: ?[]f32 = null,
    speakers: u32 = 0,
};

fn diarizeJobRun(ctx: *anyopaque) void {
    const job: *DiarizeJob = @ptrCast(@alignCast(ctx));
    const p = job.plugin orelse return;
    const r = plugin_http.diarize(p, job.allocator, job.pcm) orelse return;
    job.probs = r.probs;
    job.speakers = r.speakers;
}
