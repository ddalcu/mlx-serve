//! modelEngineName, mlxGgufClaims; ReadyCaps, readyCapsJson; loadedSupportsVideo, renderModelEntry, handleModels;
//! renderModelsBody, load progress, rescan, handleLoadModelStrict; unload

const speech_api = @import("speech.zig");
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const chat_mod = @import("../chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const radio_vision = @import("../../plugins/nemotron_h/radio_vision.zig");
const vision_mod = @import("../../abi/mlx_handles.zig");
const mlx_engine = @import("../engines/mlx.zig");
const log = @import("../../sdk/util/log.zig");
const model_registry_mod = @import("../registry/registry.zig");
const model_discovery = @import("../registry/discovery.zig");
const mlx_gguf = @import("../../sdk/quant/gguf/claim.zig");
const EngineKind = @import("../engines/served.zig").Kind;
const media_mod = @import("../media_jobs/gen.zig");
const metrics = @import("../status.zig");
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const PluginEngine = @import("../media_jobs/plugin/engine.zig").Engine;
const io_util = @import("../../sdk/util/io_util.zig");
const lan_mod = @import("../lan/lan.zig");
const nowSecs = io_util.nowSecs;
const testing = std.testing;
const http_server = @import("../http/server.zig");
const request_model = @import("request_model.zig");
const row_facts = @import("model_row_facts.zig");

const Conn = @import("../http/conn.zig").Conn;
const configuredKvQuantFor = @import("../http/server.zig").configuredKvQuantFor;
const parseModelFromBody = @import("request_model.zig").parseModelFromBody;
const resolveRequestModelId = @import("request_model.zig").resolveRequestModelId;
const embedEffectiveLimit = @import("embeddings.zig").embedEffectiveLimit;
const quantizationJson = @import("ollama_routes.zig").quantizationJson;
const getEffectiveContextLength = @import("admission.zig").getEffectiveContextLength;
const optSamplingRecJson = @import("sampling.zig").optSamplingRecJson;
const batchVerdictFor = @import("props.zig").batchVerdictFor;
const sendResponse = @import("../http/respond.zig").sendResponse;
const sendResponseFramed = @import("../http/respond.zig").sendResponseFramed;
const logHttpResponse = @import("../http/respond.zig").logHttpResponse;
const sendEnsureLoadedError = @import("../http/errors.zig").sendEnsureLoadedError;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const jsonEscape = @import("format.zig").jsonEscape;

/// Which backend serves this entry — surfaced as `meta.engine` in /v1/models
/// so the app's engine-aware Settings UI never has to INFER it from
/// `architecture`: a NATIVE deepseek_v4 safetensors dir and a DeepSeek GGUF
/// on the embedded ds4 engine report the SAME model_type. "gguf" = an
/// unloaded GGUF stub whose engine (llama vs ds4) is only known once the
/// header is read at load time. A loaded model's `config` names the MLX path's own engine: a Sushi
/// pack's EXL3 experts.
fn modelEngineName(engine: ?EngineKind, mlx_gguf_served: bool, config: ?*const model_mod.ModelConfig, path: []const u8, arch_hint: []const u8) []const u8 {
    if (engine) |k| return @tagName(k);
    if (config) |c| {
        if (c.exl3 != null) return "sushi";
    }
    if (mlx_gguf_served) return "mlx-gguf";
    if (std.mem.endsWith(u8, path, ".gguf") or std.mem.eql(u8, arch_hint, "gguf")) return "gguf";
    return "mlx";
}

/// Would src/sdk/quant/gguf claim this unloaded entry? Answered once per
/// entry: the header read behind `servablePath` is too slow for a poll.
fn mlxGgufClaims(io: std.Io, entry: *LoadedModel) bool {
    if (entry.mlx_gguf_claim == null) {
        const allocator = std.heap.page_allocator;
        const p = mlx_gguf.servablePath(io, allocator, entry.path);
        defer if (p) |v| allocator.free(v);
        entry.mlx_gguf_claim = p != null;
    }
    return entry.mlx_gguf_claim.?;
}

/// Render the JSON metadata fragment for one entry. For `.ready` entries
/// Capability flags for a READY registry entry, kept as plain booleans so the
/// JSON assembly below is hermetically testable without a LoadedModel. Every
/// engine slot on LoadedModel must have a flag here — a missing arm renders a
/// ready model with an empty capabilities list (the live `.mesh`/"3d" hole).
const ReadyCaps = struct {
    has_chat: bool = false,
    has_vision: bool = false,
    has_audio: bool = false,
    /// `/v1/audio/transcriptions` (an audio tower on a text trunk: Qwen3-ASR).
    has_transcription: bool = false,
    /// Qwen3-ForcedAligner, which times those transcripts.
    has_alignment: bool = false,
    /// Nemotron 3 Diarization, which labels their speakers.
    has_diarization: bool = false,
    has_reasoning: bool = false,
    /// Encoder-only (BERT/EmbeddingGemma) OR a decoder with a pooling contract
    /// (Qwen3-Embedding) — `config.hasEmbeddingCapability()`. Issue #116.
    has_embedding: bool = false,
    has_image_engine: bool = false,
    has_audio_engine: bool = false,
    /// Audio engine's backend is the ACE-Step music generator (advertises
    /// "music" ADDITIVELY beside "audio", the ready-model "3d" precedent).
    has_music_backend: bool = false,
    /// Stable Audio 3 text-to-audio: "sound" beside "audio", the same rule.
    has_sound_backend: bool = false,
    has_video_engine: bool = false,
    has_mesh_engine: bool = false,
    has_decision_engine: bool = false,
    has_speech: bool = false,
    /// It speaks its reply to a request with tools (`SpeechFacts.tools`).
    has_speech_tools: bool = false,
};

/// Chat capability for a READY entry. Template presence is NOT the gate for
/// embedded-engine (ds4/llama) models: a GGUF without a chat_template in its
/// header still serves chat via fallback formatting, and gating on the
/// template made a loaded DSV4-Flash advertise capabilities:[] — LAN clients
/// hid the peer's model as "no chat models" while chatting on it (live
/// ). The unloaded GGUF stub path already advertises the chat set
/// unconditionally; loaded must never advertise less than its stub.
fn readyHasChat(is_encoder_only: bool, chat_template_len: usize, has_embedded_lm: bool) bool {
    if (is_encoder_only) return false;
    return chat_template_len > 0 or has_embedded_lm;
}

/// `capabilities` JSON array for a ready model. Caller deinits.
fn readyCapsJson(allocator: std.mem.Allocator, c: ReadyCaps) !std.ArrayList(u8) {
    var caps = std.ArrayList(u8).empty;
    errdefer caps.deinit(allocator);
    try caps.append(allocator, '[');
    var n_caps: usize = 0;
    const append_cap = struct {
        fn call(a: std.mem.Allocator, b: *std.ArrayList(u8), n: *usize, name: []const u8) !void {
            if (n.* > 0) try b.append(a, ',');
            try b.append(a, '"');
            try b.appendSlice(a, name);
            try b.append(a, '"');
            n.* += 1;
        }
    }.call;
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "chat");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "tool_use");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "streaming");
    if (c.has_vision) try append_cap(allocator, &caps, &n_caps, "vision");
    if (c.has_audio) try append_cap(allocator, &caps, &n_caps, "audio");
    if (c.has_transcription) try append_cap(allocator, &caps, &n_caps, "transcription");
    if (c.has_speech) try append_cap(allocator, &caps, &n_caps, "speech");
    if (c.has_speech_tools) try append_cap(allocator, &caps, &n_caps, "speech_tools");
    if (c.has_alignment) try append_cap(allocator, &caps, &n_caps, "alignment");
    if (c.has_diarization) try append_cap(allocator, &caps, &n_caps, "diarization");
    if (c.has_reasoning) try append_cap(allocator, &caps, &n_caps, "reasoning");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "json_schema");
    if (c.has_embedding) try append_cap(allocator, &caps, &n_caps, "embeddings");
    // Native media-generation engines (resident).
    if (c.has_image_engine) try append_cap(allocator, &caps, &n_caps, "image");
    if (c.has_audio_engine and !c.has_audio) try append_cap(allocator, &caps, &n_caps, "audio");
    if (c.has_music_backend) try append_cap(allocator, &caps, &n_caps, "music");
    if (c.has_sound_backend) try append_cap(allocator, &caps, &n_caps, "sound");
    if (c.has_video_engine) try append_cap(allocator, &caps, &n_caps, "video");
    if (c.has_mesh_engine) try append_cap(allocator, &caps, &n_caps, "3d");
    if (c.has_decision_engine) try append_cap(allocator, &caps, &n_caps, "decisions");
    try caps.append(allocator, ']');
    return caps;
}

pub fn loadedSupportsVideo(entry: *const LoadedModel) bool {
    return mlx_engine.api().supports_video(entry);
}

/// pulls full capabilities/dimensions off the resident config/chat_config;
/// for non-ready entries renders a lightweight stub with state +
/// bytes_on_disk only. Returns an allocator-owned string; caller frees.
/// Called from `handleModels` per registry entry. Registry mutex must be
/// held by the caller (entry fields are read directly).
pub fn renderModelEntry(
    allocator: std.mem.Allocator,
    io: std.Io,
    entry: *LoadedModel,
) ![]u8 {
    var quant_buf: [16]u8 = undefined;
    if (entry.state == .ready and entry.config != null and entry.chat_config != null) {
        const config = entry.config.?;
        const chat_config = entry.chat_config.?;
        const ctx_len = getEffectiveContextLength(config);
        const ctx_str = if (ctx_len > 0)
            try std.fmt.allocPrint(allocator, "{d}", .{ctx_len})
        else
            try std.fmt.allocPrint(allocator, "null", .{});
        defer allocator.free(ctx_str);

        const has_chat = readyHasChat(
            config.is_encoder_only,
            chat_config.chat_template.len,
            entry.engineBacked(),
        );
        const mf = mlx_engine.facts(entry);
        const has_vision = mf.images;
        const has_audio = mf.audio;
        var caps = try readyCapsJson(allocator, .{
            .has_chat = has_chat,
            .has_vision = has_vision,
            .has_audio = has_audio,
            .has_transcription = config.forced_aligner == null and mf.hears,
            .has_alignment = config.forced_aligner != null,
            .has_reasoning = has_chat and model_discovery.templateSupportsThinking(chat_config.chat_template),
            .has_embedding = config.hasEmbeddingCapability(),
            .has_image_engine = PluginEngine.serves(entry.media_plugin, media_mod.Modality.image),
            .has_audio_engine = PluginEngine.serves(entry.media_plugin, media_mod.Modality.audio),
            .has_music_backend = PluginEngine.servesRoute(entry.media_plugin, "music"),
            .has_sound_backend = PluginEngine.servesRoute(entry.media_plugin, "sound"),
            .has_video_engine = PluginEngine.serves(entry.media_plugin, media_mod.Modality.video),
            .has_mesh_engine = PluginEngine.serves(entry.media_plugin, media_mod.Modality.mesh),
            .has_decision_engine = PluginEngine.serves(entry.media_plugin, media_mod.Modality.decision),
            .has_diarization = PluginEngine.serves(entry.media_plugin, media_mod.Modality.diarization),
            .has_speech = config.speech() != null,
            .has_speech_tools = if (config.speech()) |f| f.tools else false,
        });
        defer caps.deinit(allocator);
        // A model that speaks answers in audio too (`modalities` on chat, `/v1/audio/speech` in its voices).
        const speech_part = if (config.speech() != null) blk: {
            const voices = try speech_api.voicesJson(allocator);
            defer allocator.free(voices);
            break :blk try std.fmt.allocPrint(allocator, ",\"output_modalities\":[\"text\",\"audio\"],\"voices\":{s}", .{voices});
        } else try allocator.dupe(u8, "");
        defer allocator.free(speech_part);

        const mods = try inputModalities(allocator, has_vision, mf.video, has_audio);
        defer allocator.free(mods);

        // Loaded and unloaded rows report the same `architecture` — the
        // registry's arch hint (the pack's real model_type), never the stub
        // config's modality marker ("flux2" for every image backend).
        const arch_label: []const u8 = if (entry.arch_hint.len > 0) entry.arch_hint else config.model_type;
        const facts = try row_facts.forEntry(allocator, io, entry, arch_label, if (has_chat) ctx_len else null, config.quant_mode == .gguf);
        defer allocator.free(facts);
        const model_id_json = try jsonEscape(allocator, if (entry.id.len > 0) entry.id else config.model_type);
        defer allocator.free(model_id_json);
        const drafter_loaded = entry.hasDrafter();
        const mtp_loaded = entry.mtp != null or if (entry.engine) |e| e.kind == .llama and e.caps().drafts else false;
        const drafter_path_json = if (drafter_loaded)
            try jsonEscape(allocator, entry.drafter_path)
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(drafter_path_json);
        const bytes_on_disk_str = if (entry.bytes_on_disk) |b|
            try std.fmt.allocPrint(allocator, "{d}", .{b})
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(bytes_on_disk_str);

        // Model-author sampling recommendations from generation_config.json,
        // surfaced so the Swift Settings UI can show "model recommends" pills
        // next to the per-request sampling sliders. `null` when the model
        // ships no value for that field.
        const gen_temp_str = try optSamplingRecJson(allocator, f32, config.gen_temperature);
        defer allocator.free(gen_temp_str);
        const gen_top_p_str = try optSamplingRecJson(allocator, f32, config.gen_top_p);
        defer allocator.free(gen_top_p_str);
        const gen_top_k_str = try optSamplingRecJson(allocator, u32, config.gen_top_k);
        defer allocator.free(gen_top_k_str);

        // Effective embedding input ceiling (issue #117): surfaced so clients
        // can discover the limit instead of learning it from a 400. null for
        // models with no embedding capability; 0 = embedding-capable, unbounded.
        const embed_limit_str = if (config.hasEmbeddingCapability())
            try std.fmt.allocPrint(allocator, "{d}", .{embedEffectiveLimit(http_server.embedding_max_length, config.max_position_embeddings)})
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(embed_limit_str);

        return std.fmt.allocPrint(allocator,
            \\{{"id":{s},"object":"model","created":{d},"owned_by":"mlx-serve","loaded":true,"state":"ready","bytes_resident":{d},"bytes_on_disk":{s},"context_length":{s},"max_model_len":{s},"batched_decode":{s},"capabilities":{s},"input_modalities":{s}{s}{s},"meta":{{"architecture":"{s}","engine":"{s}","vocab_size":{d},"hidden_size":{d},"num_layers":{d},"quantization":{s},"context_length":{s},"model_max_tokens":{d},"embedding_max_length":{s},"is_moe":{s},"drafter_loaded":{s},"drafter_path":{s},"mtp_loaded":{s},"mtp_available":{s},"spec_exact":{s},"kv_quant":"{s}","gen_temperature":{s},"gen_top_p":{s},"gen_top_k":{s}}}}}
        , .{
            model_id_json,
            nowSecs(io),
            entry.bytes_resident,
            bytes_on_disk_str,
            // Top-level twins of meta.context_length: openai-models-list
            // discovery clients (oh-my-pi, vLLM-shaped readers) look for
            // row-level max_model_len/context_length and substitute their own
            // defaults when absent (issue #188).
            ctx_str,
            ctx_str,
            if (batchVerdictFor(entry) == .ok) "true" else "false",
            caps.items,
            mods,
            speech_part,
            facts,
            arch_label,
            modelEngineName(if (entry.engine) |e| e.kind else null, entry.config != null and entry.config.?.quant_mode == .gguf, entry.config, entry.path, entry.arch_hint),
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            quantizationJson(&quant_buf, config.quant_bits),
            ctx_str,
            config.max_position_embeddings,
            embed_limit_str,
            if (config.isMoe()) "true" else "false",
            if (drafter_loaded) "true" else "false",
            drafter_path_json,
            if (mtp_loaded) "true" else "false",
            if (mtp_loaded or model_discovery.readStubMeta(io, allocator, entry.path).has_mtp) "true" else "false",
            if (config.rowExactDecode()) "true" else "false",
            configuredKvQuantFor(config).wireName(),
            gen_temp_str,
            gen_top_p_str,
            gen_top_k_str,
        });
    }

    // Non-ready entry: stub form with state + bytes_on_disk.
    const state_str = switch (entry.state) {
        .unloaded => "unloaded",
        .loading => "loading",
        .ready => "ready",
        .error_state => "error",
        .evicting => "evicting",
    };
    const bytes_on_disk_str = if (entry.bytes_on_disk) |b|
        try std.fmt.allocPrint(allocator, "{d}", .{b})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(bytes_on_disk_str);
    const err_part: []const u8 = if (entry.error_name) |name| blk: {
        // Inline escape to avoid double allocation; the names we emit
        // never contain quotes/backslashes (they're @errorName output).
        break :blk try std.fmt.allocPrint(
            allocator,
            ",\"error\":\"{s}\"",
            .{name},
        );
    } else &[_]u8{};
    defer if (err_part.len > 0) allocator.free(err_part);
    // Stub metadata sourced from config.json (+ chat-template presence) WITHOUT
    // faulting in weights: dims, context window, MoE-ness, and capabilities.
    // `/v1/models` isn't hot, so reading a small JSON per unloaded entry is fine.
    const sm = model_discovery.readStubMeta(io, allocator, entry.path);

    // Capabilities. Encoder-only models advertise "embeddings"; chat models get
    // chat/tool_use/streaming/json_schema (gated on chat-template presence, the
    // same rule the loaded path uses) plus "vision" when a vision config is
    // present. Reasoning/audio stay load-gated (they need the live template /
    // encoder), so an unloaded stub may under-report those two.
    const is_encoder_stub = std.mem.eql(u8, entry.arch_hint, "bert") or
        (sm.found and sm.is_encoder);
    // Decoder embedding stubs (issue #116): StubMeta's pooling signal
    // (config key / sidecar), plus the known-family name rule for the
    // metadata-less mlx-community conversions — the SAME `poolingFromDirName`
    // the load path uses, so stub and loaded capability can't disagree.
    const stub_has_embedding = (sm.found and sm.has_embedding) or
        model_mod.poolingFromDirName(std.fs.path.basename(entry.path), entry.arch_hint) != null;
    // GGUF discovery stub (issue #59): no config.json to read StubMeta from,
    // but the embedded llama/ds4 engines always serve chat (the GGUF's own
    // template is adopted at load), so advertise the chat capability set the
    // ready path would.
    const is_gguf_stub = std.mem.eql(u8, entry.arch_hint, "gguf");
    // Native media-gen stub (image/audio/video) — advertise the modality from
    // the discovery-peeked arch_hint so the app can find it before loading.
    const media_modality = media_mod.modalityFromType(entry.arch_hint);
    const caps_part: []const u8 = blk: {
        if (media_modality) |m| {
            // The music backend advertises "music" beside "audio" on the stub
            // too (matches the ready-path readyCapsJson additive rule).
            if (m == .audio and media_mod.servesAudioRoute(io, entry.path, entry.arch_hint, .music))
                break :blk try allocator.dupe(u8, ",\"capabilities\":[\"audio\",\"music\"]");
            if (m == .audio and media_mod.servesAudioRoute(io, entry.path, entry.arch_hint, .sound))
                break :blk try allocator.dupe(u8, ",\"capabilities\":[\"audio\",\"sound\"]");
            break :blk try std.fmt.allocPrint(allocator, ",\"capabilities\":[\"{s}\"]", .{m.capability()});
        }
        if (is_encoder_stub) break :blk try allocator.dupe(u8, ",\"capabilities\":[\"embeddings\"]");
        if (is_gguf_stub) break :blk try allocator.dupe(u8, ",\"capabilities\":[\"chat\",\"tool_use\",\"streaming\",\"json_schema\"]");
        if (sm.found and sm.has_transcription) break :blk try allocator.dupe(u8, ",\"capabilities\":[\"transcription\"]");
        if (sm.found and sm.is_aligner) break :blk try allocator.dupe(u8, ",\"capabilities\":[\"alignment\"]");
        if (!sm.found or !(sm.has_chat or sm.has_vision or stub_has_embedding)) break :blk try allocator.dupe(u8, "");
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(allocator);
        try b.appendSlice(allocator, ",\"capabilities\":[");
        var n: usize = 0;
        const add = struct {
            fn call(a: std.mem.Allocator, list: *std.ArrayList(u8), cnt: *usize, name: []const u8) !void {
                if (cnt.* > 0) try list.append(a, ',');
                try list.append(a, '"');
                try list.appendSlice(a, name);
                try list.append(a, '"');
                cnt.* += 1;
            }
        }.call;
        if (sm.has_chat) {
            try add(allocator, &b, &n, "chat");
            try add(allocator, &b, &n, "tool_use");
            try add(allocator, &b, &n, "streaming");
            try add(allocator, &b, &n, "json_schema");
        }
        if (sm.has_vision) try add(allocator, &b, &n, "vision");
        if (stub_has_embedding) try add(allocator, &b, &n, "embeddings");
        try b.append(allocator, ']');
        break :blk try b.toOwnedSlice(allocator);
    };
    defer allocator.free(caps_part);

    const stub_chat = media_modality == null and !is_encoder_stub and
        (is_gguf_stub or (sm.found and !sm.has_transcription and !sm.is_aligner and sm.has_chat));
    const facts = try row_facts.forEntry(allocator, io, entry, entry.arch_hint, if (stub_chat) (if (sm.found) sm.max_position_embeddings else 0) else null, mlxGgufClaims(io, entry));
    defer allocator.free(facts);

    const mods_part: []const u8 = if (sm.found and (sm.has_vision or sm.has_audio or sm.has_chat) or is_gguf_stub) blk: {
        const m = try inputModalities(allocator, sm.found and sm.has_vision, sm.found and sm.has_vision and sm.has_video, sm.found and sm.has_audio);
        defer allocator.free(m);
        break :blk try std.fmt.allocPrint(allocator, ",\"input_modalities\":{s}", .{m});
    } else try allocator.dupe(u8, "");
    defer allocator.free(mods_part);
    // The same speech facts a loaded row carries: the app picks its native-voice mic before the model loads.
    const stub_speech = if (sm.found and sm.has_speech) blk: {
        const voices = try speech_api.voicesJson(allocator);
        defer allocator.free(voices);
        break :blk try std.fmt.allocPrint(allocator, ",\"output_modalities\":[\"text\",\"audio\"],\"voices\":{s}", .{voices});
    } else try allocator.dupe(u8, "");
    defer allocator.free(stub_speech);

    const arch_part: []const u8 = if (entry.arch_hint.len > 0) blk: {
        // arch_hint comes from config.json's model_type via discovery — the
        // supported-type allowlist guarantees no JSON metacharacters.
        break :blk try std.fmt.allocPrint(allocator, "\"architecture\":\"{s}\",", .{entry.arch_hint});
    } else &[_]u8{};
    defer if (arch_part.len > 0) allocator.free(arch_part);

    // Unloaded entries have no engine attached yet — "mlx-gguf" where the
    // MLX GGUF engine would claim the file, else "gguf" (undetermined
    // llama-vs-ds4) for GGUF paths/stubs, "mlx" for everything else.
    const engine_part = try std.fmt.allocPrint(allocator, "\"engine\":\"{s}\",", .{
        modelEngineName(null, mlxGgufClaims(io, entry), null, entry.path, entry.arch_hint),
    });
    defer allocator.free(engine_part);

    // Top-level twins of meta.context_length (issue #188): openai-models-list
    // discovery clients read row-level max_model_len/context_length. Unloaded
    // stubs advertise the architectural max — the same number meta carries here.
    const top_ctx_part: []const u8 = if (sm.found) blk: {
        break :blk try std.fmt.allocPrint(allocator, ",\"context_length\":{d},\"max_model_len\":{d}", .{
            sm.max_position_embeddings,
            sm.max_position_embeddings,
        });
    } else &[_]u8{};
    defer if (top_ctx_part.len > 0) allocator.free(top_ctx_part);

    // Dimensions/context/quant/MoE — emitted only when config.json was readable.
    const dims_part: []const u8 = if (sm.found) blk: {
        break :blk try std.fmt.allocPrint(allocator, "\"vocab_size\":{d},\"hidden_size\":{d},\"num_layers\":{d},\"quantization\":{s},\"context_length\":{d},\"model_max_tokens\":{d},\"is_moe\":{s},\"mtp_available\":{s},", .{
            sm.vocab_size,
            sm.hidden_size,
            sm.num_hidden_layers,
            quantizationJson(&quant_buf, sm.quant_bits),
            sm.max_position_embeddings,
            sm.max_position_embeddings,
            if (sm.is_moe) "true" else "false",
            if (sm.has_mtp) "true" else "false",
        });
    } else &[_]u8{};
    defer if (dims_part.len > 0) allocator.free(dims_part);

    const id_json = try jsonEscape(allocator, entry.id);
    defer allocator.free(id_json);
    return std.fmt.allocPrint(allocator,
        \\{{"id":{s},"object":"model","created":0,"owned_by":"mlx-serve","loaded":false,"state":"{s}","bytes_resident":0,"bytes_on_disk":{s}{s}{s}{s}{s}{s}{s},"meta":{{{s}{s}{s}"bytes_on_disk":{s}}}}}
    , .{ id_json, state_str, bytes_on_disk_str, err_part, top_ctx_part, caps_part, mods_part, stub_speech, facts, arch_part, engine_part, dims_part, bytes_on_disk_str });
}

pub fn handleModels(
    allocator: std.mem.Allocator,
    stream: *Conn,
    /// True for a keyless LAN peer: list only shared models, and never the
    /// remote stubs (mirroring a peer's peer invites multi-hop loops).
    lan_filtered: bool,
) !void {
    const body = renderModelsBody(allocator, stream.io, lan_filtered) catch |err| switch (err) {
        error.RegistryNotReady => return sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503),
        else => return err,
    };
    defer allocator.free(body);
    try sendModelsResponse(stream, body);
}

/// The `/v1/models` body.
pub fn renderModelsBody(allocator: std.mem.Allocator, io: std.Io, lan_filtered: bool) ![]u8 {
    // Plan 05 Phase E: emit every registry entry (loaded + unloaded), not
    // just the default model + flat discovery list. Default model is sorted
    // first so single-model clients reading `data[0]` continue to work.
    // The mutex is held for the whole render so eviction can't fire under
    // us; each rendered entry is at most a few hundred bytes. This handler
    // does NOT route through `scheduler.ensureLoaded` — listing metadata
    // shouldn't trigger a cold load of the default model.
    const registry = http_server.global_registry orelse return error.RegistryNotReady;

    var entries_buf = std.ArrayList(u8).empty;
    defer entries_buf.deinit(allocator);

    // Collect entries while holding the mutex so iteration is safe.
    // Sort: default first, then by last_used_ns desc.
    var ordered = std.ArrayList(*LoadedModel).empty;
    defer ordered.deinit(allocator);
    var alias_buf: [std.fs.max_path_bytes]u8 = undefined;
    // Any re-read happens here, not under the registry lock below.
    request_model.g_model_aliases.refreshNow(io);
    {
        registry.mutex.lockUncancelable(io);
        defer registry.mutex.unlock(io);
        try ordered.ensureTotalCapacity(allocator, registry.entries.count());
        var it = registry.entries.valueIterator();
        while (it.next()) |entry_ptr| ordered.appendAssumeCapacity(entry_ptr.*);
        const default_id = registry.default_id;
        const Cmp = struct {
            fn lt(ctx: []const u8, a: *LoadedModel, b: *LoadedModel) bool {
                const a_def = std.mem.eql(u8, a.id, ctx);
                const b_def = std.mem.eql(u8, b.id, ctx);
                if (a_def != b_def) return a_def;
                return a.last_used_ns > b.last_used_ns;
            }
        };
        std.sort.pdq(*LoadedModel, ordered.items, default_id, Cmp.lt);

        for (ordered.items) |entry| {
            if (lan_filtered and !http_server.g_lan.?.sharedAllows(entry.id)) continue;
            if (entries_buf.items.len > 0) try entries_buf.append(allocator, ',');
            const json = try renderModelEntry(allocator, io, entry);
            defer allocator.free(json);
            try appendAliasedRow(allocator, &entries_buf, json, request_model.g_model_aliases.aliasForPath(io, entry.path, &alias_buf));
        }
    }

    // Discovered LAN models and configured providers ride the same list for
    // local clients; neither is re-exported to the LAN.
    if (!lan_filtered) if (http_server.g_lan) |l| try l.appendRemoteEntries(allocator, &entries_buf);
    if (!lan_filtered) if (http_server.g_providers) |p| try p.appendEntries(allocator, &entries_buf);

    // What this machine is; keyed peers we hold no working key for, so the app can ask for
    // one; and every peer with what it said about itself.
    var extras = std.ArrayList(u8).empty;
    defer extras.deinit(allocator);
    const machine = try machineJson(allocator);
    defer allocator.free(machine);
    try extras.appendSlice(allocator, machine);
    if (!lan_filtered) if (http_server.g_lan) |l| {
        var list = std.ArrayList(u8).empty;
        defer list.deinit(allocator);
        try l.appendPeersNeedingKey(allocator, &list);
        if (list.items.len > 0) try extras.print(allocator, ",\"lan_needs_key\":[{s}]", .{list.items});
        list.clearRetainingCapacity();
        try l.appendPeerAddresses(allocator, &list);
        if (list.items.len > 0) try extras.print(allocator, ",\"lan_peers\":[{s}]", .{list.items});
    };
    return std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[{s}]{s}}}
    , .{ entries_buf.items, extras.items });
}

/// `/v1/models`' "machine" (behind the API key, unlike `/health`): the OS, the device model
/// id the app draws its icon from, its memory (the pair card's split estimate), and the
/// Thunderbolt pair this server leads, if any.
fn machineJson(allocator: std.mem.Allocator) ![]u8 {
    const pair = mlx_engine.api().pair_fields(allocator) orelse return error.OutOfMemory;
    defer allocator.free(pair);
    var model_buf: [64]u8 = undefined;
    const status_mod = @import("../status.zig");
    return std.fmt.allocPrint(allocator, ",\"machine\":{{\"os\":\"{s}\",\"model\":{f},\"free_mib\":{d},\"total_mib\":{d}{s}}}", .{
        @tagName(builtin.os.tag), std.json.fmt(machineModel(&model_buf), .{}),
        status_mod.getAvailableMemBytes() >> 20, status_mod.getTotalMemBytes() >> 20, pair,
    });
}

/// The device model id ("Mac16,5", "iPhone17,1"); empty off Apple platforms. iOS keeps it
/// in `hw.machine`, where macOS has the CPU architecture.
fn machineModel(buf: *[64]u8) []const u8 {
    const key = switch (comptime builtin.os.tag) {
        .macos => "hw.model",
        .ios, .visionos => "hw.machine",
        else => return "",
    };
    var len: usize = buf.len;
    if (std.c.sysctlbyname(key, buf, &len, null, 0) != 0) return "";
    return std.mem.sliceTo(buf[0..len], 0);
}

/// A `/v1/models` row with its Model Settings `alias` as the first field; `id`
/// stays the canonical registry id.
fn appendAliasedRow(allocator: std.mem.Allocator, out: *std.ArrayList(u8), row: []const u8, alias: ?[]const u8) !void {
    const a = alias orelse return out.appendSlice(allocator, row);
    const esc = try jsonEscape(allocator, a);
    defer allocator.free(esc);
    try out.print(allocator, "{{\"alias\":{s},{s}", .{ esc, row[1..] });
}

/// `/v1/models` responses carry the per-process LAN token
/// (`X-MLX-LAN-Token`) when LAN mode is on, so a discovering server can
/// recognize a fetch that landed on ITSELF: a stale Bonjour record of a
/// former self (same name + port, different TXT token) walks through the
/// resolve-time TXT check, and the loopback-first fetch would install our
/// own models as a "peer" (live self-mirror after a restart).
fn sendModelsResponse(stream: *Conn, body: []const u8) !void {
    logHttpResponse("200 OK", "application/json", body);
    const l = http_server.g_lan orelse return sendResponseFramed(stream, "200 OK", "application/json", body);
    if (stream.ws_mode != null) return sendResponseFramed(stream, "200 OK", "application/json", body);
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\nX-MLX-LAN-Token: {s}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n", .{
        body.len,
        &l.token_hex,
    }) catch return error.Overflow;
    try stream.writeAll(hdr);
    stream.length_framed = true;
    if (body.len > 0) try stream.writeAll(body);
}

/// `GET /v1/load-progress`: every model mid-load and what its load is doing (`load_progress.zig`).
/// Cheap enough to poll while a load runs: it reads no configs.
pub fn renderLoadProgressBody(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    return loadProgressBody(allocator, io, http_server.global_registry orelse return error.RegistryNotReady);
}

pub fn loadProgressBody(allocator: std.mem.Allocator, io: std.Io, registry: *ModelRegistry) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    const now = io_util.nowMsMonotonic(io);
    const footprint = metrics.getAppMemFootprintMb();
    try buf.appendSlice(allocator, "{\"loading\":[");
    registry.mutex.lockUncancelable(io);
    defer registry.mutex.unlock(io);
    var it = registry.entries.valueIterator();
    var first = true;
    while (it.next()) |e| {
        const entry = e.*;
        if (entry.state != .loading) continue;
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "{\"id\":");
        try chat_mod.appendJsonString(allocator, &buf, entry.id);
        try buf.appendSlice(allocator, ",\"path\":");
        try chat_mod.appendJsonString(allocator, &buf, entry.path);
        try buf.appendSlice(allocator, ",\"progress\":");
        var jb: [256]u8 = undefined;
        var w: std.Io.Writer = .fixed(&jb);
        try entry.load_progress.snapshot(now, footprint).writeJson(&w);
        try buf.appendSlice(allocator, w.buffered());
        try buf.append(allocator, '}');
    }
    try buf.appendSlice(allocator, "]}");
    return buf.toOwnedSlice(allocator);
}

/// `POST /v1/models/rescan`: re-walk the boot roots and register stubs for
/// new dirs, clearing failed loads (`ModelRegistry.rescan`). Answers `{"added":N}`.
pub fn handleModelsRescan(allocator: std.mem.Allocator, stream: *Conn) !void {
    const registry = http_server.global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
        return;
    };
    const added = registry.rescan() catch |err| {
        log.warn("/v1/models/rescan failed: {s}\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "internal_error", "Model rescan failed", 500);
        return;
    };
    if (added > 0) log.info("[registry] rescan added {d} model(s)\n", .{added});
    const body = try std.fmt.allocPrint(allocator, "{{\"added\":{d}}}", .{added});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Plan 05 Phase E: `POST /v1/load-model`. Renders a status payload for the
/// resolved-and-loaded `lm` (the dispatcher already called
/// `scheduler.ensureLoaded` against the body's `model` field, so the load
/// has actually happened by the time we get here). Returns the single
/// loaded entry with resident bytes + state so clients can confirm.
/// Plan 05 Phase E: explicit cold-load. Strict on unknown ids (404 — the
/// caller asked for a specific id, not "anything"). Routes through
/// scheduler.ensureLoaded so eviction/back-pressure kicks in.
pub fn handleLoadModelStrict(allocator: std.mem.Allocator, stream: *Conn, request_body: []const u8) !void {
    const scheduler = http_server.launch().global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Parse the model field with std.json (NOT the quick scanner): clients
    // like Swift's JSONSerialization legally escape '/' as '\/', and the raw
    // scanner slice would read "\/Users\/…" — missing the absolute-path
    // branch below and 404ing on the mangled id (live failure).
    var requested_id: []const u8 = "";
    // `"default": true` — promote the loaded model to the server default
    // (requests that omit `model` / the "mlx-serve" alias, and /v1/models'
    // default-first sort). Explicit opt-in: the app's model SWITCH sends it;
    // media-gen side-loads must never steal the chat default.
    var make_default = false;
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, request_body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            if (parsed.value.object.get("model")) |m| switch (m) {
                .string => |s| requested_id = s,
                .null => {},
                // Loading the default for an id the client misspelled as a number reads as success.
                else => {
                    try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'model' must be a string", 400);
                    return;
                },
            };
            if (parsed.value.object.get("default")) |d| {
                if (d == .bool) make_default = d.bool;
            }
        }
    } else |_| {
        requested_id = parseModelFromBody(request_body) orelse "";
    }
    // A LAN-discovered remote id: nothing to load here — the peer loads on
    // demand when the first proxied request arrives. Answer 200 with the
    // mirrored entry so client flows (load → generate → unload) work
    // unchanged on network models. Unknown peer/model falls through to
    // ensureLoaded's honest 404.
    if (lan_mod.splitRemoteId(requested_id) != null) {
        const remote_entry: ?[]u8 = if (http_server.g_providers) |p| p.entryFor(allocator, requested_id) else null;
        const entry_opt = remote_entry orelse if (http_server.g_lan) |l| l.remoteEntryFor(allocator, requested_id) else null;
        if (entry_opt) |entry| {
            defer allocator.free(entry);
            const body = try std.fmt.allocPrint(allocator, "{{\"model\":{s}}}", .{entry});
            defer allocator.free(body);
            try sendResponse(stream, "200 OK", "application/json", body);
            return;
        }
    }
    // Register-by-path: an absolute path to a model directory OUTSIDE the
    // --model-dir scan (e.g. the app's auto-downloaded embedding encoder).
    // The dir is validated exactly like discovery (config.json, supported
    // model_type + quant mode) before anything is registered; the load then
    // proceeds under the directory-basename id.
    if (std.mem.startsWith(u8, requested_id, "/")) {
        const registry = http_server.global_registry orelse {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
            return;
        };
        requested_id = registry.registerByPath(stream.io, requested_id, null) catch |err| switch (err) {
            error.ModelDirNotFound, error.InvalidModelPath => {
                try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "No loadable model directory at that path", 404);
                return;
            },
            error.UnsupportedArch => {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Model at that path has an unsupported model_type", 400);
                return;
            },
            error.UnsupportedQuantMode => {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Model at that path has an unsupported quantization mode", 400);
                return;
            },
            error.IncompleteMediaPack => {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Model at that path is an incomplete media pack (weights still downloading, or a partial copy)", 400);
                return;
            },
            else => return err,
        };
    } else if (http_server.global_registry) |r| switch (resolveRequestModelId(r, &request_model.g_model_aliases, requested_id, "/v1/load-model")) {
        .id => |id| requested_id = id,
        else => {},
    };
    const lm = scheduler.ensureLoaded(requested_id) catch |err| {
        try sendEnsureLoadedError(allocator, stream, scheduler, requested_id, err);
        return;
    };
    defer scheduler.release(lm);
    // Promotion happens only past ensureLoaded's error switch: a load that
    // FAILED must never steal the default from a model that answers.
    if (make_default) {
        if (http_server.global_registry) |registry| {
            registry.setDefault(lm.id) catch {};
            log.info("[registry] default model -> {s} (load-model request)\n", .{lm.id});
        }
    }
    const config = lm.config orelse {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_not_ready", "Loaded model has no parsed config", 500);
        return;
    };
    const model_id_json = try jsonEscape(allocator, if (lm.id.len > 0) lm.id else config.model_type);
    defer allocator.free(model_id_json);
    const bytes_resident = lm.bytes_resident;
    const bytes_on_disk_str = if (lm.bytes_on_disk) |b|
        try std.fmt.allocPrint(allocator, "{d}", .{b})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(bytes_on_disk_str);
    const drafter_loaded = lm.hasDrafter();
    const drafter_path_json = if (drafter_loaded)
        try jsonEscape(allocator, lm.drafter_path)
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(drafter_path_json);

    const body = try std.fmt.allocPrint(allocator,
        \\{{"model":{{"id":{s},"object":"model","loaded":true,"state":"ready","bytes_resident":{d},"bytes_on_disk":{s},"drafter_loaded":{s},"drafter_path":{s}}}}}
    , .{
        model_id_json,
        bytes_resident,
        bytes_on_disk_str,
        if (drafter_loaded) "true" else "false",
        drafter_path_json,
    });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Does an unload body carry keys but no usable `"model"`? `{"model_id": id}`
/// and `{"id": id}` parse fine, leave the id empty, and so resolve to the
/// DEFAULT model — which, when that default is already unloaded, answers 200
/// with `{"id":"mlx-serve","state":"unloaded"}`. That is a success-shaped
/// payload for an unload that never happened, and no client can tell it apart
/// from a real one (and on a server whose default IS resident, it unloads the
/// wrong model). Same class as the context-overflow 400: an unload that did
/// not do what was asked has to say so.
///
/// An EMPTY object (or no body at all) is left alone — "the default model" is
/// the documented shorthand this endpoint has always taken.
fn unloadBodyNamesNoModel(obj: std.json.ObjectMap) bool {
    if (obj.count() == 0) return false;
    if (obj.get("model")) |m| return m != .string;
    return true;
}

/// `POST /v1/unload-model` `{"model": id}`. Frees the model's resident GPU
/// state (the stub stays registered so it can reload). Idempotent. Returns a
/// small status payload confirming the model is unloaded.
pub fn handleUnloadModelStrict(allocator: std.mem.Allocator, stream: *Conn, request_body: []const u8) !void {
    const scheduler = http_server.launch().global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Parse the model id with std.json (handles '\/'-escaped absolute paths
    // the same way /v1/load-model does), then reduce an absolute path to its
    // basename — the registry keys media + by-path models by dir basename.
    var requested_id: []const u8 = "";
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, request_body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            if (unloadBodyNamesNoModel(parsed.value.object)) {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Unload requires the model id as a string under the \"model\" key", 400);
                return;
            }
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested_id = m.string;
            }
        }
    } else |_| {
        requested_id = parseModelFromBody(request_body) orelse "";
    }
    if (std.mem.startsWith(u8, requested_id, "/")) {
        var trimmed = requested_id;
        while (trimmed.len > 0 and trimmed[trimmed.len - 1] == '/') trimmed = trimmed[0 .. trimmed.len - 1];
        // A discovered entry is keyed `org/name`, so the path is the only exact handle.
        const by_path = if (http_server.global_registry) |r| r.peekByPath(trimmed) else null;
        requested_id = if (by_path) |e| e.id else std.fs.path.basename(trimmed);
    } else if (http_server.global_registry) |r| switch (resolveRequestModelId(r, &request_model.g_model_aliases, requested_id, "/v1/unload-model")) {
        .id => |id| requested_id = id,
        else => {},
    };

    // Remote ids hold no residency on THIS host — idempotent 200, matching
    // the load-model no-op (the peer's owner controls its memory).
    if ((http_server.g_lan != null or http_server.g_providers != null) and lan_mod.splitRemoteId(requested_id) != null and
        (http_server.global_registry == null or http_server.global_registry.?.peek(requested_id) == null))
    {
        try sendResponse(stream, "200 OK", "application/json", "{\"status\":\"ok\"}");
        return;
    }

    scheduler.unloadModel(requested_id) catch |err| switch (err) {
        error.UnknownModelId => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        },
        error.NoDefaultModel => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503);
            return;
        },
        error.Shutdown => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
            return;
        },
        else => {
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_unload_failed", "Model unload failed", 500);
            return;
        },
    };

    const id_for_body = try jsonEscape(allocator, if (requested_id.len > 0) requested_id else "mlx-serve");
    defer allocator.free(id_for_body);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"model":{{"id":{s},"object":"model","loaded":false,"state":"unloaded"}}}}
    , .{id_for_body});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

// ── tests ──

test "renderModelEntry: a Windows path id stays valid JSON" {
    const a = std.testing.allocator;
    var e = model_registry_mod.LoadedModel{
        .allocator = a,
        .id = "C:\\Users\\d\\.mlx-serve\\models\\org\\m",
        .path = "C:\\Users\\d\\.mlx-serve\\models\\org\\m",
        .bytes_on_disk = null,
        .arch_hint = "",
        .config = null,
        .weights = null,
        .transformer = null,
        .tokenizer = null,
        .chat_config = null,
        .vision_encoder = null,
        .drafter = null,
        .drafter_path = "",
        .drafter_block_size = 0,
        .prefix_cache = null,
        .refcount = std.atomic.Value(u32).init(0),
        .last_used_ns = 0,
        .bytes_resident = 0,
        .state = .unloaded,
        .error_name = null,
    };
    const row = try renderModelEntry(a, std.testing.io, &e);
    defer a.free(row);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, row, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(e.id, parsed.value.object.get("id").?.string);
}

test "machineJson names the OS and the device model, as valid JSON" {
    const a = std.testing.allocator;
    const field = try machineJson(a);
    defer a.free(field);
    const j = try std.fmt.allocPrint(a, "{{\"data\":[]{s}}}", .{field});
    defer a.free(j);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, j, .{});
    defer parsed.deinit();
    const m = parsed.value.object.get("machine").?.object;
    try std.testing.expectEqualStrings(@tagName(builtin.os.tag), m.get("os").?.string);
    if (builtin.os.tag == .macos) try std.testing.expect(m.get("model").?.string.len > 0);
    try std.testing.expect(m.get("total_mib").?.integer >= m.get("free_mib").?.integer);
}

test "an unload body naming an unrecognised key is a 400, not the default model" {
    const Case = struct { body: []const u8, rejected: bool };
    for ([_]Case{
        .{ .body = "{\"model\":\"org/repo\"}", .rejected = false },
        .{ .body = "{\"model\":\"/abs/path/org-repo\"}", .rejected = false },
        // The shorthand every other endpoint honours must keep working.
        .{ .body = "{}", .rejected = false },
        // The live bug: 200 + "mlx-serve" with the named model still resident.
        .{ .body = "{\"model_id\":\"org/repo\"}", .rejected = true },
        .{ .body = "{\"id\":\"org/repo\"}", .rejected = true },
        // A present-but-unusable `model` took the same silent path.
        .{ .body = "{\"model\":123}", .rejected = true },
        .{ .body = "{\"model\":null}", .rejected = true },
    }) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        try testing.expectEqual(c.rejected, unloadBodyNamesNoModel(parsed.value.object));
    }
}

test "appendAliasedRow: the alias rides the row, the id stays canonical" {
    const a = std.testing.allocator;
    var out = std.ArrayList(u8).empty;
    defer out.deinit(a);
    try appendAliasedRow(a, &out, "{\"id\":\"org/Q\"}", "q");
    try appendAliasedRow(a, &out, "{\"id\":\"g\"}", null);
    try std.testing.expectEqualStrings("{\"alias\":\"q\",\"id\":\"org/Q\"}{\"id\":\"g\"}", out.items);
}

test "input_modalities renders resident video weights and native unloaded capabilities" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const native = "{\"model_type\":\"nemotron_h_omni\",\"vision_config\":{\"model_type\":\"radio\",\"num_attention_heads\":16},\"llm_config\":{\"model_type\":\"nemotron_h_puzzle\"}}";
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = native });
    var path_buf: [512]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const reg = try model_registry_mod.ModelRegistry.init(a, io, null, 1, 0, null);
    defer reg.deinit();
    const entry = try reg.registerStub("video", path, 0);
    defer {
        entry.config = null;
        entry.chat_config = null;
        entry.vision_encoder = null;
        entry.state = .unloaded;
    }
    var qwen_config = model_mod.ModelConfig{ .qwen_vision = true, .video_token_id = 7 };
    var radio_config = try model_mod.parseConfigFromJson(a, native);
    defer radio_config.deinit(a);
    var chat_config = chat_mod.ChatConfig{
        .chat_template = "template", .bos_token = null, .eos_token = null,
        .add_bos_token = false, .allocator = a,
    };
    var encoder = std.mem.zeroInit(vision_mod.VisionEncoder, .{ .allocator = a, .config = qwen_config });
    var radio = std.mem.zeroInit(radio_vision.RadioVision, .{ .allocator = a });
    const video_weight = mlx.mlx_array_new_float(1);
    defer _ = mlx.mlx_array_free(video_weight);
    entry.config = &qwen_config;
    entry.chat_config = &chat_config;
    entry.vision_encoder = &encoder;
    entry.state = .ready;
    const expect_modalities = struct {
        fn check(alloc: std.mem.Allocator, runtime_io: std.Io, model: *LoadedModel, want: []const []const u8) !void {
            const json = try renderModelEntry(alloc, runtime_io, model);
            defer alloc.free(json);
            const parsed = try std.json.parseFromSlice(std.json.Value, alloc, json, .{});
            defer parsed.deinit();
            const modalities = parsed.value.object.get("input_modalities").?.array.items;
            try testing.expectEqual(want.len, modalities.len);
            for (want, modalities) |expected, actual| try testing.expectEqualStrings(expected, actual.string);
        }
    }.check;
    try expect_modalities(a, io, entry, &.{ "text", "image", "video" });
    entry.config = &radio_config;
    encoder.config = radio_config;
    radio.video_w = video_weight;
    encoder.radio = radio;
    try expect_modalities(a, io, entry, &.{ "text", "image", "video" });
    encoder.radio.?.video_w = .{ .ctx = null };
    try expect_modalities(a, io, entry, &.{ "text", "image" });
    entry.state = .unloaded;
    try expect_modalities(a, io, entry, &.{ "text", "image", "video" });
}

/// A row's `input_modalities`, loaded or not: text, then what the model reads besides.
fn inputModalities(a: std.mem.Allocator, vision: bool, video: bool, audio: bool) ![]u8 {
    return std.fmt.allocPrint(a, "[\"text\"{s}{s}{s}]", .{
        if (vision) ",\"image\"" else "", if (video) ",\"video\"" else "", if (audio) ",\"audio\"" else "",
    });
}

test "an unloaded pack that sees and hears lists both, as its loaded row does" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/LiquidAI_LFM3-Omni-2B-MLX-8bit.json") });
    var path_buf: [512]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const reg = try model_registry_mod.ModelRegistry.init(a, io, null, 1, 0, null);
    defer reg.deinit();
    const entry = try reg.registerStub("omni", path, 0);
    const json = try renderModelEntry(a, io, entry);
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const mods = parsed.value.object.get("input_modalities").?.array.items;
    try testing.expectEqual(@as(usize, 4), mods.len);
    try testing.expectEqualStrings("audio", mods[3].string);
}

test "an unloaded row carries its agent budget, media modality and engines" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "chat");
    try tmp.dir.createDirPath(io, "image");
    try tmp.dir.writeFile(io, .{ .sub_path = "chat/config.json", .data = "{\"model_type\":\"llama\",\"max_position_embeddings\":24576}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "chat/chat_template.jinja", .data = "{{ messages }}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "image/config.json", .data = "{\"model_type\":\"flux2\"}" });
    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    const chat_path = try std.fs.path.join(a, &.{ root, "chat" });
    defer a.free(chat_path);
    const image_path = try std.fs.path.join(a, &.{ root, "image" });
    defer a.free(image_path);
    const reg = try model_registry_mod.ModelRegistry.init(a, io, null, 1, 0, null);
    defer reg.deinit();
    const chat = try reg.registerStubWithArch("chat", chat_path, 0, "llama");
    const image = try reg.registerStubWithArch("image", image_path, 0, "flux2");

    const chat_json = try renderModelEntry(a, io, chat);
    defer a.free(chat_json);
    const cp = try std.json.parseFromSlice(std.json.Value, a, chat_json, .{});
    defer cp.deinit();
    const budget = cp.value.object.get("agent_budget").?.object;
    try testing.expectEqual(@as(i64, 24576), budget.get("context").?.integer);
    try testing.expectEqual(@as(i64, 12288), budget.get("output").?.integer);
    try testing.expectEqual(@as(i64, 6144), budget.get("compaction_reserve").?.integer);
    try testing.expect(cp.value.object.get("modality") == null);
    try testing.expectEqualStrings("mlx", cp.value.object.get("engines").?.array.items[0].string);

    const image_json = try renderModelEntry(a, io, image);
    defer a.free(image_json);
    const ip = try std.json.parseFromSlice(std.json.Value, a, image_json, .{});
    defer ip.deinit();
    try testing.expect(ip.value.object.get("agent_budget") == null);
    try testing.expectEqualStrings("image", ip.value.object.get("modality").?.string);
    const engines = ip.value.object.get("engines").?.array.items;
    try testing.expectEqual(@as(usize, 1), engines.len);
    try testing.expectEqualStrings("mlx", engines[0].string);
}

test "readyCapsJson: embeddings capability — encoders alone, decoders beside chat" {
    const a = std.testing.allocator;
    // Encoder-only (BERT/EmbeddingGemma): embeddings, nothing else.
    var enc = try readyCapsJson(a, .{ .has_embedding = true });
    defer enc.deinit(a);
    try std.testing.expectEqualStrings("[\"embeddings\"]", enc.items);
    // A pooling-contracted decoder (Qwen3-Embedding, issue #116) keeps its
    // chat set AND advertises embeddings — a READY model never advertises
    // less capability than its stub.
    var both = try readyCapsJson(a, .{ .has_chat = true, .has_embedding = true });
    defer both.deinit(a);
    try std.testing.expect(std.mem.indexOf(u8, both.items, "\"chat\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, both.items, "\"embeddings\"") != null);
}

test "readyCapsJson: an audio tower on a text trunk advertises transcription" {
    const a = std.testing.allocator;
    var asr = try readyCapsJson(a, .{ .has_transcription = true });
    defer asr.deinit(a);
    try std.testing.expectEqualStrings("[\"transcription\"]", asr.items);
    var aligner = try readyCapsJson(a, .{ .has_alignment = true });
    defer aligner.deinit(a);
    try std.testing.expectEqualStrings("[\"alignment\"]", aligner.items);
}

test "readyCapsJson: every resident media engine surfaces its capability (mesh -> 3d)" {
    const a = std.testing.allocator;

    // A ready 3D shape model (no chat template, no encoder) must advertise
    // "3d" — the stub path already did; the READY path shipped without the
    // mesh arm and rendered [] (live, test_3d_gen.sh check 1).
    var mesh = try readyCapsJson(a, .{ .has_mesh_engine = true });
    defer mesh.deinit(a);
    try std.testing.expectEqualStrings("[\"3d\"]", mesh.items);

    // The other three engine slots keep their existing capability names.
    var img = try readyCapsJson(a, .{ .has_image_engine = true });
    defer img.deinit(a);
    try std.testing.expectEqualStrings("[\"image\"]", img.items);
    var aud = try readyCapsJson(a, .{ .has_audio_engine = true });
    defer aud.deinit(a);
    try std.testing.expectEqualStrings("[\"audio\"]", aud.items);
    // A music-backend audio engine advertises "music" ADDITIVELY beside "audio".
    var mus = try readyCapsJson(a, .{ .has_audio_engine = true, .has_music_backend = true });
    defer mus.deinit(a);
    try std.testing.expectEqualStrings("[\"audio\",\"music\"]", mus.items);
    var vid = try readyCapsJson(a, .{ .has_video_engine = true });
    defer vid.deinit(a);
    try std.testing.expectEqualStrings("[\"video\"]", vid.items);

    // Chat-class flags are unaffected by the media arms.
    var chat = try readyCapsJson(a, .{ .has_chat = true, .has_reasoning = true });
    defer chat.deinit(a);
    try std.testing.expectEqualStrings(
        "[\"chat\",\"tool_use\",\"streaming\",\"reasoning\",\"json_schema\"]",
        chat.items,
    );
}

test "readyHasChat: embedded-engine GGUF without a chat template still advertises chat" {
    const t = std.testing;
    // MLX model with a template — chat.
    try t.expect(readyHasChat(false, 1234, false));
    // Encoder-only never chats, template or not.
    try t.expect(!readyHasChat(true, 1234, false));
    try t.expect(!readyHasChat(true, 0, true));
    // The live bug: a loaded DSV4-Flash GGUF ships no
    // chat_template in its header (fallback formatting serves chat fine), but
    // the ready path gated "chat" on template presence — the peer advertised
    // capabilities:[] and LAN clients hid the model ("No models yet" in the
    // tray while actively chatting on it). The unloaded GGUF stub path
    // already advertises the chat set; loaded must not advertise LESS.
    try t.expect(readyHasChat(false, 0, true));
    // Templateless pure-MLX entry keeps the existing conservative behavior.
    try t.expect(!readyHasChat(false, 0, false));
}

test "modelEngineName: native dsv4 reports mlx, embedded engines report themselves" {
    // Loaded entries: the attached engine pointer decides.
    try testing.expectEqualStrings("ds4", modelEngineName(.ds4, false, null, "/m/DeepSeek-V4-Flash.gguf", ""));
    try testing.expectEqualStrings("llama", modelEngineName(.llama, false, null, "/m/qwen.gguf", ""));
    try testing.expectEqualStrings("ane", modelEngineName(.ane, false, null, "/m/Qwen3.8-27B-MLX-Serve-4bit", "qwen3_5"));
    // A GGUF on the MLX path (src/sdk/quant/gguf) is its own engine: the app
    // reads "gguf" as llama.cpp.
    try testing.expectEqualStrings("mlx-gguf", modelEngineName(null, true, null, "/m/qwen.gguf", ""));
    // The MLX path's own engines, from a loaded model's config.
    var sushi: model_mod.ModelConfig = .{};
    sushi.exl3 = @as(@typeInfo(@TypeOf(sushi.exl3)).optional.child, undefined);
    try testing.expectEqualStrings("sushi", modelEngineName(null, false, &sushi, "/m/GLM-5.3-Flash-Sushi", "glm5_next"));
    try testing.expectEqualStrings("mlx", modelEngineName(null, false, &(model_mod.ModelConfig{}), "/m/gemma-4-12b", "gemma4"));
    // NATIVE deepseek_v4 (safetensors dir): architecture alone can't
    // distinguish it from the ds4 GGUF — meta.engine must.
    try testing.expectEqualStrings("mlx", modelEngineName(null, false, null, "/m/ddalcu/DeepSeek-V4-Flash-MLX-Serve", "deepseek_v4"));
    // Unloaded GGUF stubs: engine undetermined until the header is read.
    try testing.expectEqualStrings("gguf", modelEngineName(null, false, null, "/m/x.gguf", ""));
    try testing.expectEqualStrings("gguf", modelEngineName(null, false, null, "/m/dir", "gguf"));
    try testing.expectEqualStrings("mlx", modelEngineName(null, false, null, "/m/gemma-4-12b", "gemma4"));
}
