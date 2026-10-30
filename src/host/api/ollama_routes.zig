//! Ollama sink, early routes, quantization json; tags, show, pull, chat, generate, embed

const std = @import("std");
const env = @import("../../env.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const model_registry_mod = @import("../registry/registry.zig");
const model_discovery = @import("../registry/discovery.zig");
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const io_util = @import("../../sdk/util/io_util.zig");
const ollama_mod = @import("ollama/translate.zig");
const cli_mod = @import("../cli/cli.zig");
const build_options = @import("build_options");
const nowMs = io_util.nowMs;
const testing = std.testing;
const http_server = @import("../http/server.zig");
const mlx_engine = @import("../engines/mlx.zig");
const request_model = @import("request_model.zig");

const Conn = @import("../http/conn.zig").Conn;
const isRoute = @import("../http/router.zig").isRoute;
const resolveRequestModelId = @import("request_model.zig").resolveRequestModelId;
const getEffectiveContextLength = @import("admission.zig").getEffectiveContextLength;
const handleEmbeddings = @import("embeddings.zig").handleEmbeddings;
const handleChatCompletions = @import("chat_completions.zig").handleChatCompletions;
const handleCompletions = @import("completions.zig").handleCompletions;
const sendResponse = @import("../http/respond.zig").sendResponse;

/// Sink output path: writes translated bytes DIRECTLY through the Conn's
/// writer interface, bypassing the `ollama_sink` hook in writeAll.
fn ollamaSinkOut(impl: *anyopaque, data: []const u8) anyerror!void {
    const c: *Conn = @ptrCast(@alignCast(impl));
    try c.writer().writeAll(data);
    try c.writer().flush();
}

fn ollamaSinkNowMs(impl: *anyopaque) i64 {
    const c: *Conn = @ptrCast(@alignCast(impl));
    return nowMs(c.io);
}

pub fn sendOllamaError(allocator: std.mem.Allocator, stream: *Conn, status: []const u8, message: []const u8) !void {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try out.writer.writeAll("{\"error\":");
    try ollama_mod.writeJsonString(&out.writer, message);
    try out.writer.writeAll("}");
    try sendResponse(stream, status, "application/json", out.written());
}

/// /api/* endpoints that must not trigger a model load. Returns true when
/// the request was fully handled.
pub fn handleOllamaEarly(allocator: std.mem.Allocator, stream: *Conn, method: []const u8, path: []const u8, body: []const u8) !bool {
    const is_get = std.mem.eql(u8, method, "GET");
    const is_post = std.mem.eql(u8, method, "POST");
    if ((is_get or std.mem.eql(u8, method, "HEAD")) and isRoute(path, "/api/version")) {
        log.debug("{s} /api/version -> 200\n", .{method});
        const vbody = try std.fmt.allocPrint(allocator, "{{\"version\":\"{s}\"}}", .{build_options.version});
        defer allocator.free(vbody);
        try sendResponse(stream, "200 OK", "application/json", if (is_get) vbody else "");
        return true;
    }
    if (is_get and isRoute(path, "/api/tags")) {
        log.debug("GET  /api/tags -> 200\n", .{});
        try handleOllamaTags(allocator, stream, false);
        return true;
    }
    if (is_get and isRoute(path, "/api/ps")) {
        log.debug("GET  /api/ps -> 200\n", .{});
        try handleOllamaTags(allocator, stream, true);
        return true;
    }
    if (is_post and isRoute(path, "/api/show")) {
        try handleOllamaShow(allocator, stream, body);
        return true;
    }
    if (is_post and isRoute(path, "/api/pull")) {
        try handleOllamaPull(allocator, stream, body);
        return true;
    }
    // Registry-mutating Ollama endpoints we deliberately don't support get
    // an explicit, actionable error instead of a bare 404.
    const unsupported = [_][]const u8{ "/api/create", "/api/copy", "/api/delete", "/api/push", "/api/blobs" };
    for (unsupported) |prefix| {
        if (std.mem.startsWith(u8, path, prefix)) {
            log.warn("{s} {s} -> 501 (unsupported ollama endpoint)\n", .{ method, path });
            try sendOllamaError(allocator, stream, "501 Not Implemented", "this Ollama endpoint is not supported by mlx-serve; manage models via /v1/load-model, /api/pull, or the MLX-Serve app");
            return true;
        }
    }
    return false;
}

/// Ollama-style model name → registered model id, or null. Registry ids
/// are stable for the process lifetime (unload keeps the stub), so the
/// returned slice stays valid after the mutex drops.
pub fn ollamaResolveRegistryId(io: std.Io, registry: *ModelRegistry, name: []const u8) ?[]const u8 {
    registry.mutex.lockUncancelable(io);
    defer registry.mutex.unlock(io);
    var ids_buf: [128][]const u8 = undefined;
    var n: usize = 0;
    var it = registry.entries.valueIterator();
    while (it.next()) |ep| {
        if (n >= ids_buf.len) break;
        ids_buf[n] = ep.*.id;
        n += 1;
    }
    const idx = ollama_mod.resolveName(name, ids_buf[0..n]) orelse return null;
    return ids_buf[idx];
}

fn ollamaQuantOf(id: []const u8) []const u8 {
    if (std.ascii.findIgnoreCase(id, "4bit") != null) return "4bit";
    if (std.ascii.findIgnoreCase(id, "8bit") != null) return "8bit";
    if (std.ascii.findIgnoreCase(id, "bf16") != null) return "BF16";
    if (std.ascii.findIgnoreCase(id, "nvfp4") != null) return "NVFP4";
    if (std.ascii.findIgnoreCase(id, "q4") != null) return "Q4";
    if (std.ascii.findIgnoreCase(id, "q8") != null) return "Q8";
    return "";
}

/// `meta.quantization` as a JSON value: `"8-bit"`, or null when the root config declares no
/// width (media packs, bf16 and per-module mixed packs).
pub fn quantizationJson(buf: *[16]u8, bits: u32) []const u8 {
    if (bits == 0) return "null";
    return std.fmt.bufPrint(buf, "\"{d}-bit\"", .{bits}) catch "null";
}

/// Snapshot one registry entry into the pure TagEntry shape. Caller holds
/// the registry mutex; id/arch_hint slices are entry-owned and stable.
fn ollamaTagEntryOf(io: std.Io, e: *LoadedModel) ollama_mod.TagEntry {
    // `releaseRetainedCpuState`'s reader contract. Not `== .ready`: an
    // unloaded entry retains its config and must stay fully listable.
    const ready = e.state != .loading;
    const family: []const u8 = if (ready and e.config != null) e.config.?.model_type else (if (e.arch_hint.len > 0) e.arch_hint else "unknown");
    // arch_hint "gguf" covers unloaded discovery stubs whose PATH is a
    // directory of .gguf files (issue #59) — no engine yet, no .gguf suffix.
    const is_gguf = e.servesGguf() or
        std.mem.endsWith(u8, e.path, ".gguf") or std.mem.eql(u8, e.arch_hint, "gguf");
    var modified_ms: i64 = 0;
    // config.json mtime; .gguf entries fall back to 0 (epoch) rather than
    // paying a parent-dir walk. Guard the absolute-path precondition —
    // openDirAbsolute on a non-absolute path is ReleaseFast UB (CLAUDE.md).
    if (!is_gguf and e.path.len > 0 and std.fs.path.isAbsolute(e.path)) {
        if (std.Io.Dir.openDirAbsolute(io, e.path, .{})) |d| {
            var dir = d;
            defer dir.close(io);
            if (dir.statFile(io, "config.json", .{})) |st| {
                modified_ms = st.mtime.toMilliseconds();
            } else |_| {}
        } else |_| {}
    }
    return .{
        .id = e.id,
        .size_bytes = e.bytes_on_disk orelse 0,
        .modified_ms = modified_ms,
        .family = family,
        .format = if (is_gguf) "gguf" else "safetensors",
        .quant = ollamaQuantOf(e.id),
    };
}

/// GET /api/tags (`ps_only=false`: every registered model) and GET /api/ps
/// (`ps_only=true`: only GPU-resident entries, with residency bytes).
fn handleOllamaTags(allocator: std.mem.Allocator, stream: *Conn, ps_only: bool) !void {
    const registry = http_server.global_registry orelse {
        try sendOllamaError(allocator, stream, "503 Service Unavailable", "registry not ready");
        return;
    };
    var body: []u8 = undefined;
    {
        registry.mutex.lockUncancelable(stream.io);
        defer registry.mutex.unlock(stream.io);
        if (ps_only) {
            var entries = std.ArrayList(ollama_mod.PsEntry).empty;
            defer entries.deinit(allocator);
            var it = registry.entries.valueIterator();
            while (it.next()) |ep| {
                const e = ep.*;
                if (e.state != .ready) continue;
                try entries.append(allocator, .{
                    .tag = ollamaTagEntryOf(stream.io, e),
                    .resident_bytes = e.bytes_resident,
                });
            }
            body = try ollama_mod.renderPsJson(allocator, entries.items);
        } else {
            var entries = std.ArrayList(ollama_mod.TagEntry).empty;
            defer entries.deinit(allocator);
            var it = registry.entries.valueIterator();
            while (it.next()) |ep| {
                try entries.append(allocator, ollamaTagEntryOf(stream.io, ep.*));
            }
            body = try ollama_mod.renderTagsJson(allocator, entries.items);
        }
    }
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// POST /api/show — model metadata + capabilities.
fn handleOllamaShow(allocator: std.mem.Allocator, stream: *Conn, body: []const u8) !void {
    const registry = http_server.global_registry orelse {
        try sendOllamaError(allocator, stream, "503 Service Unavailable", "registry not ready");
        return;
    };
    var requested: []const u8 = "";
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            // "model" is current; "name" is the pre-0.5 client field.
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested = m.string;
            } else if (parsed.value.object.get("name")) |m| {
                if (m == .string) requested = m.string;
            }
        }
    } else |_| {}
    if (requested.len == 0) {
        try sendOllamaError(allocator, stream, "400 Bad Request", "model is required");
        return;
    }
    switch (resolveRequestModelId(registry, &request_model.g_model_aliases, requested, "/api/show")) {
        .id => |id| requested = id,
        else => {},
    }

    var rendered: ?[]u8 = null;
    {
        registry.mutex.lockUncancelable(stream.io);
        defer registry.mutex.unlock(stream.io);
        var ids_buf: [128][]const u8 = undefined;
        var n: usize = 0;
        var entry_buf: [128]*LoadedModel = undefined;
        var it = registry.entries.valueIterator();
        while (it.next()) |ep| {
            if (n >= ids_buf.len) break;
            ids_buf[n] = ep.*.id;
            entry_buf[n] = ep.*;
            n += 1;
        }
        if (ollama_mod.resolveName(requested, ids_buf[0..n])) |idx| {
            const e = entry_buf[idx];
            // Same contract as `ollamaTagEntryOf`, and it matters more here:
            // `chat_template` is copied into the response body.
            const e_ready = e.state != .loading;
            const template: []const u8 = if (e_ready and e.chat_config != null) e.chat_config.?.chat_template else "";
            const is_encoder = if (e_ready and e.config != null) e.config.?.is_encoder_only else std.mem.eql(u8, e.arch_hint, "bert");
            // Embedding capability is wider than encoder-ness: a pooling-
            // contracted decoder (Qwen3-Embedding) reports it too (issue #116).
            const has_embedding = if (e_ready and e.config != null) e.config.?.hasEmbeddingCapability() else std.mem.eql(u8, e.arch_hint, "bert");
            const has_chat = !is_encoder;
            // An entry that is not resident has no live template or tower; its files still say what it can do.
            const sm: model_discovery.StubMeta = if (e.state != .ready) model_discovery.readStubMeta(stream.io, allocator, e.path) else .{};
            rendered = try ollama_mod.renderShowJson(allocator, .{
                .tag = ollamaTagEntryOf(stream.io, e),
                .context_length = if (e_ready and e.config != null) getEffectiveContextLength(e.config.?) else 0,
                .template = template,
                .has_chat = has_chat,
                .has_tools = has_chat,
                .has_vision = mlx_engine.facts(e).images or sm.has_vision,
                .has_thinking = has_chat and (model_discovery.templateSupportsThinking(template) or sm.has_thinking),
                .has_embedding = has_embedding,
            });
        }
    }
    if (rendered) |r| {
        defer allocator.free(r);
        log.debug("POST /api/show -> 200 ({s})\n", .{requested});
        try sendResponse(stream, "200 OK", "application/json", r);
    } else {
        log.warn("POST /api/show -> 404 (unknown model {s})\n", .{requested});
        try sendOllamaError(allocator, stream, "404 Not Found", "model not found");
    }
}

/// Progress reporter for /api/pull: each status line becomes an Ollama
/// NDJSON `{"status":"…"}` chunk on the wire.
const OllamaPullSink = struct {
    stream: *Conn,
    allocator: std.mem.Allocator,
    quiet: bool = false,
    headers_sent: bool = false,
    write_failed: bool = false,

    fn report(impl: *anyopaque, line: []const u8) void {
        const self: *OllamaPullSink = @ptrCast(@alignCast(impl));
        if (self.quiet) return;
        self.emit("status", line) catch {
            self.write_failed = true;
        };
    }

    fn emit(self: *OllamaPullSink, key: []const u8, line: []const u8) !void {
        if (!self.headers_sent) {
            self.headers_sent = true;
            try self.stream.writeAll("HTTP/1.1 200 OK\r\n" ++
                "Content-Type: application/x-ndjson\r\n" ++
                "Cache-Control: no-cache\r\n" ++
                "Connection: close\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "\r\n");
        }
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        try out.writer.writeAll("{\"");
        try out.writer.writeAll(key);
        try out.writer.writeAll("\":");
        try ollama_mod.writeJsonString(&out.writer, line);
        try out.writer.writeAll("}\n");
        try self.stream.writeAll(out.written());
    }
};

/// POST /api/pull — native HF download into ~/.mlx-serve/models (same
/// resolver + layout as `mlx-serve pull`), then register-by-path so the
/// model is immediately loadable by name. Streams NDJSON status lines
/// unless the client passed `stream:false`.
fn handleOllamaPull(allocator: std.mem.Allocator, stream: *Conn, body: []const u8) !void {
    // No `curl`, no arbitrary-path downloads in the App Store build — the Swift
    // app owns model downloads via URLSession into the container.
    if (@import("build_options").mas) {
        try sendOllamaError(allocator, stream, "501 Not Implemented", "model pull is unavailable in this build");
        return;
    }
    var requested: []const u8 = "";
    var wants_stream = true;
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested = m.string;
            } else if (parsed.value.object.get("name")) |m| {
                if (m == .string) requested = m.string;
            }
            if (parsed.value.object.get("stream")) |s| {
                wants_stream = s == .bool and s.bool;
            }
        }
    } else |_| {}
    if (requested.len == 0) {
        try sendOllamaError(allocator, stream, "400 Bad Request", "model is required");
        return;
    }
    const resolved = cli_mod.resolveShortName(requested) orelse {
        try sendOllamaError(allocator, stream, "404 Not Found", "unknown model name; use a known short name or a HuggingFace 'org/repo' id");
        return;
    };
    const home = std.mem.span(env.get(.HOME) orelse "/tmp");
    const dest = try cli_mod.modelDestPath(allocator, home, resolved.repo);
    defer allocator.free(dest);

    log.info("POST /api/pull {s} -> {s}\n", .{ requested, dest });
    var sink = OllamaPullSink{ .stream = stream, .allocator = allocator, .quiet = !wants_stream };
    const reporter = cli_mod.Reporter{ .impl = &sink, .reportFn = &OllamaPullSink.report };
    if (!cli_mod.modelPresent(stream.io, dest)) {
        cli_mod.pullRepo(allocator, stream.io, resolved, dest, reporter, false) catch {
            if (sink.headers_sent) {
                sink.emit("error", "pull failed (partials kept — retry to resume)") catch {};
            } else {
                try sendOllamaError(allocator, stream, "500 Internal Server Error", "pull failed (partials kept — retry to resume)");
            }
            return;
        };
    }
    // Make it loadable by name right away. GGUF-only dirs (no config.json)
    // aren't registerable this way — they still work via --model / the app.
    if (http_server.global_registry) |registry| {
        _ = registry.registerByPath(stream.io, dest, resolved.repo) catch {};
    }
    sink.quiet = false;
    sink.emit("status", "success") catch {};
}

pub fn handleOllamaChat(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *LoadedModel) !void {
    var tr = ollama_mod.translateChatRequest(allocator, body) catch |err| switch (err) {
        error.InvalidRequest => {
            log.warn("POST /api/chat -> 400 (invalid request)\n", .{});
            try sendOllamaError(allocator, stream, "400 Bad Request", "invalid chat request: model and messages are required");
            return;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer tr.deinit(allocator);
    log.debug("POST /api/chat (stream={any}) -> inner /v1/chat/completions\n", .{tr.wants_stream});
    var sink = ollama_mod.Sink.init(allocator, .{
        .mode = .chat,
        .wants_stream = tr.wants_stream,
        .model = tr.model,
        .out_impl = stream,
        .outFn = &ollamaSinkOut,
        .nowMsFn = &ollamaSinkNowMs,
    });
    defer sink.deinit();
    stream.ollama_sink = &sink;
    defer stream.ollama_sink = null;
    try handleChatCompletions(allocator, stream, tr.body, lm);
    stream.ollama_sink = null;
    try sink.finish();
}

pub fn handleOllamaGenerate(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *LoadedModel) !void {
    var tr = ollama_mod.translateGenerateRequest(allocator, body) catch |err| switch (err) {
        error.InvalidRequest => {
            log.warn("POST /api/generate -> 400 (invalid request)\n", .{});
            try sendOllamaError(allocator, stream, "400 Bad Request", "invalid generate request: prompt is required");
            return;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer tr.deinit(allocator);
    if (tr.load_only) {
        log.info("POST /api/generate (no prompt) -> load handshake for {s}\n", .{tr.model});
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();
        var iso_buf: [32]u8 = undefined;
        try out.writer.writeAll("{\"model\":");
        try ollama_mod.writeJsonString(&out.writer, tr.model);
        try out.writer.writeAll(",\"created_at\":");
        try ollama_mod.writeJsonString(&out.writer, ollama_mod.formatIso8601(&iso_buf, nowMs(stream.io)));
        try out.writer.writeAll(",\"response\":\"\",\"done\":true,\"done_reason\":\"load\"}");
        try sendResponse(stream, "200 OK", "application/json", out.written());
        return;
    }
    log.debug("POST /api/generate (stream={any}, raw={any}) -> inner {s}\n", .{ tr.wants_stream, tr.raw, if (tr.raw) "/v1/completions" else "/v1/chat/completions" });
    var sink = ollama_mod.Sink.init(allocator, .{
        .mode = .generate,
        .wants_stream = tr.wants_stream,
        .model = tr.model,
        .out_impl = stream,
        .outFn = &ollamaSinkOut,
        .nowMsFn = &ollamaSinkNowMs,
    });
    defer sink.deinit();
    stream.ollama_sink = &sink;
    defer stream.ollama_sink = null;
    if (tr.raw) {
        try handleCompletions(allocator, stream, tr.body, lm);
    } else {
        try handleChatCompletions(allocator, stream, tr.body, lm);
    }
    stream.ollama_sink = null;
    try sink.finish();
}

pub fn handleOllamaEmbed(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *LoadedModel, legacy: bool) !void {
    var tr = ollama_mod.translateEmbedRequest(allocator, body, legacy) catch |err| switch (err) {
        error.InvalidRequest => {
            log.warn("POST /api/embed -> 400 (invalid request)\n", .{});
            try sendOllamaError(allocator, stream, "400 Bad Request", if (legacy) "invalid request: prompt is required" else "invalid request: input is required");
            return;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer tr.deinit(allocator);
    var sink = ollama_mod.Sink.init(allocator, .{
        .mode = .chat, // unused for embed; finishEmbed re-renders the capture
        .wants_stream = false,
        .model = tr.model,
        .out_impl = stream,
        .outFn = &ollamaSinkOut,
        .nowMsFn = &ollamaSinkNowMs,
    });
    defer sink.deinit();
    stream.ollama_sink = &sink;
    defer stream.ollama_sink = null;
    try handleEmbeddings(allocator, stream, tr.body, lm);
    stream.ollama_sink = null;
    try sink.finishEmbed(legacy);
}

// ── tests ──

test "ollamaTagEntryOf: reads config unless the entry is mid-load" {
    // Bar: a `.loading` entry's retained CPU state can be freed off-mutex, so it must not be read.
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try model_registry_mod.ModelRegistry.init(std.testing.allocator, io, null, 3, 0, null);
    defer reg.deinit();
    const e = try reg.registerStubWithArch("m", "/path/to/m", 1024, "arch-hint");

    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.model_type = "from-config";
    e.config = &cfg;
    // A borrowed stack config must not reach `reg.deinit`, failure or not.
    defer {
        e.config = null;
        e.state = .unloaded;
    }

    // Unloaded but retained: still fully listable, which is the retention contract.
    e.state = .unloaded;
    try std.testing.expectEqualStrings("from-config", ollamaTagEntryOf(io, e).family);

    e.state = .ready;
    try std.testing.expectEqualStrings("from-config", ollamaTagEntryOf(io, e).family);

    // Mid-load: the pointer may be freed under us, so fall back to the hint.
    e.state = .loading;
    try std.testing.expectEqualStrings("arch-hint", ollamaTagEntryOf(io, e).family);
}

test "quantizationJson: a declared width is N-bit, an undeclared one is null" {
    var buf: [16]u8 = undefined;
    try testing.expectEqualStrings("\"8-bit\"", quantizationJson(&buf, 8));
    try testing.expectEqualStrings("null", quantizationJson(&buf, 0));
}
