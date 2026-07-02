//! Ollama-compatible API (/api/*) — a pure translation layer.
//!
//! mlx-serve speaks the Ollama wire so the Ollama client ecosystem
//! (Raycast, Obsidian, Enchanted, ollama-python/js, Open WebUI, …) works
//! against a faster MLX/GGUF backend with zero client changes.
//!
//! Design: NO duplicated inference pipeline. An inbound Ollama request is
//! translated into the OpenAI request shape (`translateChatRequest` /
//! `translateGenerateRequest`, always inner-streaming with usage) and fed
//! to the EXISTING /v1 handlers; their SSE byte stream is re-framed into
//! Ollama NDJSON by a `Sink` installed on the connection
//! (`Conn.ollama_sink` — the same interception pattern as `WsBridge`).
//! Everything in this file is hermetically testable; server.zig owns the
//! glue (routing, registry iteration, Conn hook).
//!
//! Intentional gaps (documented, not silent): `options.num_ctx` is ignored
//! (context is auto-budgeted server-side), `/api/generate`'s deprecated
//! `context` token array is ignored and echoed back empty, and `keep_alive`
//! is ignored (residency is managed by the registry's LRU).

const std = @import("std");

// ── Request translation ─────────────────────────────────────────────────

/// A translated request: the OpenAI-shaped body to feed the existing /v1
/// handler, plus the outer framing facts the Sink needs.
pub const Translated = struct {
    /// OpenAI-shaped JSON body (owned). For chat/generate this always has
    /// `"stream":true` + `stream_options.include_usage` — the Sink is the
    /// single response parser regardless of what the outer client wanted.
    body: []u8,
    /// Model name echoed back in every NDJSON line (owned).
    model: []u8,
    /// Whether the OUTER client gets NDJSON lines (true) or one JSON object.
    wants_stream: bool,
    /// generate-only: `raw:true` routes to /v1/completions instead of chat.
    raw: bool = false,

    pub fn deinit(self: *Translated, allocator: std.mem.Allocator) void {
        allocator.free(self.body);
        allocator.free(self.model);
    }
};

pub const TranslateError = error{ InvalidRequest, OutOfMemory };

/// /api/chat → /v1/chat/completions body.
pub fn translateChatRequest(allocator: std.mem.Allocator, ollama_body: []const u8) TranslateError!Translated {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, ollama_body, .{}) catch return error.InvalidRequest;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidRequest;
    const root = parsed.value.object;

    const messages_val = root.get("messages") orelse return error.InvalidRequest;
    if (messages_val != .array) return error.InvalidRequest;

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;

    w.writeAll("{") catch return error.OutOfMemory;
    writeModelField(w, root) catch return error.OutOfMemory;
    w.writeAll(",\"stream\":true,\"stream_options\":{\"include_usage\":true}") catch return error.OutOfMemory;
    writeMessagesField(allocator, w, messages_val) catch return error.OutOfMemory;
    if (root.get("tools")) |tools| {
        if (tools == .array and tools.array.items.len > 0) {
            w.writeAll(",\"tools\":") catch return error.OutOfMemory;
            writeJsonValue(w, tools) catch return error.OutOfMemory;
        }
    }
    writeCommonFields(w, root) catch return error.OutOfMemory;
    w.writeAll("}") catch return error.OutOfMemory;

    return .{
        .body = allocator.dupe(u8, out.written()) catch return error.OutOfMemory,
        .model = allocator.dupe(u8, modelNameOf(root)) catch return error.OutOfMemory,
        .wants_stream = wantsStream(root),
    };
}

/// /api/generate → /v1/chat/completions body (template applied), or
/// /v1/completions when `raw:true`.
pub fn translateGenerateRequest(allocator: std.mem.Allocator, ollama_body: []const u8) TranslateError!Translated {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, ollama_body, .{}) catch return error.InvalidRequest;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidRequest;
    const root = parsed.value.object;

    const prompt: []const u8 = if (root.get("prompt")) |p| (if (p == .string) p.string else return error.InvalidRequest) else "";
    const raw = if (root.get("raw")) |r| (r == .bool and r.bool) else false;

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;

    w.writeAll("{") catch return error.OutOfMemory;
    writeModelField(w, root) catch return error.OutOfMemory;
    w.writeAll(",\"stream\":true,\"stream_options\":{\"include_usage\":true}") catch return error.OutOfMemory;
    if (raw) {
        w.writeAll(",\"prompt\":") catch return error.OutOfMemory;
        writeJsonString(w, prompt) catch return error.OutOfMemory;
        if (root.get("suffix")) |sv| {
            if (sv == .string and sv.string.len > 0) {
                w.writeAll(",\"suffix\":") catch return error.OutOfMemory;
                writeJsonString(w, sv.string) catch return error.OutOfMemory;
            }
        }
    } else {
        // system? + single user message (carrying any images).
        w.writeAll(",\"messages\":[") catch return error.OutOfMemory;
        if (root.get("system")) |sysv| {
            if (sysv == .string and sysv.string.len > 0) {
                w.writeAll("{\"role\":\"system\",\"content\":") catch return error.OutOfMemory;
                writeJsonString(w, sysv.string) catch return error.OutOfMemory;
                w.writeAll("},") catch return error.OutOfMemory;
            }
        }
        w.writeAll("{\"role\":\"user\",") catch return error.OutOfMemory;
        const images = imagesOf(root);
        writeContentWithImages(w, prompt, images) catch return error.OutOfMemory;
        w.writeAll("}]") catch return error.OutOfMemory;
    }
    writeCommonFields(w, root) catch return error.OutOfMemory;
    w.writeAll("}") catch return error.OutOfMemory;

    return .{
        .body = allocator.dupe(u8, out.written()) catch return error.OutOfMemory,
        .model = allocator.dupe(u8, modelNameOf(root)) catch return error.OutOfMemory,
        .wants_stream = wantsStream(root),
        .raw = raw,
    };
}

/// /api/embed (`input`: string|array) and legacy /api/embeddings
/// (`prompt`: string) → /v1/embeddings body. Input is normalized to an
/// array so batching semantics are uniform.
pub fn translateEmbedRequest(allocator: std.mem.Allocator, ollama_body: []const u8, legacy: bool) TranslateError!Translated {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, ollama_body, .{}) catch return error.InvalidRequest;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidRequest;
    const root = parsed.value.object;

    const input_val = if (legacy) root.get("prompt") else (root.get("input") orelse root.get("prompt"));
    const iv = input_val orelse return error.InvalidRequest;

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;

    w.writeAll("{") catch return error.OutOfMemory;
    writeModelField(w, root) catch return error.OutOfMemory;
    w.writeAll(",\"input\":") catch return error.OutOfMemory;
    switch (iv) {
        .string => |s| {
            w.writeAll("[") catch return error.OutOfMemory;
            writeJsonString(w, s) catch return error.OutOfMemory;
            w.writeAll("]") catch return error.OutOfMemory;
        },
        .array => writeJsonValue(w, iv) catch return error.OutOfMemory,
        else => return error.InvalidRequest,
    }
    w.writeAll("}") catch return error.OutOfMemory;

    return .{
        .body = allocator.dupe(u8, out.written()) catch return error.OutOfMemory,
        .model = allocator.dupe(u8, modelNameOf(root)) catch return error.OutOfMemory,
        .wants_stream = false,
    };
}

fn modelNameOf(root: std.json.ObjectMap) []const u8 {
    if (root.get("model")) |m| {
        if (m == .string and m.string.len > 0) return m.string;
    }
    return "mlx-serve";
}

fn wantsStream(root: std.json.ObjectMap) bool {
    // Ollama defaults to streaming — the opposite of OpenAI.
    if (root.get("stream")) |s| return s == .bool and s.bool;
    return true;
}

fn imagesOf(root: std.json.ObjectMap) ?std.json.Value {
    if (root.get("images")) |iv| {
        if (iv == .array and iv.array.items.len > 0) return iv;
    }
    return null;
}

fn writeModelField(w: *std.Io.Writer, root: std.json.ObjectMap) !void {
    try w.writeAll("\"model\":");
    try writeJsonString(w, modelNameOf(root));
}

/// `content` + optional Ollama base64 `images` → OpenAI content (plain
/// string, or a content-block array with data-URL image_url entries).
fn writeContentWithImages(w: *std.Io.Writer, content: []const u8, images: ?std.json.Value) !void {
    if (images) |iv| {
        try w.writeAll("\"content\":[{\"type\":\"text\",\"text\":");
        try writeJsonString(w, content);
        try w.writeAll("}");
        for (iv.array.items) |img| {
            if (img != .string) continue;
            try w.writeAll(",{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,");
            // Escape defensively — a hostile "b64" string must not break
            // out of the JSON string (same class as the control-byte gotcha).
            try writeJsonStringBody(w, img.string);
            try w.writeAll("\"}}");
        }
        try w.writeAll("]");
    } else {
        try w.writeAll("\"content\":");
        try writeJsonString(w, content);
    }
}

fn writeMessagesField(allocator: std.mem.Allocator, w: *std.Io.Writer, messages_val: std.json.Value) !void {
    try w.writeAll(",\"messages\":[");
    var first = true;
    for (messages_val.array.items) |m| {
        if (m != .object) continue;
        const obj = m.object;
        const role: []const u8 = if (obj.get("role")) |r| (if (r == .string) r.string else "user") else "user";
        if (!first) try w.writeAll(",");
        first = false;
        try w.writeAll("{\"role\":");
        try writeJsonString(w, role);
        try w.writeAll(",");
        const content: []const u8 = if (obj.get("content")) |cv| (if (cv == .string) cv.string else "") else "";
        try writeContentWithImages(w, content, blk: {
            if (obj.get("images")) |iv| {
                if (iv == .array and iv.array.items.len > 0) break :blk iv;
            }
            break :blk null;
        });
        if (std.mem.eql(u8, role, "assistant")) {
            if (obj.get("tool_calls")) |tcv| {
                if (tcv == .array and tcv.array.items.len > 0) {
                    try w.writeAll(",\"tool_calls\":[");
                    var tc_first = true;
                    for (tcv.array.items, 0..) |tc, idx| {
                        if (tc != .object) continue;
                        const func = tc.object.get("function") orelse continue;
                        if (func != .object) continue;
                        const name: []const u8 = if (func.object.get("name")) |n| (if (n == .string) n.string else "") else "";
                        if (!tc_first) try w.writeAll(",");
                        tc_first = false;
                        try w.writeAll("{\"id\":");
                        if (tc.object.get("id")) |idv| {
                            if (idv == .string and idv.string.len > 0) {
                                try writeJsonString(w, idv.string);
                            } else {
                                try w.print("\"call_{d}\"", .{idx});
                            }
                        } else {
                            try w.print("\"call_{d}\"", .{idx});
                        }
                        try w.writeAll(",\"type\":\"function\",\"function\":{\"name\":");
                        try writeJsonString(w, name);
                        try w.writeAll(",\"arguments\":");
                        // Ollama stores arguments as an OBJECT; OpenAI wants
                        // a JSON STRING. Stringify then escape.
                        if (func.object.get("arguments")) |av| {
                            switch (av) {
                                .string => try writeJsonString(w, av.string),
                                else => {
                                    const rendered = try stringifyValue(allocator, av);
                                    defer allocator.free(rendered);
                                    try writeJsonString(w, rendered);
                                },
                            }
                        } else {
                            try w.writeAll("\"{}\"");
                        }
                        try w.writeAll("}}");
                    }
                    try w.writeAll("]");
                }
            }
        }
        try w.writeAll("}");
    }
    try w.writeAll("]");
}

/// Fields shared by chat + generate: sampling options, stop, format, think.
fn writeCommonFields(w: *std.Io.Writer, root: std.json.ObjectMap) !void {
    if (root.get("options")) |ov| {
        if (ov == .object) {
            const opts = ov.object;
            if (opts.get("num_predict")) |v| {
                if (v == .integer and v.integer > 0) try w.print(",\"max_tokens\":{d}", .{v.integer});
            }
            try passNumberOpt(w, opts, "temperature", "temperature");
            try passNumberOpt(w, opts, "top_p", "top_p");
            try passNumberOpt(w, opts, "top_k", "top_k");
            try passNumberOpt(w, opts, "repeat_penalty", "repeat_penalty");
            try passNumberOpt(w, opts, "presence_penalty", "presence_penalty");
            try passNumberOpt(w, opts, "frequency_penalty", "frequency_penalty");
            try passNumberOpt(w, opts, "seed", "seed");
            if (opts.get("stop")) |sv| {
                switch (sv) {
                    .string, .array => {
                        try w.writeAll(",\"stop\":");
                        try writeJsonValue(w, sv);
                    },
                    else => {},
                }
            }
            // num_ctx deliberately ignored — context is auto-budgeted.
        }
    }
    if (root.get("format")) |fv| {
        switch (fv) {
            .string => |s| {
                if (std.mem.eql(u8, s, "json")) {
                    try w.writeAll(",\"response_format\":{\"type\":\"json_object\"}");
                }
            },
            .object => {
                try w.writeAll(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"format\",\"schema\":");
                try writeJsonValue(w, fv);
                try w.writeAll("}}");
            },
            else => {},
        }
    }
    if (root.get("think")) |tv| {
        const on = switch (tv) {
            .bool => |b| b,
            // "low" / "medium" / "high" effort levels all mean "on".
            .string => |s| !std.mem.eql(u8, s, "false"),
            else => false,
        };
        if (on) try w.writeAll(",\"enable_thinking\":true");
    }
}

fn passNumberOpt(w: *std.Io.Writer, opts: std.json.ObjectMap, ollama_key: []const u8, openai_key: []const u8) !void {
    if (opts.get(ollama_key)) |v| {
        switch (v) {
            .integer, .float => {
                try w.writeAll(",\"");
                try w.writeAll(openai_key);
                try w.writeAll("\":");
                try writeJsonValue(w, v);
            },
            else => {},
        }
    }
}

// ── JSON writing helpers ────────────────────────────────────────────────

/// Escape + quote. `\u`-escapes ALL control bytes (see the hand-rolled-JSON
/// control-byte gotcha in CLAUDE.md — one raw byte < 0x20 breaks nlohmann).
pub fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    try writeJsonStringBody(w, s);
    try w.writeByte('"');
}

fn writeJsonStringBody(w: *std.Io.Writer, s: []const u8) !void {
    for (s) |ch| {
        switch (ch) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try w.print("\\u{x:0>4}", .{ch});
                } else {
                    try w.writeByte(ch);
                }
            },
        }
    }
}

fn writeJsonValue(w: *std.Io.Writer, v: std.json.Value) !void {
    var jws: std.json.Stringify = .{ .writer = w, .options = .{} };
    try v.jsonStringify(&jws);
}

fn stringifyValue(allocator: std.mem.Allocator, v: std.json.Value) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try writeJsonValue(&out.writer, v);
    return allocator.dupe(u8, out.written());
}

// ── Time formatting ─────────────────────────────────────────────────────

/// Fixed-width RFC3339 UTC with millisecond precision, e.g.
/// "2026-07-01T12:34:56.789Z". Buffer must hold >= 24 bytes.
pub fn formatIso8601(buf: []u8, ms: i64) []const u8 {
    const clamped: i64 = if (ms < 0) 0 else ms;
    const secs: u64 = @intCast(@divFloor(clamped, 1000));
    const msec: u64 = @intCast(@mod(clamped, 1000));
    const es = std.time.epoch.EpochSeconds{ .secs = secs };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}Z", .{
        yd.year,
        md.month.numeric(),
        @as(u32, md.day_index) + 1,
        ds.getHoursIntoDay(),
        ds.getMinutesIntoHour(),
        ds.getSecondsIntoMinute(),
        msec,
    }) catch "1970-01-01T00:00:00.000Z";
}

// ── Name resolution ─────────────────────────────────────────────────────

/// Strip an Ollama-style ":tag" suffix ("gemma:latest" → "gemma").
/// A ':' followed by '/' is not a tag separator.
pub fn stripTag(name: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, name, ':')) |i| {
        if (std.mem.indexOfScalarPos(u8, name, i, '/') == null) return name[0..i];
    }
    return name;
}

/// Resolve an Ollama-style model name against registry ids. Order:
/// exact → case-insensitive exact → unique case-insensitive basename
/// (id "org/name" matches "name") → unique case-insensitive substring.
/// Ambiguity returns null (caller falls back to the default model).
pub fn resolveName(candidate: []const u8, ids: []const []const u8) ?usize {
    const base = stripTag(candidate);
    if (base.len == 0) return null;
    for (ids, 0..) |id, i| {
        if (std.mem.eql(u8, id, base)) return i;
    }
    for (ids, 0..) |id, i| {
        if (std.ascii.eqlIgnoreCase(id, base)) return i;
    }
    {
        var found: ?usize = null;
        for (ids, 0..) |id, i| {
            const bn = if (std.mem.lastIndexOfScalar(u8, id, '/')) |p| id[p + 1 ..] else id;
            if (std.ascii.eqlIgnoreCase(bn, base)) {
                if (found != null) {
                    found = null;
                    break;
                }
                found = i;
            }
        }
        if (found) |i| return i;
    }
    {
        var found: ?usize = null;
        for (ids, 0..) |id, i| {
            if (std.ascii.indexOfIgnoreCase(id, base) != null) {
                if (found != null) return null; // ambiguous
                found = i;
            }
        }
        return found;
    }
}

// ── Listing / metadata renderers ────────────────────────────────────────

pub const TagEntry = struct {
    id: []const u8,
    size_bytes: u64 = 0,
    modified_ms: i64 = 0,
    family: []const u8 = "",
    format: []const u8 = "safetensors",
    param_size: []const u8 = "",
    quant: []const u8 = "",
};

/// GET /api/tags — {"models":[…]}. `name` carries the ":latest" tag Ollama
/// clients expect; `resolveName` strips it on the way back in.
pub fn renderTagsJson(allocator: std.mem.Allocator, entries: []const TagEntry) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"models\":[");
    for (entries, 0..) |e, i| {
        if (i > 0) try w.writeAll(",");
        try writeTagModel(w, e, null);
    }
    try w.writeAll("]}");
    return allocator.dupe(u8, out.written());
}

pub const PsEntry = struct {
    tag: TagEntry,
    resident_bytes: u64 = 0,
};

/// GET /api/ps — running models. `expires_at` is a far-future sentinel:
/// residency is LRU-managed, not TTL-managed.
pub fn renderPsJson(allocator: std.mem.Allocator, entries: []const PsEntry) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"models\":[");
    for (entries, 0..) |e, i| {
        if (i > 0) try w.writeAll(",");
        try writeTagModel(w, e.tag, e.resident_bytes);
    }
    try w.writeAll("]}");
    return allocator.dupe(u8, out.written());
}

fn writeTagModel(w: *std.Io.Writer, e: TagEntry, resident: ?u64) !void {
    var iso_buf: [32]u8 = undefined;
    try w.writeAll("{\"name\":");
    try writeTaggedName(w, e.id);
    try w.writeAll(",\"model\":");
    try writeTaggedName(w, e.id);
    try w.writeAll(",\"modified_at\":");
    try writeJsonString(w, formatIso8601(&iso_buf, e.modified_ms));
    try w.print(",\"size\":{d},\"digest\":\"", .{e.size_bytes});
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(e.id, &digest, .{});
    for (digest) |b| try w.print("{x:0>2}", .{b});
    try w.writeAll("\",\"details\":");
    try writeDetails(w, e);
    if (resident) |r| {
        try w.print(",\"expires_at\":\"9999-01-01T00:00:00.000Z\",\"size_vram\":{d}", .{r});
    }
    try w.writeAll("}");
}

fn writeTaggedName(w: *std.Io.Writer, id: []const u8) !void {
    try w.writeByte('"');
    try writeJsonStringBody(w, id);
    try w.writeAll(":latest\"");
}

fn writeDetails(w: *std.Io.Writer, e: TagEntry) !void {
    try w.writeAll("{\"parent_model\":\"\",\"format\":");
    try writeJsonString(w, e.format);
    try w.writeAll(",\"family\":");
    try writeJsonString(w, e.family);
    try w.writeAll(",\"families\":[");
    try writeJsonString(w, e.family);
    try w.writeAll("],\"parameter_size\":");
    try writeJsonString(w, e.param_size);
    try w.writeAll(",\"quantization_level\":");
    try writeJsonString(w, e.quant);
    try w.writeAll("}");
}

pub const ShowInfo = struct {
    tag: TagEntry,
    context_length: u32 = 0,
    template: []const u8 = "",
    has_chat: bool = true,
    has_tools: bool = false,
    has_vision: bool = false,
    has_thinking: bool = false,
    has_embedding: bool = false,
};

/// POST /api/show — enough surface for Open WebUI + ollama-python
/// (`details`, `model_info`, `capabilities`).
pub fn renderShowJson(allocator: std.mem.Allocator, info: ShowInfo) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"modelfile\":\"\",\"parameters\":\"\",\"template\":");
    try writeJsonString(w, info.template);
    try w.writeAll(",\"details\":");
    try writeDetails(w, info.tag);
    try w.writeAll(",\"model_info\":{\"general.architecture\":");
    try writeJsonString(w, info.tag.family);
    try w.writeAll(",\"general.basename\":");
    try writeJsonString(w, info.tag.id);
    if (info.context_length > 0) {
        try w.writeAll(",\"");
        try writeJsonStringBody(w, info.tag.family);
        try w.print(".context_length\":{d}", .{info.context_length});
    }
    try w.writeAll("},\"capabilities\":[");
    var n: usize = 0;
    const caps = [_]struct { on: bool, name: []const u8 }{
        .{ .on = info.has_chat, .name = "completion" },
        .{ .on = info.has_tools, .name = "tools" },
        .{ .on = info.has_vision, .name = "vision" },
        .{ .on = info.has_thinking, .name = "thinking" },
        .{ .on = info.has_embedding, .name = "embedding" },
    };
    for (caps) |c| {
        if (!c.on) continue;
        if (n > 0) try w.writeAll(",");
        try writeJsonString(w, c.name);
        n += 1;
    }
    try w.writeAll("]}");
    return allocator.dupe(u8, out.written());
}

/// /api/embed + legacy /api/embeddings response from the /v1/embeddings
/// OpenAI body.
pub fn renderEmbedResponse(allocator: std.mem.Allocator, model: []const u8, openai_body: []const u8, legacy: bool, total_duration_ns: u64) ![]u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, openai_body, .{}) catch return error.InvalidRequest;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidRequest;
    const root = parsed.value.object;
    const data = root.get("data") orelse return error.InvalidRequest;
    if (data != .array) return error.InvalidRequest;

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;

    if (legacy) {
        try w.writeAll("{\"embedding\":");
        if (data.array.items.len > 0 and data.array.items[0] == .object) {
            if (data.array.items[0].object.get("embedding")) |emb| {
                try writeJsonValue(w, emb);
            } else try w.writeAll("[]");
        } else try w.writeAll("[]");
        try w.writeAll("}");
    } else {
        try w.writeAll("{\"model\":");
        try writeJsonString(w, model);
        try w.writeAll(",\"embeddings\":[");
        for (data.array.items, 0..) |item, i| {
            if (i > 0) try w.writeAll(",");
            if (item == .object) {
                if (item.object.get("embedding")) |emb| {
                    try writeJsonValue(w, emb);
                    continue;
                }
            }
            try w.writeAll("[]");
        }
        var prompt_tokens: i64 = 0;
        if (root.get("usage")) |u| {
            if (u == .object) {
                if (u.object.get("prompt_tokens")) |pt| {
                    if (pt == .integer) prompt_tokens = pt.integer;
                }
            }
        }
        try w.print("],\"total_duration\":{d},\"load_duration\":0,\"prompt_eval_count\":{d}}}", .{ total_duration_ns, prompt_tokens });
    }
    return allocator.dupe(u8, out.written());
}

// ── Response sink (SSE → NDJSON re-framing) ─────────────────────────────

pub const SinkMode = enum { chat, generate };

pub const SinkOpts = struct {
    mode: SinkMode,
    wants_stream: bool,
    /// Borrowed; must outlive the sink (the Translated struct does).
    model: []const u8,
    out_impl: *anyopaque,
    outFn: *const fn (impl: *anyopaque, data: []const u8) anyerror!void,
    /// Clock, called with `out_impl` (which carries whatever io context the
    /// host needs — the Conn in server glue, a test fixture in tests).
    nowMsFn: *const fn (impl: *anyopaque) i64,
};

/// Installed on `Conn.ollama_sink` while an inner /v1 handler runs; every
/// byte the handler writes lands in `feed`. Re-frames the inner HTTP/SSE
/// response into the Ollama NDJSON (or single-JSON) shape on the real
/// socket via `outFn`. Call `finish()` after the inner handler returns.
pub const Sink = struct {
    allocator: std.mem.Allocator,
    mode: SinkMode,
    wants_stream: bool,
    model: []const u8,
    out_impl: *anyopaque,
    outFn: *const fn (impl: *anyopaque, data: []const u8) anyerror!void,
    nowMsFn: *const fn (impl: *anyopaque) i64,

    buf: std.ArrayList(u8) = .empty,
    headers_done: bool = false,
    inner_is_sse: bool = false,
    /// Status line after "HTTP/1.1 " (owned), e.g. "400 Bad Request".
    inner_status: ?[]u8 = null,
    started: bool = false,
    finish_reason: ?[]u8 = null,
    prompt_tokens: u64 = 0,
    completion_tokens: u64 = 0,
    content: std.ArrayList(u8) = .empty,
    thinking: std.ArrayList(u8) = .empty,
    /// Rendered Ollama tool-call objects, comma-joined (no brackets).
    tool_calls: std.ArrayList(u8) = .empty,
    t_start: i64,
    t_first: ?i64 = null,

    pub fn init(allocator: std.mem.Allocator, opts: SinkOpts) Sink {
        return .{
            .allocator = allocator,
            .mode = opts.mode,
            .wants_stream = opts.wants_stream,
            .model = opts.model,
            .out_impl = opts.out_impl,
            .outFn = opts.outFn,
            .nowMsFn = opts.nowMsFn,
            .t_start = opts.nowMsFn(opts.out_impl),
        };
    }

    pub fn deinit(self: *Sink) void {
        self.buf.deinit(self.allocator);
        self.content.deinit(self.allocator);
        self.thinking.deinit(self.allocator);
        self.tool_calls.deinit(self.allocator);
        if (self.inner_status) |s| self.allocator.free(s);
        if (self.finish_reason) |f| self.allocator.free(f);
    }

    /// Receives every byte the inner /v1 handler writes to the Conn.
    pub fn feed(self: *Sink, bytes: []const u8) anyerror!void {
        try self.buf.appendSlice(self.allocator, bytes);
        try self.drain();
    }

    fn drain(self: *Sink) !void {
        if (!self.headers_done) {
            const he = std.mem.indexOf(u8, self.buf.items, "\r\n\r\n") orelse return;
            const head = self.buf.items[0..he];
            if (std.mem.startsWith(u8, head, "HTTP/1.1 ")) {
                const line_end = std.mem.indexOf(u8, head, "\r\n") orelse head.len;
                self.inner_status = try self.allocator.dupe(u8, head["HTTP/1.1 ".len..line_end]);
            } else {
                self.inner_status = try self.allocator.dupe(u8, "500 Internal Server Error");
            }
            self.inner_is_sse = std.ascii.indexOfIgnoreCase(head, "text/event-stream") != null;
            self.consume(he + 4);
            self.headers_done = true;
        }
        if (!self.inner_is_sse) return; // non-SSE (error body): buffer until finish()
        while (std.mem.indexOf(u8, self.buf.items, "\n\n")) |ee| {
            // Copy the event out so handleEvent can't be invalidated by
            // buffer mutation.
            const event = try self.allocator.dupe(u8, self.buf.items[0..ee]);
            defer self.allocator.free(event);
            self.consume(ee + 2);
            try self.handleEvent(event);
        }
    }

    fn consume(self: *Sink, n: usize) void {
        const rest = self.buf.items.len - n;
        std.mem.copyForwards(u8, self.buf.items[0..rest], self.buf.items[n..]);
        self.buf.shrinkRetainingCapacity(rest);
    }

    fn handleEvent(self: *Sink, event: []const u8) !void {
        var lines = std.mem.splitScalar(u8, event, '\n');
        while (lines.next()) |line_raw| {
            const line = std.mem.trimEnd(u8, line_raw, "\r");
            if (line.len == 0) continue;
            if (line[0] == ':') continue; // SSE comment (keepalive)
            if (!std.mem.startsWith(u8, line, "data: ")) continue;
            const payload = line["data: ".len..];
            if (std.mem.eql(u8, payload, "[DONE]")) continue;
            self.handleChunk(payload) catch continue; // tolerate odd chunks
        }
    }

    fn handleChunk(self: *Sink, payload: []const u8) !void {
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, payload, .{}) catch return;
        defer parsed.deinit();
        if (parsed.value != .object) return;
        const root = parsed.value.object;

        if (root.get("usage")) |u| {
            if (u == .object) {
                if (u.object.get("prompt_tokens")) |pt| {
                    if (pt == .integer and pt.integer >= 0) self.prompt_tokens = @intCast(pt.integer);
                }
                if (u.object.get("completion_tokens")) |ct| {
                    if (ct == .integer and ct.integer >= 0) self.completion_tokens = @intCast(ct.integer);
                }
            }
        }

        const choices = root.get("choices") orelse return;
        if (choices != .array or choices.array.items.len == 0) return;
        if (choices.array.items[0] != .object) return;
        const c0 = choices.array.items[0].object;

        if (c0.get("finish_reason")) |fr| {
            if (fr == .string) {
                if (self.finish_reason) |old| self.allocator.free(old);
                self.finish_reason = try self.allocator.dupe(u8, fr.string);
            }
        }

        var content: []const u8 = "";
        var thinking: []const u8 = "";
        var tool_calls_val: ?std.json.Value = null;
        if (c0.get("delta")) |dv| {
            if (dv == .object) {
                if (dv.object.get("content")) |cv| {
                    if (cv == .string) content = cv.string;
                }
                if (dv.object.get("reasoning_content")) |rv| {
                    if (rv == .string) thinking = rv.string;
                }
                if (dv.object.get("tool_calls")) |tv| {
                    if (tv == .array and tv.array.items.len > 0) tool_calls_val = tv;
                }
            }
        } else if (c0.get("text")) |tv| {
            // /v1/completions chunk shape (generate raw mode).
            if (tv == .string) content = tv.string;
        }

        if (content.len == 0 and thinking.len == 0 and tool_calls_val == null) return;
        if (self.t_first == null) self.t_first = self.nowMsFn(self.out_impl);

        var tc_items: ?[]u8 = null;
        defer if (tc_items) |t| self.allocator.free(t);
        if (tool_calls_val) |tv| {
            tc_items = try self.renderOllamaToolCalls(tv);
        }

        if (self.wants_stream) {
            try self.emitLine(content, thinking, if (tc_items) |t| t else null, false);
        } else {
            try self.content.appendSlice(self.allocator, content);
            try self.thinking.appendSlice(self.allocator, thinking);
            if (tc_items) |t| {
                if (t.len > 0) {
                    if (self.tool_calls.items.len > 0) try self.tool_calls.append(self.allocator, ',');
                    try self.tool_calls.appendSlice(self.allocator, t);
                }
            }
        }
    }

    /// OpenAI tool_calls (arguments as JSON string) → Ollama tool_calls
    /// (arguments as object). Items joined by ',' without brackets.
    fn renderOllamaToolCalls(self: *Sink, tv: std.json.Value) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        const w = &out.writer;
        var first = true;
        for (tv.array.items) |tc| {
            if (tc != .object) continue;
            const func = tc.object.get("function") orelse continue;
            if (func != .object) continue;
            const name: []const u8 = if (func.object.get("name")) |n| (if (n == .string) n.string else "") else "";
            if (name.len == 0) continue;
            const args: []const u8 = if (func.object.get("arguments")) |a| (if (a == .string) a.string else "{}") else "{}";
            if (!first) try w.writeAll(",");
            first = false;
            try w.writeAll("{\"function\":{\"name\":");
            try writeJsonString(w, name);
            try w.writeAll(",\"arguments\":");
            // arguments are guaranteed-valid JSON by the parseToolCalls
            // chokepoint; validate anyway and fall back to a wrapped string.
            if (std.json.validate(self.allocator, args) catch false) {
                try w.writeAll(args);
            } else {
                try w.writeAll("{\"raw\":");
                try writeJsonString(w, args);
                try w.writeAll("}");
            }
            try w.writeAll("}}");
        }
        return self.allocator.dupe(u8, out.written());
    }

    fn ensureStreamHeaders(self: *Sink) !void {
        if (self.started) return;
        self.started = true;
        try self.outFn(self.out_impl,
            "HTTP/1.1 200 OK\r\n" ++
            "Content-Type: application/x-ndjson\r\n" ++
            "Cache-Control: no-cache\r\n" ++
            "Connection: close\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
            "\r\n");
    }

    fn emitLine(self: *Sink, content: []const u8, thinking: []const u8, tool_items: ?[]const u8, final: bool) !void {
        try self.ensureStreamHeaders();
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        const w = &out.writer;
        try self.writeBodyObject(w, content, thinking, tool_items, final);
        try w.writeAll("\n");
        try self.outFn(self.out_impl, out.written());
    }

    fn writeBodyObject(self: *Sink, w: *std.Io.Writer, content: []const u8, thinking: []const u8, tool_items: ?[]const u8, final: bool) !void {
        var iso_buf: [32]u8 = undefined;
        try w.writeAll("{\"model\":");
        try writeJsonString(w, self.model);
        try w.writeAll(",\"created_at\":");
        try writeJsonString(w, formatIso8601(&iso_buf, self.nowMsFn(self.out_impl)));
        switch (self.mode) {
            .chat => {
                try w.writeAll(",\"message\":{\"role\":\"assistant\",\"content\":");
                try writeJsonString(w, content);
                if (thinking.len > 0) {
                    try w.writeAll(",\"thinking\":");
                    try writeJsonString(w, thinking);
                }
                if (tool_items) |t| {
                    if (t.len > 0) {
                        try w.writeAll(",\"tool_calls\":[");
                        try w.writeAll(t);
                        try w.writeAll("]");
                    }
                }
                try w.writeAll("}");
            },
            .generate => {
                try w.writeAll(",\"response\":");
                try writeJsonString(w, content);
                if (thinking.len > 0) {
                    try w.writeAll(",\"thinking\":");
                    try writeJsonString(w, thinking);
                }
            },
        }
        if (final) {
            const t_end = self.nowMsFn(self.out_impl);
            const t_first = self.t_first orelse t_end;
            const total_ns: u64 = msToNs(t_end - self.t_start);
            const prompt_ns: u64 = msToNs(t_first - self.t_start);
            const eval_ns: u64 = msToNs(t_end - t_first);
            try w.writeAll(",\"done\":true,\"done_reason\":");
            try writeJsonString(w, doneReason(self.finish_reason));
            if (self.mode == .generate) try w.writeAll(",\"context\":[]");
            try w.print(",\"total_duration\":{d},\"load_duration\":0,\"prompt_eval_count\":{d},\"prompt_eval_duration\":{d},\"eval_count\":{d},\"eval_duration\":{d}", .{
                total_ns, self.prompt_tokens, prompt_ns, self.completion_tokens, eval_ns,
            });
            try w.writeAll("}");
        } else {
            try w.writeAll(",\"done\":false}");
        }
    }

    /// Call after the inner handler returns; writes whatever the outer
    /// client is still owed (final NDJSON line, aggregate JSON, or a
    /// translated error).
    pub fn finish(self: *Sink) anyerror!void {
        if (!self.headers_done) {
            try self.sendError("500 Internal Server Error", "empty upstream response");
            return;
        }
        const status = self.inner_status orelse "500 Internal Server Error";
        if (!std.mem.startsWith(u8, status, "200")) {
            // buf holds the (non-SSE) error body: {"error":{"message":…}}
            const msg = self.extractErrorMessage() orelse "upstream error";
            try self.sendError(status, msg);
            return;
        }
        if (self.wants_stream) {
            try self.emitLine("", "", null, true);
            return;
        }
        // Aggregate single-JSON response.
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        const w = &out.writer;
        try self.writeBodyObject(w, self.content.items, self.thinking.items, if (self.tool_calls.items.len > 0) self.tool_calls.items else null, true);
        try self.sendHttpJson("200 OK", out.written());
    }

    /// Embed variant of `finish`: the inner /v1/embeddings handler writes a
    /// plain JSON response (no SSE); the buffered body is re-rendered into
    /// the Ollama embed shape. `legacy` selects the /api/embeddings
    /// single-vector form.
    pub fn finishEmbed(self: *Sink, legacy: bool) anyerror!void {
        if (!self.headers_done) {
            try self.sendError("500 Internal Server Error", "empty upstream response");
            return;
        }
        const status = self.inner_status orelse "500 Internal Server Error";
        if (!std.mem.startsWith(u8, status, "200")) {
            const msg = self.extractErrorMessage() orelse "upstream error";
            try self.sendError(status, msg);
            return;
        }
        const t_end = self.nowMsFn(self.out_impl);
        const rendered = renderEmbedResponse(self.allocator, self.model, self.buf.items, legacy, msToNs(t_end - self.t_start)) catch {
            try self.sendError("500 Internal Server Error", "malformed upstream embeddings response");
            return;
        };
        defer self.allocator.free(rendered);
        try self.sendHttpJson("200 OK", rendered);
    }

    fn extractErrorMessage(self: *Sink) ?[]const u8 {
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, self.buf.items, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        const err = parsed.value.object.get("error") orelse return null;
        if (err != .object) return null;
        const msg = err.object.get("message") orelse return null;
        if (msg != .string) return null;
        // Copy into buf-independent storage: reuse content buffer (unused
        // on error paths) so the slice stays valid after parsed.deinit().
        self.content.clearRetainingCapacity();
        self.content.appendSlice(self.allocator, msg.string) catch return null;
        return self.content.items;
    }

    fn sendError(self: *Sink, status: []const u8, message: []const u8) !void {
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        const w = &out.writer;
        try w.writeAll("{\"error\":");
        try writeJsonString(w, message);
        try w.writeAll("}");
        try self.sendHttpJson(status, out.written());
    }

    fn sendHttpJson(self: *Sink, status: []const u8, body: []const u8) !void {
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        const w = &out.writer;
        try w.writeAll("HTTP/1.1 ");
        try w.writeAll(status);
        try w.writeAll("\r\nContent-Type: application/json\r\n");
        try w.print("Content-Length: {d}\r\n", .{body.len});
        try w.writeAll("Connection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n");
        try w.writeAll(body);
        try self.outFn(self.out_impl, out.written());
    }
};

fn msToNs(ms: i64) u64 {
    if (ms <= 0) return 0;
    return @as(u64, @intCast(ms)) * std.time.ns_per_ms;
}

pub fn doneReason(finish_reason: ?[]const u8) []const u8 {
    const fr = finish_reason orelse return "stop";
    if (std.mem.eql(u8, fr, "length")) return "length";
    // "tool_calls" and everything else Ollama reports as a plain stop.
    return "stop";
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

fn parseTestJson(allocator: std.mem.Allocator, body: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, body, .{});
}

test "ollama: chat request translates to OpenAI shape with stream forced" {
    const allocator = testing.allocator;
    var tr = try translateChatRequest(allocator,
        \\{"model":"gemma:latest","messages":[{"role":"user","content":"hi"}]}
    );
    defer tr.deinit(allocator);

    try testing.expect(tr.wants_stream);
    try testing.expectEqualStrings("gemma:latest", tr.model);

    var parsed = try parseTestJson(allocator, tr.body);
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expect(root.get("stream").?.bool);
    try testing.expect(root.get("stream_options").?.object.get("include_usage").?.bool);
    try testing.expectEqualStrings("gemma:latest", root.get("model").?.string);
    const msgs = root.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 1), msgs.len);
    try testing.expectEqualStrings("user", msgs[0].object.get("role").?.string);
    try testing.expectEqualStrings("hi", msgs[0].object.get("content").?.string);
}

test "ollama: chat request honors stream:false and maps options" {
    const allocator = testing.allocator;
    var tr = try translateChatRequest(allocator,
        \\{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}],
        \\ "options":{"num_predict":64,"temperature":0.5,"top_p":0.25,"top_k":40,"seed":7,
        \\            "repeat_penalty":1.5,"stop":["END"],"num_ctx":4096}}
    );
    defer tr.deinit(allocator);

    try testing.expect(!tr.wants_stream);
    var parsed = try parseTestJson(allocator, tr.body);
    defer parsed.deinit();
    const root = parsed.value.object;
    // Inner body ALWAYS streams; wants_stream only affects outer framing.
    try testing.expect(root.get("stream").?.bool);
    try testing.expectEqual(@as(i64, 64), root.get("max_tokens").?.integer);
    try testing.expectEqual(@as(f64, 0.5), root.get("temperature").?.float);
    try testing.expectEqual(@as(f64, 0.25), root.get("top_p").?.float);
    try testing.expectEqual(@as(i64, 40), root.get("top_k").?.integer);
    try testing.expectEqual(@as(i64, 7), root.get("seed").?.integer);
    try testing.expectEqual(@as(f64, 1.5), root.get("repeat_penalty").?.float);
    try testing.expectEqualStrings("END", root.get("stop").?.array.items[0].string);
    // num_ctx is deliberately dropped.
    try testing.expect(root.get("num_ctx") == null);
}

test "ollama: chat think, format and tools translate" {
    const allocator = testing.allocator;
    var tr = try translateChatRequest(allocator,
        \\{"model":"m","think":true,"format":"json",
        \\ "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],
        \\ "messages":[{"role":"user","content":"hi"}]}
    );
    defer tr.deinit(allocator);

    var parsed = try parseTestJson(allocator, tr.body);
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expect(root.get("enable_thinking").?.bool);
    try testing.expectEqualStrings("json_object", root.get("response_format").?.object.get("type").?.string);
    const tools = root.get("tools").?.array.items;
    try testing.expectEqualStrings("get_weather", tools[0].object.get("function").?.object.get("name").?.string);

    // Schema-object format → json_schema response_format.
    var tr2 = try translateChatRequest(allocator,
        \\{"model":"m","format":{"type":"object","properties":{"x":{"type":"integer"}}},
        \\ "messages":[{"role":"user","content":"hi"}]}
    );
    defer tr2.deinit(allocator);
    var parsed2 = try parseTestJson(allocator, tr2.body);
    defer parsed2.deinit();
    const rf = parsed2.value.object.get("response_format").?.object;
    try testing.expectEqualStrings("json_schema", rf.get("type").?.string);
    const schema = rf.get("json_schema").?.object.get("schema").?.object;
    try testing.expectEqualStrings("object", schema.get("type").?.string);
}

test "ollama: chat history tool_calls, images and tool role translate" {
    const allocator = testing.allocator;
    var tr = try translateChatRequest(allocator,
        \\{"model":"m","messages":[
        \\  {"role":"user","content":"pic","images":["QUJD"]},
        \\  {"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},
        \\  {"role":"tool","content":"sunny","tool_name":"get_weather"}
        \\]}
    );
    defer tr.deinit(allocator);

    var parsed = try parseTestJson(allocator, tr.body);
    defer parsed.deinit();
    const msgs = parsed.value.object.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 3), msgs.len);

    // images → content-block array with a data-URL image_url entry.
    const blocks = msgs[0].object.get("content").?.array.items;
    try testing.expectEqualStrings("text", blocks[0].object.get("type").?.string);
    try testing.expectEqualStrings("pic", blocks[0].object.get("text").?.string);
    const url = blocks[1].object.get("image_url").?.object.get("url").?.string;
    try testing.expectEqualStrings("data:image/jpeg;base64,QUJD", url);

    // Ollama object arguments → OpenAI JSON-string arguments.
    const tc = msgs[1].object.get("tool_calls").?.array.items[0].object;
    try testing.expectEqualStrings("function", tc.get("type").?.string);
    const func = tc.get("function").?.object;
    try testing.expectEqualStrings("get_weather", func.get("name").?.string);
    const args_str = func.get("arguments").?.string;
    var args_parsed = try parseTestJson(allocator, args_str);
    defer args_parsed.deinit();
    try testing.expectEqualStrings("Paris", args_parsed.value.object.get("city").?.string);

    // tool role passes through.
    try testing.expectEqualStrings("tool", msgs[2].object.get("role").?.string);
    try testing.expectEqualStrings("sunny", msgs[2].object.get("content").?.string);
}

test "ollama: chat content control bytes stay valid JSON" {
    const allocator = testing.allocator;
    var tr = try translateChatRequest(allocator, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"a\\u001b[0mb\\nc\"}]}");
    defer tr.deinit(allocator);
    var parsed = try parseTestJson(allocator, tr.body);
    defer parsed.deinit();
    const content = parsed.value.object.get("messages").?.array.items[0].object.get("content").?.string;
    try testing.expectEqualStrings("a\x1b[0mb\nc", content);
}

test "ollama: generate request maps to chat unless raw" {
    const allocator = testing.allocator;
    var tr = try translateGenerateRequest(allocator,
        \\{"model":"m","prompt":"P","system":"S"}
    );
    defer tr.deinit(allocator);
    try testing.expect(!tr.raw);
    var parsed = try parseTestJson(allocator, tr.body);
    defer parsed.deinit();
    const msgs = parsed.value.object.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 2), msgs.len);
    try testing.expectEqualStrings("system", msgs[0].object.get("role").?.string);
    try testing.expectEqualStrings("S", msgs[0].object.get("content").?.string);
    try testing.expectEqualStrings("user", msgs[1].object.get("role").?.string);
    try testing.expectEqualStrings("P", msgs[1].object.get("content").?.string);

    var tr2 = try translateGenerateRequest(allocator,
        \\{"model":"m","prompt":"P","raw":true}
    );
    defer tr2.deinit(allocator);
    try testing.expect(tr2.raw);
    var parsed2 = try parseTestJson(allocator, tr2.body);
    defer parsed2.deinit();
    try testing.expectEqualStrings("P", parsed2.value.object.get("prompt").?.string);
    try testing.expect(parsed2.value.object.get("messages") == null);
}

test "ollama: embed request translation normalizes input" {
    const allocator = testing.allocator;
    var tr = try translateEmbedRequest(allocator,
        \\{"model":"m","input":"hello"}
    , false);
    defer tr.deinit(allocator);
    var parsed = try parseTestJson(allocator, tr.body);
    defer parsed.deinit();
    try testing.expectEqualStrings("hello", parsed.value.object.get("input").?.array.items[0].string);

    var tr2 = try translateEmbedRequest(allocator,
        \\{"model":"m","prompt":"legacy"}
    , true);
    defer tr2.deinit(allocator);
    var parsed2 = try parseTestJson(allocator, tr2.body);
    defer parsed2.deinit();
    try testing.expectEqualStrings("legacy", parsed2.value.object.get("input").?.array.items[0].string);
}

test "ollama: embed response render (new + legacy)" {
    const allocator = testing.allocator;
    const openai_body =
        \\{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.5,-0.25]}],"usage":{"prompt_tokens":3,"total_tokens":3}}
    ;
    const rendered = try renderEmbedResponse(allocator, "m", openai_body, false, 42);
    defer allocator.free(rendered);
    var parsed = try parseTestJson(allocator, rendered);
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expectEqualStrings("m", root.get("model").?.string);
    const emb = root.get("embeddings").?.array.items[0].array.items;
    try testing.expectEqual(@as(f64, 0.5), emb[0].float);
    try testing.expectEqual(@as(i64, 3), root.get("prompt_eval_count").?.integer);

    const legacy = try renderEmbedResponse(allocator, "m", openai_body, true, 42);
    defer allocator.free(legacy);
    var lp = try parseTestJson(allocator, legacy);
    defer lp.deinit();
    try testing.expectEqual(@as(f64, -0.25), lp.value.object.get("embedding").?.array.items[1].float);
}

// ── Sink test harness ──────────────────────────────────────────────────

const TestOut = struct {
    allocator: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,

    fn write(impl: *anyopaque, data: []const u8) anyerror!void {
        const self: *TestOut = @ptrCast(@alignCast(impl));
        try self.buf.appendSlice(self.allocator, data);
    }

    fn deinit(self: *TestOut) void {
        self.buf.deinit(self.allocator);
    }

    /// Body after the outer HTTP headers.
    fn body(self: *TestOut) []const u8 {
        const he = std.mem.indexOf(u8, self.buf.items, "\r\n\r\n") orelse return "";
        return self.buf.items[he + 4 ..];
    }
};

var test_clock_ms: i64 = 0;
fn testNow(impl: *anyopaque) i64 {
    _ = impl;
    test_clock_ms += 100;
    return test_clock_ms;
}

const SSE_HEADERS = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n";

fn makeTestSink(out: *TestOut, mode: SinkMode, wants_stream: bool) Sink {
    test_clock_ms = 0;
    return Sink.init(out.allocator, .{
        .mode = mode,
        .wants_stream = wants_stream,
        .model = "m",
        .out_impl = out,
        .outFn = &TestOut.write,
        .nowMsFn = &testNow,
    });
}

fn feedStandardChatStream(sink: *Sink) !void {
    try sink.feed(SSE_HEADERS);
    try sink.feed("data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n");
    try sink.feed(": keepalive\n\n");
    try sink.feed("data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n");
    try sink.feed("data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n");
    try sink.feed("data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
    try sink.feed("data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5,\"total_tokens\":8}}\n\n");
    try sink.feed("data: [DONE]\n\n");
}

fn ndjsonLines(allocator: std.mem.Allocator, body_text: []const u8) !std.ArrayList([]const u8) {
    var lines = std.ArrayList([]const u8).empty;
    errdefer lines.deinit(allocator);
    var it = std.mem.splitScalar(u8, body_text, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        try lines.append(allocator, line);
    }
    return lines;
}

test "ollama: sink streams chat NDJSON" {
    const allocator = testing.allocator;
    var out = TestOut{ .allocator = allocator };
    defer out.deinit();
    var sink = makeTestSink(&out, .chat, true);
    defer sink.deinit();

    try feedStandardChatStream(&sink);
    try sink.finish();

    try testing.expect(std.mem.startsWith(u8, out.buf.items, "HTTP/1.1 200 OK"));
    try testing.expect(std.mem.indexOf(u8, out.buf.items, "application/x-ndjson") != null);

    var lines = try ndjsonLines(allocator, out.body());
    defer lines.deinit(allocator);
    try testing.expectEqual(@as(usize, 3), lines.items.len);

    var first = try parseTestJson(allocator, lines.items[0]);
    defer first.deinit();
    try testing.expectEqualStrings("m", first.value.object.get("model").?.string);
    try testing.expect(!first.value.object.get("done").?.bool);
    const msg1 = first.value.object.get("message").?.object;
    try testing.expectEqualStrings("assistant", msg1.get("role").?.string);
    try testing.expectEqualStrings("Hel", msg1.get("content").?.string);
    try testing.expect(first.value.object.get("created_at").?.string.len >= 20);

    var last = try parseTestJson(allocator, lines.items[lines.items.len - 1]);
    defer last.deinit();
    const lroot = last.value.object;
    try testing.expect(lroot.get("done").?.bool);
    try testing.expectEqualStrings("stop", lroot.get("done_reason").?.string);
    try testing.expectEqual(@as(i64, 3), lroot.get("prompt_eval_count").?.integer);
    try testing.expectEqual(@as(i64, 5), lroot.get("eval_count").?.integer);
    try testing.expect(lroot.get("total_duration").?.integer > 0);
}

test "ollama: sink aggregates non-stream chat" {
    const allocator = testing.allocator;
    var out = TestOut{ .allocator = allocator };
    defer out.deinit();
    var sink = makeTestSink(&out, .chat, false);
    defer sink.deinit();

    try feedStandardChatStream(&sink);
    try sink.finish();

    try testing.expect(std.mem.indexOf(u8, out.buf.items, "Content-Type: application/json") != null);
    try testing.expect(std.mem.indexOf(u8, out.buf.items, "Content-Length:") != null);
    var parsed = try parseTestJson(allocator, out.body());
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expect(root.get("done").?.bool);
    try testing.expectEqualStrings("Hello", root.get("message").?.object.get("content").?.string);
    try testing.expectEqual(@as(i64, 5), root.get("eval_count").?.integer);
}

test "ollama: sink translates tool calls to objects" {
    const allocator = testing.allocator;
    var out = TestOut{ .allocator = allocator };
    defer out.deinit();
    var sink = makeTestSink(&out, .chat, true);
    defer sink.deinit();

    try sink.feed(SSE_HEADERS);
    try sink.feed("data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"x\\\":1}\"}}]},\"finish_reason\":null}]}\n\n");
    try sink.feed("data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n");
    try sink.feed("data: [DONE]\n\n");
    try sink.finish();

    var lines = try ndjsonLines(allocator, out.body());
    defer lines.deinit(allocator);
    try testing.expectEqual(@as(usize, 2), lines.items.len);

    var first = try parseTestJson(allocator, lines.items[0]);
    defer first.deinit();
    const tcs = first.value.object.get("message").?.object.get("tool_calls").?.array.items;
    const func = tcs[0].object.get("function").?.object;
    try testing.expectEqualStrings("f", func.get("name").?.string);
    // arguments must be an OBJECT, not a string.
    try testing.expectEqual(@as(i64, 1), func.get("arguments").?.object.get("x").?.integer);

    var last = try parseTestJson(allocator, lines.items[1]);
    defer last.deinit();
    try testing.expectEqualStrings("stop", last.value.object.get("done_reason").?.string);
}

test "ollama: sink reasoning_content becomes thinking" {
    const allocator = testing.allocator;
    var out = TestOut{ .allocator = allocator };
    defer out.deinit();
    var sink = makeTestSink(&out, .chat, true);
    defer sink.deinit();

    try sink.feed(SSE_HEADERS);
    try sink.feed("data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hmm\"},\"finish_reason\":null}]}\n\n");
    try sink.feed("data: [DONE]\n\n");
    try sink.finish();

    var lines = try ndjsonLines(allocator, out.body());
    defer lines.deinit(allocator);
    var first = try parseTestJson(allocator, lines.items[0]);
    defer first.deinit();
    try testing.expectEqualStrings("hmm", first.value.object.get("message").?.object.get("thinking").?.string);
}

test "ollama: sink generate mode uses response field and completions chunks" {
    const allocator = testing.allocator;
    var out = TestOut{ .allocator = allocator };
    defer out.deinit();
    var sink = makeTestSink(&out, .generate, true);
    defer sink.deinit();

    try sink.feed(SSE_HEADERS);
    // /v1/completions chunk shape: choices[0].text
    try sink.feed("data: {\"id\":\"cmpl-1\",\"object\":\"text_completion.chunk\",\"choices\":[{\"index\":0,\"text\":\"Hi\",\"finish_reason\":null}]}\n\n");
    try sink.feed("data: {\"id\":\"cmpl-1\",\"object\":\"text_completion.chunk\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":9}}\n\n");
    try sink.feed("data: [DONE]\n\n");
    try sink.finish();

    var lines = try ndjsonLines(allocator, out.body());
    defer lines.deinit(allocator);
    try testing.expectEqual(@as(usize, 2), lines.items.len);
    var first = try parseTestJson(allocator, lines.items[0]);
    defer first.deinit();
    try testing.expectEqualStrings("Hi", first.value.object.get("response").?.string);
    try testing.expect(first.value.object.get("message") == null);

    var last = try parseTestJson(allocator, lines.items[1]);
    defer last.deinit();
    const lroot = last.value.object;
    try testing.expectEqualStrings("length", lroot.get("done_reason").?.string);
    try testing.expectEqual(@as(usize, 0), lroot.get("context").?.array.items.len);
    try testing.expectEqual(@as(i64, 9), lroot.get("eval_count").?.integer);
}

test "ollama: sink maps inner error to ollama error shape" {
    const allocator = testing.allocator;
    var out = TestOut{ .allocator = allocator };
    defer out.deinit();
    var sink = makeTestSink(&out, .chat, true);
    defer sink.deinit();

    try sink.feed("HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 71\r\n\r\n");
    try sink.feed("{\"error\":{\"message\":\"boom\",\"type\":\"invalid_request_error\",\"code\":400}}");
    try sink.finish();

    try testing.expect(std.mem.startsWith(u8, out.buf.items, "HTTP/1.1 400 Bad Request"));
    var parsed = try parseTestJson(allocator, out.body());
    defer parsed.deinit();
    try testing.expectEqualStrings("boom", parsed.value.object.get("error").?.string);
}

test "ollama: sink survives split feeds across header and event boundaries" {
    const allocator = testing.allocator;
    var out = TestOut{ .allocator = allocator };
    defer out.deinit();
    var sink = makeTestSink(&out, .chat, true);
    defer sink.deinit();

    // Split mid-header and mid-event to exercise buffering.
    try sink.feed("HTTP/1.1 200 OK\r\nContent-Type: text/ev");
    try sink.feed("ent-stream\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"con");
    try sink.feed("tent\":\"A\"},\"finish_reason\":null}]}\n");
    try sink.feed("\ndata: [DONE]\n\n");
    try sink.finish();

    var lines = try ndjsonLines(allocator, out.body());
    defer lines.deinit(allocator);
    try testing.expectEqual(@as(usize, 2), lines.items.len);
    var first = try parseTestJson(allocator, lines.items[0]);
    defer first.deinit();
    try testing.expectEqualStrings("A", first.value.object.get("message").?.object.get("content").?.string);
}

test "ollama: sink finishEmbed renders captured embeddings body" {
    const allocator = testing.allocator;
    var out = TestOut{ .allocator = allocator };
    defer out.deinit();
    var sink = makeTestSink(&out, .chat, false);
    defer sink.deinit();

    try sink.feed("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 120\r\n\r\n");
    try sink.feed("{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[1.5]}],\"usage\":{\"prompt_tokens\":2,\"total_tokens\":2}}");
    try sink.finishEmbed(false);

    var parsed = try parseTestJson(allocator, out.body());
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expectEqual(@as(f64, 1.5), root.get("embeddings").?.array.items[0].array.items[0].float);
    try testing.expectEqual(@as(i64, 2), root.get("prompt_eval_count").?.integer);
}

test "ollama: resolveName exact, tag strip, basename, substring, ambiguous" {
    const ids = [_][]const u8{
        "mlx-community/gemma-4-e4b-it-4bit",
        "mlx-community/gemma-4-e2b-it-4bit",
        "ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve",
    };
    try testing.expectEqual(@as(?usize, 0), resolveName("mlx-community/gemma-4-e4b-it-4bit", &ids));
    try testing.expectEqual(@as(?usize, 0), resolveName("mlx-community/gemma-4-e4b-it-4bit:latest", &ids));
    try testing.expectEqual(@as(?usize, 0), resolveName("gemma-4-e4b-it-4bit", &ids));
    try testing.expectEqual(@as(?usize, 0), resolveName("GEMMA-4-E4B-IT-4BIT:latest", &ids));
    try testing.expectEqual(@as(?usize, 2), resolveName("qwen3.6", &ids));
    // "gemma-4" matches two ids → ambiguous → null.
    try testing.expectEqual(@as(?usize, null), resolveName("gemma-4", &ids));
    try testing.expectEqual(@as(?usize, null), resolveName("nonexistent", &ids));
}

test "ollama: renderTagsJson shape" {
    const allocator = testing.allocator;
    const entries = [_]TagEntry{.{
        .id = "mlx-community/gemma-4-e4b-it-4bit",
        .size_bytes = 5_200_000_000,
        .modified_ms = 1_735_689_600_000, // 2025-01-01T00:00:00Z
        .family = "gemma4",
        .format = "safetensors",
        .quant = "4bit",
    }};
    const rendered = try renderTagsJson(allocator, &entries);
    defer allocator.free(rendered);
    var parsed = try parseTestJson(allocator, rendered);
    defer parsed.deinit();
    const m0 = parsed.value.object.get("models").?.array.items[0].object;
    try testing.expectEqualStrings("mlx-community/gemma-4-e4b-it-4bit:latest", m0.get("name").?.string);
    try testing.expectEqualStrings("mlx-community/gemma-4-e4b-it-4bit:latest", m0.get("model").?.string);
    try testing.expectEqual(@as(i64, 5_200_000_000), m0.get("size").?.integer);
    try testing.expectEqualStrings("2025-01-01T00:00:00.000Z", m0.get("modified_at").?.string);
    try testing.expectEqual(@as(usize, 64), m0.get("digest").?.string.len);
    const details = m0.get("details").?.object;
    try testing.expectEqualStrings("gemma4", details.get("family").?.string);
    try testing.expectEqualStrings("gemma4", details.get("families").?.array.items[0].string);
    try testing.expectEqualStrings("4bit", details.get("quantization_level").?.string);
}

test "ollama: renderShowJson capabilities and model_info" {
    const allocator = testing.allocator;
    const rendered = try renderShowJson(allocator, .{
        .tag = .{ .id = "org/model", .family = "qwen3_5" },
        .context_length = 32768,
        .template = "<tmpl>",
        .has_tools = true,
        .has_thinking = true,
    });
    defer allocator.free(rendered);
    var parsed = try parseTestJson(allocator, rendered);
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expectEqualStrings("<tmpl>", root.get("template").?.string);
    const info = root.get("model_info").?.object;
    try testing.expectEqualStrings("qwen3_5", info.get("general.architecture").?.string);
    try testing.expectEqual(@as(i64, 32768), info.get("qwen3_5.context_length").?.integer);
    const caps = root.get("capabilities").?.array.items;
    try testing.expectEqual(@as(usize, 3), caps.len);
    try testing.expectEqualStrings("completion", caps[0].string);
    try testing.expectEqualStrings("tools", caps[1].string);
    try testing.expectEqualStrings("thinking", caps[2].string);
}

test "ollama: renderPsJson includes residency" {
    const allocator = testing.allocator;
    const entries = [_]PsEntry{.{
        .tag = .{ .id = "org/model", .family = "gemma4" },
        .resident_bytes = 1234,
    }};
    const rendered = try renderPsJson(allocator, &entries);
    defer allocator.free(rendered);
    var parsed = try parseTestJson(allocator, rendered);
    defer parsed.deinit();
    const m0 = parsed.value.object.get("models").?.array.items[0].object;
    try testing.expectEqual(@as(i64, 1234), m0.get("size_vram").?.integer);
    try testing.expect(m0.get("expires_at") != null);
}

test "ollama: iso8601 formatting" {
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("1970-01-01T00:00:00.000Z", formatIso8601(&buf, 0));
    try testing.expectEqualStrings("2025-01-01T00:00:00.000Z", formatIso8601(&buf, 1_735_689_600_000));
    try testing.expectEqualStrings("2026-07-01T12:34:56.789Z", formatIso8601(&buf, 1_782_909_296_789));
}

test "ollama: stripTag and doneReason" {
    try testing.expectEqualStrings("gemma", stripTag("gemma:latest"));
    try testing.expectEqualStrings("org/name", stripTag("org/name"));
    try testing.expectEqualStrings("hf.co/org/name", stripTag("hf.co/org/name:q4"));
    try testing.expectEqualStrings("stop", doneReason(null));
    try testing.expectEqualStrings("stop", doneReason("stop"));
    try testing.expectEqualStrings("stop", doneReason("tool_calls"));
    try testing.expectEqualStrings("length", doneReason("length"));
}
