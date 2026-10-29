//! parseModelFromBody; RequestModel, aliases, resolveRequestModelId, query parsing

const std = @import("std");
const model_registry_mod = @import("../registry/registry.zig");
const ModelRegistry = model_registry_mod.ModelRegistry;
const multipart = @import("../http/multipart.zig");
const model_settings_mod = @import("../registry/model_settings.zig");
const testing = std.testing;

const ollamaResolveRegistryId = @import("ollama_routes.zig").ollamaResolveRegistryId;

/// Plan 05 — extract the `"model":"..."` value from a JSON request body
/// without doing a full parse. Returns a borrowed slice into `body` (valid
/// until the body buffer is freed) or null if the field is missing or
/// malformed. Used by `handleConnection` to route each POST to the right
/// LoadedModel before invoking the handler.
///
/// Cheap (single linear scan, ~tens of nanoseconds for typical bodies);
/// the per-handler full JSON parse runs immediately after and rejects any
/// truly malformed body, so robustness here is fine.
pub fn parseModelFromBody(body: []const u8) ?[]const u8 {
    var idx: usize = 0;
    while (idx < body.len) {
        const key_start = std.mem.indexOfPos(u8, body, idx, "\"model\"") orelse return null;
        // Must be JSON object key: preceded by `{` or `,` (skipping whitespace).
        var prev = key_start;
        while (prev > 0) {
            prev -= 1;
            const c = body[prev];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') continue;
            if (c == '{' or c == ',') break;
            // Not a top-level key — keep searching after this match.
            idx = key_start + 1;
            return parseModelFromBody(body[idx..]);
        }
        // Skip past the key + the colon.
        var pos = key_start + "\"model\"".len;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t')) pos += 1;
        if (pos >= body.len or body[pos] != ':') return null;
        pos += 1;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t' or body[pos] == '\n' or body[pos] == '\r')) pos += 1;
        if (pos >= body.len) return null;
        // Accept "string" only — null/numbers/etc fall back to default.
        if (body[pos] != '"') return null;
        pos += 1;
        const val_start = pos;
        while (pos < body.len and body[pos] != '"') {
            if (body[pos] == '\\' and pos + 1 < body.len) pos += 1; // skip escape
            pos += 1;
        }
        if (pos >= body.len) return null;
        return body[val_start..pos];
    }
    return null;
}

pub const RequestModel = union(enum) {
    /// A registered id, or "" / "mlx-serve" (the default), passed through.
    id: []const u8,
    /// Not registered: dispatch serves the default so SDK names like "gpt-4" work.
    unknown_name,
    /// An absolute path no entry lives at.
    unknown_path,
};

/// Model Settings aliases, parsed once and re-read when the file changes.
pub var g_model_aliases_path: [std.fs.max_path_bytes]u8 = undefined;

pub var g_model_aliases: model_settings_mod.Cache = .{ .path = "" };

/// Which model a request's `model` field names: an id, a path, a Model Settings
/// alias, then on `/api/` the untagged alias and Ollama's short name. Dispatch,
/// the LAN gate, load, unload and `/api/show` all read this. The registry is
/// NOT locked here.
pub fn resolveRequestModelId(registry: *ModelRegistry, aliases: *model_settings_mod.Cache, id: []const u8, path: []const u8) RequestModel {
    if (id.len == 0 or std.mem.eql(u8, id, "mlx-serve")) return .{ .id = id };
    if (registry.peek(id)) |e| return .{ .id = e.id };
    // A path is never an SDK marketing name, and a default-model answer to it
    // would serve a pack that failed to load with another model's weights.
    if (std.fs.path.isAbsolute(id)) {
        if (registry.peekByPath(id)) |e| return .{ .id = e.id };
        return .unknown_path;
    }
    const ollama = std.mem.startsWith(u8, path, "/api/");
    const untagged = if (ollama) id[0 .. std.mem.lastIndexOfScalar(u8, id, ':') orelse id.len] else id;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    for ([_][]const u8{ id, untagged }) |n| {
        if (aliases.pathForAlias(registry.io, n, &buf)) |p| if (registry.peekByPath(p)) |e| return .{ .id = e.id };
    }
    if (ollama) if (ollamaResolveRegistryId(registry.io, registry, id)) |r| return .{ .id = r };
    return .unknown_name;
}

/// The requested model id from a request body of EITHER shape.
///
/// Every endpoint we serve takes JSON except `/v1/images/edits`, which is
/// `multipart/form-data` — and model resolution runs BEFORE that route
/// translates the form to JSON, so a JSON-only scan reads no id and the request
/// silently falls back to the default model. Live via Open WebUI:
/// an edit naming a Mage-Flow model ran against the default CHAT model and 400'd
/// "Target model does not support this media modality"; on a headless boot with
/// no default it 503'd "No default model configured" instead. Both existing edit
/// tests boot with `--model <the image model>`, so the default was always the
/// right one and neither could see it.
///
/// Every consumer of the requested id must go through here, so the LAN gate and
/// dispatch can't disagree about which model a request names.
/// `?model=<id>` for bodiless GETs (`/props`), percent-decoded into `buf`.
pub fn queryModel(buf: []u8, raw_path: []const u8) ?[]const u8 {
    return queryDecoded(buf, raw_path, "model");
}

/// A non-empty query parameter, percent-decoded into `buf`.
pub fn queryDecoded(buf: []u8, raw_path: []const u8, name: []const u8) ?[]const u8 {
    const q = raw_path[(std.mem.indexOfScalar(u8, raw_path, '?') orelse return null) + 1 ..];
    return percentDecoded(buf, pairValue(q, '&', name) orelse return null);
}

/// The raw value of `name=` in a `sep`-separated list: a query string, a form body, a Cookie header.
pub fn pairValue(list: []const u8, sep: u8, name: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, list, sep);
    while (it.next()) |raw| {
        const kv = std.mem.trim(u8, raw, " ");
        if (kv.len <= name.len + 1 or !std.mem.startsWith(u8, kv, name) or kv[name.len] != '=') continue;
        return kv[name.len + 1 ..];
    }
    return null;
}

pub fn percentDecoded(buf: []u8, v: []const u8) ?[]const u8 {
    if (v.len > buf.len) return null;
    @memcpy(buf[0..v.len], v);
    return std.Uri.percentDecodeInPlace(buf[0..v.len]);
}

pub fn parseModelFromRequest(body: []const u8, content_type: []const u8) ?[]const u8 {
    if (multipart.boundaryFromContentType(content_type)) |boundary| {
        var it = multipart.Iterator.init(body, boundary) catch return null;
        while (it.next()) |part| {
            if (std.mem.eql(u8, part.name, "model") and part.data.len > 0) return part.data;
        }
        return null;
    }
    return parseModelFromBody(body);
}

// ── tests ──

test "resolveRequestModelId: an alias names its model; an id beats it; /api/ strips the tag" {
    const a = std.testing.allocator;
    const reg = try ModelRegistry.init(a, std.Io.Threaded.global_single_threaded.io(), null, 8, 0, null);
    defer reg.deinit();
    _ = try reg.registerStub("org/Qwen3.6-27B-4bit", "/m/q", 1);
    _ = try reg.registerStub("qwen", "/m/other", 1);
    _ = try reg.registerStub("gemma-4-e4b-it-4bit", "/m/g", 1);
    var aliases: model_settings_mod.Cache = .{ .path = "", .settings = try model_settings_mod.parse(a,
        \\{"/m/q/": {"alias": "q"}, "/m/g": {"alias": "qwen"}, "/m/gone": {"alias": "ghost"}}
    ) };
    defer aliases.deinit();
    const chat = "/v1/chat/completions";
    try std.testing.expectEqualStrings("org/Qwen3.6-27B-4bit", resolveRequestModelId(reg, &aliases, "q", chat).id);
    try std.testing.expectEqualStrings("qwen", resolveRequestModelId(reg, &aliases, "qwen", chat).id);
    try std.testing.expectEqualStrings("org/Qwen3.6-27B-4bit", resolveRequestModelId(reg, &aliases, "q:latest", "/api/chat").id);
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "q:latest", chat) == .unknown_name);
    // An alias for a path nothing registered falls back like any unknown name.
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "ghost", chat) == .unknown_name);
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "gpt-4", chat) == .unknown_name);
}

test "resolveRequestModelId: a path names its own entry, never the default model" {
    const a = std.testing.allocator;
    const reg = try ModelRegistry.init(a, std.Io.Threaded.global_single_threaded.io(), null, 8, 0, null);
    defer reg.deinit();
    const good = try reg.registerStub("org/good", "/m/org/good", 1);
    reg.default_id = good.id;
    _ = try reg.registerStub("org/broken", "/m/org/broken", 1);
    var aliases: model_settings_mod.Cache = .{ .path = "" };
    defer aliases.deinit();

    try std.testing.expectEqualStrings("org/broken", resolveRequestModelId(reg, &aliases, "/m/org/broken", "/v1/chat/completions").id);
    try std.testing.expectEqualStrings("org/broken", resolveRequestModelId(reg, &aliases, "/m/org/broken/", "/v1/chat/completions").id);
    try std.testing.expectEqualStrings("org/broken", resolveRequestModelId(reg, &aliases, "org/broken", "/v1/chat/completions").id);
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "/m/org/missing", "/v1/chat/completions") == .unknown_path);
    // Plain unknown names and the alias keep the default-model fallback.
    try std.testing.expect(resolveRequestModelId(reg, &aliases, "gpt-4", "/v1/chat/completions") == .unknown_name);
    try std.testing.expectEqualStrings("mlx-serve", resolveRequestModelId(reg, &aliases, "mlx-serve", "/v1/chat/completions").id);
    try std.testing.expectEqualStrings("", resolveRequestModelId(reg, &aliases, "", "/v1/chat/completions").id);
}

test "parseModelFromRequest reads the model out of a multipart form, not just JSON" {
    // `/v1/images/edits` is the ONE endpoint whose body is multipart, and model
    // resolution runs BEFORE the route translates that form to JSON. A JSON-only
    // scan finds no `"model":` key, so the request silently ran against the
    // DEFAULT model: live via Open WebUI an edit naming a Mage-Flow
    // model hit the default chat model and 400'd "Target model does not support
    // this media modality"; headless with no default it 503'd instead.
    const ct = "multipart/form-data; boundary=abc123";
    // aiohttp (Open WebUI's client) writes the scalar fields first, each with
    // its own Content-Type line, and the file part last.
    const body =
        "--abc123\r\nContent-Type: text/plain; charset=utf-8\r\n" ++
        "Content-Disposition: form-data; name=\"model\"\r\n\r\nddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit\r\n" ++
        "--abc123\r\nContent-Type: text/plain; charset=utf-8\r\n" ++
        "Content-Disposition: form-data; name=\"prompt\"\r\n\r\nmake it winter\r\n" ++
        "--abc123\r\nContent-Disposition: form-data; name=\"image\"; filename=\"i.png\"\r\n" ++
        "Content-Type: image/png\r\n\r\n\x89PNG\r\n\x1a\n\x00\r\n--abc123--\r\n";
    try std.testing.expectEqualStrings(
        "ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit",
        parseModelFromRequest(body, ct) orelse "",
    );

    // A form with no `model` part is "use the default", same as JSON.
    const no_model = "--abc123\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhi\r\n--abc123--\r\n";
    try std.testing.expect(parseModelFromRequest(no_model, ct) == null);
    // An empty value is not an id either (aiohttp sends str(None) as "None",
    // but an unset field arrives empty).
    const empty = "--abc123\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n\r\n--abc123--\r\n";
    try std.testing.expect(parseModelFromRequest(empty, ct) == null);

    // JSON bodies are untouched — same answers as the JSON-only scanner, for
    // every other endpoint we serve.
    try std.testing.expectEqualStrings("m1", parseModelFromRequest("{\"model\":\"m1\"}", "application/json") orelse "");
    try std.testing.expectEqualStrings("m1", parseModelFromRequest("{\"model\":\"m1\"}", "") orelse "");
    try std.testing.expect(parseModelFromRequest("{\"prompt\":\"x\"}", "application/json") == null);
    // A multipart content-type with a JSON body (or a truncated form) degrades
    // to "no id" rather than misreading one.
    try std.testing.expect(parseModelFromRequest("{\"model\":\"m1\"}", ct) == null);
}

test "queryModel: GET /props?model=<id> routes to that model, percent-decoded" {
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings("mlx-community/Qwen3.5-0.8B", queryModel(&buf, "/props?x=1&model=mlx-community%2FQwen3.5-0.8B").?);
    try testing.expect(queryModel(&buf, "/props") == null);
    try testing.expect(queryModel(&buf, "/props?models=a") == null);
}
