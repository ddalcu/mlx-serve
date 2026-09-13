//! Upstream OpenAI-compatible chat providers (`~/.mlx-serve/providers.json`).
//!
//! A provider is any server speaking `POST /v1/chat/completions` — a cloud
//! API, another box on the LAN, a local runtime — named by a `@<name>` suffix
//! exactly like a LAN peer. Its `/v1/models` is probed in the background and
//! the rows ride our own `/v1/models` while the provider answers. A config
//! `models` list filters that list to the ids named (and is the whole list
//! for a provider that exposes none).
//! Transport is the system `curl` (TLS, HTTP/2, proxies for free — the same
//! choice `cli.zig` made for Hugging Face).
const std = @import("std");
const log = @import("log.zig");
const lan = @import("lan.zig");

pub const Model = lan.PeerModel;
pub const freeModels = lan.freePeerModels;

pub const Config = struct {
    name: []u8,
    /// Base URL with the trailing `/` trimmed; `/models` and `/chat/completions` append.
    url: []u8,
    api_key: ?[]u8,
    /// Declared model ids, used when the probe answers but lists nothing.
    models: [][]u8,

    pub fn deinit(c: *Config, alloc: std.mem.Allocator) void {
        alloc.free(c.name);
        alloc.free(c.url);
        if (c.api_key) |k| alloc.free(k);
        for (c.models) |m| alloc.free(m);
        alloc.free(c.models);
    }
};

pub fn freeConfigs(alloc: std.mem.Allocator, configs: []Config) void {
    for (configs) |*c| c.deinit(alloc);
    alloc.free(configs);
}

pub const EnvLookup = *const fn (name: []const u8) ?[]const u8;

pub fn processEnv(name: []const u8) ?[]const u8 {
    var buf: [256]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const v = std.c.getenv(buf[0..name.len :0]) orelse return null;
    const s = std.mem.span(v);
    return if (s.len == 0) null else s;
}

fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64) return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.')) return false;
    return true;
}

/// Parse the config file. Invalid entries are skipped with a warning; a body
/// that is not a JSON array (or `{"providers":[...]}`) is an error. Disabled
/// entries are dropped here so nothing downstream asks about `enabled`.
pub fn parseConfigs(alloc: std.mem.Allocator, body: []const u8, env: EnvLookup) ![]Config {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return error.BadProvidersJson;
    defer parsed.deinit();
    const list = switch (parsed.value) {
        .array => |a| a,
        .object => |o| blk: {
            const p = o.get("providers") orelse return error.BadProvidersJson;
            if (p != .array) return error.BadProvidersJson;
            break :blk p.array;
        },
        else => return error.BadProvidersJson,
    };
    var out: std.ArrayList(Config) = .empty;
    errdefer {
        for (out.items) |*c| c.deinit(alloc);
        out.deinit(alloc);
    }
    for (list.items) |item| {
        if (item != .object) continue;
        const o = item.object;
        if (o.get("enabled")) |e| if (e == .bool and !e.bool) continue;
        const name_v = o.get("name") orelse continue;
        const url_v = o.get("url") orelse continue;
        if (name_v != .string or url_v != .string) continue;
        if (!validName(name_v.string)) {
            log.warn("[providers] skipping \"{s}\": name must be [A-Za-z0-9_.-]\n", .{name_v.string});
            continue;
        }
        const url = std.mem.trimEnd(u8, std.mem.trim(u8, url_v.string, " \t"), "/");
        if (!std.mem.startsWith(u8, url, "http://") and !std.mem.startsWith(u8, url, "https://")) {
            log.warn("[providers] skipping \"{s}\": url must start with http:// or https://\n", .{name_v.string});
            continue;
        }
        var dup = false;
        for (out.items) |c| if (std.mem.eql(u8, c.name, name_v.string)) {
            dup = true;
        };
        if (dup) {
            log.warn("[providers] skipping duplicate \"{s}\"\n", .{name_v.string});
            continue;
        }
        var key: ?[]u8 = null;
        if (o.get("api_key_env")) |ev| if (ev == .string and ev.string.len > 0) {
            if (env(ev.string)) |v| key = try alloc.dupe(u8, v);
        };
        if (key == null) if (o.get("api_key")) |kv| if (kv == .string and kv.string.len > 0) {
            key = try alloc.dupe(u8, kv.string);
        };
        errdefer if (key) |k| alloc.free(k);
        var models: std.ArrayList([]u8) = .empty;
        errdefer {
            for (models.items) |m| alloc.free(m);
            models.deinit(alloc);
        }
        if (o.get("models")) |mv| if (mv == .array) for (mv.array.items) |m| {
            if (m == .string and m.string.len > 0) try models.append(alloc, try alloc.dupe(u8, m.string));
        };
        const name = try alloc.dupe(u8, name_v.string);
        errdefer alloc.free(name);
        const url_owned = try alloc.dupe(u8, url);
        errdefer alloc.free(url_owned);
        try out.append(alloc, .{ .name = name, .url = url_owned, .api_key = key, .models = try models.toOwnedSlice(alloc) });
    }
    return out.toOwnedSlice(alloc);
}

fn appendEntry(alloc: std.mem.Allocator, out: *std.ArrayList(Model), id: []const u8, provider: []const u8, ctx_len: ?u64) !void {
    const bare = try alloc.dupe(u8, id);
    errdefer alloc.free(bare);
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(alloc);
    var w: std.Io.Writer.Allocating = .fromArrayList(alloc, &buf);
    defer buf = w.toArrayList();
    var js: std.json.Stringify = .{ .writer = &w.writer, .options = .{} };
    try js.beginObject();
    try js.objectField("id");
    try js.print("\"{s}@{s}\"", .{ id, provider });
    try js.objectField("object");
    try js.write("model");
    try js.objectField("owned_by");
    try js.write(provider);
    try js.objectField("provider");
    try js.write(provider);
    // Nothing is resident here for a provider row; "state" says where it runs.
    try js.objectField("loaded");
    try js.write(false);
    try js.objectField("state");
    try js.write("remote");
    try js.objectField("capabilities");
    try js.write(&[_][]const u8{"chat"});
    if (ctx_len) |n| {
        // Twinned at the top level and under meta like every local row (#188).
        try js.objectField("context_length");
        try js.write(n);
        try js.objectField("max_model_len");
        try js.write(n);
        try js.objectField("meta");
        try js.beginObject();
        try js.objectField("context_length");
        try js.write(n);
        try js.endObject();
    }
    try js.endObject();
    const entry = try w.toOwnedSlice();
    errdefer alloc.free(entry);
    try out.append(alloc, .{ .id = bare, .entry_json = entry });
}

fn contextLengthOf(item: std.json.ObjectMap) ?u64 {
    for ([_][]const u8{ "context_length", "max_model_len" }) |k| if (item.get(k)) |v| if (v == .integer and v.integer > 0) return @intCast(v.integer);
    if (item.get("meta")) |m| if (m == .object) if (m.object.get("context_length")) |v| if (v == .integer and v.integer > 0) return @intCast(v.integer);
    return null;
}

/// Parse a provider's `/v1/models` body into rows named `<id>@<provider>`.
/// Anything without a `data` array of objects with string ids is an error.
pub fn parseModels(alloc: std.mem.Allocator, body: []const u8, provider: []const u8) ![]Model {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return error.BadModelsJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadModelsJson;
    const data = parsed.value.object.get("data") orelse return error.BadModelsJson;
    if (data != .array) return error.BadModelsJson;
    var out: std.ArrayList(Model) = .empty;
    errdefer {
        for (out.items) |m| {
            alloc.free(m.id);
            alloc.free(m.entry_json);
        }
        out.deinit(alloc);
    }
    for (data.array.items) |item| {
        if (item != .object) continue;
        const id_v = item.object.get("id") orelse continue;
        if (id_v != .string or id_v.string.len == 0) continue;
        try appendEntry(alloc, &out, id_v.string, provider, contextLengthOf(item.object));
    }
    return out.toOwnedSlice(alloc);
}

pub fn declaredModels(alloc: std.mem.Allocator, ids: []const []const u8, provider: []const u8) ![]Model {
    var out: std.ArrayList(Model) = .empty;
    errdefer {
        for (out.items) |m| {
            alloc.free(m.id);
            alloc.free(m.entry_json);
        }
        out.deinit(alloc);
    }
    for (ids) |id| try appendEntry(alloc, &out, id, provider, null);
    return out.toOwnedSlice(alloc);
}

/// Does `url` name THIS server (loopback host + our bound port)? A server
/// added as its own provider mirrors itself and every proxied request lands
/// back on it — refused at reload, never probed.
pub fn isSelfUrl(url: []const u8, own_port: u16) bool {
    const rest = if (std.mem.startsWith(u8, url, "http://")) url["http://".len..] else if (std.mem.startsWith(u8, url, "https://")) url["https://".len..] else return false;
    const authority = rest[0 .. std.mem.indexOfScalar(u8, rest, '/') orelse rest.len];
    const colon = std.mem.lastIndexOfScalar(u8, authority, ':') orelse return false;
    const host = authority[0..colon];
    const port = std.fmt.parseInt(u16, authority[colon + 1 ..], 10) catch return false;
    if (port != own_port) return false;
    return std.mem.eql(u8, host, "localhost") or std.mem.eql(u8, host, "[::1]") or
        std.mem.eql(u8, host, "0.0.0.0") or std.mem.startsWith(u8, host, "127.");
}

/// A bare host (`http://host:port`) is a common way to write a provider;
/// the OpenAI base lives at `/v1`. Chosen after probing: the bare URL is
/// tried first and `/v1` only adopted when it answers with a model list.
pub fn v1Candidate(alloc: std.mem.Allocator, url: []const u8) !?[]u8 {
    if (std.mem.endsWith(u8, url, "/v1")) return null;
    return try std.fmt.allocPrint(alloc, "{s}/v1", .{url});
}

fn listsModels(alloc: std.mem.Allocator, answer: ProbeAnswer, name: []const u8) bool {
    const a = answer orelse return false;
    if (a.status != 200) return false;
    const rows = parseModels(alloc, a.body, name) catch return false;
    defer freeModels(alloc, rows);
    return rows.len > 0;
}

/// One probe's raw outcome: `null` = no HTTP answer at all (refused, DNS,
/// TLS, timeout); otherwise the status and body the provider returned.
pub const ProbeAnswer = ?struct { status: u16, body: []const u8 };

/// Rows a probe outcome earns. Reachable + a usable list wins; reachable
/// with anything else (404, empty list, non-JSON) falls back to the declared
/// models — a server with no `/v1/models` is still up; unreachable = no rows.
/// A declared list FILTERS a listed provider: only those ids ride, with the
/// listed row's metadata where the provider knows the id.
pub fn rowsFor(alloc: std.mem.Allocator, answer: ProbeAnswer, cfg: Config) ![]Model {
    const a = answer orelse return alloc.alloc(Model, 0);
    if (a.status == 200) {
        if (parseModels(alloc, a.body, cfg.name)) |rows| {
            if (rows.len > 0 and cfg.models.len == 0) return rows;
            defer freeModels(alloc, rows);
            if (rows.len > 0) return filterDeclared(alloc, rows, cfg);
        } else |_| {}
    }
    return declaredModels(alloc, cfg.models, cfg.name);
}

fn filterDeclared(alloc: std.mem.Allocator, listed: []Model, cfg: Config) ![]Model {
    var out: std.ArrayList(Model) = .empty;
    errdefer freeModels(alloc, out.items);
    for (cfg.models) |id| {
        const hit = for (listed) |m| {
            if (std.mem.eql(u8, m.id, id)) break m;
        } else null;
        if (hit) |m| {
            const bare = try alloc.dupe(u8, m.id);
            errdefer alloc.free(bare);
            const entry = try alloc.dupe(u8, m.entry_json);
            try out.append(alloc, .{ .id = bare, .entry_json = entry });
        } else try appendEntry(alloc, &out, id, cfg.name, null);
    }
    return out.toOwnedSlice(alloc);
}

pub const UpstreamHead = struct { status: u16, reason: []const u8, content_type: []const u8 };

/// Read the status + content-type off a `curl -i` response head. Handles
/// `HTTP/2 200` (no reason phrase) and `HTTP/1.1 200 OK` alike.
pub fn parseUpstreamHead(head: []const u8) ?UpstreamHead {
    const line_end = std.mem.indexOf(u8, head, "\r\n") orelse head.len;
    const line = head[0..line_end];
    if (!std.mem.startsWith(u8, line, "HTTP/")) return null;
    const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return null;
    const rest = line[sp + 1 ..];
    const code_end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
    const status = std.fmt.parseInt(u16, rest[0..code_end], 10) catch return null;
    const reason = if (code_end < rest.len) rest[code_end + 1 ..] else reasonFor(status);
    const ct = lan.headerValueCI(head, "content-type") orelse "application/json";
    return .{ .status = status, .reason = reason, .content_type = ct };
}

fn reasonFor(status: u16) []const u8 {
    return switch (status) {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        else => "",
    };
}

/// Resolved target for one request: owned copies so the table lock is never
/// held across a network call.
pub const Upstream = struct {
    url: []u8,
    api_key: ?[]u8,
    bare: []u8,

    pub fn deinit(u: *Upstream, alloc: std.mem.Allocator) void {
        alloc.free(u.url);
        if (u.api_key) |k| alloc.free(k);
        alloc.free(u.bare);
    }
};

const PROBE_INTERVAL_S: u32 = 60;

const Slot = struct { cfg: Config, rows: []Model = &.{}, up: bool = false, probed: bool = false };

pub const Providers = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    path: []u8,
    own_port: u16 = 0,
    mu: std.c.pthread_mutex_t = .{},
    slots: []Slot = &.{},
    thread: ?std.Thread = null,
    stop_flag: std.atomic.Value(bool) = .init(false),
    probe_asap: std.atomic.Value(bool) = .init(false),

    /// Reads `path` (missing = no providers) and starts the probe thread.
    pub fn start(alloc: std.mem.Allocator, io: std.Io, path: []const u8, own_port: u16) !*Providers {
        const p = try alloc.create(Providers);
        errdefer alloc.destroy(p);
        p.* = .{ .alloc = alloc, .io = io, .path = try alloc.dupe(u8, path), .own_port = own_port };
        errdefer alloc.free(p.path);
        _ = p.reload() catch |err| log.warn("[providers] {s}: {s}\n", .{ path, @errorName(err) });
        p.thread = try std.Thread.spawn(.{}, threadMain, .{p});
        return p;
    }

    pub fn shutdown(p: *Providers) void {
        p.stop_flag.store(true, .release);
        if (p.thread) |th| th.join();
        p.freeSlots(p.slots);
        p.alloc.free(p.path);
        const alloc = p.alloc;
        alloc.destroy(p);
    }

    fn freeSlots(p: *Providers, slots: []Slot) void {
        for (slots) |*s| {
            s.cfg.deinit(p.alloc);
            freeModels(p.alloc, s.rows);
        }
        p.alloc.free(slots);
    }

    /// Re-read the config file and schedule an immediate probe. Returns the
    /// number of enabled providers, or the parse error. Rows of a provider
    /// that survives the reload under the same name are kept until re-probed.
    pub fn reload(p: *Providers) !usize {
        const body = std.Io.Dir.cwd().readFileAlloc(p.io, p.path, p.alloc, .limited(1 << 20)) catch |err| switch (err) {
            error.FileNotFound => try p.alloc.alloc(u8, 0),
            else => return err,
        };
        defer p.alloc.free(body);
        const configs = if (body.len == 0) try p.alloc.alloc(Config, 0) else try parseConfigs(p.alloc, body, processEnv);
        errdefer freeConfigs(p.alloc, configs);
        const slots_all = try p.alloc.alloc(Slot, configs.len);
        var kept: usize = 0;
        for (configs) |c| {
            if (isSelfUrl(c.url, p.own_port)) {
                log.warn("[providers] skipping \"{s}\": {s} is this server\n", .{ c.name, c.url });
                var dead = c;
                dead.deinit(p.alloc);
                continue;
            }
            slots_all[kept] = .{ .cfg = c };
            kept += 1;
        }
        p.alloc.free(configs);
        const slots = p.alloc.realloc(slots_all, kept) catch slots_all[0..kept];
        _ = std.c.pthread_mutex_lock(&p.mu);
        for (slots) |*s| for (p.slots) |*old| if (std.mem.eql(u8, old.cfg.name, s.cfg.name)) {
            s.rows = old.rows;
            s.up = old.up;
            old.rows = &.{};
        };
        const stale = p.slots;
        p.slots = slots;
        _ = std.c.pthread_mutex_unlock(&p.mu);
        p.freeSlots(stale);
        p.probe_asap.store(true, .release);
        return slots.len;
    }

    pub fn count(p: *Providers) usize {
        _ = std.c.pthread_mutex_lock(&p.mu);
        defer _ = std.c.pthread_mutex_unlock(&p.mu);
        return p.slots.len;
    }

    pub fn isProvider(p: *Providers, name: []const u8) bool {
        _ = std.c.pthread_mutex_lock(&p.mu);
        defer _ = std.c.pthread_mutex_unlock(&p.mu);
        for (p.slots) |s| if (std.mem.eql(u8, s.cfg.name, name)) return true;
        return false;
    }

    /// Owned upstream for a `<bare>@<name>` id whose suffix names a configured
    /// provider. Any bare id is accepted: a probe is a snapshot and the
    /// provider's own 404 is the honest answer for an unknown model.
    pub fn lookup(p: *Providers, alloc: std.mem.Allocator, id: []const u8) ?Upstream {
        const rid = lan.splitRemoteId(id) orelse return null;
        _ = std.c.pthread_mutex_lock(&p.mu);
        defer _ = std.c.pthread_mutex_unlock(&p.mu);
        for (p.slots) |s| if (std.mem.eql(u8, s.cfg.name, rid.peer)) {
            const url = alloc.dupe(u8, s.cfg.url) catch return null;
            const key = if (s.cfg.api_key) |k| alloc.dupe(u8, k) catch {
                alloc.free(url);
                return null;
            } else null;
            const bare = alloc.dupe(u8, rid.bare) catch {
                alloc.free(url);
                if (key) |k| alloc.free(k);
                return null;
            };
            return .{ .url = url, .api_key = key, .bare = bare };
        };
        return null;
    }

    pub fn entryFor(p: *Providers, alloc: std.mem.Allocator, id: []const u8) ?[]u8 {
        const rid = lan.splitRemoteId(id) orelse return null;
        _ = std.c.pthread_mutex_lock(&p.mu);
        defer _ = std.c.pthread_mutex_unlock(&p.mu);
        for (p.slots) |s| if (std.mem.eql(u8, s.cfg.name, rid.peer)) {
            for (s.rows) |m| if (std.mem.eql(u8, m.id, rid.bare)) return alloc.dupe(u8, m.entry_json) catch null;
        };
        return null;
    }

    pub fn appendEntries(p: *Providers, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        _ = std.c.pthread_mutex_lock(&p.mu);
        defer _ = std.c.pthread_mutex_unlock(&p.mu);
        for (p.slots) |s| for (s.rows) |m| {
            if (buf.items.len > 0) try buf.append(alloc, ',');
            try buf.appendSlice(alloc, m.entry_json);
        };
    }

    /// `{"providers":[{"name","url","up","models":N}]}` for the app's health dots.
    pub fn statusJson(p: *Providers, alloc: std.mem.Allocator) ![]u8 {
        _ = std.c.pthread_mutex_lock(&p.mu);
        defer _ = std.c.pthread_mutex_unlock(&p.mu);
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(alloc);
        var w: std.Io.Writer.Allocating = .fromArrayList(alloc, &buf);
        defer buf = w.toArrayList();
        var js: std.json.Stringify = .{ .writer = &w.writer, .options = .{} };
        try js.beginObject();
        try js.objectField("providers");
        try js.beginArray();
        for (p.slots) |s| {
            try js.beginObject();
            try js.objectField("name");
            try js.write(s.cfg.name);
            try js.objectField("url");
            try js.write(s.cfg.url);
            try js.objectField("up");
            try js.write(s.up);
            try js.objectField("probed");
            try js.write(s.probed);
            try js.objectField("models");
            try js.write(s.rows.len);
            try js.endObject();
        }
        try js.endArray();
        try js.endObject();
        return w.toOwnedSlice();
    }

    fn probeAll(p: *Providers) void {
        // Snapshot the configs so the network calls run unlocked.
        _ = std.c.pthread_mutex_lock(&p.mu);
        const n = p.slots.len;
        var names = p.alloc.alloc([]u8, n) catch {
            _ = std.c.pthread_mutex_unlock(&p.mu);
            return;
        };
        var ok: usize = 0;
        for (p.slots, 0..) |s, i| {
            names[i] = p.alloc.dupe(u8, s.cfg.name) catch break;
            ok += 1;
        }
        _ = std.c.pthread_mutex_unlock(&p.mu);
        defer {
            for (names[0..ok]) |nm| p.alloc.free(nm);
            p.alloc.free(names);
        }
        for (names[0..ok]) |name| {
            if (p.stop_flag.load(.acquire)) return;
            p.probeOne(name);
        }
    }

    fn probeOne(p: *Providers, name: []const u8) void {
        var cfg_copy: ?Config = null;
        {
            _ = std.c.pthread_mutex_lock(&p.mu);
            defer _ = std.c.pthread_mutex_unlock(&p.mu);
            for (p.slots) |s| if (std.mem.eql(u8, s.cfg.name, name)) {
                cfg_copy = dupeConfig(p.alloc, s.cfg) catch null;
            };
        }
        var cfg = cfg_copy orelse return;
        defer cfg.deinit(p.alloc);

        var answer = curlProbe(p.alloc, p.io, cfg);
        defer if (answer) |a| p.alloc.free(a.body);
        var adopted_v1: ?[]u8 = null;
        defer if (adopted_v1) |u| p.alloc.free(u);
        if (!listsModels(p.alloc, answer, cfg.name)) if (v1Candidate(p.alloc, cfg.url) catch null) |v1| {
            var alt = cfg;
            alt.url = v1;
            const alt_answer = curlProbe(p.alloc, p.io, alt);
            if (listsModels(p.alloc, alt_answer, cfg.name)) {
                if (answer) |a| p.alloc.free(a.body);
                answer = alt_answer;
                adopted_v1 = v1;
                log.info("[providers] {s}: using {s}\n", .{ cfg.name, v1 });
            } else {
                if (alt_answer) |a| p.alloc.free(a.body);
                p.alloc.free(v1);
            }
        };
        const rows = rowsFor(p.alloc, answer, cfg) catch return;

        _ = std.c.pthread_mutex_lock(&p.mu);
        defer _ = std.c.pthread_mutex_unlock(&p.mu);
        for (p.slots) |*s| if (std.mem.eql(u8, s.cfg.name, name)) {
            const was_up = s.up;
            const had = s.rows.len;
            freeModels(p.alloc, s.rows);
            s.rows = rows;
            s.up = answer != null;
            if (adopted_v1) |u| if (p.alloc.dupe(u8, u)) |owned| {
                p.alloc.free(s.cfg.url);
                s.cfg.url = owned;
            } else |_| {};
            if (!s.probed or was_up != s.up or had != rows.len) {
                if (s.up)
                    log.info("[providers] {s}: up, {d} models ({s})\n", .{ name, rows.len, if (answer.?.status == 200 and !declaredOnly(cfg, rows)) "listed" else "declared" })
                else
                    log.warn("[providers] {s}: unreachable at {s}\n", .{ name, cfg.url });
            }
            s.probed = true;
            return;
        };
        freeModels(p.alloc, rows); // provider vanished in a reload mid-probe
    }

    fn declaredOnly(cfg: Config, rows: []Model) bool {
        if (rows.len != cfg.models.len) return false;
        for (rows, cfg.models) |r, m| if (!std.mem.eql(u8, r.id, m)) return false;
        return true;
    }
};

fn dupeConfig(alloc: std.mem.Allocator, c: Config) !Config {
    const name = try alloc.dupe(u8, c.name);
    errdefer alloc.free(name);
    const url = try alloc.dupe(u8, c.url);
    errdefer alloc.free(url);
    const key = if (c.api_key) |k| try alloc.dupe(u8, k) else null;
    errdefer if (key) |k| alloc.free(k);
    const models = try alloc.alloc([]u8, c.models.len);
    var filled: usize = 0;
    errdefer {
        for (models[0..filled]) |m| alloc.free(m);
        alloc.free(models);
    }
    for (c.models, 0..) |m, i| {
        models[i] = try alloc.dupe(u8, m);
        filled += 1;
    }
    return .{ .name = name, .url = url, .api_key = key, .models = models };
}

fn threadMain(p: *Providers) void {
    var since_probe: u32 = PROBE_INTERVAL_S; // probe at once on boot
    while (!p.stop_flag.load(.acquire)) {
        if (p.probe_asap.swap(false, .acq_rel) or since_probe >= PROBE_INTERVAL_S) {
            since_probe = 0;
            p.probeAll();
        }
        const ts = std.c.timespec{ .sec = 1, .nsec = 0 };
        _ = std.c.nanosleep(&ts, null);
        since_probe += 1;
    }
}

const CURL_MAX_TIME_PROBE_S = 10;

/// GET `<url>/models`; `null` when curl got no HTTP answer at all.
fn curlProbe(alloc: std.mem.Allocator, io: std.Io, cfg: Config) ProbeAnswer {
    const url = std.fmt.allocPrint(alloc, "{s}/models", .{cfg.url}) catch return null;
    defer alloc.free(url);
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(alloc);
    argv.appendSlice(alloc, &.{ "curl", "-s", "-S", "--max-time", std.fmt.comptimePrint("{d}", .{CURL_MAX_TIME_PROBE_S}), "-w", "\n%{http_code}" }) catch return null;
    const auth = authHeader(alloc, cfg.api_key) catch return null;
    defer if (auth) |a| alloc.free(a);
    if (auth) |a| argv.appendSlice(alloc, &.{ "-H", a }) catch return null;
    argv.append(alloc, url) catch return null;
    const result = std.process.run(alloc, io, .{ .argv = argv.items, .stdout_limit = .limited(8 << 20), .stderr_limit = .limited(64 << 10) }) catch return null;
    defer alloc.free(result.stderr);
    errdefer alloc.free(result.stdout);
    switch (result.term) {
        .exited => |code| if (code != 0) {
            log.debug("[providers] probe {s}: curl exit {d}: {s}\n", .{ cfg.name, code, std.mem.trim(u8, result.stderr, " \n") });
            alloc.free(result.stdout);
            return null;
        },
        else => {
            alloc.free(result.stdout);
            return null;
        },
    }
    const split = splitStatusTrailer(result.stdout) orelse {
        alloc.free(result.stdout);
        return null;
    };
    // Shrink in place: the body is a prefix of stdout.
    const body = alloc.realloc(result.stdout, split.body_len) catch result.stdout[0..split.body_len];
    return .{ .status = split.status, .body = body };
}

/// `-w "\n%{http_code}"` leaves the status on the last line of stdout.
pub fn splitStatusTrailer(stdout: []const u8) ?struct { status: u16, body_len: usize } {
    const nl = std.mem.lastIndexOfScalar(u8, stdout, '\n') orelse return null;
    const status = std.fmt.parseInt(u16, stdout[nl + 1 ..], 10) catch return null;
    if (status == 0) return null; // curl prints 000 when no response arrived
    return .{ .status = status, .body_len = nl };
}

fn authHeader(alloc: std.mem.Allocator, key: ?[]const u8) !?[]u8 {
    const k = key orelse return null;
    return try std.fmt.allocPrint(alloc, "Authorization: Bearer {s}", .{k});
}

const CURL_MAX_TIME_CHAT_S = 900;

/// POST the (model-rewritten) body to `<url>/chat/completions` and pump the
/// answer back. Status + content-type are relayed, the body streams as it
/// arrives, the connection closes at upstream EOF. `error.ProviderUnreachable`
/// is returned BEFORE anything reaches `conn`; a later failure ends the stream.
pub fn proxyChat(alloc: std.mem.Allocator, io: std.Io, up: Upstream, body: []const u8, conn: anytype) error{ProviderUnreachable}!void {
    const url = std.fmt.allocPrint(alloc, "{s}/chat/completions", .{up.url}) catch return error.ProviderUnreachable;
    defer alloc.free(url);
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(alloc);
    argv.appendSlice(alloc, &.{
        "curl",       "-s",                                                    "-S", "-N", "-i", "-X", "POST",
        "--max-time", std.fmt.comptimePrint("{d}", .{CURL_MAX_TIME_CHAT_S}),
        "-H",         "Content-Type: application/json",
        "-H",         "Accept: */*",
        "-H",         "Expect:",
        "--data-binary", "@-",
    }) catch return error.ProviderUnreachable;
    const auth = authHeader(alloc, up.api_key) catch return error.ProviderUnreachable;
    defer if (auth) |a| alloc.free(a);
    if (auth) |a| argv.appendSlice(alloc, &.{ "-H", a }) catch return error.ProviderUnreachable;
    argv.append(alloc, url) catch return error.ProviderUnreachable;

    var child = std.process.spawn(io, .{ .argv = argv.items, .stdin = .pipe, .stdout = .pipe, .stderr = .ignore }) catch return error.ProviderUnreachable;
    defer {
        child.kill(io);
    }
    {
        var in_buf: [4096]u8 = undefined;
        var stdin_w = child.stdin.?.writer(io, &in_buf);
        stdin_w.interface.writeAll(body) catch return error.ProviderUnreachable;
        stdin_w.interface.flush() catch return error.ProviderUnreachable;
        child.stdin.?.close(io);
        child.stdin = null;
    }
    const fd = child.stdout.?.handle;

    // Head first: everything up to the blank line, then relay our own head
    // (curl already decoded any chunking, so the upstream framing headers
    // must not be forwarded).
    var acc: std.ArrayList(u8) = .empty;
    defer acc.deinit(alloc);
    var buf: [16 * 1024]u8 = undefined;
    const head_end = while (true) {
        if (std.mem.indexOf(u8, acc.items, "\r\n\r\n")) |at| break at;
        if (acc.items.len > 64 * 1024) return error.ProviderUnreachable;
        const n = pollRead(fd, &buf, conn) catch return error.ProviderUnreachable;
        if (n == 0) return error.ProviderUnreachable;
        acc.appendSlice(alloc, buf[0..n]) catch return error.ProviderUnreachable;
    };
    const head = parseUpstreamHead(acc.items[0..head_end]) orelse return error.ProviderUnreachable;
    var head_buf: [512]u8 = undefined;
    const our_head = std.fmt.bufPrint(&head_buf, "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n", .{ head.status, head.reason, head.content_type }) catch return error.ProviderUnreachable;
    conn.writeAll(our_head) catch return;
    if (acc.items.len > head_end + 4) conn.writeAll(acc.items[head_end + 4 ..]) catch return;
    while (true) {
        const n = pollRead(fd, &buf, conn) catch return;
        if (n == 0) return;
        conn.writeAll(buf[0..n]) catch return;
    }
}

/// Blocking read with a 1 s client-disconnect probe, so an abandoned chat
/// tears curl down instead of paying the provider for the whole answer.
fn pollRead(fd: std.posix.fd_t, buf: []u8, conn: anytype) !usize {
    while (true) {
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const ready = std.posix.poll(&fds, 1000) catch return error.ReadFailed;
        if (ready == 0) {
            if (conn.peerClosed()) return error.ClientGone;
            continue;
        }
        const n = std.c.read(fd, buf.ptr, buf.len);
        if (n < 0) return error.ReadFailed;
        return @intCast(n);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

const t = std.testing;

fn fakeEnv(name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "OPENAI_API_KEY")) return "sk-from-env";
    return null;
}

test "providers: parseConfigs reads the file, env key wins, bad rows are skipped" {
    const a = t.allocator;
    const body =
        \\[
        \\ {"name":"openai","url":"https://api.openai.com/v1/","api_key":"sk-literal","api_key_env":"OPENAI_API_KEY"},
        \\ {"name":"local","url":"http://127.0.0.1:1234/v1","models":["llama","", 3]},
        \\ {"name":"off","url":"http://x/v1","enabled":false},
        \\ {"name":"bad name","url":"http://x/v1"},
        \\ {"name":"ftp","url":"ftp://x/v1"},
        \\ {"name":"openai","url":"http://dup/v1"},
        \\ {"name":"unsetenv","url":"http://y/v1","api_key_env":"NOPE_UNSET","api_key":"fallback"},
        \\ "junk"
        \\]
    ;
    const cfgs = try parseConfigs(a, body, fakeEnv);
    defer freeConfigs(a, cfgs);
    try t.expectEqual(@as(usize, 3), cfgs.len);
    try t.expectEqualStrings("openai", cfgs[0].name);
    try t.expectEqualStrings("https://api.openai.com/v1", cfgs[0].url); // trailing slash trimmed
    try t.expectEqualStrings("sk-from-env", cfgs[0].api_key.?);
    try t.expectEqualStrings("local", cfgs[1].name);
    try t.expect(cfgs[1].api_key == null);
    try t.expectEqual(@as(usize, 1), cfgs[1].models.len);
    try t.expectEqualStrings("llama", cfgs[1].models[0]);
    // An unset env var falls back to the literal key rather than to no key.
    try t.expectEqualStrings("fallback", cfgs[2].api_key.?);
    // Object wrapper form is accepted too; anything else is an error.
    const wrapped = try parseConfigs(a, "{\"providers\":[{\"name\":\"a\",\"url\":\"http://a/v1\"}]}", fakeEnv);
    defer freeConfigs(a, wrapped);
    try t.expectEqual(@as(usize, 1), wrapped.len);
    try t.expectError(error.BadProvidersJson, parseConfigs(a, "{\"nope\":1}", fakeEnv));
    try t.expectError(error.BadProvidersJson, parseConfigs(a, "not json", fakeEnv));
}

test "providers: parseModels suffixes ids, badges the provider, twins context_length" {
    const a = t.allocator;
    const body =
        \\{"object":"list","data":[
        \\ {"id":"gpt-5","object":"model","owned_by":"openai"},
        \\ {"id":"openrouter/auto","context_length":200000},
        \\ {"id":"peer-model","meta":{"context_length":94000}},
        \\ {"id":""}, {"nope":1}, 7
        \\]}
    ;
    const rows = try parseModels(a, body, "openai");
    defer freeModels(a, rows);
    try t.expectEqual(@as(usize, 3), rows.len);
    try t.expectEqualStrings("gpt-5", rows[0].id);
    try t.expect(std.mem.indexOf(u8, rows[0].entry_json, "\"id\":\"gpt-5@openai\"") != null);
    try t.expect(std.mem.indexOf(u8, rows[0].entry_json, "\"provider\":\"openai\"") != null);
    try t.expect(std.mem.indexOf(u8, rows[0].entry_json, "\"capabilities\":[\"chat\"]") != null);
    try t.expect(std.mem.indexOf(u8, rows[0].entry_json, "context_length") == null);
    try t.expect(std.mem.indexOf(u8, rows[1].entry_json, "\"context_length\":200000") != null);
    try t.expect(std.mem.indexOf(u8, rows[1].entry_json, "\"max_model_len\":200000") != null);
    try t.expect(std.mem.indexOf(u8, rows[1].entry_json, "\"meta\":{\"context_length\":200000}") != null);
    try t.expect(std.mem.indexOf(u8, rows[2].entry_json, "\"context_length\":94000") != null);
    try t.expectError(error.BadModelsJson, parseModels(a, "{\"error\":\"nope\"}", "x"));
    try t.expectError(error.BadModelsJson, parseModels(a, "<html>", "x"));
}

test "providers: rowsFor — listed rides undeclared, reachable-but-listless falls back, unreachable is empty" {
    const a = t.allocator;
    var declared = [_][]u8{ try a.dupe(u8, "llama"), try a.dupe(u8, "phi") };
    var cfg = Config{ .name = try a.dupe(u8, "local"), .url = try a.dupe(u8, "http://l/v1"), .api_key = null, .models = &declared };
    defer {
        a.free(cfg.name);
        a.free(cfg.url);
        for (declared) |m| a.free(m);
    }
    // Nothing declared: the provider's own list rides as-is.
    cfg.models = &.{};
    const listed = try rowsFor(a, .{ .status = 200, .body = "{\"data\":[{\"id\":\"real\"}]}" }, cfg);
    defer freeModels(a, listed);
    try t.expectEqual(@as(usize, 1), listed.len);
    try t.expectEqualStrings("real", listed[0].id);
    cfg.models = &declared;

    // 404 (no /v1/models), an empty list, and non-JSON at 200 all mean "up": declared rows.
    for ([_]struct { s: u16, b: []const u8 }{
        .{ .s = 404, .b = "not found" },
        .{ .s = 200, .b = "{\"data\":[]}" },
        .{ .s = 200, .b = "<html>" },
        .{ .s = 401, .b = "{\"error\":\"bad key\"}" },
    }) |c| {
        const rows = try rowsFor(a, .{ .status = c.s, .body = c.b }, cfg);
        defer freeModels(a, rows);
        try t.expectEqual(@as(usize, 2), rows.len);
        try t.expectEqualStrings("phi", rows[1].id);
        try t.expect(std.mem.indexOf(u8, rows[1].entry_json, "\"id\":\"phi@local\"") != null);
    }
    // No answer at all: nothing, even with declared models — the provider is down.
    const down = try rowsFor(a, null, cfg);
    defer freeModels(a, down);
    try t.expectEqual(@as(usize, 0), down.len);
    // Reachable, nothing listed, nothing declared: zero rows, no error.
    cfg.models = &.{};
    const bare = try rowsFor(a, .{ .status = 404, .body = "" }, cfg);
    defer freeModels(a, bare);
    try t.expectEqual(@as(usize, 0), bare.len);
}

test "providers: rowsFor — a declared list FILTERS a listed provider, keeping listed metadata" {
    const a = t.allocator;
    var declared = [_][]u8{ try a.dupe(u8, "gpt-5"), try a.dupe(u8, "phi") };
    const cfg = Config{ .name = try a.dupe(u8, "or"), .url = try a.dupe(u8, "http://o/v1"), .api_key = null, .models = &declared };
    defer {
        a.free(cfg.name);
        a.free(cfg.url);
        for (declared) |m| a.free(m);
    }
    const body = "{\"data\":[{\"id\":\"junk\"},{\"id\":\"gpt-5\",\"context_length\":200000},{\"id\":\"other\"}]}";
    const rows = try rowsFor(a, .{ .status = 200, .body = body }, cfg);
    defer freeModels(a, rows);
    try t.expectEqual(@as(usize, 2), rows.len);
    try t.expectEqualStrings("gpt-5", rows[0].id);
    try t.expect(std.mem.indexOf(u8, rows[0].entry_json, "\"context_length\":200000") != null);
    // Declared but not listed still rides, bare.
    try t.expectEqualStrings("phi", rows[1].id);
    try t.expect(std.mem.indexOf(u8, rows[1].entry_json, "context_length") == null);
}

test "providers: isSelfUrl names only loopback + our own port; v1Candidate appends once" {
    try t.expect(isSelfUrl("http://localhost:11234", 11234));
    try t.expect(isSelfUrl("http://127.0.0.1:11234/v1", 11234));
    try t.expect(isSelfUrl("http://[::1]:11234/v1", 11234));
    try t.expect(!isSelfUrl("http://localhost:11235/v1", 11234));
    try t.expect(!isSelfUrl("http://192.168.1.20:11234/v1", 11234));
    try t.expect(!isSelfUrl("https://api.openai.com/v1", 443));
    const a = t.allocator;
    const v1 = (try v1Candidate(a, "http://localhost:1234")).?;
    defer a.free(v1);
    try t.expectEqualStrings("http://localhost:1234/v1", v1);
    try t.expect((try v1Candidate(a, "http://localhost:1234/v1")) == null);
}

test "providers: parseUpstreamHead reads HTTP/1.1 and HTTP/2 status lines" {
    const h1 = parseUpstreamHead("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nTransfer-Encoding: chunked\r\n").?;
    try t.expectEqual(@as(u16, 200), h1.status);
    try t.expectEqualStrings("OK", h1.reason);
    try t.expectEqualStrings("text/event-stream; charset=utf-8", h1.content_type);
    const h2 = parseUpstreamHead("HTTP/2 429\r\ncontent-type: application/json\r\nretry-after: 2\r\n").?;
    try t.expectEqual(@as(u16, 429), h2.status);
    try t.expectEqualStrings("Too Many Requests", h2.reason);
    try t.expectEqualStrings("application/json", h2.content_type);
    // No content-type defaults to JSON (error bodies from odd servers).
    try t.expectEqualStrings("application/json", parseUpstreamHead("HTTP/1.1 502\r\n").?.content_type);
    try t.expect(parseUpstreamHead("garbage") == null);
    try t.expect(parseUpstreamHead("HTTP/1.1 abc\r\n") == null);
}

test "providers: splitStatusTrailer peels curl's -w status off stdout" {
    const s = splitStatusTrailer("{\"data\":[]}\n200").?;
    try t.expectEqual(@as(u16, 200), s.status);
    try t.expectEqual(@as(usize, 11), s.body_len);
    try t.expectEqual(@as(u16, 404), splitStatusTrailer("\n404").?.status);
    try t.expect(splitStatusTrailer("{}\n000") == null);
    try t.expect(splitStatusTrailer("no newline") == null);
}

test "providers: lookup / entryFor / appendEntries key on the @suffix; declared rows are visible" {
    const a = t.allocator;
    var p = Providers{ .alloc = a, .io = undefined, .path = try a.dupe(u8, "/nonexistent") };
    defer {
        p.freeSlots(p.slots);
        a.free(p.path);
    }
    const cfgs = try parseConfigs(a, "[{\"name\":\"cloud\",\"url\":\"https://c/v1\",\"api_key\":\"k\"},{\"name\":\"box\",\"url\":\"http://b/v1\",\"models\":[\"m1\"]}]", fakeEnv);
    p.slots = try a.alloc(Slot, cfgs.len);
    for (p.slots, cfgs) |*s, c| s.* = .{ .cfg = c };
    a.free(cfgs);
    p.slots[1].rows = try declaredModels(a, p.slots[1].cfg.models, "box");
    p.slots[1].up = true;

    try t.expect(p.isProvider("cloud"));
    try t.expect(!p.isProvider("studio"));
    var up = p.lookup(a, "gpt-5@cloud").?;
    defer up.deinit(a);
    try t.expectEqualStrings("https://c/v1", up.url);
    try t.expectEqualStrings("k", up.api_key.?);
    try t.expectEqualStrings("gpt-5", up.bare);
    // A LAN peer's id is not ours to answer.
    try t.expect(p.lookup(a, "gemma@studio") == null);
    try t.expect(p.lookup(a, "no-suffix") == null);

    const entry = p.entryFor(a, "m1@box").?;
    defer a.free(entry);
    try t.expect(std.mem.indexOf(u8, entry, "\"id\":\"m1@box\"") != null);
    try t.expect(p.entryFor(a, "m2@box") == null);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try buf.appendSlice(a, "{\"id\":\"local\"}");
    try p.appendEntries(a, &buf);
    try t.expectEqualStrings("{\"id\":\"local\"},{\"id\":\"m1@box\",\"object\":\"model\",\"owned_by\":\"box\",\"provider\":\"box\",\"loaded\":false,\"state\":\"remote\",\"capabilities\":[\"chat\"]}", buf.items);

    const st = try p.statusJson(a);
    defer a.free(st);
    try t.expect(std.mem.indexOf(u8, st, "{\"name\":\"cloud\",\"url\":\"https://c/v1\",\"up\":false,\"probed\":false,\"models\":0}") != null);
    try t.expect(std.mem.indexOf(u8, st, "{\"name\":\"box\",\"url\":\"http://b/v1\",\"up\":true,\"probed\":false,\"models\":1}") != null);
}

/// Duck-typed stand-in for server.Conn.
const TestSink = struct {
    alloc: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,
    pub fn writeAll(self: *TestSink, data: []const u8) !void {
        try self.buf.appendSlice(self.alloc, data);
    }
    pub fn peerClosed(_: *TestSink) bool {
        return false;
    }
};

test "providers: proxyChat relays status + content-type, streams the decoded body, swaps in the key" {
    // A tiny HTTP/1.1 upstream on loopback: asserts the Authorization header
    // and the rewritten model, answers chunked SSE — which must reach the
    // client de-chunked under OUR head (curl decodes; forwarding the
    // upstream's Transfer-Encoding would corrupt the client's parse).
    const a = t.allocator;
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const listener = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    try t.expect(listener >= 0);
    defer _ = std.c.close(listener);
    var sa: std.posix.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    try t.expect(std.c.bind(listener, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) == 0);
    var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    try t.expect(std.c.getsockname(listener, @ptrCast(&sa), &len) == 0);
    try t.expect(std.c.listen(listener, 1) == 0);
    const port = std.mem.bigToNative(u16, sa.port);

    const Upstream1 = struct {
        fn run(l: std.posix.fd_t, seen: *std.ArrayList(u8), alloc: std.mem.Allocator) void {
            const c = std.c.accept(l, null, null);
            if (c < 0) return;
            defer _ = std.c.close(c);
            var buf: [8192]u8 = undefined;
            var total: usize = 0;
            while (total < buf.len) {
                const n = std.c.read(c, buf[total..].ptr, buf.len - total);
                if (n <= 0) break;
                total += @intCast(n);
                const head_end = std.mem.indexOf(u8, buf[0..total], "\r\n\r\n") orelse continue;
                const cl = lan.headerValueCI(buf[0..head_end], "content-length") orelse break;
                const want = std.fmt.parseInt(usize, cl, 10) catch break;
                if (total >= head_end + 4 + want) break;
            }
            seen.appendSlice(alloc, buf[0..total]) catch {};
            const resp = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n" ++
                "f\r\ndata: {\"a\":1}\n\n\r\n" ++ "e\r\ndata: [DONE]\n\n\r\n" ++ "0\r\n\r\n";
            _ = std.c.write(c, resp, resp.len);
        }
    };
    var seen: std.ArrayList(u8) = .empty;
    defer seen.deinit(a);
    const th = try std.Thread.spawn(.{}, Upstream1.run, .{ listener, &seen, a });

    const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1", .{port});
    var up = Upstream{ .url = url, .api_key = try a.dupe(u8, "secret-key"), .bare = try a.dupe(u8, "gpt") };
    defer up.deinit(a);
    var sink = TestSink{ .alloc = a };
    defer sink.buf.deinit(a);
    try proxyChat(a, io, up, "{\"model\":\"gpt\",\"stream\":true}", &sink);
    th.join();

    try t.expect(std.mem.indexOf(u8, seen.items, "POST /v1/chat/completions HTTP/1.1") != null);
    try t.expect(std.mem.indexOf(u8, seen.items, "Authorization: Bearer secret-key") != null);
    try t.expect(std.mem.indexOf(u8, seen.items, "{\"model\":\"gpt\",\"stream\":true}") != null);
    try t.expectEqualStrings(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n" ++
            "data: {\"a\":1}\n\ndata: [DONE]\n\n",
        sink.buf.items,
    );
}

test "providers: proxyChat on a dead port is ProviderUnreachable with nothing written" {
    const a = t.allocator;
    var threaded: std.Io.Threaded = .init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    // Bind-then-close: the port is ours a moment ago, so nothing listens there now.
    const s = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    var sa: std.posix.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    _ = std.c.bind(s, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in));
    var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    _ = std.c.getsockname(s, @ptrCast(&sa), &len);
    _ = std.c.close(s);
    const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1", .{std.mem.bigToNative(u16, sa.port)});
    var up = Upstream{ .url = url, .api_key = null, .bare = try a.dupe(u8, "x") };
    defer up.deinit(a);
    var sink = TestSink{ .alloc = a };
    defer sink.buf.deinit(a);
    try t.expectError(error.ProviderUnreachable, proxyChat(a, io, up, "{}", &sink));
    try t.expectEqual(@as(usize, 0), sink.buf.items.len);
}
