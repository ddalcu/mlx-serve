//! OpenAI Responses API helpers.
//!
//! Pure data-handling for `POST /v1/responses` — input-item parsing, tool-shape
//! translation, output-item JSON builders, and the in-memory response store.
//! HTTP plumbing and generation orchestration live in `server.zig`.

const std = @import("std");
const chat_mod = @import("chat.zig");

// ─── small json helpers (intentionally duplicated from server.zig to avoid
// ─── a circular import; identical behavior) ──────────────────────────────

/// Escape into a JSON string literal. Every string here is built from model
/// bytes, and a token is a BPE fragment — see the same chokepoint in server.zig.
pub fn jsonEscape(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    if (!std.unicode.utf8ValidateSlice(input)) {
        const clean = try chat_mod.utf8Sanitize(allocator, input);
        defer allocator.free(clean);
        return jsonEscapeValid(allocator, clean);
    }
    return jsonEscapeValid(allocator, input);
}

fn jsonEscapeValid(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '"');
    for (input) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            0x08 => try buf.appendSlice(allocator, "\\b"),
            0x0C => try buf.appendSlice(allocator, "\\f"),
            0...0x07, 0x0B, 0x0E...0x1F => {
                var hex_buf: [8]u8 = undefined;
                const s = try std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c});
                try buf.appendSlice(allocator, s);
            },
            else => try buf.append(allocator, c),
        }
    }
    try buf.append(allocator, '"');
    return try buf.toOwnedSlice(allocator);
}

pub fn serializeJsonValue(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), value: std.json.Value) !void {
    switch (value) {
        .null => try buf.appendSlice(allocator, "null"),
        .bool => |b| try buf.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| {
            var n: [24]u8 = undefined;
            const s = std.fmt.bufPrint(&n, "{d}", .{i}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .float => |f| {
            var n: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&n, "{d}", .{f}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .string => |s| {
            const e = try jsonEscape(allocator, s);
            defer allocator.free(e);
            try buf.appendSlice(allocator, e);
        },
        .array => |arr| {
            try buf.append(allocator, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try buf.append(allocator, ',');
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

// ─── ID generation ────────────────────────────────────────────────────────

var id_counter: std.atomic.Value(u64) = .{ .raw = 0 };

pub fn makeId(io: std.Io, allocator: std.mem.Allocator, prefix: []const u8) ![]u8 {
    const ms = std.Io.Timestamp.now(io, .real).toMilliseconds();
    const seq = id_counter.fetchAdd(1, .monotonic);
    return std.fmt.allocPrint(allocator, "{s}_{d}_{x}", .{ prefix, ms, seq });
}

// ─── reasoning effort ────────────────────────────────────────────────────

pub const ReasoningConfig = struct {
    enable: bool,
    budget: i32,
    /// The client's raw `reasoning.effort` string, borrowed from the parsed
    /// request JSON — dsv4-family templates map it into the render
    /// (`chat.dsv4EffortFor`); null when the request didn't send one.
    effort: ?[]const u8 = null,
};

/// Map `reasoning.effort` → (enable_thinking, reasoning_budget).
/// "none" is an explicit off, matching the chat and Anthropic surfaces;
/// `null` / non-object → thinking disabled, budget unchanged.
pub fn parseReasoning(reasoning_val: ?std.json.Value, default_budget: i32) ReasoningConfig {
    const v = reasoning_val orelse return .{ .enable = false, .budget = default_budget };
    if (v != .object) return .{ .enable = false, .budget = default_budget };
    const effort_val = v.object.get("effort") orelse return .{ .enable = true, .budget = default_budget };
    if (effort_val != .string) return .{ .enable = true, .budget = default_budget };
    const word = effort_val.string;
    if (std.mem.eql(u8, word, "none")) return .{ .enable = false, .budget = default_budget, .effort = word };
    return .{ .enable = true, .budget = effortBudget(word, default_budget), .effort = word };
}

/// Effort → thinking-budget mapping shared by the Responses `reasoning.effort`
/// object and the chat-completions `reasoning_effort` string. Unknown efforts
/// (model-dependent spec values like "xhigh") fall back to the default budget.
pub fn effortBudget(effort: []const u8, default_budget: i32) i32 {
    if (std.mem.eql(u8, effort, "minimal")) return 1024;
    if (std.mem.eql(u8, effort, "low")) return 2048;
    if (std.mem.eql(u8, effort, "medium")) return 8192;
    return default_budget;
}

// ─── text.format → schema constraint ──────────────────────────────────────

pub const TextFormat = struct {
    /// "text" | "json_object" | "json_schema"
    kind: []const u8,
    /// When kind == "json_schema": the schema value to enforce.
    schema_value: ?std.json.Value,
};

/// Decode the `text` field of a Responses request. Accepts both shapes:
///   • flat (current OpenAI Responses spec):
///       text.format = {type, name, schema, strict}
///   • nested (chat-completions-style, used by some clients/benches):
///       text.format = {type, json_schema: {name, schema, strict}}
/// Returns text-format ("text" by default) and the schema value to enforce.
pub fn parseTextFormat(text_val: ?std.json.Value) TextFormat {
    const v = text_val orelse return .{ .kind = "text", .schema_value = null };
    if (v != .object) return .{ .kind = "text", .schema_value = null };
    const fmt_val = v.object.get("format") orelse return .{ .kind = "text", .schema_value = null };
    if (fmt_val != .object) return .{ .kind = "text", .schema_value = null };
    const t_val = fmt_val.object.get("type") orelse return .{ .kind = "text", .schema_value = null };
    const t = if (t_val == .string) t_val.string else "text";
    // Prefer flat `schema`; fall back to nested `json_schema.schema`.
    var schema = fmt_val.object.get("schema");
    if (schema == null) {
        if (fmt_val.object.get("json_schema")) |js| if (js == .object) {
            schema = js.object.get("schema");
        };
    }
    return .{ .kind = t, .schema_value = schema };
}

/// Decode a chat-completions-style `response_format` field as an alternative to
/// `text.format` on /v1/responses. Some clients/benches send their /v1/chat/
/// completions adapter body to /v1/responses; accept it as an alias to avoid
/// silently dropping the schema constraint.
///   response_format = {type, json_schema: {name, schema, strict}}
///   response_format = {type, schema, name, strict}        (flat, also accepted)
pub fn parseResponseFormatAlias(rf_val: ?std.json.Value) TextFormat {
    const v = rf_val orelse return .{ .kind = "text", .schema_value = null };
    if (v != .object) return .{ .kind = "text", .schema_value = null };
    const t_val = v.object.get("type") orelse return .{ .kind = "text", .schema_value = null };
    const t = if (t_val == .string) t_val.string else "text";
    var schema = v.object.get("schema");
    if (schema == null) {
        if (v.object.get("json_schema")) |js| if (js == .object) {
            schema = js.object.get("schema");
        };
    }
    return .{ .kind = t, .schema_value = schema };
}

pub fn inputContainsFunctionCallOutput(input_val: std.json.Value) bool {
    if (input_val != .array) return false;
    for (input_val.array.items) |item| {
        if (item != .object) continue;
        const t_val = item.object.get("type") orelse continue;
        if (t_val == .string and std.mem.eql(u8, t_val.string, "function_call_output")) return true;
    }
    return false;
}

// ─── tools: Responses (flat) → OpenAI (nested) ───────────────────────────

/// The `(namespace, name)` pair a client resolves a namespaced call by;
/// both borrow from the parsed request (alive for the handler). The keys of
/// `aliases` are the names a model may spell the call with.
pub const NamespaceEntry = struct {
    namespace: []const u8,
    name: []const u8,
};

/// A group whose bare join was already held at declaration, so its declared
/// wire name took a `_N` suffix. `namespace`/`child` borrow the request and
/// `wire` borrows the alias-map key; nothing here is freed. Collisions are
/// rare, so this stays a small linear list rather than a hashed map.
const CollisionEntry = struct { namespace: []const u8, child: []const u8, wire: []const u8 };
const CollisionList = std.ArrayList(CollisionEntry);

/// `call_name -> (namespace, name)` plus the declaration-time collisions needed
/// to resolve a namespaced echo back to the name the current turn declared.
pub const NamespaceAliases = struct {
    aliases: std.StringHashMap(NamespaceEntry),
    collisions: CollisionList,

    pub fn init(allocator: std.mem.Allocator) NamespaceAliases {
        return .{
            .aliases = std.StringHashMap(NamespaceEntry).init(allocator),
            .collisions = .empty,
        };
    }
};

/// Free the owned wire-name keys, then the map and the (borrowed) collisions.
pub fn freeNamespaceAliases(allocator: std.mem.Allocator, aliases: *NamespaceAliases) void {
    var it = aliases.aliases.iterator();
    while (it.next()) |e| allocator.free(e.key_ptr.*);
    aliases.aliases.deinit();
    aliases.collisions.deinit(allocator);
}

/// The wire name for (namespace, child): the measured Codex join, suffixed
/// `_N` if a flat tool (or an earlier member) already holds the name. Returns
/// the owned wire name, registers it in `taken`, and reports whether a suffix
/// was needed (so the caller can record the collision for echo resolution).
fn namespaceWireName(
    allocator: std.mem.Allocator,
    namespace: []const u8,
    child: []const u8,
    taken: *std.StringHashMap(void),
) !struct { wire: []u8, collided: bool } {
    const joined = try joinNsChild(allocator, namespace, child);
    defer allocator.free(joined);
    var wire = try allocator.dupe(u8, joined);
    errdefer allocator.free(wire);
    var collided = false;
    var suffix: usize = 2;
    while (taken.contains(wire)) {
        collided = true;
        // Allocate the next candidate BEFORE freeing the current one: a
        // failure here unwinds the still-owned `wire`, never a freed one.
        const next = try std.fmt.allocPrint(allocator, "{s}_{d}", .{ joined, suffix });
        allocator.free(wire);
        wire = next;
        suffix += 1;
    }
    _ = try taken.put(wire, {});
    return .{ .wire = wire, .collided = collided };
}

/// The measured Codex join rule: trailing `_` off the namespace, leading `_`
/// off the child, joined by `__`.
fn joinNsChild(allocator: std.mem.Allocator, namespace: []const u8, child: []const u8) ![]u8 {
    var ns = namespace;
    while (ns.len > 0 and ns[ns.len - 1] == '_') ns = ns[0 .. ns.len - 1];
    var ch = child;
    while (ch.len > 0 and ch[0] == '_') ch = ch[1..ch.len];
    return std.fmt.allocPrint(allocator, "{s}__{s}", .{ ns, ch });
}

/// Write one tool as the nested OpenAI form under `name` (which may differ from
/// the tool's own name when a namespace member is expanded to a wire name).
fn appendNestedFunctionTool(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    tool: std.json.ObjectMap,
    name: []const u8,
) !void {
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
    if (tool.get("parameters")) |params_val| {
        try serializeJsonValue(allocator, buf, params_val);
    } else {
        try buf.appendSlice(allocator, "{}");
    }
    try buf.appendSlice(allocator, "}}");
}

/// Re-shape Responses tools (`{type:"function", name, parameters, description}`)
/// into the nested OpenAI form (`{type:"function", function:{name, parameters,
/// description}}`) that `chat_mod.formatChat` expects. Returns owned JSON.
///
/// A `namespace` tool is a container of ordinary client-executed function tools
/// (the shape Codex uses to wrap each MCP server), so its members are expanded
/// under a joined wire name instead of being skipped whole. When `aliases` is
/// given it is filled with `call_name -> (namespace, name)` so a resulting call
/// can be returned with its namespace intact. A model that drops the namespace
/// prefix (the history echoes the bare child name back at it) is answered too:
/// we register a unique bare child name as an alias next to its wire name.
pub fn buildToolsJson(
    allocator: std.mem.Allocator,
    tools_array: std.json.Array,
    aliases: ?*NamespaceAliases,
) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    // A null `aliases` still gets a map so every expanded wire name is owned and
    // freed exactly once here; `taken` borrows those keys, so they are freed only
    // after `taken` is deinited (defer LIFO).
    var owned_aliases = NamespaceAliases.init(allocator);
    const aliases_ref: *NamespaceAliases = if (aliases) |a| a else &owned_aliases;
    defer freeNamespaceAliases(allocator, &owned_aliases);

    // Flat function names are reserved first so an expanded member never
    // shadows a tool the client already resolves by its bare name.
    var taken = std.StringHashMap(void).init(allocator);
    defer taken.deinit();
    var child_counts = std.StringHashMap(u32).init(allocator);
    defer child_counts.deinit();
    for (tools_array.items) |tool_val| {
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        const t = if (tool.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
        if (std.mem.eql(u8, t, "namespace")) {
            const members = if (tool.get("tools")) |tv| (if (tv == .array) tv.array.items else null) else null;
            for (members orelse continue) |member_val| {
                if (member_val != .object) continue;
                const member = member_val.object;
                const mt = if (member.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
                if (!std.mem.eql(u8, mt, "function")) continue;
                const child = if (member.get("name")) |v| (if (v == .string) v.string else "") else "";
                if (child.len == 0) continue;
                const n = child_counts.get(child) orelse 0;
                try child_counts.put(child, n + 1);
            }
            continue;
        }
        if (!std.mem.eql(u8, t, "function")) continue;
        if (tool.get("name")) |v| if (v == .string and v.string.len > 0) {
            _ = try taken.put(v.string, {});
        };
    }

    try buf.append(allocator, '[');
    var emitted: usize = 0;
    for (tools_array.items) |tool_val| {
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        const t = if (tool.get("type")) |tv| (if (tv == .string) tv.string else "") else "";

        if (std.mem.eql(u8, t, "namespace")) {
            const ns_name = if (tool.get("name")) |v| (if (v == .string) v.string else "") else "";
            if (ns_name.len == 0) continue;
            const members = if (tool.get("tools")) |tv| (if (tv == .array) tv.array.items else null) else null;
            const members_items = members orelse continue;
            for (members_items) |member_val| {
                if (member_val != .object) continue;
                const member = member_val.object;
                const mt = if (member.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
                if (!std.mem.eql(u8, mt, "function")) continue;
                const child = if (member.get("name")) |v| (if (v == .string) v.string else "") else "";
                if (child.len == 0) continue;
                const ns_wire = try namespaceWireName(allocator, ns_name, child, &taken);
                const wire = ns_wire.wire;
                // `wire` is in `taken` (borrowed) and handed to the alias map below,
                // which frees it after `taken` is gone; on a handoff error it is still
                // ours, so free it here. (Same shape as the `bare` registration.)
                var wire_owned = true;
                defer if (wire_owned) allocator.free(wire);
                if (emitted > 0) try buf.append(allocator, ',');
                emitted += 1;
                try appendNestedFunctionTool(allocator, &buf, member, wire);
                {
                    const al = aliases_ref;
                    // Put before any later fallible step; the map owns the key
                    // on success, so error paths free it (never a live key).
                    try al.aliases.put(wire, .{ .namespace = ns_name, .name = child });
                    wire_owned = false;
                    // Record a group whose bare join was already held so an
                    // echo of that join resolves to the declared name. Keep the
                    // first declaration; the wire name borrows the map key.
                    if (ns_wire.collided) {
                        var declared = false;
                        for (al.collisions.items) |c| {
                            if (std.mem.eql(u8, c.namespace, ns_name) and std.mem.eql(u8, c.child, child)) {
                                declared = true;
                                break;
                            }
                        }
                        if (!declared) try al.collisions.append(allocator, .{
                            .namespace = ns_name,
                            .child = child,
                            .wire = wire,
                        });
                    }
                    // A bare child name seen in one namespace only is how the
                    // model often spells the call; register it too. If a flat
                    // tool holds it already, or two namespaces share it, leave
                    // it to the flat/client resolution.
                    const unique = (child_counts.get(child) orelse 0) == 1;
                    if (unique and !taken.contains(child)) {
                        const bare = try allocator.dupe(u8, child);
                        var bare_owned = true;
                        defer if (bare_owned) allocator.free(bare);
                        try taken.put(bare, {});
                        try al.aliases.put(bare, .{ .namespace = ns_name, .name = child });
                        bare_owned = false;
                    }
                }
            }
            continue;
        }

        // Only function tools are supported locally (web_search/file_search/computer_use are not)
        if (!std.mem.eql(u8, t, "function")) continue;
        const name = if (tool.get("name")) |v| (if (v == .string) v.string else "") else "";

        if (emitted > 0) try buf.append(allocator, ',');
        emitted += 1;
        try appendNestedFunctionTool(allocator, &buf, tool, name);
    }
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

/// Restore `(namespace, name)` for a call made by wire name or by a uniquely
/// registered bare child name. Flat tools and names the model invented return
/// `null` (emit a bare name, no namespace).
pub fn splitNamespaceToolName(aliases: ?*const NamespaceAliases, name: []const u8) ?NamespaceEntry {
    const al = aliases orelse return null;
    return al.aliases.get(name);
}

/// The name the current turn declared for (namespace, child), or null when the
/// group is not declared. Returns a slice borrowed from a live alias/collision
/// entry (alive for the handler): the bare join when it was free, else the `_N`
/// suffix recorded at declaration. One join is built only for the lookup.
fn declaredWireName(
    allocator: std.mem.Allocator,
    aliases: *const NamespaceAliases,
    namespace: []const u8,
    child: []const u8,
) ?[]const u8 {
    const joined = joinNsChild(allocator, namespace, child) catch return null;
    defer allocator.free(joined);
    if (aliases.aliases.get(joined)) |e| {
        if (std.mem.eql(u8, e.namespace, namespace) and std.mem.eql(u8, e.name, child))
            return aliases.aliases.getKeyPtr(joined).?.*;
    }
    for (aliases.collisions.items) |c| {
        if (std.mem.eql(u8, c.namespace, namespace) and std.mem.eql(u8, c.child, child))
            return c.wire;
    }
    return null;
}

// ─── output-item JSON builders ────────────────────────────────────────────

pub fn appendOutputTextMessage(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    item_id: []const u8,
    text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, text);
    defer allocator.free(esc_text);
    try buf.appendSlice(allocator, "{\"type\":\"message\",\"id\":");
    try buf.appendSlice(allocator, esc_id);
    try buf.appendSlice(allocator, ",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":");
    try buf.appendSlice(allocator, esc_text);
    try buf.appendSlice(allocator, ",\"annotations\":[]}]}");
}

pub fn appendReasoningItem(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    item_id: []const u8,
    summary_text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, summary_text);
    defer allocator.free(esc_text);
    try buf.appendSlice(allocator, "{\"type\":\"reasoning\",\"id\":");
    try buf.appendSlice(allocator, esc_id);
    try buf.appendSlice(allocator, ",\"summary\":[{\"type\":\"summary_text\",\"text\":");
    try buf.appendSlice(allocator, esc_text);
    try buf.appendSlice(allocator, "}]}");
}

pub fn appendFunctionCallItem(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    item_id: []const u8,
    call_id: []const u8,
    name: []const u8,
    arguments_json: []const u8,
    namespace: ?[]const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_call = try jsonEscape(allocator, call_id);
    defer allocator.free(esc_call);
    const esc_name = try jsonEscape(allocator, name);
    defer allocator.free(esc_name);
    const esc_args = try jsonEscape(allocator, arguments_json);
    defer allocator.free(esc_args);
    try buf.appendSlice(allocator, "{\"type\":\"function_call\",\"id\":");
    try buf.appendSlice(allocator, esc_id);
    try buf.appendSlice(allocator, ",\"call_id\":");
    try buf.appendSlice(allocator, esc_call);
    try buf.appendSlice(allocator, ",\"name\":");
    try buf.appendSlice(allocator, esc_name);
    if (namespace) |ns| {
        const esc_ns = try jsonEscape(allocator, ns);
        defer allocator.free(esc_ns);
        try buf.appendSlice(allocator, ",\"namespace\":");
        try buf.appendSlice(allocator, esc_ns);
    }
    try buf.appendSlice(allocator, ",\"arguments\":");
    try buf.appendSlice(allocator, esc_args);
    try buf.appendSlice(allocator, ",\"status\":\"completed\"}");
}

// ─── input-item parser ────────────────────────────────────────────────────

/// Owns parsed messages and their backing buffers. Free with `deinit`.
pub const ParsedInput = struct {
    messages: std.ArrayList(chat_mod.Message),
    /// Owned heap allocations backing message fields (tool_calls slices, image
    /// pixel bufs, concatenated content, etc.). Not arena-allocated because
    /// some pieces (image pixels) are freed by other paths.
    owned_strings: std.ArrayList([]const u8),
    owned_tool_calls: std.ArrayList([]chat_mod.ToolCall),
    owned_images: std.ArrayList([]chat_mod.ImageData),
    allocator: std.mem.Allocator,
    image_decode_failed: bool = false,

    pub fn deinit(self: *ParsedInput) void {
        for (self.owned_strings.items) |s| self.allocator.free(s);
        for (self.owned_tool_calls.items) |tcs| self.allocator.free(tcs);
        for (self.owned_images.items) |imgs| {
            for (imgs) |img| self.allocator.free(img.pixels);
            self.allocator.free(imgs);
        }
        self.owned_strings.deinit(self.allocator);
        self.owned_tool_calls.deinit(self.allocator);
        self.owned_images.deinit(self.allocator);
        self.messages.deinit(self.allocator);
    }
};

/// Decode a single image_url string into preprocessed pixels. Provided as a
/// callback because the actual decoder lives in `server.zig` (uses stb_image
/// + libwebp). Returns whether it appended anything; a false is recorded as
/// `image_decode_failed` so the surface can refuse the request by name.
/// Appends one entry per tower call an `image_url` expands into — usually one,
/// but LFM2-VL splits a large source into tiles plus a thumbnail. Appending
/// rather than returning is what lets a single URL produce several.
pub const ImageUrlDecoder = *const fn (
    allocator: std.mem.Allocator,
    list: *std.ArrayList(chat_mod.ImageData),
    url: []const u8,
    vp: chat_mod.VisionPreproc,
) bool;

/// Translate a Responses `input` value (string or array of input items) into
/// `chat_mod.Message`s. Optionally prepends `instructions` as the single leading
/// `system` msg. If `previous_messages` already contains a stored system message
/// and fresh instructions are provided, the fresh instructions replace it so
/// templates like Qwen's never see a non-leading/duplicate system message.
/// `previous_messages` are deep-referenced (not copied) into the result if
/// non-null — caller must keep them alive.
/// `namespace_aliases` (from `buildToolsJson`) rewrites echoed `function_call`
/// items — which carry the namespace Codex declared, not the wire name — so
/// the history shows the same name the current turn declares.
pub fn parseInput(
    allocator: std.mem.Allocator,
    input_val: std.json.Value,
    instructions: ?[]const u8,
    previous_messages: ?[]const chat_mod.Message,
    namespace_aliases: ?*const NamespaceAliases,
    image_decoder: ?ImageUrlDecoder,
    vp: chat_mod.VisionPreproc,
) !ParsedInput {
    var pi: ParsedInput = .{
        .messages = std.ArrayList(chat_mod.Message).empty,
        .owned_strings = std.ArrayList([]const u8).empty,
        .owned_tool_calls = std.ArrayList([]chat_mod.ToolCall).empty,
        .owned_images = std.ArrayList([]chat_mod.ImageData).empty,
        .allocator = allocator,
    };
    errdefer pi.deinit();

    const fresh_instructions = if (instructions) |ins| (if (ins.len > 0) ins else null) else null;
    if (fresh_instructions) |ins| {
        try pi.messages.append(allocator, .{
            .role = "system",
            .content = ins,
        });
    }

    if (previous_messages) |prev| {
        for (prev) |m| {
            if (fresh_instructions != null and std.mem.eql(u8, m.role, "system")) {
                continue;
            }
            try pi.messages.append(allocator, m);
        }
    }

    switch (input_val) {
        .string => |s| {
            try pi.messages.append(allocator, .{ .role = "user", .content = s });
        },
        .array => |arr| {
            for (arr.items) |item| {
                if (item != .object) continue;
                const obj = item.object;
                const t_val = obj.get("type") orelse {
                    // Bare {role, content} (some clients omit "type":"message")
                    try appendMessageItem(allocator, &pi, obj, image_decoder, vp);
                    continue;
                };
                if (t_val != .string) continue;
                const t = t_val.string;
                if (std.mem.eql(u8, t, "message")) {
                    try appendMessageItem(allocator, &pi, obj, image_decoder, vp);
                } else if (std.mem.eql(u8, t, "function_call")) {
                    try appendFunctionCallInputItem(allocator, &pi, obj, namespace_aliases);
                } else if (std.mem.eql(u8, t, "function_call_output")) {
                    try appendFunctionCallOutputItem(allocator, &pi, obj);
                } else if (std.mem.eql(u8, t, "reasoning")) {
                    // Drop on input — model regenerates its own reasoning.
                    continue;
                } else if (std.mem.eql(u8, t, "compaction")) {
                    appendCompactionInputItem(allocator, &pi, obj) catch {};
                } else {
                    // Unknown item type — skip silently.
                    continue;
                }
            }
        },
        else => {},
    }

    // Templates we serve require the system turn first; fold any system past
    // index 0 into the leading one — the same unconditional fold /v1/messages
    // applies — so the native template renders a multi-system Responses input.
    if (try chat_mod.foldSystemMessages(allocator, &pi.messages)) |joined| {
        errdefer allocator.free(joined);
        try pi.owned_strings.append(allocator, joined);
    }

    return pi;
}

/// Join the text parts (`input_text`/`text`/`output_text`) of a Responses
/// content array into `dest` in order, newline-separated; empty parts are
/// skipped, matching the `joinedTextParts` convention in server.zig.
fn appendTextParts(
    allocator: std.mem.Allocator,
    dest: *std.ArrayList(u8),
    parts: []const std.json.Value,
) !void {
    for (parts) |part| {
        if (part != .object) continue;
        const pt_val = part.object.get("type") orelse continue;
        if (pt_val != .string) continue;
        const pt = pt_val.string;
        if (!std.mem.eql(u8, pt, "input_text") and !std.mem.eql(u8, pt, "text") and !std.mem.eql(u8, pt, "output_text")) continue;
        const tx = part.object.get("text") orelse continue;
        if (tx != .string or tx.string.len == 0) continue;
        if (dest.items.len > 0) try dest.append(allocator, '\n');
        try dest.appendSlice(allocator, tx.string);
    }
}

fn appendMessageItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
    image_decoder: ?ImageUrlDecoder,
    vp: chat_mod.VisionPreproc,
) !void {
    const role_val = obj.get("role") orelse return;
    if (role_val != .string) return;
    const role = chat_mod.canonicalRole(role_val.string);

    const content_val = obj.get("content") orelse return;
    var content: []const u8 = "";
    var images: ?[]chat_mod.ImageData = null;

    switch (content_val) {
        .string => |s| content = s,
        .array => |arr| {
            var text_parts = std.ArrayList(u8).empty;
            defer text_parts.deinit(allocator);
            var image_list = std.ArrayList(chat_mod.ImageData).empty;
            var declared_images: usize = 0;
            errdefer {
                for (image_list.items) |img| allocator.free(img.pixels);
                image_list.deinit(allocator);
            }
            try appendTextParts(allocator, &text_parts, arr.items);
            for (arr.items) |part| {
                if (part != .object) continue;
                const pt_val = part.object.get("type") orelse continue;
                if (pt_val != .string) continue;
                if (!std.mem.eql(u8, pt_val.string, "input_image")) continue;
                declared_images += 1;
                const url_val = part.object.get("image_url") orelse continue;
                const url = switch (url_val) {
                    .string => |s| s,
                    .object => |io| if (io.get("url")) |u| (if (u == .string) u.string else continue) else continue,
                    else => continue,
                };
                if (image_decoder) |dec| if (!dec(allocator, &image_list, url, vp)) {
                    pi.image_decode_failed = true;
                };
            }
            if (vp.mode == .qwen and declared_images != image_list.items.len) return error.InvalidImage;
            if (text_parts.items.len > 0) {
                const owned = try allocator.dupe(u8, text_parts.items);
                try pi.owned_strings.append(allocator, owned);
                content = owned;
            }
            if (image_list.items.len > 0) {
                const owned = try image_list.toOwnedSlice(allocator);
                try pi.owned_images.append(allocator, owned);
                images = owned;
            } else {
                image_list.deinit(allocator);
            }
        },
        else => {},
    }

    if (content.len == 0 and images == null) return;
    try pi.messages.append(allocator, .{
        .role = role,
        .content = content,
        .images = if (images) |im| im else null,
    });
}

fn appendFunctionCallInputItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
    namespace_aliases: ?*const NamespaceAliases,
) !void {
    const call_id = if (obj.get("call_id")) |v| (if (v == .string) v.string else "") else "";
    var name: []const u8 = if (obj.get("name")) |v| (if (v == .string) v.string else "") else "";
    const args = if (obj.get("arguments")) |v| (if (v == .string) v.string else "{}") else "{}";

    // A namespaced echo (`name`=child + `namespace`=declared namespace) is
    // rendered as the name the current turn declares. The declared name is
    // borrowed from a live alias entry, so it is not owned here.
    if (namespace_aliases) |al| {
        if (obj.get("namespace")) |nv| if (nv == .string and nv.string.len > 0 and name.len > 0) {
            if (declaredWireName(allocator, al, nv.string, name)) |declared| name = declared;
        };
    }

    const tcs = try allocator.alloc(chat_mod.ToolCall, 1);
    errdefer allocator.free(tcs);
    tcs[0] = .{ .id = call_id, .name = name, .arguments = args };
    try pi.owned_tool_calls.append(allocator, tcs);
    try pi.messages.append(allocator, .{
        .role = "assistant",
        .content = "",
        .tool_calls = tcs,
    });
}

fn appendFunctionCallOutputItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
) !void {
    const call_id = if (obj.get("call_id")) |v| (if (v == .string) v.string else "") else "";
    const output_val = obj.get("output");
    var output: []const u8 = "";
    if (output_val) |v| {
        switch (v) {
            .string => |s| output = s,
            // Codex echoes `output` as an array of content parts; join its text
            // parts in order the same way message content is joined.
            .array => |arr| {
                var parts = std.ArrayList(u8).empty;
                defer parts.deinit(allocator);
                try appendTextParts(allocator, &parts, arr.items);
                if (parts.items.len > 0) {
                    const owned = try allocator.dupe(u8, parts.items);
                    try pi.owned_strings.append(allocator, owned);
                    output = owned;
                }
            },
            else => {},
        }
    }
    try pi.messages.append(allocator, .{
        .role = "tool",
        .content = output,
        .tool_call_id = call_id,
    });
}

// ─── compaction (round-trippable opaque blob) ────────────────────────────
//
// The OpenAI Responses spec treats `encrypted_content` as opaque, provider-
// defined data. We synthesize a self-describing blob: base64-encoded JSON of
// `{v:1, msgs:[{role, content}, ...]}`. No LLM call is required — the message
// list is a faithful (lossy on tool-calls / images) snapshot of the resolved
// input that round-trips back into a fresh response.create as `input`.

/// Encode a sequence of messages as a compaction blob (base64 over JSON).
/// Caller owns the returned slice. Tool calls / images are dropped — the blob
/// only carries text-form turns. That matches the spec's "summarized" framing
/// while staying self-contained.
pub fn encodeCompactionBlob(allocator: std.mem.Allocator, messages: []const chat_mod.Message) ![]u8 {
    var json_buf = std.ArrayList(u8).empty;
    defer json_buf.deinit(allocator);
    try json_buf.appendSlice(allocator, "{\"v\":1,\"msgs\":[");
    var emitted: usize = 0;
    for (messages) |m| {
        // Skip empty turns and tool-role messages (no faithful round-trip path).
        if (m.content.len == 0) continue;
        if (emitted > 0) try json_buf.append(allocator, ',');
        emitted += 1;
        const esc_role = try jsonEscape(allocator, m.role);
        defer allocator.free(esc_role);
        const esc_content = try jsonEscape(allocator, m.content);
        defer allocator.free(esc_content);
        try json_buf.appendSlice(allocator, "{\"role\":");
        try json_buf.appendSlice(allocator, esc_role);
        try json_buf.appendSlice(allocator, ",\"content\":");
        try json_buf.appendSlice(allocator, esc_content);
        try json_buf.append(allocator, '}');
    }
    try json_buf.appendSlice(allocator, "]}");

    const enc = std.base64.standard.Encoder;
    const out_len = enc.calcSize(json_buf.items.len);
    const out = try allocator.alloc(u8, out_len);
    _ = enc.encode(out, json_buf.items);
    return out;
}

fn appendCompactionInputItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
) !void {
    const enc_val = obj.get("encrypted_content") orelse return;
    if (enc_val != .string) return;
    const enc_str = enc_val.string;
    if (enc_str.len == 0) return;

    const dec = std.base64.standard.Decoder;
    const dec_len = dec.calcSizeForSlice(enc_str) catch return;
    const decoded = try allocator.alloc(u8, dec_len);
    defer allocator.free(decoded);
    dec.decode(decoded, enc_str) catch return;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, decoded, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .object) return;
    const root = parsed.value.object;
    const v_val = root.get("v") orelse return;
    if (v_val != .integer or v_val.integer != 1) return;
    const msgs_val = root.get("msgs") orelse return;
    if (msgs_val != .array) return;

    for (msgs_val.array.items) |m_val| {
        if (m_val != .object) continue;
        const role_val = m_val.object.get("role") orelse continue;
        if (role_val != .string) continue;
        const content_val = m_val.object.get("content") orelse continue;
        if (content_val != .string) continue;

        // Inner JSON values are owned by `parsed` (freed at scope end).
        // Dupe both fields so they outlive this function.
        const role_owned = try allocator.dupe(u8, role_val.string);
        try pi.owned_strings.append(allocator, role_owned);
        const content_owned = try allocator.dupe(u8, content_val.string);
        try pi.owned_strings.append(allocator, content_owned);

        try pi.messages.append(allocator, .{
            .role = role_owned,
            .content = content_owned,
        });
    }
}

// ─── tool_choice → instruction string ────────────────────────────────────

pub const ToolChoice = struct {
    /// When false, tools are dropped from the request entirely.
    include_tools: bool,
    /// Owned by the caller (free with allocator) when non-null.
    instruction: ?[]const u8,
};

pub fn parseToolChoice(allocator: std.mem.Allocator, choice_val: ?std.json.Value) !ToolChoice {
    const v = choice_val orelse return .{ .include_tools = true, .instruction = null };
    switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "none")) return .{ .include_tools = false, .instruction = null };
            if (std.mem.eql(u8, s, "required")) {
                const ins = try allocator.dupe(u8, "\nYou MUST call one of the available functions. Do not respond with text.");
                return .{ .include_tools = true, .instruction = ins };
            }
            return .{ .include_tools = true, .instruction = null }; // "auto" default
        },
        .object => |obj| {
            const t = if (obj.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
            if (!std.mem.eql(u8, t, "function")) return .{ .include_tools = true, .instruction = null };
            const name = if (obj.get("name")) |nv| (if (nv == .string) nv.string else "") else "";
            if (name.len == 0) return .{ .include_tools = true, .instruction = null };
            const ins = try std.fmt.allocPrint(allocator, "\nYou MUST call the function \"{s}\". Do not respond with text.", .{name});
            return .{ .include_tools = true, .instruction = ins };
        },
        else => return .{ .include_tools = true, .instruction = null },
    }
}

// ─── in-memory response store ────────────────────────────────────────────

pub const StoredResponse = struct {
    id: []u8,
    created_at: i64,
    model: []u8,
    status: []u8, // "completed" | "failed" | "incomplete"
    /// Pre-rendered final response JSON envelope (full body returned by
    /// `GET /v1/responses/{id}`). Owned by the arena.
    body_json: []u8,
    /// Snapshot of input + assistant messages used to produce this response.
    /// Used when a later request supplies `previous_response_id` — the saved
    /// messages are concatenated in front of the new input items. Owned by
    /// the arena (including all inner []const u8 slices and tool_calls).
    history: []chat_mod.Message,
    /// Pixel buffers are deliberately not copied into this count-bounded
    /// store. An incomplete history must never become a blind continuation.
    media_omitted: bool = false,

    arena: std.heap.ArenaAllocator,

    list_node: std.DoublyLinkedList.Node = .{},

    pub fn continuationHistory(self: *const StoredResponse) ![]const chat_mod.Message {
        if (self.media_omitted) return error.IncompleteMediaHistory;
        return self.history;
    }

    pub fn deinit(self: *StoredResponse) void {
        var arena = self.arena;
        const gpa = arena.child_allocator;
        arena.deinit();
        gpa.destroy(self);
    }
};

pub const ResponseStore = struct {
    mu: std.Io.Mutex = .init,
    map: std.StringHashMapUnmanaged(*StoredResponse) = .{},
    lru: std.DoublyLinkedList = .{},
    cap: usize,
    gpa: std.mem.Allocator,
    io: std.Io,

    pub fn init(io: std.Io, gpa: std.mem.Allocator, cap: usize) ResponseStore {
        return .{ .io = io, .gpa = gpa, .cap = cap };
    }

    pub fn deinit(self: *ResponseStore) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        var node = self.lru.first;
        while (node) |n| {
            const next = n.next;
            const sr: *StoredResponse = @fieldParentPtr("list_node", n);
            sr.deinit();
            node = next;
        }
        self.map.deinit(self.gpa);
        self.lru = .{};
    }

    /// Take ownership of `sr`. Evicts the LRU tail if at capacity.
    /// `sr.id` must already be set.
    pub fn put(self: *ResponseStore, sr: *StoredResponse) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);

        // If id already exists, evict the old entry first
        if (self.map.fetchRemove(sr.id)) |kv| {
            const old = kv.value;
            self.lru.remove(&old.list_node);
            old.deinit();
        }

        if (self.map.count() >= self.cap) {
            // Evict LRU tail
            if (self.lru.last) |tail_node| {
                const tail: *StoredResponse = @fieldParentPtr("list_node", tail_node);
                _ = self.map.remove(tail.id);
                self.lru.remove(tail_node);
                tail.deinit();
            }
        }

        try self.map.put(self.gpa, sr.id, sr);
        self.lru.prepend(&sr.list_node);
    }

    /// Returns a borrowed reference (do not free). Touches LRU.
    pub fn get(self: *ResponseStore, id: []const u8) ?*StoredResponse {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const sr = self.map.get(id) orelse return null;
        self.lru.remove(&sr.list_node);
        self.lru.prepend(&sr.list_node);
        return sr;
    }

    /// Returns true if removed.
    pub fn delete(self: *ResponseStore, id: []const u8) bool {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const kv = self.map.fetchRemove(id) orelse return false;
        self.lru.remove(&kv.value.list_node);
        kv.value.deinit();
        return true;
    }
};

// ─── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

test "parseReasoning maps effort levels" {
    const v_low = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"effort\":\"low\"}", .{});
    defer v_low.deinit();
    try testing.expectEqual(@as(i32, 2048), parseReasoning(v_low.value, -1).budget);

    // high is uncapped: the default budget rides through.
    const v_high = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"effort\":\"high\"}", .{});
    defer v_high.deinit();
    try testing.expectEqual(@as(i32, -1), parseReasoning(v_high.value, -1).budget);

    try testing.expectEqual(false, parseReasoning(null, -1).enable);
    try testing.expectEqual(@as(i32, -1), parseReasoning(null, -1).budget);
}

// `none` is the OpenAI/gpt-5.1 spelling of an explicit thinking-off on the chat
// and Anthropic surfaces; Responses must agree, not treat a present effort as on.
test "parseReasoning: effort none is an explicit thinking-off" {
    const v = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"effort\":\"none\"}", .{});
    defer v.deinit();
    const cfg = parseReasoning(v.value, -1);
    try testing.expectEqual(false, cfg.enable);
    try testing.expectEqualStrings("none", cfg.effort.?);
}

test "parseTextFormat extracts schema from flat shape" {
    const json =
        \\{"format":{"type":"json_schema","name":"x","schema":{"type":"object"}}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const tf = parseTextFormat(parsed.value);
    try testing.expectEqualStrings("json_schema", tf.kind);
    try testing.expect(tf.schema_value != null);
}

test "parseTextFormat default is text" {
    const tf = parseTextFormat(null);
    try testing.expectEqualStrings("text", tf.kind);
    try testing.expect(tf.schema_value == null);
}

test "parseTextFormat extracts schema from nested json_schema shape" {
    const json =
        \\{"format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"},"strict":true}}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const tf = parseTextFormat(parsed.value);
    try testing.expectEqualStrings("json_schema", tf.kind);
    try testing.expect(tf.schema_value != null);
}

test "parseResponseFormatAlias accepts chat-style nested shape" {
    const json =
        \\{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"},"strict":true}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const tf = parseResponseFormatAlias(parsed.value);
    try testing.expectEqualStrings("json_schema", tf.kind);
    try testing.expect(tf.schema_value != null);
}

test "parseResponseFormatAlias accepts flat shape too" {
    const json =
        \\{"type":"json_schema","name":"x","schema":{"type":"object"},"strict":true}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const tf = parseResponseFormatAlias(parsed.value);
    try testing.expectEqualStrings("json_schema", tf.kind);
    try testing.expect(tf.schema_value != null);
}

test "parseResponseFormatAlias default is text" {
    const tf = parseResponseFormatAlias(null);
    try testing.expectEqualStrings("text", tf.kind);
    try testing.expect(tf.schema_value == null);
}

test "inputContainsFunctionCallOutput detects tool result items" {
    const json =
        \\[
        \\  {"type":"message","role":"user","content":"hi"},
        \\  {"type":"function_call_output","call_id":"call_1","output":"{}"}
        \\]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    try testing.expect(inputContainsFunctionCallOutput(parsed.value));
    try testing.expect(!inputContainsFunctionCallOutput(.{ .string = "hi" }));
}

test "buildToolsJson nests Responses-shape into OpenAI-shape" {
    const json =
        \\[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const out = try buildToolsJson(testing.allocator, parsed.value.array, null);
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "\"function\":{\"name\":\"get_weather\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"parameters\":{") != null);
    // Make sure top-level "type" wraps it
    try testing.expect(std.mem.startsWith(u8, out, "[{\"type\":\"function\""));
}

test "buildToolsJson skips non-function tools" {
    const json =
        \\[{"type":"web_search"},{"type":"function","name":"f","parameters":{}}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const out = try buildToolsJson(testing.allocator, parsed.value.array, null);
    defer testing.allocator.free(out);
    // Only the function tool should be emitted, no leading comma
    try testing.expect(std.mem.startsWith(u8, out, "[{"));
    try testing.expect(std.mem.indexOf(u8, out, "web_search") == null);
}

test "buildToolsJson expands namespace groups under wire names" {
    const json =
        \\[{"type":"namespace","name":"mcp__demo__","description":"Demo","tools":[{"type":"function","name":"get_weather","description":"W","parameters":{"type":"object","required":["city"]}},{"type":"function","name":"get_time"}]}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    var aliases = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases);
    const out = try buildToolsJson(testing.allocator, parsed.value.array, &aliases);
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "\"function\":{\"name\":\"mcp__demo__get_weather\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"function\":{\"name\":\"mcp__demo__get_time\"") != null);
    // Wire name joins namespace + child (the measured Codex join); the
    // returned namespace keeps the client's original spelling, trailing '_'
    // included.
    try testing.expectEqualStrings("mcp__demo__", splitNamespaceToolName(&aliases, "mcp__demo__get_weather").?.namespace);
    try testing.expectEqualStrings("get_weather", splitNamespaceToolName(&aliases, "mcp__demo__get_weather").?.name);
}

test "buildToolsJson wire name survives a flat collision" {
    const json =
        \\[{"type":"function","name":"mcp__demo__get_weather"},{"type":"namespace","name":"mcp__demo__","tools":[{"type":"function","name":"get_weather"}]}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    var aliases = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases);
    const out = try buildToolsJson(testing.allocator, parsed.value.array, &aliases);
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "\"name\":\"mcp__demo__get_weather_2\"") != null);
    try testing.expect(splitNamespaceToolName(&aliases, "mcp__demo__get_weather") == null);
    try testing.expectEqualStrings("get_weather", splitNamespaceToolName(&aliases, "mcp__demo__get_weather_2").?.name);
}

test "splitNamespaceToolName passes flat and invented names through" {
    try testing.expect(splitNamespaceToolName(null, "get_weather") == null);
    var aliases = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases);
    try testing.expect(splitNamespaceToolName(&aliases, "get_weather") == null);
}

test "buildToolsJson registers a unique bare child name as alias" {
    const json =
        \\[{"type":"namespace","name":"mcp__demo__","tools":[{"type":"function","name":"search"}]}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    var aliases = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases);
    const out = try buildToolsJson(testing.allocator, parsed.value.array, &aliases);
    defer testing.allocator.free(out);
    // The model often drops the namespace prefix; the bare call still returns namespaced.
    const e = splitNamespaceToolName(&aliases, "search").?;
    try testing.expectEqualStrings("mcp__demo__", e.namespace);
    try testing.expectEqualStrings("search", e.name);
}

test "buildToolsJson skips the bare alias on ambiguity or flat collision" {
    const ambiguous =
        \\[{"type":"namespace","name":"a","tools":[{"type":"function","name":"go"}]},{"type":"namespace","name":"b","tools":[{"type":"function","name":"go"}]}]
    ;
    const parsed_a = try std.json.parseFromSlice(std.json.Value, testing.allocator, ambiguous, .{});
    defer parsed_a.deinit();
    var aliases_a = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases_a);
    const out_a = try buildToolsJson(testing.allocator, parsed_a.value.array, &aliases_a);
    defer testing.allocator.free(out_a);
    try testing.expect(splitNamespaceToolName(&aliases_a, "go") == null);
    try testing.expect(splitNamespaceToolName(&aliases_a, "a__go") != null);
    try testing.expect(splitNamespaceToolName(&aliases_a, "b__go") != null);

    const colliding =
        \\[{"type":"function","name":"go"},{"type":"namespace","name":"ns","tools":[{"type":"function","name":"go"}]}]
    ;
    const parsed_c = try std.json.parseFromSlice(std.json.Value, testing.allocator, colliding, .{});
    defer parsed_c.deinit();
    var aliases_c = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases_c);
    const out_c = try buildToolsJson(testing.allocator, parsed_c.value.array, &aliases_c);
    defer testing.allocator.free(out_c);
    // The flat tool owns the bare name; only the wire form is namespaced.
    try testing.expect(splitNamespaceToolName(&aliases_c, "go") == null);
    try testing.expect(splitNamespaceToolName(&aliases_c, "ns__go") != null);
}

// Fixed-pool, LIFO free-list allocator: a freed <=64B slot is reused by the next
// same-size alloc. SafeAllocator never reuses a just-freed slot, so only this
// reproduces the `taken` use-after-free.
const SlotAllocator = struct {
    const Self = @This();
    const SLOT = 64;
    const NSLOT = 1024;
    const NFREE = 1024;
    pool: [SLOT * NSLOT]u8,
    bump: usize,
    free_stack: [NFREE]usize,
    free_top: usize,
    const Allocator = std.mem.Allocator;
    const Alignment = std.mem.Alignment;
    fn init() Self {
        return .{ .pool = undefined, .bump = 0, .free_stack = undefined, .free_top = 0 };
    }
    fn selfOf(ctx: *anyopaque) *Self {
        return @ptrCast(@alignCast(ctx));
    }
    fn basePtr(self: *Self) [*]u8 {
        return @ptrCast(&self.pool[0]);
    }
    fn rawAlloc(ctx: *anyopaque, len: usize, alignment: Alignment, ra: usize) ?[*]u8 {
        const self = selfOf(ctx);
        if (len <= SLOT and alignment.toByteUnits() <= SLOT) {
            if (self.free_top > 0) {
                self.free_top -= 1;
                return self.basePtr() + (self.free_stack[self.free_top] * SLOT);
            }
            if (self.bump + SLOT > self.pool.len) return null;
            const idx = self.bump / SLOT;
            self.bump += SLOT;
            return self.basePtr() + (idx * SLOT);
        }
        return std.heap.page_allocator.rawAlloc(len, alignment, ra);
    }
    fn rawFree(ctx: *anyopaque, memory: []u8, alignment: Alignment, ra: usize) void {
        const self = selfOf(ctx);
        const b = @intFromPtr(self.basePtr());
        const p = @intFromPtr(memory.ptr);
        if (p >= b and (p - b) < self.pool.len) {
            self.free_stack[self.free_top] = (p - b) / SLOT;
            self.free_top += 1;
        } else {
            std.heap.page_allocator.rawFree(memory, alignment, ra);
        }
    }
    fn rawResize(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ra: usize) bool {
        _ = ctx;
        _ = memory;
        _ = alignment;
        _ = new_len;
        _ = ra;
        return false;
    }
    fn rawRemap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ra: usize) ?[*]u8 {
        _ = ctx;
        _ = memory;
        _ = alignment;
        _ = new_len;
        _ = ra;
        return null;
    }
    const VTable = Allocator.VTable;
    const vtable = VTable{ .alloc = rawAlloc, .resize = rawResize, .remap = rawRemap, .free = rawFree };
    fn allocator(self: *Self) Allocator {
        return Allocator{ .ptr = self, .vtable = &vtable };
    }
};

test "buildToolsJson namespaced output is independent of the aliases map" {
    // The two children's joined names share a length and a Wyhash fingerprint, so
    // a dangling `taken` key (freed wire, slot reused) reads the second as a false
    // collision and suffixes it. Bar: both emit unsuffixed, the `_2` form absent.
    var slot_alloc: SlotAllocator = .init();
    const alloc = slot_alloc.allocator();
    const json =
        \\[{"type":"namespace","name":"mcp__demo__","tools":[{"type":"function","name":"00000000","parameters":{}},{"type":"function","name":"00000623","parameters":{}}]}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();

    const out_null = try buildToolsJson(alloc, parsed.value.array, null);
    defer alloc.free(out_null);

    try testing.expect(std.mem.indexOf(u8, out_null, "\"name\":\"mcp__demo__00000000\"") != null);
    try testing.expect(std.mem.indexOf(u8, out_null, "\"name\":\"mcp__demo__00000623\"") != null);
    try testing.expect(std.mem.indexOf(u8, out_null, "mcp__demo__00000623_2") == null);
}

test "parseInput rewrites a namespaced function_call echo to its declared wire name" {
    const tools =
        \\[{"type":"namespace","name":"mcp__demo__","description":"D","tools":[{"type":"function","name":"get_weather","parameters":{}},{"type":"function","name":"get_time"}]}]
    ;
    const tp = try std.json.parseFromSlice(std.json.Value, testing.allocator, tools, .{});
    defer tp.deinit();
    var aliases = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases);
    const out = try buildToolsJson(testing.allocator, tp.value.array, &aliases);
    defer testing.allocator.free(out);

    const input =
        \\[
        \\  {"type":"function_call","call_id":"call_1","name":"get_weather","namespace":"mcp__demo__","arguments":"{\"city\":\"sf\"}"},
        \\  {"type":"function_call_output","call_id":"call_1","output":"sunny"}
        \\]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, &aliases, null, .{});
    defer pi.deinit();
    // History must echo the name the model was shown this turn (the wire name),
    // so the next call spells the declared name — not the bare child name.
    try testing.expectEqualStrings("mcp__demo__get_weather", pi.messages.items[0].tool_calls.?[0].name);
}

test "parseInput rewrites a colliding echo to its suffixed declared name" {
    const tools =
        \\[{"type":"function","name":"mcp__demo__get_weather","parameters":{}},{"type":"namespace","name":"mcp__demo__","tools":[{"type":"function","name":"get_weather","parameters":{}}]}]
    ;
    const tp = try std.json.parseFromSlice(std.json.Value, testing.allocator, tools, .{});
    defer tp.deinit();
    var aliases = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases);
    const out = try buildToolsJson(testing.allocator, tp.value.array, &aliases);
    defer testing.allocator.free(out);

    const input =
        \\[{"type":"function_call","call_id":"call_1","name":"get_weather","namespace":"mcp__demo__","arguments":"{}"}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, &aliases, null, .{});
    defer pi.deinit();
    // The bare join belongs to the FLAT tool; the member's declared name is
    // the suffixed one, so the echo must carry `..._2`.
    try testing.expectEqualStrings("mcp__demo__get_weather_2", pi.messages.items[0].tool_calls.?[0].name);
}

test "parseInput resolves an echo declared past a multi-suffix collision" {
    const tools =
        \\[{"type":"function","name":"mcp__demo__get_weather","parameters":{}},{"type":"function","name":"mcp__demo__get_weather_2","parameters":{}},{"type":"namespace","name":"mcp__demo__","tools":[{"type":"function","name":"get_weather","parameters":{}}]}]
    ;
    const tp = try std.json.parseFromSlice(std.json.Value, testing.allocator, tools, .{});
    defer tp.deinit();
    var aliases = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases);
    const out = try buildToolsJson(testing.allocator, tp.value.array, &aliases);
    defer testing.allocator.free(out);

    const input =
        \\[{"type":"function_call","call_id":"call_1","name":"get_weather","namespace":"mcp__demo__","arguments":"{}"}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, &aliases, null, .{});
    defer pi.deinit();
    // Flat holds base and _2, so the member's declared name is the _3 suffix;
    // the echo must resolve to it, not stop at the absent _2.
    try testing.expect(std.mem.indexOf(u8, out, "\"name\":\"mcp__demo__get_weather_3\"") != null);
    try testing.expectEqualStrings("mcp__demo__get_weather_3", pi.messages.items[0].tool_calls.?[0].name);
}

test "parseInput resolves echoes of two groups that share one join" {
    const tools =
        \\[{"type":"namespace","name":"a__b","tools":[{"type":"function","name":"c","parameters":{}}]},{"type":"namespace","name":"a","tools":[{"type":"function","name":"b__c","parameters":{}}]}]
    ;
    const tp = try std.json.parseFromSlice(std.json.Value, testing.allocator, tools, .{});
    defer tp.deinit();
    var aliases = NamespaceAliases.init(testing.allocator);
    defer freeNamespaceAliases(testing.allocator, &aliases);
    const out = try buildToolsJson(testing.allocator, tp.value.array, &aliases);
    defer testing.allocator.free(out);

    // (a__b,c) keeps the bare join; (a,b__c) collides onto it and is suffixed.
    try testing.expect(std.mem.indexOf(u8, out, "\"name\":\"a__b__c\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"name\":\"a__b__c_2\"") != null);

    const input =
        \\[{"type":"function_call","call_id":"call_1","name":"c","namespace":"a__b","arguments":"{}"},{"type":"function_call","call_id":"call_2","name":"b__c","namespace":"a","arguments":"{}"}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input, .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, &aliases, null, .{});
    defer pi.deinit();
    try testing.expectEqualStrings("a__b__c", pi.messages.items[0].tool_calls.?[0].name);
    try testing.expectEqualStrings("a__b__c_2", pi.messages.items[1].tool_calls.?[0].name);
}

test "buildToolsJson with null aliases still emits wire names and leaks nothing" {
    const json =
        \\[{"type":"namespace","name":"mcp__demo__","tools":[{"type":"function","name":"get_weather"},{"type":"function","name":"get_time"}]},{"type":"function","name":"mcp__demo__get_time"}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    // std.testing.allocator fails the test on any wire name not freed here.
    const out = try buildToolsJson(testing.allocator, parsed.value.array, null);
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "\"name\":\"mcp__demo__get_weather\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"name\":\"mcp__demo__get_time_2\"") != null);
}

test "appendFunctionCallItem serializes namespace only when set" {
    var set_buf = std.ArrayList(u8).empty;
    defer set_buf.deinit(testing.allocator);
    try appendFunctionCallItem(testing.allocator, &set_buf, "fc_1", "call_1", "get_weather", "{}", "mcp__demo__");
    try testing.expect(std.mem.indexOf(u8, set_buf.items, "\"name\":\"get_weather\",\"namespace\":\"mcp__demo__\"") != null);

    var flat_buf = std.ArrayList(u8).empty;
    defer flat_buf.deinit(testing.allocator);
    try appendFunctionCallItem(testing.allocator, &flat_buf, "fc_2", "call_2", "get_weather", "{}", null);
    try testing.expect(std.mem.indexOf(u8, flat_buf.items, "namespace") == null);
}

test "parseInput string becomes single user message" {
    const v: std.json.Value = .{ .string = "hello" };
    var pi = try parseInput(testing.allocator, v, null, null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 1), pi.messages.items.len);
    try testing.expectEqualStrings("user", pi.messages.items[0].role);
    try testing.expectEqualStrings("hello", pi.messages.items[0].content);
}

test "parseInput reads a developer item as the system turn" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\[{"role":"developer","content":"You are S."},{"role":"user","content":"hi"}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 2), pi.messages.items.len);
    try testing.expectEqualStrings("system", pi.messages.items[0].role);
    try testing.expectEqualStrings("You are S.", pi.messages.items[0].content);
}

fn testRejectingDecoder(_: std.mem.Allocator, _: *std.ArrayList(chat_mod.ImageData), _: []const u8, _: chat_mod.VisionPreproc) bool {
    return false;
}

test "parseInput records an input_image the decoder could not read" {
    const allocator = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\[{"role":"user","content":[{"type":"input_text","text":"what is this"},{"type":"input_image","image_url":"http://example.invalid/x.png"}]}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(allocator, parsed.value, null, null, null, testRejectingDecoder, .{});
    defer pi.deinit();
    try std.testing.expect(pi.image_decode_failed);
    try std.testing.expectEqual(@as(usize, 1), pi.messages.items.len);
}

test "parseInput with instructions prepends system" {
    const v: std.json.Value = .{ .string = "hi" };
    var pi = try parseInput(testing.allocator, v, "You are a pirate", null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 2), pi.messages.items.len);
    try testing.expectEqualStrings("system", pi.messages.items[0].role);
    try testing.expectEqualStrings("user", pi.messages.items[1].role);
}

test "parseInput replaces stored system when fresh instructions are provided" {
    const v: std.json.Value = .{ .string = "next" };
    const prev = [_]chat_mod.Message{
        .{ .role = "system", .content = "old instructions" },
        .{ .role = "user", .content = "first" },
        .{ .role = "assistant", .content = "answer" },
    };
    var pi = try parseInput(testing.allocator, v, "new instructions", &prev, null, null, .{});
    defer pi.deinit();

    try testing.expectEqual(@as(usize, 4), pi.messages.items.len);
    try testing.expectEqualStrings("system", pi.messages.items[0].role);
    try testing.expectEqualStrings("new instructions", pi.messages.items[0].content);
    try testing.expectEqualStrings("user", pi.messages.items[1].role);
    try testing.expectEqualStrings("assistant", pi.messages.items[2].role);
    try testing.expectEqualStrings("user", pi.messages.items[3].role);
    for (pi.messages.items[1..]) |m| {
        try testing.expect(!std.mem.eql(u8, m.role, "system"));
    }
}

test "parseInput folds a non-leading system into the leading one" {
    const allocator = testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\[{"role":"system","content":"mid"},{"role":"user","content":"hi"}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(allocator, parsed.value, "You are S.", null, null, null, .{});
    defer pi.deinit();
    // Qwen's own template raises on a system that is not first, so a second
    // system must fold into the leading one instead of reaching the render.
    try testing.expectEqual(@as(usize, 2), pi.messages.items.len);
    try testing.expectEqualStrings("system", pi.messages.items[0].role);
    try testing.expectEqualStrings("You are S.\n\nmid", pi.messages.items[0].content);
    try testing.expectEqualStrings("user", pi.messages.items[1].role);
    for (pi.messages.items[1..]) |m| {
        try testing.expect(!std.mem.eql(u8, m.role, "system"));
    }
}

test "parseInput function_call + function_call_output round-trip" {
    const json =
        \\[
        \\  {"type":"message","role":"user","content":"what's the weather?"},
        \\  {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"sf\"}"},
        \\  {"type":"function_call_output","call_id":"call_1","output":"sunny"}
        \\]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 3), pi.messages.items.len);
    try testing.expectEqualStrings("user", pi.messages.items[0].role);
    try testing.expectEqualStrings("assistant", pi.messages.items[1].role);
    try testing.expect(pi.messages.items[1].tool_calls != null);
    try testing.expectEqualStrings("call_1", pi.messages.items[1].tool_calls.?[0].id);
    try testing.expectEqualStrings("tool", pi.messages.items[2].role);
    try testing.expectEqualStrings("call_1", pi.messages.items[2].tool_call_id.?);
}

test "parseInput joins content-parts function_call_output in order" {
    // Codex's echo sends `output` as an array of content parts, never a string.
    const json =
        \\[
        \\  {"type":"function_call_output","call_id":"call_1","output":[{"type":"input_text","text":"Wall time: 1.0 seconds"},{"type":"input_text","text":""},{"type":"input_text","text":"Title: HELLO"}]}
        \\]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 1), pi.messages.items.len);
    try testing.expectEqualStrings("tool", pi.messages.items[0].role);
    try testing.expectEqualStrings("Wall time: 1.0 seconds\nTitle: HELLO", pi.messages.items[0].content);
}

test "compaction blob round-trips through encode + parseInput" {
    const msgs = [_]chat_mod.Message{
        .{ .role = "user", .content = "hello there" },
        .{ .role = "assistant", .content = "hi back" },
    };
    const blob = try encodeCompactionBlob(testing.allocator, &msgs);
    defer testing.allocator.free(blob);

    // Build the input array with a compaction item carrying the blob.
    const input_json = try std.fmt.allocPrint(testing.allocator,
        \\[{{"type":"compaction","id":"cmp_1","encrypted_content":"{s}"}}]
    , .{blob});
    defer testing.allocator.free(input_json);

    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input_json, .{});
    defer parsed.deinit();

    var pi = try parseInput(testing.allocator, parsed.value, null, null, null, null, .{});
    defer pi.deinit();

    try testing.expectEqual(@as(usize, 2), pi.messages.items.len);
    try testing.expectEqualStrings("user", pi.messages.items[0].role);
    try testing.expectEqualStrings("hello there", pi.messages.items[0].content);
    try testing.expectEqualStrings("assistant", pi.messages.items[1].role);
    try testing.expectEqualStrings("hi back", pi.messages.items[1].content);
}

test "compaction with malformed envelope is silently skipped" {
    // Bad base64 + bogus inner JSON shouldn't crash, just produces no messages.
    const inputs = [_][]const u8{
        \\[{"type":"compaction","encrypted_content":"!!!not-base64!!!"}]
        ,
        \\[{"type":"compaction","encrypted_content":""}]
        ,
    };
    for (inputs) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        var pi = try parseInput(testing.allocator, parsed.value, null, null, null, null, .{});
        defer pi.deinit();
        try testing.expectEqual(@as(usize, 0), pi.messages.items.len);
    }
}

test "parseToolChoice none drops tools, required emits instruction" {
    {
        const v: std.json.Value = .{ .string = "none" };
        const tc = try parseToolChoice(testing.allocator, v);
        defer if (tc.instruction) |i| testing.allocator.free(i);
        try testing.expectEqual(false, tc.include_tools);
    }
    {
        const v: std.json.Value = .{ .string = "required" };
        const tc = try parseToolChoice(testing.allocator, v);
        defer if (tc.instruction) |i| testing.allocator.free(i);
        try testing.expectEqual(true, tc.include_tools);
        try testing.expect(tc.instruction != null);
        try testing.expect(std.mem.indexOf(u8, tc.instruction.?, "MUST") != null);
    }
}

fn makeTestStored(gpa: std.mem.Allocator, id: []const u8) !*StoredResponse {
    const sr = try gpa.create(StoredResponse);
    var arena = std.heap.ArenaAllocator.init(gpa);
    sr.* = .{
        .id = try arena.allocator().dupe(u8, id),
        .created_at = 0,
        .model = try arena.allocator().dupe(u8, "m"),
        .status = try arena.allocator().dupe(u8, "completed"),
        .body_json = try arena.allocator().dupe(u8, "{}"),
        .history = &[_]chat_mod.Message{},
        .arena = arena,
    };
    return sr;
}

test "media: Responses rejects every undecodable declared Qwen image" {
    const bodies = [_][]const u8{
        "[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"describe\"},{\"type\":\"input_image\"}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"bad\"}]}]",
    };
    for (bodies) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        if (parseInput(testing.allocator, parsed.value, null, null, null, null, .{ .mode = .qwen })) |result| {
            var owned = result;
            owned.deinit();
            return error.MissingImageWasAccepted;
        } else |err| try testing.expectEqual(error.InvalidImage, err);
    }
}

test "media: stored Responses cannot continue omitted pixels" {
    const sr = try makeTestStored(testing.allocator, "media_history");
    defer sr.deinit();
    try testing.expectEqual(@as(usize, 0), (try sr.continuationHistory()).len);
    sr.media_omitted = true;
    try testing.expectError(error.IncompleteMediaHistory, sr.continuationHistory());
}

test "ResponseStore basic put/get/delete" {
    const gpa = testing.allocator;
    var store = ResponseStore.init(testing.io, gpa, 4);
    defer store.deinit();

    const sr = try makeTestStored(gpa, "resp_1");
    try store.put(sr);

    try testing.expect(store.get("resp_1") != null);
    try testing.expect(store.get("missing") == null);
    try testing.expectEqual(true, store.delete("resp_1"));
    try testing.expect(store.get("resp_1") == null);
}

test "ResponseStore evicts LRU at cap" {
    const gpa = testing.allocator;
    var store = ResponseStore.init(testing.io, gpa, 2);
    defer store.deinit();

    var ids: [3][]const u8 = undefined;
    inline for (0..3) |i| ids[i] = std.fmt.comptimePrint("id_{d}", .{i});
    for (ids) |id| {
        const sr = try makeTestStored(gpa, id);
        try store.put(sr);
    }
    // Cap is 2, inserted 3 — first one should be evicted
    try testing.expect(store.get("id_0") == null);
    try testing.expect(store.get("id_1") != null);
    try testing.expect(store.get("id_2") != null);
}
