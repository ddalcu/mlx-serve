//! OpenAI Responses API helpers.
//!
//! Pure data-handling for `POST /v1/responses` — input-item parsing, tool-shape
//! translation, output-item JSON builders, and the in-memory response store.
//! HTTP plumbing and generation orchestration live in `server.zig`.

const std = @import("std");
const chat_mod = @import("chat.zig");

// ─── small json helpers (intentionally duplicated from server.zig to avoid
// ─── a circular import; identical behavior) ──────────────────────────────

pub fn jsonEscape(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
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

pub fn makeId(allocator: std.mem.Allocator, prefix: []const u8) ![]u8 {
    const ms = std.time.milliTimestamp();
    const seq = id_counter.fetchAdd(1, .monotonic);
    return std.fmt.allocPrint(allocator, "{s}_{d}_{x}", .{ prefix, ms, seq });
}

// ─── reasoning effort ────────────────────────────────────────────────────

pub const ReasoningConfig = struct {
    enable: bool,
    budget: i32,
};

/// Map `reasoning.effort` → (enable_thinking, reasoning_budget).
/// `null` / unknown → thinking disabled, budget unchanged.
pub fn parseReasoning(reasoning_val: ?std.json.Value, default_budget: i32) ReasoningConfig {
    const v = reasoning_val orelse return .{ .enable = false, .budget = default_budget };
    if (v != .object) return .{ .enable = false, .budget = default_budget };
    const effort_val = v.object.get("effort") orelse return .{ .enable = true, .budget = default_budget };
    if (effort_val != .string) return .{ .enable = true, .budget = default_budget };
    const e = effort_val.string;
    const budget: i32 = if (std.mem.eql(u8, e, "minimal")) 128 else if (std.mem.eql(u8, e, "low")) 512 else if (std.mem.eql(u8, e, "medium")) 2048 else if (std.mem.eql(u8, e, "high")) 8192 else default_budget;
    return .{ .enable = true, .budget = budget };
}

// ─── text.format → schema constraint ──────────────────────────────────────

pub const TextFormat = struct {
    /// "text" | "json_object" | "json_schema"
    kind: []const u8,
    /// When kind == "json_schema": the schema value to enforce.
    schema_value: ?std.json.Value,
};

/// Decode the `text` field of a Responses request. Returns text-format ("text"
/// by default) — `json_schema` is FLAT here: `text.format = {type, name,
/// schema, strict}` — not nested under another `json_schema` key.
pub fn parseTextFormat(text_val: ?std.json.Value) TextFormat {
    const v = text_val orelse return .{ .kind = "text", .schema_value = null };
    if (v != .object) return .{ .kind = "text", .schema_value = null };
    const fmt_val = v.object.get("format") orelse return .{ .kind = "text", .schema_value = null };
    if (fmt_val != .object) return .{ .kind = "text", .schema_value = null };
    const t_val = fmt_val.object.get("type") orelse return .{ .kind = "text", .schema_value = null };
    const t = if (t_val == .string) t_val.string else "text";
    const schema = fmt_val.object.get("schema");
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

/// Re-shape Responses tools (`{type:"function", name, parameters, description}`)
/// into the nested OpenAI form (`{type:"function", function:{name, parameters,
/// description}}`) that `chat_mod.formatChat` expects. Returns owned JSON.
pub fn buildToolsJson(allocator: std.mem.Allocator, tools_array: std.json.Array) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    var emitted: usize = 0;
    for (tools_array.items) |tool_val| {
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        // Only function tools are supported locally (web_search/file_search/computer_use are not)
        const t = if (tool.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
        if (!std.mem.eql(u8, t, "function")) continue;

        if (emitted > 0) try buf.append(allocator, ',');
        emitted += 1;
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
        if (tool.get("parameters")) |params_val| {
            try serializeJsonValue(allocator, &buf, params_val);
        } else {
            try buf.appendSlice(allocator, "{}");
        }
        try buf.appendSlice(allocator, "}}");
    }
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
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
/// + libwebp). Returning null is fine — the input item will lack images.
pub const ImageUrlDecoder = *const fn (allocator: std.mem.Allocator, url: []const u8) ?chat_mod.ImageData;

/// Translate a Responses `input` value (string or array of input items) into
/// `chat_mod.Message`s. Optionally prepends `instructions` as the single leading
/// `system` msg. If `previous_messages` already contains a stored system message
/// and fresh instructions are provided, the fresh instructions replace it so
/// templates like Qwen's never see a non-leading/duplicate system message.
/// `previous_messages` are deep-referenced (not copied) into the result if
/// non-null — caller must keep them alive.
pub fn parseInput(
    allocator: std.mem.Allocator,
    input_val: std.json.Value,
    instructions: ?[]const u8,
    previous_messages: ?[]const chat_mod.Message,
    image_decoder: ?ImageUrlDecoder,
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
                    try appendMessageItem(allocator, &pi, obj, image_decoder);
                    continue;
                };
                if (t_val != .string) continue;
                const t = t_val.string;
                if (std.mem.eql(u8, t, "message")) {
                    try appendMessageItem(allocator, &pi, obj, image_decoder);
                } else if (std.mem.eql(u8, t, "function_call")) {
                    try appendFunctionCallInputItem(allocator, &pi, obj);
                } else if (std.mem.eql(u8, t, "function_call_output")) {
                    try appendFunctionCallOutputItem(allocator, &pi, obj);
                } else if (std.mem.eql(u8, t, "reasoning")) {
                    // Drop on input — model regenerates its own reasoning.
                    continue;
                } else {
                    // Unknown item type — skip silently.
                    continue;
                }
            }
        },
        else => {},
    }

    return pi;
}

fn appendMessageItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
    image_decoder: ?ImageUrlDecoder,
) !void {
    const role_val = obj.get("role") orelse return;
    if (role_val != .string) return;
    const role = role_val.string;

    const content_val = obj.get("content") orelse return;
    var content: []const u8 = "";
    var images: ?[]chat_mod.ImageData = null;

    switch (content_val) {
        .string => |s| content = s,
        .array => |arr| {
            var text_parts = std.ArrayList(u8).empty;
            defer text_parts.deinit(allocator);
            var image_list = std.ArrayList(chat_mod.ImageData).empty;
            errdefer {
                for (image_list.items) |img| allocator.free(img.pixels);
                image_list.deinit(allocator);
            }
            for (arr.items) |part| {
                if (part != .object) continue;
                const pt_val = part.object.get("type") orelse continue;
                if (pt_val != .string) continue;
                const pt = pt_val.string;
                if (std.mem.eql(u8, pt, "input_text") or std.mem.eql(u8, pt, "text") or std.mem.eql(u8, pt, "output_text")) {
                    const tx = part.object.get("text") orelse continue;
                    if (tx == .string) {
                        if (text_parts.items.len > 0) try text_parts.append(allocator, '\n');
                        try text_parts.appendSlice(allocator, tx.string);
                    }
                } else if (std.mem.eql(u8, pt, "input_image")) {
                    const url_val = part.object.get("image_url") orelse continue;
                    const url = switch (url_val) {
                        .string => |s| s,
                        .object => |io| if (io.get("url")) |u| (if (u == .string) u.string else continue) else continue,
                        else => continue,
                    };
                    if (image_decoder) |dec| {
                        if (dec(allocator, url)) |img| {
                            try image_list.append(allocator, img);
                        }
                    }
                }
            }
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
) !void {
    const call_id = if (obj.get("call_id")) |v| (if (v == .string) v.string else "") else "";
    const name = if (obj.get("name")) |v| (if (v == .string) v.string else "") else "";
    const args = if (obj.get("arguments")) |v| (if (v == .string) v.string else "{}") else "{}";

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
    const output = if (obj.get("output")) |v| (if (v == .string) v.string else "") else "";
    try pi.messages.append(allocator, .{
        .role = "tool",
        .content = output,
        .tool_call_id = call_id,
    });
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

    arena: std.heap.ArenaAllocator,

    list_node: std.DoublyLinkedList.Node = .{},

    pub fn deinit(self: *StoredResponse) void {
        var arena = self.arena;
        const gpa = arena.child_allocator;
        arena.deinit();
        gpa.destroy(self);
    }
};

pub const ResponseStore = struct {
    mu: std.Thread.Mutex = .{},
    map: std.StringHashMapUnmanaged(*StoredResponse) = .{},
    lru: std.DoublyLinkedList = .{},
    cap: usize,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, cap: usize) ResponseStore {
        return .{ .gpa = gpa, .cap = cap };
    }

    pub fn deinit(self: *ResponseStore) void {
        self.mu.lock();
        defer self.mu.unlock();
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
        self.mu.lock();
        defer self.mu.unlock();

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
        self.mu.lock();
        defer self.mu.unlock();
        const sr = self.map.get(id) orelse return null;
        self.lru.remove(&sr.list_node);
        self.lru.prepend(&sr.list_node);
        return sr;
    }

    /// Returns true if removed.
    pub fn delete(self: *ResponseStore, id: []const u8) bool {
        self.mu.lock();
        defer self.mu.unlock();
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
    try testing.expectEqual(@as(i32, 512), parseReasoning(v_low.value, -1).budget);

    const v_high = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"effort\":\"high\"}", .{});
    defer v_high.deinit();
    try testing.expectEqual(@as(i32, 8192), parseReasoning(v_high.value, -1).budget);

    try testing.expectEqual(false, parseReasoning(null, -1).enable);
    try testing.expectEqual(@as(i32, -1), parseReasoning(null, -1).budget);
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
    const out = try buildToolsJson(testing.allocator, parsed.value.array);
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
    const out = try buildToolsJson(testing.allocator, parsed.value.array);
    defer testing.allocator.free(out);
    // Only the function tool should be emitted, no leading comma
    try testing.expect(std.mem.startsWith(u8, out, "[{"));
    try testing.expect(std.mem.indexOf(u8, out, "web_search") == null);
}

test "parseInput string becomes single user message" {
    const v: std.json.Value = .{ .string = "hello" };
    var pi = try parseInput(testing.allocator, v, null, null, null);
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 1), pi.messages.items.len);
    try testing.expectEqualStrings("user", pi.messages.items[0].role);
    try testing.expectEqualStrings("hello", pi.messages.items[0].content);
}

test "parseInput with instructions prepends system" {
    const v: std.json.Value = .{ .string = "hi" };
    var pi = try parseInput(testing.allocator, v, "You are a pirate", null, null);
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
    var pi = try parseInput(testing.allocator, v, "new instructions", &prev, null);
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
    var pi = try parseInput(testing.allocator, parsed.value, null, null, null);
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 3), pi.messages.items.len);
    try testing.expectEqualStrings("user", pi.messages.items[0].role);
    try testing.expectEqualStrings("assistant", pi.messages.items[1].role);
    try testing.expect(pi.messages.items[1].tool_calls != null);
    try testing.expectEqualStrings("call_1", pi.messages.items[1].tool_calls.?[0].id);
    try testing.expectEqualStrings("tool", pi.messages.items[2].role);
    try testing.expectEqualStrings("call_1", pi.messages.items[2].tool_call_id.?);
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

test "ResponseStore basic put/get/delete" {
    const gpa = testing.allocator;
    var store = ResponseStore.init(gpa, 4);
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
    var store = ResponseStore.init(gpa, 2);
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
