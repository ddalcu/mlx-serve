const std = @import("std");
const jinja = @import("vibe_jinja");
const tokenizer_mod = @import("tokenizer.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

/// Chat template configuration loaded from tokenizer_config.json.
pub const ChatConfig = struct {
    chat_template: []const u8,
    bos_token: ?[]const u8,
    eos_token: ?[]const u8,
    add_bos_token: bool,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ChatConfig) void {
        self.allocator.free(self.chat_template);
        if (self.bos_token) |t| self.allocator.free(t);
        if (self.eos_token) |t| self.allocator.free(t);
    }
};

/// Load chat template configuration from tokenizer_config.json.
pub fn loadChatConfig(allocator: std.mem.Allocator, model_dir: []const u8) !ChatConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;

    const chat_template = if (root.get("chat_template")) |v|
        try allocator.dupe(u8, v.string)
    else
        try allocator.dupe(u8, "");

    const bos_token: ?[]const u8 = if (root.get("bos_token")) |v|
        (if (v == .string) try allocator.dupe(u8, v.string) else null)
    else
        null;

    const eos_token: ?[]const u8 = if (root.get("eos_token")) |v|
        (if (v == .string) try allocator.dupe(u8, v.string) else null)
    else
        null;

    const add_bos_token = if (root.get("add_bos_token")) |v|
        (if (v == .bool) v.bool else false)
    else
        false;

    return .{
        .chat_template = chat_template,
        .bos_token = bos_token,
        .eos_token = eos_token,
        .add_bos_token = add_bos_token,
        .allocator = allocator,
    };
}

/// Format chat messages into token IDs using the model's Jinja chat template.
pub fn formatChat(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    messages: []const Message,
    chat_config: *const ChatConfig,
) ![]u32 {
    // Render the chat template using vibe-jinja
    const rendered = try renderChatTemplate(allocator, messages, chat_config);
    defer allocator.free(rendered);

    // Encode the rendered text
    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);

    // Handle BOS token
    if (chat_config.add_bos_token) {
        if (tok.bos_id) |bos| {
            try ids.append(allocator, bos);
        }
    }

    // Split on special tokens and encode each segment
    try encodeWithSpecialTokens(allocator, tok, rendered, &ids);

    return ids.toOwnedSlice(allocator);
}

/// Render the Jinja chat template with the given messages.
fn renderChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
) ![]const u8 {
    // HuggingFace chat templates use complex Jinja2 features (list slicing, namespace,
    // reverse iteration, `is string` tests) that aren't reliably supported by current
    // Zig-native Jinja implementations. Use hardcoded formatters for known template types.
    return try fallbackFormatChat(allocator, messages, chat_config);
}

/// Encode text that may contain special tokens (like <|im_start|>, <bos>, etc.).
/// Splits the text on known special tokens, encodes text segments normally,
/// and inserts the special token IDs directly.
fn encodeWithSpecialTokens(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    text: []const u8,
    ids: *std.ArrayList(u32),
) !void {
    // Collect special tokens sorted by length (longest first for greedy matching)
    var specials = std.ArrayList(SpecialEntry).empty;
    defer specials.deinit(allocator);

    var sit = tok.special_tokens.iterator();
    while (sit.next()) |entry| {
        try specials.append(allocator, .{ .text = entry.key_ptr.*, .id = entry.value_ptr.* });
    }

    // Sort by length descending (greedy match)
    std.mem.sort(SpecialEntry, specials.items, {}, struct {
        fn lessThan(_: void, a: SpecialEntry, b: SpecialEntry) bool {
            return a.text.len > b.text.len;
        }
    }.lessThan);

    var pos: usize = 0;
    while (pos < text.len) {
        // Try to match a special token at current position
        var matched = false;
        for (specials.items) |special| {
            if (pos + special.text.len <= text.len and
                std.mem.eql(u8, text[pos .. pos + special.text.len], special.text))
            {
                // Encode any text before this special token
                // (already handled by the loop structure)
                try ids.append(allocator, special.id);
                pos += special.text.len;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // Find the next special token
        var next_special_pos: usize = text.len;
        for (specials.items) |special| {
            if (std.mem.indexOf(u8, text[pos..], special.text)) |offset| {
                const abs_pos = pos + offset;
                if (abs_pos < next_special_pos) {
                    next_special_pos = abs_pos;
                }
            }
        }

        // Encode the text segment before the next special token
        if (next_special_pos > pos) {
            const segment = text[pos..next_special_pos];
            const segment_ids = try tok.encode(allocator, segment);
            defer allocator.free(segment_ids);
            try ids.appendSlice(allocator, segment_ids);
        }
        pos = next_special_pos;
    }
}

const SpecialEntry = struct {
    text: []const u8,
    id: u32,
};

/// Fallback chat formatting for when Jinja rendering fails.
/// Detects known template patterns and generates the expected format.
fn fallbackFormatChat(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    // Detect template type from eos_token or template content
    const is_chatml = chat_config.eos_token != null and
        std.mem.indexOf(u8, chat_config.eos_token.?, "<|im_end|>") != null;

    if (is_chatml) {
        // ChatML format (Qwen3, etc.):
        // <|im_start|>role\ncontent<|im_end|>\n
        for (messages) |msg| {
            try result.appendSlice(allocator, "<|im_start|>");
            try result.appendSlice(allocator, msg.role);
            try result.appendSlice(allocator, "\n");
            try result.appendSlice(allocator, msg.content);
            try result.appendSlice(allocator, "<|im_end|>\n");
        }
        try result.appendSlice(allocator, "<|im_start|>assistant\n");
    } else {
        // Gemma-style format:
        // <bos><start_of_turn>role\ncontent<end_of_turn>\n
        if (chat_config.bos_token) |bos| {
            try result.appendSlice(allocator, bos);
        }
        for (messages) |msg| {
            try result.appendSlice(allocator, "<start_of_turn>");
            try result.appendSlice(allocator, msg.role);
            try result.appendSlice(allocator, "\n");
            try result.appendSlice(allocator, msg.content);
            try result.appendSlice(allocator, "<end_of_turn>\n");
        }
        try result.appendSlice(allocator, "<start_of_turn>model\n");
    }

    return result.toOwnedSlice(allocator);
}
