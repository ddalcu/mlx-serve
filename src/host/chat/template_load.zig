//! Chat template loading: include stubs, the model dir sniff, `loadChatConfig`, and `formatChat`.

const std = @import("std");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const log = @import("../../sdk/util/log.zig");
const mlx_gguf = @import("../../sdk/quant/gguf/claim.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const testing = std.testing;

const ImageData = @import("types.zig").ImageData;
const VideoData = @import("types.zig").VideoData;
const Message = @import("types.zig").Message;
const ChatConfig = @import("types.zig").ChatConfig;
const renderChatTemplate = @import("render.zig").renderChatTemplate;
const templateRendersToolTurn = @import("render.zig").templateRendersToolTurn;

/// True when `<model_dir>/config.json` declares the given model_type. Any
/// read/parse failure is false — this only ARMS a family fallback, never
/// blocks a load.
fn dirModelTypeIs(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, want: []const u8) bool {
    const cfg_path = std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir}) catch return false;
    defer allocator.free(cfg_path);
    const file = std.Io.Dir.openFileAbsolute(io, cfg_path, .{}) catch return false;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var rs = file.reader(io, &buf);
    const content = rs.interface.allocRemaining(allocator, .limited(1 << 22)) catch return false;
    defer allocator.free(content);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return false;
    defer parsed.deinit();
    const mt = parsed.value.object.get("model_type") orelse return false;
    return mt == .string and std.mem.eql(u8, mt.string, want);
}

/// Is this `chat_template` entry just an `{% include %}` POINTER at the
/// sidecar file? transformers >= 5 saves the real template to
/// `chat_template.jinja` and writes that one-liner into tokenizer_config.json
/// (mlx-community's Laguna-S-2.1-oQ4e-fast / -oQ5e, issue #169). jinja.cpp has
/// no `include` statement, so rendering the pointer fails and drops SILENTLY to
/// the generic format — the model then answers in another family's markers.
/// The file it names is always the sidecar the caller already falls back to.
fn isIncludeStub(template: []const u8) bool {
    const t = std.mem.trim(u8, template, " \t\r\n");
    if (!std.mem.startsWith(u8, t, "{%") or !std.mem.endsWith(u8, t, "%}")) return false;
    const body = std.mem.trim(u8, t[2 .. t.len - 2], " \t\r\n-");
    if (!std.mem.startsWith(u8, body, "include")) return false;
    // Exactly one statement: a second `{%` would make it a real template.
    return std.mem.indexOf(u8, t[2..], "{%") == null;
}

/// The `chat_template` entry of tokenizer_config.json, as a string.
///
/// HF allows TWO shapes and we only handled one: a bare string, or a LIST of
/// `{name, template}` objects (Mistral-7B-Instruct-v0.3 ships `default` +
/// `tool_use`). Reading the list as `v.string` is not a graceful failure — it
/// is a Zig union panic that kills the process at LOAD, so the checkpoint
/// could not be served at all (live: `access of union field
/// 'string' while field 'array' is active`).
///
/// Selection follows transformers: the entry named "default", which is what
/// `apply_chat_template` uses unless the caller names another. Falls back to
/// the first entry when nothing is named "default", and returns null for any
/// other shape so the caller drops to its chat_template.jinja / family
/// fallback instead of panicking.
fn chatTemplateFromValue(v: ?std.json.Value) ?[]const u8 {
    const val = v orelse return null;
    switch (val) {
        .string => |s| return if (isIncludeStub(s)) null else s,
        .array => |arr| {
            var first: ?[]const u8 = null;
            for (arr.items) |entry| {
                if (entry != .object) continue;
                const tpl = entry.object.get("template") orelse continue;
                if (tpl != .string) continue;
                if (first == null) first = tpl.string;
                const name = entry.object.get("name") orelse continue;
                if (name == .string and std.mem.eql(u8, name.string, "default")) return tpl.string;
            }
            return first;
        },
        else => return null,
    }
}

/// Load chat template configuration from tokenizer_config.json.
pub fn loadChatConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ChatConfig {
    const content = (try mlx_gguf.sidecar(io, allocator, model_dir, .tokenizer_config)) orelse blk: {
        const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
        defer allocator.free(path);

        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    };
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;

    // DeepSeek's V4.1 template merges dicts with `**` (jinja.cpp cannot parse it)
    // and the repack's departs from encoding.py; ours is pinned to the encoder.
    const dsv41 = dirModelTypeIs(io, allocator, model_dir, "deepseek_v41");
    if (dsv41) log.info("chat: deepseek_v41 — using the embedded V4.1 transcription\n", .{});
    const chat_template: []const u8 = if (dsv41)
        try allocator.dupe(u8, @embedFile("../../fixtures/dsv41_chat_template.jinja"))
    else if (chatTemplateFromValue(root.get("chat_template"))) |t|
        try allocator.dupe(u8, t)
    else blk: {
        // Fall back to chat_template.jinja file (e.g. Qwen3.5 models)
        const jinja_path = try std.fmt.allocPrint(allocator, "{s}/chat_template.jinja", .{model_dir});
        defer allocator.free(jinja_path);
        if (std.Io.Dir.openFileAbsolute(io, jinja_path, .{})) |f| {
            defer f.close(io);
            var jinja_buf: [4096]u8 = undefined;
            var jinja_reader = f.reader(io, &jinja_buf);
            break :blk try jinja_reader.interface.allocRemaining(allocator, .limited(1 * 1024 * 1024));
        } else |_| {
            // Family fallback: a deepseek_v4 checkpoint that ships NO
            // template at all (pipenetwork REAP builds — their python
            // package does its own encoding) gets OUR byte-pinned DSV4
            // transcription. The generic fallback renders wrong-family
            // markers the model then echoes into degeneration (the
            // silent-fallback class, live), and DSML tool
            // calling depends on the real template.
            if (dirModelTypeIs(io, allocator, model_dir, "deepseek_v4")) {
                log.info("chat: deepseek_v4 checkpoint ships no chat_template — using the embedded DSV4 transcription\n", .{});
                break :blk try allocator.dupe(u8, @embedFile("../../fixtures/dsv4_chat_template.jinja"));
            }
            break :blk try allocator.dupe(u8, "");
        }
    };

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

    const renders_tool_turn: ?bool = templateRendersToolTurn(allocator, chat_template) catch null;
    return .{
        .chat_template = chat_template,
        .bos_token = bos_token,
        .eos_token = eos_token,
        .add_bos_token = add_bos_token,
        .allocator = allocator,
        .renders_tool_turn = renders_tool_turn,
        .image_placeholder = if (dsv41) "<\xef\xbd\x9cdeepseek_image\xef\xbd\x9c>" else null,
    };
}

/// Format chat messages into token IDs using the model's Jinja chat template.
/// `effort` is the client's raw `reasoning_effort` string (see
/// `serializeExtraContext`) — null when the request didn't send one.
pub fn formatChat(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    /// Extend the trailing assistant message instead of answering after it.
    continue_final: bool,
) ![]u32 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const marked = try markUntrustedContent(arena.allocator(), tok, messages, chat_config.chat_template);
    const rendered = try renderChatTemplate(allocator, marked, chat_config, tools_json, tool_choice_instruction, enable_thinking, effort, continue_final);
    defer allocator.free(rendered);

    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);

    if (chat_config.add_bos_token) {
        if (tok.bos_id) |bos| {
            try ids.append(allocator, bos);
        }
    }

    const body = try tok.encodeMarked(allocator, rendered);
    defer allocator.free(body);
    try ids.appendSlice(allocator, body);
    log.debug("  prompt: {d} chars -> {d} tokens\n", .{ rendered.len, ids.items.len });

    return ids.toOwnedSlice(allocator);
}

/// A control token inside a user, system or tool turn is TEXT, never a turn marker (transformers'
/// `sanitize_control_tokens`). Assistant turns are the model's own output, and the server writes their
/// think tags itself (the Responses history), so they keep their specials.
fn markUntrustedContent(arena: std.mem.Allocator, tok: *const Tokenizer, messages: []const Message, template: []const u8) ![]const Message {
    var out: ?[]Message = null;
    for (messages, 0..) |m, i| {
        if (std.mem.eql(u8, m.role, "assistant")) continue;
        const marked = try tok.markControlTokens(arena, m.content, template) orelse continue;
        if (out == null) out = try arena.dupe(Message, messages);
        out.?[i].content = marked;
    }
    return out orelse messages;
}

fn radioTestTokenizer(allocator: std.mem.Allocator) !Tokenizer {
    var tok = Tokenizer.initEmptyForTests(allocator, .byte_level_bpe);
    errdefer tok.deinit();
    for (tok.byte_to_unicode, 0..) |cp, byte| {
        var buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(cp, &buf);
        const text = try allocator.dupe(u8, buf[0..n]);
        try tok.vocab.put(text, @intCast(byte));
        try tok.id_to_token.put(@intCast(byte), text);
        try tok.unicode_to_byte.put(cp, @intCast(byte));
    }
    for ([_][]const u8{ "<image>", "<img>", "</img>" }, 1000..) |name, id| {
        const text = try allocator.dupe(u8, name);
        try tok.special_tokens.put(text, @intCast(id));
        try tok.id_to_token.put(@intCast(id), text);
    }
    return tok;
}

// ── tests ──

test "chat_template accepts HF's list-of-named-templates shape" {
    // Mistral-7B-Instruct-v0.3 ships a LIST (`default` + `tool_use`). Reading
    // it as `.string` panicked the process at load, so the checkpoint could
    // not be served at all.
    const a = std.testing.allocator;

    {   // the shape that panicked: pick the entry named "default"
        const json =
            \\{"chat_template":[{"name":"tool_use","template":"TOOLS"},
            \\{"name":"default","template":"PLAIN"}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("PLAIN", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // no "default" named: fall back to the first usable entry
        const json =
            \\{"chat_template":[{"name":"tool_use","template":"TOOLS"}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("TOOLS", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // the ordinary string shape is untouched
        const json =
            \\{"chat_template":"BARE"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("BARE", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // junk shapes return null so the caller uses its jinja/family fallback
        const json =
            \\{"chat_template":[{"name":"x"},{"nope":1}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    try std.testing.expect(chatTemplateFromValue(null) == null);

    {   // transformers >= 5 saves the template to chat_template.jinja and leaves
        // an `{% include %}` POINTER in tokenizer_config.json (mlx-community
        // Laguna-S-2.1-oQ4e-fast, issue #169). jinja.cpp has no `include`, so
        // taking the pointer literally fails the render and SILENTLY drops to
        // the generic fallback — wrong-family markers the model then echoes.
        // Read it as "no inline template" so the sidecar file is used.
        const json =
            \\{"chat_template":"{% include 'chat_template.jinja' %}"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    {   // whitespace/dash variants are the same pointer
        const json =
            \\{"chat_template":"\n  {%- include \"chat_template.jinja\" -%}\n"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    {   // a real template that merely CONTAINS the word include is untouched
        const json =
            \\{"chat_template":"{% if x %}include{% endif %}"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) != null);
    }
}

test "RADIO video placeholders: literal HTML stays text in every role" {
    const a = std.testing.allocator;
    var tok = try radioTestTokenizer(a);
    defer tok.deinit();
    const config = ChatConfig{
        .chat_template = "{% for m in messages %}{{ m.role + ':' }}{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}{% if p.type == 'image' %}{{ '<image>' }}{% elif p.type == 'video' %}{{ '<video>' }}{% elif p.type == 'text' %}{{ p.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}",
        .bos_token = null, .eos_token = null, .add_bos_token = false,
        .allocator = a, .renders_tool_turn = true, .video_placeholder = "<img>",
    };
    for ([_][]const u8{ "system", "user", "assistant", "tool" }) |role| {
        const messages = [_]Message{.{ .role = role, .content = "How do I autoplay a <video> element?" }};
        const ids = try formatChat(a, &tok, &messages, &config, null, null, true, null, false);
        defer a.free(ids);
        try std.testing.expectEqual(@as(usize, 0), std.mem.count(u32, ids, &.{1001}));
        const text = try tok.decode(a, ids, false);
        defer a.free(text);
        try std.testing.expect(std.mem.indexOf(u8, text, "<video>") != null);
    }
}

test "RADIO video placeholders: only attached clips create video atoms" {
    const a = std.testing.allocator;
    var tok = try radioTestTokenizer(a);
    defer tok.deinit();
    const config = ChatConfig{
        .chat_template = "{% for m in messages %}{% for p in m.content %}{% if p.type == 'image' %}{{ '<image>' }}{% elif p.type == 'video' %}{{ '<video>' }}{% elif p.type == 'text' %}{{ p.text }}{% endif %}{% endfor %}{% endfor %}",
        .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .video_placeholder = "<img>",
    };
    const clips = [_]VideoData{.{ .pixels = "", .grid_t = 128, .grid_h = 32, .grid_w = 32 }};
    const images = [_]ImageData{.{ .pixels = "", .width = 32, .height = 32 }};
    const messages = [_]Message{.{ .role = "user", .content = "Explain this <video> element and the clip.", .images = &images, .videos = &clips }};
    const ids = try formatChat(a, &tok, &messages, &config, null, null, true, null, false);
    defer a.free(ids);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u32, ids, &.{1001}));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u32, ids, &.{1000}));
    const text = try tok.decode(a, ids, false);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "<video>") != null);
}

test "loadChatConfig: template-less deepseek_v4 checkpoint gets the embedded DSV4 template" {
    // pipenetwork REAP builds ship NO chat_template (their python package
    // does its own encoding) — the generic fallback renders wrong-family
    // markers and the model degenerates into echoing them (live:
    // pi against REAP37 looped `<start_of_turn>model`). A deepseek_v4
    // checkpoint with no template must get OUR byte-pinned transcription.
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\": \"e\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\": \"deepseek_v4\"}" });
    var pbuf: [512]u8 = undefined;
    const plen = try tmp.dir.realPath(io, &pbuf);
    const dir_path = pbuf[0..plen];

    var cc = try loadChatConfig(io, allocator, dir_path);
    defer cc.deinit();
    try testing.expect(std.mem.indexOf(u8, cc.chat_template, "thinking_mode") != null);
    try testing.expect(std.mem.indexOf(u8, cc.chat_template, "DSML") != null);

    // Any other arch keeps the empty fallback (fallbackFormatChat territory).
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\": \"llama\"}" });
    var cc2 = try loadChatConfig(io, allocator, dir_path);
    defer cc2.deinit();
    try testing.expectEqual(@as(usize, 0), cc2.chat_template.len);
}

test "loadChatConfig: deepseek_v41 always gets the embedded V4.1 transcription" {
    // DeepSeek's own template merges dicts with `**`, which jinja.cpp cannot
    // parse, and the repack's differs from the encoder: ours is the pinned one.
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\": \"e\", \"chat_template\": \"{{ shipped }}\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\": \"deepseek_v41\"}" });
    var pbuf: [512]u8 = undefined;
    const plen = try tmp.dir.realPath(io, &pbuf);
    const embedded = @embedFile("../../fixtures/dsv41_chat_template.jinja");

    var cc = try loadChatConfig(io, allocator, pbuf[0..plen]);
    defer cc.deinit();
    try testing.expectEqualStrings(embedded, cc.chat_template);

    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\": \"e\"}" });
    var cc2 = try loadChatConfig(io, allocator, pbuf[0..plen]);
    defer cc2.deinit();
    try testing.expectEqualStrings(embedded, cc2.chat_template);
}
