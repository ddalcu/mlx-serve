const std = @import("std");
const jinja_c = @import("jinja_c");
const tokenizer_mod = @import("tokenizer.zig");
const arch_ds4 = if (@import("build_options").ios) @import("arch/ds4_stub.zig") else @import("arch/ds4.zig");
const ds4_ffi = if (@import("build_options").ios) @import("ds4_ffi_stub.zig") else @import("ds4_ffi.zig");
const arch_llama = if (@import("build_options").ios) @import("arch/llama_stub.zig") else @import("arch/llama.zig");
const log = @import("log.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8, // JSON string
};

/// Raw image pixel data for vision encoder (float32, CHW format).
pub const ImageData = struct {
    pixels: []const u8, // Gemma: CHW float32 [3*H*W*4]. Qwen: merge-order patches [N*1536*4].
    width: u32,
    height: u32,
    // Qwen3-VL only: full patch grid (h=resized_H/patch, w=resized_W/patch). 0 ⇒
    // Gemma CHW layout. When >0, `pixels` holds the processor's merge-order
    // pixel_values and the encoder is QwenVision (see src/qwen_vision.zig).
    grid_h: u32 = 0,
    grid_w: u32 = 0,
    /// LFM2-VL tiling: a source image past the single-tile budget is split into
    /// a `tile_rows` x `tile_cols` grid plus a thumbnail, and EACH piece is its
    /// own entry here — separate patch grid, separate encoder call. `tile_index`
    /// runs row-major over the tiles, and `tile_index == rows*cols` is the
    /// thumbnail. All three stay 0 for an untiled image, which is every other
    /// tower we serve.
    tile_rows: u16 = 0,
    tile_cols: u16 = 0,
    tile_index: u16 = 0,
};

/// Per-request image preprocessing selector, derived from the loaded model's
/// config. Threaded into `parseImageUrlContent`/`decodeImageToPixels` so decode
/// stays race-safe (no global state) under `--max-concurrent ≥ 2`.
pub const VisionPreproc = struct {
    /// Which processor produced `ImageData.pixels`: Gemma's fixed CHW square,
    /// or one of the patch-grid towers (each with its own resize + patch order).
    mode: enum { gemma, qwen, muse, lfm2 } = .gemma,
    patch: u32 = 16,
    tps: u32 = 2,
    merge: u32 = 2,
    min_pixels: u32 = 0,
    max_pixels: u32 = 0,
    /// muse/lfm2: the resize cap is on MERGED tokens, not pixels.
    max_tokens: u32 = 0,
    /// lfm2: the budget has a FLOOR too — a small image is upscaled to it.
    min_tokens: u32 = 0,
    /// lfm2 tiling. A source past `max_tokens * pixels_tolerance` is split into
    /// a grid of `tile_size` tiles (plus a thumbnail) and encoded piece by
    /// piece. `tile_size == 0` disables splitting, which is every other tower.
    tile_size: u32 = 0,
    min_tiles: u32 = 0,
    max_tiles: u32 = 0,
    use_thumbnail: bool = false,
    pixels_tolerance: f32 = 0,
};

/// Raw mono 16 kHz audio samples for the Gemma 4 12B unified audio embedder.
/// Bytes are little-endian float32 PCM; the encoder frames them into 640-sample
/// tokens (40 ms @ 16 kHz) and projects each straight into language-model space.
pub const AudioData = struct {
    samples: []const u8, // Raw float32-LE bytes (n_samples * 4)
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
    tool_calls: ?[]const ToolCall = null,
    tool_call_id: ?[]const u8 = null,
    images: ?[]const ImageData = null, // Preprocessed image data for vision
    audio: ?[]const AudioData = null, // Raw PCM for the unified audio embedder
    /// Reasoning the client round-trips on assistant HISTORY messages
    /// (`reasoning_content`/`reasoning` on chat completions, `thinking`
    /// blocks on /v1/messages). Templates that persist reasoning across
    /// turns (laguna) read it; templates that strip history reasoning
    /// (Qwen, Gemma) never reference the field and render unchanged.
    reasoning_content: ?[]const u8 = null,
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
/// could not be served at all (live 2026-08-07: `access of union field
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
        const json = \\{"chat_template":[{"name":"tool_use","template":"TOOLS"}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("TOOLS", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // the ordinary string shape is untouched
        const json = \\{"chat_template":"BARE"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("BARE", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // junk shapes return null so the caller uses its jinja/family fallback
        const json = \\{"chat_template":[{"name":"x"},{"nope":1}]}
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
        const json = \\{"chat_template":"{% include 'chat_template.jinja' %}"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    {   // whitespace/dash variants are the same pointer
        const json = \\{"chat_template":"\n  {%- include \"chat_template.jinja\" -%}\n"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    {   // a real template that merely CONTAINS the word include is untouched
        const json = \\{"chat_template":"{% if x %}include{% endif %}"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) != null);
    }
}

/// Load chat template configuration from tokenizer_config.json.
pub fn loadChatConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ChatConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;

    const chat_template: []const u8 = if (chatTemplateFromValue(root.get("chat_template"))) |t|
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
            // silent-fallback class, live 2026-08-01), and DSML tool
            // calling depends on the real template.
            if (dirModelTypeIs(io, allocator, model_dir, "deepseek_v4")) {
                log.info("chat: deepseek_v4 checkpoint ships no chat_template — using the embedded DSV4 transcription\n", .{});
                break :blk try allocator.dupe(u8, @embedFile("fixtures/dsv4_chat_template.jinja"));
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

    return .{
        .chat_template = chat_template,
        .bos_token = bos_token,
        .eos_token = eos_token,
        .add_bos_token = add_bos_token,
        .allocator = allocator,
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
    const rendered = try renderChatTemplate(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, effort, continue_final);
    defer allocator.free(rendered);


    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);

    if (chat_config.add_bos_token) {
        if (tok.bos_id) |bos| {
            try ids.append(allocator, bos);
        }
    }

    try encodeWithSpecialTokens(allocator, tok, rendered, &ids);
    log.debug("  prompt: {d} chars -> {d} tokens\n", .{ rendered.len, ids.items.len });

    return ids.toOwnedSlice(allocator);
}

/// Render + encode chat messages via the embedded ds4 engine's chat template.
/// ds4 owns its own tokenizer and chat-template renderer — we just map our
/// `Message` array to ds4's `ChatTurn` shape and append the assistant prefix.
///
/// Tool plumbing: the DSV4 chat template doesn't model the `tools` argument
/// natively, so when `tools_json` is non-null we synthesize a fallback —
/// inject the tool catalog into the system message (same machinery the Jinja
/// path uses via `synthesizeToolFallbackMessages`), and rewrite `role:"tool"`
/// turns as `<tool_response>…</tool_response>` user turns, and rewrite
/// assistant `tool_calls` as inline `<tool_call>…</tool_call>` text. Result:
/// the model sees the full tool catalog AND the live tool-result history,
/// which is what makes Agent mode + MCP tool calling work on ds4.
/// Pure prep step extracted so the tool-synthesis decision is unit-testable
/// without booting the ds4 engine. Returns the system content (or null) and
/// the post-synthesis turns ready for `Ds4Engine.encodeChatTranscript`.
/// The returned arena owns all string memory; caller must `deinit()` it.
pub const Ds4PromptPrep = struct {
    arena: std.heap.ArenaAllocator,
    system: ?[]const u8,
    turns: []arch_ds4.Ds4Engine.ChatTurn,

    pub fn deinit(self: *Ds4PromptPrep) void {
        self.arena.deinit();
    }
};

/// Pure helper: run the ds4 chat-template fallback synthesis (tool catalog
/// into system, tool-role rewrite, assistant tool_call inlining) and split
/// the result into (system, turns). Exists so the tool-plumbing decision is
/// covered by a unit test that doesn't need a live engine.
pub fn prepareDs4Prompt(
    parent_allocator: std.mem.Allocator,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
) !Ds4PromptPrep {
    var arena = std.heap.ArenaAllocator.init(parent_allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    const messages_have_tool_content = messagesHaveToolContent(messages);
    var effective_messages = messages;
    if (tools_json != null or messages_have_tool_content) {
        effective_messages = try synthesizeToolFallbackMessages(
            arena_alloc,
            messages,
            tools_json,
            tool_choice_instruction,
            tools_json != null,
            messages_have_tool_content,
            messages_have_tool_content,
        );
    }

    var system_msg: ?[]const u8 = null;
    var turns_list = std.ArrayList(arch_ds4.Ds4Engine.ChatTurn).empty;
    for (effective_messages) |msg| {
        if (system_msg == null and std.mem.eql(u8, msg.role, "system")) {
            system_msg = msg.content;
            continue;
        }
        // After synthesis only system / user / assistant roles remain.
        try turns_list.append(arena_alloc, .{ .role = msg.role, .content = msg.content });
    }

    return .{
        .arena = arena,
        .system = system_msg,
        .turns = try turns_list.toOwnedSlice(arena_alloc),
    };
}

pub fn encodeChatViaDs4(
    allocator: std.mem.Allocator,
    engine: *arch_ds4.Ds4Engine,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]u32 {
    // ── Tool synthesis fallback. ds4's chat template doesn't reference
    //    `tools` or `role == 'tool'`, so without this rewrite:
    //      * tool definitions never reach the model (model can't call them)
    //      * tool-result messages render as empty (model loses prior context)
    //      * assistant tool_call history disappears (multi-turn loops break)
    //    Run via the pure `prepareDs4Prompt` helper so the same logic is
    //    exercised by the unit test at the bottom of this file.
    var prep = try prepareDs4Prompt(allocator, messages, tools_json, tool_choice_instruction);
    defer prep.deinit();

    const think_mode: ds4_ffi.ThinkMode = if (enable_thinking) .high else .none;
    const i32_ids = try engine.encodeChatTranscript(allocator, prep.system, prep.turns, think_mode);
    defer allocator.free(i32_ids);

    const u32_ids = try allocator.alloc(u32, i32_ids.len);
    for (i32_ids, 0..) |t, i| u32_ids[i] = @intCast(t);
    log.debug("  prompt (ds4): {d} messages -> {d} tokens (tools={s})\n", .{
        messages.len,
        u32_ids.len,
        if (tools_json != null) "yes" else "no",
    });
    return u32_ids;
}

/// Detokenize a sequence of token IDs via the ds4 engine. Mirrors
/// `Tokenizer.decode` so server handlers can switch on the LoadedModel without
/// touching the call sites for each path.
pub fn decodeViaDs4(
    allocator: std.mem.Allocator,
    engine: *arch_ds4.Ds4Engine,
    ids: []const u32,
) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    for (ids) |id| {
        const piece = try engine.detokenizeOne(allocator, @intCast(id));
        defer allocator.free(piece);
        try out.appendSlice(allocator, piece);
    }
    return out.toOwnedSlice(allocator);
}

/// Render + encode chat messages for a llama.cpp-backed (GGUF) model.
///
/// Unlike ds4 (which has its own template renderer), we reuse mlx-serve's Jinja
/// engine via `renderChatTemplate` — `chat_config.chat_template` is populated
/// from the GGUF's embedded template at load time (see
/// `Scheduler.doLoadLlamaOnInferenceThread`). That path already handles the
/// tool-synthesis fallback for templates that don't model `tools` natively, so
/// tool calling works across the GGUF zoo. The rendered prompt is then tokenized
/// through libllama's own vocab with `add_special = false` (the template owns
/// any BOS) and `parse_special = true` (so `<|im_start|>` etc. become real
/// special tokens).
pub fn encodeChatViaLlama(
    allocator: std.mem.Allocator,
    engine: *arch_llama.LlamaEngine,
    chat_config: *const ChatConfig,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    /// Extend the trailing assistant message instead of answering after it.
    continue_final: bool,
) ![]u32 {
    const rendered = try renderChatTemplate(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, effort, continue_final);
    defer allocator.free(rendered);

    const i32_ids = try engine.tokenizeText(allocator, rendered, false);
    defer allocator.free(i32_ids);

    const u32_ids = try allocator.alloc(u32, i32_ids.len);
    for (i32_ids, 0..) |t, i| u32_ids[i] = @intCast(t);
    log.debug("  prompt (llama): {d} messages -> {d} tokens (tools={s})\n", .{
        messages.len,
        u32_ids.len,
        if (tools_json != null) "yes" else "no",
    });
    return u32_ids;
}

/// Detokenize a sequence of token IDs via the llama.cpp engine. Mirrors
/// `decodeViaDs4` so server handlers switch on the LoadedModel uniformly.
pub fn decodeViaLlama(
    allocator: std.mem.Allocator,
    engine: *arch_llama.LlamaEngine,
    ids: []const u32,
) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    for (ids) |id| {
        const piece = try engine.detokenizeOne(allocator, @intCast(id));
        defer allocator.free(piece);
        try out.appendSlice(allocator, piece);
    }
    return out.toOwnedSlice(allocator);
}

/// Render the Jinja chat template with the given messages.
fn renderChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    /// Extend the trailing assistant message instead of answering after it —
    /// the prompt ends mid-turn and generation resumes from its last word.
    continue_final: bool,
) ![]const u8 {
    // Split the partial reply off FIRST, so every transform below (tool
    // fallback synthesis, muse's reasoning drop) sees exactly the message list
    // an ordinary turn would have. The template must open the assistant turn
    // once, which it only does when the trailing assistant is not in `messages`.
    const prefill = continuationPrefill(messages, continue_final);
    const msgs = if (prefill != null) messages[0 .. messages.len - 1] else messages;

    if (chat_config.chat_template.len == 0) {
        const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
        if (prefill) |partial| {
            defer allocator.free(base);
            return std.mem.concat(allocator, u8, &.{ base, partial });
        }
        return base;
    }

    // Some chat templates (e.g. DeepSeek V4) don't reference `tools` or `role == 'tool'`
    // at all — Jinja silently drops the tool definitions, and tool-result messages render
    // as empty. Detect that gap and synthesize an equivalent system-prompt + user-message
    // form so the model still sees the tool context.
    const tpl = chat_config.chat_template;
    const tpl_has_tools = std.mem.indexOf(u8, tpl, "tools") != null;
    const tpl_has_tool_role = templateReferencesToolRole(tpl);
    const needs_inject_tools = tools_json != null and !tpl_has_tools;
    const needs_rewrite_tool_role = !tpl_has_tool_role and messagesHaveToolContent(msgs);

    var fallback_arena: ?std.heap.ArenaAllocator = null;
    defer if (fallback_arena) |*a| a.deinit();
    var effective_messages = msgs;
    var effective_tools_json = tools_json;
    // Optional keys a template may dump blindly (see fillOptionalToolDefKeys).
    // Arena-owned, so the rewritten bytes outlive the render with no free path
    // of their own.
    if (tools_json) |tj| {
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        if (try fillOptionalToolDefKeys(fallback_arena.?.allocator(), tj)) |filled| {
            effective_tools_json = filled;
        }
    }
    if (needs_inject_tools or needs_rewrite_tool_role) {
        // NEVER re-init: the tool-def fill above may already own this arena,
        // and clobbering it orphans everything allocated there (a leak here, a
        // use-after-free the moment `effective_tools_json` points into it).
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const arena_alloc = fallback_arena.?.allocator();
        effective_messages = try synthesizeToolFallbackMessages(
            arena_alloc,
            msgs,
            tools_json,
            tool_choice_instruction,
            needs_inject_tools,
            !tpl_has_tool_role,
            !tpl_has_tools,
        );
        if (needs_inject_tools) {
            effective_tools_json = null; // already inlined as system content
        }
    }

    // Muse (harmony convention): reasoning belongs to the CURRENT turn only.
    // The app round-trips `reasoning_content` on assistant history, and this
    // template renders it as a to=self segment — so one bad chain (a loop-cut
    // tail) would replay in the prompt on every later turn. Drop it on
    // assistant messages before the last user message; tool rounds after it
    // keep theirs. Families whose templates NEED history reasoning (laguna,
    // inkling) are untouched — same sniff `noThinkTailSuffix` uses.
    if (std.mem.indexOf(u8, tpl, "reasoning_strength") != null) {
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        if (try dropPriorTurnReasoning(fallback_arena.?.allocator(), effective_messages)) |stripped| {
            effective_messages = stripped;
        }
    }

    // Serialize messages to JSON — Gemma 4 templates handle role:"tool" natively
    // (producing <|turn>tool in the rendered output). No transformation needed.
    const messages_json = try serializeMessagesJson(allocator, effective_messages);
    defer allocator.free(messages_json);

    // Build extra context (bos_token, eos_token, enable_thinking, effort)
    const extra_json = try serializeExtraContext(allocator, chat_config, enable_thinking, effort);
    defer allocator.free(extra_json);

    // Null-terminate strings for C
    const tmpl_z = try allocator.dupeSentinel(u8, chat_config.chat_template, 0);
    defer allocator.free(tmpl_z);
    const msgs_z = try allocator.dupeSentinel(u8, messages_json, 0);
    defer allocator.free(msgs_z);
    const extra_z = try allocator.dupeSentinel(u8, extra_json, 0);
    defer allocator.free(extra_z);

    var tools_z: ?[:0]const u8 = null;
    defer if (tools_z) |tz| allocator.free(tz);
    if (effective_tools_json) |tj| {
        tools_z = try allocator.dupeSentinel(u8, tj, 0);
    }

    const result_ptr = jinja_c.jinja_render_chat(
        tmpl_z.ptr,
        msgs_z.ptr,
        if (tools_z) |tz| tz.ptr else null,
        extra_z.ptr,
        1,
    );

    if (result_ptr) |ptr| {
        defer jinja_c.jinja_str_free(ptr);
        const collapsed = try collapseDoubledThinkTags(allocator, std.mem.span(ptr));
        // A continuation commits the content channel and then hands the model
        // its own unfinished sentence. It runs INSTEAD of the thinking-off
        // tail, never beside it — both append the same channel commit, and
        // twice is a malformed turn header.
        if (prefill) |partial| {
            defer allocator.free(collapsed);
            return std.mem.concat(allocator, u8, &.{
                collapsed, contentChannelTail(tpl, collapsed, true), partial,
            });
        }
        if (noThinkTailSuffix(tpl, collapsed, enable_thinking, tools_json != null)) |suffix| {
            defer allocator.free(collapsed);
            return std.mem.concat(allocator, u8, &.{ collapsed, suffix });
        }
        return collapsed;
    }

    // WARN, not debug: a failed render silently swaps the prompt into
    // fallbackFormatChat's generic format — for families whose stop/turn
    // tokens differ (Gemma 4's <|turn> vs the fallback's <start_of_turn>)
    // that means degenerate generation, so the downgrade must be visible
    // at the default log level.
    if (jinja_c.jinja_last_error()) |e| {
        log.warn("jinja render failed ({s}), falling back to generic chat format\n", .{std.mem.span(e)});
    }
    const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    if (prefill) |partial| {
        defer allocator.free(base);
        return std.mem.concat(allocator, u8, &.{ base, partial });
    }
    return base;
}

/// Null `reasoning_content` on assistant messages BEFORE the last user
/// message (harmony convention: prior-turn analysis is dropped from the
/// prompt; the current turn's tool rounds keep theirs). Returns null when
/// nothing would change, so the common no-history-reasoning case allocates
/// nothing.
fn dropPriorTurnReasoning(arena: std.mem.Allocator, messages: []const Message) !?[]const Message {
    var last_user: ?usize = null;
    for (messages, 0..) |m, i| {
        if (std.mem.eql(u8, m.role, "user")) last_user = i;
    }
    const cut = last_user orelse return null;
    var any = false;
    for (messages[0..cut]) |m| {
        if (std.mem.eql(u8, m.role, "assistant") and m.reasoning_content != null) {
            any = true;
            break;
        }
    }
    if (!any) return null;
    const out = try arena.dupe(Message, messages);
    for (out[0..cut]) |*m| {
        if (std.mem.eql(u8, m.role, "assistant")) m.reasoning_content = null;
    }
    return out;
}

/// DSV4's chat template emits `</think></think>` between every user→assistant
/// pair in chat mode — the user role appends `</think>` after `<|Assistant|>`
/// and the assistant role prepends `</think>` before its content. The doubling
/// is structurally invalid (no legitimate rendered prompt should contain
/// `</think></think>`), and at 2-bit DQ it pushes the model into prompt-echo
/// / phrase-loop collapse on the second turn (a 24-token "Hello / Hi! / What
/// is 2+2?" chat reliably degenerates). The HF tokenizer renders the same
/// doubling so this is a template-side artifact, not our renderer; mlx-lm
/// must be tolerating it via reduction-order quirks we don't share. Strip
/// it here. Safe across all model families because `</think></think>` never
/// appears legitimately.
fn collapseDoubledThinkTags(allocator: std.mem.Allocator, rendered: []const u8) ![]u8 {
    const tag = "</think>";
    const needle = "</think></think>";
    if (std.mem.indexOf(u8, rendered, needle) == null) {
        return allocator.dupe(u8, rendered);
    }
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.ensureTotalCapacity(allocator, rendered.len);
    var i: usize = 0;
    while (i < rendered.len) {
        // If we see `</think></think>...`, drop the FIRST `</think>` and let
        // the next iteration handle whatever follows — that way 3x, 4x, …
        // also collapse to a single tag without a separate pass.
        if (i + needle.len <= rendered.len and
            std.mem.eql(u8, rendered[i .. i + needle.len], needle))
        {
            i += tag.len; // skip the first `</think>`; next iter writes the survivor
        } else {
            try out.append(allocator, rendered[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(allocator);
}

/// True if the template contains a branch keyed on the literal `tool` role string
/// (`'tool'` or `"tool"` as a standalone token, not `tool_calls` / `tool_call_id`).
fn templateReferencesToolRole(tpl: []const u8) bool {
    const patterns = [_][]const u8{ "'tool'", "\"tool\"" };
    for (patterns) |p| {
        if (std.mem.indexOf(u8, tpl, p) != null) return true;
    }
    return false;
}

/// True if any message has `role: "tool"` or an assistant message with tool_calls.
fn messagesHaveToolContent(messages: []const Message) bool {
    for (messages) |m| {
        if (std.mem.eql(u8, m.role, "tool")) return true;
        if (m.tool_calls != null) return true;
    }
    return false;
}

/// Synthesize a messages array suitable for templates that have no tool/role support.
/// - inject_tool_prompt: prepend (or merge into) a system message with the tool prompt
/// - rewrite_tool_role: rewrite role:"tool" messages to role:"user" with <tool_response> wrapping
/// - rewrite_tool_calls: rewrite assistant messages that carry tool_calls into plain
///   content emitting <tool_call>...</tool_call> blocks (matches what we instruct the
///   model to produce, so multi-turn context stays consistent).
fn synthesizeToolFallbackMessages(
    arena: std.mem.Allocator,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    inject_tool_prompt: bool,
    rewrite_tool_role: bool,
    rewrite_tool_calls: bool,
) ![]Message {
    var out = std.ArrayList(Message).empty;

    var tool_prompt: ?[]const u8 = null;
    if (inject_tool_prompt and tools_json != null) {
        var buf = std.ArrayList(u8).empty;
        try appendToolSystemPrompt(arena, &buf, tools_json.?, tool_choice_instruction);
        tool_prompt = try buf.toOwnedSlice(arena);
    }

    var injected = false;
    if (tool_prompt) |tp| {
        if (messages.len == 0 or !std.mem.eql(u8, messages[0].role, "system")) {
            try out.append(arena, .{ .role = "system", .content = tp });
            injected = true;
        }
    }

    for (messages, 0..) |msg, i| {
        if (tool_prompt) |tp| {
            if (!injected and i == 0 and std.mem.eql(u8, msg.role, "system")) {
                const merged = if (msg.content.len > 0)
                    try std.fmt.allocPrint(arena, "{s}\n\n{s}", .{ msg.content, tp })
                else
                    try arena.dupe(u8, tp);
                try out.append(arena, .{
                    .role = "system",
                    .content = merged,
                    .tool_calls = msg.tool_calls,
                    .tool_call_id = msg.tool_call_id,
                    .images = msg.images,
                });
                injected = true;
                continue;
            }
        }

        if (rewrite_tool_role and std.mem.eql(u8, msg.role, "tool")) {
            const wrapped = try std.fmt.allocPrint(arena, "<tool_response>\n{s}\n</tool_response>", .{msg.content});
            try out.append(arena, .{
                .role = "user",
                .content = wrapped,
            });
            continue;
        }

        if (rewrite_tool_calls and std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            var buf = std.ArrayList(u8).empty;
            if (msg.content.len > 0) {
                try buf.appendSlice(arena, msg.content);
                try buf.append(arena, '\n');
            }
            for (msg.tool_calls.?) |tc| {
                try buf.appendSlice(arena, "<tool_call>\n{\"name\": \"");
                try buf.appendSlice(arena, tc.name);
                try buf.appendSlice(arena, "\", \"arguments\": ");
                try buf.appendSlice(arena, tc.arguments);
                try buf.appendSlice(arena, "}\n</tool_call>\n");
            }
            const owned = try buf.toOwnedSlice(arena);
            try out.append(arena, .{
                .role = "assistant",
                .content = std.mem.trimEnd(u8, owned, "\n"),
                .images = msg.images,
            });
            continue;
        }

        try out.append(arena, msg);
    }

    return out.toOwnedSlice(arena);
}

pub fn serializeMessagesJson(allocator: std.mem.Allocator, messages: []const Message) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '[');
    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "{\"role\":");
        try appendJsonString(allocator, &buf, msg.role);

        try buf.appendSlice(allocator, ",\"content\":");
        if (msg.content.len > 0) {
            try appendJsonString(allocator, &buf, msg.content);
        } else {
            try buf.appendSlice(allocator, "null");
        }

        if (msg.tool_calls) |tcs| {
            try buf.appendSlice(allocator, ",\"tool_calls\":[");
            for (tcs, 0..) |tc, ti| {
                if (ti > 0) try buf.append(allocator, ',');
                try buf.appendSlice(allocator, "{\"type\":\"function\",");
                if (tc.id.len > 0) {
                    // Inkling's template names a tool RESULT by matching
                    // `message.tool_call_id` against the history call's `id`
                    // (`tc.id is defined and tc.id == message.tool_call_id`) —
                    // without the id the result renders nameless. Inert for
                    // templates that never read it.
                    try buf.appendSlice(allocator, "\"id\":");
                    try appendJsonString(allocator, &buf, tc.id);
                    try buf.append(allocator, ',');
                }
                try buf.appendSlice(allocator, "\"function\":{\"name\":");
                try appendJsonString(allocator, &buf, tc.name);
                try buf.appendSlice(allocator, ",\"arguments\":");
                // Embed arguments as a JSON OBJECT when it parses cleanly — Qwen
                // 3.5/3.6 templates do `tool_call.arguments|items` which requires
                // a dict. Templates that need a string (e.g. Gemma 4 with `tojson`
                // or `string`) still get a usable value. Falls back to a string
                // for malformed arguments so downstream still sees something.
                if (std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{})) |parsed| {
                    defer parsed.deinit();
                    if (parsed.value == .object) {
                        try buf.appendSlice(allocator, tc.arguments);
                    } else {
                        try appendJsonString(allocator, &buf, tc.arguments);
                    }
                } else |_| {
                    try appendJsonString(allocator, &buf, tc.arguments);
                }
                try buf.appendSlice(allocator, "}}");
            }
            try buf.append(allocator, ']');
        }

        if (msg.tool_call_id) |tid| {
            try buf.appendSlice(allocator, ",\"tool_call_id\":");
            try appendJsonString(allocator, &buf, tid);
            // No tool_responses field needed — Gemma 4 templates handle role:"tool"
            // natively via the content field. Adding tool_responses causes the template
            // to render duplicate content, wasting tokens.
        }

        if (msg.reasoning_content) |rc| {
            // Key omitted entirely when absent: templates gate on
            // `message.reasoning_content is string`, and an explicit null
            // would still pass `is defined`-style checks.
            try buf.appendSlice(allocator, ",\"reasoning_content\":");
            try appendJsonString(allocator, &buf, rc);
        }

        try buf.append(allocator, '}');
    }
    try buf.append(allocator, ']');

    return buf.toOwnedSlice(allocator);
}

/// Fill the OPTIONAL tool-definition keys a template may dump blindly.
///
/// `description` is optional in OpenAI's function schema and a no-arg tool
/// needs no `parameters`, but a template that renders `fn.description | tojson`
/// (muse) or `tool | tojson` wholesale hits `tojson` on Undefined, which is a
/// jinja RAISE — i.e. a silent `fallbackFormatChat` with no tools preamble at
/// all, for a request that was perfectly legal. Returns null when every
/// function already carries both keys, so the common path re-emits nothing and
/// the caller keeps the client's own bytes (provenance, not content, decides
/// ownership).
///
/// Defaults are the empty ones: `""` says the client supplied no description,
/// and an empty-properties object is what a no-argument tool means. Neither
/// invents a capability the client did not declare.
pub fn fillOptionalToolDefKeys(allocator: std.mem.Allocator, tools_json: []const u8) !?[]u8 {
    // Own the intermediates: growing a parsed ObjectMap allocates OUTSIDE the
    // parse's arena, so mutating with the caller's allocator leaks unless the
    // caller happens to be an arena too. Only the returned bytes are theirs.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, a, tools_json, .{}) catch return null;
    if (parsed.value != .array) return null;

    var changed = false;
    for (parsed.value.array.items) |*tool| {
        if (tool.* != .object) continue;
        const fn_val = tool.object.getPtr("function") orelse continue;
        if (fn_val.* != .object) continue;
        if (fn_val.object.get("description") == null) {
            try fn_val.object.put(a, "description", .{ .string = "" });
            changed = true;
        }
        if (fn_val.object.get("parameters") == null) {
            var params: std.json.ObjectMap = .empty;
            try params.put(a, "type", .{ .string = "object" });
            try params.put(a, "properties", .{ .object = .empty });
            try fn_val.object.put(a, "parameters", .{ .object = params });
            changed = true;
        }
    }
    if (!changed) return null;

    return try std.json.Stringify.valueAlloc(allocator, parsed.value, .{});
}

/// Whether the template READS `reasoning_effort` — i.e. the word is a
/// BEHAVIORAL lever on this checkpoint (dsv4's preamble, Qwen3.8's and
/// Inkling's effort arms) rather than a string we hand over and it ignores.
/// Where it is, the server must NOT also derive a token budget from the same
/// word: the budget only truncates what the client is shown, so the two
/// levers fight — pi asking for `medium` on Qwen3.8 got the model's default
/// (unguided, long) thinking AND a 2048-token display cut, then 25.9k tokens
/// of invisible generation (live 2026-08-14). An explicit
/// `reasoning_budget_tokens` or `--reasoning-budget` is someone asking for a
/// cap on purpose and still applies.
///
/// Substring on the KEY, so a family reading a different one is untouched:
/// muse's `reasoning_strength` is a near-miss the corpus pins.
pub fn templateConsumesEffort(tpl: []const u8) bool {
    return std.mem.indexOf(u8, tpl, "reasoning_effort") != null;
}

/// OpenAI's effort vocabulary → DeepSeek's low|high|max. `medium` maps to
/// low deliberately: DeepSeek's `high` text is "Absolute maximum with no
/// shortcuts permitted" and its own default is low, so nothing verbose is
/// injected unless the client explicitly asked for it. Unknown strings also
/// fall back to low for the same reason.
pub fn dsv4EffortFor(effort: ?[]const u8) []const u8 {
    const e = effort orelse return "low";
    if (std.mem.eql(u8, e, "high")) return "high";
    if (std.mem.eql(u8, e, "xhigh") or std.mem.eql(u8, e, "max") or std.mem.eql(u8, e, "ultra")) return "max";
    return "low";
}

/// OpenAI's effort vocabulary -> Qwen3.8's xhigh|medium|low. The template
/// raise_exception's on any other string and its own default is xhigh, so an
/// absent or unrecognized effort keeps the checkpoint default. Thinking-off
/// maps to low for the variants that REFUSE it (2.4T-A95B): the request is
/// served on the cheapest arm with the think block closed in the prompt by
/// `noThinkTailSuffix`. On a variant that accepts thinking-off (27B) the
/// template never reads the key on that arm, so the value is inert.
fn qwen38EffortFor(effort: ?[]const u8, enable_thinking: bool) []const u8 {
    if (!enable_thinking) return "low";
    // An ABSENT effort takes OUR default, not the checkpoint's. The template
    // defaults to xhigh, whose preamble asks the model to validate assumptions
    // and consider alternatives, and nothing bounds the result: a pi agent turn
    // that sent no effort ran 16803 reasoning tokens before the client gave up
    // (live 2026-08-14). The server-side `reasoning_effort` budget is NOT the
    // answer to that — it truncates rather than shortens, and on exhaustion the
    // model is still inside its block, so the rest of the thought streams as
    // CONTENT. Low is the only lever that makes the model itself stop early
    // ("keep your thinking brief and focused, moving directly to the
    // conclusion"). A client that wants the checkpoint default still gets it by
    // asking for it; only silence changed meaning.
    const e = effort orelse return "low";
    if (std.mem.eql(u8, e, "medium")) return "medium";
    if (std.mem.eql(u8, e, "low") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "none")) return "low";
    return "xhigh";
}

/// `effort` is the client's raw `reasoning_effort` string (null when the
/// request didn't send one) — today only the dsv4 family maps it into the
/// template; other families keep their fixed vocabulary.
fn serializeExtraContext(allocator: std.mem.Allocator, chat_config: *const ChatConfig, enable_thinking: bool, effort: ?[]const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '{');
    var need_comma = false;

    if (chat_config.bos_token) |bos| {
        try buf.appendSlice(allocator, "\"bos_token\":");
        try appendJsonString(allocator, &buf, bos);
        need_comma = true;
    }
    if (chat_config.eos_token) |eos| {
        if (need_comma) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "\"eos_token\":");
        try appendJsonString(allocator, &buf, eos);
        need_comma = true;
    }
    // Qwen3.8's effort vocabulary, sniffed on the jinja STRING LITERAL 'xhigh'
    // — it appears in both the `reasoning_effort|default('xhigh')` line and the
    // accepted-values tuple, so a reworded raise message can't drift the
    // detection. EVERY 3.8 template raises on OpenAI's "high", so this keys the
    // mapping for the whole family.
    const qwen38_style = std.mem.indexOf(u8, chat_config.chat_template, "'xhigh'") != null;
    // Whether the template REFUSES thinking-off is a SEPARATE question from the
    // effort vocabulary, and the two split inside one family: 2.4T-A95B raises
    // ("Disabling thinking is not supported"), the 27B answers that arm the 3.6
    // way with a closed `<think>\n\n</think>` and NO reasoning preamble. Keying
    // the override on the shared 'xhigh' literal forced the 27B's thinking-off
    // requests onto the thinking-on arm, injecting a system line the checkpoint
    // never renders there. Only a refusing template gets the flag withheld;
    // thinking-off is then committed in the prompt by `noThinkTailSuffix`.
    const refuses_nothink = std.mem.indexOf(u8, chat_config.chat_template, "Disabling thinking is not supported") != null;
    if (need_comma) try buf.append(allocator, ',');
    if (enable_thinking or refuses_nothink) {
        try buf.appendSlice(allocator, "\"enable_thinking\":true");
    } else {
        try buf.appendSlice(allocator, "\"enable_thinking\":false");
    }
    // Hy3 templates control thinking via `reasoning_effort` (high/low/
    // no_think), not enable_thinking — and default HIGH when the var is
    // absent. Always supply the mapped value so thinking-off requests
    // actually close the think block; templates that don't read it ignore it.
    //
    // Inkling's template ALSO reads `reasoning_effort` but with a different
    // vocabulary (none/minimal/low/medium/high/max) and raise_exception's on
    // anything else — hy3's "no_think" would fail the render and silently
    // swap in fallbackFormatChat (wrong-family tags). Its thinking-off word
    // is "none"; sniffed by the effort header string unique to that family.
    // Qwen3.8 reads it too, with a THIRD vocabulary (xhigh|medium|low,
    // default xhigh) and its own raise_exception — hy3's "high"/"no_think"
    // fail the render on every request, thinking on or off.
    // DeepSeek-V4-0731's encoder made `reasoning_effort` a THREE-level
    // vocabulary (low|high|max) whose high/max levels PREPEND a verbose
    // "Reasoning Effort: …" preamble to the whole conversation, with low (=
    // nothing) as the reference default. hy3's blanket "high" would inject
    // that preamble into every thinking request on this family, so dsv4 maps
    // the CLIENT's effort string through `dsv4EffortFor` — high/max opt in,
    // everything else (incl. absent) stays the reference default.
    const dsv4_style = std.mem.indexOf(u8, chat_config.chat_template, "thinking_mode") != null;
    const inkling_style = std.mem.indexOf(u8, chat_config.chat_template, "Thinking effort level") != null;
    if (dsv4_style) {
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, dsv4EffortFor(effort));
        try buf.append(allocator, '"');
    } else if (qwen38_style) {
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, qwen38EffortFor(effort, enable_thinking));
        try buf.append(allocator, '"');
    } else try buf.appendSlice(allocator, if (enable_thinking)
        ",\"reasoning_effort\":\"high\""
    else if (inkling_style)
        ",\"reasoning_effort\":\"none\""
    else
        ",\"reasoning_effort\":\"no_think\"");

    // DeepSeek-V4's template switches on `thinking_mode` ("chat"|"thinking",
    // the reference encoder's own vocabulary). Only emitted when the template
    // reads it — other families never see the key.
    if (std.mem.indexOf(u8, chat_config.chat_template, "thinking_mode") != null) {
        try buf.appendSlice(allocator, if (enable_thinking)
            ",\"thinking_mode\":\"thinking\""
        else
            ",\"thinking_mode\":\"chat\"");
    }

    // Muse-Glimmer reads `reasoning_strength` (its template default is
    // 'high' and it has no thinking-off switch — the model always emits a
    // to=self segment). Map the client's effort onto low|medium|high;
    // thinking-off drops to 'low'. Only emitted when the template reads it.
    if (std.mem.indexOf(u8, chat_config.chat_template, "reasoning_strength") != null) {
        const strength: []const u8 = if (!enable_thinking)
            "low"
        else if (effort) |e| blk: {
            if (std.mem.eql(u8, e, "none") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "low")) break :blk "low";
            if (std.mem.eql(u8, e, "medium")) break :blk "medium";
            break :blk "high";
        } else "high";
        try buf.appendSlice(allocator, ",\"reasoning_strength\":\"");
        try buf.appendSlice(allocator, strength);
        try buf.append(allocator, '"');
    }

    try buf.append(allocator, '}');
    return buf.toOwnedSlice(allocator);
}

/// Encode text that may contain special tokens (like <|im_start|>, <bos>, etc.).
fn encodeWithSpecialTokens(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    text: []const u8,
    ids: *std.ArrayList(u32),
) !void {
    // `Tokenizer.encode` already splits around special tokens (earliest
    // occurrence, longest at a position) with an O(text) first-byte-bucketed
    // scan. This wrapper used to duplicate that split with its own
    // O(specials x text) re-search per segment — ~11 s per 66 KB rendered
    // prompt on gemma-3's 6415-special vocabulary (the tokenizer-side twin
    // of the same class was fixed in tokenizer.zig; keep both on the shared
    // fast path so they can't drift apart again).
    const segment_ids = try tok.encode(allocator, text);
    defer allocator.free(segment_ids);
    try ids.appendSlice(allocator, segment_ids);
}

const SpecialEntry = struct {
    text: []const u8,
    id: u32,
};

/// Fallback chat formatting for when Jinja rendering fails.
fn fallbackFormatChat(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    const is_chatml = chat_config.eos_token != null and
        std.mem.indexOf(u8, chat_config.eos_token.?, "<|im_end|>") != null;

    const has_system = messages.len > 0 and std.mem.eql(u8, messages[0].role, "system");

    if (is_chatml) {
        // ChatML format (Qwen3, etc.)
        if (tools_json != null and !has_system) {
            try result.appendSlice(allocator, "<|im_start|>system\n");
            try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
            try result.appendSlice(allocator, "<|im_end|>\n");
        }

        for (messages) |msg| {
            if (std.mem.eql(u8, msg.role, "system") and tools_json != null) {
                try result.appendSlice(allocator, "<|im_start|>system\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "\n\n");
                try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
                try result.appendSlice(allocator, "<|im_end|>\n");
            } else if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
                try result.appendSlice(allocator, "<|im_start|>assistant\n");
                if (msg.content.len > 0) {
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "\n");
                }
                for (msg.tool_calls.?) |tc| {
                    try result.appendSlice(allocator, "<tool_call>\n");
                    try result.appendSlice(allocator, "{\"name\": \"");
                    try result.appendSlice(allocator, tc.name);
                    try result.appendSlice(allocator, "\", \"arguments\": ");
                    try result.appendSlice(allocator, tc.arguments);
                    try result.appendSlice(allocator, "}\n</tool_call>");
                }
                try result.appendSlice(allocator, "<|im_end|>\n");
            } else if (std.mem.eql(u8, msg.role, "tool")) {
                try result.appendSlice(allocator, "<|im_start|>user\n");
                try result.appendSlice(allocator, "<tool_response>\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "\n</tool_response>");
                try result.appendSlice(allocator, "<|im_end|>\n");
            } else {
                try result.appendSlice(allocator, "<|im_start|>");
                try result.appendSlice(allocator, msg.role);
                try result.appendSlice(allocator, "\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "<|im_end|>\n");
            }
        }
        try result.appendSlice(allocator, "<|im_start|>assistant\n");
        if (std.mem.indexOf(u8, chat_config.chat_template, "enable_thinking") != null) {
            if (enable_thinking) {
                try result.appendSlice(allocator, "<think>\n");
            } else {
                try result.appendSlice(allocator, "<think>\n\n</think>\n\n");
            }
        }
    } else {
        // Gemma/Llama-style format
        if (chat_config.bos_token) |bos| {
            try result.appendSlice(allocator, bos);
        }

        const is_llama = std.mem.indexOf(u8, chat_config.chat_template, "start_header_id") != null;

        if (is_llama) {
            // Llama 3 format with tool support
            try result.appendSlice(allocator, "<|start_header_id|>system<|end_header_id|>\n\n");
            if (tools_json != null) {
                try result.appendSlice(allocator, "Environment: ipython\n");
            }
            if (has_system) {
                try result.appendSlice(allocator, messages[0].content);
            } else {
                try result.appendSlice(allocator, "You are a helpful assistant.");
            }
            if (tools_json != null) {
                try result.appendSlice(allocator, "\n\n");
                try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
            }
            try result.appendSlice(allocator, "<|eot_id|>");

            const start_idx: usize = if (has_system) 1 else 0;
            for (messages[start_idx..]) |msg| {
                if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
                    try result.appendSlice(allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
                    for (msg.tool_calls.?) |tc| {
                        try result.appendSlice(allocator, "{\"name\": \"");
                        try result.appendSlice(allocator, tc.name);
                        try result.appendSlice(allocator, "\", \"parameters\": ");
                        try result.appendSlice(allocator, tc.arguments);
                        try result.appendSlice(allocator, "}");
                    }
                    try result.appendSlice(allocator, "<|eot_id|>");
                } else if (std.mem.eql(u8, msg.role, "tool")) {
                    try result.appendSlice(allocator, "<|start_header_id|>ipython<|end_header_id|>\n\n");
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "<|eot_id|>");
                } else {
                    try result.appendSlice(allocator, "<|start_header_id|>");
                    try result.appendSlice(allocator, msg.role);
                    try result.appendSlice(allocator, "<|end_header_id|>\n\n");
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "<|eot_id|>");
                }
            }
            try result.appendSlice(allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
        } else {
            // Gemma-style format
            if (tools_json != null and !has_system) {
                try result.appendSlice(allocator, "<start_of_turn>user\n");
                try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
                try result.appendSlice(allocator, "<end_of_turn>\n");
            }
            for (messages) |msg| {
                if (std.mem.eql(u8, msg.role, "tool")) {
                    try result.appendSlice(allocator, "<start_of_turn>user\n");
                    try result.appendSlice(allocator, "Tool result: ");
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "<end_of_turn>\n");
                } else {
                    try result.appendSlice(allocator, "<start_of_turn>");
                    try result.appendSlice(allocator, msg.role);
                    try result.appendSlice(allocator, "\n");
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "<end_of_turn>\n");
                }
            }
            try result.appendSlice(allocator, "<start_of_turn>model\n");
        }
    }

    return result.toOwnedSlice(allocator);
}

/// The FIRST unclosed think/thought opener: the earliest `<|channel>thought`
/// or `<think>` in `text` with no matching close tag anywhere after it.
/// Gemma 4 opens a NEW thought channel at the very end of a prose turn (its
/// channel-thought tail behavior) — and sometimes SEVERAL in a row, none of
/// them closed (seen live from the 26B GGUF via pi). Everything from the
/// first such opener onward is dangling thought; cutting at the LAST opener
/// instead leaks the earlier raw tags into visible content. A pos-0 opener is
/// reported too: callers (the split arms / stripTrailingThinkOpen) strip
/// their leading CLOSED block first, so an unclosed opener at the start of the
/// remainder is a genuine dangling re-open (`…<channel|>\n<|channel>thought\n`,
/// 2026-06-19 live) — excluding it leaked the bare opener into content.
const TrailingThinkOpen = struct { pos: usize, after: usize };
fn lastUnclosedThinkOpen(text: []const u8) ?TrailingThinkOpen {
    var from: usize = 0;
    while (nextThinkOpen(text, from)) |o| {
        const close = if (o.is_think_style)
            indexOfThinkCloseTag(text, o.after)
        else if (std.mem.indexOfPos(u8, text, o.after, "<channel|>")) |p|
            TagAt{ .pos = p, .len = "<channel|>".len }
        else
            null;
        if (close) |c| {
            // This block IS closed — keep scanning past its close.
            from = c.pos + c.len;
            continue;
        }
        return .{ .pos = o.pos, .after = o.after };
    }
    return null;
}

// ── Suffixed think/tool tag matching (Hy3 / Hunyuan 3) ──
//
// Hy3 templates suffix every control tag with a release marker:
// `<think:opensource>`, `</think:opensource>`, `<tool_call:opensource>`, …
// (HYTK in chat_template.jinja; future releases may carry a different
// suffix). The helpers below match `BASE>` or `BASE:[A-Za-z0-9_-]+>` so the
// shared think machinery handles both the canonical and suffixed families
// without hardcoding any specific suffix.

pub const TagAt = struct { pos: usize, len: usize };

fn tagSuffixChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
        (c >= '0' and c <= '9') or c == '_' or c == '-';
}

/// Length of a complete tag at the START of `text` whose base is `base`
/// (e.g. base "<think" matches "<think>" and "<think:opensource>").
fn suffixedTagLenAt(text: []const u8, comptime base: []const u8) ?usize {
    if (!std.mem.startsWith(u8, text, base)) return null;
    var i: usize = base.len;
    if (i < text.len and text[i] == '>') return i + 1;
    if (i >= text.len or text[i] != ':') return null;
    i += 1;
    const suffix_start = i;
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if (c == '>') return if (i > suffix_start) i + 1 else null;
        if (!tagSuffixChar(c)) return null;
    }
    return null;
}

/// True when `text` could still GROW into a `base`-tag with more streamed
/// bytes (strict prefix: "<th", "<think", "<think:", "<think:opensou").
fn isPartialSuffixedTag(text: []const u8, comptime base: []const u8) bool {
    if (text.len == 0) return false;
    if (text.len <= base.len) return std.mem.startsWith(u8, base, text);
    if (!std.mem.startsWith(u8, text, base)) return false;
    if (text[base.len] != ':') return false;
    for (text[base.len + 1 ..]) |c| {
        if (!tagSuffixChar(c)) return false;
    }
    return true;
}

pub fn thinkOpenTagLenAt(text: []const u8) ?usize {
    return suffixedTagLenAt(text, "<think");
}

pub fn thinkCloseTagLenAt(text: []const u8) ?usize {
    return suffixedTagLenAt(text, "</think");
}

fn indexOfThinkOpenTag(text: []const u8, from: usize) ?TagAt {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, "<think")) |p| {
        if (thinkOpenTagLenAt(text[p..])) |l| return .{ .pos = p, .len = l };
        i = p + "<think".len;
    }
    return null;
}

pub fn indexOfThinkCloseTag(text: []const u8, from: usize) ?TagAt {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, "</think")) |p| {
        if (thinkCloseTagLenAt(text[p..])) |l| {
            if (!thinkCloseIsToolCallPayload(text, p)) return .{ .pos = p, .len = l };
        }
        i = p + "</think".len;
    }
    return null;
}

/// Whether a `</think>` at `pos` is an ARGUMENT VALUE rather than a block close.
///
/// The think split runs before tool parse, and this is the shared close-tag
/// scan every surface uses — so a model writing about its own prompt format
/// (`<arg_value>…closes a thought with </think>…</arg_value>`) had the split cut
/// straight through the call, losing it entirely and leaking the fragments.
/// Agent traffic writes files about prompts.
///
/// Payload ⟺ the nearest tool opener before `pos` is still OPEN there (no
/// `</tool_call` between it and `pos`) AND the block does close afterwards.
/// Both halves are load-bearing:
///   • without the first, a call the model emitted and CLOSED inside its
///     thought would make the real `</think>` after it look like payload;
///   • without the second, an unclosed opener inside a thought (the leaked-
///     markup case, which `trimLeakedToolMarkup` handles downstream) would
///     swallow the answer that follows the close.
fn thinkCloseIsToolCallPayload(text: []const u8, pos: usize) bool {
    // `streamThinkGate2` calls the close scan once per TOKEN on a growing
    // buffer, so the common case — text with no tool markup at all — must cost
    // one substring scan, not a walk per opener spelling. Both spellings share
    // this needle.
    if (std.mem.indexOf(u8, text[0..pos], "tool_call") == null) return false;
    // Nearest `<tool_call`-family opener before `pos`. Only this family is
    // considered: it is the one whose bodies carry free-form argument text.
    var open: ?usize = null;
    for ([_][]const u8{ "<tool_call", "<|tool_call" }) |m| {
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, text, i, m)) |p| {
            if (p >= pos) break;
            if (open == null or p > open.?) open = p;
            i = p + 1;
        }
    }
    const o = open orelse return false;
    if (std.mem.indexOfPos(u8, text, o, "</tool_call")) |c| {
        if (c < pos) return false; // that call already closed — a real block close
        return true;
    }
    return false; // never closes — treat as leaked markup, not payload
}

/// Length of the trailing bytes of `buf` that could still GROW into a think
/// close tag with more streamed bytes — either family: `</think>`, the
/// Hy3-suffixed `</think:opensource>`, or Gemma's `<channel|>`. Streaming
/// reasoning flushes must hold this suffix back: flushing half a close tag
/// splits it across the flush boundary, so the retained remainder never
/// matches and the stream stays "inside thinking" forever (tag bytes leak
/// into reasoning_content). 0 = safe to flush everything.
pub fn partialThinkCloseSuffixLen(buf: []const u8) usize {
    // Any real close tag is well under 32 bytes; a longer '<'-run is prose.
    const window_start = if (buf.len > 32) buf.len - 32 else 0;
    var i = buf.len;
    while (i > window_start) {
        i -= 1;
        if (buf[i] != '<') continue;
        const tail = buf[i..];
        if (isPartialSuffixedTag(tail, "</think")) return tail.len;
        if (tail.len < "<channel|>".len and std.mem.startsWith(u8, "<channel|>", tail)) return tail.len;
    }
    return 0;
}

test "partialThinkCloseSuffixLen holds back growing close tags, ignores prose" {
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("plain reasoning text"));
    try testing.expectEqual(@as(usize, 7), partialThinkCloseSuffixLen("thinking…</think"));
    try testing.expectEqual(@as(usize, 15), partialThinkCloseSuffixLen("thinking…</think:opensou"));
    try testing.expectEqual(@as(usize, 5), partialThinkCloseSuffixLen("thinking…<chan"));
    // A COMPLETE tag is not a partial — the scan-side indexOf owns it.
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("done</think:opensource>"));
    // A '<' too far back is prose, not a growing tag.
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("a < b and then lots of ordinary words follow here"));
}

/// Tag length when `text` ENDS with a complete think-close tag.
fn endsWithThinkCloseTag(text: []const u8) ?usize {
    if (text.len == 0 or text[text.len - 1] != '>') return null;
    const lt = std.mem.lastIndexOfScalar(u8, text, '<') orelse return null;
    const l = thinkCloseTagLenAt(text[lt..]) orelse return null;
    return if (lt + l == text.len) l else null;
}

/// Tag length when `text` ENDS with a complete think-open tag.
fn endsWithThinkOpenTag(text: []const u8) ?usize {
    if (text.len == 0 or text[text.len - 1] != '>') return null;
    const lt = std.mem.lastIndexOfScalar(u8, text, '<') orelse return null;
    const l = thinkOpenTagLenAt(text[lt..]) orelse return null;
    return if (lt + l == text.len) l else null;
}

/// Tool-call WRAPPER openers that must never survive into text a client
/// renders. Every entry is the opening marker of a call block, so whatever
/// follows it is call payload, not prose — parseToolCalls has already run by
/// the time these matter, so a marker still present means the block did not
/// parse (a mangled opener, a bare marker with no call behind it, or a call
/// the model emitted inside its think block, which the parser deliberately
/// skips). The whole tail is cut rather than excised: a wrapper we could not
/// parse has no reliable end, and shipping half of it is the same leak.
const tool_markup_openers = [_][]const u8{
    "<｜DSML｜", // DeepSeek-V4 native (covers tool_calls/invoke/parameter)
    "<|tool_call", // Gemma 4
    "<tool_call", // Hermes XML + the plural/suffixed variants
    "<tool_calls:", // Hy3 suffixed wrapper
    "<atem:", // Muse-Glimmer ATEM (covers function_calls/invoke/parameter)
    INKLING_INVOKE_TAG,
};

/// Cut visible text at the first unparsed tool-call marker. Returns a prefix
/// slice (no alloc); text with no marker is returned untouched.
pub fn trimLeakedToolMarkup(text: []const u8) []const u8 {
    var cut: usize = text.len;
    var is_inkling = false;
    for (tool_markup_openers) |m| {
        if (std.mem.indexOf(u8, text, m)) |p| {
            if (p < cut) {
                cut = p;
                is_inkling = std.mem.eql(u8, m, INKLING_INVOKE_TAG);
            }
        }
    }
    if (cut == text.len) return text;
    // Inkling puts the tool NAME immediately BEFORE its invoke marker, so the
    // cut takes the trailing identifier run with it — otherwise the bare name
    // is what the client renders as the answer.
    if (is_inkling) {
        while (cut > 0 and inklingIsNameChar(text[cut - 1])) cut -= 1;
    }
    return std.mem.trimEnd(u8, text[0..cut], "\n\r\t ");
}

// NOTE: there is deliberately NO strip-think helper anymore. Reasoning the
// model generated is ALWAYS delivered (reasoning_content / thinking blocks) —
// thinking-off is enforced on the PROMPT side (`noThinkTailSuffix`), so every
// delivery site takes `splitThinkBlock(text, true, …)` and ships both halves.

/// Trim trailing think/channel/tool CLOSE markers (and surrounding whitespace)
/// from visible content. A close marker is never valid at the tail of content —
/// but a degenerate model can spam them (live: a Gemma variant emitted 16 bare
/// `<channel|>` after its answer; the leading strip cuts the FIRST close, the
/// trailing-OPEN strip ignores closes, so they leaked). The Gemma tool CLOSE
/// `<tool_call|>` is here too: a degenerate 1-token bare close with no
/// `<|tool_call>` opener leaked as the entire content (live 2026-07-16 soak,
/// gemma-4-26B). parseToolCalls runs BEFORE this strip and extracts any real
/// call, so any residual `<tool_call|>` reaching here is orphan by construction.
/// Loops so a run of closers is fully removed. Returns a prefix slice (no alloc).
fn trimTrailingThinkClosers(content: []const u8) []const u8 {
    var s = content;
    while (true) {
        const t = std.mem.trimEnd(u8, s, "\n \t\r");
        if (std.mem.endsWith(u8, t, "<channel|>")) {
            s = t[0 .. t.len - "<channel|>".len];
        } else if (std.mem.endsWith(u8, t, "<tool_call|>")) {
            s = t[0 .. t.len - "<tool_call|>".len];
        } else if (endsWithThinkCloseTag(t)) |l| {
            s = t[0 .. t.len - l];
        } else {
            return t;
        }
    }
}

/// Truncate a trailing unclosed thought opener (and its dangling thought)
/// out of visible content. Used by the split arms after the leading block has
/// been handled.
fn stripTrailingThinkOpen(content: []const u8) []const u8 {
    if (lastUnclosedThinkOpen(content)) |o| {
        return std.mem.trimEnd(u8, content[0..o.pos], "\n ");
    }
    return content;
}

pub const ThinkSplit = struct {
    reasoning_content: ?[]const u8,
    content: []const u8,
};

/// True when a rendered generation prompt ends inside a think block the
const INKLING_THINKING_TAG = "<|content_thinking|>";
const INKLING_TEXT_TAG = "<|content_text|>";
const INKLING_END_TAG = "<|end_message|>";
const INKLING_MODEL_TAG = "<|message_model|>";
const INKLING_INVOKE_TAG = "<|content_invoke_tool_json|>";

/// Control-channel marker TOKENS that must never appear in a visible stream
/// delta. Every entry is a single special token in its family's vocab, so
/// exact-match filtering at the delta level is complete — a marker can't be
/// split across deltas. Shared by every streaming surface's flush path.
pub fn isChannelMarkerToken(tt: []const u8) bool {
    return std.mem.eql(u8, tt, "<|channel>") or std.mem.eql(u8, tt, "<channel|>") or
        std.mem.eql(u8, tt, "<think>") or std.mem.eql(u8, tt, "</think>") or
        std.mem.eql(u8, tt, INKLING_TEXT_TAG) or std.mem.eql(u8, tt, INKLING_END_TAG) or
        std.mem.eql(u8, tt, INKLING_MODEL_TAG) or std.mem.eql(u8, tt, INKLING_THINKING_TAG) or
        std.mem.eql(u8, tt, MUSE_START_TAG) or std.mem.eql(u8, tt, MUSE_MSG_TAG) or
        std.mem.eql(u8, tt, MUSE_EOM_TAG) or std.mem.eql(u8, tt, MUSE_EOT_TAG);
}

/// Strip a leading `<|message_model|>` (and, on raw completions, a leading
/// `<|end_message|>` closing the prompt's implicit message) so the channel
/// marker is at position 0.
fn inklingStripMessageHead(text: []const u8) []const u8 {
    var t = std.mem.trimStart(u8, text, "\n ");
    if (std.mem.startsWith(u8, t, INKLING_END_TAG)) t = t[INKLING_END_TAG.len..];
    if (std.mem.startsWith(u8, t, INKLING_MODEL_TAG)) t = t[INKLING_MODEL_TAG.len..];
    return t;
}

/// Content runs to its `<|end_message|>` (anything after is a re-opened
/// message the stop token cut, or trailing markers — dropped).
fn inklingTrimContentTail(content_in: []const u8) []const u8 {
    var content = content_in;
    if (std.mem.indexOf(u8, content, INKLING_END_TAG)) |em| content = content[0..em];
    return std.mem.trim(u8, content, "\n ");
}

/// The Inkling arm of splitThinkBlock: null when the text carries no Inkling
/// channel markers (every other family falls through).
fn splitInklingChannels(text: []const u8) ?ThinkSplit {
    const t = inklingStripMessageHead(text);
    if (std.mem.startsWith(u8, t, INKLING_THINKING_TAG)) {
        const body = t[INKLING_THINKING_TAG.len..];
        if (std.mem.indexOf(u8, body, INKLING_END_TAG)) |em| {
            const reasoning = std.mem.trim(u8, body[0..em], "\n ");
            var content = inklingStripMessageHead(body[em + INKLING_END_TAG.len ..]);
            if (std.mem.startsWith(u8, content, INKLING_TEXT_TAG)) content = content[INKLING_TEXT_TAG.len..];
            return .{
                .reasoning_content = if (reasoning.len > 0) reasoning else null,
                .content = inklingTrimContentTail(content),
            };
        }
        // Length-truncated mid-thought: reasoning, never content.
        const reasoning = std.mem.trim(u8, body, "\n ");
        return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
    }
    if (std.mem.startsWith(u8, t, INKLING_TEXT_TAG)) {
        return .{ .reasoning_content = null, .content = inklingTrimContentTail(t[INKLING_TEXT_TAG.len..]) };
    }
    return null;
}

// ── Muse-Glimmer (muse_glimmer) channel format ──────────────────────────
//
// The model emits harmony-style SEGMENTS after the prompt's bare
// `<|start|>assistant`:
//   ` to=self<|message|>REASONING<|eom|>` then
//   `<|start|>assistant to=user<|message|>ANSWER<|eot|>` — or a tool segment
//   `<|start|>assistant to=<fn><|message|><atem:function_calls>…<|eot|>`.
// <|start|>/<|message|>/<|eom|>/<|eot|> are single special tokens (ids
// 200022/200023/200007/200008); the `assistant to=<recipient>` header BETWEEN
// them is ORDINARY text, which is what makes an unresolved header something
// every streaming path must hold rather than flush.
const MUSE_START_TAG = "<|start|>";
const MUSE_MSG_TAG = "<|message|>";
const MUSE_EOM_TAG = "<|eom|>";
const MUSE_EOT_TAG = "<|eot|>";

fn museIsRecipientChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
        (c >= '0' and c <= '9') or c == '_' or c == '-' or c == '.';
}

const MuseSegment = struct { recipient: []const u8, body: []const u8 };

/// Recipient + body of the segment in text[seg_start..seg_end] (seg_start
/// sits on a `<|start|>` marker, or 0 for the headerless first segment).
/// null when the header never resolved — a growing or truncated header is
/// never content.
fn museSegmentAt(text: []const u8, seg_start: usize, seg_end: usize) ?MuseSegment {
    const seg = text[seg_start..seg_end];
    const msg_rel = std.mem.indexOf(u8, seg, MUSE_MSG_TAG) orelse return null;
    const header = seg[0..msg_rel];
    const recipient: []const u8 = if (std.mem.indexOf(u8, header, "to=")) |t|
        std.mem.trim(u8, header[t + 3 ..], " \n")
    else
        "";
    var body = seg[msg_rel + MUSE_MSG_TAG.len ..];
    if (std.mem.indexOf(u8, body, MUSE_EOM_TAG)) |e| body = body[0..e];
    if (std.mem.indexOf(u8, body, MUSE_EOT_TAG)) |e| body = body[0..e];
    return .{ .recipient = recipient, .body = body };
}

/// Muse channel split. Claims the text only when `<|message|>` is present —
/// no other family emits it. First segment of each channel wins (the template
/// renders one per turn). A tool segment's body (ATEM markup) rides out as
/// content ONLY when no user segment exists, so the KeepingMarkup callers can
/// still see it and the trim wrapper cuts it from display.
fn splitMuseChannels(text: []const u8) ?ThinkSplit {
    if (std.mem.indexOf(u8, text, MUSE_MSG_TAG) == null) return null;
    var reasoning: ?[]const u8 = null;
    var content: ?[]const u8 = null;
    var tool_body: ?[]const u8 = null;
    var routed = false;

    var seg_start: usize = 0;
    while (true) {
        const next = std.mem.indexOfPos(u8, text, seg_start + 1, MUSE_START_TAG);
        const seg_end = next orelse text.len;
        if (museSegmentAt(text, seg_start, seg_end)) |s| {
            routed = true;
            if (std.mem.eql(u8, s.recipient, "self")) {
                if (reasoning == null) {
                    const r = std.mem.trim(u8, s.body, "\n ");
                    if (r.len > 0) reasoning = r;
                }
            } else if (s.recipient.len == 0 or std.mem.eql(u8, s.recipient, "user")) {
                if (content == null) content = std.mem.trim(u8, s.body, "\n ");
            } else {
                if (tool_body == null) tool_body = s.body;
            }
        } else if (seg_start == 0) {
            // Headerless first segment: a prompt-committed `to=user<|message|>`
            // header puts the model's first bytes straight into BODY. Claim it
            // as content — but never text that still reads as a (truncated)
            // recipient header.
            var body = text[0..seg_end];
            if (std.mem.indexOf(u8, body, MUSE_EOM_TAG)) |e| body = body[0..e];
            if (std.mem.indexOf(u8, body, MUSE_EOT_TAG)) |e| body = body[0..e];
            const c = std.mem.trim(u8, body, "\n ");
            if (c.len > 0 and museThinkOpenerAt(body) == .not_muse) {
                routed = true;
                content = c;
            }
        }
        seg_start = next orelse break;
    }
    if (!routed) return null;
    return .{
        .reasoning_content = reasoning,
        .content = content orelse (tool_body orelse ""),
    };
}

/// Muse-Glimmer streaming verdict, decided from the channel markers alone.
/// null = not muse traffic (fall through to the generic think gate). The
/// recipient header between <|start|> and <|message|> is ORDINARY text, so an
/// unresolved header always HOLDS — flushing it leaks ` to=user` fragments.
fn museStreamVerdict(buf: []const u8) ?StreamThinkGate {
    const has_marker = std.mem.indexOf(u8, buf, MUSE_MSG_TAG) != null or
        std.mem.indexOf(u8, buf, MUSE_START_TAG) != null or
        std.mem.indexOf(u8, buf, MUSE_EOM_TAG) != null;
    if (!has_marker) {
        // A growing bare header (` to=sel`) carries no marker yet: claim it
        // only while every byte fits the recipient grammar, so prose that
        // merely starts with "to=" is released to the generic gate. The
        // first tokens can be SHORTER than "to=" (" to" leaked live
        // 2026-08-10) — a strict prefix of it still holds.
        const t = std.mem.trimStart(u8, buf, " \n");
        if (t.len < "to=".len) {
            return if (t.len > 0 and std.mem.startsWith(u8, "to=", t)) .hold_thinking else null;
        }
        if (!std.mem.startsWith(u8, t, "to=")) return null;
        for (t["to=".len..]) |c| {
            if (!museIsRecipientChar(c)) return null;
        }
        return .hold_thinking;
    }
    const seg_start = std.mem.lastIndexOf(u8, buf, MUSE_START_TAG) orelse 0;
    const seg = buf[seg_start..];
    const msg_rel = std.mem.indexOf(u8, seg, MUSE_MSG_TAG) orelse return .hold_thinking;
    const header = seg[0..msg_rel];
    const recipient: []const u8 = if (std.mem.indexOf(u8, header, "to=")) |t|
        std.mem.trim(u8, header[t + 3 ..], " \n")
    else
        "";
    if (std.mem.eql(u8, recipient, "self")) {
        const closed = std.mem.indexOf(u8, seg, MUSE_EOM_TAG) != null or
            std.mem.indexOf(u8, seg, MUSE_EOT_TAG) != null;
        return if (closed) .split_think else .hold_thinking;
    }
    if (recipient.len == 0 or std.mem.eql(u8, recipient, "user")) return .split_think;
    // Tool recipient: the tools path buffers before this gate is consulted;
    // hold defensively on a no-tools request rather than leak ATEM markup.
    return .hold_thinking;
}

/// Muse streaming tool hold: true while a segment header is unresolved (the
/// next token decides whether it names a tool) or resolved to a TOOL
/// recipient (the body is ATEM payload — parse runs on the end flush).
fn museHeaderHoldsForTools(buf: []const u8) bool {
    var header_from: usize = 0;
    if (std.mem.lastIndexOf(u8, buf, MUSE_START_TAG)) |p| {
        header_from = p + MUSE_START_TAG.len;
    } else {
        const lead = std.mem.trimStart(u8, buf, " \n");
        // A strict prefix of "to=" could still become a header (" to" leaked
        // as a text delta live 2026-08-10) — hold it.
        if (lead.len < "to=".len) return lead.len > 0 and std.mem.startsWith(u8, "to=", lead);
        if (!std.mem.startsWith(u8, lead, "to=")) return false;
        header_from = buf.len - lead.len;
    }
    const seg = buf[header_from..];
    if (std.mem.indexOf(u8, seg, MUSE_MSG_TAG)) |msg_rel| {
        const header = seg[0..msg_rel];
        const recipient: []const u8 = if (std.mem.indexOf(u8, header, "to=")) |t|
            std.mem.trim(u8, header[t + 3 ..], " \n")
        else
            "";
        if (recipient.len == 0) return false;
        if (std.mem.eql(u8, recipient, "self") or std.mem.eql(u8, recipient, "user")) return false;
        return true;
    }
    // Unresolved: hold only while the bytes still fit `assistant to=<r>`.
    var rest = std.mem.trimStart(u8, seg, " \n");
    if (std.mem.startsWith(u8, rest, "assistant")) rest = rest["assistant".len..];
    rest = std.mem.trimStart(u8, rest, " \n");
    if (rest.len == 0) return true;
    if (rest.len < "to=".len) return std.mem.startsWith(u8, "to=", rest);
    if (!std.mem.startsWith(u8, rest, "to=")) return false;
    for (rest["to=".len..]) |c| {
        if (!museIsRecipientChar(c)) return false;
    }
    return true;
}

/// Muse-Glimmer opener classification for the no-tools streaming think arm.
/// The model's first bytes after the prompt's `<|start|>assistant` are a
/// recipient header in ORDINARY text: ` to=self<|message|>` opens reasoning
/// (closes at <|eom|>), any other resolved header means the model went
/// straight to content. Payload sizes are the byte offset just past the
/// header's `<|message|>`.
pub const MuseThinkOpener = union(enum) { not_muse, growing, self_opened: usize, direct: usize };

pub fn museThinkOpenerAt(buf: []const u8) MuseThinkOpener {
    var rest = std.mem.trimStart(u8, buf, " \n");
    var saw_ctx = false;
    if (std.mem.startsWith(u8, rest, MUSE_START_TAG)) {
        rest = rest[MUSE_START_TAG.len..];
        saw_ctx = true;
    }
    if (std.mem.startsWith(u8, rest, "assistant")) {
        rest = rest["assistant".len..];
        rest = std.mem.trimStart(u8, rest, " \n");
        saw_ctx = true;
    } else if (saw_ctx) {
        return if (std.mem.startsWith(u8, "assistant", rest)) .growing else .not_muse;
    }
    if (std.mem.startsWith(u8, rest, MUSE_MSG_TAG)) {
        return .{ .direct = buf.len - rest.len + MUSE_MSG_TAG.len };
    }
    if (!std.mem.startsWith(u8, rest, "to=")) {
        if (rest.len < "to=".len and std.mem.startsWith(u8, "to=", rest)) return .growing;
        return .not_muse;
    }
    const after_to = rest["to=".len..];
    if (std.mem.indexOf(u8, after_to, MUSE_MSG_TAG)) |msg_rel| {
        const recipient = std.mem.trim(u8, after_to[0..msg_rel], " \n");
        const consumed = (buf.len - after_to.len) + msg_rel + MUSE_MSG_TAG.len;
        if (std.mem.eql(u8, recipient, "self")) return .{ .self_opened = consumed };
        return .{ .direct = consumed };
    }
    for (after_to) |c| {
        if (!museIsRecipientChar(c)) return .not_muse;
    }
    return .growing;
}

/// Length of a COMPLETE next-segment header at the start of `text`
/// (`<|start|>assistant to=<r><|message|>`), or null when text doesn't start
/// with one or the header hasn't fully arrived yet.
pub fn museContentHeaderSkip(text: []const u8) ?usize {
    if (!std.mem.startsWith(u8, text, MUSE_START_TAG)) return null;
    const seg = text[MUSE_START_TAG.len..];
    const msg_rel = std.mem.indexOf(u8, seg, MUSE_MSG_TAG) orelse return null;
    return MUSE_START_TAG.len + msg_rel + MUSE_MSG_TAG.len;
}

/// Plain-arm segment-header skip for muse streams: the `<|start|>` marker
/// token arms it, `<|message|>` disarms — everything between is the
/// role+recipient header, never content. Both markers are single special
/// tokens, so exact-match at the delta level is complete.
pub fn museHeaderSkipNext(skipping: bool, token_text: []const u8) bool {
    if (std.mem.eql(u8, token_text, MUSE_START_TAG)) return true;
    if (std.mem.eql(u8, token_text, MUSE_MSG_TAG)) return false;
    return skipping;
}

/// template opened (Qwen 3.5/3.6 style: `…assistant\n<think>\n`). Callers
/// decode the last few prompt tokens and pass the tail here. A thinking-off
/// render ends with a CLOSED `</think>` block and must not match.
pub fn promptTailOpensThink(tail: []const u8) bool {
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");
    return endsWithThinkOpenTag(trimmed) != null;
}

/// Suffix that COMMITS the no-think channel in the rendered prompt, or null.
/// Thinking-off on an always-thinking template otherwise means generating the
/// whole reasoning pass and discarding it — the prompt-side commit is what
/// makes thinking-off a real skip.
///   • Muse: the generation prompt ends bare `<|start|>assistant`; spending
///     the header (` to=user<|message|>`, byte-exactly what the template
///     renders for history content turns) makes `to=self` unreachable. Never
///     with tools — a tool call is a `to=<fn>` header.
///   • Unconditional think opener (LFM2.5 class): close the block the
///     template opened. Thinking-off Qwen-style renders end CLOSED and never
///     match; no tools gate — tool calls ride outside the think block.
fn noThinkTailSuffix(tpl: []const u8, rendered: []const u8, enable_thinking: bool, has_tools: bool) ?[]const u8 {
    if (enable_thinking) return null;
    const tail = contentChannelTail(tpl, rendered, !has_tools);
    return if (tail.len > 0) tail else null;
}

/// What must be appended to a rendered prompt so generation resumes in the
/// assistant's VISIBLE CONTENT channel rather than inside a reasoning block.
///
/// Two callers, one rule, deliberately: thinking-off enforces the content
/// channel so no reasoning pass runs, and a CONTINUATION needs the same thing
/// for a different reason — the partial text is prose the user has already
/// read, so a prompt that lands inside a think block would return the rest of
/// their answer as reasoning. Written twice, the muse and LFM2.5 cases would
/// drift apart the first time a third family joined.
///
/// `allow_channel_commit` is thinking-off's extra gate: with tools present it
/// declines to commit muse's `to=user`, since that would forbid the tool call
/// the model may be about to make. A continuation always passes true — it is
/// resuming prose by construction.
fn contentChannelTail(tpl: []const u8, rendered: []const u8, allow_channel_commit: bool) []const u8 {
    if (allow_channel_commit and std.mem.indexOf(u8, tpl, "reasoning_strength") != null and
        std.mem.endsWith(u8, rendered, "<|start|>assistant"))
    {
        return " to=user<|message|>";
    }
    if (promptTailOpensThink(rendered[rendered.len -| 64 ..])) return "</think>";
    return "";
}

/// Whether this conversation SHAPE is a continuation: it ends with an
/// assistant message carrying prose. The one predicate every surface asks, so
/// "what counts as continuable" cannot differ between the endpoint that
/// implies it and the endpoint that takes a flag — and so the answer matches
/// what `continuationPrefill` will actually do with it.
pub fn continuationRequested(messages: []const Message) bool {
    return continuationPrefill(messages, true) != null;
}

/// The partial assistant reply a continuation request is asking to extend, or
/// null when this is an ordinary turn.
///
/// The flag describes the SHAPE of the request, so a client that sets it on a
/// conversation not ending in an assistant message gets an ordinary turn —
/// never the user's own words committed into the assistant's mouth.
///
/// The prefill is trimmed of trailing whitespace: a tokenizer merges a trailing
/// space into the word that follows, so a prompt ending in one tokenizes
/// differently than the same text does mid-reply. (Anthropic's API rejects such
/// a prefill outright; trimming keeps the request serviceable, and only the
/// PROMPT is trimmed — the stored reply keeps its whitespace.)
fn continuationPrefill(messages: []const Message, continue_final: bool) ?[]const u8 {
    if (!continue_final or messages.len == 0) return null;
    const last = messages[messages.len - 1];
    if (!std.mem.eql(u8, last.role, "assistant")) return null;
    // A reply that ended in a tool call has no prose to resume — splicing a
    // continuation onto it would re-run the parse chain over half a call.
    if (last.tool_calls != null) return null;
    const trimmed = std.mem.trimEnd(u8, last.content, " \t\r\n");
    return if (trimmed.len > 0) trimmed else null;
}

/// Split model output into reasoning_content and content.
/// Handles both `<think>...</think>` and Gemma 4's `<|channel>thought\n...<channel|>`.
/// `opened_by_template`: the generation prompt ended with a template-injected
/// think opener (see `promptTailOpensThink`), so output with no literal opener
/// and no close tag is still reasoning — generation started inside the block.
pub fn splitThinkBlock(text: []const u8, thinking: bool, opened_by_template: bool) ThinkSplit {
    const split = splitThinkBlockKeepingMarkup(text, thinking, opened_by_template);
    const reasoning: ?[]const u8 = if (split.reasoning_content) |r| blk: {
        const t = trimLeakedToolMarkup(r);
        break :blk if (t.len > 0) t else null;
    } else null;
    return .{ .reasoning_content = reasoning, .content = trimLeakedToolMarkup(split.content) };
}

/// The split WITHOUT the leaked-markup cut — for the one caller that feeds the
/// content back to `parseToolCalls` (the /v1/messages non-streaming path).
/// Every arm below returns raw slices; the cut is applied ONCE in the wrapper
/// above so a new arm cannot forget it.
pub fn splitThinkBlockKeepingMarkup(text: []const u8, thinking: bool, opened_by_template: bool) ThinkSplit {
    // Muse-Glimmer channel segments (`to=self` / `to=user` / tool). Keyed on
    // the `<|message|>` marker no other family emits; runs regardless of the
    // `thinking` flag because the headers need stripping either way.
    if (splitMuseChannels(text)) |split| return split;
    // Inkling (inkling_mm_model) message channels: the model emits role-less
    // MESSAGES — `<|content_thinking|>R<|end_message|>` then
    // `<|message_model|><|content_text|>C<|end_message|>` — not a tag pair.
    // Keyed on channel openers no other family emits; runs regardless of the
    // `thinking` flag because the content marker needs stripping either way.
    if (splitInklingChannels(text)) |split| return split;
    // Gemma 4 style: <|channel>thought\n...<channel|>\n<|channel>\ncontent
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        const think_tag = "<|channel>thought\n";
        const reasoning_start: usize = if (std.mem.startsWith(u8, text, think_tag)) think_tag.len else if (std.mem.startsWith(u8, text, "<|channel>thought")) "<|channel>thought".len else 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..end], "\n ");
        var content = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
        // Strip the content channel tag: <|channel>\n or <|channel>. A
        // re-opened THOUGHT channel (`<|channel>thought…`) is NOT a content
        // opener — stripping its `<|channel>` prefix here left a bare "thought"
        // in visible content (2026-06-19 live); leave it for stripTrailingThinkOpen.
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>") and !std.mem.startsWith(u8, content, "<|channel>thought")) {
            content = content[10..];
        }
        content = std.mem.trimStart(u8, content, "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = trimTrailingThinkClosers(stripTrailingThinkOpen(content)),
        };
    }
    // Standard style: <think>...</think> (or the Hy3-suffixed variant)
    if (indexOfThinkCloseTag(text, 0)) |close| {
        const reasoning_start: usize = thinkOpenTagLenAt(text) orelse 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..close.pos], "\n ");
        const content = std.mem.trimStart(u8, text[close.pos + close.len ..], "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = trimTrailingThinkClosers(stripTrailingThinkOpen(content)),
        };
    }
    if (thinking) {
        // Unclosed think block: split policy depends on whether the model's
        // output begins with a literal opener.
        //   • Literal opener present → model definitely entered thinking but
        //     ran out of tokens / didn't close. Treat as reasoning.
        //   • No literal opener + template-injected opener → generation began
        //     INSIDE the block (Qwen 3.5/3.6 render `…assistant\n<think>\n`);
        //     an unclosed tail is truncated reasoning, never content.
        //   • No literal opener + no template opener → the model answered
        //     directly (Gemma style); keep the answer visible as content.
        if (thinkOpenTagLenAt(text) != null or std.mem.startsWith(u8, text, "<|channel>thought")) {
            const start: usize = if (thinkOpenTagLenAt(text)) |l| l else if (std.mem.startsWith(u8, text, "<|channel>thought\n")) "<|channel>thought\n".len else "<|channel>thought".len;
            const reasoning = std.mem.trimStart(u8, text[start..], "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        if (opened_by_template) {
            const reasoning = std.mem.trimStart(u8, text, "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        // Prose answer that ENDS by opening a new, unclosed thought block
        // (Gemma 12B tail behavior): the text before the opener is the
        // answer; the dangling thought is reasoning. The raw opener tag must
        // never leak into visible content.
        if (lastUnclosedThinkOpen(text)) |o| {
            const content = std.mem.trimEnd(u8, std.mem.trimStart(u8, text[0..o.pos], "\n "), "\n ");
            const reasoning = std.mem.trim(u8, text[o.after..], "\n ");
            return .{
                .reasoning_content = if (reasoning.len > 0) reasoning else null,
                .content = content,
            };
        }
        // No thought block, but the model may have emitted (or been truncated
        // right after) a dangling Gemma 4 *content* channel opener `<|channel>`
        // / `<|channel>\n`. Strip it so a cut-off reply never leaks the raw
        // control tag into visible content; whatever follows is the answer.
        var content = std.mem.trimStart(u8, text, "\n ");
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>")) {
            content = content[10..];
        }
        return .{ .reasoning_content = null, .content = std.mem.trimStart(u8, content, "\n ") };
    }
    return .{ .reasoning_content = null, .content = text };
}

const ThinkOpen = struct { pos: usize, after: usize, is_think_style: bool };

/// Earliest think/thought opener at or after `from` (either tag family;
/// the think style covers both `<think>` and the Hy3-suffixed form).
fn nextThinkOpen(text: []const u8, from: usize) ?ThinkOpen {
    const chan = std.mem.indexOfPos(u8, text, from, "<|channel>thought");
    const think = indexOfThinkOpenTag(text, from);
    if (chan == null and think == null) return null;
    if (think == null or (chan != null and chan.? < think.?.pos)) {
        return .{ .pos = chan.?, .after = chan.? + "<|channel>thought".len, .is_think_style = false };
    }
    return .{ .pos = think.?.pos, .after = think.?.pos + think.?.len, .is_think_style = true };
}

/// Earliest close tag at or after `from` (either tag family).
fn nextThinkClose(text: []const u8, from: usize) ?ThinkOpen {
    const chan = std.mem.indexOfPos(u8, text, from, "<channel|>");
    const think = indexOfThinkCloseTag(text, from);
    if (chan == null and think == null) return null;
    if (think == null or (chan != null and chan.? < think.?.pos)) {
        return .{ .pos = chan.?, .after = chan.? + "<channel|>".len, .is_think_style = false };
    }
    return .{ .pos = think.?.pos, .after = think.?.pos + think.?.len, .is_think_style = true };
}

/// Skip an optional Gemma 4 CONTENT channel opener (`<|channel>` not followed
/// by `thought`) plus surrounding newlines, right after a thought close.
fn skipContentChannelTag(text: []const u8, start: usize) usize {
    var pos = start;
    while (pos < text.len and (text[pos] == '\n' or text[pos] == ' ')) pos += 1;
    const tag = "<|channel>";
    if (pos + tag.len <= text.len and std.mem.eql(u8, text[pos .. pos + tag.len], tag)) {
        const rest = text[pos + tag.len ..];
        if (!std.mem.startsWith(u8, rest, "thought")) {
            pos += tag.len;
            while (pos < text.len and (text[pos] == '\n' or text[pos] == ' ')) pos += 1;
        }
    }
    return pos;
}

/// Merge ALL closed think/thought blocks in `text` into one leading block.
///
/// The split/strip layer understands exactly one leading block plus an
/// optional UNCLOSED trailing opener. Gemma 4 12B, however, re-opens a thought
/// channel mid-turn and closes it again (`…content<|channel>thought\n…
/// <channel|>more content`) — observed live via Claude Code on /v1/messages,
/// where the raw pair leaked verbatim into the visible text block. This pass
/// rewrites such output to `<|channel>thought\n{all thought text}<channel|>
/// {all content}` so every downstream consumer (splitThinkBlock,
/// parseToolCalls) handles it unchanged.
///
/// Returns null when no rewrite is needed (zero or one LEADING closed block —
/// the overwhelmingly common case, zero-cost). An unclosed trailing opener is
/// left in place for the existing trailing-strip logic.
pub fn normalizeEmbeddedThinkBlocks(allocator: std.mem.Allocator, text: []const u8) !?[]u8 {
    var reasoning_parts = std.ArrayList([]const u8).empty;
    defer reasoning_parts.deinit(allocator);
    var content_parts = std.ArrayList([]const u8).empty;
    defer content_parts.deinit(allocator);

    var style_think = false;
    var style_set = false;
    var closed_blocks: usize = 0;
    var first_block_leading = false;
    var pos: usize = 0;

    // Template-opened leading block: a close tag BEFORE any opener (Qwen
    // renders `…assistant\n<think>\n` into the prompt, so output starts
    // mid-thought and contains only the close).
    if (nextThinkClose(text, 0)) |lc| {
        const open_pos = if (nextThinkOpen(text, 0)) |o| o.pos else text.len;
        if (lc.pos < open_pos) {
            try reasoning_parts.append(allocator, std.mem.trim(u8, text[0..lc.pos], "\n "));
            style_think = lc.is_think_style;
            style_set = true;
            closed_blocks += 1;
            first_block_leading = true;
            pos = skipContentChannelTag(text, lc.after);
        }
    }

    while (true) {
        const o = nextThinkOpen(text, pos) orelse {
            try content_parts.append(allocator, text[pos..]);
            break;
        };
        const close_tag: []const u8 = if (o.is_think_style) "</think>" else "<channel|>";
        const close_pos = std.mem.indexOfPos(u8, text, o.after, close_tag) orelse {
            // Unclosed trailing opener — leave verbatim for the trailing-strip
            // logic in splitThinkBlock.
            try content_parts.append(allocator, text[pos..]);
            break;
        };
        try content_parts.append(allocator, text[pos..o.pos]);
        try reasoning_parts.append(allocator, std.mem.trim(u8, text[o.after..close_pos], "\n "));
        if (!style_set) {
            style_think = o.is_think_style;
            style_set = true;
        }
        closed_blocks += 1;
        if (o.pos == 0) first_block_leading = true;
        pos = skipContentChannelTag(text, close_pos + close_tag.len);
    }

    // Rewrite only when a closed block exists beyond the single leading one.
    if (closed_blocks == 0) return null;
    if (closed_blocks == 1 and first_block_leading) return null;

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, if (style_think) "<think>\n" else "<|channel>thought\n");
    var first = true;
    for (reasoning_parts.items) |r| {
        if (r.len == 0) continue;
        if (!first) try out.appendSlice(allocator, "\n\n");
        try out.appendSlice(allocator, r);
        first = false;
    }
    try out.appendSlice(allocator, if (style_think) "</think>\n" else "<channel|>\n");
    first = true;
    for (content_parts.items) |c| {
        const trimmed = std.mem.trim(u8, c, "\n ");
        if (trimmed.len == 0) continue;
        if (!first) try out.append(allocator, '\n');
        try out.appendSlice(allocator, trimmed);
        first = false;
    }
    return try out.toOwnedSlice(allocator);
}

/// Streaming-only: true when the buffer TAIL is a partial prefix of a think
/// opener (`<think>` / `<|channel>thought`). The buffered-stream flush must
/// hold these bytes back until the tag completes or diverges — flushing them
/// leaks tag fragments as visible content (a pi session showed prose ending
/// in a glued "thought" because `<|channel>` flushed before "thought"
/// arrived and completed the opener).
pub fn endsWithPartialThinkOpen(buf: []const u8) bool {
    const tags = [_][]const u8{ "<|channel>thought", "<think>" };
    for (tags) |tag| {
        // Strictly-partial prefixes only — a COMPLETE opener in the buffer is
        // the caller's contains-check's job; ours is the growing tail.
        var l = @min(buf.len, tag.len - 1);
        while (l > 0) : (l -= 1) {
            if (std.mem.endsWith(u8, buf, tag[0..l])) return true;
        }
    }
    // Hy3-suffixed opener mid-stream (`…<think:opensou`): everything from the
    // last `<` must be a valid growing prefix of `<think:suffix>`.
    if (std.mem.lastIndexOfScalar(u8, buf, '<')) |lt| {
        const tail = buf[lt..];
        if (tail.len > "<think".len and isPartialSuffixedTag(tail, "<think")) return true;
    }
    return false;
}

/// Streaming-only: should the chat-completion SSE path defer flushing
/// because `buf` is on track to become a tool call?
///
/// True when:
///   * `buf` contains `<tool` followed by a valid tag terminator
///     (`>`, ` `, `\t`, `\n`, `_`, `|`) somewhere — that's any of the
///     accepted families: `<tool>`, `<tool …>`, `<tool_call…>`, `<tool_calls…>`,
///     `<tool_request…>`, `<tool_requests…>` (mirrors `parseToolCalls`).
///   * `buf` contains the Gemma 4 `<|tool_call` substring.
///   * `buf[0] == '{'` and `buf` contains `"name"` (raw JSON tool-call shape).
///   * `buf` ends with a partial prefix that could grow into one of the above
///     in the next token (`<`, `<t`, `<to`, `<too`, `<tool`, or any prefix
///     of `<|tool_call`).
///
/// Conservatively false on `<toolkit>`, `<toolbar>`, etc. — anything where
/// the char after `<tool` isn't a valid tag terminator. That keeps prose
/// that happens to mention HTML-ish tags flowing through normally.
pub fn streamShouldBufferForTools(buf: []const u8) bool {
    if (buf.len == 0) return false;

    // Raw JSON tool-call shape: starts with `{` and has `"name"` somewhere.
    if (buf[0] == '{' and std.mem.indexOf(u8, buf, "\"name\"") != null) return true;

    // Gemma 4 fully-formed canonical open.
    if (std.mem.indexOf(u8, buf, "<|tool_call") != null) return true;

    // Inkling: the invoke marker is a SINGLE special token (arrives whole) —
    // once present, everything from here on is call payload. Without this the
    // NAME + full JSON streamed as visible content deltas (live 2026-07-30,
    // first real pi session; the leak landed in pi's transcript and
    // contaminated every later turn's history).
    if (std.mem.indexOf(u8, buf, INKLING_INVOKE_TAG) != null) return true;

    // DeepSeek-V4 DSML: everything from `<｜DSML｜` on is a tool-call block.
    if (std.mem.indexOf(u8, buf, "<｜DSML｜") != null) return true;

    // Muse-Glimmer ATEM: everything from the wrapper on is call payload.
    if (std.mem.indexOf(u8, buf, "<atem:") != null) return true;
    // Muse segment-header hold: between <|start|> and <|message|> only the
    // role+recipient text flows, and a non-self/user recipient names a TOOL.
    if (museHeaderHoldsForTools(buf)) return true;

    // Inkling NAME hold: with a message-boundary marker present (a family
    // signal no other template emits), a segment that is still a bare
    // identifier run may be the NAME directly before an invoke marker. Prose
    // disambiguates within a token or two (space/punctuation); a call
    // disambiguates at the invoke marker. An EMPTY segment never holds, so
    // thinking splits stay prompt and marker-only flushes keep flowing.
    if (inklingSegmentCouldBeToolName(buf)) return true;

    // `<tool…` family: walk every `<tool` occurrence, accept the first one
    // whose terminator is valid (mirrors parseToolCalls' acceptance rule).
    var scan: usize = 0;
    while (std.mem.indexOf(u8, buf[scan..], "<tool")) |rel| {
        const after = scan + rel + "<tool".len;
        if (after >= buf.len) return true; // truncated mid-prefix → in progress
        const c = buf[after];
        if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '_' or c == '|') return true;
        scan = after;
    }

    // Trailing partial prefixes — the next streamed token could complete any
    // of these into a real tool open. Order doesn't matter; first endsWith
    // hit wins. Listed shortest-first for legibility.
    const tail_prefixes = [_][]const u8{
        "<",     "<t",     "<to",     "<too",     "<tool",
        "<|",    "<|t",    "<|to",    "<|too",    "<|tool",
        "<|tool_", "<|tool_c", "<|tool_ca", "<|tool_cal",
        // Muse ATEM (a fused multi-char BPE fragment can end mid-marker)
        "<a", "<at", "<ate", "<atem", "<atem:",
        // DSML fullwidth-bar prefixes (`｜` = 3 bytes; cover mid-codepoint
        // splits too in case the tokenizer spells the marker in pieces)
        "<\xef",  "<\xef\xbd", "<｜",  "<｜D", "<｜DS", "<｜DSM",
        "<｜DSML", "<｜DSML\xef", "<｜DSML\xef\xbd",
    };
    for (tail_prefixes) |p| {
        if (std.mem.endsWith(u8, buf, p)) return true;
    }
    return false;
}

/// Inkling streaming NAME hold: true when the text after the last message
/// boundary (<|end_message|> / <|message_model|> / <|content_text|>) is a
/// NON-EMPTY bare identifier run — the shape of a tool NAME right before its
/// invoke marker. A thinking message never matches (no boundary marker →
/// whole buffer, which carries `<|`); other families never emit these
/// markers at all.
fn inklingSegmentCouldBeToolName(buf: []const u8) bool {
    var seg_start: usize = 0;
    var saw_marker = false;
    inline for (.{ INKLING_END_TAG, INKLING_MODEL_TAG, INKLING_TEXT_TAG }) |tag| {
        if (std.mem.lastIndexOf(u8, buf, tag)) |p| {
            saw_marker = true;
            seg_start = @max(seg_start, p + tag.len);
        }
    }
    if (!saw_marker) return false;
    const seg = buf[seg_start..];
    if (seg.len == 0 or seg.len > 64) return false;
    for (seg) |c| {
        if (!inklingIsNameChar(c)) return false;
    }
    return true;
}

/// Streaming-only: what should a tools-enabled SSE path do with the buffered
/// text so far, with respect to thinking? Shared by the chat-completions and
/// Anthropic /v1/messages stream handlers — the two paths drifted apart once
/// already: /v1/messages only recognized think OPENERS present in the output,
/// so Qwen-family template-opened thinking (opener injected into the PROMPT)
/// streamed as visible text and a raw `</think>` leaked into Claude Code
/// transcripts (2026-06-10 live).
///
///   .hold_thinking — inside an unclosed think block; buffer, emit nothing
///   .split_think   — close tag arrived; splitThinkBlock once, emit
///                    reasoning + visible remainder, clear the buffer, and
///                    set think_closed for the rest of the turn
///   .flush_text    — plain visible prose; stream it
///
/// `think_closed` releases the enable_thinking hold after the one split —
/// without it the visible answer sits in the buffer until end-of-stream and
/// gets misfiled as reasoning (the pi hidden-answer bug).
pub const StreamThinkGate = enum { flush_text, hold_thinking, split_think };

/// The gate WITHOUT a prompt-opened think block — every family whose template
/// gates its `<think>` on `enable_thinking` renders the closed `<think></think>`
/// signature (or no tag at all) when thinking is off, so the flag is false for
/// them and this stays their exact behavior.
pub fn streamThinkGate(buf: []const u8, enable_thinking: bool, think_closed: bool) StreamThinkGate {
    return streamThinkGate2(buf, enable_thinking, think_closed, false);
}

/// `prompt_opened_think`: the rendered generation prompt ended with an OPEN
/// think tag (`server.promptOpensThink`). That is a fact about the prompt
/// BYTES, not about our request flag — LFM2.5's template pre-opens `<think>`
/// unconditionally, so with thinking off the model's reasoning arrives as
/// tag-free prose and the 3-arg gate flushed the whole chain-of-thought as the
/// visible answer (live 2026-08-04). It can only ever ADD a hold: when
/// thinking is on `enable_thinking` already held, and once `think_closed` the
/// term is dropped so the answer after the block still flushes.
/// Memoized marker scan for the streaming think gate, owned BESIDE the stream
/// buffer and reset with it.
///
/// Without it `streamThinkGate2` re-scans the whole accumulated buffer on every
/// token for as long as thinking markup is present and no close has arrived —
/// O(buffer) per token, O(n²) over a reasoning block. Measured on a hermetic
/// 4000-token stream growing to 113 KB: 47.99 µs/token with no close in the
/// buffer (every scan runs to the end), 16.98 µs/token with a close near the
/// front (the close scan stops early). ~0.3% of a 15 ms/token model, ~1% at
/// 5 ms/token, and it GROWS with the thought.
///
/// Exactness rests on one property of the marker grammar: `tagSuffixChar`
/// excludes `<`, so every recognized marker contains exactly ONE `<`, at its
/// start. A marker straddling the scanned/unscanned boundary must therefore
/// begin at the LAST `<` in the buffer — and if that `<` cannot still grow into
/// a marker, nothing straddles the boundary at all. So the resume point is
/// exact and bounded, with no overlap constant to get wrong.
pub const ThinkScan = struct {
    /// Bytes already examined. Never trusted when the buffer is SHORTER than
    /// this (the loop reset the buffer without resetting the scan).
    checked_upto: usize = 0,
    /// Index of the last `<` seen, or null. The only position a marker can
    /// straddle the boundary from.
    last_lt: ?usize = null,
    saw_channel_thought: bool = false,
    saw_open: bool = false,
    saw_channel_close: bool = false,
    /// Tool markup anywhere in the buffer. This is what makes a found close
    /// UNSTABLE and therefore un-latchable: `thinkCloseIsToolCallPayload`
    /// reclassifies a close as an argument value once a `</tool_call` arrives
    /// AFTER it, so a close that was real at token N is payload at token N+k.
    /// With no tool markup the close is stable and latching is exact.
    saw_tool_marker: bool = false,
    /// First real close tag, once found and stable.
    close: ?TagAt = null,
    /// Bytes examined by the last call, INCLUDING the exact full re-scan the
    /// tool-markup case falls back to. Test-visible: the invariant is that this
    /// stays FLAT as the buffer grows for traffic with no tool markup — which
    /// is the arm the 47.99 µs/token bench measured.
    last_scan_span: usize = 0,

    /// Longest plain substring needle minus one — the overlap a windowed
    /// substring search needs so a needle split across two arrivals is caught.
    /// `<|channel>thought` is the longest at 17.
    const substr_overlap: usize = 16;

    pub fn reset(self: *ThinkScan) void {
        self.* = .{};
    }

    /// Where the next scan must start so no marker spanning the boundary is
    /// missed: the last `<` if it can still grow into a tag (see the type's
    /// doc comment for why no earlier `<` can), and far enough back for the
    /// plain substring needles either way.
    fn resumeFrom(self: *const ThinkScan, buf: []const u8) usize {
        const upto = @min(self.checked_upto, buf.len);
        const substr_from = upto -| substr_overlap;
        const lt = self.last_lt orelse return substr_from;
        if (lt >= upto) return substr_from;
        const tail = buf[lt..upto];
        const growing = isPartialSuffixedTag(tail, "<think") or
            isPartialSuffixedTag(tail, "</think") or
            (tail.len < "<|channel>thought".len and std.mem.startsWith(u8, "<|channel>thought", tail)) or
            (tail.len < "<channel|>".len and std.mem.startsWith(u8, "<channel|>", tail));
        return if (growing) @min(lt, substr_from) else substr_from;
    }

    fn advance(self: *ThinkScan, buf: []const u8) void {
        if (buf.len < self.checked_upto) self.reset(); // buffer was cleared under us
        if (buf.len == self.checked_upto and self.checked_upto != 0) {
            self.last_scan_span = 0;
        } else {
            const from = self.resumeFrom(buf);
            self.last_scan_span = buf.len - from;
            const win = buf[from..];
            if (std.mem.lastIndexOfScalar(u8, win, '<')) |rel| self.last_lt = from + rel;
            if (!self.saw_channel_thought and std.mem.indexOf(u8, win, "<|channel>thought") != null)
                self.saw_channel_thought = true;
            if (!self.saw_channel_close and std.mem.indexOf(u8, win, "<channel|>") != null)
                self.saw_channel_close = true;
            if (!self.saw_tool_marker and std.mem.indexOf(u8, win, "tool_call") != null)
                self.saw_tool_marker = true;
            // The open scan takes the FULL buffer with a start offset: it
            // consults nothing before the match, but a windowed slice would
            // shift the reported position. Monotone — a complete opener stays
            // one — so latching is exact.
            if (!self.saw_open and indexOfThinkOpenTag(buf, from) != null) self.saw_open = true;
            if (self.close == null and !self.saw_tool_marker)
                self.close = indexOfThinkCloseTag(buf, from);
            self.checked_upto = buf.len;
        }
        if (self.saw_tool_marker) {
            // Exact, and only here: the payload classification looks FORWARD
            // past the close for a `</tool_call`, so it must be recomputed on
            // the whole buffer. The caller latches `think_closed` right after a
            // split, so this runs for a handful of tokens, not a whole block.
            self.close = indexOfThinkCloseTag(buf, 0);
            self.last_scan_span = buf.len;
        }
    }
};

/// `streamThinkGate2` with the per-token rescan memoized. Byte-identical
/// verdicts — pinned prefix-by-prefix against the fresh gate over a corpus that
/// includes every marker family, a close buried in a tool argument, and a
/// suffix longer than the partial-scan's tail window.
pub fn streamThinkGateScan(
    buf: []const u8,
    enable_thinking: bool,
    think_closed: bool,
    prompt_opened_think: bool,
    scan: *ThinkScan,
) StreamThinkGate {
    if (museStreamVerdict(buf)) |v| return v;
    {
        const ihead = inklingStripMessageHead(buf);
        if (std.mem.startsWith(u8, ihead, INKLING_TEXT_TAG)) return .flush_text;
        if (std.mem.startsWith(u8, ihead, INKLING_THINKING_TAG)) {
            return if (std.mem.indexOf(u8, ihead, INKLING_END_TAG) != null) .split_think else .hold_thinking;
        }
    }
    scan.advance(buf);
    const has_thinking = ((enable_thinking or prompt_opened_think) and !think_closed) or
        scan.saw_channel_thought or
        scan.saw_open or
        (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
        (std.mem.startsWith(u8, buf, "<think") and buf.len < 7) or
        endsWithPartialThinkOpen(buf);
    if (!has_thinking) return .flush_text;
    return if (scan.saw_channel_close or scan.close != null) .split_think else .hold_thinking;
}

pub fn streamThinkGate2(buf: []const u8, enable_thinking: bool, think_closed: bool, prompt_opened_think: bool) StreamThinkGate {
    // Muse channels decide early — the recipient header is ordinary text and
    // must never flush unresolved.
    if (museStreamVerdict(buf)) |v| return v;
    // Inkling channels decide early: every marker is a SINGLE special token,
    // so a leading <|content_text|> is a whole marker (the flush path's
    // marker-token skip keeps it out of visible deltas), and a thinking
    // message holds until its <|end_message|> arrives.
    {
        const ihead = inklingStripMessageHead(buf);
        if (std.mem.startsWith(u8, ihead, INKLING_TEXT_TAG)) return .flush_text;
        if (std.mem.startsWith(u8, ihead, INKLING_THINKING_TAG)) {
            return if (std.mem.indexOf(u8, ihead, INKLING_END_TAG) != null) .split_think else .hold_thinking;
        }
    }
    const has_thinking = ((enable_thinking or prompt_opened_think) and !think_closed) or
        std.mem.indexOf(u8, buf, "<|channel>thought") != null or
        indexOfThinkOpenTag(buf, 0) != null or
        (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
        (std.mem.startsWith(u8, buf, "<think") and buf.len < 7) or
        // A partial opener at the buffer TAIL (mid-text re-opened channel
        // arriving token by token) must hold the flush — flushing leaks tag
        // fragments like a glued "thought".
        endsWithPartialThinkOpen(buf);
    if (!has_thinking) return .flush_text;
    const has_close = std.mem.indexOf(u8, buf, "<channel|>") != null or
        indexOfThinkCloseTag(buf, 0) != null;
    return if (has_close) .split_think else .hold_thinking;
}

/// End-of-stream classification for a think buffer that never saw a close tag:
/// is it reasoning, or was it content all along?
///
/// The streaming loop starts `in_think_block` from `enable_thinking`, i.e. it
/// ASSUMES the model begins inside a think block whenever thinking was
/// requested. That holds only for templates that pre-inject the opener. Gemma 4
/// renders a bare `<|turn>model\n` and lets the MODEL decide, so a turn it
/// answers directly ("391") carries no think markup at all — and the whole
/// answer was flushed as `reasoning_content` with EMPTY content, while the
/// non-streaming path (`splitThinkBlock`, no opener + no close + not
/// template-opened ⇒ content) returned it correctly (live 2026-08-04).
///
/// So require POSITIVE evidence a block was open: the PROMPT opened one, or the
/// model emitted a literal opener. This is what makes the two paths agree.
pub fn streamTailIsReasoning(in_think_block: bool, prompt_opened_think: bool, saw_think_open: bool) bool {
    return in_think_block and (prompt_opened_think or saw_think_open);
}

/// The reasoning bytes not yet streamed, or null when there is nothing new.
///
/// The tools streaming path emits reasoning INCREMENTALLY (see the
/// `.hold_thinking` arm), so every later site that emits reasoning for the same
/// Streaming counterpart to the leading `trimStart(content, "\n ")` every
/// non-streaming split already applies.
///
/// The two surfaces disagreed on the BYTES of the same answer: LFM2.5's template
/// opens `<think>` unconditionally, so its first generated token is the newline
/// pair that follows the closer, and `splitThinkBlock` drops it while a stream
/// flushing tokens verbatim shipped it — `'One, two…'` non-streaming,
/// `'\n\nOne, two…'` streaming (live 2026-08-13). Any model whose first visible
/// token is whitespace is in this class, not just LFM2.5.
///
/// Leading whitespace is the ONE thing a stream can still withhold: nothing
/// visible has been sent yet, so this is a suppression, never a retraction.
///
/// It suppresses a chunk that is ENTIRELY whitespace, never a prefix of one. A
/// partial trim would ship a logprobs entry whose `token` no longer appears in
/// `content` — the collector describes whole tokens and cannot split one — so
/// the two contracts are kept consistent by never cutting inside a token. A
/// single token mixing leading whitespace with visible text is the residual
/// case, and it keeps its whitespace on both surfaces' terms: the stream ships
/// it, and the caller flips `content_started` so nothing later is touched.
///
/// `content_started` must flip on the first NON-EMPTY emission, so a suppressed
/// whitespace token leaves the next one still eligible.
pub fn streamContentLead(chunk: []const u8, content_started: bool) []const u8 {
    if (content_started) return chunk;
    if (std.mem.trimStart(u8, chunk, "\n ").len == 0) return "";
    return chunk;
}

/// turn must send only the remainder or the client sees the thought twice.
///
/// `already >= reasoning.len` returns null rather than underflowing: the split
/// can SHRINK when a tool marker appears mid-thought and `trimLeakedToolMarkup`
/// cuts the tail. An SSE delta cannot be retracted, so the honest behavior is to
/// send nothing further — never to resend from the top.
pub fn unstreamedReasoning(reasoning: []const u8, already: usize) ?[]const u8 {
    if (already >= reasoning.len) return null;
    return reasoning[already..];
}

/// Parse tool calls from model output text.
pub fn parseToolCalls(allocator: std.mem.Allocator, text: []const u8) !?[]ParsedToolCall {
    // Strip thinking blocks if present
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (indexOfThinkCloseTag(text, 0)) |close| {
        effective_text = std.mem.trimStart(u8, text[close.pos + close.len ..], "\n ");
    }

    var calls = std.ArrayList(ParsedToolCall).empty;
    errdefer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        calls.deinit(allocator);
    }

    // Look for `<tool…>…</tool…>` patterns. DSV4-Flash hallucinates the
    // spec aggressively; we accept the entire training-bias family:
    //   • `<tool_call>JSON</tool_call>`                      (canonical Hermes)
    //   • `<tool_call name="X">JSON</tool_call>`             (attribute style)
    //   • `<tool_call …>JSON</tool_request>`                 (mismatched close)
    //   • `<tool_calls>JSON</tool_calls>`                    (plural form)
    //   • `<tool_calls name="X">{args}</tool_calls>`         (plural + attr)
    //   • `<tool name="X">{args}</tool_calls>`               (bare-tool open, plural close — nested wrappers)
    //   • `<tool>{json}</tool>`                              (bare tag, body has name+args)
    //   • `<tool name="X" arguments="{...}"/>`               (SELF-CLOSING — DSV4 emits this routinely;
    //                                                          parseSelfClosingToolTag handles broken-quote
    //                                                          variants like `…"}'/>` too)
    // Approach: search for the prefix `<tool` (5 chars). Skip the Gemma 4
    // close marker `<tool_call|>`. Try the self-closing form first (cheap
    // attribute scan + balanced-JSON extraction; tolerates the model's
    // mismatched quotes). If that doesn't match, fall through to the
    // open/close-marker path: scan to the next `>`, optionally capturing
    // `name="X"` / `name='X'`, then locate the earliest of `</tool>`,
    // `</tool_call>`, `</tool_calls>`, `</tool_request>`, `</tool_requests>`.
    // When the body fails to parse, advance only past the OPENING `>` so any
    // inner `<tool…>` blocks get a chance — this is what makes the outer
    // `<tool_calls>` wrapper case work.
    //
    // Inkling (inkling_mm_model) is tried FIRST: its calls are MESSAGES —
    // `NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>` —
    // whose distinctive marker no other family emits; the generic `<tool`
    // scans never see the shape and other families never carry the marker.
    try parseInklingToolCalls(allocator, text, &calls);
    // Muse-Glimmer ATEM (`<atem:invoke name="X">`): distinctive namespace
    // prefix no other family emits; the generic `<tool` scan never sees it
    // (the bytes after `<` are `atem`, not `tool`). Runs on the FULL text —
    // tool segments arrive wrapped in channel headers the think-strip above
    // would misjudge.
    try parseAtemToolCalls(allocator, text, &calls);
    // DeepSeek-V4 native DSML (`<｜DSML｜invoke …>`): distinctive fullwidth-bar
    // marker no other family emits, and the generic `<tool` scan never sees
    // it (the byte before "tool_calls" is `｜`, not `<`).
    try parseDsmlToolCalls(allocator, effective_text, &calls);
    // LFM2.5 pythonic (`<|tool_call_start|>[fn(a='x')]<|tool_call_end|>`):
    // distinctive marker no other family emits, and the generic `<tool` scan
    // never sees it (the opener is `<|tool_call_start|>`, not `<|tool_call>`).
    try parsePythonicToolCalls(allocator, effective_text, &calls);
    // Hy3 (Hunyuan 3) SUFFIXED tag format is tried FIRST among the tag
    // families: its wrapper (`<tool_calls:opensource>`) would also trip the
    // generic `<tool` scan below, which would misread the non-JSON
    // arg_key/arg_value body. When it matched, the generic scans are skipped
    // but the shared safety net at the bottom still runs.
    try parseHy3ToolCalls(allocator, effective_text, &calls);
    var search_pos: usize = if (calls.items.len > 0) effective_text.len else 0;
    while (search_pos < effective_text.len) {
        const rel = std.mem.indexOf(u8, effective_text[search_pos..], "<tool") orelse break;
        const after_tool = search_pos + rel + "<tool".len;
        if (after_tool >= effective_text.len) break;

        // Reject anything that isn't actually a tool open tag — the char
        // right after `<tool` must be `>`, whitespace, `_` (suffix coming),
        // or `|` (Gemma 4 close marker). Anything else is text like
        // `<toolkit>` or `<toolbar>`; advance past `<tool` and keep scanning.
        const next = effective_text[after_tool];
        if (next != '>' and next != ' ' and next != '\t' and next != '\n' and next != '_' and next != '|') {
            search_pos = after_tool;
            continue;
        }
        // Gemma 4 close marker: `<tool_call|>`. Detected when after `<tool`
        // we see `_call|`. Let the Gemma 4 branch below pick this up.
        if (next == '_'
            and after_tool + 6 <= effective_text.len
            and std.mem.eql(u8, effective_text[after_tool .. after_tool + 6], "_call|"))
        {
            search_pos = after_tool + 6;
            continue;
        }
        // If suffix is `_`, the only accepted continuations are
        // `_call`, `_calls`, `_request`, `_requests`. Any OTHER suffix gets
        // one more chance as the XML-element-TAG form before being rejected:
        // DSV4 embeds the tool name in the tag itself —
        //   `<tool_read>\n<path>mlx.html</path>\n</tool_read>`
        // (2026-06-10 pi html-ds4 capture; both calls leaked as text and pi
        // scored 0/4). Conditions kept tight: tag must close with `>` right
        // after the name (no attributes), an EXACT `</tool_NAME>` close must
        // exist, the name must not be a result-ish marker (`<tool_output>`
        // is DSV4 hallucinating a result, not calling a tool named
        // "output"), and the body must be entirely `<key>value</key>` args.
        if (next == '_') {
            const accepted = [_][]const u8{ "_call", "_calls", "_request", "_requests" };
            var matched = false;
            for (accepted) |suf| {
                if (after_tool + suf.len > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[after_tool .. after_tool + suf.len], suf)) continue;
                // Char right after the suffix must be tag-terminating; otherwise
                // `_call` could swallow `<tool_called_X>`.
                const post = after_tool + suf.len;
                if (post >= effective_text.len) break;
                const c = effective_text[post];
                if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '/' or c == '|') {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                const origin = after_tool - "<tool".len;
                if (try parseXmlElementTagToolCall(allocator, effective_text[origin..])) |etc| {
                    try calls.append(allocator, .{ .name = etc.call.name, .arguments = etc.call.arguments });
                    search_pos = origin + etc.consumed;
                    continue;
                }
                search_pos = after_tool;
                continue;
            }
        }

        // Self-closing `<tool …/>` form first — DSV4 emits this routinely
        // and the open/close scan below won't find a `</tool*>` close, so
        // without this branch the call gets dropped silently.
        const tag_origin = after_tool - "<tool".len;
        if (parseSelfClosingToolTag(effective_text[tag_origin..])) |sc| {
            const name_owned = try allocator.dupe(u8, sc.name);
            errdefer allocator.free(name_owned);
            const args_owned = try allocator.dupe(u8, sc.arguments);
            try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
            search_pos = tag_origin + sc.consumed;
            continue;
        }

        // Scan the opening tag to its `>`, capturing optional name attr.
        var attr_name: ?[]const u8 = null;
        var content_start: usize = 0;
        {
            var i: usize = after_tool;
            // Bound the scan so a malformed input can't run wild.
            const limit = @min(effective_text.len, after_tool + 256);
            while (i < limit and effective_text[i] != '>') : (i += 1) {
                if (effective_text[i] != 'n') continue;
                if (i + 5 > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[i .. i + 5], "name=")) continue;
                const q_pos = i + 5;
                if (q_pos >= effective_text.len) break;
                const quote = effective_text[q_pos];
                if (quote != '"' and quote != '\'') continue;
                const val_start = q_pos + 1;
                const val_end_rel = std.mem.indexOfScalar(u8, effective_text[val_start..], quote) orelse break;
                attr_name = effective_text[val_start .. val_start + val_end_rel];
                i = val_start + val_end_rel;
            }
            if (i >= limit or effective_text[i] != '>') {
                // Unclosed opening tag — skip past `<tool` and keep going.
                search_pos = after_tool;
                continue;
            }
            content_start = i + 1;
        }
        // Find the close marker. Prefer the close matching the OPENING tag's
        // name (`<tool_calls>` closes at `</tool_calls>`, never at a
        // `</tool_name>` child element inside the body — the XML-element
        // form puts `</tool…>`-shaped children first). When no exact match
        // exists, fall back to the earliest `</tool…>` with ANY suffix of
        // word characters: DSV4 hallucinates closes freely (`</tool_action>`
        // was captured live closing a `<tool_call>` open), and pinning an
        // exact-name list drops the whole call and leaks it as visible text.
        var close_rel: ?usize = null;
        var close_len: usize = 0;
        {
            const hay = effective_text[content_start..];
            // Opening tag name: "tool" + suffix up to the attr/`>` delimiter.
            var name_end = after_tool;
            while (name_end < effective_text.len and
                (std.ascii.isAlphanumeric(effective_text[name_end]) or effective_text[name_end] == '_')) : (name_end += 1)
            {}
            const open_name = effective_text[tag_origin + 1 .. name_end];
            var exact_buf: [40]u8 = undefined;
            if (open_name.len + 3 <= exact_buf.len) {
                const exact = std.fmt.bufPrint(&exact_buf, "</{s}>", .{open_name}) catch unreachable;
                if (std.mem.indexOf(u8, hay, exact)) |found| {
                    close_rel = found;
                    close_len = exact.len;
                }
            }
            if (close_rel == null) {
                var cpos: usize = 0;
                while (std.mem.indexOf(u8, hay[cpos..], "</tool")) |found| {
                    const open_at = cpos + found;
                    var j = open_at + "</tool".len;
                    const jlimit = @min(hay.len, j + 24);
                    var word_only = true;
                    while (j < jlimit and hay[j] != '>') : (j += 1) {
                        const c = hay[j];
                        if (!std.ascii.isAlphanumeric(c) and c != '_') {
                            word_only = false;
                            break;
                        }
                    }
                    if (word_only and j < jlimit and hay[j] == '>') {
                        close_rel = open_at;
                        close_len = j + 1 - open_at;
                        break;
                    }
                    cpos = open_at + "</tool".len;
                }
            }
        }
        if (close_rel == null) {
            // No `</tool…>` close in sight. Two ways this happens in the
            // wild: (1) the model emitted EOS mid-tool-call (`<tool_call>\n{
            // …well-formed args…}` then nothing); (2) max_tokens truncated.
            // The args object itself is usually intact — snap a balanced
            // JSON object from right after the open tag and try to parse it.
            if (balancedJsonObject(effective_text[content_start..])) |json_body| {
                const json_off = @intFromPtr(json_body.ptr) - @intFromPtr(effective_text[content_start..].ptr);
                const advance_to = content_start + json_off + json_body.len;
                // Hermes shape (top-level "name" + "arguments"): use as-is.
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                    search_pos = advance_to;
                    continue;
                }
                // Attribute shape — body is JUST the args, take name from attr.
                if (attr_name) |an| {
                    const name_owned = try allocator.dupe(u8, an);
                    errdefer allocator.free(name_owned);
                    const args_owned = try allocator.dupe(u8, json_body);
                    try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                    search_pos = advance_to;
                    continue;
                }
            }
            // Truncated Hermes/XML tool call: the model emitted
            // `<tool_call><function=NAME><parameter=KEY>…` (Hermes function-tag)
            // or the XML-element `<tool_name>NAME</tool_name>…` shape and ran
            // out of tokens before ANY closing tag — so there's no balanced
            // JSON to snap. The OPENING tags still carry the tool NAME; recover
            // that (parseHermesToolCall breaks out of its parameter loop on a
            // missing `</parameter>`, yielding name + `{}` args) so the call is
            // recognized as a truncated writeFile instead of being DROPPED and
            // leaked into visible content (live JFK-novel capture, 2026-06-20:
            // a 19k-char writeFile cut off mid-content was silently lost and
            // the app fired the wrong "malformed tag" ghost nudge). Recovering
            // the name is enough for the client to fire the right chunk/append
            // nudge; we do NOT salvage the partial content (a half-written file
            // is worse than a re-issued chunked write). The text is truncated
            // to its end here, so advance past it to terminate the scan.
            if (parseHermesToolCall(allocator, effective_text[content_start..])) |tc| {
                try calls.append(allocator, tc);
                search_pos = effective_text.len;
                continue;
            }
            if (parseXmlElementToolCall(allocator, effective_text[content_start..])) |tc| {
                try calls.append(allocator, tc);
                search_pos = effective_text.len;
                continue;
            }
            break;
        }
        const content = std.mem.trim(u8, effective_text[content_start .. content_start + close_rel.?], " \t\n\r");

        // Pre-clean the body before the parse attempts. Two DSV4 quirks:
        //   • The args object may be wrapped in `<parameters>…</parameters>`.
        //   • The model sometimes emits an extra trailing `}` after the
        //     proper JSON close. Find the first balanced JSON object via
        //     depth tracking and use just that.
        const unwrapped = stripParametersWrapper(content);
        // Some models (often when echoing a Jinja `{{ }}` example) wrap the args
        // object in an extra brace layer: `<tool_call>{{"name":…}}</tool_call>`.
        // Strip one layer so the body parses as JSON. Only fires on the `{{…}}`
        // shape (which otherwise fails to parse), so valid single-brace bodies are
        // untouched. Mirrors the unwrap in `parseGemma4ToolCall`.
        const balanced: []const u8 = unwrapDoubleBraces(balancedJsonObject(unwrapped) orelse unwrapped);

        // Try to extract a tool call from the body. Three shapes, tried in
        // priority order:
        //   1. Attribute form (`name="X"` captured, body is JUST args).
        //   2. Canonical Hermes JSON ({"name":"X","arguments":{...}}).
        //   3. Hermes function-tag format (parseHermesToolCall).
        var parsed_ok = false;
        // 0. Function-tag form FIRST when the body actually carries a
        // `<function=` opener. The dialect qwen 3.5/3.6's own template
        // mandates rides inside the SAME `<tool_call>` wrapper the JSON form
        // uses, and a `<parameter=…>` VALUE is arbitrary bytes — a written
        // package.json made `balancedJsonObject` snap the FILE CONTENT, whose
        // "name" key then became the tool name (live 2026-08-12: pi answered
        // "Tool voxel-pagoda-garden not found" and the model looped on it).
        // The JSON shapes below can never contain that opener, so this only
        // reorders the case that was being read wrong.
        if (std.mem.indexOf(u8, content, "<function=") != null) {
            if (parseHermesToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (attr_name) |an| {
                if (std.json.parseFromSlice(std.json.Value, allocator, balanced, .{})) |parsed| {
                    defer parsed.deinit();
                    if (parsed.value == .object) {
                        const name_owned = try allocator.dupe(u8, an);
                        errdefer allocator.free(name_owned);
                        const args_owned = try allocator.dupe(u8, balanced);
                        try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                        parsed_ok = true;
                    }
                } else |_| {}
            }
        }
        if (!parsed_ok) {
            if (tryParseJsonToolCall(allocator, balanced)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseHermesToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseXmlElementToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }

        if (parsed_ok) {
            // Body consumed cleanly — advance past the close marker.
            search_pos = content_start + close_rel.? + close_len;
        } else {
            // Body wasn't a tool call. Advance only past the opening `>`
            // so inner `<tool…>` blocks (if any) still get parsed. This is
            // what makes the outer-wrapper case work: the outer
            // `<tool_calls>` has no JSON body, but its content contains
            // real `<tool name="X">` inner calls.
            search_pos = content_start;
        }
    }

    // Gemma 4 format: <|tool_call>call:name{args}<tool_call|>
    if (calls.items.len == 0) {
        search_pos = 0;
        while (search_pos < effective_text.len) {
            const tag_start = std.mem.indexOf(u8, effective_text[search_pos..], "<|tool_call>") orelse break;
            const content_start = search_pos + tag_start + "<|tool_call>".len;
            const tag_end_opt = std.mem.indexOf(u8, effective_text[content_start..], "<tool_call|>");
            const content = if (tag_end_opt) |tag_end|
                std.mem.trim(u8, effective_text[content_start .. content_start + tag_end], " \t\n\r")
            else
                // Incomplete tool call (model hit EOS before closing tag) — use rest of text
                std.mem.trim(u8, effective_text[content_start..], " \t\n\r");

            search_pos = if (tag_end_opt) |tag_end|
                content_start + tag_end + "<tool_call|>".len
            else
                effective_text.len;

            // No <tool_call|> close = the generation was CUT mid-call (EOS,
            // max_tokens, or the degenerate-tail-loop guard) — fragment values
            // are dropped inside the parse, only completed args ship.
            if (parseGemma4ToolCall(allocator, content, tag_end_opt == null)) |tc| {
                try calls.append(allocator, tc);
            } else {
                log.info("  [tool-parse] Gemma4 parse FAILED for: {s}\n", .{content[0..@min(content.len, 200)]});
            }
        }
    }

    // Bare Hermes function tag with NO <tool_call> wrapper. The outer scan
    // triggers on the substring `<tool`, which a lone `<function=…></function>`
    // (even with a trailing `</tool_call>`, since that is `</too…`, not `<tool`)
    // never provides — so a model that drops the OPENING <tool_call> would leak
    // the whole call as visible text. Only fire when a `<function=` opener AND a
    // `<parameter=` are both present, so prose that merely mentions the tag can't
    // false-fire (parseHermesToolCall needs a real parameter to recover args).
    if (calls.items.len == 0) {
        if (std.mem.indexOf(u8, effective_text, "<function=") != null and
            std.mem.indexOf(u8, effective_text, "<parameter=") != null)
        {
            if (parseHermesToolCall(allocator, effective_text)) |tc| {
                try calls.append(allocator, tc);
            }
        }
    }

    // If no <tool_call> tags, try to find raw JSON tool call(s). Never on
    // text carrying the Inkling invoke marker: that format OWNED the parse,
    // and a call it deliberately skipped (no recoverable name) must not be
    // resurrected from its payload bytes by the bare-JSON heuristic — the
    // comment below says "no tag syntax anywhere", and there IS tag syntax.
    if (calls.items.len == 0 and std.mem.indexOf(u8, text, INKLING_INVOKE_TAG) == null) {
        var trimmed = std.mem.trim(u8, effective_text, " \t\n\r");
        if (std.mem.startsWith(u8, trimmed, "</tool_call>")) {
            trimmed = std.mem.trim(u8, trimmed["</tool_call>".len..], " \t\n\r");
        }
        // JSON ARRAY of {name, arguments} objects — parallel tool calls from
        // models without a trained tool format. Gemma 3 emits a ```json fence
        // around the array; skip one leading fence line before the check so
        // the array (not the first object inside it) is what we parse —
        // otherwise only the first call survives and the rest silently drop.
        var array_probe = trimmed;
        if (std.mem.startsWith(u8, array_probe, "```")) {
            if (std.mem.indexOfScalar(u8, array_probe, '\n')) |nl| {
                array_probe = std.mem.trim(u8, array_probe[nl + 1 ..], " \t\n\r");
            }
        }
        if (array_probe.len > 0 and array_probe[0] == '[') {
            if (balancedJsonArray(array_probe)) |arr_body| {
                try appendJsonToolCallArray(allocator, arr_body, &calls);
            }
        }
        if (calls.items.len == 0) {
            if (std.mem.indexOf(u8, trimmed, "{")) |brace_pos| {
                const json_start = trimmed[brace_pos..];
                // Snap the first balanced object so trailing garbage can't poison
                // the parse — models without a trained tool format (Gemma 3) emit
                // the call inside a ```json fence, leaving "\n```" after the `}`.
                const json_body = balancedJsonObject(json_start) orelse json_start;
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                }
            }
        }
        // Everything appended in this block was inferred from BARE JSON — no
        // tag syntax anywhere. Mark provenance so the request chokepoint
        // (server.parseToolCallsForRequest → filterInferredBySchema) can
        // validate the name against the declared tools: the first balanced
        // {"name": …} object in truncated prose/code is routinely a DATA
        // record, not a call (the George Washington class).
        for (calls.items) |*tc| tc.inferred = true;
    }

    if (calls.items.len == 0) return null;

    // Final safety net: EVERY emitted call carries valid-JSON arguments, whatever
    // converter built them. The direct-construction converters (Gemma custom
    // format, Hermes) can still emit invalid JSON on a pathological value — e.g. a
    // JSON-style string with a bad escape (`\q`) copied verbatim. std.json is the
    // arbiter: if the args don't parse, try the tolerant re-serialize (which
    // re-escapes lone backslashes / control bytes / inner quotes), and if THAT
    // still fails, fall back to `{}` — a client can retry a named call with empty
    // args, but invalid JSON it cannot parse at all. This makes the "args are
    // always valid JSON" invariant structural, not per-converter.
    for (calls.items) |*tc| {
        if (jsonIsValidObject(allocator, tc.arguments)) continue;
        if (looseRepairToolCallJson(allocator, tc.arguments)) |repaired| {
            if (jsonIsValidObject(allocator, repaired)) {
                allocator.free(tc.arguments);
                tc.arguments = repaired;
                continue;
            }
            allocator.free(repaired);
        }
        log.info("  [tool-parse] unrepairable args for '{s}' → empty object (never ship invalid JSON)\n", .{tc.name});
        const empty = try allocator.dupe(u8, "{}");
        allocator.free(tc.arguments);
        tc.arguments = empty;
    }

    return try calls.toOwnedSlice(allocator);
}

/// True when `s` strict-parses as a JSON object. Cheap arbiter for the
/// parseToolCalls safety net.
fn jsonIsValidObject(allocator: std.mem.Allocator, s: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, s, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object;
}

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8, // JSON string
    /// True when the call was HEURISTICALLY inferred from bare JSON in the
    /// output (no tag syntax — the model never said "this is a tool call").
    /// The request chokepoint validates inferred names against the declared
    /// tools schema: a truncated DATA object like {"name": "George
    /// Washington", "num": 1, …} must never become a tool call (live pi
    /// capture 2026-07-13). Explicit tag-format calls keep flowing even with
    /// unknown names — "tool not found" is feedback the model corrects from.
    inferred: bool = false,
};

/// Locate a tool's `properties` map in an OpenAI-shaped tools array. Accepts
/// both the wrapped (`{"type":"function","function":{…}}`) and flat forms.
/// The tool's JSON-Schema `parameters` object (Anthropic `input_schema` too).
/// Callers that need only the property map use `toolPropertiesFor`; the hoist
/// also needs `required`, which lives beside `properties` here.
fn toolParametersFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    if (tools != .array) return null;
    for (tools.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv != .string or !std.mem.eql(u8, nv.string, name)) continue;
        const params = func.get("parameters") orelse func.get("input_schema") orelse return null;
        if (params != .object) return null;
        return params.object;
    }
    return null;
}

fn toolPropertiesFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    const params = toolParametersFor(tools, name) orelse return null;
    const props = params.get("properties") orelse return null;
    if (props != .object) return null;
    return props.object;
}

/// The types a misplaced value may be hoisted on. A required OBJECT/ARRAY param
/// found inside another container is too speculative to move — only scalars.
fn isScalarJsonType(want: []const u8) bool {
    return std.mem.eql(u8, want, "string") or std.mem.eql(u8, want, "integer") or
        std.mem.eql(u8, want, "number") or std.mem.eql(u8, want, "boolean");
}

/// The JSON type a property declares. `"type"` may be a union array
/// (`["string","null"]`) — the first non-null entry wins. Absent/odd → null,
/// which means "leave the value alone".
fn declaredJsonType(prop: std.json.Value) ?[]const u8 {
    if (prop != .object) return null;
    const t = prop.object.get("type") orelse return null;
    switch (t) {
        .string => |s| return s,
        .array => |arr| {
            for (arr.items) |item| {
                if (item != .string) continue;
                if (std.mem.eql(u8, item.string, "null")) continue;
                return item.string;
            }
            return null;
        },
        else => return null,
    }
}

/// Tolerant boolean spelling — the union of what weak models actually emit:
/// Python's `False`, shell's `0`/`1`, prose `yes`/`no`, plus the trailing-comma
/// weld the app's `appendFlagIsTrue` already tolerates. Anything else → null
/// (never guess; an honest client validation error beats a wrong value).
fn looseBoolSpelling(raw: []const u8) ?bool {
    const s = std.mem.trim(u8, raw, " \t\r\n,");
    if (s.len == 0 or s.len > 5) return null;
    var buf: [5]u8 = undefined;
    const lower = std.ascii.lowerString(buf[0..s.len], s);
    if (std.mem.eql(u8, lower, "true") or std.mem.eql(u8, lower, "yes") or
        std.mem.eql(u8, lower, "on") or std.mem.eql(u8, lower, "1")) return true;
    if (std.mem.eql(u8, lower, "false") or std.mem.eql(u8, lower, "no") or
        std.mem.eql(u8, lower, "off") or std.mem.eql(u8, lower, "0")) return false;
    return null;
}

/// Coerce ONE value to the declared type. Returns null when the value already
/// matches or the spelling is undecidable. Allocated replacements are appended
/// to `owned`; parsed sub-documents to `docs` — both are freed by the caller
/// AFTER the object is re-serialized, because the returned Value borrows them.
fn coerceValueToType(
    allocator: std.mem.Allocator,
    v: std.json.Value,
    want: []const u8,
    owned: *std.ArrayList([]const u8),
    docs: *std.ArrayList(std.json.Parsed(std.json.Value)),
) !?std.json.Value {
    // A CONTAINER param carried through a tag format arrives as a STRING — no tag
    // format types its values. Live (15 hits in one session): pi's `edit.edits`,
    // an array of {oldText,newText}, shipped as `"[{\"oldText\":…}]"` on every
    // call. Strict parse only: a value that isn't well-formed JSON of the
    // DECLARED kind is left alone rather than guessed at.
    if (std.mem.eql(u8, want, "array") or std.mem.eql(u8, want, "object")) {
        if (v != .string) return null;
        const t = std.mem.trim(u8, v.string, " \t\r\n");
        const want_array = want[0] == 'a';
        if (t.len == 0 or t[0] != (if (want_array) @as(u8, '[') else @as(u8, '{'))) return null;

        // Strict first; a well-formed container is the common case.
        if (std.json.parseFromSlice(std.json.Value, allocator, t, .{})) |doc| {
            const kind_ok = if (want_array) doc.value == .array else doc.value == .object;
            if (kind_ok) {
                try docs.append(allocator, doc);
                return doc.value;
            }
            var d = doc;
            d.deinit();
            return null;
        } else |_| {}

        // The container string is itself MANGLED (live: pi's `edits` array with a
        // missing comma between two object values). Same big-file escaping class
        // as looseRepairToolCallJson — re-serialize tolerantly, then STRICT-parse
        // the result so a mis-repair that yields invalid JSON is discarded.
        const repaired = looseRepairContainer(allocator, t) orelse return null;
        defer allocator.free(repaired);
        var rdoc = std.json.parseFromSlice(std.json.Value, allocator, repaired, .{}) catch return null;
        const rkind_ok = if (want_array) rdoc.value == .array else rdoc.value == .object;
        if (!rkind_ok) {
            rdoc.deinit();
            return null;
        }
        try docs.append(allocator, rdoc); // ownership transfers; never deinit here
        return rdoc.value;
    }
    if (std.mem.eql(u8, want, "boolean")) {
        return switch (v) {
            .bool => null,
            .string => |s| if (looseBoolSpelling(s)) |b| std.json.Value{ .bool = b } else null,
            .integer => |i| if (i == 0 or i == 1) std.json.Value{ .bool = i == 1 } else null,
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "integer")) {
        return switch (v) {
            .string => |s| blk: {
                const n = std.fmt.parseInt(i64, std.mem.trim(u8, s, " \t\r\n"), 10) catch break :blk null;
                break :blk std.json.Value{ .integer = n };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "number")) {
        return switch (v) {
            .string => |s| blk: {
                const t = std.mem.trim(u8, s, " \t\r\n");
                if (std.fmt.parseInt(i64, t, 10)) |n| break :blk std.json.Value{ .integer = n } else |_| {}
                const f = std.fmt.parseFloat(f64, t) catch break :blk null;
                if (std.math.isNan(f) or std.math.isInf(f)) break :blk null; // not representable in JSON
                break :blk std.json.Value{ .float = f };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "string")) {
        // The inverse repair: the tag/custom formats infer type from the value's
        // SPELLING (isJsonLiteral), so a string param whose content happens to
        // read as `false`/`42` was shipped as a bool/number. The schema is the
        // only thing that can tell those apart.
        return switch (v) {
            .bool => |b| std.json.Value{ .string = if (b) "true" else "false" },
            .integer => |i| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{i});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .float => |f| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{f});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .number_string => |s| std.json.Value{ .string = s },
            else => null,
        };
    }
    return null; // a type we don't model — never guessed
}

fn jsonTypeMatches(v: std.json.Value, want: []const u8) bool {
    if (v == .null) return true; // an explicit null satisfies any optional field
    if (std.mem.eql(u8, want, "boolean")) return v == .bool;
    if (std.mem.eql(u8, want, "integer")) return v == .integer or v == .number_string;
    if (std.mem.eql(u8, want, "number")) return v == .integer or v == .float or v == .number_string;
    if (std.mem.eql(u8, want, "string")) return v == .string;
    if (std.mem.eql(u8, want, "array")) return v == .array;
    if (std.mem.eql(u8, want, "object")) return v == .object;
    return true; // a type we don't model constrains nothing
}

/// True when every argument whose type the tool DECLARES actually carries that
/// JSON type. This is the invariant strict clients (Claude Code, pi, opencode)
/// enforce and reject on; the format corpus asserts it universally so a new
/// entry in any family is covered without a bespoke assertion. Undeclared keys
/// and unmodelled types constrain nothing.
pub fn toolCallConformsToSchema(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer tools.deinit();
    const props = toolPropertiesFor(tools.value, call.name) orelse return true;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .object) return true;

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const prop = props.get(entry.key_ptr.*) orelse continue;
        const want = declaredJsonType(prop) orelse continue;
        if (!jsonTypeMatches(entry.value_ptr.*, want)) return false;
    }
    return true;
}

/// True when a REQUIRED scalar param is MISSING at the top level while sitting
/// buried inside one of the tool's declared container args — exactly the shape
/// `hoistMisplacedRequiredParams` exists to repair, and exactly what a strict
/// client rejects with "must have required properties X". The format corpus
/// asserts this is FALSE for every entry that declares tools, so a future family
/// that produces the buried shape is covered without a bespoke assertion.
pub fn requiredParamIsBuried(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return false;
    defer tools.deinit();
    const params = toolParametersFor(tools.value, call.name) orelse return false;
    const props_v = params.get("properties") orelse return false;
    const required_v = params.get("required") orelse return false;
    if (props_v != .object or required_v != .array) return false;
    const props = props_v.object;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;

    for (required_v.array.items) |req_v| {
        if (req_v != .string) continue;
        const name = req_v.string;
        if (parsed.value.object.get(name) != null) continue;
        const want = declaredJsonType(props.get(name) orelse continue) orelse continue;
        if (!isScalarJsonType(want)) continue;

        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            const prop = props.get(e.key_ptr.*) orelse continue;
            if (containerItemDeclares(prop, name)) continue;
            if (buriedParamIn(e.value_ptr, name, want, false) != null) return true;
        }
    }
    return false;
}

/// Coerce every parsed call's `arguments` to the types its tool DECLARES.
///
/// Class: **type inference from value spelling.** Neither the tag formats
/// (`<parameter=x>False</parameter>`, Gemma's `key:false`) nor a model writing
/// JSON by hand carries type information a parser can trust — `isJsonLiteral`
/// has to guess from the bytes, and it guesses wrong in both directions:
///   • `False` / `"false"` on a boolean param → shipped as the STRING "false";
///   • `false` / `42` as a string param's CONTENT → shipped as a bool/number.
/// Strict clients (Claude Code, pi, opencode) reject both and the model, which
/// cannot see its own serialized request, loops forever "fixing" a value that
/// was already correct. The tool schema is the only disambiguator, and it is
/// already threaded to every parse site as `tools_json`.
///
/// Contract: only where the schema declares a type, and only when the spelling
/// is unambiguous. An undecidable value is left alone so the client's
/// validation error stays honest. Correct calls from strong models are
/// byte-unchanged (nothing re-serializes unless a coercion actually fired).
///
/// Scalars AND containers: a tag format types nothing, so a declared
/// `array`/`object` arrives as a string of JSON and `coerceValueToType` parses
/// it back (strict, then one tolerant repair re-validated strictly). This
/// comment used to say "only SCALARS are touched" — it was two weeks stale by
/// the time it was read as a spec and filed as issue #94. A contract comment is
/// load-bearing documentation: pin it with a test (`issue #94: a container
/// param carried as literal TEXT…`), don't let it drift.
pub fn coerceToolArgsToSchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const props = toolPropertiesFor(tools.value, call.name) orelse continue;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var owned = std.ArrayList([]const u8).empty;
        defer {
            for (owned.items) |s| allocator.free(s);
            owned.deinit(allocator);
        }
        // Parsed container sub-documents: the rewritten object BORROWS from these,
        // so they must outlive the Stringify below.
        var docs = std.ArrayList(std.json.Parsed(std.json.Value)).empty;
        defer {
            for (docs.items) |*d| d.deinit();
            docs.deinit(allocator);
        }

        var changed = false;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const prop = props.get(entry.key_ptr.*) orelse continue;
            const want = declaredJsonType(prop) orelse continue;
            const new_val = try coerceValueToType(allocator, entry.value_ptr.*, want, &owned, &docs) orelse continue;
            entry.value_ptr.* = new_val;
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        log.info("  [tool-parse] coerced {s} arguments to the declared schema types\n", .{call.name});
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does this container's ITEM schema declare `name` as one of its own
/// properties? If it does, a `name` sitting inside it BELONGS there and must
/// never be lifted out — a multi-file edit tool would lose its per-item paths.
/// This is the whole safety argument for the hoist below, so it is deliberately
/// conservative: no declared item `properties` map ⇒ nothing is proven ⇒ false.
fn containerItemDeclares(prop: std.json.Value, name: []const u8) bool {
    if (prop != .object) return false;
    // Array property → the item schema. Object property → the schema itself.
    const inner = if (prop.object.get("items")) |items| items else prop;
    if (inner != .object) return false;
    const props = inner.object.get("properties") orelse return false;
    if (props != .object) return false;
    return props.object.get(name) != null;
}

/// Scalar equality — the only shapes we ever hoist on. A container value is
/// never a candidate, so a structural compare is not needed.
fn jsonScalarEql(a: std.json.Value, b: std.json.Value) bool {
    return switch (a) {
        .string => |s| b == .string and std.mem.eql(u8, s, b.string),
        .integer => |i| b == .integer and i == b.integer,
        .bool => |x| b == .bool and x == b.bool,
        else => false,
    };
}

/// Find `name` buried one level down inside `container`, requiring UNANIMITY:
/// every object in the container carries `name`, all with the same scalar value,
/// and that value matches the type the schema declares for the top-level param.
/// Anything less is ambiguous → null. With `remove`, also deletes it from each
/// object (the returned value keeps pointing into the parse arena, which
/// outlives the removal).
fn buriedParamIn(
    container: *std.json.Value,
    name: []const u8,
    want: []const u8,
    remove: bool,
) ?std.json.Value {
    switch (container.*) {
        .array => |*arr| {
            if (arr.items.len == 0) return null;
            var found: ?std.json.Value = null;
            for (arr.items) |*item| {
                if (item.* != .object) return null;
                const v = item.object.get(name) orelse return null; // must be in EVERY item
                if (v == .null or !jsonTypeMatches(v, want)) return null;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) return null; // items disagree → ambiguous
                } else found = v;
            }
            if (remove) for (arr.items) |*item| {
                _ = item.object.orderedRemove(name);
            };
            return found;
        },
        .object => |*obj| {
            const v = obj.get(name) orelse return null;
            if (v == .null or !jsonTypeMatches(v, want)) return null;
            if (remove) _ = obj.orderedRemove(name);
            return v;
        },
        else => return null,
    }
}

/// Lift a REQUIRED top-level parameter the model BURIED inside a container arg.
///
/// Class: **misplaced required param.** A weak model that has internalized "the
/// edit object holds everything about the edit" writes the file path INSIDE each
/// `edits[]` item and never emits it at the top level:
///
///     {"edits":[{"newText":"…","oldText":"…","path":"us_presidents/generate_site.sh"}]}
///
/// The call is valid JSON and correctly escaped, so no repair path fires and no
/// type is wrong — it is purely in the wrong PLACE, which only the schema can
/// know. Strict clients reject it ("path: must have required properties path"),
/// and the model, which cannot see its own serialized request, re-emits the same
/// call until it gives up and falls back to rewriting the whole file with
/// `write` — three dead multi-thousand-token rounds in the live 2026-07-13 pi
/// session (gemma-4-26B-A4B). Same shape as the Claude Code replace_all loop.
///
/// Contract — every condition is read off the SCHEMA, never guessed:
///   • the param is declared REQUIRED at the top level and is ABSENT there;
///   • it is declared a SCALAR (hoisting a container is too speculative);
///   • it sits inside a DECLARED container arg whose item schema does NOT
///     declare it (`containerItemDeclares` — otherwise it belongs where it is);
///   • every object in that container carries it with the SAME value and the
///     declared type.
/// Ambiguity of any kind (items disagreeing, two containers offering different
/// values, a wrong-typed value) leaves the call untouched so the client's
/// validation error stays honest. A compliant call never re-serializes.
pub fn hoistMisplacedRequiredParams(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const params = toolParametersFor(tools.value, call.name) orelse continue;
        const props_v = params.get("properties") orelse continue;
        const required_v = params.get("required") orelse continue;
        if (props_v != .object or required_v != .array) continue;
        const props = props_v.object;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var changed = false;
        for (required_v.array.items) |req_v| {
            if (req_v != .string) continue;
            const name = req_v.string; // borrows from `tools`, which outlives the Stringify
            if (parsed.value.object.get(name) != null) continue; // present — nothing misplaced

            const want = declaredJsonType(props.get(name) orelse continue) orelse continue;
            if (!isScalarJsonType(want)) continue;

            // Probe every declared container WITHOUT mutating: a value found in
            // two containers that disagree is ambiguous, and we must not have
            // already gutted the first one by then.
            var found: ?std.json.Value = null;
            var ambiguous = false;
            var it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                const v = buriedParamIn(e.value_ptr, name, want, false) orelse continue;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) {
                        ambiguous = true;
                        break;
                    }
                } else found = v;
            }
            if (ambiguous or found == null) continue;

            // Commit: strip it from the containers, install it at the top level.
            it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                _ = buriedParamIn(e.value_ptr, name, want, true);
            }
            try parsed.value.object.put(parsed.arena.allocator(), name, found.?);
            log.info("  [tool-parse] hoisted misplaced required arg '{s}' to the top level for {s}\n", .{ name, call.name });
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does the OpenAI-shaped tools array declare a function with this name?
/// Accepts both the wrapped ({"type":"function","function":{…}}) and flat
/// forms, mirroring `toolPropertiesFor`. An unparseable/odd `tools_json`
/// returns TRUE — never drop a call because OUR schema parse failed.
pub fn toolNameIsDeclared(allocator: std.mem.Allocator, tools_json: []const u8, name: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .array) return true;
    for (parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv == .string and std.mem.eql(u8, nv.string, name)) return true;
    }
    return false;
}

/// Drop HEURISTICALLY-inferred raw-JSON calls (`ParsedToolCall.inferred`)
/// whose name the request never declared. The George Washington class: a
/// generation truncated mid-data-script leaves `{"name": "George Washington",
/// "num": 1, …}` as the first balanced object, and the flat-shape synthesis
/// promoted it to a tool call the client can only answer with "tool not
/// found" — burning the turn. Explicit tag-format calls are NEVER dropped
/// here (undeclared name → honest client feedback the model corrects from).
///
/// Returns the retained slice (shrunk in place), or null when every call was
/// dropped — the caller treats that exactly like "no tool calls", so the text
/// stays visible content and finish_reason (e.g. "length") is preserved.
pub fn filterInferredBySchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !?[]ParsedToolCall {
    var kept: usize = 0;
    for (calls) |tc| {
        if (!tc.inferred or toolNameIsDeclared(allocator, tools_json, tc.name)) {
            calls[kept] = tc;
            kept += 1;
        } else {
            log.warn("  [tool-parse] dropped inferred raw-JSON call '{s}' — name not among the request's tools (data object, not a call)\n", .{tc.name});
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    if (kept == calls.len) return calls;
    if (kept == 0) {
        allocator.free(calls);
        return null;
    }
    return try allocator.realloc(calls, kept);
}

/// Last-resort tool-call inference for models that emit JUST the arguments
/// object — no tool name, no wrapper syntax. Observed live from Gemma 4 12B
/// via Claude Code: ```` ```json\n{"file_path": …, "content": …}\n``` ````
/// with the Write tool defined; parseToolCalls finds nothing (no "name" key)
/// and the un-executed JSON leaked into the visible text.
///
/// Conservative contract — fires ONLY when parseToolCalls returned null:
///   • the visible content (after a leading think block) must START with the
///     JSON object, optionally wrapped in a markdown fence;
///   • the object's keys must satisfy exactly ONE tool in `tools_json`
///     (OpenAI shape): required ⊆ keys AND keys ⊆ properties;
///   • zero or multiple matching tools → null (ambiguity never guesses).
pub fn inferBareJsonToolCalls(allocator: std.mem.Allocator, text: []const u8, tools_json: []const u8) !?[]ParsedToolCall {
    // Strip a leading think block, mirroring parseToolCalls.
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        effective_text = std.mem.trimStart(u8, text[think_end + 8 ..], "\n ");
    }
    var trimmed = std.mem.trimStart(u8, effective_text, "\n \t");
    // Skip one opening markdown fence line (``` or ```json etc.).
    if (std.mem.startsWith(u8, trimmed, "```")) {
        const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return null;
        trimmed = std.mem.trimStart(u8, trimmed[nl + 1 ..], "\n \t");
    }
    // The object must LEAD the visible content — a JSON example mid-prose is
    // never promoted to a call.
    if (!std.mem.startsWith(u8, trimmed, "{")) return null;
    const obj_slice = balancedJsonObject(trimmed) orelse return null;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, obj_slice, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;
    if (obj.count() == 0) return null;

    const tools_parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return null;
    defer tools_parsed.deinit();
    if (tools_parsed.value != .array) return null;

    var match_name: ?[]const u8 = null;
    for (tools_parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func_val = tool_val.object.get("function") orelse continue;
        if (func_val != .object) continue;
        const name_val = func_val.object.get("name") orelse continue;
        if (name_val != .string) continue;
        const params_val = func_val.object.get("parameters") orelse continue;
        if (params_val != .object) continue;
        const props_val = params_val.object.get("properties") orelse continue;
        if (props_val != .object) continue;
        const props = props_val.object;
        if (props.count() == 0) continue;

        // keys ⊆ properties
        var keys_ok = true;
        var it = obj.iterator();
        while (it.next()) |kv| {
            if (props.get(kv.key_ptr.*) == null) {
                keys_ok = false;
                break;
            }
        }
        if (!keys_ok) continue;

        // required ⊆ keys
        if (params_val.object.get("required")) |req_val| {
            if (req_val == .array) {
                var req_ok = true;
                for (req_val.array.items) |r| {
                    if (r != .string) continue;
                    if (obj.get(r.string) == null) {
                        req_ok = false;
                        break;
                    }
                }
                if (!req_ok) continue;
            }
        }

        if (match_name != null) return null; // ambiguous — never guess
        match_name = name_val.string;
    }

    const name = match_name orelse return null;
    const calls = try allocator.alloc(ParsedToolCall, 1);
    errdefer allocator.free(calls);
    const name_owned = try allocator.dupe(u8, name);
    errdefer allocator.free(name_owned);
    calls[0] = .{ .name = name_owned, .arguments = try allocator.dupe(u8, obj_slice) };
    log.info("  [tool-parse] inferred bare-args call to '{s}' via unique schema match\n", .{name});
    return calls;
}

/// Strip a leading `<parameters>` / trailing `</parameters>` pair if both
/// are present. DSV4 occasionally wraps the args object in this XML
/// container — leaving it in place would defeat the JSON parser. Trims
/// surrounding whitespace.
fn stripParametersWrapper(content: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const open = "<parameters>";
    const close = "</parameters>";
    if (trimmed.len < open.len + close.len) return trimmed;
    if (!std.mem.startsWith(u8, trimmed, open)) return trimmed;
    if (!std.mem.endsWith(u8, trimmed, close)) return trimmed;
    const inner = trimmed[open.len .. trimmed.len - close.len];
    return std.mem.trim(u8, inner, " \t\n\r");
}

/// Return the substring of `content` covering the first brace-balanced JSON
/// object, or null if no balanced object is present. Handles quoted strings
/// (so `}` inside a string doesn't close the object) and escape sequences.
/// Used to tolerate the DSV4 `{"…"}}` extra-trailing-`}` artifact.
/// Try to parse a tool tag whose body lives entirely in XML-ish
/// attributes. Two terminator shapes are accepted:
///
///   Self-closing /> :
///     <tool name="shell" arguments="{...}"/>
///     <tool name="X" arguments='{...}' />
///     <tool name="X" arguments="{...}'/>             (broken quote — DSV4)
///
///   Empty-body explicit close ></tool*> :
///     <tool_call name="cwd" arguments="{...}"></tool_call>
///     <tool name="X" arguments="{...}"></tool_calls>  (mismatched close OK)
///
/// `slice` MUST start with `<tool` (the caller has already validated the
/// terminator char after that prefix). Returns name + arguments slices into
/// `slice` plus the number of bytes consumed (up to and including the final
/// `>` or `</tool*>` close), or null if this isn't a shape we can parse.
///
/// We never trust the XML quoting around `arguments=` — the model frequently
/// uses `"` to open and `'` to close, or doesn't escape inner `"` at all.
/// Instead we expect a `{` immediately (optionally after a single opening
/// quote) and snap a balanced JSON object via `balancedJsonObject`. That
/// strategy is what makes the DSV4 fragment parse cleanly even though the
/// surrounding XML is malformed.
fn parseSelfClosingToolTag(slice: []const u8) ?struct { name: []const u8, arguments: []const u8, consumed: usize } {
    if (slice.len < "<tool".len) return null;
    if (!std.mem.startsWith(u8, slice, "<tool")) return null;
    // Bound the scan window so a runaway tag can't pin us forever.
    const limit = @min(slice.len, 8192);

    // 1. Locate `name="X"` (or `'X'`) attribute. We don't enforce ordering
    //    against `arguments=` — DSV4 sometimes interleaves them.
    var name: ?[]const u8 = null;
    var name_end: usize = "<tool".len;
    {
        var i: usize = "<tool".len;
        while (i + "name=".len < limit) : (i += 1) {
            if (slice[i] != 'n') continue;
            if (!std.mem.eql(u8, slice[i .. i + "name=".len], "name=")) continue;
            const q_pos = i + "name=".len;
            if (q_pos >= limit) break;
            const quote = slice[q_pos];
            if (quote != '"' and quote != '\'') continue;
            const val_start = q_pos + 1;
            const val_end_rel = std.mem.indexOfScalar(u8, slice[val_start..limit], quote) orelse break;
            name = slice[val_start .. val_start + val_end_rel];
            name_end = val_start + val_end_rel + 1;
            break;
        }
    }

    // 2. Locate `arguments=` and snap the next balanced JSON object as
    //    the value. We deliberately don't parse the XML quote — see
    //    function docs.
    var args: ?[]const u8 = null;
    var args_end: usize = 0;
    {
        var i: usize = "<tool".len;
        while (i + "arguments=".len < limit) : (i += 1) {
            if (slice[i] != 'a') continue;
            if (!std.mem.eql(u8, slice[i .. i + "arguments=".len], "arguments=")) continue;
            var j: usize = i + "arguments=".len;
            // Skip an opening quote if present.
            if (j < limit and (slice[j] == '"' or slice[j] == '\'')) j += 1;
            // Skip whitespace between the quote and the JSON object.
            while (j < limit and (slice[j] == ' ' or slice[j] == '\t')) j += 1;
            if (j >= limit or slice[j] != '{') break;
            const sub = slice[j..limit];
            const json = balancedJsonObject(sub) orelse break;
            args = json;
            const json_off = @intFromPtr(json.ptr) - @intFromPtr(sub.ptr);
            args_end = j + json_off + json.len;
            break;
        }
    }

    if (name == null or args == null) return null;

    // 3. Scan forward from the end of whichever attribute landed last to
    //    the terminator. Tolerate a stray closing quote (model often closes
    //    args with `'` even though the opener was `"`) and any intervening
    //    whitespace. We accept either:
    //      a) `/>` — self-closing.
    //      b) `>` followed (possibly across whitespace) by `</tool*>` —
    //         empty body, explicit close marker.
    var k: usize = @max(name_end, args_end);
    while (k < limit and (slice[k] == ' ' or slice[k] == '\t' or slice[k] == '\n' or slice[k] == '\r' or slice[k] == '"' or slice[k] == '\'')) k += 1;
    if (k >= limit) return null;

    if (slice[k] == '/') {
        k += 1;
        if (k >= limit or slice[k] != '>') return null;
        return .{ .name = name.?, .arguments = args.?, .consumed = k + 1 };
    }
    if (slice[k] == '>') {
        var w: usize = k + 1;
        while (w < slice.len and (slice[w] == ' ' or slice[w] == '\t' or slice[w] == '\n' or slice[w] == '\r')) w += 1;
        const close_markers = [_][]const u8{
            "</tool_call>",   "</tool_calls>",
            "</tool_request>", "</tool_requests>",
            "</tool>",
        };
        for (close_markers) |m| {
            if (w + m.len <= slice.len and std.mem.eql(u8, slice[w .. w + m.len], m)) {
                return .{ .name = name.?, .arguments = args.?, .consumed = w + m.len };
            }
        }
    }
    return null;
}

fn balancedJsonObject(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var depth: i32 = 0;
    var in_string: bool = false;
    var escape: bool = false;
    var i: usize = start;
    while (i < trimmed.len) : (i += 1) {
        const c = trimmed[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            depth += 1;
        } else if (c == '}') {
            depth -= 1;
            if (depth == 0) return trimmed[start .. i + 1];
            if (depth < 0) return null; // mismatched — give up
        }
    }
    return null;
}

/// DSV4-Flash XML-element tool form: the body of a `<tool_calls>`-style
/// wrapper carries the tool name and each argument as plain child elements —
/// no JSON anywhere (captured live, 2026-06-10):
///     <tool_name>shell</tool_name>
///     <command>df -h / | grep -v "Filesystem"</command>
/// The name comes from a <tool_name>, <name>, or <tool> element; every other
/// simple `<key>text</key>` pair becomes a string argument. Strict on shape:
/// non-whitespace between elements, an unclosed element, a duplicate name
/// element, or a missing name returns null so prose-ish markup can't
/// half-execute.
/// Tag suffixes after `tool_` that mark tool RESULTS or metadata, never tool
/// names. `<tool_output>…</tool_output>` is DSV4 hallucinating the result of
/// a call it never made — mapping it onto a tool named "output" would
/// fabricate a call out of thin air. `name` guards the bare XML-element-form
/// child (`<tool_name>shell</tool_name>`) appearing outside its wrapper.
const xml_tag_reserved_names = [_][]const u8{
    "output", "outputs", "result", "results", "response", "responses", "error", "errors", "name",
};

/// XML-element-TAG tool form: `<tool_NAME><key>value</key>…</tool_NAME>`,
/// the tool name embedded in the tag itself (DSV4 training-bias family;
/// captured live 2026-06-10: `<tool_read>` / `<tool_edit>`). `slice` starts
/// at `<tool`. Returns the parsed call plus bytes consumed through the close
/// tag, or null when the shape doesn't hold (no attributes allowed, exact
/// close required, reserved names rejected, body must be all elements).
fn parseXmlElementTagToolCall(
    allocator: std.mem.Allocator,
    slice: []const u8,
) !?struct { call: ParsedToolCall, consumed: usize } {
    std.debug.assert(std.mem.startsWith(u8, slice, "<tool"));
    const name_start = "<tool_".len;
    if (slice.len <= name_start) return null;
    var name_end = name_start;
    while (name_end < slice.len and
        (std.ascii.isAlphanumeric(slice[name_end]) or slice[name_end] == '_')) : (name_end += 1)
    {}
    if (name_end == name_start) return null; // bare `<tool_>`
    if (name_end >= slice.len or slice[name_end] != '>') return null; // attributes → not this form
    const tool_name = slice[name_start..name_end];
    for (xml_tag_reserved_names) |reserved| {
        if (std.ascii.eqlIgnoreCase(tool_name, reserved)) return null;
    }
    var close_buf: [72]u8 = undefined;
    if (tool_name.len + "</tool_>".len > close_buf.len) return null;
    const close_tag = std.fmt.bufPrint(&close_buf, "</tool_{s}>", .{tool_name}) catch return null;
    const body_start = name_end + 1;
    const close_rel = std.mem.indexOf(u8, slice[body_start..], close_tag) orelse return null;
    const body = slice[body_start .. body_start + close_rel];
    const args = parseXmlElementArgsJson(allocator, body) orelse blk: {
        // JSON-args body variant of the same form, also captured live:
        // `<tool_write>\n{"path": …, "content": …}\n</tool_write>`. The whole
        // object IS the args. The trimmed body must START with `{` so prose
        // that merely contains braces never produces arguments.
        const trimmed = std.mem.trim(u8, body, " \t\n\r");
        if (trimmed.len == 0 or trimmed[0] != '{') return null;
        const json_body = balancedJsonObject(trimmed) orelse return null;
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_body, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        break :blk allocator.dupe(u8, json_body) catch return null;
    };
    const name_owned = allocator.dupe(u8, tool_name) catch {
        allocator.free(args);
        return null;
    };
    return .{
        .call = .{ .name = name_owned, .arguments = args },
        .consumed = body_start + close_rel + close_tag.len,
    };
}

/// Parse a body consisting ENTIRELY of `<key>value</key>` child elements into
/// a JSON args string. Values are kept verbatim, so nested markup (DSV4's
/// `<edits><oldText>…</oldText>…</edits>`) survives as the arg's string
/// value. Returns null unless every non-whitespace byte belongs to an element
/// and at least one element is present — a plain-text body must never
/// produce arguments.
fn parseXmlElementArgsJson(allocator: std.mem.Allocator, body: []const u8) ?[]u8 {
    var args_map: std.json.ObjectMap = .empty;
    defer args_map.deinit(allocator);
    var count: usize = 0;
    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
        if (i >= body.len) break;
        if (body[i] != '<') return null;
        const key_start = i + 1;
        var j = key_start;
        while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
        if (j == key_start or j >= body.len or body[j] != '>') return null;
        const key = body[key_start..j];
        const val_start = j + 1;
        var close_buf: [64]u8 = undefined;
        if (key.len + 3 > close_buf.len) return null;
        const close_tag = std.fmt.bufPrint(&close_buf, "</{s}>", .{key}) catch return null;
        const rel = std.mem.indexOf(u8, body[val_start..], close_tag) orelse return null;
        args_map.put(allocator, key, .{ .string = body[val_start .. val_start + rel] }) catch return null;
        count += 1;
        i = val_start + rel + close_tag.len;
    }
    if (count == 0) return null;
    return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{}) catch null;
}

/// Hy3 (Hunyuan 3) tag-format tool calls (chat_template.jinja spec):
///   <tool_calls:SFX>
///   <tool_call:SFX>NAME<tool_sep:SFX>
///   <arg_key:SFX>KEY</arg_key:SFX>
///   <arg_value:SFX>VALUE</arg_value:SFX> …
///   </tool_call:SFX> …
///   </tool_calls:SFX>
/// Only the SUFFIXED form (`<tool_call:` …) is handled here — bare
/// `<tool_call>` is Hermes JSON and stays with the generic scan. Values are
/// kept as RAW STRINGS: the template `tojson`s non-string values on the way
/// in, and the schema-driven coercion at the server chokepoint types them on
/// the way back (types come from the SCHEMA, never the value's spelling).
/// Args build through ObjectMap + Stringify — keys escaped, duplicates
/// first-wins — per the tag-format-converter class rule. Truncation
/// (max_tokens mid-call, the big-file-write class): a call whose name is
/// delimited by `<tool_sep` recovers with its CLOSED key/value pairs only;
/// partial values are never salvaged.
/// Earliest position at/after `from` where any of `needles` occurs, or null.
fn earliestIndexOfAny(text: []const u8, from: usize, needles: []const []const u8) ?usize {
    var best: ?usize = null;
    for (needles) |n| {
        if (std.mem.indexOfPos(u8, text, from, n)) |q| {
            if (best == null or q < best.?) best = q;
        }
    }
    return best;
}

/// A tool-name character per the identifier alphabet every template family's
/// names draw from. The Inkling NAME is glued directly to the invoke marker,
/// so anything outside this set (a channel marker's `>`, a payload's `}`,
/// prose punctuation) terminates the name run.
fn inklingIsNameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '.' or c == '-';
}

/// The trailing identifier run ending at `end` (exclusive). Live 2026-07-30:
/// the boundary-based prefix scan swallowed the `<|content_text|>` marker
/// (name `<|content_text|>bash`), and a dropped `<|end_message|>` between
/// calls produced `}}write` — both resolve to the bare NAME here.
fn inklingTrailingNameRun(text: []const u8, end: usize) []const u8 {
    var start = end;
    while (start > 0 and inklingIsNameChar(text[start - 1])) start -= 1;
    return text[start..end];
}

/// Inkling tool calls: `NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`
/// per call (one message each; parallel calls are consecutive messages). The
/// payload's "name" is authoritative (the prefix NAME is the same string in
/// well-formed output, and is the fallback when the payload is truncated).
/// Truncation salvage per the hard rule: recover NAME + `{}` — never partial
/// argument values.
// DeepSeek-V4 native DSML markers. `｜` is the FULLWIDTH vertical bar
// (U+FF5C, 3 bytes) — the dsml_token "｜DSML｜" appears in no other family.
const DSML_INVOKE_TAG = "<｜DSML｜invoke";
const DSML_INVOKE_CLOSE = "</｜DSML｜invoke>";
const DSML_PARAM_TAG = "<｜DSML｜parameter";
const DSML_PARAM_CLOSE = "</｜DSML｜parameter>";

/// Tolerant quoted-attribute extraction from a tag-attribute segment:
/// `name="X"` / `name='X'` / mangled unquoted `name=X` (to whitespace/`>`).
fn dsmlAttr(seg: []const u8, comptime key: []const u8) ?[]const u8 {
    const pat = key ++ "=";
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, seg, start, pat)) |at| {
        if (at > 0 and (std.ascii.isAlphanumeric(seg[at - 1]) or seg[at - 1] == '_')) {
            start = at + pat.len;
            continue;
        }
        var v = at + pat.len;
        if (v >= seg.len) return null;
        const q = seg[v];
        if (q == '"' or q == '\'') {
            v += 1;
            const end = std.mem.indexOfScalarPos(u8, seg, v, q) orelse seg.len;
            return seg[v..end];
        }
        const end = std.mem.indexOfAnyPos(u8, seg, v, " \t\n>") orelse seg.len;
        return seg[v..end];
    }
    return null;
}

/// DeepSeek-V4 DSML tool calls:
///   `<｜DSML｜tool_calls>` wrapping
///   `<｜DSML｜invoke name="X">` +
///   `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`.
/// string="true" values are raw text (JSON-escaped here); string="false"
/// values are already JSON (validated; invalid degrades to an honest string).
/// House conventions: the parser keys on the INVOKE marker (a dropped
/// wrapper or invoke-close still parses — delimiter-drop class), values run
/// to the CONFIRMED closing tag (rich values with `<`/newlines survive),
/// truncation salvages NAME + completed pairs (fragments dropped), keys are
/// escaped + deduped (last wins) so emitted arguments are ALWAYS valid JSON.
fn parseDsmlToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    const Param = struct { key: []const u8, value: []const u8, is_string: bool };
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, DSML_INVOKE_TAG)) |inv| {
        var p = inv + DSML_INVOKE_TAG.len;
        const tag_end = std.mem.indexOfScalarPos(u8, text, p, '>') orelse break;
        const name = dsmlAttr(text[p..tag_end], "name") orelse {
            pos = tag_end + 1;
            continue;
        };
        p = tag_end + 1;

        var params = std.ArrayList(Param).empty;
        defer params.deinit(allocator);
        while (true) {
            const next_param = std.mem.indexOfPos(u8, text, p, DSML_PARAM_TAG);
            const next_close = std.mem.indexOfPos(u8, text, p, DSML_INVOKE_CLOSE);
            const next_invoke = std.mem.indexOfPos(u8, text, p, DSML_INVOKE_TAG);
            const np = next_param orelse std.math.maxInt(usize);
            const nc = next_close orelse std.math.maxInt(usize);
            const ni = next_invoke orelse std.math.maxInt(usize);
            if (np == std.math.maxInt(usize) and nc == std.math.maxInt(usize) and ni == std.math.maxInt(usize)) {
                // truncated invoke: ship NAME + completed pairs
                p = text.len;
                break;
            }
            if (nc <= np and nc <= ni) {
                p = nc + DSML_INVOKE_CLOSE.len;
                break;
            }
            if (ni < np) {
                // dropped invoke-close — the next call starts here; the outer
                // loop picks it up from `p`.
                p = ni;
                break;
            }
            // parameter
            var q = np + DSML_PARAM_TAG.len;
            const ptag_end = std.mem.indexOfScalarPos(u8, text, q, '>') orelse {
                p = text.len;
                break;
            };
            const attrs = text[q..ptag_end];
            const pkey = dsmlAttr(attrs, "name");
            const pstring = dsmlAttr(attrs, "string");
            const vstart = ptag_end + 1;
            const vclose = std.mem.indexOfPos(u8, text, vstart, DSML_PARAM_CLOSE) orelse {
                // unterminated value = server-cut fragment; drop it, end scan
                p = text.len;
                break;
            };
            q = vclose + DSML_PARAM_CLOSE.len;
            if (pkey) |k| {
                const value = text[vstart..vclose];
                // string attr absent (mangled opener): treat JSON-parseable
                // values as JSON, everything else as a string.
                const is_string = if (pstring) |sf|
                    !std.mem.eql(u8, sf, "false")
                else
                    !dsmlValueIsJson(allocator, value);
                // dedup: last wins
                var replaced = false;
                for (params.items) |*prm| {
                    if (std.mem.eql(u8, prm.key, k)) {
                        prm.value = value;
                        prm.is_string = is_string;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) try params.append(allocator, .{ .key = k, .value = value, .is_string = is_string });
            }
            p = q;
        }

        // Build the arguments object: escaped keys, escaped string values,
        // raw (validated) JSON values.
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);
        try out.append(allocator, '{');
        for (params.items, 0..) |prm, i| {
            if (i > 0) try out.append(allocator, ',');
            try appendJsonString(allocator, &out, prm.key);
            try out.append(allocator, ':');
            if (!prm.is_string and dsmlValueIsJson(allocator, prm.value)) {
                try out.appendSlice(allocator, std.mem.trim(u8, prm.value, " \t\n\r"));
            } else {
                try appendJsonString(allocator, &out, prm.value);
            }
        }
        try out.append(allocator, '}');
        const name_owned = try allocator.dupe(u8, name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = try out.toOwnedSlice(allocator) });
        pos = p;
    }
}

fn dsmlValueIsJson(allocator: std.mem.Allocator, value: []const u8) bool {
    const trimmed = std.mem.trim(u8, value, " \t\n\r");
    if (trimmed.len == 0) return false;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch return false;
    parsed.deinit();
    return true;
}

fn parseInklingToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, INKLING_INVOKE_TAG)) |inv| {
        const prefix_name = inklingTrailingNameRun(text, inv);

        // Body = the balanced JSON object right after the marker (string-aware
        // brace scan). The next <|end_message|> is NOT a reliable terminator:
        // the model drops it between back-to-back calls (live 2026-07-30,
        // `{…}}write<|content_invoke_tool_json|>{…}`), and an end-tag-bounded
        // body then swallows the following call whole. No balance by end of
        // text = truncation → the NAME + `{}` salvage below.
        const body_start = inv + INKLING_INVOKE_TAG.len;
        var body: []const u8 = "";
        const after = std.mem.trimStart(u8, text[body_start..], "\n ");
        if (std.mem.startsWith(u8, after, "{")) {
            if (balancedJsonObject(after)) |obj| {
                body = obj;
                pos = (body_start + (text.len - body_start - after.len)) + obj.len;
            } else {
                pos = text.len; // truncated object — nothing further to scan
            }
        } else {
            // No object after the marker at all: keep the old end-tag-bounded
            // read so a non-JSON payload still gets a parse attempt.
            const body_end = std.mem.indexOfPos(u8, text, body_start, INKLING_END_TAG) orelse text.len;
            body = std.mem.trim(u8, text[body_start..body_end], "\n ");
            pos = body_end;
        }

        var name: []const u8 = prefix_name;
        var args_json: ?[]const u8 = null; // allocated when set
        if (std.json.parseFromSlice(std.json.Value, allocator, body, .{})) |parsed| {
            defer parsed.deinit();
            if (parsed.value == .object) {
                if (parsed.value.object.get("name")) |nv| {
                    if (nv == .string and nv.string.len > 0) {
                        name = nv.string;
                        // Marker-echo repair: pi's "Tool <|content_text|>bash
                        // not found" reply taught the model to copy the garbage
                        // name INTO its payloads (live 2026-07-30). No real
                        // tool name contains `<|` — keep the trailing
                        // identifier run; empty → fall back to the prefix name.
                        if (std.mem.indexOf(u8, name, "<|") != null) {
                            name = inklingTrailingNameRun(name, name.len);
                            if (name.len == 0) name = prefix_name;
                        }
                    }
                }
                if (parsed.value.object.get("args")) |av| {
                    if (av == .object) {
                        // Re-serialize through Stringify: escaping + key dedup
                        // by construction (the tag-converter invariant).
                        args_json = std.json.Stringify.valueAlloc(allocator, av, .{}) catch null;
                    }
                }
                // `name` borrows from `parsed` — dupe before deinit.
                name = try allocator.dupe(u8, name);
            } else {
                name = try allocator.dupe(u8, name);
            }
        } else |_| {
            // Truncated / mangled payload: NAME + {} (never ship fragments).
            name = try allocator.dupe(u8, name);
        }
        errdefer allocator.free(name);
        if (name.len == 0) {
            allocator.free(name);
            if (args_json) |a| allocator.free(a);
            continue;
        }
        const arguments = args_json orelse try allocator.dupe(u8, "{}");
        try calls.append(allocator, .{ .name = name, .arguments = arguments, .inferred = false });
    }
}

// ── ATEM tool calls (Muse-Glimmer / `muse_glimmer`) ────────────────────
//
//   <atem:function_calls>
//   <atem:invoke name="weather.get">
//   <atem:parameter name="city">Paris</atem:parameter>
//   <atem:parameter name="detailed">false</atem:parameter>
//   </atem:invoke>
//   </atem:function_calls>
//
// String parameters are RAW bytes (the reference explicitly does not strip
// spaces — the Hermes trim gap stays Hermes'); lists/objects/bools/null/
// numbers are spelled as JSON and keep their spelled type (the schema
// coercion chokepoint still gets the final word). The parser keys on the
// INVOKE tag so a dropped wrapper still parses (delimiter-drop class), and
// truncation salvages NAME + completed parameters — never fragment values.
const ATEM_INVOKE_TAG = "<atem:invoke";
const ATEM_PARAM_TAG = "<atem:parameter";
const ATEM_PARAM_CLOSE = "</atem:parameter>";
const ATEM_INVOKE_CLOSE = "</atem:invoke>";

fn parseAtemToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, ATEM_INVOKE_TAG)) |inv| {
        const after_tag = inv + ATEM_INVOKE_TAG.len;
        const seg = text[after_tag..];
        // Attribute scan bounded so a runaway unclosed quote can't eat the
        // rest of the output.
        const raw_name = dsmlAttr(seg[0..@min(seg.len, 512)], "name") orelse {
            pos = after_tag;
            continue;
        };
        // No real tool name carries a channel marker (corpus invariant —
        // the inkling marker-echo class).
        var name_slice = raw_name;
        if (std.mem.indexOf(u8, name_slice, "<|") != null) {
            name_slice = inklingTrailingNameRun(name_slice, name_slice.len);
        }
        if (name_slice.len == 0) {
            pos = after_tag;
            continue;
        }

        var args_json: ?[]const u8 = null;
        if (std.mem.indexOfScalar(u8, seg, '>')) |te| {
            const body_all = seg[te + 1 ..];
            const body_end = std.mem.indexOf(u8, body_all, ATEM_INVOKE_CLOSE) orelse body_all.len;
            args_json = try atemParamsToJson(allocator, body_all[0..body_end]);
            pos = if (body_end < body_all.len)
                after_tag + te + 1 + body_end + ATEM_INVOKE_CLOSE.len
            else
                text.len; // truncated body — completed params already salvaged
        } else {
            pos = text.len; // truncated inside the opening tag: NAME + {}
        }

        const name = try allocator.dupe(u8, name_slice);
        errdefer allocator.free(name);
        const arguments = args_json orelse try allocator.dupe(u8, "{}");
        try calls.append(allocator, .{ .name = name, .arguments = arguments, .inferred = false });
    }
}

/// Assemble `<atem:parameter name="k">v</atem:parameter>` runs into a JSON
/// object string. A value that parses as COMPLETE JSON keeps its spelled type;
/// everything else is a raw string, exact bytes. Keys are escaped + deduped
/// (last wins) via ObjectMap+Stringify, so emitted arguments are ALWAYS valid
/// JSON. Returns null when no complete parameter was found.
fn atemParamsToJson(allocator: std.mem.Allocator, body: []const u8) !?[]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var map: std.json.ObjectMap = .empty;
    var pos: usize = 0;
    var found = false;
    while (std.mem.indexOfPos(u8, body, pos, ATEM_PARAM_TAG)) |p| {
        const after = p + ATEM_PARAM_TAG.len;
        const seg = body[after..];
        const te = std.mem.indexOfScalar(u8, seg, '>') orelse break; // truncated open tag
        const pname = dsmlAttr(seg[0 .. te + 1], "name") orelse {
            pos = after;
            continue;
        };
        const vstart = after + te + 1;
        // The value runs to the CONFIRMED closing tag — rich values with
        // newlines/quotes/`<` survive; no close by end of text = fragment,
        // dropped.
        const vend_rel = std.mem.indexOf(u8, body[vstart..], ATEM_PARAM_CLOSE) orelse break;
        const raw = body[vstart .. vstart + vend_rel];
        const v: std.json.Value = std.json.parseFromSliceLeaky(std.json.Value, aa, raw, .{}) catch
            std.json.Value{ .string = raw };
        try map.put(aa, pname, v);
        found = true;
        pos = vstart + vend_rel + ATEM_PARAM_CLOSE.len;
    }
    if (!found) return null;
    return try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = map }, .{});
}

// ── Pythonic tool calls (LFM2.5 / `lfm2`) ──────────────────────────────
//
//   <|tool_call_start|>[get_weather(city='Paris', days=3, metric=True,
//                                   tags=['trip', 'eu'])]<|tool_call_end|>
//
// A CALL-EXPRESSION grammar, not a JSON one: values are Python literals. The
// model emits full repr — `True`, single-quoted lists — even though its own
// template renders history containers via `tojson`, so a JSON-only value
// reader mis-reads the two most common non-string types. Parsing the literals
// here (rather than shipping them as text for `coerceToolArgsToSchema` to
// rescue) is what keeps a declared `array`/`boolean` parameter from arriving
// as a string — the value types are knowable from the SPELLING in this
// grammar, so the schema never has to be consulted.
//
// Marker-gated: `<|tool_call_start|>` is emitted by no other family, and the
// generic `<tool` scan never sees it, so this arm cannot touch existing
// traffic. The Gemma 4 arm keys on the exact `<|tool_call>` (with `>`), which
// this never matches.
const PYTHONIC_CALL_START = "<|tool_call_start|>";
const PYTHONIC_CALL_END = "<|tool_call_end|>";

fn pythonicIsNameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '.' or c == '-';
}

/// Index of the first byte in `s` at nesting depth 0 that satisfies `pred`,
/// skipping anything inside a quoted string or a nested ()/[]/{} group.
/// One scanner for every structural search in this grammar — the arg
/// separator, the `=`, the dict `:`, and the closing paren all need the same
/// blindness to separators that live inside a value.
fn pythonicScan(s: []const u8, start: usize, pred: *const fn (u8, i32) bool) ?usize {
    var depth: i32 = 0;
    var quote: u8 = 0;
    var i = start;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        if (quote != 0) {
            if (c == '\\') {
                i += 1;
            } else if (c == quote) {
                quote = 0;
            }
            continue;
        }
        if (c == '\'' or c == '"') {
            quote = c;
            continue;
        }
        if (c == '(' or c == '[' or c == '{') {
            depth += 1;
        } else if (c == ')' or c == ']' or c == '}') {
            if (pred(c, depth)) return i;
            depth -= 1;
            continue;
        }
        if (pred(c, depth)) return i;
    }
    return null;
}

fn pythonicIsCloser(c: u8, depth: i32) bool {
    return c == ')' and depth == 0;
}
fn pythonicIsComma(c: u8, depth: i32) bool {
    return c == ',' and depth == 0;
}
fn pythonicIsEquals(c: u8, depth: i32) bool {
    return c == '=' and depth == 0;
}
fn pythonicIsColon(c: u8, depth: i32) bool {
    return c == ':' and depth == 0;
}

/// Unescape a Python string literal body. Unknown escapes keep the backslash,
/// which is Python's own behavior (`'\d' == '\\d'`) and the forgiving reading
/// of a model that under-escaped a path.
fn pythonicUnescape(arena: std.mem.Allocator, body: []const u8) ![]const u8 {
    var out = std.ArrayList(u8).empty;
    var i: usize = 0;
    while (i < body.len) : (i += 1) {
        if (body[i] != '\\' or i + 1 >= body.len) {
            try out.append(arena, body[i]);
            continue;
        }
        i += 1;
        switch (body[i]) {
            'n' => try out.append(arena, '\n'),
            'r' => try out.append(arena, '\r'),
            't' => try out.append(arena, '\t'),
            '\\', '\'', '"' => try out.append(arena, body[i]),
            else => {
                try out.append(arena, '\\');
                try out.append(arena, body[i]);
            },
        }
    }
    return out.toOwnedSlice(arena);
}

/// A Python literal → its JSON value. Anything undecidable stays a string
/// VERBATIM rather than being guessed at, so a client's own validation error
/// stays honest (the `coerceToolArgsToSchema` contract, one layer down).
fn pythonicLiteral(arena: std.mem.Allocator, raw: []const u8) !std.json.Value {
    const s = std.mem.trim(u8, raw, " \t\n\r");
    if (s.len == 0) return .{ .string = "" };

    if ((s[0] == '\'' or s[0] == '"') and s.len >= 2 and s[s.len - 1] == s[0]) {
        return .{ .string = try pythonicUnescape(arena, s[1 .. s.len - 1]) };
    }
    if (std.mem.eql(u8, s, "True")) return .{ .bool = true };
    if (std.mem.eql(u8, s, "False")) return .{ .bool = false };
    if (std.mem.eql(u8, s, "None")) return .null;

    if (s[0] == '[' and s[s.len - 1] == ']') {
        var arr = std.json.Array.init(arena);
        var it = PythonicParts{ .s = s[1 .. s.len - 1] };
        while (it.next()) |part| {
            if (std.mem.trim(u8, part, " \t\n\r").len == 0) continue;
            try arr.append(try pythonicLiteral(arena, part));
        }
        return .{ .array = arr };
    }
    if (s[0] == '{' and s[s.len - 1] == '}') {
        var obj: std.json.ObjectMap = .empty;
        var it = PythonicParts{ .s = s[1 .. s.len - 1] };
        while (it.next()) |part| {
            const colon = pythonicScan(part, 0, pythonicIsColon) orelse continue;
            const key_v = try pythonicLiteral(arena, part[0..colon]);
            const key = switch (key_v) {
                .string => |k| k,
                else => continue,
            };
            if (key.len == 0 or obj.getEntry(key) != null) continue;
            try obj.put(arena, key, try pythonicLiteral(arena, part[colon + 1 ..]));
        }
        return .{ .object = obj };
    }

    if (std.fmt.parseInt(i64, s, 10)) |n| return .{ .integer = n } else |_| {}
    if (std.fmt.parseFloat(f64, s)) |f| return .{ .float = f } else |_| {}
    return .{ .string = s };
}

/// Iterator over top-level comma-separated parts.
const PythonicParts = struct {
    s: []const u8,
    i: usize = 0,

    fn next(self: *PythonicParts) ?[]const u8 {
        if (self.i > self.s.len) return null;
        if (self.i == self.s.len) {
            self.i += 1;
            return self.s[self.s.len..];
        }
        const start = self.i;
        if (pythonicScan(self.s, start, pythonicIsComma)) |c| {
            self.i = c + 1;
            return self.s[start..c];
        }
        self.i = self.s.len + 1;
        return self.s[start..];
    }
};

fn parsePythonicToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, PYTHONIC_CALL_START)) |p| {
        const body_start = p + PYTHONIC_CALL_START.len;
        const close = std.mem.indexOfPos(u8, text, body_start, PYTHONIC_CALL_END);
        pos = if (close) |c| c + PYTHONIC_CALL_END.len else text.len;

        // A dropped end marker (truncation, or the weak-model delimiter-drop
        // class) must not cost the call — the body simply runs to the end.
        var body = std.mem.trim(u8, text[body_start .. close orelse text.len], " \t\n\r");
        if (body.len > 0 and body[0] == '[') body = body[1..];
        if (body.len > 0 and body[body.len - 1] == ']') body = body[0 .. body.len - 1];

        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var i: usize = 0;
        while (i < body.len) {
            while (i < body.len and (std.ascii.isWhitespace(body[i]) or body[i] == ',')) i += 1;
            const name_start = i;
            while (i < body.len and pythonicIsNameChar(body[i])) i += 1;
            const name = body[name_start..i];
            if (name.len == 0) break;

            while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
            // No `(`, or no matching `)`: the cut landed inside the call. Ship
            // the NAME with EMPTY args — never a partial value (a half-written
            // file body is worse than a re-issued call).
            const args = blk: {
                if (i >= body.len or body[i] != '(') break :blk null;
                const arg_start = i + 1;
                const arg_end = pythonicScan(body, arg_start, pythonicIsCloser) orelse break :blk null;
                i = arg_end + 1;
                break :blk body[arg_start..arg_end];
            };

            var args_map: std.json.ObjectMap = .empty;
            if (args) |a| {
                var parts = PythonicParts{ .s = a };
                while (parts.next()) |part| {
                    const eq = pythonicScan(part, 0, pythonicIsEquals) orelse continue;
                    const key = std.mem.trim(u8, part[0..eq], " \t\n\r");
                    if (key.len == 0 or args_map.getEntry(key) != null) continue;
                    try args_map.put(arena, key, try pythonicLiteral(arena, part[eq + 1 ..]));
                }
            }

            const args_str = try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{});
            errdefer allocator.free(args_str);
            const name_owned = try allocator.dupe(u8, name);
            errdefer allocator.free(name_owned);
            try calls.append(allocator, .{ .name = name_owned, .arguments = args_str });

            if (args == null) break;
        }
    }
}

fn parseHy3ToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, "<tool_call")) |p| {
        const after_base = p + "<tool_call".len;
        var name_start: usize = undefined;
        if (after_base < text.len and text[after_base] == ':') {
            // Canonical singular per-call opener `<tool_call:sfx>`.
            const open_len = suffixedTagLenAt(text[p..], "<tool_call") orelse {
                pos = after_base;
                continue;
            };
            name_start = p + open_len;
        } else if (std.mem.startsWith(u8, text[p..], "<tool_calls:")) {
            // Suffixed plural WRAPPER `<tool_calls:sfx>` (hy3 only — the BARE
            // DSV4/generic `<tool_calls>` wrapper is a different format and must
            // fall through to its own parser). Normally an inner singular
            // `<tool_call:sfx>` opener follows — defer to it (the next loop
            // iteration parses that). A weak model (Hy3-REAP62, live 2026-07-16
            // raw capture) DROPS the singular opener and jumps straight to the
            // NAME, leaking the whole (otherwise well-formed) call; recover by
            // treating the wrapper's end as the opener. Same weak-model
            // delimiter-drop class as the <tool_sep> drop below, one delimiter over.
            const wrap_len = suffixedTagLenAt(text[p..], "<tool_calls") orelse {
                pos = after_base;
                continue;
            };
            var probe = p + wrap_len;
            while (probe < text.len and std.ascii.isWhitespace(text[probe])) probe += 1;
            const inner_singular = std.mem.startsWith(u8, text[probe..], "<tool_call") and
                probe + "<tool_call".len < text.len and text[probe + "<tool_call".len] == ':';
            if (inner_singular) {
                pos = p + wrap_len;
                continue;
            }
            name_start = p + wrap_len;
        } else if (after_base < text.len and text[after_base] == '>' and blk: {
            // Bare `<tool_call>` has two sub-formats sharing this opener:
            //   • Hermes JSON `<tool_call>{…}` / function-tag `<tool_call><function=…>`
            //     → the generic scan reads those; fall through.
            //   • GLM (Laguna, tokenizer tool_parser_type "glm47"):
            //     `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…`
            //     — bare opener, NAME, then arg_key/arg_value pairs, NO plural
            //     wrapper. Route HERE only when an `<arg_key` precedes this
            //     call's close (the unambiguous GLM signal a JSON/function body
            //     never has), so the Hermes path is untouched.
            const body = after_base + 1;
            const this_close = std.mem.indexOfPos(u8, text, body, "</tool_call") orelse text.len;
            const ak_at = std.mem.indexOfPos(u8, text, body, "<arg_key") orelse text.len;
            break :blk ak_at < this_close;
        }) {
            name_start = after_base + 1;
        } else {
            // Bare `<tool_call>` Hermes JSON / function-tag — not this format.
            pos = after_base;
            continue;
        }

        // NAME runs to the first structural marker after the opener. Canonically
        // that's <tool_sep:sfx>; a mangled call (weak model dropped <tool_sep> —
        // live 2026-07-16, Hy3-REAP62 via pi) instead closes the name with
        // </arg_value>/<arg_key>/</tool_call>, so accept any of them. No marker
        // at all → the cut happened inside the name; nothing to recover.
        const name_end = earliestIndexOfAny(text, name_start, &.{
            "<tool_sep", "<arg_key", "<arg_value", "</tool_call", "</arg_key", "</arg_value",
        }) orelse {
            pos = name_start;
            continue;
        };
        const name = std.mem.trim(u8, text[name_start..name_end], " \t\n\r");
        if (name.len == 0) {
            pos = name_start;
            continue;
        }

        var args_map: std.json.ObjectMap = .empty;
        defer args_map.deinit(allocator);

        // Consume a valid <tool_sep:sfx> if present; a mangled call that dropped
        // it (weak model — live REAP capture) starts the arg scan right at the
        // name's (wrong) close tag instead.
        var i: usize = name_end;
        if (suffixedTagLenAt(text[name_end..], "<tool_sep")) |sep_len| {
            i = name_end + sep_len;
        }
        // Parse <arg_key>/<arg_value> pairs, bounded by this call's </tool_call>.
        // SCAN to the next <arg_key> (rather than requiring it right here) so a
        // stray name-close tag between the name and the first key is skipped, and
        // match the KEY block's close TOLERANTLY (</arg_key> OR </arg_value>) —
        // REAP closes it with </arg_value> (live 2026-07-16 raw capture: bash /
        // command / "ls -la"). The well-formed path is unchanged: its <tool_sep>
        // is consumed above, <arg_key> is found immediately, and </arg_key> is the
        // earliest close.
        const call_end = std.mem.indexOfPos(u8, text, i, "</tool_call") orelse text.len;
        while (true) {
            const ak_at = std.mem.indexOfPos(u8, text, i, "<arg_key") orelse break;
            if (ak_at >= call_end) break;
            const ak_len = suffixedTagLenAt(text[ak_at..], "<arg_key") orelse {
                i = ak_at + "<arg_key".len;
                continue;
            };
            const key_start = ak_at + ak_len;
            const ak_close = earliestIndexOfAny(text, key_start, &.{ "</arg_key", "</arg_value" }) orelse break;
            const ak_close_len = suffixedTagLenAt(text[ak_close..], "</arg_key") orelse
                suffixedTagLenAt(text[ak_close..], "</arg_value") orelse break;
            const key = std.mem.trim(u8, text[key_start..ak_close], " \t\n\r");
            i = ak_close + ak_close_len;
            while (i < text.len and std.ascii.isWhitespace(text[i])) i += 1;
            const av_len = suffixedTagLenAt(text[i..], "<arg_value") orelse break;
            const val_start = i + av_len;
            const av_close = std.mem.indexOfPos(u8, text, val_start, "</arg_value") orelse break;
            const av_close_len = suffixedTagLenAt(text[av_close..], "</arg_value") orelse break;
            const value = text[val_start..av_close];
            i = av_close + av_close_len;
            if (key.len > 0 and args_map.getEntry(key) == null) {
                try args_map.put(allocator, key, .{ .string = value });
            }
        }

        const args_str = try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{});
        errdefer allocator.free(args_str);
        const name_owned = try allocator.dupe(u8, name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = args_str });

        if (std.mem.indexOfPos(u8, text, i, "</tool_call")) |cl| {
            const cl_len = suffixedTagLenAt(text[cl..], "</tool_call") orelse 0;
            pos = if (cl_len > 0) cl + cl_len else cl + "</tool_call".len;
        } else {
            pos = i;
        }
    }
}

fn parseXmlElementToolCall(allocator: std.mem.Allocator, body: []const u8) ?ParsedToolCall {
    var name: ?[]const u8 = null;
    var args_map: std.json.ObjectMap = .empty;
    defer args_map.deinit(allocator);

    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
        if (i >= body.len) break;
        if (body[i] != '<') return null;
        const key_start = i + 1;
        var j = key_start;
        while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
        if (j == key_start or j >= body.len or body[j] != '>') return null;
        const key = body[key_start..j];
        const val_start = j + 1;
        var close_buf: [64]u8 = undefined;
        if (key.len + 3 > close_buf.len) return null;
        const close_tag = std.fmt.bufPrint(&close_buf, "</{s}>", .{key}) catch return null;
        const rel = std.mem.indexOf(u8, body[val_start..], close_tag) orelse return null;
        const value = body[val_start .. val_start + rel];
        i = val_start + rel + close_tag.len;

        if (std.mem.eql(u8, key, "tool_name") or std.mem.eql(u8, key, "name") or std.mem.eql(u8, key, "tool")) {
            if (name != null) return null;
            const trimmed_name = std.mem.trim(u8, value, " \t\n\r");
            if (trimmed_name.len == 0) return null;
            name = trimmed_name;
        } else {
            args_map.put(allocator, key, .{ .string = value }) catch return null;
        }
    }

    const n = name orelse return null;
    const args_str = std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{}) catch return null;
    const name_owned = allocator.dupe(u8, n) catch {
        allocator.free(args_str);
        return null;
    };
    return .{ .name = name_owned, .arguments = args_str };
}

/// Repair a JSON object whose trailing closer(s) were dropped — e.g. the
/// model wrote `{"name":"edit","arguments":{…,"edits":[{…}]}` and went
/// straight to its close tag, one `}` short. String/escape-aware bracket
/// stack; appends exactly the missing closers in nesting order. Returns null
/// when the text doesn't start with `{`, is already balanced (trailing
/// garbage is someone else's problem), ends mid-string, or nests deeper
/// than the stack (not worth guessing).
fn completeUnbalancedJsonObject(allocator: std.mem.Allocator, content: []const u8) ?[]u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    if (trimmed.len == 0 or trimmed[0] != '{') return null;
    var stack: [16]u8 = undefined;
    var depth: usize = 0;
    var in_string = false;
    var escape = false;
    for (trimmed) |c| {
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '{' => {
                if (depth == stack.len) return null;
                stack[depth] = '}';
                depth += 1;
            },
            '[' => {
                if (depth == stack.len) return null;
                stack[depth] = ']';
                depth += 1;
            },
            '}', ']' => {
                if (depth == 0 or stack[depth - 1] != c) return null;
                depth -= 1;
                // Balanced before the end → trailing garbage, not a
                // truncated tail. balancedJsonObject owns that case.
                if (depth == 0) return null;
            },
            else => {},
        }
    }
    if (in_string or depth == 0) return null;
    var out = allocator.alloc(u8, trimmed.len + depth) catch return null;
    @memcpy(out[0..trimmed.len], trimmed);
    var i: usize = 0;
    while (i < depth) : (i += 1) {
        out[trimmed.len + i] = stack[depth - 1 - i];
    }
    return out;
}

/// Snap a balanced JSON array starting at the first `[`. Mirrors
/// balancedJsonObject: string/escape aware, tracks BOTH bracket kinds so
/// objects nested in the array can't fool the depth count.
fn balancedJsonArray(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '[') orelse return null;
    var depth: i32 = 0;
    var in_string: bool = false;
    var escape: bool = false;
    var i: usize = start;
    while (i < trimmed.len) : (i += 1) {
        const c = trimmed[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '[', '{' => depth += 1,
            ']', '}' => {
                depth -= 1;
                if (depth == 0) return trimmed[start .. i + 1];
                if (depth < 0) return null; // mismatched — give up
            },
            else => {},
        }
    }
    return null;
}

/// Parse a JSON array of tool-call objects (`[{"name":…,"arguments":…}, …]`)
/// and append EVERY call. All-or-nothing: if any element fails to parse as a
/// tool call, nothing is appended — a prose-ish array must pass through as
/// text rather than half-execute.
fn appendJsonToolCallArray(
    allocator: std.mem.Allocator,
    arr_text: []const u8,
    calls: *std.ArrayList(ParsedToolCall),
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, arr_text, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .array) return;
    const items = parsed.value.array.items;
    if (items.len == 0) return;

    var pending = std.ArrayList(ParsedToolCall).empty;
    var ok = true;
    defer {
        for (pending.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        pending.deinit(allocator);
    }
    for (items) |item| {
        if (item != .object) {
            ok = false;
            break;
        }
        // Reuse tryParseJsonToolCall (leaf-name walk, flat-shape synthesis)
        // by round-tripping the element through its text form.
        const item_text = std.json.Stringify.valueAlloc(allocator, item, .{}) catch {
            ok = false;
            break;
        };
        defer allocator.free(item_text);
        const tc = tryParseJsonToolCall(allocator, item_text) orelse {
            ok = false;
            break;
        };
        try pending.append(allocator, tc);
    }
    if (!ok) return;
    try calls.appendSlice(allocator, pending.items);
    pending.clearRetainingCapacity();
}

/// Lenient recovery for tool-call argument JSON that small models mangle when
/// emitting large string values (file contents). Two failure modes, both of
/// which make strict std.json reject the WHOLE blob so the call would be dropped
/// and leak as visible text:
///   • raw control bytes (literal newlines/tabs) inside a string instead of
///     `\n`/`\t` — the dominant big-file failure;
///   • unescaped inner double-quotes (`<meta charset="UTF-8">`) and invalid
///     backslash escapes (Windows paths, regex) inside a string.
/// Re-serializes a CLEAN object by walking the input with a position-aware
/// tolerant parser (key vs value context): control bytes are re-escaped, lone/
/// invalid backslashes are escaped, and a `"` closes a string only at a
/// structural delimiter (`:` after a key; `,`/`}`/`]`/end after a value) — any
/// other `"` is an inner content quote and gets escaped. The CALLER strict-parses
/// the result, so a mis-recovery that yields invalid JSON is discarded; the only
/// residual risk is a value string closed early on pathological content (a
/// literal `"}` / `",` byte-sequence inside the file), which still beats dropping
/// the call. Does NOT tolerate truncation (no auto-close of open containers) —
/// that stays with completeUnbalancedJsonObject. Returns an allocator-owned
/// normalized JSON string, or null when no well-formed leading object recovers.
fn looseRepairToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const obj_start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    _ = looseEmitObject(allocator, &out, trimmed, obj_start, 0) catch {
        out.deinit(allocator);
        return null;
    };
    return out.toOwnedSlice(allocator) catch null;
}

/// Tolerant re-serialize of a top-level JSON CONTAINER (`[…]` or `{…}`) — the
/// array-aware sibling of looseRepairToolCallJson. Used by the schema coercion to
/// salvage a container-typed argument the model mangled (e.g. a missing comma
/// inside an `edits` array). Same discipline: the caller strict-parses the
/// result, so a mis-repair yielding invalid JSON is discarded. Returns null when
/// the text doesn't start with `[`/`{` or can't be re-serialized.
fn looseRepairContainer(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0 or (trimmed[0] != '[' and trimmed[0] != '{')) return null;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    _ = looseEmitValue(allocator, &out, trimmed, 0, 0, .object_value) catch {
        out.deinit(allocator);
        return null;
    };
    return out.toOwnedSlice(allocator) catch null;
}

const LooseStringCtx = enum { key, object_value, array_value };
const LooseError = error{ Malformed, OutOfMemory };
const loose_max_depth = 32;

fn looseSkipWs(body: []const u8, start: usize) usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ' ', '\t', '\n', '\r' => {},
            else => break,
        }
    }
    return i;
}

fn looseSkipWsCommas(body: []const u8, start: usize) usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ' ', '\t', '\n', '\r', ',' => {},
            else => break,
        }
    }
    return i;
}

fn looseIsHex4(s: []const u8) bool {
    if (s.len != 4) return false;
    for (s) |c| {
        const ok = (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
        if (!ok) return false;
    }
    return true;
}

fn looseEmitObject(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
) LooseError!usize {
    if (depth > loose_max_depth) return error.Malformed;
    var i = looseSkipWs(body, start);
    if (i >= body.len or body[i] != '{') return error.Malformed;
    i += 1;
    try out.append(allocator, '{');
    var first = true;
    while (true) {
        i = looseSkipWsCommas(body, i);
        if (i >= body.len) return error.Malformed; // unterminated (not truncation-tolerant)
        if (body[i] == '}') {
            try out.append(allocator, '}');
            return i + 1;
        }
        if (body[i] != '"') return error.Malformed; // key must be a string
        if (!first) try out.append(allocator, ',');
        first = false;
        i = try looseEmitString(allocator, out, body, i, .key);
        i = looseSkipWs(body, i);
        if (i >= body.len or body[i] != ':') return error.Malformed;
        i += 1;
        try out.append(allocator, ':');
        i = try looseEmitValue(allocator, out, body, i, depth, .object_value);
    }
}

fn looseEmitArray(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
) LooseError!usize {
    if (depth > loose_max_depth) return error.Malformed;
    var i = looseSkipWs(body, start);
    if (i >= body.len or body[i] != '[') return error.Malformed;
    i += 1;
    try out.append(allocator, '[');
    var first = true;
    while (true) {
        i = looseSkipWsCommas(body, i);
        if (i >= body.len) return error.Malformed;
        if (body[i] == ']') {
            try out.append(allocator, ']');
            return i + 1;
        }
        if (!first) try out.append(allocator, ',');
        first = false;
        i = try looseEmitValue(allocator, out, body, i, depth, .array_value);
    }
}

fn looseEmitValue(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    str_ctx: LooseStringCtx,
) LooseError!usize {
    const i = looseSkipWs(body, start);
    if (i >= body.len) return error.Malformed;
    return switch (body[i]) {
        '"' => try looseEmitString(allocator, out, body, i, str_ctx),
        '{' => try looseEmitObject(allocator, out, body, i, depth + 1),
        '[' => try looseEmitArray(allocator, out, body, i, depth + 1),
        else => try looseEmitScalar(allocator, out, body, i),
    };
}

fn looseEmitScalar(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
) LooseError!usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ',', '}', ']', ' ', '\t', '\n', '\r' => break,
            else => try out.append(allocator, body[i]),
        }
    }
    if (i == start) return error.Malformed;
    return i; // strict re-parse validates the token was a real number/bool/null
}

fn looseEmitString(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    ctx: LooseStringCtx,
) LooseError!usize {
    // body[start] == '"'
    try out.append(allocator, '"');
    var i = start + 1;
    while (i < body.len) {
        const c = body[i];
        if (c == '\\') {
            if (i + 1 >= body.len) {
                try out.appendSlice(allocator, "\\\\"); // trailing backslash → literal
                i += 1;
                continue;
            }
            const n = body[i + 1];
            switch (n) {
                '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => {
                    try out.append(allocator, '\\');
                    try out.append(allocator, n);
                    i += 2;
                },
                'u' => {
                    if (i + 6 <= body.len and looseIsHex4(body[i + 2 .. i + 6])) {
                        try out.appendSlice(allocator, body[i .. i + 6]);
                        i += 6;
                    } else {
                        try out.appendSlice(allocator, "\\\\"); // invalid \u → literal backslash
                        i += 1;
                    }
                },
                else => {
                    try out.appendSlice(allocator, "\\\\"); // invalid escape → literal backslash
                    i += 1;
                },
            }
            continue;
        }
        if (c == '"') {
            const j = looseSkipWs(body, i + 1);
            const is_close = switch (ctx) {
                .key => j < body.len and body[j] == ':',
                .object_value => j >= body.len or body[j] == ',' or body[j] == '}',
                .array_value => j >= body.len or body[j] == ',' or body[j] == ']',
            };
            if (is_close) {
                try out.append(allocator, '"');
                return i + 1;
            }
            try out.appendSlice(allocator, "\\\""); // inner content quote → escape
            i += 1;
            continue;
        }
        if (c < 0x20) {
            switch (c) {
                '\n' => try out.appendSlice(allocator, "\\n"),
                '\t' => try out.appendSlice(allocator, "\\t"),
                '\r' => try out.appendSlice(allocator, "\\r"),
                0x08 => try out.appendSlice(allocator, "\\b"),
                0x0c => try out.appendSlice(allocator, "\\f"),
                else => {
                    var buf: [6]u8 = undefined;
                    const esc = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try out.appendSlice(allocator, esc);
                },
            }
            i += 1;
            continue;
        }
        try out.append(allocator, c);
        i += 1;
    }
    // Ran off the end with no structural close — unterminated. The enclosing
    // container won't find its close either → looseEmitObject returns Malformed
    // and the whole recovery is discarded (truncation handled elsewhere). Emit
    // a closing quote so the discarded buffer stays well-formed.
    try out.append(allocator, '"');
    return i;
}

fn tryParseJsonToolCall(allocator: std.mem.Allocator, text: []const u8) ?ParsedToolCall {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch blk: {
        // Strict parse failed. Try a chain of repairs for known shapes:
        //   1. {"name":"shell", {"command":"ls"}}           — missing `"arguments":` key entirely (Qwen MoE)
        //   2. {"name":"shell", arguments":{"command":..}}   — missing OPENING quote on `arguments` (Qwen MoE)
        //   3. {"name":"edit", "arguments": {…}]}            — truncated tail, final closer(s) dropped
        //      before the close tag (DSV4-Flash, captured live)
        // Repairs are cheap and run only on the parse-failure path.
        if (repairBrokenToolCallJson(allocator, text)) |repaired| {
            defer allocator.free(repaired);
            if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
                break :blk reparsed;
            } else |_| {}
        }
        //   4. Mangled big-file content: raw control bytes (literal newlines/
        //      tabs) and unescaped inner quotes inside a string value — what
        //      small models emit when writing a large file in one shot. The
        //      re-escaped copy is re-validated by the strict parse below, so a
        //      mis-recovery that yields invalid JSON is silently discarded.
        if (looseRepairToolCallJson(allocator, text)) |repaired| {
            defer allocator.free(repaired);
            if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
                log.info("  [tool-parse] loose-repair recovered mangled tool-call JSON (raw control bytes / unescaped quotes)\n", .{});
                break :blk reparsed;
            } else |_| {}
        }
        const completed = completeUnbalancedJsonObject(allocator, text) orelse return null;
        defer allocator.free(completed);
        const reparsed = std.json.parseFromSlice(std.json.Value, allocator, completed, .{}) catch return null;
        break :blk reparsed;
    };
    defer parsed.deinit();

    if (parsed.value != .object) return null;
    var obj = parsed.value.object;

    // Qwen 3.5/3.6 MoE sometimes emits nested-name garbage like
    //   {"name":{"name":{"name":"write","arguments":{...}}}}
    // Walk down through up to a few levels of nested "name" objects to find the
    // leaf object that has a string name + arguments/parameters. Observed in the
    // wild with pi + Qwen3.6-35B in non-thinking mode.
    {
        var depth: u8 = 0;
        while (depth < 4) : (depth += 1) {
            const nv = obj.get("name") orelse return null;
            switch (nv) {
                .string => break,
                .object => |inner| obj = inner,
                else => return null,
            }
        }
    }

    const name_val = obj.get("name") orelse return null;
    if (name_val != .string) return null;

    const args_str: []const u8 = blk: {
        if (obj.get("arguments") orelse obj.get("parameters")) |args_val| {
            break :blk switch (args_val) {
                .object => std.json.Stringify.valueAlloc(allocator, args_val, .{}) catch return null,
                .string => |s| allocator.dupe(u8, s) catch return null,
                else => return null,
            };
        }
        // Flat shape (e.g. Qwen MoE): {"name":"shell","command":"ls"} —
        // parameters live at top level. Synthesize an arguments object from
        // every non-metadata key.
        var flat_map: std.json.ObjectMap = .empty;
        defer flat_map.deinit(allocator);
        var it = obj.iterator();
        while (it.next()) |entry| {
            const k = entry.key_ptr.*;
            if (std.mem.eql(u8, k, "name") or
                std.mem.eql(u8, k, "id") or
                std.mem.eql(u8, k, "type"))
            {
                continue;
            }
            flat_map.put(allocator, k, entry.value_ptr.*) catch return null;
        }
        if (flat_map.count() == 0) return null;
        const flat_value = std.json.Value{ .object = flat_map };
        break :blk std.json.Stringify.valueAlloc(allocator, flat_value, .{}) catch return null;
    };

    return .{
        .name = allocator.dupe(u8, name_val.string) catch {
            allocator.free(args_str);
            return null;
        },
        .arguments = args_str,
    };
}

/// Run known Qwen-MoE-broken-JSON repairs in sequence; return the first
/// successfully-repaired string (allocator-owned), or null if none match.
fn repairBrokenToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    if (repairFlatBraceToolCallJson(allocator, text)) |s| return s;
    if (repairUnquotedArgsKey(allocator, text)) |s| return s;
    return null;
}

/// Repair `{"name":"x", arguments":{...}}` (missing OPENING quote on
/// `arguments`/`parameters`) by injecting the quote.
fn repairUnquotedArgsKey(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    // Look for the unquoted-key pattern. Both `arguments` and `parameters`
    // have been observed; both are short fixed strings, so direct search is fine.
    const candidates = [_][]const u8{ ", arguments\":", ",arguments\":", ", parameters\":", ",parameters\":" };
    for (candidates) |needle| {
        if (std.mem.indexOf(u8, trimmed, needle)) |at| {
            const insert_at = at + 1; // right after the comma
            // Skip leading whitespace after the comma so the injected `"` lands on the identifier.
            var p = insert_at;
            while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
            return std.fmt.allocPrint(allocator, "{s}\"{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
        }
    }
    return null;
}

/// Repair the Qwen-MoE-broken shape `{"name":"x", {"k":"v"}}` by injecting
/// the missing `"arguments":` key between the comma and the inner object.
/// Returns an allocator-owned repaired JSON string, or null if the pattern
/// doesn't match.
fn repairFlatBraceToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    if (trimmed.len < 4 or trimmed[0] != '{') return null;

    // Locate `"name"` key.
    const name_at = std.mem.indexOf(u8, trimmed, "\"name\"") orelse return null;
    var p = name_at + 6;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ':') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '"') return null;

    // Skip the string value.
    p += 1;
    while (p < trimmed.len and trimmed[p] != '"') {
        if (trimmed[p] == '\\' and p + 1 < trimmed.len) p += 2 else p += 1;
    }
    if (p >= trimmed.len) return null;
    p += 1;

    // Expect `, {` (any whitespace).
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ',') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '{') return null;

    return std.fmt.allocPrint(allocator, "{s}\"arguments\":{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
}

/// Strip one layer of outer braces from a `{{…}}`-wrapped JSON object. Models
/// sometimes emit args with a doubled brace layer (a literal-brace artifact from
/// Jinja `{{ }}` templates). Only touches strings that start with `{{` and end
/// with `}}`, so well-formed single-brace JSON passes through unchanged.
fn unwrapDoubleBraces(s: []const u8) []const u8 {
    if (s.len >= 4 and std.mem.startsWith(u8, s, "{{") and std.mem.endsWith(u8, s, "}}")) {
        return s[1 .. s.len - 1];
    }
    return s;
}

/// Strip artifacts models sometimes append to a tool name — surrounding
/// whitespace and a trailing ':' (Gemma 4 12B leaks one via its `call:NAME:`
/// format). A valid OpenAI tool name matches `^[A-Za-z0-9_-]{1,64}$`, so a
/// trailing colon is never legitimate and is always safe to drop.
fn sanitizeToolName(raw: []const u8) []const u8 {
    var name = std.mem.trim(u8, raw, " \t\n\r");
    while (name.len > 0 and name[name.len - 1] == ':') {
        name = std.mem.trim(u8, name[0 .. name.len - 1], " \t\n\r");
    }
    return name;
}

/// Parse Gemma 4 tool call format: "call:function_name{json_args}"
/// `input_truncated`: the call had NO `<tool_call|>` close tag — the
/// generation was cut (EOS mid-call, max_tokens, or the scheduler's
/// degenerate-tail-loop guard). Under truncation, a value whose scan runs to
/// end-of-body without its terminator is a FRAGMENT and is dropped (the
/// Hermes-truncation rule: a half-written file is worse than a re-issued
/// write); with the close tag present, behavior is byte-identical to before.
fn parseGemma4ToolCall(allocator: std.mem.Allocator, content: []const u8, input_truncated: bool) ?ParsedToolCall {
    const prefix = "call:";
    if (!std.mem.startsWith(u8, content, prefix)) return null;
    const after_prefix = content[prefix.len..];

    // Find the opening brace
    const brace_pos = std.mem.indexOf(u8, after_prefix, "{") orelse return null;
    const name = sanitizeToolName(after_prefix[0..brace_pos]);
    if (name.len == 0) return null;

    var args_str = after_prefix[brace_pos..];

    // Gemma 4 uses {{ }} (double braces) for literal braces in Jinja templates.
    // The model often generates {{"key":"value"}} — unwrap the outer braces.
    args_str = unwrapDoubleBraces(args_str);

    // Try JSON first (model sometimes outputs valid JSON arguments)
    if (std.json.parseFromSlice(std.json.Value, allocator, args_str, .{})) |parsed| {
        defer parsed.deinit();
        if (parsed.value == .object) {
            return .{
                .name = allocator.dupe(u8, name) catch return null,
                .arguments = allocator.dupe(u8, args_str) catch return null,
            };
        }
    } else |_| {}

    // Strict JSON failed — but the model may have emitted standard JSON with
    // mangled escaping (raw newlines / unescaped quotes in big content) rather
    // than the custom <|"|> format. Recover that BEFORE the custom-format
    // converter, which would garble standard JSON. looseRepair returns null for
    // the bare-key `{key:<|"|>v<|"|>}` form, so the custom path still runs.
    if (looseRepairToolCallJson(allocator, args_str)) |repaired| {
        var keep = false;
        defer if (!keep) allocator.free(repaired);
        if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
            defer reparsed.deinit();
            if (reparsed.value == .object) {
                const name_owned = allocator.dupe(u8, name) catch return null;
                keep = true;
                log.info("  [tool-parse] loose-repair recovered mangled Gemma 4 tool-call JSON\n", .{});
                return .{ .name = name_owned, .arguments = repaired };
            }
        } else |_| {}
    }

    // Convert Gemma 4 custom format to JSON:
    // {key:<|"|>value<|"|>,key2:<|"|>value2<|"|>} → {"key":"value","key2":"value2"}
    const json = convertGemma4ArgsToJson(allocator, args_str, input_truncated) orelse {
        log.info("  [tool-parse] convertGemma4ArgsToJson FAILED for: {s}\n", .{args_str[0..@min(args_str.len, 200)]});
        return null;
    };
    return .{
        .name = allocator.dupe(u8, name) catch return null,
        .arguments = json,
    };
}

/// Convert Gemma 4's custom key-value format to JSON.
/// Input:  {key:<|"|>value<|"|>,nested:{k:<|"|>v<|"|>},arr:[<|"|>a<|"|>]}
/// Output: {"key":"value","nested":{"k":"v"},"arr":["a"]}
///
/// Gemma 4 emits its own object/array/string syntax (bare keys, `<|"|>…<|"|>`
/// string delimiters) that can nest arbitrarily, so every value is converted
/// recursively — see `convertGemma4Value`. Nested structures that already use
/// valid JSON (regular `"…"` strings, numbers, bools, null) pass through
/// unchanged. The args are always an object, with or without a literal `{…}`.
fn convertGemma4ArgsToJson(allocator: std.mem.Allocator, input: []const u8, input_truncated: bool) ?[]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    // `cut` = the truncation landed INSIDE a value somewhere below; the pair
    // holding it has already been rolled back (fragment values never ship).
    var cut = false;
    _ = convertGemma4Object(allocator, &result, input, 0, 0, input_truncated, &cut) orelse return null;
    return result.toOwnedSlice(allocator) catch return null;
}

const gemma4_str_delim = "<|\"|>";
/// Recursion-depth guard against adversarial deeply-nested model output.
const gemma4_max_depth = 64;

fn gemma4SkipWs(body: []const u8, start: usize) usize {
    var pos = start;
    while (pos < body.len) : (pos += 1) {
        switch (body[pos]) {
            ' ', '\t', '\n', '\r' => {},
            else => break,
        }
    }
    return pos;
}

fn gemma4SkipWsCommas(body: []const u8, start: usize) usize {
    var pos = start;
    while (pos < body.len) : (pos += 1) {
        switch (body[pos]) {
            ' ', '\t', '\n', '\r', ',' => {},
            else => break,
        }
    }
    return pos;
}

/// Parse a Gemma 4 object beginning at `start` (an optional leading `{` is
/// consumed; absent braces are tolerated so a brace-less body still parses).
/// Appends the JSON object to `result`; returns the index just past it.
fn convertGemma4Object(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    if (depth >= gemma4_max_depth) return null;
    var pos = gemma4SkipWs(body, start);
    if (pos < body.len and body[pos] == '{') pos += 1;
    result.append(allocator, '{') catch return null;

    // Dedup keys within this object (a repeated key would make std.json reject
    // the args with error.DuplicateField — same class as parseHermesToolCall's
    // dup-param bug). Keys are slices into the stable input `body`.
    var seen = std.ArrayList([]const u8).empty;
    defer seen.deinit(allocator);

    var closed = false;
    var first = true;
    while (pos < body.len) {
        pos = gemma4SkipWsCommas(body, pos);
        if (pos >= body.len) break;
        if (body[pos] == '}') {
            pos += 1;
            closed = true;
            break;
        }

        // Key: everything up to the first ':'.
        const colon = std.mem.indexOfScalar(u8, body[pos..], ':') orelse break;
        const key_raw = std.mem.trim(u8, body[pos .. pos + colon], " \t\n\r");
        // Strip surrounding quotes if present (model sometimes quotes keys).
        const key = if (key_raw.len >= 2 and key_raw[0] == '"' and key_raw[key_raw.len - 1] == '"')
            key_raw[1 .. key_raw.len - 1]
        else
            key_raw;
        pos = pos + colon + 1;

        // Tentatively emit comma+key+value; roll back if this key is a dup. The
        // value is still consumed (pos advances) so the dup's payload is skipped.
        const mark = result.items.len;
        if (!first) result.append(allocator, ',') catch return null;
        // Escape the KEY — a raw `"`/control byte in the key would emit invalid
        // JSON (the Gemma converter's output is not strict-re-validated).
        appendJsonString(allocator, result, key) catch return null;
        result.append(allocator, ':') catch return null;

        pos = convertGemma4Value(allocator, result, body, pos, depth, input_truncated, cut) orelse return null;

        // The truncation landed INSIDE this value — it is a fragment (partial
        // file content, partial url) and must never ship as a real argument
        // (the Hermes-truncation rule, extended to the Gemma arm 2026-07-14:
        // php.html loop-stop capture). Roll the whole pair back; the scan
        // consumed to end-of-body, so this object is done.
        if (cut.*) {
            result.shrinkRetainingCapacity(mark); // undo comma+key+value; keep `first`
            break;
        }

        var is_dup = false;
        for (seen.items) |s| {
            if (std.mem.eql(u8, s, key)) {
                is_dup = true;
                break;
            }
        }
        if (is_dup) {
            result.shrinkRetainingCapacity(mark); // undo comma+key+value; keep `first`
            continue;
        }
        seen.append(allocator, key) catch return null;
        first = false;
    }

    // A NESTED object the cut landed inside (it never saw its `}`) is itself a
    // fragment — even when every pair inside completed, more were coming; the
    // parent rolls the whole container back. The ROOT object (depth 0) is
    // exempt: pairs completed before the cut are intact values and survive —
    // only the value holding the cut drops.
    if (input_truncated and depth > 0 and !closed) cut.* = true;

    result.append(allocator, '}') catch return null;
    return pos;
}

/// Parse a Gemma 4 array beginning at `start` (`body[start] == '['`). Appends
/// the JSON array to `result`; returns the index just past it.
fn convertGemma4Array(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    if (depth >= gemma4_max_depth) return null;
    var pos = start + 1; // consume '['
    result.append(allocator, '[') catch return null;

    var closed = false;
    var first = true;
    while (pos < body.len) {
        pos = gemma4SkipWsCommas(body, pos);
        if (pos >= body.len) break;
        if (body[pos] == ']') {
            pos += 1;
            closed = true;
            break;
        }
        if (!first) result.append(allocator, ',') catch return null;
        first = false;
        pos = convertGemma4Value(allocator, result, body, pos, depth, input_truncated, cut) orelse return null;
        // Fragment below: stop consuming; the parent object rolls the whole
        // container pair back, so no local cleanup is needed.
        if (cut.*) break;
    }

    // Same fragment rule as nested objects: an unclosed array under truncation
    // is a partial LIST — more elements were coming (a partial edits[] applies
    // fragmentary work, the same hazard as partial file content).
    if (input_truncated and !closed) cut.* = true;

    result.append(allocator, ']') catch return null;
    return pos;
}

/// Convert one Gemma 4 value (custom string / JSON string / object / array /
/// bare literal) beginning at `start`. String branches are checked first so
/// braces inside a string stay string content. Returns the index past it.
fn convertGemma4Value(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    var pos = gemma4SkipWs(body, start);
    if (pos >= body.len) {
        // `key:` then end-of-body. Under truncation the value is a fragment
        // that never arrived — drop the pair (pre-fix this shipped
        // {"path":""} and a client would act on the bogus empty value).
        if (input_truncated) {
            cut.* = true;
            return pos;
        }
        result.appendSlice(allocator, "\"\"") catch return null;
        return pos;
    }

    // Gemma custom string: <|"|>…<|"|> (closing delimiter may be missing if
    // the model output was truncated).
    if (pos + gemma4_str_delim.len <= body.len and
        std.mem.eql(u8, body[pos .. pos + gemma4_str_delim.len], gemma4_str_delim))
    {
        pos += gemma4_str_delim.len;
        const end_idx = std.mem.indexOf(u8, body[pos..], gemma4_str_delim);
        const value = if (end_idx) |e| body[pos .. pos + e] else blk: {
            // Closing <|"|> missing.
            if (input_truncated) {
                // The call itself was cut (no <tool_call|>): the truncation
                // landed INSIDE this string — a fragment (partial file
                // content, partial url) that must never ship as a real
                // argument (live 2026-07-14 php.html: the loop-stop guard cut
                // a write call mid-`content` and the 1.1 KB fragment shipped).
                cut.* = true;
                return body.len;
            }
            // Complete call (close tag present), delimiter just dropped: a
            // plain to-end-of-body scan swallows the args object's own `}`
            // and any stray fence garbage the model tacked on — a real write
            // call reached disk as "mlx_pi1.html`}". Trim one trailing `}`
            // (the enclosing object's closer) plus backtick/whitespace junk.
            var v = std.mem.trimEnd(u8, body[pos..], " \t\n\r");
            if (std.mem.endsWith(u8, v, "}")) v = v[0 .. v.len - 1];
            v = std.mem.trimEnd(u8, v, "` \t\n\r");
            break :blk v;
        };
        pos = if (end_idx) |e|
            pos + e + gemma4_str_delim.len
        else
            body.len;
        appendJsonString(allocator, result, value) catch return null;
        return pos;
    }

    // Regular JSON string: already valid, copy verbatim (respecting \" escapes).
    if (body[pos] == '"') {
        var end = pos + 1;
        var closed_quote = false;
        while (end < body.len) : (end += 1) {
            if (body[end] == '\\' and end + 1 < body.len) {
                end += 1;
                continue;
            }
            if (body[end] == '"') {
                closed_quote = true;
                end += 1;
                break;
            }
        }
        // Unclosed JSON string at end-of-body under truncation: same fragment
        // rule — and copying it verbatim would emit an UNCLOSED string, i.e.
        // invalid JSON that only the {}-fallback safety net could catch.
        if (!closed_quote and input_truncated) {
            cut.* = true;
            return body.len;
        }
        result.appendSlice(allocator, body[pos..end]) catch return null;
        return end;
    }

    if (body[pos] == '{') return convertGemma4Object(allocator, result, body, pos, depth + 1, input_truncated, cut);
    if (body[pos] == '[') return convertGemma4Array(allocator, result, body, pos, depth + 1, input_truncated, cut);

    // Bare value — a JSON literal (number/bool/null) terminates at the
    // enclosing separator and is emitted verbatim.
    const sep_rel = std.mem.indexOfAny(u8, body[pos..], ",}]");
    const first_sep = sep_rel orelse (body.len - pos);
    const head = std.mem.trim(u8, body[pos .. pos + first_sep], " \t\n\r");
    if (isJsonLiteral(head)) {
        // A literal that ran to end-of-body under truncation has no
        // terminator either — uniform fragment rule (a cut `5` may have been
        // `50`; the client re-asks and gets the real value).
        if (sep_rel == null and input_truncated) {
            cut.* = true;
            return body.len;
        }
        result.appendSlice(allocator, head) catch return null;
        return pos + first_sep;
    }

    // Unquoted STRING. The bare scan above stops at the first `,`/`}`/`]`, which
    // truncates rich content (HTML/markdown) that legitimately contains those
    // bytes. Observed live on gemma-4-e4b-it writing a full page: it dropped the
    // OPENING <|"|> on `content` but kept the CLOSING one
    // (`content:<!DOCTYPE…>…</html><|"|>,path:…`), so content got cut at the
    // viewport meta's comma and the rest became bogus keys → invalid args. Only
    // rich content (multi-line or markup) gets the wider scan; a plain short
    // bare token keeps the first-separator behavior so it can't swallow a
    // sibling field (e.g. `command:ls -la`).
    const rich = std.mem.indexOfScalar(u8, head, '\n') != null or
        std.mem.indexOfScalar(u8, head, '<') != null;
    if (rich) {
        // Prefer a CLOSING <|"|> as the boundary (dropped-opener case). Confirm
        // it's a closer — the byte after it (past whitespace) must be a
        // separator/closer — so a LATER field's OPENING delimiter isn't grabbed.
        if (std.mem.indexOf(u8, body[pos..], gemma4_str_delim)) |close_rel| {
            const after = gemma4SkipWs(body, pos + close_rel + gemma4_str_delim.len);
            const is_closer = after >= body.len or body[after] == ',' or body[after] == '}' or body[after] == ']';
            if (is_closer) {
                const value = std.mem.trim(u8, body[pos .. pos + close_rel], " \t\n\r");
                appendJsonString(allocator, result, value) catch return null;
                return pos + close_rel + gemma4_str_delim.len;
            }
        }
        // No usable closing delimiter at all. Under truncation the rest of the
        // body IS the fragment: consuming to the first comma would ship a
        // partial value AND shred the remainder into bogus keys, and the
        // last-`}` heuristic below would cut a truncated page at some interior
        // CSS brace — drop the whole value instead.
        if (input_truncated) {
            cut.* = true;
            return body.len;
        }
        // Complete call, both delimiters dropped. At the TOP level the value
        // runs to the object's closing `}` (the last one); nested values keep
        // the narrow scan since the outer `}` isn't theirs.
        if (depth == 0) {
            if (std.mem.lastIndexOfScalar(u8, body, '}')) |last_brace| {
                if (last_brace > pos) {
                    const value = std.mem.trim(u8, body[pos..last_brace], " \t\n\r");
                    appendJsonString(allocator, result, value) catch return null;
                    return last_brace;
                }
            }
        }
    }

    // A short bare string that ran to end-of-body under truncation is a
    // fragment too (`command:ls -la` cut mid-flags) — uniform rule.
    if (sep_rel == null and input_truncated) {
        cut.* = true;
        return body.len;
    }

    // Plain short bare string — terminate at the first separator (unchanged).
    appendJsonString(allocator, result, head) catch return null;
    return pos + first_sep;
}

fn parseHermesToolCall(allocator: std.mem.Allocator, block: []const u8) ?ParsedToolCall {
    const fn_start_tag = "<function=";
    const fn_start = std.mem.indexOf(u8, block, fn_start_tag) orelse return null;
    const name_start = fn_start + fn_start_tag.len;
    const name_end = std.mem.indexOf(u8, block[name_start..], ">") orelse return null;
    const fn_name = std.mem.trim(u8, block[name_start .. name_start + name_end], " \n");

    var args_map = std.ArrayList(u8).empty;
    defer args_map.deinit(allocator);
    args_map.append(allocator, '{') catch return null;

    const fn_body_start = name_start + name_end + 1;
    const fn_end = std.mem.indexOf(u8, block[fn_body_start..], "</function>") orelse block.len - fn_body_start;
    const fn_body = block[fn_body_start .. fn_body_start + fn_end];

    // Track emitted parameter names so a repeated `<parameter=NAME>` can't
    // produce a duplicate JSON key (std.json rejects those with DuplicateField —
    // invalid JSON to the client; live soak record 443, a 0.8B model emitting two
    // `<parameter=edits>` blocks). First occurrence wins.
    var seen_names = std.ArrayList([]const u8).empty;
    defer seen_names.deinit(allocator);

    var param_search: usize = 0;
    var first_param = true;
    while (std.mem.indexOf(u8, fn_body[param_search..], "<parameter=")) |ps| {
        const p_name_start = param_search + ps + "<parameter=".len;
        const p_name_end = std.mem.indexOf(u8, fn_body[p_name_start..], ">") orelse break;
        const p_name = std.mem.trim(u8, fn_body[p_name_start .. p_name_start + p_name_end], " \n");

        // A well-formed `<parameter=NAME>` name is a clean token. When the model
        // malforms the tag — e.g. `<parameter=limit=1` with no closing `>` — the
        // `>`-scan spills across a newline into `</parameter`, and interpolating
        // that raw would emit INVALID JSON (live soak record 317). Detect the
        // malformed name and skip just this opener, so the well-formed sibling
        // parameter after it is still recovered. Advancing to p_name_start (past
        // the `<parameter=` we matched) guarantees forward progress.
        if (!isPlausibleParamName(p_name)) {
            param_search = p_name_start;
            continue;
        }

        const p_val_start = p_name_start + p_name_end + 1;
        const p_val_end = std.mem.indexOf(u8, fn_body[p_val_start..], "</parameter>") orelse break;
        const p_val = std.mem.trim(u8, fn_body[p_val_start .. p_val_start + p_val_end], "\n");

        // Skip a duplicate name (first wins); still advance past its block.
        var dup = false;
        for (seen_names.items) |seen| {
            if (std.mem.eql(u8, seen, p_name)) {
                dup = true;
                break;
            }
        }
        if (dup) {
            param_search = p_val_start + p_val_end + "</parameter>".len;
            continue;
        }
        seen_names.append(allocator, p_name) catch return null;

        if (!first_param) args_map.append(allocator, ',') catch return null;
        first_param = false;

        // Escape the NAME too — belt-and-braces so no parameter name can ever
        // produce invalid JSON, even if a future malformed shape slips the guard.
        appendJsonString(allocator, &args_map, p_name) catch return null;
        args_map.append(allocator, ':') catch return null;

        const trimmed_val = std.mem.trim(u8, p_val, " ");
        if (isJsonLiteral(trimmed_val)) {
            args_map.appendSlice(allocator, trimmed_val) catch return null;
        } else {
            appendJsonString(allocator, &args_map, trimmed_val) catch return null;
        }

        param_search = p_val_start + p_val_end + "</parameter>".len;
    }

    args_map.append(allocator, '}') catch return null;

    return .{
        .name = allocator.dupe(u8, fn_name) catch return null,
        .arguments = allocator.dupe(u8, args_map.items) catch return null,
    };
}

/// A parameter name from a well-formed `<parameter=NAME>` tag is a short token
/// with no whitespace or markup. Reject names carrying a newline, angle bracket,
/// quote, or whitespace — those signal the `>`-scan spilled past a malformed tag.
fn isPlausibleParamName(name: []const u8) bool {
    if (name.len == 0 or name.len > 128) return false;
    for (name) |c| {
        switch (c) {
            '\n', '\r', '\t', ' ', '<', '>', '"' => return false,
            else => {},
        }
    }
    return true;
}

fn isJsonLiteral(s: []const u8) bool {
    if (s.len == 0) return false;
    if (std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "null")) return true;
    // Objects/arrays are handled by brace-matching in the caller, not here
    _ = std.fmt.parseFloat(f64, s) catch return false;
    return true;
}

/// THE JSON string escaper (quoted, control bytes \u-escaped). Public so other
/// request builders reuse it instead of hand-rolling a second one — a duplicate
/// escaper is exactly how the control-byte class shipped twice before.
pub fn appendJsonString(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), s: []const u8) !void {
    try buf.append(allocator, '"');
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            // Every other control char (e.g. ESC from ANSI codes in tool
            // results) must be \u-escaped — nlohmann inside jinja_render_chat
            // rejects raw control bytes, and the render failure silently
            // downgrades the prompt to fallbackFormatChat.
            0...8, 0x0B, 0x0C, 0x0E...0x1F => {
                var esc: [6]u8 = undefined;
                const n = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch unreachable;
                try buf.appendSlice(allocator, n);
            },
            else => try buf.append(allocator, c),
        }
    }
    try buf.append(allocator, '"');
}

/// Append tool definitions as a system prompt section.
fn appendToolSystemPrompt(allocator: std.mem.Allocator, result_buf: *std.ArrayList(u8), tools_json: []const u8, tool_choice_instruction: ?[]const u8) !void {
    try result_buf.appendSlice(allocator,
        \\You are a helpful assistant with access to the following functions. To call a function, respond with a JSON object in the following format:
        \\<tool_call>
        \\{"name": "function_name", "arguments": {"arg1": "value1"}}
        \\</tool_call>
        \\
        \\Available functions:
        \\
    );
    try result_buf.appendSlice(allocator, tools_json);
    if (tool_choice_instruction) |instr| {
        try result_buf.appendSlice(allocator, instr);
    }
}

// ── Tests ──

const testing = std.testing;

test "collapseDoubledThinkTags collapses 2x → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, "<|Assistant|></think></think>Hi!");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<|Assistant|></think>Hi!", out);
}

test "collapseDoubledThinkTags collapses triple → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, "</think></think></think>X");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("</think>X", out);
}

test "collapseDoubledThinkTags leaves single </think> unchanged" {
    const out = try collapseDoubledThinkTags(testing.allocator, "<think>r</think>Hello");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<think>r</think>Hello", out);
}

test "collapseDoubledThinkTags handles multiple separated doublings" {
    const out = try collapseDoubledThinkTags(testing.allocator,
        "A</think></think>B</think></think>C");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("A</think>B</think>C", out);
}

test "split content: think tags never reach content" {
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>Hello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>\nHello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>\n\nHello", true, false).content);
}

test "split content: empty for open think tag (truncated thought is never content)" {
    try testing.expectEqualStrings("", splitThinkBlock("<think>still thinking...", true, false).content);
}

test "indexOfThinkCloseTag: a close inside an OPEN tool call is argument payload, not a block close" {
    // Agent traffic writes files about prompts, and the think split runs
    // BEFORE the tool parse — so this cut used to destroy the whole call.
    const payload = "<tool_call>write\n<arg_key>content</arg_key>\n" ++
        "<arg_value>a thought ends with </think> here</arg_value>\n</tool_call>";
    try testing.expect(indexOfThinkCloseTag(payload, 0) == null);

    // A call the model emitted and CLOSED inside its thought: the `</think>`
    // after it is a REAL close and the answer stays content.
    const in_thought = "<think>let me try <tool_call>read</tool_call> hmm</think>The answer is 4.";
    const c = indexOfThinkCloseTag(in_thought, 0) orelse return error.MissingClose;
    try testing.expectEqualStrings("The answer is 4.", in_thought[c.pos + c.len ..]);
    try testing.expectEqualStrings("The answer is 4.", splitThinkBlock(in_thought, true, false).content);

    // An opener that NEVER closes is leaked markup, not a payload container:
    // the close still ends the block, and trimLeakedToolMarkup takes the tail.
    const unclosed = "<think>starting <tool_call>partial</think>Answer.";
    try testing.expect(indexOfThinkCloseTag(unclosed, 0) != null);

    // Ordinary shapes are untouched.
    try testing.expect(indexOfThinkCloseTag("<think>r</think>a", 0).?.pos == 8);
    try testing.expect(indexOfThinkCloseTag("<think>r</think>a<tool_call>x</tool_call>", 0).?.pos == 8);
}

test "split content: text without think tags passes through" {
    try testing.expectEqualStrings("Hello world", splitThinkBlock("Hello world", true, false).content);
}

test "splitThinkBlock with complete think block" {
    const result = splitThinkBlock("<think>reasoning here</think>answer here", false, false);
    try testing.expectEqualStrings("reasoning here", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock with empty reasoning" {
    const result = splitThinkBlock("<think>\n\n</think>\n\nactual content", false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("actual content", result.content);
}

test "splitThinkBlock thinking=true no close tag, literal opener present" {
    // Model entered thinking but ran out of tokens before closing.
    const result = splitThinkBlock("<think>partial reasoning", true, false);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, template-opened block" {
    // Qwen 3.6 truncated-thinking leak (regression): the chat template injects
    // `<think>\n` into the generation prompt, so the model's output starts
    // INSIDE the think block with no literal opener. If generation stops
    // (length) before `</think>`, every token so far is reasoning — it must
    // land in reasoning_content, never in content. Matches the streaming path.
    const result = splitThinkBlock("The user wants 17*23. Let me compute", true, true);
    try testing.expectEqualStrings("The user wants 17*23. Let me compute", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, no opener, template did NOT open" {
    // Gemma-style direct answer: thinking enabled but the template injects no
    // opener and the model answered without a thought channel. The answer must
    // stay visible as content.
    const result = splitThinkBlock("It is currently 8:15 AM PDT.", true, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("It is currently 8:15 AM PDT.", result.content);
}

test "splitThinkBlock template-opened block with close tag still splits" {
    // Normal Qwen 3.6 round: no literal opener (template-injected), close tag
    // present — reasoning before it, content after.
    const result = splitThinkBlock("compute 340+51=391</think>\n\n391.", true, true);
    try testing.expectEqualStrings("compute 340+51=391", result.reasoning_content.?);
    try testing.expectEqualStrings("391.", result.content);
}

test "splitThinkBlock thinking=false no tags" {
    const result = splitThinkBlock("just content", false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("just content", result.content);
}

test "splitThinkBlock strips think prefix in thinking mode" {
    const result = splitThinkBlock("<think>my reasoning", true, false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "promptTailOpensThink detects template-injected opener" {
    // Qwen 3.6 generation prompt tail with thinking on
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>\n"));
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>"));
    // Thinking off renders a CLOSED empty block — must not match
    try testing.expect(!promptTailOpensThink("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    // Gemma 4 prompt tail (no injected opener)
    try testing.expect(!promptTailOpensThink("<|turn>model\n"));
    try testing.expect(!promptTailOpensThink(""));
}

test "splitThinkBlock: Inkling message channels (thinking + text + truncation)" {
    // Real REAP25 captures (2026-07-30): the model emits role-less MESSAGES —
    // a thinking message, then a fresh <|message_model|> text message — not a
    // tag pair. Channel markers must never leak into content or reasoning.
    {
        const out = "<|content_thinking|>Shorter wavelengths scatter more.<|end_message|><|message_model|><|content_text|>Rayleigh scattering.<|end_message|>";
        const split = splitThinkBlock(out, true, false);
        try testing.expectEqualStrings("Shorter wavelengths scatter more.", split.reasoning_content.?);
        try testing.expectEqualStrings("Rayleigh scattering.", split.content);
    }
    // Thinking-off shape (live capture: chat 2+2)
    {
        const out = "<|content_text|>4<|end_message|>";
        const split = splitThinkBlock(out, false, false);
        try testing.expect(split.reasoning_content == null);
        try testing.expectEqualStrings("4", split.content);
    }
    // Length-truncated mid-thought: reasoning, never content
    {
        const out = "<|content_thinking|>The user is asking for";
        const split = splitThinkBlock(out, true, false);
        try testing.expectEqualStrings("The user is asking for", split.reasoning_content.?);
        try testing.expectEqualStrings("", split.content);
    }
    // Raw-completion shape: leading <|end_message|><|message_model|> before
    // the first channel marker (no template context).
    {
        const out = "<|end_message|><|message_model|><|content_thinking|>Standard implementation.<|end_message|><|message_model|><|content_text|>def fib(n): ...<|end_message|>";
        const split = splitThinkBlock(out, true, false);
        try testing.expectEqualStrings("Standard implementation.", split.reasoning_content.?);
        try testing.expectEqualStrings("def fib(n): ...", split.content);
    }
}

test "parseToolCalls: Inkling invoke_tool_json family" {
    const allocator = testing.allocator;
    // Canonical shape from the chat template: NAME<|content_invoke_tool_json|>
    // {"name":...,"args":{...}}<|end_message|>, optionally after a thinking message.
    {
        const text = "<|content_thinking|>Need the time.<|end_message|><|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{\"timezone\":\"UTC\"}}<|end_message|>";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("get_time", calls[0].name);
        try testing.expect(!calls[0].inferred);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("UTC", parsed.value.object.get("timezone").?.string);
    }
    // Two calls in one turn (parallel) — each its own message.
    {
        const text = "<|message_model|>alpha<|content_invoke_tool_json|>{\"name\":\"alpha\",\"args\":{\"a\":1}}<|end_message|><|message_model|>beta<|content_invoke_tool_json|>{\"name\":\"beta\",\"args\":{}}<|end_message|>";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 2), calls.len);
        try testing.expectEqualStrings("alpha", calls[0].name);
        try testing.expectEqualStrings("beta", calls[1].name);
    }
    // Truncated mid-args: salvage NAME + {} — never partial values.
    {
        const text = "<|message_model|>write_file<|content_invoke_tool_json|>{\"name\":\"write_file\",\"args\":{\"path\":\"/tmp/x\",\"content\":\"half-writ";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("write_file", calls[0].name);
        try testing.expectEqualStrings("{}", calls[0].arguments);
    }
}

test "parseToolCalls: Inkling name is the trailing identifier run" {
    const allocator = testing.allocator;
    // Live 2026-07-30 pi session: the tool message opens with the
    // <|content_text|> marker glued to the NAME. When the payload is
    // truncated, the salvage name must be the bare identifier — never
    // `<|content_text|>bash` (pi's "Tool <|content_text|>bash not found"
    // reply taught the model to echo that garbage name back).
    {
        const text = "<|message_model|><|content_text|>bash<|content_invoke_tool_json|>{\"name\":\"bash\",\"args\":{\"command\":\"ls";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("bash", calls[0].name);
        try testing.expectEqualStrings("{}", calls[0].arguments);
    }
    // Marker echoed INTO the payload's "name" (the self-reinforcing loop's
    // second stage): sanitize to the trailing identifier run.
    {
        const text = "<|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"<|content_text|>bash\",\"args\":{}}<|end_message|>";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("bash", calls[0].name);
    }
    // Payload name that is ONLY a marker, no prefix name either → the call
    // has no recoverable name and is skipped.
    {
        const text = "<|message_model|><|content_invoke_tool_json|>{\"name\":\"<|content_text|>\",\"args\":{}}<|end_message|>";
        const calls = try parseToolCalls(allocator, text);
        try testing.expect(calls == null);
    }
}

test "parseToolCalls: Inkling back-to-back invokes without <|end_message|>" {
    const allocator = testing.allocator;
    // Live 2026-07-30: the model drops the <|end_message|> between two calls
    // (`{…}}write<|content_invoke_tool_json|>{…}`). Body extraction must stop
    // at the balanced object — an end-tag-bounded body swallows call 2 whole.
    const text = "<|message_model|><|content_text|>write<|content_invoke_tool_json|>{\"name\":\"write\",\"args\":{\"path\":\"a.js\",\"content\":\"x\"}}write<|content_invoke_tool_json|>{\"name\":\"write\",\"args\":{\"path\":\"b.js\",\"content\":\"y\"}}<|end_message|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    try testing.expectEqualStrings("write", calls[1].name);
    const p0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer p0.deinit();
    try testing.expectEqualStrings("a.js", p0.value.object.get("path").?.string);
    const p1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer p1.deinit();
    try testing.expectEqualStrings("b.js", p1.value.object.get("path").?.string);
}

test "streamShouldBufferForTools: Inkling invoke marker and NAME hold" {
    // The invoke marker is a single special token — once present, everything
    // from here is call payload and must never flush as content.
    try testing.expect(streamShouldBufferForTools("<|message_model|>bash<|content_invoke_tool_json|>"));
    try testing.expect(streamShouldBufferForTools("<|content_invoke_tool_json|>{\"name\":\"bash\""));
    // NAME hold: an Inkling boundary marker plus a bare identifier segment
    // could be the NAME right before the invoke marker — hold until
    // disambiguated (prose adds a space/punct within a token or two).
    try testing.expect(streamShouldBufferForTools("<|message_model|><|content_text|>bash"));
    try testing.expect(streamShouldBufferForTools("<|message_model|>get_time"));
    // Prose disambiguates → flows.
    try testing.expect(!streamShouldBufferForTools("<|content_text|>The answer"));
    // Completed message (empty segment after the end marker) → flows.
    try testing.expect(!streamShouldBufferForTools("<|content_text|>4<|end_message|>"));
    // A thinking message never trips the NAME hold (the think gate owns it).
    try testing.expect(!streamShouldBufferForTools("<|content_thinking|>Need the time"));
    // No Inkling markers → other families completely unaffected.
    try testing.expect(!streamShouldBufferForTools("bash"));
    try testing.expect(!streamShouldBufferForTools("plain prose mentioning get_time here"));
}

test "parseToolCalls JSON format" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // arguments should be valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Qwen3.6 MoE nested-name garbage (real capture)" {
    // Captured verbatim from Qwen3.6-35B-A3B-6bit generating a tool call in
    // streaming mode with enable_thinking=false and tools present.
    // parseToolCalls previously returned null here because the top-level "name"
    // is an object, not a string — tokens then leaked as plain-text content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": {"name": {"name":  "write", "arguments": {"path": "/tmp/x/app.test.js", "content": "hello"}}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/x/app.test.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hello", parsed.value.object.get("content").?.string);
}

test "parseToolCalls Qwen MoE double-nested name" {
    // Two layers of nesting — defensive test for a slightly different shape.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name":{"name":"shell","arguments":{"command":"ls"}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls flat Qwen MoE shape (no arguments wrapper)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\", \"command\": \"mkdir -p 2ddungeon\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p 2ddungeon", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("name") == null);
}

test "parseToolCalls flat shape with multiple top-level args" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"writeFile\", \"path\": \"/tmp/a\", \"content\": \"hi\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hi", parsed.value.object.get("content").?.string);
}

test "parseToolCalls flat shape ignores id/type metadata" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"id\": \"call_1\", \"type\": \"function\", \"name\": \"shell\", \"command\": \"ls\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("id") == null);
    try testing.expect(parsed.value.object.get("type") == null);
}

test "parseToolCalls repairs Qwen MoE missing-arguments-key shape" {
    const allocator = testing.allocator;
    // Real broken output observed from Qwen3.6-35B-A3B-6bit:
    // {"name": "shell",  {"command":"ls"}}  — `, {` instead of `, "arguments": {`
    const text = "<tool_call>\n{\"name\":  \"shell\",     {\"command\":\"ls -la\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls repairs Qwen MoE missing-opening-quote on arguments key" {
    const allocator = testing.allocator;
    // Real broken output observed from Qwen3.6-35B-A3B-6bit:
    // {"name": "shell", arguments": {"command":"mkdir -p src/app"}}
    // — missing the OPENING `"` on the `arguments` key.
    const text = "<tool_call>\n{\"name\": \"shell\", arguments\": {\"command\": \"mkdir -p src/app\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p src/app", parsed.value.object.get("command").?.string);
}

test "parseToolCalls recovers writeFile content with RAW newlines (small-model big-file escaping)" {
    const allocator = testing.allocator;
    // Small models writing a big file often emit literal newlines inside the
    // JSON `content` string instead of `\n` — strict JSON rejects raw control
    // bytes, so pre-fix the whole call was dropped and the file leaked as text.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"a.js\",\"content\":\"const x = 1;\nconst y = 2;\n\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("const x = 1;\nconst y = 2;\n", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with RAW tab" {
    const allocator = testing.allocator;
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"m.py\",\"content\":\"def f():\n\treturn 1\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("def f():\n\treturn 1", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with UNESCAPED inner quotes (HTML)" {
    const allocator = testing.allocator;
    // HTML/code with attribute quotes — `<meta charset="UTF-8">` — is the other
    // half of the escaping class: the model forgets to backslash the inner `"`.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"i.html\",\"content\":\"<meta charset=\"UTF-8\"><a href=\"x\">go</a>\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("i.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("<meta charset=\"UTF-8\"><a href=\"x\">go</a>", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with BOTH raw newlines and unescaped quotes" {
    const allocator = testing.allocator;
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"p.html\",\"content\":\"<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<title>Hi</title>\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<title>Hi</title>", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers content with lone backslash (Windows path / regex)" {
    const allocator = testing.allocator;
    // `\U` and `\d` are invalid JSON escapes — strict parse rejects them.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"C:\\Users\\app.js\",\"content\":\"re.match(\\d+)\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("C:\\Users\\app.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("re.match(\\d+)", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers Gemma 4 call:NAME{json} with raw newline content" {
    const allocator = testing.allocator;
    // Gemma 4's JSON-first branch hits the same strict-parse wall on raw bytes.
    const text = "<|tool_call>call:writeFile{\"path\":\"g.txt\",\"content\":\"line1\nline2\"}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("line1\nline2", parsed.value.object.get("content").?.string);
}

test "parseToolCalls leaves valid escaped content untouched (no regression)" {
    const allocator = testing.allocator;
    // Already-correct JSON must pass straight through — the recovery path only
    // runs after strict parse fails, so this never touches looseRepair.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"ok.js\",\"content\":\"a\\nb\\t\\\"q\\\"\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a\nb\t\"q\"", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers truncated <function=writeFile> (max_tokens mid-content)" {
    const allocator = testing.allocator;
    // Live JFK-novel capture (2026-06-20): a Hermes-format writeFile dumped a
    // 19k-char novel into one <parameter=content> and hit the token cap before
    // any closing tag. Pre-fix the close_rel==null branch only tried JSON, so
    // the whole call was DROPPED and leaked as visible text. We must recover at
    // least the tool NAME (args may be empty — content is truncated, not
    // salvaged) so the client fires the chunk/append nudge instead of "use JSON".
    const text = "<tool_call>\n<function=writeFile>\n<parameter=content>\n# THE LION OF MASSACHUSETTS\n\nChapter 1. The young senator rose before dawn";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    // Args are valid JSON (empty object) — the parameter never closed.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
}

test "parseToolCalls recovers EOS-before-close-tag <function=> with full args" {
    const allocator = testing.allocator;
    // Bonus from the same fix: a Hermes call that closed </parameter></function>
    // but hit EOS before </tool_call> now recovers WITH its args, not just the
    // name (parseHermesToolCall reads the closed parameter).
    const text = "<tool_call>\n<function=shell>\n<parameter=command>ls -la</parameter>\n</function>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls recovers a bare <function=> call with NO <tool_call> opener" {
    const allocator = testing.allocator;
    // Live capture (soak, 2026-07-09, Qwen3.6-class via Claude Code): the model
    // emitted the CLOSING </tool_call> but DROPPED the opening <tool_call>, so the
    // whole call arrived as `<function=Write>…</function></tool_call>`. The outer
    // scan triggers on the substring `<tool` — which `</tool_call>` does NOT
    // contain (it is `</too…`) — so parseHermesToolCall was never reached and the
    // entire Write (with its raw <function=/<parameter= markup) leaked into the
    // user-visible text. Same big-file-write dropped-delimiter CLASS as the
    // truncated-opener bugs; recover the call so the markup never leaks.
    const text = "\n\n<function=Write>\n<parameter=file_path>\n/tmp/index.html\n</parameter>\n" ++
        "<parameter=content>\n<!DOCTYPE html>\n<html></html>\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("Write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/index.html", parsed.value.object.get("file_path").?.string);
    try testing.expectEqualStrings("<!DOCTYPE html>\n<html></html>", parsed.value.object.get("content").?.string);
}

test "parseHermesToolCall never emits invalid JSON on a malformed <parameter=> tag" {
    // Live soak capture (record 317): the model wrote `<parameter=limit=1` — using
    // `=1` instead of closing the tag with `>` and a value. The `>`-scan then
    // spilled the 'name' across a newline into `</parameter`, and the raw
    // (unescaped) name interpolation produced INVALID JSON args that flowed to the
    // client. Two guarantees: (1) args are ALWAYS valid JSON; (2) the well-formed
    // sibling `<parameter=path>` is still recovered.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=cat>\n<parameter=limit=1\n</parameter>\n<parameter=path>\n./parse.py\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cat", calls[0].name);
    // Hard invariant: valid JSON object, no matter how malformed the tag was.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    // The clean sibling parameter is recovered.
    try testing.expectEqualStrings("./parse.py", parsed.value.object.get("path").?.string);
}

test "parseHermesToolCall dedups a repeated <parameter=> name (no duplicate JSON key)" {
    // Live soak capture (record 443, Qwen3.5-0.8B): the model emitted TWO
    // `<parameter=edits>` blocks in one call. Pre-fix that produced
    // `{"edits":…,"edits":…,"path":…}`, which std.json rejects with
    // error.DuplicateField — invalid JSON to the client. Dedup so the args are
    // always a valid object.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=edit>\n<parameter=edits>\nfirst\n</parameter>\n" ++
        "<parameter=edits>\nsecond\n</parameter>\n<parameter=path>\n./notes.md\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expectEqualStrings("./notes.md", parsed.value.object.get("path").?.string);
    // Exactly one `edits` key survives.
    try testing.expect(parsed.value.object.get("edits") != null);
}

test "parseToolCalls does NOT treat prose mentioning <function=> as a call" {
    const allocator = testing.allocator;
    // The bare-<function=> fallback must not fire on prose. Without a matching
    // </function> AND a real <parameter=, parseHermesToolCall recovers nothing
    // useful — but guard against a false positive on a lone mention.
    const text = "To define one you write `<function=name>` in the Hermes format.";
    const calls = try parseToolCalls(allocator, text);
    if (calls) |cs| {
        defer {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        }
        // If it parsed anything, it must at least be well-formed JSON args —
        // never leak the raw prose as a tool name.
        try testing.expect(cs.len == 0);
    }
}

test "looseRepair does not fabricate a tool call from non-JSON prose" {
    const allocator = testing.allocator;
    // A stray `{` in prose must not become a tool call via the recovery path.
    const text = "Here is the plan {step one, step two} and we proceed.";
    const calls = try parseToolCalls(allocator, text);
    if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    }
    try testing.expect(calls == null);
}

test "endsWithPartialThinkOpen holds back partial opener tails (pi GGUF leak repro)" {
    // The exact failure shape: `<|channel>` flushed as content before
    // "thought" arrived; the buffer tail was a partial opener at every step.
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|channel>"));
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|chan"));
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|channel>thoug"));
    try testing.expect(endsWithPartialThinkOpen("prose <"));
    try testing.expect(endsWithPartialThinkOpen("prose <think"));
    try testing.expect(endsWithPartialThinkOpen("<th"));
    // Prose and HTML-ish tags keep flowing.
    try testing.expect(!endsWithPartialThinkOpen("just prose."));
    try testing.expect(!endsWithPartialThinkOpen("prose <table"));
    try testing.expect(!endsWithPartialThinkOpen("a > b"));
    try testing.expect(!endsWithPartialThinkOpen(""));
}

const test_tools_write_bash =
    \\[{"type":"function","function":{"name":"Write","description":"Write a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},
    \\ {"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
;

test "inferBareJsonToolCalls maps bare-args fenced JSON to the unique matching tool (Claude Code capture)" {
    const allocator = testing.allocator;
    // Shape captured live from gemma-4-12b via Claude Code /v1/messages: thought
    // block, then a ```json fence holding ONLY the Write tool's arguments.
    const text = "<|channel>thought\nI will create this file using Write.<channel|>```json\n{\n  \"file_path\": \"/Users/david/mlx_info.html\",\n  \"content\": \"<h1>MLX</h1>\"\n}\n```\nI've created the file.";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("Write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/Users/david/mlx_info.html", parsed.value.object.get("file_path").?.string);
}

test "inferBareJsonToolCalls unfenced bare object at content start" {
    const allocator = testing.allocator;
    const text = "{\"command\": \"ls -la\"}";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("Bash", calls[0].name);
}

test "inferBareJsonToolCalls refuses ambiguous and non-matching objects" {
    const allocator = testing.allocator;
    // Two tools share the same key → ambiguous → null.
    const dup_tools =
        \\[{"type":"function","function":{"name":"a","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}},
        \\ {"type":"function","function":{"name":"b","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}}]
    ;
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"x\": \"1\"}", dup_tools)) == null);
    // Keys not in any tool's properties → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"zzz\": 1}", test_tools_write_bash)) == null);
    // Missing a required key → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"file_path\": \"a.txt\"}", test_tools_write_bash)) == null);
}

test "inferBareJsonToolCalls ignores JSON that does not lead the content" {
    const allocator = testing.allocator;
    // Example object mid-prose must never be promoted to a call.
    const text = "Here is an example of the payload you could send: {\"command\": \"ls\"} — adjust as needed.";
    try testing.expect((try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) == null);
}

test "parseToolCalls returns null for name-only object (no real args)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\"}\n</tool_call>";
    const result = try parseToolCalls(allocator, text);
    try testing.expect(result == null);
}

test "parseToolCalls marks raw-JSON fallback calls as inferred, tag calls as explicit" {
    const allocator = testing.allocator;
    // Tagged call → explicit (never schema-name-filtered).
    {
        const calls = (try parseToolCalls(allocator, "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expect(!calls[0].inferred);
    }
    // Bare raw-JSON object → inferred (flat shape included).
    {
        const calls = (try parseToolCalls(allocator, "prefix text {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None\"} suffix")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expect(calls[0].inferred);
        try testing.expectEqualStrings("George Washington", calls[0].name);
    }
}

test "filterInferredBySchema drops undeclared inferred calls, keeps declared and explicit ones" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    // Inferred + undeclared → dropped; slice becomes null.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "George Washington"),
            .arguments = try allocator.dupe(u8, "{\"num\":1}"),
            .inferred = true,
        };
        try testing.expect((try filterInferredBySchema(allocator, calls, tools)) == null);
    }
    // Inferred + declared → kept.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "write"),
            .arguments = try allocator.dupe(u8, "{\"path\":\"a\"}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
    }
    // Explicit + undeclared → kept (client feedback, not our guess); the
    // inferred sibling with an undeclared name is dropped and the slice shrinks.
    {
        var calls = try allocator.alloc(ParsedToolCall, 2);
        calls[0] = .{
            .name = try allocator.dupe(u8, "searchWeb"),
            .arguments = try allocator.dupe(u8, "{\"q\":\"zig\"}"),
            .inferred = false,
        };
        calls[1] = .{
            .name = try allocator.dupe(u8, "John Adams"),
            .arguments = try allocator.dupe(u8, "{\"num\":2}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
        try testing.expectEqualStrings("searchWeb", kept[0].name);
    }
}

test "toolNameIsDeclared: wrapped + flat forms; unparseable schema never drops" {
    const allocator = testing.allocator;
    const wrapped =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{}}}}]
    ;
    const flat =
        \\[{"name":"write","parameters":{"type":"object","properties":{}}}]
    ;
    try testing.expect(toolNameIsDeclared(allocator, wrapped, "write"));
    try testing.expect(toolNameIsDeclared(allocator, flat, "write"));
    try testing.expect(!toolNameIsDeclared(allocator, wrapped, "George Washington"));
    // Broken schema JSON → safe default TRUE (never drop on OUR parse failure).
    try testing.expect(toolNameIsDeclared(allocator, "not json", "anything"));
}

test "parseToolCalls multiple calls" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "func_a", "arguments": {"x": 1}}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "func_b", "arguments": {"y": 2}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("func_a", calls[0].name);
    try testing.expectEqualStrings("func_b", calls[1].name);
}

test "parseToolCalls returns null for no tool calls" {
    const allocator = testing.allocator;
    const result = try parseToolCalls(allocator, "Hello, how can I help you?");
    try testing.expect(result == null);
}

// ── DSV4 training-bias tool-call shapes ──
// DSV4-Flash via ds4 emits tool calls in two non-standard variants the
// reference parser previously dropped on the floor:
//   1. Attribute form: `<tool_call name="X">{args}</tool_call>` — name as an
//      XML attribute, body is JUST the args object.
//   2. Mismatched closing tag: `<tool_call>{full json}</tool_request>` — open
//      tag is `tool_call`, close is `tool_request`. Hermes-style trained on a
//      mixed corpus.
// These tests are the regression guard. Without the fix they fail because
// the parser hard-matched on `<tool_call>…</tool_call>` exactly.

test "parseToolCalls DSV4 attribute form: <tool_call name=\"X\">{args}</tool_call>" {
    const allocator = testing.allocator;
    const text = "<tool_call name=\"webSearch\">{\"query\":\"top open source inference applications\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("top open source inference applications", parsed.value.object.get("query").?.string);
}

test "parseToolCalls DSV4 mismatched close: <tool_call>{json}</tool_request>" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "mkdir -p prisma src/lib"}}
        \\</tool_request>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p prisma src/lib", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 attribute + mismatched close combined" {
    // Worst-case combo observed from DSV4: attribute-style open, body is
    // just the args object (no top-level "name" key), and the closing tag
    // is `</tool_request>` instead of `</tool_call>`.
    const allocator = testing.allocator;
    const text = "<tool_call name=\"shell\">{\"command\":\"ls /tmp\"}</tool_request>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls /tmp", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Hermes double-brace body: <tool_call>{{json}}</tool_call>" {
    // Some models (seen on small GGUF instruct models echoing a Jinja `{{ }}`
    // example) wrap the args object in an extra brace layer. The body must still
    // parse into a tool call rather than leaking through as raw content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{{"name": "get_weather", "arguments": {"city": "Paris"}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Paris", parsed.value.object.get("city").?.string);
}

test "parseToolCalls DSV4 attribute single-quoted: <tool_call name='X'>" {
    // Some captures used single quotes around the attribute value. Same
    // semantics — accept both quote styles.
    const allocator = testing.allocator;
    const text = "<tool_call name='browse'>{\"action\":\"navigate\",\"url\":\"https://example.com\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
}

test "parseToolCalls DSV4 plural-tag form: <tool_calls>{json}</tool_calls>" {
    // Real capture from DSV4-Flash in MLX Core ChatView:
    //   <tool_calls>
    //   {"name": "shell", "arguments": {"command": "df -h /"}}
    //   </tool_calls>
    // Plural open + plural close, single JSON object inside. Without the
    // fix the parser hard-matches `<tool_call` (the suffix `s>` makes the
    // `>`-scan walk past the tool name), the body never extracts, and the
    // raw XML leaks into the chat bubble.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\{"name": "shell", "arguments": {"command": "df -h /"}}
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("df -h /", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 plural-tag with attribute: <tool_calls name=\"X\">{args}</tool_calls>" {
    const allocator = testing.allocator;
    const text = "<tool_calls name=\"webSearch\">{\"query\":\"hi\"}</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
}

test "qwen xml: the live <function=…>/<parameter=…> capture (package.json class)" {
    // Verbatim from ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve via pi (2026-08-12).
    // Qwen 3.5/3.6's OWN template dictates this dialect ("ONLY reply in the
    // following format"), so it is the checkpoint's contract, not a mangle.
    // The JSON-only reader mined the first balanced object out of the body —
    // the file CONTENT — and promoted its "name" key to the tool name, so pi
    // answered "Tool voxel-pagoda-garden not found" and the model looped.
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n" ++
        "/Users/d/pi-lfm/package.json\n</parameter>\n<parameter=content>\n" ++
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}\n" ++
        "</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("/Users/d/pi-lfm/package.json", o.get("path").?.string);
    // The value is RAW BYTES: a JSON file stays a STRING (typing by spelling
    // is what turned this call into `{"name": "voxel-pagoda-garden", …}`).
    // Types come from the schema at the request chokepoint, never from here.
    try testing.expectEqualStrings(
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}",
        o.get("content").?.string,
    );
}

test "qwen xml: truncated mid-parameter ships NAME + the completed params only" {
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n/tmp/a.ts\n</parameter>\n" ++
        "<parameter=content>\nexport const half = ";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a.ts", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "pythonic: the real LFM2.5 capture, with every literal type" {
    // Verbatim from mlx-community/LFM2.5-2.6B-8bit via /v1/completions
    // (2026-08-04). The model emits full Python repr — note `True` and the
    // SINGLE-quoted list, which is NOT the `tojson` form its own template
    // renders history with. A JSON-only value reader gets both wrong.
    const allocator = testing.allocator;
    const text = "...I should call the get_weather function.</think>" ++
        "<|tool_call_start|>[get_weather(city='Paris', days=3, metric=True, tags=['trip', 'eu'])]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("Paris", o.get("city").?.string);
    try testing.expectEqual(@as(i64, 3), o.get("days").?.integer);
    // A schema `boolean` must arrive as a real bool, not the string "True".
    try testing.expectEqual(true, o.get("metric").?.bool);
    // A schema `array` must arrive as a real array — the #94 defect in the
    // grammar where it is easiest to ship: the value is literal text.
    const tags = o.get("tags").?.array;
    try testing.expectEqual(@as(usize, 2), tags.items.len);
    try testing.expectEqualStrings("trip", tags.items[0].string);
    try testing.expectEqualStrings("eu", tags.items[1].string);
}

test "pythonic: several calls in one bracket list, and None/float/dict values" {
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[read_file(path='a.txt'), " ++
        "plot(scale=1.5, title=None, opts={'grid': True, 'dpi': 96})]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("read_file", calls[0].name);
    try testing.expectEqualStrings("plot", calls[1].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqual(@as(f64, 1.5), o.get("scale").?.float);
    try testing.expectEqual(std.json.Value.null, o.get("title").?);
    try testing.expectEqual(true, o.get("opts").?.object.get("grid").?.bool);
    try testing.expectEqual(@as(i64, 96), o.get("opts").?.object.get("dpi").?.integer);
}

test "pythonic: separators inside a string value are not separators" {
    // `)` `,` `[` inside the quoted value must not end the arg or the call —
    // the delimiter-drop class in reverse (a naive scan cuts here).
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[shell(cmd='ls -la (tmp), [x]', dir='/a')]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la (tmp), [x]", parsed.value.object.get("cmd").?.string);
    try testing.expectEqualStrings("/a", parsed.value.object.get("dir").?.string);
}

test "pythonic: escapes are unescaped, and a double-quoted string is accepted" {
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[write(body='line1\\nit\\'s \\\\ done', path=\"b.txt\")]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("line1\nit's \\ done", parsed.value.object.get("body").?.string);
    try testing.expectEqualStrings("b.txt", parsed.value.object.get("path").?.string);
}

test "pythonic: a truncated call recovers the NAME with EMPTY args, never a partial value" {
    // max_tokens cut mid-argument. Shipping half a file body is worse than
    // re-issuing the call (the truncated-opener rule); the name is enough for
    // the client to see a tool_calls finish and retry.
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[write_file(path='a.html', content='<!DOCTYPE html>\\n<p>cut mid-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}

test "pythonic: tolerates a dropped outer bracket and a dropped end marker" {
    const allocator = testing.allocator;
    // No `[`, no `<|tool_call_end|>` — one delimiter drop must not cost the call.
    const calls = (try parseToolCalls(allocator, "<|tool_call_start|>get_time(tz='UTC')")).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_time", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("UTC", parsed.value.object.get("tz").?.string);
}

test "pythonic: an unparseable value stays honest as a string, and keys dedup" {
    const allocator = testing.allocator;
    const text = "<|tool_call_start|>[run(mode=fast, mode=slow)]<|tool_call_end|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    // Bare identifier is not a Python literal — kept verbatim rather than
    // guessed at, so a client's own validation error stays honest. First wins.
    try testing.expectEqualStrings("fast", parsed.value.object.get("mode").?.string);
}

test "pythonic: the marker is required — prose that mentions a call is not one" {
    const allocator = testing.allocator;
    try testing.expect((try parseToolCalls(allocator, "Call get_weather(city='Paris') to do that.")) == null);
}

test "issue #94: a container param carried as literal TEXT coerces to a real array/object" {
    // #94 reports this as an open gap on the strength of the doc comment
    // ("only SCALARS are touched"), which was stale — `coerceValueToType` has
    // had its array/object arm since 60ba5ec (2026-07-09), two weeks before
    // the issue was filed. Pinned as a TEST rather than left as a code read,
    // so the claim is settled by evidence and cannot silently regress.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"stats","parameters":{"type":"object","properties":{"nums":{"type":"array","items":{"type":"integer"}},"opts":{"type":"object"},"label":{"type":"string"}}}}}]
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "stats"),
        .arguments = try allocator.dupe(u8, "{\"nums\":\"[1, 2, 3]\",\"opts\":\"{\\\"grid\\\": true}\",\"label\":\"[not an array]\"}"),
    }};
    defer {
        allocator.free(calls[0].name);
        allocator.free(calls[0].arguments);
    }

    try coerceToolArgsToSchema(allocator, &calls, tools);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;

    const nums = o.get("nums").?.array;
    try testing.expectEqual(@as(usize, 3), nums.items.len);
    try testing.expectEqual(@as(i64, 2), nums.items[1].integer);
    try testing.expectEqual(true, o.get("opts").?.object.get("grid").?.bool);
    // A STRING param whose content merely looks bracket-y is untouched — the
    // schema decides, never the spelling.
    try testing.expectEqualStrings("[not an array]", o.get("label").?.string);
}

test "parseToolCalls DSV4 nested <tool name=\"X\"> inside <tool_calls> wrapper" {
    // Real capture from DSV4-Flash:
    //   <tool_calls>
    //   <tool name="webSearch">{"query": "top 10 ..."}</tool_calls>
    //   <tool name="webSearch">{"query": "best ..."}</tool_calls>
    //   </tool_calls>
    // Two inner calls, each opening with `<tool name="X">` and closing with
    // `</tool_calls>` (the model conflates open and close tag names). The
    // outer `<tool_calls>` wrapper has no content of its own — it's a
    // decorative parent.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="webSearch">{"query": "top 10 open source AI inference engines 2024 2025"}</tool_calls>
        \\<tool name="webSearch">{"query": "best open source AI inference engines vllm tensorrt mlx comparison"}</tool_calls>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Both inner calls must be recovered.
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    try testing.expectEqualStrings("webSearch", calls[1].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expect(std.mem.indexOf(u8, a0.value.object.get("query").?.string, "top 10") != null);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expect(std.mem.indexOf(u8, a1.value.object.get("query").?.string, "best") != null);
}

test "parseToolCalls bare <tool>{json}</tool>" {
    // Hypothetical minimal form — `<tool>` open with no attribute, JSON
    // body carries the name. Belt-and-suspenders for future model output.
    const allocator = testing.allocator;
    const text = "<tool>{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}</tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls DSV4 <parameters> wrapper inside <tool name=\"X\">" {
    // Real capture from DSV4-Flash:
    //   <tool_calls>
    //   <tool name="browse"><parameters>{"action":"navigate","url":"…"}</parameters></tool>
    //   …
    //   </tool_calls>
    // The args object is wrapped in <parameters>…</parameters> before being
    // placed inside the <tool> body. Strip that wrapper before JSON-parsing.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.com"}</parameters></tool>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.org"}</parameters></tool>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
    try testing.expectEqualStrings("https://example.com", a0.value.object.get("url").?.string);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expectEqualStrings("https://example.org", a1.value.object.get("url").?.string);
}

test "parseToolCalls DSV4 trailing extra closing brace in args body" {
    // DSV4 occasionally emits an extra `}` after the args object — likely a
    // training-data artifact. Be tolerant: find the first balanced JSON
    // object and ignore trailing garbage.
    const allocator = testing.allocator;
    const text =
        "<tool name=\"browse\"><parameters>{\"action\":\"navigate\",\"url\":\"https://dev.to/agdex_ai/post\"}}</parameters></tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
}

test "splitThinkBlock with think block before tool call" {
    const text = "<think>I need to call the calculator</think>\n<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"a\": 5}}\n</tool_call>";
    const result = splitThinkBlock(text, false, false);
    try testing.expectEqualStrings("I need to call the calculator", result.reasoning_content.?);
    // `content` is VISIBLE text, so the call block is cut: parseToolCalls runs
    // on the raw output, and whatever it leaves behind is markup the client
    // must never render.
    try testing.expectEqualStrings("", result.content);
    // The keeping-markup variant is what the one parse-from-content caller
    // (/v1/messages non-streaming) uses, and it still hands over the block.
    const raw = splitThinkBlockKeepingMarkup(text, false, false);
    try testing.expect(std.mem.startsWith(u8, raw.content, "<tool_call>"));
}

test "splitThinkBlock with think block before regular content" {
    const text = "<think>Let me think about this</think>\n\nThe answer is 42.";
    const result = splitThinkBlock(text, false, false);
    try testing.expectEqualStrings("Let me think about this", result.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 42.", result.content);
}

test "splitThinkBlock with empty think block" {
    const text = "<think>\n\n</think>\n\nJust content here.";
    const result = splitThinkBlock(text, false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("Just content here.", result.content);
}

test "splitThinkBlock no think tags with tool call" {
    const text = "<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call>";
    const result = splitThinkBlock(text, false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("", result.content);
    try testing.expectEqualStrings(text, splitThinkBlockKeepingMarkup(text, false, false).content);
}

test "parseToolCalls with think block" {
    const allocator = testing.allocator;
    const text = "<think>reasoning</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"test\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("search", calls[0].name);
}

test "parseToolCalls raw JSON without tags" {
    const allocator = testing.allocator;
    const text = "{\"name\": \"get_time\", \"arguments\": {}}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_time", calls[0].name);
}

test "parseToolCalls Hermes format" {
    const allocator = testing.allocator;
    const text = "<tool_call><function=get_weather><parameter=location>Tokyo</parameter></function></tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // Should have {"location":"Tokyo"}
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "split content: Gemma 4 channel tags never reach content" {
    try testing.expectEqualStrings("Hello", splitThinkBlock("<|channel>thought\nreasoning here<channel|>Hello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<|channel>thought\nreasoning<channel|>\nHello", true, false).content);
    try testing.expectEqualStrings("", splitThinkBlock("<|channel>thought\nstill thinking...", true, false).content);
}

test "splitThinkBlock Gemma 4 channel tags" {
    const result = splitThinkBlock("<|channel>thought\nmy reasoning<channel|>answer here", false, false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock Gemma 4 thinking in progress" {
    const result = splitThinkBlock("<|channel>thought\npartial reasoning", true, false);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock truncated mid-thinking does not leak channel tag" {
    // Truncation regression: the model emitted the Gemma 4 *content* channel
    // opener (`<|channel>\n…`) directly — no thought block and no `<channel|>`
    // close — then hit the output cap. The raw `<|channel>` control tag must
    // never reach visible content (it used to leak straight through).
    {
        const r = splitThinkBlock("<|channel>\nThe answer is 42.", true, false);
        try testing.expect(r.reasoning_content == null);
        try testing.expectEqualStrings("The answer is 42.", r.content);
    }
    // Bare dangling opener (cut off right after the tag) → nothing visible.
    {
        const r = splitThinkBlock("<|channel>", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    {
        const r = splitThinkBlock("<|channel>\n", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    // The template-injected-but-no-tags case must still pass through untouched.
    {
        const r = splitThinkBlock("It is currently 8:15 AM PDT.", true, false);
        try testing.expect(r.reasoning_content == null);
        try testing.expectEqualStrings("It is currently 8:15 AM PDT.", r.content);
    }
}

test "splitThinkBlock trailing unclosed thought opener does not leak (Gemma 12B pi regression)" {
    // Gemma 4 12B answers in prose, then opens a NEW thought channel right
    // before the turn ends (its known channel-thought tail behavior). The raw
    // `<|channel>thought` opener — and any unclosed thought text after it —
    // must never reach visible content; pi rendered the literal tag to users.
    {
        // Bare trailing opener, nothing after.
        const r = splitThinkBlock("Here is the design.\n<|channel>thought", true, false);
        try testing.expectEqualStrings("Here is the design.", r.content);
        try testing.expect(r.reasoning_content == null);
    }
    {
        // Trailing opener with unclosed thought text → thought is reasoning.
        const r = splitThinkBlock("Here is the design.\n<|channel>thought\nI should now write the file", true, false);
        try testing.expectEqualStrings("Here is the design.", r.content);
        try testing.expectEqualStrings("I should now write the file", r.reasoning_content.?);
    }
    {
        // Same shape for the <think> family.
        const r = splitThinkBlock("Done.\n<think>wait, maybe I", true, false);
        try testing.expectEqualStrings("Done.", r.content);
        try testing.expectEqualStrings("wait, maybe I", r.reasoning_content.?);
    }
}

test "split content: trailing unclosed thought opener does not leak" {
    // The visible text must be truncated at a trailing unclosed opener (the
    // tag and dangling thought are reasoning, never content).
    try testing.expectEqualStrings("Here is the design.", splitThinkBlock("Here is the design.\n<|channel>thought", true, false).content);
    try testing.expectEqualStrings("Here is the design.", splitThinkBlock("Here is the design.\n<|channel>thought\nI should now write", true, false).content);
    try testing.expectEqualStrings("Done.", splitThinkBlock("Done.\n<think>hmm", true, false).content);
}

test "trimLeakedToolMarkup cuts at the first unparsed tool-call opener" {
    // Untouched when there is no marker (the overwhelmingly common case —
    // the returned slice must be the input itself, not a trimmed copy).
    try testing.expectEqualStrings("plain prose", trimLeakedToolMarkup("plain prose"));
    try testing.expectEqualStrings("ends in a tag <b>", trimLeakedToolMarkup("ends in a tag <b>"));

    // Every wrapper family cuts, and the trailing whitespace before the
    // marker goes with it.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"x\">"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<|tool_call>{}"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<tool_call>{}"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<tool_calls:opensource>"));
    // Inkling's NAME sits before the marker, so it goes with the cut.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done. shell<|content_invoke_tool_json|>{}"));

    // A marker with nothing before it leaves nothing behind — a bare marker
    // is not an answer (live: DSV4 answered a question and then emitted a
    // lone `<｜DSML｜` with no call behind it).
    try testing.expectEqualStrings("", trimLeakedToolMarkup("\n\n<｜DSML｜\n"));

    // Earliest marker wins when several families appear.
    try testing.expectEqualStrings("a", trimLeakedToolMarkup("a<tool_call>b<｜DSML｜c"));
}

test "in-thought tool markup never rides out as reasoning" {
    // Live 2026-08-01 (DeepSeek-V4-Flash-0731 agent session): the model
    // emitted a complete DSML call INSIDE its think block, closed the block,
    // then issued a different call in content. parseToolCalls works on the
    // post-think text, so the in-thought call is deliberately skipped — but
    // it rode out verbatim in reasoning_content, and the client round-tripped
    // it into every later prompt.
    const raw = "Let me check the directory.<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"listFiles\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">quake</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls></think>I'll build it.";
    const split = splitThinkBlock(raw, true, true);
    try testing.expectEqualStrings("Let me check the directory.", split.reasoning_content.?);
    try testing.expectEqualStrings("I'll build it.", split.content);

    // Reasoning that is NOTHING but markup reports no reasoning at all.
    const only_markup = splitThinkBlock("<｜DSML｜tool_calls>\n</｜DSML｜tool_calls></think>answer", true, true);
    try testing.expectEqual(@as(?[]const u8, null), only_markup.reasoning_content);
    try testing.expectEqualStrings("answer", only_markup.content);
}

test "a tool marker split across tokens is only visible once concatenated" {
    // The streaming tools flush holds a tail that looked like a tool call and
    // ships it when nothing parses. DSV4 spells the DSML marker as `<` then
    // `｜DSML｜` (two tokens), so a per-token filter sees neither piece as a
    // marker and both went out as visible content (live 2026-08-01). The
    // flush concatenates first for exactly this reason.
    const pieces = [_][]const u8{ "My name is", "<", "｜DSML｜", "\n" };
    var joined: [64]u8 = undefined;
    var n: usize = 0;
    for (pieces) |p| {
        @memcpy(joined[n .. n + p.len], p);
        n += p.len;
    }
    for (pieces) |p| try testing.expectEqualStrings(p, trimLeakedToolMarkup(p));
    try testing.expectEqualStrings("My name is", trimLeakedToolMarkup(joined[0..n]));
}

test "splitThinkBlock re-opened thought opener right after close does not leak" {
    // 2026-06-19 live regression (gemma-4): the model closed its thought
    // channel and IMMEDIATELY re-opened a fresh one with nothing between, then
    // the turn ended. The leading-strip consumed the first closed block,
    // leaving the bare re-opened opener at pos 0 of the remainder — it must
    // still vanish, and `<|channel>thought` must never be mis-stripped as a
    // CONTENT channel (`<|channel>`), which leaked a glued "thought".
    {
        const r = splitThinkBlock("<|channel>thought\nLet me plan.<channel|>\n<|channel>thought\n", true, false);
        try testing.expectEqualStrings("", r.content);
        try testing.expectEqualStrings("Let me plan.", r.reasoning_content.?);
    }
    {
        // <think> family equivalent.
        const r = splitThinkBlock("<think>plan</think>\n<think>", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    {
        // Content BETWEEN the close and the re-open must survive the cut.
        const r = splitThinkBlock("<|channel>thought\nPlan.<channel|>\nReady.<|channel>thought\n", true, false);
        try testing.expectEqualStrings("Ready.", r.content);
    }
}

test "split content: re-opened thought opener right after close does not leak" {
    // Same regression, content side — this is the exact form that reached
    // chat-history.json (`<|channel>thought\n` as the whole content).
    try testing.expectEqualStrings("", splitThinkBlock("<|channel>thought\nLet me plan.<channel|>\n<|channel>thought\n", true, false).content);
    try testing.expectEqualStrings("Ready.", splitThinkBlock("<|channel>thought\nPlan.<channel|>\nReady.<|channel>thought\n", true, false).content);
    try testing.expectEqualStrings("", splitThinkBlock("<think>plan</think>\n<think>", true, false).content);
}

test "parseToolCalls Gemma 4 format" {
    const allocator = testing.allocator;
    const text = "<|tool_call>call:get_weather{\"location\": \"Tokyo\"}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Gemma 4 with channel thinking" {
    const allocator = testing.allocator;
    const text = "<|channel>thought\nLet me check the weather<channel|>\n<|tool_call>call:get_weather{\"city\": \"Paris\"}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
}

test "parseToolCalls Gemma 4 truncated (no closing tag)" {
    const allocator = testing.allocator;
    // Model hit EOS before generating <tool_call|>
    const text = "<|tool_call>call:browse{action:<|\"|>browse<|\"|>,url:<|\"|>https://finance.yahoo.com<|\"|>}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    // Should have parsed at least the action argument
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("browse", parsed.value.object.get("action").?.string);
}

test "parseToolCalls Gemma 4 truncated mid-value drops the partial value" {
    const allocator = testing.allocator;
    // Model stopped mid-URL: no closing <|"|>, no `}`, no <tool_call|>. POLICY
    // (2026-07-14, aligned with the Hermes truncation rule): a value whose scan
    // runs to end-of-body without its terminator is a truncation FRAGMENT —
    // partial bytes must never ship as a real argument (a client executes them:
    // a fragmentary url browses garbage, a fragmentary content writes a corrupt
    // file "successfully"). The completed pair before it survives, and the tool
    // NAME is always recovered so the client's steering fires.
    const text = "<|tool_call>call:browse{action:<|\"|>navigate<|\"|>,url:<|\"|>https://finance.";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("navigate", parsed.value.object.get("action").?.string);
    // The truncated URL must be DROPPED, never shipped as partial bytes.
    try testing.expect(parsed.value.object.get("url") == null);
}

test "parseToolCalls Gemma 4 loop-stop truncated content dropped, name recovered (php.html capture)" {
    const allocator = testing.allocator;
    // Live 2026-07-14 (pi → gemma-4-26B-A4B-it-qat-4bit, plang/php.html): the
    // model repetition-looped INSIDE the write call's content string and the
    // scheduler's degenerate-tail-loop guard cut generation mid-word — the
    // buffered call arrived with NO closing delimiter, NO `}`, NO <tool_call|>,
    // and `path` never emitted at all. Pre-fix the salvage shipped
    // {"content":"<1.1 KB of loop garbage>"}; pi rejected the call on the
    // missing path and echoed the garbage back into the model's context.
    const text = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <title>PHP</title>\n</head>\n<body>\n    <p>PHP is a widely-used general-purpose scripting language. It is a server-side scripting language, server-side scripting language, server-side scripting language, server-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.get("content") == null);
    try testing.expect(parsed.value.object.get("path") == null);
}

test "parseToolCalls Gemma 4 truncated content after a complete path keeps path, drops fragment" {
    const allocator = testing.allocator;
    // The ordering that would CORRUPT a file: `path` completed BEFORE the cut.
    // Pre-fix salvage = {path, <partial garbage content>} — a schema-valid call
    // the client executes, writing a fragment to a real path and reporting
    // success. The completed path survives (the client's missing-content error
    // steers a clean re-issue); the fragment never ships.
    const text = "<|tool_call>call:write{path:<|\"|>plang/php.html<|\"|>,content:<|\"|><!DOCTYPE html>\n<p>PHP is a server-side scripting language, server-side scripting language, server-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("plang/php.html", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 bare rich value truncated at end-of-body is dropped" {
    const allocator = testing.allocator;
    // Both <|"|> delimiters dropped AND the generation truncated (no `}`, no
    // close tag anywhere in the rest): the run-to-end scan must not ship the
    // fragment as content.
    const text = "<|tool_call>call:write{content:<!DOCTYPE html>\n<h1>Hi</h1>\n<p>cut mid-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 truncated edits container dropped, complete path kept" {
    const allocator = testing.allocator;
    // Cut INSIDE a container arg (pi-style edits array): the list is a fragment
    // — more edits were coming — and applying a partial edit list is the same
    // fragmentary-work hazard as partial content. The whole container drops;
    // the completed scalar before it survives.
    const text = "<|tool_call>call:edit{path:<|\"|>a.sh<|\"|>,edits:[{oldText:<|\"|>x<|\"|>,newText:<|\"|>y<|\"|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.sh", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("edits") == null);
}

test "parseToolCalls Gemma 4 EOS after a complete pair keeps the pair (no over-drop)" {
    const allocator = testing.allocator;
    // Truncation landing BETWEEN pairs (after a completed value + comma) must
    // not over-drop: only the value the cut landed INSIDE is a fragment.
    const text = "<|tool_call>call:write{path:<|\"|>a.html<|\"|>,";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.html", parsed.value.object.get("path").?.string);
}

test "parseToolCalls Gemma 4 EOS right after the key colon drops the key" {
    const allocator = testing.allocator;
    // `path:` then nothing — pre-fix this shipped {"path":""} and a client
    // would create a file with an EMPTY name (or reject on a bogus value).
    // An absent value is a fragment; drop the key, recover the name.
    const text = "<|tool_call>call:write{path:";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("path") == null);
}

test "parseToolCalls Gemma 4 JSON-style truncated string value is dropped" {
    const allocator = testing.allocator;
    // Same truncation class through the JSON-quoted spelling: the closing `"`
    // never arrived. Pre-fix the verbatim copy emitted an UNCLOSED JSON string
    // (invalid args, caught only by the {}-fallback safety net); the complete
    // first pair must survive and the fragment must drop.
    const text = "<|tool_call>call:write{\"path\":\"plang/php.html\",\"content\":\"<!DOCTYPE html>\n<p>cut mid-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("plang/php.html", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 unterminated string must not swallow the closing brace (pi write regression)" {
    const allocator = testing.allocator;
    // Gemma 4 12B emitted a write call whose LAST string value was missing
    // its closing <|"|> delimiter and carried a stray markdown backtick
    // before the args object's `}`. The unterminated-string scan ran to end
    // of body, so the parsed path was literally "mlx_pi1.html`}" — and pi
    // created a file with that name on disk.
    const text = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html><html></html><|\"|>,path:<|\"|>mlx_pi1.html`}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx_pi1.html", parsed.value.object.get("path").?.string);

    // Plain missing-delimiter variant (no backtick): same brace exclusion.
    const text2 = "<|tool_call>call:write{content:<|\"|>X<|\"|>,path:<|\"|>out.html}<tool_call|>";
    const calls2 = (try parseToolCalls(allocator, text2)).?;
    defer {
        for (calls2) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls2);
    }
    const parsed2 = try std.json.parseFromSlice(std.json.Value, allocator, calls2[0].arguments, .{});
    defer parsed2.deinit();
    try testing.expectEqualStrings("out.html", parsed2.value.object.get("path").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with custom delimiters" {
    const allocator = testing.allocator;
    // Model mixes JSON-style quoted keys with <|"|> delimiters — JSON parse fails,
    // convertGemma4ArgsToJson must strip quotes from keys to produce valid JSON.
    const text =
        \\<|tool_call>call:shell{"command":<|"|>ls -la<|"|>}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with bare values" {
    const allocator = testing.allocator;
    // Model uses quoted keys but bare (non-JSON, non-delimited) values
    const text = "<|tool_call>call:shell{\"command\":ls -la}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Gemma 4 dropped OPENING delimiter on big content (E4B live)" {
    const allocator = testing.allocator;
    // Verbatim shape captured from gemma-4-e4b-it-4bit writing a full HTML page
    // (test_tool_matrix_small.sh): the model dropped the OPENING <|"|> on
    // `content` but kept the CLOSING one, so the bare-value scan cut content at
    // the FIRST comma (inside `<meta ... content="width=device-width, ...">`)
    // and shredded the rest of the markup into bogus keys → invalid args. The
    // closing <|"|> (followed by `,path`) is the real boundary.
    const text = "<|tool_call>call:write_file{content:<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html><|\"|>,path:<|\"|>mars.html<|\"|>}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mars.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings(
        "<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html>",
        parsed.value.object.get("content").?.string,
    );
}

test "parseToolCalls Gemma 4 dropped BOTH delimiters on single content field" {
    const allocator = testing.allocator;
    // Both delimiters dropped, content is the only/last field → run to the
    // object's closing brace.
    const text = "<|tool_call>call:write_file{content:<h1>Hi</h1>\n<p>a, b, c</p>}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("<h1>Hi</h1>\n<p>a, b, c</p>", parsed.value.object.get("content").?.string);
}

test "convertGemma4ArgsToJson nested braces in value" {
    const allocator = testing.allocator;
    // Content value contains JSON-like structures (e.g., JavaScript code with objects)
    const input = "{path:<|\"|>server.js<|\"|>,content:<|\"|>const x = {a: 1, b: {c: 2}};<|\"|>}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    // Verify it produces valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("server.js", parsed.value.object.get("path").?.string);
    const content = parsed.value.object.get("content").?.string;
    try testing.expect(std.mem.indexOf(u8, content, "{a: 1") != null);
}

test "convertGemma4ArgsToJson bare array value" {
    const allocator = testing.allocator;
    // Bare array value should be preserved via brace-matching
    const input = "{stops:[\"Rome\",\"Venice\",\"Athens\"],price:1200}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 3), parsed.value.object.get("stops").?.array.items.len);
    try testing.expectEqual(@as(i64, 1200), parsed.value.object.get("price").?.integer);
}

test "convertGemma4ArgsToJson bare nested object value" {
    const allocator = testing.allocator;
    // Bare nested object should be preserved via brace-matching
    const input = "{name:test,config:{\"port\":3000,\"host\":\"localhost\"}}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config").?.object;
    try testing.expectEqual(@as(i64, 3000), config.get("port").?.integer);
}

// Issue #16: nested objects/arrays that themselves use Gemma 4's custom
// `<|"|>`-delimited string format (or bare keys) must be converted recursively,
// not passed through verbatim.
test "convertGemma4ArgsToJson issue#16 nested array of custom strings" {
    const allocator = testing.allocator;
    const input =
        \\{nested_array:[<|"|>foo<|"|>,<|"|>bar<|"|>]}
    ;
    const expected =
        \\{"nested_array":["foo","bar"]}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested object custom format" {
    const allocator = testing.allocator;
    const input =
        \\{nested_object:{foo:<|"|>bar<|"|>}}
    ;
    const expected =
        \\{"nested_object":{"foo":"bar"}}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested object inside array" {
    const allocator = testing.allocator;
    const input =
        \\{nested_object_in_array:[{foo:<|"|>bar<|"|>}]}
    ;
    const expected =
        \\{"nested_object_in_array":[{"foo":"bar"}]}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested array inside object" {
    const allocator = testing.allocator;
    const input =
        \\{nested_array_in_object:{foo:[<|"|>bar<|"|>]}}
    ;
    const expected =
        \\{"nested_array_in_object":{"foo":["bar"]}}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

// Issue #16, real capture: this is the verbatim output of gemma-4-e4b-it-8bit
// for a tool with nested-object + nested-array params. The model nests its
// custom `<|"|>`/bare-key format, so the client must still receive valid,
// fully-converted JSON arguments (no surviving <|"|> delimiters / bare keys).
test "parseToolCalls Gemma 4 nested object + array args (issue#16 real capture)" {
    const allocator = testing.allocator;
    const text =
        \\<|tool_call>call:send_notification{message:<|"|>Deploy complete<|"|>,metadata:{priority:<|"|>high<|"|>,tags:[<|"|>ci<|"|>,<|"|>release<|"|>]},recipients:[<|"|>alice<|"|>,<|"|>bob<|"|>]}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("send_notification", calls[0].name);
    // The arguments must be valid JSON (would fail to parse with raw <|"|> in them).
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Deploy complete", parsed.value.object.get("message").?.string);
    const metadata = parsed.value.object.get("metadata").?.object;
    try testing.expectEqualStrings("high", metadata.get("priority").?.string);
    const tags = metadata.get("tags").?.array;
    try testing.expectEqual(@as(usize, 2), tags.items.len);
    try testing.expectEqualStrings("ci", tags.items[0].string);
    try testing.expectEqualStrings("release", tags.items[1].string);
    const recipients = parsed.value.object.get("recipients").?.array;
    try testing.expectEqual(@as(usize, 2), recipients.items.len);
    try testing.expectEqualStrings("alice", recipients.items[0].string);
    try testing.expectEqualStrings("bob", recipients.items[1].string);
}

// Gemma 4 12B leaks a trailing ':' into the function name (`call:shell:{…}`),
// producing an unresolvable tool name `shell:`. The parser must strip it so the
// client resolves a real tool instead of looping on "Unknown tool 'shell:'".
test "parseToolCalls Gemma 4 strips trailing colon from tool name" {
    const allocator = testing.allocator;
    const text =
        \\<|tool_call>call:shell:{"command":<|"|>ls<|"|>}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
}

test "sanitizeToolName strips trailing colons and whitespace" {
    try testing.expectEqualStrings("shell", sanitizeToolName("shell:"));
    try testing.expectEqualStrings("shell", sanitizeToolName("  shell  "));
    try testing.expectEqualStrings("shell", sanitizeToolName("shell : "));
    try testing.expectEqualStrings("shell", sanitizeToolName("shell::"));
    try testing.expectEqualStrings("cwd", sanitizeToolName("cwd"));
    try testing.expectEqualStrings("", sanitizeToolName(":"));
}

test "isJsonLiteral" {
    try testing.expect(isJsonLiteral("true"));
    try testing.expect(isJsonLiteral("false"));
    try testing.expect(isJsonLiteral("null"));
    // Objects/arrays are now handled by brace-matching in convertGemma4ArgsToJson, not isJsonLiteral
    try testing.expect(!isJsonLiteral("{\"key\":1}"));
    try testing.expect(!isJsonLiteral("[1,2,3]"));
    try testing.expect(isJsonLiteral("42"));
    try testing.expect(isJsonLiteral("3.14"));
    try testing.expect(!isJsonLiteral("hello"));
    try testing.expect(!isJsonLiteral(""));
}

test "appendJsonString escapes special chars" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "hello \"world\"\nnew\\line");
    try testing.expectEqualStrings("\"hello \\\"world\\\"\\nnew\\\\line\"", buf.items);
}

test "appendJsonString empty string" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "");
    try testing.expectEqualStrings("\"\"", buf.items);
}

test "appendJsonString escapes ALL control characters (2026-06-11 ESC-byte regression)" {
    // Live failure: a tool result carrying raw ANSI terminal codes (ESC = 0x1B
    // from `\x1b[?25l`) passed through unescaped, nlohmann::json rejected the
    // messages JSON inside jinja_render_chat, and the prompt silently fell
    // back to the wrong-format fallbackFormatChat — gemma-4-31b then
    // hallucinated entire conversations. JSON strings must escape EVERY
    // control char < 0x20, not just \n \r \t.
    const allocator = testing.allocator;

    var input: [0x20]u8 = undefined;
    for (&input, 0..) |*c, i| c.* = @intCast(i);

    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try appendJsonString(allocator, &buf, &input);

    // No raw control byte may survive in the serialized form.
    for (buf.items) |c| {
        try testing.expect(c >= 0x20);
    }

    // And the result must round-trip through a strict JSON parser byte-exact.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf.items, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(&input, parsed.value.string);
}

test "renderChatTemplate: tool result with raw ANSI escapes still renders via Jinja" {
    // Regression for the 2026-06-11 pi/gemma-4-31b session: the third request
    // was the first whose history held a tool result with a raw ESC byte (the
    // interactive `npx sv create` output). The Jinja render failed on the
    // unescaped messages JSON and silently downgraded to fallbackFormatChat's
    // Gemma-2/3 `<start_of_turn>` text format — not special tokens for Gemma
    // 4, so generation never stopped at turn end. The render must SUCCEED via
    // the template (template marker present, fallback marker absent).
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'tool' -%}<|tool_response>{{ message['content'] }}<turn|>
        \\{%- else -%}<|turn>{{ message['role'] }}
        \\{{ message['content'] }}{% if message.tool_calls %}{% for tc in message.tool_calls %}<|tool_call>{{ tc.function.name }}<tool_call|>{% endfor %}{% endif %}<turn|>
        \\{%- endif -%}
        \\{%- endfor -%}
        \\{# tools referenced so no synthesis kicks in: {{ tools }} #}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "bash", .arguments = "{\"command\": \"npx sv create .\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "make me a sveltekit app" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        // Verbatim shape from the live failure: hide-cursor ANSI code + prompt UI.
        .{ .role = "tool", .content = "\x1b[?25l\u{2502}\n\u{25c6}  Which template would you like?", .tool_call_id = "tc_0" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, "[]", null, false, null, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "<|tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "Which template") != null);
    // Fallback format must NOT have been used.
    try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
}

test "renderChatTemplate: Hy3 constructs — str.format tokens, arguments.items(), tojson(ensure_ascii=False)" {
    // Hy3's chat_template.jinja builds EVERY special token via
    // `'<think{}>'.format(HYTK)` on line 1 — an engine without str.format
    // throws there, and renderChatTemplate silently downgrades to
    // fallbackFormatChat (the wrong-family prompt-format class). This pins the
    // three constructs the template leans on: positional {} format, dict
    // .items() iteration over tool-call arguments (requires args serialized as
    // an OBJECT), and `| tojson(ensure_ascii=False)` on non-string values.
    const allocator = testing.allocator;
    const tpl =
        \\{%- set HYTK = ':opensource' %}
        \\{%- set eos_token = '<eos{}>'.format(HYTK) %}
        \\{%- set user_token = '<user{}>'.format(HYTK) %}
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'user' -%}{{ user_token }}{{ message['content'] }}
        \\{%- elif message['role'] == 'assistant' -%}<asst>{{ message['content'] }}
        \\{%- if message['tool_calls'] %}{% for tool in message['tool_calls'] %}{%- set arguments = tool['function']['arguments'] %}<call>{{ tool['function']['name'] }}{% for key, value in arguments.items() %}<k>{{ key }}</k>{% if value is not string %}{%- set value = value | tojson(ensure_ascii=False) %}{% endif %}<v>{{ value }}</v>{% endfor %}</call>{% endfor %}{{ eos_token }}{% endif %}
        \\{%- endif -%}
        \\{%- endfor -%}
        \\{# {{ tools }} #}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "write_file", .arguments = "{\"path\": \"a.txt\", \"count\": 2}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "write it" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, "[]", null, false, null, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "<user:opensource>write it") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<eos:opensource>") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<call>write_file") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<k>path</k><v>a.txt</v>") != null);
    // Non-string value rendered through tojson.
    try testing.expect(std.mem.indexOf(u8, rendered, "<k>count</k><v>2</v>") != null);
}

test "renderChatTemplate: REAL Hy3 chat_template.jinja renders without fallback (HY3_MODEL_DIR)" {
    // Env-gated: HY3_MODEL_DIR=<dir containing chat_template.jinja>. Renders
    // the actual shipped template with system + tools + a tool round and
    // pins: no silent fallback (family tokens present), template-opened think
    // (thinking on), closed think (thinking off), tool-call args rendered via
    // arguments.items(), tool_response wrapping for role:"tool".
    const dir = std.c.getenv("HY3_MODEL_DIR") orelse return error.SkipZigTest;
    const allocator = testing.allocator;
    const path = try std.fmt.allocPrint(allocator, "{s}/chat_template.jinja", .{std.mem.span(dir)});
    defer allocator.free(path);
    const io = std.Io.Threaded.global_single_threaded.io();
    const tpl = blk: {
        const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.SkipZigTest;
        defer f.close(io);
        var read_buf: [4096]u8 = undefined;
        var reader_state = f.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(1024 * 1024));
    };
    defer allocator.free(tpl);

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };

    // Thinking ON: generation prompt must end inside a think block.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "hy_begin_of_sentence:opensource") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "hy_User:opensource") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<tool_call:opensource>get_time<tool_sep:opensource>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<arg_key:opensource>timezone</arg_key:opensource>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<arg_value:opensource>UTC</arg_value:opensource>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<tool_response:opensource>") != null);
        // No fallback-format markers.
        try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
        // Template-opened think at the generation prompt.
        try testing.expect(promptTailOpensThink(rendered));
    }
    // Thinking OFF (reasoning_effort no_think): think opened AND closed.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(!promptTailOpensThink(rendered));
        try testing.expect(std.mem.indexOf(u8, rendered, "<think:opensource></think:opensource>") != null);
    }
}

test "renderChatTemplate: REAL Inkling chat_template.jinja renders without fallback (hermetic)" {
    // src/fixtures/inkling_chat_template.jinja is the verbatim template from
    // pipenetwork/Inkling-Small-MLX-* (Thinking Machines Inkling Small). It
    // exercises jinja features no other family does — namespace(), dict/tuple
    // literals, tojson(sort_keys=true, separators=(",", ":")), `is mapping` —
    // and it RAISES on any reasoning_effort string it doesn't know (including
    // hy3's "no_think") and on tool-call arguments passed as a JSON STRING. A
    // raise = silent fallbackFormatChat = wrong-family tags, so the render is
    // pinned hermetically for both thinking arms.
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/inkling_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };

    // Thinking ON: effort header 0.9 (our "high"), tool declare, tool call as
    // {"name":...,"args":{...}} with args as an OBJECT (sorted, compact), tool
    // result named via tool_call_id lookup, generation prompt at the tail.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_system|>tool_declare<|content_xml|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "\"name\":\"get_time\"") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_user|><|content_text|>What time is it?<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{\"timezone\":\"UTC\"}}<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|content_model_end_sampling|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_tool|>get_time<|content_text|>12:34 UTC<|end_message|>") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|message_model|>"));
        // No fallback-format markers (a raise inside the template silently falls back).
        try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
    }
    // Thinking OFF: the effort header must be the template's "0" form — hy3's
    // "no_think" is UNKNOWN to this template and raises.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Thinking effort level: 0<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
    }
    // History reasoning round-trip (laguna class): this template PERSISTS
    // assistant reasoning as its own <|content_thinking|> message. If the
    // field ever stops reaching the render, prior turns silently lose their
    // thinking blocks while turn 1 still works.
    {
        const hist = [_]Message{
            .{ .role = "user", .content = "Why is the sky blue?" },
            .{ .role = "assistant", .content = "Rayleigh scattering.", .reasoning_content = "Shorter wavelengths scatter more." },
            .{ .role = "user", .content = "And sunsets?" },
        };
        const rendered = try renderChatTemplate(allocator, &hist, &config, null, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_model|><|content_thinking|>Shorter wavelengths scatter more.<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_model|><|content_text|>Rayleigh scattering.<|end_message|>") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|message_model|>"));
    }
}

test "templateConsumesEffort: the real templates that READ the effort word" {
    // The three families whose templates act on `reasoning_effort`. For these
    // the word is the lever and an effort-derived TOKEN budget on top is pure
    // truncation (see the fn doc).
    try testing.expect(templateConsumesEffort(@embedFile("fixtures/qwen38_chat_template.jinja")));
    try testing.expect(templateConsumesEffort(@embedFile("fixtures/qwen38_27b_chat_template.jinja")));
    try testing.expect(templateConsumesEffort(@embedFile("fixtures/dsv4_chat_template.jinja")));
    try testing.expect(templateConsumesEffort(@embedFile("fixtures/inkling_chat_template.jinja")));

    // Muse reads `reasoning_strength` — the near-miss. Matching it would strip
    // the budget from a family the word never reaches, silently uncapping it.
    try testing.expect(!templateConsumesEffort(@embedFile("fixtures/muse_chat_template.jinja")));
    // Everything else on disk (gemma 4, LFM2.5, Qwen3.5/3.6, laguna, Ling,
    // Nemotron, llama, mistral) reads neither: thinking is a bool there.
    try testing.expect(!templateConsumesEffort(
        "{%- if enable_thinking %}<think>\n{%- endif %}"));
    try testing.expect(!templateConsumesEffort(""));
}

test "renderChatTemplate: a tool with no description/parameters still renders (no silent fallback)" {
    // OpenAI's schema makes `description` optional and a no-arg tool needs no
    // `parameters`, but muse's template dumps all three through `tojson`, and
    // `tojson` RAISES on Undefined — so a description-less tool killed the
    // render and fell back to ChatML with no tools preamble at all (found
    // 2026-08-14 by a sweep whose own test tool omitted the key). Every family
    // that dumps `tool | tojson` wholesale is one missing key from the same
    // class, so the defaults are filled for ALL templates, not just this one.
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = @embedFile("fixtures/muse_chat_template.jinja"),
        .bos_token = null,
        .eos_token = "<|eot|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    // No `description`, no `parameters` — both legal over the wire.
    const bare =
        \\[{"type":"function","function":{"name":"note"}}]
    ;
    const rendered = try renderChatTemplate(allocator, &messages, &config, bare, null, true, null, false);
    defer allocator.free(rendered);
    // The muse tools preamble proves the REAL template ran; the fallback emits
    // ChatML markers instead and would carry no tools text at all.
    try testing.expect(std.mem.indexOf(u8, rendered, "In this environment you have access to a set of tools") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "\"name\": \"note\"") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);

    // A COMPLETE declaration re-emits nothing: null means the caller keeps the
    // client's own bytes, so no request that works today is re-serialized
    // (key order, number formatting and unicode escaping all stay theirs).
    const complete =
        \\[{"type":"function","function":{"name":"note","description":"d","parameters":{"type":"object","properties":{}}}}]
    ;
    try testing.expect(try fillOptionalToolDefKeys(allocator, complete) == null);
    // Only the missing keys are added, and only to the function that lacks them.
    const partial =
        \\[{"type":"function","function":{"name":"a","description":"d"}},{"type":"function","function":{"name":"b","description":"e","parameters":{"type":"object"}}}]
    ;
    const filled = (try fillOptionalToolDefKeys(allocator, partial)).?;
    defer allocator.free(filled);
    try testing.expect(std.mem.indexOf(u8, filled, "\"parameters\"") != null);
    try testing.expect(std.mem.indexOf(u8, filled, "\"description\":\"d\"") != null);
    try testing.expect(std.mem.indexOf(u8, filled, "\"description\":\"e\"") != null);
}

test "renderChatTemplate: REAL Qwen3.8 chat_template.jinja renders without fallback (hermetic)" {
    // src/fixtures/qwen38_chat_template.jinja is verbatim from
    // Qwen/Qwen3.8-2.4T-A95B, the first public Qwen3.8 checkpoint (the VL
    // siblings keep this body and re-add the vision macro). Qwen3.8 added two
    // raise_exception gates that make our previous extra-context values
    // unrenderable on this family:
    //   - `enable_thinking is false` -> 'Disabling thinking is not supported.'
    //   - reasoning_effort outside (xhigh, medium, low) -> raise, and we sent
    //     "high" on every thinking request, "no_think" on every off one.
    // A raise is a silent fallbackFormatChat, and this family's fallback is
    // ChatML too, so the downgrade shows up only as a missing tools preamble.
    // Both arms pinned.
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/qwen38_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };

    // Thinking ON, no client effort: OUR default, which is deliberately NOT
    // the checkpoint's (see qwen38EffortFor) — xhigh reasons without a bound
    // and a client that sends nothing gets no bound either.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to low.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "set to xhigh") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<function=get_time>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<parameter=timezone>\nUTC\n</parameter>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<tool_response>\n12:34 UTC\n</tool_response>") != null);
        // Unconditional think opener at the generation prompt.
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n<think>\n"));
        try testing.expect(promptTailOpensThink(rendered));
    }
    // OpenAI's "high" is this family's "xhigh"; "medium" injects no preamble.
    {
        const hi = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, "high", false);
        defer allocator.free(hi);
        try testing.expect(std.mem.indexOf(u8, hi, "Reasoning effort is set to xhigh.") != null);

        const med = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, "medium", false);
        defer allocator.free(med);
        try testing.expect(std.mem.indexOf(u8, med, "Reasoning effort is set to") == null);
        try testing.expect(std.mem.indexOf(u8, med, "You have access to the following functions:") != null);
    }
    // Thinking OFF: the template REFUSES to disable thinking, so the render
    // must stay on the low-effort arm and the block the template opened is
    // closed in the prompt (noThinkTailSuffix) instead.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to low.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "</think>"));
        try testing.expect(!promptTailOpensThink(rendered));
    }
}

test "renderChatTemplate: Qwen3.8-27B ACCEPTS thinking-off natively (hermetic)" {
    // Same family, DIFFERENT gates: `qwen38_chat_template.jinja` (2.4T-A95B)
    // raises on `enable_thinking is false`, while the 27B's template answers
    // that arm the 3.6 way — a CLOSED `<think>\n\n</think>\n\n` and no
    // reasoning-instructions preamble at all (its effort validation lives
    // INSIDE the thinking-on branch). Sniffing the shared `'xhigh'` literal
    // for both decisions forced every thinking-off request onto the
    // thinking-on arm at effort low, which injects a system line the
    // checkpoint never renders for that arm and leaves the block to be closed
    // by `noThinkTailSuffix` instead of by the template. The effort mapping is
    // still keyed on `'xhigh'` (both templates raise on OpenAI's "high"); only
    // the enable_thinking override is keyed on the refusal itself.
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/qwen38_27b_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const messages = [_]Message{
        .{ .role = "user", .content = "What time is it?" },
    };

    // Thinking ON is unchanged: OpenAI's "high" still has to become this
    // family's "xhigh" or the render raises and silently falls back.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, "high", false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to xhigh.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n<think>\n"));
        try testing.expect(promptTailOpensThink(rendered));
    }
    // Absent client effort takes OUR default (low), not the template's xhigh.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to low.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "set to xhigh") == null);
    }
    // Thinking OFF renders the template's OWN no-think prompt: closed block,
    // no reasoning preamble, and nothing appended after it.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        try testing.expect(!promptTailOpensThink(rendered));
    }
}

test "serializeExtraContext with thinking enabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, true, null);
    defer allocator.free(result);
    // Should contain enable_thinking:true
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":true") != null);
}

test "serializeExtraContext with thinking disabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, false, null);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":false") != null);
}

test "dsv4EffortFor: OpenAI effort vocabulary maps onto DeepSeek's low|high|max" {
    // medium deliberately maps LOW: DeepSeek's "high" text is a verbose
    // "absolute maximum, no shortcuts" preamble and the reference default is
    // low — nothing verbose is injected unless the client explicitly asked.
    try testing.expectEqualStrings("low", dsv4EffortFor(null));
    try testing.expectEqualStrings("low", dsv4EffortFor("none"));
    try testing.expectEqualStrings("low", dsv4EffortFor("minimal"));
    try testing.expectEqualStrings("low", dsv4EffortFor("low"));
    try testing.expectEqualStrings("low", dsv4EffortFor("medium"));
    try testing.expectEqualStrings("low", dsv4EffortFor("banana")); // unknown → default
    try testing.expectEqualStrings("high", dsv4EffortFor("high"));
    try testing.expectEqualStrings("max", dsv4EffortFor("xhigh"));
    try testing.expectEqualStrings("max", dsv4EffortFor("max"));
    // `ultra` is an EXPLICIT ask for the top tier — the unknown-string fallback would invert it into low.
    try testing.expectEqualStrings("max", dsv4EffortFor("ultra"));
}

test "serializeExtraContext: dsv4 gets the reference's default reasoning effort" {
    // DeepSeek-V4-0731 turned `reasoning_effort` into low|high|max where
    // high/max PREPEND a verbose preamble to the conversation and low adds
    // nothing (the reference default). The hy3 convention of sending "high"
    // for every thinking-enabled request would silently attach that preamble
    // to every dsv4 thinking turn; the family is sniffed by `thinking_mode`,
    // which only this template reads.
    const allocator = testing.allocator;
    var dsv4 = ChatConfig{
        .chat_template = "{%- set mode = thinking_mode|default('chat') -%}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    for ([_]bool{ true, false }) |thinking| {
        const r = try serializeExtraContext(allocator, &dsv4, thinking, null);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, "\"reasoning_effort\":\"low\"") != null);
        // thinking_mode still carries the on/off switch for this family.
        try testing.expect(std.mem.indexOf(u8, r, if (thinking)
            "\"thinking_mode\":\"thinking\""
        else
            "\"thinking_mode\":\"chat\"") != null);
    }
    // A client-supplied effort string reaches the template MAPPED through
    // `dsv4EffortFor` — high/max opt into the verbose preamble, medium stays
    // at the reference default (never silently verbose).
    const mapped_cases = [_]struct { in: ?[]const u8, out: []const u8 }{
        .{ .in = "high", .out = "\"reasoning_effort\":\"high\"" },
        .{ .in = "max", .out = "\"reasoning_effort\":\"max\"" },
        .{ .in = "xhigh", .out = "\"reasoning_effort\":\"max\"" },
        .{ .in = "medium", .out = "\"reasoning_effort\":\"low\"" },
    };
    for (mapped_cases) |case| {
        const r = try serializeExtraContext(allocator, &dsv4, true, case.in);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, case.out) != null);
    }

    // Other families are untouched: hy3-style keeps high/no_think.
    var other = ChatConfig{
        .chat_template = "{{ reasoning_effort }}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const on = try serializeExtraContext(allocator, &other, true, null);
    defer allocator.free(on);
    try testing.expect(std.mem.indexOf(u8, on, "\"reasoning_effort\":\"high\"") != null);
    const off = try serializeExtraContext(allocator, &other, false, null);
    defer allocator.free(off);
    try testing.expect(std.mem.indexOf(u8, off, "\"reasoning_effort\":\"no_think\"") != null);
}

test "serializeMessagesJson basic" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Parse it back to verify valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.array.items.len);
    try testing.expectEqualStrings("user", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("Hello", parsed.value.array.items[0].object.get("content").?.string);
}

test "serializeMessagesJson with tool_calls" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "get_weather", .arguments = "{\"location\":\"Tokyo\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Should contain tool_calls array
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_calls\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "get_weather") != null);
}

test "serializeMessagesJson carries assistant reasoning_content" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const asst = parsed.value.array.items[1].object;
    try testing.expectEqualStrings("two plus two is four", asst.get("reasoning_content").?.string);
    // A message without the field must not carry the key at all — templates
    // gate on `is string`, and an explicit null would still pass `is defined`.
    try testing.expect(parsed.value.array.items[0].object.get("reasoning_content") == null);
}

test "assistant history reasoning_content reaches a template that reads it (laguna round-trip)" {
    // laguna's chat_template.jinja persists reasoning across turns: history
    // assistant messages render <think>{message.reasoning|reasoning_content}</think>.
    // When the parse layer drops the field, every history turn renders the
    // empty <think></think> nothink signature and the model stops thinking
    // from turn 2 of a session (live 2026-07-29, pi agent on Laguna XS: one
    // reasoning delta on the 2-msg turn, zero across the next 13k chunks).
    // Template fragment mirrors laguna's assistant-history branch.
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "{%- for message in messages -%}" ++
            "{%- if message.role == 'assistant' -%}" ++
            "{%- set rc = '' -%}" ++
            "{%- if message.reasoning is string -%}{%- set rc = message.reasoning -%}" ++
            "{%- elif message.reasoning_content is string -%}{%- set rc = message.reasoning_content -%}" ++
            "{%- endif -%}" ++
            "<think>{{ rc }}</think>{{ message.content }}" ++
            "{%- endif -%}{%- endfor -%}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "<think>two plus two is four</think>4") != null);
}

test "serializeMessagesJson embeds valid-JSON arguments as object (not string)" {
    // Required by Qwen 3.5/3.6 templates that do `tool_call.arguments|items`.
    // Without this, the Jinja `items` filter fails on a string and the server
    // falls back to ChatML — losing the `<think>\n` injection that primes the
    // model's reasoning + close-tag behavior on the next turn.
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Object form: ..."arguments":{"command":"date"}...
    // String form (rejected): ..."arguments":"{\"command\":\"date\"}"...
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":{\"command\":\"date\"}") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":\"{") == null);
}

test "serializeMessagesJson keeps malformed arguments as string" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "not valid json" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":\"not valid json\"") != null);
}

test "serializeMessagesJson tool response has tool_call_id and content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt", .tool_call_id = "call_1" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Must contain tool_call_id
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"call_1\"") != null);
    // Must contain the response content
    try testing.expect(std.mem.indexOf(u8, result, "file1.txt") != null);
    // Must NOT contain tool_responses (templates handle role:"tool" natively)
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

test "serializeMessagesJson full tool calling conversation" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_99_0", .name = "shell", .arguments = "{\"command\":\"echo hello\"}" },
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Run echo hello" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "hello", .tool_call_id = "call_99_0" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 4), parsed.value.array.items.len);

    // Verify roles
    try testing.expectEqualStrings("system", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("user", parsed.value.array.items[1].object.get("role").?.string);
    try testing.expectEqualStrings("assistant", parsed.value.array.items[2].object.get("role").?.string);
    try testing.expectEqualStrings("tool", parsed.value.array.items[3].object.get("role").?.string);

    // Verify assistant has tool_calls
    const assistant = parsed.value.array.items[2].object;
    try testing.expect(assistant.get("tool_calls") != null);

    // Verify tool message has tool_call_id and content, no tool_responses
    const tool_msg = parsed.value.array.items[3].object;
    try testing.expectEqualStrings("call_99_0", tool_msg.get("tool_call_id").?.string);
    try testing.expectEqualStrings("hello", tool_msg.get("content").?.string);
    try testing.expect(tool_msg.get("tool_responses") == null);
}

test "serializeMessagesJson multiple parallel tool responses" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "readFile", .arguments = "{\"path\":\"a.txt\"}" },
        .{ .id = "call_2", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file content", .tool_call_id = "call_1" },
        .{ .role = "tool", .content = "Fri Apr 4", .tool_call_id = "call_2" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();

    // Both tool messages should have tool_call_id and content
    const tool1 = parsed.value.array.items[1].object;
    try testing.expectEqualStrings("call_1", tool1.get("tool_call_id").?.string);
    try testing.expectEqualStrings("file content", tool1.get("content").?.string);

    const tool2 = parsed.value.array.items[2].object;
    try testing.expectEqualStrings("call_2", tool2.get("tool_call_id").?.string);
}

test "serializeMessagesJson tool response with empty content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"mkdir test\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "", .tool_call_id = "call_1" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Should have tool_call_id but no tool_responses
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

// ── Fallback formatter tests ──

test "fallbackFormatChat ChatML with tool calls and responses" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "List files" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const result = try fallbackFormatChat(allocator, &messages, &config, tools_json, null, false);
    defer allocator.free(result);

    // Should have <tool_call> block for assistant
    try testing.expect(std.mem.indexOf(u8, result, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"name\": \"shell\"") != null);
    // Should have <tool_response> block for tool result
    try testing.expect(std.mem.indexOf(u8, result, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "file1.txt") != null);
    // Should end with assistant prompt
    try testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n"));
}

test "fallbackFormatChat ChatML tool response uses user role" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "tool", .content = "42" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Tool responses in ChatML use <|im_start|>user role
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user\n<tool_response>") != null);
    // Should NOT have <|im_start|>tool (invalid role)
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>tool") == null);
}

test "fallbackFormatChat Gemma tool response uses user role" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "Run ls" },
        .{ .role = "tool", .content = "file1.txt" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<eos>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Gemma tool results should be in user turn
    try testing.expect(std.mem.indexOf(u8, result, "<start_of_turn>user\nTool result: file1.txt") != null);
    // Should NOT have <start_of_turn>tool (invalid role)
    try testing.expect(std.mem.indexOf(u8, result, "<start_of_turn>tool") == null);
}

test "fallbackFormatChat Llama tool response uses ipython role" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "tool", .content = "hello" },
    };
    var config = ChatConfig{
        .chat_template = "start_header_id",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|eot_id|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Llama tool results use ipython role
    try testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>ipython<|end_header_id|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "hello") != null);
}

test "fallbackFormatChat ChatML assistant with empty content and tool_calls" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Empty content should not produce a blank line before <tool_call>
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>assistant\n<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"name\": \"shell\"") != null);
}

test "fallbackFormatChat multi-round tool calling" {
    const allocator = testing.allocator;
    const tc1 = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const tc2 = [_]ToolCall{
        .{ .id = "call_2", .name = "readFile", .arguments = "{\"path\":\"main.py\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "Read main.py" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc1 },
        .{ .role = "tool", .content = "main.py" },
        .{ .role = "assistant", .content = "Found it.", .tool_calls = &tc2 },
        .{ .role = "tool", .content = "print('hello')" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Should have two <tool_call> blocks
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOf(u8, result[pos..], "<tool_call>")) |offset| {
        count += 1;
        pos += offset + 11;
    }
    try testing.expectEqual(@as(usize, 2), count);

    // Should have two <tool_response> blocks
    count = 0;
    pos = 0;
    while (std.mem.indexOf(u8, result[pos..], "<tool_response>")) |offset| {
        count += 1;
        pos += offset + 15;
    }
    try testing.expectEqual(@as(usize, 2), count);

    // Second assistant should have content before tool_call
    try testing.expect(std.mem.indexOf(u8, result, "Found it.\n<tool_call>") != null);
}

// ── Tool fallback (DSV4 + any template lacking tools support) ──

test "templateReferencesToolRole detects role branches" {
    try testing.expect(templateReferencesToolRole("{% if message['role'] == 'tool' %}"));
    try testing.expect(templateReferencesToolRole("{% if message.role == \"tool\" %}"));
    try testing.expect(!templateReferencesToolRole("{% if message['role'] == 'user' %}{{ message['tool_calls'] }}{% endif %}"));
    try testing.expect(!templateReferencesToolRole(""));
}

test "messagesHaveToolContent" {
    const tc = [_]ToolCall{.{ .id = "x", .name = "n", .arguments = "{}" }};
    const m_tool = [_]Message{.{ .role = "tool", .content = "x" }};
    const m_tc = [_]Message{.{ .role = "assistant", .content = "", .tool_calls = &tc }};
    const m_plain = [_]Message{.{ .role = "user", .content = "hi" }};
    try testing.expect(messagesHaveToolContent(&m_tool));
    try testing.expect(messagesHaveToolContent(&m_tc));
    try testing.expect(!messagesHaveToolContent(&m_plain));
}

test "synthesizeToolFallbackMessages: inject tool prompt with no existing system" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "List files" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    // System content carries the tool prompt + the tools_json blob.
    try testing.expect(std.mem.indexOf(u8, out[0].content, "tool_call") != null);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "shell") != null);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("List files", out[1].content);
}

test "synthesizeToolFallbackMessages: merge tool prompt into existing system" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "List files" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    // Existing system content preserved, tool prompt appended.
    try testing.expect(std.mem.startsWith(u8, out[0].content, "You are helpful."));
    try testing.expect(std.mem.indexOf(u8, out[0].content, "Available functions") != null);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "shell") != null);
}

test "synthesizeToolFallbackMessages: rewrite role:tool to role:user" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "do it" },
        .{ .role = "tool", .content = "result: 42" },
    };
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, false, true, false);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "result: 42") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "</tool_response>") != null);
}

test "synthesizeToolFallbackMessages: rewrite assistant tool_calls into <tool_call> blocks" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc = [_]ToolCall{
        .{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "ls" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
    };
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, false, false, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("assistant", out[1].role);
    try testing.expect(out[1].tool_calls == null); // moved into content
    try testing.expect(std.mem.indexOf(u8, out[1].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "\"name\": \"shell\"") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "</tool_call>") != null);
}

test "renderChatTemplate: DSV4-style template gets tool fallback applied" {
    const allocator = testing.allocator;
    // Minimal template that mirrors DSV4 — has system/user/assistant, no tools, no role:tool.
    const tpl =
        \\<bos>
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'system' -%}{{ message['content'] }}{%- elif message['role'] == 'user' -%}<U>{{ message['content'] }}</U><A>{%- elif message['role'] == 'assistant' -%}{{ message['content'] }}<EOS>{%- endif -%}
        \\{%- endfor -%}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"run a shell command\"}}]";
    const messages = [_]Message{
        .{ .role = "user", .content = "list my files" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
    defer allocator.free(rendered);
    // Tool prompt should have been injected as a system message.
    try testing.expect(std.mem.indexOf(u8, rendered, "Available functions") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "shell") != null);
    // User message still rendered after the synthesized system content.
    try testing.expect(std.mem.indexOf(u8, rendered, "list my files") != null);
}

// ── DSV4 agentic-flow compositions (round-trip + multi-turn + edge cases) ──

test "DSV4 agentic round-trip: model-generated <tool_call> parses + replays" {
    // Simulates the full agent loop step:
    //   1. Model emits a `<tool_call>` block in its generated content.
    //   2. parseToolCalls extracts the structured ToolCall(s).
    //   3. The agent builds the next prompt with role:assistant(tool_calls=...) +
    //      role:tool(content=result).
    //   4. synthesizeToolFallbackMessages rewrites both for the DSV4 template
    //      (assistant tool_calls → <tool_call> block, tool → user
    //      <tool_response>).
    //   5. The same call should round-trip: the rewritten assistant content
    //      should contain a parseable <tool_call> matching the original.
    const allocator = testing.allocator;

    // 1. Pretend the model wrote this:
    const generated =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "date"}}
        \\</tool_call>
    ;
    const parsed_calls = (try parseToolCalls(allocator, generated)).?;
    defer {
        for (parsed_calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(parsed_calls);
    }
    try testing.expectEqual(@as(usize, 1), parsed_calls.len);
    try testing.expectEqualStrings("shell", parsed_calls[0].name);

    // 2. Agent builds the next-turn history with the parsed call + the result.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc = [_]ToolCall{
        .{ .id = "tc_0", .name = parsed_calls[0].name, .arguments = parsed_calls[0].arguments },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "what is the date?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "Mon May 13 2026", .tool_call_id = "tc_0" },
    };

    // 3. Run through the DSV4 fallback. inject_tool_prompt=true mirrors how
    // renderChatTemplate calls it when the template lacks tool support.
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);

    // Should produce: system(tool-prompt), user, assistant(<tool_call>), user(<tool_response>).
    try testing.expectEqual(@as(usize, 4), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("assistant", out[2].role);
    try testing.expectEqualStrings("user", out[3].role);

    // 4. The assistant's rewritten content must contain a <tool_call> block
    //    that re-parses back to the same name + arguments. This is the
    //    closed-loop test.
    try testing.expect(out[2].tool_calls == null);
    const reparsed = (try parseToolCalls(allocator, out[2].content)).?;
    defer {
        for (reparsed) |t| {
            allocator.free(t.name);
            allocator.free(t.arguments);
        }
        allocator.free(reparsed);
    }
    try testing.expectEqual(@as(usize, 1), reparsed.len);
    try testing.expectEqualStrings("shell", reparsed[0].name);
    try testing.expect(std.mem.indexOf(u8, reparsed[0].arguments, "date") != null);

    // 5. The tool-result rewrite must wrap the result in <tool_response>...
    try testing.expect(std.mem.indexOf(u8, out[3].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "Mon May 13 2026") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "</tool_response>") != null);
}

test "DSV4 multi-turn agent history: every role correctly rewritten" {
    // Realistic 6-message agent transcript:
    //   1. user (initial question)
    //   2. assistant tool_call #1 (date)
    //   3. tool result #1
    //   4. assistant intermediate reasoning + tool_call #2 (ls)
    //   5. tool result #2
    //   6. user follow-up
    // The synthesizer must:
    //   - inject the tool-prompt system message at the top
    //   - rewrite each role:tool into role:user with <tool_response>
    //   - rewrite each role:assistant with tool_calls into <tool_call> blocks
    //   - preserve user messages untouched
    //   - preserve assistant intermediate text (msg #4 has BOTH content and tool_calls)
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc1 = [_]ToolCall{.{ .id = "t1", .name = "shell", .arguments = "{\"command\":\"date\"}" }};
    const tc2 = [_]ToolCall{.{ .id = "t2", .name = "shell", .arguments = "{\"command\":\"ls\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "what files exist as of today?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc1 },
        .{ .role = "tool", .content = "Mon May 13 2026", .tool_call_id = "t1" },
        .{ .role = "assistant", .content = "Got the date. Now listing files.", .tool_calls = &tc2 },
        .{ .role = "tool", .content = "README.md\nsrc/\ntests/", .tool_call_id = "t2" },
        .{ .role = "user", .content = "summarize" },
    };

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);

    // Expected layout: system + 6 rewritten messages.
    try testing.expectEqual(@as(usize, 7), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "Available functions") != null);

    // Original index 0 (user) → out[1] user, unchanged.
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("what files exist as of today?", out[1].content);

    // Original index 1 (assistant tool_call) → out[2] assistant with <tool_call> in content, no tool_calls.
    try testing.expectEqualStrings("assistant", out[2].role);
    try testing.expect(out[2].tool_calls == null);
    try testing.expect(std.mem.indexOf(u8, out[2].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[2].content, "\"command\":\"date\"") != null);

    // Original index 2 (tool) → out[3] user with <tool_response>.
    try testing.expectEqualStrings("user", out[3].role);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "Mon May 13 2026") != null);

    // Original index 3 (assistant content + tool_call) → out[4] preserves intermediate text.
    try testing.expectEqualStrings("assistant", out[4].role);
    try testing.expect(out[4].tool_calls == null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "Got the date. Now listing files.") != null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "\"command\":\"ls\"") != null);

    // Original index 4 (tool) → out[5] user with <tool_response>.
    try testing.expectEqualStrings("user", out[5].role);
    try testing.expect(std.mem.indexOf(u8, out[5].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[5].content, "README.md") != null);

    // Original index 5 (user follow-up) → out[6] user, unchanged.
    try testing.expectEqualStrings("user", out[6].role);
    try testing.expectEqualStrings("summarize", out[6].content);
}

test "DSV4 fallback: null tools_json with inject_tool_prompt does not synthesize" {
    // When a request has NO tools (chat without tools param), the synthesizer
    // must NOT inject a system message even with inject_tool_prompt=true.
    // Only role:tool / assistant tool_calls rewrites still fire (in case the
    // template still lacks support and history was injected by a previous turn).
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "hello" },
    };
    // tools_json = null → no synthesis.
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, true, true, true);
    try testing.expectEqual(@as(usize, 1), out.len);
    try testing.expectEqualStrings("user", out[0].role);
    try testing.expectEqualStrings("hello", out[0].content);
}

// ── prepareDs4Prompt: end-to-end unit coverage for the tool-plumbing fix ──
// These tests guard against the regression where ds4 silently dropped
// `tools_json` / `tool_choice_instruction` / `role:"tool"` content. If the
// `encodeChatViaDs4` call sites stop threading `tools_json` through, the
// catalog won't reach `prep.system` and these tests fail.

test "prepareDs4Prompt: tools_json injects the catalog into the system message" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "list /tmp" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"Run a command\"}}]";

    var prep = try prepareDs4Prompt(allocator, &messages, tools_json, null);
    defer prep.deinit();

    try testing.expect(prep.system != null);
    // The synthesized system message MUST contain the tool name so the model
    // knows the tool exists. If `encodeChatViaDs4` stops threading
    // `tools_json` to `synthesizeToolFallbackMessages`, this fails.
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "shell") != null);
    // User turn passes through unchanged.
    try testing.expectEqual(@as(usize, 1), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    try testing.expectEqualStrings("list /tmp", prep.turns[0].content);
}

test "prepareDs4Prompt: tools_json merges into existing system message" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "hi" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"webSearch\"}}]";

    var prep = try prepareDs4Prompt(allocator, &messages, tools_json, null);
    defer prep.deinit();

    try testing.expect(prep.system != null);
    // Both the original system content AND the tool catalog must be present
    // in the merged system message — losing either side would silently
    // strip the agent prompt or the tool list.
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "You are helpful.") != null);
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "webSearch") != null);
}

test "prepareDs4Prompt: role:tool gets rewritten to user with <tool_response>" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "run ls" },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    try testing.expectEqual(@as(usize, 2), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    // Tool result MUST be wrapped as a user turn with <tool_response>; ds4's
    // chat template doesn't model role:"tool" natively so leaving it as
    // role:"tool" makes the model unaware of any prior tool output.
    try testing.expectEqualStrings("user", prep.turns[1].role);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "file1.txt") != null);
}

test "prepareDs4Prompt: assistant tool_calls get inlined as <tool_call> text" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"ls /tmp\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "list tmp" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "ok", .tool_call_id = "c1" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    // Assistant turn must carry the inlined <tool_call> so multi-turn
    // history stays coherent on ds4. Without the rewrite the model sees
    // an empty assistant turn followed by a tool result with no context.
    try testing.expectEqual(@as(usize, 3), prep.turns.len);
    try testing.expectEqualStrings("assistant", prep.turns[1].role);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "\"name\": \"shell\"") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "ls /tmp") != null);
}

test "prepareDs4Prompt: no tools, no tool content → passthrough" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "be terse" },
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "hello" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    // Nothing to synthesize — system should be the original verbatim, turns
    // should be the non-system messages in order.
    try testing.expect(prep.system != null);
    try testing.expectEqualStrings("be terse", prep.system.?);
    try testing.expectEqual(@as(usize, 2), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    try testing.expectEqualStrings("hi", prep.turns[0].content);
    try testing.expectEqualStrings("assistant", prep.turns[1].role);
    try testing.expectEqualStrings("hello", prep.turns[1].content);
}

test "streamShouldBufferForTools: empty + plain prose" {
    try testing.expect(!streamShouldBufferForTools(""));
    try testing.expect(!streamShouldBufferForTools("hello world"));
    try testing.expect(!streamShouldBufferForTools("Let me look at this."));
}

test "streamShouldBufferForTools: trailing partial prefixes of <tool" {
    // The exact regression from the screenshot: model emits `<tool` as a
    // single BPE token, server must buffer instead of streaming it as
    // content (so the next token can either complete the open or be flushed
    // together as harmless prose).
    try testing.expect(streamShouldBufferForTools("<"));
    try testing.expect(streamShouldBufferForTools("<t"));
    try testing.expect(streamShouldBufferForTools("<to"));
    try testing.expect(streamShouldBufferForTools("<too"));
    try testing.expect(streamShouldBufferForTools("<tool"));
    try testing.expect(streamShouldBufferForTools("Let me look at the CLI help:\n<tool"));
}

test "streamShouldBufferForTools: trailing partial prefixes of <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|"));
    try testing.expect(streamShouldBufferForTools("<|t"));
    try testing.expect(streamShouldBufferForTools("<|too"));
    try testing.expect(streamShouldBufferForTools("<|tool"));
    try testing.expect(streamShouldBufferForTools("<|tool_"));
    try testing.expect(streamShouldBufferForTools("<|tool_c"));
    try testing.expect(streamShouldBufferForTools("<|tool_cal"));
}

test "streamShouldBufferForTools: bare <tool> open (DSV4)" {
    try testing.expect(streamShouldBufferForTools("<tool>"));
    try testing.expect(streamShouldBufferForTools("<tool>{\"name\":\"shell\""));
    try testing.expect(streamShouldBufferForTools("<tool name=\"shell\">"));
    try testing.expect(streamShouldBufferForTools("hello <tool>{}"));
}

test "streamShouldBufferForTools: canonical <tool_call*> opens" {
    try testing.expect(streamShouldBufferForTools("<tool_call>"));
    try testing.expect(streamShouldBufferForTools("<tool_call>{\"name\":"));
    try testing.expect(streamShouldBufferForTools("<tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<tool_request>"));
    try testing.expect(streamShouldBufferForTools("prose first then <tool_call>"));
}

test "streamShouldBufferForTools: Gemma 4 <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|tool_call>"));
    try testing.expect(streamShouldBufferForTools("hello <|tool_call name=\"x\">"));
}

test "streamShouldBufferForTools: false positives — <toolkit> etc." {
    // `<toolkit>` and `<toolbar>` have `<tool` as a prefix but the char
    // after is `k` / `b` — not a tag terminator. Must NOT buffer.
    try testing.expect(!streamShouldBufferForTools("here is <toolkit>"));
    try testing.expect(!streamShouldBufferForTools("here is <toolbar>"));
    try testing.expect(!streamShouldBufferForTools("HTML: <toolset>"));
    // But a later real `<tool>` should still trip detection:
    try testing.expect(streamShouldBufferForTools("HTML: <toolset> then <tool>{"));
}

test "streamShouldBufferForTools: raw JSON shape" {
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\",\"arguments\":{}}"));
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\""));
    // Just `{` or JSON without `"name"` shouldn't false-trigger.
    try testing.expect(!streamShouldBufferForTools("{"));
    try testing.expect(!streamShouldBufferForTools("{\"foo\":1}"));
}

test "streamThinkGate: template-opened thinking holds tag-free prose" {
    // The Claude Code leak class: with thinking on and the opener injected by
    // the template, the model's prose has NO tags — it must still be held,
    // never flushed as visible text.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("The user is asking about their system specs.", true, false));
    // …and split exactly when the close tag arrives.
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("The user is asking.\n</think>\n\nI don't have direct access.", true, false));
}

test "streamThinkGate: after the split, prose flushes (hidden-answer guard)" {
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("The visible answer streams normally.", true, true));
}

test "streamThinkGate: explicit openers hold even with thinking off or closed" {
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("Sure.<|channel>thought\nlet me reconsider", true, true));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("<think>hmm", false, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("<|channel>thought\nplan<channel|>done", false, false));
}

test "streamThinkGate: partial opener at the buffer tail holds" {
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("The answer is 391.\n<|channel>", false, false));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("Done. <thi", false, false));
}

test "streamThinkGate: plain prose with thinking off flushes" {
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("17 × 23 = 391.", false, false));
}

test "streamThinkGate: a prompt-opened block is held even with thinking OFF" {
    // LFM2.5 renders `…assistant\n<think>` UNCONDITIONALLY — there is no
    // enable_thinking branch in its template, so the model always reasons and
    // the opener is never in the OUTPUT. The gate saw tag-free prose with
    // thinking off and flushed the entire chain-of-thought as the answer
    // (live 2026-08-04, streaming only — non-streaming strips it because the
    // `</think>` is present by the time the text is split).
    //
    // Whether the prompt ends inside a think block is a fact about the
    // RENDERED BYTES, not about our request flag, which is why it needs its
    // own argument rather than riding `enable_thinking`.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2("The user wants me to say hi in 5 words.", false, false, true));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate2("The user wants 5 words.</think>Hi there, how are you?", false, false, true));

    // Containment, both directions. Same buffers with the prompt flag FALSE
    // behave exactly as before — every other family's template renders the
    // closed `<think></think>` signature (or no tag) when thinking is off, so
    // `promptTailOpensThink` is false for them and this term can never fire.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate2("The user wants me to say hi in 5 words.", false, false, false));
    // And once the block has closed, a prompt-opened flag must not re-hold the
    // visible answer — that would hide every reply after the first.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate2("Hi there, how are you?", false, true, true));
}

test "unstreamedReasoning: only the remainder, and never a resend" {
    try testing.expectEqualStrings("cde", unstreamedReasoning("abcde", 2).?);
    try testing.expectEqualStrings("abcde", unstreamedReasoning("abcde", 0).?);
    // Nothing new since the last delta.
    try testing.expect(unstreamedReasoning("abcde", 5) == null);
    // The split SHRANK (a tool marker appeared mid-thought and the leaked-markup
    // cut took the tail). Deltas cannot be retracted, so send nothing more —
    // resending from the top would duplicate the whole thought.
    try testing.expect(unstreamedReasoning("ab", 5) == null);
    try testing.expect(unstreamedReasoning("", 0) == null);
}

test "streamTailIsReasoning: an unopened block was content all along" {
    // Gemma 4 with enable_thinking:true answering "391" directly — no opener
    // in the prompt (its template renders a bare `<|turn>model\n`) and none in
    // the output. Streamed, the whole answer arrived as reasoning_content with
    // EMPTY content while non-streaming returned it as content (live
    // 2026-08-04). Content is the correct reading: nothing ever opened.
    try testing.expect(!streamTailIsReasoning(true, false, false));

    // A template-opened block that never closed IS reasoning — the truncated
    // thought must not leak into content (Qwen/LFM2.5 pre-inject the opener).
    try testing.expect(streamTailIsReasoning(true, true, false));
    // A model-emitted opener is the same evidence (Gemma's `<|channel>thought`,
    // Inkling's `<|content_thinking|>`), even with no prompt opener.
    try testing.expect(streamTailIsReasoning(true, false, true));
    // Already closed and split ⇒ the tail is the visible answer.
    try testing.expect(!streamTailIsReasoning(false, true, true));
}

test "promptTailOpensThink does not match Gemma's channel opener" {
    // The containment boundary for the prompt-opened-think stream arm: it
    // detects the close with `</think>` ONLY, which is sound exactly because
    // the flag can only be set by a `<think`-family opener. Gemma 4 opens with
    // `<|channel>thought` and closes with `<channel|>`, so it never enters
    // that arm and its streaming is byte-unchanged.
    //
    // If `thinkOpenTagLenAt` ever grows the channel opener, this test fails —
    // and `server.zig`'s arm then needs `<channel|>` in its close check, or a
    // Gemma turn would be held forever waiting for a tag that never arrives.
    try testing.expect(!promptTailOpensThink("<|turn>model\n<|channel>thought\n"));
    try testing.expect(!promptTailOpensThink("<|turn>model\n"));
    // The `<think>` family it DOES match — bare and suffixed (hy3).
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>"));
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think:opensource>"));
}

test "streamThinkGate: the 3-arg form is the 4-arg form with no prompt opener" {
    // The old signature is kept as the no-prompt-opener case so every existing
    // call site and test pins the SAME behavior it always did.
    const cases = [_][]const u8{
        "The user is asking about their system specs.",
        "The user is asking.\n</think>\n\nI don't have direct access.",
        "<think>hmm",
        "17 × 23 = 391.",
        "The answer is 391.\n<|channel>",
    };
    for (cases) |c| {
        for ([_]bool{ true, false }) |think| {
            for ([_]bool{ true, false }) |closed| {
                try testing.expectEqual(streamThinkGate(c, think, closed), streamThinkGate2(c, think, closed, false));
            }
        }
    }
}

test "parseToolCalls: self-closing <tool name=... arguments=... />" {
    const allocator = testing.allocator;
    // Clean form. We use the literal byte sequence the model actually emits —
    // unescaped `"` inside the attribute value. That's XML-invalid but
    // perfectly JSON-parseable once we snap to the inner `{…}`.
    const text =
        \\<tool name="shell" arguments="{"command": "ls -la"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: DSV4 broken-quote self-closing fragment (regression repro)" {
    // Verbatim capture from the failing turn in test_agent_stop_repro.py.
    // The model opened arguments with `"`, closed with `'`, embedded
    // unescaped `"` inside the JSON, and finished with `'/>`. The parser
    // must still recover the JSON object and produce a single tool call.
    const allocator = testing.allocator;
    const text =
        "\n\n<tool_calls>\n" ++
        "<tool name=\"shell\" arguments=\"{\"command\": \"ls -la && cat package.json 2>/dev/null || echo 'no package.json yet'\"}'/>\n" ++
        "</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    // Args should be the balanced JSON object — starts with `{` and ends `}`.
    try testing.expect(calls[0].arguments.len > 0);
    try testing.expectEqual(@as(u8, '{'), calls[0].arguments[0]);
    try testing.expectEqual(@as(u8, '}'), calls[0].arguments[calls[0].arguments.len - 1]);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: multiple self-closing tool tags in one response" {
    const allocator = testing.allocator;
    const text =
        \\<tool name="cwd" arguments="{"path": "src"}"/>
        \\<tool name="listFiles" arguments="{"recursive": "true"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expectEqualStrings("listFiles", calls[1].name);
}

test "parseToolCalls: canonical <tool_call>JSON</tool_call> still parses" {
    // Regression guard — adding the self-closing branch must not break the
    // canonical Hermes form.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls: empty body + attribute-args + explicit </tool_call> close" {
    // Verbatim capture from the user's failed chat-history.json. The model
    // emitted args via XML attribute and used `></tool_call>` (not `/>`)
    // as the terminator, with empty body. Without explicit handling this
    // shape parses as "empty body" → no JSON to extract → no tool call.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool_call name="cwd" arguments="{"path": "/Users/david/.mlx-serve/workspace"}"></tool_call>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, ".mlx-serve/workspace") != null);
}

test "parseToolCalls: <tool_call>{JSON} truncated before </tool_call>" {
    // The model emitted `<tool_call>` + well-formed args JSON, then hit
    // EOS before completing the close marker. Output looks like:
    //   "\n\n<tool_call>\n{\"name\": \"writeFile\", \"arguments\":{...}}\n</tool_cal"
    // Previously this fell through to the orphan branch and got silently
    // dropped — that's the second failure mode the agent-stops repro
    // surfaced. We snap to a balanced JSON body and parse it.
    const allocator = testing.allocator;
    const text = "\n\n<tool_call>\n" ++
        \\{"name": "writeFile", "arguments": {"path": "vite.config.js", "content": "ok"}}
        ++ "\n</tool_cal";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"content\"") != null);
}

test "parseToolCalls: mismatched </tool_action> close still parses" {
    // Verbatim DSV4-Flash capture (2026-06-10 pi html-ds4 turn 2): opened
    // with <tool_call>, closed with the hallucinated </tool_action>. The
    // edit call must parse; pre-fix it leaked into visible text and the
    // agent executed nothing.
    const allocator = testing.allocator;
    const text = "<tool_call>\n" ++
        \\{"name": "edit", "arguments": {"path":"mlx.html", "edits":[{"oldText": "  </ul>\n</body>", "newText": "  </ul>\n  <button onclick=\"alert('Hello from MLX')\">Click me</button>\n</body>"}]}
        ++ "\n</tool_action>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "mlx.html") != null);
}

test "parseToolCalls: fenced JSON array of parallel calls parses all" {
    const allocator = testing.allocator;
    const text = "```json\n[\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris, France\"}},\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo, Japan\"}}\n]\n```";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "Paris, France") != null);
    try testing.expect(std.mem.indexOf(u8, calls[1].arguments, "Tokyo, Japan") != null);
}

test "parseToolCalls: XML-element tool form (tool_name + arg children)" {
    // Verbatim DSV4-Flash capture (2026-06-10, MLX Core agent chat): the
    // tool name and each argument arrive as XML child elements — no JSON.
    const allocator = testing.allocator;
    const text = "Let me check the available disk space on this device.\n\n<tool_calls>\n<tool_name>shell</tool_name>\n<command>df -h / | grep -v \"Filesystem\"</command>\n</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "df -h / | grep -v \\\"Filesystem\\\"") != null);
}

test "parseToolCalls: XML-element-TAG form (<tool_NAME> with arg children)" {
    // Verbatim DSV4-Flash capture (2026-06-10 pi html-ds4 turn 2): the tool
    // name rides in the tag itself — <tool_read>, <tool_edit> — with each
    // argument as an XML child element. Nested elements (the <edits> body)
    // stay verbatim as the arg's string value, matching what working models
    // send pi for the same tool.
    const allocator = testing.allocator;
    const text = "Let me read the current file first.\n\n<tool_read>\n<path>mlx.html</path>\n</tool_read>Now I'll add a button:\n\n<tool_edit>\n<path>mlx.html</path>\n<edits>\n  <oldText>    <h1>MLX</h1></oldText>\n  <newText>    <h1>MLX</h1>\n    <button onclick=\"alert('hi')\">Say Hello</button></newText>\n</edits>\n</tool_edit>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("read", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "mlx.html") != null);
    try testing.expectEqualStrings("edit", calls[1].name);
    // Args must be valid JSON with the nested XML preserved as a string value.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx.html", parsed.value.object.get("path").?.string);
    const edits = parsed.value.object.get("edits").?.string;
    try testing.expect(std.mem.indexOf(u8, edits, "<oldText>") != null);
    try testing.expect(std.mem.indexOf(u8, edits, "Say Hello") != null);
}

test "parseToolCalls: XML-element-TAG form with JSON args body" {
    // Verbatim-shape DSV4-Flash capture (2026-06-10 pi html-ds4, second
    // sampling): same name-in-tag form but the body is a bare JSON args
    // object instead of XML elements — `<tool_write>\n{…}\n</tool_write>`.
    const allocator = testing.allocator;
    const text = "Here's the HTML page:\n\n<tool_write>\n{\"path\": \"/tmp/ws/mlx.html\", \"content\": \"<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<body>\\n</body>\\n</html>\"}\n</tool_write>\n\npage ready";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/ws/mlx.html", parsed.value.object.get("path").?.string);
    try testing.expect(std.mem.indexOf(u8, parsed.value.object.get("content").?.string, "<!DOCTYPE html>") != null);
}

test "parseToolCalls: <tool_NAME> with prose body containing braces is NOT a tool call" {
    // Prose with incidental {braces} inside an unknown tool-ish tag must not
    // produce arguments — the JSON-body variant requires the body to BE the
    // args object, not merely contain one.
    const allocator = testing.allocator;
    const text = "<tool_summary>The config {a: 1} was applied.</tool_summary>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: hallucinated <tool_output> result tag is NOT a tool call" {
    // Verbatim DSV4-Flash capture (same session, turn 1): the model invented
    // a tool RESULT without calling anything. A plain-text body must never
    // produce a call to a tool named "output".
    const allocator = testing.allocator;
    const text = "Here's the page I created for you:\n\n<tool_output>Page ready: mlx.html</tool_output>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: <tool_output> with element children is still NOT a tool call" {
    // A hallucinated result tag can carry element-shaped content (e.g. an
    // echoed HTML fragment). Result-ish names are denylisted outright.
    const allocator = testing.allocator;
    const text = "<tool_output>\n<status>ok</status>\n</tool_output>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: XML-element form without a tool_name is NOT a tool call" {
    // Prose-ish markup inside a <tool_calls> wrapper must not half-execute.
    const allocator = testing.allocator;
    const text = "<tool_calls>\n<note>this has no tool name element</note>\n</tool_calls>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: prose list array is NOT a tool call" {
    // All-or-nothing: an array whose elements aren't {name, arguments}
    // objects must pass through as text.
    const allocator = testing.allocator;
    const text = "[\"apples\", \"oranges\", \"pears\"]";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: <toolkit> is not a tool tag" {
    // The self-closing scan must not false-fire on HTML-ish prose. `<toolkit>`
    // has `<tool` as a prefix but the terminator (`k`) isn't valid, so the
    // outer scan rejects it before parseSelfClosingToolTag even runs.
    const allocator = testing.allocator;
    const text = "Here's a doc about <toolkit> and <toolbar>. No actual tool call.";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

// ── Schema-aware argument coercion (value-spelling type-inference class) ──

/// The Edit tool as Claude Code declares it (OpenAI shape, post
/// `buildOpenAIToolsJson`): `replace_all` is a BOOLEAN, the rest are strings.
const edit_tools_json_test =
    \\[{"type":"function","function":{"name":"Edit","description":"Edit a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean","default":false}},"required":["file_path","old_string","new_string"]}}}]
;

fn parseArgsObj(allocator: std.mem.Allocator, args: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, args, .{});
}

test "coerceToolArgsToSchema: Python-style False on a boolean param becomes JSON false" {
    // VERBATIM capture, 2026-07-09, Qwen3.6-35B-A3B via Claude Code on
    // ~/.mlx-serve/logs/mlx-serve-11234.log:109471. The model emits the Hermes
    // XML parameter form with Python's `False`; parseHermesToolCall's
    // isJsonLiteral only knows lowercase, so it shipped the STRING "False" and
    // Claude Code rejected every Edit with
    //   "The parameter `replace_all` type is expected as `boolean` but provided as `string`"
    // — six times in a row, until the model gave up and rewrote the whole file.
    const allocator = testing.allocator;
    const raw =
        "\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/Users/david/doom/index.html\n</parameter>\n" ++
        "<parameter=old_string>\n  <script src=\"game.js\"></script>\n</parameter>\n" ++
        "<parameter=new_string>\n  <script src=\"game.js\" type=\"module\"></script>\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const v = parsed.value.object.get("replace_all").?;
    try testing.expect(v == .bool);
    try testing.expectEqual(false, v.bool);
}

test "coerceToolArgsToSchema: a STRING param spelled `false` stays a string" {
    // The inverse half of the same class: isJsonLiteral guesses the type from
    // the value's SPELLING, so a code edit whose old_string is literally the
    // token `false` was shipped as JSON `false` (boolean) — "expected string,
    // provided boolean". The schema is the only thing that can disambiguate.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
        "<parameter=old_string>\nfalse\n</parameter>\n<parameter=new_string>\n42\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const old = parsed.value.object.get("old_string").?;
    try testing.expect(old == .string);
    try testing.expectEqualStrings("false", old.string);
    const new = parsed.value.object.get("new_string").?;
    try testing.expect(new == .string);
    try testing.expectEqualStrings("42", new.string);
}

test "coerceToolArgsToSchema: quoted boolean in a JSON body is coerced too" {
    // Same class, different producer: the model emits well-formed JSON but
    // quotes the boolean. Strict parse succeeds, so no repair path ever runs.
    const allocator = testing.allocator;
    const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\"," ++
        "\"old_string\":\"a\",\"new_string\":\"b\",\"replace_all\":\"true\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const v = parsed.value.object.get("replace_all").?;
    try testing.expect(v == .bool);
    try testing.expectEqual(true, v.bool);
}

test "coerceToolArgsToSchema: unknown tool / unknown key / undecidable value pass through" {
    const allocator = testing.allocator;
    // Unknown tool name → args untouched.
    {
        const raw = "<tool_call>{\"name\":\"Nope\",\"arguments\":{\"replace_all\":\"False\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        try testing.expect(parsed.value.object.get("replace_all").? == .string);
    }
    // Known tool, boolean param, value that is not a recognizable boolean →
    // left alone so the client's validation error stays honest.
    {
        const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"replace_all\":\"maybe\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const v = parsed.value.object.get("replace_all").?;
        try testing.expect(v == .string);
        try testing.expectEqualStrings("maybe", v.string);
    }
}

/// pi's `edit` tool, verbatim from its own schema
/// (@earendil-works/pi-coding-agent dist/core/tools/edit.js — typebox
/// `editSchema`). Two facts the tests below lean on: `path` is REQUIRED at the
/// TOP level, and the array's item schema declares ONLY oldText/newText — it
/// never declares `path`. That asymmetry is what makes a buried `path`
/// provably misplaced rather than a judgment call.
const pi_edit_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to edit (relative or absolute)"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]},"description":"One or more targeted replacements."}},"required":["path","edits"]}}}]
;

const edit_array_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object"}},"dry_run":{"type":"boolean"}},"required":["path","edits"]}}}]
;

test "coerceToolArgsToSchema: an ARRAY param passed through Hermes XML is not left a string" {
    // Live capture class (15 occurrences in one session): `<parameter=edits>` holds
    // a JSON array, but parseHermesToolCall's isJsonLiteral only recognizes
    // scalars, so the whole array shipped as a STRING. Strict clients reject it
    // ("edits: want array, got str"); lenient ones silently mis-execute.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n/tmp/a.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"const a = 1;\", \"newText\": \"const a = 2;\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("const a = 2;", edits.array.items[0].object.get("newText").?.string);
    // sibling scalars still coerce/pass through
    try testing.expectEqualStrings("/tmp/a.js", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: a STRING param whose content is JSON stays a string" {
    // The inverse guard for arrays/objects: `old_string` may legitimately BE the
    // text `[1,2]`. Only the schema decides.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
        "<parameter=old_string>\n[1,2]\n</parameter>\n<parameter=new_string>\n{\"k\":1}\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("old_string").? == .string);
    try testing.expectEqualStrings("[1,2]", parsed.value.object.get("old_string").?.string);
    try testing.expect(parsed.value.object.get("new_string").? == .string);
}

test "coerceToolArgsToSchema: already-conforming args are byte-identical (no-op guard)" {
    // The auto-correct layer must never touch a good call. Property: when every
    // declared type already matches, `arguments` comes out byte-for-byte the same.
    const allocator = testing.allocator;
    const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\",\"old_string\":\"a\"," ++
        "\"new_string\":\"b\",\"replace_all\":true}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

// ── Auto-correct safety properties (deterministic fuzz) ─────────────────────
// The repair layer is only ever allowed to touch input that is actually broken.
// These generate well-formed tool calls whose values deliberately SPELL other
// JSON types ("false", "42", "[1,2]", "null") — exactly the strings the
// value-spelling inference used to mis-type — and assert round-trip identity.

fn jsonValueEql(a: std.json.Value, b: std.json.Value) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    return switch (a) {
        .null => true,
        .bool => a.bool == b.bool,
        .integer => a.integer == b.integer,
        .float => a.float == b.float,
        .number_string => std.mem.eql(u8, a.number_string, b.number_string),
        .string => std.mem.eql(u8, a.string, b.string),
        .array => blk: {
            if (a.array.items.len != b.array.items.len) break :blk false;
            for (a.array.items, b.array.items) |x, y| if (!jsonValueEql(x, y)) break :blk false;
            break :blk true;
        },
        .object => blk: {
            if (a.object.count() != b.object.count()) break :blk false;
            var it = a.object.iterator();
            while (it.next()) |e| {
                const other = b.object.get(e.key_ptr.*) orelse break :blk false;
                if (!jsonValueEql(e.value_ptr.*, other)) break :blk false;
            }
            break :blk true;
        },
    };
}

/// Value strings chosen to collide with every JSON literal spelling the
/// tag-format parsers used to guess from.
const adversarial_strings = [_][]const u8{
    "false",      "true",         "False",   "True",    "null",   "None",
    "42",         "-7",           "3.14",    "0",       "1",      "",
    "[1,2]",      "{\"k\":1}",    "nan",     "inf",     "  ",     "yes",
    "a\"b",       "line\nline",   "tab\there", "back\\slash", "café ☕", "0x1f",
    "{not json",  "[unclosed",    "1e400",   "00",      "+5",     "-",
};

test "fuzz: a conforming tool call round-trips byte-identical through parse+coerce" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE_1234);
    const rand = prng.random();

    const types = [_][]const u8{ "string", "integer", "number", "boolean", "array", "object" };

    var iter: usize = 0;
    while (iter < 400) : (iter += 1) {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Build a random schema + a CONFORMING args object for it.
        var props: std.json.ObjectMap = .empty;
        var args: std.json.ObjectMap = .empty;
        const nprops = 1 + rand.uintLessThan(usize, 5);
        var p: usize = 0;
        while (p < nprops) : (p += 1) {
            const key = try std.fmt.allocPrint(arena, "p{d}", .{p});
            const want = types[rand.uintLessThan(usize, types.len)];

            var prop: std.json.ObjectMap = .empty;
            try prop.put(arena, "type", .{ .string = want });
            try props.put(arena, key, .{ .object = prop });

            const v: std.json.Value = if (std.mem.eql(u8, want, "string"))
                .{ .string = adversarial_strings[rand.uintLessThan(usize, adversarial_strings.len)] }
            else if (std.mem.eql(u8, want, "integer"))
                .{ .integer = @as(i64, @intCast(rand.uintLessThan(u32, 1000))) - 500 }
            else if (std.mem.eql(u8, want, "number"))
                .{ .float = 1.5 }
            else if (std.mem.eql(u8, want, "boolean"))
                .{ .bool = rand.boolean() }
            else if (std.mem.eql(u8, want, "array")) blk: {
                var arr = std.json.Array.init(arena);
                try arr.append(.{ .string = "x" });
                try arr.append(.{ .integer = 1 });
                break :blk .{ .array = arr };
            } else blk: {
                var o: std.json.ObjectMap = .empty;
                try o.put(arena, "k", .{ .string = "v" });
                break :blk .{ .object = o };
            };
            try args.put(arena, key, v);
        }

        var func: std.json.ObjectMap = .empty;
        try func.put(arena, "name", .{ .string = "t" });
        var params: std.json.ObjectMap = .empty;
        try params.put(arena, "type", .{ .string = "object" });
        try params.put(arena, "properties", .{ .object = props });
        try func.put(arena, "parameters", .{ .object = params });
        var tool: std.json.ObjectMap = .empty;
        try tool.put(arena, "type", .{ .string = "function" });
        try tool.put(arena, "function", .{ .object = func });
        var tools_arr = std.json.Array.init(arena);
        try tools_arr.append(.{ .object = tool });
        const tools_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .array = tools_arr }, .{});

        // Serialize as a canonical Hermes JSON call — always VALID input.
        const args_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .object = args }, .{});
        const raw = try std.fmt.allocPrint(arena,
            "<tool_call>{{\"name\":\"t\",\"arguments\":{s}}}</tool_call>", .{args_json});

        const calls = (try parseToolCalls(allocator, raw)) orelse {
            std.debug.print("\n[fuzz iter {d}] valid tool call did not parse\n  {s}\n", .{ iter, raw });
            return error.FuzzFailed;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }

        const before = try arena.dupe(u8, calls[0].arguments);
        try coerceToolArgsToSchema(allocator, calls, tools_json);

        // Property 1: a conforming call is left BYTE-identical.
        if (!std.mem.eql(u8, before, calls[0].arguments)) {
            std.debug.print("\n[fuzz iter {d}] auto-correct mutated a conforming call\n  before: {s}\n  after : {s}\n", .{ iter, before, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 2: the values survive verbatim (no spelling-based retyping).
        var got = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer got.deinit();
        if (!jsonValueEql(std.json.Value{ .object = args }, got.value)) {
            std.debug.print("\n[fuzz iter {d}] value mismatch\n  want: {s}\n  got : {s}\n", .{ iter, args_json, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 3: conformance holds, and coercion is idempotent.
        try testing.expect(toolCallConformsToSchema(allocator, calls[0], tools_json));
        try coerceToolArgsToSchema(allocator, calls, tools_json);
        try testing.expectEqualStrings(before, calls[0].arguments);
    }
}

test "coerceToolArgsToSchema: a MANGLED array-typed param is tolerantly repaired" {
    // Live soak capture (record 268): pi's `edits` array with a MISSING COMMA
    // between two object values — `"newText": "..." "path": "..."`. Strict parse
    // rejects it, so pre-fix `edits` stayed a STRING (schema contradiction). The
    // tolerant container repair recovers a real array.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n./game.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"  speed: 8,\", \"newText\": \"  vy: 0,\" \"extra\": \"x\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("  speed: 8,", edits.array.items[0].object.get("oldText").?.string);
}

// ── Misplaced required param (buried-`path` class) ──────────────────────────
//
// VERBATIM captured arguments, 2026-07-13 pi session (gemma-4-26B-A4B-it-qat-4bit,
// ~/.mlx-serve/logs/mlx-serve-11234.log around the us_presidents run). The model
// put `path` INSIDE each edits[] item and emitted no top-level `path`, so pi
// answered
//     Validation failed for tool "edit":
//       - path: must have required properties path
// three times in a row. Each rejection was a full multi-thousand-token
// generation; the model then abandoned `edit` and rewrote the whole file with
// `write` — the same "give up on the cheap tool, re-emit the expensive one"
// degradation as the Claude Code replace_all loop.
//
// The parse layer is NOT at fault (verified: convertGemma4Object/Array are a
// structural walk that preserves the model's own nesting and key order, and
// coerceToolArgsToSchema only rewrites values in place) — the model genuinely
// emitted it this way. The schema is what makes it provably fixable: `path` is
// required at the TOP level and the item schema declares only oldText/newText.
//
// The script bodies are the real ones, excerpted for readability — their LENGTH
// is not load-bearing (the JSON was already valid and correctly escaped; that is
// exactly why no repair path fired). The nesting and the escaping are verbatim.
const pi_buried_path_args_captured =
    \\{"edits":[{"newText":"#!/bin/bash\n\n# Get the directory where the script is located\nSCRIPT_DIR=\"$( cd \"$( dirname \"${BASH_SOURCE[0]}\" )\" && pwd )\"\ncd \"$SCRIPT_DIR\"\n","oldText":"#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n","path":"us_presidents/generate_site.sh"}]}
;

test "hoistMisplacedRequiredParams: pi's buried `path` is lifted out of the edits items" {
    const allocator = testing.allocator;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, pi_buried_path_args_captured),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }

    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();

    // The required param now satisfies the schema at the top level…
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expect(path == .string);
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);

    // …and is gone from the item, whose schema never declared it.
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    const item = edits.array.items[0].object;
    try testing.expect(item.get("path") == null);

    // The payload the model worked for survives byte-exact, escaping intact.
    try testing.expectEqualStrings(
        "#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n",
        item.get("oldText").?.string,
    );
    try testing.expect(std.mem.startsWith(u8, item.get("newText").?.string, "#!/bin/bash\n"));
    try testing.expect(std.mem.indexOf(u8, item.get("newText").?.string, "${BASH_SOURCE[0]}") != null);

    // The whole call now conforms — which is the point: pi accepts it.
    try testing.expect(toolCallConformsToSchema(allocator, calls[0], pi_edit_tools_json_test));
}

test "hoistMisplacedRequiredParams: a buried `path` survives the Gemma 4 parse path end to end" {
    // The same class through the REAL parse chain (Gemma's call:name{...} form —
    // this session's model). The raw pre-parse bytes were not dumped (the server
    // was not at --log-level debug), so the tag wrapper here is reconstructed;
    // the ARGUMENT SHAPE it produces is the verbatim captured one asserted above,
    // which is what the hoist actually operates on.
    const allocator = testing.allocator;
    const raw = "<|tool_call>call:edit{edits:[{oldText:<|\"|>old line<|\"|>,newText:<|\"|>new line<|\"|>," ++
        "path:<|\"|>us_presidents/generate_site.sh<|\"|>}]}<tool_call|>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try hoistMisplacedRequiredParams(allocator, calls, pi_edit_tools_json_test);
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits.array.items[0].object.get("path") == null);
    try testing.expectEqualStrings("old line", edits.array.items[0].object.get("oldText").?.string);
}

test "hoistMisplacedRequiredParams: a compliant call is byte-identical" {
    // The strong-model path: nothing may re-serialize when nothing was misplaced.
    const allocator = testing.allocator;
    const args =
        \\{"path":"a.js","edits":[{"oldText":"a","newText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: never touches a key the item schema DECLARES" {
    // Safety gate. A tool whose items legitimately carry their own `path` (a
    // multi-file edit) must never have it stripped out from under them — hoisting
    // there would DESTROY data, not repair it. The item schema declaring the key
    // is the proof that it belongs where it is.
    const allocator = testing.allocator;
    const multifile_schema =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"path":{"type":"string"},"oldText":{"type":"string"}},"required":["path","oldText"]}}},"required":["path","edits"]}}}]
    ;
    const args =
        \\{"edits":[{"path":"a.js","oldText":"a"},{"path":"b.js","oldText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, multifile_schema);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: items that DISAGREE on the value stay honest" {
    // Two items, two different paths, and the tool edits ONE file. There is no
    // correct hoist here — picking either would silently write the wrong file.
    // Leave it broken so the client's validation error reaches the model.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":"a.js"},{"oldText":"c","newText":"d","path":"b.js"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: a wrong-typed buried value is never hoisted" {
    // `path` is declared a string; an object spelled `path` inside the item is
    // not the missing param. Hoisting it would just move the schema violation.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":{"nested":"x"}}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "coerceToolArgsToSchema: quoted integer/number strings coerce to numeric" {
    // A model emitting `"limit":"5"` for an integer param → strict clients reject
    // "expected integer, got string". Coerce string→numeric per the schema.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"scale":{"type":"number"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"Read\",\"arguments\":{\"path\":\"a.txt\",\"limit\":\"5\",\"scale\":\"1.5\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 5), parsed.value.object.get("limit").?.integer);
    const scale = parsed.value.object.get("scale").?;
    try testing.expect(scale == .float or scale == .integer);
    // A string param is untouched.
    try testing.expectEqualStrings("a.txt", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: nullable union type [\"string\",\"null\"] coerces to the non-null member" {
    // Real schemas declare optional params as a type UNION. declaredJsonType
    // picks the first non-null member, so a boolean-spelled value under
    // ["boolean","null"] still coerces.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"flag":{"type":["boolean","null"]},"note":{"type":["string","null"]}}}}}]
    ;
    const raw = "<tool_call>\n<function=E>\n<parameter=flag>\nFalse\n</parameter>\n<parameter=note>\nhi\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("flag").? == .bool);
    try testing.expectEqual(false, parsed.value.object.get("flag").?.bool);
    try testing.expectEqualStrings("hi", parsed.value.object.get("note").?.string);
}

test "coerceToolArgsToSchema: an explicit JSON null satisfies any typed param (not coerced away)" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"limit":{"type":"integer"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"limit\":null}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    // null is a valid absent-value marker for any type — left byte-identical.
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: coercion is SHALLOW — values nested inside a param are untouched" {
    // Documented boundary: the layer coerces TOP-LEVEL params against the tool's
    // declared property types. It does NOT recurse into array/object element
    // properties (the tool schema's `items`/nested `properties` are the client's
    // to validate). A nested `"port":"8080"` stays a string — byte-identical.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"W","parameters":{"type":"object","properties":{"config":{"type":"object"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"W\",\"arguments\":{\"config\":{\"port\":\"8080\",\"tls\":\"false\"}}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: an unmodelled declared type leaves the value alone" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"x":{"type":"weird"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"x\":\"False\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "convertGemma4 dedups a repeated key + escapes a key with special chars (valid JSON)" {
    // Same class as the parseHermesToolCall dup-key / raw-name bugs, but in the
    // Gemma `call:name{...}` converter: a repeated key produced a DUPLICATE JSON
    // key (std.json → error.DuplicateField), and a key with a `"` was appended
    // raw → invalid JSON. Both must yield a valid JSON object.
    const allocator = testing.allocator;
    // Duplicate key.
    {
        const raw = "<|tool_call>call:shell{command:<|\"|>ls<|\"|>,command:<|\"|>pwd<|\"|>}<tool_call|>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expect(parsed.value == .object);
        try testing.expect(parsed.value.object.get("command") != null);
    }
    // Key containing a double-quote must not break the JSON.
    {
        const raw = "<|tool_call>call:shell{a\"b:<|\"|>v<|\"|>}<tool_call|>";
        const calls = try parseToolCalls(allocator, raw);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };
        if (calls) |cs| {
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, cs[0].arguments, .{});
            defer parsed.deinit();
            try testing.expect(parsed.value == .object);
        }
    }
}

test "parseXmlElementToolCall is already dup-key + escape safe (characterization)" {
    // The DSV4 XML-element path builds args via std.json.ObjectMap.put + Stringify,
    // which dedups keys (last wins) and escapes values — so it is structurally
    // immune to the dup-key/raw-interpolation class that bit parseHermesToolCall
    // and convertGemma4Object. This characterization test pins that guarantee.
    const allocator = testing.allocator;
    const raw = "<tool_calls>\n<tool_name>shell</tool_name>\n<command>echo \"a\"</command>\n<command>echo \"b\"</command>\n</tool_calls>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Valid JSON despite the duplicate <command> and the inner quotes.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.get("command") != null);
}

test "parseToolCalls: NO path emits invalid JSON args (final safety net)" {
    // Bullet-proof invariant: whatever a converter builds, the arguments a client
    // receives are ALWAYS valid JSON. convertGemma4Value copies a JSON-style
    // string verbatim, so a bad escape (`\q`) would otherwise emit invalid JSON.
    const allocator = testing.allocator;
    const cases = [_][]const u8{
        // Gemma regular-JSON-string value with an INVALID escape.
        "<|tool_call>call:write{path:\"a\\qb\"}<tool_call|>",
        // Gemma custom-string is fine, but mix a bad JSON string sibling.
        "<|tool_call>call:write{path:<|\"|>ok<|\"|>,x:\"y\\z\"}<tool_call|>",
    };
    for (cases) |text| {
        const calls = try parseToolCalls(allocator, text);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };
        if (calls) |cs| {
            for (cs) |tc| {
                const parsed = std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{}) catch {
                    std.debug.print("\nINVALID JSON args emitted: {s}\n", .{tc.arguments});
                    return error.InvalidJsonEmitted;
                };
                parsed.deinit();
            }
        }
    }
    // The `\q` case is REPAIRABLE (looseRepair doubles the bad backslash), so the
    // path survives as the literal `a\qb` — the safety net repairs, only falling
    // back to `{}` when even tolerant repair fails.
    {
        const calls = (try parseToolCalls(allocator, "<|tool_call>call:write{path:\"a\\qb\"}<tool_call|>")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("a\\qb", parsed.value.object.get("path").?.string);
    }
}

test "split content: trailing <channel|> close-marker spam never leaks (degenerate model output)" {
    // Live soak capture (record 2151, a Gemma reasoning variant): the model emitted
    // reasoning, one <channel|> close, a scrap of content, then SPAMMED 16 more bare
    // <channel|> close markers. The leading cut takes the FIRST close, and the
    // trailing-open cut only handles unclosed OPENERS — so the 16 stray CLOSE markers
    // leaked into visible content. A close marker is never valid at the tail of
    // content; trailing closers (and openers) are trimmed so <channel|> can't leak.
    const text = "Reasoning here about the file.\n<channel|>`glob` `./game.js`\n\n" ++
        "<channel|><channel|><channel|><channel|><channel|>";
    const content = splitThinkBlock(text, true, false).content;
    try testing.expect(std.mem.indexOf(u8, content, "<channel|>") == null);
    // The scrap of real content before the spam survives.
    try testing.expect(std.mem.indexOf(u8, content, "glob") != null);
}

test "split content: trailing </think> close spam never leaks" {
    const text = "<think>plan</think>The answer.</think></think></think>";
    const content = splitThinkBlock(text, true, false).content;
    try testing.expect(std.mem.indexOf(u8, content, "</think>") == null);
    try testing.expect(std.mem.indexOf(u8, content, "The answer.") != null);
}

test "split content: orphan Gemma <tool_call|> close never leaks (no-tag-leak)" {
    // Live 2026-07-16 soak (gemma-4-26B-A4B-it-qat-4bit, tools present, a "no
    // tools needed" probe at temp 0.7): the model degenerated into a bare
    // 1-token <tool_call|> CLOSE with NO <|tool_call> opener, so parseToolCalls
    // found no call and the orphan control token leaked as the ENTIRE visible
    // content (server response content == "<tool_call|>"). A tool CLOSE marker is
    // never valid at the tail of content — same class as the <channel|> spam.
    // parseToolCalls already extracted any real call before this split runs, so
    // any residual <tool_call|> here is orphan by construction.
    try testing.expectEqualStrings("", splitThinkBlock("<tool_call|>", true, false).content);
    try testing.expectEqualStrings("Sure.", splitThinkBlock("Sure.<tool_call|>", true, false).content);
    // Legit prose (no control token) is untouched.
    try testing.expectEqualStrings("2 + 2 = 4", splitThinkBlock("2 + 2 = 4", true, false).content);
}

test "splitThinkBlock content never keeps trailing channel close spam" {
    const text = "<|channel>thought\nplan<channel|>\n<|channel>\nThe answer.<channel|><channel|><channel|>";
    const split = splitThinkBlock(text, true, false);
    try testing.expect(std.mem.indexOf(u8, split.content, "<channel|>") == null);
    try testing.expect(std.mem.indexOf(u8, split.content, "The answer.") != null);
}

// ── Hy3 (hy_v3 / Hunyuan 3) suffixed think tags + tag-format tool calls ──

test "hy3 think: splitThinkBlock splits on </think:opensource>" {
    const out = "I reason here.</think:opensource>The answer is 4.";
    const split = splitThinkBlock(out, true, true);
    try testing.expectEqualStrings("I reason here.", split.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 4.", split.content);
}

test "hy3 think: literal suffixed opener + close" {
    const out = "<think:opensource>hmm</think:opensource>Answer.";
    const split = splitThinkBlock(out, true, false);
    try testing.expectEqualStrings("hmm", split.reasoning_content.?);
    try testing.expectEqualStrings("Answer.", split.content);
}

test "hy3 think: trailing suffixed close spam trimmed from content" {
    const out = "r</think:opensource>Answer.</think:opensource></think:opensource>";
    const split = splitThinkBlock(out, true, true);
    try testing.expectEqualStrings("Answer.", split.content);
}

test "hy3 think: dangling re-opened suffixed opener never leaks" {
    const out = "r</think:opensource>Answer.\n<think:opensource>\nnew thought";
    const split = splitThinkBlock(out, true, true);
    try testing.expectEqualStrings("Answer.", split.content);
    try testing.expect(std.mem.indexOf(u8, split.content, "<think") == null);
}

test "hy3 think: promptTailOpensThink recognizes the suffixed opener, not a closed no_think tail" {
    try testing.expect(promptTailOpensThink("<\xEF\xBD\x9Chy_Assistant:opensource\xEF\xBD\x9C><think:opensource>"));
    try testing.expect(!promptTailOpensThink("<\xEF\xBD\x9Chy_Assistant:opensource\xEF\xBD\x9C><think:opensource></think:opensource>"));
}

test "hy3 think: streamThinkGate splits on suffixed close and holds on a partial suffixed re-open" {
    // Template-opened thinking: reasoning streams in with no literal opener.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("partial reasoning", true, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("reasoning</think:opensource>ans", true, false));
    // Mid-text re-open arriving token by token: the partial tail must hold the
    // flush (leaking "<think:opensou" into visible text is the tag-leak class).
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("visible<think:opensou", false, true));
    // Prose that merely mentions "<think about it" is NOT a partial opener.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("let's <think about it", false, true));
}

test "parseToolCalls hy3 arg_key/arg_value format (single call, hostile value bytes)" {
    const raw = "<tool_calls:opensource>\n" ++
        "<tool_call:opensource>write_file<tool_sep:opensource>\n" ++
        "<arg_key:opensource>path</arg_key:opensource>\n" ++
        "<arg_value:opensource>jfk.txt</arg_value:opensource>\n" ++
        "<arg_key:opensource>content</arg_key:opensource>\n" ++
        "<arg_value:opensource>Hello \"world\"\nline2\ttabbed</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expect(!calls[0].inferred);
    // Arguments must be VALID JSON with both keys, hostile bytes escaped.
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("jfk.txt", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("Hello \"world\"\nline2\ttabbed", parsed.value.object.get("content").?.string);
}

test "parseToolCalls hy3: two calls in one wrapper" {
    const raw = "<tool_calls:opensource>\n" ++
        "<tool_call:opensource>alpha<tool_sep:opensource>\n" ++
        "<arg_key:opensource>a</arg_key:opensource>\n<arg_value:opensource>1</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n" ++
        "<tool_call:opensource>beta<tool_sep:opensource>\n" ++
        "<arg_key:opensource>b</arg_key:opensource>\n<arg_value:opensource>two</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("alpha", calls[0].name);
    try testing.expectEqualStrings("beta", calls[1].name);
}

test "parseToolCalls hy3: duplicate arg key — first wins, args stay valid JSON" {
    const raw = "<tool_call:opensource>edit<tool_sep:opensource>\n" ++
        "<arg_key:opensource>path</arg_key:opensource>\n<arg_value:opensource>a.txt</arg_value:opensource>\n" ++
        "<arg_key:opensource>path</arg_key:opensource>\n<arg_value:opensource>b.txt</arg_value:opensource>\n" ++
        "</tool_call:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.txt", parsed.value.object.get("path").?.string);
}

test "parseToolCalls hy3: truncated mid-value recovers name + closed args only" {
    // max_tokens cut the generation inside the second value — recover the call
    // with the CLOSED pair only; never salvage the partial content (the
    // truncated-opener class: a half-written file is worse than a retry).
    const raw = "<tool_calls:opensource>\n" ++
        "<tool_call:opensource>write_file<tool_sep:opensource>\n" ++
        "<arg_key:opensource>path</arg_key:opensource>\n<arg_value:opensource>x.txt</arg_value:opensource>\n" ++
        "<arg_key:opensource>content</arg_key:opensource>\n<arg_value:opensource>a very long novel that never clos";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("x.txt", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls hy3: dropped <tool_sep> + mangled key-close still recovers name AND args" {
    // Live 2026-07-16 RAW capture (pipenetwork/Hy3-REAP62 via the running server,
    // MLX_SERVE_RAW_DUMP_FILE): the pruned model drops <tool_sep> (closes the NAME
    // with </arg_value:opensource>) and closes the arg KEY block with
    // </arg_value:opensource> instead of </arg_key:opensource> — the VALUE block is
    // well-formed. Before the fix the strict parser bailed at the missing
    // <tool_sep> and the whole call LEAKED as content (finish_reason stop), so pi
    // saw bash({}) → "command required" and looped. The corruption is small and
    // regular, so recover the FULL call: name=bash, {"command":"ls -la"}.
    const raw = "<tool_calls:opensource>\n" ++
        "<tool_call:opensource>bash</arg_value:opensource>\n" ++
        "<arg_key:opensource>command</arg_value:opensource>\n" ++
        "<arg_value:opensource>ls -la</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls hy3: dropped singular <tool_call> opener (plural wrapper only) still recovers" {
    // Live 2026-07-16 RAW capture (pipenetwork/Hy3-REAP62 via the running server,
    // MLX_SERVE_RAW_DUMP_FILE): the pruned model emitted the PLURAL wrapper
    // <tool_calls:opensource> and jumped STRAIGHT to the NAME, dropping the
    // singular per-call <tool_call:opensource> opener the parser keys on — so the
    // whole (well-structured, complete) call LEAKED as content (finish_reason
    // stop). Same weak-model delimiter-drop class as the <tool_sep> drop above,
    // one delimiter over. The call is otherwise regular (name/key closed with
    // </arg_value>), so recover the FULL call incl. the quote-bearing content.
    const raw = "<tool_calls:opensource>\n" ++
        "write_file</arg_value:opensource>\n" ++
        "<arg_key:opensource>path</arg_value:opensource>\n" ++
        "<arg_value:opensource>page.html</arg_value:opensource>\n" ++
        "<arg_key:opensource>content</arg_value:opensource>\n" ++
        "<arg_value:opensource><meta charset=\"UTF-8\"><a href=\"/x\">L</a><div class=\"hero\">Hi</div></arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("page.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings(
        "<meta charset=\"UTF-8\"><a href=\"/x\">L</a><div class=\"hero\">Hi</div>",
        parsed.value.object.get("content").?.string,
    );
}

test "parseToolCalls hy3: suffixed think close before the wrapper still parses" {
    const raw = "planning the call</think:opensource>\n<tool_calls:opensource>\n" ++
        "<tool_call:opensource>list_files<tool_sep:opensource>\n" ++
        "<arg_key:opensource>dir</arg_key:opensource>\n<arg_value:opensource>.</arg_value:opensource>\n" ++
        "</tool_call:opensource>\n</tool_calls:opensource>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("list_files", calls[0].name);
}

// ── DeepSeek-V4 native DSML tool calls ─────────────────────────────────

fn freeParsedCalls(calls: []ParsedToolCall) void {
    for (calls) |tc| {
        testing.allocator.free(tc.name);
        testing.allocator.free(tc.arguments);
    }
    testing.allocator.free(calls);
}

test "parseToolCalls dsml: canonical two-call block, string and JSON params" {
    const raw = "I'll check the weather.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"get_weather\">\n" ++
        "<｜DSML｜parameter name=\"city\" string=\"true\">San Francisco</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "<｜DSML｜invoke name=\"get_time\">\n" ++
        "<｜DSML｜parameter name=\"tz\" string=\"true\">PST</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"San Francisco\",\"days\":3}", calls[0].arguments);
    try testing.expectEqualStrings("get_time", calls[1].name);
    try testing.expectEqualStrings("{\"tz\":\"PST\"}", calls[1].arguments);
    try testing.expect(!calls[0].inferred);
}

test "parseToolCalls dsml: dropped invoke close before the next call still parses both" {
    // Delimiter-drop tolerance (hy3 class): the model omits </｜DSML｜invoke>
    // between back-to-back calls.
    const raw = "<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"first\">\n" ++
        "<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>\n" ++
        "<｜DSML｜invoke name=\"second\">\n" ++
        "<｜DSML｜parameter name=\"b\" string=\"false\">true</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("first", calls[0].name);
    try testing.expectEqualStrings("{\"a\":\"x\"}", calls[0].arguments);
    try testing.expectEqualStrings("second", calls[1].name);
    try testing.expectEqualStrings("{\"b\":true}", calls[1].arguments);
}

test "parseToolCalls dsml: truncation salvages NAME + completed pairs, never fragments" {
    // Server-side cut mid-value: the unterminated pair is dropped (a half-
    // written file is worse than a re-issued write), completed pairs survive.
    const raw = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a.txt</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"content\" string=\"true\">first half of a long fi";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expectEqualStrings("{\"path\":\"/tmp/a.txt\"}", calls[0].arguments);
}

test "parseToolCalls dsml: rich string value runs to the confirmed close" {
    // Newlines, quotes, braces, `<` inside a string value — value ends ONLY
    // at the closing parameter tag (the Gemma dropped-quote class).
    const raw = "<｜DSML｜invoke name=\"bash\">\n" ++
        "<｜DSML｜parameter name=\"cmd\" string=\"true\">grep -r \"a < b\" src/\necho {done}</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("grep -r \"a < b\" src/\necho {done}", parsed.value.object.get("cmd").?.string);
}

test "parseToolCalls dsml: duplicate keys dedup (last wins) and invalid JSON value degrades to string" {
    const raw = "<｜DSML｜invoke name=\"t\">\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">old</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">new</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"n\" string=\"false\">not[json</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("new", parsed.value.object.get("k").?.string);
    try testing.expectEqualStrings("not[json", parsed.value.object.get("n").?.string);
    try testing.expectEqual(@as(usize, 2), parsed.value.object.count());
}

test "parseToolCalls dsml: zero-arg call and thinking prefix" {
    const raw = "thinking about it</think>Sure.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"list_files\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("list_files", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}

test "streamShouldBufferForTools: DSML marker and partial fullwidth prefixes hold" {
    try testing.expect(streamShouldBufferForTools("Sure.\n\n<｜DSML｜tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<｜DSML｜invoke name=\"x\">"));
    // partial fullwidth-bar prefix at the tail — next token may complete it
    try testing.expect(streamShouldBufferForTools("answer text <｜DSML"));
    try testing.expect(streamShouldBufferForTools("answer text <｜"));
    // plain prose with a fullwidth bar elsewhere must NOT hold
    try testing.expect(!streamShouldBufferForTools("the ｜ character is fun"));
}

test "loadChatConfig: template-less deepseek_v4 checkpoint gets the embedded DSV4 template" {
    // pipenetwork REAP builds ship NO chat_template (their python package
    // does its own encoding) — the generic fallback renders wrong-family
    // markers and the model degenerates into echoing them (live 2026-08-01:
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

test "renderChatTemplate: muse_glimmer shipped template renders byte-identical to python jinja2 (hermetic)" {
    // src/fixtures/muse_chat_template.jinja is the VERBATIM template
    // meta-models/Muse-Glimmer-30B ships; muse_render_reference.txt is python
    // jinja2's render of the exact message set below (regenerate with the
    // script in the fixture header if either changes). Byte equality pins:
    // <|start|>role<|message|>…<|eot|> framing, PRIOR-turn reasoning dropped
    // before the template runs (harmony convention — the reference is jinja2's
    // render of the message set with that field already nulled; the
    // current-turn keep arm is pinned by the drops-PRIOR-turn test below),
    // tool calls as to=<fn> ATEM blocks with dict arguments
    // (string params RAW, bools bare), tool results as
    // "<|start|>tool <name><|message|><tool_output …>", the namespaced tool
    // catalog + "# Valid recipients" line, and the bare "<|start|>assistant"
    // generation prompt (NO <|message|> — the model emits its own recipient).
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/muse_chat_template.jinja");
    const expected = @embedFile("fixtures/muse_render_reference.txt");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type": "function", "function": {"name": "weather.get", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    ;
    // Alphabetical argument keys on purpose: a sorted-map JSON layer must
    // produce the same parameter order as python's insertion-ordered dicts.
    const tc = [_]ToolCall{.{ .id = "call_1", .name = "weather.get", .arguments = "{\"city\": \"Paris\", \"detailed\": false, \"units\": \"celsius\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are a helpful assistant." },
        .{ .role = "user", .content = "What's the weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc, .reasoning_content = "Need the weather tool." },
        .{ .role = "tool", .content = "{\"temp\": 21}", .tool_call_id = "call_1" },
        .{ .role = "user", .content = "thanks, and in kelvin?" },
    };

    const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
    defer allocator.free(rendered);
    try testing.expectEqualStrings(expected, rendered);
}

test "renderChatTemplate: muse thinking-off no-tools commits the to=user channel" {
    // Thinking off must SKIP the reasoning pass, not generate-and-discard it:
    // spending the recipient header in the prompt makes a `to=self` segment
    // unreachable. Tools stay bare — a tool call is a `to=<fn>` header.
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/muse_chat_template.jinja");
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    const off = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(off);
    try testing.expect(std.mem.endsWith(u8, off, "<|start|>assistant to=user<|message|>"));

    const on = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(on);
    try testing.expect(std.mem.endsWith(u8, on, "<|start|>assistant"));

    const tools_json =
        \\[{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    ;
    const off_tools = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
    defer allocator.free(off_tools);
    try testing.expect(std.mem.endsWith(u8, off_tools, "<|start|>assistant"));
}

test "renderChatTemplate: muse template dedups a client-supplied reasoning directive" {
    // Upstream a4e59da (#35): a system prompt already carrying a "Reasoning
    // effort/strength" line used to get the template's own "Reasoning
    // strength: high." appended after it — two conflicting directives, ours
    // stomping the client's. The template now normalizes the spelling and
    // skips its own line. Rendered through OUR jinja.cpp because the fix
    // rides the `replace`/`lower` filters and the `in` operator — a silently
    // no-op filter keeps the duplicate while the render still succeeds.
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/muse_chat_template.jinja");
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "Be terse.\nReasoning effort: low." },
        .{ .role = "user", .content = "hi" },
    };
    const r = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(r);
    try testing.expect(std.mem.indexOf(u8, r, "Reasoning strength: low.") != null);
    try testing.expect(std.mem.indexOf(u8, r, "Reasoning effort") == null);
    const first = std.mem.indexOf(u8, r, "Reasoning strength").?;
    try testing.expect(std.mem.indexOfPos(u8, r, first + 1, "Reasoning strength") == null);
}

test "renderChatTemplate: thinking-off closes a template-opened think block (LFM2.5 class)" {
    // LFM2.5's template opens `<think>` unconditionally; thinking-off used to
    // mean generating the whole reasoning pass and discarding it. Closing the
    // block in the prompt skips it. Thinking ON keeps the opener untouched.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    const off = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(off);
    try testing.expect(std.mem.endsWith(u8, off, "<think></think>"));

    const on = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(on);
    try testing.expect(std.mem.endsWith(u8, on, "<think>"));
}

test "renderChatTemplate: continuation appends the partial reply into the OPEN assistant turn" {
    // Continuing a reply means the prompt ends mid-assistant-turn. The trailing
    // assistant message is a PREFIX to extend, not history: rendering it as
    // history and then letting add_generation_prompt open a second assistant
    // turn is what the server did before this existed, and the model answers
    // the doubled turn by starting over.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "count to three" },
        .{ .role = "assistant", .content = "one, two," },
    };

    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    // Ends INSIDE the turn, on the partial text — no close tag after it.
    try testing.expect(std.mem.endsWith(u8, cont, "<|im_start|>assistantone, two,"));
    // Exactly one assistant turn was opened.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cont, "<|im_start|>assistant"));
    // The partial appears once — as the prefill, not also as history.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cont, "one, two,"));

    // Same messages WITHOUT the flag stay history + a fresh turn, unchanged.
    const plain = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(plain);
    try testing.expect(std.mem.endsWith(u8, plain, "<|im_start|>assistant"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, plain, "<|im_start|>assistant"));
}

test "renderChatTemplate: continuation resumes in the CONTENT channel, not a think block" {
    // A template that opens `<think>` unconditionally (LFM2.5 class) would
    // otherwise put the partial reply — visible prose the user already read —
    // inside a reasoning block, and the continuation would come back as
    // reasoning. The same tail that thinking-off uses closes it first.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Hello ther" },
    };

    // Thinking ON is the hard case: nothing else would close the block.
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "<think></think>Hello ther"));
}

test "renderChatTemplate: a continuation prefill is trimmed of trailing whitespace" {
    // The tokenizer merges a trailing space with the word that follows it, so
    // a prompt ending in one splits differently than the same text mid-reply —
    // the hazard Anthropic's API rejects prefills outright for. Trim it: the
    // stored reply keeps its whitespace, only the PROMPT is trimmed.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Once upon a time \n" },
    };
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "Once upon a time"));
}

test "renderChatTemplate: continuation is declined when the last message is not an assistant" {
    // The flag describes the SHAPE of the request; a client that sets it on a
    // user-final conversation gets an ordinary turn rather than a prompt with
    // the user's own words committed into the assistant's mouth.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "<|im_start|>assistant"));
}

test "renderChatTemplate: muse commits its content channel before the partial reply" {
    // Muse's turn header is `<|start|>assistant to=<recipient><|message|>`, and
    // an uncommitted header means the model picks the channel — including
    // `to=self`, which would make the continuation reasoning.
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/muse_chat_template.jinja");
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|return|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Hello, I was say" },
    };
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, " to=user<|message|>Hello, I was say"));
}

test "renderChatTemplate: muse drops PRIOR-turn reasoning, keeps the current tool round's" {
    // Harmony (muse's format ancestor) drops analysis from turns before the
    // last user message. Without that, the app's reasoning round-trip renders
    // every old chain — a loop-cut tail included — as a to=self history
    // segment the model reads back on every later turn (live 2026-08-10: one
    // repetition-primed turn poisoned the rest of the chat).
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/muse_chat_template.jinja");
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    // Prior turn: assistant reasoning BEFORE the last user message → dropped;
    // its content survives.
    const prior = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "hello there", .reasoning_content = "waffling chain" },
        .{ .role = "user", .content = "and again?" },
    };
    const r1 = try renderChatTemplate(allocator, &prior, &config, null, null, true, null, false);
    defer allocator.free(r1);
    try testing.expect(std.mem.indexOf(u8, r1, "to=self") == null);
    try testing.expect(std.mem.indexOf(u8, r1, "waffling chain") == null);
    try testing.expect(std.mem.indexOf(u8, r1, "hello there") != null);

    // Current turn: a tool round AFTER the last user message keeps its chain —
    // that is where the model needs it.
    const tools_json =
        \\[{"type": "function", "function": {"name": "weather.get", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "call_1", .name = "weather.get", .arguments = "{\"city\": \"Paris\"}" }};
    const current = [_]Message{
        .{ .role = "user", .content = "weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc, .reasoning_content = "Need the weather tool." },
        .{ .role = "tool", .content = "{\"temp\": 21}", .tool_call_id = "call_1" },
    };
    const r2 = try renderChatTemplate(allocator, &current, &config, tools_json, null, true, null, false);
    defer allocator.free(r2);
    try testing.expect(std.mem.indexOf(u8, r2, "<|start|>assistant to=self<|message|>Need the weather tool.<|eom|>") != null);

    // A non-muse template that persists history reasoning (laguna class) is
    // untouched — the round-trip rule stays per-family.
    const generic_tpl = "{%- for m in messages -%}{%- if m.get('reasoning_content') -%}<think>{{ m['reasoning_content'] }}</think>{%- endif -%}{{ m['role'] }}: {{ m['content'] }}\n{%- endfor -%}";
    var generic = ChatConfig{
        .chat_template = generic_tpl,
        .bos_token = "",
        .eos_token = "",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const r3 = try renderChatTemplate(allocator, &prior, &generic, null, null, true, null, false);
    defer allocator.free(r3);
    try testing.expect(std.mem.indexOf(u8, r3, "<think>waffling chain</think>") != null);
}

test "splitMuseChannels: headerless first segment (prompt-committed header) is content" {
    // With `to=user<|message|>` committed in the prompt, the model's first
    // bytes are BODY. If it then re-opens a segment, the leading text must
    // survive as content, not vanish.
    const split = splitMuseChannels("answer<|eom|><|start|>assistant to=self<|message|>notes").?;
    try testing.expectEqualStrings("answer", split.content);
    try testing.expectEqualStrings("notes", split.reasoning_content.?);
    // A truncated bare header is still never content.
    const hdr = splitMuseChannels(" to=self<|eom|><|start|>assistant to=user<|message|>hi").?;
    try testing.expectEqualStrings("hi", hdr.content);
    try testing.expect(hdr.reasoning_content == null);
}

test "splitMuseChannels: to=self reasoning + to=user content, headers stripped" {
    // The exact shape muse emits after the prompt's `<|start|>assistant`.
    const text = " to=self<|message|>Let me think.<|eom|><|start|>assistant to=user<|message|>The answer is 4.";
    const split = splitThinkBlock(text, true, false);
    try testing.expectEqualStrings("Let me think.", split.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 4.", split.content);

    // Direct answer: no reasoning segment at all.
    const direct = splitThinkBlock(" to=user<|message|>Hi there.", true, false);
    try testing.expect(direct.reasoning_content == null);
    try testing.expectEqualStrings("Hi there.", direct.content);

    // Length-truncated mid-thought: reasoning so far, never content.
    const cut = splitThinkBlock(" to=self<|message|>Half a thou", true, false);
    try testing.expectEqualStrings("Half a thou", cut.reasoning_content.?);
    try testing.expectEqualStrings("", cut.content);

    // A trailing partial header after <|eom|> is dropped, not content.
    const partial = splitThinkBlock(" to=self<|message|>R done.<|eom|><|start|>assistant to=us", true, false);
    try testing.expectEqualStrings("R done.", partial.reasoning_content.?);
    try testing.expectEqualStrings("", partial.content);

    // Non-muse text (no <|message|> marker) is never claimed.
    const plain = splitThinkBlock("just prose with to=user in it", true, false);
    try testing.expect(plain.reasoning_content == null);
    try testing.expectEqualStrings("just prose with to=user in it", plain.content);
}

test "splitMuseChannels: tool segment markup never rides out as content (wrapper cut)" {
    const text = " to=self<|message|>Need the tool.<|eom|><|start|>assistant to=weather.get<|message|><atem:function_calls>\n<atem:invoke name=\"weather.get\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    // Display split: the ATEM markup is CUT from content.
    const split = splitThinkBlock(text, true, false);
    try testing.expectEqualStrings("Need the tool.", split.reasoning_content.?);
    try testing.expectEqualStrings("", split.content);
    // KeepingMarkup split: the markup survives for the tool parser.
    const keeping = splitThinkBlockKeepingMarkup(text, true, false);
    try testing.expect(std.mem.indexOf(u8, keeping.content, "<atem:invoke") != null);
}

test "parseAtemToolCalls: canonical block — raw strings, spelled types, schema untouched" {
    const allocator = testing.allocator;
    const text = " to=weather.get<|message|><atem:function_calls>\n<atem:invoke name=\"weather.get\">\n<atem:parameter name=\"city\">Paris, France</atem:parameter>\n<atem:parameter name=\"detailed\">false</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n<atem:parameter name=\"tags\">[\"trip\", \"eu\"]</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("weather.get", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try testing.expectEqualStrings("Paris, France", obj.get("city").?.string);
    try testing.expectEqual(false, obj.get("detailed").?.bool);
    try testing.expectEqual(@as(i64, 3), obj.get("days").?.integer);
    try testing.expectEqual(@as(usize, 2), obj.get("tags").?.array.items.len);
}

test "parseAtemToolCalls: multiline string values keep exact bytes (no trimming)" {
    const allocator = testing.allocator;
    const text = "<atem:invoke name=\"fs.write\">\n<atem:parameter name=\"content\">line one\n  \"quoted\" line\n</atem:parameter>\n</atem:invoke>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    // Exact bytes incl. the trailing newline — the reference does not strip.
    try testing.expectEqualStrings("line one\n  \"quoted\" line\n", parsed.value.object.get("content").?.string);
}

test "parseAtemToolCalls: truncation salvages NAME + completed params, never fragments" {
    const allocator = testing.allocator;
    // Cut mid-value: the completed param ships, the fragment is dropped.
    {
        const text = "<atem:invoke name=\"weather.get\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n<atem:parameter name=\"units\">celsi";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("weather.get", calls[0].name);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("Paris", parsed.value.object.get("city").?.string);
        try testing.expect(parsed.value.object.get("units") == null);
    }
    // Cut right after the opener: NAME + {}.
    {
        const text = "<atem:function_calls>\n<atem:invoke name=\"weather.get\">";
        const calls = (try parseToolCalls(allocator, text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("weather.get", calls[0].name);
        try testing.expectEqualStrings("{}", calls[0].arguments);
    }
}

test "parseAtemToolCalls: parallel invokes each parse; dropped wrapper still parses" {
    const allocator = testing.allocator;
    const text = "<atem:invoke name=\"a.one\">\n<atem:parameter name=\"x\">1</atem:parameter>\n</atem:invoke>\n<atem:invoke name=\"a.two\">\n<atem:parameter name=\"y\">two</atem:parameter>\n</atem:invoke>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("a.one", calls[0].name);
    try testing.expectEqualStrings("a.two", calls[1].name);
}

test "muse streaming gate: headers hold, self holds then splits, user splits" {
    // Growing bare header — never flush the ` to=…` text. The sub-"to="
    // prefixes are the live 2026-08-10 leak (" to" shipped as a text delta
    // on a thinking-off /v1/messages stream).
    try testing.expect(museStreamVerdict(" t") == StreamThinkGate.hold_thinking);
    try testing.expect(museStreamVerdict(" to") == StreamThinkGate.hold_thinking);
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2(" to", true, false, false));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2(" to=se", true, false, false));
    // Reasoning in progress streams via hold_thinking; <|eom|> splits.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2(" to=self<|message|>thinking…", true, false, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate2(" to=self<|message|>done<|eom|>", true, false, false));
    // Direct answer resolves to a split (the handler strips the header once).
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate2(" to=user<|message|>Hi", true, false, false));
    // A next-segment header mid-stream holds again.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2(" to=self<|message|>d<|eom|><|start|>assistant to=w", true, false, false));
    // Prose that merely starts with to= releases at the first non-recipient byte.
    try testing.expect(museStreamVerdict("to=5 grams of flour") == null);
    // The memoized gate agrees on the muse arms.
    var scan = ThinkScan{};
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGateScan(" to=self<|message|>done<|eom|>", true, false, false, &scan));
}

test "muse streamShouldBufferForTools: tool headers hold, self/user release" {
    // Unresolved header could still name a tool — including sub-"to="
    // prefixes (the live " to" text-delta leak).
    try testing.expect(streamShouldBufferForTools(" t"));
    try testing.expect(streamShouldBufferForTools(" to"));
    try testing.expect(streamShouldBufferForTools(" to=weath"));
    try testing.expect(streamShouldBufferForTools(" to=self<|message|>r<|eom|><|start|>assistant to=wea"));
    // Resolved tool recipient keeps holding (body is ATEM payload).
    try testing.expect(streamShouldBufferForTools(" to=weather.get<|message|>"));
    // ATEM markup anywhere holds.
    try testing.expect(streamShouldBufferForTools("<atem:function_calls>"));
    try testing.expect(streamShouldBufferForTools("prefix <atem"));
    // Resolved self/user releases so reasoning/content can stream.
    try testing.expect(!streamShouldBufferForTools(" to=self<|message|>thinking about tools"));
    try testing.expect(!streamShouldBufferForTools(" to=user<|message|>plain answer"));
    // Ordinary prose that starts with to= releases at the first space.
    try testing.expect(!streamShouldBufferForTools("to=5 grams"));
}

test "museThinkOpenerAt: self opens reasoning, others answer directly, partials grow" {
    try testing.expectEqual(MuseThinkOpener.growing, museThinkOpenerAt(" to=se"));
    switch (museThinkOpenerAt(" to=self<|message|>")) {
        .self_opened => |hl| try testing.expectEqual(@as(usize, " to=self<|message|>".len), hl),
        else => return error.TestUnexpectedResult,
    }
    switch (museThinkOpenerAt(" to=user<|message|>Hi")) {
        .direct => |hl| try testing.expectEqual(@as(usize, " to=user<|message|>".len), hl),
        else => return error.TestUnexpectedResult,
    }
    try testing.expectEqual(MuseThinkOpener.not_muse, museThinkOpenerAt("Okay, the user wants X."));
    // Complete next-segment header skip; incomplete returns null.
    try testing.expectEqual(@as(usize, "<|start|>assistant to=user<|message|>".len), museContentHeaderSkip("<|start|>assistant to=user<|message|>Hello").?);
    try testing.expect(museContentHeaderSkip("<|start|>assistant to=us") == null);
}

test "serializeExtraContext: muse maps effort onto reasoning_strength" {
    const allocator = testing.allocator;
    var muse = ChatConfig{
        .chat_template = "…{%- set rs = reasoning_strength if reasoning_strength is defined else 'high' -%}…",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const cases = [_]struct { thinking: bool, effort: ?[]const u8, want: []const u8 }{
        .{ .thinking = true, .effort = null, .want = "\"reasoning_strength\":\"high\"" },
        .{ .thinking = true, .effort = "low", .want = "\"reasoning_strength\":\"low\"" },
        .{ .thinking = true, .effort = "medium", .want = "\"reasoning_strength\":\"medium\"" },
        .{ .thinking = true, .effort = "xhigh", .want = "\"reasoning_strength\":\"high\"" },
        .{ .thinking = false, .effort = null, .want = "\"reasoning_strength\":\"low\"" },
    };
    for (cases) |case| {
        const r = try serializeExtraContext(allocator, &muse, case.thinking, case.effort);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, case.want) != null);
    }
    // Non-muse templates never see the key.
    var plain = ChatConfig{
        .chat_template = "{{ messages }}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const r = try serializeExtraContext(allocator, &plain, true, null);
    defer allocator.free(r);
    try testing.expect(std.mem.indexOf(u8, r, "reasoning_strength") == null);
}

test "renderChatTemplate: dsv4 template renders tools + DSML history + tool_result (hermetic)" {
    // src/fixtures/dsv4_chat_template.jinja is OUR faithful transcription of
    // DeepSeek-V4's encoding_dsv4.py (the checkpoint ships none): TOOLS block
    // into the system position, assistant history tool calls as DSML
    // invoke/parameter markup (string params raw, JSON params via tojson —
    // whose default ", "/": " separators match python json.dumps), tool role
    // as <tool_result> user turns (consecutive results merge), thinking_mode
    // chat|thinking with the reference's drop-thinking rule (reasoning kept
    // only for assistant turns AFTER the last user/tool message).
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/dsv4_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_weather", .arguments = "{\"city\": \"Paris\", \"days\": 3}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "Sunny, 22C", .tool_call_id = "tc_0" },
    };

    // Thinking OFF (chat mode): assistant turns open with the bare </think>.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "You are helpful.\n\n## Tools") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "\"<｜DSML｜tool_calls>\" block") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "\"name\": \"get_weather\"") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜User｜>Weather in Paris?") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "</｜DSML｜tool_calls><｜end▁of▁sentence｜>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜User｜><tool_result>Sunny, 22C</tool_result>") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<｜Assistant｜></think>"));
        // No fallback-format markers (a raise inside the template silently falls back).
        try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
    }
    // Thinking ON with TOOLS: the reference DISABLES drop_thinking whenever
    // tools are declared (agent sessions keep reasoning on every assistant
    // turn — the laguna round-trip class), so the tool-calling assistant
    // renders <think>…</think> even though a tool result follows it.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.endsWith(u8, rendered, "<｜Assistant｜><think>"));
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜Assistant｜><think></think>\n\n<｜DSML｜tool_calls>") != null);
        // 0731's reasoning-effort preamble must NOT appear at the serializer's
        // dsv4 default (the reference default is "low" = nothing). hy3's
        // blanket "high" would have prepended it to every thinking request.
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning Effort:") == null);
    }
    // Client-supplied effort reaches the render MAPPED (dsv4EffortFor): the
    // raw CLIENT vocabulary goes in ("xhigh"), the template sees "max" and
    // prepends the max preamble. This is the end-to-end pin the unit tests
    // can't give — the string must survive renderChatTemplate →
    // serializeExtraContext → Jinja.
    {
        const user_only = [_]Message{.{ .role = "user", .content = "prove it" }};
        const r_max = try renderChatTemplate(allocator, &user_only, &config, null, null, true, "xhigh", false);
        defer allocator.free(r_max);
        try testing.expect(std.mem.indexOf(u8, r_max, "Reasoning Effort: Beyond maximum") != null);
        const r_high = try renderChatTemplate(allocator, &user_only, &config, null, null, true, "high", false);
        defer allocator.free(r_high);
        try testing.expect(std.mem.indexOf(u8, r_high, "Reasoning Effort: Absolute maximum") != null);
        // Outside thinking mode the preamble never applies, whatever the effort.
        const r_chat = try renderChatTemplate(allocator, &user_only, &config, null, null, false, "high", false);
        defer allocator.free(r_chat);
        try testing.expect(std.mem.indexOf(u8, r_chat, "Reasoning Effort:") == null);
    }
    // Old-turn reasoning DROPS (the reference's drop_thinking): an assistant
    // turn with a LATER user message renders without its reasoning.
    {
        const hist = [_]Message{
            .{ .role = "user", .content = "Why is the sky blue?" },
            .{ .role = "assistant", .content = "Rayleigh scattering.", .reasoning_content = "Shorter wavelengths scatter more." },
            .{ .role = "user", .content = "And sunsets?" },
        };
        const rendered = try renderChatTemplate(allocator, &hist, &config, null, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜Assistant｜></think>Rayleigh scattering.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "Shorter wavelengths") == null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<｜Assistant｜><think>"));
    }
    // No system message + tools: the reference delivers tools through a
    // system/developer turn and renders `system_content + "\n\n" + tools`, so
    // an EMPTY system turn still emits the two newlines (measured against
    // 0731's encoder — it silently DROPS tools attached to a user message, so
    // an empty system turn is its only way to say "tools, no system prompt").
    // Clients routinely send user+tools with no system message, and dropping
    // that separator shifts every token after the bos away from what the
    // model was trained on. Pinned byte-exact by tests/dsv4_template_ab.py.
    {
        const no_sys = [_]Message{
            .{ .role = "user", .content = "hi" },
        };
        const rendered = try renderChatTemplate(allocator, &no_sys, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.startsWith(u8, rendered, "<｜begin▁of▁sentence｜>\n\n## Tools"));
        try testing.expect(std.mem.indexOf(u8, rendered, "tool name and parameter schemas to invoke tool calls.\n<｜User｜>hi") != null);
    }
}

test "ThinkScan: incremental gate matches the fresh gate at every prefix length" {
    // The gate is O(buffer) per token today (measured 47.99 us/token at 113 KB
    // with no close in the buffer), i.e. O(n^2) over a reasoning block. A
    // cursor makes it O(new bytes) — but only if it is EXACT, and the failure
    // a naive cursor introduces is a marker split across two token arrivals.
    // Feeding prefix-by-prefix IS that case, at every possible split point.
    const corpus = [_][]const u8{
        "plain visible prose with no markup at all",
        "<think>reasoning here</think>the answer",
        "<think:opensource>hy3 reasoning</think:opensource>answer",
        // A </think> inside a tool ARGUMENT is payload, not a block close.
        "<think>plan<tool_call>f<arg_key>k</arg_key><arg_value>closes with </think> inside</arg_value></tool_call>done</think>visible",
        "<|channel>thought hidden<channel|>visible",
        "prose where a < b and nothing else happens",
        "trailing partial opener at the end <thi",
        "trailing partial close </thin",
        // A suffix longer than the 32-byte tail window the partial-scan uses:
        // the fresh scan still recognizes it, so the cursor must too.
        "<think>x</think:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>tail",
        "<|message_thinking|>inkling thought<|end_message|>after",
        "<|message_text|>inkling visible",
    };
    const flags = [_][2]bool{ .{ true, false }, .{ false, false }, .{ false, true } };
    for (corpus) |full| {
        for (flags) |f| {
            const enable_thinking = f[0];
            const prompt_opened = f[1];
            var scan: ThinkScan = .{};
            var n: usize = 0;
            while (n <= full.len) : (n += 1) {
                const buf = full[0..n];
                const want = streamThinkGate2(buf, enable_thinking, false, prompt_opened);
                const got = streamThinkGateScan(buf, enable_thinking, false, prompt_opened, &scan);
                if (want != got) {
                    std.debug.print("prefix {d} of \"{s}\" (think={}, opened={}): {any} != {any}\n", .{ n, full, enable_thinking, prompt_opened, got, want });
                    return error.IncrementalGateDiverged;
                }
            }
        }
    }
}

test "ThinkScan: per-token work is FLAT in buffer size, not linear" {
    // The whole point. A smaller constant is not the fix — the re-examined
    // span must not grow with the buffer, or a 32K-token thought pays 8x what
    // the 113 KB bench measured.
    var scan: ThinkScan = .{};
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(testing.allocator);
    var max_span: usize = 0;
    var i: usize = 0;
    while (i < 4000) : (i += 1) {
        // A think block that never closes — the expensive arm of the bench.
        try buf.appendSlice(testing.allocator, if (i == 0) "<think>" else "token words and more ");
        _ = streamThinkGateScan(buf.items, true, false, false, &scan);
        if (i > 0 and scan.last_scan_span > max_span) max_span = scan.last_scan_span;
    }
    try testing.expect(buf.items.len > 60_000);
    // Bounded by one token's bytes plus the trailing partial-marker candidate.
    try testing.expect(max_span <= 64);

    // A growing partial marker at the tail is re-examined, and only it.
    var scan2: ThinkScan = .{};
    var buf2 = std.ArrayList(u8).empty;
    defer buf2.deinit(testing.allocator);
    try buf2.appendSlice(testing.allocator, "<think>");
    var j: usize = 0;
    while (j < 500) : (j += 1) try buf2.appendSlice(testing.allocator, "filler ");
    _ = streamThinkGateScan(buf2.items, true, false, false, &scan2);
    for ("</think:opensou") |c| {
        try buf2.append(testing.allocator, c);
        _ = streamThinkGateScan(buf2.items, true, false, false, &scan2);
        try testing.expect(scan2.last_scan_span <= 64);
    }
}

test "ThinkScan: reset() re-arms the cursor when the stream buffer is cleared" {
    // The buffer is NOT append-only — the stream loop trims and resets it at
    // every emit. A stale cursor then points past the end or into different
    // bytes, which is the second failure mode a naive cursor introduces.
    var scan: ThinkScan = .{};
    const first = "<think>reasoning</think>";
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGateScan(first, true, false, false, &scan));
    scan.reset();
    // Same scan struct, fresh (shorter) buffer: must not carry the old close.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGateScan("<think>more", true, false, false, &scan));
}

test "the streaming handlers route the think gate through a persistent ThinkScan" {
    // A memoized scan nobody threads through is a silent no-op that is
    // output-identical to the unmemoized gate — the exact shape of the
    // hardcoded-use_drafter=false class. Both streaming surfaces must hold a
    // scan beside their buffer AND reset it where they clear that buffer.
    // Needles are ++-split so this test's own source cannot satisfy the scan.
    const src = @embedFile("server.zig");
    const call = "streamThinkGateScan(buf, enable_thinking, think_closed, prompt_opened_think, &think" ++ "_scan)";
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, src, call));
    // The unmemoized 4-arg gate must be GONE from the streaming paths.
    const old = "chat_mod.streamThinkGate2(" ++ "buf,";
    try testing.expect(std.mem.indexOf(u8, src, old) == null);
    // One reset per buffer clear in those handlers.
    const reset = "think_scan" ++ ".reset();";
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, src, reset));
}

test "bench: streaming think gate scans BYTES, memoized vs fresh" {
    // The 5.10 acceptance bar is that per-token work is FLAT in buffer size,
    // not merely smaller. Wall clock is unavailable in a hermetic test under
    // Zig 0.17 (clocks live under std.Io) and would be noise anyway — bytes
    // examined is the thing that was quadratic, and it is deterministic.
    // Mirrors the 4b.4 harness: 4000 tokens growing to ~104 KB.
    const a = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(a);
    var scan: ThinkScan = .{};
    try buf.appendSlice(a, "<think>");
    var memo_bytes: u64 = 0;
    var fresh_bytes: u64 = 0;
    var i: usize = 0;
    while (i < 4000) : (i += 1) {
        try buf.appendSlice(a, "some reasoning words here ");
        const g = streamThinkGateScan(buf.items, true, false, false, &scan);
        try testing.expectEqual(StreamThinkGate.hold_thinking, g);
        memo_bytes += scan.last_scan_span;
        // What the unmemoized gate re-reads every token: the whole buffer, and
        // more than once (open scan + close scan + two substring scans).
        fresh_bytes += buf.items.len;
    }
    std.debug.print(
        "[think-gate] {d} tokens to {d} KB: fresh >= {d} KB scanned, memoized {d} KB ({d}x)\n",
        .{ 4000, buf.items.len / 1024, fresh_bytes / 1024, memo_bytes / 1024, fresh_bytes / @max(memo_bytes, 1) },
    );
    // Linear-in-total, not quadratic: the memoized scan reads each byte a
    // bounded number of times.
    try testing.expect(memo_bytes < buf.items.len * 4);
}

test "streaming content agrees with the non-streaming split on leading whitespace" {
    // LFM2.5 class: the template opens <think> unconditionally, so the first
    // generated token is the newline pair after the closer. The non-streaming
    // split drops it; the stream must reach the same bytes.
    const out = "\n\nOne, two, three.";
    const expected = splitThinkBlock(out, true, false).content;
    try testing.expectEqualStrings("One, two, three.", expected);

    const tokens = [_][]const u8{ "\n\n", "One", ", two", ", three." };
    var streamed: std.ArrayList(u8) = .empty;
    defer streamed.deinit(testing.allocator);
    var content_started = false;
    for (tokens) |tt| {
        const emit = streamContentLead(tt, content_started);
        if (emit.len == 0) continue;
        content_started = true;
        try streamed.appendSlice(testing.allocator, emit);
    }
    try testing.expectEqualStrings(expected, streamed.items);
}
test "streamContentLead never touches whitespace inside the answer" {
    // Only the LEAD is suppressed: once anything visible has been sent, a
    // newline token is ordinary content (markdown paragraphs, code blocks).
    try testing.expectEqualStrings("\n\n", streamContentLead("\n\n", true));
    try testing.expectEqualStrings("  x", streamContentLead("  x", true));
    // And a lead that is not whitespace is untouched.
    try testing.expectEqualStrings("Hello", streamContentLead("Hello", false));
    // A token is never cut in half: one carrying whitespace AND text rides
    // whole, so its logprobs entry still describes bytes that reached content.
    try testing.expectEqualStrings("\nHello", streamContentLead("\nHello", false));
}
