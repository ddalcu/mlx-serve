//! Format corpus — hermetic, cross-family format-correctness tests.
//!
//! A table of REAL captured model outputs (plus a few minimal synthetic
//! variants of real failures) run through the pure post-processing layer:
//! `chat.splitThinkBlock` / `chat.stripThinkBlock` / `chat.parseToolCalls` —
//! and back through the INPUT layer (`chat.serializeMessagesJson`), since
//! every output re-enters the next request's history.
//! No model weights, no server — runs in CI on every `zig build test`.
//!
//! Run just this corpus:
//!     zig build test -Dtest-filter="format corpus"
//!
//! ## Harvesting new entries
//!
//! 1. Start the server with `--log-level debug`. Every tools-enabled request
//!    dumps the model's raw output before tool parsing:
//!        raw generated text before tool parse (NNNb): <text>
//!    (two sites in src/server.zig — streaming and non-streaming). The inline
//!    dump caps at 4KB; for mega-tool-calls also set
//!    MLX_SERVE_RAW_DUMP_FILE=<abs path> to write the FULL pre-parse buffer
//!    of the last streamed tools request (how the 2026-07-03 timeout-guillotine
//!    class was captured).
//! 2. Grep the server log for that line (or for the misbehaving output).
//! 3. Paste the raw text into a new `Expect` entry below with the family it
//!    came from and what SHOULD happen. The universal invariants (no control
//!    tags in visible content, tool args must be valid JSON) apply
//!    automatically; add per-entry expectations for the specific behavior.
//!
//! Origin: the 2026-06-10 live pi-agent session caught five format bugs unit
//! tests missed. Three are pure-function bugs pinned here (truncated
//! template-opened thinking leaking into content; a trailing raw
//! `<|channel>thought` tag leaking into visible output; an unterminated
//! `<|"|>` string swallowing the args' closing brace — a file literally named
//! "mlx_pi1.html`}" reached disk). The other two (final answer misfiled as
//! reasoning_content in tools+thinking streams; omitted max_tokens defaulting
//! to 256) live in server.zig request handling and are pinned by
//! tests/test_format_matrix.sh checks 4 and 7 plus tests/test_thinking_split.sh.

const std = @import("std");
const testing = std.testing;
const chat = @import("chat.zig");

const Expect = struct {
    family: []const u8,
    name: []const u8,
    raw: []const u8,
    /// Request had thinking enabled (selects splitThinkBlock vs stripThinkBlock).
    thinking: bool = false,
    /// Generation prompt ended with a template-injected think opener
    /// (Qwen 3.5/3.6 render `…assistant\n<think>\n`).
    opened_by_template: bool = false,
    content_contains: ?[]const u8 = null,
    content_exact: ?[]const u8 = null,
    reasoning_contains: ?[]const u8 = null,
    /// Expected name of the FIRST parsed tool call.
    tool_name: ?[]const u8 = null,
    /// Expected key/value (string-typed) in the first call's arguments.
    tool_arg_key: ?[]const u8 = null,
    tool_arg_value: ?[]const u8 = null,
    /// Assert parseToolCalls returns null (prose that merely looks tag-ish).
    no_tool_calls: bool = false,
    /// Expected number of parsed tool calls (parallel-call outputs).
    tool_count: ?usize = null,
    /// Expected value of `tool_arg_key` in the LAST parsed call (asserts
    /// parallel calls each kept their own arguments).
    last_tool_arg_value: ?[]const u8 = null,
    /// The tools the request declared (OpenAI shape, exactly what server.zig
    /// threads to the parse sites as `tools_json`). When set, the corpus runs
    /// `chat.coerceToolArgsToSchema` and enforces the universal
    /// declared-type invariant below.
    tools_json: ?[]const u8 = null,
    /// Expected BOOLEAN-typed argument in the first call (schema entries only).
    tool_bool_key: ?[]const u8 = null,
    tool_bool_value: ?bool = null,
    /// Assert this key is ABSENT from the first call's arguments. Used by the
    /// truncation-salvage entries: a value the cut landed inside is a FRAGMENT
    /// and must be dropped, never shipped as a real argument (a client executes
    /// what it receives — fragmentary content writes a corrupt file
    /// "successfully").
    tool_arg_absent: ?[]const u8 = null,
};

/// Claude Code's Edit tool, post `server.buildOpenAIToolsJson`. `replace_all`
/// is the boolean; everything else is a string. Shared by the entries below so
/// the string-vs-boolean confusion is exercised in BOTH directions.
const edit_tool_schema =
    \\[{"type":"function","function":{"name":"Edit","description":"Edit a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean","default":false}},"required":["file_path","old_string","new_string"]}}}]
;

/// pi's `edit` tool, verbatim from its own schema (@earendil-works/pi-coding-agent
/// dist/core/tools/edit.js). Two facts the entries below lean on: the tag formats
/// carry no type information, so the whole `edits` array arrives as a STRING; and
/// `path` is required at the TOP level while the item schema declares only
/// oldText/newText — which is what makes a buried `path` provably misplaced.
const pi_edit_tool_schema =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to edit (relative or absolute)"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]},"description":"One or more targeted replacements."}},"required":["path","edits"]}}}]
;

/// Weather tool with a boolean arg — used by the LIVE Hy3 capture to pin the
/// tag-format string→bool schema coercion.
const weather_tool_schema =
    \\[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"celsius":{"type":"boolean"}},"required":["city"]}}}]
;

/// pi-style write/read pair — used by the hallucinated-raw-JSON (George
/// Washington) entries to exercise the inferred-name-must-be-declared filter.
const write_read_tools_schema =
    \\[{"type":"function","function":{"name":"write","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
;

const corpus = [_]Expect{
    // ── Qwen 3.5/3.6 (<think> family, template-injected opener) ─────────────
    .{
        .family = "qwen",
        .name = "full think round, template-opened (close tag only in output)",
        .raw = "The user wants 17*23. 17*20=340, 17*3=51, total 391.</think>\n\n17 × 23 = **391**.",
        .thinking = true,
        .opened_by_template = true,
        .content_contains = "391",
        .reasoning_contains = "17*20=340",
    },
    .{
        // BUG 1 (2026-06-10 pi session): generation hit max_tokens before
        // `</think>`, so the output has NO think tags at all. Pre-fix the
        // truncated reasoning was dumped into visible content.
        .family = "qwen",
        .name = "template-opened truncated thinking stays out of content",
        .raw = "The user asks for 17*23. Let me compute: 17*20 = 340, then 17*3 =",
        .thinking = true,
        .opened_by_template = true,
        .content_exact = "",
        .reasoning_contains = "17*20 = 340",
    },
    .{
        // Prose answer that ENDS by opening a new, unclosed think block.
        .family = "qwen",
        .name = "trailing <think> opener truncated out of content",
        .raw = "The answer is 391.\n<think>wait, should I double-check the carry",
        .thinking = true,
        .content_exact = "The answer is 391.",
        .reasoning_contains = "double-check",
    },
    .{
        .family = "qwen",
        .name = "thinking-off prose passes through verbatim",
        .raw = "17 × 23 = 391.",
        .content_exact = "17 × 23 = 391.",
    },
    .{
        // Raw JSON tool call with no wrapper tags (Qwen emits this when the
        // template's <tool_call> markers get sampled away).
        .family = "qwen",
        .name = "raw JSON tool call, no wrapper tags",
        .raw = "{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}",
        .tool_name = "get_time",
        .tool_arg_key = "timezone",
        .tool_arg_value = "UTC",
    },
    // ── Qwen 3.6 MoE (broken-JSON repair paths) ─────────────────────────────
    .{
        // Real broken output from Qwen3.6-35B-A3B-6bit: `, {` instead of
        // `, "arguments": {` — repairFlatBraceToolCallJson path.
        .family = "qwen-moe",
        .name = "flat-brace missing-arguments-key repair",
        .raw = "<tool_call>\n{\"name\":  \"shell\",     {\"command\":\"ls -la\"}}\n</tool_call>",
        .tool_name = "shell",
        .tool_arg_key = "command",
        .tool_arg_value = "ls -la",
    },
    .{
        // Real broken output from Qwen3.6-35B-A3B-6bit: missing the OPENING
        // quote on the `arguments` key.
        .family = "qwen-moe",
        .name = "missing-opening-quote on arguments key repair",
        .raw = "<tool_call>\n{\"name\": \"shell\", arguments\": {\"command\": \"mkdir -p src/app\"}}\n</tool_call>",
        .tool_name = "shell",
        .tool_arg_key = "command",
        .tool_arg_value = "mkdir -p src/app",
    },
    // ── Gemma 4 (<|channel> family, call:name{...} tools) ───────────────────
    .{
        .family = "gemma4",
        .name = "full channel round: thought + content channel",
        .raw = "<|channel>thought\nCompute 17*23: 340+51=391.<channel|>\n<|channel>\n17 × 23 = 391.",
        .thinking = true,
        .content_contains = "391",
        .reasoning_contains = "340+51",
    },
    .{
        // BUG 3 (2026-06-10 pi session): Gemma 4 12B answers in prose, then
        // opens a NEW thought channel right before the turn ends. The raw
        // opener tag leaked into visible output; pi rendered it to the user.
        .family = "gemma4",
        .name = "trailing <|channel>thought opener never leaks (thinking on)",
        .raw = "The page is saved and ready to view.\n\n<|channel>thought\nThe user might also want",
        .thinking = true,
        .content_exact = "The page is saved and ready to view.",
        .reasoning_contains = "might also want",
    },
    .{
        // Same tail behavior with thinking OFF → stripThinkBlock path.
        .family = "gemma4",
        .name = "trailing <|channel>thought opener never leaks (thinking off)",
        .raw = "Here is the design.\n<|channel>thought\nI should now write the file",
        .content_exact = "Here is the design.",
    },
    .{
        // Truncation right after the bare CONTENT channel opener.
        .family = "gemma4",
        .name = "bare content-channel opener stripped on truncation",
        .raw = "<|channel>\nThe answer is 42.",
        .thinking = true,
        .content_exact = "The answer is 42.",
    },
    .{
        // Live 2026-07-16 soak: gemma-4-26B degenerated into a bare 1-token
        // <tool_call|> CLOSE with NO <|tool_call> opener (a "no tools needed"
        // probe with tools present, temp 0.7). parseToolCalls found no call, so
        // the orphan control token used to leak as the WHOLE content. A tool
        // CLOSE is never valid at the tail of content (universal no-tag-leak).
        .family = "gemma4",
        .name = "orphan <tool_call|> close never leaks into content",
        .raw = "<tool_call|>",
        .content_exact = "",
    },
    .{
        // BUG 4 (2026-06-10 pi session, verbatim capture): the LAST string
        // value lost its closing <|"|> delimiter and carried a stray markdown
        // backtick. The unterminated-string scan used to run to end of body,
        // so the parsed path was literally "mlx_pi1.html`}" — and pi created
        // a file with that name on disk. Path must round-trip byte-exact.
        .family = "gemma4",
        .name = "unterminated <|\"|> string must not swallow the closing brace",
        .raw = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html><html></html><|\"|>,path:<|\"|>mlx_pi1.html`}<tool_call|>",
        .tool_name = "write",
        .tool_arg_key = "path",
        .tool_arg_value = "mlx_pi1.html",
    },
    .{
        .family = "gemma4",
        .name = "tool call after closed thought channel",
        .raw = "<|channel>thought\nLet me check the weather<channel|>\n<|tool_call>call:get_weather{\"city\": \"Paris\"}<tool_call|>",
        .thinking = true,
        .tool_name = "get_weather",
        .tool_arg_key = "city",
        .tool_arg_value = "Paris",
    },
    .{
        // Model mixes JSON-style quoted keys with Gemma's <|"|> delimiters.
        .family = "gemma4",
        .name = "quoted keys with custom string delimiters",
        .raw = "<|tool_call>call:shell{\"command\":<|\"|>ls -la<|\"|>}<tool_call|>",
        .tool_name = "shell",
        .tool_arg_key = "command",
        .tool_arg_value = "ls -la",
    },
    .{
        // Jinja literal-brace artifact: args wrapped in {{ }}.
        .family = "gemma4",
        .name = "double-brace wrapped args unwrap",
        .raw = "<|tool_call>call:shell{{\"command\": \"pwd\"}}<tool_call|>",
        .tool_name = "shell",
        .tool_arg_key = "command",
        .tool_arg_value = "pwd",
    },
    // ── DSV4-Flash (self-closing XML-attribute tool form) ───────────────────
    .{
        // Verbatim capture: opened arguments with `"`, closed with `'`,
        // unescaped `"` inside the JSON, finished with `'/>`.
        .family = "dsv4",
        .name = "broken-quote self-closing tool tag",
        .raw = "\n\n<tool_calls>\n<tool name=\"shell\" arguments=\"{\"command\": \"echo hello\"}'/>\n</tool_calls>",
        .tool_name = "shell",
        .tool_arg_key = "command",
        .tool_arg_value = "echo hello",
    },
    // ── Hermes XML (canonical <tool_call>JSON</tool_call>) ──────────────────
    .{
        .family = "hermes",
        .name = "canonical tool_call JSON body",
        .raw = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
        .tool_name = "get_weather",
        .tool_arg_key = "city",
        .tool_arg_value = "Paris",
    },
    .{
        // Double-brace Jinja artifact on the Hermes body.
        .family = "hermes",
        .name = "double-brace wrapped tool_call body",
        .raw = "<tool_call>{{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}}</tool_call>",
        .tool_name = "shell",
        .tool_arg_key = "command",
        .tool_arg_value = "ls",
    },
    .{
        // Claude Code capture (2026-06-10, gemma-4-12b via /v1/messages): the
        // model closed its thought, emitted content, then RE-OPENED an empty
        // thought channel mid-text and closed it immediately. The raw
        // `<|channel>thought\n<channel|>` pair leaked verbatim into the text
        // block Claude Code displayed. Both halves of the surrounding content
        // must stay visible; the pair must vanish.
        .family = "gemma4",
        .name = "mid-text re-opened thought channel pair never leaks",
        .raw = "<|channel>thought\nThe user wants an HTML file.<channel|>Here is the file.```<|channel>thought\n<channel|>I've created a minimal HTML file for you.",
        .thinking = true,
        .content_contains = "I've created a minimal HTML file",
        .reasoning_contains = "user wants an HTML file",
    },
    .{
        // Same shape with a NON-empty second thought: its text is reasoning,
        // never content.
        .family = "gemma4",
        .name = "mid-text thought pair with text routes to reasoning",
        .raw = "<|channel>thought\nPlan the answer.<channel|>The answer is 391.<|channel>thought\nShould I add more detail? No.<channel|>Let me know if you need more.",
        .thinking = true,
        .content_contains = "Let me know if you need more",
        .reasoning_contains = "Should I add more detail",
    },
    .{
        // pi capture (2026-06-10, gemma-4-26B-A4B GGUF via llama engine, same
        // shared split code): the model emits its answer, then opens TWO
        // thought channels in a row, neither ever closed. The cut must happen
        // at the FIRST unclosed opener — cutting at the last one leaks the
        // earlier raw tag into visible content (seen live in pi).
        .family = "gemma4",
        .name = "multiple unclosed thought openers cut at the FIRST one",
        .raw = "I'll start by listing the files in the current directory to see what the project is about.\n<|channel>thought\nI need to understand what system I'm supposed to create specs for.\n<|channel>thought\nWait, I should check the directory once more.",
        .thinking = true,
        .content_exact = "I'll start by listing the files in the current directory to see what the project is about.",
        .reasoning_contains = "check the directory once more",
    },
    .{
        // Same shape, thinking OFF → stripThinkBlock path must also cut at
        // the first unclosed opener.
        .family = "gemma4",
        .name = "multiple unclosed thought openers stripped (thinking off)",
        .raw = "Here is the summary.\n<|channel>thought\nMore ideas\n<|channel>thought\nEven more",
        .content_exact = "Here is the summary.",
    },
    .{
        // 2026-06-19 live Claude Code agentic session (gemma-4): the model
        // CLOSED its thought channel and IMMEDIATELY re-opened a fresh one with
        // NOTHING between, then the turn ended. The leading-strip consumed the
        // first closed block, leaving the bare re-opened opener at the START
        // (pos 0) of the remainder — the trailing-strip bailed on a pos==0
        // opener, so the raw `<|channel>thought\n` leaked verbatim into visible
        // content (it reached chat-history.json as the entire assistant reply).
        .family = "gemma4",
        .name = "re-opened thought opener right after close never leaks (thinking on)",
        .raw = "<|channel>thought\nLet me plan the answer.<channel|>\n<|channel>thought\n",
        .thinking = true,
        .content_exact = "",
        .reasoning_contains = "Let me plan the answer.",
    },
    .{
        // Same shape, thinking OFF → stripThinkBlock path. THIS is the exact
        // form captured live: visible content was the literal `<|channel>thought\n`.
        .family = "gemma4",
        .name = "re-opened thought opener right after close never leaks (thinking off)",
        .raw = "<|channel>thought\nLet me plan the answer.<channel|>\n<|channel>thought\n",
        .content_exact = "",
    },
    .{
        // Inverse guard: real content BETWEEN the close and a trailing
        // re-opened opener must survive — the cut applies only to the dangling
        // re-open, never to the answer that preceded it.
        .family = "gemma4",
        .name = "content between close and re-opened opener survives",
        .raw = "<|channel>thought\nPlan it.<channel|>\nThe file is ready.<|channel>thought\n",
        .thinking = true,
        .content_exact = "The file is ready.",
        .reasoning_contains = "Plan it.",
    },
    .{
        // Live soak capture (2026-07-09, record 2151, a Gemma reasoning variant):
        // the model emitted reasoning, one close, a content scrap, then SPAMMED
        // 16 more bare `<channel|>` close markers. The leading strip cut the FIRST
        // close; the trailing-strip only handled unclosed OPENERS — so the stray
        // CLOSE markers leaked. A close marker is never valid at the tail of
        // content; the universal no-tag-leak invariant pins this.
        .family = "gemma4",
        .name = "trailing <channel|> close-marker spam never leaks (thinking on)",
        .raw = "<|channel>thought\nFind the file.<channel|>\nrunning glob\n\n" ++
            "<channel|><channel|><channel|><channel|><channel|><channel|>",
        .thinking = true,
        .content_contains = "running glob",
        .reasoning_contains = "Find the file.",
    },
    .{
        // Same shape, thinking OFF → stripThinkBlock path.
        .family = "gemma4",
        .name = "trailing <channel|> close-marker spam never leaks (thinking off)",
        .raw = "Reasoning about the file.\n<channel|>running glob\n\n" ++
            "<channel|><channel|><channel|><channel|><channel|>",
        .content_contains = "running glob",
    },
    // ── Gemma 3 (no native tool syntax — markdown-fenced JSON) ──────────────
    .{
        // Verbatim capture from gemma-3-12b-it-qat-4bit on the live matrix
        // (2026-06-10): models without a trained tool format emit the call as
        // a ```json fence. The raw-JSON fallback must tolerate the fence.
        .family = "gemma3",
        .name = "markdown-fenced raw JSON tool call",
        .raw = "```json\n{\"name\": \"write\", \"arguments\": {\"path\": \"report_v2.html\", \"content\": \"<h1>Report</h1>\"}}\n```",
        .tool_name = "write",
        .tool_arg_key = "path",
        .tool_arg_value = "report_v2.html",
    },
    .{
        // Verbatim capture from gemma-3-12b-it-qat-4bit (2026-06-10 llmprobe
        // tool-parallel): asked for parallel calls, the model emits a fenced
        // JSON ARRAY of {name, arguments} objects. Pre-fix only the first
        // object parsed — the second call was silently dropped on all three
        // API surfaces.
        .family = "gemma3",
        .name = "fenced JSON array of parallel tool calls parses ALL calls",
        .raw = "```json\n[\n  {\n    \"name\": \"get_weather\",\n    \"arguments\": {\n      \"location\": \"Paris, France\"\n    }\n  },\n  {\n    \"name\": \"get_weather\",\n    \"arguments\": {\n      \"location\": \"Tokyo, Japan\"\n    }\n  }\n]\n```",
        .tool_name = "get_weather",
        .tool_count = 2,
        .tool_arg_key = "location",
        .tool_arg_value = "Paris, France",
        .last_tool_arg_value = "Tokyo, Japan",
    },
    .{
        // Unfenced variant of the same shape.
        .family = "gemma3",
        .name = "bare JSON array of parallel tool calls parses ALL calls",
        .raw = "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris, France\"}}, {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo, Japan\"}}]",
        .tool_name = "get_weather",
        .tool_count = 2,
        .tool_arg_key = "location",
        .tool_arg_value = "Paris, France",
        .last_tool_arg_value = "Tokyo, Japan",
    },
    // ── Small-model big-file escaping recovery (looseRepairToolCallJson) ────
    // Class: a model writing a large file in one shot mangles the JSON `content`
    // string — raw control bytes instead of `\n`/`\t`, and/or unescaped inner
    // quotes — which strict std.json rejects, so PRE-FIX the whole writeFile
    // call was dropped and the file leaked as visible text. The valid-JSON
    // invariant + byte-exact content assertion below pin the recovery; reverting
    // looseRepairToolCallJson turns each of these red (call → null → "expected a
    // tool call, got none"). New entries are covered automatically.
    .{
        .family = "qwen",
        .name = "writeFile content with RAW newlines (small-model big-file)",
        .raw = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"app.js\",\"content\":\"const a = 1;\nconst b = 2;\nmodule.exports = { a, b };\n\"}}</tool_call>",
        .tool_name = "writeFile",
        .tool_arg_key = "content",
        .tool_arg_value = "const a = 1;\nconst b = 2;\nmodule.exports = { a, b };\n",
    },
    .{
        .family = "qwen",
        .name = "writeFile HTML with UNESCAPED inner quotes + raw newlines",
        .raw = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"brevard.html\",\"content\":\"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n<title>Brevard, NC</title>\n</head>\n</html>\"}}</tool_call>",
        .tool_name = "writeFile",
        .tool_arg_key = "content",
        .tool_arg_value = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n<title>Brevard, NC</title>\n</head>\n</html>",
    },
    .{
        .family = "gemma4",
        .name = "Gemma 4 call:writeFile{json} with raw newlines + inner quotes",
        .raw = "<|tool_call>call:writeFile{\"path\":\"page.html\",\"content\":\"<div class=\"box\">\nhello\n</div>\"}<tool_call|>",
        .tool_name = "writeFile",
        .tool_arg_key = "content",
        .tool_arg_value = "<div class=\"box\">\nhello\n</div>",
    },
    .{
        // Live gemma-4-e4b-it-4bit (test_tool_matrix_small.sh): on a big HTML
        // page it DROPPED the opening <|"|> on `content` but kept the closing
        // one. Pre-fix the bare-value scan cut content at the viewport meta's
        // comma and shredded the rest into bogus keys → invalid args; the
        // closing <|"|> (followed by `,path`) is the true boundary.
        .family = "gemma4",
        .name = "Gemma 4 dropped opening <|\"|> on big content keeps full file",
        .raw = "<|tool_call>call:write_file{content:<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html><|\"|>,path:<|\"|>mars.html<|\"|>}<tool_call|>",
        .tool_name = "write_file",
        .tool_arg_key = "content",
        .tool_arg_value = "<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html>",
    },
    .{
        // Windows path / regex in content — `\U`, `\d` are invalid JSON escapes
        // that strict parse rejects; looseRepair treats them as literal
        // backslashes (the model meant a path, not an escape).
        .family = "qwen",
        .name = "writeFile content with invalid backslash escapes (path/regex)",
        .raw = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"out.py\",\"content\":\"p = r\"C:\\Users\\dev\"\nm = re.match(\\d+)\"}}</tool_call>",
        .tool_name = "writeFile",
        .tool_arg_key = "path",
        .tool_arg_value = "out.py",
    },
    .{
        // Verbatim capture from DeepSeek-V4-Flash via the ds4 engine
        // (2026-06-10, MLX Core agent chat): tool name and each argument as
        // XML child elements, no JSON anywhere. Pre-fix this leaked as
        // visible text and the app's ghost-tool-call nudge fired
        // ("your last response contained a malformed tool-call tag").
        .family = "dsv4",
        .name = "XML-element tool form (<tool_name>/<command> children)",
        .raw = "Let me check the available disk space on this device.\n\n<tool_calls>\n<tool_name>shell</tool_name>\n<command>df -h / | grep -v \"Filesystem\"</command>\n</tool_calls>",
        .tool_name = "shell",
        .tool_count = 1,
        .tool_arg_key = "command",
        .tool_arg_value = "df -h / | grep -v \"Filesystem\"",
    },
    .{
        // Verbatim capture from DeepSeek-V4-Flash via the ds4 engine
        // (2026-06-10 pi html-ds4 turn 2): opened with <tool_call>, closed
        // with the hallucinated </tool_action>. The edit call must parse;
        // pre-fix it leaked as visible text and pi executed nothing.
        .family = "dsv4",
        .name = "mismatched </tool_action> close still parses",
        .raw = "<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\":\"mlx.html\", \"edits\":[{\"oldText\": \"  </ul>\\n</body>\", \"newText\": \"  </ul>\\n  <button onclick=\\\"alert('Hello from MLX')\\\">Click me</button>\\n</body>\"}]}\n</tool_action>",
        .tool_name = "edit",
        .tool_count = 1,
        .tool_arg_key = "path",
        .tool_arg_value = "mlx.html",
    },
    .{
        // Verbatim capture from DeepSeek-V4-Flash via the ds4 engine
        // (2026-06-10 validator-matrix pi html-ds4 turn 2): the tool NAME is
        // embedded in the tag itself (<tool_read>, <tool_edit>) with XML
        // child elements as args. Pre-fix the `<tool_*>` suffix gate only
        // accepted _call/_calls/_request/_requests, so BOTH calls leaked as
        // visible text and pi executed nothing (scored 0/4).
        .family = "dsv4",
        .name = "XML-element-TAG form (<tool_read>/<tool_edit>) parses both calls",
        .raw = "\n\nLet me read the current file first.\n\n<tool_read>\n<path>mlx.html</path>\n</tool_read>Now I'll add a button with inline JavaScript:\n\n<tool_edit>\n<path>mlx.html</path>\n<edits>\n  <oldText>    <h1>MLX Framework on Mac</h1>\n    <ul>\n      <li>Apple silicon–optimized array framework</li>\n      <li>Blazing fast on M-series chips</li>\n      <li>Feels like NumPy, but for Metal</li>\n      <li>Great for ML research and experimentation</li>\n    </ul></oldText>\n  <newText>    <h1>MLX Framework on Mac</h1>\n    <ul>\n      <li>Apple silicon–optimized array framework</li>\n      <li>Blazing fast on M-series chips</li>\n      <li>Feels like NumPy, but for Metal</li>\n      <li>Great for ML research and experimentation</li>\n    </ul>\n    <button onclick=\"alert('Hello from MLX')\">Say Hello</button></newText>\n</edits>\n</tool_edit>",
        .tool_name = "read",
        .tool_count = 2,
        .tool_arg_key = "path",
        .tool_arg_value = "mlx.html",
        .last_tool_arg_value = "mlx.html",
    },
    .{
        // Verbatim-shape capture from the SAME pi case, second sampling
        // (2026-06-10): name-in-tag form again, but the body is a bare JSON
        // args object — `<tool_write>\n{…}\n</tool_write>` — followed by
        // trailing prose. Both body shapes are live DSV4 behavior.
        .family = "dsv4",
        .name = "XML-element-TAG form with JSON args body (<tool_write>{json})",
        .raw = "Here's the HTML page:\n\n<tool_write>\n{\"path\": \"/private/tmp/pi_mlx_workspaces/html-ds4/mlx.html\", \"content\": \"<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<head>\\n  <title>MLX on Mac</title>\\n</head>\\n<body>\\n  <h1>MLX</h1>\\n</body>\\n</html>\"}\n</tool_write>\n\npage ready",
        .tool_name = "write",
        .tool_count = 1,
        .tool_arg_key = "path",
        .tool_arg_value = "/private/tmp/pi_mlx_workspaces/html-ds4/mlx.html",
    },
    .{
        // Verbatim capture, same session turn 1: DSV4 hallucinated a tool
        // RESULT tag without ever calling a tool. Must stay prose — mapping
        // `<tool_output>` onto a tool named "output" would fabricate a call
        // out of thin air.
        .family = "dsv4",
        .name = "hallucinated <tool_output> result tag is not a tool call",
        .raw = "Here's the page I created for you:\n\n<tool_output>Page ready: mlx.html</tool_output>",
        .content_contains = "Page ready",
        .no_tool_calls = true,
    },
    // ── Truncated tool-call OPENER recovery (close_rel==null branch) ────────
    // Class: a model dumps a huge file into ONE Hermes/XML tool call and hits
    // the token cap mid-content, so the call arrives with an OPENING tag but no
    // close (`</parameter>`/`</function>`/`</tool_call>`). Pre-fix the
    // close_rel==null branch only tried JSON shapes, so the whole writeFile was
    // DROPPED and leaked as visible text (live JFK-novel capture, 2026-06-20),
    // and the app misclassified it as a "malformed tag" ghost call. We recover
    // the tool NAME (content is intentionally NOT salvaged — a half-written file
    // is worse than a re-issued chunked write) so the client fires the right
    // chunk/append nudge. The no-tag-leak invariant below auto-confirms the
    // `<tool_call>`/`<function=` markup no longer leaks once the call parses;
    // reverting the recovery turns these red ("expected a tool call, got none").
    .{
        .family = "hermes",
        .name = "truncated <function=writeFile> mid-content recovers the tool name",
        .raw = "<tool_call>\n<function=writeFile>\n<parameter=content>\n# THE LION OF MASSACHUSETTS\n\nChapter 1. The young senator rose before dawn, the Cape light still grey over the water, and thought of all the speeches yet unwritten",
        .tool_name = "writeFile",
    },
    .{
        // EOS-before-close-tag variant: the parameter+function CLOSED but the
        // outer </tool_call> was cut — recovers WITH args (bonus of the fix).
        .family = "hermes",
        .name = "EOS before </tool_call> recovers <function=> call with args",
        .raw = "<tool_call>\n<function=shell>\n<parameter=command>ls -la</parameter>\n</function>",
        .tool_name = "shell",
        .tool_arg_key = "command",
        .tool_arg_value = "ls -la",
    },
    // ── Schema-declared argument types (value-spelling inference class) ──────
    // Class: the tag formats carry NO type information, so the parser infers it
    // from the value's SPELLING (`isJsonLiteral`) — and guesses wrong in both
    // directions. Only the tool schema disambiguates, so every entry with a
    // `tools_json` is coerced (chat.coerceToolArgsToSchema) and then checked by
    // the universal declared-type invariant below. Reverting the coercion turns
    // both entries red.
    .{
        // VERBATIM capture, 2026-07-09 (~/.mlx-serve/logs/mlx-serve-11234.log:109471):
        // Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled via Claude Code.
        // The model writes Python's `False` for a boolean param. isJsonLiteral
        // only knows lowercase `false`, so the arg shipped as the STRING
        // "False" and Claude Code rejected every Edit with
        //   "The parameter `replace_all` type is expected as `boolean` but provided as `string`"
        // The model cannot see its own serialized request, so it "fixed" a
        // value that was already correct — six dead rounds, then it gave up on
        // Edit and rewrote whole files.
        .family = "qwen",
        .name = "Python-style False on a boolean param is coerced to JSON false",
        .raw = "\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
            "<parameter=file_path>\n/Users/david/doom/index.html\n</parameter>\n" ++
            "<parameter=old_string>\n<script src=\"game.js\"></script>\n</parameter>\n" ++
            "<parameter=new_string>\n<script src=\"game.js\" type=\"module\"></script>\n</parameter>\n" ++
            "</function>\n</tool_call>",
        .tool_name = "Edit",
        .tool_arg_key = "file_path",
        .tool_arg_value = "/Users/david/doom/index.html",
        .tools_json = edit_tool_schema,
        .tool_bool_key = "replace_all",
        .tool_bool_value = false,
    },
    .{
        // The INVERSE half of the class: a string-typed param whose content
        // happens to spell a JSON literal. isJsonLiteral promoted it to a real
        // boolean/number, so a code edit touching the token `false` (or a bare
        // `42`) shipped as `"old_string": false` — "expected string, provided
        // boolean". The schema is the only thing that can tell these apart.
        .family = "qwen",
        .name = "string param whose content spells `false` stays a string",
        .raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
            "<parameter=old_string>\nfalse\n</parameter>\n" ++
            "<parameter=new_string>\n42\n</parameter>\n</function>\n</tool_call>",
        .tool_name = "Edit",
        .tool_arg_key = "old_string",
        .tool_arg_value = "false",
        .tools_json = edit_tool_schema,
    },
    .{
        // The CONTAINER half of the class, and the most frequent one in live
        // traffic (15 hits in one pi session): a `<parameter=edits>` holding a
        // JSON array. isJsonLiteral only knows scalars, so the whole array
        // shipped as a STRING — "edits: want array, got str".
        .family = "qwen",
        .name = "array-typed param through Hermes XML is not left a string",
        .raw = "<tool_call>\n<function=edit>\n<parameter=path>\n/tmp/a.js\n</parameter>\n" ++
            "<parameter=edits>\n[{\"oldText\": \"const a = 1;\", \"newText\": \"const a = 2;\"}]\n</parameter>\n" ++
            "</function>\n</tool_call>",
        .tool_name = "edit",
        .tool_arg_key = "path",
        .tool_arg_value = "/tmp/a.js",
        .tools_json = pi_edit_tool_schema,
    },
    .{
        // Same class, different producer: well-formed JSON that merely QUOTES
        // the boolean. Strict parse succeeds, so no repair path ever runs and
        // only the schema pass catches it.
        .family = "qwen",
        .name = "quoted boolean in a JSON tool body is coerced",
        .raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\",\"old_string\":\"a\"," ++
            "\"new_string\":\"b\",\"replace_all\":\"true\"}}</tool_call>",
        .tool_name = "Edit",
        .tools_json = edit_tool_schema,
        .tool_bool_key = "replace_all",
        .tool_bool_value = true,
    },
    // ── Misplaced required param (buried-`path` class) ──────────────────────
    // Class: a weak model that has internalized "the edit object holds everything
    // about the edit" writes the required top-level `path` INSIDE each edits[]
    // item. The args are valid JSON with correctly-typed values — nothing to
    // repair, nothing to coerce — they are simply in the wrong PLACE, which only
    // the schema knows. Strict clients answer "must have required properties
    // path" and the model, blind to its own serialized request, re-emits the same
    // call. The universal buried-param invariant below pins the whole class.
    .{
        // Live pi session 2026-07-13 (gemma-4-26B-A4B-it-qat-4bit, us_presidents):
        // three consecutive rejections, each a full multi-thousand-token
        // generation, then the model abandoned `edit` and rewrote the entire file
        // with `write`. The raw pre-parse bytes were not dumped (the server was
        // not at --log-level debug), so the tag wrapper here is reconstructed; the
        // ARGUMENT SHAPE it produces is the verbatim captured one (pinned
        // byte-for-byte by the hoistMisplacedRequiredParams tests in chat.zig).
        .family = "gemma4",
        .name = "required `path` buried in the edits items is hoisted to the top level",
        .raw = "<|tool_call>call:edit{edits:[{oldText:<|\"|>old line<|\"|>,newText:<|\"|>new line<|\"|>," ++
            "path:<|\"|>us_presidents/generate_site.sh<|\"|>}]}<tool_call|>",
        .tools_json = pi_edit_tool_schema,
        .tool_name = "edit",
        .tool_arg_key = "path",
        .tool_arg_value = "us_presidents/generate_site.sh",
    },
    // ── Loop-stop truncated Gemma call (partial-value salvage class) ────────
    // Class: the server's degenerate-tail-loop guard (scheduler.runSingleDecodeTick)
    // cuts a repetition-looping generation mid-tool-call, so a Gemma-format
    // call arrives with an unterminated value, no `}`, and no <tool_call|>.
    // The salvage must recover the tool NAME and DROP the fragment value — the
    // Hermes-truncation rule ("a half-written file is worse than a re-issued
    // write") now applied to the Gemma arm too. The cut itself reports
    // finish_reason "length" (server truncation; scheduler.loopStopReason), so
    // client truncation recovery fires instead of validating a fragment.
    .{
        // Live 2026-07-14 (pi → gemma-4-26B-A4B-it-qat-4bit, plang/php.html):
        // the model looped "server-side scripting language, " (a ~6-token
        // cycle) inside `content`; the guard cut at its 16-rep threshold
        // mid-word. `path` was never generated — no parse layer can conjure
        // it; what must NOT happen is the 1.1 KB loop fragment shipping as a
        // real argument (pi echoed it back into context verbatim).
        .family = "gemma4",
        .name = "loop-stop truncated write: fragment content dropped, name recovered",
        .raw = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <title>PHP</title>\n</head>\n<body>\n    <p>PHP is a widely-used general-purpose scripting language. It is a server-side scripting language, server-side scripting language, server-side scripting language, server-",
        .tools_json = write_read_tools_schema,
        .tool_name = "write",
        .tool_arg_absent = "content",
    },
    .{
        // The ordering that would CORRUPT a file: `path` completed BEFORE the
        // cut. Pre-fix salvage = {path, <partial garbage>} — schema-valid, so
        // the client writes the fragment to a real file and reports success.
        // The complete path survives; the fragment never ships.
        .family = "gemma4",
        .name = "loop-stop truncated write after complete path: path kept, fragment dropped",
        .raw = "<|tool_call>call:write{path:<|\"|>plang/php.html<|\"|>,content:<|\"|><!DOCTYPE html>\n<p>PHP is a server-side scripting language, server-side scripting language, server-",
        .tools_json = write_read_tools_schema,
        .tool_name = "write",
        .tool_arg_key = "path",
        .tool_arg_value = "plang/php.html",
        .tool_arg_absent = "content",
    },
    // ── Negatives ────────────────────────────────────────────────────────────
    .{
        // Prose containing a `<tool…>`-ish tag that is NOT a tool call.
        .family = "prose",
        .name = "prose with <toolbar> markup is not a tool call",
        .raw = "Click the <toolbar> icon, then choose Settings from the menu.",
        .content_contains = "Settings",
        .no_tool_calls = true,
    },
    // ── Hallucinated raw-JSON tool calls (George Washington class) ──────────
    .{
        // Live pi capture 2026-07-13 (Qwen3.6-35B-A3B distilled): generation
        // hit max_tokens midway through a presidents data script. The raw-JSON
        // fallback found the first balanced object — {"name": "George
        // Washington", "num": 1, …} — and the flat-shape synthesis promoted it
        // to a TOOL CALL named "George Washington". pi answered "Tool George
        // Washington not found"; the model retried the identical mega-write and
        // the session burned two 16K-token turns making zero progress. With the
        // request's tools schema present, an INFERRED call whose name is
        // undeclared is dropped: the text stays visible content and the
        // client's own truncation recovery (finish_reason="length") fires.
        .family = "qwen",
        .name = "truncated data dict with a name field is not a hallucinated tool call",
        .raw = "Now let me build a generator script that creates all 46 president pages:\n\n" ++
            "presidents = [\n" ++
            "  {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None (Federalist-leaning)\", \"term\": \"1789\u{2013}1797\", \"vice\": \"John Adams\"},\n" ++
            "  {\"name\": \"John Adams\", \"num\": 2, \"party\": \"Federalist\",",
        .tools_json = write_read_tools_schema,
        .no_tool_calls = true,
        .content_contains = "presidents = [",
    },
    .{
        // The counterweight: models WITHOUT a trained tool format (Gemma 3)
        // emit fenced raw-JSON calls — a declared name must keep parsing.
        .family = "gemma3",
        .name = "fenced raw-JSON call with a DECLARED name still parses",
        .raw = "```json\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"hi\"}}\n```",
        .tools_json = write_read_tools_schema,
        .tool_name = "write",
        .tool_arg_key = "path",
        .tool_arg_value = "a.txt",
    },
    // ── Hy3 / Hunyuan 3 (hy_v3): suffixed think tags + arg_key/arg_value tool
    // format. Entries are template-spec-shaped (chat_template.jinja, HYTK
    // ":opensource"); replace/extend with harvested live bytes once the 295B
    // runs locally (MLX_SERVE_RAW_DUMP_FILE workflow above). Thinking is
    // template-opened by default (generation prompt ends with the opener when
    // reasoning_effort is high/low). ─────────────────────────────────────────
    .{
        .family = "hy3",
        .name = "full think round, template-opened, suffixed close tag",
        .raw = "The user wants 17*24. 17*24 = 408.</think:opensource>17 × 24 = **408**.",
        .thinking = true,
        .opened_by_template = true,
        .content_contains = "408",
        .reasoning_contains = "17*24 = 408",
    },
    .{
        .family = "hy3",
        .name = "template-opened truncated thinking stays out of content (suffixed family)",
        .raw = "Let me work through the request step by step: first",
        .thinking = true,
        .opened_by_template = true,
        .content_exact = "",
        .reasoning_contains = "step by step",
    },
    .{
        .family = "hy3",
        .name = "arg_key/arg_value tool call after thinking, string→bool schema coercion",
        .raw = "I should edit the file.</think:opensource><tool_calls:opensource>\n" ++
            "<tool_call:opensource>Edit<tool_sep:opensource>\n" ++
            "<arg_key:opensource>file_path</arg_key:opensource>\n" ++
            "<arg_value:opensource>src/main.py</arg_value:opensource>\n" ++
            "<arg_key:opensource>old_string</arg_key:opensource>\n" ++
            "<arg_value:opensource>x = 1</arg_value:opensource>\n" ++
            "<arg_key:opensource>new_string</arg_key:opensource>\n" ++
            "<arg_value:opensource>x = 2</arg_value:opensource>\n" ++
            "<arg_key:opensource>replace_all</arg_key:opensource>\n" ++
            "<arg_value:opensource>false</arg_value:opensource>\n" ++
            "</tool_call:opensource>\n</tool_calls:opensource>",
        .thinking = true,
        .opened_by_template = true,
        .tools_json = edit_tool_schema,
        .tool_name = "Edit",
        .tool_arg_key = "file_path",
        .tool_arg_value = "src/main.py",
        .tool_bool_key = "replace_all",
        .tool_bool_value = false,
    },
    .{
        .family = "hy3",
        .name = "parallel calls in one wrapper keep their own args",
        .raw = "<tool_calls:opensource>\n" ++
            "<tool_call:opensource>read<tool_sep:opensource>\n" ++
            "<arg_key:opensource>path</arg_key:opensource>\n" ++
            "<arg_value:opensource>a.txt</arg_value:opensource>\n" ++
            "</tool_call:opensource>\n" ++
            "<tool_call:opensource>read<tool_sep:opensource>\n" ++
            "<arg_key:opensource>path</arg_key:opensource>\n" ++
            "<arg_value:opensource>b.txt</arg_value:opensource>\n" ++
            "</tool_call:opensource>\n</tool_calls:opensource>",
        .tools_json = write_read_tools_schema,
        .tool_count = 2,
        .tool_name = "read",
        .tool_arg_key = "path",
        .tool_arg_value = "a.txt",
        .last_tool_arg_value = "b.txt",
    },
    .{
        // Big-file-write truncation class, hy3 shape: max_tokens landed inside
        // the `content` value. Recover the call with the CLOSED pair only —
        // the fragment must never ship as a real argument.
        .family = "hy3",
        .name = "truncated mid-arg_value recovers name + closed args, drops the fragment",
        .raw = "<tool_calls:opensource>\n" ++
            "<tool_call:opensource>write<tool_sep:opensource>\n" ++
            "<arg_key:opensource>path</arg_key:opensource>\n" ++
            "<arg_value:opensource>novel.txt</arg_value:opensource>\n" ++
            "<arg_key:opensource>content</arg_key:opensource>\n" ++
            "<arg_value:opensource>Chapter 1. It was a dark and stormy night and the",
        .tools_json = write_read_tools_schema,
        .tool_name = "write",
        .tool_arg_key = "path",
        .tool_arg_value = "novel.txt",
        .tool_arg_absent = "content",
    },
    .{
        // Prose mentioning the format's pieces (without an actual opener tag —
        // the control tags are special tokens a real generation can't casually
        // reproduce mid-prose) must not parse as a call.
        .family = "hy3",
        .name = "prose about arg_key/arg_value is not a tool call",
        .raw = "Hy3 encodes each argument as an arg_key/arg_value pair inside the call block.",
        .tools_json = write_read_tools_schema,
        .no_tool_calls = true,
        .content_contains = "arg_key/arg_value pair",
    },
    .{
        // LIVE capture 2026-07-14 — first Hy3 (295B, 2-bit) run on this
        // engine; raw bytes verbatim from the debug log ("Weather in Tokyo in
        // celsius please", temp 0). Confirms the shipped model emits the
        // template-spec format exactly; the "true" arg is a STRING in the tag
        // format and the schema coercion must type it.
        .family = "hy3",
        .name = "LIVE: get_weather call, wrapper + sep + arg tags, string→bool coercion",
        .raw = "<tool_calls:opensource>\n" ++
            "<tool_call:opensource>get_weather<tool_sep:opensource>\n" ++
            "<arg_key:opensource>city</arg_key:opensource>\n" ++
            "<arg_value:opensource>Tokyo</arg_value:opensource>\n" ++
            "<arg_key:opensource>celsius</arg_key:opensource>\n" ++
            "<arg_value:opensource>true</arg_value:opensource>\n" ++
            "</tool_call:opensource>\n</tool_calls:opensource>",
        .tools_json = weather_tool_schema,
        .tool_name = "get_weather",
        .tool_arg_key = "city",
        .tool_arg_value = "Tokyo",
        .tool_bool_key = "celsius",
        .tool_bool_value = true,
    },
    .{
        // LIVE capture 2026-07-16 (pipenetwork/Hy3-REAP62 via MLX_SERVE_RAW_DUMP_FILE,
        // the soak): the pruned model emitted the PLURAL wrapper
        // <tool_calls:opensource> and jumped STRAIGHT to the NAME, dropping the
        // singular per-call <tool_call:opensource> opener the parser keys on — so
        // the whole (well-formed, complete) call LEAKED as content. Same
        // weak-model delimiter-drop class as the dropped-<tool_sep> entry, one
        // delimiter over. Recover the full call incl. the quote-bearing content
        // (the universal no-tag-leak + valid-JSON-args invariants cover it).
        .family = "hy3",
        .name = "LIVE: dropped singular <tool_call> opener (plural wrapper only) still recovers",
        .raw = "<tool_calls:opensource>\n" ++
            "write_file</arg_value:opensource>\n" ++
            "<arg_key:opensource>path</arg_value:opensource>\n" ++
            "<arg_value:opensource>page.html</arg_value:opensource>\n" ++
            "<arg_key:opensource>content</arg_value:opensource>\n" ++
            "<arg_value:opensource><meta charset=\"UTF-8\"><a href=\"/x\">L</a><div class=\"hero\">Hi</div></arg_value:opensource>\n" ++
            "</tool_call:opensource>\n</tool_calls:opensource>",
        .tool_name = "write_file",
        .tool_arg_key = "path",
        .tool_arg_value = "page.html",
    },
};

/// Control tags that must never appear in visible content, regardless of
/// family. `<|"|>` is Gemma 4's string delimiter; the rest are think/tool
/// markers from every supported template family.
const leak_tags = [_][]const u8{
    "<think>", "</think>", "<|channel>", "<channel|>", "<|tool_call", "<tool_call", "<|\"|>",
};

fn fail(entry: Expect, comptime what: []const u8, got: []const u8) !void {
    std.debug.print("\n[{s}] {s}: " ++ what ++ "\n  got: {s}\n", .{ entry.family, entry.name, got });
    return error.FormatCorpusExpectFailed;
}

test "format corpus: recorded model outputs across families" {
    const allocator = testing.allocator;

    for (corpus) |entry| {
        // ── Normalize first (mirrors the server: re-opened mid-text thought
        // channels merge into one leading block before any parse/split). ──
        const normalized = try chat.normalizeEmbeddedThinkBlocks(allocator, entry.raw);
        defer if (normalized) |n| allocator.free(n);
        const raw: []const u8 = normalized orelse entry.raw;

        // ── Tool calls (when calls parse, content is suppressed and only
        // tool deltas + reasoning are emitted). ──
        var calls = try chat.parseToolCalls(allocator, raw);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };

        // Mirror the server chokepoint (server.parseToolCallsForRequest): when
        // the request declared tools, (1) heuristically-inferred raw-JSON calls
        // must name a DECLARED tool — a truncated data object is never a call
        // (George Washington class; every entry with a tools_json is covered
        // automatically) — then (2) a required param the model BURIED inside a
        // container arg is hoisted back to the top level, and (3) arguments are
        // coerced to the schema's types before any client sees them.
        if (entry.tools_json) |tj| {
            if (calls) |cs| calls = try chat.filterInferredBySchema(allocator, cs, tj);
            if (calls) |cs| try chat.hoistMisplacedRequiredParams(allocator, cs, tj);
            if (calls) |cs| try chat.coerceToolArgsToSchema(allocator, cs, tj);
        }

        if (entry.no_tool_calls and calls != null) {
            try fail(entry, "expected NO tool calls but got some", calls.?[0].name);
        }
        if (entry.tool_name) |want_name| {
            const cs = calls orelse return fail(entry, "expected a tool call, got none", entry.raw);
            if (!std.mem.eql(u8, cs[0].name, want_name)) {
                try fail(entry, "tool name mismatch", cs[0].name);
            }
        }
        if (entry.tool_count) |want_count| {
            const cs = calls orelse return fail(entry, "expected tool calls, got none", entry.raw);
            if (cs.len != want_count) {
                var buf: [32]u8 = undefined;
                try fail(entry, "tool call count mismatch", std.fmt.bufPrint(&buf, "{d}", .{cs.len}) catch "?");
            }
        }

        // Valid-JSON invariant: EVERY parsed call's arguments must round-trip.
        if (calls) |cs| {
            for (cs) |tc| {
                const parsed = std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{}) catch {
                    try fail(entry, "tool arguments are not valid JSON", tc.arguments);
                    unreachable;
                };
                defer parsed.deinit();
                if (parsed.value != .object) try fail(entry, "tool arguments are not a JSON object", tc.arguments);

                // Universal declared-type invariant: every argument whose type
                // the tool declares must actually carry that JSON type. This is
                // what strict clients validate and reject on. Any future entry
                // that supplies a `tools_json` is covered automatically.
                if (entry.tools_json) |tj| {
                    if (!chat.toolCallConformsToSchema(allocator, tc, tj)) {
                        try fail(entry, "tool argument type contradicts the declared schema", tc.arguments);
                    }

                    // Universal buried-param invariant: a REQUIRED scalar the
                    // model stuffed inside a container arg (while omitting it at
                    // the top level) is what strict clients reject with "must
                    // have required properties X". The chokepoint hoists it, so
                    // nothing may still be buried here. Any future entry with a
                    // tools_json is covered automatically.
                    if (chat.requiredParamIsBuried(allocator, tc, tj)) {
                        try fail(entry, "a required param is still buried inside a container arg", tc.arguments);
                    }
                }
            }

            if (entry.tool_bool_key) |key| {
                const parsed = try std.json.parseFromSlice(std.json.Value, allocator, cs[0].arguments, .{});
                defer parsed.deinit();
                const val = parsed.value.object.get(key) orelse {
                    try fail(entry, "expected boolean arg key missing", cs[0].arguments);
                    unreachable;
                };
                if (val != .bool or val.bool != entry.tool_bool_value.?) {
                    try fail(entry, "boolean arg is not the expected JSON boolean", cs[0].arguments);
                }
            }
            if (entry.tool_arg_absent) |key| {
                const parsed = try std.json.parseFromSlice(std.json.Value, allocator, cs[0].arguments, .{});
                defer parsed.deinit();
                if (parsed.value == .object and parsed.value.object.get(key) != null) {
                    try fail(entry, "fragment arg shipped — key must be ABSENT after truncation salvage", cs[0].arguments);
                }
            }
            if (entry.tool_arg_key) |key| {
                const parsed = try std.json.parseFromSlice(std.json.Value, allocator, cs[0].arguments, .{});
                defer parsed.deinit();
                const val = parsed.value.object.get(key) orelse {
                    try fail(entry, "expected arg key missing", cs[0].arguments);
                    unreachable;
                };
                if (entry.tool_arg_value) |want| {
                    if (val != .string or !std.mem.eql(u8, val.string, want)) {
                        try fail(entry, "arg value mismatch (must be byte-exact)", cs[0].arguments);
                    }
                }
                if (entry.last_tool_arg_value) |want| {
                    const last_parsed = try std.json.parseFromSlice(std.json.Value, allocator, cs[cs.len - 1].arguments, .{});
                    defer last_parsed.deinit();
                    const last_val = last_parsed.value.object.get(key) orelse {
                        try fail(entry, "expected arg key missing in LAST call", cs[cs.len - 1].arguments);
                        unreachable;
                    };
                    if (last_val != .string or !std.mem.eql(u8, last_val.string, want)) {
                        try fail(entry, "LAST call arg value mismatch", cs[cs.len - 1].arguments);
                    }
                }
            }
        }

        // ── Visible content / reasoning split (server's no-tool-call path). ──
        const split: chat.ThinkSplit = if (entry.thinking)
            chat.splitThinkBlock(raw, true, entry.opened_by_template)
        else
            .{ .reasoning_content = null, .content = chat.stripThinkBlock(raw) };
        // When tool calls parsed, the server emits NO content from this text.
        const content: []const u8 = if (calls != null) "" else split.content;

        // Universal leak invariant: visible content never carries control tags.
        for (leak_tags) |tag| {
            if (std.mem.indexOf(u8, content, tag) != null) {
                try fail(entry, "control tag leaked into visible content", content);
            }
        }

        if (entry.content_exact) |want| {
            if (!std.mem.eql(u8, content, want)) {
                try fail(entry, "content not byte-exact", content);
            }
        }
        if (entry.content_contains) |want| {
            if (std.mem.indexOf(u8, content, want) == null) {
                try fail(entry, "content missing expected substring", content);
            }
        }
        if (entry.reasoning_contains) |want| {
            const reasoning = split.reasoning_content orelse {
                try fail(entry, "expected reasoning_content, got null", content);
                unreachable;
            };
            if (std.mem.indexOf(u8, reasoning, want) == null) {
                try fail(entry, "reasoning missing expected substring", reasoning);
            }
        }
    }
}

test "format corpus: streaming think-gate never leaks thinking mid-stream" {
    // Replay every recorded output byte-by-byte through the shared streaming
    // gate (chat.streamThinkGate — used by both the chat-completions and
    // /v1/messages SSE handlers with tools present). Invariants:
    //   1. With thinking enabled, NOTHING flushes as visible text before the
    //      think close tag has fully arrived — the 2026-06-10 Claude Code
    //      failure streamed Qwen's template-opened thinking as text_deltas,
    //      raw `</think>` included.
    //   2. The split fires only once the close tag is actually in the buffer.
    //   3. After the split (think_closed), plain prose flushes — the inverse
    //      failure hid the visible answer in the buffer until end-of-stream.
    for (corpus) |entry| {
        if (!entry.thinking) continue;

        // Earliest end position of a think close tag, any family (the chat
        // helper covers both `</think>` and the Hy3-suffixed variant).
        const close_end: ?usize = blk: {
            var best: ?usize = null;
            if (chat.indexOfThinkCloseTag(entry.raw, 0)) |c| best = c.pos + c.len;
            if (std.mem.indexOf(u8, entry.raw, "<channel|>")) |p| {
                const e = p + "<channel|>".len;
                if (best == null or e < best.?) best = e;
            }
            break :blk best;
        };

        var think_closed = false;
        var i: usize = 1;
        while (i <= entry.raw.len) : (i += 1) {
            const buf = entry.raw[0..i];
            const gate = chat.streamThinkGate(buf, true, think_closed);
            if (think_closed) break; // post-split buffers start fresh in the real path
            if (close_end == null or i < close_end.?) {
                if (gate == .flush_text) {
                    try fail(entry, "gate flushed visible text before think close", buf);
                }
            }
            if (gate == .split_think) {
                if (close_end == null or i < close_end.?) {
                    try fail(entry, "gate split before the close tag arrived", buf);
                }
                think_closed = true;
            }
        }

        // Truncated thinking (no close tag at all) must hold to the very end —
        // end-of-stream handling owns it from there.
        if (close_end == null) {
            const gate = chat.streamThinkGate(entry.raw, true, false);
            if (gate == .flush_text) {
                try fail(entry, "gate flushed truncated thinking as text", entry.raw);
            }
        }
    }

    // Invariant 3, directly: once think_closed, prose streams.
    try testing.expectEqual(chat.StreamThinkGate.flush_text, chat.streamThinkGate("The visible answer.", true, true));
}

test "format corpus: history round-trip serialization survives any byte content" {
    // Inverse direction of the corpus: everything a model emits (and every
    // tool result an agent echoes back) re-enters the NEXT request's history
    // and is serialized by chat.serializeMessagesJson into the JSON that the
    // C++ Jinja engine (nlohmann, strict) parses. 2026-06-11 pi/gemma-4-31b
    // failure: a tool result with a raw ESC byte (`\x1b[?25l`, ANSI
    // hide-cursor from an interactive npm CLI) produced invalid JSON →
    // jinja_render_chat returned NULL → silent fallback to the wrong prompt
    // format → the model hallucinated whole conversations.
    //
    // Invariants, for every corpus entry's raw text AND hostile tool-result
    // samples:
    //   1. The serialized form contains NO raw control byte (< 0x20) — the
    //      strictest parser downstream must accept it.
    //   2. A strict JSON parse round-trips every content byte exactly.
    const allocator = testing.allocator;

    // Tool-result shapes that have to survive verbatim: ANSI codes from the
    // live failure, plus every control byte 0x00–0x1F in one payload.
    var all_ctrl: [0x20]u8 = undefined;
    for (&all_ctrl, 0..) |*c, i| c.* = @intCast(i);
    const hostile_tool_results = [_][]const u8{
        "\x1b[?25l\u{2502}\n\u{25c6}  Which template would you like?\n\u{2502}  \u{25cf} SvelteKit minimal", // verbatim live failure
        &all_ctrl,
    };

    for (corpus) |entry| {
        for (hostile_tool_results) |tool_result| {
            const tc = [_]chat.ToolCall{
                .{ .id = "tc_0", .name = "bash", .arguments = "{\"command\": \"npx sv create .\"}" },
            };
            const messages = [_]chat.Message{
                .{ .role = "user", .content = "make me a sveltekit app" },
                // The model's own raw output goes back in as assistant content.
                .{ .role = "assistant", .content = entry.raw, .tool_calls = &tc },
                .{ .role = "tool", .content = tool_result, .tool_call_id = "tc_0" },
            };

            const serialized = try chat.serializeMessagesJson(allocator, &messages);
            defer allocator.free(serialized);

            for (serialized) |c| {
                if (c < 0x20) {
                    std.debug.print("\n[{s}] {s}: raw control byte 0x{x:0>2} in serialized history\n", .{ entry.family, entry.name, c });
                    return error.FormatCorpusExpectFailed;
                }
            }

            const parsed = std.json.parseFromSlice(std.json.Value, allocator, serialized, .{}) catch {
                std.debug.print("\n[{s}] {s}: serialized history is not valid JSON\n  got: {s}\n", .{ entry.family, entry.name, serialized });
                return error.FormatCorpusExpectFailed;
            };
            defer parsed.deinit();

            const msgs = parsed.value.array.items;
            const assistant_content = msgs[1].object.get("content").?.string;
            const tool_content = msgs[2].object.get("content").?.string;
            try testing.expectEqualStrings(entry.raw, assistant_content);
            try testing.expectEqualStrings(tool_result, tool_content);
        }
    }
}
