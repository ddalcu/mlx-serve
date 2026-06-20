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
//!    (two sites in src/server.zig — streaming and non-streaming).
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
};

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
    // ── Negatives ────────────────────────────────────────────────────────────
    .{
        // Prose containing a `<tool…>`-ish tag that is NOT a tool call.
        .family = "prose",
        .name = "prose with <toolbar> markup is not a tool call",
        .raw = "Click the <toolbar> icon, then choose Settings from the menu.",
        .content_contains = "Settings",
        .no_tool_calls = true,
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
        const calls = try chat.parseToolCalls(allocator, raw);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };

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

        // Earliest end position of a think close tag, either family.
        const close_end: ?usize = blk: {
            var best: ?usize = null;
            if (std.mem.indexOf(u8, entry.raw, "</think>")) |p| best = p + "</think>".len;
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
