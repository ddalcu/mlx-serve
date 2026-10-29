# Tool calling and model-output formats: gotchas

One story per gotcha: the defect, the cause, the fix, the guard. The rules live in the root `AGENTS.md` (`## Rules`);
what is still open lives in `docs/known-issues.md`. Chat code is under `src/host/chat/` (dialect parsers in `tools/`,
reasoning splits in `reasoning/`, the corpus in `format_corpus_test.zig`, the traffic replay in
`tool_traffic_replay_test.zig`).

## Silent template fallback

### pi "stalls" on Qwen: the system prompt arrived as role `developer` (#348)
Defect: every pi request logged `jinja render failed … Unexpected message role`, fell back to the generic format and
ended turns empty or malformed. Cause: pi sends the system prompt as role `developer` unless its config says
otherwise, and Qwen's template raises on the role. Fix: `chat.canonicalRole` maps `developer` to `system` at the
chat-completions and Responses parse sites; the cache-key system hash reads it too. Guard: `canonicalRole` +
`requestCacheKey` tests, `parseInput reads a developer item`.

### One control byte in history downgraded the whole prompt format
Defect: gemma-4-31b in pi degenerated on turn 3, hallucinating both sides. Cause: a tool result carried ANSI codes;
nlohmann rejects one raw byte < 0x20, the render failed, and `fallbackFormatChat` served tags that are not even
special tokens for Gemma 4, so the model lost its stop token. Fix: every string reaches the template through
`appendJsonString`, which `\u`-escapes all control bytes and replaces ill-formed UTF-8 with U+FFFD; a render failure
logs at warn (`jinja render failed`). Guard: corpus `history round-trip serialization survives any byte content`.
Tell: wrong-family tags in output means a silent fallback first.

### A NUL byte in a message truncated the rendered prompt
Defect: an opencode agent grepped its own binary; the next request logged `prompt=88113` and streamed `+0` tokens, a
200 with nothing in it, and every later turn rendered the same count. Cause: jinja decodes `\u0000` to a real NUL and
the shim's result was read with `std.mem.span`, cutting the prompt at the first NUL. Fix: `jinja_render_chat` reports
its length and Zig slices `ptr[0..len]`; the tokenizer round-trips 0x00. Guard: the render test with a NUL mid-message
keeps the text after it and the assistant header. Tell: the same `prompt=` on consecutive turns.

### A `chat_template` can be a pointer to the sidecar file (#169)
Defect: Laguna answered in Gemma markers with its reasoning in content. Cause: transformers ≥ 5 writes `{% include
'chat_template.jinja' %}` as the `tokenizer_config.json` value; we read that first, jinja.cpp has no `include`, and
the render fell back. Fix: `isIncludeStub` reads an include-only value as "no inline template" so the sidecar loads; a
template with an include AND other text is left to fail by name. Guard: the include-stub cases in `chat_template
accepts HF's list-of-named-templates shape`. Grep the log for `jinja` before touching the arch.

### Templates that raise on our own extra-context values (Inkling, Qwen3.8)
Defect: 100% of Qwen3.8 requests and many Inkling requests rendered the generic fallback; Qwen3.8's fallback is
ChatML, so nothing looked wrong except a model that had never heard of the declared tools. Cause: Inkling maps
`reasoning_effort` through its own table and raises on hy3's `"no_think"`, raises on string `arguments`, and needed
`tojson(sort_keys=true)`; Qwen3.8 raises on any effort outside `xhigh|medium|low` and (the 2.4T release, not the 27B)
on thinking-off. Fix: `sort_keys` in jinja.cpp; `serializeExtraContext` sniffs the family (`Thinking effort level`,
the `'xhigh'` literal) and maps the effort (`qwen38EffortFor`); `enable_thinking:false` is withheld only where the
template carries the refusal string, and `noThinkTailSuffix` closes the opened `<think>`; history calls carry `"id"`
so results are named. Guard: hermetic real-template render tests over `src/fixtures/inkling_chat_template.jinja`,
`qwen38_chat_template.jinja` and `qwen38_27b_chat_template.jinja`.

### Late system messages were folded ahead of cached history
Defect: Claude Code's runtime reminders arrive as late system messages; folding them to the front invalidated almost
all RAM/SSD prefix reuse every tool round. Cause: unconditional folding, because strict templates reject late system
turns. Fix: parsers merge only the leading run; the renderer probes each late-system boundary with the request's tools
and kwargs (`templatePreservesMidSystem`) and folds only where the template raises, drops or reorders; byte-pinned
Qwen revisions render later ChatML system turns. Guard: `mid-system:` tests, the corpus late-note invariant,
`tests/test_mid_system_cache.sh`.

### A transcribed template's whitespace is a token-level contract (DeepSeek-V4)
Defect: our transcription of the DSV4 encoder emitted `<bos>## Tools` where the model was trained on `<bos>\n\n##
Tools` for every no-system tools request. Cause: the separator was conditional on a system turn; the reference renders
an empty system turn. Fix: unconditional separator; A/B the transcription against the reference over the shapes the
server emits, converting the INPUT to each side's contract (the reference `json.loads` string arguments) and comparing
only the output bytes. Guard: `tests/dsv4_template_ab.py` + the hermetic render test.

### DeepSeek-V4.1's own template does not render in jinja.cpp
Defect: every V4.1 request fell back (`Unexpected token: *`). Cause: `dict(schema.d, **{k: v})`, `dict(d, key=v)` and
chained comparisons are not in jinja.cpp, and block tags differ on trailing newlines. Fix: V4.1 always renders our
transcription of `encoding/encoding.py` (`src/fixtures/dsv41_chat_template.jinja`); the parser accepts V4.1's spaced
DSML tag names. Guard: `tests/dsv41_template_ab.py` dumps the cases the render test replays; corpus family
`dsv41-dsml`.

### jinja.cpp gaps: `x.0` and `\xHH`
Defect: GLM-5.3 tool history and every Flint turn rendered the generic fallback (Gemma turn markers, wrong stop
tokens). Cause: jinja.cpp parsed the `0` in `x.0` as a name, and its lexer knew no `\x` escape (Flint's
`replace("\x00", "")` raised at lex time). Fix: an integer member access indexes like `x[0]`; `\xHH` lexes as U+00HH.
Guard: the GLM tool-history and Flint render tests (byte-exact against jinja2), `tests/test_glm5_next.sh` [4],
`tests/test_flint.sh` [4].

### A generic role header rewrote tool results into user turns (MiMo)
Defect: MiMo-V2.6 got every tool result as a `<tool_response>` user turn. Cause: its template renders `<|im_start|>{{
message.role }}` and never spells `'tool'`; the literal check read that as no tool role. Fix: a template that renders
assistant `tool_calls` but names no tool role is probe-rendered with one tool message (`templateRendersToolTurn`).
Guard: `renderChatTemplate: a tool-aware template with a generic role header renders tool turns natively`.

## Parsing tool calls

### Mangled argument JSON dropped the whole call (big-file escaping)
Defect: a weak model writing a big file put raw newlines and unescaped quotes in `content`; strict parse failed, the
call was dropped and the file leaked into chat. Fix: `looseRepairToolCallJson` re-escapes control bytes, inner quotes
(a `"` closes a string only at a structural delimiter) and bad backslashes, runs only after strict parse fails, and is
re-validated by a strict parse. Guard: `parseToolCalls recovers …` tests, corpus "Small-model big-file escaping
recovery" + the valid-JSON-args invariant.

### A truncated opener is a truncation, not a ghost call
Defect: a 19k-char `writeFile` cut by `max_tokens` inside a Hermes call was dropped and leaked as text; the app sent
the "malformed tag" nudge and the model retried the same one-shot write. Cause: the `close_rel == null` branch tried
only the JSON shapes. Fix: the truncated branch also runs the XML-function and element parsers, recovering NAME +
`{}`; partial values never ship (a half-written file is worse than a re-issued chunked write); the app routes
`maxTokensHit` + an unclosed opener to the truncation nudge. Guard: `recovers truncated <function=…>`, corpus
"Truncated tool-call OPENER recovery", `ChatTurnEngineTruncationTests`.

### The Hermes JSON arm had no truncation salvage
Defect: a 4.4 KB qwen4_exp `edit` that hit EOS inside a string came back as an empty `stop`. Cause: only the
balanced-object case was handled, and the stream flush cut the text at `<tool_call>`. Fix: `truncatedJsonCallName`
finds a depth-1 `"name"` with a complete string value and ships NAME + `{}`. Shipping the raw buffer as content
re-opens the leak class and is not the fix. Guard: `<tool_call>{JSON} truncated mid-string recovers NAME + {}`.

### Weak models drop one delimiter (Hy3 REAP)
Defect: `pipenetwork/Hy3-REAP62` closed the NAME with `</arg_value>` (no `<tool_sep>`), closed keys with
`</arg_value>`, or skipped the singular `<tool_call:opensource>` after the plural wrapper; the call leaked and pi
looped on `bash({})`. Fix: the name ends at the earliest structural marker, the arg loop scans to the next `<arg_key>`
and accepts either close; a plural-wrapper arm keys strictly on the SUFFIXED `<tool_calls:` so the bare DSV4
`<tool_calls>` wrapper still reaches its own parser. Guard: the two hy3 capture tests in `host/chat/chat.zig` +
corpus.

### A doubled key opener kept markup in the key (Flint)
Defect: Claude Code got the argument key `<arg_key>file_path` and rejected the `Edit`. Cause: the GLM pair loop took
everything between the first opener and the close. Fix: the key starts after the LAST opener before its close. Guard:
corpus family `flint`; universal invariant: no argument key carries a line break or markup.

### A dropped `>` after the function name became the name (#748)
Defect: `<function=read_file` with no `>` shipped the name `read_file\n</parameter`; the streamed chat delta was
invalid JSON and its newline split the SSE line. Fix: a name ends at `>`, a newline or `<`; the stream delta goes
through `server.streamToolCallDelta`, escaping both strings. Guard: corpus invariant (no line break or markup in a
name), `streamed tool-call delta escapes the name and arguments`.

### A parameter value decided the call (Qwen `<function=` dialect)
Defect: pi on Qwen3.6-27B kept calling tools named `pi-lfm`, then `voxel-pagoda-garden`. Cause: the template-mandated
`<function=write>` call rides inside `<tool_call>`, and the JSON arm ran first: it took the package.json being written
as a flat call and promoted its `"name"`. Fix: a body carrying `<function=` is parsed by the function-tag arm FIRST. A
value is arbitrary bytes and never decides the call. Guard: "qwen xml: the live `<function=…>` capture (package.json
class)" + corpus entry.

### A parameter value that spells the dialect's own close tags
Defect: a model writing a file that documents the Qwen tool format got an empty file (reported as success), a dropped
required parameter, or a call cut short. Cause: the value end, `</function>` and `</tool_call>` scans each took the
FIRST occurrence; the truncated branch ran the JSON arm before the `<function=` guard; `isJsonLiteral` typed
`0755`/`Infinity` via `parseFloat`. Fix: `hermesValueEnd` = the LAST `</parameter>` before the next opener,
`hermesParamSpanEnclosing` skips values when scanning for outer closes, the guard is hoisted, `isJsonNumber` is JSON's
grammar. Guard: hostile-value corpus loop asserting byte equality on `args.content`.

### A Hermes value keeps its own whitespace
Defect: `Edit` calls matched the wrong, un-indented occurrence or failed "string not found". Cause:
`parseHermesToolCall` trimmed all newlines and spaces; the template frames a value with exactly one newline per side.
Fix: `stripHermesValueFraming` removes at most one leading and one trailing newline (CRLF as one); the space trim
survives only for the literal probe (` 5 ` still emits `5`). Guard: the three whitespace tests and the `old_string`
assertions on the live capture.

### Gemma dropped its `<|"|>` opener on big content
Defect: gemma-4-e4b writing an HTML page dropped the opening `<|"|>`; the bare-value scan cut `content` at the first
`,` in the markup and shredded the rest into keys. Fix: a rich bare value (newline or `<`) runs to the confirmed
closing `<|"|>` (followed by `,`/`}`/`]`), else at top level to the final `}`. Guard: `Gemma 4 dropped … delimiter`
tests + corpus entry, found by `tests/test_tool_matrix_small.sh`.

### MiniCPM5 attribute XML, and a hand-written prefix ladder missing a rung
Defect: a MiniCPM5 call streamed a `<funct` fragment as content, then arrived again as a structured call. Cause: the
stream gate's `tail_prefixes` ladder for `<function` omitted `<funct`, and its unit test was a second copy of the same
list. Fix: `parseMiniCpm5ToolCalls` (CDATA-aware, dedups first-wins, truncation keeps complete pairs); the ladder test
is DERIVED by slicing the marker; the gate defers to the parser (`functionOpenerHoldsForTools`) so it stays a strict
superset. Guard: `EVERY growing prefix of <function holds`, `the gate is a strict SUPERSET of parseMiniCpm5ToolCalls`,
corpus `no flush boundary lands inside a tool-call opener, any family`.

### Qwen3.8 Flash Next wrote Claude's tool markup (#928)
Defect: long pi sessions got a complete `<invoke name="bash">…</invoke>` block as content with `stop`, and the agent
ended. Cause: the checkpoint samples Claude's dialect; no parser, gate or scrubber knew a bare `<invoke`. Fix:
`BARE_INVOKE` (ATEM's grammar without the namespace) parses after every native dialect as `.inferred`; the gate holds
`<function_calls>` / `<invoke …` prefixes; `bareInvokeMarkupCut` cuts only beside `<parameter ` or `</invoke>`. Guard:
corpus family `qwen4_exp`, the prose and in-value negative tests.

### LFM2.5's pythonic call grammar
Defect: LFM2.5 tool requests came back empty. Cause: `<|tool_call_start|>[get_weather(city='Paris',
metric=True)]<|tool_call_end|>` matched no parser, and the leak trim cut it. Fix: `pythonicLiteral` parses Python
literals (types live in the spelling) through one quote- and depth-aware scanner; the marker is unique, so the arm is
additive; truncation ships NAME + `{}`. Guard: pythonic parse tests + corpus.

### Inkling: single-token message markers and a self-reinforcing name loop
Defect: Inkling's first real pi session streamed the whole call as content, then pi's "Tool <|content_text|>bash not
found" taught the model to emit that name itself; back-to-back calls merged into one `{}`. Cause: the stream gate did
not know the invoke marker; the salvage name included a marker; the body scan ran to the next end tag; Inkling ships
no `generation_config.json`, so pi ran at 1.0/1.0. Fix: buffer on the invoke marker and hold bare identifier segments;
NAME = the trailing identifier run (`inklingTrailingNameRun`), never `<|`; body = the balanced object;
`applyFamilySamplingDefaults` top_p 0.95. Guard: four capture corpus entries, the no-`<|`-in-NAME invariant, the
stream replay. A format validated by curl has not met an agent.

### A duplicated key defeated container coercion (#402)
Defect: Qwen Code's `questions` array arrived as a string: `Parameter "questions" must be an array`. Cause: std.json
rejects duplicate keys and the tolerant emitter kept the repeat. Fix: `parseContainerAllowingRepeats` accepts the
document only when first-wins and last-wins parses agree. Guard: the identical-vs-conflicting duplicate test.

## The parse chokepoint and its invariants

### Types come from the schema, never the value's spelling
Defect: Claude Code rejected every `Edit` with `replace_all` "expected boolean, provided string" (Python's `False`),
and the inverse promoted an `old_string` of `false` to a boolean. The model cannot see its serialized request, so it
burned rounds "fixing" a correct value. Fix: `coerceToolArgsToSchema` runs after every parse at the ONE chokepoint
`server.parseToolCallsForRequest`, coercing both ways by the declared type; undecidable values stay as sent;
`--no-tool-autocorrect` gates only the coercion and hoist. Guard: coercion tests on captured bytes + the corpus
declared-type invariant (`toolCallConformsToSchema`).

### A required argument buried in a container is misplaced, not malformed
Defect: Gemma wrote the required `path` inside each `edits[]` item; pi refused three times and the model fell back to
rewriting the whole file. Pristine args are the tell that the parse layer is innocent. Fix:
`hoistMisplacedRequiredParams` lifts a required scalar only when every schema condition holds (absent at top level,
declared container whose items do not declare it, every item agrees on value and type); any ambiguity leaves the call
alone. Guard: chat tests, the chokepoint test, the corpus buried-param invariant.

### A heuristically inferred call must name a declared tool
Defect: a truncated data script produced a call named "George Washington"; pi answered "tool not found" and the model
retried the same 16K-token write. Cause: the raw-JSON fallback accepts any object with a string `"name"`. Fix: such
calls are `.inferred` and `filterInferredBySchema` drops undeclared names (ungated: it corrects our heuristic);
explicit tag calls are never filtered. Guard: the chokepoint tests and corpus "Hallucinated raw-JSON tool calls".

### The parse layer's hard invariants, pinned by real traffic
Defect: an 8-hour agent soak found five more weak-model parse bugs: a malformed `<parameter=limit=1` spilled a name
across lines, a repeated parameter made a duplicate key (std.json rejects it), a bare `<function=` with the wrapper
dropped leaked, a mangled container stayed a string. Fix: escape and dedup every name in every converter (Hermes,
Gemma; element dialect builds through `Stringify`); a bare `<function=` + `<parameter=` fallback;
`looseRepairContainer`; a final net in `parseToolCalls` that repairs or falls back to `{}`, so emitted args are always
valid JSON; broken output stays honest. Guard: `src/host/chat/tool_traffic_replay_test.zig` over
`src/fixtures/tool_traffic.jsonl` (grow it with `MLX_SERVE_RAW_DUMP_FILE` + `tests/harvest_tool_traffic.py`), the
conforming-call fuzz.

### `tool_choice: "required"` was advice, not a constraint
Defect: tool-eval-bench TC-45 got text for a required call. Cause: the instruction line reached only templates that
inline our tool prompt, and a line is advice anyway. Fix: the line reaches every template, and `armToolForce` commits
the dialect's opener (`forcedToolOpener`) after the think block closes, closing a thought that reaches 3/4 of
`max_tokens`; the request decodes plain. The opener stops before `=` (Qwen joins `=name` into one token) and skips the
model's pending token when it matches. Guard: `tests/test_tool_choice_required.sh`, `ToolForce` unit tests.

## Thinking and streams

### `</think>` inside a tool argument destroyed the call
Defect: a corpus entry writing a file about prompt formats lost its call. Cause: `indexOfThinkCloseTag` took the first
valid `</think>` anywhere. Fix: `thinkCloseIsToolCallPayload`: a close is payload when the nearest preceding
`<tool_call` opener is still open AND the block closes afterwards (each half protects a case the other breaks). Guard:
the unit test's ordinary and edge shapes.

### Think-tag leaks: a re-opened opener and trailing closer spam
Defect: replies whose whole visible content was a bare `<|channel>thought`, `thought`, sixteen `<channel|>` closes, or
an orphan `<tool_call|>`. Cause: the trailing strip ignored a pos-0 opener, the content-channel strip treated
`<|channel>thought` as content, and only openers were trimmed. Fix: report pos-0 unclosed openers;
`trimTrailingThinkClosers` loops off trailing `<channel|>`, `</think>` and `<tool_call|>`. Guard: corpus entries for
each shape + the universal no-tag-leak invariant.

### Tool markup leaked into reasoning and content (DSV4)
Defect: a DSML call the model wrote inside its thought came back in `reasoning_content`, was round-tripped and taught
the model its own malformed markup; a marker split across tokens and a mangled opener leaked as content. Fix:
`trimLeakedToolMarkup` cuts at the first wrapper opener, once, in the split wrapper; the end-of-stream flush
concatenates before cutting; `/v1/messages` non-stream cuts at emission because it parses the split text. Guard: three
capture corpus entries + the reasoning-side markup invariant.

### No `tools` = raw text on both paths
Defect: LFM2.5 answering in its own call format with no `tools` field returned `content: ""` non-streamed but raw text
streamed. Fix: the non-stream split keeps markup when the request has no tools, as vLLM, SGLang, llama.cpp and Ollama
do. Guard: `tests/test_no_tools_markup_passthrough.sh`.

### A template that always thinks (LFM2.5)
Defect: LFM2.5 streamed its whole chain of thought as the answer, thinking on or off. Cause: its template renders
`<think>` unconditionally; `streamThinkGate` keyed on `enable_thinking`, and every call site ANDed `promptOpensThink`
with it. Fix: whether the prompt ends inside a think block is a property of the rendered bytes (`promptOpensThink`),
passed to the gate as its own fact. Guard: `test_thinking_streaming.sh`, `test_thinking_tools.sh`,
`test_messages_stream_thinking_tools.sh`.

### `in_think_block` started from the request flag (Gemma)
Defect: Gemma streamed `content: ''`, `reasoning: '391'` where non-stream had it right. Cause: the stream assumed the
model starts inside a think block whenever thinking was requested; Gemma's template lets the model decide. Fix: the
stream tail is reasoning only on positive evidence (`streamTailIsReasoning`: the prompt opened a block or the model
emitted a real opener, `saw_think_open`). Guard: the stream/non-stream split invariant per arch. Test lesson: assert
the server invariant and branch on the model's choice, never "the model must think".

### A model that opens its own think block had no way in
Defect: Gemma 4, LFM2.5-8B-A1B and Muse streamed their thought as the answer. Cause: the model's own opener
(`<|channel>` + `thought`, a bare `<think>`, Muse's ` to=self<|message|>`) was swallowed by the marker skip before the
plain arm saw it. Fix: `modelThinkOpener` latches before the marker skip; `promptOpensMuseHeader` arms the header skip
at stream start. Guard: `test_thinking_tools.sh` 6b, the smoke matrix stream-vs-non-stream thinking check.

### `.hold_thinking` streamed nothing for seconds with tools
Defect: with tools and thinking the first reasoning byte arrived after 4.4 s instead of 0.05 s. Cause: the hold arm
was empty, so the thought went out in one delta at the end. Fix: reaching the gate proves the buffer holds no tool
markup, so reasoning streams incrementally; every later emit site sends only `unstreamedReasoning` (null when a cut
shrinks it: an SSE delta cannot be retracted); a reasoning budget keeps buffering. `/v1/responses` got the same arms
later. Guard: corpus streaming replay, `test_messages_stream_thinking_tools.sh`, `test_responses_streaming.sh` [B2].

### Muse channel headers are ordinary text between single-token markers
Defect: a thinking-off Muse stream shipped a bare `" to"` before the tool block. Cause: the header `assistant
to=<recipient>` is plain BPE text, and the hold checked `startsWith(lead, "to=")` against a two-byte buffer. Fix: an
unresolved header, including strict prefixes of `to=`, always holds; `to=self` is reasoning, `to=user`/bare is
content, any other recipient is an ATEM tool call (raw string values, JSON keeps its type). Guard: `museStreamVerdict`
tests, `tests/test_muse_glimmer.sh`.

### Thinking-off is enforced in the prompt; reasoning is always delivered
Defect: thinking-off on Muse and LFM2.5 still generated the whole reasoning pass, then discarded it. Fix:
`noThinkTailSuffix` commits ` to=user<|message|>` (Muse, no tools) or `</think>` (an unconditional opener) at the
render's tail; every delivery site splits with `splitThinkBlock(text, true, …)` and ships whatever reasoning exists.
`splitMuseChannels` classifies the eom-cut body, not the raw chunk (it once dropped the whole answer). Guard: the
render tail-commit tests, `tests/test_muse_glimmer.sh` [2]/[2b].

### Assistant-history reasoning must round-trip (Laguna)
Defect: Laguna thought on turn 1 and never again in a pi session. Cause: `chat.Message` had no `reasoning_content`, so
the template rendered every prior turn as `<think></think>`, the nothink signature. Fix: `Message.reasoning_content`
from the chat parser (and vLLM's `reasoning`) and from `/v1/messages` thinking blocks, omitted when absent, in the
tokenize-cache key. Guard: the serialize and laguna render round-trip tests.

### Muse renders round-tripped reasoning as history, so one loop poisoned every later turn
Defect: after one cut reasoning loop, every later Muse turn looped. Cause: Muse's template renders history reasoning
as a full `to=self` segment, so the model re-read its own loop. Fix: `dropPriorTurnReasoning` keeps reasoning only
after the last user message, Muse family only. Guard: the drop/keep render tests and the regenerated byte-parity
fixture. Tell: prompt tokens grow by the prior turn's reasoning length.

### `preserve_thinking` keeps the template's default (Qwen3.8)
Defect: long-context Qwen3.8 pi turns hit `near_repeat`; forcing `preserve_thinking:false` fixed that pack and then
made every follow-up re-prefill up to 44k tokens. Cause: dropping prior reasoning moves the prefix. Fix: the template
default stands; `chat_template_kwargs` `preserve_thinking:false` opts out. Guard: the `preserve_thinking` test beside
`serializeExtraContext`, `resolveChatThinking`.

## Loop stops

### A loop cut is an intentional stop, and fragment values never ship
Defect: gemma-4-26B looped inside a write call's `content`; the guard cut it silently, `finish_reason "stop"` was
upgraded to `tool_calls`, and the salvage shipped 1.1 KB of loop garbage without `path`. Fix: a loop cut reports
`stop` with `finish_details.type "repetition_loop"`, logs `[loop-stop]`, suppresses tool parsing; a truncated Gemma
call drops any value that runs to end-of-body. Guard: `loopStopDecision`, `shouldParseToolCalls`, the Gemma truncation
tests, corpus "Loop-stop truncated Gemma call".

### Short cycles and low-entropy files convict on span
Defect: Qwen3.8 designing a map wrote `"111111111111111111111111"` (24 per-digit tokens) and the exact tier cut it; a
turn later five tile maps were cut by the near-repeat tier at window fill. Cause: no content measure tells a file of
identical rows from a loop; what a file does and a loop never does is end. Fix: the exact tier needs
`degenerate_loop_min_span` (128), the long exact tier `degenerate_loop_long_min_span` (1024), near-repeat
`near_repeat_min_span` (4096). Guard: `degenerateTail acquits a low-entropy STRUCTURED file that ends inside the span
bar`, `tests/test_loop_stop_signal.sh`.

### A contract comment is read as a spec (#94)
Defect: an issue was filed against `coerceToolArgsToSchema` because its doc said "only SCALARS are touched", two weeks
after the container arm shipped. Fix: the comment describes the code and the container coercion is pinned by a test
named for the issue. Guard: that test.

## A DSML tool call came back as bare closers with finish "stop" (#980)
Defect: DeepSeek-V4.1 agent turns ended with text like `</｜DSML｜ calls>` and no `tool_calls`; the client took it as the final answer and went idle. Cause: the scrubber and the stream gate knew only the OPENER `<｜DSML｜`; once the opener and payload were gone (a call inside the think block, or never written) the closers matched nothing and rode out as content, and live they streamed because nothing held them. Fix: `</｜DSML｜` is a cut marker in `trimLeakedToolMarkup` and a hold (with its split lead-in `</`) in `streamShouldBufferForTools`. Guard: the corpus entry `lone DSML closers with no opener never leak into content` plus the two `#980` unit tests. Not fixed: a call the model writes inside its think block is still not recovered as a call.

