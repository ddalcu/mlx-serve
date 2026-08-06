# Tool calling & model-output formats — war stories (moved out of CLAUDE.md)

Full histories: live failures, measurements, diagnosis ladders, dead ends. The distilled RULES live in the root CLAUDE.md "Rules" section — when a rule changes, update the story here too. New gotchas in this domain: add the 1-3 line rule to root, the full story here.

### Hand-rolled JSON and control bytes (silent prompt-format downgrade)
Everything fed to the C++ Jinja engine is hand-serialized JSON (`chat.serializeMessagesJson` → `appendJsonString`), and nlohmann is strict: ONE raw control byte (< 0x20) anywhere in the history makes the whole render fail. `renderChatTemplate` then silently downgrades to `fallbackFormatChat`, whose generic tags are NOT the family's trained format — for Gemma 4 the fallback's `<start_of_turn>`/`<end_of_turn>` aren't even special tokens (its template uses `<|turn>`/`<turn|>`), so the model loses its stop token and degenerates into hallucinating both sides of the conversation. Live failure 2026-06-11: gemma-4-31b via pi went insane on turn 3 because turn 2's tool result captured an interactive npm CLI's ANSI codes (`\x1b[?25l`). Rules:
- Any string that can carry arbitrary bytes (tool results, model output echoed into history) must be escaped by a helper that `\u`-escapes ALL control chars. `appendJsonString` does this now; never add a new field with ad-hoc escaping (compare `server.zig jsonEscape`, `responses.zig jsonEscape`, `json_schema.zig writeJsonString` — all already correct).
- The corpus test "history round-trip serialization survives any byte content" (`src/format_corpus_test.zig`) pins the invariant for every corpus entry plus hostile ANSI/control-byte tool results — new corpus entries are covered automatically.
- A Jinja render failure logs at **warn** (`jinja render failed`), not debug. If a model suddenly emits wrong-family tags (`<start_of_turn>` from a Gemma 4, raw role names like `assistant`), suspect a silent fallback render before suspecting the model or spec-decode.

### Mangled tool-call argument JSON drops the whole call (small-model big-file escaping)
The OUTBOUND mirror of the inbound gotcha above: when a model emits a tool call, its `arguments` are hand-written JSON the model has to escape itself, and `std.json` (in `tryParseJsonToolCall`/`parseGemma4ToolCall`) is strict. A weak model writing a large file in one shot routinely mangles the `content` string two ways — (1) **raw control bytes**, literal newlines/tabs instead of `\n`/`\t` (the dominant case: code/HTML is full of newlines); (2) **unescaped inner double-quotes** (`<meta charset="UTF-8">`) and invalid backslash escapes (Windows paths, regex `\d`). Strict parse rejects the whole blob, so PRE-FIX the entire `writeFile` call was DROPPED and the file leaked into visible content — wasting the turn. Symptom signature: a writeFile/editFile that "didn't fire", the file content appearing as the assistant's chat text, and (app-side) a `SALVAGED_PATH` log with empty content. Fix: `looseRepairToolCallJson` (`src/chat.zig`) — a position-aware tolerant re-serializer in the parse-failure chain that re-escapes control bytes + inner quotes (a `"` closes a string only at a structural delimiter: `:` after a key, `,`/`}`/`]`/end after a value) and doubles invalid backslashes. It runs ONLY after strict parse fails and the result is re-validated by a strict re-parse, so valid JSON is untouched and a mis-recovery yielding invalid JSON is discarded (residual risk: a value string closed early on pathological literal `"}`/`",` content — still beats dropping the call). It does NOT handle truncation (that stays with `completeUnbalancedJsonObject`). Because `parseToolCalls` is the single chokepoint the server re-serializes from (`server.zig` `jsonEscape(tc.arguments)`), this one fix covers all four HTTP surfaces, streaming + non-streaming, every client. This is the model-agnostic answer to "the model fails the tool call due to escaping" — invisible to the model, helps 1B models enormously, leaves 100B models' valid JSON alone (a system-prompt writeFile-vs-heredoc steer fixes neither, and heredoc is the MORE fragile encoding since it double-escapes through shell + JSON). Guarded by the `parseToolCalls recovers …` instance tests in `chat.zig`, the "Small-model big-file escaping recovery" corpus section + the universal valid-JSON-args invariant in `src/format_corpus_test.zig` (red on revert: call → null → "expected a tool call, got none"), and this gotcha.

### Truncated tool-call OPENER is a truncation, not a ghost/malformed call (big-file write class)
The THIRD failure in the big-file-write class (after the escaping-recovery and Gemma-dropped-delimiter gotchas above): a model dumps a whole file into ONE Hermes/XML tool call and hits the token cap *mid-content*, so the call arrives as an OPENING tag with NO close (`<tool_call><function=writeFile><parameter=content>…novel…` — no `</parameter>`/`</function>`/`</tool_call>`). Live JFK-novel capture (2026-06-20): a 19k-char `writeFile` was cut off, **silently DROPPED, and leaked into chat as text**; the app then fired the WRONG nudge ("malformed tool-call tag… call it with proper JSON" — useless, the JSON was fine, just too long), the model retried identically, and the turn died with nothing written. Two bugs, two layers:
- **Server (`src/chat.zig` `parseToolCalls`, the `close_rel == null` branch):** it used to only try the JSON shapes (`balancedJsonObject`→`tryParseJsonToolCall`, `attr_name`). A truncated **JSON** writeFile salvaged its path; a truncated **Hermes/XML** one fell through to `break` and was dropped — a format-specific hole. Fix: in that branch, after the JSON attempts, also try `parseHermesToolCall` then `parseXmlElementToolCall` on `effective_text[content_start..]`. `parseHermesToolCall` breaks out of its parameter loop on a missing `</parameter>`, so it recovers the tool NAME with empty `{}` args (a *closed* parameter/function before EOS recovers WITH args — bonus). Recovering the NAME is enough; do **not** salvage the partial content (a half-written file is worse than a re-issued chunked write — the user explicitly rejected fragmentary writes).
- **App (`ChatTurnEngine.runAgentLoop`):** the existing truncation path is gated on `!receivedToolCalls.isEmpty`, so a *dropped* call (empty calls + `maxTokensHit`) never reached it and fell to the ghost path. Fix: before the `looksLikeGhostToolCall` block, route `maxTokensHit && receivedToolCalls.isEmpty && hasUnclosedToolCallOpener(content)` to the truncation nudge (`allowTruncationRetry()`, budget 2, the shared `truncatedToolCallNudge` = chunk + `append:"true"`), not the ghost nudge (budget 1, "use proper JSON"). `hasUnclosedToolCallOpener` = opener present with no matching close (`<function=`→`</function>`, `<tool_call>`→`</tool_call>`, `<|tool_call>`→`<tool_call|>`). This is defense-in-depth — with the server fix the truncated Hermes call now PARSES (so it takes the existing `!isEmpty` truncation path), but a future format that escapes the parser still gets the right nudge.
- Guarded by the `parseToolCalls recovers truncated <function=…>` / `recovers EOS-before-close-tag` instance tests (`chat.zig`), the "Truncated tool-call OPENER recovery" corpus section + the universal no-tag-leak invariant (`src/format_corpus_test.zig`, red on revert: call → null → "expected a tool call, got none"), and `ChatTurnEngineTruncationTests` (Swift). **Caveat:** the model-compliance gap (it *says* it'll chunk, then one-shots the whole file anyway) is only *mitigated* (right nudge + budget line), not solved — a hard server-side "single tool-call content exceeds remaining budget" guard is the heavier lever if it recurs.

### Hy3 tool call with a dropped `<tool_sep>` + mangled key-close leaks/loses args (weak-model delimiter-drop class)
Hunyuan 3's native tool format is `<tool_call:opensource>NAME<tool_sep:opensource><arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>…</tool_call:opensource>` (parsed by `parseHy3ToolCalls` in `src/chat.zig`). A weak/heavily-pruned model mangles it two ways at once — RAW capture 2026-07-16 (`pipenetwork/Hy3-REAP62`, via `MLX_SERVE_RAW_DUMP_FILE` on a live server): `<tool_call:opensource>bash</arg_value:opensource>\n<arg_key:opensource>command</arg_value:opensource>\n<arg_value:opensource>ls -la</arg_value:opensource></tool_call:opensource>` — it (1) DROPS `<tool_sep>` (closes the NAME with `</arg_value>`) and (2) closes the arg KEY block with `</arg_value>` instead of `</arg_key>`. The VALUE block and the key/value CONTENT are correct. The old parser keyed the NAME on `<tool_sep>` and **bailed entirely when absent**, so the whole call leaked as content (`finish_reason=stop`) → pi saw `bash({})` → "command required" → infinite retry loop (the model emits the format CORRECTLY on most turns — the server log shows many `[tool_calls]` finishes — so this is degraded adherence on SOME turns, the concrete cost of 62%-expert REAP pruning; the full `Hy3-oQ2e`/ox-ox builds are reliable). Fix, in two parts: (1) the name ends at the earliest of `<tool_sep`/`<arg_key`/`<arg_value`/`</tool_call`/`</arg_key`/`</arg_value` (`earliestIndexOfAny`), recovering it without `<tool_sep>`; (2) the arg loop runs whether or not a valid `<tool_sep>` was consumed, SCANS to the next `<arg_key>` (bounded by `</tool_call>`, so the stray name-close tag is skipped), and matches the KEY block's close TOLERANTLY (`</arg_key>` OR `</arg_value>`). The corruption is small and regular, so the FULL call recovers — `{"command":"ls -la"}`, live-verified end-to-end (client receives the command, not `{}`). The good-format path is byte-unchanged (its `<tool_sep>` is consumed, `<arg_key>` is found immediately, `</arg_key>` is the earliest close). Rule: a tag-format tool parser must never bail the whole call on ONE missing/mismatched delimiter — recover from the next structural marker and match closes tolerantly. Guarded by `parseToolCalls hy3: dropped <tool_sep> + mangled key-close still recovers name AND args` (chat.zig, real captured bytes, red-on-revert: call → null → `.?` panic; asserts `command == "ls -la"`), all four existing hy3 tool tests, and the corpus/traffic-replay invariants.

**Variant — dropped singular `<tool_call:opensource>` opener (2026-07-16 soak, same REAP62 model, RAW capture):** the format nests a PLURAL wrapper `<tool_calls:opensource>` around one or more singular per-call `<tool_call:opensource>` openers, and `parseHy3ToolCalls` keys the call start on the SINGULAR opener (the plural wrapper explicitly falls through — it shares the `<tool_call` prefix but has `s:`/`>` where the singular has `:`). REAP sometimes emits the plural wrapper then jumps STRAIGHT to the NAME, dropping the singular opener (`<tool_calls:opensource>\nwrite_file</arg_value:opensource>\n<arg_key:opensource>path…`), so the parser found no call-start and the whole (complete, well-formed) call LEAKED as content with `finish_reason=stop`. Same weak-model delimiter-drop class as the `<tool_sep>` drop above, one delimiter over. Fix: a plural-wrapper recovery arm keyed strictly on the SUFFIXED form `<tool_calls:` (the BARE DSV4/generic `<tool_calls>` wrapper must still fall through to its own parser — gating on `suffixedTagLenAt` alone regressed 7 DSV4/XML tests because it accepts bare `<tool_calls>` too) — if a real inner `<tool_call:sfx>` opener follows, defer to it (normal path, byte-unchanged); else treat the wrapper's end as the opener. Rule reinforced: recover from the next structural marker AND never let hy3 recovery steal the bare `<tool_calls>` wrapper of another format. Guarded by `parseToolCalls hy3: dropped singular <tool_call> opener (plural wrapper only) still recovers` (chat.zig, real captured bytes incl. quote-bearing content, red-on-revert: call → null → `.?` panic) + the `LIVE: dropped singular <tool_call> opener` corpus entry (`format_corpus_test.zig`, universal no-tag-leak + valid-JSON-args invariants).

### A server-side loop cut is a TRUNCATION: finish as "length", log it, and never ship fragment values (loop-stop truncation class)
The SIXTH failure in the big-file-write class (live 2026-07-14, pi → gemma-4-26B-A4B, plang/php.html): the model repetition-looped INSIDE a write call's `content` string ("server-side scripting language, " — a ~6-token cycle), the scheduler's degenerate-tail-loop guard (`scheduler.runSingleDecodeTick`, period ≤ 8 / 16 reps) correctly cut the generation mid-word — and then THREE downstream layers each told a small lie that compounded: the cut was **silent** (no log line — the post-mortem took log archaeology), it finished as **"stop"** (so `toolCallFinishReason` upgraded a server-cut fragment to `"tool_calls"` — a completed-looking call), and the Gemma truncation salvage shipped the **partial value** (`{"content":"<1.1 KB of loop garbage>"}`, no `path` — pi rejected on the missing required prop and echoed the garbage back into context). Symptom signature: a client "missing required property" rejection whose received args carry an obviously-degenerate, mid-word-truncated value, on a round whose token count sits far below max_tokens, with no server log line explaining the early end. Three rules, each fixed + pinned:
- **A loop cut reports `finish_reason "length"`, never "stop"** — the server truncated; "length" is the one reason that survives the tool-parse chokepoint, so client truncation recovery (pi's, and the app's `maxTokensHit` path) fires instead of schema-validating a fragment. Pure decision in `scheduler.loopStopReason` (null = healthy), pinned by its unit test (red on "stop").
- **Never cut silently** — the guard logs `[loop-stop] degenerate tail loop cut after N generated tokens`.
- **The Gemma truncation salvage obeys the Hermes rule: fragment values never ship.** When a `<|tool_call>` call has NO `<tool_call|>` close (`parseGemma4ToolCall(…, input_truncated=true)`), any value whose scan runs to end-of-body without its terminator — unterminated `<|"|>` string, unclosed JSON `"` string, bare/rich value with no separator, missing value after `key:`, and unclosed nested containers (a partial `edits[]` is fragmentary work too) — is DROPPED via the converter's `cut` signal; completed pairs before it survive (so `{path}` without the fragment `content` steers a clean re-issue, and the dangerous path-first ordering can never write a garbage file "successfully"). With the close tag present, behavior is byte-identical (the mlx_pi1 ends-with-`}` trim and mars.html dropped-delimiter salvage are complete-call paths and unchanged). Guards: the Gemma truncation instance tests in chat.zig (php.html capture, path-first, bare-rich, edits container, no-over-drop, JSON-quoted), the "Loop-stop truncated Gemma call" corpus section + the `tool_arg_absent` assertion (`src/format_corpus_test.zig` — red on revert: "fragment arg shipped"), and `loopStopReason` (scheduler.zig).

### Tool-arg types must come from the SCHEMA, never from the value's spelling (strict-client rejection class)
The tag tool formats carry no type information — `<parameter=replace_all>False</parameter>`, Gemma's `key:false` — so the parsers inferred the JSON type from the value's BYTES (`chat.isJsonLiteral`, used by `parseHermesToolCall` + `convertGemma4Value`; `parseXmlElementToolCall` was worse and typed *everything* as a string). That guess is wrong in **both** directions, and strict clients (Claude Code, pi, opencode) reject both:
- **String where a boolean belongs.** `isJsonLiteral` only knew lowercase `true`/`false`/`null`, so Python's `False` fell through to `appendJsonString` and shipped as `"replace_all":"False"`. Live 2026-07-09 (Qwen3.6-35B-A3B-Claude-4.7-Opus-Distilled via Claude Code, `~/.mlx-serve/logs/mlx-serve-11234.log:109471`): every `Edit` died on `InputValidationError: The parameter 'replace_all' type is expected as 'boolean' but provided as 'string'`. The model **cannot see its own serialized request**, so it burned six rounds "fixing" a value that was already correct, then abandoned `Edit` and rewrote whole files. Symptom signature: a client type-validation error naming a parameter the model demonstrably sent correctly, plus reasoning that spirals into "maybe there's a quoting issue in how I'm constructing the request".
- **Boolean/number where a string belongs.** The inverse: `<parameter=old_string>false</parameter>` (or `42`) — a code edit whose *content* spells a JSON literal — was promoted to a real bool/number → "expected string, provided boolean".
The schema is the only disambiguator, and it is already threaded to every parse site as `tools_json`. Fix: `chat.coerceToolArgsToSchema` runs after every parse, coercing SCALARS both ways per the declared `type` (tolerant boolean spellings `False`/`0`/`yes`/`true,` mirroring the app's `appendFlagIsTrue`); an undecidable value is left alone so the client's validation error stays honest, and a call that needs no coercion is byte-unchanged. All five HTTP surfaces go through the ONE chokepoint `server.parseToolCallsForRequest` (parse → bare-JSON inference → coerce) — **never call `chat_mod.parseToolCalls` directly from a handler**, or that surface silently regresses alone (the drafter-dispatch-hole lesson: output-equality tests cannot see it). **Escape hatch:** `--no-tool-autocorrect` (`server.g_tool_autocorrect = false`) gates ONLY the coercion at that chokepoint — args then pass through as the model emitted them. The parse-repair + valid-JSON safety net (below) always run, so this can never make args invalid; it only re-exposes the mistyped-value class to strict clients. Guard: `parseToolCallsForRequest: --no-tool-autocorrect leaves args verbatim` (server.zig). Guards: `coerceToolArgsToSchema` instance tests in `chat.zig` (verbatim captured bytes, both directions, pass-through cases) + the **universal declared-type invariant** in `src/format_corpus_test.zig` — any corpus entry that supplies a `tools_json` is auto-checked via `chat.toolCallConformsToSchema`, so new families are covered for free (red on revert: `tool argument type contradicts the declared schema … {"replace_all":"False",…}`).
- KNOWN GAP (same function, different bug): `parseHermesToolCall` does `std.mem.trim(u8, p_val, " ")` on every parameter value, so an `old_string`/`new_string` with meaningful leading indentation loses it. It survives today only because the un-indented needle still matches mid-line; a value that must begin or end with whitespace will silently mis-edit.

### A required tool arg the model BURIED in a container is misplaced, not malformed (buried-param class)
The sibling of the schema-coercion class above, and the reason it needs its own fix: the args are **valid JSON with correctly-typed values** — nothing to repair, nothing to coerce — they are simply in the wrong PLACE, which only the schema knows. A weak model that has internalized "the edit object holds everything about the edit" writes the required top-level `path` INSIDE each `edits[]` item:
```json
{"edits":[{"newText":"…","oldText":"…","path":"us_presidents/generate_site.sh"}]}
```
Live 2026-07-13 (pi, gemma-4-26B-A4B-it-qat-4bit): pi answered `Validation failed for tool "edit": - path: must have required properties path` three times; each rejection cost a full multi-thousand-token generation, and the model — which **cannot see its own serialized request** — re-emitted the identical call, then abandoned `edit` entirely and rewrote the whole file with `write`. Symptom signature: a client "missing required property X" error where X *is* present in the args, one level down inside a container; identical retries; eventual fallback from a surgical tool to a whole-file rewrite. **The parse layer is innocent here and you will waste time there** — `convertGemma4Object`/`convertGemma4Array` are a structural walk that preserves the model's own nesting and key order, and `coerceToolArgsToSchema` only rewrites values in place; nothing in the pipeline can relocate a key. Pristine, correctly-escaped args are the tell that no repair path fired.
- Fix: `chat.hoistMisplacedRequiredParams`, run at the ONE chokepoint `server.parseToolCallsForRequest` BEFORE coercion (so the lifted value is type-checked like any other top-level arg) and gated by `--no-tool-autocorrect` (it corrects the MODEL's output — unlike `filterInferredBySchema`, which corrects OUR heuristic and is deliberately ungated).
- Every condition is READ OFF THE SCHEMA, never guessed: the param is declared REQUIRED at top level and ABSENT there; it is declared a SCALAR (hoisting a container is too speculative); it sits in a DECLARED container arg whose item schema does NOT declare it (`containerItemDeclares` — a multi-file edit tool whose items legitimately carry their own `path` must never have it stripped out, which would DESTROY data rather than repair it); and every object in that container carries it with the SAME value and the declared type. **Any ambiguity — items disagreeing, two containers offering different values, a wrong-typed value — leaves the call untouched so the client's validation error stays honest.** A compliant call never re-serializes (verified: the hoist fires on 0 of the 4,480 real captured calls in the replay corpus).
- Guards: verbatim-captured-args + Gemma-parse-path + four over-reach tests in chat.zig, the chokepoint/escape-hatch test in server.zig, and the **universal buried-param invariant** in `src/format_corpus_test.zig` (`chat.requiredParamIsBuried` — every entry with a `tools_json` is checked automatically, so a future family producing this shape is covered for free). Red-on-revert verified at all three layers.

### The tool-call parse+coerce layer has HARD invariants; a replay harness pins them against real traffic
An 8-hour agentic soak (`claude -p`/`pi`/`opencode` × every local arch, 2026-07-09) surfaced FIVE more parse-robustness bugs beyond the schema-coercion one above — all in `parseHermesToolCall`, all triggered by weak models (0.5–4B) mangling the Hermes `<function=…>/<parameter=…>` form, all caught by replaying captured traffic through the real parse path. The invariants the layer must NEVER break, and how each bug violated one:
- **Emitted `arguments` are always valid JSON** (a client parses them). Violated two ways: a malformed `<parameter=limit=1` tag (no closing `>`) made the `>`-scan spill the "name" across a newline into `</parameter`, and the raw (unescaped) name interpolation produced invalid JSON — fix: `isPlausibleParamName` skips the malformed opener (recovering the well-formed sibling param) + the name is `appendJsonString`-escaped. And a repeated `<parameter=edits>` produced a DUPLICATE JSON key, which `std.json` rejects with `error.DuplicateField` — fix: dedup parameter names (first wins). **The Gemma `call:name{…}` converter (`convertGemma4Object`) had the SAME two bugs** — raw-interpolated keys + no dedup — fixed the same way (escape the key via `appendJsonString`, dedup by rolling back `result` on a repeat while still consuming the value). The class is now CLOSED across all three tag-format converters: Hermes + Gemma fixed, DSV4 `parseXmlElementToolCall` already immune (it builds args via `ObjectMap.put` + `Stringify`, which dedups + escapes — pinned by a characterization test). Any NEW tag-format converter must either build through `ObjectMap`/`Stringify` or escape+dedup by hand.
- **A bare `<function=…>` with the opening `<tool_call>` DROPPED must still parse.** The outer scan triggers on the substring `<tool`, which a lone `<function=Write>…</function></tool_call>` never provides (`</tool_call>` is `</too…`, not `<tool`) — so the whole Write leaked as visible text. Fix: a fallback that runs `parseHermesToolCall` on the whole text when a `<function=` opener AND a `<parameter=` are both present (so prose mentioning the tag can't false-fire).
- **Coercion never makes conformance WORSE and repairs what's safely fixable.** A container-typed arg (`edits` array) the model mangled with a missing comma stayed a string under strict parse; `looseRepairContainer` (the array-aware sibling of `looseRepairToolCallJson`) re-serializes it tolerantly, re-validated by a strict parse so a mis-repair is discarded.
- **Genuinely-broken model output is left HONEST, never fabricated.** An unbalanced/nested `edits` array or a `<tool_call>{invalid json}</tool_call>` from a 0.6B model that no tolerant repair can recover stays a string / stays unparsed — the client gets an honest type error and retries, which beats inventing data. These are counted, not failed.
- **Final safety net (structural, not per-converter):** `parseToolCalls` ends with a pass that strict-parses EVERY built call's arguments; if invalid, it runs `looseRepairToolCallJson` (re-escapes lone backslashes / control bytes / inner quotes), and if THAT still fails, falls back to `{}` (keeping the tool name — a client can retry a named call, but cannot parse invalid JSON at all). This makes "emitted args are always valid JSON" a property of `parseToolCalls` itself, so a pathological value a direct-construction converter copies verbatim (found live: a Gemma JSON-style string with a bad escape `\q` → `{"path":"a\qb"}`) can never reach a client. Any new converter is covered for free. Guard: `parseToolCalls: NO path emits invalid JSON args` (chat.zig) + the replay R1 invariant.
- Harness: `src/tool_traffic_replay_test.zig` replays `src/fixtures/tool_traffic.jsonl` (real `(tools schema, raw output)` pairs) through parse+coerce and asserts the HARD invariants (valid JSON, no-regression, byte-identical no-op on conforming calls, idempotence, no think/delimiter-tag leak); soft signals (broken-JSON non-conformance, unparseable-wrapper display leaks) are reported, not failed. Grow it by pointing `MLX_SERVE_RAW_DUMP_FILE=<path>` at the server (framed dump written by `server.appendRawToolDump` — schema + raw TOGETHER, because the 16 KB debug-log line cap makes scraping bodies unsound), driving agents, then `tests/harvest_tool_traffic.py --dump <path> --out src/fixtures/tool_traffic.jsonl`. Plus a deterministic fuzz (`fuzz: a conforming tool call round-trips…` in chat.zig) that generates 400 conforming calls whose values deliberately SPELL other JSON types and asserts byte-identity through parse+coerce.

### A heuristically-inferred tool call must name a DECLARED tool (hallucinated raw-JSON call class)
`parseToolCalls`' raw-JSON fallback (no tag syntax anywhere) takes the FIRST balanced `{…}` object in the text and — via `tryParseJsonToolCall`'s flat-shape synthesis — accepts ANY object with a string `"name"` key, treating every other key as arguments. That means a generation truncated by max_tokens mid-DATA-script hands the parser something like `{"name": "George Washington", "num": 1, …}` and the client receives a tool call named "George Washington" (live pi capture 2026-07-13, Qwen3.6-35B-A3B distilled writing a presidents site: pi answered `Tool George Washington not found`, the model retried the identical mega-write, two full 16K-token turns burned with zero progress). Symptom signature: a client-side "tool not found" error naming a piece of the model's DATA (not any real tool), right after a max-token truncation, with the "call"'s arguments being the rest of that data record. Fix: `ParsedToolCall.inferred` marks calls born from the bare raw-JSON fallback (array + single-object paths; tag/Hermes/Gemma converters stay explicit), and the chokepoint `server.parseToolCallsForRequest` runs `chat.filterInferredBySchema` — an inferred call whose name isn't declared in the request's `tools_json` (`chat.toolNameIsDeclared`, wrapped + flat forms, unparseable schema never drops) is discarded, so the text stays visible content and `finish_reason="length"` reaches the client untouched (its truncation recovery fires instead of a bogus tool loop). Rules: (1) EXPLICIT tag-format calls are never name-filtered — "tool not found" on a tagged call is model-visible feedback the model corrects from; a heuristic guess is not; (2) the filter is deliberately NOT gated on `--no-tool-autocorrect` (it corrects OUR heuristic's false positive, not the model's output); (3) any new heuristic inference path must set `.inferred = true`. Guards: the George Washington chokepoint tests in server.zig (drop + declared-name-keeps-parsing + tag-undeclared-kept), `filterInferredBySchema`/`toolNameIsDeclared`/provenance-marking unit tests in chat.zig, and the "Hallucinated raw-JSON tool calls" corpus entries — the corpus runner mirrors the chokepoint (filter → coerce), so every future entry with a `tools_json` is covered automatically (verified red-on-revert: `got: George Washington`).

### Gemma 4 tool calling
Templates render `role: "tool"` natively as `<|turn>tool` — no transformation. Don't add `tool_responses` field (causes duplicate content). Args serialized as JSON strings.

Gemma's custom arg format delimits strings with `<|"|>…<|"|>`. On LARGE content the model sometimes DROPS the opening `<|"|>` while keeping the closing one (live, gemma-4-e4b-it writing a full HTML page: `call:write_file{content:<!DOCTYPE…>…</html><|"|>,path:<|"|>x<|"|>}`). `convertGemma4Value`'s bare-value scan then terminates `content` at the FIRST `,`/`}`/`]` inside the markup (a viewport-meta comma, a CSS brace) and shreds the rest into bogus keys → invalid args → the write call carries garbage (or is dropped). Fix: in the bare-value branch, a non-literal value that is "rich" (contains a newline or `<`) runs to the CLOSING `<|"|>` when present (confirmed a closer by a `,`/`}`/`]` right after it, so a later field's opener isn't grabbed), else — at the top level only — to the object's final `}`. Plain short bare tokens (`command:ls -la`) keep the first-separator behavior. Same big-file-tool-call CLASS as the no-tag-leak / escaping-recovery work; guarded by the `Gemma 4 dropped … delimiter` instance tests in `chat.zig` + the `Gemma 4 dropped opening <|"|> on big content` corpus entry (`src/format_corpus_test.zig`). Surfaced by `tests/test_tool_matrix_small.sh` (the sub-4B cross-model tool-call matrix). The complementary mitigation is app-side: `writeFile` takes `append:"true"` and the tool description tells the model to chunk large files (~200 lines/call) so no single call truncates.

### Streaming with tools + thinking
Server buffers tokens to detect tool patterns. With thinking enabled, `<|channel>thought` is buffered (not flushed) until closing `<channel|>`. After generation, thinking is split into `reasoning_content`; channel tags stripped from visible content.

### Re-opened thought channel right after a close leaks the bare opener (think-tag-leak class)
Symptom signature: an assistant reply whose **entire visible content is a bare opener** — `<|channel>thought\n` (thinking off) or a glued `thought` (thinking on) — reaching the user / chat-history.json. 2026-06-19 live (gemma-4 agentic): the model CLOSED its thought channel and IMMEDIATELY re-opened a fresh one with nothing between (`…<channel|>\n<|channel>thought\n`), then the turn ended. The post-processing layer (`chat.stripThinkBlock`/`splitThinkBlock`) strips the leading CLOSED block first, leaving the re-opened opener at **position 0** of the remainder. Two traps combined: (1) `lastUnclosedThinkOpen` used to bail on a `pos==0` opener (it assumed leading openers were already handled), so the trailing-strip never cut it; (2) `splitThinkBlock`'s content-channel strip treated `<|channel>thought` as a `<|channel>` *content* opener and shaved off the prefix, leaving `thought`. `normalizeEmbeddedThinkBlocks` does NOT save this — it returns null for one-leading-closed-block + trailing-unclosed-opener, delegating to the (then-broken) trailing-strip. Rule: the trailing-strip must report a pos-0 unclosed opener (callers strip their leading block first), and a re-opened `<|channel>thought` is never a content channel. This is the same no-tag-leak class as the truncated-template-opener and mid-text-reopened-pair bugs; guarded by the `re-opened thought opener right after close` corpus entries (`src/format_corpus_test.zig`, both thinking on/off) + the universal no-tag-leak invariant, plus the instance tests in `chat.zig`.

**Variant — trailing CLOSE-marker spam (2026-07-09 soak, a Gemma reasoning variant, record 2151):** the model emitted reasoning, one `<channel|>` close, a content scrap, then SPAMMED 16 more bare `<channel|>` close markers. The leading strip cut the FIRST close; the trailing-strip only handled unclosed OPENERS (`lastUnclosedThinkOpen`), so the stray CLOSES leaked. A close marker is never valid at the tail of visible content. Fix: `trimTrailingThinkClosers` loops off trailing `<channel|>` / `</think>` (+ whitespace), applied by BOTH `stripThinkBlock` and `splitThinkBlock`'s content (returns a prefix slice, no alloc). This is why the soak found it and the tiny models didn't — large reasoning models degenerate into tag-spam under load. Guarded by the `trailing <channel|> close-marker spam` corpus entries (both thinking on/off) + `stripThinkBlock`/`splitThinkBlock` instance tests. Rule: strip trailing think/channel CLOSE markers, not just unclosed openers.

**Variant — orphan Gemma tool CLOSE `<tool_call|>` (2026-07-16 soak, gemma-4-26B-A4B):** with tools present and a trivial "no tools needed" probe (temp 0.7), the model degenerated into a bare 1-token `<tool_call|>` CLOSE with NO `<|tool_call>` opener. Tool-call detection keys on the OPENER, so `parseToolCalls` found no call and the orphan control token leaked as the ENTIRE visible content (server response `content == "<tool_call|>"`, `finish_reason=stop`). Same trailing-orphan-close class as the `<channel|>` spam — a tool CLOSE is never valid at the tail of content, and `parseToolCalls` runs BEFORE the strip and extracts any real call, so any residual `<tool_call|>` reaching content is orphan by construction. Fix: `trimTrailingThinkClosers` also loops off a trailing `<tool_call|>`. Degenerate + stochastic (didn't reproduce in 6 re-runs — a single log artifact), so the guard is deterministic/hermetic, not a live red-on-revert: the `stripThinkBlock removes orphan Gemma <tool_call|> close` instance test + the `orphan <tool_call|> close never leaks` corpus entry (universal no-tag-leak invariant).

### Dropped assistant-history reasoning starves reasoning-persisting templates into nothink (laguna, 2026-07-29)
Symptom signature: a model whose template PRE-OPENS `<think>` thinks on the FIRST turn of a session and never again — while every part of the flag wiring reads correct (request logs `thinking=true`, the template reads `enable_thinking`, the render succeeds, no jinja fallback WARN). Live pi agent on Laguna XS: ONE `reasoning_content` delta on the 2-msg opening turn (log line 87925, port 11234), then ZERO across the next 13k chunks of the same session. Mechanism: pi (like vLLM clients) round-trips `reasoning_content` on assistant HISTORY messages, but `chat.Message` had no field for it — the chat parser dropped it, `serializeMessagesJson` never emitted it, and laguna's `chat_template.jinja` (which persists reasoning across turns: history assistants render `<think>{message.reasoning|reasoning_content}</think>`) rendered EVERY prior turn as the empty `<think></think>` — the GLM-family nothink signature. Sitting inside the pre-opened `<think>` of the current turn, the model's argmax continuation after a history of empty thinks is an immediate `</think>`; the stream gate consumes that lone close, so the output shows pure content and reads as "thinking never enabled". Qwen/Gemma never hit this: their templates strip history reasoning by DESIGN and the models are trained for it. Fix is universal plumbing, template-decided behavior: `Message.reasoning_content` carried from the chat parser (`server.messageReasoningFromObj`: `reasoning_content` then vLLM's `reasoning` spelling, non-empty strings only — an empty string would render the exact signature the field exists to avoid) and from `/v1/messages` history `thinking` blocks, emitted by `serializeMessagesJson` (key OMITTED when absent — templates gate on `is string`), and hashed into `TokenizeCache.keyFor` (two histories differing only in reasoning must not collide on one cached tokenization). Templates that never reference the field render byte-identical prompts. Clients that DON'T round-trip reasoning still see reasoning-persisting models go quiet after turn 1 — that is the model/template contract, not our wiring. Guards: `serializeMessagesJson carries assistant reasoning_content` + the laguna-fragment render round-trip (chat.zig), `messageReasoningFromObj` cases (server.zig), `TokenizeCache key distinguishes assistant reasoning_content` (tokenize_cache.zig).

### Inkling's template raise_exceptions on our own extra-context values, and its output is MESSAGES, not tag pairs (2026-07-30)
Adding `inkling_mm_model` (Thinking Machines Inkling Small) surfaced a new sub-class of the silent-fallback family: the failure lives in the TEMPLATE's input contract, not the model's output bytes. `chat_template.jinja` (verbatim in `src/fixtures/inkling_chat_template.jinja`) maps `reasoning_effort` strings through its own table (`none/minimal/low/medium/high/max` → a numeric "Thinking effort level: N" system line) and `raise_exception`s on anything else — including the `"no_think"` `serializeExtraContext` sends for hy3. It also raises when a history tool call's `arguments` is a JSON STRING (we already serialize objects — the test at "embeds valid-JSON arguments as object" is now load-bearing for a second family), and it renders tool declares/calls with `tojson(sort_keys=true, separators=(",", ":"))`, which jinja_cpp threw NotImplemented on. Any one of these = render failure = silent `fallbackFormatChat` = wrong-family tags = degeneration, with the flag wiring reading perfectly. Fixes: `sort_keys` implemented in jinja_cpp's `value_to_json` (byte-wise sort == Python's code-point sort for UTF-8); `serializeExtraContext` sniffs the family ("Thinking effort level" in the template) and sends `"none"` for thinking-off; `serializeMessagesJson` now emits the tool call `"id"` — the template names a tool RESULT by matching `message.tool_call_id` against history `tc.id`, and without it every result rendered nameless. All pinned by the hermetic real-template render test (both thinking arms + tool round + reasoning round-trip).

Output side: the model emits role-less MESSAGES — `<|content_thinking|>R<|end_message|>`, then `<|message_model|><|content_text|>C<|end_message|>`, tool calls as their own `NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>` messages, EOS `<|content_model_end_sampling|>` (200006). Every marker is a SINGLE special token, which makes streaming tractable: a marker can never split across deltas, so `chat.isChannelMarkerToken` (shared by all flush paths) filters exactly, `streamThinkGate` decides early off the leading marker, and the three per-surface `in_think_block` machines just needed the opener (`<|content_thinking|>`) and close (`<|end_message|>`) registered plus content-marker strips after the close. `splitInklingChannels` serves both `splitThinkBlock` AND `stripThinkBlock` — the thinking-OFF non-stream path uses the latter, which is exactly where the first live run leaked `<|content_text|>4<|end_message|>` as content. `parseInklingToolCalls` runs ahead of the tag families (its marker is unmistakable); the payload's `"name"` is authoritative with the message-prefix NAME as fallback, and a truncated payload salvages NAME + `{}` per the hard rule. Corpus family "inkling" (6 live-captured/live-shaped entries) + the Inkling markers in `leak_tags` auto-cover future shapes.

### The first REAL Inkling agent session: a compounding four-mechanism loop (2026-07-30)
The curl-validated Inkling tool support above broke on its first real workload (pi v0.83.0, "build quake 1 in threejs", REAP25 on the app server). Four mechanisms, in causal order, each amplifying the next — raw captures in `mlx-serve-11234.log` lines 61961/62828/63285/63543, now corpus entries:

1. **Streaming leaked the whole call as content.** Under pi the model opens tool turns `<|message_model|><|content_text|>bash<|content_invoke_tool_json|>{…}`. `streamThinkGate` saw the `<|content_text|>` head → `.flush_text`, and `streamShouldBufferForTools` didn't know the invoke marker — so the NAME and the full JSON streamed as visible deltas (only the marker TOKENS were filtered). End-of-turn parse still extracted the calls, so pi got BOTH leaked text and tool_calls; the leak landed in assistant history and contaminated every later turn. Fix: `streamShouldBufferForTools` buffers on the invoke marker (a single special token — arrives whole) and HOLDS while the segment after the last boundary marker is a bare identifier run 1..64 (`inklingSegmentCouldBeToolName`) — prose disambiguates within a token (space/punct), a call at the invoke marker; an EMPTY segment never holds, so thinking splits stay prompt. The /v1/messages stream had drifted back to its own inline subset predicate — re-unified on the shared function.
2. **The salvage name manufactured garbage, and pi's error echo taught the model the garbage.** The fallback prefix-name was "text since the last `<|message_model|>`/`<|end_message|>`", which included the `<|content_text|>` marker → tool name `<|content_text|>bash` → pi: "Tool <|content_text|>edit not found" → the model began emitting `{"name":"<|content_text|>bash","args":{}}` ITSELF (capture 63285). A self-reinforcing loop the parser started. Fix: NAME = trailing identifier run `[A-Za-z0-9_.-]` before the invoke marker (`inklingTrailingNameRun`); payload names containing `<|` get the same treatment; universal corpus invariant: a parsed NAME never contains `<|`. Related: the bare-JSON inference block is now suppressed on invoke-marker text — it resurrected (as an `.inferred` call) the exact payload the Inkling parser had deliberately skipped for having no recoverable name.
3. **Back-to-back calls without `<|end_message|>`** (capture 62828: `{…}}write<|content_invoke_tool_json|>{…}`): body extraction scanned to the NEXT end tag, so call 1's "body" swallowed call 2 and the JSON parse failed → both calls degraded to one `{}` salvage. Fix: body = the balanced `{…}` object at body_start (string-aware brace scan, the `balancedJsonObject` helper), `pos` advances past the object; no balance by end of text = truncation → NAME + `{}` as before.
4. **Wild sampling**: pi omits temperature/top_p/top_k (confirmed null in the logged request) and Inkling ships NO generation_config.json anywhere — requests ran 1.0/1.0/off, the exact 2026-07-13 pi-budget-burn class, showing up as duplicate identical calls (capture 61961). TM publishes no recommendation (their tooling is greedy-only), so the `applyFamilySamplingDefaults` inkling arm's top_p 0.95 is documented as OUR tail cut; body > flags > this default unchanged.

Guards: 4 verbatim-capture corpus entries (duplicate-with-separators, back-to-back-no-separator, marker-echoed name, content_text-prefixed single), the `last_tool_name` corpus field, the universal no-`<|`-in-NAME invariant, the "streaming tool buffer never flushes Inkling call text" replay (atomic-marker tokenization — markers are single tokens, so the replay feeds them whole), and unit tests for the name-run/balanced-body/hold logic. Meta-lesson: a format validated by curl smoke tests has not met an AGENT — the failure modes only compose under multi-turn history contamination, client error echo, and omitted sampling params.

## A transcribed template's whitespace is a token-level contract (dsv4/0731, 2026-07-31)

DeepSeek-V4-Flash ships no `chat_template`, so `src/fixtures/dsv4_chat_template.jinja`
is our transcription of the release's `encoding/encoding_dsv4.py`, and the converter
injects it into every mirror we build. Session 2 validated it 5/5 by hand; when 0731
landed (its ONLY encoder change being `reasoning_effort` → low|high|max) that ad-hoc
check was rebuilt as a checked-in guard, `tests/dsv4_template_ab.py`, rendering both
sides over the shapes the server actually emits and demanding BYTE equality.

It went 8/14 on the first run, and every failure was informative:

1. **The reference SILENTLY DROPS tools attached to a user message.** It reads
   `msg.get("tools")` off the first message and renders the block only from a
   system/developer turn. Its canonical way to express "tools, no system prompt" is
   an EMPTY system turn — which renders `content + "\n\n" + tools`, i.e. a leading
   `\n\n` before `## Tools`. Our template emitted the separator only `if has_system`,
   so every no-system tool request — the shape most clients send — produced
   `<bos>## Tools` where the model was trained on `<bos>\n\n## Tools`. Two bytes,
   but they retokenize everything after the bos. Fixed to unconditional; pinned in
   both the A/B and the hermetic Zig render test.

2. **`encode_arguments_to_dsml` `json.loads()`es the arguments** — the reference wants
   a JSON STRING where we serialize OBJECTS (the Inkling rule: a string breaks other
   families, so we keep objects). Handing it a dict does not raise: it lands in the
   except branch and renders ONE parameter literally named `arguments` wrapping the
   whole JSON. A harness that fed both sides "the same" message would have reported
   our per-key rendering as the bug.

3. A conversation ENDING on an assistant turn while still asking for a generation
   prompt is not a shape the server produces (the reference has its own `wo_eos`
   continuation path). The parallel-call case now ends with both tool results — the
   real agent shape, which also exercises consecutive-`tool_result` merging.

The lesson generalizes past this family: when A/B-ing a transcription against a
reference implementation, convert the INPUT shapes to each side's own contract and
compare only the OUTPUT. Every remaining difference is then a real defect in one of
the two — and here the whitespace one was ours, silently, in production shapes.

## reasoning_content is a client-visible, history-round-tripped field (DSV4, 2026-08-01)

A live agent session on DeepSeek-V4-Flash-0731 came back with a full DSML call block
sitting inside `reasoning_content`:

```
"reasoning_content": "…Let me first check the working directory structure.
<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"listFiles\">\n
<｜DSML｜parameter name=\"path\" string=\"true\">quake</｜DSML｜parameter>\n
</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
```

The model had emitted a complete tool call INSIDE its think block, closed the block,
then issued two different calls in content. `parseToolCalls` works on the post-think
text by design, so the in-thought call is deliberately skipped — but nothing then
removed it, and `splitThinkBlock` handed the whole thing back as reasoning. The client
round-tripped it into the next request's history (the assistant-history reasoning
rule), so from that turn on the model was being shown its own malformed markup as an
example of what thinking looks like. Same family as the Inkling error-echo loop: the
server taught the model the mistake.

Two more shapes from the same log, same class, different field:

- **A marker split across tokens beats a per-token filter.** DSV4 spells the marker as
  `<` then `｜DSML｜`. With tools declared the stream buffers correctly (`<` is a
  tail-prefix), but the END-of-stream flush — reached because nothing parsed — looped
  the held tokens straight out as content deltas. `isChannelMarkerToken` cannot help:
  neither piece IS a marker. The flush now concatenates first, then cuts.
- **A mangled opener leaks the whole block.** `<｜DSML｜toolinvoke name">…` (the model
  fused `tool_calls>\n<｜DSML｜invoke name=`) parses as nothing and became the visible
  answer verbatim.

Fix: `chat.trimLeakedToolMarkup` cuts visible text at the first tool-call WRAPPER
opener (`<｜DSML｜`, `<|tool_call`, `<tool_call`, `<tool_calls:`, Inkling's invoke
marker — plus, for Inkling, the trailing identifier run before it, since the NAME
precedes the marker there). It is applied ONCE, in a wrapper around `splitThinkBlock`
/ `stripThinkBlock`, so a new split arm cannot forget it. The whole tail goes rather
than just the block: a wrapper we could not parse has no reliable end, and shipping
half of it is the same leak.

The one caller that must NOT get the cut is `/v1/messages` non-streaming — alone among
the surfaces it reassigns its working text from the split result and then hands that
to `parseToolCallsForRequest`, so cutting first would make a real call unparseable.
That path uses `splitThinkBlockKeepingMarkup` / `stripThinkBlockKeepingMarkup` and cuts
at the text-block emission instead. Every other surface (chat non-stream + stream,
`/v1/responses`, `/v1/messages` streaming) already parses from the raw text.

Guards: three verbatim-capture corpus entries and a SECOND universal corpus invariant —
`reasoning_content` is checked against the tool-markup list exactly as content is
checked against `leak_tags`. KNOWN GAP recorded in the test: a re-opened
`<|channel>thought` marker can still sit INSIDE reasoning (excising an interior marker
needs an allocation the alloc-free `ThinkSplit` contract doesn't have, and Gemma's
template strips history reasoning, so it is a rendering wart, not a prompt
contaminant). Second gap: streaming with NO tools declared has no buffer to cut from,
so a model emitting call markup unprompted still streams it.

## LFM2.5: a pythonic call grammar, and a template that always thinks (2026-08-04)

Two separate bugs, both surfaced by getting `mlx-community/LFM2.5-2.6B-8bit`
serving. Neither is a parser tolerance failure — both are cases where a new
template family carries a fact our pipeline had no way to learn.

### 1. The tool-call grammar

A tools request came back with empty content and no `tool_calls`. Raw output,
captured by rendering the model's own template offline and posting it to
`/v1/completions` so nothing in the tool pipeline could touch it:

```
</think><|tool_call_start|>[get_weather(city='Paris', days=3, metric=True, tags=['trip', 'eu'])]<|tool_call_end|>
```

Nothing in `parseToolCalls` speaks that. The wrapper was being cut by
`trimLeakedToolMarkup` (it already lists the `<|tool_call` prefix for Gemma),
which is why the failure presented as empty content rather than a leak — the
safety net did its job and there was nothing behind it.

The load-bearing detail is the VALUES. `format_arg_value` in the template
renders strings as Python reprs but containers via `tojson`, so a reasonable
guess is "strings are pythonic, everything else is JSON". The model does not
agree: it emits `True` and `['trip', 'eu']`, full repr. A JSON-only value
reader gets the boolean and the array — the two commonest non-string argument
types — wrong on the first real call. So `pythonicLiteral` parses the literal
set (quoted strings either way, `True`/`False`/`None`, ints, floats, lists,
dicts) and types the value at parse time. In this grammar the type is
knowable from the spelling, so the schema never has to be consulted.

Everything structural runs through one quote-and-depth-aware scanner
(`pythonicScan`), because the arg separator, the `=`, the dict `:` and the
closing paren all need the same blindness to a separator sitting inside a
value — `shell(cmd='ls -la (tmp), [x]')` breaks a naive scan in three places
at once.

Additive by construction, which was the whole point given how much live
tolerance the existing arms encode: `<|tool_call_start|>` is emitted by no
other family, the generic `<tool` scan never sees it (Gemma keys the exact
`<|tool_call>`, with the `>`), and both the streaming buffer gate and the
leaked-markup cut already covered it through the `<|tool_call` prefix. The
only wiring was one call in `parseToolCalls`.

### 2. The template thinks whether or not you asked

With the parser in, streaming still delivered the model's entire
chain-of-thought as the answer — with and without tools — while non-streaming
was clean. The split:

```
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n<think>" -}}
{%- endif -%}
```

No `enable_thinking` branch anywhere in the template. LFM2.5 always reasons.
So with thinking off the opener is in the PROMPT and never in the output, and
the model's first tokens are tag-free prose that happens to be reasoning.
Non-streaming survives because by the time it splits, the `</think>` has
arrived and the leading strip finds it. Streaming has to decide live, and
`streamThinkGate`'s only signal was `enable_thinking`.

`server.promptOpensThink` already computed exactly the missing fact, but every
call site ANDed it with `enable_thinking` — so the one case that needed it was
the one case it was suppressed in. Whether a prompt ends inside a think block
is a property of the rendered bytes, not of our request flag.

Two changes, both shaped for containment because this is the layer that must
not move:

- `chat.streamThinkGate2` takes the fact as a 4th argument; the 3-arg
  `streamThinkGate` stays as "no prompt opener", so every existing call site
  and test pins the behavior it always had (a test asserts the two agree over
  the old cases, both flags, both directions).
- The thinking-OFF case gets its OWN stream arm rather than widening the
  reasoning arm — that one EMITS `reasoning_content`, and with thinking off
  the block must be dropped. Nothing in the new arm can run when thinking is
  on.

The term can only fire on thinking-OFF plus a literal open tag at the prompt
tail. A scan of every local checkpoint's template found LFM2.5 is the only one
where that combination is reachable — everything else renders the closed
`<think></think>` signature when thinking is off, which `promptTailOpensThink`
already returns false for. Verified live on Qwen3.6-27B across all four
stream × thinking × tools combinations, plus `test_thinking_streaming.sh`
(13/13), `test_thinking_tools.sh` (27/27) and
`test_messages_stream_thinking_tools.sh` (6/6).

Lesson for the next family: check whether its think opener is CONDITIONAL
before assuming the request flag describes what the model is doing.

### 3. Issue #94 was a stale comment

Filed against `coerceToolArgsToSchema` on the strength of its doc comment
("Contract: only SCALARS are touched"). The array/object arm had shipped in
60ba5ec two weeks earlier; the comment never got updated. The container
coercion is now pinned by a test named for the issue, and the comment
describes the code. A contract comment is read as a specification — by people
and by whatever is comparing your implementation against another engine's.

## `in_think_block` started from the request flag, not the prompt (Gemma, 2026-08-04)

Found while regression-testing the LFM2.5 work against `gemma-4-e4b-it-4bit`.
Streaming and non-streaming disagreed on the same request:

```
enable_thinking: true, "What is 17 * 23? Just the number after thinking."
  non-stream → content '391', reasoning None      # correct
  stream     → content '',    reasoning '391'     # the whole answer misfiled
```

The streaming loop initialises `in_think_block = enable_thinking` — it ASSUMES
the model begins inside a think block whenever thinking was requested. That is
true only for templates that pre-inject the opener. Gemma renders a bare
`<|turn>model\n` and lets the MODEL decide, so a turn it answers directly
carries no think markup at all: no opener to recognise, no close tag to split
on, and at end-of-stream the buffer was flushed as `reasoning_content` on the
strength of `in_think_block` alone. Content came back EMPTY. Any client
rendering `content` showed nothing.

Note the shape of the miss: the answer was short ("391"), so the opener-skip
logic — gated on `think_buf.items.len >= 7` — never even ran. A longer
non-thinking answer would have reached its final else ("not a known opener —
the template must have injected one") and been misfiled just the same, for a
different reason.

The non-streaming path had it right all along, because `splitThinkBlock` asks
for evidence: no opener + no close + not template-opened ⇒ content. So the fix
is to make the stream flush use the same rule
(`chat.streamTailIsReasoning(in_think_block, prompt_opened_think,
saw_think_open)`) — reasoning only with POSITIVE evidence a block was open:
the prompt opened one, or the model emitted a literal opener.

`saw_think_open` is new and deliberately distinct from the existing
`skipped_think_open`, whose else-branch also fires for "no known opener, assume
the template injected one" — precisely the case that has to be told apart. It
is set only in the three branches that recognise a REAL opener (`<think`
family, Inkling's `<|content_thinking|>`, Gemma's `<|channel>thought`).

Truncated thoughts are unaffected: a pre-injecting template sets
`prompt_opened_think`, so a thought cut by max_tokens is still reasoning and
still never leaks into content.

### The tests were asserting model behavior

Three integration assertions failed on models that were behaving correctly, all
the same class — asserting what the MODEL chooses rather than what the server
guarantees:

- `test_thinking_streaming.sh` Test 2 demanded reasoning >50 chars. Gemma
  answers that prompt directly. Now: either a streamed think block, or a direct
  answer as content — never the broken third state (answer filed as reasoning
  with empty content), which is what the old assertion let through unnoticed.
- `test_thinking_tools.sh` Test 2 demanded content. Laguna-XS spends >500
  tokens thinking about 15x17 and ends at `finish_reason: length` still inside
  the block — empty content is the truncated-thought rule working.
- Tests 4/8 demanded reasoning before a tool call. Laguna-XS closes its
  pre-opened block empty and calls the tool in ~35 tokens; LFM2.5 sometimes
  spends the whole budget thinking and emits neither (~2 runs in 3, so the arm
  was also nondeterministic).

Each now asserts the invariant and branches on the model's choice. The general
rule: an integration assertion that a model MUST think, MUST answer, or MUST
call a tool is a checkpoint-specific expectation wearing a server test's
clothes — and it either fails on the next family or, worse, passes while
hiding a real defect.

`test_format_matrix.sh` also learned to look under the sibling model root
(`~/.mlx-serve/models` vs `~/.lmstudio/models`): its gemma4-e4b arm had been
skipping while the checkpoint was present, which is missing coverage that reads
as a pass.

## `.hold_thinking` was an empty block, so tools+thinking streamed nothing for seconds (2026-08-04)

Reported as "with tools and thinking, streaming takes a long time before I see
anything; without tools I see content right away". Measured on LFM2.5-2.6B:

```
                            1st reasoning   1st content   prefill
thinking ON,  no tools           0.05s          n/a         8ms
thinking ON,  WITH tools         4.40s          n/a        27ms
```

Prefill is 8-52 ms in every combination, so the wait was not prefill. With
`tools` present every token goes into a buffer for tool-call detection, and the
think gate returns `.hold_thinking` until `</think>` arrives — and that arm was
literally empty:

```zig
.hold_thinking => {
    // Incomplete thinking block — keep buffering until closed
},
```

So the whole thought landed in ONE delta at the end. 4.4 s here, and it scales
with the length of the thought.

The fix is small because **the tool side was already fine-grained**:
`streamShouldBufferForTools` runs immediately above and holds on partial
prefixes down to a bare `<`. Reaching the gate at all therefore PROVES the
buffer contains no tool markup and no partial marker at its tail — those bytes
are reasoning and nothing else, and they can go out now. 4.40 s → 0.09 s,
matching the no-tools path.

What the change actually has to get right is not the emission, it is that
**three other sites emit reasoning for the same turn**: the `.split_think` arm,
the end-of-stream tool-call path, and the end-of-stream no-tool-call path. Each
now sends only the remainder via `chat.unstreamedReasoning(reasoning,
reasoning_streamed)`. Its one interesting case is a split that SHRINKS — a tool
marker appearing mid-thought moves `trimLeakedToolMarkup`'s cut backwards, so
the reasoning gets shorter than what was already sent. It returns null there:
an SSE delta cannot be retracted, so sending nothing further is honest, and
resending from the top would duplicate the entire thought.

A reasoning BUDGET keeps the old buffering. Capping what the client is allowed
to see cannot be reconciled with having already streamed it, and the guard
(`reasoning_budget < 0`) also means `reasoning_streamed` is only ever advanced
when no budget is set — which is what leaves the budget-truncation branches at
the two end sites provably untouched.

Verification worth copying: stream vs non-stream reasoning compared
BYTE-IDENTICAL at temperature 0 on LFM2.5 and Gemma, including the tool-call
turn. That comparison is UNSOUND on Qwen3.6-27B-4bit — INT4 near-tie argmax
plus MTP makes two temp-0 runs diverge past ~30-80 tokens (documented), and the
diff looks alarming until you notice the two runs generated different text. For
that model the property to test is self-duplication within ONE stream (the
reasoning must not contain its own head twice): 637 reasoning deltas, head
count 1, no tag leak, tool calls intact.

This is a stopgap for the symptom. The real fix — an incremental parser that
emits diffs and holds back only the minimal ambiguous suffix, which is what
vLLM's `extract_tool_calls_streaming` and llama.cpp's `common/chat.cpp` partial
parse do — is in TODO.md.

---

## A `</think>` inside a tool ARGUMENT destroyed the whole call (2026-08-05)

Found by WRITING a corpus family, not by a live failure — which is the point of
having one.

Writing a `format_corpus_test.zig` family for a GLM-tag arch, one entry gave a tool call a
realistic argument value: a model writing a file about its own prompt format.

```
<tool_call>write
<arg_key>content</arg_key>
<arg_value>The model closes a thought with </think> and "quotes" it.</arg_value>
</tool_call>
```

Expected a parsed call; got `expected a tool call, got none`, with the raw text
falling through as content.

The parse chain works on POST-think text, and `chat.indexOfThinkCloseTag` is
the single scan every surface uses to find the block boundary — split, strip,
and the streaming gate. It takes the FIRST syntactically valid `</think>`
anywhere in the text, with no reference to what encloses it. So the split cut
through the middle of the call: everything before the close became reasoning
(or was dropped, thinking-off), and the fragments after it leaked as content.
The call was simply gone.

This is general to every `<think>` family — qwen, laguna, dsv4 — and the
traffic that hits it is ordinary: coding agents write files about prompts.

Fix (`chat.thinkCloseIsToolCallPayload`): a close is payload iff the nearest
preceding `<tool_call`-family opener is still OPEN at that point (no
`</tool_call` between them) AND the block does close afterwards. Both halves
are load-bearing and each protects a case the other breaks:

- **Without the first**, a call the model emitted and CLOSED inside its thought
  (`<think>let me try <tool_call>read</tool_call> hmm</think>The answer is 4.`)
  would make the real `</think>` after it look like payload, and the answer
  would vanish into reasoning.
- **Without the second**, an unclosed opener inside a thought — the documented
  leaked-markup case that `trimLeakedToolMarkup` handles downstream —
  (`<think>starting <tool_call>partial</think>Answer.`) would swallow the answer
  that follows the close.

Only the `<tool_call` family is considered: it is the one whose bodies carry
free-form argument text. Ordinary shapes (`<think>r</think>a`, and a think
block followed by a real call) are untouched, pinned in the unit test.
