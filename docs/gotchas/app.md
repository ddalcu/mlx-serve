# Swift app — war stories (moved out of CLAUDE.md)

Full histories: live failures, measurements, diagnosis ladders, dead ends. The distilled RULES live in the root CLAUDE.md "Rules" section — when a rule changes, update the story here too. New gotchas in this domain: add the 1-3 line rule to root, the full story here.

### writeFile body crammed into the `append` flag; missing-content error must not claim truncation (big-file write class)
The FOURTH failure in the big-file-write class, and the one that actually bit a user (live gemma-4-12b, 2026-06-20, captured in `chat-history.json`). Turn 1 (`{"path","content"}`) wrote fine; then on "append to it" the model emitted `{"append":"true,\n<ENTIRE file body>","path":"jfk.txt"}` — **the whole body jammed into the `append` value, with NO `content` key** — five times in a row, each rejected, until the stuck-detector gave up. Two distinct bugs:
- **The `append:"true"` string param invites conflation.** A mid-size model merges the flag and the body into one string (`"true,\n…text…"`) instead of closing `append` and opening `content`. Fix (`AgentEngine.normalizeWriteFileArgs`, called in `executeBuiltinTool` before validation + handler): when `content` is absent/blank AND `append` is a boolean keyword FOLLOWED by a separator and more text, peel the leading `true`/`false` (flag) and move the remainder to `content`. A clean `"true"` is left alone so a genuinely missing body still errors (don't fabricate an empty file). Also tightened the schema: `append` is described as "Flag ONLY … NEVER put file text here — the body ALWAYS goes in `content`".
- **The missing-content error LIED about the cause.** It unconditionally returned "content was truncated — your output got cut off" whenever `content` was missing + `path` present — even though only ~500 tokens were generated (the user's exact complaint: "why did it get interrupted if we did not reach max tokens?"). Real `max_tokens` truncation is intercepted UPSTREAM in `ChatTurnEngine.runAgentLoop` (the `maxTokensHit && !receivedToolCalls.isEmpty` branch) BEFORE execution, so reaching `executeBuiltinTool` with missing content means the body was misplaced/omitted, never cut off. Fix: the error now says the body must go in `content` (and `append` is only a flag), with no false "cut off" claim. **Rule: never let a tool-executor error infer truncation — only the loop, which knows `maxTokensHit`, may say "cut off".**
- Guarded by the `normalizeWriteFileArgs` unit tests + the `testNormalizedAppendActuallyWritesContent` handler test (`WriteFileHandlerTests`) and the end-to-end `AgentWriteFileRecoveryTests` (drives the real `AgentEngine.executeToolCall` with the EXACT captured append-jam bytes → file written; and asserts the missing-content error names `content` and never says "cut off"/"truncated"). **Testing lesson:** the prior fix in this class was validated against a *different* (Hermes) format on a *steered* model, not gemma-4-12b's real JSON output — always reproduce from the actual failing model's captured bytes (`chat-history.json` / `last-agent-request.json`), not a synthesized analogue.

A LIVE follow-up run (gemma-4-12b via TestServer, 2026-06-20) surfaced two MORE variants of the same `append`-vs-content confusion, fixed together:
- **Dirty flag value silently OVERWROTE.** The model emitted `{"append":"true,","content":"<body>"}` — a trailing comma on the flag. `WriteFileHandler`'s exact `== "true"` match failed, so it overwrote (the file SHRANK) instead of appending. Fix: `WriteFileHandler.appendFlagIsTrue` parses the flag tolerantly (`"true"`, `"true,"`, `"True"`, boolean `true`→`"1"`, `"yes"` → append; `"truely"`/`"false"`/`""` → not). Never gate append on an exact string match.
- **Structural weld → empty args (the root kill).** The model emitted `{"append":"true\",content:","<body>"}` — it over-escapes the flag's closing quote and welds the `content:` key-boundary INTO the flag's string value, orphaning the real body. This is JSON so broken it parses to **empty args at the server before the app sees it** (`PARSED keys=`), so no app-side recovery can touch it, and `looseRepairToolCallJson` can't reconstruct the *missing* `content` key (it preserves structure). The clean fix is at the SOURCE, not another JSON-repair: the schema declared `append` as **`"type":"string"`**, which literally tells the model to emit `"true"` — the exact string it welds into. Changing it to **`"type":"boolean"`** steers the model to emit a bareword `true` (which can't absorb a content string). Verified live: 12 consecutive gemma-4-12b appends across 3 stress runs, **zero welds, zero empty-keys parses** (was failing ~1-in-4 before). `appendFlagIsTrue` still accepts both forms so no model regresses. **Rule: a boolean flag adjacent to a large free-text value is a JSON-mangle magnet for weak models — type it as a boolean, never a string, so there's no quoted value to weld the next key into.**

### editFile endLine-without-startLine loop + the empty `Example: {}` steer (error-steering class)
The FIFTH failure in the weak-model tool-call class (live 2026-07-02, alex.html session, captured in `chat-history.json` + `tool-calls.log`): after a legitimate "Pattern not found" (the model hallucinated a `class` attribute into its `find` text), it switched to line-based editing but emitted `{path, endLine, replace}` with NO `startLine` — **12 near-identical retries across three user turns**, each answered by the same error, until the stuck-detector killed each turn (a whole-file `writeFile` finally rescued it). The `tool-calls.log` EMIT lines prove the server parse was innocent: the raw `arguments` bytes already lacked `startLine` in every key order the model tried — so this is NOT a `format_corpus_test.zig` case; the fix is app-side. Two compounding Swift bugs:
- **`AgentEngine.toolExample` extracted the example by the literal marker `"Example: "`.** editFile's description says `"Example line-based: {…}"`, so the match failed and every editFile error shipped a literal **`Example: {}`** — a null steer. Contrast writeFile, whose real example recovered the model in ONE retry. Fix: slice from the first `{` after any `"Example"` marker; class guard = `testEveryToolDescriptionExampleIsExtractable` (every tool with an example in its description must yield a real one; no-arg tools like listProcesses legitimately example as `{}`).
- **The missing-param error didn't name the gap relative to the keys sent.** "Either 'find' or 'startLine' is required" restates the modes; a fixated model needs "you sent endLine=65 but no startLine — resend the same call with startLine added". Fix in `EditFileHandler`: the endLine-present branch throws exactly that, with a concrete example built from the model's own path/endLine values.
- Bonus hardening (the `appendFlagIsTrue` lesson applied to line numbers): `EditFileHandler.parseLineNumber` reads `"45,"`/`" 45 "` as 45 so a dirty `startLine` can't demote a line-based edit into the find branch, and a dirty `endLine` can't silently collapse the range to `startLine..startLine`.
- Guarded by `EditFileRecoveryTests` (captured-bytes end-to-end via the real `AgentEngine.executeToolCall`, the extraction class guard, both dirty-number cases). **Rule: an error string sent back to a model must never contain an empty/placeholder example, and must name the missing key RELATIVE to the keys actually sent — a restated schema is not a steer.**

### ServerOptions defaults must mirror the Zig server defaults
`ServerOptions.toCLIArgs` (app/Sources/MLXServe/Models/ServerOptions.swift) emits most launch flags ONLY when the value differs from an *assumed* server default (`if maxConcurrent > 1`, `if llamaCacheEntries != 4`, etc.) to keep the CLI tail readable. That assumption is a silent contract with the Zig defaults: the Swift `ServerOptions` field default AND the emit-guard's constant must BOTH equal the real server default, which lives in `src/main.zig` (`var x = default` for CLI-parsed flags) or `src/server.zig` (`pub var` globals like `prefix_cache_capacity`/`llama_cache_entries`/`tokenize_cache_entries`/`max_concurrent`). If they drift, the app omits the flag and the server silently runs ITS default — the UI shows one value, the process runs another. Live surprises: `prefix-cache-entries` (Swift assumed 1, server 32 → filled 16 GB Macs) and `llama-cache-entries` (Swift assumed 1, server 4). Rules: (1) when you change a server default in main.zig/server.zig, update the matching ServerOptions field default AND the `toCLIArgs` guard constant in the same change; (2) for memory-critical caps where a wrong silent default causes OOM, prefer ALWAYS-emitting the flag (prefix-cache-entries does this — it also RAM-clamps, so it must always send the computed value) so the server default is irrelevant. The guard test `testDefaultLaunchOmitsAllMatchDefaultFlags` (ServerOptionsTests) fails if a default ServerOptions() launch emits any match-default flag — i.e. if a Swift default or guard drifts — and names each Zig source of truth in a comment.

### An LSUIElement app must go `.regular` BEFORE it presents UI, never after (semi-focused-window class)
Symptom signature: you open a window (or a folder picker) and it comes up **half-focused** — no blinking caret, title bar unemphasized, clicks land nowhere — and then "if I start typing it gets activated" / "if I play around with it, it eventually gets focused". Fixed 2026-07-13.
- ROOT: MLX Core is `LSUIElement`, so it launches `.accessory`, and **an accessory app that isn't active has NO key window**. `ActivationPolicyManager` flips to `.regular` reactively, on `NSWindow.didBecomeKeyNotification` — which cannot fire while the app is `.accessory` and inactive. Chicken-and-egg: the window is ordered on screen but never becomes key, so the policy never flips, so the app never properly activates… until the user's click/keystroke activates it, which fires the notification, which flips the policy. The old `openAndFocus` did `openWindow(id:)` → `NSApp.activate()` with the policy still `.accessory` — the inverted order (and ChatView's `onAppear` comment even asserted the broken assumption: "the .regular flip is handled app-wide by ActivationPolicyManager when this window becomes key").
- FIX: `AppActivation` (Services/AppActivation.swift) is the ONE chokepoint. `focus()` sets `.regular` first, THEN activates — synchronously, because a modal panel spins its own run loop (`NSModalPanelRunLoopMode`) and a main-queue block deferred past it lands too late. `AppActivation.openWindow(id:using:)` focuses → opens → re-asserts + raises by title next runloop turn; `AppActivation.runModal(panel)` / `beginPanel` focus first, then `reapply()` after (a cancelled picker with no windows open must fall back to `.accessory`, or the eager flip leaves a permanent Dock icon — verified live).
- The `NSOpenPanel` half is the same bug: `ActivationPolicyManager.countsAsUserWindow` deliberately EXCLUDES panels (the ⌃Space quick launcher is one), so it will never flip the policy for a picker — the presenter must. There were 15 raw `panel.runModal()`/`panel.begin` sites, none of which activated first; `launchClaudeCodeWithPicker` (tray → code editor → folder picker) was the one users hit.
- **Rule: any new window or panel is presented through `AppActivation` — never a bare `openWindow(id:)` or `panel.runModal()`.** Guards: `AppActivationTests` — the fake-`NSApp` ordering tests (`[.setPolicy(.regular), .activate]`, order is the bug) plus two SOURCE-AUDIT class guards that fail on any new raw `openWindow(id:` / `panel.runModal()` site. The ⌃Space launcher's own `.nonactivatingPanel` is built directly and deliberately stays outside this path.

### Two binaries in the app bundle
`.app` contains `MLXCore` (Swift UI) AND `mlx-serve` (Zig server). Both must be updated together. Forgetting one after rebuild is a common "still doesn't work" cause.

### Agent Sandbox = Virtualization.framework Linux guest (no libcontain)
The Settings "Agent Sandbox" routes the agent's foreground shell commands into a Linux VM built directly on Apple's Virtualization.framework (`VzGuest`) — the former `../contain` (libcontain/CContain) dependency is gone. Load-bearing facts:
- **Entitlement**: the PROCESS needs `com.apple.security.virtualization` (in `MLXCore.entitlements`; MAS-compatible — UTM ships with it). `app/build.sh` applies it in BOTH signing branches — ad-hoc dev builds included — because VZ refuses to start otherwise. Plain `swift build` binaries and the `xctest` host can't boot VMs; the live-boot guard is `SANDBOX_SMOKE=1 <MLXCore binary>` (phases: direct guest → ShellHandler routing → Sandbox Terminal; `SANDBOX_SMOKE_REAL_PROVISION=1` also exercises the real kernel fetch + OCI pull). Hermetic counterparts: VzGuestTests / OCIClientTests / ShellSentinelTests / AgentSandboxTests.
- **Boot shape**: kernel = contain's prebuilt minimal 6.6 arm64 Image (fetched once from the `ddalcu/contain` `kernels-v2` release, gunzip'd by `OCIClient.gunzip` — Compression framework, no Process). Rootfs = the base OCI image unpacked to `~/.mlx-serve/sandbox/images/<ref>` and mounted as the guest ROOT over virtiofs (`root=rootfs rootfstype=virtiofs`, cmdline in `VzGuest.kernelCommandLine`) — NO initramfs, so the image is demand-paged and guest RAM is 1 GiB workload headroom (was 6 GiB when the whole rootfs rode in RAM). PID 1 is a host-written `/.vz-init` script (`VzGuest.buildInitScript`) that mounts /proc /sys /dev + the `workspace` virtiofs share, exports the image's Env (from the `.vz-image-config.json` sidecar `OCIClient.pull` writes), and hands a persistent dash the shell channel.
- **Transport**: the kernel has NO vsock; the shell rides the SECOND virtio-console port (`serialPorts[1]` → guest `/dev/hvc1`, index order is hvc order), driven by the echo-proof `ShellSentinel` framing. hvc0 stays kernel-printk-only so boot failures are diagnosable (`consoleSnapshot`) and the shell stream is never polluted. hvc1 is a real tty → `stty -echo` works and 0x03 delivers SIGINT on exec timeout.
- **Kernel cache self-heals**: a cached `~/.mlx-serve/sandbox/kernel` that predates the virtiofs config (the pre-VZ fetch) cannot mount the root and would panic; `AgentSandbox.kernelHasVirtiofsSupport` (literal "virtiofs" bytes) detects and re-fetches. The legacy `~/.mlx-serve/contain` cache dir is migrated by rename on first touch; rootfs caches without the config sidecar re-pull once.
- **hvc1 is a real tty → tools emit ANSI + progress animations**: npm's braille spinner, `\e[1G\e[0K` line rewrites, curl's `\e[1m…\e[0m` bold headers all appear because CLI tools detect an interactive terminal (the host path uses a pipe, so they stay quiet). `TerminalOutput.sanitize` (VzGuest.swift, applied to `exec` output in BOTH `AgentSandbox.runForeground`/`runUserCommand`) strips CSI/OSC/charset escapes and resolves carriage-return + cursor-to-column-1 (`…G`) overwrites so only each line's final content survives. GOTCHA (same tty-ONLCR class as hvc2): the tty maps `\n`→`\r\n`, so you MUST normalize `\r\n`→`\n` BEFORE the bare-`\r` progress collapse — otherwise the trailing `\r` reads as a line-overwrite and wipes every line (live-caught: `echo a; uname; pwd` came back as one run-on line; pinned by `testSanitizeNormalizesCrlfWithoutEatingLines`). Sentinel scanning stays on RAW bytes; sanitize runs only on the returned/recorded/displayed text.
- **Guest background commands are tracked + killable like host ones**: the `.sandboxBackground` route backgrounds inside the guest (`(cmd) </dev/null >>/tmp/mlx-bg-<ms>.log 2>&1 & echo __CTN_BGPID=$!`), parses the guest pid, and registers a `ManagedProcess.Kind.sandbox(logPath:)` in `ProcessRegistry` (no host `Process`) so the chat card shows the same running badge + kill X; kill routes a guest `kill -TERM`→grace→`-KILL` (`AgentSandbox.killGuestProcess`, into the already-booted guest — never a new boot), `readProcessOutput` tails the guest log. LIMITATION: no host `terminationHandler` for a guest process, so a self-terminating bg process shows "running" until killed or polled (a `kill -0` liveness check in readProcessOutput would close it).
- **Guest networking + live port map (default ON, Settings toggle)**: the kernel itself DHCPs (`ip=dhcp` on the cmdline — the prebuilt kernel has CONFIG_IP_PNP), so networking works with ANY image, no userspace DHCP client needed; DNS is copied from `/proc/net/pnp` to resolv.conf. With the toggle OFF the VM gets NO network device at all. A guest monitor loop streams `/proc/meminfo` + the guest IP + `/proc/net/tcp(6)` once a second over a THIRD console port (hvc2, `=EOS=`-framed); the host parses it (`GuestNetParser`) into (a) the tray's RAM readout (`AgentSandbox.guestMemoryText`, 16 MB-quantized + published only on change — tray-churn class) and (b) `SandboxPortForwarder`, which mirrors every non-loopback LISTEN port onto HOST LOOPBACK `<same port>` live (an Express app on guest 8080 is `localhost:8080`; ports in use on the host are skipped with a log). The mirror binds BOTH loopback families — `127.0.0.1` AND `::1`, one NWListener each, the pair dropped together if either fails — because `localhost` resolves to ::1 FIRST in modern clients and a v4-only listener reads as "server is down" to anything that doesn't fall back (live 2026-07-02: guest python http.server, map line shown, browser refused). LAN exposure stays deliberately off. Companion model-side rule (two layers): the served-app URL FORM is environment-specific, so it rides `AgentPrompt.executionEnvironmentSection` — host: `http://<local-ip>:<port>` (LAN-reachable 0.0.0.0 bind); sandbox: exactly `http://localhost:<port>` with an explicit NEVER-a-LAN/guest-IP countermand — and the base user-editable prompt (`defaultPromptFile` "Serving apps" / `~/.mlx-serve/system-prompt.md`) must NOT hardcode a URL form. Live 2026-07-02: the base prompt's `http://<local-ip>:<port>` directive (IP from the grounding line) beat the env section's localhost hint and the agent handed the user the Mac's LAN IP, which the loopback-only map can never serve — when two prompt layers conflict, the model follows the more specific formatting directive, so per-environment facts must never be stated unconditionally in the shared base prompt. Belt-and-braces: the sandbox background-start TOOL RESULT (`ShellMessages.sandboxBackgroundStarted`) repeats the "reachable at http://localhost:<port>, never a LAN/guest IP" steer in the bytes the model replies from. Pinned by `testServedUrlFormRidesTheEnvironmentSectionNotTheBasePrompt` + `testSandboxBackgroundStartMessageSteersToLocalhostUrl`. GOTCHA: hvc2 is a tty, so lines arrive `\r\n` — and Swift treats `"\r\n"` as ONE grapheme cluster, so `split(separator: "\n")` does NOT split; parse with `split(whereSeparator: \.isNewline)` (live-smoke-caught; pinned by `testParseSnapshotSurvivesTtyCrlfLineEndings`). Hermetic forwarder tests can't bind the fake service and the listener to the same port (NWListener probes BOTH address families) — they use the `targetPortOverride` seam; the same-port identity is proven by SandboxSmoke phase 4 (outbound HTTPS + guest:8123 → localhost:8123 + [::1]:8123 + RAM readout).

### Multi-input AVAssetWriter deadlocks if you push all of one track before the other (mux toy-green/real-hang class)
A two-input `AVAssetWriter` (video + audio) keeps its tracks interleaved by applying BACKPRESSURE: once one input leads the other on the timeline by the muxer's window, that input's `isReadyForMoreMediaData` goes false and STAYS false until the lagging track is fed. `VideoGenService.writeMP4` busy-waits per frame (`while !input.isReadyForMoreMediaData { usleep(500) }`) and originally appended the WHOLE audio track only AFTER the video loop — so once video led the still-empty audio track, the video input wedged, the audio that would un-wedge it was never appended, and the loop spun forever. Live symptom (2026-06-20, LTX-2 with audio): the UI sat at "Encoding mp4…" indefinitely; the server had already produced frames + PCM and unloaded (so it's NOT a server bug — the hang is entirely client-side in the mux). Video-only never hit it (a single input has no sibling to wait on). **Toy-scale stays green**: a 3-frame 16×16 clip fits under the backpressure window and completes, so a small unit test passes while a real ~97-frame 256×256+ clip deadlocks — the guard MUST run at realistic frame count/timeline. Fix: append the full audio track (one CMSampleBuffer) + `markAsFinished()` BEFORE the video loop; a finished input is no longer an active sibling the muxer waits on, so the video input never wedges. Guarded by `testWriteMP4WithAudioDoesNotDeadlockAtRealisticScale` (MediaGenServiceTests — 97×256×256 + 4s stereo on a detached thread with a 30s `wait()`; red = timeout on the old ordering, verified). Rule: never feed a multi-input AVAssetWriter one track to completion before starting the other; finish the single-shot track up front, or drive both inputs via `requestMediaDataWhenReady`.

### SwiftUI ToolbarItems whose content changes size break NSToolbar (» eviction class)
The AppKit bridge does NOT re-measure a `ToolbarItem` whose SwiftUI content changes size after insertion. Live bite (2026-07-02, chat window): enabling Agent mode grew the toolbar strip (sandbox shield + workspace chip appear), NSToolbar kept stale measurements, decided "doesn't fit" with hundreds of points of free space, and shoved controls into a mangled `»` overflow menu (custom capsules render as garbage rows there) until a window RESIZE forced a fresh layout pass. Three structures all failed the same way: separate sibling items (evicts the neighbor), `ViewThatFits` inside an item (measured once at ideal size — never re-proposed narrower, so it can't adapt), and one merged item (evicted wholesale when its content grew). Rule: **any toolbar-like strip whose members appear/disappear or change density at runtime must be rendered as a plain view row (`.safeAreaInset(edge: .top)`), not NSToolbar items** — outside NSToolbar there is no overflow mechanism to misfire. The chat window's `chatHeaderBar` (ChatView.swift) is this pattern; its pill density (full vs icon-only) is decided from the MEASURED pane width via `ChatMetrics.useCompactToggles(forDetailWidth:)` (pinned by ChatMetricsTests), never by system fitting negotiation. Static, never-changing items (a lone gear icon) are still fine as real ToolbarItems.

### Per-tab chat state must be keyed by session id, never a shared flag (reused-ChatDetailView class)
The chat window's detail pane is `ChatDetailView(sessionId:)` with NO `.id(sessionId)`, so SwiftUI **reuses ONE instance** across tab switches — it swaps the `sessionId` `let`, it does NOT rebuild the view or reset its `@State`. Any per-conversation UI state stored as a single shared `@State` therefore (a) leaks the active tab's value into every other tab, and (b) only "re-arms" if an explicit `.onChange(of: sessionId)` wipes it — which then forgets the value the moment you switch away and back. Two live bugs of this class (2026-06-20): "Allow all tools this session" was a `Bool` reset on every switch (forgotten across tabs), and the Stop button read the engine's single `isGenerating` (every tab showed Stop while any chat generated). Rule: store per-conversation state in a structure keyed by session id — a `Set<UUID>`/dict in `@State` (e.g. `SessionToolAllowList`) for view-local memory, persisted fields on `ChatSession` for state that should survive relaunch (the three toolbar toggles: `mode`=Agent, `enableThinking`=Think, `useMCP`=MCP), or query the app-level engine per session (`ChatTurnEngine.composerState(for:)`, which gates on `activeTurnSessionId == sessionId`). For a persisted toggle, load it from the session BOTH in `.onAppear` AND in `.onChange(of: sessionId)` (the reused view's `.onAppear` fires only once) via one `syncTogglesFromSession()`, and write back via per-toggle `.onChange`. Do NOT "fix" forgetting by adding a `.onChange(of: sessionId)` that RESETS a flag — reset-on-switch is the anti-pattern that caused the allow-all forgetting (the allow-list is deliberately NOT reset there; it re-arms only on its own Agent-toggle-off). MCP moved from the app-global `appState.mcpMode` to per-session here; the global remains as the new-chat seed + Voice's source. Guarded by `PerSessionUIStateTests` (allow-all per tab, composer state, ChatSession Think/MCP Codable round-trip + legacy-decode-to-off).

### Ghost turn on the shared ChatTurnEngine (deleting a chat ≠ stopping its turn)
`appState.chatEngine` runs ONE turn at a time app-wide (chat window + Quick Launcher share it; Tasks and Telegram each have their OWN engine instance). A turn whose session gets REMOVED keeps running as a ghost: every append/update no-ops against the gone session, the empty-response check then reads `""` from the missing session and PAD-RETRIES — each retry a full multi-minute generation against a system-only prompt — while `isGenerating` stays true. Symptom signature: every chat surface says "The model is answering another chat" (or send silently no-ops via `runTurn`'s guard), NO visible chat shows a streaming bubble or Stop button, `~/.mlx-serve/last-agent-request.json` keeps updating with no UI activity, and **stopping/restarting the SERVER doesn't clear it** — the turn is app-side and reconnects to the new server process. Live 2026-07-03: user deleted an errored agent chat; the ghost held the engine 10+ minutes across a server restart, kept the 27B model decoding, and only an app relaunch cleared it. Fix: `AppState.deleteSession` (the single runtime session-removal chokepoint) calls `chatEngine.stopIfOrphaned()` — pure decision `ChatTurnEngine.turnOrphaned` (an idle engine is never orphaned; `activeTurnSessionId` deliberately keeps its last value) — and `runAgentLoop` re-checks `session(sessionId) != nil` at the top of every iteration as defense in depth, which also short-circuits the pad-retry burn (`continue` re-enters through the guard). Rules: (1) any new session-removal site must stop the owning turn; (2) when "everything says busy but nothing is generating", check `last-agent-request.json` mtime BEFORE suspecting the server. Guarded by `OrphanedTurnTests`.

### `TaskRun.summary` is a 280-char UI-row cap, never the full result
`TaskScheduler.lastAssistantText` caps the final answer at `prefix(280)` for the Tasks timeline ROW; that capped string lands in `run.summary`. Relaying `run.summary` to any full-text surface (Telegram, export) silently truncates the result — the live bug was Telegram task delivery sending the 280-char version. Rule: for the whole answer use `TaskScheduler.fullLastAssistantText(transcript(taskId:runId:))` (no cap); `run.summary`/desktop notifications keep the short form. Guarded by `testFullLastAssistantTextIsNotTruncatedButTimelineRowIs` + `testFullTaskResultSplitsIntoMultipleTelegramMessages`. (Telegram's own 4096-char per-message cap is handled separately by `TelegramAPI.splitForTelegram` at every send site — `sendReply` and `deliverTaskResult`.)

### Stale task runs persist as `.running` after a crash/quit
A run is created `.running` and only flips terminal when `driveRun` finishes; an app quit mid-run leaves it `.running` on disk forever (forever-spinner in the Tasks list). `TaskScheduler.start()` heals these via `reconcileStaleRuns` (→ `.failed`) since `activeRun` is always nil at launch. `.needsApproval` is durable/resumable and explicitly preserved. User-facing controls in `TasksView`: per-run Stop (`cancelRun` — finalizes the LIVE run as `.cancelled` via `cancelledRunIds` so the engine's in-flight completion can't clobber it) / Delete (`deleteRun`, refuses the live run), and per-task "Clear finished" (`clearFinishedRuns` / pure `runsAfterClearingFinished`).

### Launch-eager audio objects = mic permission prompt at APP LAUNCH (TCC class)
`VoiceModeController` is constructed at every launch (the menu-bar label observes `appState.voice`), and anything audio it builds eagerly brings up CoreAudio in-process — whose voice-isolation/"chat flavor" evaluation consults `kTCCServiceMicrophone`. Live 2026-07-05: tccd logged `AUTHREQ_PROMPTING service=kTCCServiceMicrophone` 1.9 s after launch, traced to the eager `AVSpeechSynthesizer` + the chime/cue `AVAudioEngine` graphs built in their inits. Amplifier: `SKIP_NOTARIZE` ad-hoc builds re-sign per build, tccd logs `Failed to match existing code requirement` and RE-prompts on every rebuild's first access — so a launch-time access prompts on every dev build. Rules: (1) any audio object owned by a launch-eager singleton builds its AVSpeechSynthesizer/AVAudioEngine graph lazily on FIRST USE (`SystemSpeechSynthesizer.synthStorage`, `SystemWakeChime`/`SystemLoadingCue.ensureEngine`) — a bare `AVAudioEngine()`/`AVSpeechSynthesisVoice` catalog read is safe, graph/unit instantiation is not; (2) `stop()`-style idle paths must not force creation (`synthStorage?`, `guard running`). Pinned by the lazy-init tests in `VoiceActivityTests`. Diagnosis: `/usr/bin/log show --predicate 'process == "tccd"'` around launch (NOTE: zsh's `log` builtin shadows `/usr/bin/log` — bare `log show` silently returns nothing).

### Fixed-threshold VAD wedges on ambient noise (voice hears you, never answers)
Voice mode's turn endpointing is a hand-rolled energy VAD: a turn finalizes after 1.1 s of mic RMS below threshold. With a FIXED absolute threshold (0.015), any continuous ambient above it (GPU-fan MacBook mic, AC, music) refreshes `lastVoiceAt` forever → the utterance NEVER finalizes → nothing is ever submitted to the LLM, while SFSpeechRecognizer happily keeps transcribing. Symptom signature: run-on partials gluing multiple sentences ("Hello how are you can you hear me"), the tray stuck on "Listening…" with a live partial, ONE long recognition task in the speech-daemon log (a healthy session creates a new task per finalized utterance), and no user message ever landing in `chat-history.json`. Fix (src `VoiceActivity.swift` + `BaseSpeechRecognizer`): (1) the threshold is RELATIVE — `max(0.015, ambient_floor × 2)` where the floor is a minimum-statistics tracker (`AmbientFloor`: min RMS over a sliding ~8 s window; between-word dips keep it at ambient during speech, and it re-converges within one window when noise starts/stops mid-session); (2) a transcript-stall backstop finalizes when recognition has words but produced no NEW word for 2 s, regardless of RMS (never fires with an empty transcript — noise that transcribes to nothing has nothing to submit). Rule: never gate audio endpointing on an absolute level; track the floor and threshold relative to it, and keep a recognition-derived backstop for whatever the energy VAD misreads. Pinned by `VoiceActivityTests`.

### WebSearch + Browse
`webSearch` navigates DuckDuckGo HTML, extracts results via JS. `browse.readText` navigates first then extracts — ensures correct page (not previous).

### WKWebView main thread
`BrowserManager` is `@MainActor`. All WKWebView ops (navigate, readText, evaluateJS) on main thread. Created eagerly at app launch so tools work without Browser window open.

### Swift JSONSerialization quirks
- `[String: Any]` non-deterministic key order
- `""` stays `""` in JSON (not `null`); server treats both as empty
- `Double` like `0.7` → `0.69999999999999996` — fine
- `arguments` in tool_calls must be a JSON String (e.g., `"{\"command\":\"ls\"}"`), not nested dict; server checks `if (v == .string)`

### A guest-side payload staged in only ONE packaging path breaks the feature for every OTHER build variant (issue #89)
Live 2026-07-18 (GitHub issue #89): sandboxed MCP failed for every Developer ID user with "MCP servers need the vsock guest transport (kernel kernels-v3 + vz-agent); the guest booted the legacy console shell" — while working fine in dev. The vsock transport needs TWO halves: a kernels-v3 kernel (downloaded + tag-version-cached by the provisioner — this half was fine) and the `vz-agent` static aarch64 ELF injected into the rootfs. The agent was BUILT on every `app/build.sh` run but COPIED into `Contents/Resources/guest/` only in the `MAS=1` branch, and the GitHub release workflow (which assembles its own bundle, not via build.sh) never even built it. `agentBinaryPath()`'s fallback chain (VZ_AGENT_PATH → bundle → `zig-out/guest/` dev path) meant dev machines always found the zig-out copy — so the breakage was invisible locally and total in the field. Three fixes:
- Both packagers stage `Resources/guest/vz-agent` unconditionally AND fail the build if it's absent from the finished bundle (the MAS bundle additionally carries kernel+rootfs; guideline 2.5.2 forbids downloading those, while Developer ID keeps downloading them to stay small — the agent is our own ~200 KB ELF, so it always rides the bundle).
- `AgentSandbox.transportFallbackReason` names the MISSING half in the MCP error ("vz-agent missing from the app bundle (reinstall)" vs "kernel predates kernels-v3 (delete ~/.mlx-serve/sandbox)") — the collective "kernel + vz-agent" wording sent the reporter debugging the wrong half. Same distinct-failure-modes discipline as the LAN proxy 404s.
- Verified end-to-end: `SANDBOX_SMOKE=1` with the freshly built bundle boots `transport=vsock` off the BUNDLED agent (production kernel + rootfs), all phases green.
Rule: any file a feature needs at runtime that is produced by the build must be staged by EVERY packager (build.sh non-MAS, build.sh MAS, release.yml) with an existence check in each — a dev-path fallback in the lookup chain guarantees the gap ships silently. Guards: the SandboxTransportTests fallback-reason cases; the existence checks in build.sh + release.yml.

### Agent CLIs inside the sandbox ride ssh — and the traps that shaped it (issue #89 follow-up)
2026-07-19, design + v1 implementation. The issue #89 reporter runs coding agents in docker "to keep them isolated from my Mac"; the ask is pi/hermes running INSIDE the Agent Sandbox with a real TUI. The constraint that shapes everything: vsock connections can only be opened by the VM-owning process, so an external terminal can never reach the guest directly — and neither the ShellSentinel console nor the exec-per-command vsock channel can host a full-screen TUI. One transport serves both the embedded terminal and the user's own Terminal.app: dropbear BAKED into the OCI image, `/usr/bin/ssh` spawned on a PTY (SwiftTerm's `LocalProcessTerminalView`, wrapped by the ONLY file that imports SwiftTerm — `EmbeddedTerminalView`, the seam libghostty can replace later), mirrored by a DEDICATED `SandboxPortForwarder` (`targetPortOverride { _ in 22 }`, `localhost:<first free from 2222>` → guest `:22`). vz-agent untouched; no PTY-over-vsock.
Traps hit while building it:
- **/dev/pts**: the guest init mounts devtmpfs, which does NOT auto-mount devpts — without `mount -t devpts devpts /dev/pts` every ssh session dies with "PTY allocation request failed", while a tty-less `ssh <cmd>` works fine, which makes it look like a TUI/SwiftTerm bug instead of a mount gap. The dropbear arm of `buildInitScript` owns the mount; pinned by the VzGuestTests ssh-arm characterization.
- **port-22 double-mirror**: dropbear binds 0.0.0.0:22, so the GENERAL port forwarder would also mirror it to localhost:22 — where a host sshd usually owns the bind, so the mapping fails or thrashes on every snapshot. The general forwarder now filters 22; the dedicated ssh mirror is the one stable address.
- **the host's address inside the guest**: what the agent must dial is the NAT gateway, knowable only in-guest and per-boot. Config files are materialized HOST-side with the literal `__MLX_HOST__` and sed'd by `/.vz-bootstrap-<agent>` after `ip route` resolves the gateway. And because guest→host traffic arrives NON-loopback at the server, the REAL `--api-key` must ride the generated configs — the host launcher's "mlx-serve" placeholder only works for loopback-trusted clients.
- **known-hosts pollution — and the re-pull MITM banner**: guest host keys must never touch the user's `~/.ssh`; the entire key world is app-owned under `~/.mlx-serve/sandbox/ssh/` (ed25519 keypair, own known_hosts, `StrictHostKeyChecking=accept-new`). The subtle half, caught testing against a locally built image: dropbear's `-R` host key lives IN the rootfs, so every re-pull (exactly what the stale-image flow triggers) presents a NEW key at the SAME `[localhost]:<port>` — and accept-new only auto-accepts UNKNOWN hosts; a CHANGED key hard-fails ssh with the REMOTE HOST IDENTIFICATION HAS CHANGED banner, stranding every session right after the re-pull that was supposed to fix things. Fix: `SandboxSSH.resetKnownHosts()` at each guest boot — TOFU-per-boot; the trust anchor is our own loopback bind to our own VM, not key continuity.
Session semantics: a live session refcounts a pin (`pinCliSession`); `ensureBooted`'s remount path THROWS `remountBlockMessage` ("a sandboxed pi session is using the VM — close it before switching the workspace") instead of rebooting the VM under a live TUI. A cached rootfs that predates dropbear gets the distinct `staleImageMessage` + a one-click re-pull (`repullBaseImage`). App quit kills the VM and every session in it — accepted v1 semantics, no confirm dialog. MAS: spawning `/usr/bin/ssh` under the App Sandbox is a release-validation item; the disposition lives in HostEscapeAuditTests's `known` entry (if review/seatbelt blocks it → Developer-ID-only with an explicit message, never silently broken).

### Sandbox terminal round 2: unmount kills the session; agent docs lie about their own config (issue #89 follow-up)
2026-07-19, first live session testing. Three bites in one afternoon:
- **A SwiftUI unmount is a SIGTERM**: the Terminal/Activity switch was a `switch tab` in `body`, so flipping to Activity REMOVED the `EmbeddedTerminalView` from the hierarchy → `dismantleNSView` → `terminate()` → the live pi session died and the tab came back to a fresh prompt. Rule (the LIFETIME RULE in SandboxTerminalView): every live terminal stays MOUNTED in a ZStack and hides via `.opacity` + `.allowsHitTesting`; the same applies across session tabs. Corollary: nothing in an always-mounted hidden pane may grab focus in `.onAppear` (the Activity input now takes focus in `.onChange(of: pane)`), and no hidden pane may hold a bare `.keyboardShortcut` that would swallow keys from the visible terminal.
- **Multi-session costs nothing at the transport layer**: N concurrent pi/hermes/shell sessions are just N ssh connections into the SAME dropbear behind the SAME mirror port — the whole "tabs" feature is pure UI state (`SandboxSessionTabs`: stable never-renumbered display names "pi"/"pi 2", neighbor-selection on close) plus per-tab runtimes. The pin list was a refcount from day one, so pinning needed zero changes.
- **Configure an agent from its SOURCE, not its docs**: hermes's docs say `model: {provider: custom, model: …}`; the shipped CLI warned "Unknown provider 'custom'… falling back to auto detection", showed "Active provider: none", and ran the first-run wizard anyway. The real contract (read from `hermes_cli/model_setup_flows.py` `_model_flow_custom` + `main.py `_save_custom_provider`/`_has_any_provider_configured`): model name lives under `model.default`; `api_mode: chat_completions` skips URL heuristics; `context_length` rides a `custom_providers` entry; and the wizard's gate is satisfied by `OPENAI_BASE_URL` in `~/.hermes/.env` ALONE ("local models often don't require an API key" — their comment). We now write exactly what the wizard's own save path writes, so the first `hermes` run starts configured. Rule: when pre-seeding a third-party CLI's config, mirror its setup flow's SAVE code path verbatim and pin the shape in a test — docs drift, save paths don't.

### A parent onTapGesture swallows child Buttons on macOS (sandbox chip ✕)
2026-07-19, same-day live report on the session pills: the chip's ✕ "did nothing" — no alert, no close, no error. The chip was an HStack containing a plain-style ✕ Button, with `.contentShape(Capsule())` + `.onTapGesture { select }` on the PARENT for tab selection. On macOS that parent click gesture claims the whole capsule INCLUDING the child button's pixels; the Button never fires and there is no diagnostic anywhere. (The short-lived bottom-bar "End Session" button that also "did nothing" was removed the same day as redundant with the ✕.) Fix: composite interactive rows are built from REAL side-by-side Buttons — select = one Button (dot + name), close = another — each with its own `.contentShape(Rectangle())`, and NO gesture on the container. Rule in CLAUDE.md; the pattern to grep for in review is `.onTapGesture` on any container that has a `Button` descendant.

### A legacy .alert(item:) on an ancestor shadows a descendant's alert (sandbox ✕-confirm, round 2)
2026-07-19, the very next bite on the same chip: with the ✕ now a real Button that demonstrably fired, the confirm-before-close alert STILL never appeared — `requestCloseTab` set its item binding and nothing happened, no warning anywhere. The close-confirm used a SECOND deprecated `.alert(item: $confirmClose)` attached to the session strip, deliberately moved off the root VStack because "two .alert modifiers on the same node shadow each other." That understanding was half the class: with the legacy `Alert` machinery, a modifier on an ANCESTOR also swallows a descendant's presentation — the root VStack already carried `.alert(item: $alert)` for window-level errors, so the strip's alert was dead on arrival. (Sibling-branch alerts, as in ModelBrowserView, are fine — it's the ancestor chain that kills.) Fix: ONE presentation path per window — `SandboxAlert.Action` grew a `confirmClose(UUID)` arm and the ✕ confirm rides the same window-level `$alert` item; the switch in the root modifier renders the destructive End Session dialog. Guard: `testSandboxWindowHasExactlyOneAlertPresentationPath` (source audit, comment-lines excluded) pins the file to exactly one `.alert(` so a second nested modifier can't sneak back.

### Zig 0.17 caches configure-time subprocess output — a removed CLT keeps linking against a ghost SDK
2026-07-19, first build after the macOS 27 beta + Zig 0.17 upgrade. `app/build.sh` "succeeded" locally but the zig link failed with "unable to find framework IOKit/Metal…" searching `/Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk` — a path that DID NOT EXIST: the macOS 27 upgrade had removed the CommandLineTools entirely (only Xcode-beta remained). Two independent traps stacked:
- **build.sh hard-forced `DEVELOPER_DIR=/Library/Developer/CommandLineTools`** (historical zig-vs-Xcode SDK clash), so once the CLT vanished, `xcrun` inside build.zig failed outright. Fix: build.sh + scripts/xcode-build-server.sh now prefer the CLT when present and fall back to `xcode-select -p` (the same Xcode SDK CI links against on macos-26).
- **The stale path survived the env fix**: with `DEVELOPER_DIR` unset and `xcrun --show-sdk-path` correctly returning the Xcode SDK, `zig build` STILL emitted `-F <CLT path>` — Zig 0.17's build runner caches configure-time subprocess results (`b.runAllowFail`, here build.zig's `xcrun` call) in `.zig-cache`, and kept serving the answer from when the CLT existed. `rm -rf .zig-cache` was the only reset. Corollary to watch: configure-time reads like `readLlamaTag`/`readMlxcPin` ride the same `b.graph.io` — after any toolchain/SDK/pin change, a stale configure answer means `rm -rf .zig-cache` before deeper diagnosis.
Rule: when a zig link references a path that no longer exists on disk, suspect the configure cache before the environment — and never hard-code a DEVELOPER_DIR without an existence fallback; macOS upgrades remove the CLT.

### A renamed default strands upgraders on the old value forever — the persisted-settings trap (sandbox base image)
2026-07-19, live report right after the sandbox terminal shipped. Repro: factory-reset the sandbox, turn it on with the "default" image, MCP in ChatView hangs, and opening a terminal session shows the stale-image dialog naming `ddalcu/agent-shell` — with Re-pull "doing nothing" and reset not helping. Root cause: the sandbox image was RENAMED (`ddalcu/agent-shell` → `ddalcu/agent-shell-mlxserve`, the one with dropbear baked in) by changing the struct DEFAULT — but the tolerant settings decoder faithfully restores whatever the blob stored, so every pre-rename user kept the old ref. Factory reset deletes `~/.mlx-serve/sandbox` (the image CACHE) yet the setting lives in the config blob and survives; Re-pull faithfully re-pulls the stored old ref. Both "fixes" were correct and useless. Resolution: the base image is PINNED (`ServerOptions.SandboxConfig.baseImage` static, setting + Settings field deleted, stored values ignored at decode) — the ssh terminal and in-guest MCP servers depend on what that exact image ships, so "any arm64 image" was already a broken promise. Guard: `testStoredBaseImageIsIgnoredImagePinned`. Rule: changing a persisted setting's DEFAULT does nothing for existing users — ship an explicit migration, or pin the value if the feature actually requires one specific artifact.

### Enabling the sandbox left already-running MCP servers on the host (spawn-time placement class)
2026-07-19, user-verified live: start MCP servers with the sandbox OFF, then enable it in Settings — the servers keep executing on the host with full user permissions while the UI says "sandboxed"; only unchecking + re-checking each server moved them into the guest. Mechanism: where a stdio server runs is decided ONCE, at spawn (`MCPSpawnerRouter.spawner(sandboxEnabled:)` inside `spawnAndConnect`), and `startEnabled()` is deliberately idempotent (`sessions[id] != nil → continue`), so no later reconcile ever revisits the decision. The reverse flip was silently worse: disabling the sandbox tears down the guest VM under live guest-side bridges — the sessions still look connected but their transport is dead. Fix: `AgentSandbox.configure` computes `mcpRestartNeeded` (toggle flipped either direction, or a guest re-provision under a live sandbox) and posts `placementChanged`; `MCPManager` observes and runs `restartStdioForSandboxChange` — stop every RUNNING stdio session, then `startEnabled()` respawns through the router's current answer. Strictly respawn-what-was-running: with zero sessions it's a no-op, because the launch-time `configure()` also announces and must never eagerly start servers before ChatView's first `startEnabled()`. HTTP sessions have no placement and are untouched. Rule: any behavior chosen at spawn/creation time from a mutable setting needs a subscriber that re-evaluates the live instances when that setting changes — idempotent reconcilers ("skip if exists") are exactly the loops that will never fix it for you.

### The sandbox only gated shell — file tools wrote to the host regardless (file-tool bypass class)
2026-07-20, live from chat-history.json: with the Agent Sandbox ON and the VM pinned to a DIFFERENT workspace by a live CLI session, the same turn produced both behaviors — `writeFile` wrote 23 KB to the host workspace, then `shell uname -a` was correctly declined with the remount-block message. The sandbox routing lived exclusively in `ShellHandler.route`; the five file tools (`readFile`/`writeFile`/`editFile`/`searchFiles`/`listFiles`) ran host-side via FileManager with `resolveAndConfine` as their ONLY containment — and that helper silently returns the path UNCONFINED when `workingDirectory` is nil (the Scheduled-Tasks yolo contract). So "sandbox ON" promised confinement the file tools never enforced: a nil-wd session could write anywhere the app can reach, and a session whose folder the pinned VM wasn't even sharing still got writes.
Fix, three parts:
- **`FileToolSandboxGate`** (ToolExecutor.swift): pure `rejectReason` + injectable closures (same seam pattern as `ShellHandler.sandboxEnabled`), checked FIRST by all five file handlers. Sandbox ON + nil wd → hard fail with a steer (never unconfined host file access); sandbox ON + live pinned guest whose share doesn't cover the session's folder (`AgentSandbox.needsRemount` on `pinnedWorkspace`) → decline naming the session, both paths, and the way out. Sandbox OFF preserves today's behavior exactly, including yolo. Deliberately NOT routed through the guest: virtiofs makes in-workspace bytes identical either way, and guest routing would break exactly when the VM is pinned elsewhere. browse/webSearch/saveMemory are not file tools and stay ungated.
- **The default agent workspace became a SETTING** (`ChatSession.defaultWorkingDirectory`, UserDefaults-backed, builtin `~/.mlx-serve/workspace` fallback; Settings → Agent Sandbox row): changing it retargets sessions still on the OLD default (per-session picks untouched — `ChatSession.retargeted`), stores + starts the global `SecurityScopedBookmark.defaultWorkspaceName` bookmark (MAS reachability; started each turn beside the per-session slot in ChatTurnEngine), all via `AppState.setDefaultAgentWorkspace`.
- **Eager VM remount on workspace pick**: `AgentSandbox.noteWorkspaceChanged` (pure rule `workspaceChangeAction`, mirroring `ensureBooted`'s remount decision) tears down a live guest whose share no longer covers the new folder — next command reboots sharing the right one, same teardown `configure` does for image/network changes — and returns the shell-identical decline message while pinned (surfaced in the Settings row's explainer; the chat toolbar pick calls the same hook and lets the next command's error explain).
Guards: FileToolSandboxGateTests (pure rule + wiring for all five handlers + nothing-written-on-deny + yolo preservation), AgentWorkspaceDefaultTests (stored default, fallback, retarget rules), AgentSandboxTests `testWorkspaceChangeActionRules`. Rule: a confinement toggle must gate EVERY tool that touches the resource it confines, not the one tool it was built around — and each new file tool takes the gate.

Follow-up ruling (same day): the working directory is MANDATORY for file tools in EVERY mode — the "yolo passes nil wd = unconfined" lever was retired, not preserved. `resolveAndConfine` now throws `workspaceRequiredMessage` on a nil wd regardless of the sandbox toggle (the gate keeps only the PIN question); yolo task runs anchor at the default agent workspace (`TaskScheduler.workDir` — yolo's meaning is now purely the APPROVAL matrix in ApprovalPolicy: auto-allow everything, never ask; shell remains command-unrestricted). And the answer to "shouldn't file tools run IN the VM when the sandbox is on?": routing them through the guest would push the same bytes through a shell/base64 pipe into the same virtiofs folder — the guest's /workspace IS the host workspace folder; what the VM adds for file I/O is only kernel-enforced path containment, which the resolver now provides (symlink-resolving prefix check + mandatory wd + pin agreement) minus exotic TOCTOU. The VM stays the boundary for CODE (shell/MCP/CLI sessions); the resolver is the boundary for the app's own file primitives. Guards: `testEveryFileToolRequiresAWorkingDirectoryInEveryMode`, `testEveryAutonomyLevelGetsARealWorkingDirectory`.

Round 2 (2026-07-20, live report): the Settings workspace pick was DECLINED under a live terminal session — the pin path returned the block message, the only surface was a ⚠️ line in the row explainer, and `ls /workspace` in the open session kept showing the old folder until an app restart. The pin exists to protect sessions from IMPLICIT switches (a chat command from a different-workspace tab must not reboot the VM under a live TUI) — but a Settings pick is EXPLICIT intent to re-anchor the sandbox, so declining it optimizes for exactly the wrong thing. Fix: `workspaceChangeAction` gained `restartPinnedSessions` — the Settings path tears down even when pinned and posts `workspaceRemounted`; SandboxTerminalView restarts every living tab IN PLACE (`SandboxSessionTabs.restart` keeps the tab id + display name — never a new "pi 2" for what the user still calls "pi"; CLI conversation state is honestly lost with the VM). Both teardown arms also post `placementChanged` (guest-side MCP bridges die with the VM — the spawn-time-placement class again). Trap dodged in review: the respawn races the old ssh's `onExit` — a stale exit callback would have killed the REPLACEMENT runtime; `sessionExited` now requires the exiting runtime to be the CURRENT one (identity check), which also subsumes the old closed-tab no-op. Chat-tab picks keep the decline (chat-scoped intent; the next tool call explains). Guards: `testWorkspaceChangeActionForceRestartsPinnedSessions`, `testRestartKeepsIdentityAndReturnsToPreparing`, `testRestartLeavesExitedTabsAlone`.

### Picking a voice backend did not download it, and the bundle it reused made a complete download look incomplete (2026-07-26)
Kokoro-82M shipped fully working (engine, phonemizer, `/v1/audio/speech`, Settings picker, tray picker, web console) but selecting it in Settings ▸ Voice did nothing you could act on: the weights were a hand-run local conversion, the preset still carried the old `local/` repo id, and Settings ▸ Voice was the one surface offering a backend with no way to fetch it (every gen pane has had a `BundleDownloadBar` all along). Publishing the weights (`ddalcu/Kokoro-82M-MLX-Serve`, f32, ~345 MB, `g2p/` included) is only half of it. Three things had to follow.

**The bundle.** `AudioModelPreset.bundle` returned `MediaBundle.tts(...)` for every audio preset, and that bundle's ready markers demand a `speech_tokenizer/` directory Kokoro does not have. A fully downloaded Kokoro would therefore have read as permanently incomplete: the bar never collapses, the pane offers Download forever, and clicking it size-matches every file, skips them all, finishes instantly and re-checks still false (the same shape as the Mage-Flow `existingModelDir` bug). `supportsCloning` already existed on the preset as the UI's discriminator, so it became the bundle's too: `MediaBundle.kokoro` is recursive (the dictionaries live in `g2p/`) with markers `config.json` + `model.safetensors` + `voices.safetensors` + `g2p`, because either of the last two missing breaks the engine AT LOAD rather than degrading.

**The catalog.** The interrupted session had moved Kokoro to the FRONT of `AudioModelPreset.all`, i.e. made it the default audio model. That list is what the MEDIA panes offer, and both AudioGenView's reference-clip control and VideoGenView's "Speak text" composer send `ref_audio`, which Kokoro answers with a named 400. Ordering cannot prevent that; absence can. `.all` went back to the four cloning-capable Qwen3-TTS presets and `allIncludingVoiceOnly` carries the full list for the model browser and the catalogue guards. Nothing in voice mode regressed because every path that speaks with Kokoro (`ClonedVoiceSynthesizer.synthesize`, `VoiceCloneMenuModel.kokoroModelDownloaded`, `SettingsView.kokoroBundle`, `KokoroVoiceCatalog.grouped`) names `.kokoro82M` directly.

**The readiness probe.** Both tray checks asked `ServerManager.resolveModelDir(repo:) != nil`, which is true as soon as `config.json` exists. That was harmless while the weights arrived by hand-run script (they were complete or absent), but the app pulls them now, and `config.json` lands in the first second of a 345 MB transfer. Mid-download the tray would have listed all 54 voices as available while Settings correctly showed a progress bar, and picking one gets silence. Both probes now go through `bundleOnDisk`, i.e. the same ready markers the download bar reads, so the two surfaces cannot disagree. Related: the tray snapshots those probes in `.onAppear` (the menu body re-evaluates ~20 Hz while speaking, far too often to stat the disk), so a download finishing with the tray open left Kokoro looking unavailable until you closed and reopened it. Re-stat on a `DownloadManager.downloads` publish, keyed on a status projection so the comparison stays cheap. Note the chat window's voice sheet needed its own `.environmentObject(appState.downloads)`: a sheet does not inherit the presenter's environment objects, and a missing one is a runtime crash, not a compile error.

Guards: `testKokoroBundleIsNotTheTtsBundle`, `testAudioBundleDispatchFollowsTheDeclaredCapability`, `testKokoroIsAbsentFromTheMediaGenCatalog`, `testKokoroReadinessNeedsTheG2pDictionaries`, `testKokoroUnavailableReasonPointsAtTheDownload`. Rule: a preset's declared capability decides its bundle as well as its UI, a catalog is a capability contract rather than an ordering preference, and once a model can arrive incrementally, "is it here?" means ready markers, never one file.

### Parallel chunked downloads: the `.partial` SIZE stops being the resume cursor (2026-07-26)
`DownloadManager` pulled files serially, one `dataTask` each, with a FRESH `URLSession` per file per retry — so every file re-paid TCP + TLS + the `resolve/main` 302, and a single stream had to carry the whole transfer. The fix is two independent things, and only one of them is a speedup you can quote.

**Chunking.** `ChunkedFileDownloader` opens up to `DownloadChunking.defaultConnections` ranged requests against the same URL, each writing its own region of the `.partial`. Writes go through `PositionalFile` (`pwrite`), not `FileHandle` — a shared write cursor is exactly the state N connections must not share. Measured against the HF CDN on a single-stream-limited line (2026-07-25, same file, same minute): 1 conn 22.6 MB/s, 8 conns 41.5, 16 conns 46.3. Re-measured 2026-07-26 on a line that saturates: 1 conn 55.8 MB/s vs 8 conns 53.7 — a wash. So this is insurance for the constrained case, NOT a speedup everywhere, and quoting it without naming the link is the same trap as quoting a decode win without naming the engine.

**The shared session.** `DownloadSession.shared` is one `URLSession` for every transfer, with `httpMaximumConnectionsPerHost` raised — URLSession's default is 6, which would have silently queued chunks 7 and 8 of an 8-way plan and made the whole feature look like it did nothing. Measured on a repo's 10 small config/tokenizer files: 2.30 s with a session per file vs 1.34 s shared (1.71x), i.e. ~100 ms of handshake per file. That one IS a real win and it's the half nobody would have thought to measure.

**What the parallelism actually broke.** The old resume rule was "the `.partial`'s size is how far we got" — true only while one stream appends. With N connections writing at their own offsets the file is SPARSE, its size is the highest offset touched, and using it as a resume cursor would send `Range: bytes=<near-EOF>-` and skip everything in between. Per-chunk progress therefore lives in a `<name>.partial.parts` sidecar, written AFTER the bytes land so it can under-report (those bytes get refetched and rewritten at the same offsets — harmless) but never over-report, which would leave a hole in a file we then call complete. The sidecar and the `.partial` are one unit: `commitPartial` retires both, and `removeGgufQuant` deletes both, or the next download resumes against a plan for bytes that have already moved.

Three fallbacks the live protocol forces, each a distinct failure: a `200` answer to a ranged chunk means the origin ignored `Range`, so the whole plan is void → wipe and re-stream; a `Content-Range` total that contradicts the size we planned against means every boundary past chunk 0 is wrong → same treatment (HF's tree listing and the CDN are two different sources of truth); and a chunk that closes CLEANLY having sent less than its range asked for is the one that no socket error reports, so `runChunked` compares banked bytes against the file size before declaring success. An upgrade mid-download keeps its bytes: with no sidecar present, an existing `.partial` is a contiguous prefix from the old path, and `planAdopting` turns it into a completed leading chunk instead of throwing away however many GB were on disk.

Also shipped here: the app finally sends an HF token (`hfToken` → `HF_TOKEN`, then `$HF_HOME/token`, then `~/.cache/huggingface/token`). A Finder-launched bundle has NO shell environment, so the env var the Zig CLI reads is almost never set in the app and the `huggingface-cli login` file is the one that actually works. Buys gated repos and API rate limit, not speed.

**The entry-point question.** The app has six download entry points (`start`, both `startGguf` overloads, `startBundle`, `download`, `downloadGguf`) reached from ~14 call sites across the views, plus DocumentIndex's bge auto-pull. All of them funnel into `download` or `downloadGguf`, which are the only two callers of `transferFile` — so the change is app-wide by construction, not by inspection. A seventh entry point that hand-rolled its own `dataTask` would work PERFECTLY and silently opt out of chunking, the shared session and the token; nothing would fail, it would just be slow again, which is why it's source-audited rather than trusted. The same audit one level up ("no service fetches huggingface.co on `URLSession.shared`") had to be written at FILE scope: at LINE scope it passed vacuously, because the URL is built a line or two above the fetch — and that false negative was hiding `HFSearchService`, i.e. the Model Browser's search and per-repo size lookups, which had never sent the token and were the surface most likely to be rate-limited. Non-model downloads stay on their own paths and are unchanged: `OCIClient` (sandbox guest image, a different registry and protocol) and `UpdateChecker` (the app DMG).

Guards: `ChunkedDownloadTests` — planning invariants, sidecar validation, and an end-to-end suite against a `URLProtocol` stub that speaks real Range semantics, asserting the ASSEMBLED BYTES (a chunk writing at the wrong offset is precisely what arithmetic-only tests miss). `DownloadManagerTransferTests` drives the REAL `download(repoId:)` loop against a stub HF origin — a working transport and a commit path that strands a `.partial` both look like "it downloaded fine" — covering retry-then-land, skip-what's-present, cancel-leaves-zero-footprint, and both source audits. Two opt-in live measurements (`MLX_SERVE_LIVE_DOWNLOAD=1`) keep both halves re-measurable, one of them verifying the shipped 16-way reassembly byte-for-byte against the CDN. Known untested: HF's response to sustained 16-way range requests (a 429 degrades to the retry loop, i.e. slower, not broken).

### Agents (personas): who you're talking to, and the five places a setting can be read from (2026-07-26)
The iPhone app had a small, well-liked concept — an Agent is a saved system prompt with a name, a symbol, a web flag and a Kokoro voice — and the Mac had none of it. What the Mac had instead was the opposite: a tray row of chips (wake phrase / Agent / MCP / Think), an auto-approve toggle and a voice picker, all global, all of which the user set by hand before every kind of conversation. So the port is also a simplification: define an agent once (persona, voice, model, capabilities, workspace, sampling, wake phrase), pick who you're talking to, and let their config drive the turn. The tray collapses to who / what it's doing / three transport buttons.

**The one real design risk was the read sites, not the object.** Five surfaces start turns — chat tab, voice tray, scheduled task, Telegram, Quick Launcher — and each one had grown its own reads of `serverOptions`, `appState.maxTokens`, the session's working directory and its own toggles. A sixth read added later, or one surface left behind, is a feature that works everywhere except the place you happened not to check. This is the same class as the LAN rule (chat routes through `server.chatModelId`, never `modelInfo?.name`), so it got the same shape: `AgentResolution.resolve(agent:defaults:)` folds the agent's overrides into the surface's own values, `TurnConfig.from(resolved)` is the only builder, and every field on `TurnConfig` is DECIDED before the engine sees it. `agent == nil` returns the defaults verbatim and has its own test — that's the upgrade guarantee, and it's what let the whole thing land without a behavior change for anyone who never opens the window.

**Two things that looked like details and weren't.**

*Sampling defaults are not shared.* The tool loop has always run at a hardcoded 0.7 while plain chat uses the user's `defaultTemperature`. A `ResolvedAgentSettings.temperature` with the defaults folded in — which is the obvious design — would have handed 0.8 to the agent loop for every existing install, silently, with nothing in the diff that reads like a behavior change. So the resolution carries the agent's RAW `temperatureOverride`/`maxTokensOverride` alongside the decided values, `TurnConfig` carries the overrides, and each path keeps its own fallback (`agentLoopTemperature` vs `serverOptions.defaultTemperature`).

*The persona belongs in the stable prefix.* `composeSystemPrompt` exists because a per-minute timestamp at the front of the prompt used to miss the server's KV prefix cache at token 0 and cold-re-prefill every agent turn. A persona in `volatileTail` would have done the same thing at a smaller scale — a re-prefill per MESSAGE. In front of `stable` it costs exactly one re-prefill per agent switch. Plain chat deliberately has NO system message (a synthesized one used to be read by models as the user's input), so the persona there is one system message shared with voice-mode's style guidance, persona first, and only when there is something to say.

**Capability gating needed two halves.** Filtering the advertised tool list is the obvious one, and it has a trap: `AgentPrompt.toolDefinitionsJSON` is a hand-ordered literal whose key order is load-bearing (`path` before `content`, so a call truncated at max_tokens still carries the path), and a JSONSerialization round-trip would reorder it. One tool per line makes a line-based filter trivial and order-preserving, and allowing everything reproduces the literal BYTE for byte — which matters, because that block sits in front of the whole cached prefix. The second half is that models call tools that were never advertised, so `AgentEngine.disallowedToolRefusal` runs at the top of `executeToolCall`, ahead of the meta-tools (`createTask`, `generate_*`) that aren't ToolHandlers and would otherwise have executed before any handler-level check. Writing that filter also surfaced that `AgentToolKind` was missing FOUR tools the JSON actually ships (`createTask`, `generate_image/audio/video`) — i.e. four tools no capability list could ever have gated. That's a class, not an instance, so the guard asserts the enum and the parsed definitions are equal in BOTH directions, plus that every kind belongs to a coarse group (`loopTools` / `webTools` / the docs gate) so nothing can become unreachable from the UI. `searchDocuments` is deliberately outside the capability toggles: its gate is whether a folder is attached, which is stronger, so it always rides the resolved set (docs-only mode has Tools off and still needs it).

**Three per-agent settings that had to reach outside the turn.**
- *Model.* An agent may pin one, and "Current" (nil) means whatever is running. The outcome that must never be silent is a multi-GB download, so `AgentModelSwitch` returns `.needsDownload` (greyed row + a Download button) rather than fetching, `.unavailable` for an `id@peer` whose peer is gone, and passes a live LAN id straight through with no local load. A SPOKEN switch to an unavailable agent is declined out loud ("Chef needs its model downloaded first") — answering as whoever was active is the failure the user cannot see.
- *Workspace.* Agents carry their own folders, so a switch can move the ground under a live pinned CLI session. That's the case the pin was built to decline — but only for IMPLICIT switches (a chat command from a different-workspace tab). An agent pick is explicit, so it takes the Settings path: `noteWorkspaceChanged(_:restartPinnedSessions: true)`, which tears down, posts `placementChanged` (guest MCP bridges respawn) and `workspaceRemounted` (Sandbox tabs restart in place). What it must NOT do is call `setDefaultAgentWorkspace`: that rewrites a user SETTING and retargets every session still on the old default, with nothing in the UI to show it. A source audit pins that, because it's a one-line mistake that would look like a fix.
- *Voice.* No new plumbing was needed at all: `ClonedVoiceSynthesizer`'s production `voice:` closure already re-read the saved settings ONCE PER UTTERANCE (so a Settings change applies to the next sentence), and that re-read is the seam. `ActiveAgentVoice` is a lock-guarded holder — the closure isn't main-actor bound — consulted before the settings. One asymmetry matters: `.system` must WIN rather than fall through, or an agent asking for the plain Apple voice against a global Kokoro setting still sounds neural; but a half-saved EMPTY value defers to the global setting, because the alternative is silence.

**Multi-agent wake words are a longest-first sweep.** `WakeWord.strip` stays single-phrase; `WakeWord.match` sorts candidates by token count DESCENDING, because "hey loki" is a prefix of "hey loki coder" and a naive sweep makes the specific agent unreachable by voice forever. The Loki homophone table stays Loki-specific (a custom name gets exact + greeting-prefixed matching only, or everyday speech starts waking agents). Collisions are refused at SAVE time and keyed on the phrase's LAST WORD, not the whole phrase: greetings are universal, so "hey chef" and "ok chef" are one gate, and two agents sharing it makes BOTH unreachable with nothing to see until you try talking.

Also: the tray picker and the chat chip observe `AppState`, not the store, so `AgentStore.objectWillChange` is forwarded into `AppState`'s (the same wiring `server` has had) or a newly created agent doesn't appear until an unrelated publish. The chat chip lives in the COMPOSER row, never the toolbar band — its label is agent-name-sized, i.e. runtime-variable, which is precisely what re-triggers the » eviction class. The Quick Launcher takes the agent's persona and sampling but deliberately NOT its tool loop: the panel has no tool-call cards and no approval surface, and "Open in chat" is one keystroke.

Guards: `AgentResolutionTests` (nil-agent parity field for field, each override, coarse→set mapping, Advanced verbatim), `AgentCapabilityGateTests` (byte-identical unfiltered literal, key order after filtering, refusal at dispatch incl. meta-tools, the enum/JSON sync guard both ways), `AgentStoreTests` (round-trip, a file with only the iPhone's keys, unknown future keys, an unknown voice engine costing the voice and not the agent), `AgentWriterTests` (ported from the phone), `WakeWordMultiAgentTests`, `AgentModelSwitchTests`, `AgentWorkspaceSwitchTests` (+ the source audit), `AgentVoiceOverrideTests`, `AgentTurnConfigTests` (one builder + where the persona lands), `PerSessionUIStateTests` (agentId per tab, absent key → nil).

Follow-up, same day (live): **each agent needs its own chat thread.** Voice originally resolved its session as `activeChatId ?? newChatSession()`, which with personas means talking to Chef lands in whatever tab happened to be open — and the first version of the spoken handover made that worse by calling `setAgent` on that session, i.e. rebranding someone else's conversation as Chef's. Routing now goes by AGENT (`AppState.sessionForAgent`): continue that agent's most recent thread, or create one on first use, and a handover moves the conversation instead of renaming the one you were having. Three constraints the wiring forced. The chosen thread must become `activeChatId`, because the controller speaks the ACTIVE session's trailing assistant message — a turn running in any other session generates silently and is never read aloud. An ACTIVE thread of the same agent has to beat a more recently *updated* one, or opening an older Chef thread and speaking yanks you to a different one. And task-run / Telegram-bridge sessions must never be adopted: task runs now carry an `agentId` too (the scheduler sets it), and they're hidden, transient and harvested into a transcript, so a voice turn landing in one would corrupt a run and disappear from the sidebar. Launching voice from a chat tab adopts THAT tab's agent (otherwise a tray default of "Chef" pulls the conversation out of the tab you launched from), and the tray's "New" makes a fresh thread for the same agent while leaving the old one in the list, as it always did. Guard: `AgentSessionThreadTests`.

### The composer's Think/Tools/MCP discs disagreed with what the turn actually ran (agent-lock class, 2026-07-30)

`AgentResolution` decides `toolsEnabled` from the agent's capabilities and `mcpEnabled` from `capabilities.mcp`, ignoring the chat's own toggles entirely — that is the chokepoint working as designed. What was missing is that the composer never asked it: `toolbarToggles` read the session's `mode`/`useMCP`/`enableThinking` and nothing else. So the discs showed one thing and the turn ran another, and the controls were dead on top of that — a click wrote a session field the resolver would throw away.

The instance that makes it concrete: `AgentCapabilities.web` defaults **true**, so `resolvedTools()` is non-empty for every agent ever created, `toolsEnabled: !capabilityTools.isEmpty` is therefore true, and selecting ANY agent — including one with `tools: false` — silently ran the tool loop while the wrench rendered grey/OFF. Nothing in the UI could say so, and the same shape ran in reverse for MCP.

Fix is the display reading the same resolution the turn does. `ChatDetailView.agentModeLock` calls `resolvedAgentSettings` (a pure fold, cheap per render) and hands `ChatModeToggles.resolve` an `AgentModeLock`; the disc renders the agent's value with a dashed inset ring and its menu becomes "Set by \<name\>" + "Edit Agent…". Reading `agent.capabilities` there instead would have been shorter and is exactly how the two drifted in the first place, so a source audit pins the call.

Three things the wiring forced. **Lock per control, not per chat**: thinking is the one an agent may leave unset (`enableThinking: nil`), where resolution falls back to the surface's own value — locking it anyway takes away a control nobody is deciding for you. **A locked control still shows ON/OFF**, because that is what the disc's colour has always said; the lock is a ring, not a dimming, and the hover card carries the "why". **A locked disc still opens something**: a control that does nothing on click is the dead-control class, so the primary action goes away and the menu explains and offers the editor. The pre-send intent nudge is suppressed for a locked mode too (accepting it would change nothing and send the message unchanged), and both setters refuse — the nudge calls into them.

Same session, unrelated bug in the same row: **opening a menu over the composer does not deliver a hover-exit**, so `ComposerTipHover`'s card — a non-hit-testing overlay whose only dismissal was `onHover(false)` — sat under the open menu until you hovered the control again. `NSMenu.didBeginTrackingNotification` is the one signal that covers every way in (SwiftUI's `Menu` and `.contextMenu` are both NSMenus, so it fires for left-click on the agent/attach menus, right-click on the Tools/MCP discs, and press-and-hold). The dismiss must BUMP the reveal token, not just clear `shown`: hover, then click before the 0.45 s delay elapses, and the in-flight reveal would otherwise pop the card up on top of the menu you just opened. That token behaviour is the whole reason `ComposerTipHoverState` was pulled out of the view.

Guards: `ChatModeTogglesTests` (per-control lock, thinking-unset falls through, bridge sessions), `ComposerTipTests` (locked card names the agent, drops "click to turn it on" and the right-click line, keeps ON/OFF and the workspace; the four dismissal cases), `ComposerModeControlTests` (lock built from `resolvedAgentSettings`, setters and nudge refuse while locked, the NSMenu observer).

Follow-up, same day: **the picker went with the lock.** Once the discs are the agent's, the agent chip sitting beside them in the composer row was the last thing that could still change all three mid-conversation — and it did it silently, re-pointing the prompt, tools, model and voice of a thread already underway with only the transcript to show where the seam was. The choice belongs at the moment the conversation starts, so the picker is now `ChatSidebar.newAgentChatMenu`, beside New Chat, and `AppState.setAgent(_:forSession:)` is deleted rather than merely unreferenced — a session's `agentId` is written at creation and nowhere else, which is a structural guarantee instead of a convention. `startChat(withAgent:)` is the one call, because creating the session is only half of it: model, workspace and voice all live OUTSIDE the turn, so a session carrying just the id would run the persona against whatever model happened to be loaded. Changing an agent's behaviour mid-conversation still works, and is the supported route — every turn re-reads `AgentResolution`, so flipping thinking on in the editor flips it on for the chat already in progress.

That made the "Edit Agent…" row load-bearing, and it was landing on the wrong agent: the Agents window is a single reused `Window` whose `onAppear` selected `store.allAgents.first`, so a card that had just said "Set by Chef" opened on whoever sorted first — and if the window was already open it didn't move at all. `openAgentSettings` sets `pendingAgentSelection` before opening, and `AgentsWindowFocus.selection(pending:current:first:)` is the three-way precedence: a request wins over an open selection, no request + nothing selected falls back to first, no request + a live selection returns nil so a re-publish can't yank the user to the top of the list mid-edit. A request for the agent ALREADY showing still counts as a selection, because `onChange(of: selectedId)` cannot fire for an unchanged id and the draft would never reload — the click would look broken in exactly the case the user is most likely to try twice. `ComposerTip.agent(name:)` was deleted with the chip: a card for a control that no longer renders is a sentence nobody can reach.

### First run: a welcome screen that links to fourteen models, and a chat window that dead-ends (2026-07-30)

The app assumed a technical user. Someone installing it for the first time got a welcome window whose only lead was a link to a **Model Browser showing 14 models across 4 vendor sections** (Gemma 4 / Qwen / Laguna / "Largest models 96 GB+"), so before they could chat they had to make a taxonomy decision, on rows carrying capability chips ("Fast replies", "Balanced", "Coding help") that repeated the blurb and said nothing comparative. No server ran (`autoStartServer` was `UserDefaults.bool` on a key nobody had ever set, i.e. false), no chat window opened on dismiss, and opening Chat with no model gave an empty transcript, a "Select a model" pill and a disabled Send — a dead end with nothing on screen saying why.

**One function makes the recommendation.** `RecommendedModelPick.starterPick(physicalMemoryBytes:)` is RAM-tiered (≤8 GB E2B, 8–16 E4B, 16–32 12B, 32+ Qwen 3.6 27B), total by construction, bands upper-INCLUSIVE so a boundary machine — the one with the least headroom in its band — takes the smaller side. Three surfaces read it: the welcome window, the chat gate, and the browser's "Best for your Mac" card. The two first-run ones render the same view (`RecommendedStarterCard`) rather than two similar ones, because they are shown minutes apart to the same person and two copies is how they start naming different models. Every tier is checked against `meetsSystemRequirements` at the BOTTOM of its band, since a recommendation the machine can't load is worse than none.

**The card's job is a working chat, not a downloaded folder.** Its `onFinish` selects the model and starts the server (`useModelAndAwaitReady`), and the welcome window then opens Chat. It also routes a `ggufFilename` pick through the quant path even though no tier is one today — the assumption is what breaks the first time a tier is.

**A blocking sheet still needs exactly one door.** The chat gate was first presented on `.constant(true)`: correctly un-dismissable by Esc/click-away, and Cancel did NOTHING. AppKit refuses to close a window with an attached sheet, so `dismissWindow(id: "chat")` was a silent no-op — measured through the accessibility API, window count 1 before the click and 1 after, sheet still up. The binding's SETTER is what blocks SwiftUI's own dismissals; Cancel flips a `@State` that ends the sheet and only then closes the window, and that order is the whole fix (1 → 0 windows once corrected). The gate also counts LAN peer chat models (`ChatGateState.resolve`, matching `trayHasNoUsableModels`): locking someone out of a conversation their peer can already serve is worse than the composer it replaces. It fires on chat-CAPABLE models, so a Mac whose only download is an image backend is still gated, and it clears itself — `localModels` is `@Published` and the card refreshes it.

**`UserDefaults.bool` on an unset key is not "the user said no".** `autoStartServer` defaults to true now (`object(forKey:) as? Bool ?? true`). It's safe because the launch gate is `autoStartServer && !selectedModelPath.isEmpty` — a no-op until a model exists, so nothing boots eagerly on a fresh install — and the first download's completion hook is what actually starts the server. No migration, deliberately: existing users who never touched the toggle get it on.

**The bars: intelligence is someone else's number, speed is ours.** The chips became three slim tracks (Intelligence / Speed / Context) with no numeric readout. Intelligence is the Artificial Analysis Intelligence Index rescaled `round(index/60 × 100)`, read 2026-07-30, reasoning variant, describing the ORIGINAL weights rather than our 4-bit build. Speed is never theirs: their figure is measured on cloud GPUs and that ordering does not survive the move to Apple Silicon, where decode is bandwidth-bound and ACTIVE parameters dominate — same M4 Max in the 26.7.12 bench (`docs/perf-csvs/all-26.7.12.csv`, in git history), `gemma4-26b-a4b` 118 tok/s against `gemma4-31b` 25 tok/s, a 4.7× gap a cloud comparison shows as nearly level. So the speed scores are hand-tuned against that bench, and `activeParamsB` (a fact per checkpoint, read from each repo's config or model card) exists to CHECK them: a model that wakes more parameters per token can never be scored faster than one that wakes fewer. Ties are unconstrained, which is where 4-bit vs 8-bit and different expert-bank sizes legitimately differ, and is also where the real crossings land (Hunyuan 3 at 21B active measured 26 tok/s against Qwen 3.6 27B's 28 — scored equal rather than fudged). Speculative decode is excluded from the score on purpose: it is a property of how we run a model, it moves some picks 2-3× on their own, and every pick that has one already says so in its blurb. Models with no site entry (both Lagunas, Hunyuan 3) carry our estimate and render "estimated" beside the bar — there is no unrated state, because a missing bar reads as "bad" rather than "unknown". Context is the checkpoint's own `max_position_embeddings`, on a log scale between 32K and 1M, and deliberately NOT the RAM-clamped effective window: the bars compare models to each other, and the clamp is a property of the user's Mac.

Verified by simulation rather than by reading the code: the three model roots (`~/.mlx-serve/models`, LM Studio's folder, the HF hub cache) moved aside under a restore trap, the app launched against the empty tree, and the flow driven through the accessibility API — welcome names the RAM-matched model, the gate's own AX contents read back verbatim, Cancel takes the window count to 0, and Download (with the real checkpoint staged where the transfer lands, so it size-matches and skips) reaches `"state":"ready"` on `/v1/models` with the sheet gone in under 5 s.

### The shipped app advertised a version the release didn't have, so it nagged forever (v26.8.1, 2026-07-31)

`v26.8.1` was cut by a `workflow_dispatch` at **01:26 UTC on Aug 1**, while it was still **21:26 Jul 31** in the maintainer's zone. The workflow computed CalVer with `date -u +%y.%-m`, so it minted `26.8.1` against a CHANGELOG, an `app/Info.plist`, perf CSVs, website PNGs and a merge commit that all said **26.7.12**. That's the visible half, and it's the boring half.

The half that actually broke users: **the `.app`'s version came from a committed file while the tag and the CLI binary came from a CI-computed value, and nothing kept them equal.** `app/build.sh` PlistBuddy-stamps `Info.plist` at build time, but the release workflow only ever did `cp app/Info.plist "$CONTENTS/"` — verbatim, no stamp. So the DMG published under `v26.8.1` contained an app reporting `26.7.12`, while `zig build -Dversion=…` baked `26.8.1` into the `mlx-serve` beside it. One bundle, two versions, neither matching the release it shipped in.

`UpdateChecker` then does exactly what it should: `releases/latest` returns `v26.8.1`, `isNewer("v26.8.1", than: "26.7.12")` compares `[26,8,1]` against `[26,7,12]`, hits `8 > 7` at index 1 and reports an update. Installing it does not help — the new DMG also reports `26.7.12` — so the prompt returns on the next check, forever. Self-perpetuating, invisible to every existing guard, and impossible to diagnose from the app: the tray is telling the truth about a bundle that was mislabeled upstream.

Note which comparison saved it from being worse. CalVer `YY.M.N` is not semver-ordered across a month rollover: `26.8.1` sorts above `26.7.12`, but `26.7.12` sorts above `26.7.9`. Component-wise numeric comparison gets both right; a string compare would have gotten the second one wrong and hidden this class behind a different bug.

Fixes, both in `.github/workflows/release.yml`:

- **Stamp the bundle's Info.plist with the version the run is releasing**, right after the `cp` and before the bundle is codesigned (a stamp after signing invalidates the signature). Stamp the **copy** under `$CONTENTS`, never the repo file — the Homebrew step later runs `git rebase origin/main`, which refuses to run against a dirty tree, so stamping the source would fail the release at the very last step.
- **Pin the CalVer timezone** (`TZ: America/New_York` on the version step, `TZ=… date` in `app/build.sh`) so CI and a local build resolve `YY.M` from the same clock. Runners are UTC; without the pin the two disagree for a few hours around every month boundary, which is precisely when a release cut in the evening lands in the wrong month.

Guard: `tests/test_release_workflow_gates.sh` — the stamp exists, uses `steps.version.outputs.version`, targets `$CONTENTS/Info.plist`, lands before `codesign`, and the two CalVer sites name the same zone. Hermetic (PyYAML parse, no runners). It folds backslash-continuations before scanning, because a guard should pin what a command does, not how it's wrapped.

The `26.8.1` tag, release and Homebrew bump were deleted and the release re-cut as `v26.7.12` from a tag push, which takes the workflow's `else` branch (`version=${GITHUB_REF_NAME#v}`) and so never consults the clock at all.

## The workaround the app could not reach (`--max-resident-mem`, 2026-08-07)

The server has had a residency cap since Plan 05 Phase D, and every diagnosis of
a refused load ends with "raise `--max-resident-mem`" — the 503 body says so, the
refusal log says so. `ServerOptions.toCLIArgs` emitted forty-odd flags and not
that one, and there is no extra-args passthrough, so from the GUI the cap was
always `auto` (80% of Metal's recommended working set at startup) and a model
that cap refuses was unloadable at any setting.

`--skip-mem-preflight` looks like the escape hatch and is not: it gates
`doLoadGenOnInferenceThread`'s media preflight and the MLX loader's, both of
which run AFTER `ensureLoaded`'s registry gate. The gate is what returns the 503.

Two things the field needed beyond the obvious:

- **A value that cannot be malformed needs no validator.** `main.zig` calls
  `std.process.exit(1)` on a `--max-resident-mem` it cannot parse, so a typo in a
  free-text Settings field would stop the server from starting at all, with the
  cause buried in a log nobody opens because the app never came up. The first cut
  answered that with `ServerOptions.parsableSizeArg`, a Swift mirror of
  `parseSizeArg`'s grammar (bare bytes, optional B/KB/MB/GB, `off`, `0`, plus
  `auto` → omit) that dropped anything else. That is a second copy of a grammar,
  in a second language, with its own test enumerating junk strings — all of it
  defending a text field nobody needed. The field is a **slider** now:
  `maxResidentMemGB: Int`, 0 = Auto, snapping to `residentMemPresets`, whose
  ladder stops at the machine's physical RAM because a cap above it can never be
  reached. The validator, its test, and the whole malformed-input class went with
  it. Prefer deleting the input over defending it.

  `snappingSlider` moved from `RequestDefaultsSectionContent` to file scope
  rather than being copied — two sections render ladder-valued settings now, and
  a second copy is how the two drift.

- **A new `ServerOptions` field needs a `SettingsReset` section.**
  `testEveryServerOptionsFieldBelongsToExactlyOneSection` fails otherwise — a
  field owned by no section can never be restored by "Reset section to
  defaults". Good tripwire; it caught this within one test run.

## `mlx-serve` survived the app quitting it (#133, 2026-08-07)

Reported against 26.8.2: quit MLX Core with ⌘Q or the Quit menu and the
`mlx-serve` process stays up with a model loaded, holding gigabytes. The
reporter also found the workaround, which is the diagnosis: "pressing the power
button in the menubar pull-down menu quits the main GUI and closes mlx-serve."

That button is the only quit path with an explicit teardown:

```swift
Button {
    server.stop()
    NSApplication.shared.terminate(nil)
}
```

Every other route — ⌘Q, the app menu's Quit, Dock ▸ Quit, any other
`terminate(nil)` — goes straight to termination, and nothing signals the child.

`ServerManager.deinit` is the trap. It cancels the pollers and calls
`proc.terminate()`, so it reads exactly like the safety net for this, and it is
not: `ServerManager` is owned by an `AppState` held as a `@StateObject`, and
those are not deallocated at app termination. The process image is torn down,
`deinit` never runs, and the child is reparented to launchd still holding the
model.

The fix belongs to the object that spawned the process, not to a button:
`ServerManager.init` observes `NSApplication.willTerminateNotification`. One
`ServerManager` exists, so every quit route is covered by construction and a
future quit affordance cannot forget the teardown.

`queue: nil` is deliberate. `addObserver(forName:object:queue:)` with a queue
ENQUEUES the block; during `applicationWillTerminate` the app may never turn its
runloop again, so the teardown would be scheduled and then dropped — the same
leak with extra steps. `queue: nil` delivers synchronously on the posting thread,
which AppKit guarantees is the main one, so `MainActor.assumeIsolated` is sound.

What makes the guard real rather than decorative: the test posts the
notification and asserts `status == .stopped` with NO runloop turn, no `await`
and no expectation in between. A `.main`-queued version of the fix fails it.

`stop()` sends SIGTERM and the server handles it — verified against a running
server with a model resident: it logs "Shutting down gracefully..." and exits.
The tray button keeps its explicit `stop()`; it is idempotent, and belt-and-
braces on the one path users already know works is cheap.

## The truncation banner rode history as assistant prose (2026-08-11)

The live capture behind the muse loop investigation had a second, app-side
half: `TruncationNotice.text(...)` was APPENDED into `message.content` via
`updateLastMessage(content:)`, and content is exactly what `plainHistoryDict`
and `buildAgentHistory` send back. So after one repetition cut, every later
request carried "⚠️ *Stopped — the model started repeating itself and the
server cut the reply…*" as something the assistant SAID — the model reads a
warning about looping, verbatim, every turn, in a chat already primed to
repeat. Same class as the error-echo rule: our own UI text became the error.

Fix: the notice is a FIELD (`ChatMessage.truncationNotice`, a
`TruncationNotice.Notice` carrying cause + cap), set by the plain-chat path,
both agent-loop exits and TestServer, and drawn by `MessageBubble` as a
footnote under the reply — content never carries it, so history builders
can't resend it by construction. Sessions saved BEFORE the change still hold
the banner inside content (the capture does), so both builders scrub it at
build time with `TruncationNotice.stripped(from:)`; the strip markers derive
from the legacy `text(...)` string itself, so the two cannot drift apart.
Decode is tolerant both directions: absent on every old message, and an
unknown future cause nils the notice instead of failing the whole message.

Relation to PR #147: its diagnosis (doubled banner) was a SERVER wire-shape
bug — the include_usage chunk restated the ending (docs/gotchas/server-http.md)
— and its `APIClient.TruncationGate` remains worthwhile hardening for
third-party backends. Its ChatTurnEngine append-after-stream hunk is
superseded by the field.

Guards: `TruncationNoticeTests` (field + clean content, both-cause strip,
history builders carry neither field nor legacy text, tolerant decode).

## The badges stopped where the small window had ended (fullscreen, 2026-08-13)

⌘-held numbers the sidebar's conversation rows, and only the rows in the
CLEAR get one — the list scrolls under the pinned destination block, so a
number behind frosted glass names a shortcut you cannot read. That band was
measured from three `frame(in: .global)` readers: the block's bottom edge, the
column's bottom edge, and each row's own span.

In fullscreen the badge count stopped tracking the window. Reproduced in an
isolated SwiftUI harness with the same structure (`ScrollView` of rows +
`.safeAreaInset(.top)` block + the two background readers), stepped through
900x600 → 900x1300 → fullscreen:

```
windowed    blockMaxY=370.0  rows 378…406, 408…436, …   in band 16
TALL        blockMaxY=370.0  rows 378…406, …            in band 16
FULLSCREEN  blockMaxY=370.0  rows 396…424, 426…454, …   in band 18
```

`blockMaxY` is the same 370.0 in every state while the rows genuinely move
(378 → 396). **A `.global` frame is not re-published when a view merely MOVES
rather than resizes** — entering fullscreen translates the whole column, and
the pinned block keeps whatever global maxY it had in the window that last
laid it out. The band's top edge is then a number from a different window
size, which is exactly what it looked like: badges based on the geometry the
feature happened to be written at.

The second `SidebarClearBandTopKey` publisher, `frame.minY +
safeAreaInsets.top`, was documented as the same line derived a second way. It
never was — measured, it reports **104** against a frost line at **370**,
because that reader sits below the toolbar (it has already lost that inset)
and the block's height is not part of what it reads back. It was harmless only
because the key reduces by `max` and the real number was always larger. A
fallback that can only ever be wrong is worse than no fallback: deleted.

Fix: one named coordinate space on the column (`ChatSidebar.bandSpace`), and
all three measurements taken in it. In the column's own space the top edge is
the block's HEIGHT and the bottom edge is the column's HEIGHT — neither can be
invalidated by the window moving, and both re-publish when there is genuinely
a new layout. Same harness, same three states, container space: top constant
at 318, bottom 825 → 827 → 897, and fullscreen numbers 19 of 19 measured rows
where the global version numbered 18.

The filter itself moved out of the view to `ChatQuickSwitch.numbering(rowSpans:
clearBandTop:clearBandBottom:)`, which is what makes "a taller window numbers
more rows" a test rather than a screenshot. Guards: `ChatQuickSwitchTests`
(band cases + a source scan pinning zero `.global` frames and exactly three
readers in the named space — verified red by reverting one reader).

## Continuing a reply filed itself as a new version of it (2026-08-14)

Three features landed together and two of them meet on one message. Regenerate
puts a version pager under a reply; Continue hands the reply back to the model
to finish. Both end at the same place — `ChatTurnEngine.endTurn` →
`AppState.finishRevisions`, the ONE turn exit — and that exit could not tell
them apart:

```swift
let seed = pendingRevisionSeed.removeValue(forKey: sessionId)   // nil after a continuation
guard seed != nil || !msg.revisions.isEmpty else { return }     // …but revisions is not
```

So a reply that had been regenerated once (pager reading 2/2) went to **3/3**
the moment you continued it, with 2/3 holding the same reply minus the ending
that had just been written. Nothing errors and nothing is lost — the text is in
v3 — which is what makes it the quiet kind: the pager silently grew a page that
is not another answer to the question.

The rule is that **the pager counts REGENERATIONS**, and a continuation is not
one. It is the reply you are reading, carrying on — the same relationship an
edit has to the version it changes, which is why `MessageRevisions.applyingEdit`
already exists and says so: stepping away and back reloads `content` from the
stored revision, so any in-place change that does not sync into the list is
discarded the first time you touch an arrow.

The fact has to be HELD, for the same reason `pendingRevisionSeed` is held: the
turn exit is the only place that knows the turn is over, and by then the request
that started it is gone. `AppState.markContinuing(_:)` is that marker, and it
carries the seed's ordering hazard too — `continueReply` opens with
`stop(sessionId:)`, which IS a turn exit, so a mark set before it is consumed
immediately and the continuation records itself as a new version anyway. Set
after `stop`, pinned by a scan that reads the two statements in order.

Two more things describing a message that a continuation changes and the first
cut did not carry over. The truncation notice was already cleared (the reply was
cut; it is being un-cut). Token usage was not: `updateLastMessage` **replaces**
`completionTokens`, so a 900-token reply finished by a 42-token continuation
reported 42 in its footnote. `addingCompletionTokens:` adds instead, driven by
`runPlainTurn`'s own `continuing` flag — the prompt count and the rate stay the
latest generation's, since a continuation's prompt already contains everything
before it and a rate is not a quantity to sum.

And the affordance itself: `ContinueReply.isEligible` takes the `ServerEngine`
now. ds4 renders its chat template inside the embedded engine, where there is
nowhere to append a prefill, so the server refuses by name
(`continuationRejectReason`) — a live button over a guaranteed 400 is the
dead-control class, and the same rule as a locked composer disc.

Guards: `ContinuedReplyBookkeepingTests`, `ContinueReplyTests` engine cases,
`RegenerationSeedWiringTests` continuation cases (both verified red by
reverting each half separately — the AppState branch and the `markContinuing`
ordering fail independently).

## A typed `onDrop(of:)` never saw the Create panes' drops (2026-08-13)

The media panes' shared drop target (`MediaDropTarget.swift`) shipped as
`onDrop(of: [.fileURL])` with the type filtering done in the completion
handler — after the providers had RESOLVED, which is after SwiftUI has
accepted the drop and animated the file in. So a dropped `.txt` lit the
dashed border, flew home, and nothing happened: the pane had already said
yes to a file it then silently discarded.

The obvious fix — name the kind's own UTTypes in `of:` (`.image` / `.movie` /
`.audio`), the way the chat composer's drop reads — made it worse in the way
that is hardest to argue with: the panes stopped accepting ANY file. A drag
out of Finder puts `public.file-url` on the pasteboard and nothing else; the
file's own content type is not there to match against. A target registered
for `.image` is therefore never offered the drag, and no amount of correct
filtering downstream matters. (The chat composer gets away with the typed
list because its other source — an image dragged out of a browser — really is
`public.png` DATA, and its fallback branch reads that with `NSImage`. It has
never been the same question: the panes need a PATH on disk.)

The two requirements are not in conflict, they just belong to different
hooks. The target registers `.fileURL`, which is what a file drag actually
is, and the refusal moves to `DropDelegate.validateDrop` — the one hook that
answers while the drag is still in the air. It reads the drag pasteboard
(`NSPasteboard(name: .drag)`, `.urlReadingFileURLsOnly`) and gets the real
URLs synchronously, so the verdict is the pane's own extension allow-list
rather than a UTType approximation of it: a `.tiff` the picker opens is
accepted, a `.txt` bounces, and a full slot bounces too. `dropEntered` fires
on validated drops only, so one verdict both refuses the file and decides
whether the border lights.

A drag the pasteboard tells us nothing about (`urls` empty while the drag
still claims to carry files) is NO INFORMATION, not a refusal: it falls back
to accepting and filtering after the resolve, which is exactly the old
behaviour. Between the two failures, bouncing everything is the worse one —
it is the one that makes the pane look broken.

Guards: `MediaDropTests` (the verdict per kind and for the mixed H3 target,
every allow-listed extension passing it, the full-slot and unreadable-drag
cases, a real pasteboard round-trip proving the read recovers a
`public.file-url` item and skips a web URL, and a source scan pinning
`.onDrop(of: [.fileURL]` + `validateDrop` in the modifier).

## The Hugging Face cache was "supported" and never worked (2026-08-14)

Reported as "we don't account for `HF_HOME`". We did — `DownloadManager.huggingFaceRoot`
read `HF_HUB_CACHE`, then `$HF_HOME/hub`, then `~/.cache/huggingface/hub`. The code
was right and the feature was dead, because the read was
`ProcessInfo.processInfo.environment` and a bundle launched from Finder or the Dock
has none of the user's shell environment. Anything exported from `~/.zshrc` is simply
not there. So the env branches only ever fired when the binary was run from a terminal,
and everyone else silently got the default cache — an empty folder if they had moved
theirs to an external drive.

The tell that this was already known and had been solved twice, in two other places:

* `hfToken`'s own comment — *"the token file `huggingface-cli login` writes is the one
  that actually works in the app, because a bundle launched from Finder has NO shell
  environment"*.
* `CLIInstaller.userShellPathEntries()` — spawns `$SHELL -l -i -c` to print `$PATH`,
  because *"Finder-launched apps get a minimal PATH that never contains
  `~/.local/bin`"*.

Neither generalized, so the third instance shipped broken. `LoginShellEnv` is now the
one probe and `CLIInstaller` delegates to it.

Mechanics worth keeping:

* **Markers, not line parsing.** An interactive rc file prints banners. Each value comes
  back between `__MLX_ENV_<NAME>_BEGIN__` / `__MLX_ENV_<NAME>_END__`.
* **An unset variable is omitted, never returned as `""`.** Otherwise the shell's empty
  answer shadows a value the process genuinely has, and the merge inverts.
* **Process env wins.** A launch that does carry the variable (terminal, or an explicit
  override) is authoritative; the shell only fills gaps.
* **Off-main, watchdogged.** The probe spawns a shell and can be wedged by a pathological
  rc file, so it is primed from a detached task at launch with a 5 s terminate, and
  `refreshModels()` re-runs only if the resolved root actually moved. `huggingFaceRoot`
  had to stop being a `let` for that.
* **A configured-but-missing root is nil.** Falling back to `~/.cache/huggingface/hub`
  when `HF_HOME` points at an unmounted drive would list the wrong library and read as
  "your models vanished" rather than "that drive isn't mounted".
* **`XDG_CACHE_HOME` was missing entirely** — `huggingface_hub` defaults `HF_HOME` to
  `$XDG_CACHE_HOME/huggingface`, so a user who moves only that var was in the same hole.
  It is now in the precedence for both the cache root and the token file.

Under MAS the probe is gated off (`BuildFeatures.customModelFolders`): the app is
sandboxed, `~` is the container, and a cache outside it is unreadable no matter what the
shell reports — so there is nothing to find and no reason to spawn `/bin/zsh` (which is
also why `HostEscapeAuditTests` needed a disposition entry for the new file).

The class guard is the one that matters: a source scan asserting no file outside
`LoginShellEnv.swift` reads an HF variable straight from `ProcessInfo`. That is exactly
how the bug got in, and it is what the next `HF_*` variable will trip over. Verified red
by injecting a single direct read into `CLIInstaller.swift`.

### The pill showed a commit hash, then kept showing the wrong model for a minute (2026-08-14)

Two reports about the composer's model picker, one cause each.

**A registry id is not a display name.** A model in the Hugging Face cache lives at
`models--<org>--<repo>/snapshots/<commit>/`, and nothing points the server at that root:
it reaches an HF model by absolute path, either `--model` at launch or `/v1/load-model`
on a hot switch. Both key the entry by DIRECTORY BASENAME (`ModelRegistry.registerByPath`
→ `std.fs.path.basename`), and for a snapshot dir that basename is the commit hash. So
`/v1/models` reports `9f0ea3c1d2`, and the pill — which preferred `chatModelInfo?.name`
over everything else, deliberately, because the resident model is the one answering —
rendered a hex string. The tray was fine and that is the tell: its rows come from our own
scan (`LocalModel.displayLabel`), which reads the repo id out of the cache dir name. The
fix is not to stop trusting the server: where the resident id names the model we PICKED
(its `name`, or its path's last component — the two shapes a registry id can take), our
label wins; where it names something else, the server's id stays, because a model another
surface loaded is not ours to rename.

**A hot switch is invisible from the outside.** It never moves `server.status` off
`.running` (the process doesn't restart), and `chatModelInfo` keeps reporting the OLD
model until the new one is resident — which on a 27B is a minute. The pill therefore sat
on the previous model's name with a green dot beside it while the menu's checkmark had
already moved to the new one, and nothing anywhere said a load was running.
`AppState.pendingModelLoadTask` knew, but it was private and unpublished. It is now
`@Published var loadingModelPath`, set beside that task and cleared in its `defer`, and
the pill names the model being loaded with a spinner in place of the dot. A RESTART is
deliberately not tracked there: that one does move the status, and
`ChatServerStartControl` already reports it.

Both answers live in one pure function (`ChatModelSelection.pillState`), next to the tag
rules, for the reason the tag rules are there: a second copy is how one picker starts
naming a model the other doesn't.


### The download bar filled up once per file (2026-08-14)

`DownloadState` carried two fractions: `progress` (bytes banked across the whole repo over
its total) and `fileProgress` (the file currently in flight). Every bar in the app —
the model pill's hairline, the browser rows, the welcome card, the starter card, the
bundle bar, the chat gate — rendered `fileProgress`, and so did `percentFormatted`. A
four-shard 18 GB pack therefore ran 0→100% four times, and the reports were all the same
shape: "it says 100% and starts again, I don't think it's working". The bigger the model,
the worse it reads, because more shards means more resets.

Nothing was wrong with the producer: `progress` was already live and correct, fed by
`(baseDownloaded + fileBytesTotal) / totalSize` on every transfer callback. It was simply
the field nobody rendered.

The fix is a deletion. Keeping both fractions in the struct and pointing the views at the
right one leaves the wrong one sitting there for the next surface to reach for — and eight
independent sites had already made exactly that mistake, which says the reflex is stronger
than the comment would be. With `fileProgress` gone the compiler names every site, and the
per-file detail is carried where it belongs: `currentFile` and `fileIndex`/`fileCount`,
words rather than a bar that appears to lose its place.

Two details the deletion had to preserve. `ChunkedFileDownloader.onProgress` reports bytes
*of this file including resumed ones*, so the callback's arithmetic is the transfer's true
position and needed no change. And the resume branch, which used to set only the per-file
number, now banks its existing bytes into `progress` — otherwise the bar sits at the last
completed file's mark for the fraction of a second before the first callback, which is not
wrong but is not where the disk already is.

Same file, same round, the other half of the same class: the hairline drew for ANY chat
transfer, so downloading a second model in the background put a progress bar under the
model that was already answering. Excluding media bundles had fixed the loudest instance of
that and left the general one, because both times the check was "is something downloading"
rather than "is THIS model downloading". Every other surface in the app keys its download
by its own repo id; the pill now does too, matching against `selectedModelPath` through the
layout the downloader writes (`<root>/<org>/<name>`, one level up for a `.gguf` file) since
a model still arriving has no `LocalModel` to match on. The one exception is what the
hairline was added for: with nothing chat-pickable on disk, the composer cannot answer at
all and whatever is arriving is the reason.

Media BUNDLES stay per-component: a bundle is N repos and the manager learns a repo's total
only when it starts, so a weighted bundle fraction would need sizes it doesn't have. That
bar resets once per model, and its label already says which one ("Downloading model 1/2").

## The abandoned connect() leak (MCP stdio connect race, 2026-08-15)

Symptom: user's RAG chat hung after 3-4 turns. Server idle and healthy (`/health` sub-2ms, `requests_running=0`, `last-agent-request.json` untouched 11+ min — rules out the server and the "ghost turn" class). App at ~160% CPU over 23h (spikes ~700%), 5326 threads. `sample <pid>`: two threads parked in `Client.connect(transport:)`, awaiting the stdio transport's `AsyncThrowingStream`. A completed connect RETURNS from that call — a thread still inside it after 23h never finished.

Root cause: `connectOrFailFast` races `client.connect(transport:)` (Path A) against a death-watcher (Path B, 10 Hz) and a 30s hard cap. The winner resumes the outer continuation; Path A's Task is abandoned, never cancelled ("the loser keeps running" — deliberate, to avoid `withThrowingTaskGroup`'s destructor hanging on an unresponsive child). The gap: nothing on the losing paths called `client.disconnect()`. Per the swift-sdk (`Client.swift:287-320`), `disconnect()` is the ONLY thing that resumes a pending `withCheckedThrowingContinuation` — cancelling the Task or killing the child does not unstick it. `executeToolCall`'s watchdog, a few hundred lines above, already documents and applies this exact break-glass for tool calls; the connect race never got the same treatment.

Each failed connect (server dead or no answer within 30s) leaks one permanently-scheduled task. They accumulate over the session, starve the cooperative thread pool, and the agent's next turn — which awaits a tool call on that same pool — can't complete. The spin and thread count are the leak; the hung chat is the starvation.

Fix: both losing paths call `client.disconnect()` (via `Task.detached` so it doesn't block the resume closure); the timeout path also terminates the child first. `connectOrFailFast` gained an injectable `hardCapSeconds` (default 30) and `onPathASettled` hook, and went `private`→`internal` for `@testable import`. Guard: `MCPConnectLeakTests.testAbandonedConnectSettlesAfterLosingTheRace` — real `sleep 60` child (accepts pipes, never speaks MCP, so the death-watcher never fires), 0.3s cap, asserts Path A settles within 5s.

Rule: any race that abandons one arm must ask whether the abandoned arm can unstick itself — here, only `disconnect()` does. A "loser keeps running" comment is a code smell: re-read it whenever a related watchdog is added nearby, in case the treatments have diverged (they did, for ~months, between the tool-call watchdog and this race). Diagnosis: `/health` + `requests_running` rule out the server in under a second; `sample <pid>` on the app is the fastest way to catch a CPU-spinning leak — a thread still inside a one-shot async call hours in is the tell.

## MLX Core crashed on every equation: a SwiftPM resource bundle no signed .app can hold (issue #233, 2026-08-20)

v26.8.9 shipped LaTeX rendering. A reporter on an M5 MacBook Pro could not launch the app
at all: it restored a chat containing math and died on the spot, `EXC_BREAKPOINT` in
`_assertionFailure` under `KaTeXFontProvider.makeUnitFont` → `Bundle.module`. Reproduced
here in seconds by asking a model to show a formula — the display-math path
(`MathCanvasContent.body`) trapped identically. It looked user-specific and was universal.

The first read was that their copy of the app was missing the font bundle, because the
shipped DMG has it: `Contents/Resources/SwaTex_SwaTexRender.bundle` is present, sealed in
`CodeResources`, `codesign -v --deep --strict` passes, and the same binary UUID on the same
OS build (26.5 25F71) resolved the fonts from a scratch Swift snippet. All true, and all
beside the point.

The accessor is what SwiftPM generates for a target with resources:

```swift
let mainPath = Bundle.main.bundleURL.appendingPathComponent("SwaTex_SwaTexRender.bundle").path
let buildPath = "/Users/runner/work/mlx-serve/mlx-serve/app/.build/.../SwaTex_SwaTexRender.bundle"
guard let bundle = Bundle(path: mainPath) ?? Bundle(path: buildPath) else { Swift.fatalError(...) }
```

For an app bundle `Bundle.main.bundleURL` is the **.app itself** (measured with a throwaway
.app: `bundleURL=/…/T.app`, `resourceURL=/…/T.app/Contents/Resources`), so it looks for
`MLX Core.app/SwaTex_SwaTexRender.bundle`. The second candidate is a CI runner's build
directory. Neither exists on any user's machine, and the miss is a `fatalError` rather than
a nil the caller can absorb — `makeUnitFont` itself degrades to a system font perfectly
well, it just never gets the chance.

The obvious fix does not exist. Copying the bundle to the .app root and re-signing:

```
MLX Core.app: unsealed contents present in the bundle root      # codesign -v --deep --strict, rc=1
MLX Core.app: rejected (unsealed contents present in the bundle root)   # spctl
```

Only `Contents/` may live in a bundle root, so the one place the accessor looks is the one
place nothing can ship. An Xcode-built app is fine because Xcode emits a different accessor
that also searches `Bundle.main.resourceURL`; every mlx-serve path — the DMG lane in
release.yml and the MAS lane in build.sh — uses `swift build`, so both were broken.

The fix patches the dependency's single call site before it is compiled
(`scripts/patch-swatex-font-lookup.sh`, wired into build.sh, release.yml and ci.yml ahead of
`swift build`): `Bundle.module.url(...)` becomes a candidate search over
`Bundle.main.resourceURL`, `Bundle.main.bundleURL`, the reading bundle's own two, and that
bundle's parent (the `swift test` layout, where the resource bundle is the .xctest's
sibling). It is idempotent, `chmod u+w`s the read-only checkout, and REFUSES rather than
silently no-opping when the upstream source stops matching — a patch that quietly fails to
apply ships a build that crashes exactly where it did before. Proof it works is a probe
binary placed inside a real assembled bundle, so `Bundle.main` is the .app:

```
KaTeX_Main-Regular -> /…/MLX Core.app/Contents/Resources/SwaTex_SwaTexRender.bundle/Fonts/KaTeX_Main-Regular.ttf
old Bundle.module path would be -> /…/MLX Core.app/SwaTex_SwaTexRender.bundle (MISSING -> fatalError)
```

App-side, `LaTeXFonts.isAvailable` asks the same question once and both entry points
(`InlineLaTeXRenderer.attributedAttachment`, `DisplayLaTeXRenderer.canRender`) decline when
it is false, so the segment renders as its exact source — the fallback malformed TeX already
takes. A missing resource must cost the math, not the process.

Two lessons worth more than the bug. **A packaging guard has to assert the destination the
code READS, not that a copy happens**: `testSwaTexFontBundleShipsInBothDeveloperIDPackagingPaths`
scanned build.sh for the string `SwaTex_SwaTexRender.bundle` and was green through every
crashing release, because the bundle really was being copied — just to a path nothing looks
in. And **"the artifact is correct" does not answer "the artifact works"**: the DMG passed
every structural check that could be run against it while being unable to render a single
equation.
