# Swift app: gotchas

One story per gotcha: the defect, the cause, the fix, the guard. The rules live in `app/CLAUDE.md`; what is still open
lives in `docs/known-issues.md`.

## Agent tools and weak models

### writeFile: the body crammed into the `append` flag, and an error that claimed truncation
Defect: gemma-4-12b wrote `{"append":"true,\n<whole body>","path":…}` five times, each rejected with "content was
truncated"; later `"true,"` overwrote instead of appending and `"true\",content:"` welded the key into the flag (empty
args at the server). Cause: `append` was typed as a string, inviting the weld; real `max_tokens` truncation is caught
upstream in the loop, so the executor's "cut off" was a lie. Fix: `normalizeWriteFileArgs` peels a leading flag word
into `content`; `appendFlagIsTrue` parses tolerantly; `append` is typed boolean (12 appends, zero welds); the error
names `content`. Only the loop, which knows `maxTokensHit`, may say "cut off". Reproduce from the failing model's
captured bytes, never a synthesized analogue. Guard: `AgentWriteFileRecoveryTests`, `WriteFileHandlerTests`.

### editFile: endLine without startLine, and an empty `Example: {}` steer
Defect: 12 identical retries of `{path, endLine, replace}` across three turns. Cause: `toolExample` matched the
literal `"Example: "`, so editFile's errors shipped `Example: {}`; the error restated the modes instead of naming the
gap. Fix: slice from the first `{` after any "Example"; the error says "you sent endLine=65 but no startLine" with a
concrete example; `parseLineNumber` reads `"45,"` as 45. Guard: `EditFileRecoveryTests`,
`testEveryToolDescriptionExampleIsExtractable`.

### `ServerOptions` defaults must mirror the Zig defaults
Defect: the app showed one value and the server ran another (`prefix-cache-entries`: Swift assumed 1, server 32,
filling 16 GB Macs). Cause: `toCLIArgs` omits a flag equal to an assumed server default. Fix: change both together;
memory-critical caps are always emitted (RAM-clamped). Guard: `testDefaultLaunchOmitsAllMatchDefaultFlags`.

### The hybrid prefix cache filled 16 GB Macs
See `engine-mlx.md` "The hybrid prefix cache retains far more than it reports": the app now always emits
`--prefix-cache-entries` via `ramCappedPrefixCacheEntries`.

## Windows, focus, SwiftUI

### An LSUIElement app must go `.regular` before it presents UI
Defect: windows and folder pickers came up half-focused until the user clicked. Cause: an inactive `.accessory` app
has no key window, and the policy flipped on `didBecomeKey`, which cannot fire. Fix: `AppActivation` sets `.regular`,
then activates, synchronously; every window and panel goes through it (a cancelled picker falls back to `.accessory`).
Guard: `AppActivationTests` (ordering + source audits for raw `openWindow(id:` / `panel.runModal()`).

### Two binaries in the app bundle
Defect: "still doesn't work" after a rebuild. Cause: `MLXCore` and `mlx-serve` both live in the bundle. Fix: update
them together.

### SwiftUI ToolbarItems whose content changes size break NSToolbar
Defect: enabling Agent mode pushed controls into a mangled `»` overflow until a resize. Cause: the bridge never
re-measures a ToolbarItem whose content grows; `ViewThatFits` and merged items fail the same way. Fix: dynamic strips
are a plain view row (`.safeAreaInset(edge: .top)`), density from the measured width
(`ChatMetrics.useCompactToggles`).

### Per-tab chat state is keyed by session id
Defect: "allow all tools" was forgotten across tabs, and every tab showed Stop while any chat generated. Cause:
`ChatDetailView` is reused across tabs, so a shared `@State` leaks and a reset-on-switch forgets. Fix: state keyed by
session (`SessionToolAllowList`), persisted toggles on `ChatSession` loaded in `.onAppear` and `.onChange(of:
sessionId)`, `composerState(for:)`. Guard: `PerSessionUIStateTests`.

### A parent `onTapGesture` swallows child Buttons on macOS
Defect: a session chip's ✕ did nothing. Cause: the container's tap gesture claims the child button's pixels. Fix:
side-by-side real Buttons, no gesture on the container.

### A legacy `.alert(item:)` on an ancestor shadows a descendant's alert
Defect: the close confirmation never appeared though its binding was set. Cause: legacy alerts on an ancestor swallow
a descendant's. Fix: one presentation path per window. Guard: `testSandboxWindowHasExactlyOneAlertPresentationPath`.

## Engines, tasks, voice

### Ghost turn on the shared ChatTurnEngine
Defect: every chat surface said "answering another chat" for 10+ minutes after a chat was deleted, surviving a server
restart. Cause: the deleted session's turn kept running and pad-retried against a missing session. Fix:
`deleteSession` stops an orphaned turn; the agent loop re-checks the session each iteration. When everything says busy
and nothing streams, check `last-agent-request.json`'s mtime first. Guard: `OrphanedTurnTests`.

### `TaskRun.summary` is a 280-char row cap
Defect: Telegram delivered task results truncated to 280 chars. Fix: full text via `fullLastAssistantText`; Telegram
splits at 4096 (`splitForTelegram`). Guard: the two task-result tests.

### Stale task runs persisted as `.running`
Defect: a quit mid-run left a forever-spinner. Fix: `reconcileStaleRuns` at start marks them failed (`.needsApproval`
kept); per-run Stop/Delete and "Clear finished".

### Launch-eager audio objects prompted for the microphone at launch
Defect: tccd prompted for the mic 1.9 s after launch, and on every ad-hoc rebuild. Cause: the always-constructed voice
controller built `AVSpeechSynthesizer` and `AVAudioEngine` graphs eagerly. Fix: build audio graphs on first use; idle
paths never create them. Diagnose with `/usr/bin/log show --predicate 'process == "tccd"'` (zsh's `log` builtin
shadows it). Guard: the lazy-init tests in `VoiceActivityTests`.

### A fixed-threshold VAD wedged on ambient noise
Defect: voice mode transcribed forever and never answered (one long recognition task, run-on partials). Cause: an
absolute RMS threshold; fan noise refreshed `lastVoiceAt` forever. Fix: threshold relative to a minimum-statistics
ambient floor, plus a longest-utterance cap (`maxUtterance`, server ASR has no partials to stall on). Guard:
`VoiceActivityTests`.

### Multi-input AVAssetWriter deadlocks if one track is pushed before the other
Defect: LTX with audio sat at "Encoding mp4…" forever. Cause: the writer applies backpressure across inputs, and audio
was appended after the video loop. A 3-frame toy clip fits under the window and passes. Fix: append the whole audio
track and finish it before the video loop. Guard: `testWriteMP4WithAudioDoesNotDeadlockAtRealisticScale` (97 frames).

### Browser tools, JSON quirks
Defect: (contracts) WKWebView work must run on the main thread (`BrowserManager` is `@MainActor`, created at launch);
`browse.readText` navigates first; `[String: Any]` key order is non-deterministic; tool-call `arguments` must be a
JSON string.

## Agent Sandbox (Virtualization.framework)

### The sandbox guest: entitlement, boot shape, tty traps
Defect: (design traps) the process needs `com.apple.security.virtualization` (ad-hoc builds too; plain `swift build`
and xctest cannot boot VMs: the live guard is `SANDBOX_SMOKE=1`); the rootfs mounts over virtiofs with no initramfs; a
cached kernel without virtiofs is detected and re-fetched. hvc1 is a real tty, so tools emit ANSI and progress:
`TerminalOutput.sanitize` must normalize `\r\n` before collapsing bare `\r` (else every line is wiped); Swift treats
`"\r\n"` as one grapheme, so `split(separator: "\n")` does not split (`split(whereSeparator: \.isNewline)`). The port
mirror binds both `127.0.0.1` and `::1` (`localhost` resolves to ::1 first). The served-app URL form is per
environment and lives only in the environment section; a base prompt that hardcoded `http://<local-ip>:<port>` beat it
and the agent handed out an unreachable LAN IP.

### A guest payload staged by only one packager broke sandboxed MCP for every other build (#89)
Defect: sandboxed MCP failed for every Developer ID user and worked in dev. Cause: `vz-agent` was copied into the
bundle only in the MAS branch, the release workflow never built it, and a dev-path fallback found `zig-out` locally.
Fix: every packager stages it and fails without it; the error names the missing half. Any build-produced runtime file
is staged and checked by every packager.

### Agent CLIs inside the sandbox ride ssh
Defect: (design traps) only the VM-owning process can open vsock, so TUIs ride dropbear baked into the image over a
dedicated forwarder. Without a devpts mount every PTY session dies while tty-less ssh works; the general forwarder
must skip port 22; the guest reaches the host via the per-boot NAT gateway (configs sed'd at boot, with the real API
key: guest traffic is not loopback); every re-pull changes dropbear's host key at the same port, so known_hosts is
app-owned and reset per boot.

### Sandbox terminals: an unmount is a SIGTERM; configure agents from their save path
Defect: switching to the Activity tab killed a live pi session; hermes ignored a config written from its docs. Cause:
a `switch` in `body` dismantled the terminal view; hermes' docs drift from its setup flow. Fix: live terminals stay
mounted in a ZStack (hidden by opacity); pre-seed a third-party CLI exactly as its setup flow's save code writes it,
pinned by a test.

### A renamed default strands upgraders on the old value
Defect: after the sandbox image was renamed by changing the struct default, existing users kept the old ref, and
factory reset and Re-pull both "did nothing". Fix: the base image is pinned (stored values ignored). Changing a
persisted setting's default needs a migration, or a pin. Guard: `testStoredBaseImageIsIgnoredImagePinned`.

### Enabling the sandbox left running MCP servers on the host
Defect: servers started before the toggle kept running on the host while the UI said "sandboxed"; disabling it killed
guest bridges that still looked connected. Cause: placement is decided at spawn and `startEnabled()` skips existing
sessions. Fix: `placementChanged` restarts running stdio sessions through the router. Anything chosen at spawn from a
mutable setting needs a subscriber that re-evaluates live instances.

### The sandbox gated only shell; file tools wrote to the host
Defect: with the sandbox on, `writeFile` wrote 23 KB to the host while `shell` was declined. Cause: routing lived only
in `ShellHandler`; `resolveAndConfine` returned paths unconfined when the working directory was nil. Fix:
`FileToolSandboxGate` on all five file tools; the working directory is mandatory in every mode; the default agent
workspace is a setting; a Settings workspace pick restarts pinned sessions in place (a stale ssh `onExit` must not
kill the replacement: identity check). A confinement toggle gates every tool that touches what it confines. Guard:
`FileToolSandboxGateTests`, `AgentWorkspaceDefaultTests`.

### Zig caches configure-time subprocess output
Defect: after a macOS upgrade removed the Command Line Tools, the link still searched the ghost CLT SDK. Cause:
`build.sh` forced `DEVELOPER_DIR` to the CLT, and Zig's build runner cached the old `xcrun` answer in `.zig-cache`.
Fix: prefer the CLT only when present; `rm -rf .zig-cache` after any toolchain or SDK change.

## Downloads and model presets

### Picking a voice backend did not download it, and its reused bundle made a complete download look incomplete
Defect: Kokoro in Settings ▸ Voice could not be fetched; reusing the TTS bundle (whose markers demand
`speech_tokenizer/`) would have read a full download as incomplete forever; moving Kokoro to the front made it the
default media model, which answers `ref_audio` with a 400; tray probes called it ready once `config.json` landed. Fix:
a preset's declared capability picks its bundle (`MediaBundle.kokoro`), the media catalog lists only cloning-capable
presets, readiness reads the bundle's ready markers, never one file. Guard: the Kokoro bundle and catalog tests.

### Parallel chunked downloads: the `.partial` size stops being the resume cursor
Defect: (design) N ranged connections write a sparse `.partial` whose size is its highest offset, so the old "size is
how far we got" resume would skip everything between. Fix: per-chunk progress in a `.partial.parts` sidecar written
after the bytes (may under-report, never over-report); `pwrite` per connection; a `200` to a ranged chunk, a
contradicting `Content-Range` total, or a short clean chunk each void the plan; one shared `URLSession` with more
connections per host (the default 6 queued chunks 7-8; the shared session was the real win, 1.71x on small files;
chunking is insurance on single-stream-limited links); the HF token read from `$HF_HOME/token` and
`~/.cache/huggingface/token` (a Finder-launched app has no shell env). Every entry point funnels through
`transferFile`, source-audited. Guard: `ChunkedDownloadTests` (assembled bytes), `DownloadManagerTransferTests`.

### Agents (personas): one resolution, five surfaces
Defect: (design) five surfaces start turns, each with its own reads of settings. Fix: `AgentResolution.resolve` +
`TurnConfig.from` are the only builders. Traps: sampling defaults differ per path (the tool loop's 0.7 vs chat's
default), so overrides travel raw; the persona sits in the stable prompt prefix (a volatile placement re-prefills
every message); tool filtering keeps the hand-ordered literal byte for byte, and a disallowed tool called anyway is
refused at dispatch; a pinned model never downloads silently; an agent pick is an explicit workspace switch but never
rewrites the default-workspace setting; wake words match longest first, collisions refused at save by the last word;
each agent has its own chat thread (`sessionForAgent`). Guard: the `Agent*Tests` family, `WakeWordMultiAgentTests`.

### The composer's Think/Tools/MCP discs disagreed with the turn
Defect: selecting any agent ran the tool loop while the Tools disc showed OFF (`web` defaults true, so every agent has
tools); clicks wrote fields the resolver ignored. Fix: the discs read `resolvedAgentSettings` and lock per control
(thinking may stay unset); a locked disc still shows ON/OFF and opens "Set by <name>"; the agent is chosen at chat
creation only (`setAgent` deleted). A menu opening over the composer delivers no hover-exit: dismiss on
`NSMenu.didBeginTrackingNotification` and bump the reveal token. Guard: `ChatModeTogglesTests`, `ComposerTipTests`.

### First run: a welcome screen linking to fourteen models, and a chat window that dead-ended
Defect: a new user faced a taxonomy decision, no server ran, and Chat with no model was a dead end; the blocking
sheet's Cancel did nothing. Cause: `autoStartServer` read an unset key as false; AppKit refuses to close a window with
an attached sheet, so close must end the sheet first. Fix: one RAM-tiered `starterPick` rendered by one shared card
that selects, starts and opens chat; Cancel ends the sheet, then closes the window; auto-start defaults on. Speed bars
are our measurements (decode is bandwidth-bound on active parameters), not a cloud leaderboard's.

### The shipped app advertised a version the release did not have, so it nagged forever
Defect: the DMG under `v26.8.1` reported `26.7.12`, so the update prompt returned forever. Cause: CalVer was computed
in UTC after local midnight on a month boundary, and the workflow copied `Info.plist` verbatim while the CLI baked the
computed version. Fix: stamp the bundle copy before codesign; pin the CalVer timezone in CI and `app/build.sh`. Guard:
`tests/test_release_workflow_gates.sh`.

### The workaround the app could not reach (`--max-resident-mem`)
Defect: every refused load said "raise `--max-resident-mem`", and the app could not emit it. `--skip-mem-preflight`
runs after the registry gate and does not help. Fix: a slider (0 = Auto) snapping to presets capped at physical RAM:
delete the input rather than validate a free-text grammar the server `exit(1)`s on. A new `ServerOptions` field needs
a `SettingsReset` section (`testEveryServerOptionsFieldBelongsToExactlyOneSection`).

### `mlx-serve` survived the app quitting it (#133)
Defect: ⌘Q left the server running with a model loaded; only the tray's power button stopped it. Cause:
`ServerManager.deinit` never runs at termination (a `@StateObject` is not deallocated). Fix: `ServerManager` observes
`willTerminateNotification` with `queue: nil` (a queued block may never run during termination). Guard: the test posts
the notification and asserts stopped with no runloop turn.

## Chat window

### The truncation banner rode history as assistant prose
Defect: after one repetition cut, every later request carried "⚠️ Stopped — the model started repeating itself…" as
something the assistant said, in a chat already primed to repeat. Cause: the notice was appended into
`message.content`, which history builders resend. Fix: `ChatMessage.truncationNotice` is a field drawn as a footnote;
builders strip the legacy text from old sessions (markers derived from the legacy string). Our own UI text must never
become model input. Guard: `TruncationNoticeTests`.

### The ⌘ quick-switch badges stopped where the small window had ended
Defect: in fullscreen the numbered sidebar rows did not track the window. Cause: a `.global` frame is not re-published
when a view merely moves; a second "fallback" reader was always wrong (104 vs 370) and hidden by a `max`. Fix: one
named coordinate space on the column, all three measurements in it; the filter moved into `ChatQuickSwitch.numbering`.
Guard: `ChatQuickSwitchTests`.

### Continuing a reply filed itself as a new version
Defect: continuing a regenerated reply grew its pager from 2/2 to 3/3; a continuation's usage replaced the reply's
token count. Cause: the one turn exit could not tell a continuation from a regeneration; `continueReply` opens with
`stop()`, itself a turn exit, so a mark set before it is consumed at once. Fix: `markContinuing` set after `stop`,
usage added (`addingCompletionTokens:`), and Continue is offered only where the engine can continue (ds4 cannot).
Guard: `ContinuedReplyBookkeepingTests`, `ContinueReplyTests`.

### A typed `onDrop(of:)` never saw the Create panes' drops
Defect: a dropped `.txt` lit the pane and vanished; typing the target as `.image` then refused every file. Cause: a
Finder drag carries only `public.file-url`; filtering after resolve happens after the drop was accepted. Fix: register
`.fileURL` and refuse in `DropDelegate.validateDrop`, reading the drag pasteboard's URLs against the pane's extension
list; a drag with no readable URLs falls back to accept-then-filter. Guard: `MediaDropTests`.

### The Hugging Face cache was "supported" and never worked
Defect: `HF_HOME` and `HF_HUB_CACHE` were honoured only when the app was run from a terminal. Cause: a Finder-launched
bundle has none of the shell environment; the token file and `PATH` had each solved this separately. Fix:
`LoginShellEnv` is the one probe (marker-delimited values, unset omitted, process env wins, off-main with a 5 s
watchdog, a configured-but-missing root is nil, `XDG_CACHE_HOME` added; gated off under MAS). Guard: a source scan
that no file outside `LoginShellEnv.swift` reads an HF variable from `ProcessInfo`.

### The model pill showed a commit hash, then the wrong model for a minute
Defect: an HF-cache model's pill read a hex string; after a hot switch it kept the old name for a minute. Cause: the
registry keys by directory basename (a snapshot dir is the commit); a hot switch never leaves `.running`. Fix: our
label wins where the resident id names the model we picked; `@Published loadingModelPath` shows a spinner; both in one
pure `ChatModelSelection.pillState`.

### The download bar filled up once per file
Defect: a four-shard pack ran 0 → 100% four times ("it says 100% and starts again"). Cause: eight surfaces rendered
the per-file fraction. Fix: delete `fileProgress`, so the compiler names every site; per-file detail is words
(`fileIndex`/`fileCount`); resume banks existing bytes; the pill shows a download only for its own model.

### The abandoned `connect()` leak (MCP stdio connect race)
Defect: RAG chats hung after 3-4 turns; the app sat at 160% CPU with 5326 threads. Cause: `connectOrFailFast`
abandoned the losing `client.connect` arm, and only `disconnect()` resumes its continuation, so every failed connect
leaked a scheduled task until the pool starved. Fix: both losing paths call `client.disconnect()`. A "loser keeps
running" comment is a smell; `sample <pid>` catches a thread still inside a one-shot call hours later. Guard:
`MCPConnectLeakTests`.

### MLX Core crashed on every equation: a SwiftPM resource bundle no signed .app can hold (#233)
Defect: any chat with math killed the app (`fatalError` in `Bundle.module`). Cause: SwiftPM's accessor looks at the
.app root and a CI build path; only `Contents/` may live in a bundle root (codesign rejects anything else); Xcode
emits a different accessor. Fix: `scripts/patch-swatex-font-lookup.sh` rewrites the dependency's call site before
`swift build` (idempotent, refuses when upstream changes); `LaTeXFonts.isAvailable` renders source when fonts are
missing. A packaging guard asserts the path the code reads, not that a copy happens.

### A dropped frame is a failed mux (#170)
Defect: long H3 renders intermittently produced a ~28 KB mp4 of black frames reported as success. Cause: a nil pixel
buffer was skipped with `continue` and `adaptor.append`'s Bool discarded. Fix: either is a failed mux, surfaced as an
error.

### Sandbox terminals moved into the chat sidebar; closing the window killed them
Defect: closing the sandbox window SIGTERM'd live TUIs (`dismantleNSView` called `terminate()`). Fix: the process is
owned by `EmbeddedTerminalView.Handle` held in `TerminalSessionStore` on `AppState`; views attach and detach.

### Every quantized repo read 4x too big in the Model Browser
Defect: gemma-4-e2b-4bit (3.55 GB) read 18.1 GB. Cause: Hugging Face changed `safetensors.parameters` so U32 counts
logical elements, not packed words. Fix: price packed elements by the width the repo id names, `(bits + 0.5) / 8`
bytes; no width → the tree-API fallback. Metadata from a third party is a contract it can change: keep one live test
per external contract (`HFSearchIntegrationTests`).

### The chat column became a setting, and the transcript was laid out around it (PR #339)
Defect: prose, tables and bubbles had three different edges in one column. Fix: every element takes the user's column
width. Traps on the way: a paragraph indent cannot reach an `NSTextTableBlock`; table column shares by content length
rendered "Life spa n" (sqrt-compressed, floored by the longest word); a list regex that matched any digit and `. `;
inline code detected by font trait missed the monospaced system font; the render cache keyed without leading; tool
cards rendered a truncated summary string; `bg1` is reassigned every launch, so process ownership is the transcript's
last announcement scoped to the chat; ⌃ shortcuts steal keys from embedded terminals (interface shortcuts carry ⌘).

### One checkbox started a server and loaded 26 GB (#214)
Defect: "Auto-start on launch" loaded the selected checkpoint at every login and made it the registry default. Fix:
`StartupModelChoice.launch` (`.doNothing`/`.headless`/`.load`): auto-start alone is headless; "Load a model at start"
is a separate opt-in with its own mode key. Guard: `StartupModelChoiceTests`.

### The transcript rendered blank until the mouse moved
Defect: after a chat switch or a large height change the transcript showed white space and filled in from the bottom
on mouse moves. Cause: a `LazyVStack`'s height is an estimate that changed hundreds of times on long chats. Fix: a
`VStack` over a window of the last 150 rows (an index fixed when the chat opens), open-at-end by layout
(`defaultScrollAnchor`), a shrinking row keeps the clicked control in place, and a negative bottom distance during a
shrink is not arrival. A `mask` renders an offscreen layer as tall as the content: the fade is an overlay.

### Audio clips rode the history as float32
Defect: a minute of speech was 5 MB of base64 inside `chat-history.json`, rewritten every save. Fix: `AudioClipFile`
stores a 16 kHz mono 16-bit WAV under attachments, as images already were.

### An interrupted turn had no footer, and the trash under a reply took one row of a dozen
Defect: a turn stopped mid-thought ended on a bare thinking block with no footer; deleting an agent reply left its
tool rounds standing, which the model read back as calls with no answer. Cause: "there is no turn in this app":
nothing named the segment a reader means by "this reply". Fix: `ChatTurn`, one seam beside `ChatFork`, sharing its
boundary rule (a message the model could be handed back; a row carrying media is a boundary, or a fork trimmed the
generated picture).

### The markdown a model writes, and the code tint that ran to the margin
Defect: indented list items, `---`, checklists and `~~strike~~` rendered flat; a wrapped inline code span tinted to
the right margin. Cause: an item's continuation joined with `\n` starts a new paragraph in TextKit (U+2028 is a break
inside one); `enumerateEnclosingRects` answers selection geometry. Fix: indent-stop depth (`ListDepth`), table-cell
rules, the strikethrough intent, `InlineCodeGround.rects` over line fragments. A headless layout test must hold the
`NSTextStorage` (the back reference is weak). Guard: `MarkdownListTests`, `InlineCodeGroundTests`.

### Headless browser: `outerWidth` 0
Defect: pages driven by the agent's browse tools read `window.outerWidth` as 0, a bot signal, even in a hidden window;
the URL bar stopped following the page; `localhost:3000` was unreachable. Cause: WebKit returns an empty window frame
unless the UI delegate implements the private `_webView:getWindowFrameWithCompletionHandler:`; `navigate()` replaced
the coordinator's delegate per call and forced `https://`. Fix: `WindowFrameUIDelegate` (compiled out under MAS), one
persistent navigation delegate with KVO on the URL, `resolveURL` per target.

### The Create panes relaid model-first, and four SwiftUI traps
Defect: (traps) `.menuStyle(.borderlessButton)` drops a chip label's background (`.button` keeps it); a drag handle
below its box must measure in `.global` (in `.local` it chases the pointer at half speed); a focused number box that
refuses every bound change also refuses a value picked from a menu (`NumberFieldRepaint.shouldRepaint` compares with
what the box itself would read); `.frame(maxWidth: .infinity)` centres without an alignment and a `LazyVGrid` spreads
cells into equal columns (`FlowLayout` packs at own widths).

### SwiftUI layout traps in the panes
Defect: (traps, one fix each) a clipped `.fill` image still takes the pointer beyond its clip (`contentShape` beside
the clip); an overlay is proposed the size of the view it sits on, so a preview capped itself at the tile (explicit
fitted frame); `onGeometryChange` fires whenever its value changes, so return the answer (a Bool, an Int, a quantised
rect) and never measure what depends on the answer; a segmented control's narrowest width flickers, so `ViewThatFits`
flipped it into a menu while scrolling (`SegmentFloors` keeps the smallest floor per label set); a stack that will not
compress is centred by its `ScrollView` (a full-width leading frame with the padding inside). Guards:
`ReferencePreviewTests`, `SegmentedOrMenuTests`.

### Media panes: a truncated mixed drop, a rounded-up canvas, a draft not in the settings blob
Defect: dropping fourteen files landed nothing (the loader took the first `limit`, all duplicates); the video canvas
snapped 460 to 512 though the video handlers refuse off-grid sizes rather than round; the Video pane lost its prompt,
anchors and references on a trip to Chat and four knobs on every launch. Fix: the mixed slot hands the router
everything (`MediaDrop.deliverable`) and every way in dedupes; `ResolutionGrid.Rounding` per grid (`.nearest` for
video); the whole pane state is one saved `stickySnapshot`, files as paths checked on hydrate. Guard:
`MediaDropTests`, `CustomResolutionTests`, `MediaGenSettingsTests`.

### The UI process pegged a core while a reply streamed
Defect: MLXCore sat near 100% of a core during any generation. Cause: a per-frame `TimelineView` re-rendered the
window's hosting view; the `App` struct observed `AppState` and rebuilt every scene per batch; one growing reasoning
`Text` was re-measured whole per batch; every row re-rendered and re-ran LaTeX and markdown. Fix: `CAShapeLayer`
activity rings, the App holds its objects and small views observe, `StreamingLines`, an `Equatable` `MessageBubble`,
cached inline rendering: 74% → ~45%. Owed: a per-message observable for the streaming text. Guard:
`MessageBubbleEquatableTests`, `StreamingLinesTests`.

### A semantic font is a name, not a size
Defect: a percentage Text Size moved the chat transcript and nothing else. Cause: macOS has no dynamic type:
`.dynamicTypeSize` does not move a semantic font (measured with `ImageRenderer`); tests asserted the mapping, never a
rendered size. Fix: `AppType`, a ladder of macOS's own sizes; `Font.app(_:)` is the only way a view states a size.
Guard: `SystemTypeTests`; the console keeps the same rule in `rem` (`app-web/test/css.test.ts`).

### A control's title is not your text
Defect: 183 button titles ignored the type ladder. Cause: `.font()` does not reach a Button's title on macOS, nor a
menu, automatic or segmented Picker's. Fix: the title as a label view (`Button { … } label: { Text(…).font(.app(…))
}`). Swift will not mix a labelled `action:` with a trailing `label:`.

### A guard that scans too much passes for the wrong reason
Defect: 62 bare `Text`s sat in main with the font guard green. Cause: a sibling's `.app(…)` anywhere in the block
vouched for them. Fix: inheritance is read only where the trailing-modifier idiom puts it; a probe in a previously
waved-through view now fails.

### Deleting a provider crashed Settings
Defect: deleting the last Providers row crashed (`Array._checkSubscript`); on other rows it cleared the next
provider's text. Cause: a `ForEach($array) { $entry in }` binding reads by index, and the closure read `entry.id`
after the removal. Fix: capture the id when the closure is built (`[id = entry.id]`).

### Full-size pictures decoded in a view body stall the window
Defect: window moves and typing stuttered with media on screen. Cause: full-size `NSImage`s decoded on the main
thread, some in `body`. Fix: `MediaImage` downsamples off the main thread with a cache. Guard: `MediaImageTests`.

### A material background fills the safe areas
Defect: the AI Radio lock banner showed under the sidebar. Cause: `.background(.regularMaterial)` (a shape style)
fills the safe areas. Fix: draw the material `in:` a shape inside the column. Guard: `RadioLockBannerTests`.

### Composer note recovery must wait for the IME
Defect: recovering a steering note during marked (IME) text lost it. Cause: the redraw guard refused to overwrite the
composition and the commit overwrote the binding. Fix: paused notes in a session-keyed store, restored after
composition ends. Guard: `SteeringNotesTests`, `ComposerMarkedTextTests`.

### Qwen-Image infographics came back as gibberish
Defect: every infographic's small lettering was garbled. Cause: the prompt named no exact words, so the model invented
labels; Enhance asked for "one or two sentences" and its cleanup trimmed a closing quote. Fix: the format of Qwen's
own prompt enhancer (every string quoted with its place, "There is no other text in the image") in templates and
Enhance; our tokenizer was innocent.

### A placeholder over a `TextEditor` ran under its scroller
Defect: with legacy scrollers the prompt placeholder ran under the scroll bar. Fix: `.clearOfEditorScroller()` pads by
the scroller width and follows style changes. Guard: `EditorScrollerInsetTests`.

### A nested shard the index names was never downloaded
Defect: a pack finished downloading and read "Missing shards". Cause: the downloaders kept nested files only from a
hand-kept allowlist, while `weightDefect` reads the index. Fix: the root index is fetched first and every file it
names rides along at any depth (`indexedShards`); nested copies no index names stay out.
Defect: Qwen-Image-2.1 garbled the lettering of every infographic, while the title and headline numbers came out right. Cause: the prompt asked for "icons and statistics" without giving their words, so the model invented filler labels, and those render as gibberish; our tokenizer matched Hugging Face's token for token. Enhance made it worse: it asked the chat model for "one or two sentences", which drops the exact words, and `clean` trimmed every quote off both ends, cutting the closing quote of a prompt that ended on quoted text. Fix: the format of Qwen's own prompt enhancer (every string in double quotes with where it sits, exact words for every label, "There is no other text in the image"), shipped as Qwen's Infographics templates and as Enhance's layout rules for a `rendersDenseText` model; `clean` strips only a wrapping pair. Same seed and size, the quoted layout rendered every word. Guards: `PromptRewriterTests`, `testQwenImageOffersInfographicTemplates`.

## A placeholder over a `TextEditor` ran under its scroller (2026-10-13)
With "Show scroll bars: Always" (or a mouse attached), the Video prompt's placeholder ran under the editor's scroll bar: "Click" sat on the track. A legacy scroller takes 17 pt from the column the editor's text wraps in; the placeholder is a `Text` laid over the editor at full width. Fix: `.clearOfEditorScroller()` pads it by `EditorScrollerInset.trailing(for: NSScroller.preferredScrollerStyle)` and follows the style-change notification; every laid-over text uses it (Video prompt, Music lyrics and score, New Task goal, the Enhance sheet's thought). Guard: `EditorScrollerInsetTests`.

## A nested shard the index names was never downloaded, so the folder read "Missing shards" (2026-10-16)
Defect: `mlx-community/humanizer-12b-OptiQ-4bit` finished downloading and the Models list flagged it "Missing shards". Cause: its `model.safetensors.index.json` names `optiq/optiq_vision.safetensors`, and the downloader (app and `mlx-serve pull`) keeps nested files only from a hand-kept allowlist (`mtp/`, `drafter/`, `engram/`, `optiq/mtp.safetensors`), so each new pack layout was a new miss. `weightDefect` reads the index, so the two disagreed by construction. Fix: the root index is fetched first and any file it names rides along at any depth (`indexedShards`, `Resolved.indexed`); a nested copy no index names (`original/`) stays out. The server only loads top-level shards, so this pack's vision tower is still not used. Guards: `testSelectNeededFilesPullsEveryShardTheIndexNames`, `cli: a nested shard the index names is part of the pack`.

## An exited agent shell command kept a core per pipe busy until the app quit (2026-10-25)
Defect: with a model loaded and no requests, `MLXCore` burned CPU while `mlx-serve` idled. Cause: the `readabilityHandler` on each agent shell process's stdout and stderr pipe never cleared itself. Once the child exits, Foundation re-fires the handler without pause with empty data; `ManagedProcess` retains the `Process`, so the pipes outlive the exit, and only `kill()` called `detachCapture` (it returns early for an exited process, so the card's kill X could not help). Fix: `ProcessRegistry.capture(into:)` sets `readabilityHandler = nil` when `availableData` is empty, which on a pipe is EOF and the last callback, so trailing output still lands first; `terminationHandler` is the wrong place, it can run before the final read. `ClusterWorker` and `ServerManager` carry the same check in their stderr readers. Guard: `ProcessCaptureTests`.
