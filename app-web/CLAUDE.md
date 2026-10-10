# app-web — Studio console source

Svelte 5 (runes) + TypeScript, Vite 8, Vitest + happy-dom. `npm run build` writes the ONE committed file `src/html/index.html`. Never hand-edit that file: change the source here, rebuild, commit both.

## Layout

- `src/lib/core/` pure logic (HTTP client, SSE, request builders, tool loop, library, metrics). DOM-free.
- `src/lib/state/` `$state` classes in `*.svelte.ts`: `Connection`, preferences, router, and one workspace per pane (`ChatWorkspace`, `ImageWorkspace`, `AudioWorkspace`, `VideoWorkspace`, `MonitorWorkspace`) holding that pane's drafts and generation; `MediaRun` is the shared run controller.
- `src/lib/{markdown,icons,mp4,i18n}` first-party parser (AST), icons, MP4 muxer, language store + zh-Hans table.
- `src/panes/*.svelte` one pane each, thin: they draw a workspace. `src/components/` shared pieces. `src/lib/app.svelte.ts` builds the `App` they share.

## Rules

- **No raw HTML from data, ever.** No `{@html}`, `innerHTML`, `outerHTML` or `insertAdjacentHTML` with model, server or user text: there is no CSP behind it. Markdown renders from an AST to elements, links are http(s) only (`rel="noopener noreferrer"`), images show as alt text. Guard: `test/hostile.test.ts` pushes a hostile corpus through every text surface (`test/support/inert.ts`).
- **Same persisted keys.** localStorage `studio.interface|servers|activeServer|activeChat`, `mlx-serve-lang`, and the IndexedDB databases `mlx-serve-studio` (v1, store `items`), `mlx-serve-studio-{image,audio,video}-drafts`, `studio.monitor` are a user-data contract; renaming one orphans a Library or a preference.
- **`t("English")` is keyed by the exact English string** (`src/lib/i18n/zh-hans.ts`). Mark a string translated later with `N("…")`. Tests fail on a marked string with no entry AND on an entry nothing uses.
- **Sizes in `rem`, never `px`** (`test/css.test.ts`). Colors through the custom properties in `app.css`.
- **The generation forms are `novalidate`**: the pane validates itself, and native validation silently blocks Generate over a control inside a closed section. Bind a select with `<select value={…}>`, not `selected` per option.
- **Tests live here**, next to the code they cover; there is no UI test under `tests/`. Pure logic: `test/*.test.ts`; rune classes: `*.svelte.test.ts`; whole panes: `mountApp()` (`test/support/mount.ts`) against `mockApi()`. A browser API happy-dom lacks (SpeechRecognition, WebCodecs, MediaRecorder, IndexedDB) sits behind a seam and is faked: `App(library, drafts)`, `VideoController.request/encode`.
- **Behavior the server and tools depend on**: ONE media generation attempt per user turn (a budget, not a round cap); the edit tool's model enum lists exactly the edit-capable models; every request goes to the selected server under the page's mount prefix (`apiPrefix`); metrics are "on" when `GET /metrics.json` is not 503.
- **Size budget**: `test/size.test.ts` builds in memory and fails above its ratchet (lower it when the page shrinks); the page stays one file with no outside reference.
- Dev servers bind `localhost` (IPv6 on macOS): `curl http://localhost:5173`, not `127.0.0.1`. In a happy-dom test, submit a form with a `submit` event, not `requestSubmit()` (its URL validation is stricter than a browser's).
