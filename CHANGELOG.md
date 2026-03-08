# Changelog

## [0.0.2] (unreleased)

### Added

- **Prompt cache**: KV (and SSM for hybrid/MoE) state is cached after each request; follow-up requests whose prompt is an exact prefix of the previous one reuse the cache and only prefill the new suffix. Logs `prompt cache hit: N/M tokens` when used.
- **TUI status bar metrics**: Background thread samples system stats; status bar now shows RSS (M/G), Mem %, CPU %, and GPU % (right-aligned).

### Changed

- **Build**: Link IOKit and CoreFoundation on macOS (required for status bar metrics).
- **Streaming**: Decoded token bytes are buffered until complete UTF-8 codepoints; fixes split multi-byte characters (e.g. emojis) across tokens with byte-level BPE. Thinking/reasoning chunks also flush on valid UTF-8 boundaries only.
- **MoE / recurrence**: Prefill evaluates MoE layers every 4 layers (was 48). SSM recurrence state is evaluated every 32 steps to limit graph size.

### Fixed

- Streaming and thinking output no longer split multi-byte UTF-8 characters mid-sequence.

---

## [0.0.1]

### Added

- **Jinja_cpp**: Chat templates rendered via lib/jinja_cpp (replaces vibe-based Jinja; fixes infinite loop with macros).
- **Qwen3.5**: Support for `qwen3_5`, `qwen3_5_moe`, and related model types; ChatML with optional Jinja2 `chat_template.jinja`.
- **TUI status bar**: Optional status line (idle/prefill/decode) when running the HTTP server.
- **Streaming**: `POST /v1/chat/completions` with `"stream": true` returns SSE chunks (OpenAI-compatible).
- **Thinking tags**: Models that expose reasoning (e.g. `<think>...</think>`) can stream and return `reasoning_content`; strip/filter handled in chat and API responses.

### Changed

- Chat template loading: tokenizer_config.json plus fallback to `chat_template.jinja` for Qwen3.5-style configs.
- Server and generation paths updated for streaming and thinking-aware formatting.

### Dependencies

- build.zig.zon: added jinja_cpp (lib/jinja_cpp with nlohmann/json, ujson).
