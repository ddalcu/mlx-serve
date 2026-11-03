# Add an API field

1. Parse it where its surface parses (`src/host/api/`: chat_completions, completions, anthropic, responses, ollama);
   a field every surface takes is read once and passed down (`sampling.zig`, `submit.zig`).
2. A field that changes dispatch is readable from every body shape (JSON and multipart).
3. Clamp request integers, escape every string at the sink, and refuse an unhonored value with a named 400.
4. Stream and non-stream deliver the same bytes: test both.
5. Unit test in the file, plus an integration script in `tests/` when the field is HTTP-observable; a line in
   `docs/api.md`.
