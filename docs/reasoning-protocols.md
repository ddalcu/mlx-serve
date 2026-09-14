# Reasoning with constrained JSON

The MLX generation path uses one bounded protocol state machine followed by the
existing JSON grammar. Format differences are delimiter/header descriptions;
HTTP handlers share one output router. Selection uses the actual chat template,
rendered prompt tail, and tokenizer capabilities, never checkpoint names.

| Format | Reasoning | Final entry |
| --- | --- | --- |
| Bare think tags | `<think>…</think>` | JSON after the matching close |
| Suffixed think tags | `<think:S>…</think:S>` | Same suffix required |
| Gemma | `<|channel>thought…<channel|>` | Direct JSON or `<|channel>\n` |
| Inkling | `<|content_thinking|>…<|end_message|>` | Model message with `<|content_text|>` |
| Harmony | Assistant `analysis` channel, ending in `<|end|>` | Assistant `final`/`commentary` header |
| Muse | Assistant `to=self`, ending in `<|eom|>` | Assistant `to=user` or unaddressed message |

Prompt-opened and generated headers are supported, including repeated channel
reasoning segments. While the channel is unresolved, masks admit the supported
reasoning and final alternatives. Once JSON starts, all bytes belong to the schema,
even when string data looks like a reasoning or tool marker. Unsupported/dynamic
headers retain the thinking-off fallback; this is a bounded set of safe protocol
spellings, not a permissive parser for arbitrary headers or tool recipients.

Tokenizer candidate indexes and exact recovery encodings are cached once per
marker on the loaded model. Ordinary reasoning tokens need no grammar snapshot;
only candidates crossing into JSON are trial-validated. Invalid crossing tokens
are masked before sampling, including tokens that carry a closer, final header,
and JSON together. A constrained protocol failure never disables the grammar.

`max_tokens` counts reasoning, structural headers, and JSON together. Exhausting
it during reasoning can produce empty content with a length finish reason. There
is no answer reserve. Early loop/EOS recovery requires room for the complete
remaining transition and at least one answer token. Finite response-side reasoning
budgets keep the existing thinking-off fallback. Tools still bypass the schema
path; constrained generation still disables speculation.

Validation: protocol tests cover all six formats, candidate rejection, every
two-token split of representative transitions, partial-header recovery, special
IDs, direct answers, thinking-off entry, UTF-8 carry, and clipped payload spans.
The HTTP regression runs Qwen on Chat Completions, Messages, and Responses, with
both streaming modes and thinking enabled/disabled (12 combinations). The other formats currently have deterministic coverage;
they still need live validation against installed checkpoints before performance
or model-specific compatibility claims can be made.
