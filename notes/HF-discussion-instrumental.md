**Title:** Has anyone gotten a consistently vocal-free track out of the open weights?

---

We run MiniMax-Music3 on a native Metal/MLX engine ([mlx-serve](https://github.com/ddalcu/mlx-serve)) and it sounds great. We cannot get it to stop singing.

The hosted API has `is_instrumental`. The open weights have no such parameter — `scripts/end_to_end/minimax_ttm_test.py` posts only `input` and `instructions` — so whatever it does has to be expressible in those two text fields.

**What we tried.** Same seed and instructions each time, on the released weights at 8-bit:

| `input` (lyrics) | `instructions` | result |
|---|---|---|
| `[Instrumental]` | prompt | wordless vocalizations |
| `[Instrumental]` | prompt + "Instrumental only: no vocals, no singing, no lyrics." | wordless vocalizations |
| `[Inst]` | prompt + same clause | wordless vocalizations |
| **empty** (`[start]` alone) | prompt + same clause | wordless vocalizations |

The `instructions` already opened with **"Instrumental ambient field-recording piece, freely paced, no vocals."** — so the request said "no vocals" three ways at once, in the genre, in an explicit clause, and via the lyric tag.

No intelligible lyrics in any take, but clear sung vocal texture throughout, and one opened with an audible word. The empty-lyrics row is the interesting one: with no lyric text at all, it still sings.

One partial result worth mentioning: a take at **BPM 167** came out genuinely vocal-free, where the same prompt at 60 BPM did not. Single sample, seed not controlled, so we are not claiming it.

**Ruled out already**, so nobody has to suggest it:

- No instrumental special token — the 32 added tokens are `<|caption_start/end|>`, `<|lyrics_start/end|>`, `<|audio_start/end|>`, `<|audio_cfg|>` and stock Qwen3 markers.
- `tokenizer/chat_template.jinja` in the release is the stock Qwen3 tool-calling template, no music path.
- Our prompt assembly matches the reference, and the engine matches fp32 at prefill cos 0.9999 / DiT velocity 0.999 / vocoder 1.000000, so this is not a numerics bug on our side.

**Two questions:**

1. Has anyone gotten **consistently** instrumental output — no vocalizations at all — from these weights? If so, what did your `input` and `instructions` look like?
2. Does `is_instrumental` map to something in those two fields at all, or does it select a different conditioning path / checkpoint that is not in this release? If it is text, what does the serving stack put there?

Happy to run any prompt shape you want and post the audio.
