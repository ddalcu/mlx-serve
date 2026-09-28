# mlx-serve decisions (Laya, Kev): `POST /v1/decisions`

A decision model (capability `decisions`) answers typed questions about a STATE
with probabilities instead of free text. Use it for game logic an LLM is too
slow or too unpredictable for: NPC intent, dialogue routing, moderation, "is
the player stuck", "which quest fits this situation", difficulty scoring. It
never generates text; pair it with a chat model when you also need words.

Two families take the same request:

- **Laya** (model type `laya`, ~0.35 GB): a few milliseconds per request.
  Pick it when you call often and speed matters most.
- **Kev** (model type `kev`, ~4 GB): about 60 ms per question on an M4 Max,
  and often more accurate on nuanced text (on one 2,394-headline news-labeling
  test: Laya 55%, Kev-4B 79%). Pick it when getting the answer right matters
  more than speed.

`GET /v1/models` shows which one a model is in `meta.architecture`; use whichever is installed.

## Request

```json
{
  "model": "<id with the decisions capability>",
  "state": "The player has died 5 times at the bridge and keeps jumping into the river.",
  "questions": {
    "stuck":   {"type": "noul",   "instructions": "Is the player stuck?"},
    "hint":    {"type": "choice", "instructions": "Which hint helps most?",
                "criteria": {"grapple": "the grappling hook crosses gaps",
                             "timing": "wait for the log to float past",
                             "none": "no hint needed"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the player?",
                    "criteria": ["calm", "annoyed", "angry", "rage-quitting"]}
  }
}
```

- `state`: a string, or any JSON object/array (serialized for the model, so pass
  your game state struct as is; keep it relevant and small).
- `questions`: object keyed by your own ids, answered in request order. Max 64
  per request; batch related questions into one call.
- `type`:
  - `noul`: yes/no. Optional `criteria: {"false": "...", "true": "..."}`
    describing each side.
  - `choice`: pick one label. `criteria` is a list of unique labels, or an
    object `{label: description}` (descriptions improve accuracy).
  - `score`: ordinal scale. `criteria` is a list from low to high; items may be
    strings or objects.
- `instructions`: the question, in plain words. Required for Laya; Kev
  accepts a question without it, but always send it.

## Response

```json
{"model": "...",
 "answers": {
   "stuck": {"type": "noul", "confidence": 0.93, "action": {"act_probability": 0.88}, "noul": 0.93},
   "hint":  {"type": "choice", "confidence": 0.71, "action": {"act_probability": 0.8},
             "choice": "timing", "probabilities": {"grapple": 0.2, "timing": 0.77, "none": 0.03}},
   "frustration": {"type": "score", "confidence": 0.52, "action": {"act_probability": 0.6},
                   "score": 2.1, "legend": {"0": "calm", "1": "annoyed", "2": "angry", "3": "rage-quitting"},
                   "probabilities": {"0": 0.05, "1": 0.2, "2": 0.4, "3": 0.35}}
 },
 "usage": {"input_tokens": 142, "output_tokens": 0}}
```

- `noul`: probability of true. `choice`: the argmax label plus every label's
  probability. `score`: expected index on the scale (0 = first criterion).
- `confidence` (0-1) says how sure the model is; `action.act_probability` is
  its probability that the answer is safe to act on rather than escalate. Gate
  game behavior on them: act above a threshold you tune, fall back to a default
  below it.
- Kev answers have no `action`, and its `noul` answers no `confidence`; gate
  on the probability itself (`noul`, or the chosen label's probability).
- The numbers are illustrative; always read them from the response.

## Usage notes

- Fast enough for per-event calls (NPC turn, room enter, chat message), not per
  frame. Debounce and cache by state hash. Kev answers questions one after
  another, so its request time grows with the number of questions.
- Same state + same questions = same answers. Change the wording of
  `instructions` or add criteria descriptions to steer it, then re-test.
- Errors are 400s naming the problem (`question 'type' must be one of choice,
  score, noul`, `choice 'criteria' must be ...`).
