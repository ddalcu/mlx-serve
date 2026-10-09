#!/usr/bin/env python3
"""Dump reference EmbeddingGemma sentence embeddings for the mlx-serve parity check.

Reference = sentence-transformers in float32 (Transformer -> mean pool ->
dense head -> normalize), NO prompt prefix (default_prompt_name is null, and the
/v1/embeddings surface serves raw text).

Run (downloads torch + the fp32 checkpoint on first use):
    uv run --with sentence-transformers python3 tests/dump_embeddinggemma_fixtures.py \
        [--model google/embeddinggemma-300m] [--out /tmp/embeddinggemma_ref.json]

google/embeddinggemma-300m is gated on HF; the unsloth/embeddinggemma-300m
mirror is the ungated default here (same weights).

EmbeddingGemma 2 (`--model google/embeddinggemma-2`) needs sentence-transformers
>= 6.1 and transformers >= 5.18 with torchvision (its processor imports the
Gemma 4 image code). Its last sentence is long enough to cross the 512-position
sliding band.

Compare against a running mlx-serve with the 8-bit MLX conversion loaded:
    python3 tests/dump_embeddinggemma_fixtures.py --compare http://127.0.0.1:11297 \
        --ref /tmp/embeddinggemma_ref.json
Expected: per-sentence cosine >= 0.98 (8-bit quant + bf16 vs fp32 reference).
"""

import argparse
import json
import math
import sys
import urllib.request

SENTENCES = [
    "The chef prepared a delicious pasta dinner for the guests.",
    "A cook made tasty spaghetti for the evening meal.",
    "Quantum entanglement violates local realism in Bell tests.",
    "The stock market rallied after the central bank's announcement.",
    "She debugged the segfault by bisecting the commit history.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "def mean(xs): return sum(xs) / len(xs)",
    "A single word",
]
# ~1000 tokens of distinct sentences: positions farther apart than the 512 band must still agree.
SENTENCES.append(" ".join(f"Item {i}: {SENTENCES[i % 7]}" for i in range(60)))


def dump(model_id: str, out_path: str) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, model_kwargs={"dtype": "float32"})
    # prompt="" forces raw text (no task prefix) regardless of config defaults.
    vecs = model.encode(SENTENCES, prompt="", normalize_embeddings=True)
    with open(out_path, "w") as f:
        json.dump({"model": model_id, "sentences": SENTENCES, "embeddings": [v.tolist() for v in vecs]}, f)
    print(f"wrote {len(SENTENCES)} reference embeddings ({len(vecs[0])} dims) to {out_path}")


def compare(server: str, ref_path: str) -> int:
    with open(ref_path) as f:
        ref = json.load(f)
    body = json.dumps({"model": "embeddinggemma", "input": ref["sentences"]}).encode()
    req = urllib.request.Request(server.rstrip("/") + "/v1/embeddings", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        got = json.load(r)
    ours = [d["embedding"] for d in got["data"]]
    worst = 1.0
    for i, (a, b) in enumerate(zip(ref["embeddings"], ours)):
        cos = sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))
        worst = min(worst, cos)
        print(f"  [{i}] cos={cos:.5f}  {ref['sentences'][i][:60]!r}")
    print(f"worst cosine: {worst:.5f}")
    if worst < 0.98:
        print("FAIL: below 0.98 parity threshold")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/embeddinggemma-300m")
    ap.add_argument("--out", default="/tmp/embeddinggemma_ref.json")
    ap.add_argument("--compare", help="mlx-serve base URL; compares --ref instead of dumping")
    ap.add_argument("--ref", default="/tmp/embeddinggemma_ref.json")
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(args.compare, args.ref))
    dump(args.model, args.out)
