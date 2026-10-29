#!/usr/bin/env python3
"""Dump reference Qwen3-Embedding sentence embeddings for the mlx-serve parity check.

Reference = sentence-transformers (Transformer -> LAST-TOKEN pool -> normalize;
issue #116). The Qwen3-Embedding tokenizer's TemplateProcessing post-processor
appends <|endoftext|> to every encode and the pooling reads THAT position —
mlx-serve mirrors it by appending config.json's eos_token_id for last-token
pooling models (`wrapEncoderIds` eos-only arm in host/http/server.zig).

No prompt/instruction prefix on either side: /v1/embeddings serves raw text,
and the reference is encoded with prompt="" for the same reason as the
EmbeddingGemma fixture.

Run (downloads torch + the bf16 checkpoint on first use):
    uv run --with sentence-transformers python3 tests/dump_qwen3_embedding_fixtures.py \
        [--model Qwen/Qwen3-Embedding-0.6B] [--out /tmp/qwen3_embedding_ref.json]

Compare against a running mlx-serve with an MLX conversion loaded:
    python3 tests/dump_qwen3_embedding_fixtures.py --compare http://127.0.0.1:8199 \
        --ref /tmp/qwen3_embedding_ref.json
Expected: 4-bit DWQ worst cosine ~0.958 vs the bf16 reference — measured
2026-08-02, and QUANTIZATION-limited: a same-weights mlx-lm probe (last-token
pool on the appended eos) agrees with the server at >= 0.9996 per sentence.
Default threshold 0.95 admits the 4-bit conversion; pass --threshold 0.98 for
8-bit. Also reusable for CLS-pooling BERTs: --model BAAI/bge-small-en-v1.5.
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


def dump(model_id: str, out_path: str) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    # prompt="" forces raw text (no instruction prefix) — the surface contract.
    vecs = model.encode(SENTENCES, prompt="", normalize_embeddings=True)
    with open(out_path, "w") as f:
        json.dump({"model": model_id, "sentences": SENTENCES, "embeddings": [v.tolist() for v in vecs]}, f)
    print(f"wrote {len(SENTENCES)} reference embeddings ({len(vecs[0])} dims) to {out_path}")


def compare(server: str, ref_path: str, threshold: float) -> int:
    with open(ref_path) as f:
        ref = json.load(f)
    body = json.dumps({"input": ref["sentences"]}).encode()
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
    if worst < threshold:
        print(f"FAIL: below {threshold} parity threshold")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", default="/tmp/qwen3_embedding_ref.json")
    ap.add_argument("--compare", help="mlx-serve base URL; compares --ref instead of dumping")
    ap.add_argument("--ref", default="/tmp/qwen3_embedding_ref.json")
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(args.compare, args.ref, args.threshold))
    dump(args.model, args.out)
