#!/usr/bin/env python3
"""Dump T5 parity fixtures for the FLUX.1 text encoder (src/t5.zig).

NOT run in CI — run manually in an ML env (torch + transformers + tokenizers +
numpy). Produces two independent things:

  1. Tokenizer cases (JSONL {"text","ids"}) straight from the pack's
     tokenizer_2/tokenizer.json — the exact T5 Unigram tokenizer, no download.
     `ids` excludes the trailing </s> (the Zig test appends + checks eos).
     Consumed by the env-gated test in src/t5_tokenizer.zig:
       T5_TOK_JSON=<pack>/tokenizer_2/tokenizer.json  T5_TOK_CASES=<out>/tok_cases.jsonl

  2. Encoder reference — fp32 hidden states from HF T5EncoderModel run on CPU on
     the ORIGINAL (unquantized) T5 weights, for one prompt. The Zig encoder runs
     the 4-bit pack, so the bar is cosine + rms_ratio (see the test), not exact.
     Consumed by the env-gated test in src/t5.zig:
       T5_TEST_MODEL=<pack>  T5_REF_IDS=<out>/ref_ids.i32  T5_REF_HIDDEN=<out>/ref_hidden.f32

Reference weights: pass --hf-model pointing at a directory (or HF id) whose
text_encoder_2 is the original google/t5-v1_1-xxl (e.g. a diffusers FLUX.1-dev
checkout's text_encoder_2, or "google/t5-v1_1-xxl"). FLUX uses max_sequence_length
512 and NO attention mask (the full padded sequence is encoded).

Usage:
  python tests/dump_t5_fixtures.py \
    --pack ~/.mlx-serve/models/mflux-community/flux-1-dev-mflux-q4 \
    --hf-model google/t5-v1_1-xxl \
    --out ~/claude-tmp/t5-fixtures \
    --prompt "a photo of a cat sitting on a mat"
"""
import argparse
import json
import os

MAX_LEN = 512
PROMPTS = [
    "a cat",
    "A photo of a cat sitting on a mat.",
    "hello world",
    "Renaissance oil painting of an astronaut riding a horse",
]


def dump_tok_cases(pack, out):
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(os.path.join(pack, "tokenizer_2", "tokenizer.json"))
    path = os.path.join(out, "tok_cases.jsonl")
    with open(path, "w") as f:
        for p in PROMPTS:
            ids = tk.encode(p).ids
            # Strip trailing </s> (id 1) — the Zig test appends + checks eos itself.
            if ids and ids[-1] == 1:
                ids = ids[:-1]
            f.write(json.dumps({"text": p, "ids": ids}) + "\n")
    print(f"wrote {path} ({len(PROMPTS)} cases)")


def dump_encoder_ref(pack, hf_model, out, prompt):
    import numpy as np
    import torch
    from transformers import T5EncoderModel, AutoTokenizer

    # Tokenize with the pack's own T5 tokenizer for id-parity with the server.
    tok = AutoTokenizer.from_pretrained(os.path.join(pack, "tokenizer_2")) \
        if os.path.isdir(os.path.join(pack, "tokenizer_2")) else AutoTokenizer.from_pretrained(hf_model)
    enc_in = tok(prompt, padding="max_length", max_length=MAX_LEN,
                 truncation=True, return_tensors="pt")
    ids = enc_in.input_ids  # [1, MAX_LEN]

    # HF FLUX passes NO attention mask to text_encoder_2.
    model = T5EncoderModel.from_pretrained(
        hf_model, subfolder="text_encoder_2" if os.path.isdir(os.path.join(str(hf_model), "text_encoder_2")) else None,
        torch_dtype=torch.float32).eval()
    with torch.no_grad():
        hidden = model(input_ids=ids)[0]  # [1, MAX_LEN, 4096], fp32

    ids_np = ids[0].to(torch.int32).cpu().numpy()
    hidden_np = hidden[0].to(torch.float32).cpu().numpy()  # [MAX_LEN, 4096]
    ids_np.tofile(os.path.join(out, "ref_ids.i32"))
    hidden_np.tofile(os.path.join(out, "ref_hidden.f32"))
    print(f"wrote ref_ids.i32 [{ids_np.shape}] + ref_hidden.f32 [{hidden_np.shape}] for prompt {prompt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="FLUX.1 mflux pack dir")
    ap.add_argument("--hf-model", default=None, help="original T5 reference (dir or HF id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="A photo of a cat sitting on a mat.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dump_tok_cases(args.pack, args.out)
    if args.hf_model:
        dump_encoder_ref(args.pack, args.hf_model, args.out, args.prompt)
    else:
        print("(skipped encoder reference — pass --hf-model to dump it)")


if __name__ == "__main__":
    main()
