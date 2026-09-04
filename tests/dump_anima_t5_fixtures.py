"""Reference fixture dump for Anima's T5 (t5xxl) unigram tokenizer.

Uses the HF `tokenizers` library directly against the real
`comfy/text_encoders/t5_tokenizer/tokenizer.json` (avoids `transformers`,
whose T5TokenizerFast wrapper pulls in an unrelated broken `regex` version
check in some environments). Dumps {prompt, ids} pairs for a set of test
prompts covering: plain ASCII words, punctuation, multiple consecutive
spaces, leading/trailing whitespace, a single character, and an empty
string — the cases the Zig port (src/t5_tokenizer.zig) is expected to match
exactly (see its module doc for the one documented gap: full Unicode NFKC
normalization is not implemented, so exotic composed-Unicode prompts are
excluded here).

Run: venv/bin/python dump_anima_t5_fixtures.py <tokenizer.json> <out_dir>
"""
import json
import sys

from tokenizers import Tokenizer

PROMPTS = [
    "a cat sitting on a mat",
    "Hello, world!",
    "anime girl with long blue hair, detailed eyes, masterpiece",
    "  extra   spaces   here  ",
    "a",
    "",
    "a-b_c 123",
    "1girl, solo, looking at viewer, best quality, ultra-detailed",
    "   ",
    "The quick brown fox jumps over the lazy dog.",
]


def main():
    tok_path, out_dir = sys.argv[1], sys.argv[2]
    tok = Tokenizer.from_file(tok_path)
    cases = []
    for prompt in PROMPTS:
        enc = tok.encode(prompt)
        cases.append({"prompt": prompt, "ids": enc.ids, "tokens": enc.tokens})
        print(f"{prompt!r:60s} -> {enc.ids}")
    out_path = f"{out_dir}/anima_t5_fixture.json"
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
