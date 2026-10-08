#!/usr/bin/env python3
"""DeepSeek-V4.1's Engram token map, built from the tokenizer exactly as DeepSeek's inference/engram.py
(`build_compressed_token_map`, MIT) builds it, packed into src/fixtures/dsv41_engram_token_map.bin. Packs that
ship no map (oMLX's) are served from it.

Format: one bit per token id (LSB first) set where the token opens a new compressed id (ids are handed out in
token order), then every other token's compressed id as a little-endian u32.

  venv/bin/python -I tests/dump_dsv41_token_map.py <dir with tokenizer.json> [<pack dir whose map must match>]
"""
import json
import os
import struct
import sys

from tokenizers import Regex, Tokenizer, normalizers


def build_compressed_token_map(tok):
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_new = {}
    lookup = []
    for token_id in range(tok.get_vocab_size(with_added_tokens=True)):
        text = tok.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        lookup.append(key_to_new.setdefault(key, len(key_to_new)))
    return lookup, len(key_to_new)


def pack(lookup):
    bits = bytearray((len(lookup) + 7) // 8)
    repeats = []
    nxt = 0
    for t, v in enumerate(lookup):
        if v == nxt:
            bits[t // 8] |= 1 << (t % 8)
            nxt += 1
        else:
            repeats.append(v)
    return bytes(bits) + struct.pack(f"<{len(repeats)}I", *repeats)


def pack_map(d):
    u32 = os.path.join(d, "engram-token-map.u32")
    if os.path.exists(u32):
        raw = open(u32, "rb").read()
        return list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return json.load(open(os.path.join(d, "engram_token_map.json")))


lookup, n = build_compressed_token_map(Tokenizer.from_file(os.path.join(sys.argv[1], "tokenizer.json")))
print(f"{len(lookup)} tokens -> {n} compressed ids")
if len(sys.argv) > 2:
    other = pack_map(sys.argv[2])
    diff = sum(a != b for a, b in zip(lookup, other)) + abs(len(lookup) - len(other))
    print(f"vs {sys.argv[2]}: {diff} differences")
    if diff:
        sys.exit(1)
out = os.path.join(os.path.dirname(__file__), "..", "src", "fixtures", "dsv41_engram_token_map.bin")
open(out, "wb").write(pack(lookup))
print(f"wrote {out} ({os.path.getsize(out)} bytes)")
