#!/usr/bin/env python3
r"""A DeepSeek-V4.1 DSpark draft vocabulary: the K token ids a training corpus uses most, plus the ids generation must be
able to end on (end of sequence, <think>, </think> and the DSML tool marker).

    python3 -I dsv41_draft_vocab.py <tokenizer.json> <out_prefix> <ks> <train files> -- <eval files>

<ks> is a comma-separated list of sizes K. For each K it writes <out_prefix>-<K>.txt (the ids ascending, 16 per
line, space-separated) and prints each eval file's coverage by that set against a prefix cut of K ids. It tokenizes
with the pack's byte-level BPE (its merges and its three split regexes, \p classes approximated for Python's re).

The shipped scripts/dsv41_draft_vocab.txt, which dsv41_nvfp4_pack.py writes into every pack as draft-vocab.txt, is K = 65536 over about 5.3M training tokens: the CPython
standard library, the Zig standard library, the MLX C++ sources and docs, the macOS SDK C headers and one prose set.
Every source file that the benchmark prompts are built from, and every file that shares a line with one, was left
out of the corpus.
"""
import json, re, sys, unicodedata

tj = json.load(open(sys.argv[1]))
out_prefix, ks = sys.argv[2], [int(k) for k in sys.argv[3].split(",")]
args = sys.argv[4:]
cut = args.index("--")
train, evals = args[:cut], args[cut + 1:]
vocab = tj["model"]["vocab"]
ranks = {}
for i, m in enumerate(tj["model"]["merges"]):
    a, b = m.split(" ") if isinstance(m, str) else m
    ranks[(a, b)] = i

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

B2U = bytes_to_unicode()
L = r"[^\W\d_]"
PS = r"(?:[^\w\s]|_)"
SPLITS = [
    re.compile(r"\d{1,3}"),
    re.compile(r"[一-龥぀-ゟ゠-ヿ]+"),
    re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[ \t\f\v]?" + L + r"+| ?" + PS + r"+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
]

def split_isolated(rx, pieces):
    out = []
    for p in pieces:
        pos = 0
        for mt in rx.finditer(p):
            if mt.start() > pos:
                out.append(p[pos:mt.start()])
            out.append(mt.group(0))
            pos = mt.end()
        if pos < len(p):
            out.append(p[pos:])
    return out

cache = {}

def bpe(word):
    if word in cache:
        return cache[word]
    parts = list(word)
    while len(parts) > 1:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get((parts[i], parts[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    ids = [vocab[p] for p in parts if p in vocab]
    cache[word] = ids
    return ids

def encode(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    for piece in split_isolated(SPLITS[2], split_isolated(SPLITS[1], split_isolated(SPLITS[0], [text]))):
        yield from bpe("".join(B2U[b] for b in piece.encode("utf-8")))

import collections
freq = collections.Counter()
n_train = 0
for path in train:
    c = collections.Counter(encode(path))
    freq.update(c)
    n_train += sum(c.values())
    print(f"train {path}: {sum(c.values())} tokens", flush=True)
added = {a["content"]: a["id"] for a in tj["added_tokens"]}
forced = sorted({1} | {added[t] for t in ("<think>", "</think>", "\uff5cDSML\uff5c") if t in added})
print(f"train: {n_train} tokens, {len(freq)} distinct ids; forced ids {forced}", flush=True)
ranked = [t for t, _ in freq.most_common()]
subsets = {}
for k in ks:
    s = set(forced)
    for t in ranked:
        if len(s) >= k:
            break
        s.add(t)
    for t in range(len(vocab)):
        if len(s) >= k:
            break
        s.add(t)
    subsets[k] = s
    ids = sorted(s)
    with open(f"{out_prefix}-{k}.txt", "w") as f:
        f.write("".join(" ".join(str(t) for t in ids[i:i + 16]) + "\n" for i in range(0, len(ids), 16)))
for path in evals:
    ids = list(encode(path))
    n = max(len(ids), 1)
    row = " ".join(f"K{k}: set {100.0 * sum(1 for t in ids if t in subsets[k]) / n:.3f}% prefix {100.0 * sum(1 for t in ids if t < k) / n:.3f}%" for k in ks)
    print(f"eval {path}: {len(ids)} tokens; {row}", flush=True)
