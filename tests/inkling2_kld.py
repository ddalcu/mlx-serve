#!/usr/bin/env python3
"""How far an Inkling 2 Mini pack is from the release: teacher-forced top-K logprobs on held-out text.

One server at a time (the release and a re-quantized pack do not fit a small Mac together), so two steps:

    ./zig-out/bin/mlx-serve --model <release> --serve --port 11300 &
    python3 tests/inkling2_kld.py collect --url http://127.0.0.1:11300 --out release.json
    # stop it, serve the pack under test
    python3 tests/inkling2_kld.py collect --url http://127.0.0.1:11300 --out pack.json
    python3 tests/inkling2_kld.py compare release.json pack.json

`collect` sends each text once to /v1/completions with `echo` and `logprobs` (every prompt position's top-K). `compare`
reports top-1 agreement and KL(release || pack) per position over the union of both top-K lists, a token one list
lacks priced at that list's K-th logprob (a floor: the estimate is a lower bound on a long tail, comparable between
packs of the same corpus). The corpus is files of this checkout, one per domain, cut to `--chars` characters: `--root` reads them from a frozen
copy, since a file that changes between two runs tokenizes differently and `compare` refuses it.
"""
import argparse
import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = [
    "docs/reference.md",
    "src/plugins/inkling2/kernels.zig",
    "CHANGELOG.md",
    "tests/convert_inkling2.py",
    "app/CLAUDE.md",
    "README.md",
    "src/host/chat/AGENTS.md",
    "docs/recipes/add-a-model.md",
]


def post(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def collect(args):
    docs = []
    for name in args.files or CORPUS:
        path = Path(args.root) / name
        if not path.exists():
            print(f"skip {name}: missing", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[args.skip : args.skip + args.chars]
        # max_tokens 1: the echoed prompt is what is scored; one sampled token is not.
        r = post(args.url, {"prompt": text, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": args.top, "stream": False})
        lp = r["choices"][0]["logprobs"]
        n = len(lp["tokens"]) - 1  # the sampled token is not part of the text
        docs.append({"file": name, "tokens": lp["tokens"][:n], "top": [t or {} for t in lp["top_logprobs"][:n]], "logprob": lp["token_logprobs"][:n]})
        print(f"{name}: {n} positions", flush=True)
    Path(args.out).write_text(json.dumps({"top": args.top, "docs": docs}))


def kl(ref, cand):
    """KL(ref || cand) over the union of two top-K dicts {token: logprob}, a missing side at its own floor."""
    if not ref or not cand:
        return None
    floor_r, floor_c = min(ref.values()), min(cand.values())
    keys = set(ref) | set(cand)
    p = {k: math.exp(ref.get(k, floor_r)) for k in keys}
    q = {k: math.exp(cand.get(k, floor_c)) for k in keys}
    sp, sq = sum(p.values()), sum(q.values())
    return sum((p[k] / sp) * math.log((p[k] / sp) / (q[k] / sq)) for k in keys)


def compare(args):
    a, b = json.load(open(args.ref)), json.load(open(args.pack))
    allk, top1, nll_a, nll_b, n = [], 0, 0.0, 0.0, 0
    for da, db in zip(a["docs"], b["docs"]):
        assert da["tokens"] == db["tokens"], f"{da['file']}: the packs tokenize the text differently"
        ks = []
        for i in range(1, len(da["tokens"])):  # position 0 has no logprob
            k = kl(da["top"][i], db["top"][i])
            if k is None:
                continue
            ks.append(k)
            top1 += max(da["top"][i], key=da["top"][i].get) == max(db["top"][i], key=db["top"][i].get)
            nll_a -= da["logprob"][i]
            nll_b -= db["logprob"][i]
            n += 1
        allk += ks
        print(f"{da['file']:40s} KLD {statistics.fmean(ks):.4f}  p99 {sorted(ks)[int(0.99 * len(ks))]:.3f}  ({len(ks)} positions)")
    se = statistics.stdev(allk) / math.sqrt(len(allk))
    print(f"\nmean KLD {statistics.fmean(allk):.4f} +/- {se:.4f}   median {statistics.median(allk):.4f}   top-1 agreement {top1 / n:.1%}   "
          f"ppl release {math.exp(nll_a / n):.3f} pack {math.exp(nll_b / n):.3f}   ({n} positions)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--top", type=int, default=20)
    c.add_argument("--root", default=str(ROOT), help="where the corpus files are read from: copy them to compare packs over time")
    c.add_argument("--chars", type=int, default=6000)
    c.add_argument("--skip", type=int, default=0, help="characters to skip at the start of each file")
    c.add_argument("files", nargs="*", help="repo-relative files (default: the built-in corpus)")
    c.set_defaults(fn=collect)
    d = sub.add_parser("compare")
    d.add_argument("ref")
    d.add_argument("pack")
    d.set_defaults(fn=compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
