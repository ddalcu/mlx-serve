#!/usr/bin/env python3
"""How far a DeepSeek-V4.1-Flash server is from the hosted model: teacher-forced top-20 against the official API.

The references are ds4's `deepseek-v4.1-flash-20261026-api` continuations (100 short prompts, 64 sampled tokens each,
top-20 logprobs, the hosted 1017 checkpoint; `~/github/ds4/gguf-tools/quality-testing/`). For every position the server is
given the prompt plus the API's tokens so far (/v1/completions, token ids, one token, 20 logprobs) and its top-20 is
compared with the API's: top-1 agreement and KL(API || server) over the union of the two lists, ds4's estimate.

    ./zig-out/bin/mlx-serve --model <pack> --serve --port 11300 &
    python3 tests/dsv41_api_kld.py --cases 0-49 --out /tmp/run.jsonl
    python3 tests/dsv41_api_kld.py --cases 0-49 --out /tmp/new.jsonl --against /tmp/run.jsonl   # paired, same positions

M3 Ultra, Lossless pack: top-1 97.4%, mean KLD 0.0086 on cases 0-49 (ds4's MXFP4 GGUF: 97.6%, 0.0077). A change in
numerics moves the paired mean by about its standard error (0.0003); read the difference with its error, not alone.
The completions endpoint returns token strings, so an alternative that is part of a UTF-8 sequence is dropped from the
server's side (the API side keeps it): one position of 3,200 reads 2.0 nats for that reason alone.
"""
import argparse, json, math, os, statistics, sys, time, urllib.request

BOS, USER, ASSISTANT = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜></think>"
REFS = "~/github/ds4/gguf-tools/quality-testing/deepseek-v4.1-flash-20261026-api"


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def reference(refs, i):
    prompt = open(f"{refs}/prompts/case_{i:03d}.txt", encoding="utf-8").read()
    resp = json.load(open(f"{refs}/responses/case_{i:03d}.json", encoding="utf-8"))
    rows = []
    for e in resp["choices"][0]["logprobs"]["content"]:
        top = {}
        for alt in e.get("top_logprobs") or []:
            if alt.get("bytes") is not None:
                top.setdefault(bytes(alt["bytes"]), alt["logprob"])
        top.setdefault(bytes(e["bytes"] or []), e["logprob"])
        rows.append((e["token"], bytes(e["bytes"] or []), top))
    return prompt, rows


def probe_case(base, prompt, api, max_pos):
    """The server's top-20 at each of the first `max_pos` positions: [(position, {token bytes: logprob})], or None."""
    p_ids = post(base, "/tokenize", {"content": BOS + USER + prompt + ASSISTANT, "add_special": False})["tokens"]
    cont = b"".join(b for _, b, _ in api).decode("utf-8", "replace")
    c_ids = post(base, "/tokenize", {"content": cont, "add_special": False})["tokens"]
    pieces = [post(base, "/detokenize", {"tokens": [t]}).get("content", "") for t in c_ids]
    if len(c_ids) != len(api) or not all(p == s or "�" in p or "�" in s for p, (s, _, _) in zip(pieces, api)):
        return None
    out = []
    for pos in range(min(len(api), max_pos)):
        r = post(base, "/v1/completions", {"model": "x", "prompt": p_ids + c_ids[:pos], "max_tokens": 1, "temperature": 0, "logprobs": 20})
        top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        out.append((pos, {k.encode(): v for k, v in top.items() if "�" not in k}))
    return out


def truncated_kld(ref, cand):
    """KL(ref || cand) over the union of the two top-k sets; a token missing from one side gets that side's smallest."""
    a = {k: math.exp(v) for k, v in ref.items()}
    b = {k: math.exp(v) for k, v in cand.items()}
    fa, fb = min(a.values()), min(b.values())
    union = set(a) | set(b)
    pa = [a.get(k, fa) for k in union]
    pb = [b.get(k, fb) for k in union]
    za, zb = sum(pa), sum(pb)
    return max(0.0, sum(x / za * math.log((x / za) / (y / zb)) for x, y in zip(pa, pb) if x > 0))


def score(local):
    """{(case, pos): (kld, top1 agrees)} over the positions both sides have and whose emitted token matches."""
    out = {}
    for cid, (api, rows) in local.items():
        for pos, top in rows:
            _, emitted, api_top = api[pos]
            top = dict(top)
            top.setdefault(emitted, min(top.values()))
            out[(cid, pos)] = (truncated_kld(api_top, top), max(api_top, key=api_top.get) == max(top, key=top.get))
    return out


def summary(label, s):
    k = sorted(v[0] for v in s.values())
    print(f"{label}: {len(k)} positions · top-1 {100 * sum(v[1] for v in s.values()) / len(k):.1f}% · KLD mean {sum(k) / len(k):.4f} "
          f"median {k[len(k) // 2]:.4f} p95 {k[int(.95 * len(k))]:.4f} max {k[-1]:.3f} nats")


def cases_arg(spec):
    ids = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        ids += range(int(lo), int(hi or lo) + 1)
    return ids


def load(path, refs):
    local = {}
    for line in open(path):
        row = json.loads(line)
        i = int(row["id"].split("_")[1])
        api = local.setdefault(row["id"], (reference(refs, i)[1], []))
        api[1].append((row["pos"], {bytes(k): v for k, v in row["top"]}))
    return local


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:11300")
    ap.add_argument("--refs", default=os.path.expanduser(REFS))
    ap.add_argument("--cases", default="0-99", help="a-b or a comma list of case numbers")
    ap.add_argument("--max-pos", type=int, default=64)
    ap.add_argument("--out", help="JSONL of the server's top-20 per position (resumes: finished cases are kept)")
    ap.add_argument("--against", help="a previous --out: report the paired difference over the shared positions")
    a = ap.parse_args()
    done = load(a.out, a.refs) if a.out and os.path.exists(a.out) else {}
    sink = open(a.out, "a") if a.out else None
    t0 = time.time()
    for i in cases_arg(a.cases):
        cid = f"case_{i:03d}"
        if cid in done:
            continue
        prompt, api = reference(a.refs, i)
        rows = probe_case(a.base, prompt, api, a.max_pos)
        if rows is None:
            print(f"{cid}: token boundaries differ from the API's, skipped", file=sys.stderr)
            continue
        done[cid] = (api, rows)
        if sink:
            for pos, top in rows:
                sink.write(json.dumps({"id": cid, "pos": pos, "top": [[list(k), v] for k, v in top.items()]}) + "\n")
            sink.flush()
        print(f"{cid}: {len(rows)} positions ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    wanted = {f"case_{i:03d}" for i in cases_arg(a.cases)}
    mine = score({c: v for c, v in done.items() if c in wanted})
    summary("server ", mine)
    if a.against:
        other = score({c: v for c, v in load(a.against, a.refs).items() if c in wanted})
        both = [k for k in mine if k in other]
        d = [mine[k][0] - other[k][0] for k in both]
        gained = sum(1 for k in both if mine[k][1] and not other[k][1])
        lost = sum(1 for k in both if other[k][1] and not mine[k][1])
        print(f"paired vs {a.against}: {len(both)} positions · mean dKLD {statistics.mean(d):+.5f} ± {statistics.stdev(d) / math.sqrt(len(d)):.5f} (1 se) "
              f"· top-1 {gained} gained, {lost} lost")


main()
