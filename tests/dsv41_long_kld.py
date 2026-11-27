#!/usr/bin/env python3
"""dsv41_long_kld.py: how far an opt-in prompt-pass switch moves DeepSeek-V4.1-Flash's next-token distributions after a
16K-token prompt, against the exact server. Three arms, each its own mlx-serve server on one build:

  E  the exact server (default env);
  I  the opt-in switch under test: MLX_SERVE_DSV41_PF_INT8=3 (the trunk's wq_b and wo_b on int8 in the wide prompt
     pass) or =4 (also the shared expert and wq_a+wkv);
  F  the floor: MLX_SERVE_DSV41_FLASH_ATTN=0, an already-shipped rounding-class switch, so I's distance from E reads
     beside a rounding change that was accepted.

GLM-5.3-Flash's 32-document design (12 code, 10 PG-19, 10 UltraChat documents of about 16,384 tokens, none sharing a
64-token run with the benchmark prompts), scored the way DeepSeek-V4.1 allows: its CED prompt pass computes logits for a
prompt's last row only, so there is no per-row prompt scoring. Instead every position is its own one-token request,
teacher-forced on one continuation C per document:

  (a) arm E's greedy continuation: /v1/completions, prompt = the document's ids, temperature 0, max_tokens 1,024. The
      text E emits, tokenized back through /tokenize, is C (checked piece by piece against the emitted tokens' text).
  (b) every arm, every position pos in 0..len(C)-1: /v1/completions, prompt = ids + C[:pos], max_tokens 1,
      temperature 0, logprobs = TOPK (1,024, the server's MAX_TOP_LOGPROBS). A stop finish adds pos = len(C), whose
      target is the stop (scored for KL and top-1 only, where the server returns logprobs for it; it returns none
      for a stop it does not emit, so such a position is recorded unscored, "nolp", and left out, counted).
      The prompt is always ids + C[:pos], never the arm's own sample, so every arm walks the same prompts.

The prefix cache (host: engines/mlx/prefill.zig restoreModulePrompt, cache/prefix_cache.zig commitModule): the
plugin snapshots a prompt at its last-but-one token (deepseek_v41.zig armPrompt) and the host commits it as an entry of
len(prompt) - 1 tokens; a later prompt that extends an entry restores it whole (exact, no window replay) and forwards
the rest. So position 0 runs the document's whole 16K prefill (the wide prompt pass, where PF_INT8 acts) and every
later position restores the previous position's entry and forwards two tokens (C[pos-1], C[pos]... the narrow path).
The driver records each response's usage.prompt_tokens and usage.prompt_tokens_details.cached_tokens and requires:
  pos 0:  cached_tokens == 0 (the prefill ran here; else the server's cache already held the document: rc 3, restart
          the server and run again, the finished documents are kept);
  pos>0:  cached_tokens == prompt_tokens - 2 (a whole restore of the previous position's entry); any other value is a
          miss, marked in the record ("hit": false), reported by the analysis and left out of it. Eight misses in a
          row stop the arm (rc 4): the cache is not serving (check --prefix-cache-entries / --prefix-cache-mem).
A cut restore (an entry cut to a pooling boundary and its window replayed, approximate) is chosen only when no whole
entry matches, so it shows as a miss unless it lands exactly on prompt_tokens - 2; the server log names each one
("cut from a longer entry"): count them there.

Phases, one server each (standard library only, python3 -I, resumable: finished documents are kept, an unfinished
one is moved aside to <file>.partial-<k> and redone from position 0):

  ids      --url U --docs <data dir> --run <run dir>   on arm E's first server: tokenizes the prep's text documents
           through POST /tokenize, cuts code and prose to 16,384 tokens (BOS + text), renders chat documents with
           DeepSeek-V4.1's template (chat mode: BOS, <｜User｜>u<｜Assistant｜></think>a<｜end▁of▁sentence｜>..., ending
           on a user turn) and drops trailing turns until it fits 16,384, checks each chat render against the
           server's own (/v1/chat/completions: the same prompt_tokens, required; a /v1/completions of the ids then
           restoring the chat request's entry, cached_tokens == len - 1, recorded), and runs the held-out check: no
           64-token run shared with the prep's reference texts (coding.json, prompts30). A code document that shares
           one stops the phase (rc 2); a prose or chat candidate is skipped, named, and the next spare taken.
  cont     --url U --run R                             the same server: (a) for every selected document.
  score    --url U --run R --arm E|I|F|<name> [--note T]   a FRESH server per arm (arm E's too: its position 0 must
           run the prefill): (b) for every document, then the control: the first document's first 64 positions again
           (its entries are gone by then), which must repeat them byte for byte.
  analyze  --run R --arms E,I,F [--out result.json]     CPU only.
  selftest <new dir> [<dsv41_template_cases.json>]      CPU only: a stand-in server, planted failures.

Analysis (the first arm is the reference E; each other arm X against it; the first other arm minus each later one,
I - F). Positions where any arm missed the cache are left out (counted). Per position:
- KL(E || X) at K = 16, 64, 256, 1024: exact over a partition of the vocabulary, each token whose text appears once in
  both top-K lists plus one bucket for the rest (GLM's KL@64 method). By the data-processing inequality it is a lower
  bound on the full-vocabulary KL; the rest bucket's mass and the smaller K show how tight it is. Tokens are keyed by
  their text (the completions endpoint returns text, not ids): a text that appears twice in a list, an empty text and
  a text holding U+FFFD (a partial UTF-8 byte token) go to the rest bucket.
- top-1 agreement of X with E (the first entry's text).
- the NLL of the target C[pos] in each arm: minus its log-probability where its text appears once in the arm's top-K,
  else (absent) minus the log of the probability outside the top K, a lower bound; left out where the text is
  ambiguous (as above). C is arm E's own greedy continuation, so the NLL differences favour E by construction: read
  KL first.
Reported per category (code, prose, chat) and pooled: means with block standard errors, the documents as blocks
(cluster SE: sqrt(D / (D - 1) * sum_d (S_d - mean * n_d)^2) / N); the documents' mean with its SE over documents;
paired X - E for NLL and I - F for every metric, each with its block SE and z. Also by position (0, 1-63, 64-255,
256+). VOID (rc 2) when a document is missing in an arm, the arms' TOPK differ, arm E's control is not byte-identical
(the exact arm is not deterministic), or arm E's top-1 matches C at under 0.9 of the positions (the records do not
line up with C).
"""
import argparse
import hashlib
import http.server
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

N = 16384
GRAM = 64
CONT = 1024
TOPK = 1024
CTRL = 64
MISS_RUN = 8
SEED = 1234
CHAT_MIN = 0.9  # a chat document's rendered prompt must reach this share of N
BOS_S, EOS_S = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
USER_S, ASSIST_S, THINK_END = "<｜User｜>", "<｜Assistant｜>", "</think>"
REPL = "�"
OFF = {"enable_mtp": False, "enable_pld": False, "enable_drafter": False}
TINY = 1e-12
BUCKETS = ((0, 1), (1, 64), (64, 256), (256, 1 << 30))
BUCKET_NAMES = ("pos0", "pos1-63", "pos64-255", "pos256+")


class Fail(Exception):
    def __init__(self, rc, msg):
        super().__init__(msg)
        self.rc = rc


class Obj(dict):
    """A JSON object that keeps its pairs in order, duplicates included (a top_logprobs object can repeat a text)."""

    def __init__(self, pairs):
        super().__init__(pairs)
        self.pairs = pairs


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"), object_pairs_hook=Obj)
    except urllib.error.HTTPError as e:
        raise Fail(1, f"POST {path} -> HTTP {e.code}: {e.read()[:400]!r}")
    except (urllib.error.URLError, OSError) as e:
        raise Fail(1, f"POST {path} -> {type(e).__name__}: {e}")


def get(url, path):
    try:
        with urllib.request.urlopen(url + path, timeout=60) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:  # provenance only
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def tokenize(url, text):
    return list(post(url, "/tokenize", {"content": text, "add_special": False})["tokens"])


def detok(url, ids):
    return post(url, "/detokenize", {"tokens": list(ids)})["content"]


def one_token(url, s):
    t = tokenize(url, s)
    if len(t) != 1:
        raise Fail(1, f"{s!r} tokenizes to {t}, not one token: not DeepSeek-V4.1's tokenizer?")
    return t[0]


def render_chat(msgs):
    """DeepSeek-V4.1's template (src/fixtures/dsv41_chat_template.jinja) in chat mode for user/assistant text turns, no
    system, no tools, with the generation prompt after the last (user) turn."""
    out, prev = [BOS_S], ""
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            out.append("\n\n" if prev == "user" else USER_S)
            out.append(m["content"])
            if i == len(msgs) - 1 or msgs[i + 1]["role"] == "assistant":
                out.append(ASSIST_S + THINK_END)
        elif m["role"] == "assistant":
            out.append(m["content"])
            out.append(EOS_S)
        else:
            raise ValueError(f"role {m['role']}")
        prev = m["role"]
    return "".join(out)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(p, obj, indent=None):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, p)


def aside(p):
    """Move an unfinished file out of the way (nothing is deleted)."""
    k = 1
    while os.path.exists(f"{p}.partial-{k}"):
        k += 1
    os.replace(p, f"{p}.partial-{k}")
    return f"{p}.partial-{k}"


def same_piece(a, b):
    return a == b or (REPL in a and REPL in b)


# ---------------------------------------------------------------------------------------------------------------- ids

def phase_ids(a):
    sel_path = os.path.join(a.run, "selection.json")
    if os.path.exists(sel_path):
        print(f"LKLD_IDS done already: {sel_path}", flush=True)
        return
    man_path = os.path.join(a.docs, "manifest.json")
    man = load_json(man_path)
    n, gram = a.n, a.gram
    os.makedirs(os.path.join(a.run, "ids"), exist_ok=True)
    bos, eos = one_token(a.url, BOS_S), one_token(a.url, EOS_S)
    grams, refinfo = set(), []
    for r in man["refs"]:
        p = os.path.join(a.docs, r["file"])
        if sha256_file(p) != r["sha256"]:
            raise Fail(1, f"{p}: sha256 is not the manifest's")
        with open(p, encoding="utf-8") as f:
            ids = tokenize(a.url, f.read())
        for i in range(len(ids) - gram + 1):
            grams.add(tuple(ids[i:i + gram]))
        refinfo.append({"name": r["name"], "sha256": r["sha256"], "tokens": len(ids)})
    print(f"LKLD_IDS held-out reference: {len(refinfo)} texts, {sum(r['tokens'] for r in refinfo)} tokens, {len(grams)} distinct {gram}-token runs", flush=True)

    def shared(ids):
        return sum(1 for i in range(len(ids) - gram + 1) if tuple(ids[i:i + gram]) in grams)

    want = man["want"]
    chosen = {k: [] for k in want}
    skipped, order, render_bad = [], [], []
    for d in man["docs"]:
        kind = d["kind"]
        if len(chosen[kind]) >= want[kind]:
            skipped.append(f"{d['id']} {kind}: a spare, not needed")
            continue
        p = os.path.join(a.docs, d["file"])
        if sha256_file(p) != d["sha256"]:
            raise Fail(1, f"{p}: sha256 is not the manifest's")
        doc = load_json(p)
        rec = {"id": d["id"], "kind": kind, "label": d["label"], "lang": d.get("lang")}
        if kind != "chat":
            tid = tokenize(a.url, doc["text"])
            if len(tid) + 1 < n:
                if kind == "code":
                    raise Fail(2, f"code {d['id']} ({d['label']}): {len(tid)} tokens, under {n - 1}: give the prep a longer source")
                skipped.append(f"{d['id']} {kind} ({d['label']}): {len(tid)} tokens, under {n - 1}")
                continue
            ids = [bos] + tid[:n - 1]
            rec.update(text_tokens=len(tid))
        else:
            msgs = [dict(m) for m in doc["messages"]]
            while True:
                while msgs and msgs[-1]["role"] != "user":
                    msgs.pop()
                if not msgs:
                    break
                ids = tokenize(a.url, render_chat(msgs))
                if len(ids) <= n:
                    break
                msgs.pop()
            if not msgs or len(ids) < CHAT_MIN * n:
                skipped.append(f"{d['id']} chat ({d['label']}): {len(ids) if msgs else 0} tokens after the cut, under {int(CHAT_MIN * n)}")
                continue
            if ids[0] != bos:
                raise Fail(1, f"{d['id']}: the rendered chat does not start with BOS")
            rec.update(messages=msgs, turns=len(msgs))
        k = shared(ids)
        if k:
            if kind == "code":
                raise Fail(2, f"code {d['id']} ({d['label']}): shares {k} {gram}-token runs with the held-out reference: pick another source")
            skipped.append(f"{d['id']} {kind} ({d['label']}): shares {k} {gram}-token runs with the held-out reference")
            continue
        if kind == "chat":
            r = post(a.url, "/v1/chat/completions", {"messages": rec["messages"], "max_tokens": 1, "temperature": 0, "seed": SEED,
                                                     "stream": False, "enable_thinking": False, **OFF})
            pt = r["usage"]["prompt_tokens"]
            r2 = post(a.url, "/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0, "seed": SEED, "stream": False, **OFF})
            c2 = (r2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            rec["render_check"] = {"server_prompt_tokens": pt, "ours": len(ids), "same_count": pt == len(ids),
                                   "completions_cached_after_chat": c2, "cache_identity": c2 == len(ids) - 1}
            print(f"LKLD_RENDER {d['id']} server {pt} tokens, ours {len(ids)}; /v1/completions of our ids restored {c2} of {len(ids)} "
                  f"({'the chat request entry: token-identical' if c2 == len(ids) - 1 else 'NOT the chat request entry'})", flush=True)
            if pt != len(ids):
                render_bad.append(d["id"])
        rec.update(n=len(ids), shared_grams=k, ids=ids)
        write_json(os.path.join(a.run, "ids", d["id"] + ".json"), rec)
        chosen[kind].append(d["id"])
        order.append(d["id"])
        print(f"LKLD_IDS {d['id']} {kind:5} {len(ids)} tokens  {d['label']}", flush=True)
    for s in skipped:
        print(f"LKLD_IDS skipped: {s}", flush=True)
    short = {k: (len(v), want[k]) for k, v in chosen.items() if len(v) != want[k]}
    if short:
        raise Fail(2, f"too few documents passed (have, want): {short}")
    if render_bad and not a.accept_own_render:
        raise Fail(2, f"the server renders chat documents {render_bad} to other token counts than ours: fix render_chat, or pass "
                      f"--accept-own-render to score our rendering anyway")
    write_json(sel_path, {"n": n, "gram": gram, "bos": bos, "eos": eos, "order": order, "by_kind": chosen, "skipped": skipped,
                          "render_mismatch": render_bad, "refs": refinfo, "grams": len(grams),
                          "docs_manifest": {"path": man_path, "sha256": sha256_file(man_path)}}, indent=1)
    print(f"LKLD_IDS selected {len(order)} documents: " + ", ".join(f"{len(v)} {k}" for k, v in chosen.items()), flush=True)


# --------------------------------------------------------------------------------------------------------------- cont

def phase_cont(a):
    sel = load_json(os.path.join(a.run, "selection.json"))
    os.makedirs(os.path.join(a.run, "cont"), exist_ok=True)
    for did in sel["order"]:
        cp = os.path.join(a.run, "cont", did + ".json")
        if os.path.exists(cp):
            continue
        ids = load_json(os.path.join(a.run, "ids", did + ".json"))["ids"]
        t0 = time.time()
        r = post(a.url, "/v1/completions", {"prompt": ids, "max_tokens": a.cont, "temperature": 0, "seed": SEED, "logprobs": 1,
                                            "stream": False, **OFF})
        secs = time.time() - t0
        u, ch = r["usage"], r["choices"][0]
        if u["prompt_tokens"] != len(ids):
            raise Fail(1, f"{did}: the server counted {u['prompt_tokens']} prompt tokens, not {len(ids)}")
        text, finish = ch.get("text") or "", ch.get("finish_reason")
        toks = list((ch.get("logprobs") or {}).get("tokens") or [])
        c_ids = tokenize(a.url, text) if text else []
        pieces = [detok(a.url, [t]) for t in c_ids]
        diffs = [i for i in range(max(len(c_ids), len(toks))) if i >= len(c_ids) or i >= len(toks) or not same_piece(pieces[i], toks[i])]
        C, texts = c_ids[:a.cont], pieces[:a.cont]
        stop = finish == "stop" and len(c_ids) <= a.cont
        P = len(C) + (1 if stop else 0)  # a stop adds the position whose greedy token is the stop: no target
        rec = {"doc": did, "C": C, "texts": texts, "finish": finish, "stop_position": stop, "positions": P,
               "completion_tokens": u.get("completion_tokens"), "emitted": len(toks), "exact": not diffs,
               "first_diff": diffs[0] if diffs else None, "diffs": len(diffs), "secs": round(secs, 2)}
        write_json(cp, rec)
        print(f"LKLD_CONT {did} {finish} emitted {len(toks)} (counted {u.get('completion_tokens')}), C {len(C)} tokens, positions {P}, "
              f"{'the emitted tokens exactly' if not diffs else f'retokenized: {len(diffs)} pieces differ from the emitted, the first at {diffs[0]}'}; {secs:.1f} s", flush=True)
    print(f"LKLD_CONT_DONE {len(sel['order'])} documents", flush=True)


# -------------------------------------------------------------------------------------------------------------- score

def score_doc(a, did, ids, C, npos, path, tag):
    t0, misses, run_miss = time.time(), 0, 0
    with open(path, "w", encoding="utf-8") as f:
        for pos in range(npos):
            prompt = ids + C[:pos]
            t = time.time()
            r = post(a.url, "/v1/completions", {"prompt": prompt, "max_tokens": 1, "temperature": 0, "seed": SEED, "logprobs": a.topk,
                                                "stream": False, **OFF})
            secs = time.time() - t
            u = r["usage"]
            pt = u["prompt_tokens"]
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            if pt != len(prompt):
                raise Fail(1, f"{tag} {did} pos {pos}: the server counted {pt} prompt tokens, not {len(prompt)}")
            lp = r["choices"][0]["logprobs"]
            if not lp or not lp.get("top_logprobs"):
                # a stop at the stop position: the server sends no logprobs for the stop it does not emit; the
                # position is recorded unscored ("nolp") and left out of the analysis
                if pos != len(C):
                    raise Fail(1, f"{tag} {did} pos {pos}: no logprobs at a position before the stop")
                f.write(json.dumps({"pos": pos, "pt": pt, "cached": cached, "hit": cached == len(prompt) - 2, "secs": round(secs, 4), "nolp": True}) + "\n")
                continue
            top = lp["top_logprobs"][0]
            pairs = [[k, v] for k, v in (top.pairs if isinstance(top, Obj) else top.items())]
            if pos == 0:
                if cached != 0:
                    raise Fail(3, f"{tag} {did} pos 0: served from the prefix cache ({cached} cached tokens), so its prefill did not run: "
                                  f"restart this arm's server and run the phase again (finished documents are kept)")
                hit = True
            else:
                hit = cached == len(prompt) - 2
            if hit:
                run_miss = 0
            else:
                misses += 1
                run_miss += 1
                print(f"LKLD_MISS {tag} {did} pos {pos}: cached {cached} of {pt} (a hit restores {pt - 2})", flush=True)
            f.write(json.dumps({"pos": pos, "pt": pt, "cached": cached, "hit": hit, "secs": round(secs, 4), "tok": lp["tokens"][0],
                                "tok_lp": lp["token_logprobs"][0], "top": pairs}, ensure_ascii=False) + "\n")
            if run_miss >= MISS_RUN:
                f.flush()
                raise Fail(4, f"{tag} {did}: {run_miss} cache misses in a row at pos {pos}: the prefix cache is not serving the positions")
        secs = time.time() - t0
        f.write(json.dumps({"done": True, "positions": npos, "misses": misses, "secs": round(secs, 2)}) + "\n")
    print(f"LKLD_SCORE {tag} {did} positions {npos} misses {misses} {secs:.1f} s ({secs / max(npos, 1):.3f} s a position)", flush=True)


def finished(path):
    if not os.path.exists(path):
        return False
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            last = line
    try:
        return bool(last) and json.loads(last).get("done") is True
    except ValueError:
        return False


def phase_score(a):
    sel = load_json(os.path.join(a.run, "selection.json"))
    out = os.path.join(a.run, "score-" + a.arm)
    os.makedirs(out, exist_ok=True)
    meta_p = os.path.join(out, "arm.json")
    if os.path.exists(meta_p):
        meta = load_json(meta_p)
        if meta["topk"] != a.topk:
            raise Fail(1, f"{out} was scored at top-{meta['topk']}, not {a.topk}")
    else:
        meta = {"arm": a.arm, "note": a.note, "topk": a.topk, "url": a.url, "models": get(a.url, "/v1/models"), "started": time.time()}
        write_json(meta_p, meta, indent=1)
    meta.setdefault("resumes", []).append({"at": time.time(), "note": a.note, "models": get(a.url, "/v1/models")})
    write_json(meta_p, meta, indent=1)
    docs = sel["order"]
    for did in docs + [None]:
        if did is None:  # the control: the first document's first positions again
            did, path, tag = docs[0], os.path.join(out, "control.jsonl"), a.arm + " control"
        else:
            path, tag = os.path.join(out, did + ".jsonl"), a.arm
        if finished(path):
            continue
        if os.path.exists(path):
            print(f"LKLD_SCORE {tag} {did}: unfinished, moved to {aside(path)}; redone from position 0", flush=True)
        ids = load_json(os.path.join(a.run, "ids", did + ".json"))["ids"]
        c = load_json(os.path.join(a.run, "cont", did + ".json"))
        npos = c["positions"] if not path.endswith("control.jsonl") else min(a.ctrl, c["positions"])
        score_doc(a, did, ids, c["C"], npos, path, tag)
    print(f"LKLD_ARM_DONE {a.arm} documents {len(docs)} and the control", flush=True)


# ------------------------------------------------------------------------------------------------------------ analyze

class Dist:
    __slots__ = ("texts", "lps", "uniq", "count", "rest", "top1")

    def __init__(self, pairs):
        self.texts = [t for t, _ in pairs]
        self.lps = [float(v) for _, v in pairs]
        count = {}
        for t in self.texts:
            count[t] = count.get(t, 0) + 1
        self.count = count
        self.uniq = {t: (r, v) for r, (t, v) in enumerate(zip(self.texts, self.lps)) if count[t] == 1 and t and REPL not in t}
        self.rest = 1.0 - sum(math.exp(v) for v in self.lps)
        self.top1 = self.texts[0] if self.texts else None


def kl_partition(e, x, ks):
    """KL(e || x) at each K of `ks` over the partition: tokens unique by text in both top-K lists, plus the rest."""
    exp, log = math.exp, math.log
    sE, sX, kl = [0.0] * len(ks), [0.0] * len(ks), [0.0] * len(ks)
    xu = x.uniq
    for t, (re_, va) in e.uniq.items():
        h = xu.get(t)
        if h is None:
            continue
        ri, vb = h
        m = re_ if re_ > ri else ri
        pa, pb = exp(va), exp(vb)
        d = pa * (va - vb)
        for q, kk in enumerate(ks):
            if m < kk:
                sE[q] += pa
                sX[q] += pb
                kl[q] += d
    for q in range(len(ks)):
        rE, rX = max(1.0 - sE[q], 0.0), max(1.0 - sX[q], TINY)
        if rE > TINY:
            kl[q] += rE * log(rE / rX)
    return kl


def nll_of(x, target):
    """(NLL, bound used) of the target text in x, or (None, None) where the text is ambiguous."""
    if target is None or not target or REPL in target:
        return None, None
    c = x.count.get(target, 0)
    if c == 1:
        return -x.uniq[target][1], False
    if c > 1:
        return None, None
    return -math.log(max(x.rest, TINY)), True


class Series:
    """A per-position quantity over a group: totals, and each document's (rows, sum) for the block SE."""

    def __init__(self):
        self.n, self.s, self.docs = 0, 0.0, {}

    def add(self, doc, v):
        self.n += 1
        self.s += v
        d = self.docs.setdefault(doc, [0, 0.0])
        d[0] += 1
        d[1] += v

    def stats(self):
        n = self.n
        if not n:
            return {"rows": 0}
        m = self.s / n
        D = len(self.docs)
        se_b = math.sqrt(D / (D - 1) * sum((s - m * k) ** 2 for k, s in self.docs.values())) / n if D > 1 else float("nan")
        dms = [s / k for k, s in self.docs.values()]
        dm = sum(dms) / D
        se_d = math.sqrt(sum((x - dm) ** 2 for x in dms) / (D - 1) / D) if D > 1 else float("nan")
        z = m / se_b if se_b and se_b > 0 and se_b == se_b else float("nan")
        return {"rows": n, "mean": m, "se_block": se_b, "z": z, "doc_mean": dm, "se_doc": se_d, "docs": D}


def fmt(st, digits=6):
    if not st.get("rows"):
        return "rows=0"
    return f"{st['mean']:+.{digits}f} se {st['se_block']:.{digits}f} (z {st['z']:+.2f}; doc mean {st['doc_mean']:+.{digits}f} se {st['se_doc']:.{digits}f}; rows {st['rows']})"


def read_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("done"):
                return
            yield r


def phase_analyze(a):
    run = a.run
    sel = load_json(os.path.join(run, "selection.json"))
    arms = a.arms.split(",")
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise Fail(1, "--arms: the reference first, then at least one other, all distinct")
    E, others = arms[0], arms[1:]
    void, notes = [], []
    metas = {}
    for arm in arms:
        mp = os.path.join(run, "score-" + arm, "arm.json")
        if not os.path.exists(mp):
            raise Fail(1, f"no {mp}: arm {arm} was not scored")
        metas[arm] = load_json(mp)
    topks = {arm: m["topk"] for arm, m in metas.items()}
    if len(set(topks.values())) != 1:
        void.append(f"the arms were scored at different top-K: {topks}")
    topk = min(topks.values())
    ks = [k for k in (16, 64, 256, 1024) if k < topk] + [topk]
    kmax = ks[-1]
    # the pairs reported: X - E for each other arm, and the first other arm minus each later one
    pairs = [(x, E) for x in others] + [(others[0], y) for y in others[1:]]
    groups = ["all", "code", "prose", "chat"] + list(BUCKET_NAMES)
    S = {}

    def ser(*key):
        return S.setdefault(key, Series())

    misses = {arm: 0 for arm in arms}
    excluded, used_rows, align, docs_out, nolp = 0, 0, [0, 0], [], 0
    nll_amb, nll_bound = 0, {arm: 0 for arm in arms}
    for did in sel["order"]:
        meta = load_json(os.path.join(run, "ids", did + ".json"))
        kind = meta["kind"]
        c = load_json(os.path.join(run, "cont", did + ".json"))
        P, texts = c["positions"], c["texts"]
        paths = {arm: os.path.join(run, "score-" + arm, did + ".jsonl") for arm in arms}
        bad = [arm for arm, p in paths.items() if not finished(p)]
        if bad:
            void.append(f"{did}: not finished in arm(s) {bad}")
            continue
        its = {arm: read_records(p) for arm, p in paths.items()}
        dmiss = {arm: 0 for arm in arms}
        dkl = {x: Series() for x in others}
        dtop = {x: Series() for x in others}
        dused = 0
        for pos in range(P):
            recs = {}
            for arm in arms:
                r = next(its[arm], None)
                if r is None or r["pos"] != pos or r["pt"] != meta["n"] + pos:
                    void.append(f"{did} {arm}: record {pos} missing or not position {pos} of {meta['n']} + {pos} tokens")
                    recs = None
                    break
                recs[arm] = r
            if recs is None:
                break
            miss = [arm for arm in arms if not recs[arm]["hit"]]
            for arm in miss:
                dmiss[arm] += 1
                misses[arm] += 1
            if miss:
                excluded += 1
                continue
            if any(recs[arm].get("nolp") for arm in arms):
                nolp += 1
                continue
            dused += 1
            used_rows += 1
            dist = {arm: Dist(recs[arm]["top"]) for arm in arms}
            target = texts[pos] if pos < len(c["C"]) else None
            gk = [kind, "all"] + [BUCKET_NAMES[i] for i, (lo, hi) in enumerate(BUCKETS) if lo <= pos < hi]
            e = dist[E]
            if target is not None:
                align[0] += 1
                align[1] += e.top1 == target
            kl = {x: kl_partition(e, dist[x], ks) for x in others}
            t1 = {x: 1.0 if dist[x].top1 == e.top1 else 0.0 for x in others}
            nl = {arm: nll_of(dist[arm], target) for arm in arms}
            nll_ok = all(v[0] is not None for v in nl.values())
            if target is not None and not nll_ok:
                nll_amb += 1
            for g in gk:
                for x in others:
                    for q, kk in enumerate(ks):
                        ser(g, "kl", kk, x).add(did, kl[x][q])
                    ser(g, "top1", x).add(did, t1[x])
                ser(g, "rest", E).add(did, max(e.rest, 0.0))
                for x, y in pairs[len(others):]:
                    ser(g, "kl-diff", x, y).add(did, kl[x][-1] - kl[y][-1])
                    ser(g, "top1-diff", x, y).add(did, t1[x] - t1[y])
                if nll_ok:
                    for arm in arms:
                        ser(g, "nll", arm).add(did, nl[arm][0])
                    for x, y in pairs:
                        ser(g, "dnll", x, y).add(did, nl[x][0] - nl[y][0])
            if nll_ok:
                for arm in arms:
                    nll_bound[arm] += bool(nl[arm][1])
            for x in others:
                dkl[x].add(did, kl[x][-1])
                dtop[x].add(did, t1[x])
        rec = {"id": did, "kind": kind, "label": meta.get("label"), "positions": P, "used": dused, "misses": dmiss, "finish": c["finish"],
               "C_exact": c["exact"], "kl": {x: dkl[x].stats().get("mean") for x in others}, "top1": {x: dtop[x].stats().get("mean") for x in others}}
        docs_out.append(rec)
        print(f"LKLD_DOC {did} {kind:5} positions {P} used {dused} misses " + "/".join(f"{arm}:{dmiss[arm]}" for arm in arms)
              + " | " + " ".join(f"KL(E||{x})@{kmax}={rec['kl'][x]:.6f}" for x in others if rec["kl"][x] is not None)
              + " | " + " ".join(f"top1 {x}={rec['top1'][x]:.4f}" for x in others if rec["top1"][x] is not None), flush=True)

    # the control: the first document's first positions on each arm, again at the end of its run
    first = sel["order"][0]
    control = {}
    for arm in arms:
        cpath, fpath = os.path.join(run, "score-" + arm, "control.jsonl"), os.path.join(run, "score-" + arm, first + ".jsonl")
        if not (finished(cpath) and finished(fpath)):
            control[arm] = None
            if arm == E:
                void.append("arm E's control is missing")
            continue
        n_same = n = 0
        for x, y in zip(read_records(cpath), read_records(fpath)):
            n += 1
            n_same += x["top"] == y["top"] and x["tok"] == y["tok"] and x["tok_lp"] == y["tok_lp"]
        control[arm] = {"positions": n, "identical": n_same == n and n > 0, "same_positions": n_same}
        print(f"LKLD_CONTROL {arm} {first} positions 0-{n - 1} again: {'byte-identical' if n_same == n and n else f'{n - n_same} of {n} positions differ'}", flush=True)
        if arm == E and not control[arm]["identical"]:
            void.append("arm E's control is not byte-identical: the exact arm is not deterministic")
    al = align[1] / max(align[0], 1)
    print(f"LKLD_ALIGN arm E's top-1 is C's token at {align[1]}/{align[0]} = {al:.5f} of the positions", flush=True)
    if al < 0.9:
        void.append(f"arm E's top-1 matches C at {al:.4f} of the positions, under 0.9: the records do not line up with C")
    tot = used_rows + excluded + nolp
    print(f"LKLD_MISSES positions left out (a miss in any arm): {excluded} of {tot}; by arm " + ", ".join(f"{k} {v}" for k, v in misses.items()), flush=True)
    print(f"LKLD_NOLP stop positions without logprobs, left out: {nolp}", flush=True)
    if excluded:
        notes.append(f"{excluded} of {tot} positions left out for cache misses")
    print(f"LKLD_NLL_COVER targets left out as ambiguous text (repeated, empty or U+FFFD): {nll_amb}; rest-mass bound used: "
          + ", ".join(f"{k} {v}" for k, v in nll_bound.items()), flush=True)

    res = {"valid": False, "arms": arms, "topk": topk, "ks": ks, "groups": {}, "docs": docs_out, "control": control, "align": al,
           "misses": misses, "excluded": excluded, "nolp": nolp, "positions": tot, "nll_ambiguous": nll_amb, "nll_bound": nll_bound,
           "scored": {arm: {"note": m.get("note"), "topk": m["topk"]} for arm, m in metas.items()}, "notes": notes}
    for g in groups:
        out = {}
        if (g, "rest", E) not in S:
            continue
        out["rest_mass_E"] = S[(g, "rest", E)].stats()
        for x in others:
            out[f"kl/{x}"] = {str(kk): S[(g, "kl", kk, x)].stats() for kk in ks}
            out[f"top1/{x}"] = S[(g, "top1", x)].stats()
        for x, y in pairs[len(others):]:
            out[f"kl-diff/{x}-{y}"] = S[(g, "kl-diff", x, y)].stats()
            out[f"top1-diff/{x}-{y}"] = S[(g, "top1-diff", x, y)].stats()
        if (g, "nll", E) in S:
            for arm in arms:
                out[f"nll/{arm}"] = S[(g, "nll", arm)].stats()
            for x, y in pairs:
                out[f"dnll/{x}-{y}"] = S[(g, "dnll", x, y)].stats()
        res["groups"][g] = out
        rows = out["rest_mass_E"]["rows"]
        print(f"LKLD_KL {g} rows={rows} E rest mass outside top-{kmax} {out['rest_mass_E']['mean']:.2e}", flush=True)
        for x in others:
            st = out[f"kl/{x}"]
            print(f"LKLD_KL {g} KL(E||{x})@{kmax} {fmt(st[str(kmax)])} | " + " ".join(f"@{kk} {st[str(kk)]['mean']:.6f}" for kk in ks[:-1]), flush=True)
        for x, y in pairs[len(others):]:
            print(f"LKLD_KL {g} {x}-{y} KL@{kmax} {fmt(out[f'kl-diff/{x}-{y}'])}", flush=True)
        print(f"LKLD_TOP1 {g} " + " | ".join(f"{x} {fmt(out[f'top1/{x}'], 5)}" for x in others)
              + "".join(f" | {x}-{y} {fmt(out[f'top1-diff/{x}-{y}'], 5)}" for x, y in pairs[len(others):]), flush=True)
        if f"nll/{E}" in out:
            print(f"LKLD_NLL {g} E nll {out[f'nll/{E}']['mean']:.5f} | " + " | ".join(f"{x}-{y} {fmt(out[f'dnll/{x}-{y}'], 6)}" for x, y in pairs), flush=True)
    valid = not void and len(docs_out) == len(sel["order"])
    res["valid"], res["void"] = valid, void
    for v in void[:12]:
        print(f"LKLD_VOID {v}", flush=True)
    print(f"LKLD_VALID {valid} ({len(docs_out)} documents; arms {','.join(arms)}; top-{topk})", flush=True)
    if a.out:
        write_json(a.out, res, indent=1)
        print(f"LKLD_RESULT {a.out}", flush=True)
    return 0 if valid else 2


# ----------------------------------------------------------------------------------------------------------- selftest

ALPHA = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:\n(){}=+-*/"
SPECIALS = [BOS_S, EOS_S, USER_S, ASSIST_S, THINK_END]
AMB = (10 + len(ALPHA), 11 + len(ALPHA))  # two ids whose text is U+FFFD (partial UTF-8 bytes)
VOCAB = 12 + len(ALPHA)


def toy_encode(text):
    out, i = [], 0
    while i < len(text):
        for k, s in enumerate(SPECIALS):
            if text.startswith(s, i):
                out.append(k)
                i += len(s)
                break
        else:
            ch = text[i]
            out.append(AMB[0] if ch == REPL else 10 + ALPHA.index(ch))
            i += 1
    return out


def toy_decode(t):
    if t < len(SPECIALS):
        return SPECIALS[t]
    if t in AMB:
        return REPL
    return ALPHA[t - 10]


def toy_logits(ctx, salt, scale):
    # keyed by the last ids' text, so the two U+FFFD ids give one distribution
    h = hashlib.sha256(json.dumps([[toy_decode(t) for t in ctx[-4:]], len(ctx)]).encode()).digest()
    rnd = random.Random(h)
    lg = [rnd.gauss(0, 3) for _ in range(VOCAB)]
    for t in range(len(SPECIALS)):
        lg[t] = -30.0
    if scale:
        r2 = random.Random(h + salt.encode())
        lg = [x + scale * r2.gauss(0, 1) for x in lg]
    m = max(lg)
    lse = m + math.log(sum(math.exp(x - m) for x in lg))
    return [x - lse for x in lg]


class Stand(http.server.BaseHTTPRequestHandler):
    cfg = None

    def log_message(self, *a):
        pass

    def reply(self, obj=None, raw=None, code=200):
        out = raw.encode("utf-8") if raw is not None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def cache(self, prompt):
        """The prefix cache as the server keeps it: entries of len - 1 tokens, the longest one inside a prompt restored."""
        st = self.cfg
        st["requests"] += 1
        if st["requests"] in st.get("fail_at", ()):
            return None
        if len(prompt) in st.get("drop_len", ()):
            st["entries"].clear()
        best = 0
        for e in st["entries"]:
            if len(e) < len(prompt) and len(e) > best and tuple(prompt[:len(e)]) == e:
                best = len(e)
        key = tuple(prompt[:-1])
        if key in st["entries"]:
            st["entries"].remove(key)
        st["entries"].append(key)
        del st["entries"][:-32]
        return best

    def do_POST(self):
        st = self.cfg
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self.reply({"tokens": toy_encode(body["content"])})
        if self.path == "/detokenize":
            return self.reply({"content": "".join(toy_decode(t) for t in body["tokens"])})
        if self.path == "/v1/chat/completions":
            prompt = toy_encode(render_chat(body["messages"]))
        else:
            prompt = body["prompt"]
        cached = self.cache(prompt)
        if cached is None:
            return self.reply({"error": "planted failure"}, code=500)
        k, n = int(body.get("logprobs") or 0), int(body.get("max_tokens", 16))
        gen, lps, finish, ctx = [], [], "length", list(prompt)
        chat = 3 in prompt
        for step in range(n):
            if chat and n > 1 and step == st["stop_after"]:
                finish = "stop"
                break
            key = tuple(ctx)
            scale = st["scale"] + (1e-3 if st.get("nondet") and key in st["seen"] else 0.0)
            st["seen"].add(key)
            lp = toy_logits(ctx, st["salt"], scale)
            t = max(range(VOCAB), key=lambda i: lp[i])
            if n > 1 and step == 3:  # a non-canonical pick: the second U+FFFD id, which retokenizes to the first
                t = AMB[1]
            gen.append(t)
            lps.append(lp)
            ctx.append(t)
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(gen), "total_tokens": len(prompt) + len(gen),
                 "prompt_tokens_details": {"cached_tokens": cached}}
        texts = [toy_decode(t) for t in gen]
        lpj = "null"
        if k:
            tops = []
            for lp in lps:
                order = sorted(range(VOCAB), key=lambda i: (-lp[i], i))[:k]
                tops.append("{" + ",".join(json.dumps(toy_decode(i), ensure_ascii=False) + ":" + f"{lp[i]:.6f}" for i in order) + "}")
            lpj = ('{"tokens":' + json.dumps(texts, ensure_ascii=False) + ',"token_logprobs":' + json.dumps([round(lp[t], 6) for lp, t in zip(lps, gen)])
                   + ',"top_logprobs":[' + ",".join(tops) + "]}")
        raw = ('{"choices":[{"index":0,"text":' + json.dumps("".join(texts), ensure_ascii=False) + ',"logprobs":' + lpj
               + ',"finish_reason":' + json.dumps(finish) + '}],"usage":' + json.dumps(usage) + "}")
        return self.reply(raw=raw)


def serve(port, **cfg):
    st = {"entries": [], "seen": set(), "requests": 0, "salt": "", "scale": 0.0, "stop_after": 25}
    st.update(cfg)
    h = type("H", (Stand,), {"cfg": st})
    s = http.server.ThreadingHTTPServer(("127.0.0.1", port), h)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, st


def selftest(base, fixture):
    os.makedirs(base)
    me = os.path.abspath(__file__)
    py = sys.executable
    n, gram, cont, topk, ctrl = 200, 16, 40, 64, 8
    rnd = random.Random(7)

    def words(k):
        return " ".join("".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(2, 8))) for _ in range(k))

    ref_text = words(400)
    # the data dir as the prep writes it: 2 code, 3 prose (the second shares a run with the reference: skipped), 3 chat
    def make_docs(dd, code_overlap):
        os.makedirs(os.path.join(dd, "docs"))
        os.makedirs(os.path.join(dd, "refs"))
        rp = os.path.join(dd, "refs", "prompt-coding.txt")
        with open(rp, "w", encoding="utf-8") as f:
            f.write(ref_text)
        docs = []
        specs = [("code", "x = (a + b) * c;\n" * 3 + words(160))]
        specs.append(("code", ref_text[:600] + words(150) if code_overlap else "f(y) = {y / 2};\n" + words(160)))
        specs += [("prose", words(170)), ("prose", words(3) + " " + ref_text[1000:1400] + " " + words(120)), ("prose", words(170))]
        for kind, text in specs:
            did = f"d{len(docs) + 1:02d}"
            p = os.path.join(dd, "docs", did + ".json")
            write_json(p, {"id": did, "kind": kind, "label": f"selftest {kind}", "text": text})
            docs.append({"id": did, "kind": kind, "label": f"selftest {kind}", "file": os.path.relpath(p, dd), "sha256": sha256_file(p)})
        for _ in range(3):
            msgs = []
            for j in range(40):
                msgs.append({"role": "user" if j % 2 == 0 else "assistant", "content": words(1)})
            did = f"d{len(docs) + 1:02d}"
            p = os.path.join(dd, "docs", did + ".json")
            write_json(p, {"id": did, "kind": "chat", "label": "selftest chat", "messages": msgs})
            docs.append({"id": did, "kind": "chat", "label": "selftest chat", "file": os.path.relpath(p, dd), "sha256": sha256_file(p)})
        write_json(os.path.join(dd, "manifest.json"), {"want": {"code": 2, "prose": 2, "chat": 2}, "docs": docs,
                                                       "refs": [{"name": "prompt-coding", "file": "refs/prompt-coding.txt", "sha256": sha256_file(rp)}]})

    def run(*args):
        r = subprocess.run([py, "-I", me] + [str(x) for x in args], capture_output=True, text=True)
        print(f"$ {' '.join(str(x) for x in args)}\n  rc={r.returncode}\n" + "".join("  " + l + "\n" for l in (r.stdout + r.stderr).splitlines()[-40:]), flush=True)
        return r

    common = ["--n", n, "--gram", gram]
    checks = {}
    dd = os.path.join(base, "data")
    make_docs(dd, False)
    R = os.path.join(base, "run")
    port = 11461
    # E's first server: ids + cont
    s, _ = serve(port)
    u = f"http://127.0.0.1:{port}"
    r1 = run("ids", "--url", u, "--docs", dd, "--run", R, *common)
    r2 = run("cont", "--url", u, "--run", R, "--cont", cont)
    s.shutdown()
    s.server_close()
    sel = load_json(os.path.join(R, "selection.json")) if r1.returncode == 0 else {}
    checks["ids_selected"] = r1.returncode == 0 and sel.get("by_kind") == {"code": ["d01", "d02"], "prose": ["d03", "d05"], "chat": ["d06", "d07"]} \
        and any(x.startswith("d04 prose") and "shares" in x for x in sel.get("skipped", []))
    conts = {d: load_json(os.path.join(R, "cont", d + ".json")) for d in sel.get("order", [])} if r2.returncode == 0 else {}
    # every continuation's step 3 is the second U+FFFD id: retokenized to the first, the same text (exact by text)
    checks["cont"] = r2.returncode == 0 and len(conts) == 6 and all(c["exact"] and c["C"][3] == AMB[0] for c in conts.values()) \
        and all(c["finish"] == "length" and c["positions"] == cont for d, c in conts.items() if d not in ("d06", "d07")) \
        and all(c["finish"] == "stop" and c["stop_position"] and c["positions"] == 26 for d, c in conts.items() if d in ("d06", "d07"))
    # E's score: a planted failure mid-document, then a fresh server resumes
    port += 1
    s, _ = serve(port, fail_at={60})
    u = f"http://127.0.0.1:{port}"
    rE1 = run("score", "--url", u, "--run", R, "--arm", "E", "--topk", topk, "--ctrl", ctrl)
    s.shutdown()
    s.server_close()
    port += 1
    s, _ = serve(port)
    u = f"http://127.0.0.1:{port}"
    rE2 = run("score", "--url", u, "--run", R, "--arm", "E", "--topk", topk, "--ctrl", ctrl)
    s.shutdown()
    s.server_close()
    checks["resume"] = rE1.returncode == 1 and rE2.returncode == 0 and os.path.exists(os.path.join(R, "score-E", "d02.jsonl.partial-1"))
    # I (noise 0.05) and F (noise 0.02, the cache dropped at prompt length n + 7: a miss at position 7 of each full doc)
    for arm, scale, extra in (("I", 0.05, {}), ("F", 0.02, {"drop_len": {n + 7}})):
        port += 1
        s, _ = serve(port, salt=arm, scale=scale, **extra)
        r = run("score", "--url", f"http://127.0.0.1:{port}", "--run", R, "--arm", arm, "--topk", topk, "--ctrl", ctrl, "--note", f"selftest {arm}")
        s.shutdown()
        s.server_close()
        checks[f"score_{arm}"] = r.returncode == 0
    shutil.copytree(os.path.join(R, "score-I"), os.path.join(R, "score-J"))
    ra = run("analyze", "--run", R, "--arms", "E,I,F", "--out", os.path.join(base, "result.json"))
    res = load_json(os.path.join(base, "result.json")) if ra.returncode == 0 else {}
    # F's planted misses: the position whose prompt is n + 7 tokens long, in every document that reaches it
    drop_pos = {d: n + 7 - load_json(os.path.join(R, "ids", d + ".json"))["n"] for d in conts}
    expect_miss = sum(1 for d, c in conts.items() if 1 <= drop_pos[d] < c["positions"])
    g = res.get("groups", {}).get("all", {})
    klI = g.get("kl/I", {}).get(str(topk), {}).get("mean", -1)
    klF = g.get("kl/F", {}).get(str(topk), {}).get("mean", -1)
    # the partition KL is a lower bound on the stand-in's full-vocabulary KL, and close to it at top-64 of 92
    full, rows = 0.0, 0
    for did, c in conts.items():
        ids = load_json(os.path.join(R, "ids", did + ".json"))["ids"]
        for pos in range(c["positions"]):
            if pos == drop_pos[did]:  # F's planted misses, left out
                continue
            ctx = ids + c["C"][:pos]
            le, li = toy_logits(ctx, "", 0.0), toy_logits(ctx, "I", 0.05)
            full += sum(math.exp(x) * (x - y) for x, y in zip(le, li))
            rows += 1
    full /= max(rows, 1)
    print(f"[selftest] KL(E||I) partition {klI:.6f}, full-vocabulary {full:.6f}; KL(E||F) {klF:.6f}; misses {res.get('misses')} (F expected {expect_miss})")
    checks["analyze"] = ra.returncode == 0 and res.get("valid") and res["misses"] == {"E": 0, "I": 0, "F": expect_miss} and expect_miss >= 4 \
        and 0 < klF < klI <= full * 1.001 + 1e-6 and klI > 0.8 * full and g["kl-diff/I-F"]["mean"] > 0 \
        and all(v and v["identical"] for v in res["control"].values()) and res["align"] >= 0.9
    # a copy of I against I: every paired difference exactly zero
    rz = run("analyze", "--run", R, "--arms", "E,I,J", "--out", os.path.join(base, "result-zero.json"))
    z = load_json(os.path.join(base, "result-zero.json")).get("groups", {}).get("all", {}) if rz.returncode == 0 else {}
    checks["zero"] = rz.returncode == 0 and z.get("kl-diff/I-J", {}).get("mean") == 0.0 and z.get("top1-diff/I-J", {}).get("mean") == 0.0 \
        and z.get("dnll/I-J", {}).get("mean") == 0.0
    # planted: a server whose cache already holds the first document (its position 0 would not run the prefill)
    port += 1
    ids1 = load_json(os.path.join(R, "ids", "d01.json"))["ids"]
    s, _ = serve(port)
    s.RequestHandlerClass.cfg["entries"].append(tuple(ids1[:-1]))
    rw = run("score", "--url", f"http://127.0.0.1:{port}", "--run", R, "--arm", "W", "--topk", topk, "--ctrl", ctrl)
    s.shutdown()
    s.server_close()
    checks["warm_stops"] = rw.returncode == 3
    # planted: arm E not deterministic (a context scored twice drifts): the analysis must come out VOID
    R2 = os.path.join(base, "run-nondet")
    os.makedirs(R2)
    for x in ("selection.json",):
        shutil.copy(os.path.join(R, x), os.path.join(R2, x))
    for x in ("ids", "cont", "score-I"):
        shutil.copytree(os.path.join(R, x), os.path.join(R2, x))
    port += 1
    s, _ = serve(port, nondet=True)
    rn = run("score", "--url", f"http://127.0.0.1:{port}", "--run", R2, "--arm", "E", "--topk", topk, "--ctrl", ctrl)
    s.shutdown()
    s.server_close()
    rv = run("analyze", "--run", R2, "--arms", "E,I")
    checks["nondet_void"] = rn.returncode == 0 and rv.returncode == 2 and "not byte-identical" in rv.stdout
    # planted: a code document that shares a run with the reference stops the ids phase
    dd3, R3 = os.path.join(base, "data-overlap"), os.path.join(base, "run-overlap")
    make_docs(dd3, True)
    port += 1
    s, _ = serve(port)
    ro = run("ids", "--url", f"http://127.0.0.1:{port}", "--docs", dd3, "--run", R3, *common)
    s.shutdown()
    s.server_close()
    checks["code_overlap_stops"] = ro.returncode == 2 and "code d02" in ro.stdout + ro.stderr
    # the chat render against the template's own cases (src/fixtures/dsv41_template_cases.json)
    if fixture:
        cases = [c for c in load_json(fixture) if c["thinking_mode"] == "chat" and not c["tools"]
                 and all(m["role"] in ("user", "assistant") and not m.get("tool_calls") for m in c["messages"])]
        checks["template"] = len(cases) >= 2 and all(render_chat(c["messages"]) == c["expected"] for c in cases)
        print(f"[selftest] template: {len(cases)} chat cases ({', '.join(c['label'] for c in cases)})")
    bad = [k for k, v in checks.items() if not v]
    print(f"[selftest] checks: {checks}")
    print("SELFTEST", "PASS" if not bad else f"FAIL {bad}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ids", "cont", "score"):
        p = sub.add_parser(name)
        p.add_argument("--url", required=True)
        p.add_argument("--run", required=True)
        if name == "ids":
            p.add_argument("--docs", required=True)
            p.add_argument("--n", type=int, default=N)
            p.add_argument("--gram", type=int, default=GRAM)
            p.add_argument("--accept-own-render", action="store_true")
        if name == "cont":
            p.add_argument("--cont", type=int, default=CONT)
        if name == "score":
            p.add_argument("--arm", required=True)
            p.add_argument("--note", default="")
            p.add_argument("--topk", type=int, default=TOPK)
            p.add_argument("--ctrl", type=int, default=CTRL)
    p = sub.add_parser("analyze")
    p.add_argument("--run", required=True)
    p.add_argument("--arms", default="E,I,F")
    p.add_argument("--out")
    p = sub.add_parser("selftest")
    p.add_argument("base")
    p.add_argument("fixture", nargs="?")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest(a.base, a.fixture)
    if a.cmd in ("ids", "cont", "score"):
        a.url = a.url.rstrip("/")
    try:
        if a.cmd == "ids":
            phase_ids(a)
        elif a.cmd == "cont":
            phase_cont(a)
        elif a.cmd == "score":
            if not all(ch.isalnum() or ch in "_-" for ch in a.arm):
                raise Fail(1, "--arm: letters, digits, _ and - only")
            phase_score(a)
        else:
            return phase_analyze(a)
    except Fail as e:
        print(f"LKLD_FAIL rc {e.rc}: {e}", flush=True)
        return e.rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
