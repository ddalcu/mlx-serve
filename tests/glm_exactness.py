#!/usr/bin/env python3
"""Exactness, sampling and logprob checks against ONE running server, plus cross-config diffs.

  glm_exactness.py run --base URL --label CFG --out DIR [--greedy-only] [--no-cache]
  glm_exactness.py compare DIR

`run` writes DIR/CFG.json: per check {ok, detail} and every greedy reply with its tokens and
top-5 logprobs. `compare` diffs each config's greedy replies against `default`: first divergent
token, the reference's top-1/top-2 margin there (a near-tie is rounding, a wide margin is a bug)
and where the other config ranked the reference's token. Written for GLM-5.3-Flash (sparse path
past the indexer's 2051-token budget) but any chat model works. Stdlib only.

Bars: greedy cold == warm == stream (bytes); temp-0 rank 1 == emitted; stream logprobs == non-
stream; seeded draws repeat across runs and transports and move with the seed; penalties change
the output; a stop string cuts the greedy answer at its first match with finish_reason stop.
"""

import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": {
    "type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                     "languages": {"type": "array", "items": {"type": "string"}}},
    "required": ["name", "age", "languages"], "additionalProperties": False}}}
NOTHINK = {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}


def source(path, chars):
    return open(os.path.join(ROOT, path)).read()[:chars]


def prompts():
    """name -> request body (no model/stream). Text prompts run thinking off so logprobs cover the
    whole reply; the agent-shaped ones keep the model's default thinking."""
    mid = "Here is a source file:\n\n" + source("src/round_cost.zig", 11000) + "\n\nSummarize what it does in five bullet points."
    long = ("The deploy password for the Aldergate cluster is ORCHID-7714.\n\nHere is a source file:\n\n"
            + source("src/round_cost.zig", 34000)
            + "\n\nFirst give the deploy password for the Aldergate cluster, then list three public functions in the file.")
    lp = {"logprobs": True, "top_logprobs": 5}
    return {
        "short": {**NOTHINK, **lp, "max_tokens": 200, "messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]},
        "mid3k": {**NOTHINK, **lp, "max_tokens": 200, "messages": [{"role": "user", "content": mid}]},
        "long9k": {**NOTHINK, **lp, "max_tokens": 200, "messages": [{"role": "user", "content": long}]},
        "think": {"max_tokens": 1500, "messages": [{"role": "user", "content": "What is 17 * 23? Show the answer only."}]},
        "tool": {"max_tokens": 2000, "tools": TOOLS, "messages": [{"role": "user", "content": "What is the weather in Paris and in Tokyo right now? Use the tool."}]},
        "schema": {**NOTHINK, "max_tokens": 300, "response_format": SCHEMA, "messages": [{"role": "user", "content": "Describe a fictional programmer named Ada as JSON."}]},
    }


class Server:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.model = self.get("/v1/models")["data"][0]["id"]

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=60) as r:
            return json.load(r)

    def post(self, path, body, stream=False):
        body = {"model": self.model, **body, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            if not stream:
                return json.load(r)
            return [json.loads(l[6:]) for l in (x.decode().strip() for x in r) if l.startswith("data: {")]

    def chat(self, body, stream=False):
        """Normalized reply: content, reasoning, tool_calls, finish, usage, logprobs entries."""
        out = {"content": "", "reasoning": "", "tool_calls": [], "finish": None, "usage": {}, "lp": []}
        if not stream:
            d = self.post("/v1/chat/completions", body)
            ch = d["choices"][0]
            m = ch["message"]
            out.update(content=m.get("content") or "", reasoning=m.get("reasoning_content") or "",
                       finish=ch.get("finish_reason"), usage=d.get("usage") or {},
                       tool_calls=[(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []],
                       lp=(ch.get("logprobs") or {}).get("content") or [])
            out["finish_details"] = ch.get("finish_details")
            return out
        calls = {}
        for d in self.post("/v1/chat/completions", body, True):
            if d.get("usage"):
                out["usage"] = d["usage"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                out["content"] += dl.get("content") or ""
                out["reasoning"] += dl.get("reasoning_content") or ""
                for t in dl.get("tool_calls") or []:
                    c = calls.setdefault(t.get("index", 0), ["", ""])
                    f = t.get("function") or {}
                    c[0] += f.get("name") or ""
                    c[1] += f.get("arguments") or ""
                out["lp"] += (ch.get("logprobs") or {}).get("content") or []
                out["finish"] = ch.get("finish_reason") or out["finish"]
        out["tool_calls"] = [tuple(calls[k]) for k in sorted(calls)]
        return out

    def completion(self, body, stream=False):
        if not stream:
            ch = self.post("/v1/completions", body)["choices"][0]
            lp = ch.get("logprobs") or {}
            return {"text": ch.get("text") or "", "finish": ch.get("finish_reason"), "tokens": lp.get("tokens") or [],
                    "token_logprobs": lp.get("token_logprobs") or [], "top": lp.get("top_logprobs") or []}
        out = {"text": "", "finish": None, "tokens": [], "token_logprobs": [], "top": []}
        for d in self.post("/v1/completions", body, True):
            for ch in d.get("choices") or []:
                out["text"] += ch.get("text") or ""
                out["finish"] = ch.get("finish_reason") or out["finish"]
                lp = ch.get("logprobs") or {}
                for k in ("tokens", "token_logprobs"):
                    out[k] += lp.get(k) or []
                out["top"] += lp.get("top_logprobs") or []
        return out


def visible(r):
    return {"content": r["content"], "reasoning": r["reasoning"], "tool_calls": r["tool_calls"]}


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None if len(a) == len(b) else n)


def lp_problems(entries):
    """Temp-0 bar: rank 1 is the emitted token (ties allowed), logprobs <= 0, top sorted."""
    bad = []
    for i, e in enumerate(entries):
        top = e.get("top_logprobs") or []
        if e["logprob"] > 1e-4:
            bad.append(f"#{i} logprob {e['logprob']} > 0")
        if top and top[0]["token"] != e["token"] and top[0]["logprob"] > e["logprob"] + 1e-5:
            bad.append(f"#{i} emitted {e['token']!r} ({e['logprob']:.4f}) but rank 1 {top[0]['token']!r} ({top[0]['logprob']:.4f})")
        if any(top[j]["logprob"] < top[j + 1]["logprob"] - 1e-6 for j in range(len(top) - 1)):
            bad.append(f"#{i} top_logprobs not sorted")
    return bad


def run(base, label, outdir, greedy_only=False, no_cache=False):
    s = Server(base)
    res = {"label": label, "base": base, "model": s.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "checks": [], "greedy": {}}

    def check(name, ok, detail=""):
        res["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  — ' + str(detail)[:300] if detail and not ok else ''}", flush=True)

    def info(name, detail):
        res["checks"].append({"name": name, "ok": None, "detail": detail})
        print(f"  info {name}: {detail}", flush=True)

    for name, body in prompts().items():
        body = {**body, "temperature": 0}
        t0 = time.time()
        cold = s.chat(body)
        warm = s.chat(body)
        st = s.chat(body, stream=True)
        res["greedy"][name] = {"reply": visible(cold), "finish": cold["finish"], "usage": cold["usage"],
                               "tokens": [e["token"] for e in cold["lp"]], "lp": cold["lp"],
                               "warm_cached": (warm["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
                               "wall_s": round(time.time() - t0, 1)}
        pt = cold["usage"].get("prompt_tokens")
        print(f" [{name}] prompt={pt} gen={cold['usage'].get('completion_tokens')} finish={cold['finish']} ({res['greedy'][name]['wall_s']}s)", flush=True)
        if greedy_only:
            continue
        check(f"{name}: answers", cold["content"].strip() or cold["tool_calls"], visible(cold))
        a, b = json.dumps(visible(cold)), json.dumps(visible(warm))
        check(f"{name}: greedy {'repeat' if no_cache else 'cold == warm'} (bytes)", a == b,
              f"first diff at char {first_diff(a, b)}: cold={a[:200]} warm={b[:200]}")
        cached = res["greedy"][name]["warm_cached"] or 0
        if no_cache:
            check(f"{name}: no cache reuse with the cache off", cached == 0, f"cached={cached}")
        else:
            check(f"{name}: warm request reuses the prefix", cached > 0, f"cached={cached}")
        c = json.dumps(visible(st))
        check(f"{name}: stream == non-stream (bytes)", c == b, f"first diff at char {first_diff(c, b)}: stream={c[:200]}")
        if name == "tool":
            check("tool: both cities called with JSON args",
                  {json.loads(a).get("city") for n, a in cold["tool_calls"] if n == "get_weather"} >= {"Paris", "Tokyo"}
                  and cold["finish"] == "tool_calls", cold["tool_calls"])
        if name == "schema":
            try:
                d = json.loads(cold["content"])
                check("schema: content parses with every required key", all(k in d for k in ("name", "age", "languages")), d)
            except ValueError:
                check("schema: content parses with every required key", False, cold["content"][:200])
        if cold["lp"]:
            check(f"{name}: logprobs rank 1 == emitted, <= 0, sorted", not lp_problems(cold["lp"]), lp_problems(cold["lp"])[:5])
            d = [i for i, (x, y) in enumerate(zip(cold["lp"], st["lp"]))
                 if x["token"] != y["token"] or abs(x["logprob"] - y["logprob"]) > 1e-4]
            check(f"{name}: stream logprobs == non-stream", len(cold["lp"]) == len(st["lp"]) and not d,
                  f"n={len(cold['lp'])}/{len(st['lp'])} first mismatch {d[:1]}")
            toks = "".join(e["token"] for e in cold["lp"])
            check(f"{name}: logprob tokens spell message.content", toks == cold["content"],
                  f"tokens={toks[:120]!r} content={cold['content'][:120]!r}")
    if greedy_only:
        return save(res, outdir)

    # /v1/completions logprobs: the integer form, both transports.
    body = {"prompt": "The capital of France is", "max_tokens": 16, "temperature": 0, "logprobs": 5}
    a, b = s.completion(body), s.completion(body, stream=True)
    bad = [f"#{i} emitted {t!r} not rank 1 of {top}" for i, (t, lp, top) in enumerate(zip(a["tokens"], a["token_logprobs"], a["top"]))
           if top and max(top.values()) > lp + 1e-5]
    bad += [f"#{i} logprob {lp} > 0" for i, lp in enumerate(a["token_logprobs"]) if lp > 1e-4]
    check("completions: logprobs present, rank 1 == emitted, <= 0", a["tokens"] and not bad, bad[:5] or a)
    check("completions: stream == non-stream (text + logprobs)",
          a["text"] == b["text"] and a["tokens"] == b["tokens"]
          and all(abs(x - y) <= 1e-4 for x, y in zip(a["token_logprobs"], b["token_logprobs"])),
          f"ns={a['text']!r} st={b['text']!r}")
    check("completions: tokens spell text", "".join(a["tokens"]) == a["text"], f"{a['tokens']} vs {a['text']!r}")

    # Seeded sampling.
    q = {**NOTHINK, "max_tokens": 120, "messages": [{"role": "user", "content": "Write a two-sentence story about a lighthouse keeper."}]}
    smp = {"temperature": 0.8, "top_p": 0.9, "top_k": 40}
    r1 = s.chat({**q, **smp, "min_p": 0.05, "seed": 7})
    r2 = s.chat({**q, **smp, "min_p": 0.05, "seed": 7})
    r3 = s.chat({**q, **smp, "min_p": 0.05, "seed": 7}, stream=True)
    r4 = s.chat({**q, **smp, "min_p": 0.05, "seed": 8})
    r5 = s.chat({**q, **smp, "seed": 7})
    check("seeded: same seed, same text (non-stream x2)", r1["content"] == r2["content"], [r1["content"][:120], r2["content"][:120]])
    check("seeded: stream == non-stream", r1["content"] == r3["content"], [r1["content"][:120], r3["content"][:120]])
    check("seeded: another seed draws other text", r1["content"] != r4["content"], r1["content"][:120])
    info("seeded: min_p 0.05 changes the draw", "no (field ignored or inert at 0.05)" if r1["content"] == r5["content"] else "yes")
    rc = s.completion({"prompt": "Once upon a time", "max_tokens": 40, **smp, "seed": 7})
    rd = s.completion({"prompt": "Once upon a time", "max_tokens": 40, **smp, "seed": 7}, stream=True)
    check("seeded completions: stream == non-stream", rc["text"] == rd["text"], [rc["text"], rd["text"]])

    # Penalties: each must change a repetitive greedy answer, on both surfaces.
    rep = {**NOTHINK, "max_tokens": 80, "temperature": 0,
           "messages": [{"role": "user", "content": "Write the word banana forty times, separated by spaces, nothing else."}]}
    base_c = s.chat(rep)["content"]
    for field, val in (("repeat_penalty", 1.5), ("presence_penalty", 1.5)):
        r = s.chat({**rep, field: val})
        check(f"chat {field}={val}: well formed and changes the output",
              r["finish"] in ("stop", "length") and r["content"].strip() and r["content"] != base_c, r["content"][:120])
    info("chat repetition_penalty (vLLM spelling) changes the output",
         "yes" if s.chat({**rep, "repetition_penalty": 1.5})["content"] != base_c else "no (field not read)")
    # The server reads frequency_penalty x as repeat_penalty 1+x: both spellings, same bytes.
    cp = {"prompt": "banana banana banana banana banana banana", "max_tokens": 40, "temperature": 0}
    base_t = s.completion(cp)["text"]
    for surface, call, body, base_out, key in (("chat", s.chat, rep, base_c, "content"), ("completions", s.completion, cp, base_t, "text")):
        f, r = call({**body, "frequency_penalty": 1.5})[key], call({**body, "repeat_penalty": 2.5})[key]
        check(f"{surface} frequency_penalty=1.5 == repeat_penalty=2.5", f == r and r != base_out, f"freq={f[:80]!r} rep={r[:80]!r} base={base_out[:80]!r}")
    r = s.completion({**cp, "repeat_penalty": 1.5})["text"]
    check("completions repeat_penalty=1.5: changes the output", r != base_t, f"base={base_t!r} pen={r!r}")
    info("completions presence_penalty=1.5 changes the output", "yes" if s.completion({**cp, "presence_penalty": 1.5})["text"] != base_t else "no")

    # Stop strings: the cut lands on the first match of the greedy answer.
    cnt = {**NOTHINK, "max_tokens": 200, "temperature": 0,
           "messages": [{"role": "user", "content": "Count from 1 to 30, separated by commas and spaces. Output only the numbers."}]}
    full = s.chat(cnt)["content"]
    stop = ", 12"
    for stream in (False, True):
        r = s.chat({**cnt, "stop": [stop]}, stream=stream)
        want = full[:full.find(stop)] if stop in full else full
        check(f"stop{' (stream)' if stream else ''}: cut at the first match, finish stop",
              r["content"] == want and r["finish"] == "stop" and stop in full, f"got={r['content'][-60:]!r} want={want[-60:]!r} finish={r['finish']}")
    return save(res, outdir)


def save(res, outdir):
    os.makedirs(outdir, exist_ok=True)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(outdir, res["label"] + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    n = [c for c in res["checks"] if c["ok"] is not None]
    print(f"{res['label']}: {sum(c['ok'] for c in n)}/{len(n)} checks pass")
    return res


def compare(outdir, ref_label="default"):
    runs = {f[:-5]: json.load(open(os.path.join(outdir, f))) for f in sorted(os.listdir(outdir)) if f.endswith(".json") and f != "compare.json"}
    ref = runs.get(ref_label)
    if not ref:
        sys.exit(f"no {ref_label}.json in {outdir}")
    rows = []
    for label, r in runs.items():
        if label == ref_label:
            continue
        for name, g in r["greedy"].items():
            rg = ref["greedy"].get(name)
            if not rg:
                continue
            row = {"config": label, "prompt": name, "exact": g["reply"] == rg["reply"]}
            if rg["tokens"] and g["tokens"]:
                i = first_diff(rg["tokens"], g["tokens"])
                row["first_token_diff"] = i
                row["of"] = len(rg["tokens"])
                if i is not None and i < len(rg["lp"]) and i < len(g["lp"]):
                    top = rg["lp"][i].get("top_logprobs") or []
                    row["ref_margin_nats"] = round(top[0]["logprob"] - top[1]["logprob"], 4) if len(top) > 1 else None
                    row["ref_token"], row["other_token"] = rg["lp"][i]["token"], g["lp"][i]["token"]
                    other = {t["token"]: t["logprob"] for t in g["lp"][i].get("top_logprobs") or []}
                    row["other_lp_of_ref_token"] = other.get(rg["lp"][i]["token"])
                    row["other_margin_nats"] = round(g["lp"][i]["logprob"] - other[rg["lp"][i]["token"]], 4) if rg["lp"][i]["token"] in other else None
            else:
                a, b = json.dumps(rg["reply"]), json.dumps(g["reply"])
                row["first_char_diff"] = first_diff(a, b)
            rows.append(row)
    with open(os.path.join(outdir, "compare.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for row in rows:
        print(json.dumps(row))
    return rows


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "compare"):
        sys.exit(__doc__)
    if a[0] == "compare":
        compare(a[1], a[2] if len(a) > 2 else "default")
        return
    opt = lambda k: a[a.index(k) + 1]
    res = run(opt("--base"), opt("--label"), opt("--out"), "--greedy-only" in a, "--no-cache" in a)
    sys.exit(1 if any(c["ok"] is False for c in res["checks"]) else 0)


if __name__ == "__main__":
    main()
