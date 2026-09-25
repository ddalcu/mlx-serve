#!/usr/bin/env python3
"""Build a directional-steering bank from two prompt sets.

A bank is one unit-norm direction per trunk layer. This captures the model's own
activations on prompts that SHOW a behaviour and on prompts that do not, takes the
difference of the means, and removes the component the two sets share:

    d = unit(mean(show) - mean(control))        per layer
    d = unit(d - (d . unit(mean(control))) * unit(mean(control)))

The subtraction is what makes it a behaviour direction rather than a "these are prompts"
direction -- both sets are prompts, and without it the shared component dominates.

Nothing here is architecture-specific: the layer count and width come from the dumped
rows, and the server validates the file's SIZE against whatever model is loaded.

The server must be started with a dump directory, and it must be the same directory
passed here:

    MLX_SERVE_STEERING_DUMP_DIR=/tmp/caps mlx-serve --model <pack> --serve --max-concurrent 1

    python3 tests/build_steering_bank.py --dump-dir /tmp/caps \\
        --show tests/fixtures/steering/succinct.txt \\
        --control tests/fixtures/steering/verbose.txt \\
        --name verbosity

Then `mlx-serve steer verbosity --ffn -0.1` for terser answers, `--ffn 0.5` for longer
ones. Stdlib only, on purpose: building a bank should need no environment.
"""

import argparse
import array
import glob
import json
import math
import os
import re
import urllib.error
import urllib.request


def capture(base, model, dump_dir, cid, prompt, component, timeout):
    """One prompt, both arms OFF, and prove the rows on disk are THIS request's.

    A capture id whose directory already exists is refused by the server, but a stale
    directory from an earlier run would otherwise be read as this run's data -- hence
    matching the response echo against the manifest rather than trusting the path.
    """
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0, "stream": False, "enable_thinking": False,
            "steering": {"ffn": 0, "attn": 0}, "steering_capture": cid}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            js = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # The server refuses by NAME; a traceback would bury it.
        detail = e.read().decode(errors="replace")[:400]
        raise SystemExit(f"{cid}: server said {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach {base}: {e.reason}. Start the server with "
                         f"MLX_SERVE_STEERING_DUMP_DIR set, or pass --base.")
    echo = js.get("steering_capture")
    if not echo:
        raise SystemExit(f"{cid}: the server returned no steering_capture echo -- was it "
                         f"started with MLX_SERVE_STEERING_DUMP_DIR?")
    man = os.path.join(dump_dir, cid, "done.json")
    if not os.path.exists(man):
        raise SystemExit(f"{cid}: no manifest at {man} -- is --dump-dir the directory the "
                         f"server was started with?")
    with open(man) as f:
        done = json.load(f)
    if not (echo["id"] == cid == done["capture_id"] and echo["prompt_sha256"] == done["prompt_sha256"]):
        raise SystemExit(f"{cid}: manifest does not match the response -- stale dump directory")
    if js["usage"]["prompt_tokens_details"]["cached_tokens"]:
        raise SystemExit(f"{cid}: capture reused a cached prefix, so the rows are not this "
                         f"prompt's alone")
    files = sorted(glob.glob(os.path.join(dump_dir, cid, f"{component}-*_pos0.bin")),
                   key=lambda f: int(f.split("-")[-1].split("_")[0]))
    if not files:
        raise SystemExit(f"{cid}: no {component} rows were dumped")
    rows = []
    for path in files:
        with open(path, "rb") as f:
            rows.append(array.array("f", f.read()))
    return rows


def mean_rows(sets):
    n_layers, width = len(sets[0]), len(sets[0][0])
    return [[sum(s[l][i] for s in sets) / len(sets) for i in range(width)]
            for l in range(n_layers)]


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def read_prompts(path):
    with open(path) as f:
        return [ln for ln in f.read().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", required=True,
                    help="prompts that SHOW the behaviour, one per line")
    ap.add_argument("--control", required=True,
                    help="prompts that do not, one per line")
    ap.add_argument("--name", help="write into ~/.mlx-serve/steering/<name>.{f32,json}")
    ap.add_argument("--out", help="explicit output path prefix (overrides --name)")
    ap.add_argument("--dump-dir", required=True,
                    help="the directory the server was started with")
    ap.add_argument("--base", default="http://127.0.0.1:11234")
    ap.add_argument("--model", default="mlx-serve", help="model id (default: the server's default)")
    ap.add_argument("--component", default="ffn_out", choices=["ffn_out", "attn_out"],
                    help="ffn_out matches the default --dir-steering-ffn arm")
    ap.add_argument("--run", default="b", help="capture id prefix, so two runs never collide")
    ap.add_argument("--no-orthogonalize", action="store_true",
                    help="keep the raw difference of means")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    if not (a.name or a.out):
        ap.error("one of --name or --out")
    # The server resolves a registry bank by this exact shape; anything else is unloadable.
    if a.name and (not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", a.name) or a.name.endswith(".f32")):
        ap.error("--name must be [A-Za-z0-9._-]{1,64}, without .f32")
    out = a.out or os.path.join(os.path.expanduser("~/.mlx-serve/steering"), a.name)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    show, control = read_prompts(a.show), read_prompts(a.control)
    if not show or not control:
        raise SystemExit("both prompt files need at least one non-empty line")
    print(f"capturing {len(show)} show + {len(control)} control prompts "
          f"({a.component}) from {a.base}", flush=True)
    # Serial on purpose: each capture forces a cold prefill, and the server is expected to
    # be running --max-concurrent 1 for this.
    S = [capture(a.base, a.model, a.dump_dir, f"{a.run}-s{i}", p, a.component, a.timeout)
         for i, p in enumerate(show)]
    C = [capture(a.base, a.model, a.dump_dir, f"{a.run}-c{i}", p, a.component, a.timeout)
         for i, p in enumerate(control)]
    if len({len(x) for x in S + C}) != 1:
        raise SystemExit("captures disagree on layer count")

    ms, mc = mean_rows(S), mean_rows(C)
    flat = array.array("f")
    for l in range(len(ms)):
        d = unit([x - y for x, y in zip(ms[l], mc[l])])
        if not a.no_orthogonalize:
            c = unit(mc[l])
            dot = sum(x * y for x, y in zip(d, c))
            d = unit([x - dot * y for x, y in zip(d, c)])
        flat.extend(d)
    with open(out + ".f32", "wb") as f:
        f.write(flat.tobytes())
    meta = {"format": "directional-steering-v1",
            "shape": [len(ms), len(ms[0])],
            "show": len(S),
            "control": len(C),
            "component": a.component,
            "orthogonalized": not a.no_orthogonalize}
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out}.f32  ({len(ms)} layers x {len(ms[0])}, "
          f"{len(flat) * 4} bytes)\n"
          f"      {out}.json ({json.dumps(meta)})")
    if a.name:
        arm = "attn" if a.component == "attn_out" else "ffn"
        print(f"\ntry it:  mlx-serve steer {a.name} --{arm} 1\n"
              f"         mlx-serve steer off")


if __name__ == "__main__":
    main()
