# Metal tracing for performance work

`./tests/bench.sh` / llmprobe say HOW FAST. A Metal System Trace says WHERE the GPU time goes: which kernels, how many dispatches, whether the GPU is saturated or starved. Use it before guessing at a fusion, and again after one to see the kernel actually shrink.

Everything here is headless: no Xcode window, no sudo, and the output is text an agent can read. Tested on macOS 27 / Xcode `xctrace` 27.0 / M5 Ultra.

**It uses the GPU.** Do not trace while someone is benchmarking on the same Mac, and never quote tok/s from a traced run (tracing perturbs timing). Traces are for attribution only; the numbers that count come from `bench.sh`.

## 1. Record

```bash
xcrun xctrace record --template "Metal System Trace" --instrument "Metal GPU Counters" \
  --output /tmp/t.trace --time-limit 40s --no-prompt \
  --launch -- zig-out/bin/mlx-serve --model <dir> --prompt "..." --max-tokens 128 --temp 0
```

- **Build with `zig build -Doptimize=ReleaseFast`** first (see AGENTS.md "Building"); a Debug binary traces fake hotspots.
- **`--instrument "Metal GPU Counters"` is required for kernel names.** The stock template alone records encoder timing but leaves the shader-timeline table empty. (Patching the template plist's `shaderprofiler` key did NOT take effect; do not retry it.)
- **One-shot `--prompt` mode is the verified recipe.** The target exits, xctrace stops and saves. Keep runs short: a 128-token run is ~50 MB, a 45 s `--serve` run was ~240 MB.
- **`--serve` mode is NOT verified for kernel names.** Launching `--serve` under xctrace (driven with curl, ended by the time limit or SIGINT) produced the encoder table (dispatch counts, busy/idle gaps) but ZERO shader-timeline samples, twice. Whether `--attach <pid>` to a warm server fixes it is untested. If you need kernel attribution for a server-only path, reproduce it as a `--prompt` run first.
- The trace records every process's GPU work; the analyzer filters to one (`--proc`, default `mlx-serve`).

## 2. Read it

Export is `xcrun xctrace export --input t.trace --xpath '/trace-toc/run[1]/data/table[@schema="<name>"]'` (XML; `--toc` lists the tables). The XML dedups values through `id`/`ref` attributes, so index every `id` before reading rows. The script below does both and prints the two summaries that matter:

| Table | Gives you |
|---|---|
| `metal-gpu-intervals` | one row per compute encoder: start, duration. Busy %, idle gaps, encoders per step. Labels are anonymous (`Compute Command 0`): MLX does not label encoders. |
| `metal-shader-profiler-intervals` | per-kernel samples with names (`affine_qmv_fast_…`, `sdpa_vector_…`, our `custom_kernel_mlxserve_*`). |
| `metal-driver-intervals`, `metal-application-command-buffer-submissions` | CPU-side submit/driver cost, command-buffer counts. |

Hardware counters (occupancy, ALU/bandwidth limiters) are NOT usable yet: `gpu-counter-value` fills with samples but the counter-name table came back empty, so the values cannot be told apart.

### Analyzer

Save as `mtrace.py` anywhere (e.g. `~/claude-tmp/`); `python3 mtrace.py /tmp/t.trace --top 20 [--from S --to S]`.

```python
#!/usr/bin/env python3
"""Summarize a Metal System Trace: GPU busy/gaps and per-kernel time for one process."""
import argparse, collections, re, subprocess, sys
import xml.etree.ElementTree as ET


def table(trace, schema, proc):
    xml = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", trace, "--xpath",
         f'/trace-toc/run[1]/data/table[@schema="{schema}"]'],
        capture_output=True, text=True, check=True).stdout
    root = ET.fromstring(xml)
    cols = [c.findtext("mnemonic") for c in root.find(".//schema").findall("col")]
    refs = {e.get("id"): e for e in root.iter() if e.get("id")}
    for r in root.iter("row"):
        d = {n: (refs[e.get("ref")] if e.get("ref") else e) for n, e in zip(cols, list(r))}
        if proc in (d["process"].get("fmt") or ""):
            yield d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--proc", default="mlx-serve")
    ap.add_argument("--from", dest="t0", type=float, default=0.0, help="window start, seconds")
    ap.add_argument("--to", dest="t1", type=float, default=1e9, help="window end, seconds")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    lo, hi = a.t0 * 1e9, a.t1 * 1e9

    iv = sorted((int(d["start"].text), int(d["start"].text) + int(d["duration"].text))
                for d in table(a.trace, "metal-gpu-intervals", a.proc)
                if d["channel-name"].get("fmt") == "Compute" and lo <= int(d["start"].text) <= hi)
    if iv:
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
            else: merged.append([s, e])
        span = merged[-1][1] - merged[0][0]
        busy = sum(e - s for s, e in merged)
        gaps = sorted((merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)), reverse=True)
        print(f"encoders={len(iv)} span={span/1e6:.1f}ms gpu-busy={100*busy/span:.1f}% "
              f"idle-gaps={len(gaps)} biggest(ms)={[round(g/1e6, 1) for g in gaps[:5]]}")

    agg = collections.defaultdict(lambda: [0, 0])
    for d in table(a.trace, "metal-shader-profiler-intervals", a.proc):
        if lo <= int(d["start"].text) <= hi:
            k = re.sub(r" \(\d+\)$", "", d["name"].get("fmt") or "?")
            agg[k][0] += 1; agg[k][1] += int(d["duration"].text)
    tot = sum(v[1] for v in agg.values())
    if not tot:
        sys.exit("no shader-timeline samples: record with --instrument 'Metal GPU Counters'")
    print(f"kernels={len(agg)} sampled={tot/1e6:.1f}ms (a SAMPLE of GPU time: rank by %, not by us)")
    for k, (n, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:a.top]:
        print(f"  {100*t/tot:5.1f}%  n={n:5d}  {k[:100]}")


main()
```

Example output (Qwen3.5-2B 4-bit, `--prompt`, 128 tokens):

```
kernels=43 sampled=28.2ms
   54.2%  n=  329  affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0
    9.5%  n=   53  argmax_float32
    8.3%  n=  310  sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks
    4.0%  n=   16  affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true
```

## 3. Interpret it (traps)

- **Kernel numbers are a SAMPLE** (about a tenth of GPU time in the runs above). Compare PERCENTAGES and ranks between an A and a B trace, never microseconds or totals.
- **`gpu-busy` depends on the window.** The default span includes model load, warmup and the idle tail, so it read 16.8% on one run and 66.3% on another for the same command. Use `--from/--to` to cut to the decode (or prefill) window before reading it. Low busy inside a pure decode window = launch-bound (dispatch count and CPU graph build); high busy = kernel-time-bound (look at the kernel table). Compare against the byte floor before calling anything bandwidth-bound.
- **Encoders per step is a dispatch-cost signal.** Fewer, fatter encoders/kernels per token is the win this repo keeps measuring (see the "custom `metal_kernel` dispatch costs" rules in AGENTS.md); the trace shows whether a fusion removed dispatches or only moved time.
- **A/B discipline is unchanged**: same model, same prompt, same flags, both traces taken in the same session, and engagement lines from the server log to prove the arm under test actually ran. The trace is evidence for WHERE; `bench.sh` decides WHETHER.
- Kernel names carry the template parameters (dtype, group size, bits, tile sizes): the name tells you which lane (`affine_qmv_fast` vs `affine_qmm_t_splitk`, our `custom_kernel_mlxserve_*`) a shape landed on, which is often the whole answer.

## 4. Not covered

- **`.gputrace` frame captures** (`MTL_CAPTURE_ENABLED=1`, shader-level counters/ISA in Xcode) stay GUI-only; there is no scripted reader.
- **`metalperftrace collect|overview`** exists on macOS 27 and has not been tried here.
- **Hardware counters and `--serve` kernel attribution**: see the notes in section 1 and 2; both are open.
