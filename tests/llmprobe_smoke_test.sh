#!/bin/bash
# llmprobe_smoke_test.sh — birds-eye conformance + speed across every MLX model
# family this machine has, under each speculative-decoding config.
#
# What it does
#   1. Boots ONE headless server over --model-dir, reads /v1/models, and groups
#      what it finds into families (gemma4, gemma4-moe, qwen3_5, qwen3_5-moe,
#      gemma3, ...). Picks one representative per family. GGUF (llama.cpp /
#      ds4), encoders and media models are out of scope — Zig MLX only.
#   2. For each representative, boots a server per CELL and runs `npx llmprobe`:
#        none      --no-pld --no-mtp       (PLD is default-ON: a real baseline
#                                           needs BOTH flags)
#        pld       --pld --no-mtp
#        mtp       --no-pld --mtp          (only when the dir ships an MTP head;
#                                           --mtp is what makes the cell real on
#                                           a MoE target, whose per-request
#                                           default is OFF)
#        drafter   --drafter <assistant>   (Gemma 4 dense only)
#      Plus, on the ONE fastest model, a KV-quant crash check at its default
#      spec config: --kv-quant 4.
#   3. Scrapes each server log for `[spec-stats] mode=` — throughput alone cannot
#      prove a spec path engaged (the regular-decode fallback is output-identical;
#      that is how a hardcoded use_drafter=false survived a month).
#   4. Prints a birds-eye table + flags any cell whose conformance dropped
#      against its own family's baseline.
#
# Usage
#   ./tests/llmprobe_smoke_test.sh
#   SMOKE_MODELS=mlx-community/gemma-4-e4b-it-4bit ./tests/llmprobe_smoke_test.sh
#   SMOKE_DEPTH=default ./tests/llmprobe_smoke_test.sh   # + capability evals (slower)
#   SMOKE_MAX_GB=200 ./tests/llmprobe_smoke_test.sh      # let the giants in
#   SMOKE_PLAN_ONLY=1 ./tests/llmprobe_smoke_test.sh     # print the plan, run nothing
#   SMOKE_CELLS=mtp ./tests/llmprobe_smoke_test.sh       # re-run one cell everywhere
#
# Env: BINARY PORT MODEL_DIR SMOKE_DEPTH SMOKE_MODELS SMOKE_CELLS SMOKE_MAX_GB
#      SMOKE_TIMEOUT SMOKE_CTX SMOKE_PREFIX_CACHE
#      LLMPROBE (default: `npx --yes llmprobe`)
#
# Output: tests/llmprobe-smoke-results/ (per-cell .json + .server.log, rows.json,
# summary.tsv). Exit 1 if any cell failed.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

BINARY="${BINARY:-$REPO/zig-out/bin/mlx-serve}"
PORT="${PORT:-11299}"
MODEL_DIR="${MODEL_DIR:-$HOME/.mlx-serve/models}"
DEPTH="${SMOKE_DEPTH:-quick}"        # quick | default | full
MAX_BYTES=$(( ${SMOKE_MAX_GB:-60} * 1000000000 ))
TIMEOUT="${SMOKE_TIMEOUT:-180}"      # per-request timeout handed to llmprobe
LLMPROBE="${LLMPROBE:-npx --yes llmprobe}"
# Every family on the same footing, and comfortably above llmprobe's context
# ladder (its top rung is a ~16.4k-token prompt — a 16384 context would 400 on
# it and the run would look like an engine failure). SMOKE_CTX=0 → omit the
# flag and let auto-context pin whatever this machine can hold.
CTX="${SMOKE_CTX:-32768}"
RESULTS="$REPO/tests/llmprobe-smoke-results"
LIB="$REPO/tests/llmprobe_smoke_lib.mjs"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'

mkdir -p "$RESULTS"
[[ -x "$BINARY" ]] || { echo "[fatal] $BINARY not found — zig build -Doptimize=ReleaseFast"; exit 1; }
command -v node >/dev/null || { echo "[fatal] node is required (llmprobe + the planner)"; exit 1; }

kill_servers() {
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
    for _ in $(seq 1 20); do
        pgrep -f "mlx-serve.*--port $PORT" >/dev/null || return 0
        sleep 0.5
    done
    pkill -9 -f "mlx-serve.*--port $PORT" 2>/dev/null || true
}
trap 'kill_servers' EXIT

# boot <logfile> <extra flags...> ; 0 = healthy and the model is resident
boot() {
    local logf="$1"; shift
    "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --log-level info "$@" >"$logf" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 600); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
        kill -0 "$SERVER_PID" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

# ── 1. Discovery: one headless boot, read the registry, plan the matrix ──────
kill_servers
echo -e "${YELLOW}== discovering models in $MODEL_DIR ==${NC}"
if ! boot "$RESULTS/discovery.server.log" --model-dir "$MODEL_DIR"; then
    echo -e "${RED}[fatal] headless server failed to boot${NC}"; tail -n 20 "$RESULTS/discovery.server.log"; exit 1
fi
curl -sf "http://127.0.0.1:$PORT/v1/models" -o "$RESULTS/models.json" || {
    echo -e "${RED}[fatal] /v1/models failed${NC}"; exit 1; }
kill_servers

node "$LIB" plan \
    --models "$RESULTS/models.json" \
    --dir "$MODEL_DIR" \
    --max-bytes "$MAX_BYTES" \
    --only "${SMOKE_MODELS:-}" > "$RESULTS/plan.json" || { echo "[fatal] planning failed"; exit 1; }

node -e '
const p = require("./tests/llmprobe-smoke-results/plan.json");
for (const m of p.plan) {
  const gb = (m.bytes / 1e9).toFixed(1);
  console.log(`  ${m.family.padEnd(16)} ${m.id}  (${gb} GB)`);
  console.log(`    cells: ${m.cells.map((c) => c.id).join(", ")}`);
}
if (!p.plan.length) console.log("  (nothing probeable)");
const dropped = p.skipped.filter((s) => !/same family/.test(s.reason));
if (dropped.length) {
  console.log("  skipped:");
  for (const s of dropped) console.log(`    - ${s.id}: ${s.reason}`);
}
'
CELL_COUNT=$(node -e 'const p=require("./tests/llmprobe-smoke-results/plan.json");console.log(p.plan.reduce((n,m)=>n+m.cells.length,0))')
echo -e "${DIM}  depth=$DEPTH · $CELL_COUNT cells total${NC}"
[[ "${SMOKE_PLAN_ONLY:-}" == "1" ]] && exit 0
[[ "$CELL_COUNT" -eq 0 ]] && { echo -e "${RED}nothing to probe${NC}"; exit 1; }

# ── 2. Run the matrix, one server boot per cell ──────────────────────────────
: > "$RESULTS/rows.jsonl"
overall_fail=0

# Unit-separator, NOT tab: tab is an IFS *whitespace* char, so bash collapses a
# run of them — an empty expectSpec field would vanish and shift `cell_flags`
# into it, silently launching the cell with NO flags. (Caught live: the "no
# speculation" cell reported pld×7 engagements.)
while IFS=$'\x1f' read -r family model_id model_dir cell_id cell_label expect_spec expect_no_spec cell_flags; do
    tag="${family}.${cell_id}"
    echo -e "\n${YELLOW}== $family · $model_id · [$cell_id] $cell_label ==${NC}"
    server_log="$RESULTS/$tag.server.log"
    probe_json="$RESULTS/$tag.json"
    probe_log="$RESULTS/$tag.probe.log"

    ctx_flag=()
    [[ "$CTX" != "0" ]] && ctx_flag=(--ctx-size "$CTX")

    # The hot prefix cache makes llmprobe's prefill/TTFT numbers meaningless: it
    # re-sends the SAME 2032-token benchmark prompt (warmup + median of 3), so
    # after the warmup every run is a cache hit and the reported rate is the
    # cache's, not the engine's (measured: 145,000 tok/s prefill, 16 ms TTFT on
    # an E2B). Same class as the metrics-panel bug where the numerator counted
    # tokens the denominator never spent time on. Turn it off so the row means
    # what it says; SMOKE_PREFIX_CACHE=1 keeps it on if you want that config.
    cache_flag=()
    [[ "${SMOKE_PREFIX_CACHE:-0}" == "0" ]] && cache_flag=(--prefix-cache-entries 0)

    # shellcheck disable=SC2086  # cell_flags is a deliberate word-split
    if ! boot "$server_log" --model "$model_dir" "${ctx_flag[@]}" "${cache_flag[@]}" $cell_flags; then
        echo -e "${RED}  server failed to boot${NC}"; tail -n 12 "$server_log"
        node -e '
          const [f,m,c,l,log] = process.argv.slice(1);
          console.log(JSON.stringify({family:f, model:m, cell:c, label:l,
            result:{error:"server failed to boot — see "+log},
            verdict:{ok:false, reasons:["server failed to boot"]}}));
        ' "$family" "$model_id" "$cell_id" "$cell_label" "$server_log" >> "$RESULTS/rows.jsonl"
        overall_fail=1
        kill_servers
        continue
    fi

    depth_flag=""
    [[ "$DEPTH" == "quick" ]] && depth_flag="--quick"
    [[ "$DEPTH" == "full" ]] && depth_flag="--full"

    # shellcheck disable=SC2086
    $LLMPROBE "http://127.0.0.1:$PORT" $depth_flag --bench --no-color \
        --timeout "$TIMEOUT" --save "$probe_json" > "$probe_log" 2>&1
    probe_rc=$?

    # The server must still be alive: a crash during the probe (the whole point
    # of the KV-quant cells) is the failure this harness exists to catch.
    alive=1; kill -0 "$SERVER_PID" 2>/dev/null || alive=0
    specs=$(grep -c '\[spec-stats\] mode=' "$server_log" 2>/dev/null || echo 0)
    kill_servers

    node --input-type=module -e '
      import { readFileSync, existsSync } from "node:fs";
      import { parseSpecStats, cellVerdict } from "./tests/llmprobe_smoke_lib.mjs";
      const [family, model, cell, label, expectSpec, expectNoSpec, serverLog, probeJson, rc, alive] =
        process.argv.slice(1);

      const stats = parseSpecStats(existsSync(serverLog) ? readFileSync(serverLog, "utf8") : "");
      let result;
      if (alive !== "1") {
        result = { error: `server died during the probe (see ${serverLog})` };
      } else if (!existsSync(probeJson)) {
        result = { error: `llmprobe produced no report (rc=${rc})` };
      } else {
        const r = JSON.parse(readFileSync(probeJson, "utf8"));
        // --quick skips the capability evals entirely, so llmprobe reports 0% —
        // "not measured", not "the model failed everything". Null it out rather
        // than print a 0 that reads as a catastrophic score.
        const measuredCapability = (r.capability?.evals?.length ?? 0) > 0;
        result = {
          conformance: { passed: r.conformance.passed, total: r.conformance.total },
          capability: measuredCapability ? { pct: r.capability.pct } : null,
          coverage: r.coverage?.byTier ?? null,
          bench: r.bench ?? null,
        };
        if (Number(rc) !== 0) result.error = `llmprobe exited rc=${rc}`;
      }
      const c = { id: cell, expectSpec: expectSpec || null, expectNoSpec: expectNoSpec === "1" };
      const verdict = cellVerdict(c, result, stats);
      console.log(JSON.stringify({ family, model, cell, label, specStats: stats, result, verdict }));
    ' "$family" "$model_id" "$cell_id" "$cell_label" "$expect_spec" "$expect_no_spec" \
      "$server_log" "$probe_json" "$probe_rc" "$alive" >> "$RESULTS/rows.jsonl"

    tail -n1 "$RESULTS/rows.jsonl" | node --input-type=module -e '
      let s=""; for await (const c of process.stdin) s+=c;
      const r = JSON.parse(s);
      const d = r.result?.bench?.decodeTokPerSec?.median;
      const conf = r.result?.conformance;
      const head = r.verdict.ok ? "\x1b[0;32mPASS\x1b[0m" : "\x1b[0;31mFAIL\x1b[0m";
      console.log(`  ${head}  ${conf ? `conformance ${conf.passed}/${conf.total}` : ""}` +
                  `${d ? `  ·  ${d} tok/s` : ""}`);
      for (const reason of r.verdict.reasons) console.log(`        ${reason}`);
      if (!r.verdict.ok) process.exit(1);
    ' || overall_fail=1
done < <(node -e '
    const p = require("./tests/llmprobe-smoke-results/plan.json");
    const only = (process.env.SMOKE_CELLS ?? "").split(",").map((s) => s.trim()).filter(Boolean);
    for (const m of p.plan)
      for (const c of m.cells) {
        if (only.length && !only.includes(c.id)) continue;
        console.log([m.family, m.id, m.dir, c.id, c.label, c.expectSpec ?? "",
                     c.expectNoSpec ? "1" : "", c.flags.join(" ")].join("\x1f"));
      }
')

# ── 3. Birds eye ────────────────────────────────────────────────────────────
node -e '
  const fs = require("node:fs");
  const rows = fs.readFileSync("tests/llmprobe-smoke-results/rows.jsonl", "utf8")
    .trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));
  fs.writeFileSync("tests/llmprobe-smoke-results/rows.json", JSON.stringify(rows, null, 2));
'
node "$LIB" summarize --rows "$RESULTS/rows.json" > "$RESULTS/summary.json"

echo ""
echo "================= llmprobe smoke — birds eye ================="
node -e '
  const rows = require("./tests/llmprobe-smoke-results/rows.json");
  const s = require("./tests/llmprobe-smoke-results/summary.json");
  const n = (v, d = 1) => (v == null ? "—" : Number(v).toFixed(d));

  const H = ["family", "cell", "conf", "capab", "decode", "ttft", "prefill", "spec", "engaged", ""];
  const body = rows.map((r) => {
    const b = r.result?.bench ?? {};
    const c = r.result?.conformance;
    const engaged = Object.entries(r.specStats ?? {}).filter(([, v]) => v > 0)
      .map(([k, v]) => `${k}×${v}`).join(" ") || "—";
    return [
      r.family, r.cell,
      c ? `${c.passed}/${c.total}` : "—",
      r.result?.capability?.pct != null ? `${r.result.capability.pct}%` : "—",
      n(b.decodeTokPerSec?.median), n(b.ttftMs?.median, 0),
      n(b.prefillTokPerSec?.median, 0),
      b.speculative ? `${n(b.speculative.ratio, 2)}×` : "—",
      engaged,
      r.verdict.ok ? "ok" : "FAIL",
    ];
  });
  const w = H.map((_, i) => Math.max(H[i].length, ...body.map((r) => String(r[i]).length)));
  const line = (r) => r.map((v, i) => String(v).padEnd(w[i])).join("  ");
  console.log(line(H));
  console.log(w.map((x) => "-".repeat(x)).join("  "));
  for (const r of body) console.log(line(r));

  console.log("");
  for (const f of s.families) {
    const sp = f.speedup && f.speedup !== 1 ? ` (${f.speedup}× over no-spec)` : "";
    console.log(`  ${f.family.padEnd(18)} best: ${String(f.bestCell).padEnd(9)} ${n(f.bestDecode)} tok/s${sp}`);
  }
  if (s.regressions.length) {
    console.log("\n  \x1b[0;31mconformance regressions vs the no-spec baseline:\x1b[0m");
    for (const r of s.regressions) console.log(`    - ${r}`);
  }
'

{
    printf "timestamp\tfamily\tmodel\tcell\tconformance\tdecode_tok_s\tttft_ms\tspec_ratio\tengaged\tverdict\n"
    node -e '
      const rows = require("./tests/llmprobe-smoke-results/rows.json");
      const ts = new Date().toISOString();
      for (const r of rows) {
        const b = r.result?.bench ?? {};
        const c = r.result?.conformance;
        const eng = Object.entries(r.specStats ?? {}).filter(([, v]) => v > 0).map(([k, v]) => `${k}x${v}`).join(",");
        console.log([ts, r.family, r.model, r.cell,
          c ? `${c.passed}/${c.total}` : "", b.decodeTokPerSec?.median ?? "",
          b.ttftMs?.median ?? "", b.speculative?.ratio ?? "", eng,
          r.verdict.ok ? "ok" : "FAIL"].join("\t"));
      }
    '
} > "$RESULTS/summary.tsv"

echo ""
echo "  reports: $RESULTS  (rows.json · summary.tsv · per-cell .json + .server.log)"
if [[ "$overall_fail" -eq 0 ]]; then
    echo -e "${GREEN}ALL GREEN${NC}"
else
    echo -e "${RED}FAILURES — see the table above${NC}"
fi
exit "$overall_fail"
