#!/bin/bash
# pi as a real agent on GLM-5.3-Flash, one server per cell: every thinking level, KV quant,
# prefix cache off and a narrow prefill chunk. Each cell runs a two-turn coding task (read a
# conventions file, write a module + unittest, then edit both) and a hidden checker grades the
# result; tests/analyze_pi_sessions.py then scores early ends, leaks, tool args and prefix reuse.
#
#   tests/glm_pi_matrix.sh                      # every cell
#   CELLS=low,kv8 tests/glm_pi_matrix.sh        # a subset
#   GLM5_PACK=<dir> OUT=<dir> PORT=11367 TIMEOUT_MIN=40 tests/glm_pi_matrix.sh
#
# Per cell under $OUT/<cell>/: server.log (debug: the raw dump is written only then), rawdump.txt
# (MLX_SERVE_RAW_DUMP_FILE), sessions/,
# work/ (the agent's project), turn{1,2}.{out,err}, cell.json, outcome.json. Needs pi on PATH.
# `max` maps pi's max level to reasoning_effort "max" in the generated models.json: the launch
# config maps only `off`, so pi would otherwise clamp max to high.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
source tests/_lib_models.sh

BIN="${BINARY:-$PWD/zig-out/bin/mlx-serve}"
MODEL="${GLM5_PACK:-$(find_model TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP)}"
PORT="${PORT:-11367}"
BASE="http://127.0.0.1:$PORT"
OUT="${OUT:-$HOME/claude-tmp/glm-validation/pi-$(date +%Y%m%d-%H%M%S)}"
TIMEOUT_MIN="${TIMEOUT_MIN:-40}"
[[ -d "$MODEL" ]] || { echo "SKIP: no GLM pack (set GLM5_PACK)"; exit 0; }
[[ -x "$BIN" ]] || { echo "FAIL: $BIN missing — zig build -Doptimize=ReleaseFast"; exit 1; }

# cell|pi thinking level|server flags
ALL=(
    "off|off|"
    "minimal|minimal|"
    "low|low|"
    "medium|medium|"
    "high|high|"
    "max|max|"
    "kv8|low|--kv-quant 8"
    "kv4|low|--kv-quant 4"
    "nocache|high|--prefix-cache-entries 0"
    "chunk2048|medium|--prefill-chunk 2048"
)

TURN1='Read README.md in this directory first and follow its conventions. Create inventory.py with a class Inventory: add(name, qty) adds stock, remove(name, qty) removes stock and raises ValueError when there is not enough, count(name) returns the quantity (0 for unknown items), total() returns the sum of all quantities. Then write test_inventory.py using unittest that covers every method, run it with `python3 -m unittest -v test_inventory`, and fix anything until it passes.'
TURN2='Now extend it: add low_stock(threshold) returning the sorted names whose quantity is below threshold, and make remove() delete an item entirely once its quantity reaches 0 (count() still returns 0). Edit the existing files rather than rewriting them, add tests for both changes, and run the tests until they pass.'
README='# Inventory service

Conventions for this project:

- Every public method has a one-line docstring.
- Quantities are ints; add() and remove() raise ValueError for a quantity <= 0.
- No third-party packages: standard library only.
'
# Hidden grader: the agent never sees it.
CHECK=$(cat <<'PY'
import importlib, inspect, json, subprocess, sys
r = {"checks": {}}
def ok(name, cond):
    r["checks"][name] = bool(cond)
try:
    sys.path.insert(0, ".")
    inv = importlib.import_module("inventory").Inventory
    i = inv(); i.add("bolt", 5); i.add("nut", 2); i.add("bolt", 3)
    ok("add/count", i.count("bolt") == 8 and i.count("nut") == 2 and i.count("gear") == 0)
    ok("total", i.total() == 10)
    i.remove("bolt", 3); ok("remove", i.count("bolt") == 5)
    try: i.remove("nut", 5); ok("remove raises", False)
    except ValueError: ok("remove raises", True)
    for bad in (0, -1):
        try: i.add("x", bad); ok(f"add rejects {bad}", False)
        except ValueError: ok(f"add rejects {bad}", True)
    i.remove("nut", 2)
    ok("remove to zero deletes", i.count("nut") == 0 and "nut" not in (i.low_stock(100) if hasattr(i, "low_stock") else []))
    i.add("washer", 1); i.add("gear", 9)
    ok("low_stock", hasattr(i, "low_stock") and i.low_stock(6) == ["bolt", "washer"])
    pub = [m for n, m in inspect.getmembers(inv, inspect.isfunction) if not n.startswith("_")]
    ok("docstrings", pub and all(inspect.getdoc(m) for m in pub))
except Exception as e:
    r["error"] = f"{type(e).__name__}: {e}"
t = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_inventory"], capture_output=True, text=True, timeout=120)
ok("agent tests pass", t.returncode == 0)
r["agent_tests_tail"] = t.stderr[-400:]
r["pass"] = "error" not in r and all(r["checks"].values())
print(json.dumps(r))
PY
)

SERVER_PID=""
stop() {
    [[ -n "$SERVER_PID" ]] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
}
trap stop EXIT

run_turn() { # $1 cell dir, $2 n, $3 prompt, $4 thinking, $5.. extra pi args
    local dir="$1" n="$2" prompt="$3" level="$4"; shift 4
    printf '%s' "$prompt" > "$dir/turn$n.prompt"
    local t0; t0=$(date +%s)
    (cd "$dir/work" && PI_CODING_AGENT_DIR="$dir/home/.mlx-serve/pi" perl -e 'alarm shift; exec @ARGV or die "exec: $!"' $((TIMEOUT_MIN * 60)) \
        "$REPO/tests/agent_eval/pi_rpc.py" "$dir/turn$n.prompt" --provider mlx --model "$MODEL_ID" --thinking "$level" \
        --session-dir "$dir/sessions" "$@" > "$dir/turn$n.out" 2> "$dir/turn$n.err")
    local rc=$?
    echo "    turn $n: exit=$rc wall=$(( $(date +%s) - t0 ))s"
    return $rc
}

mkdir -p "$OUT"
IFS=',' read -r -a WANT <<< "${CELLS:-}"
for entry in "${ALL[@]}"; do
    IFS='|' read -r cell level flags <<< "$entry"
    if [[ ${#WANT[@]} -gt 0 ]] && ! [[ ",${CELLS}," == *",$cell,"* ]]; then continue; fi
    dir="$OUT/$cell"
    rm -rf "$dir"; mkdir -p "$dir/work" "$dir/sessions" "$dir/home"
    printf '%s' "$README" > "$dir/work/README.md"
    echo "=== $cell  (thinking=$level flags='$flags')"
    t0=$(date +%s)
    # Isolated HOME: model-settings.json would outrank the cell's flags.
    # shellcheck disable=SC2086
    HOME="$dir/home" MLX_SERVE_RAW_DUMP_FILE="$dir/rawdump.txt" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 \
        --port "$PORT" --log-level debug $flags > "$dir/server.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 600); do
        curl -sf "$BASE/v1/models" 2>/dev/null | grep -q '"id"' && break
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    if ! curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
        echo "    boot failed: $(tail -3 "$dir/server.log" | tr '\n' ' ')"
        printf '{"cell":"%s","thinking":"%s","flags":"%s","boot":"failed"}\n' "$cell" "$level" "$flags" > "$dir/cell.json"
        stop; continue
    fi
    MODEL_ID=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    echo "    booted in $(( $(date +%s) - t0 ))s as $MODEL_ID"
    HOME="$dir/home" "$BIN" launch pi --url "$BASE" --model "$MODEL_ID" --print --no-start > "$dir/launch.out" 2>&1 \
        || { echo "    launch pi failed"; stop; continue; }
    if [[ "$level" == max ]]; then
        python3 - "$dir/home/.mlx-serve/pi/models.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
for prov in d["providers"].values():
    for m in prov["models"]:
        m.setdefault("thinkingLevelMap", {}).update({"xhigh": "xhigh", "max": "max"})
json.dump(d, open(p, "w"), indent=2)
PY
    fi
    ctx=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("context_length"))')
    printf '{"cell":"%s","thinking":"%s","flags":"%s","model":"%s","context_length":%s,"started":"%s"}\n' \
        "$cell" "$level" "$flags" "$MODEL_ID" "${ctx:-null}" "$(date +%FT%T)" > "$dir/cell.json"
    run_turn "$dir" 1 "$TURN1" "$level"
    run_turn "$dir" 2 "$TURN2" "$level" --continue
    (cd "$dir/work" && python3 -c "$CHECK") > "$dir/outcome.json" 2>&1 || echo '{"pass": false, "error": "checker crashed"}' > "$dir/outcome.json"
    echo "    task: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("pass" if d.get("pass") else "FAIL", {k:v for k,v in d.get("checks",{}).items() if not v}, d.get("error",""))' "$dir/outcome.json")"
    stop
    sleep 5
done

python3 tests/analyze_pi_sessions.py "$OUT"/*/ --json "$OUT/analysis.json" | tee "$OUT/analysis.md"
