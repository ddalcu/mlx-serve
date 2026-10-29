#!/bin/bash
# test_model_card_frontmatter.sh — static checks for the HF model cards our converters generate (the `README`
# constant read with `ast`, never imported: the converters import mlx): the frontmatter parses, declares `base_model`,
# `base_model_relation: quantized` (HF DEFAULTS a missing relation to `finetune`, silently) and an explicit
# `library_name` (else a diffusers-shaped repo advertises a from_pretrained snippet that cannot load it).
#
#   ./tests/test_model_card_frontmatter.sh   # hermetic: no weights, no network

set -u
cd "$(dirname "$0")/.." || exit 1

# Converters that embed a README template as a module-level constant.
#
# DISCOVERED, not listed. A hand-maintained list is the same class of bug this
# file exists to catch: `convert_kokoro_weights.py` was published-ready with no
# card at all and the list simply did not mention it, so the guard passed 4/4
# while saying nothing. Anything matching tests/convert_*.py that defines a
# README is now checked automatically, and the converters that have NO card are
# named at the end so the gap stays visible instead of silent.
CONVERTERS=()
UNCARDED=()
for f in tests/convert_*.py; do
  if python3 - "$f" <<'PROBE'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
has = any(
    isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "README" for t in n.targets)
    for n in tree.body
)
sys.exit(0 if has else 1)
PROBE
  then
    CONVERTERS+=("$f")
  else
    UNCARDED+=("$f")
  fi
done

PASS=0; FAIL=0
pass() { PASS=$((PASS+1)); }
fail() { FAIL=$((FAIL+1)); echo "FAIL: $1"; }

for script in "${CONVERTERS[@]}"; do
  if [ ! -f "$script" ]; then
    fail "$script: not found"
    continue
  fi

  # Pull the README constant by PARSING the source (ast), never importing it —
  # these scripts import mlx at module level.
  fm=$(python3 - "$script" <<'PY'
import ast, sys

src = open(sys.argv[1]).read()
tree = ast.parse(src)
readme = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "README":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    readme = node.value.value
if readme is None:
    sys.exit("no README string constant")

lines = readme.splitlines()
if not lines or lines[0].strip() != "---":
    sys.exit("README does not open with frontmatter")
end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
if end is None:
    sys.exit("frontmatter not terminated")
print("\n".join(lines[1:end]))
PY
  )
  if [ $? -ne 0 ]; then
    fail "$script: $fm"
    continue
  fi
  pass  # frontmatter parsed

  key() { echo "$fm" | grep -E "^$1:" | head -1 | sed "s/^$1:[[:space:]]*//"; }

  if [ -n "$(key base_model)" ]; then pass; else
    fail "$script: frontmatter declares no base_model"
  fi

  rel=$(key base_model_relation)
  if [ "$rel" = "quantized" ]; then pass; else
    fail "$script: base_model_relation is '${rel:-<missing>}', expected 'quantized' (HF defaults a missing relation to 'finetune')"
  fi

  if [ -n "$(key library_name)" ]; then pass; else
    fail "$script: frontmatter declares no library_name (a diffusers-shaped repo auto-advertises an unusable diffusers snippet)"
  fi
done

if [ "${#UNCARDED[@]}" -gt 0 ]; then
  echo
  echo "NOTE: ${#UNCARDED[@]} converter(s) generate no model card (not published as HF repos):"
  for f in "${UNCARDED[@]}"; do echo "  - $f"; done
  echo "  Add a README constant before publishing any of them."
fi

echo
echo "checked ${#CONVERTERS[@]} card(s)"
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
