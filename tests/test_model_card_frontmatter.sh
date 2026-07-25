#!/bin/bash
# test_model_card_frontmatter.sh — static checks for the HF model cards our
# converter scripts generate.
#
# Why this exists: Hugging Face DEFAULTS a missing `base_model_relation` to
# `finetune`. A quantized mirror that omits the field is therefore published as
# a finetune of its base model — nothing warns, the card renders fine, and the
# repo lands in the wrong list on the base model's page. Both Mage-Flow mirrors
# shipped that way (live, 2026-07-25).
#
# Verifies, without running the converter (it needs mlx + a ~16 GB checkpoint):
#   1. every generated card's frontmatter is parseable YAML-ish key/value
#   2. it declares `base_model`
#   3. it declares `base_model_relation: quantized` — these scripts quantize,
#      they never train
#   4. it declares an explicit `library_name`, so a diffusers-shaped repo does
#      not auto-advertise a `DiffusionPipeline.from_pretrained` snippet that
#      cannot work against pruned, quantized weights
#
# Usage: ./tests/test_model_card_frontmatter.sh

set -u
cd "$(dirname "$0")/.." || exit 1

# converter scripts that embed a README template as a module-level constant
CONVERTERS=(
  tests/convert_mageflow_weights.py
)

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

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
