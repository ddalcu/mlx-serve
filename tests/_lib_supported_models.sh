#!/bin/bash
# Shared helper for multi-model tests: enumerate model subdirectories under a
# root, filtering out any whose `config.json` declares a `model_type` we don't
# support. Keeps the multi-model test family from picking up partially-
# downloaded or wrong-arch checkpoints (e.g. a `deepseek_v4` directory that
# would crash mlx-serve on tokenizer load).
#
# Must stay in sync with `supported_model_types` in src/model_discovery.zig
# and the `model_type` branches in src/model.zig:parseConfigFromJson.
#
# Usage:
#   source "$(dirname "$0")/_lib_supported_models.sh"
#   readarray -t MODELS < <(list_supported_models "$ROOT" [count])
# 'count' is optional; omitted = all.

list_supported_models() {
    local root="$1"
    local limit="${2:-}"
    python3 - "$root" "$limit" <<'PY'
import json, os, sys
root, limit = sys.argv[1], sys.argv[2]
supported = {
    "gemma3", "gemma4", "gemma4_text",
    "qwen3", "qwen3_5", "qwen3_5_text",
    "qwen3_5_moe", "qwen3_5_moe_text",
    "qwen3_next",
    "llama", "mistral",
    "nemotron_h", "bert",
}
# lfm2 is a prefix match — any model_type starting with 'lfm2' is supported.
out = []
try:
    entries = sorted(os.listdir(root))
except OSError:
    sys.exit(0)
for name in entries:
    if name.startswith("."):
        continue
    cfg = os.path.join(root, name, "config.json")
    if not os.path.isfile(cfg):
        continue
    try:
        with open(cfg) as f:
            data = json.load(f)
        mt = data.get("model_type", "")
        q = data.get("quantization") or {}
        qmode = q.get("mode")
    except Exception:
        continue
    if mt not in supported and not mt.startswith("lfm2"):
        continue
    # If quantized, only affine is supported. Skips nvfp4 / other non-MLX
    # quants — they share the gemma/qwen model_type but use a weight layout
    # mlx-serve's safetensors loader can't decode.
    if qmode is not None and qmode != "affine":
        continue
    out.append(name)
n = int(limit) if limit else len(out)
for name in out[:n]:
    print(name)
PY
}
