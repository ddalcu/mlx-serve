"""The 8-bit trunk as tests/convert_inkling2.py --bits 8 makes it from the Inkling 2 release, for
EVERY tensor it quantizes: its own sanitize and quantize (group 64, on the default device, the GPU, as the converter
runs), in memory, nothing saved. Writes each quantized module's weight / scales / biases sha256, shape and dtype to a
json the plugin's TRUNK_Q8 test compares against.

Needs mlx and numpy (an mlx-lm environment). The json goes to the tests as INKLING2_Q8_REF.

usage: python -I tests/q8ref_conv.py <repo root> <release dir> <out.json>
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

tree, src, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("convert_inkling2", tree / "tests" / "convert_inkling2.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

weights = {}
for shard in sorted(src.glob("model-*.safetensors")):
    weights.update(mx.load(str(shard)))
weights = conv.sanitize(weights)
names = sorted(k for k in weights if conv.QUANTIZED.match(k))
ends = {"model.language_model.embed_tokens": 8, "lm_head": 8}
out = {}
for k in names:
    q = conv.quantize({k: weights[k]}, 8, (), {}, ends)
    base = k.rsplit(".", 1)[0]
    rec = {}
    for leaf in ("weight", "scales", "biases"):
        a = q[f"{base}.{leaf}"]
        mx.eval(a)
        raw = a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
        rec[leaf] = {"sha256": hashlib.sha256(np.array(raw).tobytes()).hexdigest(), "shape": list(a.shape), "dtype": str(a.dtype)}
    out[base] = rec
    del q
print(f"{len(out)} quantized modules", flush=True)
out_path.write_text(json.dumps(out, indent=1, sort_keys=True))
