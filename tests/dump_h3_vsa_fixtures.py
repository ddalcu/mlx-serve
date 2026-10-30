#!/usr/bin/env python3
"""Oracle for src/plugins/minimax_h3/vsa.zig: FastVideo's own MLX VSA-H3 reference
(`fastvideo/mlx_runtime/minimax_h3_vsa.py`, `impl="reference"`), run on the CPU
over a small packed sequence shaped like a FastH3 Omni request: text, an image
reference, a reference video with its soundtrack, the target audio and a
ragged target video, with the trained gate branch.

    uv run --with mlx --with numpy python -I tests/dump_h3_vsa_fixtures.py \\
        <FastVideo>/fastvideo/mlx_runtime/minimax_h3_vsa.py <out.safetensors>

`H3_VSA_FIXTURE=<out.safetensors> zig build test -Dtest-filter="h3 vsa"` reads it.
"""
import importlib.util
import logging
import sys
import types

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

# The module imports only `fastvideo.logger.init_logger`.
logger_mod = types.ModuleType("fastvideo.logger")
logger_mod.init_logger = logging.getLogger
sys.modules["fastvideo"] = types.ModuleType("fastvideo")
sys.modules["fastvideo.logger"] = logger_mod
spec = importlib.util.spec_from_file_location("minimax_h3_vsa", sys.argv[1])
vsa = importlib.util.module_from_spec(spec)
sys.modules["minimax_h3_vsa"] = vsa
spec.loader.exec_module(vsa)

# text | image ref | ref soundtrack | ref video (4,4,12) | target audio | target video (6,6,10)
SEGMENTS = (37, 20, 10, (4, 4, 12), 6, (6, 6, 10))
SPARSITY, KEEP = 0.75, 0.5
H, D = 2, 128

geo = vsa.build_h3_segment_geometry(SEGMENTS, 128)
rng = np.random.default_rng(7)
q, k, v, gate = (mx.array(rng.standard_normal((geo.total_seq_length, H, D)).astype(np.float32)) for _ in range(4))
out = vsa.h3_vsa_attention(q, k, v, geo, sparsity=SPARSITY, exempt=True, gate_compress=gate,
                           impl="reference", ref_keep_rate=KEEP)
mx.eval(out)
print(f"seq {geo.total_seq_length}, tiles {geo.num_tiles} ({geo.num_prefix_tiles} dense), "
      f"spans {geo.video_tile_spans}, vbs {geo.variable_block_sizes.tolist()}")
mx.save_safetensors(sys.argv[2], {
    "q": q, "k": k, "v": v, "gate": gate, "out": out,
    "vbs": mx.array(geo.variable_block_sizes.astype(np.int32)),
    "untile": mx.array(geo.untile_combined_index.astype(np.int32)),
})
