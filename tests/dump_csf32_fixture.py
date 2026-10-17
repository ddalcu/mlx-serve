"""Writes src/fixtures/csf32_plane.bin: one csf_32x2_v1 NVFP4 scale plane built by FlashInfer's
reference (`flashinfer/experimental/b12x/loader/_csf_blocks.py`) and its decoded bytes.
Clipped tiles (5 row blocks, 67 columns) and two exception pages.
Usage: python3 tests/dump_csf32_fixture.py <dir holding _csf_blocks.py> src/fixtures/csf32_plane.bin"""
import struct
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import _csf_blocks as csf  # noqa: E402

rows, cols = 160, 67
rng = np.random.default_rng(0x5CF)
bases = rng.integers(0, 241, rows, dtype=np.uint8)
fields = rng.integers(0, 16, (rows, cols), dtype=np.uint8)
positions = np.sort(rng.choice(rows * cols, 400, replace=False))
values = rng.integers(0, 256, positions.size, dtype=np.uint8)
plane = csf._from_logical_components(bases, fields, positions, values, "nvfp4")
plane.validate()
logical = (bases[:, None].astype(np.int32) + fields).astype(np.uint8)
logical.ravel()[positions] = values
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("<4I", rows, cols, plane.offsets.size, plane.exceptions.size))
    for a in (plane.bases, plane.codes, plane.offsets, plane.exceptions, logical):
        f.write(np.ascontiguousarray(a).tobytes())
