# inkling2

Owns `inkling2_loop` (Thinking Machines Inkling 2 Mini), text only. Packs: `mlx-community/Inkling-2-Mini-4bit`
(affine 4-bit trunk) and mlx-lm-converted 8-bit / bf16 trunks, all with the release's NVFP4 / MXFP8 experts. Oracle:
mlx-lm's `inkling2` (PR #2086), `tests/dump_inkling2_fixtures.py` (`fixtures/tiny`, `tiny_q`; `--pack` for a real
pack's reference). Live: `tests/test_inkling2.sh`. `inkling2` (Small, no loop) and the ModelOpt release
(`model.llm.*`) are not claimed.

## Rules
- Everything that counts layers counts VISITS (48 on Mini: prelude, the core times its passes, tail): one KV slot and
  one `.conv` entry per visit. A span knob that names layers is refused by name.
- Reference policy: transformers' declared bf16 rounding, not mlx-lm's: static short convs in f32, weighted RMSNorms
  on an f32 input. The dump script patches mlx-lm to this policy for every bf16 reference.
- Not bit-exact against mlx-lm, by construction: the NVFP4 expert scale is applied after `gather_qmm` in f32 (mlx-c has
  no global-scale argument), the down scale folds into the routing weights, the sorted gather may take the NAX path,
  and mlx-lm's compiled clip / SwiGLU are eager here.
- The relative and depth attention kernels (`kernels.zig`, `MLX_SERVE_INKLING2_KERNELS=0` turns both off) keep the
  composed path as fallback. The composed prefill holds `[H, L, S]` f32 logits per visit.
- `inkling2 proof: the quantized pack ...` runs with the kernels off: its `tiny_q` seed was chosen on mlx-lm's composed
  GPU run, and the kernel flips a router near-tie in it; kernel exactness is the contract's.
- Sliding visits keep a full KV buffer (band view, no ring), so at the default context the KV bill leaves the hot
  prefix cache 0 MB; the live test passes `--ctx-size 32768` to exercise it.
- `inkling2 pack: ...`'s 0.99 bar is per row on its own prompt: on another prompt, compare a low row against both of
  mlx-lm's roundings before calling it a regression.
