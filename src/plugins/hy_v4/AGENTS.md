# hy_v4 — Tencent Hy4-flash (220B-A12B multimodal MoE)

`model_type: hy_v4_flash`. The trunk is a DeepSeek-family MLA + compressed-DSA MoE with iHC streams, an Engram
n-gram PLE on 4 layers, and a DFly2 draft stack in `model.dfly_layers.*`. Plan 01 (this state): claim, parse,
bill, cache layout. Loader = Plan 02, forward = Plan 03, indexer = Plan 04, Engram = Plan 05, DFly = Plan 06,
vision = Plan 07.

## Rules

- **Layer kinds come from WEIGHT presence, never `layer_types`**: the config list is HF-offset by one. The pack's
  DSA layers are `[0,1,5,9,...,41]` (12, with `compressed_indexer.*` tensors); the rest are SWA (64 heads, window
  512). Engram tensors sit at pack layers 1/11/21/31 while `engram_layer_ids` says 2/12/22/32.
- **`q_b_proj`/`linear_gate`/`o_proj` ship 2×-padded (8192 = 2×4096)**: padding vs the 64-head SWA read is
  unresolved upstream; the loader keeps the raw rows and Plan 03 A/Bs it against the HF reference.
- **The Engram tables never go resident**: 4× `U32[165M,12]` 4-bit/g32 ≈ 39.7 GB packed. They stream from an SSD
  bank (Plan 02); the bill excludes them on purpose.
- **Fine indexer stays off** (`compressed_indexer_skip_fine_select: true`): compressed IndexPool only — 4×
  pooling → top-512 groups → top-2048 tokens. A pack with it on is refused by name.
- The DFly2 draft reads the target's indexer top-2048 snapshot at the last-accepted row ∪ its own causal block;
  its KV aliases target layers 25/29/33/37/41; fusion happens on slot 0 only. HF transformers does NOT implement
  DFly2 — the semantics live in SGLang #51159 / vLLM #68296 (see `~/claude-tmp/hy4-port/SEMANTICS.md`).
- `confidence_head` ships in the pack; Tencent's engines skip it — load it, do not wire it.

## Packs & fixtures

- Full pack: `~/.mlx-serve/models/TensorFold/Hy4-flash-MLX-4bit-MTP` (148 shards, 129.9 GB; 89.5 GB minus Engram).
- CI dev pack: `test-org-copilot/HYV4-0.7B-mlx` (~838 MB, no engram/DFly/vit weights); engram/DFly fixtures are
  synthetic. Fixture config: `src/fixtures/model-configs/hy_v4-flash.json`.
- Quant census: attention/embed at g64 4-bit, MoE experts and Engram embeddings at g32 (per-tensor overrides in
  `quantization`), iHC f32, DFly bf16. `enable_lm_head_fp32` lies — the lm_head IS quantized.
