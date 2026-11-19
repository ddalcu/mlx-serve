# ROADMAP — mlx-serve

## Phases

### Phase 01: Hy4-flash port (`hy_v4_flash`, issue #1241)

**Goal:** `TensorFold/Hy4-flash-MLX-4bit-MTP` loads and serves on Apple Silicon: text chat at quality parity with the HF reference, images accepted, DFly2 speculative decode engaged.

**Goal source:** issue #1241 + `~/claude-tmp/hy4-port/SEMANTICS.md` (evidence) + `docs/recipes/add-a-model.md` (contract).

**Requirements:**
- HY4-01 `config.zig` claims `hy_v4_flash`, refuses unimplemented knobs by name
- HY4-02 weights load: 148 shards, quant map (g64 + per-tensor g32 engram), iHC f32, DFly bf16, engram tables to the streamed bank
- HY4-03 core forward: iHC 4-stream + MLA + M-RoPE + SWA-512 + sigmoid-router MoE + dense L0
- HY4-04 compressed IndexPool indexer selects top-2048 on the 12 DSA layers (pool→group-topk→expand)
- HY4-05 Engram n-gram hash lookup injects at layers 1/11/21/31
- HY4-06 DFly2 drafter: 5 draft layers, target KV/index reuse (25/29/33/37/41), mtp_fusion slot0, confidence head
- HY4-07 SigLIP2 vision tower + projector, image tokens on chat
- HY4-08 **OpenAI-compatible serving**: `/v1/chat/completions` (stream + non-stream), `/v1/models` modality row, model discovery, the chat template (pack's jinja, jinja.cpp with fallback format) — inherited host work, needs its own verify task
- HY4-09 Sampling defaults: `generation_config.json` parity (temp 0.9/top_p 1.0 recommendation), `return_bos_score`/`return_max_logit` handling decision (implement or refuse-with-record)
- HY4-10 proofs: tiny-pack parity vs HF reference, kernel contracts, live tests, AGENTS.md + docs row, gotchas entries, release notes

**Plans:** 8 plans (01–05 wave 1 spine; 06–07 fan out; 08 verify)
