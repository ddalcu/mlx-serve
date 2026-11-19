# PLAN 07 — Vision: SigLIP2 tower + projector + M-RoPE

Phase: 01-hy4-flash-port | Wave: 3 | Depends on: 03
Requirements: HY4-07, HY4-08
Files: `src/plugins/hy_v4/vision.zig`, `src/sdk/vision/` reuse (2D-rope builder), vision kernels + contract, `tests/test_hyv4_vision.sh` (add)

## Objective

Chat requests with images work: the 27-layer `vit.*` tower + `perceive` projector encodes images into 3072-d tokens placed between image markers with correct interleaved M-RoPE spans, at parity with the HF vision forward.

## Tasks

### Task 1: tower + projector
- Reuse `sdk/vision` (gemma4/lfm2 tower shape reference): patch 16 (3×16×16→1152), learned `position_embedding` [16384,1152] (128×128 grid) with bilinear interpolation to the request grid (anyres max 2048, min 512), rope_2d (theta 1e4, full span), 27 blocks (norms with bias, gelu MLP 4304), `vision_full_attention=false` handling per config cite, 2×2 merge (4096 max tokens), `perceive_pre_norm`, `perceive.norm`+`proj.0/proj.2` MLP → 3072, `image_newline` row appended per token grid per the HF placement (spatial_merge_size 2).
- bf16 weights resident (0.43B ≈ 0.9 GB, no quant); billed at load.
- **Verify:** fixture parity: one 512×512 + one 1024×768 image, tower output tokens cos ≥ 0.999 vs `tests/dump_hyv4_fixtures.py` vision dump; contract for the rope-2d/patchify fusion.
- **Done:** vision forward proven on fixtures.

### Task 2: token placement + positions + live path
- Marker tokens from config ids (`<img/>`=120118, `</img>`=120119, `img` newline 120121, image token 120120, video ids present but video deferred — refuse `video` input by name until supported); projector output replaces image-token spans between markers; interleaved M-RoPE [12,10,10] position assignment for image grids (t=h=w=text for text; grid t fixed, h/w per patch per §9 cite); text-only path (t=h=w=pos) unchanged and its tests stay green.
- Modality caps on (`vision`), `/v1/models` row modality, registry maps the pack multimodal; chat template from the pack's `chat_template.jinja` renders with jinja.cpp (fallback format documented if template unsupported — check before claiming).
- **Verify:** `tests/test_hyv4_vision.sh`: image+text greedy answer matches the HF dump's next-token at temp 0 on the 512 GB host; text-only regression green; KV reuse invalidates after image turns correctly.
- **Done:** multimodal requests served at parity.

## Verification (plan-level)
Vision parity script green; text-only tests untouched; lint 0. Deferred: video inputs (ids exist; refuse by name).
