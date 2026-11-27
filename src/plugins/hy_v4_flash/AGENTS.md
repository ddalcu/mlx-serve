# hy_v4_flash

Owns `hy_v4_flash`: Tencent Hunyuan 4-flash (Hy4-flash, 220B-A12B multimodal MoE, text path in this port; PR #1241).

## Files
- `plugin.zig`: the table (`abi.ModelTable`). Handles `parse`, `free_parsed`, `load`, `unload`, `forward`, `bill`.
- `config.zig`: MLX-free claim + parse. Declares `hy_v4_flash`, `hy_eos = 120025`, parses both full HuggingFace `text_config` and tiny oracle configurations.
- `hy_v4_flash.zig`: The core model architecture (MLA, compressed indexer, sliding window attention, learnable sinks, hyper-connections, Engram table gather, latent MoE with FP8 embeddings and per-head E4M3 rounding).
- `proof_test.zig`: Parity and contract proof tests against `tests/hy4_flash_ref.py` and `tests/hy4_flash_quant_ref.py`.

## Facts the table declares
- Serial decode, single-flight: module-owned decode state (`owns_decode_state = true`, `prefix_cache = false`, `moe = true`).
- The stop set always holds `<｜hy_end:opensource｜>` (120025) merged directly in `config.zig`.
- External Engram table support shared via `src/plugins/deepseek_v41/engram.zig` (`openHy4`).
- No template grepping in `tokenizer.zig`; the EOS token is declared and exempt by config.
- Zero string checks in `prefix_cache.zig`; plugin declares `owns_decode_state = true` and `prefix_cache = false`.

## Rules
- Text trunk: batch 1, f32 math matching reference.
- Engram projections carry biases (`value_proj.bias` and `key_projs.*.bias`).
- Engram table row offsets advance by each head's vocabulary size across consecutive slices.
