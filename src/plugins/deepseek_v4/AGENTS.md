# deepseek_v4

DeepSeek-V4-Flash (`model_type: deepseek_v4`, 1M context) as a native, module-owned arch: the request's state
(`dec_state`: sliding window, compressed latents, indexer keys) lives on the model, never in the engine's KVCache.
Packs: `ddalcu/DeepSeek-V4-Flash-MLX-Serve` (0731+, our layout). `config.zig` is also the family's shared parser
(V4.1 extends it).

## Files
- `plugin.zig`: the table (load from the driver's map, forward at position 0 = new request, lazy GPU decode, verify
  via all-rows extend). Served: `deepseek_v4` is in `engines/mlx/plugin/served.zig`.
- `config.zig`: MLX-free claims + parse into `Parsed` + the family `Config` (ModelConfig's field names, so
  `model_config.zig`'s `toModelConfig` is mechanical while the module still reads ModelConfig).
- `bill.zig`: per-token KV, attended keys, the arch's own prefill term (`dsv4PrefillMemoryNeeded`).
- `deepseek_v4.zig`: the model. The legacy seam is deleted: a pack the plugin declines is refused by name at parse
  (`served.legacy_removed`), e.g. mlx-community's DeepSeek-V4-Flash-4bit, the withdrawn preview (MTP layers, no
  `dspark_*`). `fixtures/real/`: released configs, the preview and mlx-community's 0731 pack.
- `fixture.zig` (shared with V4.1): tiny random packs, DSpark stages on request; `proof_test.zig`: the table against
  the legacy seam's recorded values (also V4.1's); `served_test.zig`: the driver's shell through `forwardWith` and
  the generator's DSpark rounds, against tapes (both plugins). `fixtures/`: the tapes (`fixtures/MANIFEST.md`).
- `contract.zig`: the kernel contract registrations of `kernels/*.metal`.

## Facts the table declares
- `owns_decode_state`, `prefix_cache = false`, serial, single-flight (one module state per model); no batching,
  no layer capture, no Gemma/DFlash drafter (`spec.policy.drafter_ok = false`), DSpark stages from the config
  (opt-in on V4: `--dspark`). V4 has no multi-Mac path (`distributed.plan` refuses it).
- Thinking default OFF; effort vocabulary `dsv4`; DSML tools on OUR byte-pinned template (template_family).
- Missing generation_config = the wild 1.0/1.0 signature: V4 serves the reference temperature 0.6.

## P8 debt (`engines/mlx/plugin/intree.zig`, section deepseek)
- Weights come from the driver's map (`intree.takeWeights`); DSpark rounds (`spec/dspark.zig`, `generate/init.zig`)
  and `dsparkStages` reach the module through `intree.dsv4`. The ABI-clean replacement needs a native block drafter
  in `SpecHooks`: a round (committed token, accept cap) -> accepted tokens, next token, new position, with the
  module's own rewind; the stochastic arm's filtered-probs accept under the request's seed; the block width.

## Rules (moved from root AGENTS.md)
- Serial, spec off but DSpark, single-flight; DSML tools, OUR byte-pinned templates; prefix cache OFF (supported
  architectures row).
- A transcribed chat template is worth what it is PINNED against; whitespace is a token-level contract
  (`tests/dsv4_template_ab.py`, byte equality over the shapes the server emits).
- The guard bills what the ARCH reads: dsv4's own prefill term, not the dense formula (`bill.prefillBytes`).
- dsv4 comp_in decode-attention requant is EXPLICIT opt-in (`decodeAttnQuantExplicit`), never the default arm.
- dsv4 decode glue is FUSED (`dsv4_emit_win`, `dsv4_dec_chain`); A/B any new prefill kernel before default-on.
- A model whose per-request state lives outside the KVCache is excluded from prefix-cache reuse and owes an unload
  that frees it (here: `Dsv4Model.deinit`).

## Known issues (kernel contract, `contract.zig`; kernels unchanged)
- `dec_chain` kv / indexer q and `emit_win` (3 arms): a NaN in a quantized group comes out a finite code (Metal's
  max drops it from the group max, clamp from the code); float64 and the composed chain propagate it.
- `moe_gateup` (2/4/8-bit): a NaN gate sum becomes the clip limit (Metal's min drops NaN, MLX's minimum keeps it).

## Fixtures
- `conformanceFixture`: the oracle's four-layer miniature (`fixture.v4Files`, every attention arm), 8-bit gs32.
- Oracle: `tests/dsv4_mlx_ref.py`; parse fixtures: the release config (`config.release_config`, no pack config in
  `src/fixtures/model-configs/`).
