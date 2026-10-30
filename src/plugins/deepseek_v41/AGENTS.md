# deepseek_v41

DeepSeek-V4.1-Flash (`model_type: deepseek_v41`, 1M context, MLX packs) with its vision tower, as a native,
module-owned arch. Resident unpaired captures transfer to the engine's prefix cache, keyed on image rows. Packs:
`ddalcu/DeepSeek-V4.1-Flash-1017-iQ-MLX-3.0bpw`, `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`, and
OpensourceWTF's streaming repacks (`experts.bin`: routed experts as an EXL3 bank, streamed).

## Files
- `plugin.zig`: the table; `CacheHooks.aux` / live `state_bytes`; streamed/paired legacy `resume_prompt`;
  a streamed pack's bank (`openBank`) and its prompt calls (`runPrompt`).
- `config.zig`: claims + V4.1 fields on top of `../deepseek_v4/config.zig`; `bill.zig`: compressed caches per
  token, attended keys, `dsv41_mod.prefillBytes`; a streamed bank's budget (`streamBudget`).
- `media.zig`: decode, the release's processor (`vision.preprocess`), tower, placeholder run.
- `deepseek_v41.zig` (model), `engram.zig`, `vision.zig`. The legacy seam is deleted: a pack the plugin declines is
  refused by name (`load/init.zig`).
- `streamed_test.zig`: the trunk over an MXFP4 `experts.bin` bit-equal to the in-memory banks; a served EXL3 pack
  equal to the trunk over the same bank at fewer slot rows (phase change, misses through the transient bank).
- `fixtures/`: proof tapes recorded from the legacy forward (`fixtures/MANIFEST.md`).
- `contract.zig`: kernel contract registrations of `kernels/*.metal`.
- `decode/`: the fused decode kernels (`kernels/dec_*.metal`, one file a stage) a step or verify of up to `max_rows`
  rows runs through `blockFused`; `decode/fixture.zig` builds release-shape layers on random weights for their
  equivalence tests and the `DSV41_DEC_UBENCH` meters.

## Facts the table declares
- `owns_decode_state`, `slot_state` + `batches_decode` (streamed: single-flight): active state in `dec_state`,
  other live slots in `parked`; plain decode batches through `extendBatched`. Each State owns its pending capture.
- `prefix_cache` for resident packs; `aux.enabled` also excludes Thunderbolt pairs. Snapshots retain every buffer
  and restore only at their exact position; the engine owns finished history. Streamed/paired keep legacy resume.
- No spec rollback (DSpark rewinds inside the module); DSpark stages from `native_stages`;
  `spec.trunk_weight` lends the embedding and head; a pair's load re-reads `distributed.planFor` (idempotent).
- Thinking default ON; effort vocabulary `dsv41`; DSML tools on OUR byte-pinned template.
- Metal only: a CPU host stream loads on this thread's GPU stream; no GPU = refused by name.

## P8 debt (`engines/mlx/plugin/intree.zig`, section deepseek)
- Weights from the driver's map (`intree.takeWeights`; the tower stays in the engine's map, whose encoder hands the
  rows to `Forward.media`).
- DSpark rounds, the policy's confidence (`ds_want_conf`) and `ds_block` reach the module through `intree.dsv41`
  (rounds proven equal to the legacy shell's, greedy and seeded: `../deepseek_v4/served_test.zig`). ABI needed: V4's
  list plus the copy-step drafts and the policy's per-round confidence.

## Rules (moved from root AGENTS.md)
- V4.1 images: every span position carries `image_token_id` (without the request's rows that id is TEXT, as in the
  reference); the module splices the rows itself; image rows route by `gate.bias_vl` and get no Engram write
  (`Hash.DEAD`).
- The resume key includes the image rows: a different image at the same tokens is a cold prompt.
- DSpark = DFlash + Markov head: `block_size` at ROOT; `rope_is_neox_style:false` = MLX `traditional=true`; `w1`
  DENSE; sampled requests via one-hot acceptance. Per-model `mtp:false` keeps the
  stages off (`mtpSetting`).
- Engine code that drives the module directly (a DSpark round) activates its slot first (`activate_slot`); a
  resume names its slot, so it never touches another live slot's state.
- A row in a batch depends on its own state only, but two rows take the MoE's gather kernels where one row takes
  the one-expert kernels: batched logits match a solo step to kernel numerics, not bits.
- The V4.1 template, like V4's, is byte-pinned (`tests/dsv41_template_ab.py`).
- A fused decode stage is judged against an f32 model (`decode.expectNoWorse`), not against the chain's bits: greedy
  texts diverge from the chain's at bf16 near-ties. Two dependent stages merged into one ticket-ordered dispatch
  measured slower than the pair (their threadgroup shapes differ): keep one dispatch a stage.

## Streamed packs (`experts.bin`)
- M5 and newer only (`naxAvailable`), refused by name below, as are GLM-5.3's EXL3 repacks: the prompt GEMMs are built
  on MLX's NAX tiles and miss their f64 self-check on M4. The MLX pack streams anywhere.
- The routed experts come from `sdk/weights/streamed.zig` `Exl3` (`exl3_source.zig`: the EXL3 lane, event-gated
  waves, the next layer's reads predicted, the wide prompt feed, the decode forms / banked waves / hoist / device
  route); the trunk's own MoE never runs for them (`moe`'s `source` arm; the DSpark stages stay resident).
- The prompt runs in calls of up to `bill.stream_call_rows`, each one MoE call a layer over all its rows
  (`wideLayer`), its routes predicted before the attention (`readAhead`) so the reads overlap it.
- Slot rows per phase (`Routed.phase`): a call wider than `region_rows` is a prompt, else decode; the phase change
  grows the rows into what the prompt's working set held.
- The bank bills its rows at the load (`streamBudget`: the GPU working set less the host's wired margin and
  `stream_host_bytes`, the memory in use, the phase's working set at the context); the context is `ctx_size`, else
  the largest `stream_context_rungs` rung keeping `stream_min_rows`. The engine's prefill guard leaves the pack to
  it (`streamed_weights`); a prompt past the billed context is refused by name.
- A plain MLX pack streams with the per-model `stream_experts` (`openPackBank`, the `trunk` arm): one fetch serves
  `Routed.decode_ids` experts, so a wider prompt call runs its experts in groups (`slotWide`). Decode is read-bound
  (nine scattered reads an expert vs one EXL3 record): the repack stays the fast way to stream.

## Known issues
- Auxiliary snapshots persist through the disk tensor hooks as exact v10 module records; one-shot serialization
  and shortened slices still return named unsupported verdicts. SSD-only module retention is unsupported.
- Serial decode near the memory wall: 720 GB/s of ~1.1 TB/s, so kernel work buys 15-25%, not 2x.
- Prefill is chunked at 8192 (`prefillChunkMax`), one MoE call per chunk: each chunk still reads every expert bank
  once (the CED replay runs the decoder layers), so a narrower chunk pays that read twice as often.
- Past 0.95 x Metal's working set MLX's allocator evicts its buffer cache on every miss (two residency commits per
  allocation): DSpark at 8192 crosses it on a 256 GB Mac at the default `iogpu.wired_limit_mb`; raise it to bench.
- DSpark on prose falls back to plain (acceptance 1.04/round): correct, prose runs ~54 tok/s.
- Kernel contract (`contract.zig`; kernels unchanged): `idx_score` / `idx_tile` score a NaN q.k dot as 0 (relu
  via Metal's max); `rope_tail` fp4 arms and `kv_row` write a NaN in a quantized block as a finite code (max drops
  it, the fp8 encoder saturates it); `moe_gu` / `moe_act` turn a NaN gate sum into the clip limit (Metal's min);
  `engram` launches 1024 threads at 5120-wide rows where M1/M2 compile it for 928, unprobed: decode-width Engram
  writes fail on M1/M2.

## Fixtures
- `conformanceFixture`: `fixture.v41Files` (the `tests/dump_dsv41_fixtures.py` pipe layout, 8 layers, no Engram or
  DSpark). Parse fixtures: `src/fixtures/model-configs/deepseek_v41-*.json` (configs of the two packs above).

## The row from the root architecture table

DeepSeek-V4/V4.1 Flash, 1M ctx. Spec off but DSpark; V4 single-flight, V4.1 batches plain decode across slots; DSML tools, OUR byte-pinned templates. V4:
prefix cache OFF, 0731+, our layout. V4.1: MLX packs, DSpark ON, resident unpaired auxiliary snapshots keyed on
image rows; the EXL3 repack's experts streamed from its bank. V4.1 images: every span position carries `image_token_id` (without the
request's rows that id is TEXT, as in the reference), the module splices the rows itself, image rows route by
`gate.bias_vl` and get no Engram write (`Hash.DEAD`)
