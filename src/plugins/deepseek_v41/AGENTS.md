# deepseek_v41

DeepSeek-V4.1-Flash (`model_type: deepseek_v41`, 1M context, MLX packs) with its vision tower, as a native,
module-owned arch. Unpaired captures transfer to the engine's prefix cache, keyed on image rows. Packs:
`ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw`, `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`, and
OpensourceWTF's EXL3 streaming repacks through the `turboexl3_ds41` quant plugin (its trunk and its `experts.bin` bank).

## Files
- `plugin.zig`: the table; `CacheHooks.aux` / live `state_bytes`; streamed/paired legacy `resume_prompt`;
  a streamed pack's bank (`openBank`) and its prompt calls (`runPrompt`).
- `config.zig`: claims + V4.1 fields on top of `../deepseek_v4/config.zig`; `bill.zig`: compressed caches per
  token, attended keys, `dsv41_mod.prefillBytes`; a streamed bank's budget (`streamBudget`).
- `media.zig`: decode, the release's processor (`vision.preprocess`), tower, placeholder run.
- `deepseek_v41.zig` (model), `engram.zig`, `vision.zig`. The legacy seam is deleted: a pack the plugin declines is
  refused by name (`load/init.zig`).
- `streamed_test.zig`: the trunk with its experts streamed from the pack (`stream_experts`) bit-equal to the in-memory banks.
- `fixtures/`: proof tapes recorded from the legacy forward (`fixtures/MANIFEST.md`).
- `contract.zig`: kernel contract registrations of `kernels/*.metal`.
- `decode/`: the fused decode kernels (`kernels/dec_*.metal`, one file a stage) a step or verify of up to `max_rows`
  rows runs through `blockFused`; `decode/fixture.zig` builds release-shape layers on random weights for their
  equivalence tests and the `DSV41_DEC_UBENCH` meters.
- `prompt/`: the wide prompt layers' exact rewrites (each file's header names its change and gain): the gather's
  kernels (`attn_core`, `q_tail`, `scratch_attn` regions and ring), the hc and router fusions (`hc_fuse`, `gate_gemm`),
  span grouping (`idx_group`), buffers kept or reused (`transient_keep`, `scratch_x1` on `scratch`),
  Engram's direct read (`engram_direct`), the embedding pread and a plugin bank's own pieces (`wide_early`, `woa_rows`,
  `flash_attn`: the lossy flash kernel, default on beside a plugin bank only).
- `spec/markov.zig`: DSpark's Markov chain in B + 1 launches; `decode/engram_decode.zig` the Engram LRU,
  `decode/gemv_twin.zig` the batched GEMV twin of MLX's M=1 words.
- `decode/moe_fp4.zig`: the MoE's two launches over mxfp4 routed experts (the Lossless pack), rows 1..6, built only
  without NAX (unmeasured on M5); `moe_up` / `moe_down` (NAX only) serve the affine 2/3-bit packs at group 64 or
  128 (the expert-pruned packs are 2-bit g128). Otherwise `routed` runs.
- GGUF expert banks (`--mlx-gguf`, ISTA's GSQ-RCO sets) decode up to `region_rows` through `ffnDecodeGguf`
  (`moe_gu_gguf` + `moe_dn_gguf`, built per ggml type by `gguf.pluginKernel`) and prefill on the matrix units
  (`gguf.gatherNaxSwiGLU`), a K-quant bank a 32-weight slice at a time (`deq_<type>_s`); Q3_K decodes `routed`.

## Facts the table declares
- `owns_decode_state`, `slot_state` + `batches_decode` (streamed: single-flight): active state in `dec_state`, other
  live slots in `parked`; decode and DSpark rounds batch through `extendBatched` (rows a state, projections shared,
  attention per state) and `dsparkDraftBatched`. Each State owns its pending capture.
- `prefix_cache` for unpaired instances whose bank holds every expert; a streaming bank (fewer prompt or decode slot
  rows than `num_experts`, `bankStreams`) and a Thunderbolt pair keep legacy resume.
  Snapshots retain every buffer and restore whole only at their own position; the engine owns finished history. Pairs
  keep legacy resume.
- A snapshot is cut to a shorter prefix (`sliceSnap`: global rows and index keys kept, where every pooling group is
  closed) and restored by the tech report's SWA bounded replay (`rewind`, `Want.replay`: the last window run again,
  its band stopped at its first row, no compressor). Text only, no pair or tensor split; a slice is never exported.
- No spec rollback (DSpark rewinds inside the module); DSpark stages from `native_stages`;
  `spec.trunk_weight` lends the embedding and head; a pair's load re-reads `distributed.planFor` (idempotent).
- A tensor pair (`InitOpts.tp`, `--dist tensor`) halves the experts and attention heads at load (the sliced `cfg` says so);
  both ranks run every layer and `tpSum` each attention and MoE output. Every rank must build the same sums in the same
  order: `blockFused` and `batchedLayer` sum between their stages (`tpFfn`). DSpark drafts on rank 0 alone: its stages
  load whole and sum nothing, and the worker follows each verify's rollback.
- Under `tp` the worker knows a state by the gen rank 0 gave it: every state rank 0 drops must reach `tpFree`, or the
  worker keeps it; a step on another state (`st != &dec_state`) is refused.
- Under `tp` every aux snapshot rank 0 takes, restores, clones or frees reaches the worker by id (`tpSnap`); one taken
  outside a step (`armPrompt` at a restored boundary) goes as `.capture`. The ranks' caches are equal (`tpStream`
  asserts it), so the SSD tier writes rank 0's and a disk import sends the worker its tensors (`tpLoad`).
- Under `tp` rank 0 alone holds the vision tower: a chunk's image rows ride with its step (`tp_step` `images`), the
  worker splices them at offset 0, so a rank's splice must depend only on the chunk's own rows (`chunkImageRows`).
- Thinking default ON; effort vocabulary `dsv41`; DSML tools on OUR byte-pinned template.
- Metal only: a CPU host stream loads on this thread's GPU stream; no GPU = refused by name.

## P8 debt (`engines/mlx/plugin/intree.zig`, section deepseek)
- Weights from the driver's map (`intree.takeWeights`; the tower stays in the engine's map, whose encoder hands the
  rows to `Forward.media`).
- DSpark rounds, the policy's confidence (`ds_want_conf`) and `ds_block` reach the module through `intree.dsv41`
  (rounds proven equal to the legacy shell's, greedy and seeded: `../deepseek_v4/served_test.zig`); a group round's
  slot states through `intree.dsv41States`. ABI needed: V4's list plus the copy-step drafts, the policy's per-round
  confidence, and a round over several slots.

## Rules (moved from root AGENTS.md)
- V4.1 images: every span position carries `image_token_id` (without the request's rows that id is TEXT, as in the
  reference); the module splices the rows itself; image rows route by `gate.bias_vl` and get no Engram write
  (`Hash.DEAD`).
- The resume key includes the image rows: a different image at the same tokens is a cold prompt.
- DSpark = DFlash + Markov head: `block_size` at ROOT; `rope_is_neox_style:false` = MLX `traditional=true`; `w1`
  DENSE; sampled requests via one-hot acceptance. Per-model `mtp:false` keeps the
  stages off (`mtpSetting`).
- The confidence head is calibrated: a draft is accepted at about the sigmoid of its logit, at every block position.
  The policy's bins start from it (`Calibration.head`); the GLM-fitted prior read 0.93 as 0.65 and starved rows.
- Engine code that drives the module directly (a DSpark round) activates its slot first (`activate_slot`); a
  resume names its slot, so it never touches another live slot's state.
- A row in a batch depends on its own state only, but two rows take the MoE's gather kernels where one row takes
  the one-expert kernels: batched logits match a solo step to kernel numerics, not bits. Past one row they match bit
  for bit on the fixture, so a slot's greedy group rounds never depend on who shares them.
- The V4.1 template, like V4's, is byte-pinned (`tests/dsv41_template_ab.py`).
- A pack an outside quant plugin claims (`sdk/quant/outside.zig`, exllamav3's EXL3 trunk) loads its weights through it:
  `Lin.ext` and `Experts.ext` hold no MLX arrays, so a path that reads a linear's `w`/`s` checks `ext` first; the
  head counts through `hasHead`, not `head.w` (a head of `ext` alone was skipped as a tensor rank's).
- The model runs in three `-Dunits` units (plugin, engine, modules): a setting or per-prompt region the load sets is
  a `Shared` (`ps()`), never a plain `var`, or the rounds and the streamer run on another unit's defaults.
- A plugin-streamed bank's greedy DSpark round verifies the whole block (`DflashPol.pin_rows`): its 6-row verify costs
  about a plain tick, and the adaptive policy's prior ladder cut novel text from 54 to 40 tok/s.
- A streamed bank's phase follows its decode capacity (`promptPhase`): a call past the decode regions OR past
  `Routed.decode_ids` routed ids runs the prompt phase; a 9-16 token prompt on the EXL3 bank failed `NotPrefill`.
- Engram tables are read whole and past the page cache only beside a streamed bank (`Table.readDirect`): on a resident
  MLX pack the direct whole-grid read slowed warm prompts (the cache served their rows), and the bank needs that memory.
- A resident pack reads its Engram CODES past the page cache while memory is tight (`Table.openCodes`, `pastCache`,
  `engram.cacheTight`: under an eighth of RAM free, asked at each prompt chunk), its scales always through it: with
  no room the cache evicted the scales to hold codes read once and every row cost two reads; with room the cache wins.
- A fused decode stage is judged against an f32 model (`decode.expectNoWorse`), not against the chain's bits: greedy
  texts diverge from the chain's at bf16 near-ties. Stacked layers judge the median over seeds: one seed lands 1.6x
  either side of the chain (TF32 moves the f32 model itself 8% at the verify step on M5). Two dependent stages merged into one ticket-ordered dispatch
  measured slower than the pair (their threadgroup shapes differ): keep one dispatch a stage.
- A cut restore is as far from a cold prompt as another prefill chunk width is, never bit-equal past one window: its
  proof is the state (`proof_test`: a whole-prefix replay is the cold state bit for bit, the global rows are the
  snapshot's own), its quality a live pair of answers with logprobs.
- A prompt chunk of 4096 rows or more that nothing read ahead runs as its first half, then the rest (`extendCold`,
  `leadRows`; the lead is `Want.encode`: no decoder replay, which the rest makes): the drive reads the rest's Engram
  rows behind the lead. The log's `waited ... ms` is the meter.
- Construct each MPP cooperative tensor from its owning op; copying one invokes incompatible Metal layout
  interfaces on some SDKs. Fill each operand's independent storage before running the matmul.

## Streamed packs (`experts.bin`)
- An `experts.bin` bank is an outside quant plugin's (`openBank`: `outside.Format.openBank`, the `abi.Bank` as the
  trunk's `Routed` `source` arm); no plugin claiming the pack refuses it by name (`StreamedBankUnclaimed`). The
  trunk's own MoE never runs for its experts (the DSpark stages stay resident). The plugin owns the bank's rows,
  reads, phases, kernels and its own bill terms; the trunk owns the forward and the budget (`streamBudget`).
- The prompt runs in calls of up to `bill.stream_call_rows`, each one MoE call a layer over all its rows
  (`wideLayer`), its routes predicted before the attention (`readAhead`) so the reads overlap it.
- Slot rows per phase (`Routed.phase`): a call wider than `region_rows` is a prompt, else decode; the phase change
  grows the rows into what the prompt's working set held.
- The bank bills its rows at the load (`streamBudget`: the GPU working set less the host's wired margin and
  `stream_host_bytes`, the memory in use, the phase's working set at the context); the context is `ctx_size`, else
  the largest `stream_context_rungs` rung keeping `stream_min_rows` (`bank_rows`). The engine's prefill guard leaves
  the pack to it (`streamed_weights`); a prompt past the billed context is refused by name.
- A plain MLX pack streams with the per-model `stream_experts` (`openPackBank`, the `trunk` arm): one fetch serves
  `Routed.decode_ids` experts, so a wider prompt call runs its experts in groups (`slotWide`). Decode is read-bound
  (nine scattered reads an expert vs one EXL3 record): the repack stays the fast way to stream.
- A tensor pair refuses streamed experts by name (`Dsv41TpStreamedExperts`): a bank reads whole experts, only rank 0
  would open one, and each rank holds half of every expert.
- A layer split refuses `stream_experts` by name (`Dsv41PairStreamedExperts`): each Mac is billed its share of the
  layers, while the stream opens slot rows for every one.
- A run takes a heads region only when every span's gather serves (`runGathers`, decided at its first span): a span
  that wrote the region hands back no heads, so a run cannot fall back to the join halfway (unaligned second turns).

## Known issues
- Auxiliary snapshots persist through the disk tensor hooks as exact v10 module records; one-shot serialization
  still returns a named unsupported verdict. SSD-only module retention is unsupported.
- Serial decode near the memory wall: 720 GB/s of ~1.1 TB/s, so kernel work buys 15-25%, not 2x.
- Without NAX (M4 and older) `attn`, `idx_score` and the `wo_a` / `ffn_in` fusion behind them stay off: decode runs the
  pre-NAX split kernel and the compiled regions. The affine `moe_up` / `moe_down` stay off too (they lose to the
  compiled MoE there); `moe_fp4` is the non-NAX one.
- Prefill is chunked at 8192 at most (`prefillChunkMax`, an explicit `--prefill-chunk` included), one MoE call per
  chunk: each chunk still reads every expert bank once (the CED replay runs the decoder layers), so a narrower chunk
  pays that read twice as often.
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

DeepSeek-V4/V4.1 Flash, 1M ctx. Spec off but DSpark; V4 single-flight, V4.1 batches decode and DSpark rounds across slots; DSML tools, OUR byte-pinned templates. V4:
prefix cache OFF, 0731+, our layout. V4.1: MLX packs, DSpark ON, unpaired auxiliary snapshots keyed on
image rows; the EXL3 repack's experts streamed from its bank. V4.1 images: every span position carries `image_token_id` (without the
request's rows that id is TEXT, as in the reference), the module splices the rows itself, image rows route by
`gate.bias_vl` and get no Engram write (`Hash.DEAD`)
