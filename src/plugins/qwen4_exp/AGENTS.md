# qwen4_exp plugin (Qwen3.8-Flash-Next)

Owns `qwen4_exp`, `qwen4_exp_text`: a qwen3_5 GDN+MoE trunk in 4 hyper-connection streams, n-gram PLE
(`ngram_table.bin` or the n-gram rows embedded in the pack's shards), QSA sparse attention past `indexer_budget`,
an in-checkpoint MTP layer, a Qwen3-VL tower. Packs, oracle: `docs/reference.md`; fixtures: `fixtures/MANIFEST.md`.

- Table: `config.zig` (MLX-free claims + parse; `Config` carries `ModelConfig`'s names, `model_config.zig` maps them
  one to one, pinned against `parseConfigLegacy` on both shipped packs), `bill.zig`, `plugin.zig` (load + forward),
  `contract.zig` (kernel contract).
- SERVED by the driver (`served.zig`); the legacy forward is deleted (`served.legacy_removed`, refused by name). The
  plugin owns its `Trunk` (`trunk.zig`, `trunk/`, `trunk_load.zig`: weights, config, module state, MTP head), copied
  from the SDK/engine code it used with `copied_from` headers and cut to qwen4's arms; a layout an arm it dropped
  would read is refused at load (`trunk_load.unservedLayout`). It adopts the driver's map (`Resident.take`, no second
  load). Hooks: `forward_settle` (deferred PLE), `release_slot` (`ssmGroupDrop`, QSA marks), `meters` (`[qsa-arms]`);
  `lm_head_mxfp8` arrives through `LoadCtx.settings_json`; int8 prefill is inert (Hadamard packs only). Proofs:
  `proof_test.zig` and `served_test.zig` replay tapes recorded from the legacy forward (`fixtures/*.tape`, frozen),
  `oracle_test.zig`; ModelOpt/combined-GDN/ternary arms have their own tests (fixture layouts the tapes lack).
- P8 DEBT, `engines/mlx/plugin/intree.zig` (in-tree only, delete when the ABI carries these): the MTP head, group
  lanes and the grouped verify run on the plugin's `Trunk` through `MtpHeadRef.qwen4`; the cluster worker runs a
  bare `Trunk` and the plugin reads the Thunderbolt plan (`distributed.planFor`). An ABI-clean head needs:
  (1) `ForwardOut.next`, the chained hidden apart from the projected one; (2) a fact that the trunk's hidden is the
  pre-mixer stream; (3) a placed head forward (dead history rows); (4) `HeadHooks.mark` with a length; (5) head
  snapshot / bytes / serialize for the prefix cache and SSD; (6) group lanes and the grouped verify on the plugin arm;
  (7) the rerank on the head's mixer output; (8) the slot's whole M-RoPE table (`intree.enter` hands the engine's ctx,
  `plugin.engineCtx`, a `SharedThreadLocal`).
- Unserved hooks: `cache.aux` (QSA history + PLE prev in checkpoints/SSD ride the engine's SSM entries),
  `capture_stream_all` (no caller on this arch).
- QSA's generic half is the SDK's block-sparse attention (`sdk/blocks/attention/sparse_block/`: `Blocks`/`Params`,
  score sheet, exact select, masks, decode kernel, float64 reference, its own contract); the indexer (YaRN/M-RoPE,
  pooled keys, hc reads), the key history and checkpoints, the gather and verify arms stay here.
- Kernel contract: every Metal kernel registered (`contract*.zig` here and in `sparse_block/`); the multi-launch ones
  (qsa_decode, qsa_select's split pair, qsa_attn) chained against their wrappers. Unregistered: qsa_decode's CUDA pair.

## Rules (from the root AGENTS.md)

- NOT a qwen3_5 pack: hyper-connections, n-gram PLE, QSA around the trunk; HF `hidden_states[i]` is the INPUT of layer
  i. Vision rows splice BEFORE the hc tile. `isLinearLayer` reads only `full_attention_interval`.
- Module state READ-ONLY: text slots batch-decode; only the MTP slot is exclusive (`slotExclusiveDecode`).
- The spec "hidden" IS the pre-mixer stream (`[B,L,hc*hidden]`); head row r = (stream r, token r+1) at r+1.
- Per-request state outside conv/ssm rides `SSMCacheEntry.aux_state` + `ple_prev`; every reset/free via
  `ssmFreeQsaState`; restore-parity bar = the chunking class (~0.3 nats top-5).
- QSA GATHERS selected blocks, never a dense `[S, kv]` mask; exact select `msv_qsa_select` (16-way split at decode);
  the score sheet stays ONE NAX kernel at every width (`msv_qsa_score`): decode, verify, prefill pick the same blocks.
- QSA's visible tail is PER QUERY; scores in f32; `torch.topk` keeps the LOWER index on ties; n-gram eos is the TEXT
  config's (`ngram_eos`). One YaRN mscale on every indexer arm; pooled keys rope with attention's M-RoPE table.
- Past the budget a solo dense-KV decode row skips mask, gather and SDPA (`sparse_block/decode.zig` `attend`): reduce BEFORE the
  select (a shuffle in divergent code returns garbage).
- A host read inside the layer loop is a GPU barrier: PLE defers (`ple_defer` + `flushDeferredPle`) and claims its
  spec slot at build time; the GPU arm settles its n-gram history at the NEXT forward.
- PLE read path is calibrated on the inference thread before and after warming, median of 5 (`calibrateArm`); a cold
  32 GB mmap is a serial SSD fault (`PrefetchPool`, `startWarm`); `--ple-gpu` gates the no-copy wrap BEFORE wrapping.
- Residual stream is bf16; the f32 fixture is the MATH oracle (`MLX_SERVE_DIAG=qwen4-stream-f32`); `qwen4 fixture bf16` pins
  streams at 0.999. GDN state + gate in f32 (`MLX_SERVE_GDN_STATE_F32=0` = bf16), its own SSD namespace.
- Tiny MoE oracles tie everywhere: fixtures dump the reference's own margins; `Qwen4Ties` acquits by those.
- MTP head state is a `Qwen4MtpState` swapped onto the module (`qwen4MtpActivate` before EVERY head touch); the head
  projects only the consumed row; its rounds stay solo (`mtpRoundsStaySolo`).
- A fast path keyed on one pack's quant layout is silent on another's: new pack layout => `tests/qwen4_engagement.sh`.
- Decode is kernel-time and dispatch bound: a kernel that replaces ONE dispatch loses to MLX's qmv; rows per simdgroup
  is an occupancy decision (sweep in situ). Meters: `tests/fwd_ubench.sh`, `MLX_SERVE_DIAG=step-trace`.

- MTP specifics: the head projects only the consumed row (`Qwen4MtpProject`) and rides the slot's M-RoPE table on image turns; the grouped-expert NAX tile is a LOSS; solo greedy rounds pad the head history and pre-draft lazily, lagging one round at auto depth BY DESIGN.

## Nulls (measured, reverted)

One-expert gather kernels for the shared expert, tail ops folded into down+reduce, the uv hc kernel at one row,
grouped MTP rounds (#685: win only at 4 streams code), `mlx_depends` ordering of the q projection, top-k inside gate+up.
