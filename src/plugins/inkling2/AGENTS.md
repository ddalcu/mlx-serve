# inkling2

`inkling2_loop` (Thinking Machines Inkling 2 Mini, 26.4B-A3.4B), text and images: the release as published
(`thinkingmachines/Inkling-2-Mini`, `model.llm.*`) and mlx-lm's layout (`models/inkling2`, ml-explore/mlx-lm#2086, the
`mlx-community/Inkling-2-Mini-4bit` pack). `config.zig` parses `text_config`, the loop, `vision_config` and
`mtp_config`; `layers.zig` is one layer visit as an op chain (the reference: a CPU stream, any dtype the kernels
decline); `kernels.zig` the same visit as a few kernels around the matmuls; `plugin.zig` the loop over visits, the
load, the draft round and the fixture; `vision.zig` + `media.zig` the image tower; `draft.zig` the bundled drafter.

## The model

- 24 layers of weights, 48 visits per token: layers `[core_start, core_end)` run `num_passes` times. EVERY visit has
  its own cache entry (KV + conv state), so the shell's layer count, the layout and `ssm_state.entries` are visits.
- The stream is not a residual: a layer whose index divides by `n_layers_per_attn_res_block` saves its normed input
  as a depth block and restarts from its attention output; attention inputs, the MLP input and the final hidden are
  depth attention (learned queries, f32) over the blocks and the live stream. Blocks live in one forward.
- A later pass restarts from the loop's input and reads the pass before through messages (depth attention over the
  blocks that pass opened after its first, and its normed output), mixed in by the q, k, v and mlp message gates.
- Attention has NO RoPE: L2-clipped q, k and relative states, unweighted-RMS v, a hidden-conditioned relative bias,
  log-scaling of q and the bias on a global layer past the floor, a per-row sink, a sigmoid output gate.
- Short convs on k and v (static on a sliding layer; per head, static plus one projected from the input, on a global
  one) and on both sublayer outputs. A visit's conv state is ONE f32 array `[k | v | attn | mlp]`, K-1 rows.
- MoE: sigmoid scores + selection bias pick the top-k, weights are the unbiased scores over their sum times
  `route_scale`; `gate_up_proj` rows alternate gate, up; SwiGLU is soft-capped (tanh) on both arms.
- mlx-lm's packs name no expert quant scheme: a bank with u8 block scales is the release's NVFP4 (per-expert
  `global_scale`, applied as `scale / (6 x 448)` on the output) or MXFP8, told apart by geometry as mlx-lm's sanitize
  does. The config's `quantization` block is the attention's, the embedding's and the head's.
- The head trims the padded vocab BEFORE the soft cap, so the logits are a dense array.
- Every read in `plugin.zig` names its tensor twice, mlx-lm's name then the release's (`At`). Same tensors, so both
  layouts serve the same greedy text, which holds only while the release's per-expert multiplier is computed in the
  converter's order of operations (`amax x 2688 / den`, then `/ 2688`): `amax / den` is one ulp off.
- Refused by name: oMLX's oQ layout (`model.embed`, `attn.in_proj`), a `quant_method` other than the release's
  `modelopt`, shared experts, a variable loop, a sliding geometry other than the global one.

## Rules

- **A one-row visit is bound by the dependent launches on its critical path (5-7 us each, any size) and its bytes,
  not by its node count.** Independent launches overlap. A kernel that recomputes to save a launch loses unless the
  recompute is a few hundred thread steps (the router rows inside `mid.metal`).
- **A one-row forward runs under a 200 MB command-buffer split** (`command_buffer_mb`, via `command_buffer_limits`): at
  MLX's 50 each expert launch (a whole bank) committed and the encode stalled. Prompts and verifies keep the host's. It
  is process-wide: another model runs at 200 MB from inkling2's last step to its next wide forward (a known limit).
- **Dense projections stay in the checkpoint's `[out, in]` layout** (`layers.projectRows`): MLX's one-row matmul over
  a pre-transposed `[in, out]` weight costs several times the latency of `x @ W.T`. The message gates' down
  projections are held `[4, rank, 2 D]` for the same reason.
- **Every step kernel takes rows on grid y**, so `kernels.visit` serves one row, a prompt's rows and batched slots;
  the conv kernels read one state the rows chain behind (`SEQ` 1) or a state per row (`SEQ` 0, batched slots).
- **A prompt's rows (past `small_rows`) read on the NAX tensor unit** (`relattn_nax.metal`: `matmul2d` for Q K^T and
  P V, no mask built); `relattn.metal` (8x8 SIMD matrices, 4x the time a key) serves a GPU without NAX. A masked key's
  value still enters P V at weight 0 in its 32-key tile: a NaN there reaches the row.
- **Reads of at most `small_rows` rows build a mask (`relmask.metal`) and use MLX's attention**, which beats the flash
  kernel (`relattn.metal`, mlx-lm's, MIT: 16 keys at a time on one threadgroup) at a few rows' latency.
- **2 to `packed_rows` rows (63) read the packed expert banks in place, their picks GROUPED by expert** (`group`: tiles
  of up to 4 picks; `ups_grp`, `downs_grp`, then `few_tail`): an expert's bytes are decoded once a tile. One row reads
  `ups`/`downs`; MLX's gather wins only from 64 rows (its sorted tiles start there; under it, it reads a bank a pick).
- **Non-NAX wide-tile sorted MoE prefill (`moeWide`, `tile_scan.metal` + `moe_wide.metal`)**: serves non-NAX Apple9+
  from `moe_wide_min_rows` (1536, covering 2048/4096 chunks) with fused soft-capped SwiGLU and direct `rmap` reads;
  below 1536 per-expert tiles underfill and stock sorted `gather_qmm` is faster.
- **2 to 16 rows read the keys on the NAX tensor unit in one pass** (`relsplit_nax.metal`, `splitReadNax`; up to four
  slots' blocks a launch, `splitReadNaxLanes`: a kernel takes 31 buffers). `relsplit.metal` serves a GPU without NAX, its
  splits joined in `relcomb` (normalised after the sum: contract gate 4). Under 4 KB of keys MLX binds them `constant`: old path.
- **A threadgroup past 256 threads is proven per instantiation** (`unproven`): an M1/M2 compiles `pre` and `mid` for
  384 to 576 threads, under the 768 an M3+ runs. The first dispatch is evaluated and a refusal narrows
  `stream_threads` for the process; every other kernel's group is at most 256.
- **A template argument never carries a launch's shape** (rows, rows a slot, keys): each value is an instantiation, and
  past the proof table every launch pays an evaluation. A kernel reads such a count from its inputs' shapes.
- **The activation and the message gates use `precise::tanh` / `precise::exp`, the sigmoid divided once**: the fast
  tanh is NaN at an infinity and a percent off near zero, and `capped * (1 / (1 + e))` flushes a tiny sigmoid to zero.
- **`split_from` to `split_rows` rows (4 to 8) read through `relsplit` + `relcomb`**: MLX's attention reads the keys
  once per row and head, the split kernel once per KV head over threadgroups of 128 keys. At 2 or 3 rows they tie,
  and the mask path keeps them.
- **A verify's rows (2 to `open_rows`) read the cache where it lies, their own keys a second source of `relsplit`**,
  `split_rows` a launch: joined to the cache, every global visit copied all of it, and past 8 rows the mask path
  walked the keys once per row and head (at 32K a ninth row cost more than the eight before it).
- **A few-row f32 dot stays in a kernel** (`routerRows`, `gate`'s dots, 2 to `small_rows` rows): on NAX, MLX's f32
  matmul past one row is TF32 (~1e-3) at ~7x a row's latency; its one-row path is exact.
- **A kernel is more exact than its op chain**, and on the random fixture a last bit flips an expert pick:
  rows-against-one-row proofs hold on the CPU stream, a GPU proof compares forwards of the same width (batched: a
  slot's row moves with neither its place nor its neighbours), and each kernel has a direct test against its chain.
- **A raw reader of a transposed view compares physical bytes**: wrap the chain's side in `g.contiguous`.
- **Images**: `model.vision_tower.*` (mlx-lm) or `model.visual.*` (release); 40 px merge blocks, padded with -1,
  spliced at `<|unused_200054|>`. `bases.Model.placeholders` must be set at load or the rows never land.
- **One row's output gate rides the output projection** (`oGated`, `oproj.metal`): the launch it saves beats MLX's
  one-row qmv on this shape too. Affine at every width or dense (`oFormat`); anything else keeps `gated` + `projectRows`.
- **The draft head** (`mtp.safetensors`): a round drafts 7 tokens, verifies the drafts worth their rows
  (`draftsToVerify`, on the engine's ladder `Forward.round_ms`; never a stop id) as ONE forward that appends nothing
  (`kernels.Open`), and commits the rows the trunk agrees with (`kernels.commit`): no rollback. State is per slot (`Spec.select`).
- **A `drafter/` folder in the pack serves in place of the bundled head** when `config.readDflash2` runs its config
  (`incoai/Inkling-2-Mini-DFlash2`, `dflash2.zig`, z-lab's MLX model the reference): five sliding layers over the taps, a
  selector chaining top-16 candidates a row, its soft cap AFTER the edge (`after_correction`, the one mode ported).
- **A sampled block's draft starts on the GPU and its ids are read AFTER the verify graph is built** (`chainDraft`, `settleDraft`):
  the verify embeds the ids as an array, so nothing waits on the host; a stop id ends the block there and the later rows go unread.
- **A DFlash2 round is one launch chain: a draft TREE built on the GPU, its verify, the accept and the path's commit**
  (`stepGroup`, `NativeDrafter.step_group`; `MLX_SERVE_INKLING2_TREE=0` = the chain). A sampled node draws its children without
  replacement and the accept is rejection over the top 16 (`sample_tree_*`); the host reads one small result a round.
- **A lane's tree has `treeWidth` rows: 8 alone, fewer with more lanes or a sampled lane** (every row reads its experts'
  banks, the forward's largest read). A chain block never has more than 15 rows: 16 takes the NAX quantized tile lane, 3x slower.
- **A float32 stream's tree read keeps float matrices** (`relsplit_lanes`, `msv_ink2_el`): in half, one flipped expert pick
  on the fixture moved a node's logits (cosine 0.97) on a GPU without the tensor unit.
- **The DFlash2 context is the newest `window - 1` (2047) committed positions** (`Spec.keep`): a row attends a ring key
  only within `window` of its index, so past 2040 keys the block's later rows lose the oldest (`windowMask`); the
  parity test runs both sides of it. The engine's own sidecar resolver also sees `drafter/` and declines it by name.
- **The plugin times nothing and the verify stays lazy** (`verifyRound`): the engine prices rounds
  (`Forward.round_ms`) and its accept evaluates the verify with it, one evaluation a round.
- **The drafter reads its block rows through an 8-bit copy of a dense head** (`draft.headCopy`, 463 MB): the 926 MB
  head is a draft's largest read, and drafts are only proposals. At 4 bits acceptance fell.
- **A drafter layer `local_layer_ids` leaves out reads every committed position** (`appendFull`, billed as `Bill.state_per_token`),
  the others the ring's newest window - 1: read as a sliding layer, the release's layer 1 drafted blind past 1K tokens.
- **A sampled round draws its drafts from the drafter's law at the request's temperature, as many as the engine
  asks** (`Drafter.Law`, `accept_cap`): the width is chosen before any draft is drawn (a drawn token's probability, or
  its law's peak, cut rounds that land), and a draft law cooler than the request's lands less.
- **A drawn block is never cut on what was drawn**: it ends on its stop id (`draftsToVerify`), which the engine tests
  like any draft and hands back as the pending token (`stopPending`). Cut before it, a sampled turn ended less often
  than the trunk ends it.
- **A sampled, unseeded request continues the slot whose sequence it repeats** (`Forward.adopt_slot`, `Spec.adopt`):
  slot ids are per request, so the drafter re-read only the uncached tail every turn. The ring and full context are cut
  back to the restore. The engine picks the newest record agreeing over half the prompt (`adoptFor`); the state moves.
- **The draft head loads on Apple9+ only, with the kernels on** (`draftsOn`, told to the engine by `NativeDrafter.active`):
  on an M1 a few-row verify costs so much more than a step that drafted decode ran under serial, and only the kernels
  serve a verified or batched visit (`MLX_SERVE_INKLING2_FUSED=0` also batches no slots).
- **A sliding visit keeps a ring of window + 2048 rows** (`Config.ring`, declared as `CacheFacts.sliding_ring`;
  `layers.appendKv` at every append): kept whole, the 40 sliding visits held several times the KV a long prompt needs, and every
  prefix-cache entry with it. A match more than the slack behind an entry's end then misses.
- **The draft head's 3 layers and Markov chain**: `draft_bits = 8` joins `q|k|v`, `k_ctx|v_ctx`, `gate|up` and quantizes
  the layers to affine 8-bit g64; `fold` is group-64 uint8+f16 (`markov_fold_q8.metal`, `~55 MB < 96 MB` SLC) and
  `markov_pick_rnn.metal` fuses pick + next-step RNN (`LAW=0` greedy, `LAW=1` Gumbel-max + parallel law probs). `draft.markov`'s op chain is the proof reference.
- **Batched decode**: slots are rows with their own cache and conv state (`attendSlots`, `state.get`/`put`); their
  reads run as one `relslots` launch per 8 slots (each slot's K, V and K's strides a buffer: a kernel takes 31) into
  `relcomb`. A CPU stream or several rows a slot run a slot at a time.

- **A verify of several slots is ONE forward** (`openGroup`, `Spec.opens`, `NativeDrafter.verify_group`/`finish_group`): blocks stack
  `rows` a slot (shorter ones padded), the conv kernels read `RPS` rows behind each slot's own state, `attendOpen` reads each
  slot's own cache, `commitLane` keeps a slot's accepted rows. A slot's rows equal its verify alone (proof test).
- **The DFlash2 draft batches across slots** (`draftBatch`, `pickBatch`): dense work over every lane's rows, attention
  per lane over its own context; it differs from a lane alone by matmul tiling (cosine 0.95+) and accepts the same.
  The bundled head drafts a slot at a time.
- **A DFlash2 group's width is the GPU's, the bundled head's 2** (`groupWidth`, `NativeDrafter.group_width`): 16 with NAX,
  8 elsewhere; past it the slots decode plain in the batched group. The head keeps too few drafts a grouped round for a
  wider group to beat the plain batched steps it replaces.
- **A quantized pack's joined projections are whole 64-row tiles** (`At.joinQuant`: zero rows behind the parts, `widths`
  the parts alone): the tensor unit's quantized matmul takes no other output width, and a verify's rows fell to a slow lane.
- **The q, k and v gates' down projections are a matmul a gate from 16 rows** (`gateDowns`): MLX's batched matmul leaves
  its few-row kernel there for one several times slower.

## Packs and proof

- The release needs no conversion. `tests/convert_inkling2.py --bits 4|5|6|8|16` writes mlx-lm's layout with the
  attention, embedding and head affine (the experts keep the release's bytes) and copies the draft head: at 4 bits
  its names, dtypes and shapes equal `mlx-community/Inkling-2-Mini-4bit`'s.
- The DFlash2 pair: a pack with the draft's two files in `drafter/`. Fixtures: `tests/dump_inkling2_dflash2_fixtures.py`
  (its header has the command), then `INKLING2_DFLASH2_DIR` + `INKLING2_DFLASH2_FIXTURES` (`inkling2 dflash2` parity)
  and `INKLING2_DFLASH2_PACK` (the rounds).
- `--experts BITS:GROUP[,LAYERS=...]` re-quantizes the experts to mlx-lm's affine (2-6, 8 bits, groups 32/64/128; a
  width off the `--bits` block is listed per module path in `quantization`). One to `packed_rows` rows read them in
  place (`ups`/`downs` FMT 3), more go through MLX's gather. Score a pack with `tests/inkling2_kld.py`: no bf16 experts.
- **An fp16 gather is for affine banks only** (`gatherRowsDtype`, GPUs before Apple9 that lack bf16): NVFP4 and MXFP8
  raw outputs carry the per-expert multiplier's inverse and overflow fp16 (the answer was `!!`).
- `proof_test.zig`: whole == chunked == stepwise on the tiny looped fixture (CPU), planted state mutations, the media
  splice, a verified block's commit, batched slots against each slot alone (CPU) or reordered and re-neighboured
  (GPU), and each kernel against its chain.
- `contract.zig`: `gated`, `state`, `relmask`, `act`, `sum`, `topk`, `gate` (its one-row arm), `relsplit`, `relattn_nax`,
  `relcomb`, `relslots`, `markov_rnn`, `markov_fold`, `markov_pick`, `oproj`, `router`, `norm`.
- Contract entries owed (GPU proofs in `proof_test.zig` cover each): `pre`, `prep`, `mid`, `ups`, `downs`, `relattn`,
  `gate`'s prompt arm, `group`, `ups_grp`, `downs_grp`, `few_tail`, `topk_group`, `tile_scan`, `moe_wide`,
  `markov_fold_q8`, `markov_pick_rnn`, `relsplit_nax`, `relsplit_lanes`, `commit_conv`,
  `accept_tree`, `sample_tree_draw`/`_mass`/`_search`, `dflash_conv`/`_edges`/`_top16_shard`/`_top16_merge`/`_tree`.
- Not verified: an image against mlx-lm's tower, the drafter's tokens against oMLX's.
