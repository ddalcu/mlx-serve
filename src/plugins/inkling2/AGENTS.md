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
- **A step runs under a 200 MB command-buffer split** (`command_buffer_mb`, set per forward through
  `HostGpu.command_buffer_limits`): MLX counts a bound buffer's whole size, so at its 50 every expert launch (a whole
  bank) committed and the encode stalled on MLX's in-flight cap. A prompt keeps the host's split: a wider one slows it.
- **Dense projections stay in the checkpoint's `[out, in]` layout** (`layers.projectRows`): MLX's one-row matmul over
  a pre-transposed `[in, out]` weight costs several times the latency of `x @ W.T`. The message gates' down
  projections are held `[4, rank, 2 D]` for the same reason.
- **Every step kernel takes rows on grid y**, so `kernels.visit` serves one row, a prompt's rows and batched slots;
  the conv kernels read one state the rows chain behind (`SEQ` 1) or a state per row (`SEQ` 0, batched slots).
- **Reads of at most `small_rows` rows build a mask (`relmask.metal`) and use MLX's attention**: the flash kernel
  (`relattn.metal`, mlx-lm's, MIT) walks the keys 16 at a time on one threadgroup, a prompt's throughput and a few
  rows' latency.
- **2 to `packed_rows` rows read the packed expert banks in place** (`ups`, `downs`, then `sum` over one pick a
  row); past that MLX's gather, which reads a bank once for all its picks, wins.
- **A threadgroup past 256 threads is proven per instantiation** (`unproven`): an M1/M2 compiles `pre` and `mid` for
  384 to 576 threads, under the 768 an M3+ runs. The first dispatch is evaluated and a refusal narrows
  `stream_threads` for the process; every other kernel's group is at most 256.
- **The activation and the message gates use `precise::tanh` / `precise::exp`, the sigmoid divided once**: the fast
  tanh is NaN at an infinity and a percent off near zero, and `capped * (1 / (1 + e))` flushes a tiny sigmoid to zero.
- **`split_from` to `split_rows` rows (4 to 8) read through `relsplit` + `relcomb`**: MLX's attention reads the keys
  once per row and head, the split kernel once per KV head over threadgroups of 128 keys. At 2 or 3 rows they tie,
  and the mask path keeps them.
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
- **The draft head** (`mtp.safetensors`, greedy and sampled requests): a round drafts 7 tokens,
  verifies the prefix the drafter is sure of (`min_sure`, never a stop id: `draftsToVerify`) as ONE forward that appends nothing (`kernels.Open`), and
  commits the rows the trunk agrees with (`kernels.commit`): no rollback. Its taps are per engine slot (`Spec.select`).
- **A drafter layer `local_layer_ids` leaves out reads every committed position** (`appendFull`, billed as `Bill.state_per_token`),
  the others the ring's newest window - 1: read as a sliding layer, the release's layer 1 drafted blind past 1K tokens.
- **The draft head loads on Apple9+ only, with the kernels on** (`draftsOn`, told to the engine by `NativeDrafter.active`):
  on an M1 a few-row verify costs so much more than a step that drafted decode ran under serial, and only the kernels
  serve a verified or batched visit (`MLX_SERVE_INKLING2_FUSED=0` also batches no slots).
- **A sliding visit keeps a ring of window + 2048 rows** (`Config.ring`, declared as `CacheFacts.sliding_ring`;
  `layers.appendKv` at every append): kept whole, the 40 sliding visits held several times the KV a long prompt needs, and every
  prefix-cache entry with it. A match more than the slack behind an entry's end then misses.
- **The draft head's Markov chain runs three launches a step** (`markovStep`: `markov_rnn`, `markov_fold`,
  `markov_pick`) and picks in MLX's argmax order (a NaN first, the lower index on ties); `draft.markov`'s op chain
  serves any other stream or dtype and is the proof's reference.
- **Batched decode**: slots are rows with their own cache and conv state (`attendSlots`, `state.get`/`put`); their
  reads run as one `relslots` launch per 8 slots (each slot's K, V and K's strides a buffer: a kernel takes 31) into
  `relcomb`. A CPU stream or several rows a slot run a slot at a time.

## Packs and proof

- The release needs no conversion. `tests/convert_inkling2.py --bits 4|5|6|8|16` writes mlx-lm's layout with the
  attention, embedding and head affine (the experts keep the release's bytes) and copies the draft head: at 4 bits
  its names, dtypes and shapes equal `mlx-community/Inkling-2-Mini-4bit`'s.
- `--experts BITS:GROUP[,LAYERS=...]` re-quantizes the experts to mlx-lm's affine (2-6, 8 bits, groups 32/64/128; a
  width off the `--bits` block is listed per module path in `quantization`). One to `packed_rows` rows read them in
  place (`ups`/`downs` FMT 3), more go through MLX's gather. Score a pack with `tests/inkling2_kld.py`: no bf16 experts.
- **An fp16 gather is for affine banks only** (`gatherRowsDtype`, GPUs before Apple9 that lack bf16): NVFP4 and MXFP8
  raw outputs carry the per-expert multiplier's inverse and overflow fp16 (the answer was `!!`).
- `proof_test.zig`: whole == chunked == stepwise on the tiny looped fixture (CPU), planted state mutations, the media
  splice, a verified block's commit, batched slots against each slot alone (CPU) or reordered and re-neighboured
  (GPU), and each kernel against its chain.
- `contract.zig`: `gated`, `state`, `relmask`, `act`, `sum`, `topk`, `gate` (its one-row arm), `relsplit`, `relcomb`,
  `relslots`, `markov_rnn`, `markov_fold`, `markov_pick`, `oproj`, `router`, `norm`.
  Owed: `pre`, `prep`, `mid`, `ups`, `downs`, `relattn`, and `gate`'s prompt arm.
- Not verified: an image against mlx-lm's tower, the drafter's tokens against oMLX's.
