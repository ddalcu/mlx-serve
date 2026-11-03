# gpt_oss plugin

Owns `gpt_oss` (OpenAI gpt-oss 20B/120B, mlx-lm layout; MLX packs: mxfp4 experts, affine-8 attention overrides).
Served (see "Served").

- Forward: `Decoder(TableEmbed, RmsNorm, DecodeCopies(SinkAttention), Moe, AddNorm, LmHead)`, both blocks this folder's.
  `attention.zig` (copy of `sdk/blocks/attention/sinks.zig` without MiMo's arms): additive q/k/v/o `.bias` (beside
  the quantizer's `.biases`: never read one for the other), no q/k norm, learned sinks (`self_attn.sinks`, or
  `attention_sink_bias`), YaRN on every layer with the mscale over the whole head, sliding(128) layers banded at
  prefill and window-masked at decode. `moe.zig`: router `.bias`, softmax top-k, per-expert `.bias` on gate/up/down
  added in SORTED order, the clamped SwiGLU `clip(g, max=l)·sigmoid(1.702·g)·(clip(u, ±l)+1)`; the biases force the
  sorted path at every width, as the legacy (no fused decode arm has a bias slot). Serial, as the legacy.
- Parse: `num_local_experts`/`experts_per_token` spellings; the expert width IS `intermediate_size`; ONE theta for
  both layer types; YaRN mscale COMPUTED (0.1·ln(factor)+1, the config ships none); context = the YaRN window;
  terminators 200002 and 200012 joined (never 200007, which closes the analysis channel). Chat facts: thinking on
  (Harmony has no off), the harmony dialect. Declined (`proof_test.zig` lists every shape): `rope_parameters` (the
  legacy read YaRN only from `rope_scaling`), any non-YaRN scaling, a separate expert width, `swiglu_clamp`, no
  `layer_types`/`sliding_window`/`head_dim`/`swiglu_limit`, a malformed value. Read as the engine reads them:
  `text_config` nesting, `vision_config`, `pooling_mode`, a missing `vocab_size`/`intermediate_size`. The released
  `mlx-community/gpt-oss-20b-MXFP4-Q8` config at the proof geometry replays a live-recorded tape.
- Decode-attn-quant side copies through laguna's `DecodeCopies`. Load refuses by name: missing per-expert biases (only
  the biased sorted path is here).
- Legacy quirk mirrored: the config's YaRN `truncate: false` is NOT read (the engine reads `truncate` only from
  `rope_parameters`), so the frequencies are the truncated ones. Worth checking against mlx-lm before flipping.
- Proof (`proof_test.zig`): 4-bit, and bf16 with decode-attn-quant off and on, bit-equal to the recorded
  `forwardMoeWith` (steps crossing the window, captures, greedy). Mutations: no expert bias, no mscale, no sinks, no
  YaRN table: each fails. The decode window mask is all-visible over a window-sized view (dropping it is bit-equal
  at these shapes; kept).
- Drafters: `SpecHooks.trunk_weight` (the bare table and head) + `caps.dflash_capture`, so a pack with an in-dir
  DFlash sidecar binds the plugin (the legacy served it); the drafted stream replays one recorded from the legacy
  forward, whose embedding rows and draft head matched bit for bit.

## Served

In `served.zig`, on the legacy parse's `ModelConfig`. The legacy forward is deleted (`served.legacy_removed`: a pack
the plugin declines is refused by name): the gpt_oss load arms, its sink-attention entry (MiMo keeps
`sinkAttnWith`), and the legacy MoE's router bias, per-expert biases and unclamped `gptOssSwiGLU` arm (this plugin's
`moe.zig` keeps the activation). Proofs replay tapes recorded from it (`fixtures/*.tape`). A `--mlx-gguf` gpt-oss
GGUF (`sdk/quant/gguf/arch/gpt_oss.zig`, `quantization.mode: gguf`) serves here: block-quantized tensors through the
gguf kernels off their sentinels, float ones dense (the loader's bf16; `llama/load.zig` `Reader`). Proof: a pack of
q8_0 table, head and attention, mxfp4 experts and a bf16 router replays tapes recorded live against the legacy
forward (`fixtures/gguf-*.tape`, recorded on a pre-deletion tree with this plugin; dropping the projection biases
failed them there).
YaRN `truncate: false` matches the legacy (logged for after).

## Rules moved from the root AGENTS.md

- None stand alone for gpt_oss; the Harmony/channel and `ensureGptOssTerminators` rules are host chat rules.
