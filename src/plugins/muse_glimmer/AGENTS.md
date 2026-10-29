# muse_glimmer plugin

Owns `muse_glimmer` / `muse_glimmer_text` TEXT trunks (Muse-Glimmer-30B: dense GQA 32/2, hd 128). Served (see
"Served"). `vision.zig` is the tower, run by the engine's `VisionEncoder`, not by this plugin.

- Forward: `Decoder(NormedEmbed, RmsNorm, PostNormed(DecoderAttention(.sliding, .gate = .channel)),
  PostNormed(SwiGlu), ResidualTail, SoftcapHead)`. `embed.zig` is a copy of `TableEmbed` plus the weight-less RMS
  norm the reference applies to looked-up rows (never folded into the table).
- Load: sandwich norms are Gemma2-centred (`1 + w` folded at load, bf16) but `model.norm` is plain-scale; post-norms
  run at `post_norm_eps` (1e-8, `Shared.post_eps`); q/k norms are weight-less (bf16 ones `[head_dim]`) and
  `qk_scale_factor/sqrt(head_dim)` is the SDPA scale; the per-channel gate is `self_attn.gate_proj`, joined into
  q|k|v|g for one-row steps of a quantized layer; logits times `output_multiplier`, then the softcap through the same
  shapeless compiled closure the legacy load builds.
- Parse: `text_config` nesting picks the `model.language_model` prefix; ONE theta for every roped layer
  (`rope_parameters.rope_theta`) unless `rope_local_base_freq` is given; NoPE where `layer_rope_theta` is 0; the
  template's terminators 200001 and 200008 join the eos set (never `<|eom|>`). Chat facts: thinking on with tools,
  the muse reasoning dialect, prior-turn reasoning dropped; a `vision_config` sets `Caps.images`. Declined: hd 256,
  per-type rope blocks, a scale switched off.
- Proof (`proof_test.zig`): 4-bit nested and bf16 flat packs at hd 128 bit-equal to `forwardStandardWith` (steps
  crossing the window, captures, greedy) and `forwardBatchedDecode`; media rows across a chunk boundary; a greedy
  DFlash run emits the legacy stream and acceptance. Mutations: RoPE on NoPE layers, scale without
  the factor, post-norm eps = rms eps: each fails both.

## Served

DFlash: `trunk_weight` hands the bare table and head (the drafter applies no norm, multiplier or softcap, as the
legacy `rawEmbedding`/`lmHeadForDraft`); `dflash_capture` set. Media rows splice after the normed embedding at the
root's `image_token_id` (`image_token_index`), as the legacy. A vision pack is served too: the engine loads its tower
into its own map beside the plugin (`driver.loadWith`) and its `VisionEncoder` makes the rows. On the legacy parse's
`ModelConfig`. The legacy forward's muse arms are deleted (`served.legacy_removed`: a pack the plugin declines is refused
by name): the normed embedding, the plain final norm, the weight-less q/k norm, the channel gate (`attnOutGate` and
its row join) and `output_multiplier` in `forward/dense.zig`, `batched/dense.zig` and the loads. NoPE and the sandwich
norms stay there for the families that still read them. Proofs replay tapes recorded from it (`fixtures/*.tape`).
Parse: nested or flat, a missing vocab or MLP width (off the tensors), `pooling_mode`; declined (`proof_test.zig` lists
each shape with what the legacy did): `rope_scaling`, MoE and KV-sharing keys, `global_head_dim`, bidirectional
attention, `attention_k_eq_v` (keys the legacy dropped or mis-served), hd 256, a missing structural key.

## Rules moved from the root AGENTS.md

- Supported-architectures row: sandwich norms, qk-norm folded into the scale, sigmoid out-gate, NoPE every 4th layer,
  softcap 20. Harmony channels + ATEM tools; thinking ON with tools / OFF without (`defaultEnableThinking`). CASED
  Llama-3 pretokenizer. Vision `muse_vision.zig`; DFlash via `-assistant` sidecar.
- Config reads: a default only ONE family wants is pinned PER LAYER TYPE (muse `rope_local_base_freq`, guard
  `tests/test_muse_repetition.sh`).
- The muse channel-header, `to=self` reasoning and prior-turn-reasoning rules stay in the root (host chat code).

## The row from the root architecture table

Muse-Glimmer-30B (dense VL, GQA 32/2 hd128): sandwich norms, qk-norm folded into `attnScale()`, sigmoid out-gate, NoPE
every 4th layer, softcap 20. Harmony channels + ATEM tools; thinking ON with tools / OFF without
(`defaultEnableThinking`). CASED Llama-3 pretokenizer (`.llama3`). Vision `vision.zig`; DFlash via `-assistant`
sidecar.
