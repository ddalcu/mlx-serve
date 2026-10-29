# gemma4 plugin

Owns `gemma4`, `gemma4_text`, `gemma4_unified(_text)` (mlx-lm's layouts). The text decoders (E2B/E4B with
per-layer embeddings and a KV-shared tail, 12B/31B, the 26B-A4B experts), with image and audio rows spliced at their
placeholders (a unified pack's encoder-free embedder rows too), are served bit-equal to the legacy forward
(`gemma4`, `gemma4_text`, `gemma4_unified(_text)` in `served.zig`); draft-model configs (DSpark / DFlash / EAGLE-3
keys) are declined `unsupported`. `bidirectional.zig` is the encoder base of both EmbeddingGemma plugins.

## Files
- `config.zig`: MLX-free claim + parse over Gemma 3's `readCommon`/`finish` with the Gemma 4 traits (nested
  prefix, plain norms, value norms, no rope scale), `layout` shared with Gemma 3.
- `decoder.zig`: its own skeleton (Gemma 3's attention and GeGLU blocks, PLE input + per-layer PLE seam, layer
  scalars folded into the residual kernels at decode, softcap head). The weights load through `gemma3/load.zig`.
- `moe.zig`: the sigma-MoE experts (router scale folded with `hidden^-0.5`, softmax top-k, `per_expert_scale`, GeGLU
  banks): a copy of the SDK MoE block's arms that a tanh-GELU pack takes (house rule 3: the shared block is not changed).
- `gelu_ple.zig` + `kernels/gelu_ple_proj.metal`: the fused decode PLE tail; `contract.zig` registers it
  (`kernel_contract/registry.zig`, all five gates).
- `plugin.zig`: the table, `trunk_weight` (embedding table / head for the Gemma 4 assistant drafter),
  `conformanceFixture`; `proof_test.zig`: E-series 4-bit and 31B-like bf16 packs vs the legacy engine.
- Legacy engine files still here (moved in P1): `attention.zig`, `moe_tail.zig`, `ple.zig`,
  `assistant_drafter.zig`; they go when the legacy Gemma 4 path does.

## Facts the table declares
- `dflash_capture` (the forward fills `capture_layers`): the driver adopts the trunk's table and head, so the
  assistant drafter (and a DFlash sidecar) serve on the plugin path; proven by `expectDrafter` (tokens, acceptance,
  each round's draft logits bit-equal on E-series and 26B-A4B trunks).
- `batches_decode` (dense packs; a pack with experts decodes serial and declares `moe`), thinking off with the
  `gemma_channel` dialect, `caps.images`/`audio` + `placeholder_ids` with towers, `exact_resumed` (`docs/known-issues.md`: resumed yes, drafted and concurrent
  no); stop set `{config eos} + {1, 106}`; sampling defaults top_k 64 / top_p 0.95; `weights.prefix`
  `language_model.model`, probed to `model.language_model` / `model` when the pack keeps its tensors there.

## Rules (moved from root AGENTS.md)
- Gemma 4 attention runs at scale 1.0 (the q/k norms carry it); global layers rope a partial proportional table
  (`partial_rotary_factor`), sliding ones the local base; full layers may be K=V (`attention_k_eq_v`, no v_proj).
- A KV-shared layer reads the last earlier layer of its type (`Config.kvSource`); it projects only q.
- The PLE tail at one decode row is ONE dispatch (`fusedGeluPleProj`): bit-equal to the table lookup, multiply and
  qmm, and it shortens the dependent chain, which is why it pays on the small E models.
- Per-layer inputs read id 0 at image positions (the reference's `text_mask`); media rows splice before them.
- A 26B-A4B layer's router reads the RAW residual after attention, the experts its `pre_feedforward_layernorm_2`
  norm. fp4 experts are served; the legacy one- and two-row fp4 expert kernel applies SwiGLU, so a legacy decode on
  an mxfp4/nvfp4 26B-A4B pack was wrong and is no oracle (`proof_test.zig` checks decode against the verify row).
- `gemma4_assistant` is a drafter, hidden from discovery; the Gemma 4 drafter reads the trunk's table and head via
  `trunk_weight`, never a copy.

## Known issues
- `bill` returns the defaults until the engine prices from the layout (P8).

## Fixtures
See `fixtures/README.md`.
