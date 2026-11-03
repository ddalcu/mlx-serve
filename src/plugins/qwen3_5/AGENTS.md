# qwen3_5 plugin

Owns `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text` (Qwen3.5 / 3.6 / 3.8 packs) and `minicpmv4_7`
(MiniCPM-V 4.7 on this trunk; its SigLIP2 tower + canvas positions are `minicpm_vision.zig` / `minicpm_canvas.zig`,
run by the engine). `qwen3_next` reuses this plugin's load and forward (`plugins/qwen3_next`).

- Forward: `Decoder(TableEmbed, RmsNorm, Either(Gdn, GatedAttention), GatedMlp | Moe, AddNorm, LmHead)`; GDN state
  through `CacheOps.state`. Bit-equal to the layered forward through the table (`proof_test.zig`: dense / MoE,
  4-bit / bf16, cold + warm chunks, decode, captured verify rows, every recurrent state and capture) and to the GDN
  batched driver (two slots: decode, verify rows, decode). `served_test.zig` proves the table through the engine's
  plugin driver over the legacy ModelConfig: steps, partial-accept rollbacks, the lookup loop, batched ticks and a
  batched verify, an image prompt on the M-RoPE table across a chunk boundary.
- Image towers (Qwen3-VL, MiniCPM-V): the engine runs the tower and builds the position table; the plugin declares
  `placeholder_ids` and `mrope` (interleaved sections, rope dims, the theta spectrum) and splices the rows.
- Load binds what the engine's loader binds: q|k|v|z|a|b joined (quantized), q|k|v joined on NAX for dense trunks
  (the row-exact lane join), decode side copies (`--decode-attn-quant`), routed banks' one-expert views, the
  shared-expert gate dequantized to a `[hidden]` vector; compiled GDN gate + routing where a GPU runs them.
- Facts: model_type `qwen3_5_moe` (the engine's one name for every layout), `batches_decode`, `exact_resumed`,
  `cache.layout` (linear vs full per `full_attention_interval`), `ssm_state` (one bf16 entry per layer, the
  checkpoint bytes), `tool_dialect = qwen_xml`, sampling defaults top_k 20 / top_p 0.95.
- MTP head (`mtp.zig`, `SpecHooks.head`, also qwen3_next's): the qwen-layout head, loaded by `sdk/spec/mtp`'s
  loader, its forward the sidecar arm's over the plugin's embedding and the head cache behind `cache_ops`; a MoE
  trunk's head runs its experts through the plugin's MoE block (no one-expert views, routing op by op, the shared gate
  as loaded, as the arm's `MoeMlpWeights`). The readout reads the trunk head `trunk_weight` lends. `mtp_test.zig`
  proves dense, qwen3_5_moe and qwen3_next heads through the driver against the legacy arm (history, merged and
  every-row steps, draft picks, a pinned-depth round loop).
- DFlash (`trunk_weight` + `dflash_capture`, every layout). With a drafter bound an affine dense trunk runs the
  `.mma` row-exact lane (`sdk/bases/blocks/row_exact.zig`, set at load where every projection fits): projections,
  joined groups at every lane width, attention through `lane_attn`/`row_attn`. `served_test.zig` proves it against
  the legacy `rowExactDecode` and a verify row against serial decode, bit for bit (`exact_drafted`). A dense trunk
  (bf16 too, as legacy) declares `spec_tree` (DFlash2 draft trees: tree-masked attention on per-depth RoPE, the GDN
  tree recurrence, the round's input state + prework left for the engine's commit); the `.mma` lane tiles its
  projections in place when a drafter binds (`SpecHooks.tile_lanes`). Both proven against the legacy trunk (and
  serial decode along each row's ancestors) in `served_test.zig`. bf16 and MoE
  trunks take DFlash without `exact_drafted` (a bf16 matmul's bits follow its row count; legacy claimed them).
- Parse declines: an image tower without M-RoPE sections, rope scaling, no output gate,
  `mlp_only_layers`, pooling, `quantization_config.quant_method` / `expert_quant` / `modules` (ModelOpt, EXL3; Hadamard
  outside `prism_hadamard_qwen35`),
  2-bit group-128.
- ANE prefill (`Caps.ane_prefill`, not Qwen3-Next or Prism): `WeightHooks.layer_weight` lends each dense MLP and
  joined q|k|v / z to the engine's build; a prompt chunk's `Forward.offload` takes them at the legacy seams
  (`GatedMlp`, the GDN projections). `ane_test.zig` proves the build's inputs and the seam against the legacy shell.
- Not carried by the blocks (so not declared): grouped-verify `projected` slices, int8 prefill.
- Prism packs (`plugins/prism_hadamard_qwen35`) are this plugin's dense trunk with `Config.hadamard_block`: the `.prism`
  body (`Prism` base), one `rht.Registry` per model, f16 activations, an f32 GDN state.

## Rules (from root AGENTS.md)
- Every qwen3.5/3.6/3.8 checkpoint is hd 256: read the CHECKPOINT before porting an hd-128 kernel.
- A GDN trunk's `KVCache.step` is 0 forever: batched rope offsets read the slot's `moe_seq_offset`, which the batched
  tick ADVANCES. Bar: `test_batched_equivalence.sh`.
- A grouped verify row equals its solo verify only while both run the SAME arm (#727): the MoE rows arm stops at 8
  rows, past it the sorted kernels' bits differ from solo (accepted: joining is faster).
- A `<tool_call>` body carrying `<function=` is the XML dialect and is read FIRST (the 3.5+ template mandates it).
- Qwen3.8: the THIRD effort vocabulary (`xhigh|medium|low`); the thinking-off refusal is sniffed on the refusal
  string, only a refusing template gets `noThinkTailSuffix`; `preserve_thinking` keeps the template's default.
- A MiniCPM-V image is SEVERAL encoder calls (thumbnail + crops); the canvas M-RoPE table is built from the EXPANDED
  ids (`minicpm_canvas.canvasPositions`).

## Known issues
- The shared-expert gate vector (`SharedExpert.gate_vec`): dropping it is not caught by the proof (the router
  kernel's folded arm gives the same bits or is not engaged at the fixture's shapes).
