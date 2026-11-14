# prism_hadamard_qwen35 plugin

Owns `prism_hadamard_qwen35` (prism-ml Bonsai 2 packs). Table = `plugins/qwen3_5`'s load/forward with
`Config.hadamard_block`: the dense qwen3_5 trunk, every projection read through the `hadamard` quant format.

- Numerics are the pack's: f16 activations over f16 scales, an f32 GDN state; f32 unpacked tensors narrowed to f16 at
  load (`narrowHadamardPackTables`, run by the driver too); signs stay f32.
- Rotations: each `<linear>.signs` pooled in one `rht.Registry` per model (siblings share a handle, so a projection
  reading the same activation hits the memo); bases `Linear.rht` takes the rotated arm (qmv_nax2, qmv2 with the
  -scale fast path, stock qmm); `TableEmbed` and the MTP head's embedding rotate gathered rows back. A joined group
  keeps its join only where its parts share one vector; q|k|v takes no join and no decode side copy.
- Seams (`seam.zig`): `RotAddNorm` adds, norms and rotates in one dispatch (`rht.normRotate`, block 1024, <= 16 rows),
  seeding the memo; `RotNorm.want_normed` says whether the reader also needs the norm unrotated (a GDN layer, the
  final norm). The GDN decode step fuses the out projection's rotation (`gdn_decode.step`).
- The driver gives the shell a registry over the rotated table and head (`trunk_weight` lends `side`/`block`/`flags`),
  so the engine's draft readouts rotate as on the legacy path; the MTP head is grafted at load (`mtp_graft.zig`,
  served packs too), depth 2 by default (`spec.mtp_default_depth`).
- Proofs (`proof_test.zig`, tapes in `fixtures/`): steps, rollbacks, the lookup loop, a tied pack, batched ticks and a
  batched verify, an image prompt, and the MTP head (steps, picks, rounds) bit-equal to the legacy shell.
- The LOSSY `int8_prefill` setting arms the registry's int8 route (`Registry.int8`, `qmm_int8`) on prompt-width
  projections, as the legacy shell's.
- A DFlash drafter reads the trunk's layers and its rotated table and head through the shell (`lmHeadForeign`).
- Not served: routed experts (the format refuses rotated banks).
- A `qwen3_5` pack claims by marker when it is affine 2-bit group 128 and names one rotation block
  (`hadamard.block_size`, one shared `modules[].block`, or `hadamard.json`). Without that block it stays with
  `qwen3_5` and is declined. A `jang_config.json` zero-centred language norm is loaded as `1 + w` (not the GDN norm).
- A tower linear that ships `.scales` is not loaded, so an image request is the no-tower refusal.

## The row from the root architecture table

prism-ml Bonsai 2 = qwen3_5 behind Hadamard rotations (block 512-4096 per pack: `sdk/quant/hadamard/rht.zig`,
`hadamard_block` from `hadamard.block_size`, `modules[].block`, or `hadamard.json`); in the pack's numerics: f16
activations over f16 scales, f32 GDN state; fused QKV declines; MTP depth 2
