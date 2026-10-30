# qwen3_next plugin

Owns `qwen3_next` (Qwen3-Next-80B-A3B). Table = `plugins/qwen3_5`'s load/forward with `Config.next`: flat `model.*`
weights, GDN q|k|v|z and b|a projections combined per key head (`Gdn.Weights.combined`), the pipelined-prefill
table keyed `qwen3_next`, the rotary factor read at the config root, routed experts with a gated shared expert.

- Bit-equal to the legacy layered forward through the table (`proof_test.zig`: 4-bit, bf16 with side copies; chunks,
  decode, captured verify rows, every recurrent state).
- Decodes serial (`batches_decode` false: the engine batches only the qwen3_5 MoE trunk); `exact_resumed`.
- DFlash (`trunk_weight` + `dflash_capture`): the tiny assistant over two GDN layers emits the legacy stream with the
  same accepted counts (`proof_test.zig`). A pack's MTP head runs through qwen3_5's head hooks (`mtp_test.zig`).
