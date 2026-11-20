# PLAN 04 — Compressed IndexPool indexer (DSA layers)

Phase: 01-hy4-flash-port | Wave: 2 | Depends on: 01, 02, 03
Requirements: HY4-04
Files: `src/plugins/hy_v4/blocks/indexer.zig`, indexer kernels + contract, cache/state files for the index pool, `tests/test_hyv4_dsa.sh` (add)

## Objective

Replace Plan 03's DSA placeholder with the real selection: pooled key scoring → top-512 groups → expand to top-2048 token indices → MLA attention gathers only those keys. 12 layers; 30 SWA layers untouched. This is the highest-risk subsystem — keep deepseek_v41's indexer reads as the structural reference, the HF math as the spec.

## Reuse bill (copy `//! copied_from:`)

- Skeleton: `glm5_next` DSA arm — `glm5_next.zig` `Indexer`, `indexerScores`, `maskedScores`, `selectTokens`/`selectRows`/`decodeSelect`, `expandSelection`/`topkExpand` (+ `kernels/{dsa_indexer_nax,topk_expand,dsa_sparse_nax,pooled_rows,dec_score}.metal`), `withPooledKeys`/pool state + `slots.zig`.
- From `deepseek_v41`: `decode/idx_score.zig` (pool+gate semantics), `decode/topk.zig` stage pickers, `cache_codec.zig` row codec, `LState{win,pend_kv,pend_gate}`/`State` per-slot shape, `plugin.zig` `CacheHooks.aux`/`state_bytes`/`openBank` shape; `sdk/blocks/attention/sparse_block/row_mask.zig` `qsa_row_mask` as-is.
- Contract: copy `deepseek_v41/contract.zig` + `glm5_next` kernel-contract arms; `sdk/testing/kernel_contract` 5-gate per REGISTERING.md.

## Tasks

### Task 1: compressed indexer math (spec + tests first)
- Implement per SEMANTICS §3 + §11: `key_value_projection`+`key_gate_projection` (gated pooling) → 4× pooled group keys (`key_norm`); `query_projection` (16 heads × 128); per-group score = per-head dot × `head_weight_projection` weights, `intra_group_ape` [4,128] positional-encoding add inside groups; top-512 groups; expand group members as candidates; `token_key_projection`+`token_key_norm` fine score over candidates; sum across heads → top-2048 indices per query. `skip_fine_select: true` = the two-stage candidate path only.
- [AMBIGUOUS flags] head combination exact formula + indexer rope/nope — pin from the §8 evidence reader's HF citations before coding; if still open, implement behind a unit-test both arms.
- Unit tests: on fixture hidden states, selected index set vs HF reference set — exact match prefill, ≥99% overlap decode; golden fixtures from `tests/dump_hyv4_fixtures.py` indexer dumps.
- **Verify:** `zig build test -Dtest-filter="hy_v4 indexer"` green; contract for pooling/top-k/gather kernels (float64 reference, planted controls fail).
- **Done:** selection parity proven on fixtures; both stage shapes (pool, gather) have contract coverage.

### Task 2: pool cache + attention gather + prefill/decode
- Index-pool state per slot (pooled keys, token keys, counts) alongside the KV; prefix-cache invalidation rules same as KV; SSD spill consistent with latent KV tiering.
- DSA attention reads: gather selected 2048 latent rows + kv_b expand → softmax with sinks; window floor (recent tokens always in) if reference says so (§3 cite); prefill processes query blocks (stage-2 block 8192) — chunk boundaries yield to decode ticks like other prefills.
- End-to-end: full pack greedy parity on long contexts (≥32k) vs HF reference; engagement log lines per DSA layer.
- **Verify:** `tests/test_hyv4_dsa.sh` — short and long greedy parity, prefix-cache reuse across turns, memory bounds (pool ≤ configured bytes); green on 512 GB host.
- **Done:** DSA layers attend only selected keys at parity; [batched]/spec gates updated honestly (spec stays off until Plan 06).

## Verification (plan-level)
DSA parity test green; indexer selection exact-match test in CI on fixtures; lint 0.

## Execution notes (plan 04)

- The HF reference's compressed indexer is ONE stage (`compressed_indexer_skip_fine_select: true`): gated fp32
  pooling over complete groups -> group keys roped at the group start -> `relu(q @ keys^T)` per head, mixed by
  `head_weight_projection(x) * (heads*dim)^-0.5` -> best `group_topk - 1` closed groups + the query's own group.
  The plan's "fine token-key score over candidates / top-2048" belongs to the skipped stage; Task 1's spec line
  is corrected here.
- Own-group follows the query's SLOT (`query_slots // ratio`), not its absolute position: a prefix-cache restore
  keeps slots contiguous while positions carry the prefix; rope positions stay absolute.
- Selection is an additive mask over the slot axis (causal + recall), which is exactly the reference's arm
  (`_mask_unrecalled_tokens` masks a dense causal mask). Below `index_topk` (2048 tokens) the layer attends
  densely. The gather-only kernel is a perf step, deferred.
- DSA layers cache their per-token indexer key beside the latent (K row becomes
  `latent | k_rot | token_key`, 704 halves; V stays the latent). The bill says 1216 halves/token/DSA layer.
- Oracle: pooling + head-weighted recall + own-group + causal, per slot, against f64 (12-token fixture, ratio 4,
  group_topk 2), on a `TestCache`-shaped history.
- Deferred: incremental pool state (groups pooled once, not re-pooled per step), the gather kernel, batched
  decode (`batches_decode` stays off), and the `tests/test_hyv4_dsa.sh` long-context script (needs the full pack).
