# PLAN 02 — Weight loader + memory bill + stream bank

Phase: 01-hy4-flash-port | Wave: 1 | Depends on: 01
Requirements: HY4-02
Files: `src/plugins/hy_v4/weights.zig` (or per sdk pattern), `src/plugins/hy_v4/loader_test.zig` section, `src/plugins/hy_v4/kernels/` (digest registrations), `tests/dump_hyv4_fixtures.py` (stub)

## Objective

Every tensor in the 148-shard pack opens, validates against the config bill, and lands in the right place: resident affine-quant arrays (g64), the 4 engram tables through the streamed SSD bank (kit `streamed`, g32), iHC f32, DFly bf16, the `I32[120818]` engram lookup tables, and the `vit.*` tower (loaded here, used in Plan 07). Load peak must match the bill on the 512 GB host.

## Tasks

### Task 1: loader + name map
- Map HF/pack names → internal: the pack writes `model.layers.N.*` with N at the **pack index**; internal layer index is HF. Engram and indexer placement come from weight presence, not config lists (Plan 01 reconciles); `hc_attn_layer`/`hc_mlp_layer`/`dfly_layers`/`vit` keep their prefixes verbatim.
- Quant dispatch by the config quant map: default 4-bit g64 affine packed U32 + BF16 scales/biases (existing affine loader path); per-tensor g32 for `engram_layer.multi_head_embedding.embedding`; F32/BF16 verbatim for iHC, DFly, router gate, norms, sinks, `intra_group_ape`, indexer small matrices. If the pack carries NO `quantization_config` (reader flags this as possible), the loader derives the layout from the config's `quantization.quantizations` per-tensor overrides, else a documented name-regex default (g64 4-bit; g32 for `engram.*.embedding`) — and record the finding as a gotcha entry candidate.
- **33 GB `biases` tensors** (engram): shape says `[rows,3]` = a packed 96-d row (48 nibbles + scale + bias per the U8 packing), NOT per-group biases — [AMBIGUOUS, confirm vs `engram_fp8_embedding`]; loader keeps raw, Plan 05 Task 1's gather math owns the semantics; unit test asserts the packing identity on fixture rows.
- **Padded nope halves** in `q_b_proj`/`linear_gate`/`o_proj` (8192 = 2×4096): load raw, zero-pad validate; [AMBIGUOUS: padding vs 64-head SWA read] — Plan 03 resolves with an A/B, decision noted here and in gotchas.
- Engram tables (`U32[164,994,380, 12]` + scales/biases `[164994380, 3]`): stream bank rows, page-aligned; the flat row index = `(order * 8 + head) * 10,312,000 + hash_slot + 2` — loader validates total rows against the config head-vocab lists (pad rows 0/1 + per-order head tables); refuse on mismatch, do not clamp.
- Validate: every expected tensor present (census count 3,172), shapes match config math (incl. the 2×-padded nope halves in `q_b_proj`/`linear_gate`/`o_proj` — [AMBIGUOUS: padding vs 64-head SWA read] — loader keeps raw, forward decides, note in gotchas).
- **Verify:** `zig build test -Dtest-filter="hy_v4 loader"` (fixture/tiny-pack shapes); on the 512 GB host a load-to-mmap run reports resident vs bill within tolerance and engram rows in the bank.
- **Done:** `--model <pack>` progresses past load with the full resident set mapped and engram in the stream tier; a missing-tensor fixture errors with the tensor's name.

### Task 2: stream bank + eviction + preflight bill
- Register the 4 engram tables as streamed banks: LRU page policy, eviction returns memory, preflight counts resident-only vs streamed, admission never charges the 132 GB to a request peak.
- Guard tests mirroring `tests/test_ds4_ssd_streaming.sh` expectations (name-keyed banks, free-disk via `msv_volume_free_for_use`).
- **Verify:** load on a constrained-RAM box (or a test with a shrunken resident budget) shows engram pages fault in/out and RAM returns on unload; `zig build test -Dtest-filter="hy_v4 stream"` green.
- **Done:** load peak ≈ resident bill (≈ 88 GB + tower + DFly), not the 121 GB total.

## Verification (plan-level)
Load succeeds on the 512 GB host with the real pack (log line: bill == measured within tolerance); unload returns memory; unit tests green on hermetic fixtures; kernel-digest registration unchanged (no kernels yet).

## Execution notes (2026-11-19) — committed 07eb701b

- Task 1 + the Engram-open half of Task 2 landed: `weights.zig` (load/Holder/validate with the
  packed-or-dense dual-arm checks, pread `Engram` tables over shard header ranges, per-stage
  error sets so refusals name what failed), `fixture.zig` (the conformance micro-pack; the
  subject returns null until Plan 03's forward), `loader_test.zig` (real-pack census, tiny-pack
  load through the SDK loader AND the table-open arm, and a missing-tensor refusal test).
- `engram_layer_ids` and the head-vocab lists are HF (1-based): config stores pack index
  `id-1` (matches omlx's `block + 1`), and `engram_table_rows` (head-vocab sums, pack-indexed)
  is cross-checked against the pack's tensor rows at load — refused, never clamped.
- The engram row is 60 B (12 U32 codes + 3 bf16 scales + 3 bf16 biases); the bill test pins the
  39.60 GB census within 0.1% and the resident bill = shards − that figure.
- The 33 GB `biases` [AMBIGUOUS] line: the pack's tables carry per-row scale+bias pairs as
  documented above; Plan 05's gather math confirms the fp8-embedding reading.
- Carried to Task 2's second half (wave 2): eviction/LRU policy + a constrained-RAM load test;
  kernel digest registrations ride Plan 04 (first kernel copy).

