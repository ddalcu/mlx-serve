//! DeepSeek-V4.1's levers in one table (`rows`): each lever's name, stack, class, default, lossy pick, stack-off arm,
//! parent, engage line and README text. Every read site reads a lever through this file (`get`, `onUnlessZero`, `on`,
//! `int`, `span`), and the README's switch tables and counts, the engage lines and the harness check list
//! (`accepted-lines.txt`) are rendered from the table and tested against it. A lever's own variable,
//! `MLX_SERVE_DSV41_<NAME>`, stays its per-lever input and wins over everything else.
//!
//! Resolution, when the lever's own variable is unset:
//! 1. its stack's switch (`MLX_SERVE_DSV41_STACK_<STACK>=0`) is off: the row's off arm (`=0` for a default-on lever);
//! 2. `MLX_SERVE_DSV41_PRECISION=lossy` (`--precision lossy`): the row's lossy pick, where it has one;
//! 3. else the row's lossless pick; for most rows unset, so the read site's own default serves.
//! The bill, admission and the forward read the same functions, so a load's memory plan sees the lossy levers before a
//! model exists. Precision and the stack switches are read once (the first read, then each load's `check`), kept in
//! one `Shared` state.

const std = @import("std");
const env = @import("../../env.zig");
const shared = @import("../../sdk/util/shared.zig");
const log = @import("../../sdk/util/log.zig");
const acceptance = @import("../../sdk/spec/mtp_acceptance.zig");

pub const Stack = enum { prefill, attention, decode, spec, memory, loader };
/// The README's classes. `requant` is HEAD8 alone, the one accepted requantization that lossless keeps.
pub const Class = enum { exact, reorder, tf32, draft_only, requant, lossy };
/// `on`: a default-on lever; `off`: an opt-in; `setting`: a value the load decides with (no stack switch reaches it).
pub const Kind = enum { on, off, setting };
/// A resolved raw value: `site` leaves the variable unset, so the read site's own default serves.
pub const Pick = union(enum) { site, value: [:0]const u8 };
/// The line a served lever logs: `canonical` is `[dsv41] MLX_SERVE_DSV41_<NAME> engaged`, `line` a row's own
/// prefix, `none` the reason the row logs none.
pub const Engage = union(enum) { canonical, line: []const u8, none: []const u8 };

pub const Row = struct {
    name: []const u8,
    stack: Stack,
    class: Class,
    kind: Kind,
    lossless: Pick = .site,
    lossy: ?Pick = null,
    /// The value its stack's switch gives it when off; null: the switch leaves it alone.
    off: ?Pick,
    parent: ?[]const u8 = null,
    engage: Engage,
    /// Listed in `accepted-lines.txt` (false: not reached on the standard test's route).
    accepted: bool = true,
    /// The accepted-lines text when it is more than the engage line.
    accepted_line: ?[]const u8 = null,
    class_text: []const u8,
    what: []const u8,
    evidence: []const u8,
};

pub const prefix = "MLX_SERVE_DSV41_";

/// The DSpark typical rule a sampled request takes under `precision=lossy` when nothing else sets its acceptance:
/// Medusa's standard pair, a draft kept iff p > min(eps, delta e^-H) (`--mtp-typical 0.3 --mtp-typical-eps 0.09`).
/// The 16K standard cells with card sampling: decode 96.05 tok/s against exact's 88.25 (+8.8%), 3.002 tokens a round
/// against 2.797, prefill unchanged; HumanEval+ 162/164 base and 155/164 plus against exact's 161/164 and 155/164.
pub const lossy_typical: acceptance.Typical = .{ .delta = 0.3, .eps = 0.09 };

/// The typical pair a DeepSeek-V4.1 sampled request takes by default: `lossy_typical` under `precision=lossy`, else
/// null (exact). An explicit `--mtp-typical`, `--mtp-tokenv3` or model `mtp_acceptance` setting wins (the engine's).
pub fn lossyTypical() ?acceptance.Typical {
    return if (precision() == .lossy) lossy_typical else null;
}

pub const rows = [_]Row{
    .{ .name = "PROMPT_ASYNC_EVAL", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Commits the last wide prompt layer without a wait", .evidence = "test (word for word)" },
    .{ .name = "PF_CED_HC", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "CED compressor hc pre-norm once over the pass", .evidence = "test (word for word)" },
    .{ .name = "PF_ROPE_PAIRS", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Query-tail rope computes each angle once per pair", .evidence = "test (word for word)" },
    .{ .name = "PF_RMS_ROWS", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Query-latent rms reads its rows in place", .evidence = "test (word for word)" },
    .{ .name = "CED_LAZY", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .accepted = false, .class_text = "EXACT", .what = "Runs a CED replay once, for its first reader", .evidence = "test (word for word)" },
    .{ .name = "IN_PROJ_MIXR", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Verify in_proj hc mixes as one threadgroup per (mix, row)", .evidence = "test (word for word; M3 or later)" },
    .{ .name = "HC_EXPAND_NORM", .stack = .prefill, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER", .what = "FFN-side hc expand and norm in one launch", .evidence = "A/B text (its test allows 1% relative error, not counted as a bound)" },
    .{ .name = "PF_MOE_EPI", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Counting-sort route pack in place of two argsorts", .evidence = "test" },
    .{ .name = "PF_GU_SEG", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "PF_FP4_GEMM", .engage = .canonical, .class_text = "EXACT", .what = "128 x 128 paired gate/up where the planner picks it", .evidence = "test (word for word, to 8,192 rows, NVFP4 with per-expert scales; NAX)" },
    .{ .name = "PF_KV_ROWS", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Window kv rows on one simdgroup per row", .evidence = "test (word for word)" },
    .{ .name = "PF_COMP_PASS", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "PF_COMP_BF16", .engage = .canonical, .class_text = "EXACT", .what = "Compressor pooling once over the pass, each span's index-key linear on its own groups", .evidence = "test (word for word)" },
    .{ .name = "PF_CED_COMP", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "PF_CED_HC", .engage = .canonical, .class_text = "EXACT", .what = "The CED compressor at the split layer: each span's projection as before, then its norm, index keys and rope over the pass in one compressPass", .evidence = "test (word for word)" },
    .{ .name = "PF_DOWN_TOKORD", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "PF_FP4_GEMM", .engage = .canonical, .class_text = "EXACT", .what = "Staged down writes rows in token order", .evidence = "test" },
    .{ .name = "PF_DOWN_BK32", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "PF_FP4_GEMM", .engage = .canonical, .class_text = "EXACT", .what = "Staged down on the 64 x 64 tile, 32 deep a step, at 96 routed rows an expert and more", .evidence = "test (word for word)" },
    .{ .name = "PF_FP4_GEMM", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Staged matrix-unit gather for routed gate/up and down", .evidence = "test" },
    .{ .name = "WOA_JOIN", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Joins the spans' wo_a rows in one copy", .evidence = "commit" },
    .{ .name = "PF_ROUTE_PACK", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Router top-6 on a simdgroup per row; one-threadgroup scan", .evidence = "test" },
    .{ .name = "PF_ROUTER_BF16", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Router logits on a bf16 MMA, only where MLX runs one sum (1,707 rows and more)", .evidence = "test" },
    .{ .name = "PF_KEEP_WIDE", .stack = .memory, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT (memory)", .what = "Prompt buffers from the MLX patch's reserve", .evidence = "test" },
    .{ .name = "PF_ENGRAM_PASS", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Engram wkv once per 16 spans", .evidence = "test" },
    .{ .name = "MOE_TAIL", .stack = .prefill, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .accepted = false, .class_text = "EXACT", .what = "MoE combine and hc expand in one launch", .evidence = "test" },
    .{ .name = "FLASH_TRIM", .stack = .attention, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "FLASH_ATTN", .engage = .canonical, .class_text = "EXACT (to FLASH_ATTN)", .what = "FLASH_ATTN with a raw-score block max and 8-byte stores", .evidence = "test" },
    .{ .name = "FFN_IN_SINK", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "ffn_in mix tail in its own launch", .evidence = "test (word for word)" },
    .{ .name = "SPEC_ENGRAM_AHEAD", .stack = .spec, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "SPEC_SYNC", .engage = .canonical, .class_text = "EXACT", .what = "Queues the next verify's Engram kv behind the draft", .evidence = "test" },
    .{ .name = "FFN_IN_R5_COLS1", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "5-row ffn_in launch 1 with one column per simdgroup", .evidence = "test (word for word)" },
    .{ .name = "DRAFT_POS_ONCE", .stack = .spec, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "SPEC_SYNC", .engage = .canonical, .class_text = "EXACT", .what = "Makes draft ring slots and positions once", .evidence = "commit" },
    .{ .name = "FUSE_WQB_RING", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "The wq_b launch writes the window ring", .evidence = "contract" },
    .{ .name = "ENGRAM_KV_CACHE", .stack = .memory, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "ENGRAM_KV_SLOTS", .engage = .canonical, .class_text = "EXACT (to ENGRAM_KV_SLOTS)", .what = "GPU cache of Engram slot partials, at most 4 GB", .evidence = "test" },
    .{ .name = "FP4_DEC_WORD", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Decodes e2m1 codes a word at a time", .evidence = "test" },
    .{ .name = "FP4_UP_SPLIT", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Shared gate and up rows on separate simdgroups", .evidence = "test" },
    .{ .name = "FP8_FOLD", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .{ .line = "deepseek_v41: decode mxfp8 reads folded" }, .class_text = "EXACT (declines on rare codes)", .what = "Folds 2^8 into the e8m0 scale in mxfp8 reads", .evidence = "test (word for word on finite codes), load scan" },
    .{ .name = "MOE_DOWN_ORDER", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "EXACT", .what = "Verify down launch in column-block order", .evidence = "test (bit for bit, R = 2 to 6)" },
    .{ .name = "FUSE_DOWN_SCALE", .stack = .decode, .class = .exact, .kind = .on, .off = .{ .value = "0" }, .parent = "WEIGHT_PATH", .engage = .{ .line = "deepseek_v41: fused down scale" }, .class_text = "EXACT", .what = "NVFP4 down scale inside the gate/up launch", .evidence = "test (word for word)" },
    .{ .name = "CED", .stack = .prefill, .class = .reorder, .kind = .on, .off = null, .engage = .{ .none = "the chunked prompt path itself; -prefill keeps it" }, .class_text = "REORDER (row count)", .what = "The chunked prompt path: the decoder layers run again over a chunk's last window rows only", .evidence = "commit (its fixture test is bit for bit only at 16-row chunks)" },
    .{ .name = "MOE_FP4_NAX", .stack = .decode, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .{ .line = "deepseek_v41: fused fp4 MoE decode kernels on" }, .class_text = "REORDER (bounded)", .what = "Fused fp4 MoE decode kernels on the matrix units", .evidence = "test (within 1.25x of the `=0` path's rel-rms against f32, + 1e-4)" },
    .{ .name = "MOE_PAIR_FP4", .stack = .prefill, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER (bounded)", .what = "fp4 prompt gate/up (NVFP4 and MXFP4), the scales (NVFP4) and activation in one launch", .evidence = "test" },
    .{ .name = "PF_HC_V2", .stack = .prefill, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER (bounded)", .what = "Wide-layer hc in three passes, not five", .evidence = "test" },
    .{ .name = "PF_PASS_SIZE", .stack = .memory, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER (partition)", .what = "Near-equal memory-sized prompt passes", .evidence = "commit" },
    .{ .name = "PF_PASS_CEIL", .stack = .memory, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .parent = "PF_PASS_SIZE", .engage = .{ .none = "the pass plan's load line names the width it picks" }, .class_text = "REORDER (partition)", .what = "Prompt passes no wider than the machine's measured knee (39,680 rows on the M5 Ultra 512 GiB; other machines keep the memory-sized width)", .evidence = "test (the plan) and A/B text (the view at 1M +3.27%, 512K +0.84%; NVFP4 already plans under it)" },
    .{ .name = "PF_IDX_NAX", .stack = .prefill, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER (bounded)", .what = "Indexer scores on the matrix units", .evidence = "test (within 2·γ(34) of `idx_tile`, the same top-half picks except near-ties; NAX)" },
    .{ .name = "PF_COMP_BF16", .stack = .prefill, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER", .what = "Compressor projection on a bf16 MMA, K blocks summed apart", .evidence = "test" },
    .{ .name = "PF_IDX_PASS", .stack = .prefill, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER", .what = "Indexer queries and head weights once over the pass, the head weights on a shape-cut kernel (16 partial sums)", .evidence = "test (head weights no farther from float64 than MLX's GEMM per span; +0.8 GB peak, billed)" },
    .{ .name = "FLASH_ATTN", .stack = .attention, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER", .what = "Every prompt-attention call on one flash kernel where it fits", .evidence = "test" },
    .{ .name = "ENGRAM_GPU_IDS", .stack = .decode, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER (bounded)", .what = "Engram ids hashed on the GPU; fused wkv", .evidence = "test" },
    .{ .name = "ENGRAM_KV_SLOTS", .stack = .decode, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .accepted = false, .class_text = "REORDER", .what = "Slot-blocked Engram wkv", .evidence = "test" },
    .{ .name = "FP4_READS", .stack = .decode, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER (bounded; declines on rare codes)", .what = "Bit-placement fp4 reads; each 16-group summed apart", .evidence = "test (within 1.25x of the `=0` path's rel-rms against f32, + 1e-4)" },
    .{ .name = "ATTN_DEC", .stack = .decode, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER", .what = "Decode attention on a shape-cut tile kernel", .evidence = "test" },
    .{ .name = "WOA_PLAIN", .stack = .decode, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "REORDER", .what = "Splits merged once; wo_a as a plain grouped GEMV", .evidence = "test" },
    .{ .name = "FUSE_MXFP8_QMM", .stack = .prefill, .class = .reorder, .kind = .on, .off = .{ .value = "0" }, .parent = "WEIGHT_PATH", .engage = .{ .line = "deepseek_v41: fused mxfp8 prompt linears" }, .class_text = "REORDER (bounded)", .what = "Prompt mxfp8 linears on MLX's quantized matmul", .evidence = "test (within 1.25x of the `=0` path's rel-rms against f32, + 1e-4)" },
    .{ .name = "DRAFT_FUSED", .stack = .spec, .class = .draft_only, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "DRAFT-ONLY", .what = "DSpark stages on the fused decode kernels", .evidence = "test (within 1% relative rms of the op chain)" },
    .{ .name = "DRAFT_SAMPLE", .stack = .spec, .class = .draft_only, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "DRAFT-ONLY", .what = "Sampled drafts from q, kept at min(1, p / q)", .evidence = "test (engine: `spec/dspark.zig`, \"dspark sampled verdict\")" },
    .{ .name = "DRAFT_QKV", .stack = .spec, .class = .draft_only, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "DRAFT-ONLY", .what = "Fuses a stage's wq_a and wkv at load", .evidence = "test (within 1e-3 relative rms, no farther from the op chain)" },
    .{ .name = "SPEC_SYNC", .stack = .spec, .class = .draft_only, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "DRAFT-ONLY", .what = "Greedy accept on the GPU; next draft queued behind the verify", .evidence = "A/B text (its test needs a pack)" },
    .{ .name = "DRAFT_VOCAB", .stack = .spec, .class = .draft_only, .kind = .on, .off = .{ .value = "0" }, .engage = .canonical, .accepted = false, .class_text = "DRAFT-ONLY", .what = "Greedy drafts score only the pack's `draft-vocab.txt` ids (default: the pack's file, shipped as the plugin's `draft-vocab.txt`)", .evidence = "test (and `deepseek_v4/served_test.zig`)" },
    .{ .name = "DRAFT_VOCAB_PROMPT", .stack = .spec, .class = .draft_only, .kind = .on, .off = .{ .value = "0" }, .parent = "DRAFT_VOCAB", .engage = .canonical, .accepted = false, .class_text = "DRAFT-ONLY", .what = "A request's prompt ids join the draft vocabulary", .evidence = "test (`deepseek_v4/served_test.zig`)" },
    .{ .name = "HEAD8", .stack = .decode, .class = .requant, .kind = .on, .off = .{ .value = "0" }, .engage = .{ .none = "a load-time requantization; the load logs the head" }, .class_text = "Requantized (KLD within one standard error)", .what = "LM head requantized to 8-bit g64", .evidence = "KLD measurement" },
    .{ .name = "ATTN_F16", .stack = .attention, .class = .lossy, .kind = .off, .lossless = .{ .value = "0" }, .lossy = .site, .off = .{ .value = "0" }, .engage = .{ .none = "the core_calls meter counts its calls" }, .class_text = "LOSSY (measured, not gated)", .what = "f16 attention core: under precision=lossy on without NAX, off on NAX; off under lossless", .evidence = "printed float64 measurement, served-route tests" },
    .{ .name = "ENGRAM_LRU_MB", .stack = .loader, .class = .exact, .kind = .setting, .off = null, .engage = .{ .none = "a memory size" }, .class_text = "EXACT", .what = "The Engram LRU's size in MiB", .evidence = "bill test" },
    .{ .name = "ENGRAM_RESIDENT", .stack = .loader, .class = .reorder, .kind = .setting, .off = null, .engage = .{ .none = "a load-time residency choice; the load logs it" }, .class_text = "REORDER", .what = "The Engram tables held in memory at load (auto, wired, 0)", .evidence = "load log" },
    .{ .name = "PF_INT8", .stack = .prefill, .class = .lossy, .kind = .off, .lossless = .{ .value = "0" }, .lossy = .{ .value = "4" }, .off = .{ .value = "0" }, .engage = .canonical, .class_text = "LOSSY", .what = "A resident wide pass's trunk projections on w8a8's rotated int8 GEMM (3: wq_b, wo_b; 4: also the shared expert and wq_a+wkv)", .evidence = "float64 tests, the real-pack quality test, the 16K continuation KLD tool" },
    .{ .name = "MXFP4_VIEW", .stack = .loader, .class = .reorder, .kind = .off, .off = null, .engage = .{ .line = "deepseek_v41: NVFP4 routed experts served as MXFP4" }, .accepted = false, .class_text = "REORDER", .what = "An NVFP4 bank whose every 32-value span is one power of two served as MXFP4 (the experts_mxfp4 view)", .evidence = "test (value-identical banks only)" },
    .{ .name = "EXACT", .stack = .spec, .class = .reorder, .kind = .off, .off = null, .engage = .{ .none = "an opt-in mode; the load logs it" }, .class_text = "EXACT to plain decode", .what = "Greedy DSpark text equal to plain decode's bytes, one request at a time (1 or solo)", .evidence = "test (verify rows against the plain step)" },
    .{ .name = "FLASH_PSUM", .stack = .attention, .class = .reorder, .kind = .off, .off = null, .parent = "FLASH_ATTN", .engage = .canonical, .accepted = false, .class_text = "REORDER", .what = "FLASH_ATTN's row sum over the f16 P the accumulator takes; nearer float64, measured no faster", .evidence = "test (no farther from float64 than FLASH_ATTN)" },
    .{ .name = "WEIGHT_PATH", .stack = .loader, .class = .reorder, .kind = .setting, .off = null, .engage = .{ .line = "deepseek_v41: weight path" }, .accepted_line = "deepseek_v41: weight path resident (", .class_text = "REORDER", .what = "The kernel family: resident (the in-memory fusions), ssd, or auto", .evidence = "load log" },
};

/// Engine variables on the DeepSeek path: listed and reported here, read and owned by the engine or MLX.
pub const EngineRow = struct { name: []const u8, class: Class, line: ?[]const u8 = null, note: []const u8 };
pub const engine_rows = [_]EngineRow{
    .{ .name = "MLX_SERVE_DSV4_DSPARK", .class = .reorder, .note = "the verify's multi-row kernels, by reasoning" },
    .{ .name = "MLX_SERVE_DSPARK_POLICY2", .class = .draft_only, .line = "dspark policy2: next_decided", .note = "picks verify widths" },
    .{ .name = "MLX_SERVE_DSPARK_POLICY2_MODELED", .class = .draft_only, .line = "dspark policy2 modeled:", .note = "picks verify widths" },
    .{ .name = "MLX_ENABLE_TF32", .class = .tf32, .note = "MLX's TF32 matmuls; on M5 at the defaults only the drafter's P·V" },
};

pub const stacks = [_]Stack{ .prefill, .attention, .decode, .spec, .memory };

/// `MLX_SERVE_DSV41_STACK_<STACK>`, the stack's switch.
pub fn stackVar(comptime s: Stack) [:0]const u8 {
    return comptime blk: {
        const tag = @tagName(s);
        var up: [tag.len]u8 = undefined;
        for (tag, 0..) |ch, i| up[i] = std.ascii.toUpper(ch);
        const f = up;
        break :blk std.fmt.comptimePrint("{s}STACK_{s}", .{ prefix, &f });
    };
}

fn index(comptime n: []const u8) usize {
    @setEvalBranchQuota(200_000);
    for (rows, 0..) |r, i| if (std.mem.eql(u8, r.name, n)) return i;
    @compileError("levers: " ++ n ++ " is not a row of levers.zig");
}

pub fn row(comptime name: @EnumLiteral()) Row {
    return rows[comptime index(@tagName(name))];
}

// ── precision and the stack switches: read once per load ──

pub const Precision = enum { lossless, lossy };

const READY: u8 = 1;
const LOSSY: u8 = 2;
const BAD: u8 = 4;
fn stackBit(comptime s: Stack) u8 {
    return switch (s) {
        .prefill => 8,
        .attention => 16,
        .decode => 32,
        .spec => 64,
        .memory => 128,
        .loader => 0,
    };
}

/// Listed in `src/shared_all.zig`, so the host unit exports it under `-Dunits` (the server build).
pub const shared_state = shared.Shared(std.atomic.Value(u8), "dsv41_levers", .init(0));

fn read() u8 {
    var b: u8 = READY;
    if (env.span(.MLX_SERVE_DSV41_PRECISION)) |v| {
        const t = std.mem.trim(u8, v, " ");
        if (std.mem.eql(u8, t, "lossy")) b |= LOSSY else if (!(t.len == 0 or std.mem.eql(u8, t, "lossless"))) b |= BAD;
    }
    if (!env.onUnlessZero(.MLX_SERVE_DSV41_STACK_PREFILL)) b |= stackBit(.prefill);
    if (!env.onUnlessZero(.MLX_SERVE_DSV41_STACK_ATTENTION)) b |= stackBit(.attention);
    if (!env.onUnlessZero(.MLX_SERVE_DSV41_STACK_DECODE)) b |= stackBit(.decode);
    if (!env.onUnlessZero(.MLX_SERVE_DSV41_STACK_SPEC)) b |= stackBit(.spec);
    if (!env.onUnlessZero(.MLX_SERVE_DSV41_STACK_MEMORY)) b |= stackBit(.memory);
    return b;
}

/// Reads precision and the stack switches again (a model's load calls it, through `check`; a test that sets them,
/// too). Before the first load, the first read takes them: the bill and admission read the same values as the load.
pub fn refresh() void {
    shared_state.get().store(read(), .monotonic);
}

fn bits() u8 {
    const b = shared_state.get().load(.monotonic);
    if (b & READY != 0) return b;
    const r = read();
    shared_state.get().store(r, .monotonic);
    return r;
}

pub fn precision() Precision {
    return if (bits() & LOSSY != 0) .lossy else .lossless;
}

pub fn stackOn(comptime s: Stack) bool {
    return bits() & stackBit(s) == 0;
}

pub const Error = error{Dsv41Precision};

/// A load's check: an unknown precision is refused by name (`--precision` refuses it at startup too).
pub fn check() Error!void {
    refresh();
    if (bits() & BAD != 0) {
        log.err("deepseek_v41: MLX_SERVE_DSV41_PRECISION is '{s}'; it takes lossless or lossy\n", .{env.span(.MLX_SERVE_DSV41_PRECISION) orelse ""});
        return error.Dsv41Precision;
    }
}

fn raw(comptime p: Pick) ?[*:0]const u8 {
    return switch (p) {
        .site => null,
        .value => |v| v.ptr,
    };
}

/// The value a row resolves to with its own variable unset.
fn derived(comptime r: Row) ?[*:0]const u8 {
    const b = bits();
    if (comptime r.off != null) if (b & stackBit(r.stack) != 0) return comptime raw(r.off.?);
    if (comptime r.lossy != null) if (b & LOSSY != 0) return comptime raw(r.lossy.?);
    return comptime raw(r.lossless);
}

// ── the readers: the env readers' semantics over the resolved value ──

/// The lever's value: its own variable when set, else what its stack switch, the precision and the table give.
pub fn get(comptime name: @EnumLiteral()) ?[*:0]const u8 {
    return getOf(comptime row(name));
}

fn getOf(comptime r: Row) ?[*:0]const u8 {
    if (env.named(prefix ++ r.name)) |v| return v;
    return derived(r);
}

/// The lever's own variable, when set (the startup line's `changed`).
fn ownOf(comptime r: Row) ?[*:0]const u8 {
    return env.named(prefix ++ r.name);
}

pub fn span(comptime name: @EnumLiteral()) ?[:0]const u8 {
    return if (get(name)) |v| std.mem.span(v) else null;
}

/// A default-on lever: on unless its value starts with `0`.
pub fn onUnlessZero(comptime name: @EnumLiteral()) bool {
    const v = get(name) orelse return true;
    return v[0] != '0';
}

/// An opt-in flag: set and not starting with `0`.
pub fn on(comptime name: @EnumLiteral()) bool {
    return env.onValue(get(name));
}

pub fn int(comptime name: @EnumLiteral(), comptime T: type, default: T) T {
    return env.intValue(T, get(name), default);
}

/// The line `name` logs when it serves (the caller adds its details and the newline).
pub fn line(comptime name: @EnumLiteral()) []const u8 {
    const r = comptime row(name);
    return comptime lineOf(r);
}

fn lineOf(comptime r: Row) []const u8 {
    return switch (r.engage) {
        .canonical => "[dsv41] " ++ prefix ++ r.name ++ " engaged",
        .line => |l| l,
        .none => @compileError("levers: " ++ r.name ++ " logs no engage line"),
    };
}

fn eqlZ(a: ?[*:0]const u8, b: ?[*:0]const u8) bool {
    if (a == null or b == null) return a == null and b == null;
    return std.mem.eql(u8, std.mem.span(a.?), std.mem.span(b.?));
}

/// A hash of the LOSSY rows' resolved values, for the prompt cache's layout; null while every one resolves to its
/// lossless pick, so a lossless load keeps the layout hash it had before this table.
pub fn lossyKey() ?u64 {
    var h = std.hash.Wyhash.init(0x1e7e);
    var any = false;
    inline for (rows) |r| {
        if (r.class == .lossy) {
            const v = getOf(r);
            if (!eqlZ(v, comptime raw(r.lossless))) any = true;
            h.update(r.name);
            h.update("=");
            if (v) |s| h.update(std.mem.span(s));
            h.update("\x00");
        }
    }
    return if (any) h.final() else null;
}

// ── the startup line ──

/// One line a load prints: the precision, the stacks switched off, every lever variable set, and the acceptance.
pub fn settingsLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("[dsv41] settings: precision={t}", .{precision()}) catch {};
    w.writeAll(" stacks_off=") catch {};
    var n: usize = 0;
    inline for (stacks) |s| if (!stackOn(s)) {
        w.print("{s}{t}", .{ if (n > 0) "," else "", s }) catch {};
        n += 1;
    };
    if (n == 0) w.writeAll("-") catch {};
    w.writeAll(" changed=") catch {};
    n = 0;
    inline for (rows) |r| if (ownOf(r)) |v| {
        w.print("{s}{s}={s}", .{ if (n > 0) "," else "", r.name, std.mem.span(v) }) catch {};
        n += 1;
    };
    if (n == 0) w.writeAll("-") catch {};
    if (precision() == .lossy) {
        w.print(" accept=typical:delta={d},eps={d} (sampled; unless the engine's acceptance is set) NOT lossless", .{ lossy_typical.delta, lossy_typical.eps }) catch {};
    } else w.writeAll(" accept=exact") catch {};
    return w.buffered();
}

pub fn logSettings() void {
    var buf: [4096]u8 = undefined;
    log.info("{s}\n", .{settingsLine(&buf)});
}

// ── the rendered docs and the harness check list ──

fn className(c: Class) []const u8 {
    return switch (c) {
        .exact => "EXACT",
        .reorder => "REORDER",
        .tf32 => "TF32",
        .draft_only => "DRAFT-ONLY",
        .requant => "REQUANT",
        .lossy => "LOSSY",
    };
}

fn count(comptime k: Kind, comptime c: Class) usize {
    var n: usize = 0;
    for (rows) |r| {
        if (r.kind == k and r.class == c) n += 1;
    }
    return n;
}

fn defaultOn() usize {
    var n: usize = 0;
    for (rows) |r| {
        if (r.kind == .on) n += 1;
    }
    return n;
}

/// The README's counts paragraph.
pub fn renderCounts(w: *std.Io.Writer) !void {
    try w.print("Under `precision=lossless` (the default) the plugin has {d} default-on switches: {d} EXACT, {d} REORDER, " ++
        "{d} DRAFT-ONLY and `HEAD8`, the 8-bit LM head, the one accepted requantization (REQUANT). `precision=lossy` " ++
        "adds the LOSSY picks below. `src/plugins/deepseek_v41/levers.zig` holds every switch; this block and the " ++
        "tables marked `levers:` are rendered from it.\n", .{ defaultOn(), count(.on, .exact), count(.on, .reorder), count(.on, .draft_only) });
}

/// The README's class table: default-on plugin switches and the engine variables on the DeepSeek path.
pub fn renderClasses(w: *std.Io.Writer) !void {
    try w.writeAll("| Class | Plugin switches (default-on) | Engine variables |\n|---|---|---|\n");
    inline for (.{ Class.exact, Class.reorder, Class.draft_only, Class.tf32, Class.requant }) |c| {
        try w.print("| {s} | {d} |", .{ className(c), count(.on, c) });
        var n: usize = 0;
        for (engine_rows) |e| if (e.class == c) {
            try w.print("{s} `{s}` ({s})", .{ if (n > 0) ";" else "", e.name, e.note });
            n += 1;
        };
        if (n == 0) try w.writeAll(" 0");
        try w.writeAll(" |\n");
    }
}

/// The README's table of every default-on switch.
pub fn renderSwitches(w: *std.Io.Writer) !void {
    try w.writeAll("| Switch | What it does | Class | Evidence |\n|---|---|---|---|\n");
    for (rows) |r| if (r.kind == .on) try w.print("| {s} | {s} | {s} | {s} |\n", .{ r.name, r.what, r.class_text, r.evidence });
}

/// The README's table of what `precision=lossy` changes.
pub fn renderLossy(w: *std.Io.Writer) !void {
    try w.writeAll("| Switch | Under lossless | Under lossy | What it does |\n|---|---|---|---|\n");
    inline for (rows) |r| if (comptime r.lossy != null) {
        try w.print("| {s} | {s} | {s} | {s} |\n", .{ r.name, comptime pickText(r.lossless), comptime pickText(r.lossy.?), r.what });
    };
    try w.print("| typical acceptance (the engine's DSpark rule) | exact | typical at delta {d}, eps {d} (`levers.lossy_typical`), unless `--mtp-typical`, `--mtp-tokenv3` or a model `mtp_acceptance` sets it | a sampled request's drafts kept iff p > min(eps, delta e^-H) |\n", .{ lossy_typical.delta, lossy_typical.eps });
}

fn pickText(comptime p: Pick) []const u8 {
    return switch (p) {
        .site => "the switch's own default",
        .value => |v| "`" ++ v ++ "`",
    };
}

/// The README's stack table: each stack switch and the switches it turns off.
pub fn renderStacks(w: *std.Io.Writer) !void {
    try w.writeAll("| Stack switch (`=0` turns it off) | Switches it sets to their `=0` path |\n|---|---|\n");
    inline for (stacks) |s| {
        try w.print("| `{s}` |", .{comptime stackVar(s)});
        var n: usize = 0;
        for (rows) |r| if (r.stack == s and r.off != null) {
            try w.print("{s} {s}", .{ if (n > 0) "," else "", r.name });
            n += 1;
        };
        try w.writeAll(" |\n");
    }
}

/// `accepted-lines.txt`: the line each accepted switch logs on the standard test's route, then the engine's.
pub fn renderAccepted(w: *std.Io.Writer) !void {
    inline for (rows) |r| {
        if (comptime r.accepted and r.engage != .none and (r.kind == .on or r.accepted_line != null))
            try w.print("{s}\n", .{comptime r.accepted_line orelse lineOf(r)});
    }
    for (engine_rows) |e| if (e.line) |l| try w.print("{s}\n", .{l});
}

// ── checks over a table ──

/// Lossless purity: no LOSSY row is a default-on switch, every LOSSY row resolves to `0` under lossless, and the
/// table has one REQUANT row (HEAD8, which lossless keeps).
pub fn pure(comptime t: []const Row) bool {
    @setEvalBranchQuota(200_000);
    var requant: usize = 0;
    for (t) |r| {
        if (r.class == .requant) requant += 1;
        if (r.class == .lossy) {
            if (r.kind == .on) return false;
            switch (r.lossless) {
                .site => return false,
                .value => |v| if (!std.mem.eql(u8, v, "0")) return false,
            }
        }
    }
    return requant == 1;
}

/// Every default-on switch outside `loader` has an off arm for its stack switch (CED, the chunked prompt path
/// itself, is the stated exception), and every parent names a row.
pub fn sound(comptime t: []const Row) bool {
    @setEvalBranchQuota(200_000);
    for (t) |r| {
        if (r.kind == .on and r.off == null and r.stack != .loader and !std.mem.eql(u8, r.name, "CED")) return false;
        if (r.parent) |p| {
            var found = false;
            for (t) |q| {
                if (std.mem.eql(u8, q.name, p)) found = true;
            }
            if (!found) return false;
        }
    }
    return true;
}

comptime {
    if (!pure(&rows)) @compileError("levers: the table breaks lossless purity");
    if (!sound(&rows)) @compileError("levers: a default-on row has no off arm, or a parent names no row");
}

// ── tests ──

const testing = std.testing;

fn setVar(comptime n: [:0]const u8, v: ?[*:0]const u8) void {
    if (v) |s| _ = env.setForeign(n.ptr, s) else env.unsetForeign(n.ptr);
}

const control_vars = [_][:0]const u8{
    "MLX_SERVE_DSV41_PRECISION",      "MLX_SERVE_DSV41_STACK_PREFILL", "MLX_SERVE_DSV41_STACK_ATTENTION",
    "MLX_SERVE_DSV41_STACK_DECODE", "MLX_SERVE_DSV41_STACK_SPEC",    "MLX_SERVE_DSV41_STACK_MEMORY",
};

fn clearControls() void {
    inline for (control_vars) |n| env.unsetForeign(n.ptr);
    refresh();
}

test "deepseek_v41 levers: every row is a registered switch, and every DSV41 switch in env.zig is a row" {
    @setEvalBranchQuota(400_000);
    inline for (rows) |r| try testing.expect(env.registered(prefix ++ r.name));
    inline for (engine_rows) |e| try testing.expect(env.registered(e.name));
    inline for (comptime std.meta.declarations(env.vars)) |n| {
        if (comptime std.mem.startsWith(u8, n, prefix) and @field(env.vars, n).kind == .kill) {
            const own = comptime n[prefix.len..];
            const control = comptime std.mem.eql(u8, own, "PRECISION") or std.mem.startsWith(u8, own, "STACK_");
            if (!control) {
                var found = false;
                for (rows) |r| {
                    if (std.mem.eql(u8, r.name, own)) found = true;
                }
                if (!found) std.debug.print("levers: {s} has no row\n", .{n});
                try testing.expect(found);
            }
        }
    }
}

test "deepseek_v41 levers: with no precision or stack set, a lever reads as its own variable alone (ATTN_F16 and PF_INT8 resolve to 0)" {
    @setEvalBranchQuota(200_000);
    clearControls();
    defer clearControls();
    inline for (rows) |r| {
        const n = prefix ++ r.name;
        const had = env.byName(n);
        var keep: [256]u8 = undefined;
        var keep_len: usize = 0;
        if (had) |h| {
            const s = std.mem.span(h);
            keep_len = @min(s.len, keep.len - 1);
            @memcpy(keep[0..keep_len], s[0..keep_len]);
        }
        keep[keep_len] = 0;
        defer setVar(n, if (had != null) @as([*:0]const u8, @ptrCast(&keep)) else null);
        inline for (.{ "", "0", "1", "solo", "3" }) |v| {
            setVar(n, v);
            try testing.expectEqualStrings(v, std.mem.span(getOf(r).?));
        }
        setVar(n, null);
        const want: ?[]const u8 = if (comptime std.mem.eql(u8, r.name, "ATTN_F16") or std.mem.eql(u8, r.name, "PF_INT8")) "0" else null;
        const got = getOf(r);
        if (want) |w| try testing.expectEqualStrings(w, std.mem.span(got.?)) else try testing.expect(got == null);
    }
}

test "deepseek_v41 levers: lossless purity holds, and a planted LOSSY default fails the same check" {
    try testing.expect(comptime pure(&rows));
    const planted = comptime blk: {
        @setEvalBranchQuota(200_000);
        var t = rows;
        for (&t) |*r| {
            if (std.mem.eql(u8, r.name, "PF_INT8")) r.lossless = .{ .value = "4" };
        }
        break :blk t;
    };
    try testing.expect(!comptime pure(&planted));
    const planted_on = comptime blk: {
        @setEvalBranchQuota(200_000);
        var t = rows;
        for (&t) |*r| {
            if (std.mem.eql(u8, r.name, "ATTN_F16")) r.kind = .on;
        }
        break :blk t;
    };
    try testing.expect(!comptime pure(&planted_on));
    clearControls();
    defer clearControls();
    inline for (rows) |r| if (r.class == .lossy and ownOf(r) == null) {
        try testing.expectEqualStrings("0", std.mem.span(getOf(r).?));
    };
}

test "deepseek_v41 levers: each stack switch turns off its stack's default-on switches, and a lever's own variable wins" {
    clearControls();
    defer clearControls();
    inline for (stacks) |s| {
        const n = comptime stackVar(s);
        _ = env.setForeign(n.ptr, "0");
        refresh();
        try testing.expect(!stackOn(s));
        inline for (rows) |r| if (ownOf(r) == null) {
            if (comptime r.stack == s and r.off != null) {
                try testing.expectEqualStrings("0", std.mem.span(getOf(r).?));
            } else if (comptime r.stack != s) {
                try testing.expect(eqlZ(getOf(r), comptime raw(r.lossless)));
            }
        };
        env.unsetForeign(n.ptr);
        refresh();
    }
    _ = env.setForeign("MLX_SERVE_DSV41_STACK_PREFILL", "0");
    refresh();
    try testing.expect(!onUnlessZero(.PF_GU_SEG));
    try testing.expect(get(.CED) == null);
    _ = env.setForeign("MLX_SERVE_DSV41_PF_GU_SEG", "1");
    defer env.unsetForeign("MLX_SERVE_DSV41_PF_GU_SEG");
    try testing.expect(onUnlessZero(.PF_GU_SEG));
    try testing.expect(onUnlessZero(.FLASH_ATTN));
}

test "deepseek_v41 levers: precision lossy picks PF_INT8=4 and the site's ATTN_F16, a stack off wins over it, and an unknown precision refuses" {
    clearControls();
    defer clearControls();
    try testing.expectEqual(@as(u32, 0), int(.PF_INT8, u32, 0));
    _ = env.setForeign("MLX_SERVE_DSV41_PRECISION", "lossy");
    refresh();
    try testing.expectEqual(Precision.lossy, precision());
    try check();
    try testing.expectEqual(@as(u32, 4), int(.PF_INT8, u32, 0));
    try testing.expect(get(.ATTN_F16) == null);
    try testing.expectEqual(@as(f32, 0.3), lossyTypical().?.delta);
    try testing.expectEqual(@as(f32, 0.09), lossyTypical().?.eps);
    try testing.expect(lossyKey() != null);
    _ = env.setForeign("MLX_SERVE_DSV41_STACK_PREFILL", "0");
    refresh();
    try testing.expectEqual(@as(u32, 0), int(.PF_INT8, u32, 0));
    _ = env.setForeign("MLX_SERVE_DSV41_PRECISION", "lossier");
    try testing.expectError(error.Dsv41Precision, check());
    clearControls();
    try testing.expect(lossyKey() == null);
    try testing.expect(lossyTypical() == null);
    try check();
}

test "deepseek_v41 levers: the startup line names the precision, the stacks off and each lever set" {
    clearControls();
    defer clearControls();
    var buf: [4096]u8 = undefined;
    _ = env.setForeign("MLX_SERVE_DSV41_STACK_SPEC", "0");
    _ = env.setForeign("MLX_SERVE_DSV41_PF_GU_SEG", "0");
    defer env.unsetForeign("MLX_SERVE_DSV41_PF_GU_SEG");
    refresh();
    try testing.expectEqualStrings("[dsv41] settings: precision=lossless stacks_off=spec changed=PF_GU_SEG=0 accept=exact", settingsLine(&buf));
}

fn block(doc: []const u8, comptime tag: []const u8) ?[]const u8 {
    const open = "<!-- levers:" ++ tag ++ " -->\n";
    const close = "<!-- /levers:" ++ tag ++ " -->";
    const a = std.mem.indexOf(u8, doc, open) orelse return null;
    const b = std.mem.indexOfPos(u8, doc, a, close) orelse return null;
    return doc[a + open.len .. b];
}

fn expectRendered(doc: []const u8, comptime tag: []const u8, comptime render: anytype) !void {
    var buf: [64 * 1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try render(&w);
    const got = block(doc, tag) orelse {
        std.debug.print("README.md has no <!-- levers:{s} --> block; it should hold:\n{s}", .{ tag, w.buffered() });
        return error.TestExpectedEqual;
    };
    if (!std.mem.eql(u8, got, w.buffered())) std.debug.print("README.md's levers:{s} block should be:\n{s}", .{ tag, w.buffered() });
    try testing.expectEqualStrings(w.buffered(), got);
}

test "deepseek_v41 levers: the README's switch tables and counts are the table's" {
    const doc = @embedFile("README.md");
    try expectRendered(doc, "counts", renderCounts);
    try expectRendered(doc, "classes", renderClasses);
    try expectRendered(doc, "switches", renderSwitches);
    try expectRendered(doc, "lossy", renderLossy);
    try expectRendered(doc, "stacks", renderStacks);
    // A planted row the README lacks fails the comparison.
    var buf: [64 * 1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try renderSwitches(&w);
    try w.writeAll("| PLANTED | a row the README lacks | EXACT | none |\n");
    try testing.expect(!std.mem.eql(u8, block(doc, "switches").?, w.buffered()));
}

test "deepseek_v41 levers: accepted-lines.txt is the table's harness check list" {
    var buf: [16 * 1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try renderAccepted(&w);
    const file = @embedFile("accepted-lines.txt");
    if (!std.mem.eql(u8, file, w.buffered())) std.debug.print("accepted-lines.txt should be:\n{s}", .{w.buffered()});
    try testing.expectEqualStrings(w.buffered(), file);
}
