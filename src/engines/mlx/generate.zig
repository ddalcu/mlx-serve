//! Generation entry point: re-exports the generate/, sampling/, spec/, grammar/, decode/ and embed/ files,
//! and owns the launch-flag globals main.zig and server.zig write (a var cannot be aliased).
const std = @import("std");
const mtp_acceptance = @import("../../sdk/spec/mtp_acceptance.zig");
pub const explicitPrefillChunkU32 = @import("generate/prefill_chunk.zig").explicitPrefillChunkU32;
pub const mtpGreedyTailFor = @import("generate/prefill_chunk.zig").mtpGreedyTailFor;
pub const commitAdaptiveWidth = @import("generate/prefill_chunk.zig").commitAdaptiveWidth;
pub const AdaptiveWidthState = @import("generate/prefill_chunk.zig").AdaptiveWidthState;
pub const SpecDisableReason = @import("generate/prefill_chunk.zig").SpecDisableReason;
pub const effectiveMtpHistoryWindow = @import("generate/prefill_chunk.zig").effectiveMtpHistoryWindow;
pub const chunkNeedsMtpHistory = @import("generate/prefill_chunk.zig").chunkNeedsMtpHistory;
pub const PREFILL_SCORES_BUDGET_BYTES = @import("generate/prefill_chunk.zig").PREFILL_SCORES_BUDGET_BYTES;
pub const PREFILL_CHUNK_FLOOR = @import("generate/prefill_chunk.zig").PREFILL_CHUNK_FLOOR;
pub const boundedPrefillChunk = @import("generate/prefill_chunk.zig").boundedPrefillChunk;
pub const effectivePrefillChunk = @import("generate/prefill_chunk.zig").effectivePrefillChunk;
pub const decodeShareCapped = @import("generate/prefill_chunk.zig").decodeShareCapped;
pub const MtpHeadRef = @import("spec/mtp_head_ref.zig").MtpHeadRef;
pub const MtpCacheRef = @import("spec/mtp_head_ref.zig").MtpCacheRef;
pub const MtpRestored = @import("spec/mtp_head_ref.zig").MtpRestored;
pub const Constraint = @import("grammar/constraint.zig").Constraint;
pub const forcedBoundaryCanContinue = @import("grammar/constraint.zig").forcedBoundaryCanContinue;
pub const SchemaConstraint = @import("grammar/constraint.zig").SchemaConstraint;
pub const TokenLogprob = @import("sampling/logprobs.zig").TokenLogprob;
pub const LogprobResult = @import("sampling/logprobs.zig").LogprobResult;
pub const ThinkBound = @import("../../sdk/request/forced.zig").ThinkBound;
pub const ToolForce = @import("../../sdk/request/forced.zig").ToolForce;
pub const SamplingParams = @import("sampling/params.zig").SamplingParams;
pub const buildSuppressMask = @import("sampling/params.zig").buildSuppressMask;
pub const fillSuppressMask = @import("sampling/params.zig").fillSuppressMask;
pub const installSuppressMask = @import("sampling/params.zig").installSuppressMask;
pub const CommittedArgmax = @import("spec/verify.zig").CommittedArgmax;
pub const GenerationResult = @import("../../sdk/request/result.zig").GenerationResult;
pub const tokensPerSec = @import("../../sdk/request/result.zig").tokensPerSec;
pub const prefillTokensPerSec = @import("../../sdk/request/result.zig").prefillTokensPerSec;
pub const TAIL_MERGE_MAX = @import("generate/prefill_chunk.zig").TAIL_MERGE_MAX;
pub const tailMergeMax = @import("generate/prefill_chunk.zig").tailMergeMax;
pub const tailMergeMaxFor = @import("generate/prefill_chunk.zig").tailMergeMaxFor;
pub const nextChunkEnd = @import("generate/prefill_chunk.zig").nextChunkEnd;
pub const SSM_SNAPSHOT_BACKOFF = @import("generate/prefill_chunk.zig").SSM_SNAPSHOT_BACKOFF;
pub const ssmSnapshotBackoff = @import("generate/prefill_chunk.zig").ssmSnapshotBackoff;
pub const SSM_GRID_LOW_DENSE_STRIDE = @import("generate/prefill_chunk.zig").SSM_GRID_LOW_DENSE_STRIDE;
pub const SSM_GRID_LOW_DENSE_TOP = @import("generate/prefill_chunk.zig").SSM_GRID_LOW_DENSE_TOP;
pub const ssmCapturePos = @import("generate/prefill_chunk.zig").ssmCapturePos;
pub const ssmLowAnchorEnd = @import("generate/prefill_chunk.zig").ssmLowAnchorEnd;
pub const effectiveSsmCheckpointStride = @import("generate/prefill_chunk.zig").effectiveSsmCheckpointStride;
pub const reservedPrefillTokens = @import("generate/prefill_chunk.zig").reservedPrefillTokens;
pub const shouldCheckpointSsmPrefill = @import("generate/prefill_chunk.zig").shouldCheckpointSsmPrefill;
pub const countSpliceRows = @import("generate/prefill_chunk.zig").countSpliceRows;
pub const prefillChunkCount = @import("generate/prefill_chunk.zig").prefillChunkCount;
pub const CACHE_CLEAR_INTERVAL = @import("generate/prefill_chunk.zig").CACHE_CLEAR_INTERVAL;
pub const shouldClearAllocatorCache = @import("generate/prefill_chunk.zig").shouldClearAllocatorCache;
pub const clearsPoolAtPrefillEnd = @import("generate/prefill_chunk.zig").clearsPoolAtPrefillEnd;
pub const capAcceptedForTokenBudget = @import("spec/verify.zig").capAcceptedForTokenBudget;
pub const dsparkCopyLen = @import("spec/verify.zig").dsparkCopyLen;
pub const Generator = @import("generate/generator.zig").Generator;
pub const sampleTokenLazy = @import("sampling/lazy.zig").sampleTokenLazy;
pub const sampleRows = @import("sampling/lazy.zig").sampleRows;
pub const generate = @import("generate/oneshot.zig").generate;
pub const generatePld = @import("generate/oneshot.zig").generatePld;
pub const generateDrafter = @import("generate/oneshot.zig").generateDrafter;
pub const generateMtp = @import("generate/oneshot.zig").generateMtp;
pub const isEosId = @import("../../sdk/request/stops.zig").isEosId;
pub const tokenStops = @import("../../sdk/request/stops.zig").tokenStops;
pub const degenerate_loop_max_period = @import("../../sdk/request/loop_guard.zig").degenerate_loop_max_period;
pub const degenerate_loop_reps = @import("../../sdk/request/loop_guard.zig").degenerate_loop_reps;
pub const degenerate_loop_min_span = @import("../../sdk/request/loop_guard.zig").degenerate_loop_min_span;
pub const degenerate_loop_long_max_period = @import("../../sdk/request/loop_guard.zig").degenerate_loop_long_max_period;
pub const degenerate_loop_long_reps = @import("../../sdk/request/loop_guard.zig").degenerate_loop_long_reps;
pub const degenerate_loop_long_min_span = @import("../../sdk/request/loop_guard.zig").degenerate_loop_long_min_span;
pub const near_repeat_window = @import("../../sdk/request/loop_guard.zig").near_repeat_window;
pub const near_repeat_ngram = @import("../../sdk/request/loop_guard.zig").near_repeat_ngram;
pub const near_repeat_max_ngram_ratio = @import("../../sdk/request/loop_guard.zig").near_repeat_max_ngram_ratio;
pub const near_repeat_max_token_ratio = @import("../../sdk/request/loop_guard.zig").near_repeat_max_token_ratio;
pub const near_repeat_max_novelty = @import("../../sdk/request/loop_guard.zig").near_repeat_max_novelty;
pub const isNearRepeatTailLoop = @import("../../sdk/request/loop_guard.zig").isNearRepeatTailLoop;
pub const near_repeat_step = @import("../../sdk/request/loop_guard.zig").near_repeat_step;
pub const near_repeat_min_span = @import("../../sdk/request/loop_guard.zig").near_repeat_min_span;
pub const near_repeat_max_lookback = @import("../../sdk/request/loop_guard.zig").near_repeat_max_lookback;
pub const DegenerateTail = @import("../../sdk/request/loop_guard.zig").DegenerateTail;
pub const degenerateTail = @import("../../sdk/request/loop_guard.zig").degenerateTail;
pub const StallClock = @import("../../sdk/request/stops.zig").StallClock;
pub const isDegenerateTailLoop = @import("../../sdk/request/loop_guard.zig").isDegenerateTailLoop;
pub const isDegenerateTailLoopRange = @import("../../sdk/request/loop_guard.zig").isDegenerateTailLoopRange;
pub const computeEmbedding = @import("embed/pooling.zig").computeEmbedding;
pub const EMBED_MAX_BATCH = @import("embed/pooling.zig").EMBED_MAX_BATCH;
pub const EMBED_TOKEN_BUDGET = @import("embed/pooling.zig").EMBED_TOKEN_BUDGET;
pub const embedSubBatchEnd = @import("embed/pooling.zig").embedSubBatchEnd;
pub const PaddedBatch = @import("embed/pooling.zig").PaddedBatch;
pub const buildPaddedBatch = @import("embed/pooling.zig").buildPaddedBatch;
pub const buildKeyPadMask = @import("embed/pooling.zig").buildKeyPadMask;
pub const maskedMeanPool = @import("embed/pooling.zig").maskedMeanPool;
pub const maskedMeanPoolNormalize = @import("embed/pooling.zig").maskedMeanPoolNormalize;
pub const gatherTokenPool = @import("embed/pooling.zig").gatherTokenPool;
pub const l2NormalizeRows = @import("embed/pooling.zig").l2NormalizeRows;
pub const computeEmbeddingsBatch = @import("embed/pooling.zig").computeEmbeddingsBatch;
pub const computeEmbeddingsBatchWith = @import("embed/pooling.zig").computeEmbeddingsBatchWith;
pub const MAX_TOP_LOGPROBS = @import("sampling/logprobs.zig").MAX_TOP_LOGPROBS;
pub const promptLogprobs = @import("sampling/logprobs.zig").promptLogprobs;


/// The launch's prefill chunk settings, set once from the CLI before `serve()` runs (`configure.apply`); one copy for
/// every compile unit, since a plugin unit sizes chunks too (`Shared`).
pub const PrefillCli = struct {
    /// The chunk width: `--prefill-chunk`, else the default.
    override: usize = 8192,
    /// Set by `--prefill-chunk`. An operator-chosen width outranks the per-model chunk the memory sizer pins
    /// (`ModelConfig.pinned_prefill_chunk`); without this flag the default 8192 is indistinguishable from a request.
    explicit: bool = false,
    /// The widest chunk this machine runs without `--prefill-chunk` (`autoPrefillChunkCap`). Every auto width obeys
    /// it, the per-request and adaptive rungs included.
    machine_cap: usize = std.math.maxInt(usize),
};
pub const prefillCli_shared = @import("../../sdk/util/shared.zig").Shared(PrefillCli, "prefill_cli", .{});
pub inline fn prefillCli() *PrefillCli {
    return prefillCli_shared.get();
}
/// Default chunk of an arch that builds no prefill score tensor (score width 0):
/// only its MoE gather transient grows with the chunk, and the guard bills that.
pub const WIDE_PREFILL_CHUNK: usize = 16384;

pub var prefill_trace_force: bool = false;

/// Set once by the serve CLI before request construction. Direct callers may
/// override it through InitOptions; exact remains the library default.
pub var mtp_acceptance_default: mtp_acceptance.Mode = .exact;

/// `--mtp-greedy-tail`, set once by the serve CLI.
pub var mtp_greedy_tail_default: bool = false;

/// MTP prefill-history window (`--mtp-history-window`; 0 = full history).
/// Same set-once-at-CLI-parse contract as `PrefillCli`.
/// DEFAULT 0 (full): the A/B gate failed for windowing — at 64K ctx on the
/// stock Qwen3.6-27B head, window 8192 cost 14 acceptance points (68.2% ->
/// 54.0%) and 4.2 decode tok/s for ZERO prefill benefit (184.7 vs 185.1
/// tok/s); at 32K it was a wash. Qwen's stock head drafts from deep history.
pub var mtp_history_window_override: usize = 0;

/// `--max-mtp-ctx N`: MTP stays off past N context tokens (0 = unlimited). Read by the HTTP
/// admission gate and by `nextMtp` per round (`mtp.mtpCtxWithinLimit`).
pub var max_mtp_ctx: u32 = 0;

/// The text of generate.zig and every file carved from it, for the source-scan tests that pinned generate.zig.
pub const all_source = @embedFile("generate.zig") ++
    @embedFile("generate/generator.zig") ++
    @embedFile("spec/dflash.zig") ++
    @embedFile("generate/prefill_chunk.zig") ++
    @embedFile("spec/stats_log.zig") ++
    @embedFile("spec/mtp_head_ref.zig") ++
    @embedFile("grammar/constraint.zig") ++
    @embedFile("sampling/logprobs.zig") ++
    @embedFile("../../sdk/request/forced.zig") ++
    @embedFile("sampling/params.zig") ++
    @embedFile("spec/verify.zig") ++
    @embedFile("../../sdk/request/result.zig") ++
    @embedFile("generate/pipeline.zig") ++
    @embedFile("spec/tree.zig") ++
    @embedFile("spec/gates.zig") ++
    @embedFile("generate/options.zig") ++
    @embedFile("spec/pld.zig") ++
    @embedFile("spec/dspark.zig") ++
    @embedFile("generate/init.zig") ++
    @embedFile("spec/drafter.zig") ++
    @embedFile("spec/dflash_tree.zig") ++
    @embedFile("spec/mtp/chain.zig") ++
    @embedFile("spec/mtp/chain_build.zig") ++
    @embedFile("spec/mtp/batched_graph.zig") ++
    @embedFile("spec/mtp/serial.zig") ++
    @embedFile("spec/mtp/round.zig") ++
    @embedFile("spec/mtp/round_finish.zig") ++
    @embedFile("spec/mtp/depth.zig") ++
    @embedFile("spec/mtp/ev_costs.zig") ++
    @embedFile("spec/mtp/observe.zig") ++
    @embedFile("spec/mtp/ev_plan.zig") ++
    @embedFile("spec/mtp/trace.zig") ++
    @embedFile("spec/mtp/regime.zig") ++
    @embedFile("spec/mtp/adaptive.zig") ++
    @embedFile("spec/mtp/planner.zig") ++
    @embedFile("generate/step.zig") ++
    @embedFile("sampling/spec_sampling.zig") ++
    @embedFile("sampling/lazy.zig") ++
    @embedFile("generate/oneshot.zig") ++
    @embedFile("../../sdk/request/stops.zig") ++
    @embedFile("../../sdk/request/loop_guard.zig") ++
    @embedFile("embed/pooling.zig") ++
    @embedFile("sampling/sampler.zig") ++
    @embedFile("sampling/filters.zig") ++
    @embedFile("../../plugins/qwen4_exp/mtp_generate.zig") ++
    @embedFile("../../plugins/embedding_gemma2/generate.zig");

test {
    _ = @import("generate/generator.zig");
    _ = @import("spec/dflash.zig");
    _ = @import("generate/prefill_chunk.zig");
    _ = @import("spec/stats_log.zig");
    _ = @import("spec/mtp_head_ref.zig");
    _ = @import("grammar/constraint.zig");
    _ = @import("sampling/logprobs.zig");
    _ = @import("../../sdk/request/forced.zig");
    _ = @import("sampling/params.zig");
    _ = @import("spec/verify.zig");
    _ = @import("../../sdk/request/result.zig");
    _ = @import("generate/pipeline.zig");
    _ = @import("spec/tree.zig");
    _ = @import("spec/gates.zig");
    _ = @import("generate/options.zig");
    _ = @import("spec/pld.zig");
    _ = @import("spec/dspark.zig");
    _ = @import("generate/init.zig");
    _ = @import("spec/drafter.zig");
    _ = @import("spec/dflash_tree.zig");
    _ = @import("spec/mtp/chain.zig");
    _ = @import("spec/mtp/chain_build.zig");
    _ = @import("spec/mtp/batched_graph.zig");
    _ = @import("spec/mtp/serial.zig");
    _ = @import("spec/mtp/round.zig");
    _ = @import("spec/mtp/round_finish.zig");
    _ = @import("spec/mtp/depth.zig");
    _ = @import("spec/mtp/ev_costs.zig");
    _ = @import("spec/mtp/observe.zig");
    _ = @import("spec/mtp/ev_plan.zig");
    _ = @import("spec/mtp/trace.zig");
    _ = @import("spec/mtp/regime.zig");
    _ = @import("spec/mtp/adaptive.zig");
    _ = @import("spec/mtp/planner.zig");
    _ = @import("generate/step.zig");
    _ = @import("sampling/spec_sampling.zig");
    _ = @import("sampling/lazy.zig");
    _ = @import("generate/oneshot.zig");
    _ = @import("../../sdk/request/stops.zig");
    _ = @import("../../sdk/request/loop_guard.zig");
    _ = @import("embed/pooling.zig");
    _ = @import("sampling/sampler.zig");
    _ = @import("sampling/filters.zig");
    _ = @import("../../plugins/qwen4_exp/mtp_generate.zig");
    _ = @import("../../plugins/embedding_gemma2/generate.zig");
}
