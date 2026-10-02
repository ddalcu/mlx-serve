//! generate surface for builds without MLX.

const std = @import("std");
const real = @import("generate_common.zig");
pub const ThinkBound = real.ThinkBound;
pub const StallClock = @import("loop_detect.zig").StallClock;

pub const Error = error{MlxUnavailable};

pub const SamplingParams = real.SamplingParams;

pub const Constraint = real.Constraint;

pub const TokenLogprob = real.TokenLogprob;

pub const LogprobResult = real.LogprobResult;

pub const GenerationResult = real.GenerationResult;

pub const SchemaConstraint = real.SchemaConstraint;
pub const MAX_TOP_LOGPROBS: u32 = 1024; // mirrors generate.zig
pub const MtpCacheRef = union(enum) {
    qwen: @import("transformer_stub.zig").KVCache,

    pub fn step(_: *const MtpCacheRef) usize {
        return 0;
    }

    pub fn truncate(_: *MtpCacheRef, _: usize, _: anytype) anyerror!void {
        return Error.MlxUnavailable;
    }

    pub fn kv(self: *MtpCacheRef) *@import("transformer_stub.zig").KVCache {
        return &self.qwen;
    }

    pub fn deinit(_: *MtpCacheRef) void {}
};
pub const MtpHead = @import("spec_stub.zig").MtpModel;

pub const MtpHeadRef = union(enum) {
    pub fn moduleOwned(_: MtpHeadRef) bool {
        return false;
    }
    qwen: *MtpHead,
    qwen4: *@import("transformer_stub.zig").Transformer,

    pub fn makeCache(_: MtpHeadRef, _: std.mem.Allocator) anyerror!MtpCacheRef {
        return Error.MlxUnavailable;
    }
};
pub const MtpRestored = struct { cache: MtpCacheRef, base: usize };
const loop_detect = @import("loop_detect.zig");
pub const DegenerateTail = loop_detect.DegenerateTail;
pub const degenerateTail = loop_detect.degenerateTail;
pub const isNearRepeatTailLoop = loop_detect.isNearRepeatTailLoop;

pub const PREFILL_CHUNK_FLOOR: usize = 512;

pub var prefill_chunk_override: usize = 8192;
pub var prefill_chunk_explicit: bool = false;
pub var prefill_trace_force: bool = false;
pub var mtp_history_window_override: usize = 0;
pub var max_mtp_ctx: u32 = 0;
pub var mtp_acceptance_default: @import("mtp_acceptance.zig").Mode = .exact;
pub var mtp_greedy_tail_default: bool = false;
pub const degenerate_loop_reps = loop_detect.degenerate_loop_reps;
pub const near_repeat_window = loop_detect.near_repeat_window;

pub fn generate(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!GenerationResult {
    return Error.MlxUnavailable;
}

pub fn generateMtp(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!GenerationResult {
    return Error.MlxUnavailable;
}

pub fn computeEmbeddingsBatch(_: std.mem.Allocator, _: anytype, _: anytype) anyerror![][]f32 {
    return Error.MlxUnavailable;
}

pub fn visionPrefill(_: anytype) anyerror!void {
    return Error.MlxUnavailable;
}

pub fn sampleTokenLazy(_: anytype, _: SamplingParams, _: anytype) @import("mlx_stub.zig").mlx_array {
    return .{};
}

pub fn installSuppressMask(_: anytype, _: anytype, _: []const u8, _: []const u32) void {}

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| {
        if (e == id) return true;
    }
    return false;
}

pub fn tokensPerSec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0;
    return @as(f64, @floatFromInt(tokens)) * 1_000_000_000.0 / @as(f64, @floatFromInt(elapsed_ns));
}

pub fn prefillTokensPerSec(prompt_tokens: u32, cached_tokens: u32, prefill_ns: u64) f64 {
    return tokensPerSec(prompt_tokens -| cached_tokens, prefill_ns);
}

pub fn effectivePrefillChunk(_: u32, _: u32, _: usize, _: bool, _: bool, _: bool, pinned_chunk: usize) usize {
    return if (pinned_chunk != 0) pinned_chunk else prefill_chunk_override;
}

pub fn visionPrefillUnchunked(_: bool) bool {
    return true;
}

pub const Generator = struct {
    pub fn mtpAdaptiveEnabled() bool {
        return false;
    }
    pub fn mtpModuleHeadReleased(_: *const Generator) bool {
        return false;
    }
    pub fn dflashYieldToCompany(_: *Generator) void {}
    pub fn invalidateRoundClock(_: *Generator) void {}
    pub fn logQsaArms(_: *const Generator) void {}
    pub fn invalidateSerialClock(_: *Generator) void {}
    pub const CancelledCheckpointSink = struct {
        pub fn deinit(_: *@This()) void {}
    };
    pub const DFLASH_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 2.0;
    pub const DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.0;
    pub const DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.8;

    pub fn mtpDepthCapFree(configured: u32) u32 {
        return configured;
    }

    pub fn persistRoundCost(_: *Generator) void {}

    spec_cost_solo: bool = true,
    done: bool = true,
    finish_reason: []const u8 = "stop",
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    generated_ids: std.ArrayListUnmanaged(u32) = .empty,
    next_token_id: u32 = 0,
    has_pending_token: bool = false,
    has_pending_logits: bool = false,
    consecutive_pad: u32 = 0,
    last_logprob: ?LogprobResult = null,
    logprobs_n: u32 = 0,
    sampling: SamplingParams = .{},
    timeout_ns: u64 = 0,
    ctx: @import("transformer_stub.zig").ForwardCtx = .{},
    ssm_checkpoint_alloc: ?std.mem.Allocator = null,

    pld_enabled: bool = false,
    spec_disabled_runtime: bool = false,
    dspark_enabled: bool = false,
    drafter: ?*anyopaque = null,
    dflash: ?*anyopaque = null,
    dflash_ctx: ?@import("spec_stub.zig").DflashCtx = null,
    mtp_cache: ?MtpCacheRef = null,
    mtp_position_base: usize = 0,
    mtp: ?MtpHeadRef = null,

    pub const PldStepResult = struct {
        tokens: []const u32 = &.{},
        accepted_tokens: u32 = 0,
    };

    pub fn init(_: std.Io, _: std.mem.Allocator, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!Generator {
        return Error.MlxUnavailable;
    }

    pub fn initWithOptions(_: std.Io, _: std.mem.Allocator, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) anyerror!Generator {
        return Error.MlxUnavailable;
    }

    pub fn deinit(_: *Generator, _: std.mem.Allocator) void {}

    pub fn resolveMtpDepthCap(_: u32, _: bool) u32 {
        return 0;
    }

    pub fn resolveMtpDepthCapForProfile(_: u32, _: anytype) u32 {
        return 0;
    }

    pub fn next(_: *Generator, _: std.mem.Allocator) anyerror!?u32 {
        return Error.MlxUnavailable;
    }
    pub fn nextPld(_: *Generator, _: std.mem.Allocator, _: u32, _: u32) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }
    pub fn nextDrafter(_: *Generator, _: std.mem.Allocator) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }
    pub fn nextDflash(_: *Generator, _: std.mem.Allocator) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }
    pub fn nextMtp(_: *Generator, _: std.mem.Allocator) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }
    pub fn nextDspark(_: *Generator, _: std.mem.Allocator) anyerror!?PldStepResult {
        return Error.MlxUnavailable;
    }

    pub fn mtpCommittedHistoryLen(_: *const Generator) usize {
        return 0;
    }

    pub fn takeSsmCheckpoints(_: *Generator) []@import("transformer_stub.zig").SSMCheckpoint {
        return &.{};
    }

    pub fn logSpecStats(_: *const Generator) void {}

    pub fn advanceStep(_: *Generator, _: anytype) void {}

    pub fn drainPipelineForBatch(_: *Generator, _: std.mem.Allocator) anyerror!?u32 {
        return Error.MlxUnavailable;
    }
};

pub const AdaptiveWidthState = real.AdaptiveWidthState;
pub fn envPrefillChunk() usize {
    return prefill_chunk_override;
}

pub fn mtpGreedyTailFor(_: ?bool) bool {
    return false;
}
