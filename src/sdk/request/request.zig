//! The request vocabulary the host and the engines share: sampling, grammar, forced tokens, logprobs, results, stops
//! and the loop guard's thresholds. Pure data and CPU logic; no engine code.

pub const SamplingParams = @import("sampling.zig").SamplingParams;
pub const Constraint = @import("../grammar/constraint.zig").Constraint;
pub const SchemaConstraint = @import("../grammar/constraint.zig").SchemaConstraint;
pub const forcedBoundaryCanContinue = @import("../grammar/constraint.zig").forcedBoundaryCanContinue;
pub const ThinkBound = @import("forced.zig").ThinkBound;
pub const ToolForce = @import("forced.zig").ToolForce;
pub const LogprobResult = @import("logprob.zig").LogprobResult;
pub const TokenLogprob = @import("logprob.zig").TokenLogprob;
pub const MAX_TOP_LOGPROBS = @import("logprob.zig").MAX_TOP_LOGPROBS;
pub const GenerationResult = @import("result.zig").GenerationResult;
pub const tokensPerSec = @import("result.zig").tokensPerSec;
pub const prefillTokensPerSec = @import("result.zig").prefillTokensPerSec;
pub const isEosId = @import("stops.zig").isEosId;
pub const degenerate_loop_min_span = @import("loop_guard.zig").degenerate_loop_min_span;
pub const degenerate_loop_long_min_span = @import("loop_guard.zig").degenerate_loop_long_min_span;
pub const near_repeat_window = @import("loop_guard.zig").near_repeat_window;
pub const near_repeat_min_span = @import("loop_guard.zig").near_repeat_min_span;
pub const DegenerateTail = @import("loop_guard.zig").DegenerateTail;
pub const degenerateTail = @import("loop_guard.zig").degenerateTail;
