//! G2, the phase change's memory lifecycle: the order every module-owned arch keeps at its handover, the stages an
//! arch's handover supplies, and the harness-only observer. Every refusal is named and sticky; nothing here runs per
//! token, per layer or per cycle.

const std = @import("std");

/// A harness's observer of the phase change (its memory proofs), set before the first request; the served
/// path passes none. A mark records only; it never refuses inside the phase change.
pub const PhaseObserver = struct {
    ctx: *anyopaque,
    mark: *const fn (ctx: *anyopaque, stage: Stage) anyerror!void,

    pub const Stage = enum { start, released, grown };
};
