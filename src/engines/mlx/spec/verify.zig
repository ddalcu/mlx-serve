//! Verify-round invariants shared by every drafter: committed argmax, accept caps, DSpark copy length.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const testing = std.testing;
const Constraint = @import("../grammar/constraint.zig").Constraint;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const scalarLike = @import("../sampling/params.zig").scalarLike;
const tokenStops = @import("../../../sdk/request/stops.zig").tokenStops;


/// A greedy spec-verify argmax whose entries are COMMITTED as next tokens.
/// `verifyArgmax` is its only constructor and every commit site takes this type,
/// so a raw `mlx_argmax_axis` cannot reach a commit without the reserved-id mask.
pub const CommittedArgmax = struct {
    arr: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *CommittedArgmax) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }

    pub fn present(self: CommittedArgmax) bool {
        return self.arr.ctx != null;
    }

    /// The lazy handle, for an eval vector or a forced eval - never for a commit.
    pub fn lazy(self: CommittedArgmax) mlx.mlx_array {
        return self.arr;
    }

    /// The realized ids; the caller must have evaluated `lazy()` first.
    pub fn ids(self: CommittedArgmax, len: usize) ![]const i32 {
        const data = mlx.mlx_array_data_int32(self.arr) orelse return error.MlxArrayDataNull;
        return data[0..len];
    }
};

/// The greedy spec-verify argmax, over the reserved-id mask when the model has
/// one. The sampled verify filters through `applySuppressMask` and the rerank
/// draft select takes the mask, so without this the greedy verify was the one
/// path that could commit a flagged special or a padding row past the defined
/// vocabulary — ids the serial sampler can never draw.
pub fn verifyArgmax(logits: mlx.mlx_array, mask: ?mlx.mlx_array, s: mlx.mlx_stream) !CommittedArgmax {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (mask) |m| {
        const neg_inf = try scalarLike(-std.math.inf(f32), logits, s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        try mlx.check(mlx.mlx_argmax_axis(&out, masked, 2, false, s));
    } else {
        try mlx.check(mlx.mlx_argmax_axis(&out, logits, 2, false, s));
    }
    return .{ .arr = out };
}

/// A speculative decode step (PLD / drafter / MTP) cannot honor a grammar
/// constraint — the drafts bypass the per-token grammar mask — nor per-token
/// logprobs, which the draft path never captures. The scheduler and server
/// already select a spec mode only when both are absent, but `nextPld` /
/// `nextDrafter` / `nextMtp` only *documented* that with `std.debug.assert`,
/// which compiles out in the ReleaseFast build users actually run (issue #97).
/// A future dispatch bug that let a constrained request reach a spec step would
/// then read the constrained branch's placeholder token as a real commit and
/// stream silently off-schema output as a normal 200. So the spec steps gate on
/// this real, release-enforced check and return `error.SpecDecodeUnsupported`
/// instead of asserting.
pub fn specDecodeUnsupported(sampling: SamplingParams, logprobs_n: u32) bool {
    return sampling.constraint != null or logprobs_n != 0;
}

/// Number of accepted draft tokens that may accompany the always-committed
/// anchor token without crossing the request's output-token ceiling.
pub fn capAcceptedForTokenBudget(accepted: u32, completion: u32, max_tokens: u32) u32 {
    const remaining = max_tokens -| completion;
    if (remaining <= 1) return 0;
    return @min(accepted, remaining - 1);
}

/// Drafts a DSpark copy round takes from a context match, or 0 for the stage draft. The stage block
/// already lands most drafts on repeated text, so only a quoted span (a long agreement) whose copy
/// is longer than the block is worth the wider verify.
pub fn dsparkCopyLen(found: ?mtp_lookup.Match, remaining: u32, block: usize) usize {
    const f = found orelse return 0;
    if (f.suffix < mtp_lookup.STRONG_SUFFIX) return 0;
    const n = @min(f.draft.len, remaining);
    return if (n > block) n else 0;
}

// ── tests ──


test "capAcceptedForTokenBudget keeps speculative commits inside max_tokens" {
    try std.testing.expectEqual(@as(u32, 15), capAcceptedForTokenBudget(15, 0, 400));
    try std.testing.expectEqual(@as(u32, 2), capAcceptedForTokenBudget(15, 397, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 399, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 400, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 401, 400));
}

test "dsparkCopyLen copies only a long agreement that outruns the stage block" {
    const draft: [15]u32 = @splat(0);
    const strong = mtp_lookup.STRONG_SUFFIX;
    try std.testing.expectEqual(@as(usize, 15), dsparkCopyLen(.{ .draft = &draft, .suffix = strong }, 400, 5));
    try std.testing.expectEqual(@as(usize, 0), dsparkCopyLen(null, 400, 5));
    try std.testing.expectEqual(@as(usize, 0), dsparkCopyLen(.{ .draft = &draft, .suffix = strong - 1 }, 400, 5));
    try std.testing.expectEqual(@as(usize, 0), dsparkCopyLen(.{ .draft = draft[0..5], .suffix = strong }, 400, 5));
    try std.testing.expectEqual(@as(usize, 7), dsparkCopyLen(.{ .draft = &draft, .suffix = strong }, 7, 5));
    try std.testing.expectEqual(@as(usize, 0), dsparkCopyLen(.{ .draft = &draft, .suffix = strong }, 5, 5));
}

test "every speculative decoder caps accepted drafts before commit" {
    // Class guard: every speculative path commits an always-emitted anchor
    // plus zero or more accepted drafts. A new/forgotten arm must cap that
    // accepted prefix before it mutates generated ids, KV/module state, or
    // usage. Output-only clipping in the scheduler is already too late.
    const source = @import("../generate.zig").all_source;
    const names = [_][]const u8{
        "nextDspark",
        "nextPld",
        "nextDrafter",
        "nextDflash",
        "mtpRoundFinish",
    };
    for (names) |name| {
        const signature = try std.fmt.allocPrint(testing.allocator, "\npub fn {s}", .{name});
        defer testing.allocator.free(signature);
        const start = std.mem.indexOf(u8, source, signature) orelse return error.MissingSpecDecoder;
        const end = std.mem.indexOfPos(u8, source, start + signature.len, "\n//! ") orelse source.len;
        if (std.mem.indexOf(u8, source[start..end], "capAcceptedForTokenBudget(") == null) {
            std.debug.print("{s} does not cap accepted drafts before commit\n", .{name});
            return error.UncappedSpecDecoder;
        }
    }
}

test "specDecodeUnsupported: release-enforced guard for spec + constraint/logprobs (issue #97)" {
    // Speculative decode (PLD/drafter/MTP) cannot honor a grammar constraint or
    // per-token logprobs. nextPld/nextDrafter/nextMtp only asserted this, which
    // compiles out in ReleaseFast (issue #97) — this is the real check they now
    // use to fail loud instead of streaming silently off-schema output.
    try testing.expect(!specDecodeUnsupported(.{}, 0)); // plain sampling → spec OK
    try testing.expect(specDecodeUnsupported(.{}, 1)); // any logprobs → not OK
    try testing.expect(specDecodeUnsupported(.{}, 5));
    // A grammar constraint present → not OK. The guard only null-checks the
    // pointer (never dereferences), so a dummy address is sufficient here.
    var dummy: Constraint = undefined;
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 0));
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 3));
}

test "a block decoder's entry token stops the round before it drafts" {
    // Bar: an EOS or third pad as the round's entry token ends the request before any draft.
    const eos = [_]u32{ 151645, 151643 };
    var pad: u32 = 0;
    try testing.expect(tokenStops(151645, &eos, &pad));
    try testing.expect(tokenStops(151643, &eos, &pad));
    try testing.expect(!tokenStops(17, &eos, &pad));

    // A pad run stops on the third; any real token resets it.
    pad = 0;
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(tokenStops(0, &eos, &pad));
    pad = 0;
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(!tokenStops(9, &eos, &pad));
    try testing.expectEqual(@as(u32, 0), pad);
}
