//! The progress a media backend reports through (`Progress`), the chat turn it gives at each step (`Yield`), and
//! `offload`, which runs a CPU stage off the inference thread while chat keeps it.

const std = @import("std");
const shared = @import("../util/shared.zig");
const preview = @import("preview.zig");

/// Erased progress callback handed into the model code (flux/tts/ltx) so the
/// inner loops can report step/total without importing the HTTP layer.
pub const Progress = struct {
    ctx: *anyopaque,
    cb: *const fn (ctx: *anyopaque, stage: []const u8, step: u32, total: u32) void,
    /// Optional "has the client gone away?" probe. Long inner loops poll it
    /// each step and abort with error.Cancelled — without it a cancelled
    /// request burns the GPU to completion and queues everything behind it.
    cancelled_cb: ?*const fn (ctx: *anyopaque) bool = null,
    /// Opt-in per-step JPEG (issue #208). Null unless the client asked and
    /// the sink is an SSE stream. Cached-velocity H3 steps leave this unused.
    preview_cb: ?*const fn (ctx: *anyopaque, stage: []const u8, step: u32, total: u32, frame: preview.Encoded) void = null,
    preview_opts: preview.Opts = .{},
    /// False when the caller's own `cancelled_cb` already gives chat its turn (a media plugin's table `boundary`).
    yields: bool = true,
    pub fn emit(self: Progress, stage: []const u8, step: u32, total: u32) void {
        self.cb(self.ctx, stage, step, total);
    }
    pub fn emitPreview(self: Progress, stage: []const u8, step: u32, total: u32, frame: preview.Encoded) void {
        if (self.preview_cb) |f| {
            f(self.ctx, stage, step, total, frame);
        } else {
            self.emit(stage, step, total);
        }
    }
    pub fn wantsPreview(self: Progress) bool {
        return self.preview_cb != null and self.preview_opts.enabled;
    }
    pub fn cancelled(self: Progress) bool {
        const f = self.cancelled_cb orelse return false;
        return f(self.ctx);
    }
    /// The per-step poll: gives chat its turn on the inference thread (when a
    /// scheduler hook is installed), then reports whether the client left.
    pub fn boundary(self: Progress) bool {
        if (self.yields) if (genYield().*) |y| y.step(y.ctx);
        return self.cancelled();
    }
};

/// The scheduler's chat turn while a media job owns the inference thread.
pub const Yield = struct {
    ctx: *anyopaque,
    /// At a step boundary: run chat for its share of the step just finished.
    step: *const fn (ctx: *anyopaque) void,
    /// Serve chat until `done` is set; `wake` interrupts its idle wait.
    serve_until: *const fn (ctx: *anyopaque, done: *const std.atomic.Value(bool)) void,
    wake: *const fn (ctx: *anyopaque) void,
};

/// Installed by the scheduler around a media job, on the inference thread only.
/// Thread-local rather than a `Progress` field so every wrapper that rebuilds a
/// `Progress` (H3's chained windows, LTX's windows) keeps it.
pub fn genYield() *?*const Yield {
    return GenYield.get();
}
pub const GenYield = shared.SharedThreadLocal(?*const Yield, "gen_yield", null);

/// A streaming job's keepalive: `beat_until` writes a keepalive whenever the stream has been silent, until `done`.
pub const Keepalive = struct {
    ctx: *anyopaque,
    beat_until: *const fn (ctx: *anyopaque, done: *const std.atomic.Value(bool)) void,

    fn run(k: Keepalive, done: *const std.atomic.Value(bool)) void {
        k.beat_until(k.ctx, done);
    }
};

/// The streaming job's keepalive on this thread, set by the host's stream sink and cleared when the job ends: a CPU
/// stage that runs for minutes (`offload`) keeps the socket alive through it.
pub fn streamKeepalive() *?Keepalive {
    return StreamKeepalive.get();
}
pub const StreamKeepalive = shared.SharedThreadLocal(?Keepalive, "stream_keepalive", null);

fn ReturnOf(comptime func: anytype) type {
    return @typeInfo(@TypeOf(func)).@"fn".return_type.?;
}

/// Run a pure-CPU stage (zero MLX: it runs off the inference thread) on a worker
/// thread while chat keeps the inference thread. Inline when no hook is installed.
pub fn offload(comptime func: anytype, args: std.meta.ArgsTuple(@TypeOf(func))) ReturnOf(func) {
    const ka = streamKeepalive().*;
    if (genYield().* == null and ka == null) return @call(.auto, func, args);
    const Job = struct {
        args: @TypeOf(args),
        y: ?*const Yield,
        result: ReturnOf(func) = undefined,
        done: std.atomic.Value(bool) = .init(false),
        fn run(j: *@This()) void {
            j.result = @call(.auto, func, j.args);
            j.done.store(true, .release);
            if (j.y) |y| y.wake(y.ctx);
        }
    };
    var job = Job{ .args = args, .y = genYield().* };
    const t = std.Thread.spawn(.{}, Job.run, .{&job}) catch return @call(.auto, func, args);
    const beater = if (ka) |k| std.Thread.spawn(.{}, Keepalive.run, .{ k, &job.done }) catch null else null;
    if (job.y) |y| y.serve_until(y.ctx, &job.done);
    t.join();
    if (beater) |b| b.join();
    return job.result;
}
