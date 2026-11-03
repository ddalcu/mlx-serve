//! A served model's per-slot plugin calls: the slot id the driver hands in `Forward.slots`, the deferred host work a
//! `defer_host` forward leaves (`forward_settle`) behind the Transformer's deferred-PLE entry points, the slot's
//! release and its meter lines. A legacy model owes nothing.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const log = @import("../../../sdk/util/log.zig");
const ple = @import("../../../plugins/qwen4_exp/ple.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;
const Bound = @import("driver.zig").Bound;

const MAX_GROUP = 64;

/// The slot's id, given on its first served forward (ids start at 1; 0 = none yet).
pub fn slotId(b: *Bound, ctx: *ForwardCtx) u32 {
    if (ctx.plugin_slot == 0) {
        b.next_slot +%= 1;
        if (b.next_slot == 0 or b.next_slot == ForwardCtx.shell_slot) b.next_slot = 1;
        ctx.plugin_slot = b.next_slot;
    }
    return ctx.plugin_slot;
}

fn call(b: *Bound, ids: []const u32, how: abi.Settle, tokens: []const u32) !void {
    const f = b.table.forward_settle orelse return error.PluginSettleMissing;
    const v = f(b.model, .of(ids), how, &abi.Span(u32).of(tokens));
    if (!v.isOk()) {
        log.err("[plugin] {s}: settle failed: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return error.PluginSettleFailed;
    }
}

/// `Transformer.flushDeferredPle`: settles what the slot's last forward left owed, reading its tokens off the device.
pub fn flush(xfm: *Transformer, ctx: *ForwardCtx) !void {
    const b = xfm.plugin orelse return;
    if (ctx.plugin_owed == .none) return;
    try call(b, &.{ctx.plugin_slot}, .flush, &.{});
    ctx.plugin_owed = .none;
}

/// `Transformer.flushDeferredPleGroup`: every owing slot of a batched tick in one call (one device read).
pub fn flushGroup(xfm: *Transformer, ctxs: []const *ForwardCtx) !void {
    const b = xfm.plugin orelse return;
    var ids: [MAX_GROUP]u32 = undefined;
    var n: usize = 0;
    for (ctxs) |c| if (c.plugin_owed != .none) {
        if (n == ids.len) return error.PleGroupTooWide;
        ids[n] = c.plugin_slot;
        n += 1;
    };
    if (n == 0) return;
    try call(b, ids[0..n], .flush, &.{});
    for (ctxs) |c| c.plugin_owed = .none;
}

/// `Transformer.settleDeferredPle`: settles the slot's owed history with the tokens the engine already read.
pub fn settleWith(xfm: *Transformer, ctx: *ForwardCtx, tokens: []const u32) !void {
    const b = xfm.plugin orelse return;
    if (ctx.plugin_owed == .none) return;
    if (ctx.plugin_owed != .history) return error.PleLeafUnfilled;
    try call(b, &.{ctx.plugin_slot}, .flush, tokens);
    ctx.plugin_owed = .none;
}

/// `Transformer.discardDeferredPle`: drops what the slot was owed (the forward that built on it is abandoned).
pub fn discard(xfm: *Transformer, ctx: *ForwardCtx) void {
    const b = xfm.plugin orelse return;
    if (ctx.plugin_owed == .none) return;
    ctx.plugin_owed = .none;
    call(b, &.{ctx.plugin_slot}, .discard, &.{}) catch {};
}

/// `Transformer.deferredPleOnGpu`: only host history is owed, which may settle later with tokens already read.
pub fn historyOwed(ctx: *const ForwardCtx) bool {
    return ctx.plugin_owed == .history or ple.deferredPleOnGpu(ctx);
}

/// The model defers host work past its forwards (the decode step sends the sampled token out as its own batch).
pub fn defersHost(xfm: *const Transformer) bool {
    const b = xfm.plugin orelse return false;
    return b.table.forward_settle != null;
}

/// The slot is gone: the plugin frees what it keeps for it.
pub fn release(xfm: *Transformer, ctx: *ForwardCtx) void {
    const b = xfm.plugin orelse return;
    if (ctx.plugin_slot == 0) return;
    if (b.table.release_slot) |f| f(b.model, ctx.plugin_slot);
    ctx.plugin_slot = 0;
}

/// Engine code is about to drive the plugin's module directly for this slot (DSpark): its state becomes the module's.
pub fn activate(xfm: *Transformer, ctx: *ForwardCtx) !void {
    const b = xfm.plugin orelse return;
    const f = b.table.activate_slot orelse return;
    const v = f(b.model, slotId(b, ctx));
    if (!v.isOk()) {
        log.err("[plugin] {s}: slot state not activated: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return error.PluginActivateFailed;
    }
}

/// Logs the slot's meter lines (the request's summary, beside `[spec-stats]`).
pub fn logMeters(xfm: *Transformer, ctx: *const ForwardCtx) void {
    const b = xfm.plugin orelse return;
    const f = b.table.meters orelse return;
    if (ctx.plugin_slot == 0) return;
    var out: abi.Str = .{};
    f(b.model, ctx.plugin_slot, &out);
    var lines = std.mem.splitScalar(u8, out.slice(), '\n');
    while (lines.next()) |line| if (line.len > 0) log.info("  {s}\n", .{line});
}
