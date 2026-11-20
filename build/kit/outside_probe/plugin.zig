//! A kit check, not an example (`checks/outside_probe`): a model plugin built from the staged kit whose load opens its
//! pack through the outside quant table that claims it (`kit.outside_model`) and closes it again. Like every model
//! dylib it holds its own copy of the SDK's quant table list, empty, so `host/plugins/kit_test.zig` shows the claim
//! arriving through the host (`Host.quant_claimant`). It serves nothing.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const quant = kit.abi.quant;
pub const config = @import("config.zig");

pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{
    .name = "_outside_probe",
    .kind = .model,
    .claims = &config.model_types,
};

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{},
    .weights = &.{},
};

const gpa = std.heap.c_allocator;

fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

fn noTensor(_: ?*anyopaque, _: quant.Str) callconv(.c) quant.Array {
    return .{};
}

/// The pack opened through the table claiming it, then closed: a one-byte handle when a table claimed it; refused
/// `unsupported` by name when none did.
fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    const source: quant.Source = if (ctx.weights) |w| w.* else .{ .ctx = null, .get = &noTensor };
    var l = kit.outside_model.Load.open(gpa, ctx, source, .{ .ctx = ctx.stream.ctx }) catch |e| {
        out.* = abi.Verdict.named(.failed, @errorName(e));
        return null;
    };
    defer l.deinit();
    if (l.format == null) {
        out.* = abi.Verdict.refuse(.unsupported, "_outside_probe: no outside quant table claims the pack");
        return null;
    }
    return gpa.create(u8) catch {
        out.* = abi.Verdict.refuse(.failed, "_outside_probe: out of memory");
        return null;
    };
}

fn unload(h: *anyopaque) callconv(.c) void {
    gpa.destroy(@as(*u8, @ptrCast(h)));
}

fn forward(_: *anyopaque, _: *const abi.Forward, _: *abi.ForwardOut) callconv(.c) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "_outside_probe: serves nothing");
}

pub fn conformanceFixture(_: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    return .{ .files = &.{} };
}
