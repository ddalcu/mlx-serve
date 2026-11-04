//! The template quant plugin, a stub: claim, validation and bill are real; loading and the dispatches refuse by name
//! until a format fills them; `src/sdk/quant/native.zig` is a filled one. Copy the folder to start a format.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const sdk = @import("../../sdk/root.zig");
pub const config = @import("config.zig");

pub const plugin = sdk.Plugin{ .name = "_template_quant", .kind = .quant };

pub const table = abi.QuantTable{
    .name = abi.Str.of(plugin.name),
    .claims = &config.claims,
    .validate = &config.validate,
    .bill = &config.bill,
    .open = &open,
    .close = &close,
    .load_tensor = &loadTensor,
    .load_bank = &loadBank,
    .release = &release,
    .matmul = &matmul,
    .moe = &moe,
};

const unsupported = abi.Verdict.refuse(.unsupported, "_template_quant: a stub");

fn open(_: *abi.LoadCtx) callconv(.c) abi.Verdict {
    return unsupported;
}
fn close(_: *abi.LoadCtx) callconv(.c) void {}
fn loadTensor(_: *const abi.LoadCtx, _: abi.Str, _: u32, _: *abi.Weight) callconv(.c) abi.Verdict {
    return unsupported;
}
fn loadBank(_: *const abi.LoadCtx, _: abi.Str, _: u32, _: *abi.Weight) callconv(.c) abi.Verdict {
    return unsupported;
}
fn release(_: *abi.Weight) callconv(.c) void {}
fn matmul(_: *const abi.Weight, _: abi.Array, _: abi.Stream, _: *abi.Array) callconv(.c) abi.Verdict {
    return unsupported;
}
fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) abi.Verdict {
    return unsupported;
}

pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    _ = a;
    return .{ .files = &.{}, .block = "{\"mode\":\"_template_quant\",\"bits\":4,\"group_size\":64}" };
}
