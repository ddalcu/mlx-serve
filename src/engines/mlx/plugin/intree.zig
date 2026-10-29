//! P8 DEBT, in-tree only: where the engine reaches a served in-tree plugin's own module because the plugin table has
//! no hook for it yet. One section per plugin, each listing what its ABI-clean replacement needs (the plugin's
//! AGENTS.md carries the same list); delete a section when its hooks land, and the file when none is left.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const model = @import("../../../sdk/config/model.zig");
const Resident = @import("../../../sdk/weights/resident.zig").Resident;
const Transformer = @import("../transformer.zig").Transformer;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const distributed = @import("../cluster/distributed.zig");
const qwen4 = @import("../../../plugins/qwen4_exp/plugin.zig");
const Qwen4Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;

/// The map the driver already loaded, moved out of the `Resident` behind `ctx.weights` (it is left empty, so the
/// driver's `deinit` frees nothing the plugin keeps); null when the source is not the driver's resident map.
pub fn takeWeights(ctx: *const abi.LoadCtx) ?model.Weights {
    return Resident.take(ctx.weights orelse return null);
}

/// A served model's plugin module (`xfm.plugin.?.model`), for engine code that reaches it through that plugin's own
/// pub accessor; null on the legacy path.
pub fn module(xfm: *const Transformer) ?*anyopaque {
    const b = xfm.plugin orelse return null;
    return b.model;
}

// ── deepseek_v4, deepseek_v41 ──
// DSpark rounds (spec/dspark.zig, generate/init.zig) run on the module itself. The ABI-clean replacement needs a
// native block drafter in `SpecHooks`: a round taking the committed token and an accept cap and returning the
// accepted tokens, the next token and the new position (the module rewinds its own state), the stochastic arm's
// filtered-probs accept with the request's seed, the block width (`ds_block`) and the policy's per-round confidence
// (`ds_want_conf`, the copy-step drafts).

const v4plugin = @import("../../../plugins/deepseek_v4/plugin.zig");
const v41plugin = @import("../../../plugins/deepseek_v41/plugin.zig");

/// The DeepSeek-V4 module of the served plugin behind `xfm`; null for any other model.
pub fn dsv4(xfm: *const Transformer) ?*v4plugin.Module {
    const b = xfm.plugin orelse return null;
    return if (served(b.table, @import("../../../plugins/deepseek_v4/config.zig").plugin.name)) v4plugin.moduleOf(b.model) else null;
}

/// The DeepSeek-V4.1 module of the served plugin behind `xfm`; null for any other model.
pub fn dsv41(xfm: *const Transformer) ?*v41plugin.Module {
    const b = xfm.plugin orelse return null;
    return if (served(b.table, @import("../../../plugins/deepseek_v41/config.zig").plugin.name)) v41plugin.moduleOf(b.model) else null;
}

/// Whether `table` is the plugin named `name`. By name: under `-Dunits` the served table is the plugin unit's export,
/// and this unit's copy of it sits at another address.
fn served(table: *const abi.ModelTable, name: []const u8) bool {
    return std.mem.eql(u8, table.name.slice(), name);
}

// ── qwen4_exp ──
// Its forward runs through the table; the MTP head (padded placement, lazy predraft, QSA leftover marks, prefix-cache
// and SSD head history, group lanes and the grouped verify, the coarse rerank, the G17 cost profile) runs on the
// plugin's trunk module through the `.qwen4` arm (`MtpHeadRef{ .qwen4 = trunk }`, every engine site that touches the
// trunk's `qwen4` / `qwen4_mtp` goes through `Transformer.qwen4Host`). An ABI-clean head needs: (1)
// `ForwardOut.next`, the chained hidden apart from the projected one; (2) a fact that the trunk's hidden is the
// pre-mixer stream; (3) a placed head forward (dead history rows); (4) `HeadHooks.mark` with a length; (5) head
// snapshot / bytes / serialize for the prefix cache and SSD; (6) group lanes and the grouped verify on the plugin arm;
// (7) the rerank on the head's mixer output. Beside the head: (8) the slot's whole M-RoPE table, which the QSA indexer
// ropes pooled keys with at block starts the chunk does not carry (`enter`); (9) a `dist` hook (a Thunderbolt pair
// keeps the legacy path, `declines`).

fn isQwen4(table: *const abi.ModelTable) bool {
    return served(table, @import("../../../plugins/qwen4_exp/config.zig").plugin.name);
}

/// qwen4_exp builds its own MXFP8 head (`lm_head_mxfp8`, its default on for a quantized pack on an integrated CUDA
/// GPU), so the driver leaves its map alone.
pub fn buildsMxfp8Head(table: *const abi.ModelTable) bool {
    return isQwen4(table);
}

/// The plugin's own trunk module behind a served qwen4_exp shell; null for every other model.
pub fn inner(xfm: *const Transformer) ?*Qwen4Trunk {
    const b = xfm.plugin orelse return null;
    if (!isQwen4(b.table)) return null;
    return &@as(*qwen4.Model, @ptrCast(@alignCast(b.model))).trunk;
}

/// qwen4_exp's trunk module (state `qwen4`, head `qwen4_mtp`); null for every other model.
pub const trunk = inner;

/// Lends the shell the inner trunk's embedding table and head (borrowed, as `adoptTrunkWeights` does): the
/// generator's draft readouts read them off the target.
pub fn lendTrunk(xfm: *Transformer) void {
    const t = inner(xfm) orelse return;
    xfm.emb_w = t.emb_w;
    xfm.emb_s = t.emb_s;
    xfm.emb_b = t.emb_b;
    xfm.lm_head_w = t.lm_head_w;
    xfm.lm_head_s = t.lm_head_s;
    xfm.lm_head_b = t.lm_head_b;
}

/// The engine's context of the serial forward in flight, for the qwen4_exp plugin (`leave` clears it).
pub fn enter(ctx: *const ForwardCtx) void {
    qwen4.engineCtx().* = ctx;
}

pub fn leave() void {
    qwen4.engineCtx().* = null;
}

// ── tests ──

const testing = std.testing;

test "intree: takeWeights moves the driver's map out of its resident source" {
    const mlx = @import("../../../mlx/ffi.zig");
    var w = model.Weights.init(testing.allocator);
    try w.put("x.weight", mlx.mlx_array_new_float(2.0));
    var r = Resident.adopt(w);
    defer r.deinit();
    const src = r.source();
    var got = takeWeights(&.{ .model_dir = .{}, .dir = undefined, .parsed = undefined, .weights = &src }) orelse return error.NotTaken;
    defer got.deinit();
    try testing.expectEqual(@as(u32, 1), got.count());
    try testing.expect(takeWeights(&.{ .model_dir = .{}, .dir = undefined, .parsed = undefined }) == null);
}
