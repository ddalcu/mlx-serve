//! The module archs' functions the engine calls (`fns`), compiled in their own unit under `-Dunits`
//! (`src/modules_unit.zig`, `split.active`): each is exported there as `mlx_serve_mod_<name>` and called here by that
//! symbol, its arguments as a tuple, its error by name (`sdk/util/err_name.zig`). In every other build `call` is the
//! direct call. Naming a function here compiles nothing of it in the engine: only `@TypeOf` reads it.

const std = @import("std");
const err_name = @import("../../../sdk/util/err_name.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const split = @import("split.zig");

pub const fns = struct {
    // DeepSeek-V4/V4.1: a DSpark round on the module, V4.1's vision tower
    pub const dspark_round = @import("../spec/dspark.zig").moduleRound;
    pub const dsv41_vision_init = @import("../vision_encoder.zig").initDsv41Tower;
    pub const dsv41_vision_forward = Dsv41Vision.forward;
    pub const dsv41_vision_deinit = Dsv41Vision.deinit;
    // the Thunderbolt cluster's worker (`mlx-serve dist-worker`): whole models served for rank 0
    pub const cluster_worker = @import("../cluster/distributed.zig").cmdWorker;
    // Sushi packs' EXL3 routed experts
    pub const exl3_routed = @import("../../../sdk/quant/exl3/format.zig").routed;
    // qwen4_exp's MTP head and grouped verify on the plugin's trunk
    pub const qwen4_mtp_forward = Trunk.qwen4MtpForward;
    pub const qwen4_mtp_forward_placed = Trunk.qwen4MtpForwardPlaced;
    pub const qwen4_mtp_forward_batched = Trunk.qwen4MtpForwardBatched;
    pub const qwen4_draft_select = Trunk.qwen4DraftSelect;
    pub const qwen4_draft_shortlist = Trunk.qwen4DraftShortlist;
    pub const qwen4_draft_select_batched = Trunk.qwen4DraftSelectBatched;
    pub const qwen4_draft_shortlists_batched = Trunk.qwen4DraftShortlistsBatched;
    pub const qwen4_build_draft_rerank = Trunk.qwen4BuildDraftRerank;
    pub const qwen4_row_axis_verify = Trunk.forwardRowAxisVerify;
    pub const qwen4_warmup_spec_verify = Trunk.warmupSpecVerify;
};

const Dsv41Vision = @import("../../../plugins/deepseek_v41/vision.zig").Dsv41Vision;
const Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;

const Entry = *const fn (args: *const anyopaque, out: *anyopaque, err: *[]const u8) callconv(.c) bool;

fn Fn(comptime name: []const u8) type {
    return @TypeOf(@field(fns, name));
}

fn Ret(comptime name: []const u8) type {
    return @typeInfo(Fn(name)).@"fn".return_type.?;
}

fn Payload(comptime name: []const u8) type {
    return switch (@typeInfo(Ret(name))) {
        .error_union => |e| e.payload,
        else => Ret(name),
    };
}

fn Result(comptime name: []const u8) type {
    return if (@typeInfo(Ret(name)) == .error_union) anyerror!Payload(name) else Ret(name);
}

/// `fns.<name>(args...)`, in the module archs' unit when this unit calls it through its symbol.
pub fn call(comptime name: []const u8, args: std.meta.ArgsTuple(Fn(name))) Result(name) {
    if (comptime !split.active) return @call(.auto, @field(fns, name), args);
    const entry = @extern(Entry, .{ .name = "mlx_serve_mod_" ++ name });
    var out: Payload(name) = undefined;
    var err: []const u8 = "";
    const ok = entry(@ptrCast(&args), @ptrCast(&out), &err);
    if (comptime @typeInfo(Ret(name)) == .error_union) if (!ok) return err_name.toError(err);
    return out;
}

/// The module archs' unit: every `fns` entry exported under its symbol.
pub fn exportAll() void {
    inline for (comptime std.meta.declarations(fns)) |name| @export(&Export(name).run, .{ .name = "mlx_serve_mod_" ++ name });
}

fn Export(comptime name: []const u8) type {
    return struct {
        fn run(a: *const anyopaque, o: *anyopaque, e: *[]const u8) callconv(.c) bool {
            var args = @as(*const std.meta.ArgsTuple(Fn(name)), @ptrCast(@alignCast(a))).*;
            inline for (0..args.len) |i| if (@TypeOf(args[i]) == std.Io) {
                args[i] = io_util.local(args[i]);
            };
            const out: *Payload(name) = @ptrCast(@alignCast(o));
            if (comptime @typeInfo(Ret(name)) == .error_union) {
                out.* = @call(.auto, @field(fns, name), args) catch |x| {
                    e.* = @errorName(x);
                    return false;
                };
            } else out.* = @call(.auto, @field(fns, name), args);
            return true;
        }
    };
}
