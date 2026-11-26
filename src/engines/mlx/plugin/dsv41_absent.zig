//! The DeepSeek-V4.1 plugin as a build without it sees it (`dsv41_link.on` false: every target but macOS). One namespace
//! per plugin module `dsv41_link` exports, holding only the declarations the rest of the tree uses, under the same
//! names and nesting; a function keeps its parameter count and does nothing, answering null, false, 0, empty or
//! `error.Dsv41Absent`. No model is ever served here (the registry has no plugin named deepseek_v41), so none of it
//! runs; `dsv41_link`'s drift test holds each name and arity to the real module's.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const dsv4 = @import("../../../plugins/deepseek_v4/deepseek_v4.zig");
const v4_config = @import("../../../plugins/deepseek_v4/config.zig");

const A = mlx.mlx_array;

/// `plugins/deepseek_v41/deepseek_v41.zig`.
pub const deepseek_v41 = struct {
    pub const SPAN: usize = 256;
    pub const PASS_MIN_ROWS: usize = 8192;
    pub const region_rows = 16;

    pub const DsparkDraft = dsv4.DsparkDraft;
    pub const DsparkRound = dsv4.DsparkRound;
    pub const SampledDraft = struct { d: DsparkDraft, q: A };

    pub const Pair = struct {
        lo: usize,
        hi: usize,
        worker: bool,

        pub fn holds(p: Pair, li: usize) bool {
            return (li >= p.lo and li < p.hi) == p.worker;
        }
    };

    pub const State = struct {
        n: usize = 0,
    };

    pub const Dsv41Model = struct {
        s: mlx.mlx_stream,
        cfg: v4_config.Config,
        act: mlx.mlx_dtype,
        vocab: usize,
        n_mtp: usize = 0,
        ds_block: usize = 0,
        streamed: ?void = null,
        dspark: ?void = null,
        dec_state: ?State = null,
        pair: ?Pair = null,
        ds_want_conf: bool = false,
        suppress: A = .{ .ctx = null },

        pub fn deinit(_: *Dsv41Model) void {}

        pub fn imageRows(_: *const Dsv41Model, _: []const u32) usize {
            return 0;
        }
    };

    pub const SnapOp = enum(i32) { take, restore, clone, free, capture };
    pub const Parked = std.AutoHashMap(u64, State);
    pub const Snaps = std.AutoHashMap(u64, void);

    const Want = enum { last, all, replay, encode };

    const ClusterHooksState = struct {
        pair_exchange: ?*const fn (x: A, pre: A, p0: usize, snap: bool, verify: bool) anyerror![2]A = null,
        tp_sum: ?*const fn (x: A) anyerror!A = null,
        tp_step: ?*const fn (ids: []const u32, p0: usize, snap: bool, all: bool, verify: bool, gen: u64, images: ?A) anyerror!void = null,
        tp_batch: ?*const fn (gens: []const u64, p0s: []const usize, chunks: []const []const u32, verify: bool) anyerror!void = null,
        tp_free: ?*const fn (gen: u64) void = null,
        tp_load: ?*const fn (id: u64, ids: []const u32, tensors: []const A) anyerror!void = null,
        tp_snap: ?*const fn (op: SnapOp, a: u64, b: u64) void = null,
    };
    var cluster_hooks: ClusterHooksState = .{};

    pub fn clusterHooks() *ClusterHooksState {
        return &cluster_hooks;
    }

    const Spec = struct {
        rows: usize,
        width: usize,
        dtype: mlx.mlx_dtype,

        pub fn rank(_: Spec) usize {
            return 2;
        }
    };

    const Plan = struct {
        len: usize = 0,

        pub fn specs(_: *const Plan) []const Spec {
            return &.{};
        }
    };

    const DsparkPending = struct {
        vl_g: A = .{ .ctx = null },
        b: usize = 0,
        verify: [16]u32 = @splat(0),
        phases: dsv4.DsparkPhases = .{},

        pub fn lapVerify(_: *DsparkPending, _: *const Dsv41Model) void {}

        pub fn deinit(_: *DsparkPending) void {}
    };

    pub fn passPlanned(_: anytype) bool {
        return false;
    }

    pub fn passRowsLimit(_: anytype) u64 {
        return 0;
    }

    pub fn passCeil() usize {
        return 0;
    }

    pub fn passRows(rows: usize, _: usize) usize {
        return rows;
    }

    pub fn prefillBytes(_: anytype, _: u64, _: u64, _: u64, _: u64) u64 {
        return 0;
    }

    pub fn pairRange(_: anytype, _: u64, _: u64) ?[2]usize {
        return null;
    }

    pub fn init(_: std.mem.Allocator, _: anytype, _: anytype, _: mlx.mlx_stream, _: anytype) !*Dsv41Model {
        return error.Dsv41Absent;
    }

    pub fn dropParked(p: *Parked) void {
        p.deinit();
    }

    pub fn dropSnaps(s: *Snaps, _: std.mem.Allocator) void {
        s.deinit();
    }

    pub fn workerStep(_: *Dsv41Model, _: std.mem.Allocator, _: A, _: A, _: usize, _: bool, _: bool) ![2]A {
        return error.Dsv41Absent;
    }

    pub fn tpWorkerStep(_: *Dsv41Model, _: std.mem.Allocator, _: *Parked, _: u64, _: []const u32, _: usize, _: bool, _: Want, _: bool, _: ?A) !void {
        return error.Dsv41Absent;
    }

    pub fn tpWorkerBatch(_: *Dsv41Model, _: std.mem.Allocator, _: *Parked, _: []const u64, _: []const usize, _: []const []const u32, _: bool) !void {
        return error.Dsv41Absent;
    }

    pub fn tpWorkerFree(_: *Dsv41Model, _: *Parked, _: u64) void {}

    pub fn tpWorkerSnap(_: *Dsv41Model, _: std.mem.Allocator, _: *Parked, _: *Snaps, _: SnapOp, _: u64, _: u64) !void {
        return error.Dsv41Absent;
    }

    pub fn tpLoadPlan(_: *const Dsv41Model, _: usize) !Plan {
        return error.Dsv41Absent;
    }

    pub fn tpWorkerLoad(_: *const Dsv41Model, _: std.mem.Allocator, _: *Snaps, _: u64, _: []const u32, _: []const A) !void {
        return error.Dsv41Absent;
    }

    pub fn exactServes(_: *const Dsv41Model) bool {
        return false;
    }

    pub fn exactSolo(_: *const Dsv41Model) bool {
        return false;
    }

    pub fn copyBlock(_: *const Dsv41Model) usize {
        return 0;
    }

    pub fn maxVerifyDrafts(_: *const Dsv41Model) usize {
        return 0;
    }

    pub fn takeRoundShiftMs(_: *Dsv41Model) f32 {
        return 0;
    }

    pub fn draftSampleOn() bool {
        return false;
    }

    pub fn draftPromptVocab(_: *Dsv41Model, _: *State, _: []const u32, _: bool) !void {
        return error.Dsv41Absent;
    }

    pub fn draftVocabMode(_: *Dsv41Model, _: *State, _: bool) void {}

    pub fn dsparkDraft(_: *Dsv41Model, _: std.mem.Allocator, _: *State, _: u32) !DsparkDraft {
        return error.Dsv41Absent;
    }

    pub fn dsparkDraftSampled(_: *Dsv41Model, _: std.mem.Allocator, _: *State, _: u32, _: A, _: f32) !SampledDraft {
        return error.Dsv41Absent;
    }

    pub fn dsparkDraftBatched(_: *Dsv41Model, _: std.mem.Allocator, _: []const *State, _: []const u32, _: bool, _: []DsparkDraft) !void {
        return error.Dsv41Absent;
    }

    pub fn dsparkRoundWith(_: *Dsv41Model, _: std.mem.Allocator, _: *State, _: u32, _: *const DsparkDraft, _: usize) !DsparkRound {
        return error.Dsv41Absent;
    }

    pub fn dsparkBegin(_: *Dsv41Model, _: std.mem.Allocator, _: *State, _: u32) !DsparkPending {
        return error.Dsv41Absent;
    }

    pub fn dsparkBeginWith(_: *Dsv41Model, _: std.mem.Allocator, _: *State, _: u32, _: *const DsparkDraft) !DsparkPending {
        return error.Dsv41Absent;
    }

    pub fn dsparkFinish(_: *Dsv41Model, _: std.mem.Allocator, _: *State, _: *DsparkPending, _: usize, _: u32) !DsparkRound {
        return error.Dsv41Absent;
    }

    pub fn dsparkObserve(_: *Dsv41Model, _: dsv4.DsparkPhases) void {}

    pub fn extendBatched(_: *Dsv41Model, _: std.mem.Allocator, _: []const *State, _: []const u64, _: []const []const u32, _: bool) !A {
        return error.Dsv41Absent;
    }

    pub fn finishVerify(_: *Dsv41Model, _: std.mem.Allocator, _: *State, _: usize) !void {
        return error.Dsv41Absent;
    }
};

/// `plugins/deepseek_v41/plugin.zig`.
pub const plugin = struct {
    pub const Module = deepseek_v41.Dsv41Model;
    pub const State = deepseek_v41.State;

    pub fn moduleOf(mp: *anyopaque) *Module {
        return @ptrCast(@alignCast(mp));
    }

    pub fn slotStates(_: *anyopaque, _: []const u32, _: []*State, _: []u64) bool {
        return false;
    }

    pub fn lossyTypical() ?@import("../../../sdk/spec/mtp_acceptance.zig").Typical {
        return null;
    }
};

/// `plugins/deepseek_v41/config.zig`.
pub const config = struct {
    pub const plugin = struct {
        pub const name = "";
    };
    pub const of = v4_config.of;
};

/// `plugins/deepseek_v41/bill.zig`: only tests use it, and they skip without the plugin.
pub const bill = struct {};

/// `plugins/deepseek_v41/contract.zig`: the kernel contract lists it only with the plugin.
pub const contract = struct {};

/// `plugins/deepseek_v41/vision.zig`.
pub const vision = struct {
    const Pixels = struct { bytes: []u8, h: u32, w: u32, grid_h: u32, grid_w: u32 };

    pub fn preprocess(_: std.mem.Allocator, _: []const u8, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32) !Pixels {
        return error.Dsv41Absent;
    }

    pub fn spanTokens(_: u32, _: u32) usize {
        return 0;
    }

    pub const Dsv41Vision = struct {
        pub fn init(_: std.mem.Allocator, _: anytype, _: anytype) !Dsv41Vision {
            return error.Dsv41Absent;
        }

        pub fn deinit(_: *Dsv41Vision) void {}

        pub fn forward(_: *Dsv41Vision, _: A, _: u32, _: u32) !A {
            return error.Dsv41Absent;
        }
    };
};

/// `plugins/deepseek_v41/prompt/keep_wide.zig`.
pub const keep_wide = struct {
    pub fn pinned(_: usize, _: u64) void {}
};
