//! GLM-5.3's prompt-pass phase split, for PROFILE builds only (`-Dplugin-profile=true`): each phase of a prompt
//! forward's layer evaluated on its own and its wall time charged to it, accumulated over one prompt pass and printed
//! at its end. In every other build `enabled` is false and each call below compiles to nothing.

const bo = @import("../../sdk/weights/streamed/build_flags.zig");
const ops = @import("../../sdk/weights/streamed/ops.zig");
const sdk_ext = @import("../../sdk/weights/streamed/sdk_ext.zig");
const log = @import("sdk").log;

pub const enabled: bool = if (@hasDecl(bo, "glm_prefill_timers")) bo.glm_prefill_timers else false;

/// The phases of a prompt layer: the read-ahead's predictor, the q / kv / indexer projections and the cache append,
/// the indexer's selection, the attention core, the output projection, the router, the routed experts (split below),
/// the shared expert or the dense MLP, the residual and the layer's eval.
pub const Phase = enum { seed, proj, index, attn, oproj, router, routed, mlp, rest };
/// The routed call's split: the routing barrier (the ids to the host, their counts and order), the read-ahead's landing,
/// the seed and the routes (reads planned and posted), the read waits, the groups' graphs, their compute drains, the
/// staged hot experts' copies into resident rows, the join, the combine's build.
pub const Routed = enum { barrier, ahead, route, wait, encode, compute, promote, join, combine };

const n_phases = @typeInfo(Phase).@"enum".field_names.len;
const n_routed = @typeInfo(Routed).@"enum".field_names.len;

pub var ns: [n_phases]u64 = @splat(0);
pub var routed_ns: [n_routed]u64 = @splat(0);
/// Attention core per layer kind: full (with the indexer) and shared layers.
pub var attn_full_ns: u64 = 0;
pub var attn_shared_ns: u64 = 0;
var last: u64 = 0;

const clock = sdk_ext.profile.Clock(enabled);
pub const Stamp = clock.Stamp;
pub const now = clock.now;

/// The phase clock's start (a layer's first phase).
pub inline fn start() void {
    if (comptime !enabled) return;
    last = now();
}

/// `xs` evaluated, the time since the previous phase's end charged to `p`; returns the time charged.
pub inline fn phase(g: *ops.MlxOps, xs: []const ops.MlxOps.T, p: Phase) !u64 {
    if (comptime !enabled) return 0;
    if (xs.len > 0) try g.evalAll(xs);
    const t = now();
    const d = t - last;
    ns[@intFromEnum(p)] += d;
    last = t;
    return d;
}

/// The time since `t0` charged to the routed call's `r` (the routed phase's total is its own).
pub inline fn chargeRouted(r: Routed, t0: Stamp) void {
    if (comptime !enabled) return;
    routed_ns[@intFromEnum(r)] += now() - t0;
}

pub inline fn chargeAttn(full: bool, d: u64) void {
    if (comptime !enabled) return;
    if (full) attn_full_ns += d else attn_shared_ns += d;
}

fn sec(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e9;
}

/// The prompt pass's split, one line, then the counters zeroed.
pub fn report(tokens: u64) void {
    if (comptime !enabled) return;
    var total: u64 = 0;
    for (ns) |x| total += x;
    const r = struct {
        fn of(b: Routed) f64 {
            return sec(routed_ns[@intFromEnum(b)]);
        }
    }.of;
    log.info("glm_moe_dsa profile: prompt {d} tokens, phases {d:.2} s: seed {d:.2}, proj {d:.2}, index {d:.2}, attn {d:.2} (full {d:.2}, shared {d:.2}), oproj {d:.2}, router {d:.2}, routed {d:.2} (barrier {d:.2}, ahead {d:.2}, route {d:.2}, wait {d:.2}, encode {d:.2}, compute {d:.2}, promote {d:.2}, join {d:.2}, combine {d:.2}), mlp {d:.2}, rest {d:.2}\n", .{
        tokens,                            sec(total),
        sec(ns[@intFromEnum(Phase.seed)]), sec(ns[@intFromEnum(Phase.proj)]),
        sec(ns[@intFromEnum(Phase.index)]), sec(ns[@intFromEnum(Phase.attn)]),
        sec(attn_full_ns),                 sec(attn_shared_ns),
        sec(ns[@intFromEnum(Phase.oproj)]), sec(ns[@intFromEnum(Phase.router)]),
        sec(ns[@intFromEnum(Phase.routed)]), r(.barrier),
        r(.ahead),                         r(.route),
        r(.wait),                          r(.encode),
        r(.compute),                       r(.promote),
        r(.join),                          r(.combine),
        sec(ns[@intFromEnum(Phase.mlp)]),
        sec(ns[@intFromEnum(Phase.rest)]),
    });
    ns = @splat(0);
    routed_ns = @splat(0);
    attn_full_ns = 0;
    attn_shared_ns = 0;
}
