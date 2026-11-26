//! Inkling 2's visit as a few kernels around the matmuls. A one-row visit is bound by the launches on its critical
//! path, a prompt's rows by the passes over their arrays, not by the weights' bytes: each kernel here stands for a
//! whole op chain of `layers.zig`, which stays the reference and serves a CPU stream, batched slots and any dtype
//! outside the kernels' set.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const block = @import("../../sdk/bases/block.zig");
const kv = @import("../../sdk/bases/kv.zig");
const abi = @import("../../abi/plugin.zig");
const band = @import("../../sdk/bases/blocks/sliding.zig");
const slot_state = @import("../../sdk/bases/blocks/state.zig");
const sdpaSinks = @import("../../sdk/bases/blocks/decoder_attention.zig").sdpaSinks;
const moe_fp4 = @import("../../sdk/blocks/moe/moe_fp4.zig");
const mxfp8 = @import("../../sdk/blocks/qmm/mxfp8_qmv.zig");
const rel = @import("../inkling/attention.zig");
const layers = @import("layers.zig");
const Config = @import("config.zig").Config;

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;
const Layer = layers.Layer;
const Body = layers.Body;

pub const Kernel = struct {
    name: [*:0]const u8,
    ins: []const [*:0]const u8,
    outs: []const [*:0]const u8,
    source: [*:0]const u8,
    header: [*:0]const u8 = "",
    /// False = the kernel walks its inputs' strides itself (a cache view is not copied).
    contiguous: bool = true,
    built: ?mlx.mlx_fast_metal_kernel = null,
};

/// The SDK's fp4 and fp8 decodes under the packed expert kernels' chunk reader.
pub const bank_header = moe_fp4.NV_SCALE ++ moe_fp4.DOT16 ++ mxfp8.HEADER ++ kernelSource("plugins/inkling2/kernels/bank_header.metal");
/// `relsplit_lanes.metal`'s matrix element: half, and float where the stream is float32 (its read is then the chain's sums).
const lanes_header = "template <typename T> struct msv_ink2_el { typedef half type; };\ntemplate <> struct msv_ink2_el<float> { typedef float type; };\n";
/// Output rows a SIMD group of a packed expert kernel owns (one: a lane's loop sets the launch's latency, and here
/// that beats keeping x in registers across rows), and the up kernel's SIMD groups per threadgroup (each threadgroup
/// ranks its row's picks: fewer, wider ones rank fewer times).
const bank_rows: c_int = 1;
const bank_groups: c_int = 8;

pub var pre_kernel: Kernel = .{ .name = "msv_inkling2_pre", .ins = &.{ "HIN", "BLK", "QY", "NW0", "NW1", "NW2", "BANK", "BPART", "MQ", "PR" }, .outs = &.{ "X", "NORMED", "A", "MSGO", "BOTH" }, .source = kernelSource("plugins/inkling2/kernels/pre.metal") };
pub var prep_kernel: Kernel = .{ .name = "msv_inkling2_prep", .ins = &.{ "QP", "GP", "RP", "KP", "VP", "KDY", "VDY", "ST", "KW", "VW", "QS", "RS", "RJ", "PR", "PARENT" }, .outs = &.{ "Q", "GATE", "K", "V", "BIAS", "REL" }, .source = kernelSource("plugins/inkling2/kernels/prep.metal") };
pub var oproj_kernel: Kernel = .{ .name = "msv_inkling2_oproj", .ins = &.{ "ATT", "GATE", "WQ", "WS", "WB" }, .outs = &.{"Y"}, .source = kernelSource("plugins/inkling2/kernels/oproj.metal"), .header = bank_header };
pub var gated_kernel: Kernel = .{ .name = "msv_inkling2_gated", .ins = &.{ "ATT", "GATE" }, .outs = &.{"OUT"}, .source = kernelSource("plugins/inkling2/kernels/gated.metal") };
pub var mid_kernel: Kernel = .{ .name = "msv_inkling2_mid", .ins = &.{ "OP", "ST", "AW", "HIN", "NORMED", "BLK", "QY", "PN", "MSG3", "PR", "PARENT" }, .outs = &.{ "RES", "X", "X32", "M", "BOTH" }, .source = kernelSource("plugins/inkling2/kernels/mid.metal") };
pub var gate_kernel: Kernel = .{ .name = "msv_inkling2_gate", .ins = &.{ "A", "MS", "ST", "F", "G" }, .outs = &.{"Y"}, .source = kernelSource("plugins/inkling2/kernels/gate.metal") };
pub var norm_kernel: Kernel = .{ .name = "msv_inkling2_norm", .ins = &.{ "Y", "NW0", "NW1", "NW2", "RT", "PR" }, .outs = &.{ "X", "SCORES" }, .source = kernelSource("plugins/inkling2/kernels/norm.metal") };
pub var topk_kernel: Kernel = .{ .name = "msv_inkling2_topk", .ins = &.{ "SC", "RB", "PR" }, .outs = &.{ "IDX", "WTS" }, .source = kernelSource("plugins/inkling2/kernels/topk.metal") };
pub var act_kernel: Kernel = .{ .name = "msv_inkling2_act", .ins = &.{ "GU", "IDX", "SC", "PR" }, .outs = &.{"ACT"}, .source = kernelSource("plugins/inkling2/kernels/act.metal") };
pub var ups_kernel: Kernel = .{ .name = "msv_inkling2_ups", .ins = &.{ "X", "WQ", "WS", "WB", "SCR", "RB", "RS", "IDX", "SC", "PR" }, .outs = &.{ "ACT", "PIDX", "PWTS" }, .source = kernelSource("plugins/inkling2/kernels/ups.metal"), .header = bank_header };
pub var downs_kernel: Kernel = .{ .name = "msv_inkling2_downs", .ins = &.{ "ACT", "WQ", "WS", "WB", "IDX", "SC", "WTS", "RES", "ST", "MW", "PR" }, .outs = &.{ "HOUT", "XM" }, .source = kernelSource("plugins/inkling2/kernels/downs.metal"), .header = bank_header };
pub var group_kernel: Kernel = .{ .name = "msv_inkling2_group", .ins = &.{"IDX"}, .outs = &.{ "TEXP", "TCNT", "TPICK" }, .source = kernelSource("plugins/inkling2/kernels/group.metal") };
pub var ups_grp_kernel: Kernel = .{ .name = "msv_inkling2_ups_grp", .ins = &.{ "X", "WQ", "WS", "WB", "TEXP", "TCNT", "TPICK", "SC", "PR" }, .outs = &.{"ACT"}, .source = kernelSource("plugins/inkling2/kernels/ups_grp.metal"), .header = bank_header };
pub var downs_grp_kernel: Kernel = .{ .name = "msv_inkling2_downs_grp", .ins = &.{ "ACT", "WQ", "WS", "WB", "TEXP", "TCNT", "TPICK" }, .outs = &.{"DN"}, .source = kernelSource("plugins/inkling2/kernels/downs_grp.metal"), .header = bank_header };
pub var relattn_kernel: Kernel = .{ .name = "msv_inkling2_relattn", .ins = &.{ "q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale" }, .outs = &.{"out"}, .source = kernelSource("plugins/inkling2/kernels/relattn.metal"), .contiguous = false };
pub var router_kernel: Kernel = .{ .name = "msv_inkling2_router", .ins = &.{ "X", "RT" }, .outs = &.{"SC"}, .source = kernelSource("plugins/inkling2/kernels/router.metal") };
pub var relattn_nax_kernel: Kernel = .{ .name = "msv_inkling2_relattn_nax", .ins = &.{ "q", "k", "v", "rel", "rproj", "sinks", "sizes", "scale" }, .outs = &.{"out"}, .source = kernelSource("plugins/inkling2/kernels/relattn_nax.metal"), .header = kernelSource("kernels/common/mpp_tensor_ops.metal"), .contiguous = false };
pub var relsplit_nax_kernel: Kernel = .{ .name = "msv_inkling2_relsplit_nax", .ins = &.{ "q", "k0", "k1", "k2", "k3", "v0", "v1", "v2", "v3", "kb", "vb", "rel", "sinks", "GATE", "TMETA", "sizes", "params" }, .outs = &.{ "PART", "ML", "OUT" }, .source = kernelSource("plugins/inkling2/kernels/relsplit_nax.metal"), .header = kernelSource("kernels/common/mpp_tensor_ops.metal"), .contiguous = false };
pub var relsplit_lanes_kernel: Kernel = .{ .name = "msv_inkling2_relsplit_lanes", .ins = &.{ "q", "k0", "k1", "k2", "k3", "v0", "v1", "v2", "v3", "kb", "vb", "rel", "sinks", "GATE", "TMETA", "sizes", "params" }, .outs = &.{ "PART", "ML", "OUT" }, .source = kernelSource("plugins/inkling2/kernels/relsplit_lanes.metal"), .header = lanes_header, .contiguous = false };
pub var relsplit_kernel: Kernel = .{ .name = "msv_inkling2_relsplit", .ins = &.{ "q", "k", "v", "kb", "vb", "rel", "sinks", "sizes", "scale" }, .outs = &.{ "PART", "ML" }, .source = kernelSource("plugins/inkling2/kernels/relsplit.metal"), .contiguous = false };
pub var relcomb_kernel: Kernel = .{ .name = "msv_inkling2_relcomb", .ins = &.{ "PART", "ML", "GATE" }, .outs = &.{"OUT"}, .source = kernelSource("plugins/inkling2/kernels/relcomb.metal") };
/// The Markov kernels' argmax order.
const markov_header = kernelSource("plugins/inkling2/kernels/markov_header.metal");
pub var markov_rnn_kernel: Kernel = .{ .name = "msv_inkling2_markov_rnn", .ins = &.{ "W", "B", "ST", "PREV", "ROWS", "NEW", "TOK", "AT" }, .outs = &.{"P"}, .source = kernelSource("plugins/inkling2/kernels/markov_rnn.metal") };
pub var markov_fold_kernel: Kernel = .{ .name = "msv_inkling2_markov_fold", .ins = &.{ "P", "FOLD", "FB", "BASE", "AT", "CAP" }, .outs = &.{ "PM", "PI", "PS" }, .source = kernelSource("plugins/inkling2/kernels/markov_fold.metal"), .header = markov_header };
pub var markov_pick_kernel: Kernel = .{ .name = "msv_inkling2_markov_pick", .ins = &.{ "PM", "PI", "PS", "P", "ST", "CAP" }, .outs = &.{ "TOKN", "CONF", "NST" }, .source = kernelSource("plugins/inkling2/kernels/markov_pick.metal"), .header = markov_header };
pub var markov_fold_batch_kernel: Kernel = .{ .name = "msv_inkling2_markov_fold_batch", .ins = &.{ "P", "FOLD", "FB", "BASE", "AT", "CAP" }, .outs = &.{ "PM", "PI", "PS" }, .source = kernelSource("plugins/inkling2/kernels/markov_fold_batch.metal"), .header = markov_header };
pub var markov_pick_batch_kernel: Kernel = .{ .name = "msv_inkling2_markov_pick_batch", .ins = &.{ "PM", "PI", "PS", "P", "ST", "CAP" }, .outs = &.{ "TOKN", "CONF", "NST" }, .source = kernelSource("plugins/inkling2/kernels/markov_pick_batch.metal"), .header = markov_header };
pub var markov_pick_sample_kernel: Kernel = .{ .name = "msv_inkling2_markov_pick_sample", .ins = &.{ "PM", "PI", "P", "ST", "CAP", "LPARAMS", "SEEDS", "AT" }, .outs = &.{ "TOKN", "CANDS", "PROBS", "DRAW", "NST" }, .source = kernelSource("plugins/inkling2/kernels/markov_pick_sample.metal"), .header = markov_header };
pub var relslots_kernel: Kernel = .{ .name = "msv_inkling2_relslots", .ins = &.{ "q", "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "rel", "sinks", "KEYS", "scale" }, .outs = &.{ "PART", "ML" }, .source = kernelSource("plugins/inkling2/kernels/relslots.metal"), .contiguous = false };
pub var relmask_kernel: Kernel = .{ .name = "msv_inkling2_relmask", .ins = &.{ "rel", "sinks", "sizes" }, .outs = &.{"M"}, .source = kernelSource("plugins/inkling2/kernels/relmask.metal") };
pub var sum_kernel: Kernel = .{ .name = "msv_inkling2_sum", .ins = &.{ "DN", "IDX", "INV", "SC", "WTS", "RES", "ST", "MW", "PR", "PARENT" }, .outs = &.{ "HOUT", "XM" }, .source = kernelSource("plugins/inkling2/kernels/sum.metal") };
pub var few_tail_kernel: Kernel = .{ .name = "msv_inkling2_few_tail", .ins = &.{ "DN", "IDX", "SC", "WTS", "RES", "ST", "MW", "PR", "KP", "VP", "OP", "PARENT" }, .outs = &.{ "HOUT", "XM", "WIN" }, .source = kernelSource("plugins/inkling2/kernels/few_tail.metal") };
pub var state_kernel: Kernel = .{ .name = "msv_inkling2_state", .ins = &.{ "ST", "KP", "VP", "OP", "XM" }, .outs = &.{"NST"}, .source = kernelSource("plugins/inkling2/kernels/state.metal") };
pub var commit_conv_kernel: Kernel = .{ .name = "msv_inkling2_commit_conv", .ins = &.{ "WIN", "IDX" }, .outs = &.{"NST"}, .source = kernelSource("plugins/inkling2/kernels/commit_conv.metal") };
pub var sample_mass_kernel: Kernel = .{ .name = "msv_inkling2_sample_mass", .ins = &.{ "LOGITS", "SUPPRESS", "LPARAMS" }, .outs = &.{"PMASS"}, .source = kernelSource("plugins/inkling2/kernels/sample_tree_mass.metal") };
pub var sample_search_kernel: Kernel = .{ .name = "msv_inkling2_sample_search", .ins = &.{ "LOGITS", "SUPPRESS", "LPARAMS", "PMASS" }, .outs = &.{"VROW"}, .source = kernelSource("plugins/inkling2/kernels/sample_tree_search.metal") };
pub var sample_draw_kernel: Kernel = .{ .name = "msv_inkling2_sample_draw", .ins = &.{ "LOGITS", "SUPPRESS", "LPARAMS", "VROW", "SEEDS", "TMETA", "CANDS", "NEXT_Q", "DRAWS" }, .outs = &.{"SELECTED"}, .source = kernelSource("plugins/inkling2/kernels/sample_tree_draw.metal") };
pub var accept_tree_kernel: Kernel = .{ .name = "msv_inkling2_accept_tree", .ins = &.{ "TOKENS", "PARENT", "SELECTED", "ACCEPT_CAP", "STOPS", "N_STOPS" }, .outs = &.{"RESULT"}, .source = kernelSource("plugins/inkling2/kernels/accept_tree.metal") };
pub const enabled = @import("config.zig").fusedOn;

pub const TArg = union(enum) { dtype: Dtype, int: c_int };
pub const Out = struct { shape: []const c_int, dtype: Dtype };
/// An output a kernel variant never writes.
pub const unused: Out = .{ .shape = &.{1}, .dtype = .float32 };

pub const TArgs = []const struct { [*:0]const u8, TArg };

/// `outs` of `k` over `grid` threads in threadgroups of `tg`, owned by `g`.
pub fn launch(g: *Graph, k: *Kernel, inputs: []const Array, comptime n: usize, outs: [n]Out, grid: c_int, tg: c_int, targs: TArgs) ![n]Array {
    return launchGrid(g, k, inputs, n, outs, .{ grid, 1, 1 }, .{ tg, 1, 1 }, targs);
}

pub fn launchGrid(g: *Graph, k: *Kernel, inputs: []const Array, comptime n: usize, outs: [n]Out, grid: [3]c_int, tg: [3]c_int, targs: TArgs) ![n]Array {
    const kernel = k.built orelse blk: {
        const iv = mlx.mlx_vector_string_new_data(k.ins.ptr, k.ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(k.outs.ptr, k.outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new(k.name, iv, ov, k.source, k.header, k.contiguous, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        k.built = made;
        break :blk made;
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg[0], tg[1], tg[2]));
    for (targs) |t| switch (t[1]) {
        .dtype => |d| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t[0], d)),
        .int => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], v)),
    };
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, g.s));
    if (unproven(k, tg, targs)) |key| if (!mlx.errorPending()) {
        _ = mlx.mlx_eval(ov);
        if (mlx.takeErrorIf("maximum allowed threads per threadgroup")) return error.ThreadLimit;
        // A full table re-proves: correct, one evaluation a launch.
        if (!mlx.errorPending() and probed_n < probed.len) {
            probed[probed_n] = key;
            probed_n += 1;
        }
    };
    var res: [n]Array = undefined;
    for (&res, 0..) |*r, i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, ov, i));
        r.* = try g.adopt(o);
    }
    return res;
}

fn floating(dt: Dtype) bool {
    return dt == .bfloat16 or dt == .float16 or dt == .float32;
}

/// The most threads a stream kernel's threadgroup takes. An M1 or M2 compiles a pipeline for what its registers
/// allow, under 768 for some instantiations: a launch one refuses lowers this for the process (`narrow`).
pub var stream_threads: u32 = 768;

/// The threadgroup of a kernel that sums over the stream: SIMD groups of 32 that tile the width, as few groups as
/// `stream_threads` allow (every group redoes the sums).
pub fn streamGroup(hidden: u32) ?c_int {
    var tg: u32 = stream_threads;
    while (tg >= 32) : (tg -= 32) if (hidden % tg == 0) return @intCast(tg);
    return null;
}

/// After a refused stream launch: whether a narrower threadgroup is left to try.
fn narrow(hidden: u32) bool {
    const now = streamGroup(hidden) orelse return false;
    if (now <= 32) return false;
    stream_threads = @intCast(now - 32);
    return streamGroup(hidden) != null;
}

/// Threads a group every pipeline runs; a wider launch is evaluated at each instantiation's first dispatch, where a
/// pipeline that refuses it says so.
const sure_threads = 256;
var probed: [2048]u64 = @splat(0);
var probed_n: usize = 0;
/// The threadgroup past which a launch is proven; a test lowers it to count every instantiation.
pub var prove_past: c_int = sure_threads;

/// The instantiations proven so far. One a launch's template arguments must not mint per shape: every prompt length
/// would be a new one, and past the table each launch pays an evaluation.
pub fn instantiations() usize {
    return probed_n;
}

/// Whether this instantiation's first dispatch still has to show its pipeline takes the launch.
fn unproven(k: *const Kernel, tg: [3]c_int, targs: TArgs) ?u64 {
    if (tg[0] * tg[1] * tg[2] <= prove_past) return null;
    var h = std.hash.Wyhash.init(@intFromPtr(k));
    h.update(std.mem.asBytes(&tg));
    for (targs) |t| switch (t[1]) {
        .dtype => |d| h.update(std.mem.asBytes(&d)),
        .int => |v| h.update(std.mem.asBytes(&v)),
    };
    const key = h.final() | 1;
    return if (std.mem.indexOfScalar(u64, probed[0..probed_n], key) == null) key else null;
}

/// Whether the kernels serve a one-row visit of `L` on the stream `h`.
pub fn runs(g: *const Graph, L: *const Layer, cfg: *const Config, h: Array) bool {
    if (!mlx.streamIsGpu(g.s) or !enabled()) return false;
    // The top-k kernel's threadgroup is the experts.
    if (streamGroup(cfg.hidden_size) == null or cfg.num_experts > sure_threads or cfg.top_k > cfg.num_experts) return false;
    const kw = if (L.k_conv == .dynamic) L.k_conv.dynamic.static else L.k_conv.static;
    const vw = if (L.v_conv == .dynamic) L.v_conv.dynamic.static else L.v_conv.static;
    for ([_]Array{ h, L.attn_conv, L.mlp_conv, L.post_norm, L.in_norm[0], L.in_norm[1], L.in_norm[2], L.q_scale, L.rel_scale, kw, vw }) |x|
        if (!floating(ops.dtype(x))) return false;
    if (ops.dtype(kw) != ops.dtype(vw) or ops.dtype(L.q_scale) != ops.dtype(L.rel_scale)) return false;
    if (L.gates) |gt| if (!floating(ops.dtype(gt.down))) return false;
    return true;
}

/// A later loop pass's inputs to `pre.metal`: the last pass's bank and normed output, the layer's message queries
/// and its gates' dtype.
const Messaged = struct { bank: Array, partial: Array, query: Array, gt: Dtype };

/// `pre.metal` on the stream `h` `[1, S, D]`: its outputs in order.
fn streams(g: *Graph, cfg: *const Config, consts: ?*const layers.Consts, h: Array, blocks: ?Array, query: Array, nw: [3]Array, inputs: c_int, msg: ?Messaged) ![5]Array {
    const d: c_int = @intCast(cfg.hidden_size);
    const s = ops.shape(h)[1];
    const dt = ops.dtype(h);
    const q: c_int = if (query.ctx != null) ops.shape(query)[1] else 0;
    const eps = if (consts != null and consts.?.eps.ctx != null) consts.?.eps else try g.fromSlice(f32, &.{cfg.rms_norm_eps}, &.{1});
    const m: Messaged = msg orelse .{ .bank = h, .partial = h, .query = h, .gt = .float32 };
    const on = msg != null;
    while (true) {
        const tg = streamGroup(cfg.hidden_size) orelse return error.ThreadLimit;
        return launchGrid(g, &pre_kernel, &.{ h, blocks orelse h, if (q > 0) query else h, nw[0], nw[1], nw[2], m.bank, m.partial, m.query, eps }, 5, .{
            if (on) unused else .{ .shape = &.{ inputs, s, d }, .dtype = dt },
            .{ .shape = &.{ 1, s, d }, .dtype = .float32 },
            if (on) .{ .shape = &.{ 3, s, d }, .dtype = .float32 } else unused,
            if (on) .{ .shape = &.{ 4, s, d }, .dtype = .float32 } else unused,
            if (on) .{ .shape = &.{ 3, s, 2 * d }, .dtype = m.gt } else unused,
        }, .{ d, s, 1 }, .{ tg, 1, 1 }, &.{
            .{ "T", .{ .dtype = dt } },                                    .{ "GT", .{ .dtype = m.gt } },
            .{ "TG", .{ .int = tg } },                                     .{ "D", .{ .int = d } },
            .{ "Q", .{ .int = q } },                                       .{ "J", .{ .int = inputs } },
            .{ "NB", .{ .int = if (blocks) |b| ops.shape(b)[2] else 0 } }, .{ "MSG", .{ .int = @intFromBool(on) } },
            .{ "NBK", .{ .int = if (on) ops.shape(m.bank)[2] else 0 } },
        }) catch |e| if (e == error.ThreadLimit and narrow(cfg.hidden_size)) continue else return e;
    }
}

/// Gates `from .. from + n` of a looped layer over the streams `a` `[n, S, D]` and their messages `ms` (its first `n`
/// streams), from each gate's state `st` `[n, S, rank]`: `[n, S, D]` f32. A few rows' threads take the forget and
/// message projections themselves (MLX's f32 matmul at two rows is several launches' latency); a prompt's rows take
/// them from two matmuls.
pub fn gate(g: *Graph, gates: *const layers.Gates, a: Array, ms: Array, st: Array, from: c_int, n: c_int) !Array {
    const sh = ops.shape(gates.forget);
    const s = ops.shape(st)[1];
    var forget = try g.sliceAxis(gates.forget, 0, from, from + n);
    var message = try g.sliceAxis(gates.message, 0, from, from + n);
    const projected = s > packed_rows;
    if (projected) {
        const st32 = try g.astype(st, .float32);
        forget = try g.matmul(st32, try g.transpose(try g.astype(forget, .float32), &.{ 0, 2, 1 }));
        message = try g.matmul(st32, try g.transpose(try g.astype(message, .float32), &.{ 0, 2, 1 }));
    }
    const o = try launch(g, &gate_kernel, &.{ a, ms, st, forget, message }, 1, .{.{ .shape = &.{ n, s, sh[1] }, .dtype = .float32 }}, n * s * sh[1], 64, &.{
        .{ "D", .{ .int = sh[1] } }, .{ "RANK", .{ .int = sh[2] } }, .{ "PROJ", .{ .int = @intFromBool(projected) } },
    });
    return o[0];
}

/// The threads a stream kernel adds to a row for the router: a SIMD group per expert, in whole threadgroups.
fn routerThreads(cfg: *const Config, tg: c_int) c_int {
    return @divTrunc(@as(c_int, @intCast(cfg.num_experts)) * 32 + tg - 1, tg) * tg;
}

/// The `n` streams `y` `[n, S, D]` f32 under their weighted RMS norms, `[n, S, D]` in `dt`; with a `router`, its
/// scores `[S, E]` f32 over the first.
pub fn norm(g: *Graph, cfg: *const Config, y: Array, nw: [3]Array, n: c_int, dt: Dtype, router: ?Array) ![2]Array {
    return normWithConsts(g, cfg, null, y, nw, n, dt, router);
}

fn normWithConsts(g: *Graph, cfg: *const Config, consts: ?*const layers.Consts, y: Array, nw: [3]Array, n: c_int, dt: Dtype, router: ?Array) ![2]Array {
    const d: c_int = @intCast(cfg.hidden_size);
    const s = ops.shape(y)[1];
    const experts_n: c_int = if (router != null) @intCast(cfg.num_experts) else 0;
    const eps = if (consts != null and consts.?.eps.ctx != null) consts.?.eps else try g.fromSlice(f32, &.{cfg.rms_norm_eps}, &.{1});
    while (true) {
        const tg = streamGroup(cfg.hidden_size) orelse return error.ThreadLimit;
        return launchGrid(g, &norm_kernel, &.{ y, nw[0], nw[1], nw[2], router orelse y, eps }, 2, .{
            .{ .shape = &.{ n, s, d }, .dtype = dt },
            if (router != null) .{ .shape = &.{ s, experts_n }, .dtype = .float32 } else unused,
        }, .{ d + if (router != null) routerThreads(cfg, tg) else 0, s, 1 }, .{ tg, 1, 1 }, &.{
            .{ "T", .{ .dtype = dt } }, .{ "TG", .{ .int = tg } }, .{ "D", .{ .int = d } }, .{ "J", .{ .int = n } }, .{ "E", .{ .int = experts_n } },
        }) catch |e| if (e == error.ThreadLimit and narrow(cfg.hidden_size)) continue else return e;
    }
}

/// A visit's projections with what its second half reads: the stream's unweighted norm `[1, S, D]`, and on a later
/// loop pass the visit's messages `[4, S, D]` f32.
pub const Fore = struct { proj: [6]Array, dyn: ?[2]Array, normed: Array, messages: ?Array };

/// Gates `from .. from + n`'s down projections of their inputs `x` `[n, S, 2 hidden]`, `[n, S, rank]`. From 16 rows a
/// matmul a gate over the gate's own rows: MLX's batched matmul leaves its few-row kernel there for a slow general one.
fn gateDowns(g: *Graph, x: Array, gates: *const layers.Gates, from: c_int, n: c_int) !Array {
    const sh = ops.shape(x);
    if (sh[1] < 16) return g.matmul(x, try layers.downs(g, gates, from, n));
    const rank = ops.shape(gates.down)[1];
    var parts: [3]Array = undefined;
    for (parts[0..@intCast(n)], 0..) |*part, j| {
        const at: c_int = @intCast(j);
        const w = try g.reshape(try g.sliceAxis(gates.down, 0, from + at, from + at + 1), &.{ rank, sh[2] });
        const y = try g.matmul(try g.reshape(try g.sliceAxis(x, 0, at, at + 1), &.{ sh[1], sh[2] }), try g.transpose(w, &.{ 1, 0 }));
        part.* = try g.reshape(y, &.{ 1, sh[1], rank });
    }
    return if (n == 1) parts[0] else g.concat(parts[0..@intCast(n)], 0);
}

/// `layers.project`. `bank` is the last pass's bank and normed output on a later loop pass.
pub fn pre(g: *Graph, L: *const Layer, w: *const Body, h: Array, blocks: ?Array, bank: ?[2]Array) !Fore {
    const cfg = &w.cfg;
    const dt = ops.dtype(h);
    var x: Array = undefined;
    var normed: Array = undefined;
    var messages: ?Array = null;
    if (bank) |bk| {
        const gates = &(L.gates orelse return error.Inkling2NoMessageGate);
        const o = try streams(g, cfg, &w.consts, h, blocks, L.attn_query, L.in_norm, 3, .{ .bank = bk[0], .partial = bk[1], .query = L.message_query, .gt = ops.dtype(gates.down) });
        const st = try gateDowns(g, o[4], gates, 0, 3);
        x = (try normWithConsts(g, cfg, &w.consts, try gate(g, gates, o[2], o[3], st, 0, 3), L.in_norm, 3, dt, null))[0];
        normed = o[1];
        messages = o[3];
    } else {
        const o = try streams(g, cfg, &w.consts, h, blocks, L.attn_query, L.in_norm, if (L.global) 3 else 1, null);
        x = o[0];
        normed = o[1];
    }
    var qkv: [3]Array = @splat(x);
    if (ops.shape(x)[0] == 3) for (&qkv, 0..) |*r, i| {
        r.* = try g.sliceAxis(x, 0, @intCast(i), @intCast(i + 1));
    };
    const ps = try layers.projections(g, L, qkv);
    return .{ .proj = ps.proj, .dyn = ps.dyn, .normed = normed, .messages = messages };
}

/// The final stream: `layers.finalInput` under the final norm; null where the kernel does not run.
pub fn finalStream(g: *Graph, w: *const Body, h: Array, blocks: ?Array, norm_weight: Array) !?Array {
    if (!mlx.streamIsGpu(g.s) or !enabled() or streamGroup(w.cfg.hidden_size) == null) return null;
    if (!floating(ops.dtype(h)) or !floating(ops.dtype(norm_weight))) return null;
    return (try streams(g, &w.cfg, &w.consts, h, blocks, w.final_query, @splat(norm_weight), 1, null))[0];
}

/// What the attention read takes: q (clipped; `[1, H, 1, hd]` log-scaled at one row, `[1, S, H, hd]` for more), the
/// k and v rows the cache appends `[1, Hkv, S, hd]`, the output gate's sigmoid `[S * H * hd]` f32, and at one row its
/// additive bias `[1, H, 1, keys]`, for more the clipped relative states `[1, S, H, d_rel]`.
pub const Prep = struct { q: Array, k: Array, v: Array, gate: Array, bias: Array, rel: Array };

/// The keys a one-row read of `L` sees once its row makes the sequence `total` long.
pub fn keysSeen(L: *const Layer, cfg: *const Config, total: c_int) c_int {
    return band.view(if (L.global) 0 else cfg.sliding_window, @intCast(cfg.head_dim), total, 1).kv_len;
}

/// `layers.prepare` as one launch. One row takes its log-scaling factor `tau` and the count of `keys` it will read,
/// for its bias row; more rows leave both to the read. Null where the projections are not one float dtype.
pub fn prep(g: *Graph, L: *const Layer, cfg: *const Config, proj: *const [6]Array, taps: ?[2]Array, st: Array, tau: f32, keys: c_int) !?Prep {
    return prepWith(g, L, cfg, proj, taps, st, tau, keys, null);
}

pub fn prepWith(g: *Graph, L: *const Layer, cfg: *const Config, proj: *const [6]Array, taps: ?[2]Array, st: Array, tau: f32, keys: c_int, parent: ?Array) !?Prep {
    return prepWithConsts(g, L, cfg, null, proj, taps, st, tau, keys, parent);
}

fn prepWithConsts(g: *Graph, L: *const Layer, cfg: *const Config, consts: ?*const layers.Consts, proj: *const [6]Array, taps: ?[2]Array, st: Array, tau: f32, keys: c_int, parent: ?Array) !?Prep {
    const dt = ops.dtype(proj[0]);
    for (proj) |x| if (ops.dtype(x) != dt) return null;
    // A head is one threadgroup.
    if (!floating(dt) or cfg.head_dim > sure_threads) return null;
    const dynamic = L.k_conv == .dynamic;
    const kw = if (dynamic) L.k_conv.dynamic.static else L.k_conv.static;
    const vw = if (dynamic) L.v_conv.dynamic.static else L.v_conv.static;
    const dyn: [2]Array = taps orelse .{ proj[4], proj[5] };
    if (ops.dtype(dyn[0]) != dt or ops.dtype(dyn[1]) != dt) return null;

    const s = ops.shape(proj[0])[1];
    const wide = s > 1;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const d_rel: c_int = @intCast(cfg.d_rel);
    const kt: c_int = @intCast(cfg.sconv_kernel);
    const kv_dim = kv_heads * hd;
    const conv_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(if (dynamic) 2 * kt + 1 else kt + 1)));
    const cached_pr = if (wide and tau == 1.0 and keys == 0 and consts != null)
        (if (dynamic) consts.?.prep_pr_dyn else consts.?.prep_pr_static)
    else
        Array{};
    const pr = if (cached_pr.ctx != null) cached_pr else try g.fromSlice(f32, &.{ cfg.qk_clip, cfg.rel_clip, cfg.rms_norm_eps, tau, conv_scale, @floatFromInt(keys) }, &.{6});
    const par = parent orelse (if (consts != null and consts.?.zero_i32.ctx != null) consts.?.zero_i32 else try g.zeros(&.{1}, .int32));
    const o = try launchGrid(g, &prep_kernel, &.{ proj[0], proj[1], proj[2], proj[4], proj[5], dyn[0], dyn[1], st, kw, vw, L.q_scale, L.rel_scale, L.rel_proj_rev, pr, par }, 6, .{
        .{ .shape = if (wide) &.{ 1, s, heads, hd } else &.{ 1, heads, 1, hd }, .dtype = dt },
        .{ .shape = &.{s * heads * hd}, .dtype = .float32 },
        .{ .shape = &.{ 1, kv_heads, s, hd }, .dtype = dt },
        .{ .shape = &.{ 1, kv_heads, s, hd }, .dtype = dt },
        if (wide) unused else .{ .shape = &.{ 1, heads, 1, keys }, .dtype = dt },
        if (wide) .{ .shape = &.{ 1, s, heads, d_rel }, .dtype = dt } else unused,
    }, .{ heads * hd + 2 * kv_dim + heads * (if (wide) d_rel else keys), s, 1 }, .{ hd, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },                  .{ "WT", .{ .dtype = ops.dtype(kw) } },                .{ "S", .{ .dtype = ops.dtype(L.q_scale) } },
        .{ "H", .{ .int = heads } },                 .{ "HD", .{ .int = hd } },                             .{ "HKV", .{ .int = kv_heads } },
        .{ "DREL", .{ .int = d_rel } },              .{ "KT", .{ .int = kt } },                             .{ "DYN", .{ .int = @intFromBool(dynamic) } },
        .{ "WIDE", .{ .int = @intFromBool(wide) } }, .{ "TREE", .{ .int = @intFromBool(parent != null) } },
    });
    return .{ .q = o[0], .gate = o[1], .k = o[2], .v = o[3], .bias = o[4], .rel = o[5] };
}

/// `x` (any shape) times the output gate's sigmoid element by element, `[1, S, H * hd]` in its dtype.
/// How `oproj` reads an output projection: an affine weight at a width the bank reader takes (FMT 3) or a dense one in
/// the activation's dtype (FMT 4); null for any other (an outside or rotated weight, a bias), which keeps `gated` and
/// `projectRows`.
pub const OFormat = struct { fmt: c_int, bits: u32, gs: u32, k: c_int };

pub fn oFormat(o: *const layers.Linear, dt: Dtype) ?OFormat {
    if (o.ext != null or o.rht != null or o.ternary or o.bias.ctx != null or !floating(dt)) return null;
    const sh = ops.shape(o.w);
    if (sh.len != 2) return null;
    if (o.s.ctx == null) {
        if (ops.dtype(o.w) != dt or @rem(sh[1], 16) != 0) return null;
        return .{ .fmt = 4, .bits = 16, .gs = 16, .k = sh[1] };
    }
    if (o.mode != .affine or o.b.ctx == null or !affineWidth(o.bits, o.group_size)) return null;
    const k: c_int = @intCast(@divExact(@as(u32, @intCast(sh[1])) * 32, o.bits));
    return .{ .fmt = 3, .bits = o.bits, .gs = o.group_size, .k = k };
}

/// One row's output projection of its attention read `att` `[H * HD]` under the output gate `gate_sigmoid` (f32):
/// `projectRows(gated(att, gate))` as one launch, the gated read rounded to the activation's dtype as `gated` leaves it.
pub fn oGated(g: *Graph, o: *const layers.Linear, att: Array, gate_sigmoid: Array) !Array {
    const dt = ops.dtype(att);
    const f = oFormat(o, dt) orelse return error.Inkling2OutputLayout;
    const n = ops.shape(o.w)[0];
    const affine = f.fmt == 3;
    const y = try launchGrid(g, &oproj_kernel, &.{ att, gate_sigmoid, o.w, if (affine) o.s else o.w, if (affine) o.b else o.w }, 1, .{.{ .shape = &.{ 1, 1, n }, .dtype = dt }}, .{ 32, n, 1 }, .{ 32, 4, 1 }, &.{
        .{ "T", .{ .dtype = dt } },            .{ "S", .{ .dtype = if (affine) ops.dtype(o.s) else dt } }, .{ "K", .{ .int = f.k } },
        .{ "GS", .{ .int = @intCast(f.gs) } }, .{ "BITS", .{ .int = @intCast(f.bits) } },                  .{ "FMT", .{ .int = f.fmt } },
    });
    return y[0];
}

pub fn gated(g: *Graph, x: Array, gate_sigmoid: Array, rows: c_int) !Array {
    const n: c_int = @intCast(mlx.mlx_array_size(gate_sigmoid));
    const o = try launch(g, &gated_kernel, &.{ x, gate_sigmoid }, 1, .{.{ .shape = &.{ 1, rows, @divExact(n, rows) }, .dtype = ops.dtype(x) }}, n, 64, &.{.{ "T", .{ .dtype = ops.dtype(x) } }});
    return o[0];
}

/// A few rows' additive mask `[1, H, S, keys + 1]` in the bias's dtype: each row's `bias` `[1, S, H, R]` where a key
/// is visible (the rows are the last S of the `keys`; `window` > 0 keeps the keys fewer than it behind a row), -inf
/// elsewhere, its sink `[1, S, H]` f32 in the last column; without `sink` that column is left out.
pub fn relMask(g: *Graph, bias: Array, sinks: Array, keys: c_int, window: c_int, sink: bool) !Array {
    const sh = ops.shape(bias);
    const dt = ops.dtype(bias);
    const extra: c_int = @intFromBool(sink);
    const mask = try launchGrid(g, &relmask_kernel, &.{ bias, sinks, try g.fromSlice(i32, &.{ sh[1], keys }, &.{2}) }, 1, .{.{ .shape = &.{ 1, sh[2], sh[1], keys + extra }, .dtype = dt }}, .{ @divTrunc(keys + extra + 31, 32) * 32, sh[1], sh[2] }, .{ 32, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } }, .{ "HQ", .{ .int = sh[2] } }, .{ "R", .{ .int = sh[3] } }, .{ "WINDOW", .{ .int = window } }, .{ "SINK", .{ .int = extra } },
    });
    return mask[0];
}

/// The rows a read hands MLX's attention under a built mask: the flash kernel walks the keys 16 at a time on one
/// threadgroup, which is a prompt's throughput and a draft verify's latency.
pub const small_rows = 32;

/// `layers.wideRead`: the rows `q` `[1, S, H, hd]` read `k` and `v` (the view after the append, the rows at its last
/// S keys) under the bias of their clipped relative `states` `[1, S, H, d_rel]` and their `sinks` `[1, S, H]`:
/// `[1, S, H, hd]`. `offset` is the first row's position. One flash launch with no mask built, or (at most
/// `small_rows` rows, unless `flash`) a mask launch and MLX's attention.
pub fn relative(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, k: Array, v: Array, offset: c_int, flash: bool) !Array {
    if (ops.shape(q_rows)[1] > small_rows) if (try relativeNax(g, L, cfg, q_rows, rel_states, sinks, k, v, offset)) |o| return o;
    const dt = ops.dtype(q_rows);
    const hd: c_int = @intCast(cfg.head_dim);
    if (ops.dtype(k) != dt or ops.dtype(v) != dt or @rem(hd, 8) != 0) return error.Inkling2StepDtype;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const s = ops.shape(q_rows)[1];
    const keys = ops.shape(k)[2];
    const in = try readInputs(g, L, cfg, q_rows, rel_states, sinks, offset);
    const q = in.q;
    const bias = in.bias;
    const sink_rows = in.sinks;
    const window: c_int = if (L.global) 0 else @intCast(cfg.sliding_window);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    if (s <= small_rows and !flash) {
        const zero = try g.zeros(&.{ 1, @intCast(cfg.num_key_value_heads), 1, hd }, dt);
        const read = try g.sdpa(q, try g.concat(&.{ k, zero }, 2), try g.concat(&.{ v, zero }, 2), scale, .{ .array = try relMask(g, bias, sink_rows, keys, window, true) });
        return g.transpose(read, &.{ 0, 2, 1, 3 });
    }
    const o = try launchGrid(g, &relattn_kernel, &.{
        q, k, v, bias, sink_rows, try g.zeros(&.{1}, .int32), try g.fromSlice(i32, &.{ s, keys }, &.{2}), try g.scalar(scale, .float32),
    }, 1, .{.{ .shape = &.{ 1, s, heads, hd }, .dtype = dt }}, .{ @divTrunc(s + 31, 32) * 128, heads, 1 }, .{ 128, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },                     .{ "D", .{ .int = hd } },
        .{ "HQ", .{ .int = heads } },                   .{ "HKV", .{ .int = @intCast(cfg.num_key_value_heads) } },
        .{ "R", .{ .int = ops.shape(L.rel_proj)[1] } }, .{ "WINDOW", .{ .int = window } },
        .{ "BQ", .{ .int = 32 } },                      .{ "TG", .{ .int = 128 } },
    });
    return o[0];
}

fn relProjT(g: *Graph, L: *const Layer, dt: Dtype) !Array {
    if (L.rel_proj_t.ctx != null and ops.dtype(L.rel_proj_t) == dt) return L.rel_proj_t;
    return g.contiguous(try g.astype(try g.transpose(L.rel_proj, &.{ 1, 0 }), dt));
}

/// `relative`'s read of a prompt's rows on the NAX tensor unit (`relattn_nax.metal`): `[1, S, H, hd]`, null where it does
/// not serve the read (a GPU without NAX, a dtype but half or bfloat16, a head width off 16).
pub fn relativeNax(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, k: Array, v: Array, offset: c_int) !?Array {
    const dt = ops.dtype(q_rows);
    const hd: c_int = @intCast(cfg.head_dim);
    if ((dt != .bfloat16 and dt != .float16) or ops.dtype(k) != dt or ops.dtype(v) != dt or @rem(hd, 16) != 0) return null;
    if (!mlx.streamIsGpu(g.s) or !enabled() or !@import("../../mlx/device.zig").naxAvailable()) return null;
    const s = ops.shape(q_rows)[1];
    var q = try g.transpose(q_rows, &.{ 0, 2, 1, 3 });
    var states = rel_states;
    if (L.global and offset + s > @as(c_int, @intCast(cfg.log_n_floor))) {
        const tau = try g.adopt(try rel.inklingLogScaleTau(g.s, offset, s, cfg.log_n_floor, cfg.log_alpha));
        q = try g.astype(try g.mul(try g.astype(q, .float32), tau), dt);
        states = try g.astype(try g.mul(try g.astype(states, .float32), try g.reshape(tau, &.{ 1, s, 1, 1 })), dt);
    }
    const rproj = try relProjT(g, L, dt);
    const sink_rows = try g.contiguous(try g.astype(sinks, .float32));
    return try naxRead(g, q, k, v, try g.contiguous(states), rproj, sink_rows, if (L.global) 0 else @intCast(cfg.sliding_window));
}

/// `relattn_nax.metal` over prepared inputs: `q` `[1, H, S, hd]`, `k` and `v` `[1, Hkv, keys, hd]`, `rel_states`
/// `[1, S, H, d_rel]`, `rproj` `[R, d_rel]` and `sinks` `[1, S, H]` f32; `window` 0 on a global layer. `[1, S, H, hd]`.
pub fn naxRead(g: *Graph, q: Array, k: Array, v: Array, rel_states: Array, rproj: Array, sinks: Array, window: c_int) !Array {
    const dt = ops.dtype(q);
    const sh = ops.shape(q);
    const heads = sh[1];
    const s = sh[2];
    const hd = sh[3];
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const o = try launchGrid(g, &relattn_nax_kernel, &.{
        q, k, v, rel_states, rproj, sinks, try g.fromSlice(i32, &.{ s, ops.shape(k)[2] }, &.{2}), try g.scalar(scale, .float32),
    }, 1, .{.{ .shape = &.{ 1, s, heads, hd }, .dtype = dt }}, .{ @divTrunc(s + 63, 64) * 128, heads, 1 }, .{ 128, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },                .{ "D", .{ .int = hd } },
        .{ "HQ", .{ .int = heads } },              .{ "HKV", .{ .int = ops.shape(k)[1] } },
        .{ "R", .{ .int = ops.shape(rproj)[0] } }, .{ "DREL", .{ .int = ops.shape(rproj)[1] } },
        .{ "WINDOW", .{ .int = window } },
    });
    return o[0];
}

/// The fold kernel's threads a threadgroup and vocab columns a thread.
const markov_tg = 256;
const markov_cols = 8;

/// Whether `markovStep` serves a drafter of `rank` whose fold is `fold` `[rank, V]` and RNN weight `rnn` on `g`'s
/// stream: half dtypes, the fold read 8 columns at a time.
pub fn markovRuns(g: *const Graph, rnn: Array, fold: Array, rank: c_int) bool {
    const dt = ops.dtype(fold);
    if (!mlx.streamIsGpu(g.s) or !enabled() or (dt != .bfloat16 and dt != .float16) or ops.dtype(rnn) != dt) return false;
    return @rem(ops.shape(fold)[1], markov_cols) == 0 and @rem(3 * rank, 8) == 0;
}

/// One step of the drafter's Markov chain as three launches: the RNN's projection of [state | the previous token's
/// row | block row `at` | the newest mix], the latent's bias over the vocab with each threadgroup's share of the pick,
/// and the pick with the next state. `base` `[1, block - 1, V]` f32 is the block rows' head logits, `softcap` the
/// head's (the probability reads the capped logits; 0 = none).
pub const MarkovStep = struct { token: Array, confidence: Array, state: Array };

pub fn markovStep(g: *Graph, rnn: Array, rnn_bias: Array, state: Array, prev_rows: Array, rows: Array, newest: Array, token: Array, fold: Array, fold_bias: Array, base: Array, at: c_int, vocab: c_int, softcap: f32) !MarkovStep {
    const step = try g.fromSlice(i32, &.{at}, &.{1});
    const cap = try g.fromSlice(f32, &.{softcap}, &.{1});
    const p = try markovRnn(g, rnn, rnn_bias, state, prev_rows, rows, newest, token, step);
    const o = try markovPick(g, try markovFold(g, p, fold, fold_bias, base, step, vocab, cap), p, state, cap);
    return .{ .token = o[0], .confidence = o[1], .state = o[2] };
}

/// `markov_rnn.metal`: the step's RNN projection `[3 R]` f32.
pub fn markovRnn(g: *Graph, rnn: Array, rnn_bias: Array, state: Array, prev_rows: Array, rows: Array, newest: Array, token: Array, step: Array) !Array {
    const rank = @divExact(ops.shape(rnn)[0], 3);
    const o = try launch(g, &markov_rnn_kernel, &.{ rnn, rnn_bias, state, prev_rows, rows, newest, token, step }, 1, .{.{ .shape = &.{3 * rank}, .dtype = .float32 }}, 32 * 3 * rank, 256, &.{
        .{ "T", .{ .dtype = ops.dtype(rnn) } }, .{ "R", .{ .int = rank } }, .{ "H", .{ .int = ops.shape(rows)[2] } },
    });
    return o[0];
}

/// `markov_fold.metal`: each fold threadgroup's (max, argmax, sum of exp) over the step's `vocab` logits.
pub fn markovFold(g: *Graph, p: Array, fold: Array, fold_bias: Array, base: Array, step: Array, vocab: c_int, cap: Array) ![3]Array {
    const groups = @divTrunc(vocab + markov_tg * markov_cols - 1, markov_tg * markov_cols);
    return launch(g, &markov_fold_kernel, &.{ p, fold, fold_bias, base, step, cap }, 3, .{
        .{ .shape = &.{groups}, .dtype = .float32 }, .{ .shape = &.{groups}, .dtype = .uint32 }, .{ .shape = &.{groups}, .dtype = .float32 },
    }, groups * markov_tg, markov_tg, &.{
        .{ "T", .{ .dtype = ops.dtype(fold) } }, .{ "R", .{ .int = ops.shape(fold)[0] } }, .{ "TG", .{ .int = markov_tg } },
        .{ "VPT", .{ .int = markov_cols } },     .{ "VOCAB", .{ .int = vocab } },
    });
}

/// `markov_pick.metal`: the step's token `[1]` int32, its probability `[1]` f32 and the next state `[1, 1, R]`.
pub fn markovPick(g: *Graph, part: [3]Array, p: Array, state: Array, cap: Array) ![3]Array {
    const rank = @divExact(ops.shape(p)[0], 3);
    return launch(g, &markov_pick_kernel, &.{ part[0], part[1], part[2], p, state, cap }, 3, .{
        .{ .shape = &.{1}, .dtype = .int32 }, .{ .shape = &.{1}, .dtype = .float32 }, .{ .shape = &.{ 1, 1, rank }, .dtype = ops.dtype(state) },
    }, 256, 256, &.{ .{ "T", .{ .dtype = ops.dtype(state) } }, .{ "R", .{ .int = rank } } });
}

fn foldBatchTile(b: c_int) struct { nb: c_int, vpt: c_int } {
    if (b <= 2) return .{ .nb = 2, .vpt = 4 };
    if (b <= 4) return .{ .nb = 4, .vpt = 4 };
    return .{ .nb = 8, .vpt = 2 };
}

/// `markov_fold_batch.metal`: batched fold over `B` lanes (`P [B, 3R]`, `BASE [B, steps, V]`).
pub fn markovFoldBatch(g: *Graph, p: Array, fold: Array, fold_bias: Array, base: Array, step: Array, vocab: c_int, cap: Array) ![3]Array {
    const b = ops.shape(p)[0];
    const tile = foldBatchTile(b);
    const groups = @divTrunc(vocab + markov_tg * tile.vpt - 1, markov_tg * tile.vpt);
    const bg = @divTrunc(b + tile.nb - 1, tile.nb);
    return launchGrid(g, &markov_fold_batch_kernel, &.{ p, fold, fold_bias, base, step, cap }, 3, .{
        .{ .shape = &.{ b, groups }, .dtype = .float32 },
        .{ .shape = &.{ b, groups }, .dtype = .uint32 },
        .{ .shape = &.{ b, groups }, .dtype = .float32 },
    }, .{ groups * markov_tg, bg, 1 }, .{ markov_tg, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(fold) } }, .{ "R", .{ .int = ops.shape(fold)[0] } }, .{ "TG", .{ .int = markov_tg } },
        .{ "VPT", .{ .int = tile.vpt } },        .{ "VOCAB", .{ .int = vocab } },          .{ "NB", .{ .int = tile.nb } },
        .{ "SG_OUT", .{ .int = 0 } },
    });
}

/// `markov_fold_batch.metal` (`SG_OUT = 1`): emits exact top-4 candidates per simdgroup (`groups * (TG/32) * 4` per lane)
/// so dense common-token BPE blocks retain competing candidates for top-K sampling.
pub fn markovFoldBatchSg(g: *Graph, p: Array, fold: Array, fold_bias: Array, base: Array, step: Array, vocab: c_int, cap: Array) ![3]Array {
    const b = ops.shape(p)[0];
    const tile = foldBatchTile(b);
    const groups = @divTrunc(vocab + markov_tg * tile.vpt - 1, markov_tg * tile.vpt);
    const nt = groups * @divExact(markov_tg, 32) * 4;
    const bg = @divTrunc(b + tile.nb - 1, tile.nb);
    return launchGrid(g, &markov_fold_batch_kernel, &.{ p, fold, fold_bias, base, step, cap }, 3, .{
        .{ .shape = &.{ b, nt }, .dtype = .float32 },
        .{ .shape = &.{ b, nt }, .dtype = .uint32 },
        unused,
    }, .{ groups * markov_tg, bg, 1 }, .{ markov_tg, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(fold) } }, .{ "R", .{ .int = ops.shape(fold)[0] } }, .{ "TG", .{ .int = markov_tg } },
        .{ "VPT", .{ .int = tile.vpt } },        .{ "VOCAB", .{ .int = vocab } },          .{ "NB", .{ .int = tile.nb } },
        .{ "SG_OUT", .{ .int = 1 } },
    });
}

/// `markov_pick_batch.metal`: batched pick over `B` lanes returning `{ token [B], confidence [B], state [B, 1, R] }`.
pub fn markovPickBatch(g: *Graph, part: [3]Array, p: Array, state: Array, cap: Array) ![3]Array {
    const b = ops.shape(state)[0];
    const rank = @divExact(ops.shape(p)[1], 3);
    return launchGrid(g, &markov_pick_batch_kernel, &.{ part[0], part[1], part[2], p, state, cap }, 3, .{
        .{ .shape = &.{b}, .dtype = .int32 }, .{ .shape = &.{b}, .dtype = .float32 }, .{ .shape = &.{ b, 1, rank }, .dtype = ops.dtype(state) },
    }, .{ 256, b, 1 }, .{ 256, 1, 1 }, &.{ .{ "T", .{ .dtype = ops.dtype(state) } }, .{ "R", .{ .int = rank } } });
}

/// Top-K sparse proposal width for GPU-sampled bundled drafter rounds.
pub const markov_sample_k: c_int = 16;

pub const MarkovSampledStep = struct {
    token: Array,
    cands: Array,
    probs: Array,
    draw: Array,
    state: Array,
};

/// `markov_pick_sample.metal`: top-K sparse proposal (with matching `lparams` `{ temp, top_k, top_p, min_p }` filter) +
/// draw + RNN state update over `B` lanes.
pub fn markovPickSample(g: *Graph, pm: Array, pi: Array, p: Array, state: Array, cap: Array, lparams: Array, seeds: Array, step: Array, k: c_int) !MarkovSampledStep {
    const b = ops.shape(state)[0];
    const rank = @divExact(ops.shape(p)[1], 3);
    const o = try launchGrid(g, &markov_pick_sample_kernel, &.{ pm, pi, p, state, cap, lparams, seeds, step }, 5, .{
        .{ .shape = &.{b}, .dtype = .int32 },
        .{ .shape = &.{ b, k }, .dtype = .int32 },
        .{ .shape = &.{ b, k }, .dtype = .float32 },
        .{ .shape = &.{b}, .dtype = .uint8 },
        .{ .shape = &.{ b, 1, rank }, .dtype = ops.dtype(state) },
    }, .{ 256, b, 1 }, .{ 256, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(state) } }, .{ "R", .{ .int = rank } }, .{ "K", .{ .int = k } },
    });
    return .{ .token = o[0], .cands = o[1], .probs = o[2], .draw = o[3], .state = o[4] };
}

/// Batched Markov step for `B` lanes: reads `fold` once per 8 lanes. `state` `[B, 1, R]`, `rows` `[B, steps, H]`,
/// `newest` `[B, 1, H]`, `tokens` `[B]` int32, `base` `[B, steps, V]` f32; returns `{ token [B], confidence [B], state [B, 1, R] }`.
pub fn markovStepBatch(g: *Graph, rnn: Array, rnn_bias: Array, state: Array, prev_rows: Array, rows: Array, newest: Array, tokens: Array, fold: Array, fold_bias: Array, base: Array, at: c_int, vocab: c_int, softcap: f32) !MarkovStep {
    const b = ops.shape(state)[0];
    const rank = @divExact(ops.shape(rnn)[0], 3);
    const step = try g.fromSlice(i32, &.{at}, &.{1});
    const cap = try g.fromSlice(f32, &.{softcap}, &.{1});
    const p = (try launchGrid(g, &markov_rnn_kernel, &.{ rnn, rnn_bias, state, prev_rows, rows, newest, tokens, step }, 1, .{.{ .shape = &.{ b, 3 * rank }, .dtype = .float32 }}, .{ 32 * 3 * rank, b, 1 }, .{ 256, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(rnn) } }, .{ "R", .{ .int = rank } }, .{ "H", .{ .int = ops.shape(rows)[2] } },
    }))[0];
    const part = try markovFoldBatch(g, p, fold, fold_bias, base, step, vocab, cap);
    const o = try markovPickBatch(g, part, p, state, cap);
    return .{ .token = o[0], .confidence = o[1], .state = o[2] };
}

/// Batched sampled Markov step for `B` lanes: shares one `fold` read per tile and emits a top-K sparse proposal for
/// `sampleAndAcceptTree` (`K = markov_sample_k`, `lparams` `[B, 4]` `{ temp, top_k, top_p, min_p }`).
pub fn markovStepBatchSampled(g: *Graph, rnn: Array, rnn_bias: Array, state: Array, prev_rows: Array, rows: Array, newest: Array, tokens: Array, fold: Array, fold_bias: Array, base: Array, at: c_int, vocab: c_int, cap: Array, lparams: Array, seeds: Array) !MarkovSampledStep {
    const b = ops.shape(state)[0];
    const rank = @divExact(ops.shape(rnn)[0], 3);
    const step = try g.fromSlice(i32, &.{at}, &.{1});
    const p = (try launchGrid(g, &markov_rnn_kernel, &.{ rnn, rnn_bias, state, prev_rows, rows, newest, tokens, step }, 1, .{.{ .shape = &.{ b, 3 * rank }, .dtype = .float32 }}, .{ 32 * 3 * rank, b, 1 }, .{ 256, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(rnn) } }, .{ "R", .{ .int = rank } }, .{ "H", .{ .int = ops.shape(rows)[2] } },
    }))[0];
    const part = try markovFoldBatchSg(g, p, fold, fold_bias, base, step, vocab, cap);
    return markovPickSample(g, part[0], part[1], p, state, cap, lparams, seeds, step, markov_sample_k);
}

/// What a read of the rows takes: q `[1, H, S, hd]` (log-scaled on a global layer past the floor), the bias rows
/// `[1, S, H, R]` and the sinks `[1, S, H]` f32.
const ReadInputs = struct { q: Array, bias: Array, sinks: Array };

fn readInputs(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, offset: c_int) !ReadInputs {
    const dt = ops.dtype(q_rows);
    const s = ops.shape(q_rows)[1];
    var q = try g.transpose(q_rows, &.{ 0, 2, 1, 3 });
    var states = rel_states;
    if (L.global and offset + s > @as(c_int, @intCast(cfg.log_n_floor))) {
        // As the reference: q and the relative states times each row's factor, rounded, before the bias projection.
        const tau = try g.adopt(try rel.inklingLogScaleTau(g.s, offset, s, cfg.log_n_floor, cfg.log_alpha));
        q = try g.astype(try g.mul(try g.astype(q, .float32), tau), dt);
        states = try g.astype(try g.mul(try g.astype(states, .float32), try g.reshape(tau, &.{ 1, s, 1, 1 })), dt);
    }
    return .{ .q = q, .bias = try g.matmul(states, try g.astype(L.rel_proj, dt)), .sinks = try g.contiguous(try g.astype(sinks, .float32)) };
}

/// Rows from which a read behind a cache splits over the keys (`splitRead`): below it MLX's attention under a mask is
/// as fast; up to `split_rows`, the kernel's one 8-row fragment.
pub const split_from = 4;
pub const split_rows = 8;
/// Keys a threadgroup of the split read takes.
const split_keys = 128;
/// The most rows a verify's read splits, a fragment at a time.
pub const open_rows = 16;

/// `gated(relative(..))` for a few half-exact rows (bf16, f16): the keys split over threadgroups that each read a K/V
/// tile once for every query head of its KV head, then the splits combined under the output gate `gate_sigmoid`
/// `[S * H * hd]` f32: `[1, S, H * hd]`. `k`, `v` hold every key (`split_from` to `split_rows` rows at their last), or
/// with `own` (a verify's keys and values `[1, Hkv, S, hd]`, which its cache does not hold) the keys before the
/// rows': 2 to `open_rows` rows, read in place, `split_rows` a launch. Null where it does not serve the read.
pub fn splitRead(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, k: Array, v: Array, own: ?[2]Array, offset: c_int, gate_sigmoid: Array) !?Array {
    const dt = ops.dtype(q_rows);
    const s = ops.shape(q_rows)[1];
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const group = @divTrunc(heads, kv_heads);
    if (own) |o| {
        if (s < 2 or s > open_rows or ops.dtype(o[0]) != dt or ops.dtype(o[1]) != dt) return null;
    } else if (s < split_from or s > split_rows) return null;
    if ((dt != .bfloat16 and dt != .float16) or ops.dtype(k) != dt or ops.dtype(v) != dt) return null;
    if (@rem(hd, 8) != 0 or @rem(heads, kv_heads) != 0 or 32 * group > sure_threads) return null;
    const in = try readInputs(g, L, cfg, q_rows, rel_states, sinks, offset);
    const window: c_int = if (L.global) 0 else @intCast(cfg.sliding_window);
    var outs: [open_rows / split_rows]Array = undefined;
    var n: usize = 0;
    var r0: c_int = 0;
    while (r0 < s) : (r0 += split_rows) {
        const r1 = @min(s, r0 + split_rows);
        // A fragment's rows see no key past their last: its read ends there.
        const mine: ?[2]Array = if (own) |o| .{ try g.sliceAxis(o[0], 2, 0, r1), try g.sliceAxis(o[1], 2, 0, r1) } else null;
        const parts = try splitParts(g, try g.sliceAxis(in.q, 2, r0, r1), k, v, mine, try g.sliceAxis(in.bias, 1, r0, r1), try g.sliceAxis(in.sinks, 1, r0, r1), window);
        outs[n] = try splitJoin(g, parts, try g.sliceAxis(gate_sigmoid, 0, r0 * heads * hd, r1 * heads * hd), r1 - r0, dt);
        n += 1;
    }
    return if (n == 1) outs[0] else try g.concat(outs[0..n], 1);
}

/// `relsplit.metal` over q `[1, H, S, hd]`, the keys and values `[1, Hkv, ., hd]` (`k`, `v`, then `own`'s behind
/// them), the bias rows `[1, S, H, R]` and the sinks `[1, S, H]` f32, the rows at the last S keys: each split's
/// unnormalised sums `[splits, S * H, hd]` and (max, sum) `[splits, S * H, 2]`, f32.
pub fn splitParts(g: *Graph, q: Array, k: Array, v: Array, own: ?[2]Array, bias: Array, sinks: Array, window: c_int) ![2]Array {
    const heads = ops.shape(q)[1];
    const s = ops.shape(q)[2];
    const hd = ops.shape(q)[3];
    const kv_heads = ops.shape(k)[1];
    const cached = ops.shape(k)[2];
    const keys = cached + if (own) |o| ops.shape(o[0])[2] else 0;
    const group = @divTrunc(heads, kv_heads);
    // The first row's oldest visible key: a split before it would read nothing.
    const first: c_int = if (window > 0) @max(0, keys - s - window + 1) else 0;
    const splits = @divTrunc(keys - first + split_keys - 1, split_keys);
    const kb, const vb = own orelse .{ k, v };
    return launchGrid(g, &relsplit_kernel, &.{ q, k, v, kb, vb, bias, sinks, try g.fromSlice(i32, &.{ s, keys, first, cached }, &.{4}), try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32) }, 2, .{
        .{ .shape = &.{ splits, s * heads, hd }, .dtype = .float32 },
        .{ .shape = &.{ splits, s * heads, 2 }, .dtype = .float32 },
    }, .{ splits * 32 * group, kv_heads, 1 }, .{ 32 * group, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(q) } },     .{ "D", .{ .int = hd } },
        .{ "HQ", .{ .int = heads } },             .{ "HKV", .{ .int = kv_heads } },
        .{ "R", .{ .int = ops.shape(bias)[3] } }, .{ "WINDOW", .{ .int = window } },
        .{ "KPS", .{ .int = split_keys } },
    });
}

/// `relcomb.metal`: the splits `parts` of `s` rows joined, normalised and times the output gate, `[1, S, H * hd]`
/// in `dt`.
pub fn splitJoin(g: *Graph, parts: [2]Array, gate_sigmoid: Array, s: c_int, dt: Dtype) !Array {
    const rows = ops.shape(parts[0])[1];
    const hd = ops.shape(parts[0])[2];
    const o = try launchGrid(g, &relcomb_kernel, &.{ parts[0], parts[1], gate_sigmoid }, 1, .{.{ .shape = &.{ 1, s, @divExact(rows, s) * hd }, .dtype = dt }}, .{ 32, rows, 1 }, .{ 32, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } }, .{ "D", .{ .int = hd } },
    });
    return o[0];
}

/// Lanes one launch of `relsplit_nax.metal` reads: each lane's K and V take a buffer, and a kernel has 31.
pub const nax_lanes = 4;

/// Keys per split for long-context attention: saturates wide GPUs up to 128 splits solo (bounded per-launch partial
/// memory across multi-lane batches so `splits * here <= 128`).
fn splitKps(most: c_int, here: usize) c_int {
    const max_splits: c_int = @max(@as(c_int, 16), @divTrunc(@as(c_int, 128), @as(c_int, @intCast(here))));
    const base: c_int = if (most <= 1040) 64 else if (most <= 4096) 128 else if (most <= 16384) 256 else 512;
    const floor_kps: c_int = @divTrunc(@divTrunc(most + max_splits - 1, max_splits) + 31, 32) * 32;
    return @max(base, floor_kps);
}

/// Tree metadata for a tree-structured verify forward: `parent` `[NSLOT, RPS]` int32 (per-row parent within slot,
/// -1 for root) and `meta` `[NSLOT * RPS, 2]` int32 (`{ depth, ancestors_mask }` per row).
pub const TreeCtx = struct { parent: Array, meta: Array };

/// `relsplit_nax.metal`: single-pass 2..16-row NAX split-K relative attention (no Q transpose, no tau ops, no sink
/// cast, no second pass for 9..16 rows, direct gated output when `splits == 1`). Null where NAX does not serve it.
pub fn splitReadNax(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, k: Array, v: Array, own: ?[2]Array, offset: c_int, gate_sigmoid: Array) !?Array {
    return splitReadNaxLanesTree(g, L, cfg, q_rows, rel_states, sinks, &.{k}, &.{v}, own, &.{offset}, gate_sigmoid, ops.shape(q_rows)[1], null);
}

/// `splitReadNax` for a verify of several slots: slot `i`'s `s` rows are rows `i * s` on of `q_rows` `[1, N * s, H, hd]`,
/// `rel_states`, `sinks`, `gate_sigmoid` and of `own`, the slots' keys and values stacked `[1, Hkv, N * s, hd]`; they read
/// the slot's cache `ks[i]`, `vs[i]` (the whole view: the kernel finds a sliding window's start) from position `offsets[i]`.
/// `[1, N * s, H * hd]`; null where the kernel does not serve the read.
pub fn splitReadNaxLanes(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, ks: []const Array, vs: []const Array, own: ?[2]Array, offsets: []const c_int, gate_sigmoid: Array, s: c_int) !?Array {
    return splitReadNaxLanesTree(g, L, cfg, q_rows, rel_states, sinks, ks, vs, own, offsets, gate_sigmoid, s, null);
}

pub fn splitReadNaxLanesTree(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, ks: []const Array, vs: []const Array, own: ?[2]Array, offsets: []const c_int, gate_sigmoid: Array, s: c_int, tree_meta: ?Array) !?Array {
    if (!mlx.streamIsGpu(g.s) or !enabled() or !@import("../../mlx/device.zig").naxAvailable()) return null;
    const n = ks.len;
    const dt = ops.dtype(q_rows);
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const group = @divTrunc(heads, kv_heads);
    if (own) |o| {
        if (s < 2 or s > open_rows or ops.dtype(o[0]) != dt or ops.dtype(o[1]) != dt) return null;
    } else if (n != 1 or s < split_from or s > open_rows) return null;
    if ((dt != .bfloat16 and dt != .float16) or ops.dtype(sinks) != dt) return null;
    for (ks, vs) |k, v| if (ops.dtype(k) != dt or ops.dtype(v) != dt) return null;
    if (@rem(hd, 16) != 0 or @rem(heads, kv_heads) != 0 or 32 * group > sure_threads) return null;

    const proj = if (ops.dtype(L.rel_proj) == dt) L.rel_proj else try g.astype(L.rel_proj, dt);
    const bias = try g.matmul(rel_states, proj);
    const window: c_int = if (L.global) 0 else @intCast(cfg.sliding_window);
    const own_rows: c_int = if (own != null) s else 0;
    var keys: [max_slots]c_int = undefined;
    var first: [max_slots]c_int = undefined;
    var most: c_int = 0;
    for (ks, 0..) |k, i| {
        const cached = ops.shape(k)[2];
        // MLX binds small inputs (< 4 KB) in Metal `constant` address space, incompatible with `device` NAX tensors.
        if (cached * kv_heads * hd * 2 < 4096 or s * kv_heads * hd * 2 < 4096) return null;
        keys[i] = cached + own_rows;
        first[i] = if (window > 0) @max(0, keys[i] - s - window + 1) else 0;
        most = @max(most, keys[i] - first[i]);
    }
    const kb, const vb = own orelse .{ ks[0], vs[0] };
    const params = try g.fromSlice(f32, &.{ 1.0 / @sqrt(@as(f32, @floatFromInt(hd))), cfg.log_alpha }, &.{2});
    const tmeta = tree_meta orelse try g.zeros(&.{ 1, 2 }, .int32);
    var outs: [max_slots / nax_lanes]Array = undefined;
    var made: usize = 0;
    var from: usize = 0;
    while (from < n) : (from += nax_lanes) {
        const here = @min(n - from, nax_lanes);
        const kps = splitKps(most, here);
        const splits: c_int = @max(1, @divTrunc(most + kps - 1, kps));
        const direct = splits == 1;
        const rows: c_int = @as(c_int, @intCast(here)) * s;
        var sizes: [35]i32 = @splat(0);
        var kin: [nax_lanes]Array = undefined;
        var vin: [nax_lanes]Array = undefined;
        for (0..nax_lanes) |z| {
            const i = from + (if (z < here) z else 0);
            kin[z] = ks[i];
            vin[z] = vs[i];
            if (z >= here) continue;
            sizes[z * 8 ..][0..7].* = .{ s, keys[i], first[i], ops.shape(ks[i])[2], offsets[i], @as(c_int, @intCast(i)) * s, @as(c_int, @intCast(z)) * s };
        }
        sizes[32] = kps;
        sizes[33] = @intCast(cfg.log_n_floor);
        sizes[34] = rows;
        const o = try launchGrid(g, &relsplit_nax_kernel, &.{ q_rows, kin[0], kin[1], kin[2], kin[3], vin[0], vin[1], vin[2], vin[3], kb, vb, bias, sinks, gate_sigmoid, tmeta, try g.fromSlice(i32, &sizes, &.{35}), params }, 3, .{
            if (direct) unused else .{ .shape = &.{ splits, rows * heads, hd }, .dtype = .float32 },
            if (direct) unused else .{ .shape = &.{ splits, rows * heads, 2 }, .dtype = .float32 },
            if (direct) .{ .shape = &.{ 1, rows, heads * hd }, .dtype = dt } else unused,
        }, .{ splits * 32 * group, kv_heads, @intCast(here) }, .{ 32 * group, 1, 1 }, &.{
            .{ "T", .{ .dtype = dt } },                               .{ "D", .{ .int = hd } },
            .{ "HQ", .{ .int = heads } },                             .{ "HKV", .{ .int = kv_heads } },
            .{ "R", .{ .int = ops.shape(bias)[3] } },                 .{ "WINDOW", .{ .int = window } },
            .{ "GLOBAL", .{ .int = @intFromBool(L.global) } },        .{ "DIRECT", .{ .int = @intFromBool(direct) } },
            .{ "TREE", .{ .int = @intFromBool(tree_meta != null) } },
        });
        outs[made] = if (direct) o[2] else blk: {
            const lo: c_int = @as(c_int, @intCast(from)) * s * heads * hd;
            const gate_rows = if (here == n) gate_sigmoid else try g.sliceAxis(gate_sigmoid, 0, lo, lo + rows * heads * hd);
            break :blk try splitJoin(g, .{ o[0], o[1] }, gate_rows, rows, dt);
        };
        made += 1;
    }
    return if (made == 1) outs[0] else try g.concat(outs[0..made], 1);
}

/// `relsplit_lanes.metal`: 2..16-row multi-lane relative attention on non-NAX GPUs (M3/M4), reading up to 4 slots and
/// 16 rows in one pass over each 16-key tile, with DIRECT gated output when `splits == 1` and optional tree masks.
pub fn splitReadLanesM4(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, ks: []const Array, vs: []const Array, own: ?[2]Array, offsets: []const c_int, gate_sigmoid: Array, s: c_int, tree_meta: ?Array) !?Array {
    if (!mlx.streamIsGpu(g.s) or !enabled()) return null;
    const n = ks.len;
    const dt = ops.dtype(q_rows);
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const group = @divTrunc(heads, kv_heads);
    if (own) |o| {
        if (s < 2 or s > open_rows or ops.dtype(o[0]) != dt or ops.dtype(o[1]) != dt) return null;
    } else if (n != 1 or s < split_from or s > open_rows) return null;
    if (((dt != .bfloat16 and dt != .float16) and !(dt == .float32 and tree_meta != null)) or ops.dtype(sinks) != dt) return null;
    for (ks, vs) |k, v| if (ops.dtype(k) != dt or ops.dtype(v) != dt) return null;
    if (@rem(hd, 8) != 0 or @rem(heads, kv_heads) != 0 or 32 * group > sure_threads) return null;

    const proj = if (ops.dtype(L.rel_proj) == dt) L.rel_proj else try g.astype(L.rel_proj, dt);
    const bias = try g.matmul(rel_states, proj);
    const window: c_int = if (L.global) 0 else @intCast(cfg.sliding_window);
    const own_rows: c_int = if (own != null) s else 0;
    var keys: [max_slots]c_int = undefined;
    var first: [max_slots]c_int = undefined;
    var most: c_int = 0;
    for (ks, 0..) |k, i| {
        const cached = ops.shape(k)[2];
        keys[i] = cached + own_rows;
        first[i] = if (window > 0) @max(0, keys[i] - s - window + 1) else 0;
        most = @max(most, keys[i] - first[i]);
    }
    const kb, const vb = own orelse .{ ks[0], vs[0] };
    const params = try g.fromSlice(f32, &.{ 1.0 / @sqrt(@as(f32, @floatFromInt(hd))), cfg.log_alpha }, &.{2});
    const tmeta = tree_meta orelse try g.zeros(&.{ 1, 2 }, .int32);
    const nr: c_int = if (s > 8) 2 else 1;
    var outs: [max_slots / nax_lanes]Array = undefined;
    var made: usize = 0;
    var from: usize = 0;
    while (from < n) : (from += nax_lanes) {
        const here = @min(n - from, nax_lanes);
        const kps = splitKps(most, here);
        const splits: c_int = @max(1, @divTrunc(most + kps - 1, kps));
        const direct = splits == 1;
        const rows: c_int = @as(c_int, @intCast(here)) * s;
        var sizes: [35]i32 = @splat(0);
        var kin: [nax_lanes]Array = undefined;
        var vin: [nax_lanes]Array = undefined;
        for (0..nax_lanes) |z| {
            const i = from + (if (z < here) z else 0);
            kin[z] = ks[i];
            vin[z] = vs[i];
            if (z >= here) continue;
            sizes[z * 8 ..][0..7].* = .{ s, keys[i], first[i], ops.shape(ks[i])[2], offsets[i], @as(c_int, @intCast(i)) * s, @as(c_int, @intCast(z)) * s };
        }
        sizes[32] = kps;
        sizes[33] = @intCast(cfg.log_n_floor);
        sizes[34] = rows;
        const o = try launchGrid(g, &relsplit_lanes_kernel, &.{ q_rows, kin[0], kin[1], kin[2], kin[3], vin[0], vin[1], vin[2], vin[3], kb, vb, bias, sinks, gate_sigmoid, tmeta, try g.fromSlice(i32, &sizes, &.{35}), params }, 3, .{
            if (direct) unused else .{ .shape = &.{ splits, rows * heads, hd }, .dtype = .float32 },
            if (direct) unused else .{ .shape = &.{ splits, rows * heads, 2 }, .dtype = .float32 },
            if (direct) .{ .shape = &.{ 1, rows, heads * hd }, .dtype = dt } else unused,
        }, .{ splits * 32 * group, kv_heads, @intCast(here) }, .{ 32 * group, 1, 1 }, &.{
            .{ "T", .{ .dtype = dt } },                               .{ "D", .{ .int = hd } },
            .{ "HQ", .{ .int = heads } },                             .{ "HKV", .{ .int = kv_heads } },
            .{ "R", .{ .int = ops.shape(bias)[3] } },                 .{ "WINDOW", .{ .int = window } },
            .{ "GLOBAL", .{ .int = @intFromBool(L.global) } },        .{ "DIRECT", .{ .int = @intFromBool(direct) } },
            .{ "TREE", .{ .int = @intFromBool(tree_meta != null) } }, .{ "NR", .{ .int = nr } },
        });
        outs[made] = if (direct) o[2] else blk: {
            const lo: c_int = @as(c_int, @intCast(from)) * s * heads * hd;
            const gate_rows = if (here == n) gate_sigmoid else try g.sliceAxis(gate_sigmoid, 0, lo, lo + rows * heads * hd);
            break :blk try splitJoin(g, .{ o[0], o[1] }, gate_rows, rows, dt);
        };
        made += 1;
    }
    return if (made == 1) outs[0] else try g.concat(outs[0..made], 1);
}

/// A few or a prompt's rows read behind the keys `k`, `v` (the rows at their last S), gated.
fn readRows(g: *Graph, L: *const Layer, cfg: *const Config, p: *const Prep, sinks: Array, k: Array, v: Array, offset: c_int, s: c_int) !Array {
    if (try splitReadNax(g, L, cfg, p.q, p.rel, sinks, k, v, null, offset, p.gate)) |o| return o;
    if (try splitRead(g, L, cfg, p.q, p.rel, sinks, k, v, null, offset, p.gate)) |o| return o;
    return gated(g, try relative(g, L, cfg, p.q, p.rel, sinks, k, v, offset, false), p.gate, s);
}

/// A draft verify's visit: its rows read the cache without appending and leave here, kept by `g`, what a commit
/// needs: their keys and values `[1, Hkv, S, hd]` and the conv windows `[1, K - 1 + S, .]` (the state before them,
/// then every row's conv inputs). When `tree` is set, the verify rows form a per-slot tree.
pub const Open = struct { g: *Graph, k: Array = .{}, v: Array = .{}, window: Array = .{}, tree: ?TreeCtx = null };

/// Keeps the first `rows` rows of slot `lane` of a verify of `lanes` slots: each visit's keys and values appended, its
/// conv state the window behind them.
pub fn commit(c: *block.Ctx, cfg: *const Config, opens: []const Open, rows: c_int, lane: usize, lanes: usize) !void {
    const kt: c_int = @intCast(cfg.sconv_kernel);
    const slot: c_int = @intCast(lane);
    for (opens, 0..) |*o, v| {
        c.layer = @intCast(v);
        const per = @divExact(ops.shape(o.k)[2], @as(c_int, @intCast(lanes)));
        const at = slot * per;
        var view = try layers.appendKv(c, c.cache(), cfg.global[cfg.visit(@intCast(v)).layer], cfg, try c.g.sliceAxis(o.k, 2, at, at + rows), try c.g.sliceAxis(o.v, 2, at, at + rows), 0);
        kv.release(c, &view);
        const window = if (lanes == 1) o.window else try c.g.sliceAxis(o.window, 0, slot, slot + 1);
        try kv.setState(c, .conv, try c.g.sliceAxis(window, 1, rows, rows + kt - 1));
    }
}

/// Keeps the accepted tree path `path` (row indices within slot `lane`, root 0 first) of a verify of `lanes` slots:
/// gathers the path's K/V rows into each visit's cache and gathers the last `KT - 1` conv window rows along `path`.
pub fn commitPath(c: *block.Ctx, cfg: *const Config, opens: []const Open, path: []const u32, lane: usize, lanes: usize) !void {
    const kept: c_int = @intCast(path.len);
    var linear = true;
    for (path, 0..) |r, idx| {
        if (r != idx) {
            linear = false;
            break;
        }
    }
    if (linear) return commit(c, cfg, opens, kept, lane, lanes);
    const kt: usize = @intCast(cfg.sconv_kernel);
    const slot: c_int = @intCast(lane);
    const per = @divExact(ops.shape(opens[0].k)[2], @as(c_int, @intCast(lanes)));
    const at: i32 = @intCast(slot * per);
    var kv_idx_buf: [open_rows]i32 = undefined;
    for (path, kv_idx_buf[0..path.len]) |r, *dst| dst.* = at + @as(i32, @intCast(r));
    const kv_idx = try c.g.fromSlice(i32, kv_idx_buf[0..path.len], &.{kept});
    var conv_idx_buf: [8]i32 = undefined;
    for (0..kt - 1) |j| {
        const seq_pos = path.len + j;
        conv_idx_buf[j] = if (seq_pos < kt - 1) @intCast(seq_pos) else @intCast((kt - 1) + path[seq_pos - (kt - 1)]);
    }
    const conv_idx = try c.g.fromSlice(i32, conv_idx_buf[0 .. kt - 1], &.{@intCast(kt - 1)});
    const sw = ops.shape(opens[0].window)[2];
    const use_gpu = mlx.streamIsGpu(c.g.s) and enabled();
    for (opens, 0..) |*o, v| {
        c.layer = @intCast(v);
        const ks = try c.g.take(o.k, kv_idx, 2);
        const vs = try c.g.take(o.v, kv_idx, 2);
        var view = try layers.appendKv(c, c.cache(), cfg.global[cfg.visit(@intCast(v)).layer], cfg, ks, vs, 0);
        kv.release(c, &view);
        if (use_gpu) {
            const st = try launch(c.g, &commit_conv_kernel, &.{ o.window, conv_idx }, 1, .{.{ .shape = &.{ 1, @intCast(kt - 1), sw }, .dtype = .float32 }}, @as(c_int, @intCast(kt - 1)) * sw, 64, &.{.{ "SLOT", .{ .int = slot } }});
            try kv.setState(c, .conv, st[0]);
        } else {
            const window = if (lanes == 1) o.window else try c.g.sliceAxis(o.window, 0, slot, slot + 1);
            try kv.setState(c, .conv, try c.g.take(window, conv_idx, 1));
        }
    }
}

/// `layers.attend`: appends the visit's k and v, reads under the bias and the `sinks`, gates. An `open` verify reads
/// the cache's rows with its own behind them and appends nothing.
/// `raw`: one row's read is returned without its output gate, which `oGated` folds into the output projection.
fn attend(c: *block.Ctx, L: *const Layer, cfg: *const Config, p: *const Prep, sinks: Array, open: ?*Open, raw: bool) !Array {
    const g = c.g;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const s = c.rows;
    const offset: c_int = @intCast(c.offset());
    const span: u32 = if (L.global) 0 else cfg.sliding_window;
    if (open) |o| {
        if (s < 2) return error.Inkling2DraftVerify;
        o.k = try o.g.keep(p.k);
        o.v = try o.g.keep(p.v);
        return attendOpen(c, L, cfg, p, sinks, if (o.tree) |t| t.meta else null);
    }
    var view = try layers.appendKv(c, c.cache(), L.global, cfg, p.k, p.v, if (span > 0) band.view(span, hd, offset + s, s).span else 0);
    defer kv.release(c, &view);
    const dv = kv.dense(view);
    if (s > 1) return readRows(g, L, cfg, p, sinks, dv.k, dv.v, offset, s);
    if (ops.shape(dv.k)[2] != ops.shape(p.bias)[3]) return error.Inkling2KvView;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const row_sinks = try g.reshape(sinks, &.{heads});
    const attn = (try layers.packedRead(g, view, p.q, p.bias, row_sinks, scale)) orelse try sdpaSinks(g, p.q, dv.k, dv.v, scale, "array", p.bias, row_sinks, false);
    return if (raw) attn else gated(g, attn, p.gate, 1);
}

/// An open verify's read: each slot's `c.rows` rows read its own cache where it lies, its own rows behind it.
fn attendOpen(c: *block.Ctx, L: *const Layer, cfg: *const Config, p: *const Prep, sinks: Array, tree_meta: ?Array) !Array {
    const g = c.g;
    const n = c.caches.len;
    const s = c.rows;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const span: u32 = if (L.global) 0 else cfg.sliding_window;
    var views: [max_slots]abi.KvView = undefined;
    var made: usize = 0;
    defer for (views[0..made]) |*v| kv.release(c, v);
    var outs: [max_slots]Array = undefined;
    var whole_k: [max_slots]Array = undefined;
    var whole_v: [max_slots]Array = undefined;
    var whole_at: [max_slots]c_int = undefined;
    for (c.caches, c.positions, 0..) |cache, position, i| {
        views[i] = try kv.view(c, cache, c.layer);
        made += 1;
        const dv = kv.dense(views[i]);
        whole_k[i] = dv.k;
        whole_v[i] = dv.v;
        whole_at[i] = @intCast(position);
    }
    // The tensor unit reads every slot's block in a launch up to 4 slots wide; non-NAX GPUs use the 4-lane M4 kernel.
    if (try splitReadNaxLanesTree(g, L, cfg, p.q, p.rel, sinks, whole_k[0..n], whole_v[0..n], .{ p.k, p.v }, whole_at[0..n], p.gate, s, tree_meta)) |read| return read;
    if (try splitReadLanesM4(g, L, cfg, p.q, p.rel, sinks, whole_k[0..n], whole_v[0..n], .{ p.k, p.v }, whole_at[0..n], p.gate, s, tree_meta)) |read| return read;
    if (tree_meta != null) return error.Inkling2TreeAttend;
    for (c.positions, 0..) |position, i| {
        const dv = kv.dense(views[i]);
        const keys = ops.shape(dv.k)[2];
        // A sliding layer's rows reach at most `window - 1` keys before the block.
        const from = if (span > 0) @max(0, keys - @as(c_int, @intCast(span - 1))) else 0;
        const cached = [2]Array{ try g.sliceAxis(dv.k, 2, from, keys), try g.sliceAxis(dv.v, 2, from, keys) };
        const offset: c_int = @intCast(position);
        // The slot's rows of the stacked projections.
        const at: c_int = @intCast(i * @as(usize, @intCast(s)));
        const solo = n == 1;
        const q = if (solo) p.q else try g.sliceAxis(p.q, 1, at, at + s);
        const rel_states = if (solo) p.rel else try g.sliceAxis(p.rel, 1, at, at + s);
        const sink_rows = if (solo) sinks else try g.sliceAxis(sinks, 1, at, at + s);
        const gate_rows = if (solo) p.gate else try g.sliceAxis(p.gate, 0, at * heads * hd, (at + s) * heads * hd);
        const own = [2]Array{ if (solo) p.k else try g.sliceAxis(p.k, 2, at, at + s), if (solo) p.v else try g.sliceAxis(p.v, 2, at, at + s) };
        // The cache is read where it lies; a read the split kernel does not serve joins the rows to a copy of it.
        outs[i] = if (try splitReadNax(g, L, cfg, q, rel_states, sink_rows, cached[0], cached[1], own, offset, gate_rows)) |read| read else if (try splitRead(g, L, cfg, q, rel_states, sink_rows, cached[0], cached[1], own, offset, gate_rows)) |read| read else blk: {
            const slot_p: Prep = .{ .q = q, .k = own[0], .v = own[1], .gate = gate_rows, .bias = p.bias, .rel = rel_states };
            break :blk try readRows(g, L, cfg, &slot_p, sink_rows, try g.concat(&.{ cached[0], own[0] }, 2), try g.concat(&.{ cached[1], own[1] }, 2), offset, s);
        };
    }
    return if (n == 1) outs[0] else g.concat(outs[0..n], 1);
}

/// A few rows' router logits `[S, E]` f32: `x` `[1, S, D]` f32 over the router's rows `[E, D]` f32. MLX's GPU f32
/// matmul past one row is ~7x a row's latency and good to ~1e-3; its one-row path, and this, are exact.
pub fn routerRows(g: *Graph, x: Array, router: Array) !Array {
    const sh = ops.shape(router);
    const s = ops.shape(x)[1];
    const o = try launchGrid(g, &router_kernel, &.{ x, router }, 1, .{.{ .shape = &.{ s, sh[0] }, .dtype = .float32 }}, .{ 32 * sh[0], s, 1 }, .{ 32, 1, 1 }, &.{ .{ "D", .{ .int = sh[1] } }, .{ "E", .{ .int = sh[0] } } });
    return o[0];
}

/// The router's picks for the rows' logits `scores` `[S, E]` (or `[1, S, E]`) f32: the `k` experts of the largest
/// sigmoid plus `bias` `[E]`, by rank, `[1, S, k]` uint32, and their sigmoids over their sum times `scale` (`[1]`),
/// `[1, S, k]` f32.
pub fn topK(g: *Graph, scores: Array, bias: Array, scale: Array, k: c_int) ![2]Array {
    const sh = ops.shape(scores);
    const s = sh[sh.len - 2];
    const e = sh[sh.len - 1];
    return launchGrid(g, &topk_kernel, &.{ scores, bias, scale }, 2, .{
        .{ .shape = &.{ 1, s, k }, .dtype = .uint32 },
        .{ .shape = &.{ 1, s, k }, .dtype = .float32 },
    }, .{ e, s, 1 }, .{ e, 1, 1 }, &.{ .{ "E", .{ .int = e } }, .{ "K", .{ .int = k } } });
}

/// Batched slots at once (`batched_rows`, one row a slot).
pub const max_slots = 64;

/// Whether batched slots ride the kernels as rows, each a sequence of its own, on the stream `h` is on.
pub fn batches(g: *const Graph, w: *const Body, h: Array, slots: c_int) bool {
    if (slots > max_slots) return false;
    for (w.layers) |*L| if (!runs(g, L, &w.cfg, h)) return false;
    return true;
}

/// `attend` for batched slots: row i of the visit's q, k, v and relative states is slot i's, appended to its own
/// cache and read through its own view under its own bias and sink.
fn attendSlots(c: *block.Ctx, L: *const Layer, cfg: *const Config, consts: ?*const layers.Consts, p: *const Prep, sinks: Array) !Array {
    const g = c.g;
    const n = c.caches.len;
    const hd: c_int = @intCast(cfg.head_dim);
    const dt = ops.dtype(p.q);
    const span: u32 = if (L.global) 0 else cfg.sliding_window;
    var views: [max_slots]abi.KvView = undefined;
    var made: usize = 0;
    defer for (views[0..made]) |*v| kv.release(c, v);
    var ks: [max_slots]Array = undefined;
    var vs: [max_slots]Array = undefined;
    for (c.caches, c.positions, 0..) |cache, position, i| {
        const at: c_int = @intCast(i);
        const offset: c_int = @intCast(position);
        views[i] = try layers.appendKv(c, cache, L.global, cfg, try g.sliceAxis(p.k, 2, at, at + 1), try g.sliceAxis(p.v, 2, at, at + 1), if (span > 0) band.view(span, hd, offset + 1, 1).span else 0);
        made += 1;
        const dv = kv.dense(views[i]);
        ks[i] = dv.k;
        vs[i] = dv.v;
    }
    const use_slots = slotsRuns(g, cfg, dt, ks[0..n], vs[0..n]);
    var q = if (use_slots) p.q else try g.transpose(p.q, &.{ 0, 2, 1, 3 });
    var states = p.rel;
    if (L.global) {
        // As a one-row step: q and the relative states times the slot's factor, rounded, before the bias projection.
        var taus: [max_slots]f32 = undefined;
        var scaled = false;
        for (c.positions, taus[0..n]) |at, *t| {
            t.* = layers.stepTau(cfg, @as(c_int, @intCast(at)) + 1);
            scaled = scaled or t.* != 1.0;
        }
        if (scaled) {
            const rows: c_int = @intCast(n);
            const tau_s = try g.fromSlice(f32, taus[0..n], &.{ 1, rows, 1, 1 });
            const tau_q = if (use_slots) tau_s else try g.reshape(tau_s, &.{ 1, 1, rows, 1 });
            q = try g.astype(try g.mul(try g.astype(q, .float32), tau_q), dt);
            states = try g.astype(try g.mul(try g.astype(states, .float32), tau_s), dt);
        }
    }
    const rel_proj = if (ops.dtype(L.rel_proj) == dt) L.rel_proj else try g.astype(L.rel_proj, dt);
    const bias = try g.matmul(states, rel_proj);
    if (use_slots) {
        const sink_rows = try g.contiguous(try g.astype(sinks, .float32));
        return slotsReadWithConsts(g, consts, q, ks[0..n], vs[0..n], bias, sink_rows, @intCast(span), p.gate);
    }
    return slotsChain(g, q, ks[0..n], vs[0..n], bias, sinks, @intCast(span), p.gate);
}

/// The per-slot read `slotsRead` replaces: a mask launch and MLX's attention a slot, gated. `q` `[1, H, n, hd]`,
/// `sinks` `[1, n, H]` in the projections' dtype; the rest as `slotsRead`'s.
pub fn slotsChain(g: *Graph, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int, gate_sigmoid: Array) !Array {
    const heads = ops.shape(q)[1];
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(ops.shape(q)[3])));
    const sink_rows = try g.contiguous(try g.astype(sinks, .float32));
    var outs: [max_slots]Array = undefined;
    for (ks, vs, 0..) |k, v, i| {
        const at: c_int = @intCast(i);
        const mask = try relMask(g, try g.sliceAxis(bias, 1, at, at + 1), sink_rows, ops.shape(k)[2], window, false);
        outs[i] = try sdpaSinks(g, try g.sliceAxis(q, 2, at, at + 1), k, v, scale, "array", mask, try g.reshape(try g.sliceAxis(sinks, 1, at, at + 1), &.{heads}), false);
    }
    return gated(g, try g.transpose(try g.concat(outs[0..ks.len], 2), &.{ 0, 2, 1, 3 }), gate_sigmoid, @intCast(ks.len));
}

/// Slots one launch of `relslots.metal` reads: each slot's K, V and K's strides take a buffer, and a kernel has 31.
pub const slots_per_read = 8;
/// Its keys a threadgroup and SIMD groups a threadgroup.
const slot_keys = 256;
const slot_groups = 8;

/// Whether `slotsRead` serves batched slots of `dt` on `g`'s stream: a float dtype the caches share, head widths of
/// whole 8-value chunks, the query heads a KV head in one threadgroup's memory.
pub fn slotsRuns(g: *const Graph, cfg: *const Config, dt: Dtype, ks: []const Array, vs: []const Array) bool {
    if (!mlx.streamIsGpu(g.s) or !enabled() or !floating(dt)) return false;
    const group = cfg.num_attention_heads / cfg.num_key_value_heads;
    if (cfg.head_dim % 8 != 0 or cfg.num_attention_heads % cfg.num_key_value_heads != 0 or group > slot_groups) return false;
    for (ks, vs) |k, v| if (ops.dtype(k) != dt or ops.dtype(v) != dt) return false;
    return true;
}

/// One row a slot read behind its own cache view (`ks[i]`, `vs[i]` `[1, Hkv, keys_i, hd]`, the row at its last key,
/// V laid out as K) under its bias rows `bias` `[1, n, H, R]` and sinks `[1, n, H]` f32, `q` `[1, n, H, hd]`
/// contiguous: `relslots.metal` for up to `slots_per_read` slots a launch, joined under the output gate by
/// `relcomb.metal`, `[1, n, H * hd]`.
pub fn slotsRead(g: *Graph, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int, gate_sigmoid: Array) !Array {
    return slotsReadWithConsts(g, null, q, ks, vs, bias, sinks, window, gate_sigmoid);
}

fn slotsReadWithConsts(g: *Graph, consts: ?*const layers.Consts, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int, gate_sigmoid: Array) !Array {
    const n = ks.len;
    const width = ops.shape(q)[2] * ops.shape(q)[3];
    var outs: [max_slots / slots_per_read]Array = undefined;
    var groups: usize = 0;
    var from: usize = 0;
    while (from < n) : (groups += 1) {
        const to = @min(n, from + slots_per_read);
        const whole = from == 0 and to == n;
        const lo: c_int = @intCast(from);
        const hi: c_int = @intCast(to);
        const qg = if (whole) q else try g.sliceAxis(q, 1, lo, hi);
        const bg = if (whole) bias else try g.sliceAxis(bias, 1, lo, hi);
        const sg = if (whole) sinks else try g.sliceAxis(sinks, 1, lo, hi);
        const gg = if (whole) gate_sigmoid else try g.sliceAxis(gate_sigmoid, 0, lo * width, hi * width);
        const part = try slotPartsWithConsts(g, consts, qg, ks[from..to], vs[from..to], bg, sg, window);
        outs[groups] = try splitJoin(g, part, gg, hi - lo, ops.dtype(q));
        from = to;
    }
    return if (groups == 1) outs[0] else g.concat(outs[0..groups], 1);
}

/// `relslots.metal` for up to `slots_per_read` slots: each split's unnormalised sums `[splits, n * H, hd]` and (max,
/// sum) `[splits, n * H, 2]`, f32, as `splitParts`'.
pub fn slotParts(g: *Graph, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int) ![2]Array {
    return slotPartsWithConsts(g, null, q, ks, vs, bias, sinks, window);
}

fn slotPartsWithConsts(g: *Graph, consts: ?*const layers.Consts, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int) ![2]Array {
    const ng: c_int = @intCast(ks.len);
    const heads = ops.shape(q)[2];
    const hd = ops.shape(q)[3];
    var inputs: [21]Array = undefined;
    inputs[0] = q;
    var keys: [slots_per_read]i32 = undefined;
    var splits: c_int = 1;
    for (0..slots_per_read) |j| {
        // A launch binds every slot input; the slots past the group read the first one's, unused.
        const at = if (j < ks.len) j else 0;
        inputs[1 + j] = ks[at];
        inputs[1 + slots_per_read + j] = vs[at];
        if (j >= ks.len) continue;
        const len = ops.shape(ks[j])[2];
        keys[j] = len;
        const seen = if (window > 0) @min(len, window) else len;
        splits = @max(splits, @divTrunc(seen + slot_keys - 1, slot_keys));
    }
    inputs[17] = bias;
    inputs[18] = sinks;
    inputs[19] = try g.fromSlice(i32, keys[0..ks.len], &.{ng});
    inputs[20] = if (consts != null and consts.?.attn_scale.ctx != null) consts.?.attn_scale else try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32);
    return launchGrid(g, &relslots_kernel, &inputs, 2, .{
        .{ .shape = &.{ splits, ng * heads, hd }, .dtype = .float32 },
        .{ .shape = &.{ splits, ng * heads, 2 }, .dtype = .float32 },
    }, .{ splits * 32 * slot_groups, ops.shape(ks[0])[1], ng }, .{ 32 * slot_groups, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(q) } },        .{ "D", .{ .int = hd } },                 .{ "HQ", .{ .int = heads } },
        .{ "HKV", .{ .int = ops.shape(ks[0])[1] } }, .{ "R", .{ .int = ops.shape(bias)[3] } }, .{ "NS", .{ .int = ng } },
        .{ "NSG", .{ .int = slot_groups } },         .{ "KB", .{ .int = slot_keys } },         .{ "WINDOW", .{ .int = window } },
    });
}

/// A visit between its attention and its experts: the MLP's input, the router's logits `[S, E]` f32, the f32 residual
/// the MLP's output joins, the block stack when the layer opened a block, and the output projection (the attention
/// conv's newest inputs).
pub const Middle = struct { x: Array, scores: Array, residual: Array, blocks: ?Array, op: Array };

/// `layers.mid`: `attn` are the gated attention rows, `h` the visit's input stream, `normed` its unweighted norm,
/// `messages` `[4, S, D]` on a later loop pass.
/// `out_gate`: the output gate of a read `attend` left ungated (one row), applied inside the output projection.
pub fn mid(g: *Graph, L: *const Layer, w: *const Body, attn: Array, out_gate: ?Array, h: Array, normed: Array, blocks: ?Array, st: Array, messages: ?Array) !Middle {
    return midWith(g, L, w, attn, out_gate, h, normed, blocks, st, messages, null);
}

pub fn midWith(g: *Graph, L: *const Layer, w: *const Body, attn: Array, out_gate: ?Array, h: Array, normed: Array, blocks: ?Array, st: Array, messages: ?Array, parent: ?Array) !Middle {
    const cfg = &w.cfg;
    const d: c_int = @intCast(cfg.hidden_size);
    const s = ops.shape(h)[1];
    const kt: c_int = @intCast(cfg.sconv_kernel);
    const op = if (out_gate) |gt| try oGated(g, &L.o, attn, gt) else try layers.projectRows(g, &L.o, attn);
    const dt = ops.dtype(attn);
    if (!floating(ops.dtype(op))) return error.Inkling2StepDtype;
    const gates: ?*const layers.Gates = if (messages != null) &(L.gates orelse return error.Inkling2NoMessageGate) else null;
    const gt: Dtype = if (gates) |x| ops.dtype(x.down) else .float32;
    const msg3: Array = if (messages) |m| try g.sliceAxis(m, 0, 3, 4) else h;
    const pr = if (w.consts.mid_pr.ctx != null) w.consts.mid_pr else try g.fromSlice(f32, &.{ 1.0 / @sqrt(@as(f32, @floatFromInt(kt + 1))), cfg.rms_norm_eps }, &.{2});
    const par = parent orelse (if (w.consts.zero_i32.ctx != null) w.consts.zero_i32 else try g.zeros(&.{1}, .int32));
    const on = gates != null;
    const o = while (true) {
        const tg = streamGroup(cfg.hidden_size) orelse return error.ThreadLimit;
        break launchGrid(g, &mid_kernel, &.{ op, st, L.attn_conv, h, normed, blocks orelse h, L.mlp_query, L.post_norm, msg3, pr, par }, 5, .{
            .{ .shape = &.{ 1, s, d }, .dtype = .float32 },
            if (on) unused else .{ .shape = &.{ 1, s, d }, .dtype = dt },
            if (on) unused else .{ .shape = &.{ 1, s, d }, .dtype = .float32 },
            if (on) .{ .shape = &.{ 1, s, d }, .dtype = .float32 } else unused,
            if (on) .{ .shape = &.{ 1, s, 2 * d }, .dtype = gt } else unused,
        }, .{ d, s, 1 }, .{ tg, 1, 1 }, &.{
            .{ "T", .{ .dtype = ops.dtype(op) } },                         .{ "XT", .{ .dtype = dt } },
            .{ "WT", .{ .dtype = ops.dtype(L.attn_conv) } },               .{ "GT", .{ .dtype = gt } },
            .{ "TG", .{ .int = tg } },                                     .{ "D", .{ .int = d } },
            .{ "KT", .{ .int = kt } },                                     .{ "OFF", .{ .int = @intCast(2 * cfg.kvDim()) } },
            .{ "NB", .{ .int = if (blocks) |b| ops.shape(b)[2] else 0 } }, .{ "BOUND", .{ .int = @intFromBool(L.boundary) } },
            .{ "MSG", .{ .int = @intFromBool(on) } },                      .{ "TREE", .{ .int = @intFromBool(parent != null) } },
        }) catch |e| if (e == error.ThreadLimit and narrow(cfg.hidden_size)) continue else return e;
    };
    var x = o[1];
    // A plain visit's router is MLX's matmul; a looped one's rides its norm kernel, where the rows read the stream
    // as it is (inside `mid.metal` each row would redo the conv and the mix: slower than the launch it saves).
    var scores: Array = undefined;
    if (gates) |gs| {
        const state = try gateDowns(g, o[4], gs, 3, 1);
        const xs = try normWithConsts(g, cfg, &w.consts, try gate(g, gs, o[3], msg3, state, 3, 1), @splat(L.post_norm), 1, dt, L.router);
        x = xs[0];
        scores = xs[1];
    } else scores = if (s > 1 and s <= packed_rows) try routerRows(g, o[2], L.router) else try g.matmul(o[2], try g.transpose(L.router, &.{ 1, 0 }));
    var opened: ?Array = null;
    if (L.boundary) {
        const saved = try g.expandDims(normed, 2);
        opened = if (blocks) |stack| try g.concat(&.{ stack, saved }, 2) else saved;
    }
    return .{ .x = x, .scores = scores, .residual = o[0], .blocks = opened, .op = op };
}

/// A visit's output stream and its new conv state.
pub const Tail = struct { h: Array, state: Array };

/// The picked experts' soft-capped SwiGLU `[N, 1, I]` in `gu`'s dtype: `gu` `[N, 1, 2 I]` holds each pick's gate
/// (even) and up (odd) rows, times `scale[sel]` when `scaled`; `caps` = { 1 / gate cap, gate cap, 1 / up cap, up cap }.
pub fn activation(g: *Graph, gu: Array, sel: Array, scale: Array, caps: Array, scaled: bool) !Array {
    const sh = ops.shape(gu);
    const inter = @divExact(sh[2], 2);
    const o = try launch(g, &act_kernel, &.{ gu, sel, scale, caps }, 1, .{.{ .shape = &.{ sh[0], 1, inter }, .dtype = ops.dtype(gu) }}, sh[0] * inter, 64, &.{ .{ "T", .{ .dtype = ops.dtype(gu) } }, .{ "SCALED", .{ .int = @intFromBool(scaled) } } });
    return o[0];
}

/// A visit's output stream and MLP output `[1, S, D]`, each in `dn`'s dtype: the rows' `k` picks `dn` `[S * k, 1, D]`
/// (pair (row, pick) at `inverse[row * k + pick]`, of expert `sel` there) times `scale[sel]` when `scaled` and their
/// routing `weights` `[1, S, k]`, summed; then the MLP's short conv behind `st` with `taps`, and the residual.
pub fn pickSum(g: *Graph, dn: Array, sel: Array, inverse: Array, scale: Array, weights: Array, residual: Array, st: Array, taps: Array, conv_scale: Array, k: c_int, scaled: bool) ![2]Array {
    return pickSumWith(g, dn, sel, inverse, scale, weights, residual, st, taps, conv_scale, k, scaled, null);
}

pub fn pickSumWith(g: *Graph, dn: Array, sel: Array, inverse: Array, scale: Array, weights: Array, residual: Array, st: Array, taps: Array, conv_scale: Array, k: c_int, scaled: bool, parent: ?Array) ![2]Array {
    const s = ops.shape(residual)[1];
    const d = ops.shape(residual)[2];
    const dt = ops.dtype(dn);
    const par = parent orelse try g.zeros(&.{1}, .int32);
    const out: Out = .{ .shape = &.{ 1, s, d }, .dtype = dt };
    return launch(g, &sum_kernel, &.{ dn, sel, inverse, scale, weights, residual, st, taps, conv_scale, par }, 2, .{ out, out }, s * d, 64, &.{
        .{ "T", .{ .dtype = dt } }, .{ "WT", .{ .dtype = ops.dtype(taps) } }, .{ "K", .{ .int = k } }, .{ "KT", .{ .int = ops.shape(st)[1] + 1 } }, .{ "SCALED", .{ .int = @intFromBool(scaled) } }, .{ "TREE", .{ .int = @intFromBool(parent != null) } },
    });
}

/// The most rows the packed expert kernels serve (64 * 8 = 512 fits `group.metal`'s shared buffer).
pub const packed_rows = 64;
/// The packed layout `ups.metal` and `downs.metal` read in place, of a bank over `in_dim` inputs: 0 NVFP4, 1 MXFP4,
/// 2 MXFP8, 3 affine (2 to 6 and 8 bits, groups 32, 64, 128, scales and biases of one float dtype); null = another (a
/// dense bank), which MLX's gather serves.
pub fn packedFormat(bank: anytype, in_dim: c_int) ?c_int {
    if (bank.s.ctx == null or ops.dtype(bank.w) != .uint32) return null;
    const qp = bank.qp;
    const fp = ops.dtype(bank.s) == .uint8;
    const fmt: c_int = switch (qp.mode) {
        .nvfp4 => if (fp and qp.bits == 4 and qp.group_size == 16) 0 else return null,
        .mxfp4 => if (fp and qp.bits == 4 and qp.group_size == 32) 1 else return null,
        .mxfp8 => if (fp and qp.bits == 8 and qp.group_size == 32) 2 else return null,
        .affine => if (!fp and bank.b.ctx != null and ops.dtype(bank.b) == ops.dtype(bank.s) and floating(ops.dtype(bank.s)) and affineWidth(qp.bits, qp.group_size)) 3 else return null,
        else => return null,
    };
    const rows = ops.shape(bank.w)[1];
    return if (@rem(in_dim, 32) == 0 and @rem(rows, 2 * bank_rows) == 0 and @rem(rows, 2 * bank_groups) == 0) fmt else null;
}

fn affineWidth(bits: u32, group: u32) bool {
    return (bits >= 2 and bits <= 6 or bits == 8) and (group == 32 or group == 64 or group == 128);
}

/// A bank's biases for the packed kernels' `WB`: the affine bank's own, the scales again for the formats with none.
fn biasesOf(bank: anytype) Array {
    return if (bank.b.ctx != null) bank.b else bank.s;
}

/// Tile width for grouped expert picks: 4 up to 128 picks (low register footprint for solo/2-slot verifies), 8 past that.
pub fn groupTileWidth(n: c_int) c_int {
    return if (n <= 128) 4 else 8;
}

/// `group.metal` over `sel` `[N]` u32 expert ids (a row's picks together): the picks as tiles of at most `BM` (4 or 8) of one
/// expert, `{ expert [max_tiles] (0xFFFFFFFF past the last tile), picks in the tile [max_tiles], pick indices [max_tiles * BM] }`.
pub fn groupPicks(g: *Graph, sel: Array, num_experts: u32) ![3]Array {
    const n = ops.shape(sel)[0];
    const bm = groupTileWidth(n);
    const e: c_int = @intCast(num_experts);
    const max_tiles: c_int = @min(n, e + @divTrunc(n - 1, bm));
    const tg = @divTrunc(e + 31, 32) * 32;
    return launch(g, &group_kernel, &.{sel}, 3, .{
        .{ .shape = &.{max_tiles}, .dtype = .uint32 },
        .{ .shape = &.{max_tiles}, .dtype = .uint32 },
        .{ .shape = &.{max_tiles * bm}, .dtype = .uint32 },
    }, tg, tg, &.{ .{ "E", .{ .int = e } }, .{ "TG", .{ .int = tg } }, .{ "BM", .{ .int = bm } } });
}

/// What a packed gate_up launch reads its picks from: the router's logits `[S, E]` f32, ranked inside the launch as
/// `topK` ranks them (one row), or the picks `[S * k]` u32 a `topK` launch left (more rows).
pub const UpsRoute = union(enum) { scores: Array, picks: Array };

/// The rows' picked experts' soft-capped SwiGLU `[S * k, 1, I]` from a packed gate_up bank, in one launch: { act,
/// picks, weights }, the picks `[1, S, k]` and their weights only when `route` is the logits.
pub fn ups(g: *Graph, L: *const Layer, cfg: *const Config, x: Array, route: UpsRoute) ![3]Array {
    const dt = ops.dtype(x);
    const s = ops.shape(x)[1];
    const top_k: c_int = @intCast(cfg.top_k);
    const inter: c_int = @intCast(cfg.moe_intermediate_size);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const fmt = packedFormat(&L.gate_up, hidden) orelse return error.Inkling2ExpertLayout;
    const caps = try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4});
    const scaled = L.gate_up_scale.ctx != null;
    const ranked = route == .scores;
    // The arm not taken reads none of its inputs: any array of a float dtype stands in.
    const scores = if (ranked) route.scores else caps;
    const sh = ops.shape(scores);
    return launchGrid(g, &ups_kernel, &.{ x, L.gate_up.w, L.gate_up.s, biasesOf(&L.gate_up), scores, if (ranked) L.router_bias else caps, try g.fromSlice(f32, &.{cfg.route_scale}, &.{1}), if (ranked) caps else route.picks, if (scaled) L.gate_up_scale else caps, caps }, 3, .{
        .{ .shape = &.{ s * top_k, 1, inter }, .dtype = dt },
        if (ranked) .{ .shape = &.{ 1, s, top_k }, .dtype = .uint32 } else unused,
        if (ranked) .{ .shape = &.{ 1, s, top_k }, .dtype = .float32 } else unused,
    }, .{ 32, @divExact(inter, bank_rows), s * top_k }, .{ 32, bank_groups, 1 }, &.{
        .{ "T", .{ .dtype = dt } },                               .{ "K", .{ .int = hidden } },                            .{ "I", .{ .int = inter } },
        .{ "GS", .{ .int = @intCast(L.gate_up.qp.group_size) } }, .{ "FMT", .{ .int = fmt } },                             .{ "ROWS", .{ .int = bank_rows } },
        .{ "BITS", .{ .int = @intCast(L.gate_up.qp.bits) } },     .{ "S", .{ .dtype = ops.dtype(L.gate_up.s) } },          .{ "SCALED", .{ .int = @intFromBool(scaled) } },
        .{ "KSEL", .{ .int = top_k } },                           .{ "E", .{ .int = if (ranked) sh[sh.len - 1] else 1 } }, .{ "SGS", .{ .int = bank_groups } },
        .{ "ROUTE", .{ .int = @intFromBool(ranked) } },
    });
}

/// `layers.tail`. The visit's new conv state (off the next visit's path) is `st` behind the rows' k and v projections
/// `kp`, `vp`, their output projection and the MLP's output.
/// One row, a packed bank read in place: the router's picks, their gate_up and the soft-capped SwiGLU as one launch,
/// their down, the weighted sum, the MLP's conv behind `st` and the final add as a second. 2 to `packed_rows` rows read
/// the packed banks in place with their picks grouped by expert (`groupPicks`: `ups_grp`, `downs_grp`, so a shared
/// expert is read once for up to eight picks), then `few_tail` sums, convolves and adds, and in a verify keeps the conv
/// window. Another bank, or more rows (their pairs sorted by expert), go through `topK` and MLX's gather, with the
/// SwiGLU and the sum, conv and add as launches of their own.
pub fn experts(g: *Graph, L: *const Layer, cfg: *const Config, m: *const Middle, kp: Array, vp: Array, st: Array, open: ?*Open) !Tail {
    return expertsWithConsts(g, L, cfg, null, m, kp, vp, st, open);
}

fn expertsWithConsts(g: *Graph, L: *const Layer, cfg: *const Config, consts: ?*const layers.Consts, m: *const Middle, kp: Array, vp: Array, st: Array, open: ?*Open) !Tail {
    const dt = ops.dtype(m.x);
    const s = ops.shape(m.x)[1];
    const top_k: c_int = @intCast(cfg.top_k);
    const inter: c_int = @intCast(cfg.moe_intermediate_size);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const kt: c_int = @intCast(cfg.sconv_kernel);
    const n = s * top_k;
    const gu_fmt = packedFormat(&L.gate_up, hidden);
    const dn_fmt = packedFormat(&L.down, inter);
    const few = s > 1 and s <= packed_rows and n <= 512 and gu_fmt != null and dn_fmt != null;
    const route_scale = if (consts != null and consts.?.route_scale.ctx != null) consts.?.route_scale else try g.fromSlice(f32, &.{cfg.route_scale}, &.{1});
    const up = if (s == 1 and gu_fmt != null) try ups(g, L, cfg, m.x, .{ .scores = m.scores }) else null;
    const picks: [2]Array = if (up) |u| .{ u[1], u[2] } else try topK(g, m.scores, L.router_bias, route_scale, top_k);
    const weights = picks[1];
    const caps = if (consts != null and consts.?.caps.ctx != null) consts.?.caps else try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4});
    const conv_scale = if (consts != null and consts.?.conv_scale.ctx != null) consts.?.conv_scale else try g.fromSlice(f32, &.{1.0 / @sqrt(@as(f32, @floatFromInt(kt + 1)))}, &.{1});
    const dn_scale = if (L.down_scale.ctx != null) L.down_scale else weights;
    const dn_scaled: TArg = .{ .int = @intFromBool(L.down_scale.ctx != null) };
    const tail_out: [2]Out = .{ .{ .shape = &.{ 1, s, hidden }, .dtype = dt }, .{ .shape = &.{ 1, s, hidden }, .dtype = dt } };

    if (few) {
        const sel = try g.reshape(picks[0], &.{n});
        const bm = groupTileWidth(n);
        const grp = try groupPicks(g, sel, cfg.num_experts);
        const max_tiles = ops.shape(grp[0])[0];
        const gu_scale = if (L.gate_up_scale.ctx != null) L.gate_up_scale else caps;
        const gu_scaled: TArg = .{ .int = @intFromBool(L.gate_up_scale.ctx != null) };
        const act_out: [1]Out = .{.{ .shape = &.{ n, 1, inter }, .dtype = dt }};
        const half_inter = @divExact(inter, 2);
        const ups_sgs: c_int = if (@rem(half_inter, 4) == 0) 4 else if (@rem(half_inter, 2) == 0) 2 else 1;
        const act = try launchGrid(g, &ups_grp_kernel, &.{ m.x, L.gate_up.w, L.gate_up.s, biasesOf(&L.gate_up), grp[0], grp[1], grp[2], gu_scale, caps }, 1, act_out, .{ 32 * max_tiles, half_inter, 1 }, .{ 32, ups_sgs, 1 }, &.{
            .{ "T", .{ .dtype = dt } },                               .{ "K", .{ .int = hidden } },     .{ "I", .{ .int = inter } },
            .{ "GS", .{ .int = @intCast(L.gate_up.qp.group_size) } }, .{ "FMT", .{ .int = gu_fmt.? } }, .{ "BITS", .{ .int = @intCast(L.gate_up.qp.bits) } },
            .{ "S", .{ .dtype = ops.dtype(L.gate_up.s) } },           .{ "SCALED", gu_scaled },         .{ "KSEL", .{ .int = top_k } },
            .{ "BM", .{ .int = bm } },
        });
        const qtr_hidden = @divExact(hidden, 4);
        const dn_sgs: c_int = if (@rem(qtr_hidden, 4) == 0) 4 else if (@rem(qtr_hidden, 2) == 0) 2 else 1;
        const dn = try launchGrid(g, &downs_grp_kernel, &.{ act[0], L.down.w, L.down.s, biasesOf(&L.down), grp[0], grp[1], grp[2] }, 1, .{.{ .shape = &.{ n, 1, hidden }, .dtype = dt }}, .{ 32 * max_tiles, qtr_hidden, 1 }, .{ 32, dn_sgs, 1 }, &.{
            .{ "T", .{ .dtype = dt } },                            .{ "I", .{ .int = inter } },      .{ "D", .{ .int = hidden } },
            .{ "GS", .{ .int = @intCast(L.down.qp.group_size) } }, .{ "FMT", .{ .int = dn_fmt.? } }, .{ "BITS", .{ .int = @intCast(L.down.qp.bits) } },
            .{ "S", .{ .dtype = ops.dtype(L.down.s) } },           .{ "BM", .{ .int = bm } },
        });
        const sw = ops.shape(st)[2];
        const nslot = ops.shape(st)[0];
        const rps = @divExact(s, nslot);
        const is_open = open != null;
        const tree_parent: ?Array = if (open) |o| (if (o.tree) |t| t.parent else null) else null;
        const par = tree_parent orelse (if (consts != null and consts.?.zero_i32.ctx != null) consts.?.zero_i32 else try g.zeros(&.{1}, .int32));
        const off = sw - hidden;
        const total_threads = hidden + if (is_open) nslot * (kt - 1) * sw + s * off else nslot * (kt - 1) * off;
        const ft = try launch(g, &few_tail_kernel, &.{ dn[0], sel, dn_scale, weights, m.residual, st, L.mlp_conv, conv_scale, kp, vp, m.op, par }, 3, .{
            tail_out[0],
            tail_out[1],
            if (is_open) .{ .shape = &.{ nslot, kt - 1 + rps, sw }, .dtype = .float32 } else .{ .shape = &.{ nslot, kt - 1, sw }, .dtype = .float32 },
        }, total_threads, 64, &.{
            .{ "T", .{ .dtype = dt } },
            .{ "WT", .{ .dtype = ops.dtype(L.mlp_conv) } },
            .{ "KSEL", .{ .int = top_k } },
            .{ "KT", .{ .int = kt } },
            .{ "SCALED", dn_scaled },
            .{ "OPEN", .{ .int = @intFromBool(is_open) } },
            .{ "STATE", .{ .int = @intFromBool(!is_open) } },
            .{ "TREE", .{ .int = @intFromBool(tree_parent != null) } },
        });
        if (open) |v| {
            v.window = try v.g.keep(ft[2]);
            return .{ .h = ft[0], .state = st };
        }
        return .{ .h = ft[0], .state = ft[2] };
    }

    const pairs = try layers.pairs(g, cfg, m.x, picks[0]);
    const sel = pairs.sel;
    const sorted = pairs.inverse != null;

    const act: [1]Array = if (up) |u| .{u[0]} else blk: {
        const gu = try layers.gather(g, pairs.rows, &L.gate_up, sel, sorted);
        if (!floating(ops.dtype(gu))) return error.Inkling2StepDtype;
        break :blk .{try activation(g, gu, sel, if (L.gate_up_scale.ctx != null) L.gate_up_scale else caps, caps, L.gate_up_scale.ctx != null)};
    };

    // The down kernel's threadgroup holds every pick of its row.
    const o = if (if (s == 1 and 32 * top_k <= sure_threads) dn_fmt else null) |fmt|
        try launchGrid(g, &downs_kernel, &.{ act[0], L.down.w, L.down.s, biasesOf(&L.down), sel, dn_scale, weights, m.residual, st, L.mlp_conv, conv_scale }, 2, tail_out, .{ 32, top_k, @divExact(hidden, bank_rows) }, .{ 32, top_k, 1 }, &.{
            .{ "T", .{ .dtype = dt } },                        .{ "WT", .{ .dtype = ops.dtype(L.mlp_conv) } },        .{ "I", .{ .int = inter } },
            .{ "D", .{ .int = hidden } },                      .{ "GS", .{ .int = @intCast(L.down.qp.group_size) } }, .{ "FMT", .{ .int = fmt } },
            .{ "ROWS", .{ .int = bank_rows } },                .{ "KSEL", .{ .int = top_k } },                        .{ "KT", .{ .int = kt } },
            .{ "BITS", .{ .int = @intCast(L.down.qp.bits) } }, .{ "S", .{ .dtype = ops.dtype(L.down.s) } },           .{ "SCALED", dn_scaled },
            .{ "TAIL", .{ .int = 1 } },
        })
    else blk: {
        const dn = try layers.gather(g, act[0], &L.down, sel, sorted);
        if (!floating(ops.dtype(dn))) return error.Inkling2StepDtype;
        // Unsorted pairs sit in (row, pick) order already.
        const inverse = pairs.inverse orelse try g.arange(0, @floatFromInt(n), 1, .uint32);
        const tree_parent: ?Array = if (open) |v| (if (v.tree) |tr| tr.parent else null) else null;
        break :blk try pickSumWith(g, dn, sel, inverse, dn_scale, weights, m.residual, st, L.mlp_conv, conv_scale, top_k, L.down_scale.ctx != null, tree_parent);
    };
    if (open) |v| {
        const nslot = ops.shape(st)[0];
        const rows = try g.astype(try g.concat(&.{ kp, vp, m.op, o[1] }, 2), .float32);
        v.window = try v.g.keep(try g.concat(&.{ st, try g.reshape(rows, &.{ nslot, @divExact(s, nslot), -1 }) }, 1));
        return .{ .h = o[0], .state = st };
    }
    return .{ .h = o[0], .state = try convState(g, st, kp, vp, m.op, o[1]) };
}

/// A visit's conv state `[1, KT - 1, .]` f32 after its rows: the last KT - 1 rows of the old state `st` followed
/// by the rows' conv inputs `[kp | vp | op | xm]`. With a state per row (`[S, KT - 1, .]`, batched slots) each moves
/// on by its own row.
pub fn convState(g: *Graph, st: Array, kp: Array, vp: Array, op: Array, xm: Array) !Array {
    const sh = ops.shape(st);
    const state = try launch(g, &state_kernel, &.{ st, kp, vp, op, xm }, 1, .{.{ .shape = &.{ sh[0], sh[1], sh[2] }, .dtype = .float32 }}, sh[0] * sh[1] * sh[2], 64, &.{.{ "KT", .{ .int = sh[1] + 1 } }});
    return state[0];
}

/// One visit of one sequence's rows `[1, S, D]`: `layers.layerForward`, or null where the kernels do not run (nothing
/// is appended then). `bank` is the last pass's bank and normed output on a later loop pass; `open` a draft verify's
/// visit. Batched slots (`c.slots` > 1) are the rows, each a sequence of its own with its own cache and conv state.
pub fn visit(c: *block.Ctx, L: *const Layer, w: *const Body, h: Array, blocks: ?Array, bank: ?[2]Array, open: ?*Open) !?layers.Post {
    const g = c.g;
    const cfg = &w.cfg;
    if (!runs(g, L, cfg, h)) return null;
    const batch = c.slots > 1 and c.rows == 1;
    const held = if (c.slots > 1) try slot_state.get(c, .conv) else try kv.state(c, .conv);
    const st = held orelse try g.zeros(&.{ c.slots, @intCast(cfg.sconv_kernel - 1), @intCast(2 * cfg.kvDim() + 2 * cfg.hidden_size) }, .float32);
    const f = try pre(g, L, w, h, blocks, bank);
    const total = @as(c_int, @intCast(c.offset())) + 1;
    const one = c.rows == 1 and !batch;
    const tree_parent: ?Array = if (open) |o| (if (o.tree) |t| t.parent else null) else null;
    const p = (try prepWithConsts(g, L, cfg, &w.consts, &f.proj, f.dyn, st, if (one and L.global) layers.stepTau(cfg, total) else 1.0, if (one) keysSeen(L, cfg, total) else 0, tree_parent)) orelse return null;
    const fuse = one and open == null and oFormat(&L.o, ops.dtype(h)) != null;
    const read = if (batch) try attendSlots(c, L, cfg, &w.consts, &p, f.proj[3]) else try attend(c, L, cfg, &p, f.proj[3], open, fuse);
    const m = try midWith(g, L, w, read, if (fuse) p.gate else null, h, f.normed, blocks, st, f.messages, tree_parent);
    const t = try expertsWithConsts(g, L, cfg, &w.consts, &m, f.proj[4], f.proj[5], st, open);
    if (open == null) {
        if (batch) try slot_state.put(c, .conv, t.state) else try kv.setState(c, .conv, t.state);
    }
    return .{ .h = t.h, .blocks = m.blocks };
}

/// Per-lane sampling parameters for GPU-side tree selection + acceptance (`sampleAndAcceptTree`).
pub const SampleLane = struct {
    /// <= 0 = greedy argmax; > 0 = sampled at this temperature.
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
    seed: u64 = 0,
    accept_cap: u32 = 15,
};

/// Optional sparse-16 draft tree proposal inputs for sampled multi-child tree rejection.
pub const SampleTreeIn = struct {
    tmeta: Array,
    cands: Array,
    next_q: Array,
    draws: Array,
    r: c_int,
    k: c_int,
};

/// Fused GPU-side token selection (greedy argmax or exact temperature/top-k/top-p/min-p + multi-child sparse-16
/// rejection sampling) + tree walk for `B` lanes of `w` verify rows each over `logits` `[B * w, stride]`.
/// Returns `RESULT [B, 2 + 2 * w]` int32: `{ accepted, next_token, path_rows[0..w-1], emitted_tokens[0..w-1] }`.
pub fn sampleAndAcceptTree(
    g: *Graph,
    logits: Array,
    vocab: c_int,
    suppress_mask: Array,
    tokens: Array,
    parent: Array,
    lanes: []const SampleLane,
    w: c_int,
    stops: []const u32,
    tree_in: ?SampleTreeIn,
) !Array {
    const b: c_int = @intCast(lanes.len);
    const total_rows: c_int = b * w;
    const flat_logits = try g.reshape(logits, &.{ total_rows, -1 });
    const stride = ops.shape(flat_logits)[1];
    const dt = ops.dtype(flat_logits);
    const has_mask = suppress_mask.ctx != null;
    const mask = if (has_mask) suppress_mask else try g.zeros(&.{1}, .bool_);

    var lp: [max_slots * 4]f32 = undefined;
    var seeds: [max_slots * 2]u32 = undefined;
    var caps: [max_slots]u32 = undefined;
    for (lanes, 0..) |l, i| {
        lp[i * 4 + 0] = l.temperature;
        lp[i * 4 + 1] = @floatFromInt(l.top_k);
        lp[i * 4 + 2] = l.top_p;
        lp[i * 4 + 3] = l.min_p;
        seeds[i * 2 + 0] = @truncate(l.seed);
        seeds[i * 2 + 1] = @truncate(l.seed >> 32);
        caps[i] = l.accept_cap;
    }
    const lparams = try g.fromSlice(f32, lp[0 .. lanes.len * 4], &.{ b, 4 });
    const seed_arr = try g.fromSlice(u32, seeds[0 .. lanes.len * 2], &.{ b, 2 });
    const cap_arr = try g.fromSlice(u32, caps[0..lanes.len], &.{b});

    const shards: c_int = 16;
    const pmass = (try launchGrid(g, &sample_mass_kernel, &.{ flat_logits, mask, lparams }, 1, .{.{ .shape = &.{ total_rows, shards, 4 }, .dtype = .float32 }}, .{ shards * 256, total_rows, 1 }, .{ 256, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },
        .{ "V", .{ .int = vocab } },
        .{ "STRIDE", .{ .int = stride } },
        .{ "SHARDS", .{ .int = shards } },
        .{ "W", .{ .int = w } },
        .{ "MASK", .{ .int = @intFromBool(has_mask) } },
    }))[0];

    const vrow = (try launchGrid(g, &sample_search_kernel, &.{ flat_logits, mask, lparams, pmass }, 1, .{.{ .shape = &.{ total_rows, 4 }, .dtype = .float32 }}, .{ 256, total_rows, 1 }, .{ 256, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },
        .{ "V", .{ .int = vocab } },
        .{ "STRIDE", .{ .int = stride } },
        .{ "SHARDS", .{ .int = shards } },
        .{ "W", .{ .int = w } },
        .{ "MASK", .{ .int = @intFromBool(has_mask) } },
    }))[0];

    const has_tree = tree_in != null;
    const ti: SampleTreeIn = tree_in orelse .{
        .tmeta = try g.zeros(&.{ total_rows, 2 }, .int32),
        .cands = try g.zeros(&.{ 1, 1, 1 }, .int32),
        .next_q = try g.zeros(&.{ 1, 1, 1 }, .float32),
        .draws = try g.zeros(&.{ total_rows, 4 }, .uint8),
        .r = 1,
        .k = 1,
    };
    const selected = (try launchGrid(g, &sample_draw_kernel, &.{ flat_logits, mask, lparams, vrow, seed_arr, ti.tmeta, ti.cands, ti.next_q, ti.draws }, 1, .{.{ .shape = &.{total_rows}, .dtype = .int32 }}, .{ 256, total_rows, 1 }, .{ 256, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },
        .{ "V", .{ .int = vocab } },
        .{ "STRIDE", .{ .int = stride } },
        .{ "W", .{ .int = w } },
        .{ "MASK", .{ .int = @intFromBool(has_mask) } },
        .{ "TREE", .{ .int = @intFromBool(has_tree) } },
        .{ "R", .{ .int = ti.r } },
        .{ "K", .{ .int = ti.k } },
    }))[0];

    var stop_buf: [16]i32 = @splat(-1);
    const n_stops: usize = @min(stops.len, stop_buf.len);
    for (stops[0..n_stops], stop_buf[0..n_stops]) |s_id, *dst| dst.* = @intCast(s_id);
    const stop_arr = try g.fromSlice(i32, &stop_buf, &.{@intCast(stop_buf.len)});
    const nstop_arr = try g.fromSlice(u32, &.{@intCast(n_stops)}, &.{1});
    return (try launchGrid(g, &accept_tree_kernel, &.{ tokens, parent, selected, cap_arr, stop_arr, nstop_arr }, 1, .{.{ .shape = &.{ b, 2 + 2 * w }, .dtype = .int32 }}, .{ b, 1, 1 }, .{ 1, 1, 1 }, &.{
        .{ "W", .{ .int = w } },
    }))[0];
}

// ── tests ──

// Bar: a launch wider than the pipeline takes is refused as `ThreadLimit` at its first dispatch with no MLX error
// left behind, and the stream group narrows through the widths that tile the stream until none is left.
test "inkling2 kernels: a threadgroup past the pipeline's limit is refused by name, and the stream group narrows" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    mlx.installErrorHandler();
    var g = Graph.init(a, mlx.gpuStream());
    defer g.deinit();
    const n = 4096;
    const x = try g.ones(&.{n}, .float32);
    const out: [1]Out = .{.{ .shape = &.{n}, .dtype = .float32 }};
    const targs: TArgs = &.{.{ "T", .{ .dtype = .float32 } }};
    // 2048 threads a group: past every Apple GPU.
    try std.testing.expectError(error.ThreadLimit, launch(&g, &gated_kernel, &.{ x, x }, 1, out, n, 2048, targs));
    try std.testing.expect(!mlx.errorPending());
    const y = try launch(&g, &gated_kernel, &.{ x, x }, 1, out, n, 64, targs);
    try ops.eval(y[0]);

    const saved = stream_threads;
    defer stream_threads = saved;
    stream_threads = 768;
    var seen: [8]c_int = undefined;
    var count: usize = 0;
    while (true) {
        seen[count] = streamGroup(2304).?;
        count += 1;
        if (count == seen.len or !narrow(2304)) break;
    }
    try std.testing.expectEqualSlices(c_int, &.{ 768, 576, 384, 288, 256, 192, 128, 96 }, seen[0..count]);
    stream_threads = 64;
    try std.testing.expect(streamGroup(64).? == 64 and narrow(64) and streamGroup(64).? == 32 and !narrow(64));
}
