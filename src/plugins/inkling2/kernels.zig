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
const gather_qmm_nax = @import("../../sdk/blocks/moe/gather_qmm_nax.zig");
const verifyQmmNaxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
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

const Kernel = struct {
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
pub const bank_grouped_header = bank_header ++ kernelSource("plugins/inkling2/kernels/bank_grouped_header.metal");
/// Output rows a SIMD group of a packed expert kernel owns (one: a lane's loop sets the launch's latency, and here
/// that beats keeping x in registers across rows), and the up kernel's SIMD groups per threadgroup (each threadgroup
/// ranks its row's picks: fewer, wider ones rank fewer times).
const bank_rows: c_int = 1;
const bank_groups: c_int = 8;

pub var pre_kernel: Kernel = .{ .name = "msv_inkling2_pre", .ins = &.{ "HIN", "BLK", "QY", "NW0", "NW1", "NW2", "BANK", "BPART", "MQ", "PR" }, .outs = &.{ "X", "NORMED", "A", "MSGO", "BOTH" }, .source = kernelSource("plugins/inkling2/kernels/pre.metal") };
pub var prep_kernel: Kernel = .{ .name = "msv_inkling2_prep", .ins = &.{ "QP", "GP", "RP", "KP", "VP", "KDY", "VDY", "ST", "KW", "VW", "QS", "RS", "RJ", "PR" }, .outs = &.{ "Q", "GATE", "K", "V", "BIAS", "REL" }, .source = kernelSource("plugins/inkling2/kernels/prep.metal") };
pub var oproj_kernel: Kernel = .{ .name = "msv_inkling2_oproj", .ins = &.{ "ATT", "GATE", "WQ", "WS", "WB" }, .outs = &.{"Y"}, .source = kernelSource("plugins/inkling2/kernels/oproj.metal"), .header = bank_header };
pub var proj_rows_kernel: Kernel = .{ .name = "msv_inkling2_proj_rows", .ins = &.{ "X", "WQ", "WS", "WB" }, .outs = &.{"Y"}, .source = kernelSource("plugins/inkling2/kernels/proj_rows.metal"), .header = bank_grouped_header };
pub var gated_kernel: Kernel = .{ .name = "msv_inkling2_gated", .ins = &.{ "ATT", "GATE" }, .outs = &.{"OUT"}, .source = kernelSource("plugins/inkling2/kernels/gated.metal") };
pub var mid_kernel: Kernel = .{ .name = "msv_inkling2_mid", .ins = &.{ "OP", "ST", "AW", "HIN", "NORMED", "BLK", "QY", "PN", "MSG3", "PR" }, .outs = &.{ "RES", "X", "X32", "M", "BOTH" }, .source = kernelSource("plugins/inkling2/kernels/mid.metal") };
pub var gate_kernel: Kernel = .{ .name = "msv_inkling2_gate", .ins = &.{ "A", "MS", "ST", "F", "G" }, .outs = &.{"Y"}, .source = kernelSource("plugins/inkling2/kernels/gate.metal") };
pub var norm_kernel: Kernel = .{ .name = "msv_inkling2_norm", .ins = &.{ "Y", "NW0", "NW1", "NW2", "RT", "PR" }, .outs = &.{ "X", "SCORES" }, .source = kernelSource("plugins/inkling2/kernels/norm.metal") };
pub var topk_kernel: Kernel = .{ .name = "msv_inkling2_topk", .ins = &.{ "SC", "RB", "PR" }, .outs = &.{ "IDX", "WTS" }, .source = kernelSource("plugins/inkling2/kernels/topk.metal") };
pub var act_kernel: Kernel = .{ .name = "msv_inkling2_act", .ins = &.{ "GU", "IDX", "SC", "PR" }, .outs = &.{"ACT"}, .source = kernelSource("plugins/inkling2/kernels/act.metal") };
pub var ups_kernel: Kernel = .{ .name = "msv_inkling2_ups", .ins = &.{ "X", "WQ", "WS", "WB", "SCR", "RB", "RS", "IDX", "SC", "PR" }, .outs = &.{ "ACT", "PIDX", "PWTS" }, .source = kernelSource("plugins/inkling2/kernels/ups.metal"), .header = bank_header };
pub var downs_kernel: Kernel = .{ .name = "msv_inkling2_downs", .ins = &.{ "ACT", "WQ", "WS", "WB", "IDX", "SC", "WTS", "RES", "ST", "MW", "PR" }, .outs = &.{ "HOUT", "XM" }, .source = kernelSource("plugins/inkling2/kernels/downs.metal"), .header = bank_header };
pub var group_experts_kernel: Kernel = .{ .name = "msv_inkling2_group_experts", .ins = &.{"IDX"}, .outs = &.{ "ACT_E", "ACT_CNT", "ACT_PAIR" }, .source = kernelSource("plugins/inkling2/kernels/group_experts.metal") };
pub var ups_grouped_kernel: Kernel = .{ .name = "msv_inkling2_ups_grouped", .ins = &.{ "X", "WQ", "WS", "WB", "ACT_E", "ACT_CNT", "ACT_PAIR", "SC", "PR" }, .outs = &.{"ACT"}, .source = kernelSource("plugins/inkling2/kernels/ups_grouped.metal"), .header = bank_grouped_header };
pub var downs_grouped_kernel: Kernel = .{ .name = "msv_inkling2_downs_grouped", .ins = &.{ "ACT", "WQ", "WS", "WB", "ACT_E", "ACT_CNT", "ACT_PAIR" }, .outs = &.{"DN"}, .source = kernelSource("plugins/inkling2/kernels/downs_grouped.metal"), .header = bank_grouped_header };
pub var relattn_kernel: Kernel = .{ .name = "msv_inkling2_relattn", .ins = &.{ "q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale" }, .outs = &.{"out"}, .source = kernelSource("plugins/inkling2/kernels/relattn.metal"), .contiguous = false };
pub var relattn_nax_kernel: Kernel = .{ .name = "msv_inkling2_relattn_nax", .ins = &.{ "q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale" }, .outs = &.{"out"}, .source = kernelSource("plugins/inkling2/kernels/relattn_nax.metal"), .header = kernelSource("kernels/common/mpp_tensor_ops.metal"), .contiguous = false };
pub var router_kernel: Kernel = .{ .name = "msv_inkling2_router", .ins = &.{ "X", "RT" }, .outs = &.{"SC"}, .source = kernelSource("plugins/inkling2/kernels/router.metal") };
pub var relsplit_kernel: Kernel = .{ .name = "msv_inkling2_relsplit", .ins = &.{ "q", "k", "v", "rel", "sinks", "sizes", "scale" }, .outs = &.{ "PART", "ML" }, .source = kernelSource("plugins/inkling2/kernels/relsplit.metal"), .contiguous = false };
pub var relsplit_open_kernel: Kernel = .{ .name = "msv_inkling2_relsplit_open", .ins = &.{ "q", "kc", "vc", "kn", "vn", "rel", "sinks", "sizes", "scale" }, .outs = &.{ "PART", "ML" }, .source = kernelSource("plugins/inkling2/kernels/relsplit_open.metal"), .contiguous = false };
pub var relcomb_kernel: Kernel = .{ .name = "msv_inkling2_relcomb", .ins = &.{ "PART", "ML", "GATE" }, .outs = &.{"OUT"}, .source = kernelSource("plugins/inkling2/kernels/relcomb.metal") };
/// The Markov kernels' argmax order.
const markov_header = kernelSource("plugins/inkling2/kernels/markov_header.metal");
pub var markov_rnn_kernel: Kernel = .{ .name = "msv_inkling2_markov_rnn", .ins = &.{ "W", "B", "ST", "PREV", "ROWS", "NEW", "TOK", "AT" }, .outs = &.{"P"}, .source = kernelSource("plugins/inkling2/kernels/markov_rnn.metal") };
pub var markov_step_rnn_kernel: Kernel = .{ .name = "msv_inkling2_markov_step_rnn", .ins = &.{ "W", "KAPPA", "ST", "PREV", "TOK", "AT" }, .outs = &.{"P"}, .source = kernelSource("plugins/inkling2/kernels/markov_step_rnn.metal") };
pub var markov_fold_kernel: Kernel = .{ .name = "msv_inkling2_markov_fold", .ins = &.{ "P", "FOLD", "FB", "BASE", "AT", "CAP" }, .outs = &.{ "PM", "PI", "PS" }, .source = kernelSource("plugins/inkling2/kernels/markov_fold.metal"), .header = markov_header };
pub var markov_fold_q8_kernel: Kernel = .{ .name = "msv_inkling2_markov_fold_q8", .ins = &.{ "P", "FQ", "FSC", "BASE", "AT", "CAP" }, .outs = &.{ "PM", "PI", "PS" }, .source = kernelSource("plugins/inkling2/kernels/markov_fold_q8.metal"), .header = markov_header };
pub var markov_pick_kernel: Kernel = .{ .name = "msv_inkling2_markov_pick", .ins = &.{ "PM", "PI", "PS", "P", "ST", "CAP" }, .outs = &.{ "TOKN", "CONF", "NST" }, .source = kernelSource("plugins/inkling2/kernels/markov_pick.metal"), .header = markov_header };
pub var relslots_kernel: Kernel = .{ .name = "msv_inkling2_relslots", .ins = &.{ "q", "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "rel", "sinks", "KEYS", "scale" }, .outs = &.{ "PART", "ML" }, .source = kernelSource("plugins/inkling2/kernels/relslots.metal"), .contiguous = false };
pub var relmask_kernel: Kernel = .{ .name = "msv_inkling2_relmask", .ins = &.{ "rel", "sinks", "sizes" }, .outs = &.{"M"}, .source = kernelSource("plugins/inkling2/kernels/relmask.metal") };
pub var sum_kernel: Kernel = .{ .name = "msv_inkling2_sum", .ins = &.{ "DN", "IDX", "INV", "SC", "WTS", "RES", "ST", "MW", "PR" }, .outs = &.{ "HOUT", "XM" }, .source = kernelSource("plugins/inkling2/kernels/sum.metal") };
pub var state_kernel: Kernel = .{ .name = "msv_inkling2_state", .ins = &.{ "ST", "KP", "VP", "OP", "XM" }, .outs = &.{"NST"}, .source = kernelSource("plugins/inkling2/kernels/state.metal") };

pub const enabled = @import("config.zig").fusedOn;

const TArg = union(enum) { dtype: Dtype, int: c_int };
const Out = struct { shape: []const c_int, dtype: Dtype };
/// An output a kernel variant never writes.
const unused: Out = .{ .shape = &.{1}, .dtype = .float32 };

const TArgs = []const struct { [*:0]const u8, TArg };

/// `outs` of `k` over `grid` threads in threadgroups of `tg`, owned by `g`.
fn launch(g: *Graph, k: *Kernel, inputs: []const Array, comptime n: usize, outs: [n]Out, grid: c_int, tg: c_int, targs: TArgs) ![n]Array {
    return launchGrid(g, k, inputs, n, outs, .{ grid, 1, 1 }, .{ tg, 1, 1 }, targs);
}

fn launchGrid(g: *Graph, k: *Kernel, inputs: []const Array, comptime n: usize, outs: [n]Out, grid: [3]c_int, tg: [3]c_int, targs: TArgs) ![n]Array {
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
var probed: [256]u64 = @splat(0);
var probed_n: usize = 0;

/// Whether this instantiation's first dispatch still has to show its pipeline takes the launch.
fn unproven(k: *const Kernel, tg: [3]c_int, targs: TArgs) ?u64 {
    if (tg[0] * tg[1] * tg[2] <= sure_threads) return null;
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
fn streams(g: *Graph, cfg: *const Config, h: Array, blocks: ?Array, query: Array, nw: [3]Array, inputs: c_int, msg: ?Messaged, eps_c: Array) ![5]Array {
    const d: c_int = @intCast(cfg.hidden_size);
    const s = ops.shape(h)[1];
    const dt = ops.dtype(h);
    const q: c_int = if (query.ctx != null) ops.shape(query)[1] else 0;
    const eps = if (eps_c.ctx != null) eps_c else try g.fromSlice(f32, &.{cfg.rms_norm_eps}, &.{1});
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
    const projected = s > small_rows;
    if (projected) {
        const st32 = try g.astype(st, .float32);
        forget = try g.matmul(st32, try g.transpose(try g.astype(forget, .float32), &.{ 0, 2, 1 }));
        message = try g.matmul(st32, try g.transpose(try g.astype(message, .float32), &.{ 0, 2, 1 }));
    }
    const coop = !projected and @rem(sh[2], 8) == 0;
    const grid_x = if (coop) n * s * sh[1] * 8 else n * s * sh[1];
    const o = try launch(g, &gate_kernel, &.{ a, ms, st, forget, message }, 1, .{.{ .shape = &.{ n, s, sh[1] }, .dtype = .float32 }}, grid_x, 64, &.{
        .{ "D", .{ .int = sh[1] } }, .{ "RANK", .{ .int = sh[2] } }, .{ "PROJ", .{ .int = @intFromBool(projected) } }, .{ "COOP", .{ .int = @intFromBool(coop) } },
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
    return normWithEps(g, cfg, y, nw, n, dt, router, .{});
}

fn normWithEps(g: *Graph, cfg: *const Config, y: Array, nw: [3]Array, n: c_int, dt: Dtype, router: ?Array, eps_c: Array) ![2]Array {
    const d: c_int = @intCast(cfg.hidden_size);
    const s = ops.shape(y)[1];
    const experts_n: c_int = if (router != null) @intCast(cfg.num_experts) else 0;
    const eps = if (eps_c.ctx != null) eps_c else try g.fromSlice(f32, &.{cfg.rms_norm_eps}, &.{1});
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

/// `layers.project`. `bank` is the last pass's bank and normed output on a later loop pass.
pub fn pre(g: *Graph, L: *const Layer, w: *const Body, h: Array, blocks: ?Array, bank: ?[2]Array) !Fore {
    const cfg = &w.cfg;
    const dt = ops.dtype(h);
    var x: Array = undefined;
    var normed: Array = undefined;
    var messages: ?Array = null;
    if (bank) |bk| {
        const gates = &(L.gates orelse return error.Inkling2NoMessageGate);
        const o = try streams(g, cfg, h, blocks, L.attn_query, L.in_norm, 3, .{ .bank = bk[0], .partial = bk[1], .query = L.message_query, .gt = ops.dtype(gates.down) }, w.consts.eps);
        const st = try g.matmul(o[4], try layers.downs(g, gates, 0, 3));
        x = (try normWithEps(g, cfg, try gate(g, gates, o[2], o[3], st, 0, 3), L.in_norm, 3, dt, null, w.consts.eps))[0];
        normed = o[1];
        messages = o[3];
    } else {
        const o = try streams(g, cfg, h, blocks, L.attn_query, L.in_norm, if (L.global) 3 else 1, null, w.consts.eps);
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
    return (try streams(g, &w.cfg, h, blocks, w.final_query, @splat(norm_weight), 1, null, w.consts.eps))[0];
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
    return prepWithConsts(g, L, cfg, null, proj, taps, st, tau, keys);
}

fn prepWithConsts(g: *Graph, L: *const Layer, cfg: *const Config, consts: ?*const layers.Consts, proj: *const [6]Array, taps: ?[2]Array, st: Array, tau: f32, keys: c_int) !?Prep {
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
    const cached_pr: Array = if (consts) |cs|
        (if (wide and tau == 1.0 and keys == 0)
            (if (dynamic) cs.prep_pr_dynamic else cs.prep_pr_static)
        else if (!wide and !dynamic and tau == 1.0 and keys == @as(c_int, @intCast(cfg.sliding_window)))
            cs.prep_pr_sliding_full
        else
            .{})
    else
        .{};
    const pr = if (cached_pr.ctx != null) cached_pr else try g.fromSlice(f32, &.{ cfg.qk_clip, cfg.rel_clip, cfg.rms_norm_eps, tau, conv_scale, @floatFromInt(keys) }, &.{6});
    const o = try launchGrid(g, &prep_kernel, &.{ proj[0], proj[1], proj[2], proj[4], proj[5], dyn[0], dyn[1], st, kw, vw, L.q_scale, L.rel_scale, L.rel_proj_rev, pr }, 6, .{
        .{ .shape = if (wide) &.{ 1, s, heads, hd } else &.{ 1, heads, 1, hd }, .dtype = dt },
        .{ .shape = &.{s * heads * hd}, .dtype = .float32 },
        .{ .shape = &.{ 1, kv_heads, s, hd }, .dtype = dt },
        .{ .shape = &.{ 1, kv_heads, s, hd }, .dtype = dt },
        if (wide) unused else .{ .shape = &.{ 1, heads, 1, keys }, .dtype = dt },
        if (wide) .{ .shape = &.{ 1, s, heads, d_rel }, .dtype = dt } else unused,
    }, .{ heads * hd + 2 * kv_dim + heads * (if (wide) d_rel else keys), s, 1 }, .{ hd, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },                  .{ "WT", .{ .dtype = ops.dtype(kw) } }, .{ "S", .{ .dtype = ops.dtype(L.q_scale) } },
        .{ "H", .{ .int = heads } },                 .{ "HD", .{ .int = hd } },              .{ "HKV", .{ .int = kv_heads } },
        .{ "DREL", .{ .int = d_rel } },              .{ "KT", .{ .int = kt } },              .{ "DYN", .{ .int = @intFromBool(dynamic) } },
        .{ "WIDE", .{ .int = @intFromBool(wide) } }, .{ "SEQ", seqOf(st) },
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
        .{ "T", .{ .dtype = dt } },               .{ "S", .{ .dtype = if (affine) ops.dtype(o.s) else dt } }, .{ "K", .{ .int = f.k } },
        .{ "GS", .{ .int = @intCast(f.gs) } },    .{ "BITS", .{ .int = @intCast(f.bits) } },                    .{ "FMT", .{ .int = f.fmt } },
    });
    return y[0];
}

/// `proj_rows.metal`: multi-row quantized projection for `2..8` rows (2 output rows per SIMD group).
pub fn projRows(g: *Graph, w: *const layers.Linear, x: Array) !?Array {
    const dt = ops.dtype(x);
    if (!mlx.streamIsGpu(g.s) or !enabled() or (dt != .bfloat16 and dt != .float16)) return null;
    if (w.s.ctx == null or w.bias.ctx != null or w.rht != null or w.ext != null or w.ternary or ops.dtype(w.w) != .uint32) return null;
    const xsh = ops.shape(x);
    if (xsh.len != 3 or (xsh[0] != 1 and xsh[1] != 1)) return null;
    const m = xsh[0] * xsh[1];
    const k = xsh[2];
    if (m < 2 or m > 8 or @rem(k, 32) != 0) return null;
    const wsh = ops.shape(w.w);
    if (wsh.len != 2) return null;
    const n = wsh[0];
    if (@rem(n, 2) != 0) return null;
    const fp = ops.dtype(w.s) == .uint8;
    const fmt: c_int = switch (w.mode) {
        .nvfp4 => if (fp and w.bits == 4 and w.group_size == 16) 0 else return null,
        .mxfp4 => if (fp and w.bits == 4 and w.group_size == 32) 1 else return null,
        .mxfp8 => if (fp and w.bits == 8 and w.group_size == 32) 2 else return null,
        .affine => if (!fp and w.b.ctx != null and ops.dtype(w.b) == ops.dtype(w.s) and floating(ops.dtype(w.s)) and affineWidth(w.bits, w.group_size)) 3 else return null,
        else => return null,
    };
    const half_n = @divExact(n, 2);
    const sgs: c_int = if (@rem(half_n, 4) == 0) 4 else if (@rem(half_n, 2) == 0) 2 else 1;
    const y = try launchGrid(g, &proj_rows_kernel, &.{ x, w.w, w.s, if (w.b.ctx != null) w.b else w.s }, 1, .{.{ .shape = &.{ xsh[0], xsh[1], n }, .dtype = dt }}, .{ 32, half_n, 1 }, .{ 32, sgs, 1 }, &.{
        .{ "T", .{ .dtype = dt } },
        .{ "S", .{ .dtype = if (fmt == 3) ops.dtype(w.s) else dt } },
        .{ "K", .{ .int = k } },
        .{ "N", .{ .int = n } },
        .{ "GS", .{ .int = @intCast(w.group_size) } },
        .{ "BITS", .{ .int = @intCast(w.bits) } },
        .{ "FMT", .{ .int = fmt } },
        .{ "M", .{ .int = m } },
    });
    return y[0];
}

/// Fast multi-row dispatch for Inkling 2's 8-bit affine projections:
/// - `M = 2..5`: register-only `proj_rows.metal` (2 rows/SG, coalesced uint4 loads, zero barriers).
/// - `M = 6..16` on NAX hardware: direct `mlx_quantized_matmul` (MLX 0.32.5 `qmv_nax` beats `vqmm_nax` at 8-bit).
pub fn fastProjectRows(g: *Graph, w: *const layers.Linear, x: Array) !?Array {
    if (w.s.ctx == null or w.b.ctx == null or w.bias.ctx != null or w.rht != null or w.ext != null or w.ternary) return null;
    if (w.mode != .affine or w.bits != 8 or !mlx.streamIsGpu(g.s) or !enabled()) return null;
    const dt = ops.dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const xsh = ops.shape(x);
    if (xsh.len != 3 or (xsh[0] != 1 and xsh[1] != 1)) return null;
    const m = xsh[0] * xsh[1];
    if (m < 2 or m > 16) return null;
    if (m <= 5) return projRows(g, w, x);
    const wsh = ops.shape(w.w);
    if (wsh.len != 2) return null;
    const n = wsh[0];
    const k = xsh[2];
    if (verifyQmmNaxAvailable() and n >= 512 and k >= 1024 and @rem(n, 32) == 0 and @rem(k, 128) == 0) {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_quantized_matmul(
            &out,
            x,
            w.w,
            w.s,
            w.b,
            true,
            mlx.mlx_optional_int.some(@intCast(w.group_size)),
            mlx.mlx_optional_int.some(@intCast(w.bits)),
            "affine",
            g.s,
        ));
        return try g.adopt(out);
    }
    return null;
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

/// `relattn_nax.metal`: Metal 4 NAX 16x32x16 flash attention over `[1, H, L, 128]` queries and `[1, HKV, S, 128]` keys/values.
pub fn relAttnNax(g: *Graph, q: Array, k: Array, v: Array, bias: Array, sinks: Array, key_start: Array, sizes: Array, scale: Array, window: c_int) !Array {
    const dt = ops.dtype(q);
    const heads = ops.shape(q)[1];
    const s = ops.shape(q)[2];
    const hd = ops.shape(q)[3];
    const o = try launchGrid(g, &relattn_nax_kernel, &.{ q, k, v, bias, sinks, key_start, sizes, scale }, 1, .{.{ .shape = &.{ 1, s, heads, hd }, .dtype = dt }}, .{ @divTrunc(s + 63, 64) * 128, heads, 1 }, .{ 128, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },              .{ "D", .{ .int = hd } },
        .{ "HQ", .{ .int = heads } },            .{ "HKV", .{ .int = ops.shape(k)[1] } },
        .{ "R", .{ .int = ops.shape(bias)[3] } }, .{ "WINDOW", .{ .int = window } },
        .{ "BQ", .{ .int = 64 } },               .{ "TG", .{ .int = 128 } },
    });
    return o[0];
}

/// `layers.wideRead`: the rows `q` `[1, S, H, hd]` read `k` and `v` (the view after the append, the rows at its last
/// S keys) under the bias of their clipped relative `states` `[1, S, H, d_rel]` and their `sinks` `[1, S, H]`:
/// `[1, S, H, hd]`. `offset` is the first row's position. One flash launch with no mask built, or (at most
/// `small_rows` rows, unless `flash`) a mask launch and MLX's attention.
pub fn relative(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, k: Array, v: Array, offset: c_int, flash: bool) !Array {
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
    const key_start = try g.zeros(&.{1}, .int32);
    const sizes = try g.fromSlice(i32, &.{ s, keys }, &.{2});
    const sc = try g.scalar(scale, .float32);
    if (verifyQmmNaxAvailable() and (dt == .bfloat16 or dt == .float16) and hd == 128) {
        return relAttnNax(g, q, k, v, bias, sink_rows, key_start, sizes, sc, window);
    }
    const o = try launchGrid(g, &relattn_kernel, &.{ q, k, v, bias, sink_rows, key_start, sizes, sc }, 1, .{.{ .shape = &.{ 1, s, heads, hd }, .dtype = dt }}, .{ @divTrunc(s + 31, 32) * 128, heads, 1 }, .{ 128, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } },                     .{ "D", .{ .int = hd } },
        .{ "HQ", .{ .int = heads } },                   .{ "HKV", .{ .int = @intCast(cfg.num_key_value_heads) } },
        .{ "R", .{ .int = ops.shape(L.rel_proj)[1] } }, .{ "WINDOW", .{ .int = window } },
        .{ "BQ", .{ .int = 32 } },                      .{ "TG", .{ .int = 128 } },
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

/// Whether `markovStepQ8` serves a drafter with pre-split `rnn_step` `[3R, 2R]` and per-row Q8 `fold_q` `[V, R]`.
pub fn markovQ8Runs(g: *const Graph, rnn_step: Array, fold_q: Array, rank: c_int) bool {
    const dt = ops.dtype(rnn_step);
    if (!mlx.streamIsGpu(g.s) or !enabled() or (dt != .bfloat16 and dt != .float16) or ops.dtype(fold_q) != .int8) return false;
    return @rem(rank, 4) == 0 and @rem(3 * rank, 8) == 0;
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

/// Fast Q8 Markov step: precomputed static `kappa` `[steps, 3R]` f32 + coalesced Q8 `fold_q` `[V, R]` int8.
pub fn markovStepQ8(g: *Graph, rnn_step: Array, kappa: Array, state: Array, prev_rows: Array, token: Array, fold_q: Array, fold_sc: Array, base_b: Array, at: c_int, vocab: c_int, softcap: f32) !MarkovStep {
    const step = try g.fromSlice(i32, &.{at}, &.{1});
    const cap = try g.fromSlice(f32, &.{softcap}, &.{1});
    const p = try markovStepRnn(g, rnn_step, kappa, state, prev_rows, token, step);
    const o = try markovPick(g, try markovFoldQ8(g, p, fold_q, fold_sc, base_b, step, vocab, cap, ops.dtype(rnn_step)), p, state, cap);
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

/// `markov_step_rnn.metal`: the step's RNN projection `[3 R]` f32 over `rnn_step` `[3 R, 2 R]` + precomputed `kappa`.
pub fn markovStepRnn(g: *Graph, rnn_step: Array, kappa: Array, state: Array, prev_rows: Array, token: Array, step: Array) !Array {
    const rank = @divExact(ops.shape(rnn_step)[0], 3);
    const o = try launch(g, &markov_step_rnn_kernel, &.{ rnn_step, kappa, state, prev_rows, token, step }, 1, .{.{ .shape = &.{3 * rank}, .dtype = .float32 }}, 32 * 3 * rank, 256, &.{
        .{ "T", .{ .dtype = ops.dtype(rnn_step) } }, .{ "R", .{ .int = rank } },
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

/// `markov_fold_q8.metal`: coalesced Q8 fold over 64 vocab rows per threadgroup.
pub fn markovFoldQ8(g: *Graph, p: Array, fold_q: Array, fold_sc: Array, base_b: Array, step: Array, vocab: c_int, cap: Array, dt: Dtype) ![3]Array {
    const groups = @divTrunc(vocab + 63, 64);
    return launch(g, &markov_fold_q8_kernel, &.{ p, fold_q, fold_sc, base_b, step, cap }, 3, .{
        .{ .shape = &.{groups}, .dtype = .float32 }, .{ .shape = &.{groups}, .dtype = .uint32 }, .{ .shape = &.{groups}, .dtype = .float32 },
    }, groups * 256, 256, &.{
        .{ "T", .{ .dtype = dt } }, .{ "R", .{ .int = ops.shape(fold_q)[1] } }, .{ "VOCAB", .{ .int = vocab } },
    });
}

/// `markov_pick.metal`: the step's token `[1]` int32, its probability `[1]` f32 and the next state `[1, 1, R]`.
pub fn markovPick(g: *Graph, part: [3]Array, p: Array, state: Array, cap: Array) ![3]Array {
    const rank = @divExact(ops.shape(p)[0], 3);
    return launch(g, &markov_pick_kernel, &.{ part[0], part[1], part[2], p, state, cap }, 3, .{
        .{ .shape = &.{1}, .dtype = .int32 }, .{ .shape = &.{1}, .dtype = .float32 }, .{ .shape = &.{ 1, 1, rank }, .dtype = ops.dtype(state) },
    }, 256, 256, &.{ .{ "T", .{ .dtype = ops.dtype(state) } }, .{ "R", .{ .int = rank } } });
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
    const rp = if (dt == .bfloat16 and L.rel_proj_bf16.ctx != null) L.rel_proj_bf16 else try g.astype(L.rel_proj, dt);
    return .{ .q = q, .bias = try g.matmul(states, rp), .sinks = try g.contiguous(try g.astype(sinks, .float32)) };
}

/// Rows from which a read behind a cache splits over the keys (`splitRead`): below it MLX's attention under a mask is
/// as fast; up to `split_rows`, the kernel's one 8-row fragment.
pub const split_from = 4;
pub const split_rows = 8;
/// Keys a threadgroup of the split read takes.
const split_keys = 128;

/// `gated(relative(..))` for `split_from` to `split_rows` half-exact rows (bf16, f16): the keys split over
/// threadgroups that each read a K/V tile once for every query head of its KV head, then the splits combined under
/// the output gate `gate_sigmoid` `[S * H * hd]` f32: `[1, S, H * hd]`. Null where it does not serve the read.
pub fn splitRead(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, k: Array, v: Array, offset: c_int, gate_sigmoid: Array) !?Array {
    const dt = ops.dtype(q_rows);
    const s = ops.shape(q_rows)[1];
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const group = @divTrunc(heads, kv_heads);
    if (s < split_from or s > split_rows or (dt != .bfloat16 and dt != .float16) or ops.dtype(k) != dt or ops.dtype(v) != dt) return null;
    if (@rem(hd, 8) != 0 or @rem(heads, kv_heads) != 0 or 32 * group > sure_threads) return null;
    const in = try readInputs(g, L, cfg, q_rows, rel_states, sinks, offset);
    const window: c_int = if (L.global) 0 else @intCast(cfg.sliding_window);
    return try splitJoin(g, try splitParts(g, in.q, k, v, in.bias, in.sinks, window), gate_sigmoid, s, dt);
}

/// `relsplit.metal` over q `[1, H, S, hd]`, k, v `[1, Hkv, keys, hd]`, the bias rows `[1, S, H, R]` and the sinks
/// `[1, S, H]` f32: each split's unnormalised sums `[splits, S * H, hd]` and (max, sum) `[splits, S * H, 2]`, f32.
pub fn splitParts(g: *Graph, q: Array, k: Array, v: Array, bias: Array, sinks: Array, window: c_int) ![2]Array {
    const heads = ops.shape(q)[1];
    const s = ops.shape(q)[2];
    const hd = ops.shape(q)[3];
    const kv_heads = ops.shape(k)[1];
    const keys = ops.shape(k)[2];
    const group = @divTrunc(heads, kv_heads);
    // The first row's oldest visible key: a split before it would read nothing.
    const first: c_int = if (window > 0) @max(0, keys - s - window + 1) else 0;
    const splits = @divTrunc(keys - first + split_keys - 1, split_keys);
    return launchGrid(g, &relsplit_kernel, &.{ q, k, v, bias, sinks, try g.fromSlice(i32, &.{ s, keys, first }, &.{3}), try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32) }, 2, .{
        .{ .shape = &.{ splits, s * heads, hd }, .dtype = .float32 },
        .{ .shape = &.{ splits, s * heads, 2 }, .dtype = .float32 },
    }, .{ splits * 32 * group, kv_heads, 1 }, .{ 32 * group, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(q) } },     .{ "D", .{ .int = hd } },
        .{ "HQ", .{ .int = heads } },             .{ "HKV", .{ .int = kv_heads } },
        .{ "R", .{ .int = ops.shape(bias)[3] } }, .{ "WINDOW", .{ .int = window } },
        .{ "KPS", .{ .int = split_keys } },
    });
}

/// Zero-copy `splitRead` for an open verify block (`1..split_rows` rows): reads cached `(kc, vc)` directly alongside
/// the new rows `(kn, vn)` without concatenating the KV cache.
pub fn splitReadOpen(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, kc: Array, vc: Array, kn: Array, vn: Array, offset: c_int, gate_sigmoid: Array) !?Array {
    return splitReadOpenWithConsts(g, L, cfg, q_rows, rel_states, sinks, kc, vc, kn, vn, offset, gate_sigmoid, null);
}

fn splitReadOpenWithConsts(g: *Graph, L: *const Layer, cfg: *const Config, q_rows: Array, rel_states: Array, sinks: Array, kc: Array, vc: Array, kn: Array, vn: Array, offset: c_int, gate_sigmoid: Array, consts: ?*const layers.Consts) !?Array {
    const dt = ops.dtype(q_rows);
    const s = ops.shape(q_rows)[1];
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const group = @divTrunc(heads, kv_heads);
    if (s < 1 or s > split_rows or (dt != .bfloat16 and dt != .float16) or ops.dtype(kc) != dt or ops.dtype(vc) != dt or ops.dtype(kn) != dt or ops.dtype(vn) != dt) return null;
    if (@rem(hd, 8) != 0 or @rem(heads, kv_heads) != 0 or 32 * group > sure_threads) return null;
    const in = try readInputs(g, L, cfg, q_rows, rel_states, sinks, offset);
    const window: c_int = if (L.global) 0 else @intCast(cfg.sliding_window);
    const sc = if (consts) |cs| (if (cs.attn_scale.ctx != null) cs.attn_scale else try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32)) else try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32);
    return try splitJoin(g, try splitPartsOpenScale(g, in.q, kc, vc, kn, vn, in.bias, in.sinks, window, sc), gate_sigmoid, s, dt);
}

/// `relsplit_open.metal`: split attention directly over cached `(kc, vc)` `[1, Hkv, C, hd]` + new `(kn, vn)` `[1, Hkv, L, hd]`.
pub fn splitPartsOpen(g: *Graph, q: Array, kc: Array, vc: Array, kn: Array, vn: Array, bias: Array, sinks: Array, window: c_int) ![2]Array {
    const hd = ops.shape(q)[3];
    return splitPartsOpenScale(g, q, kc, vc, kn, vn, bias, sinks, window, try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32));
}

fn splitPartsOpenScale(g: *Graph, q: Array, kc: Array, vc: Array, kn: Array, vn: Array, bias: Array, sinks: Array, window: c_int, scale_arr: Array) ![2]Array {
    const heads = ops.shape(q)[1];
    const s = ops.shape(q)[2];
    const hd = ops.shape(q)[3];
    const kv_heads = ops.shape(kc)[1];
    const cached = ops.shape(kc)[2];
    const total = cached + s;
    const group = @divTrunc(heads, kv_heads);
    const first: c_int = if (window > 0) @max(0, cached - (window - 1)) else 0;
    const splits = @divTrunc(total - first + split_keys - 1, split_keys);
    return launchGrid(g, &relsplit_open_kernel, &.{ q, kc, vc, kn, vn, bias, sinks, try g.fromSlice(i32, &.{ s, cached, first }, &.{3}), scale_arr }, 2, .{
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
    const o = try launchGrid(g, &relcomb_kernel, &.{ parts[0], parts[1], gate_sigmoid }, 1, .{.{ .shape = &.{ 1, s, @divExact(rows, s) * hd }, .dtype = dt }}, .{ hd, rows, 1 }, .{ hd, 1, 1 }, &.{
        .{ "T", .{ .dtype = dt } }, .{ "D", .{ .int = hd } },
    });
    return o[0];
}

/// A few or a prompt's rows read behind the keys `k`, `v` (the rows at their last S), gated.
fn readRows(g: *Graph, L: *const Layer, cfg: *const Config, p: *const Prep, sinks: Array, k: Array, v: Array, offset: c_int, s: c_int) !Array {
    if (try splitRead(g, L, cfg, p.q, p.rel, sinks, k, v, offset, p.gate)) |o| return o;
    return gated(g, try relative(g, L, cfg, p.q, p.rel, sinks, k, v, offset, false), p.gate, s);
}

/// A draft verify's visit: its rows read the cache without appending and leave here, kept by `g`, what a commit
/// needs: their keys and values `[1, Hkv, S, hd]` and the conv state + projections (zero concat during verify).
pub const Open = struct { g: *Graph, k: Array = .{}, v: Array = .{}, st: Array = .{}, kp: Array = .{}, vp: Array = .{}, op: Array = .{}, xm: Array = .{} };

/// Keeps the first `rows` rows of a verify: each visit's keys and values appended, its conv state moved forward by `rows`.
pub fn commit(c: *block.Ctx, cfg: *const Config, opens: []const Open, rows: c_int) !void {
    const kt: c_int = @intCast(cfg.sconv_kernel);
    const gpu = mlx.streamIsGpu(c.g.s);
    for (opens, 0..) |*o, v| {
        c.layer = @intCast(v);
        var view = try layers.appendKv(c, c.cache(), cfg.global[cfg.visit(@intCast(v)).layer], cfg, try c.g.sliceAxis(o.k, 2, 0, rows), try c.g.sliceAxis(o.v, 2, 0, rows), 0);
        kv.release(c, &view);
        const kp = try c.g.sliceAxis(o.kp, 1, 0, rows);
        const vp = try c.g.sliceAxis(o.vp, 1, 0, rows);
        const op = try c.g.sliceAxis(o.op, 1, 0, rows);
        const xm = try c.g.sliceAxis(o.xm, 1, 0, rows);
        const nst = if (gpu) try convState(c.g, o.st, kp, vp, op, xm) else blk: {
            const win = try c.g.concat(&.{ o.st, try c.g.astype(try c.g.concat(&.{ kp, vp, op, xm }, 2), .float32) }, 1);
            break :blk try c.g.sliceAxis(win, 1, rows, rows + kt - 1);
        };
        try kv.setState(c, .conv, nst);
    }
}

/// `layers.attend`: appends the visit's k and v, reads under the bias and the `sinks`, gates. An `open` verify reads
/// the cache's rows with its own behind them and appends nothing.
/// `raw`: one row's read is returned without its output gate, which `oGated` folds into the output projection.
fn attend(c: *block.Ctx, L: *const Layer, cfg: *const Config, p: *const Prep, sinks: Array, open: ?*Open, raw: bool, consts: ?*const layers.Consts) !Array {
    const g = c.g;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const s = c.rows;
    const offset: c_int = @intCast(c.offset());
    const span: u32 = if (L.global) 0 else cfg.sliding_window;
    if (open) |o| {
        if (s < 2) return error.Inkling2DraftVerify;
        var held = try kv.view(c, c.cache(), c.layer);
        defer kv.release(c, &held);
        const dv = kv.dense(held);
        o.k = try o.g.keep(p.k);
        o.v = try o.g.keep(p.v);
        if (try splitReadOpenWithConsts(g, L, cfg, p.q, p.rel, sinks, dv.k, dv.v, p.k, p.v, offset, p.gate, consts)) |out| return out;
        const rows = ops.shape(dv.k)[2];
        // A sliding layer's rows reach at most `window - 1` keys before the block.
        const from = if (span > 0) @max(0, rows - @as(c_int, @intCast(span - 1))) else 0;
        const keys = try g.concat(&.{ try g.sliceAxis(dv.k, 2, from, rows), p.k }, 2);
        const values = try g.concat(&.{ try g.sliceAxis(dv.v, 2, from, rows), p.v }, 2);
        return readRows(g, L, cfg, p, sinks, keys, values, offset, s);
    }
    var view = try layers.appendKv(c, c.cache(), L.global, cfg, p.k, p.v, if (span > 0) band.view(span, hd, offset + s, s).span else 0);
    defer kv.release(c, &view);
    const dv = kv.dense(view);
    if (s > 1) return readRows(g, L, cfg, p, sinks, dv.k, dv.v, offset, s);
    if (ops.shape(dv.k)[2] != ops.shape(p.bias)[3]) return error.Inkling2KvView;
    const attn = try sdpaSinks(g, p.q, dv.k, dv.v, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))), "array", p.bias, try g.reshape(sinks, &.{heads}), false);
    return if (raw) attn else gated(g, attn, p.gate, 1);
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
fn attendSlots(c: *block.Ctx, L: *const Layer, cfg: *const Config, p: *const Prep, sinks: Array, consts: ?*const layers.Consts) !Array {
    const g = c.g;
    const n = c.caches.len;
    const hd: c_int = @intCast(cfg.head_dim);
    const dt = ops.dtype(p.q);
    const span: u32 = if (L.global) 0 else cfg.sliding_window;
    var q = try g.transpose(p.q, &.{ 0, 2, 1, 3 });
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
            const tau = try g.fromSlice(f32, taus[0..n], &.{ 1, 1, rows, 1 });
            q = try g.astype(try g.mul(try g.astype(q, .float32), tau), dt);
            states = try g.astype(try g.mul(try g.astype(states, .float32), try g.reshape(tau, &.{ 1, rows, 1, 1 })), dt);
        }
    }
    const rp = if (dt == .bfloat16 and L.rel_proj_bf16.ctx != null) L.rel_proj_bf16 else try g.astype(L.rel_proj, dt);
    const bias = try g.matmul(states, rp);
    const sink_rows = try g.contiguous(try g.astype(sinks, .float32));
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
    if (slotsRuns(g, cfg, dt, ks[0..n], vs[0..n]))
        return slotsReadWithConsts(g, try g.contiguous(try g.transpose(q, &.{ 0, 2, 1, 3 })), ks[0..n], vs[0..n], bias, sink_rows, @intCast(span), p.gate, consts);
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
    return slotsReadWithConsts(g, q, ks, vs, bias, sinks, window, gate_sigmoid, null);
}

fn slotsReadWithConsts(g: *Graph, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int, gate_sigmoid: Array, consts: ?*const layers.Consts) !Array {
    const n = ks.len;
    const width = ops.shape(q)[2] * ops.shape(q)[3];
    var outs: [max_slots / slots_per_read]Array = undefined;
    var groups: usize = 0;
    var from: usize = 0;
    while (from < n) : (groups += 1) {
        const to = @min(n, from + slots_per_read);
        const lo: c_int = @intCast(from);
        const hi: c_int = @intCast(to);
        const part = try slotPartsWithConsts(g, try g.sliceAxis(q, 1, lo, hi), ks[from..to], vs[from..to], try g.sliceAxis(bias, 1, lo, hi), try g.sliceAxis(sinks, 1, lo, hi), window, consts);
        outs[groups] = try splitJoin(g, part, try g.sliceAxis(gate_sigmoid, 0, lo * width, hi * width), hi - lo, ops.dtype(q));
        from = to;
    }
    return if (groups == 1) outs[0] else g.concat(outs[0..groups], 1);
}

/// `relslots.metal` for up to `slots_per_read` slots: each split's unnormalised sums `[splits, n * H, hd]` and (max,
/// sum) `[splits, n * H, 2]`, f32, as `splitParts`'.
pub fn slotParts(g: *Graph, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int) ![2]Array {
    return slotPartsWithConsts(g, q, ks, vs, bias, sinks, window, null);
}

fn slotPartsWithConsts(g: *Graph, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int, consts: ?*const layers.Consts) ![2]Array {
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
    inputs[20] = if (consts) |cs| (if (cs.attn_scale.ctx != null) cs.attn_scale else try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32)) else try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32);
    return launchGrid(g, &relslots_kernel, &inputs, 2, .{
        .{ .shape = &.{ splits, ng * heads, hd }, .dtype = .float32 },
        .{ .shape = &.{ splits, ng * heads, 2 }, .dtype = .float32 },
    }, .{ splits * 32 * slot_groups, ops.shape(ks[0])[1], ng }, .{ 32 * slot_groups, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(q) } }, .{ "D", .{ .int = hd } },                    .{ "HQ", .{ .int = heads } },
        .{ "HKV", .{ .int = ops.shape(ks[0])[1] } }, .{ "R", .{ .int = ops.shape(bias)[3] } }, .{ "NS", .{ .int = ng } },
        .{ "NSG", .{ .int = slot_groups } },  .{ "KB", .{ .int = slot_keys } },            .{ "WINDOW", .{ .int = window } },
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
    const on = gates != null;
    const o = while (true) {
        const tg = streamGroup(cfg.hidden_size) orelse return error.ThreadLimit;
        break launchGrid(g, &mid_kernel, &.{ op, st, L.attn_conv, h, normed, blocks orelse h, L.mlp_query, L.post_norm, msg3, pr }, 5, .{
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
            .{ "MSG", .{ .int = @intFromBool(on) } },                      .{ "SEQ", seqOf(st) },
        }) catch |e| if (e == error.ThreadLimit and narrow(cfg.hidden_size)) continue else return e;
    };
    var x = o[1];
    // A plain visit's router is MLX's matmul; a looped one's rides its norm kernel, where the rows read the stream
    // as it is (inside `mid.metal` each row would redo the conv and the mix: slower than the launch it saves).
    var scores: Array = undefined;
    if (gates) |gs| {
        const state = try g.matmul(o[4], try layers.downs(g, gs, 3, 1));
        const xs = try normWithEps(g, cfg, try gate(g, gs, o[3], msg3, state, 3, 1), @splat(L.post_norm), 1, dt, L.router, w.consts.eps);
        x = xs[0];
        scores = xs[1];
    } else scores = if (s > 1 and s <= small_rows) try routerRows(g, o[2], L.router) else try g.matmul(o[2], try g.transpose(L.router, &.{ 1, 0 }));
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
    const s = ops.shape(residual)[1];
    const d = ops.shape(residual)[2];
    const dt = ops.dtype(dn);
    const out: Out = .{ .shape = &.{ 1, s, d }, .dtype = dt };
    return launch(g, &sum_kernel, &.{ dn, sel, inverse, scale, weights, residual, st, taps, conv_scale }, 2, .{ out, out }, s * d, 64, &.{
        .{ "T", .{ .dtype = dt } }, .{ "WT", .{ .dtype = ops.dtype(taps) } }, .{ "K", .{ .int = k } }, .{ "KT", .{ .int = ops.shape(st)[1] + 1 } }, .{ "SCALED", .{ .int = @intFromBool(scaled) } }, .{ "SEQ", seqOf(st) },
    });
}

/// The most rows the grouped unique-expert kernels serve: covers DSpark verify (2..8), beam-tree verify (up to 16),
/// and small-to-medium batched decode (2..16).
pub const packed_rows = 16;

/// `group_experts.metal`: compacts the unique active experts across `s` rows (`idx` `[1, s, top_k]` u32).
pub fn groupExperts(g: *Graph, idx: Array, cfg: *const Config, s_cap: c_int, nmax: c_int) ![3]Array {
    const e: c_int = @intCast(cfg.num_experts);
    const tg: c_int = @divTrunc(e + 31, 32) * 32;
    return launch(g, &group_experts_kernel, &.{idx}, 3, .{
        .{ .shape = &.{nmax}, .dtype = .uint32 },
        .{ .shape = &.{nmax}, .dtype = .uint32 },
        .{ .shape = &.{ nmax, s_cap }, .dtype = .uint32 },
    }, tg, tg, &.{
        .{ "E", .{ .int = e } },       .{ "TG", .{ .int = tg } },     .{ "KSEL", .{ .int = @intCast(cfg.top_k) } },
        .{ "S_CAP", .{ .int = s_cap } }, .{ "NMAX", .{ .int = nmax } },
    });
}

/// `ups_grouped.metal`: grouped multi-row gate+up+SwiGLU over unique active experts.
pub fn upsGrouped(g: *Graph, x: Array, wq: Array, ws: Array, wb: Array, act_e: Array, act_cnt: Array, act_pair: Array, scale: Array, caps: Array, n: c_int, inter: c_int, hidden: c_int, gs: u32, fmt: c_int, bits: u32, scaled: bool, top_k: c_int, s_cap: c_int, nmax: c_int) !Array {
    const dt = ops.dtype(x);
    const up_tg: c_int = if (@rem(inter, 4) == 0) 4 else if (@rem(inter, 2) == 0) 2 else 1;
    const out = try launchGrid(g, &ups_grouped_kernel, &.{ x, wq, ws, wb, act_e, act_cnt, act_pair, scale, caps }, 1, .{.{ .shape = &.{ n, 1, inter }, .dtype = dt }}, .{ 32, inter, nmax }, .{ 32, up_tg, 1 }, &.{
        .{ "T", .{ .dtype = dt } },    .{ "K", .{ .int = hidden } },          .{ "I", .{ .int = inter } },
        .{ "GS", .{ .int = @intCast(gs) } }, .{ "FMT", .{ .int = fmt } },     .{ "BITS", .{ .int = @intCast(bits) } },
        .{ "S", .{ .dtype = ops.dtype(ws) } }, .{ "SCALED", .{ .int = @intFromBool(scaled) } }, .{ "KSEL", .{ .int = top_k } },
        .{ "S_CAP", .{ .int = s_cap } },
    });
    return out[0];
}

/// `downs_grouped.metal`: grouped multi-row down projection (2 output rows per SIMD group) over unique active experts.
pub fn downsGrouped(g: *Graph, act: Array, wq: Array, ws: Array, wb: Array, act_e: Array, act_cnt: Array, act_pair: Array, n: c_int, inter: c_int, hidden: c_int, gs: u32, fmt: c_int, bits: u32, s_cap: c_int, nmax: c_int) !Array {
    const dt = ops.dtype(act);
    const half_d = @divExact(hidden, 2);
    const dn_tg: c_int = if (@rem(half_d, 4) == 0) 4 else if (@rem(half_d, 2) == 0) 2 else 1;
    const out = try launchGrid(g, &downs_grouped_kernel, &.{ act, wq, ws, wb, act_e, act_cnt, act_pair }, 1, .{.{ .shape = &.{ n, 1, hidden }, .dtype = dt }}, .{ 32, half_d, nmax }, .{ 32, dn_tg, 1 }, &.{
        .{ "T", .{ .dtype = dt } },    .{ "I", .{ .int = inter } },           .{ "D", .{ .int = hidden } },
        .{ "GS", .{ .int = @intCast(gs) } }, .{ "FMT", .{ .int = fmt } },     .{ "BITS", .{ .int = @intCast(bits) } },
        .{ "S", .{ .dtype = ops.dtype(ws) } }, .{ "S_CAP", .{ .int = s_cap } },
    });
    return out[0];
}

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

/// What a packed gate_up launch reads its picks from: the router's logits `[S, E]` f32, ranked inside the launch as
/// `topK` ranks them (one row), or the picks `[S * k]` u32 a `topK` launch left (more rows).
pub const UpsRoute = union(enum) { scores: Array, picks: Array };

/// The rows' picked experts' soft-capped SwiGLU `[S * k, 1, I]` from a packed gate_up bank, in one launch: { act,
/// picks, weights }, the picks `[1, S, k]` and their weights only when `route` is the logits.
pub fn ups(g: *Graph, L: *const Layer, cfg: *const Config, x: Array, route: UpsRoute) ![3]Array {
    return upsWithConsts(g, L, cfg, null, x, route);
}

fn upsWithConsts(g: *Graph, L: *const Layer, cfg: *const Config, consts: ?*const layers.Consts, x: Array, route: UpsRoute) ![3]Array {
    const dt = ops.dtype(x);
    const s = ops.shape(x)[1];
    const top_k: c_int = @intCast(cfg.top_k);
    const inter: c_int = @intCast(cfg.moe_intermediate_size);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const fmt = packedFormat(&L.gate_up, hidden) orelse return error.Inkling2ExpertLayout;
    const caps = if (consts) |cs| (if (cs.caps.ctx != null) cs.caps else try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4})) else try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4});
    const rscale = if (consts) |cs| (if (cs.route_scale.ctx != null) cs.route_scale else try g.fromSlice(f32, &.{cfg.route_scale}, &.{1})) else try g.fromSlice(f32, &.{cfg.route_scale}, &.{1});
    const scaled = L.gate_up_scale.ctx != null;
    const ranked = route == .scores;
    // The arm not taken reads none of its inputs: any array of a float dtype stands in.
    const scores = if (ranked) route.scores else caps;
    const sh = ops.shape(scores);
    return launchGrid(g, &ups_kernel, &.{ x, L.gate_up.w, L.gate_up.s, biasesOf(&L.gate_up), scores, if (ranked) L.router_bias else caps, rscale, if (ranked) caps else route.picks, if (scaled) L.gate_up_scale else caps, caps }, 3, .{
        .{ .shape = &.{ s * top_k, 1, inter }, .dtype = dt },
        if (ranked) .{ .shape = &.{ 1, s, top_k }, .dtype = .uint32 } else unused,
        if (ranked) .{ .shape = &.{ 1, s, top_k }, .dtype = .float32 } else unused,
    }, .{ 32, @divExact(inter, bank_rows), s * top_k }, .{ 32, bank_groups, 1 }, &.{
        .{ "T", .{ .dtype = dt } },                               .{ "K", .{ .int = hidden } }, .{ "I", .{ .int = inter } },
        .{ "GS", .{ .int = @intCast(L.gate_up.qp.group_size) } }, .{ "FMT", .{ .int = fmt } },  .{ "ROWS", .{ .int = bank_rows } },
        .{ "BITS", .{ .int = @intCast(L.gate_up.qp.bits) } },     .{ "S", .{ .dtype = ops.dtype(L.gate_up.s) } },
        .{ "SCALED", .{ .int = @intFromBool(scaled) } },          .{ "KSEL", .{ .int = top_k } },
        .{ "E", .{ .int = if (ranked) sh[sh.len - 1] else 1 } },  .{ "SGS", .{ .int = bank_groups } },
        .{ "ROUTE", .{ .int = @intFromBool(ranked) } },
    });
}

/// `layers.tail`. The visit's new conv state (off the next visit's path) is `st` behind the rows' k and v projections
/// `kp`, `vp`, their output projection and the MLP's output.
/// One row, a packed bank read in place: the router's picks, their gate_up and the soft-capped SwiGLU as one launch,
/// their down, the weighted sum, the MLP's conv behind `st` and the final add as a second. Another bank, or more rows
/// (their pairs sorted by expert), go through `topK` and MLX's gather, with the SwiGLU and the sum, conv and add as
/// launches of their own.
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
    const caps = if (consts) |cs| (if (cs.caps.ctx != null) cs.caps else try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4})) else try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4});
    const conv_scale = if (consts) |cs| (if (cs.conv_scale.ctx != null) cs.conv_scale else try g.fromSlice(f32, &.{1.0 / @sqrt(@as(f32, @floatFromInt(kt + 1)))}, &.{1})) else try g.fromSlice(f32, &.{1.0 / @sqrt(@as(f32, @floatFromInt(kt + 1)))}, &.{1});
    const rscale = if (consts) |cs| (if (cs.route_scale.ctx != null) cs.route_scale else try g.fromSlice(f32, &.{cfg.route_scale}, &.{1})) else try g.fromSlice(f32, &.{cfg.route_scale}, &.{1});
    const gu_scale = if (L.gate_up_scale.ctx != null) L.gate_up_scale else caps;
    const dn_scaled: TArg = .{ .int = @intFromBool(L.down_scale.ctx != null) };
    const tail_out: [2]Out = .{ .{ .shape = &.{ 1, s, hidden }, .dtype = dt }, .{ .shape = &.{ 1, s, hidden }, .dtype = dt } };

    // Multi-row packed path (`2..packed_rows`): compact unique active experts on GPU so each expert's weights are
    // read from DRAM once across all rows in the verify/batch block, with zero argsort overhead.
    const gu_fmt = packedFormat(&L.gate_up, hidden);
    const dn_fmt = packedFormat(&L.down, inter);
    const grouped = s > 1 and s <= packed_rows and cfg.num_experts <= sure_threads and gu_fmt != null and dn_fmt != null;
    if (grouped) {
        const picks = try topK(g, m.scores, L.router_bias, rscale, top_k);
        const dn_scale = if (L.down_scale.ctx != null) L.down_scale else picks[1];
        const s_cap: c_int = if (s <= 8) 8 else 16;
        const nmax: c_int = @min(n, @as(c_int, @intCast(cfg.num_experts)));
        const grp = try groupExperts(g, picks[0], cfg, s_cap, nmax);
        const act = try upsGrouped(g, m.x, L.gate_up.w, L.gate_up.s, biasesOf(&L.gate_up), grp[0], grp[1], grp[2], gu_scale, caps, n, inter, hidden, L.gate_up.qp.group_size, gu_fmt.?, L.gate_up.qp.bits, L.gate_up_scale.ctx != null, top_k, s_cap, nmax);
        const dn = try downsGrouped(g, act, L.down.w, L.down.s, biasesOf(&L.down), grp[0], grp[1], grp[2], n, inter, hidden, L.down.qp.group_size, dn_fmt.?, L.down.qp.bits, s_cap, nmax);
        const flat_idx = try g.reshape(picks[0], &.{n});
        const ident = if (consts) |cs| (if (cs.ident.ctx != null and ops.shape(cs.ident)[0] >= n) try g.sliceAxis(cs.ident, 0, 0, n) else try g.arange(0, @floatFromInt(n), 1, .uint32)) else try g.arange(0, @floatFromInt(n), 1, .uint32);
        const o = try pickSum(g, dn, flat_idx, ident, dn_scale, picks[1], m.residual, st, L.mlp_conv, conv_scale, top_k, L.down_scale.ctx != null);
        if (open) |v| {
            v.st = try v.g.keep(st);
            v.kp = try v.g.keep(kp);
            v.vp = try v.g.keep(vp);
            v.op = try v.g.keep(m.op);
            v.xm = try v.g.keep(o[1]);
            return .{ .h = o[0], .state = st };
        }
        return .{ .h = o[0], .state = try convState(g, st, kp, vp, m.op, o[1]) };
    }

    const packed_up = s == 1 and gu_fmt != null;
    const up = if (packed_up) try upsWithConsts(g, L, cfg, consts, m.x, .{ .scores = m.scores }) else null;
    const picks: [2]Array = if (up) |u| .{ u[1], u[2] } else try topK(g, m.scores, L.router_bias, rscale, top_k);
    const weights = picks[1];
    const dn_scale = if (L.down_scale.ctx != null) L.down_scale else weights;

    // Prefill NAX mapped path (`s > 1`, sorted pairs): read `m.x` via `row_map` directly in `gather_qmm_nax`,
    // skipping the `top_k`× activation copy and engaging NAX even below 4 rows/expert.
    const nax = s > 1 and verifyQmmNaxAvailable();
    var sel: Array = .{};
    var inverse_opt: ?Array = null;
    var act: [1]Array = undefined;
    var nax_down_ready = false;
    if (up) |u| {
        const pairs = try layers.pairs(g, cfg, m.x, picks[0]);
        sel = pairs.sel;
        inverse_opt = pairs.inverse;
        act = .{u[0]};
    } else if (nax and n >= 64) {
        const flat = try g.reshape(picks[0], &.{n});
        const order = try g.argsort(flat, 0);
        sel = try g.take(flat, order, 0);
        const token = try g.astype(try g.call(mlx.mlx_floor_divide, .{ order, try g.astype(try g.scalar(@floatFromInt(top_k), .float32), ops.dtype(order)) }), .uint32);
        inverse_opt = try g.argsort(order, 0);
        const gq = L.gate_up.qp;
        const mapped_gu: ?Array = if (L.gate_up.s.ctx != null and (gq.mode == .affine or gq.mode.isFp4())) blk: {
            const x_tok = try g.reshape(m.x, &.{ s, 1, hidden });
            break :blk if (try gather_qmm_nax.sortedGatherMappedPrompt(x_tok, token, L.gate_up.w, L.gate_up.s, L.gate_up.b, sel, gq.bits, gq.group_size, gq.mode.isFp4(), true, g.s)) |y| try g.adopt(y) else null;
        } else null;
        const gu = mapped_gu orelse blk: {
            const rows = try g.reshape(try g.take(try g.reshape(m.x, &.{ s, hidden }), token, 0), &.{ n, 1, hidden });
            break :blk try layers.gather(g, rows, &L.gate_up, sel, true);
        };
        if (!floating(ops.dtype(gu))) return error.Inkling2StepDtype;
        act = .{try activation(g, gu, sel, gu_scale, caps, L.gate_up_scale.ctx != null)};
        nax_down_ready = true;
    } else {
        const pairs = try layers.pairs(g, cfg, m.x, picks[0]);
        sel = pairs.sel;
        inverse_opt = pairs.inverse;
        const gu = try layers.gather(g, pairs.rows, &L.gate_up, sel, pairs.inverse != null);
        if (!floating(ops.dtype(gu))) return error.Inkling2StepDtype;
        act = .{try activation(g, gu, sel, gu_scale, caps, L.gate_up_scale.ctx != null)};
    }
    const sorted = inverse_opt != null;

    // The down kernel's threadgroup holds every pick of its rows.
    const o = if (if (s == 1 and 32 * top_k <= sure_threads) dn_fmt else null) |fmt| blk: {
        break :blk try launchGrid(g, &downs_kernel, &.{ act[0], L.down.w, L.down.s, biasesOf(&L.down), sel, dn_scale, weights, m.residual, st, L.mlp_conv, conv_scale }, 2, tail_out, .{ 32, top_k, s * @divExact(hidden, bank_rows) }, .{ 32, top_k, 1 }, &.{
            .{ "T", .{ .dtype = dt } },         .{ "WT", .{ .dtype = ops.dtype(L.mlp_conv) } },        .{ "I", .{ .int = inter } },
            .{ "D", .{ .int = hidden } },       .{ "GS", .{ .int = @intCast(L.down.qp.group_size) } }, .{ "FMT", .{ .int = fmt } },
            .{ "ROWS", .{ .int = bank_rows } }, .{ "KSEL", .{ .int = top_k } },                        .{ "KT", .{ .int = kt } },
            .{ "BITS", .{ .int = @intCast(L.down.qp.bits) } }, .{ "S", .{ .dtype = ops.dtype(L.down.s) } },
            .{ "SCALED", dn_scaled },           .{ "TAIL", .{ .int = 1 } },
        });
    } else blk: {
        const dq = L.down.qp;
        const nax_dn: ?Array = if (nax_down_ready and L.down.s.ctx != null and (dq.mode == .affine or dq.mode.isFp4()))
            (if (try gather_qmm_nax.sortedGatherPrompt(act[0], L.down.w, L.down.s, L.down.b, sel, dq.bits, dq.group_size, dq.mode.isFp4(), true, g.s)) |y| try g.adopt(y) else null)
        else
            null;
        const dn = nax_dn orelse try layers.gather(g, act[0], &L.down, sel, sorted);
        if (!floating(ops.dtype(dn))) return error.Inkling2StepDtype;
        // Unsorted pairs sit in (row, pick) order already.
        const inverse = inverse_opt orelse (if (consts) |cs| (if (cs.ident.ctx != null and ops.shape(cs.ident)[0] >= n) try g.sliceAxis(cs.ident, 0, 0, n) else try g.arange(0, @floatFromInt(n), 1, .uint32)) else try g.arange(0, @floatFromInt(n), 1, .uint32));
        break :blk try pickSum(g, dn, sel, inverse, dn_scale, weights, m.residual, st, L.mlp_conv, conv_scale, top_k, L.down_scale.ctx != null);
    };
    if (open) |v| {
        v.st = try v.g.keep(st);
        v.kp = try v.g.keep(kp);
        v.vp = try v.g.keep(vp);
        v.op = try v.g.keep(m.op);
        v.xm = try v.g.keep(o[1]);
        return .{ .h = o[0], .state = st };
    }
    return .{ .h = o[0], .state = try convState(g, st, kp, vp, m.op, o[1]) };
}

/// A visit's conv state `[1, KT - 1, .]` f32 after its rows: the last KT - 1 rows of the old state `st` followed
/// by the rows' conv inputs `[kp | vp | op | xm]`. With a state per row (`[S, KT - 1, .]`, batched slots) each moves
/// on by its own row.
pub fn convState(g: *Graph, st: Array, kp: Array, vp: Array, op: Array, xm: Array) !Array {
    const sh = ops.shape(st);
    const state = try launch(g, &state_kernel, &.{ st, kp, vp, op, xm }, 1, .{.{ .shape = &.{ sh[0], sh[1], sh[2] }, .dtype = .float32 }}, sh[0] * sh[1] * sh[2], 64, &.{ .{ "KT", .{ .int = sh[1] + 1 } }, .{ "SEQ", seqOf(st) } });
    return state[0];
}

/// A conv state's rows: one state the rows chain behind (`SEQ` 1), or a state per row, each a sequence of its own.
fn seqOf(st: Array) TArg {
    return .{ .int = @intFromBool(ops.shape(st)[0] == 1) };
}

/// One visit of one sequence's rows `[1, S, D]`: `layers.layerForward`, or null where the kernels do not run (nothing
/// is appended then). `bank` is the last pass's bank and normed output on a later loop pass; `open` a draft verify's
/// visit. Batched slots (`c.slots` > 1) are the rows, each a sequence of its own with its own cache and conv state.
pub fn visit(c: *block.Ctx, L: *const Layer, w: *const Body, h: Array, blocks: ?Array, bank: ?[2]Array, open: ?*Open) !?layers.Post {
    const g = c.g;
    const cfg = &w.cfg;
    if (!runs(g, L, cfg, h)) return null;
    const batch = c.slots > 1;
    const held = if (batch) try slot_state.get(c, .conv) else try kv.state(c, .conv);
    const st = held orelse try g.zeros(&.{ c.slots, @intCast(cfg.sconv_kernel - 1), @intCast(2 * cfg.kvDim() + 2 * cfg.hidden_size) }, .float32);
    const f = try pre(g, L, w, h, blocks, bank);
    const total = @as(c_int, @intCast(c.offset())) + 1;
    const one = c.rows == 1 and !batch;
    const p = (try prepWithConsts(g, L, cfg, &w.consts, &f.proj, f.dyn, st, if (one and L.global) layers.stepTau(cfg, total) else 1.0, if (one) keysSeen(L, cfg, total) else 0)) orelse return null;
    const fuse = one and open == null and oFormat(&L.o, ops.dtype(h)) != null;
    const read = if (batch) try attendSlots(c, L, cfg, &p, f.proj[3], &w.consts) else try attend(c, L, cfg, &p, f.proj[3], open, fuse, &w.consts);
    const m = try mid(g, L, w, read, if (fuse) p.gate else null, h, f.normed, blocks, st, f.messages);
    const t = try expertsWithConsts(g, L, cfg, &w.consts, &m, f.proj[4], f.proj[5], st, open);
    if (batch) try slot_state.put(c, .conv, t.state) else if (open == null) try kv.setState(c, .conv, t.state);
    return .{ .h = t.h, .blocks = m.blocks };
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
