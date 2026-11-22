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
/// Output rows a SIMD group of a packed expert kernel owns (one: a lane's loop sets the launch's latency, and here
/// that beats keeping x in registers across rows), and the up kernel's SIMD groups per threadgroup.
const bank_rows: c_int = 1;
const bank_groups: c_int = 2;

pub var pre_kernel: Kernel = .{ .name = "msv_inkling2_pre", .ins = &.{ "HIN", "BLK", "QY", "NW0", "NW1", "NW2", "BANK", "BPART", "MQ", "PR" }, .outs = &.{ "X", "NORMED", "A", "MSGO", "BOTH" }, .source = kernelSource("plugins/inkling2/kernels/pre.metal") };
pub var prep_kernel: Kernel = .{ .name = "msv_inkling2_prep", .ins = &.{ "QP", "GP", "RP", "KP", "VP", "KDY", "VDY", "ST", "KW", "VW", "QS", "RS", "RJ", "PR" }, .outs = &.{ "Q", "GATE", "K", "V", "BIAS", "REL" }, .source = kernelSource("plugins/inkling2/kernels/prep.metal") };
pub var oproj_kernel: Kernel = .{ .name = "msv_inkling2_oproj", .ins = &.{ "ATT", "GATE", "WQ", "WS", "WB" }, .outs = &.{"Y"}, .source = kernelSource("plugins/inkling2/kernels/oproj.metal"), .header = bank_header };
pub var gated_kernel: Kernel = .{ .name = "msv_inkling2_gated", .ins = &.{ "ATT", "GATE" }, .outs = &.{"OUT"}, .source = kernelSource("plugins/inkling2/kernels/gated.metal") };
pub var mid_kernel: Kernel = .{ .name = "msv_inkling2_mid", .ins = &.{ "OP", "ST", "AW", "HIN", "NORMED", "BLK", "QY", "PN", "MSG3", "PR" }, .outs = &.{ "RES", "X", "X32", "M", "BOTH" }, .source = kernelSource("plugins/inkling2/kernels/mid.metal") };
pub var gate_kernel: Kernel = .{ .name = "msv_inkling2_gate", .ins = &.{ "A", "MS", "ST", "F", "G" }, .outs = &.{"Y"}, .source = kernelSource("plugins/inkling2/kernels/gate.metal") };
pub var norm_kernel: Kernel = .{ .name = "msv_inkling2_norm", .ins = &.{ "Y", "NW0", "NW1", "NW2", "RT", "PR" }, .outs = &.{ "X", "SCORES" }, .source = kernelSource("plugins/inkling2/kernels/norm.metal") };
pub var topk_kernel: Kernel = .{ .name = "msv_inkling2_topk", .ins = &.{ "SC", "RB", "PR" }, .outs = &.{ "IDX", "WTS" }, .source = kernelSource("plugins/inkling2/kernels/topk.metal") };
pub var act_kernel: Kernel = .{ .name = "msv_inkling2_act", .ins = &.{ "GU", "IDX", "SC", "PR" }, .outs = &.{"ACT"}, .source = kernelSource("plugins/inkling2/kernels/act.metal") };
pub var ups_kernel: Kernel = .{ .name = "msv_inkling2_ups", .ins = &.{ "X", "WQ", "WS", "WB", "IDX", "SC", "PR" }, .outs = &.{"ACT"}, .source = kernelSource("plugins/inkling2/kernels/ups.metal"), .header = bank_header };
pub var downs_kernel: Kernel = .{ .name = "msv_inkling2_downs", .ins = &.{ "ACT", "WQ", "WS", "WB", "IDX", "SC", "WTS", "RES", "ST", "MW", "PR" }, .outs = &.{ "HOUT", "XM" }, .source = kernelSource("plugins/inkling2/kernels/downs.metal"), .header = bank_header };
pub var relattn_kernel: Kernel = .{ .name = "msv_inkling2_relattn", .ins = &.{ "q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale" }, .outs = &.{"out"}, .source = kernelSource("plugins/inkling2/kernels/relattn.metal"), .contiguous = false };
pub var router_kernel: Kernel = .{ .name = "msv_inkling2_router", .ins = &.{ "X", "RT" }, .outs = &.{"SC"}, .source = kernelSource("plugins/inkling2/kernels/router.metal") };
pub var relsplit_kernel: Kernel = .{ .name = "msv_inkling2_relsplit", .ins = &.{ "q", "k", "v", "rel", "sinks", "sizes", "scale" }, .outs = &.{ "PART", "ML" }, .source = kernelSource("plugins/inkling2/kernels/relsplit.metal"), .contiguous = false };
pub var relcomb_kernel: Kernel = .{ .name = "msv_inkling2_relcomb", .ins = &.{ "PART", "ML", "GATE" }, .outs = &.{"OUT"}, .source = kernelSource("plugins/inkling2/kernels/relcomb.metal") };
/// The Markov kernels' argmax order.
const markov_header = kernelSource("plugins/inkling2/kernels/markov_header.metal");
pub var markov_rnn_kernel: Kernel = .{ .name = "msv_inkling2_markov_rnn", .ins = &.{ "W", "B", "ST", "PREV", "ROWS", "NEW", "TOK", "AT" }, .outs = &.{"P"}, .source = kernelSource("plugins/inkling2/kernels/markov_rnn.metal") };
pub var markov_fold_kernel: Kernel = .{ .name = "msv_inkling2_markov_fold", .ins = &.{ "P", "FOLD", "FB", "BASE", "AT", "CAP" }, .outs = &.{ "PM", "PI", "PS" }, .source = kernelSource("plugins/inkling2/kernels/markov_fold.metal"), .header = markov_header };
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
var stream_threads: u32 = 768;

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
fn streams(g: *Graph, cfg: *const Config, h: Array, blocks: ?Array, query: Array, nw: [3]Array, inputs: c_int, msg: ?Messaged) ![5]Array {
    const d: c_int = @intCast(cfg.hidden_size);
    const s = ops.shape(h)[1];
    const dt = ops.dtype(h);
    const q: c_int = if (query.ctx != null) ops.shape(query)[1] else 0;
    const eps = try g.fromSlice(f32, &.{cfg.rms_norm_eps}, &.{1});
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
    const d: c_int = @intCast(cfg.hidden_size);
    const s = ops.shape(y)[1];
    const experts_n: c_int = if (router != null) @intCast(cfg.num_experts) else 0;
    const eps = try g.fromSlice(f32, &.{cfg.rms_norm_eps}, &.{1});
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
        const o = try streams(g, cfg, h, blocks, L.attn_query, L.in_norm, 3, .{ .bank = bk[0], .partial = bk[1], .query = L.message_query, .gt = ops.dtype(gates.down) });
        const st = try g.matmul(o[4], try layers.downs(g, gates, 0, 3));
        x = (try norm(g, cfg, try gate(g, gates, o[2], o[3], st, 0, 3), L.in_norm, 3, dt, null))[0];
        normed = o[1];
        messages = o[3];
    } else {
        const o = try streams(g, cfg, h, blocks, L.attn_query, L.in_norm, if (L.global) 3 else 1, null);
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
    return (try streams(g, &w.cfg, h, blocks, w.final_query, @splat(norm_weight), 1, null))[0];
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
    const pr = try g.fromSlice(f32, &.{ cfg.qk_clip, cfg.rel_clip, cfg.rms_norm_eps, tau, conv_scale, @floatFromInt(keys) }, &.{6});
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
/// needs: their keys and values `[1, Hkv, S, hd]` and the conv windows `[1, K - 1 + S, .]` (the state before them,
/// then every row's conv inputs).
pub const Open = struct { g: *Graph, k: Array = .{}, v: Array = .{}, window: Array = .{} };

/// Keeps the first `rows` rows of a verify: each visit's keys and values appended, its conv state the window behind
/// them.
pub fn commit(c: *block.Ctx, cfg: *const Config, opens: []const Open, rows: c_int) !void {
    const kt: c_int = @intCast(cfg.sconv_kernel);
    for (opens, 0..) |*o, v| {
        c.layer = @intCast(v);
        var view = try layers.appendKv(c, c.cache(), cfg.global[cfg.visit(@intCast(v)).layer], cfg, try c.g.sliceAxis(o.k, 2, 0, rows), try c.g.sliceAxis(o.v, 2, 0, rows), 0);
        kv.release(c, &view);
        try kv.setState(c, .conv, try c.g.sliceAxis(o.window, 1, rows, rows + kt - 1));
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
        var held = try kv.view(c, c.cache(), c.layer);
        defer kv.release(c, &held);
        const dv = kv.dense(held);
        const rows = ops.shape(dv.k)[2];
        // A sliding layer's rows reach at most `window - 1` keys before the block.
        const from = if (span > 0) @max(0, rows - @as(c_int, @intCast(span - 1))) else 0;
        o.k = try o.g.keep(p.k);
        o.v = try o.g.keep(p.v);
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
fn attendSlots(c: *block.Ctx, L: *const Layer, cfg: *const Config, p: *const Prep, sinks: Array) !Array {
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
    const bias = try g.matmul(states, try g.astype(L.rel_proj, dt));
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
        return slotsRead(g, try g.contiguous(try g.transpose(q, &.{ 0, 2, 1, 3 })), ks[0..n], vs[0..n], bias, sink_rows, @intCast(span), p.gate);
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
    const n = ks.len;
    const width = ops.shape(q)[2] * ops.shape(q)[3];
    var outs: [max_slots / slots_per_read]Array = undefined;
    var groups: usize = 0;
    var from: usize = 0;
    while (from < n) : (groups += 1) {
        const to = @min(n, from + slots_per_read);
        const lo: c_int = @intCast(from);
        const hi: c_int = @intCast(to);
        const part = try slotParts(g, try g.sliceAxis(q, 1, lo, hi), ks[from..to], vs[from..to], try g.sliceAxis(bias, 1, lo, hi), try g.sliceAxis(sinks, 1, lo, hi), window);
        outs[groups] = try splitJoin(g, part, try g.sliceAxis(gate_sigmoid, 0, lo * width, hi * width), hi - lo, ops.dtype(q));
        from = to;
    }
    return if (groups == 1) outs[0] else g.concat(outs[0..groups], 1);
}

/// `relslots.metal` for up to `slots_per_read` slots: each split's unnormalised sums `[splits, n * H, hd]` and (max,
/// sum) `[splits, n * H, 2]`, f32, as `splitParts`'.
pub fn slotParts(g: *Graph, q: Array, ks: []const Array, vs: []const Array, bias: Array, sinks: Array, window: c_int) ![2]Array {
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
    inputs[20] = try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32);
    return launchGrid(g, &relslots_kernel, &inputs, 2, .{
        .{ .shape = &.{ splits, ng * heads, hd }, .dtype = .float32 },
        .{ .shape = &.{ splits, ng * heads, 2 }, .dtype = .float32 },
    }, .{ splits * 32 * slot_groups, ops.shape(ks[0])[1], ng }, .{ 32 * slot_groups, 1, 1 }, &.{
        .{ "T", .{ .dtype = ops.dtype(q) } }, .{ "D", .{ .int = hd } },                    .{ "HQ", .{ .int = heads } },
        .{ "HKV", .{ .int = ops.shape(ks[0])[1] } }, .{ "R", .{ .int = ops.shape(bias)[3] } }, .{ "NS", .{ .int = ng } },
        .{ "NSG", .{ .int = slot_groups } },  .{ "KB", .{ .int = slot_keys } },            .{ "WINDOW", .{ .int = window } },
    });
}

/// A visit between its attention and its experts: the MLP's input, the router's picks and weights, the f32 residual
/// the MLP's output joins, the block stack when the layer opened a block, and the output projection (the attention
/// conv's newest inputs).
pub const Middle = struct { x: Array, idx: Array, weights: Array, residual: Array, blocks: ?Array, op: Array };

/// `layers.mid`: `attn` are the gated attention rows, `h` the visit's input stream, `normed` its unweighted norm,
/// `messages` `[4, S, D]` on a later loop pass.
/// `out_gate`: the output gate of a read `attend` left ungated (one row), applied inside the output projection.
pub fn mid(g: *Graph, L: *const Layer, w: *const Body, attn: Array, out_gate: ?Array, h: Array, normed: Array, blocks: ?Array, st: Array, messages: ?Array) !Middle {
    const cfg = &w.cfg;
    const d: c_int = @intCast(cfg.hidden_size);
    const s = ops.shape(h)[1];
    const kt: c_int = @intCast(cfg.sconv_kernel);
    const top_k: c_int = @intCast(cfg.top_k);
    const op = if (out_gate) |gt| try oGated(g, &L.o, attn, gt) else try layers.projectRows(g, &L.o, attn);
    const dt = ops.dtype(attn);
    if (!floating(ops.dtype(op))) return error.Inkling2StepDtype;
    const gates: ?*const layers.Gates = if (messages != null) &(L.gates orelse return error.Inkling2NoMessageGate) else null;
    const gt: Dtype = if (gates) |x| ops.dtype(x.down) else .float32;
    const msg3: Array = if (messages) |m| try g.sliceAxis(m, 0, 3, 4) else h;
    const pr = try g.fromSlice(f32, &.{ 1.0 / @sqrt(@as(f32, @floatFromInt(kt + 1))), cfg.rms_norm_eps }, &.{2});
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
        const xs = try norm(g, cfg, try gate(g, gs, o[3], msg3, state, 3, 1), @splat(L.post_norm), 1, dt, L.router);
        x = xs[0];
        scores = xs[1];
    } else scores = if (s > 1 and s <= small_rows) try routerRows(g, o[2], L.router) else try g.matmul(o[2], try g.transpose(L.router, &.{ 1, 0 }));
    const picks = try topK(g, scores, L.router_bias, try g.fromSlice(f32, &.{cfg.route_scale}, &.{1}), top_k);
    var opened: ?Array = null;
    if (L.boundary) {
        const saved = try g.expandDims(normed, 2);
        opened = if (blocks) |stack| try g.concat(&.{ stack, saved }, 2) else saved;
    }
    return .{ .x = x, .idx = picks[0], .weights = picks[1], .residual = o[0], .blocks = opened, .op = op };
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

/// The most rows the packed expert kernels serve: past it MLX's gather, which reads a bank once for every pick of
/// it, wins on bytes; under it the gather's fixed cost is most of a row.
pub const packed_rows = 8;

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

/// `layers.tail`. The visit's new conv state (off the next visit's path) is `st` behind the rows' k and v projections
/// `kp`, `vp`, their output projection and the MLP's output.
/// One row, a packed bank read in place: the picks' gate_up and the soft-capped SwiGLU as one launch, their down, the
/// weighted sum, the MLP's conv behind `st` and the final add as a second. Another bank, or more rows (their pairs
/// sorted by expert), go through MLX's gather, with the SwiGLU and the sum, conv and add as launches of their own.
pub fn experts(g: *Graph, L: *const Layer, cfg: *const Config, m: *const Middle, kp: Array, vp: Array, st: Array, open: ?*Open) !Tail {
    const dt = ops.dtype(m.x);
    const s = ops.shape(m.x)[1];
    const top_k: c_int = @intCast(cfg.top_k);
    const inter: c_int = @intCast(cfg.moe_intermediate_size);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const kt: c_int = @intCast(cfg.sconv_kernel);
    const n = s * top_k;
    // A few rows read both packed banks in place, their picks in (row, pick) order; one row either.
    const few = s > 1 and s <= packed_rows and 32 * top_k <= sure_threads and packedFormat(&L.gate_up, hidden) != null and packedFormat(&L.down, inter) != null;
    const pairs = try layers.pairs(g, cfg, m.x, m.idx);
    const sel = if (few) try g.reshape(m.idx, &.{n}) else pairs.sel;
    const caps = try g.fromSlice(f32, &.{ 1.0 / cfg.gate_softcap, cfg.gate_softcap, 1.0 / cfg.up_softcap, cfg.up_softcap }, &.{4});
    const conv_scale = try g.fromSlice(f32, &.{1.0 / @sqrt(@as(f32, @floatFromInt(kt + 1)))}, &.{1});
    const gu_scale = if (L.gate_up_scale.ctx != null) L.gate_up_scale else caps;
    const dn_scale = if (L.down_scale.ctx != null) L.down_scale else m.weights;
    const gu_scaled: TArg = .{ .int = @intFromBool(L.gate_up_scale.ctx != null) };
    const dn_scaled: TArg = .{ .int = @intFromBool(L.down_scale.ctx != null) };
    const act_out: [1]Out = .{.{ .shape = &.{ n, 1, inter }, .dtype = dt }};
    const tail_out: [2]Out = .{ .{ .shape = &.{ 1, s, hidden }, .dtype = dt }, .{ .shape = &.{ 1, s, hidden }, .dtype = dt } };
    const sorted = pairs.inverse != null;

    const act = if (if (s == 1 or few) packedFormat(&L.gate_up, hidden) else null) |fmt|
        try launchGrid(g, &ups_kernel, &.{ m.x, L.gate_up.w, L.gate_up.s, biasesOf(&L.gate_up), sel, gu_scale, caps }, 1, act_out, .{ 32, @divExact(inter, bank_rows), n }, .{ 32, bank_groups, 1 }, &.{
            .{ "T", .{ .dtype = dt } },                               .{ "K", .{ .int = hidden } }, .{ "I", .{ .int = inter } },
            .{ "GS", .{ .int = @intCast(L.gate_up.qp.group_size) } }, .{ "FMT", .{ .int = fmt } },  .{ "ROWS", .{ .int = bank_rows } },
            .{ "BITS", .{ .int = @intCast(L.gate_up.qp.bits) } },     .{ "S", .{ .dtype = ops.dtype(L.gate_up.s) } },
            .{ "SCALED", gu_scaled },                                 .{ "KSEL", .{ .int = top_k } },
        })
    else blk: {
        const gu = try layers.gather(g, pairs.rows, &L.gate_up, sel, sorted);
        if (!floating(ops.dtype(gu))) return error.Inkling2StepDtype;
        break :blk .{try activation(g, gu, sel, gu_scale, caps, L.gate_up_scale.ctx != null)};
    };

    // The down kernel's threadgroup holds every pick of its rows.
    const o = if (if ((s == 1 and 32 * top_k <= sure_threads) or few) packedFormat(&L.down, inter) else null) |fmt| blk: {
        const packed_out = try launchGrid(g, &downs_kernel, &.{ act[0], L.down.w, L.down.s, biasesOf(&L.down), sel, dn_scale, m.weights, m.residual, st, L.mlp_conv, conv_scale }, 2, .{ if (few) unused else tail_out[0], tail_out[1] }, .{ 32, top_k, s * @divExact(hidden, bank_rows) }, .{ 32, top_k, 1 }, &.{
            .{ "T", .{ .dtype = dt } },         .{ "WT", .{ .dtype = ops.dtype(L.mlp_conv) } },        .{ "I", .{ .int = inter } },
            .{ "D", .{ .int = hidden } },       .{ "GS", .{ .int = @intCast(L.down.qp.group_size) } }, .{ "FMT", .{ .int = fmt } },
            .{ "ROWS", .{ .int = bank_rows } }, .{ "KSEL", .{ .int = top_k } },                        .{ "KT", .{ .int = kt } },
            .{ "BITS", .{ .int = @intCast(L.down.qp.bits) } }, .{ "S", .{ .dtype = ops.dtype(L.down.s) } },
            .{ "SCALED", dn_scaled },           .{ "TAIL", .{ .int = @intFromBool(!few) } },
        });
        if (!few) break :blk packed_out;
        // The rows' MLP outputs are whole: the conv over them and the add, as the wide sum over one pick a row.
        const rows_at = try g.arange(0, @floatFromInt(s), 1, .uint32);
        break :blk try pickSum(g, try g.reshape(packed_out[1], &.{ s, 1, hidden }), rows_at, rows_at, caps, try g.ones(&.{ 1, s, 1 }, .float32), m.residual, st, L.mlp_conv, conv_scale, 1, false);
    } else blk: {
        const dn = try layers.gather(g, act[0], &L.down, sel, sorted);
        if (!floating(ops.dtype(dn))) return error.Inkling2StepDtype;
        // Unsorted pairs sit in (row, pick) order already.
        const inverse = pairs.inverse orelse try g.arange(0, @floatFromInt(n), 1, .uint32);
        break :blk try pickSum(g, dn, sel, inverse, dn_scale, m.weights, m.residual, st, L.mlp_conv, conv_scale, top_k, L.down_scale.ctx != null);
    };
    if (open) |v| {
        v.window = try v.g.keep(try g.concat(&.{ st, try g.astype(try g.concat(&.{ kp, vp, m.op, o[1] }, 2), .float32) }, 1));
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
    const p = (try prep(g, L, cfg, &f.proj, f.dyn, st, if (one and L.global) layers.stepTau(cfg, total) else 1.0, if (one) keysSeen(L, cfg, total) else 0)) orelse return null;
    const fuse = one and open == null and oFormat(&L.o, ops.dtype(h)) != null;
    const read = if (batch) try attendSlots(c, L, cfg, &p, f.proj[3]) else try attend(c, L, cfg, &p, f.proj[3], open, fuse);
    const m = try mid(g, L, w, read, if (fuse) p.gate else null, h, f.normed, blocks, st, f.messages);
    const t = try experts(g, L, cfg, &m, f.proj[4], f.proj[5], st, open);
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
