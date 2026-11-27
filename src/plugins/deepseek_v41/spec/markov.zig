//! The DSpark draft's Markov chain in B + 1 launches (the plan is MLX-free in markov_plan.zig). The model's
//! draftRows hands `chain` the head's logits;
//! without it every draft position is five dependent op groups (a take, a [1, rank] x [rank, vocab] matmul in the
//! activation dtype, its cast, the add to the logits row, an argmax and its cast) and the confidence head's dot.
//!
//! The kernels (kernels/markov_*.metal): position k is one launch (`markov_step`). Each of its threadgroups first
//! reduces the previous position's threadgroup bests to that position's token (MLX's argmax order: NaN first, then the
//! larger value, then the lower index, a strict total order, so any reduction tree agrees), reads that token's
//! markov_embed row, and sums its rows of the bias as MLX's gemv sums them (lane, element order, the shuffle-down tree,
//! the round to the activation dtype), adds the logits row in f32 and keeps its best (value, index) pair. One more launch
//! (`markov_final`) reduces each position's pairs to its token and gives the confidence logit as MLX's dot_product does
//! (where every product of its dot is exact: `plan.confFused`). No host sync in between; the draft's one eval reads them.
//!
//! The bytes are the op chain's: the same products (exact: two 8- or 11-bit significands), sums in the same order, the
//! same roundings.
//!
//! An odd position reads the rows from the far end (zig-zag): the rows the position before read last are the ones the
//! system cache may still hold. Which rows a threadgroup sums does not change a row's bytes.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const plan = @import("markov_plan.zig");

const A = mlx.mlx_array;

pub const header = kernelSource("plugins/deepseek_v41/kernels/markov_header.metal");
pub const step_source = kernelSource("plugins/deepseek_v41/kernels/markov_step.metal");
pub const final_source = kernelSource("plugins/deepseek_v41/kernels/markov_final.metal");

var step_kernel: ?mlx.mlx_fast_metal_kernel = null;
var step_sample_kernel: ?mlx.mlx_fast_metal_kernel = null;
var final_kernel: ?mlx.mlx_fast_metal_kernel = null;
/// The confidence head whose weights were checked (`plan.exact16`) and the answer.
var conf_seen: ?*anyopaque = null;
var conf_exact: bool = false;
var refusal_said: bool = false;

/// A draft's inputs, the model's arrays.
pub const In = struct {
    /// f32 [B, vocab]: the head's logits
    logits: A,
    /// [B, dim]: the last stage's collapse
    hout: A,
    /// int32 [1]: the token before position 0
    t1: A,
    /// markov_embed [rows, rank] (rows >= vocab: the row of every token a step can read) and markov_head [vocab, rank]
    /// (the activation dtype)
    emb: A,
    mh: A,
    /// f32 [1, dim + rank]: the confidence head
    cw: A,
    want_conf: bool,
    /// the biased rows too (the sampled path's)
    keep: bool,
    /// Each position's token drawn from softmax(row / T) (Gumbel-max) instead of its argmax.
    sample: ?Sample = null,
};

/// A sampled draft's noise: f32 [B, vocab] uniforms in [0, 1) and f32 [1] 1 / T.
pub const Sample = struct { u: A, inv_t: A };

/// New arrays, the caller's to free: the tokens (int32 [B]), the confidence logits (f32 [B], `want_conf` and
/// `plan.confFused`), the biased rows (f32 [B, vocab], `keep`).
pub const Out = struct {
    toks: A,
    conf: ?A = null,
    rows: ?A = null,
};

fn dt(x: A) plan.Dt {
    return switch (mlx.mlx_array_dtype(x)) {
        .bfloat16 => .bf16,
        .float16 => .f16,
        .float32 => .f32,
        else => .other,
    };
}

fn dimOf(x: A, axis: usize) usize {
    const sh = mlx.getShape(x);
    return if (axis < sh.len) @intCast(sh[axis]) else 0;
}

fn buildKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, hdr: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, hdr, true, false);
    if (k.ctx == null) return error.MarkovKernel;
    return k;
}

const sample_header = header ++ "#define DSV41_MK_SAMPLE 1\n";

fn kernels(sampled: bool) !struct { step: mlx.mlx_fast_metal_kernel, final: mlx.mlx_fast_metal_kernel } {
    if (!sampled and step_kernel == null) step_kernel = try buildKernel("dsv41_markov_step", &.{ "lg", "emb", "mh", "t1", "pin" }, &.{ "pout", "row" }, step_source, header);
    if (sampled and step_sample_kernel == null) step_sample_kernel = try buildKernel("dsv41_markov_step_sample", &.{ "lg", "emb", "mh", "t1", "pin", "gu", "it" }, &.{ "pout", "row" }, step_source, sample_header);
    if (final_kernel == null) final_kernel = try buildKernel("dsv41_markov_final", &.{ "parts", "t1", "hout", "emb", "cw" }, &.{ "toks", "conf" }, final_source, header);
    return .{ .step = if (sampled) step_sample_kernel.? else step_kernel.?, .final = final_kernel.? };
}

/// Whether the confidence head's weights are all 16-bit floats' values, checked once per head.
fn confExact(cw: A) !bool {
    if (conf_seen == cw.ctx) return conf_exact;
    try mlx.check(mlx.mlx_array_eval(cw));
    const n = mlx.mlx_array_size(cw);
    const data = mlx.mlx_array_data_float32(cw) orelse return error.NoData;
    conf_exact = plan.exact16(data[0..n]);
    conf_seen = cw.ctx;
    log.info("deepseek_v41: Markov chain: the confidence head's {d} weights are {s}\n", .{ n, if (conf_exact) "16-bit floats' values: its logit is the last kernel's" else "not all 16-bit floats' values: its logit stays on MLX's ops" });
    return conf_exact;
}

fn apply(kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const A, s: mlx.mlx_stream, out: []A) !void {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    for (out, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(o, ov, i)) catch |e| {
            for (out[0 .. i + 1]) |x| _ = mlx.mlx_array_free(x);
            return e;
        };
    }
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// The fused chain's arrays for one draft (lazy), or null where `plan.refusal` keeps the op chain.
pub fn chain(s: mlx.mlx_stream, in: In) !?Out {
    const sh: plan.Shape = .{
        .act = dt(in.mh),
        .embed = dt(in.emb),
        .hout = dt(in.hout),
        .vocab = dimOf(in.mh, 0),
        .rank = dimOf(in.mh, 1),
        .dim = dimOf(in.hout, 1),
        .block = dimOf(in.logits, 0),
    };
    if (plan.refusal(sh)) |r| {
        if (!refusal_said) log.info("deepseek_v41: Markov chain: the draft keeps the op chain ({s})\n", .{@tagName(r)});
        refusal_said = true;
        return null;
    }
    // markov_embed may hold more rows than the chain scores (a draft vocabulary's: `t1` is any id's row).
    if (dimOf(in.emb, 0) < sh.vocab or dimOf(in.emb, 1) != sh.rank or dimOf(in.logits, 1) != sh.vocab or dimOf(in.hout, 0) != sh.block) return error.MarkovShape;
    const k = try kernels(in.sample != null);
    if (in.sample) |sm| if (dimOf(sm.u, 0) != sh.block or dimOf(sm.u, 1) != sh.vocab) return error.MarkovShape;
    const conf = in.want_conf and plan.confFused(sh, try confExact(in.cw));
    const B = sh.block;
    const V = sh.vocab;
    const nt = plan.step_groups;
    const act = mlx.mlx_array_dtype(in.mh);
    const edt = mlx.mlx_array_dtype(in.emb);
    // position 0's pin, never read: 8 elements, so MLX binds it in the device address space as every later pin
    const zero8: [8]u32 = @splat(0);
    var pin = mlx.mlx_array_new_data(&zero8, &[_]c_int{8}, 1, .uint32);
    defer _ = mlx.mlx_array_free(pin);
    var rows: [plan.max_block]A = undefined;
    var n_rows: usize = 0;
    defer for (rows[0..n_rows]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (0..B) |p| {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ci((p + 1) * nt * 2)}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ci(if (in.keep) V else 1)}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(nt * plan.step_threads), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(plan.step_threads), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", act));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "E", edt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", ci(sh.rank)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", ci(V)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", ci(p)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NT", ci(nt)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KEEP", if (in.keep) 1 else 0));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ZZ", 1));
        var o: [2]A = undefined;
        if (in.sample) |sm| try apply(k.step, cfg, &.{ in.logits, in.emb, in.mh, in.t1, pin, sm.u, sm.inv_t }, s, &o) else try apply(k.step, cfg, &.{ in.logits, in.emb, in.mh, in.t1, pin }, s, &o);
        _ = mlx.mlx_array_free(pin);
        pin = o[0];
        if (in.keep) {
            rows[n_rows] = o[1];
            n_rows += 1;
        } else _ = mlx.mlx_array_free(o[1]);
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ci(B)}, 1, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ci(if (conf) B else 1)}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(plan.final_threads), ci(B), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(plan.final_threads), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "B", ci(B)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NT", ci(nt)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", ci(sh.dim)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", ci(sh.rank)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "H", mlx.mlx_array_dtype(in.hout)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "E", edt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CONF", if (conf) 1 else 0));
    var f: [2]A = undefined;
    try apply(k.final, cfg, &.{ pin, in.t1, in.hout, in.emb, in.cw }, s, &f);
    var out: Out = .{ .toks = f[0] };
    errdefer {
        _ = mlx.mlx_array_free(out.toks);
        if (out.conf) |x| _ = mlx.mlx_array_free(x);
    }
    if (conf) out.conf = f[1] else _ = mlx.mlx_array_free(f[1]);
    if (in.keep) {
        const vec = mlx.mlx_vector_array_new_data(&rows, n_rows);
        defer _ = mlx.mlx_vector_array_free(vec);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_stack_axis(&r, vec, 0, s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        out.rows = r;
    }
    return out;
}

/// MLX builds a custom kernel's library at its first launch, one per template instance (B step kernels and the last
/// one: a few tens of ms each). At load, one chain over zero logits builds the ones `m`'s drafts run, so the first draft
/// runs at speed.
pub fn warm(s: mlx.mlx_stream, emb: A, mh: A, cw: A, hout_dtype: mlx.mlx_dtype, block: usize, dim: usize, want_conf: bool, sampled: bool) !void {
    if (block == 0) return;
    if (sampled) try warm(s, emb, mh, cw, hout_dtype, block, dim, want_conf, false);
    var lg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lg);
    try mlx.check(mlx.mlx_zeros(&lg, &[_]c_int{ ci(block), ci(dimOf(mh, 0)) }, 2, .float32, s));
    var hout = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hout);
    try mlx.check(mlx.mlx_zeros(&hout, &[_]c_int{ ci(block), ci(dim) }, 2, hout_dtype, s));
    const zero: i32 = 0;
    const t1 = mlx.mlx_array_new_data(&zero, &[_]c_int{1}, 1, .int32);
    defer _ = mlx.mlx_array_free(t1);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    var it = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(it);
    if (sampled) {
        const half = mlx.mlx_array_new_float(0.5);
        defer _ = mlx.mlx_array_free(half);
        try mlx.check(mlx.mlx_full(&u, &[_]c_int{ ci(block), ci(dimOf(mh, 0)) }, 2, half, .float32, s));
        const one: f32 = 1;
        _ = mlx.mlx_array_free(it);
        it = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .float32);
    }
    const o = (try chain(s, .{ .logits = lg, .hout = hout, .t1 = t1, .emb = emb, .mh = mh, .cw = cw, .want_conf = want_conf, .keep = sampled, .sample = if (sampled) .{ .u = u, .inv_t = it } else null })) orelse return;
    defer {
        _ = mlx.mlx_array_free(o.toks);
        if (o.conf) |x| _ = mlx.mlx_array_free(x);
        if (o.rows) |x| _ = mlx.mlx_array_free(x);
    }
    var ev: [3]A = .{ o.toks, o.conf orelse o.toks, o.rows orelse o.toks };
    const vec = mlx.mlx_vector_array_new_data(&ev, ev.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

// ── tests ──

const testing = std.testing;

fn normal(shape: []const c_int, dtype: mlx.mlx_dtype, scale: f32, seed: u64, s: mlx.mlx_stream) !A {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dtype, 0, scale, key, s));
    return o;
}

/// Pearson's chi-square of each position's draws against the mean of the proposals they were drawn from, over the
/// `bins` tokens the proposals weigh most and one bin for the rest.
fn chiSquare(obs: []const f64, expect: []const f64, bins: usize) f64 {
    var order: [4096]usize = undefined;
    for (order[0..expect.len], 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order[0..expect.len], expect, struct {
        fn gt(e: []const f64, a: usize, b: usize) bool {
            return e[a] > e[b];
        }
    }.gt);
    var x: f64 = 0;
    var ro: f64 = 0;
    var re: f64 = 0;
    for (order[0..expect.len], 0..) |i, r| {
        if (r < bins) {
            x += (obs[i] - expect[i]) * (obs[i] - expect[i]) / expect[i];
        } else {
            ro += obs[i];
            re += expect[i];
        }
    }
    return x + (ro - re) * (ro - re) / re;
}

test "markov chain sampled (GPU): each position's token is a draw from softmax(row / T) of its own biased row over fixed seeds; drawn at T 1 against T 0.7's proposal it fails" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const V = 2048;
    const R = 128;
    const D = 128;
    const B = 3;
    const lg = try normal(&.{ B, V }, .float32, 2.5, 1, s);
    defer _ = mlx.mlx_array_free(lg);
    const emb = try normal(&.{ V, R }, .bfloat16, 0.25, 2, s);
    defer _ = mlx.mlx_array_free(emb);
    const mh = try normal(&.{ V, R }, .bfloat16, 0.25, 3, s);
    defer _ = mlx.mlx_array_free(mh);
    const hout = try normal(&.{ B, D }, .bfloat16, 1, 4, s);
    defer _ = mlx.mlx_array_free(hout);
    const cw = try normal(&.{ 1, D + R }, .float32, 0.05, 5, s);
    defer _ = mlx.mlx_array_free(cw);
    const t1v: i32 = 7;
    const t1 = mlx.mlx_array_new_data(&t1v, &[_]c_int{1}, 1, .int32);
    defer _ = mlx.mlx_array_free(t1);
    const N = 3000;
    const inv_t: f32 = 1.0 / 0.7;
    var chi: [2][B]f64 = undefined;
    for (0..2) |arm| {
        // arm 1, planted: the kernel draws at T 1 while the proposal is read at T 0.7
        const drawn: f32 = if (arm == 0) inv_t else 1.0;
        const it = mlx.mlx_array_new_data(&drawn, &[_]c_int{1}, 1, .float32);
        defer _ = mlx.mlx_array_free(it);
        const read = mlx.mlx_array_new_float(inv_t);
        defer _ = mlx.mlx_array_free(read);
        const obs = try testing.allocator.alloc(f64, B * V);
        defer testing.allocator.free(obs);
        const expect = try testing.allocator.alloc(f64, B * V);
        defer testing.allocator.free(expect);
        @memset(obs, 0);
        @memset(expect, 0);
        for (0..N) |trial| {
            var key = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(key);
            try mlx.check(mlx.mlx_random_key(&key, 1000 + trial));
            const lo = mlx.mlx_array_new_float(0);
            defer _ = mlx.mlx_array_free(lo);
            const hi = mlx.mlx_array_new_float(1);
            defer _ = mlx.mlx_array_free(hi);
            var u = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(u);
            try mlx.check(mlx.mlx_random_uniform(&u, lo, hi, &[_]c_int{ B, V }, 2, .float32, key, s));
            const o = (try chain(s, .{ .logits = lg, .hout = hout, .t1 = t1, .emb = emb, .mh = mh, .cw = cw, .want_conf = false, .keep = true, .sample = .{ .u = u, .inv_t = it } })) orelse return error.MarkovRefused;
            defer {
                _ = mlx.mlx_array_free(o.toks);
                if (o.rows) |x| _ = mlx.mlx_array_free(x);
            }
            var scaled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scaled);
            try mlx.check(mlx.mlx_multiply(&scaled, o.rows.?, read, s));
            var q = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q);
            try mlx.check(mlx.mlx_softmax_axis(&q, scaled, -1, true, s));
            try mlx.check(mlx.mlx_array_eval(q));
            try mlx.check(mlx.mlx_array_eval(o.toks));
            const qd = mlx.mlx_array_data_float32(q) orelse return error.NoData;
            const td = mlx.mlx_array_data_int32(o.toks) orelse return error.NoData;
            for (0..B) |k| {
                obs[k * V + @as(usize, @intCast(td[k]))] += 1;
                for (0..V) |i| expect[k * V + i] += qd[k * V + i];
            }
        }
        for (0..B) |k| chi[arm][k] = chiSquare(obs[k * V ..][0..V], expect[k * V ..][0..V], 24);
    }
    std.debug.print("[markov sample] chi-square of {d} draws a position against the mean proposal (24 bins + the rest, 24 dof): at the proposal's T {d:.1}/{d:.1}/{d:.1}; drawn at T 1 {d:.1}/{d:.1}/{d:.1}\n", .{ N, chi[0][0], chi[0][1], chi[0][2], chi[1][0], chi[1][1], chi[1][2] });
    // 24 degrees of freedom: 75 is past the 1e-6 tail.
    for (chi[0]) |x| try testing.expect(x < 75);
    for (chi[1]) |x| try testing.expect(x > 300);
}
