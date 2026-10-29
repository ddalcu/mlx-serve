//! sdk/blocks/moe/router_kernel.
//! Owns RouterMode, ROUTER_MAX_EXPERTS, moeRouterSource, ROUTER_SOURCES, ROUTER_NAMES, router_kernels, and 13 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const testing = std.testing;
const testRandUniformBf16 = @import("../../testing/parity.zig").testRandUniformBf16;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const groupLimitedRouting = @import("routing.zig").groupLimitedRouting;
const MoeRouting = @import("routing.zig").MoeRouting;
const hy3RoutingChain = @import("routing.zig").hy3RoutingChain;
const kolibriRoutingChain = @import("routing.zig").kolibriRoutingChain;
const moeRoutingChain = @import("routing.zig").moeRoutingChain;

// ── Fused MoE router (top-K selection + weight normalization in ONE kernel) ──
//
// The op chains above are ~11 dispatches each — sigmoid/softmax, bias add,
// negate, argpartition (a full 256-wide sort), slice, take_along_axis, sum,
// eps add, divide, scale multiply, cast — over a [rows, E] tensor that is 512
// bytes wide at decode. Every one of them is launch-latency bound, and a MoE
// model pays the whole chain once per LAYER: 40 layers x 11 = 440 dispatches
// per token on Laguna, none of which move meaningful data.
//
// One threadgroup per row does all of it: phase 1 computes the selection key
// for every expert into threadgroup memory, phase 2 runs K rounds of
// simdgroup-wide argmax with a LOWEST-INDEX tie-break, phase 3 normalizes the
// selected weights. Selection is exact integer/compare logic, so the picked
// expert set is the same set the chain picks whenever the keys are distinct.
//
// The ARITHMETIC deliberately replicates the chain's rather than improving on
// it — same `fast::exp`, same reciprocal-multiply softmax normalization, same
// MLX `Sigmoid` two-sided form, same intermediate rounding to the output dtype
// at the same points. A "more accurate" fused router is still a DIFFERENT
// model: the first live A/B of this kernel computed the softmax weights in one
// f32 pass and diverged from the chain on gemma4-26B-A4B at token ~80, which
// is the known INT4 near-tie argmax class amplified by an avoidable numerical
// change. The tests below therefore assert BIT equality with the chain, not
// just no-worse-than-reference; only the reduction ORDER over the expert axis
// differs (a simd tree vs MLX's N_READS blocking), which cannot move a bf16
// result except at an exact rounding boundary.
//
// Engagement is LOGGED, never
// inferred — a declined kernel is output-identical to the chain.
/// `sigmoid_bias_grouped` is `sigmoid_bias` with a GROUP LIMIT spliced
/// between the bias and the top-k: the experts are split into NG equal groups,
/// each group is scored by the SUM OF ITS TOP TWO, and every expert outside the
/// best TOPKG groups is pushed to the composed chain's own finite -1e4 penalty
/// before selection. The composed chain is ~28 dispatches per MoE layer x 41 layers,
/// and a dispatch prices at 1.76 us on this arch (diag dispatch-probe
/// slope over 1000 injected ops), so that chain alone is ~2 ms of a 19 ms token.
/// `softmax_gate` is `softmax` plus the shared-expert gate's logit, a dot of one x row with a flat
/// weight vector that the simdgroups sitting out the selection compute while it runs (the matmul it
/// replaces is two dependent launches ahead of the router). It needs at least two simdgroups.
/// `logit_bias` (kolibri1) selects on `logits + bias` and weights by `sigmoid(logits)`.
pub const RouterMode = enum(u8) { softmax = 0, sigmoid_bias = 1, sigmoid_bias_grouped = 2, softmax_gate = 3, logit_bias = 4 };

/// Largest expert count the kernel will stage in threadgroup memory. The
/// sigmoid arm keeps two f32 planes (key + raw weight), so 2048 experts is
/// 16 KiB of the 32 KiB budget.
const ROUTER_MAX_EXPERTS: c_int = 2048;

fn moeRouterSource(comptime mode: RouterMode) [:0]const u8 {
    const logit_sel = mode == .logit_bias;
    const sigmoid = mode == .sigmoid_bias or mode == .sigmoid_bias_grouped or logit_sel;
    const grouped = mode == .sigmoid_bias_grouped;
    const gate = mode == .softmax_gate;
    return std.fmt.comptimePrint(kernelSource("kernels/moe/router.fmt.metal"), .{
        if (grouped)
            "threadgroup float rw[NE];\nthreadgroup float gs[NG];"
        else if (sigmoid)
            "threadgroup float rw[NE];"
        else
            "",
        if (gate)
            "threadgroup float gp[TG / 32];"
        else
            "",
        if (sigmoid)
            // MLX's `Sigmoid` unary op, verbatim (unary_ops.h): the two-sided
            // form, not 1/(1+exp(-x)) — they differ in the last ulp. `exp` is
            // spelled PRECISE because copying upstream's TEXT is not copying
            // its ARITHMETIC: MLX's metallib is built `-fno-fast-math`
            // (kernels/CMakeLists.txt), so its bare `metal::exp` is the precise
            // one, while this source is JIT-compiled at MathMode::Safe where
            // the same spelling lowers to the fast variant. Measured 13.8% of
            // 1M random f32 sigmoids 1 ulp apart, and 6.1% of the routing
            // weights that come out of them; precise is 0/1M. The softmax arm
            // below spells `fast::exp` for the mirror-image reason (softmax.h
            // asks for fast explicitly). Costs ~1% of the kernel at an 8192-row
            // prefill width and nothing measurable at decode.
            kernelSource("kernels/moe/router_sigmoid_score.metal") ++ "\n  // kolibri1 selects on the raw logit + bias; the others on sigmoid + bias.\n  rk[e] = " ++ (if (logit_sel) "l" else "sg") ++ " + float(bias[e]);\n  rw[e] = sg;"
        else
            "  rk[e] = float(logits[rbase + e]);",
        if (gate)
            // Simdgroups 1.. take the shared-expert gate's dot while simdgroup 0 selects; their partials
            // meet after the selection, so the dot never waits on it nor the selection on the dot.
            kernelSource("kernels/moe/router_gate_dot.metal")
        else
            "",
        if (grouped)
            // The group limit (`group_limited_topk` shape). One lane
            // per group scores it by the SUM OF ITS TOP TWO biased experts, then
            // TOPKG rounds of the same max-then-mask/lowest-index selection the
            // expert top-k uses, then every expert in a losing group is pushed to
            // -1e4 — the composed chain's own FINITE penalty, not -inf: biased
            // scores are sigmoid+bias so they live well inside +/-2, and a
            // non-finite sentinel in a tensor that feeds selection is how the
            // composed causal-mask arm once shipped an all-NaN path that scored a
            // perfect parity max_err of 0.
            kernelSource("kernels/moe/router_group_limit.metal")
        else
            "",
        if (sigmoid)
            ""
        else
            // MLX's `softmax_single_row`, reduction tree included. It launches
            // ceil(NE/4) threads, each summing FOUR consecutive elements in
            // order, then one `simd_sum` per simdgroup, then a final `simd_sum`
            // over the per-simdgroup partials sitting in lanes 0..SIMDS-1 with
            // zeros above. Emulating those virtual threads 32 at a time
            // reproduces the identical butterfly on identical values, which is
            // what makes the rounded bf16 probability bit-equal; a plain
            // strided sum over the same data was 1 ulp off on real inputs.
            kernelSource("kernels/moe/router_softmax_sum.metal"),
        if (sigmoid)
            "    w = rw[myidx];"
        else
            // softmax(...) writes probabilities in the OUTPUT dtype, so the
            // per-expert rounding happens here, before the top-K sum.
            "    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));",
        if (sigmoid)
            // The hy3 chain sums FLOAT32 sigmoid scores, so the accumulator is
            // float — but still ascending, still sequential.
            "  for (int j = 0; j < NK; ++j) tot += simd_shuffle(w, ushort(j));"
        else
            "  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));",
        if (sigmoid)
            kernelSource("kernels/moe/router_sigmoid_norm.metal")
        else
            // `sum` over the K probabilities lands back in the output dtype
            // before the divide, exactly as mlx_sum_axis + mlx_divide do.
            "  w = w / float(TOUT(tot));",
        if (gate)
            kernelSource("kernels/moe/router_gate_store.metal")
        else
            "",
    });
}

pub const ROUTER_SOURCES = [5][:0]const u8{ moeRouterSource(.softmax), moeRouterSource(.sigmoid_bias), moeRouterSource(.sigmoid_bias_grouped), moeRouterSource(.softmax_gate), moeRouterSource(.logit_bias) };
const ROUTER_NAMES = [5][*:0]const u8{ "mlxserve_moe_router_softmax", "mlxserve_moe_router_sigmoid", "mlxserve_moe_router_sigmoid_grouped", "mlxserve_moe_router_softmax_gate", "mlxserve_moe_router_logit_bias" };
var router_kernels: [5]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getMoeRouterKernel(mode: RouterMode) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @backingInt(mode);
    if (router_kernels[mi]) |k| return k;
    const softmax_inputs = [_][*:0]const u8{"logits"};
    const gate_inputs = [_][*:0]const u8{ "logits", "gx", "gw" };
    const sigmoid_inputs = [_][*:0]const u8{ "logits", "bias", "scale" };
    const input_names: []const [*:0]const u8 = switch (mode) {
        .softmax => &softmax_inputs,
        .softmax_gate => &gate_inputs,
        .sigmoid_bias, .sigmoid_bias_grouped, .logit_bias => &sigmoid_inputs,
    };
    const plain_outputs = [_][*:0]const u8{ "inds", "scores" };
    const gate_outputs = [_][*:0]const u8{ "inds", "scores", "glogit" };
    const output_names: []const [*:0]const u8 = if (mode == .softmax_gate) &gate_outputs else &plain_outputs;
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(output_names.ptr, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        ROUTER_NAMES[mi],
        in_vec,
        out_vec,
        ROUTER_SOURCES[mi],
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    router_kernels[mi] = kernel;
    return kernel;
}

/// Threads per row. Selection needs a full simdgroup; phase 1 wants enough
/// threads to cover the expert row without a long strided loop.
pub fn moeRouterThreadGroup(num_experts: c_int) c_int {
    const rounded = @divTrunc(num_experts + 31, 32) * 32;
    return @max(@as(c_int, 32), @min(@as(c_int, 256), rounded));
}

/// Everything the cached kernel config bakes in. A mismatch re-arms it.
/// Full-shape component for cached metal-kernel config keys. A cached config
/// bakes the OUTPUT SHAPES (and grid) of the call that built it, so the key
/// must carry the caller's actual dims: keying on a PRODUCT (rows, n) collides
/// across layouts with equal element counts and hands the colliding caller an
/// output labeled with the FIRST caller's shape. Live (Laguna XS):
/// a MoE-layer SwiGLU at a 16-token hot-cache-restore prefill is (1,16,512) =
/// 8192 elements — the same count/rank/dtype as a dense mlp_only layer at
/// decode (1,1,8192) — and the mislabeled activation killed the server in the
/// dense down-proj matmul. Every cached config key whose config bakes an
/// input-derived shape uses this instead of derived products.
pub const ShapeKey = struct {
    dims: [8]c_int = @splat(0),
    len: u8 = 0,
    pub fn from(sh: []const c_int) ShapeKey {
        var key = ShapeKey{};
        for (sh, 0..) |d, i| key.dims[i] = d;
        key.len = @intCast(sh.len);
        return key;
    }
};

const RouterCfgKey = struct {
    shape: ShapeKey,
    k: c_int,
    out_dtype: mlx.mlx_dtype,
    norm: bool,
    scale: f32,
    n_group: c_int = 0,
    topk_group: c_int = 0,
    gate_k: c_int = 0,
};
const RouterCfgSlot = struct {
    cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null },
    key: RouterCfgKey = std.mem.zeroes(RouterCfgKey),
    scale_arr: mlx.mlx_array = .{ .ctx = null },
};
var router_cfg_cache: [5]RouterCfgSlot = .{ .{}, .{}, .{}, .{}, .{} };

pub var moe_router_fused_override: ?bool = null;
var moe_router_engaged: bool = false;

pub fn moeRouterFusedEnabled() bool {
    return moe_router_fused_override orelse true;
}

/// The shared-expert gate's logit, folded into the softmax router: `x` [rows * gate_k] and the flat
/// weight vector `w` [gate_k], one logit per row.
pub const RouterGate = struct { x: mlx.mlx_array, w: mlx.mlx_array };

/// One-kernel replacement for `moeRoutingChain` / `hy3RoutingChain`.
/// Returns null when the geometry or dtype is outside the supported set, so
/// every caller keeps its chain fallback.
pub fn moeRouterTopK(
    s: mlx.mlx_stream,
    logits: mlx.mlx_array,
    bias: mlx.mlx_array,
    k: c_int,
    mode_in: RouterMode,
    route_norm: bool,
    route_scale: f32,
    out_dtype: mlx.mlx_dtype,
    n_group: c_int,
    topk_group: c_int,
    gate: ?RouterGate,
) !?MoeRouting {
    if (!moeRouterFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    const mode: RouterMode = if (gate != null and mode_in == .softmax) .softmax_gate else mode_in;
    const sigmoid_mode = mode == .sigmoid_bias or mode == .sigmoid_bias_grouped or mode == .logit_bias;
    const lsh = mlx.getShape(logits);
    if (lsh.len < 1 or lsh.len > 4) return null;
    const num_experts = lsh[lsh.len - 1];
    if (num_experts < k or num_experts > ROUTER_MAX_EXPERTS) return null;
    if (k < 1 or k > 32) return null;
    const ldt = mlx.mlx_array_dtype(logits);
    if (ldt != .bfloat16 and ldt != .float16 and ldt != .float32) return null;
    if (sigmoid_mode and bias.ctx == null) return null;
    if (mode == .sigmoid_bias_grouped) {
        // One lane per group, one bit per group in the keep mask, and a group
        // score that is a top-2 sum. Anything outside that declines to the
        // composed chain rather than computing a different model.
        if (n_group <= 1 or n_group > 32) return null;
        if (topk_group <= 0 or topk_group > n_group) return null;
        if (@rem(num_experts, n_group) != 0) return null;
        if (@divExact(num_experts, n_group) < 2) return null;
    }

    var rows: c_int = 1;
    for (lsh[0 .. lsh.len - 1]) |d| rows *= d;
    if (rows < 1) return null;
    // The gate logit: one flat weight vector, x holding one row of it per router row, in a threadgroup
    // that has a simdgroup to spare for every 32 threads.
    var gate_k: c_int = 0;
    if (gate) |gt| {
        const wn = mlx.mlx_array_size(gt.w);
        if (wn == 0 or wn > 65536 or mlx.mlx_array_size(gt.x) != @as(usize, @intCast(rows)) * wn) return null;
        const xd = mlx.mlx_array_dtype(gt.x);
        const wd = mlx.mlx_array_dtype(gt.w);
        if ((xd != .bfloat16 and xd != .float16) or (wd != .bfloat16 and wd != .float16)) return null;
        gate_k = @intCast(wn);
    }

    const tg = moeRouterThreadGroup(num_experts);
    // The gate's dot runs on the simdgroups that sit out the selection: with one there is none to run it.
    if (mode == .softmax_gate and tg < 64) return moeRouterTopK(s, logits, bias, k, .softmax, route_norm, route_scale, out_dtype, n_group, topk_group, null);

    // The whole point of the kernel is to spend ONE dispatch where the chain
    // spent a dozen; rebuilding the config every call would just move that cost
    // to the CPU side (measured +0.16 ms of graph-build per token on Laguna).
    // A model's routing geometry is fixed, so one cached config per mode covers
    // decode, and prefill re-arms it once per width.
    const key = RouterCfgKey{
        .shape = ShapeKey.from(lsh),
        .k = k,
        .out_dtype = out_dtype,
        .norm = route_norm,
        .scale = route_scale,
        .n_group = n_group,
        .topk_group = topk_group,
        .gate_k = gate_k,
    };
    const slot = &router_cfg_cache[@backingInt(mode)];
    if (slot.cfg.ctx == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(slot.cfg);
        if (slot.scale_arr.ctx != null) _ = mlx.mlx_array_free(slot.scale_arr);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        // Outputs keep the caller's leading dims with the expert axis replaced by K.
        var out_shape: [4]c_int = undefined;
        for (lsh[0 .. lsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[lsh.len - 1] = k;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, out_dtype));
        if (mode == .softmax_gate) {
            // One logit per router row.
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{rows}, 1, out_dtype));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GK", gate_k));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "TOUT", out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NE", num_experts));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NK", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TG", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NORM", @intFromBool(route_norm)));
        if (mode == .sigmoid_bias_grouped) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NG", n_group));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GSZ", @divExact(num_experts, n_group)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPKG", topk_group));
        }
        slot.* = .{ .cfg = config, .key = key, .scale_arr = mlx.mlx_array_new_float(route_scale) };
    }
    const config = slot.cfg;

    const softmax_inputs = [_]mlx.mlx_array{logits};
    const gate_inputs = [_]mlx.mlx_array{ logits, if (gate) |gt| gt.x else logits, if (gate) |gt| gt.w else logits };
    const sigmoid_inputs = [_]mlx.mlx_array{ logits, bias, slot.scale_arr };
    const inputs_arr: []const mlx.mlx_array = switch (mode) {
        .softmax => &softmax_inputs,
        .softmax_gate => &gate_inputs,
        .sigmoid_bias, .sigmoid_bias_grouped, .logit_bias => &sigmoid_inputs,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getMoeRouterKernel(mode);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != (if (mode == .softmax_gate) @as(usize, 3) else 2)) return error.MetalKernelBadOutputCount;

    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    try mlx.check(mlx.mlx_vector_array_get(&inds, outputs_vec, 0));
    var scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_vector_array_get(&scores, outputs_vec, 1));
    var gate_logit = mlx.mlx_array{ .ctx = null };
    if (mode == .softmax_gate) {
        gate_logit = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&gate_logit, outputs_vec, 2));
    }

    if (!moe_router_engaged) {
        moe_router_engaged = true;
        log.info("[moe] fused router kernel engaged: mode={s} E={d} K={d} tg={d}\n", .{ @tagName(mode), num_experts, k, tg });
    }
    return .{ .inds = inds, .norm_scores = scores, .gate_logit = gate_logit };
}

// ── tests ──

/// Host-side reference for one router row: exact top-K by key with the
/// LOWEST-INDEX tie-break, weights in f64. Returns the selected ids in
/// descending-key order, matching the kernel's emission order.
fn routerRefRow(keys: []const f64, k: usize, out_ids: []usize) void {
    var taken = std.testing.allocator.alloc(bool, keys.len) catch unreachable;
    defer std.testing.allocator.free(taken);
    @memset(taken, false);
    for (0..k) |r| {
        var best: f64 = -std.math.inf(f64);
        var bidx: usize = 0;
        var found = false;
        for (keys, 0..) |v, i| {
            if (taken[i]) continue;
            if (!found or v > best) {
                best = v;
                bidx = i;
                found = true;
            }
        }
        taken[bidx] = true;
        out_ids[r] = bidx;
    }
}

test "fused MoE router reproduces the softmax routing chain's top-K and beats its weights" {
    // Selection is discrete: a different expert set is a different model, so
    // the ids are asserted EXACTLY against an fp64 host reference. The weights
    // follow the kernel rule — no worse than the fp64 reference, never
    // kernel-vs-kernel equality (the chain rounds a full bf16 softmax; the
    // kernel rounds once).
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE11);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const ROWS: c_int = 5;
    const E: c_int = 256;
    const K: c_int = 8;
    const n: usize = @intCast(ROWS * E);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    // Spread wide enough that a bf16 round-trip leaves the top-K boundary
    // unambiguous; ties there would make "the chain's set" undefined, not wrong.
    for (buf, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * 8.0 + @as(f32, @floatFromInt(i % 7)) * 0.011;
    const lsh = [_]c_int{ ROWS, E };
    const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(l32);
    var lbf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lbf);
    try mlx.check(mlx.mlx_astype(&lbf, l32, .bfloat16, s));

    // The kernel and the reference must see the SAME bf16 values.
    const host = try allocator.alloc(f32, n);
    defer allocator.free(host);
    try testReadF32(lbf, host, s);

    const fused = (try moeRouterTopK(s, lbf, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0, null)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);
    const chain = try moeRoutingChain(lbf, K, s);
    defer _ = mlx.mlx_array_free(chain.inds);
    defer _ = mlx.mlx_array_free(chain.norm_scores);

    const kn: usize = @intCast(ROWS * K);
    const f_ids = try allocator.alloc(f32, kn);
    defer allocator.free(f_ids);
    const c_ids = try allocator.alloc(f32, kn);
    defer allocator.free(c_ids);
    const f_w = try allocator.alloc(f32, kn);
    defer allocator.free(f_w);
    const c_w = try allocator.alloc(f32, kn);
    defer allocator.free(c_w);
    try testReadF32(fused.inds, f_ids, s);
    try testReadF32(chain.inds, c_ids, s);
    try testReadF32(fused.norm_scores, f_w, s);
    try testReadF32(chain.norm_scores, c_w, s);

    const keys = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(keys);
    var ref_ids: [8]usize = undefined;
    var fused_err: f64 = 0;
    var chain_err: f64 = 0;
    for (0..@intCast(ROWS)) |r| {
        const row = host[r * @as(usize, @intCast(E)) ..][0..@intCast(E)];
        for (row, 0..) |v, i| keys[i] = @floatCast(v);
        routerRefRow(keys, @intCast(K), &ref_ids);

        // Exact ids, in the kernel's descending-key order.
        for (0..@intCast(K)) |j| {
            try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(f_ids[r * @as(usize, @intCast(K)) + j])));
        }
        // The chain's order is argpartition's, so compare it as a SET.
        var chain_set: [8]usize = undefined;
        for (0..@intCast(K)) |j| chain_set[j] = @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + j]);
        std.mem.sort(usize, &chain_set, {}, std.sort.asc(usize));
        var ref_sorted = ref_ids;
        std.mem.sort(usize, &ref_sorted, {}, std.sort.asc(usize));
        try testing.expectEqualSlices(usize, &ref_sorted, &chain_set);

        // fp64 reference weights: softmax over the selected logits.
        var denom: f64 = 0;
        var num: [8]f64 = undefined;
        for (0..@intCast(K)) |j| {
            num[j] = @exp(keys[ref_ids[j]] - keys[ref_ids[0]]);
            denom += num[j];
        }
        for (0..@intCast(K)) |j| {
            const want = num[j] / denom;
            const got = f_w[r * @as(usize, @intCast(K)) + j];
            fused_err = @max(fused_err, @abs(want - @as(f64, got)));
            // Chain rows are in argpartition order — find this id's slot.
            var slot: usize = 0;
            while (slot < @as(usize, @intCast(K))) : (slot += 1) {
                if (@as(usize, @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + slot])) == ref_ids[j]) break;
            }
            const chain_got = c_w[r * @as(usize, @intCast(K)) + slot];
            chain_err = @max(chain_err, @abs(want - @as(f64, chain_got)));
            // BIT equality with the chain: the kernel replicates the chain's
            // arithmetic, and anything else is a different model (a live A/B on
            // gemma4-26B-A4B diverged at token ~80 when it did not).
            try testing.expectEqual(chain_got, got);
        }
    }
    try testing.expect(fused_err <= chain_err);
}

test "fused softmax router folds the shared-expert gate's logit: routing unchanged, the logit within a bf16 ulp of the f64 dot" {
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6A7E10);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const rows: c_int = 3;
    const E: c_int = 512;
    const K: c_int = 10;
    const GK: c_int = 2560;
    const randBf = struct {
        fn f(a: std.mem.Allocator, r: std.Random, st: mlx.mlx_stream, shape: []const c_int, scale: f32) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try a.alloc(f32, n);
            defer a.free(buf);
            for (buf) |*v| v.* = (r.float(f32) - 0.5) * scale;
            const a32 = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(a32);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, a32, .bfloat16, st));
            return out;
        }
    }.f;
    const logits = try randBf(allocator, rnd, s, &.{ rows, E }, 8.0);
    defer _ = mlx.mlx_array_free(logits);
    const x = try randBf(allocator, rnd, s, &.{ rows, GK }, 4.0);
    defer _ = mlx.mlx_array_free(x);
    const w = try randBf(allocator, rnd, s, &.{GK}, 0.4);
    defer _ = mlx.mlx_array_free(w);

    const plain = (try moeRouterTopK(s, logits, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0, null)) orelse return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(plain.inds);
    defer _ = mlx.mlx_array_free(plain.norm_scores);
    const gated = (try moeRouterTopK(s, logits, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x, .w = w })) orelse return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(gated.inds);
    defer _ = mlx.mlx_array_free(gated.norm_scores);
    defer _ = mlx.mlx_array_free(gated.gate_logit);
    try testing.expect(gated.gate_logit.ctx != null);

    const kn: usize = @intCast(rows * K);
    inline for (.{ "inds", "norm_scores" }) |name| {
        const a = try allocator.alloc(f32, kn);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, kn);
        defer allocator.free(b);
        try testReadF32(@field(plain, name), a, s);
        try testReadF32(@field(gated, name), b, s);
        try testing.expectEqualSlices(f32, a, b);
    }

    const xn: usize = @intCast(rows * GK);
    const xh = try allocator.alloc(f32, xn);
    defer allocator.free(xh);
    const wh = try allocator.alloc(f32, @intCast(GK));
    defer allocator.free(wh);
    const got = try allocator.alloc(f32, @intCast(rows));
    defer allocator.free(got);
    try testReadF32(x, xh, s);
    try testReadF32(w, wh, s);
    try testReadF32(gated.gate_logit, got, s);
    for (0..@intCast(rows)) |r| {
        var want: f64 = 0;
        for (wh, 0..) |wv, k| want += @as(f64, wv) * xh[r * @as(usize, @intCast(GK)) + k];
        try testing.expect(@abs(@as(f64, got[r]) - want) <= 0.004 * @abs(want) + 1e-3);
    }
}

test "fused softmax router: a router with no spare simdgroup routes without the gate logit" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x6A7E11);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    // 16 experts is one simdgroup: nothing is left to dot the gate vector, and a zero logit must never ship.
    const logits = try testRandUniformBf16(rnd, &.{ 2, 16 }, -4.0, 4.0, s);
    defer _ = mlx.mlx_array_free(logits);
    const x = try testRandUniformBf16(rnd, &.{ 2, 64 }, -2.0, 2.0, s);
    defer _ = mlx.mlx_array_free(x);
    const w = try testRandUniformBf16(rnd, &.{64}, -0.2, 0.2, s);
    defer _ = mlx.mlx_array_free(w);

    const plain = (try moeRouterTopK(s, logits, .{ .ctx = null }, 4, .softmax, true, 1.0, .bfloat16, 0, 0, null)) orelse return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(plain.inds);
    defer _ = mlx.mlx_array_free(plain.norm_scores);
    const gated = (try moeRouterTopK(s, logits, .{ .ctx = null }, 4, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x, .w = w })) orelse return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(gated.inds);
    defer _ = mlx.mlx_array_free(gated.norm_scores);
    try testing.expect(gated.gate_logit.ctx == null);
    inline for (.{ "inds", "norm_scores" }) |name| {
        var a: [8]f32 = undefined;
        var b: [8]f32 = undefined;
        try testReadF32(@field(plain, name), &a, s);
        try testReadF32(@field(gated, name), &b, s);
        try testing.expectEqualSlices(f32, &a, &b);
    }
}

test "fused GROUPED MoE router reproduces groupLimitedRouting at the SHIPPED 512/8/4/8 geometry" {
    // Bit equality with the composed chain is the bar, not "close enough": the
    // first live A/B of the plain fused router computed a cleaner softmax and
    // diverged on gemma4-26B-A4B at token ~80 (the INT4 near-tie class,
    // triggered by an avoidable numerical change). The geometry is the SHIPPED
    // one on purpose — a group limit tuned at a toy 4x4 cannot see a mask that
    // is right for 4 groups and wrong for 8.
    //
    // ROWS is a SAMPLE SIZE, and a bit-equality bar is worth exactly its
    // sample: at 4 rows this compared 32 weights, which dodged a live
    // 6.1%-of-weights divergence on the dev GPU and hit it on CI's (the
    // JIT-vs-metallib `exp` above). 256 rows is 2048 weights.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x11_6C_0DE);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const ROWS: c_int = 256;
    const E: c_int = 512;
    const K: c_int = 8;
    const NG: c_int = 8;
    const TOPKG: c_int = 4;
    const GSZ: usize = @intCast(@divExact(E, NG));
    const ROUTE_SCALE: f32 = 2.5;
    const n: usize = @intCast(ROWS * E);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * 6.0 + @as(f32, @floatFromInt(i % 7)) * 0.011;
    const lsh = [_]c_int{ ROWS, E };
    const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(l32);

    const bbuf = try allocator.alloc(f32, @intCast(E));
    defer allocator.free(bbuf);
    for (bbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
    const bsh = [_]c_int{E};
    const bias = mlx.mlx_array_new_data(bbuf.ptr, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const fused = (try moeRouterTopK(s, l32, bias, K, .sigmoid_bias_grouped, true, ROUTE_SCALE, .float32, NG, TOPKG, null)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);

    moe_router_fused_override = false;
    const chain = try groupLimitedRouting(l32, bias, K, NG, TOPKG, true, ROUTE_SCALE, s);
    moe_router_fused_override = true;
    defer _ = mlx.mlx_array_free(chain.inds);
    defer _ = mlx.mlx_array_free(chain.norm_scores);

    const kn: usize = @intCast(ROWS * K);
    const f_ids = try allocator.alloc(f32, kn);
    defer allocator.free(f_ids);
    const c_ids = try allocator.alloc(f32, kn);
    defer allocator.free(c_ids);
    const f_w = try allocator.alloc(f32, kn);
    defer allocator.free(f_w);
    const c_w = try allocator.alloc(f32, kn);
    defer allocator.free(c_w);
    try testReadF32(fused.inds, f_ids, s);
    try testReadF32(chain.inds, c_ids, s);
    try testReadF32(fused.norm_scores, f_w, s);
    try testReadF32(chain.norm_scores, c_w, s);

    // Independent f64 reference for the WHOLE gate, group limit included.
    const keys = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(keys);
    const sig = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(sig);
    var ref_ids: [8]usize = undefined;
    var dropped_seen = false;
    for (0..@intCast(ROWS)) |r| {
        const row = buf[r * @as(usize, @intCast(E)) ..][0..@intCast(E)];
        for (row, 0..) |v, i| {
            sig[i] = 1.0 / (1.0 + @exp(-@as(f64, v)));
            keys[i] = sig[i] + @as(f64, bbuf[i]);
        }
        // Group score = sum of the top TWO biased experts in the group.
        var gscore: [8]f64 = undefined;
        for (0..@intCast(NG)) |g| {
            var m1: f64 = -std.math.inf(f64);
            var m2: f64 = -std.math.inf(f64);
            for (keys[g * GSZ ..][0..GSZ]) |v| {
                if (v > m1) {
                    m2 = m1;
                    m1 = v;
                } else if (v > m2) m2 = v;
            }
            gscore[g] = m1 + m2;
        }
        var kept: [8]bool = @splat(false);
        var gsel: [8]usize = undefined;
        routerRefRow(gscore[0..@intCast(NG)], @intCast(TOPKG), &gsel);
        for (gsel[0..@intCast(TOPKG)]) |g| kept[g] = true;
        for (0..@intCast(E)) |i| {
            if (!kept[i / GSZ]) {
                keys[i] = -1.0e4;
                dropped_seen = true;
            }
        }
        routerRefRow(keys, @intCast(K), &ref_ids);

        // Every selected expert must live in a surviving group — the failure
        // mode of a silently skipped group step is that low groups stay
        // reachable, and it is invisible in the weights.
        for (0..@intCast(K)) |j| {
            try testing.expect(kept[ref_ids[j] / GSZ]);
            try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(f_ids[r * @as(usize, @intCast(K)) + j])));
        }
        // The chain does not order its top-k, so compare index SETS with it.
        var chain_set: [8]usize = undefined;
        for (0..@intCast(K)) |j| chain_set[j] = @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + j]);
        std.mem.sort(usize, &chain_set, {}, std.sort.asc(usize));
        var ref_sorted = ref_ids;
        std.mem.sort(usize, &ref_sorted, {}, std.sort.asc(usize));
        try testing.expectEqualSlices(usize, &ref_sorted, &chain_set);

        for (0..@intCast(K)) |j| {
            var slot: usize = 0;
            while (slot < @as(usize, @intCast(K))) : (slot += 1) {
                if (@as(usize, @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + slot])) == ref_ids[j]) break;
            }
            try testing.expectEqual(c_w[r * @as(usize, @intCast(K)) + slot], f_w[r * @as(usize, @intCast(K)) + j]);
        }
    }
    // The test data has to actually exercise the limit, or it proves nothing.
    try testing.expect(dropped_seen);
}

test "fused MoE router reproduces the hy3 sigmoid+bias routing chain" {
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED2026);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const ROWS: c_int = 4;
    const E: c_int = 256;
    const K: c_int = 8;
    const ROUTE_SCALE: f32 = 2.5;
    const n: usize = @intCast(ROWS * E);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * 6.0 + @as(f32, @floatFromInt(i % 5)) * 0.013;
    const lsh = [_]c_int{ ROWS, E };
    const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(l32);
    var lbf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lbf);
    try mlx.check(mlx.mlx_astype(&lbf, l32, .bfloat16, s));

    const bbuf = try allocator.alloc(f32, @intCast(E));
    defer allocator.free(bbuf);
    for (bbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
    const bsh = [_]c_int{E};
    const bias = mlx.mlx_array_new_data(bbuf.ptr, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const host = try allocator.alloc(f32, n);
    defer allocator.free(host);
    try testReadF32(lbf, host, s);

    const fused = (try moeRouterTopK(s, lbf, bias, K, .sigmoid_bias, true, ROUTE_SCALE, .bfloat16, 0, 0, null)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);
    const chain = try hy3RoutingChain(lbf, bias, K, true, ROUTE_SCALE, s);
    defer _ = mlx.mlx_array_free(chain.inds);
    defer _ = mlx.mlx_array_free(chain.norm_scores);

    const kn: usize = @intCast(ROWS * K);
    const f_ids = try allocator.alloc(f32, kn);
    defer allocator.free(f_ids);
    const c_ids = try allocator.alloc(f32, kn);
    defer allocator.free(c_ids);
    const f_w = try allocator.alloc(f32, kn);
    defer allocator.free(f_w);
    const c_w = try allocator.alloc(f32, kn);
    defer allocator.free(c_w);
    try testReadF32(fused.inds, f_ids, s);
    try testReadF32(chain.inds, c_ids, s);
    try testReadF32(fused.norm_scores, f_w, s);
    try testReadF32(chain.norm_scores, c_w, s);

    const keys = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(keys);
    const sig = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(sig);
    var ref_ids: [8]usize = undefined;
    var fused_err: f64 = 0;
    var chain_err: f64 = 0;
    for (0..@intCast(ROWS)) |r| {
        const row = host[r * @as(usize, @intCast(E)) ..][0..@intCast(E)];
        for (row, 0..) |v, i| {
            sig[i] = 1.0 / (1.0 + @exp(-@as(f64, v)));
            keys[i] = sig[i] + @as(f64, bbuf[i]);
        }
        routerRefRow(keys, @intCast(K), &ref_ids);
        for (0..@intCast(K)) |j| {
            try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(f_ids[r * @as(usize, @intCast(K)) + j])));
        }
        var chain_set: [8]usize = undefined;
        for (0..@intCast(K)) |j| chain_set[j] = @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + j]);
        std.mem.sort(usize, &chain_set, {}, std.sort.asc(usize));
        var ref_sorted = ref_ids;
        std.mem.sort(usize, &ref_sorted, {}, std.sort.asc(usize));
        try testing.expectEqualSlices(usize, &ref_sorted, &chain_set);

        var denom: f64 = 1e-20;
        for (0..@intCast(K)) |j| denom += sig[ref_ids[j]];
        for (0..@intCast(K)) |j| {
            const want = sig[ref_ids[j]] / denom * @as(f64, ROUTE_SCALE);
            const got = f_w[r * @as(usize, @intCast(K)) + j];
            fused_err = @max(fused_err, @abs(want - @as(f64, got)));
            var slot: usize = 0;
            while (slot < @as(usize, @intCast(K))) : (slot += 1) {
                if (@as(usize, @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + slot])) == ref_ids[j]) break;
            }
            const chain_got = c_w[r * @as(usize, @intCast(K)) + slot];
            chain_err = @max(chain_err, @abs(want - @as(f64, chain_got)));
            try testing.expectEqual(chain_got, got);
        }
    }
    // Both round to bf16 at the end, so the bar is "no worse", not "better".
    try testing.expect(fused_err <= chain_err * 1.0001 + 1e-6);
}

test "fused MoE router reproduces the kolibri1 logit+bias routing chain" {
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4B011B21);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const ROWS: c_int = 5;
    const E: c_int = 384;
    const K: c_int = 6;
    const en: usize = @intCast(E);
    const kn: usize = @intCast(ROWS * K);

    const buf = try allocator.alloc(f32, @intCast(ROWS * E));
    defer allocator.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * 8.0;
    const lsh = [_]c_int{ ROWS, E };
    const logits = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const bbuf = try allocator.alloc(f32, en);
    defer allocator.free(bbuf);
    for (bbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const bsh = [_]c_int{E};
    const bias = mlx.mlx_array_new_data(bbuf.ptr, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const fused = (try moeRouterTopK(s, logits, bias, K, .logit_bias, false, 1.0, .float32, 0, 0, null)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);
    const chain = try kolibriRoutingChain(logits, bias, K, s);
    defer _ = mlx.mlx_array_free(chain.inds);
    defer _ = mlx.mlx_array_free(chain.norm_scores);

    const f_ids = try allocator.alloc(f32, kn);
    defer allocator.free(f_ids);
    const c_ids = try allocator.alloc(f32, kn);
    defer allocator.free(c_ids);
    const f_w = try allocator.alloc(f32, kn);
    defer allocator.free(f_w);
    const c_w = try allocator.alloc(f32, kn);
    defer allocator.free(c_w);
    try testReadF32(fused.inds, f_ids, s);
    try testReadF32(chain.inds, c_ids, s);
    try testReadF32(fused.norm_scores, f_w, s);
    try testReadF32(chain.norm_scores, c_w, s);

    // Same expert set per row, and each expert carries the chain's exact weight.
    const k: usize = @intCast(K);
    for (0..@intCast(ROWS)) |r| {
        for (0..k) |j| {
            const id = f_ids[r * k + j];
            var slot: usize = 0;
            while (slot < k and c_ids[r * k + slot] != id) : (slot += 1) {}
            try testing.expect(slot < k);
            try testing.expectEqual(c_w[r * k + slot], f_w[r * k + j]);
        }
    }
}

test "moeRouterThreadGroup keeps a full simdgroup and caps at 256" {
    try testing.expectEqual(@as(c_int, 32), moeRouterThreadGroup(8));
    try testing.expectEqual(@as(c_int, 32), moeRouterThreadGroup(32));
    try testing.expectEqual(@as(c_int, 64), moeRouterThreadGroup(33));
    try testing.expectEqual(@as(c_int, 128), moeRouterThreadGroup(128));
    try testing.expectEqual(@as(c_int, 256), moeRouterThreadGroup(256));
    try testing.expectEqual(@as(c_int, 256), moeRouterThreadGroup(2048));
}
