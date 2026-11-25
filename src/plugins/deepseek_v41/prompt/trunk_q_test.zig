//! MLX_SERVE_DSV41_PF_INT8=3's quality test, before its kernels: on the real pack's trunk weights and the inputs a
//! served p01 pass gave them (`DSV41_TRUNK_DUMP`), each dense prompt projection's output against float64, for the
//! served mxfp8 path and the two int8 schemes it could take.
//!   E: the served path, MLX's mxfp8 quantized matmul.
//!   R: w8a8 (sdk/blocks/dit): a 256-point Hadamard rotation of the input and of the weight (at load), one int8 scale
//!      a row on each side, the int32 sum scaled once; its own kernels.
//!   B: no rotation; the input one int8 scale a `qb`-column block of a row, the weight one a row (or a 128-column
//!      block), the block sums folded in f32; emulated (the quantized values' product in float64, rounded to bf16).
//! Truth: the dumped bf16 input times the mxfp8 weight's exact values, in float64. The plant (R on the input rows a
//! row late) must fail.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const dsv41 = @import("../deepseek_v41.zig");
const w8a8 = @import("../../../sdk/blocks/dit/w8a8.zig");
const pfint8 = @import("pf_int8.zig");
const world = @import("../../../sdk/testing/world.zig");

const A = mlx.mlx_array;
const Scope = dsv41.Scope;

const Proj = struct { name: []const u8, dump: []const u8, keys: []const []const u8 };
const projs = [_]Proj{
    .{ .name = "wq_b", .dump = "wq_b_in", .keys = &.{"attn.wq_b"} },
    .{ .name = "wo_b", .dump = "wo_b_in", .keys = &.{"attn.wo_b"} },
    .{ .name = "wq_a+wkv", .dump = "qkv_in", .keys = &.{ "attn.wq_a", "attn.wkv" } },
    .{ .name = "shared gate+up", .dump = "shared_in", .keys = &.{ "ffn.shared_experts.w1", "ffn.shared_experts.w3" } },
    .{ .name = "shared down", .dump = "shared_down_in", .keys = &.{"ffn.shared_experts.w2"} },
};

const Shards = struct {
    a: std.mem.Allocator,
    dir: []const u8,
    index: std.json.Parsed(std.json.Value),
    maps: std.StringHashMapUnmanaged(mlx.mlx_map_string_to_array) = .{},

    fn get(self: *Shards, key: []const u8) !A {
        const wm = self.index.value.object.get("weight_map").?.object;
        const file = (wm.get(key) orelse return error.TrunkQNoTensor).string;
        const gop = try self.maps.getOrPut(self.a, file);
        if (!gop.found_existing) {
            var pbuf: [1024]u8 = undefined;
            const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/{s}", .{ self.dir, file }, 0);
            var m = mlx.mlx_map_string_to_array_new();
            var meta = mlx.mlx_map_string_to_string_new();
            defer _ = mlx.mlx_map_string_to_string_free(meta);
            try mlx.check(mlx.mlx_load_safetensors(&m, &meta, path.ptr, mlx.mlx_default_cpu_stream_new()));
            gop.value_ptr.* = m;
        }
        var buf: [256]u8 = undefined;
        const k = try std.fmt.bufPrintSentinel(&buf, "{s}", .{key}, 0);
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_map_string_to_array_get(&x, gop.value_ptr.*, k.ptr));
        return x;
    }

    fn deinit(self: *Shards) void {
        var it = self.maps.valueIterator();
        while (it.next()) |m| _ = mlx.mlx_map_string_to_array_free(m.*);
        self.maps.deinit(self.a);
        self.index.deinit();
    }
};

fn readDump(a: std.mem.Allocator, path: []const u8) !A {
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 31));
    defer a.free(bytes);
    const n = std.mem.readInt(u32, bytes[0..4], .little);
    const k = std.mem.readInt(u32, bytes[4..8], .little);
    if (bytes.len != 8 + 2 * @as(usize, n) * k) return error.TrunkQBadDump;
    return mlx.mlx_array_new_data(bytes[8..].ptr, &[_]c_int{ @intCast(n), @intCast(k) }, 2, .bfloat16);
}

/// `x` `[n, K]` f32 quantized to int8 with one scale a `qb`-column block of a row (amax / 127, round to nearest even),
/// and back: the int8 values times their scale, f32.
fn quantDequant(sc: *Scope, x: A, qb: usize) !A {
    const sh = mlx.getShape(x);
    const n = sh[0];
    const k = sh[1];
    const xb = try sc.reshape(x, &.{ n, @divExact(k, @as(c_int, @intCast(qb))), @intCast(qb) });
    var ab = mlx.mlx_array_new();
    const ax = try sc.res(mlx.mlx_abs(&ab, xb, sc.s), &ab);
    const am = try sc.max(ax, 2, true);
    const s0 = try sc.div(am, try sc.f(127));
    const s = try sc.where(try sc.op2(mlx.mlx_greater, s0, try sc.f(0)), s0, try sc.f(1));
    var r = mlx.mlx_array_new();
    const q = try sc.res(mlx.mlx_round(&r, try sc.div(xb, s), 0, sc.s), &r);
    return sc.reshape(try sc.mul(try sc.clip(q, -127, 127), s), &.{ n, k });
}

const Err = struct { l2: f32, max: f32 };

/// `y` against float64 truth `t` (both `[n, cols]` f64, on the CPU): l2 relative to the truth's, the largest
/// difference relative to the truth's largest value.
fn errOf(cs: *Scope, y: A, t: A) !Err {
    const d = try cs.sub(y, t);
    const l2 = @sqrt(try item(cs, try cs.sum(try cs.reshape(try cs.mul(d, d), &.{-1}), 0, false)) / try item(cs, try cs.sum(try cs.reshape(try cs.mul(t, t), &.{-1}), 0, false)));
    var ad = mlx.mlx_array_new();
    const da = try cs.res(mlx.mlx_abs(&ad, d, cs.s), &ad);
    var at = mlx.mlx_array_new();
    const ta = try cs.res(mlx.mlx_abs(&at, t, cs.s), &at);
    const mx = try item(cs, try cs.max(try cs.reshape(da, &.{-1}), 0, false)) / try item(cs, try cs.max(try cs.reshape(ta, &.{-1}), 0, false));
    return .{ .l2 = @floatCast(l2), .max = @floatCast(mx) };
}

fn item(cs: *Scope, x: A) !f64 {
    const v32 = try cs.astype(x, .float32);
    try dsv41.evalAll(&.{v32});
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, v32));
    return v;
}

/// A GPU output `[n, N]` at columns `cols`, as f64 on the CPU.
fn onCpu(sc: *Scope, cs: *Scope, y: A, cols: A) !A {
    const ys = try sc.astype(try sc.take(y, cols, 1), .float32);
    try dsv41.evalAll(&.{ys});
    return cs.astype(ys, .float64);
}

test "dsv41 pf int8=3 quality (not lossless): each trunk projection on the real pack's weights and a served p01 pass's inputs (layers 3, 10, 17), the served mxfp8 path, w8a8's rotated per-row int8 (R) and per-block int8 (B: input blocks 128, 256 and 512, weight per row or per 128 block) against float64; R on rows a row late fails (DSV41_TRUNK_Q = <dump dir>:<pack dir>)" {
    const spec = std.mem.sliceTo(env.get(.DSV41_TRUNK_Q) orelse return error.SkipZigTest, 0);
    const colon = std.mem.indexOfScalar(u8, spec, ':') orelse return error.TrunkQBadSpec;
    const dump_dir = spec[0..colon];
    const pack_dir = spec[colon + 1 ..];
    mlx.installErrorHandler();
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ipath = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{pack_dir});
    defer a.free(ipath);
    const ibytes = try std.Io.Dir.cwd().readFileAlloc(io, ipath, a, .limited(1 << 26));
    defer a.free(ibytes);
    var shards = Shards{ .a = a, .dir = pack_dir, .index = try std.json.parseFromSlice(std.json.Value, a, ibytes, .{}) };
    defer shards.deinit();
    const gs = mlx.gpuStream();
    const cpu = mlx.mlx_default_cpu_stream_new();
    var worst_plant: f32 = std.math.inf(f32);
    for (dsv41.trunk_dump_layers) |li| for (projs) |pj| {
        var sc = Scope.init(a, gs);
        defer sc.deinit();
        var cs = Scope.init(a, cpu);
        defer cs.deinit();
        const dpath = try std.fmt.allocPrint(a, "{s}/L{d}_{s}.bin", .{ dump_dir, li, pj.dump });
        defer a.free(dpath);
        const x = try sc.keep(try readDump(a, dpath));
        // the weight: each key's mxfp8 codes and e8m0 scales, joined along the output rows
        var ws: [2]A = undefined;
        var ss: [2]A = undefined;
        for (pj.keys, 0..) |key, i| {
            var buf: [128]u8 = undefined;
            ws[i] = try sc.keep(try shards.get(try std.fmt.bufPrint(&buf, "layers.{d}.{s}.weight", .{ li, key })));
            ss[i] = try sc.keep(try shards.get(try std.fmt.bufPrint(&buf, "layers.{d}.{s}.scales", .{ li, key })));
        }
        const nk = pj.keys.len;
        const w = if (nk == 1) ws[0] else try sc.concat(ws[0..nk], 0);
        const s = if (nk == 1) ss[0] else try sc.concat(ss[0..nk], 0);
        var dq = mlx.mlx_array_new();
        const wd = try sc.res(mlx.mlx_dequantize(&dq, w, s, .{ .ctx = null }, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, gs), &dq);
        try dsv41.evalAll(&.{ x, w, s, wd });
        const N: usize = @intCast(mlx.getShape(wd)[0]);
        const K: usize = @intCast(mlx.getShape(wd)[1]);
        if (mlx.getShape(x)[1] != @as(c_int, @intCast(K))) return error.TrunkQShape;
        const step = @max(N / 512, 1);
        var ar = mlx.mlx_array_new();
        const cols = try sc.res(mlx.mlx_arange(&ar, 0, @floatFromInt(N), @floatFromInt(step), .int32, gs), &ar);
        // truth
        const x64 = try onCpu(&sc, &cs, x, try sc.arange(0, K));
        const wsub = try sc.take(wd, cols, 0);
        try dsv41.evalAll(&.{wsub});
        const w64 = try cs.astype(wsub, .float64);
        const t = try cs.matmul(x64, try cs.t2(w64));
        // E: the served path
        var qo = mlx.mlx_array_new();
        const ye = try sc.res(mlx.mlx_quantized_matmul(&qo, x, w, s, .{ .ctx = null }, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", gs), &qo);
        const ee = try errOf(&cs, try onCpu(&sc, &cs, ye, cols), t);
        // R: w8a8
        var wt = try w8a8.Weight.fromDense(try sc.astype(wd, .bfloat16), gs);
        defer wt.deinit();
        var act = try w8a8.quantize(x, gs);
        defer act.deinit();
        const yr = try sc.keep(try w8a8.gemm(act, &wt, .bfloat16, false, gs));
        const er = try errOf(&cs, try onCpu(&sc, &cs, yr, cols), t);
        // the plant: R on the rows a row late
        const n: usize = @intCast(mlx.getShape(x)[0]);
        const xl = try sc.concat(&.{ try sc.slice(x, 0, 1, n), try sc.slice(x, 0, 0, 1) }, 0);
        var actl = try w8a8.quantize(xl, gs);
        defer actl.deinit();
        const yp = try sc.keep(try w8a8.gemm(actl, &wt, .bfloat16, false, gs));
        const ep = try errOf(&cs, try onCpu(&sc, &cs, yp, cols), t);
        worst_plant = @min(worst_plant, ep.l2);
        // B: per-block input, per-row or per-128-block weight, emulated in float64, rounded to bf16
        const xf = try sc.astype(x, .float32);
        var eb: [5]Err = undefined;
        const bs = [5][2]usize{ .{ 128, K }, .{ 512, K }, .{ 128, 128 }, .{ 256, 128 }, .{ 512, 128 } };
        for (bs, 0..) |b, bi| {
            if (K % b[0] != 0) {
                eb[bi] = .{ .l2 = std.math.nan(f32), .max = std.math.nan(f32) };
                continue;
            }
            const xq = try sc.astype(try quantDequant(&sc, xf, b[0]), .float32);
            const wq = try sc.astype(try quantDequant(&sc, wsub, b[1]), .float32);
            try dsv41.evalAll(&.{ xq, wq });
            const yb = try cs.matmul(try cs.astype(xq, .float64), try cs.t2(try cs.astype(wq, .float64)));
            eb[bi] = try errOf(&cs, try cs.astype(try cs.astype(yb, .bfloat16), .float64), t);
        }
        std.debug.print("[trunk-q] layer {d} {s} ({d} rows x K {d} x {d} of {d} columns): float64 l2 / max: E (served mxfp8) {d:.5} / {d:.5}; R (rotated, per row) {d:.5} / {d:.5}; B in128 w-row {d:.5} / {d:.5}; B in512 w-row {d:.5} / {d:.5}; B in128 w128 {d:.5} / {d:.5}; B in256 w128 {d:.5} / {d:.5}; B in512 w128 {d:.5} / {d:.5}; plant (R a row late) {d:.4}\n", .{ li, pj.name, n, K, (N + step - 1) / step, N, ee.l2, ee.max, er.l2, er.max, eb[0].l2, eb[0].max, eb[1].l2, eb[1].max, eb[2].l2, eb[2].max, eb[3].l2, eb[3].max, eb[4].l2, eb[4].max, ep.l2 });
        try std.testing.expect(ee.l2 < 0.01);
    };
    try std.testing.expect(worst_plant > 0.3);
}

/// An mxfp8 linear `[n, k]` of random normal values (MLX's quantizer: e4m3 codes, an e8m0 scale a 32-group).
fn mxfp8Lin(sc: *Scope, n: c_int, k: c_int, seed: u64) !dsv41.Lin {
    const w = try sc.keep(try w8a8.randn(&.{ n, k }, seed, .bfloat16, sc.s));
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{ .ctx = null }, sc.s));
    var q = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&q, qv, 0));
    var s = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&s, qv, 1));
    try dsv41.evalAll(&.{ try sc.keep(q), try sc.keep(s) });
    return .{ .w = q, .s = s, .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
}

test "dsv41 pf int8=3 (not lossless): wq_b's and wo_b's shapes (level 4: the shared expert's and wq_a+wkv's too) on w8a8's int8 GEMM against float64 (the mxfp8 weight's exact values), within 3% l2 and beside the exact path's; the input a row late fails; narrow calls, another dtype and a non-mxfp8 linear decline" {
    mlx.installErrorHandler();
    if (mlx.noGpuBackend() or !@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const gs = mlx.gpuStream();
    const cpu = mlx.mlx_default_cpu_stream_new();
    // (name, K, N): wq_b 1280 -> 64 heads x 512 (a quarter of them here), wo_b 8 x 1024 -> 5120 (a fifth)
    // level 4's: wq_a+wkv 5120 -> 1792, the shared gate+up 5120 -> 4608 (a quarter here) and down 2304 -> 5120 (a quarter)
    const shapes = [_]struct { name: []const u8, k: c_int, n: c_int }{ .{ .name = "wq_b", .k = 1280, .n = 8192 }, .{ .name = "wo_b", .k = 8192, .n = 1024 }, .{ .name = "wq_a+wkv", .k = 5120, .n = 1792 }, .{ .name = "shared gate+up", .k = 5120, .n = 1152 }, .{ .name = "shared down", .k = 2304, .n = 1280 } };
    for (shapes, 0..) |shp, si| for ([_]c_int{ 1024, 1500 }) |rows| {
        var sc = Scope.init(a, gs);
        defer sc.deinit();
        var cs = Scope.init(a, cpu);
        defer cs.deinit();
        const l = try mxfp8Lin(&sc, shp.n, shp.k, 0x3100 + si);
        const x = try sc.keep(try w8a8.randn(&.{ rows, shp.k }, 0x3200 + si, .bfloat16, gs));
        const N: usize = @intCast(shp.n);
        var ar = mlx.mlx_array_new();
        const cols = try sc.res(mlx.mlx_arange(&ar, 0, @floatFromInt(N), 16, .int32, gs), &ar);
        const wd = try l.dense(&sc, .float32);
        const wsub = try sc.take(wd, cols, 0);
        try dsv41.evalAll(&.{wsub});
        const t = try cs.matmul(try onCpu(&sc, &cs, x, try sc.arange(0, @intCast(shp.k))), try cs.t2(try cs.astype(wsub, .float64)));
        var wt = (try pfint8.weightOf(&sc, &l)) orelse return error.TestDeclined;
        defer wt.deinit();
        pfint8.trunk_plant = false;
        const y = (try pfint8.apply(&sc, 4, &wt, x, .bfloat16, shp.name)) orelse return error.TestDeclined;
        const e = try errOf(&cs, try onCpu(&sc, &cs, y, cols), t);
        var qo = mlx.mlx_array_new();
        const ye = try sc.res(mlx.mlx_quantized_matmul(&qo, x, l.w, l.s, .{ .ctx = null }, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", gs), &qo);
        const ee = try errOf(&cs, try onCpu(&sc, &cs, ye, cols), t);
        pfint8.trunk_plant = true;
        const yp = (try pfint8.apply(&sc, 4, &wt, x, .bfloat16, shp.name)) orelse return error.TestDeclined;
        pfint8.trunk_plant = false;
        const ep = try errOf(&cs, try onCpu(&sc, &cs, yp, cols), t);
        std.debug.print("[trunk-i8] {s} rows {d} (K {d}, {d} of {d} columns): float64 l2 / max, int8 {d:.5} / {d:.5}, exact mxfp8 path {d:.5} / {d:.5}; planted (a row late) {d:.4}\n", .{ shp.name, rows, shp.k, (N + 15) / 16, N, e.l2, e.max, ee.l2, ee.max, ep.l2 });
        try std.testing.expect(e.l2 < 0.03 and e.l2 > ee.l2);
        try std.testing.expect(ep.l2 > 0.3);
        // declines: a narrow call, an f32 input, a linear with no mxfp8 scales
        try std.testing.expect((try pfint8.apply(&sc, 4, &wt, try sc.slice(x, 0, 0, pfint8.TRUNK_ROWS - 1), .bfloat16, shp.name)) == null);
        try std.testing.expect((try pfint8.apply(&sc, 4, &wt, try sc.astype(x, .float32), .bfloat16, shp.name)) == null);
        const plain = dsv41.Lin{ .w = try l.dense(&sc, .bfloat16) };
        try std.testing.expect((try pfint8.weightOf(&sc, &plain)) == null);
    };
}

test "dsv41 pf int8 bill: levels 3 and 4 add their int8 inputs and bf16 outputs a row to a wide pass's working set (DeepSeek-V4.1's shapes: 85,256 and 120,852 bytes a row) where the plan serves (NAX), none at level 0, without NAX or for a call no wider than a span" {
    const C = struct {
        hidden_size: u32 = 5120,
        dsv4_q_lora_rank: u32 = 1280,
        num_attention_heads: u32 = 64,
        head_dim: u32 = 512,
        dsv4_o_groups: u32 = 8,
        dsv4_o_lora_rank: u32 = 1024,
        moe_intermediate_size: u32 = 2304,
    };
    const c = C{};
    const rows: u64 = 38656;
    {
        const pin = env.override(.MLX_SERVE_DSV41_PF_INT8, "0");
        defer pin.restore();
        try std.testing.expectEqual(@as(u64, 0), pfint8.transientBytes(&c, rows, 2048));
    }
    {
        const pin = env.override(.MLX_SERVE_DSV41_PF_INT8, "3");
        defer pin.restore();
        // Without NAX the plan is inert and reserves nothing.
        const nax = world.nax();
        try std.testing.expectEqual(if (nax) rows * 85_256 else 0, pfint8.transientBytes(&c, rows, 2048));
        try std.testing.expectEqual(@as(u64, 0), pfint8.transientBytes(&c, 2048, 2048));
    }
    {
        const pin = env.override(.MLX_SERVE_DSV41_PF_INT8, "4");
        defer pin.restore();
        try std.testing.expectEqual(if (world.nax()) rows * 120_852 else 0, pfint8.transientBytes(&c, rows, 2048));
    }
}

test "dsv41 pf int8 bill: precision lossy bills level 4 with no model and no PF_INT8 set (where the plan serves, NAX); the prefill stack off and PF_INT8=3 win over it" {
    const lev = @import("../levers.zig");
    const C = struct {
        hidden_size: u32 = 5120,
        dsv4_q_lora_rank: u32 = 1280,
        num_attention_heads: u32 = 64,
        head_dim: u32 = 512,
        dsv4_o_groups: u32 = 8,
        dsv4_o_lora_rank: u32 = 1024,
        moe_intermediate_size: u32 = 2304,
    };
    const c = C{};
    const rows: u64 = 38656;
    const unset = env.override(.MLX_SERVE_DSV41_PF_INT8, null);
    defer unset.restore();
    try std.testing.expectEqual(@as(u64, 0), pfint8.transientBytes(&c, rows, 2048));
    const lossy = env.override(.MLX_SERVE_DSV41_PRECISION, "lossy");
    defer {
        lossy.restore();
        lev.refresh();
    }
    lev.refresh();
    // Without NAX the plan is inert and reserves nothing.
    const nax = @import("../../../mlx/device.zig").naxAvailable();
    try std.testing.expectEqual(if (nax) rows * 120_852 else 0, pfint8.transientBytes(&c, rows, 2048));
    {
        const pin = env.override(.MLX_SERVE_DSV41_PF_INT8, "3");
        defer pin.restore();
        try std.testing.expectEqual(if (nax) rows * 85_256 else 0, pfint8.transientBytes(&c, rows, 2048));
    }
    {
        const off = env.override(.MLX_SERVE_DSV41_STACK_PREFILL, "0");
        defer {
            off.restore();
            lev.refresh();
        }
        lev.refresh();
        try std.testing.expectEqual(@as(u64, 0), pfint8.transientBytes(&c, rows, 2048));
    }
}
