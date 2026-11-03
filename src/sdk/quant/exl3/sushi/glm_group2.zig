//! Exact shared-expert reuse of GLM's cooperative F16 dot for DFlash verify rows.
//! Routing IDs must be valid for the supplied expert bank, as in the baseline API.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const base = @import("expert_exl3_kernels.zig");
const support = base.Group2Support;
const Arr = mlx.mlx_array;
pub const Reduction = enum { parallel, serial };
pub const Layout = enum { natural, lane };

fn replace(comptime source: []const u8, comptime old: []const u8, comptime value: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    const at = comptime std.mem.indexOf(u8, source, old) orelse return source ++ "";
    return source[0..at] ++ value ++ replace(source[at + old.len ..], old, value);
}
const MEMBERS =
    @embedFile("kernels/glm_group2_members.metal");
// Three verification rows have 24 routes, so their matching slots fit one ballot.
const THREE_MEMBERS =
    @embedFile("kernels/glm_group2_three_members.metal");
const EPILOGUE =
    @embedFile("kernels/glm_group2_epilogue.metal");
// Carry two output tiles through the same k-loop to reuse each member's input loads.
fn tileKLoops(comptime source: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    const loop = "for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {";
    const at = std.mem.indexOf(u8, source, loop) orelse return source ++ "";
    const begin = at + loop.len;
    var depth: usize = 1;
    var end = begin;
    while (depth != 0) : (end += 1) {
        if (source[end] == '{') depth += 1;
        if (source[end] == '}') depth -= 1;
    }
    return source[0..begin] ++ "\nfor(uint tile=0;tile<uint(TILES);++tile){\nconst uint ot=ot0+tile;\n" ++
        replace(source[begin .. end - 1], "acc[", "acc[tile][") ++ "\n}\n}" ++ tileKLoops(source[end..]);
}
fn outputTiles(comptime source: []const u8, comptime members: usize) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    const marker = if (members == 1) "for (uint si = 0u; si < 8u; si++)" else "if constexpr(SERIAL_REDUCTION)";
    const cut = std.mem.indexOf(u8, source, marker).?;
    var body = replace(source[0..cut], "uint ot = uint(threadgroup_position_in_grid.x);", "uint ot0=uint(threadgroup_position_in_grid.x)*uint(TILES);");
    body = if (members == 1)
        replace(body, "float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};", "float acc[TILES][8] = {};")
    else
        replace(body, std.fmt.comptimePrint("float acc[{d}][8] = {{}};", .{members}), std.fmt.comptimePrint("float acc[TILES][{d}][8] = {{}};", .{members}));
    return tileKLoops(body) ++ "\nfor(uint tile=0;tile<uint(TILES);++tile){\nconst uint ot=ot0+tile;\n" ++
        replace(source[cut..], "acc[", "acc[tile][") ++ "\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}\n";
}
fn groupedSource(comptime triples: bool) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    const original = replace(support.cooperative_source, "threadgroup float partial[4 * 256];", "");
    const cut = std.mem.indexOf(u8, original, "for (uint si = 0u; si < 8u; si++)").?;
    var paired: []const u8 = original[0..cut];
    paired = replace(paired, "float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};", "float acc[2][8] = {};");
    paired = replace(paired, "const size_t xb = (size_t)slot * (size_t)(IDIM);", "const size_t xb = size_t(first)*uint(IDIM), xb1=size_t(partner)*uint(IDIM);");
    for (0..4) |r| {
        const old = std.fmt.comptimePrint("const float in{d} = float(x[xb + tk * TILE + row{d}]);", .{ r, r });
        const new = std.fmt.comptimePrint("const float2 in{d} = float2(float(x[xb + tk * TILE + row{d}]),float(x[xb1 + tk * TILE + row{d}]));", .{ r, r, r });
        paired = replace(paired, old, new);
    }
    paired = replace(paired, "const float ins[8]", "const float2 ins[8]");
    paired = replace(paired, "acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);", "acc[0][p*2u]=fma(ins[p*2u].x,w.x,acc[0][p*2u]); acc[1][p*2u]=fma(ins[p*2u].y,w.x,acc[1][p*2u]);");
    paired = replace(paired, "acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);", "acc[0][p*2u+1u]=fma(ins[p*2u+1u].x,w.y,acc[0][p*2u+1u]); acc[1][p*2u+1u]=fma(ins[p*2u+1u].y,w.y,acc[1][p*2u+1u]);");
    if (!triples) return MEMBERS ++ "\nif(partner==first) {\n" ++ original ++ "\n} else {\n" ++ paired ++ EPILOGUE ++ "\n}\n";
    var triple = replace(paired, "float acc[2][8]", "float acc[3][8]");
    triple = replace(triple, "xb1=size_t(partner)*uint(IDIM);", "xb1=size_t(partner)*uint(IDIM), xb2=size_t(third)*uint(IDIM);");
    for (0..4) |r| {
        const old = std.fmt.comptimePrint("const float2 in{d} = float2(float(x[xb + tk * TILE + row{d}]),float(x[xb1 + tk * TILE + row{d}]));", .{ r, r, r });
        const new = std.fmt.comptimePrint("const float3 in{d} = float3(float(x[xb + tk * TILE + row{d}]),float(x[xb1 + tk * TILE + row{d}]),float(x[xb2 + tk * TILE + row{d}]));", .{ r, r, r, r });
        triple = replace(triple, old, new);
    }
    triple = replace(triple, "const float2 ins[8]", "const float3 ins[8]");
    triple = replace(triple, "acc[1][p*2u]=fma(ins[p*2u].y,w.x,acc[1][p*2u]);", "acc[1][p*2u]=fma(ins[p*2u].y,w.x,acc[1][p*2u]); acc[2][p*2u]=fma(ins[p*2u].z,w.x,acc[2][p*2u]);");
    triple = replace(triple, "acc[1][p*2u+1u]=fma(ins[p*2u+1u].y,w.y,acc[1][p*2u+1u]);", "acc[1][p*2u+1u]=fma(ins[p*2u+1u].y,w.y,acc[1][p*2u+1u]); acc[2][p*2u+1u]=fma(ins[p*2u+1u].z,w.y,acc[2][p*2u+1u]);");
    const epilogue = replace(replace(replace(EPILOGUE, "j<2u", "j<3u"), "j==0u?first:partner", "j==0u?first:(j==1u?partner:third)"), "if(lid<32u)", "if(lid<48u)");
    return THREE_MEMBERS ++ "\nif(partner==first) {\n" ++ outputTiles(original, 1) ++ "\n} else if(third==first) {\n" ++ outputTiles(paired ++ EPILOGUE, 2) ++ "\n} else {\n" ++ outputTiles(triple ++ epilogue, 3) ++ "\n}\n";
}
const SOURCE = groupedSource(false);
const THREE_SOURCE = groupedSource(true);
const PAIR_SOURCE: [:0]const u8 =
    @embedFile("kernels/glm_group2_pair.metal") ++ replace(SOURCE, "uint split = uint(threadgroup_position_in_grid.z);", "uint split=0u;");
fn laneSource(comptime source: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    var result: []const u8 = source;
    const names = [_][]const u8{ "x", "y", "z", "w" };
    for (0..4) |r| {
        const single_old = std.fmt.comptimePrint("const float in{d} = float(x[xb + tk * TILE + row{d}]);", .{ r, r });
        const single_new = (if (r == 0) "const float4 in4=float4(*((const device half4*)(x+xb+tk*TILE+(lane&3u)*4u)));\n" else "") ++ std.fmt.comptimePrint("const float in{d}=in4.{s};", .{ r, names[r] });
        result = replace(result, single_old, single_new);
        const pair_old = std.fmt.comptimePrint("const float2 in{d} = float2(float(x[xb + tk * TILE + row{d}]),float(x[xb1 + tk * TILE + row{d}]));", .{ r, r, r });
        const pair_new = (if (r == 0) "const float4 in4a=float4(*((const device half4*)(x+xb+tk*TILE+(lane&3u)*4u)));\nconst float4 in4b=float4(*((const device half4*)(x+xb1+tk*TILE+(lane&3u)*4u)));\n" else "") ++ std.fmt.comptimePrint("const float2 in{d}=float2(in4a.{s},in4b.{s});", .{ r, names[r], names[r] });
        result = replace(result, pair_old, pair_new);
        const triple_old = std.fmt.comptimePrint("const float3 in{d} = float3(float(x[xb + tk * TILE + row{d}]),float(x[xb1 + tk * TILE + row{d}]),float(x[xb2 + tk * TILE + row{d}]));", .{ r, r, r, r });
        const triple_new = (if (r == 0) "const float4 in4a=float4(*((const device half4*)(x+xb+tk*TILE+(lane&3u)*4u)));\nconst float4 in4b=float4(*((const device half4*)(x+xb1+tk*TILE+(lane&3u)*4u)));\nconst float4 in4c=float4(*((const device half4*)(x+xb2+tk*TILE+(lane&3u)*4u)));\n" else "") ++ std.fmt.comptimePrint("const float3 in{d}=float3(in4a.{s},in4b.{s},in4c.{s});", .{ r, names[r], names[r], names[r] });
        result = replace(result, triple_old, triple_new);
    }
    return result ++ "";
}
const LANE_SOURCE = laneSource(SOURCE);
const LANE_PAIR_SOURCE = laneSource(PAIR_SOURCE);
const LANE_THREE_SOURCE = laneSource(THREE_SOURCE);
const LANE_THREE_PAIR_SOURCE = laneSource(
    "const bool upper=threadgroup_position_in_grid.z!=0u;\nconst device half* x=upper?xu:xg;\nconst device ushort* trellis=upper?tu:tg;\ndevice half* y=upper?yu:yg;\n" ++
        replace(THREE_SOURCE, "uint split = uint(threadgroup_position_in_grid.z);", "uint split=0u;"),
);
var three_lane_kernels: support.Slots = support.empty;
var three_pair_lane_kernels: support.Slots = support.empty;
var single_lane_kernels: support.Slots = support.empty;
var pair_lane_kernels: support.Slots = support.empty;
var single_kernels: support.Slots = support.empty;
var pair_kernels: support.Slots = support.empty;
const Key = struct { input: c_int, output: c_int, slots: c_int, rate: c_int, paired: bool, reduction: Reduction, tiles: c_int = 1 };
const Entry = struct { key: Key, config: mlx.mlx_fast_metal_kernel_config };
var cache: [64]?Entry = @splat(null);

fn config(key: Key) !struct { value: mlx.mlx_fast_metal_kernel_config, cached: bool } {
    for (cache) |entry| if (entry) |e| if (std.meta.eql(e.key, key)) return .{ .value = e.config, .cached = true };
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const sh = [_]c_int{ key.slots, key.output };
    for (0..if (key.paired) @as(usize, 2) else 1) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(key.output, 16 * key.tiles) * 128, key.slots, if (key.paired) 2 else 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
    inline for (.{ "IDIM", "ODIM", "NSLOTS", "NHW", "SERIAL_REDUCTION" }, .{ key.input, key.output, key.slots, key.rate, @as(c_int, @intFromBool(key.reduction == .serial)) }) |name, v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TILES", key.tiles));
    for (&cache) |*entry| if (entry.* == null) {
        entry.* = .{ .key = key, .config = c };
        return .{ .value = c, .cached = true };
    };
    return .{ .value = c, .cached = false };
}
/// The packed rates the cooperative reader serves: every rate the format admits.
pub fn servesRate(n: c_int) bool {
    return n >= 0 and @import("expert_exl3.zig").kFromPackedDim(@intCast(n)) != null;
}
fn eligible(x: Arr, bank: Arr, ids: Arr) bool {
    if (x.ctx == null or bank.ctx == null or ids.ctx == null) return false;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(bank);
    const is = mlx.getShape(ids);
    return xs.len == 2 and ws.len == 4 and is.len == 1 and xs[0] == is[0] and xs[0] >= 2 and xs[0] <= 128 and
        xs[1] > 0 and @mod(xs[1], 128) == 0 and ws[0] > 0 and ws[1] == @divExact(xs[1], 16) and ws[2] > 0 and ws[2] <= @divTrunc(std.math.maxInt(c_int), 128) and @mod(ws[2], 8) == 0 and
        servesRate(ws[3]) and mlx.mlx_array_dtype(x) == .float16 and mlx.mlx_array_dtype(bank) == .uint16 and
        (mlx.mlx_array_dtype(ids) == .uint32 or mlx.mlx_array_dtype(ids) == .int32);
}
pub fn project(s: mlx.mlx_stream, x: Arr, bank: Arr, ids: Arr, reduction: Reduction) !?Arr {
    return projectLayout(s, x, bank, ids, reduction, .natural);
}
pub fn projectLayout(s: mlx.mlx_stream, x: Arr, bank: Arr, ids: Arr, reduction: Reduction, layout: Layout) !?Arr {
    if (!mlx.streamIsGpu(s) or !eligible(x, bank, ids)) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(bank);
    const cfg = try config(.{ .input = xs[1], .output = ws[2] * 16, .slots = xs[0], .rate = ws[3], .paired = false, .reduction = reduction, .tiles = if (layout == .lane and xs[0] == 24) 2 else 1 });
    defer if (!cfg.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg.value);
    };
    const kernel = if (layout == .lane and xs[0] == 24) try support.makeKernel(&three_lane_kernels, "sushi_glm_exl3_group3_tile2_lane", &.{ "x", "trellis", "slots" }, &.{"y"}, LANE_THREE_SOURCE) else if (layout == .lane) try support.makeKernel(&single_lane_kernels, "sushi_glm_exl3_group2_lane", &.{ "x", "trellis", "slots" }, &.{"y"}, LANE_SOURCE) else try support.makeKernel(&single_kernels, "sushi_glm_exl3_group2", &.{ "x", "trellis", "slots" }, &.{"y"}, SOURCE);
    const iv = mlx.mlx_vector_array_new_data(&.{ x, bank, ids }, 3);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg.value, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    return y;
}
pub fn pair(s: mlx.mlx_stream, xg: Arr, xu: Arr, tg: Arr, tu: Arr, ids: Arr, reduction: Reduction) !?[2]Arr {
    return pairLayout(s, xg, xu, tg, tu, ids, reduction, .natural);
}
pub fn pairLayout(s: mlx.mlx_stream, xg: Arr, xu: Arr, tg: Arr, tu: Arr, ids: Arr, reduction: Reduction, layout: Layout) !?[2]Arr {
    if (!mlx.streamIsGpu(s) or !eligible(xg, tg, ids) or !eligible(xu, tu, ids) or !std.mem.eql(c_int, mlx.getShape(tg), mlx.getShape(tu)) or !std.mem.eql(c_int, mlx.getShape(xg), mlx.getShape(xu))) return null;
    const xs = mlx.getShape(xg);
    const ws = mlx.getShape(tg);
    const cfg = try config(.{ .input = xs[1], .output = ws[2] * 16, .slots = xs[0], .rate = ws[3], .paired = true, .reduction = reduction, .tiles = if (layout == .lane and xs[0] == 24) 2 else 1 });
    defer if (!cfg.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg.value);
    };
    const kernel = if (layout == .lane and xs[0] == 24) try support.makeKernel(&three_pair_lane_kernels, "sushi_glm_exl3_pair_group3_tile2_lane", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, LANE_THREE_PAIR_SOURCE) else if (layout == .lane) try support.makeKernel(&pair_lane_kernels, "sushi_glm_exl3_pair_group2_lane", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, LANE_PAIR_SOURCE) else try support.makeKernel(&pair_kernels, "sushi_glm_exl3_pair_group2", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, PAIR_SOURCE);
    const iv = mlx.mlx_vector_array_new_data(&.{ xg, xu, tg, tu, ids }, 5);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg.value, s));
    var output = [2]Arr{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (output) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&output, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    return output;
}

pub const Down = enum { baseline, grouped };
var fused_middle_down_calls: usize = 0;
/// Full routed chain; unsupported inputs return null so the caller falls back to moeClamped.
pub fn moe(s: mlx.mlx_stream, x: Arr, bank: @import("root.zig").Bank, indices: Arr, scores: Arr, dec: @import("expert_exl3.zig").Decode, reduction: Reduction, down: Down) !?Arr {
    return moeLayout(s, x, bank, indices, scores, dec, reduction, down, .natural);
}
pub fn moeLayout(s: mlx.mlx_stream, x: Arr, bank: @import("root.zig").Bank, indices: Arr, scores: Arr, dec: @import("expert_exl3.zig").Decode, reduction: Reduction, down: Down, layout: Layout) !?Arr {
    return moeLayoutShared(s, x, bank, indices, scores, dec, reduction, down, layout, null);
}
/// `moeLayout` plus `shared` (x's shape, BF16) added to the stored routed rows in the reduce,
/// bit for bit the separate BF16 add.
pub fn moeLayoutShared(s: mlx.mlx_stream, x: Arr, bank: @import("root.zig").Bank, indices: Arr, scores: Arr, dec: @import("expert_exl3.zig").Decode, reduction: Reduction, down: Down, layout: Layout, shared: ?Arr) !?Arr {
    if (!mlx.streamIsGpu(s)) return null;
    if (shared) |value| if (mlx.mlx_array_dtype(value) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(value), mlx.getShape(x))) return null;
    for ([_]Arr{ x, indices, scores, bank.gate.trellis, bank.gate.suh, bank.gate.svh, bank.up.trellis, bank.up.suh, bank.up.svh, bank.down.trellis, bank.down.suh, bank.down.svh }) |value| if (value.ctx == null) return null;
    const shape = mlx.getShape(x);
    const ids_shape = mlx.getShape(indices);
    const gs = mlx.getShape(bank.gate.trellis);
    if (shape.len != 3 or ids_shape.len != 3 or gs.len != 4 or shape[0] != 1 or shape[1] < 2 or shape[1] > 16 or ids_shape[0] != 1 or ids_shape[1] != shape[1] or ids_shape[2] != 8 or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const r = shape[1];
    const h = shape[2];
    const top: c_int = 8;
    const nslots = r * top;
    const inter = std.math.mul(c_int, gs[2], 16) catch return null;
    if (h <= 0 or @mod(h, 128) != 0 or inter <= 0 or !std.mem.eql(c_int, ids_shape, mlx.getShape(scores))) return null;
    const idtype = mlx.mlx_array_dtype(indices);
    const scoretype = mlx.mlx_array_dtype(scores);
    if ((idtype != .uint32 and idtype != .int32) or (scoretype != .float32 and scoretype != .float16 and scoretype != .bfloat16)) return null;
    const api = @import("root.zig");
    api.validateClampedProjection(bank.gate, gs[0], h, inter) catch return null;
    api.validateClampedProjection(bank.up, gs[0], h, inter) catch return null;
    api.validateClampedProjection(bank.down, gs[0], inter, h) catch return null;
    base.setDecodeParams(dec);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&flat, x, &.{ r, h }, 2, s));
    try mlx.check(mlx.mlx_reshape(&ids, indices, &.{nslots}, 1, s));
    try mlx.check(mlx.mlx_reshape(&sc, scores, &.{nslots}, 1, s));
    const prep = if (layout == .lane) try base.lanePairPrepare(s, flat, bank.gate.suh, bank.up.suh, ids, null, h, nslots, top) else try support.prepare(s, flat, bank.gate.suh, bank.up.suh, ids, null, h, nslots, top);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    const gu = (try pairLayout(s, prep[0], prep[1], bank.gate.trellis, bank.up.trellis, ids, reduction, layout)) orelse return null;
    defer for (gu) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const d = blk: {
        if (layout == .lane) {
            const middle = try support.lane_middle(s, gu[0], gu[1], bank.gate.svh, bank.up.svh, bank.down.suh, ids, inter, nslots, 10);
            defer _ = mlx.mlx_array_free(middle);
            break :blk if (down == .grouped) (try projectLayout(s, middle, bank.down.trellis, ids, reduction, .lane)) orelse return null else try support.lane_down(s, middle, bank.down.trellis, ids);
        }
        if (down == .baseline and h == 4096 and inter == 2048 and gs[3] == 36 and dec.codebook.servesGlmFastPaths() and dec.window == .w12)
            if (try support.fused_middle_down(s, gu[0], gu[1], bank.down.trellis, bank.gate.svh, bank.up.svh, bank.down.suh, ids, 10, 8)) |v| {
                fused_middle_down_calls += 1;
                break :blk v;
            };
        const middle = try support.middle(s, gu[0], gu[1], bank.gate.svh, bank.up.svh, bank.down.suh, ids, inter, nslots, 10);
        defer _ = mlx.mlx_array_free(middle);
        break :blk if (down == .grouped) (try project(s, middle, bank.down.trellis, ids, reduction)) orelse return null else try base.indexedGemvCoopF16(s, middle, bank.down.trellis, ids);
    };
    defer _ = mlx.mlx_array_free(d);
    const out = if (shared) |value| blk: {
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_reshape(&rows, value, &.{ r, h }, 2, s));
        break :blk try base.downFinishReduceShared(s, d, bank.down.svh, ids, sc, h, r, top, .bfloat16, rows);
    } else try base.downFinishReduce(s, d, bank.down.svh, ids, sc, h, r, top, .bfloat16);
    defer _ = mlx.mlx_array_free(out);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_reshape(&result, out, shape.ptr, shape.len, s));
    return result;
}

const Owned = struct {
    values: std.ArrayList(Arr) = .empty,
    fn deinit(self: *Owned) void {
        for (self.values.items) |a| {
            _ = mlx.mlx_array_free(a);
        }
        self.values.deinit(std.testing.allocator);
    }
    fn own(self: *Owned, a: Arr) !Arr {
        self.values.append(std.testing.allocator, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        return a;
    }
    fn weights(self: *Owned, e: c_int, k: c_int, n: c_int, rate: c_int, seed: u64) !Arr {
        const len: usize = @intCast(e * @divExact(k, 16) * @divExact(n, 16) * rate);
        const data = try std.testing.allocator.alloc(u16, len);
        defer std.testing.allocator.free(data);
        var rng = std.Random.DefaultPrng.init(seed);
        for (data) |*v| v.* = rng.random().int(u16);
        return self.own(mlx.mlx_array_new_data(data.ptr, &.{ e, @divExact(k, 16), @divExact(n, 16), rate }, 4, .uint16));
    }
    fn floats(self: *Owned, shape: []const c_int, dtype: mlx.mlx_dtype, seed: u64, scale: f32, s: mlx.mlx_stream) !Arr {
        var count: usize = 1;
        for (shape) |v| count *= @intCast(v);
        const data = try std.testing.allocator.alloc(f32, count);
        defer std.testing.allocator.free(data);
        var rng = std.Random.DefaultPrng.init(seed);
        for (data) |*v| v.* = (rng.random().float(f32) - 0.5) * scale;
        const input = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
        defer _ = mlx.mlx_array_free(input);
        var out = mlx.mlx_array_new();
        mlx.check(mlx.mlx_astype(&out, input, dtype, s)) catch |err| {
            _ = mlx.mlx_array_free(out);
            return err;
        };
        return self.own(out);
    }
};
fn exact(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const n = mlx.mlx_array_size(a);
    if (mlx.mlx_array_dtype(a) == .float16) try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float16(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float16(b).?[0..n])) else try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n]);
}

test "GLM group2 cooperative projections preserve F16 bits at all rates and ballot boundaries" {
    const s = mlx.gpuStream();
    base.setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer base.setDecodeParams(.mul1);
    const expert = @import("expert_exl3.zig");
    const rates = (expert.Rate.max_n - expert.Rate.min_n) / 2 + 1;
    for (0..rates + 1) |case| {
        const production = case == rates;
        const k: c_int = if (production) 4096 else 128;
        const n: c_int = if (production) 2048 else 128;
        const e: c_int = if (production) 32 else 128;
        const rate: c_int = if (production) 36 else @intCast(expert.Rate.min_n + case * 2);
        var owned: Owned = .{};
        defer owned.deinit();
        const tg = try owned.weights(e, k, n, rate, 132);
        const tu = try owned.weights(e, k, n, rate, 175);
        for ([_]c_int{ 3, 7, 32, 65, 128 }) |count| {
            const xg = try owned.floats(&.{ count, k }, .float16, 821, 0.3, s);
            const xu = try owned.floats(&.{ count, k }, .float16, 823, 0.5, s);
            var data: [128]u32 = undefined;
            for (data[0..@intCast(count)], 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(e)));
            for ([_]usize{ 0, 2, 6, 31, 32, 63, 64, 127 }) |i| if (i < count) {
                data[i] = 0;
            };
            const ids = try owned.own(mlx.mlx_array_new_data(&data, &.{count}, 1, .uint32));
            const rg = try owned.own(try base.indexedGemvCoopF16(s, xg, tg, ids));
            const ru = try owned.own(try base.indexedGemvCoopF16(s, xu, tu, ids));
            for ([_]Reduction{ .parallel, .serial }) |reduction| {
                const got = (try project(s, xg, tg, ids, reduction)) orelse return error.TestExpectedGroup2;
                defer _ = mlx.mlx_array_free(got);
                try exact(rg, got);
                const both = (try pair(s, xg, xu, tg, tu, ids, reduction)) orelse return error.TestExpectedGroup2;
                defer for (both) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                try exact(rg, both[0]);
                try exact(ru, both[1]);
            }
        }
    }
}

test "GLM group2 singleton and all-shared routes preserve strided input bits" {
    const s = mlx.gpuStream();
    base.setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer base.setDecodeParams(.mul1);
    var owned: Owned = .{};
    defer owned.deinit();
    const source = try owned.floats(&.{ 65, 256 }, .float16, 481, 0.2, s);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_slice(&x, source, &.{ 0, 0 }, 2, &.{ 65, 256 }, 2, &.{ 1, 2 }, 2, s));
    const bank = try owned.weights(128, 128, 128, 36, 876);
    var transposed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(transposed);
    try mlx.check(mlx.mlx_transpose_axes(&transposed, bank, &.{ 0, 2, 1, 3 }, 4, s));
    for ([_]bool{ false, true }) |shared| {
        var raw: [130]u32 = undefined;
        for (&raw, 0..) |*v, i| v.* = if (shared) 0 else @intCast(i / 2);
        const all = try owned.own(mlx.mlx_array_new_data(&raw, &.{130}, 1, .uint32));
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_slice(&ids, all, &.{1}, 1, &.{130}, 1, &.{2}, 1, s));
        const reference = try base.indexedGemvCoopF16(s, x, transposed, ids);
        defer _ = mlx.mlx_array_free(reference);
        for ([_]Reduction{ .parallel, .serial }) |reduction| {
            const output = (try project(s, x, transposed, ids, reduction)) orelse return error.TestExpectedGroup2;
            defer _ = mlx.mlx_array_free(output);
            try exact(reference, output);
        }
    }
}

test "GLM group2 routed-chain guards precede any preparation" {
    const s = mlx.gpuStream();
    var owned: Owned = .{};
    defer owned.deinit();
    const proj = @import("root.zig").Proj{ .trellis = try owned.weights(8, 128, 128, 36, 71), .suh = try owned.floats(&.{ 8, 128 }, .float16, 73, 0.1, s), .svh = try owned.floats(&.{ 8, 128 }, .float16, 77, 0.1, s) };
    const valid = @import("root.zig").Bank{ .gate = proj, .up = proj, .down = proj };
    const x = try owned.floats(&.{ 1, 2, 128 }, .bfloat16, 79, 2, s);
    var ids_data: [16]u32 = undefined;
    for (&ids_data, 0..) |*v, i| v.* = @intCast(i % 8);
    const ids = try owned.own(mlx.mlx_array_new_data(&ids_data, &.{ 1, 2, 8 }, 3, .uint32));
    const scores = try owned.floats(&.{ 1, 2, 8 }, .float32, 83, 1, s);
    var malformed = valid;
    malformed.gate.suh = try owned.floats(&.{ 8, 1 }, .float16, 81, 0.1, s);
    const dec = @import("expert_exl3.zig").Decode{ .codebook = .mcg, .window = .w12 };
    try std.testing.expect((try moe(s, x, malformed, ids, scores, dec, .parallel, .grouped)) == null);
    const bad_scores = try owned.floats(&.{ 1, 2, 7 }, .float32, 91, 1, s);
    try std.testing.expect((try moe(s, x, valid, ids, bad_scores, dec, .parallel, .grouped)) == null);
    try std.testing.expect((try moe(s, x, valid, ids, ids, dec, .parallel, .grouped)) == null);
    try std.testing.expect((try moe(s, x, valid, scores, scores, dec, .parallel, .grouped)) == null);
    var huge = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(huge);
    const scalar = try owned.own(mlx.mlx_array_new_data(&[_]u16{0}, &.{1}, 1, .uint16));
    try mlx.check(mlx.mlx_broadcast_to(&huge, scalar, &.{ 8, 8, @divTrunc(std.math.maxInt(c_int), 16) + 1, 36 }, 4, s));
    malformed = valid;
    malformed.gate.trellis = huge;
    try std.testing.expect((try moe(s, x, malformed, ids, scores, dec, .parallel, .grouped)) == null);
}

fn laneInput(owned: *Owned, x: Arr, s: mlx.mlx_stream) !Arr {
    const width: usize = @intCast(mlx.getShape(x)[1]);
    const indices = try std.testing.allocator.alloc(u32, width);
    defer std.testing.allocator.free(indices);
    for (0..width / 16) |tile| for (0..4) |q| for ([_]usize{ 0, 1, 8, 9 }, 0..) |offset, j| {
        indices[tile * 16 + q * 4 + j] = @intCast(tile * 16 + q * 2 + offset);
    };
    const ids = mlx.mlx_array_new_data(indices.ptr, &[_]c_int{@intCast(width)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    var result = mlx.mlx_array_new();
    mlx.check(mlx.mlx_take_axis(&result, x, ids, 1, s)) catch |err| {
        _ = mlx.mlx_array_free(result);
        return err;
    };
    return owned.own(result);
}

test "GLM group2 half4 composition preserves all-rate cooperative projection bits" {
    const s = mlx.gpuStream();
    base.setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer base.setDecodeParams(.mul1);
    const expert = @import("expert_exl3.zig");
    const rates = (expert.Rate.max_n - expert.Rate.min_n) / 2 + 1;
    for (0..rates + 1) |case| {
        const production = case == rates;
        const k: c_int = if (production) 4096 else 128;
        const n: c_int = if (production) 2048 else 128;
        const rate: c_int = if (production) 36 else @intCast(expert.Rate.min_n + 2 * case);
        var owned: Owned = .{};
        defer owned.deinit();
        const g = try owned.weights(32, k, n, rate, 987);
        const u = try owned.weights(32, k, n, rate, 977);
        for ([_]c_int{ 3, 24, 32, 65, 128 }) |count| {
            const xg = try owned.floats(&.{ count, k }, .float16, 877, 0.3, s);
            const xu = try owned.floats(&.{ count, k }, .float16, 857, 0.5, s);
            const pg = try laneInput(&owned, xg, s);
            const pu = try laneInput(&owned, xu, s);
            var data: [128]u32 = undefined;
            for (data[0..@intCast(count)], 0..) |*id, i| id.* = @intCast(i % @as(usize, if (count == 24) 8 else 17));
            const ids = try owned.own(mlx.mlx_array_new_data(&data, &.{count}, 1, .uint32));
            const rg = try owned.own(try base.indexedGemvCoopF16(s, xg, g, ids));
            const ru = try owned.own(try base.indexedGemvCoopF16(s, xu, u, ids));
            for ([_]Reduction{ .parallel, .serial }) |reduction| {
                const got = (try projectLayout(s, pg, g, ids, reduction, .lane)) orelse return error.TestExpectedGroup2;
                defer _ = mlx.mlx_array_free(got);
                try exact(rg, got);
                const pair_out = (try pairLayout(s, pg, pu, g, u, ids, reduction, .lane)) orelse return error.TestExpectedGroup2;
                defer for (pair_out) |v| {
                    _ = mlx.mlx_array_free(v);
                };
                try exact(rg, pair_out[0]);
                try exact(ru, pair_out[1]);
            }
        }
    }
}

test "GLM group2 serves every admitted rate, exact to the cooperative reader" {
    const s = mlx.gpuStream();
    base.setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer base.setDecodeParams(.mul1);
    var owned: Owned = .{};
    defer owned.deinit();
    const x = try owned.floats(&.{ 4, 128 }, .float16, 5, 0.3, s);
    const ids = try owned.own(mlx.mlx_array_new_data(&[_]u32{ 0, 1, 1, 2 }, &.{4}, 1, .uint32));
    const expert = @import("expert_exl3.zig");
    var n: u32 = expert.Rate.min_n;
    while (n <= expert.Rate.max_n) : (n += 1) {
        if (expert.kFromPackedDim(n) == null) {
            try std.testing.expect(!servesRate(@intCast(n)));
            continue;
        }
        const bank = try owned.weights(4, 128, 128, @intCast(n), 41);
        const got = (try project(s, x, bank, ids, .serial)) orelse return error.TestExpectedGroup2;
        defer _ = mlx.mlx_array_free(got);
        try exact(try owned.own(try base.indexedGemvCoopF16(s, x, bank, ids)), got);
    }
}

test "GLM group2 keeps the cooperative reader's bytes under PAIR1" {
    const s = mlx.gpuStream();
    base.setDecodeParams(.{ .codebook = .pair1, .window = .w16 });
    defer base.setDecodeParams(.mul1);
    var owned: Owned = .{};
    defer owned.deinit();
    const xg = try owned.floats(&.{ 7, 128 }, .float16, 5, 0.3, s);
    const xu = try owned.floats(&.{ 7, 128 }, .float16, 7, 0.5, s);
    const ids = try owned.own(mlx.mlx_array_new_data(&[_]u32{ 0, 1, 1, 2, 0, 3, 1 }, &.{7}, 1, .uint32));
    for ([_]c_int{ 24, 32, 42, 64 }) |rate| {
        const g = try owned.weights(4, 128, 128, rate, 41);
        const u = try owned.weights(4, 128, 128, rate, 43);
        const rg = try owned.own(try base.indexedGemvCoopF16(s, xg, g, ids));
        const ru = try owned.own(try base.indexedGemvCoopF16(s, xu, u, ids));
        for ([_]Reduction{ .parallel, .serial }) |reduction| {
            const got = (try project(s, xg, g, ids, reduction)) orelse return error.TestExpectedGroup2;
            defer _ = mlx.mlx_array_free(got);
            try exact(rg, got);
            const both = (try pair(s, xg, xu, g, u, ids, reduction)) orelse return error.TestExpectedGroup2;
            defer for (both) |a| {
                _ = mlx.mlx_array_free(a);
            };
            try exact(rg, both[0]);
            try exact(ru, both[1]);
        }
    }
}

// Group2 needs two rows; DFlash verify sends three or four.
test "GLM serves spec1 and std1 on group2's routed chain with the unfolded oracle's bytes, fused middle-down included" {
    const s = mlx.gpuStream();
    defer base.setDecodeParams(.mul1);
    defer base.unfolded_for_test = false;
    const e: c_int = 12;
    for ([_]@import("expert_exl3.zig").Codebook{ .spec1, .std1 }) |codebook| for ([_]@import("expert_exl3.zig").Window{ .w12, .w15 }) |window| for ([_]c_int{ 36, 40 }) |rate| for ([_]c_int{ 2, 3, 4 }) |rows| {
        var owned: Owned = .{};
        defer owned.deinit();
        var projections: [3]@import("root.zig").Proj = undefined;
        for (&projections, 0..) |*p, i| {
            const k: c_int = if (i == 2) 2048 else 4096;
            const n: c_int = if (i == 2) 4096 else 2048;
            p.* = .{ .trellis = try owned.weights(e, k, n, rate, 800 + i), .suh = try owned.floats(&.{ e, k }, .float16, 810 + i, 0.4, s), .svh = try owned.floats(&.{ e, n }, .float16, 820 + i, 0.1, s) };
        }
        const bank = @import("root.zig").Bank{ .gate = projections[0], .up = projections[1], .down = projections[2] };
        var id_data: [32]u32 = undefined;
        var score_data: [32]f32 = undefined;
        for (&id_data, &score_data, 0..) |*id, *sc, i| {
            id.* = @intCast((i / 8 + (i % 8) * 5) % 12);
            sc.* = 0.05 * @as(f32, @floatFromInt(i % 8 + 1));
        }
        const ids = try owned.own(mlx.mlx_array_new_data(&id_data, &.{ 1, rows, 8 }, 3, .uint32));
        const scores = try owned.own(mlx.mlx_array_new_data(&score_data, &.{ 1, rows, 8 }, 3, .float32));
        const dec = @import("expert_exl3.zig").Decode.tabled(codebook, window);
        const fuses = rate == 36 and window == .w12;
        for ([_]f32{ 0.01, 2 }) |x_scale| {
            const x = try owned.floats(&.{ 1, rows, 4096 }, .bfloat16, 831, x_scale, s);
            for ([_]Layout{ .natural, .lane }) |layout| for ([_]Down{ .baseline, .grouped }) |down| {
                var got: [2]Arr = undefined;
                for (&got, 0..) |*y, unfolded| {
                    base.unfolded_for_test = unfolded == 1;
                    const fused = fused_middle_down_calls;
                    y.* = try owned.own((try moeLayout(s, x, bank, ids, scores, dec, .serial, down, layout)) orelse return error.TestExpectedGroup2);
                    try std.testing.expectEqual(fused + @intFromBool(fuses and layout == .natural and down == .baseline), fused_middle_down_calls);
                }
                try exact(got[1], got[0]);
            };
        }
    };
}

test "GLM serves PAIR1 on group2's routed chain with the unfolded oracle's bytes, fused middle-down included" {
    const s = mlx.gpuStream();
    defer base.setDecodeParams(.mul1);
    defer base.unfolded_for_test = false;
    const e: c_int = 12;
    for ([_]@import("expert_exl3.zig").Codebook{.pair1}) |codebook| for ([_]@import("expert_exl3.zig").Window{.w16}) |window| for ([_]c_int{ 24, 36, 40 }) |rate| for ([_]c_int{ 2, 3, 4 }) |rows| {
        var owned: Owned = .{};
        defer owned.deinit();
        var projections: [3]@import("root.zig").Proj = undefined;
        for (&projections, 0..) |*p, i| {
            const k: c_int = if (i == 2) 2048 else 4096;
            const n: c_int = if (i == 2) 4096 else 2048;
            p.* = .{ .trellis = try owned.weights(e, k, n, rate, 800 + i), .suh = try owned.floats(&.{ e, k }, .float16, 810 + i, 0.4, s), .svh = try owned.floats(&.{ e, n }, .float16, 820 + i, 0.1, s) };
        }
        const bank = @import("root.zig").Bank{ .gate = projections[0], .up = projections[1], .down = projections[2] };
        var id_data: [32]u32 = undefined;
        var score_data: [32]f32 = undefined;
        for (&id_data, &score_data, 0..) |*id, *sc, i| {
            id.* = @intCast((i / 8 + (i % 8) * 5) % 12);
            sc.* = 0.05 * @as(f32, @floatFromInt(i % 8 + 1));
        }
        const ids = try owned.own(mlx.mlx_array_new_data(&id_data, &.{ 1, rows, 8 }, 3, .uint32));
        const scores = try owned.own(mlx.mlx_array_new_data(&score_data, &.{ 1, rows, 8 }, 3, .float32));
        const dec = @import("expert_exl3.zig").Decode.tabled(codebook, window);
        const fuses = false;
        for ([_]f32{ 0.01, 2 }) |x_scale| {
            const x = try owned.floats(&.{ 1, rows, 4096 }, .bfloat16, 831, x_scale, s);
            for ([_]Layout{ .natural, .lane }) |layout| for ([_]Down{ .baseline, .grouped }) |down| {
                var got: [2]Arr = undefined;
                for (&got, 0..) |*y, unfolded| {
                    base.unfolded_for_test = unfolded == 1;
                    const fused = fused_middle_down_calls;
                    y.* = try owned.own((try moeLayout(s, x, bank, ids, scores, dec, .serial, down, layout)) orelse return error.TestExpectedGroup2);
                    try std.testing.expectEqual(fused + @intFromBool(fuses and layout == .natural and down == .baseline), fused_middle_down_calls);
                }
                try exact(got[1], got[0]);
            };
        }
    };
}

test "GLM group2 shared expert joined in the reduce equals the separate BF16 add" {
    const s = mlx.gpuStream();
    const dec: @import("expert_exl3.zig").Decode = .{ .codebook = .mcg, .window = .w14 };
    defer base.setDecodeParams(.mul1);
    const e: c_int = 12;
    for ([_]c_int{ 36, 40 }) |rate| for ([_]c_int{ 3, 4 }) |rows| {
        var owned: Owned = .{};
        defer owned.deinit();
        var projections: [3]@import("root.zig").Proj = undefined;
        for (&projections, 0..) |*p, i| {
            const k: c_int = if (i == 2) 2048 else 4096;
            const n: c_int = if (i == 2) 4096 else 2048;
            p.* = .{ .trellis = try owned.weights(e, k, n, rate, 700 + i), .suh = try owned.floats(&.{ e, k }, .float16, 710 + i, 0.4, s), .svh = try owned.floats(&.{ e, n }, .float16, 720 + i, 0.1, s) };
        }
        const bank = @import("root.zig").Bank{ .gate = projections[0], .up = projections[1], .down = projections[2] };
        const x = try owned.floats(&.{ 1, rows, 4096 }, .bfloat16, 731, 2, s);
        // Shared outputs at the routed magnitudes, so the sum rounds in both directions.
        const shared = try owned.floats(&.{ 1, rows, 4096 }, .bfloat16, 733, 0.5, s);
        var id_data: [32]u32 = undefined;
        var score_data: [32]f32 = undefined;
        for (&id_data, &score_data, 0..) |*id, *sc, i| {
            id.* = @intCast((i / 8 + (i % 8) * 3) % 12);
            sc.* = 0.05 * @as(f32, @floatFromInt(i % 8 + 1));
        }
        const ids = try owned.own(mlx.mlx_array_new_data(&id_data, &.{ 1, rows, 8 }, 3, .uint32));
        const scores = try owned.own(mlx.mlx_array_new_data(&score_data, &.{ 1, rows, 8 }, 3, .float32));
        const routed = try owned.own((try moeLayout(s, x, bank, ids, scores, dec, .serial, .grouped, .lane)) orelse return error.TestExpectedGroup2);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_add(&want, routed, shared, s));
        const got = try owned.own((try moeLayoutShared(s, x, bank, ids, scores, dec, .serial, .grouped, .lane, shared)) orelse return error.TestExpectedGroup2);
        try exact(want, got);
    };
}
