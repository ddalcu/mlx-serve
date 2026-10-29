//! Sushi packs keep the source checkpoint's tensor layout beside their EXL3 experts. These steps
//! move the tensors to where the GLM-5-Next and MiMo loaders read them, once, right after load.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

const Weights = @import("../../weights/resident.zig").Weights;
const indexShardSet = @import("../../weights/resident.zig").indexShardSet;
const freeShardSet = @import("../../weights/resident.zig").freeShardSet;
const Owners = @import("../../weights/load.zig").Owners;
const indexOwners = @import("../../weights/load.zig").indexOwners;

fn rename(weights: *Weights, from: []const u8, to: []const u8) !void {
    const kv = weights.map.fetchRemove(from) orelse return;
    weights.allocator.free(kv.key);
    try weights.map.put(try weights.allocator.dupe(u8, to), kv.value);
}

fn collectKeys(weights: *const Weights, a: std.mem.Allocator, comptime keep: fn ([]const u8) bool) !std.ArrayList([]u8) {
    var out: std.ArrayList([]u8) = .empty;
    errdefer freeKeys(&out, a);
    var it = weights.map.keyIterator();
    while (it.next()) |k| if (keep(k.*)) try out.append(a, try a.dupe(u8, k.*));
    return out;
}

fn freeKeys(list: *std.ArrayList([]u8), a: std.mem.Allocator) void {
    for (list.items) |k| a.free(k);
    list.deinit(a);
}

const hc_tags = [_]struct { src: []const u8, dst: []const u8 }{
    .{ .src = ".hc_attn_", .dst = ".attn_hc." },
    .{ .src = ".hc_ffn_", .dst = ".ffn_hc." },
};

fn isHcKey(key: []const u8) bool {
    inline for (hc_tags) |t| if (std.mem.indexOf(u8, key, t.src) != null) return true;
    return false;
}

/// `layers.N.hc_attn_fn` is `layers.N.attn_hc.fn` in the layout the loader reads.
pub fn renameHyperConnections(weights: *Weights) !void {
    const a = weights.allocator;
    var keys = try collectKeys(weights, a, isHcKey);
    defer freeKeys(&keys, a);
    for (keys.items) |key| {
        inline for (hc_tags) |t| if (std.mem.indexOf(u8, key, t.src)) |at| {
            const to = try std.fmt.allocPrint(a, "{s}{s}{s}", .{ key[0..at], t.dst, key[at + t.src.len ..] });
            defer a.free(to);
            try rename(weights, key, to);
        };
    }
}

/// GLM's NextN layer past the trunk (`<prefix>.layers.<trunk_layers>`, as the release names it) under the head's names:
/// `eh_proj` / `enorm` / `hnorm` and `shared_head.norm` (as `norm`) at `<prefix>.mtp.0`, the rest under `.mtp.0.block`.
pub fn renameGlmMtp(weights: *Weights, prefix: []const u8, trunk_layers: u32) !void {
    const a = weights.allocator;
    var lb: [128]u8 = undefined;
    const layer = try std.fmt.bufPrint(&lb, "{s}.layers.{d}.", .{ prefix, trunk_layers });
    var keys: std.ArrayList([]u8) = .empty;
    defer freeKeys(&keys, a);
    var it = weights.map.keyIterator();
    while (it.next()) |k| if (std.mem.startsWith(u8, k.*, layer)) try keys.append(a, try a.dupe(u8, k.*));
    for (keys.items) |key| {
        var rest = key[layer.len..];
        var sub: []const u8 = ".block";
        for ([_][]const u8{ "eh_proj.", "enorm.", "hnorm.", "shared_head." }) |p| if (std.mem.startsWith(u8, rest, p)) {
            sub = "";
            if (p[0] == 's') rest = rest[p.len..];
        };
        const to = try std.fmt.allocPrint(a, "{s}.mtp.0{s}.{s}", .{ prefix, sub, rest });
        defer a.free(to);
        try rename(weights, key, to);
    }
}

const KvB = struct { heads: u32, nope: u32, v: u32, latent: u32 };

fn isKvBKey(key: []const u8) bool {
    return std.mem.endsWith(u8, key, ".self_attn.kv_b_proj.weight");
}

/// The pack stores the MLA expansion `kv_b_proj` whole; the loader reads its two halves per head:
/// `embed_q` [H, latent, nope] and `unembed_out` [H, v, latent], dense bf16 (dequantized once).
pub fn splitKvB(weights: *Weights, d: KvB, s: mlx.mlx_stream) !void {
    const a = weights.allocator;
    var keys = try collectKeys(weights, a, isKvBKey);
    defer freeKeys(&keys, a);
    for (keys.items) |key| {
        const base = key[0 .. key.len - ".kv_b_proj.weight".len];
        const w = weights.map.get(key).?;
        var sc_name: [256]u8 = undefined;
        var bi_name: [256]u8 = undefined;
        const sc = weights.get(std.fmt.bufPrint(&sc_name, "{s}.kv_b_proj.scales", .{base}) catch unreachable);
        const bi = weights.get(std.fmt.bufPrint(&bi_name, "{s}.kv_b_proj.biases", .{base}) catch unreachable);

        const rows: c_int = @intCast(d.heads * (d.nope + d.v));
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        if (sc) |scales| {
            const sh = mlx.getShape(w);
            const ssh = mlx.getShape(scales);
            if (sh.len != 2 or ssh.len != 2 or sh[0] != rows or ssh[1] <= 0 or @mod(@as(c_int, @intCast(d.latent)), ssh[1]) != 0) return error.GlmKvBGeometry;
            const group: c_int = @divExact(@as(c_int, @intCast(d.latent)), ssh[1]);
            const bits: c_int = @divExact(sh[1] * 32, @as(c_int, @intCast(d.latent)));
            try mlx.check(mlx.mlx_dequantize(&dense, w, scales, bi orelse mlx.mlx_array{ .ctx = null }, mlx.mlx_optional_int.some(group), mlx.mlx_optional_int.some(bits), "affine", .{}, .{ .value = mlx.mlx_array_dtype(scales), .has_value = true }, s));
        } else {
            try mlx.check(mlx.mlx_array_set(&dense, w));
        }
        const dsh = mlx.getShape(dense);
        if (dsh.len != 2 or dsh[0] != rows or dsh[1] != @as(c_int, @intCast(d.latent))) return error.GlmKvBGeometry;

        const h: c_int = @intCast(d.heads);
        const nope: c_int = @intCast(d.nope);
        const per_head = [_]c_int{ h, nope + @as(c_int, @intCast(d.v)), @intCast(d.latent) };
        var cube = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cube);
        try mlx.check(mlx.mlx_reshape(&cube, dense, &per_head, 3, s));

        var k_half = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_half);
        try mlx.check(mlx.mlx_slice(&k_half, cube, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ h, nope, per_head[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_swapaxes(&k_t, k_half, 1, 2, s));
        var embed_q = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(embed_q);
        try mlx.check(mlx.mlx_contiguous(&embed_q, k_t, false, s));

        var v_half = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_half);
        try mlx.check(mlx.mlx_slice(&v_half, cube, &[_]c_int{ 0, nope, 0 }, 3, &[_]c_int{ h, per_head[1], per_head[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var unembed = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(unembed);
        try mlx.check(mlx.mlx_contiguous(&unembed, v_half, false, s));

        try mlx.check(mlx.mlx_array_eval(embed_q));
        try mlx.check(mlx.mlx_array_eval(unembed));

        var name: [256]u8 = undefined;
        try weights.map.put(try a.dupe(u8, std.fmt.bufPrint(&name, "{s}.embed_q.weight", .{base}) catch unreachable), embed_q);
        try weights.map.put(try a.dupe(u8, std.fmt.bufPrint(&name, "{s}.unembed_out.weight", .{base}) catch unreachable), unembed);
        for ([_][]const u8{ "weight", "scales", "biases" }) |leaf| {
            const old = std.fmt.bufPrint(&name, "{s}.kv_b_proj.{s}", .{ base, leaf }) catch unreachable;
            if (weights.map.fetchRemove(old)) |kv| {
                a.free(kv.key);
                _ = mlx.mlx_array_free(kv.value);
            }
        }
    }
}

pub const FP8_BLOCK = 128;

/// The tensor-parallel rank count whose slabs `[q|k|v]` tile into exactly `scale_rows` grid rows
/// (each slab tiles its own rows, so a slab's last tile may be partial).
fn solveQkvTp(q: u64, k: u64, v: u64, scale_rows: u64) !u32 {
    var found: ?u32 = null;
    for ([_]u64{ 8, 4 }) |tp| {
        if (q % tp != 0 or k % tp != 0 or v % tp != 0) continue;
        const per = (q + k + v) / tp;
        if (scale_rows != tp * ((per + FP8_BLOCK - 1) / FP8_BLOCK)) continue;
        if (found != null) return error.AmbiguousQkvTensorParallelism;
        found = @intCast(tp);
    }
    return found orelse error.InvalidQkvGeometry;
}

/// e4m3 codes `[tp * per, K]` with their 128x128 f32 tile scales, as bf16 `[tp, per, K]`.
pub fn dequantSlabs(codes: mlx.mlx_array, scales: mlx.mlx_array, tp: u32, per: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(codes);
    const k: c_int = sh[1];
    const kb = @divExact(k, FP8_BLOCK);
    const nb: c_int = @intCast((per + FP8_BLOCK - 1) / FP8_BLOCK);
    const pad: c_int = nb * FP8_BLOCK - @as(c_int, @intCast(per));
    const t: c_int = @intCast(tp);
    const sc_sh = mlx.getShape(scales);
    // A plain linear may carry spare scale rows past its last tile.
    if (sh.len != 2 or sc_sh.len != 2 or sc_sh[1] != kb or (if (tp == 1) sc_sh[0] < nb else sc_sh[0] != t * nb)) return error.InvalidFp8Scales;

    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_from_fp8(&f, codes, .float32, s));
    var slabs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slabs);
    try mlx.check(mlx.mlx_reshape(&slabs, f, &[_]c_int{ t, @intCast(per), k }, 3, s));
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    try mlx.check(mlx.mlx_pad(&padded, slabs, &[_]c_int{1}, 1, &[_]c_int{0}, 1, &[_]c_int{pad}, 1, zero, "constant", s));
    var tiles = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tiles);
    try mlx.check(mlx.mlx_reshape(&tiles, padded, &[_]c_int{ t, nb, FP8_BLOCK, kb, FP8_BLOCK }, 5, s));
    var used = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(used);
    try mlx.check(mlx.mlx_slice(&used, scales, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ t * nb, kb }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var grid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grid);
    try mlx.check(mlx.mlx_reshape(&grid, used, &[_]c_int{ t, nb, 1, kb, 1 }, 5, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, tiles, grid, s));
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_reshape(&rows, scaled, &[_]c_int{ t, nb * FP8_BLOCK, k }, 3, s));
    var kept = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kept);
    try mlx.check(mlx.mlx_slice(&kept, rows, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ t, @intCast(per), k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, kept, .bfloat16, s));
    return out;
}

/// Rows `[from, to)` of each slab, the slabs stacked: one of Q, K or V as `[tp * (to - from), K]`.
fn slabPart(slabs: mlx.mlx_array, tp: u32, from: u32, to: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const k = mlx.getShape(slabs)[2];
    const t: c_int = @intCast(tp);
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_slice(&part, slabs, &[_]c_int{ 0, @intCast(from), 0 }, 3, &[_]c_int{ t, @intCast(to), k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, part, &[_]c_int{ t * @as(c_int, @intCast(to - from)), k }, 2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, flat, false, s));
    return out;
}

fn takeKey(weights: *Weights, key: []const u8) ?mlx.mlx_array {
    const kv = weights.map.fetchRemove(key) orelse return null;
    weights.allocator.free(kv.key);
    return kv.value;
}

fn putOwned(weights: *Weights, key: []const u8, arr: mlx.mlx_array) !void {
    try weights.map.put(try weights.allocator.dupe(u8, key), arr);
}

/// MiMo's MTP heads (`model.mtp.layers.{k}`) as the trunk's loader leaves them, for a loader holding them apart:
/// `qkv` the q/k/v rows of a sliding layer.
pub fn adaptMimoHeads(weights: *Weights, qkv: [3]u64, s: mlx.mlx_stream) !void {
    var buf: [160]u8 = undefined;
    var k: u32 = 0;
    while (true) : (k += 1) {
        const base = try std.fmt.bufPrint(&buf, "model.mtp.layers.{d}", .{k});
        var probe: [200]u8 = undefined;
        if (weights.get(try std.fmt.bufPrint(&probe, "{s}.eh_proj.weight", .{base})) == null) break;
        try splitFusedQkv(weights, base, qkv, s);
        try decodeDenseMlp(weights, base, s);
    }
}

/// A dense e4m3 MLP decoded to bf16 once; the loader reads it as a plain projection.
pub fn decodeDenseMlp(weights: *Weights, base: []const u8, s: mlx.mlx_stream) !void {
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        var k1: [200]u8 = undefined;
        var k2: [200]u8 = undefined;
        const wk = try std.fmt.bufPrint(&k1, "{s}.mlp.{s}.weight", .{ base, proj });
        const sk = try std.fmt.bufPrint(&k2, "{s}.mlp.{s}.weight_scale_inv", .{ base, proj });
        const codes = takeKey(weights, wk) orelse continue;
        defer _ = mlx.mlx_array_free(codes);
        const scales = takeKey(weights, sk) orelse return error.MissingFp8Scale;
        defer _ = mlx.mlx_array_free(scales);
        const rows: u32 = @intCast(mlx.getShape(codes)[0]);
        const dense = try dequantSlabs(codes, scales, 1, rows, s);
        defer _ = mlx.mlx_array_free(dense);
        var flat = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, dense, &[_]c_int{ @intCast(rows), mlx.getShape(codes)[1] }, 2, s));
        try mlx.check(mlx.mlx_array_eval(flat));
        try putOwned(weights, wk, flat);
    }
}

/// A part of the fused QKV stays e4m3 as stored: its `.weight` holds the code rows and its `.scales`
/// one f32 scale row per code row (the source's tile scale, repeated), which is what `fp8Linear` reads.
pub fn splitFusedQkv(weights: *Weights, base: []const u8, rows_qkv: [3]u64, s: mlx.mlx_stream) !void {
    var k1: [200]u8 = undefined;
    var k2: [200]u8 = undefined;
    const wk = try std.fmt.bufPrint(&k1, "{s}.self_attn.qkv_proj.weight", .{base});
    const sk = try std.fmt.bufPrint(&k2, "{s}.self_attn.qkv_proj.weight_scale_inv", .{base});
    const codes = takeKey(weights, wk) orelse return;
    defer _ = mlx.mlx_array_free(codes);
    const scales = takeKey(weights, sk) orelse return error.MissingFp8Scale;
    defer _ = mlx.mlx_array_free(scales);

    const q, const k, const v = rows_qkv;
    const sh = mlx.getShape(codes);
    if (sh.len != 2 or @as(u64, @intCast(sh[0])) != q + k + v) return error.InvalidQkvGeometry;
    const tp = try solveQkvTp(q, k, v, @intCast(mlx.getShape(scales)[0]));
    const per: u32 = @intCast((q + k + v) / tp);

    var slabs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slabs);
    try mlx.check(mlx.mlx_reshape(&slabs, codes, &[_]c_int{ @intCast(tp), @intCast(per), sh[1] }, 3, s));
    const cuts = [_]u32{ 0, @intCast(q / tp), @intCast((q + k) / tp), per };
    inline for (.{ "q_proj", "k_proj", "v_proj" }, 0..) |name, n| {
        const rows = try slabPart(slabs, tp, cuts[n], cuts[n + 1], s);
        errdefer _ = mlx.mlx_array_free(rows);
        const row_scales = try rowScales(scales, tp, per, cuts[n], cuts[n + 1], s);
        errdefer _ = mlx.mlx_array_free(row_scales);
        try mlx.check(mlx.mlx_array_eval(rows));
        try mlx.check(mlx.mlx_array_eval(row_scales));
        var key: [200]u8 = undefined;
        try putOwned(weights, try std.fmt.bufPrint(&key, "{s}.self_attn.{s}.weight", .{ base, name }), rows);
        try putOwned(weights, try std.fmt.bufPrint(&key, "{s}.self_attn.{s}.scales", .{ base, name }), row_scales);
    }
}

/// The tile scale of every row in `[from, to)` of each slab, slabs stacked: `[tp * (to - from), K / 128]`.
fn rowScales(scales: mlx.mlx_array, tp: u32, per: u32, from: u32, to: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const nb = (per + FP8_BLOCK - 1) / FP8_BLOCK;
    const idx = try std.heap.page_allocator.alloc(i32, @as(usize, tp) * (to - from));
    defer std.heap.page_allocator.free(idx);
    var at: usize = 0;
    for (0..tp) |r| for (from..to) |row| {
        idx[at] = @intCast(r * nb + row / FP8_BLOCK);
        at += 1;
    };
    const ids = mlx.mlx_array_new_data(idx.ptr, &[_]c_int{@intCast(idx.len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(ids);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_take_axis(&out, scales, ids, 0, s));
    return out;
}

/// Row-scaled e4m3 (`.weight` u8 codes `[R, K]`, `.scales` f32 `[R, K / 128]`): neither affine's
/// bf16 scales nor MX's u8 ones. Only Sushi's MiMo trunk stores this pair; a new format that does
/// must get its own test here, or `qmatmul` routes it through `fp8Linear`.
pub fn isRowScaledFp8(w: mlx.mlx_array, sc: mlx.mlx_array) bool {
    if (w.ctx == null or sc.ctx == null) return false;
    return mlx.mlx_array_dtype(w) == .uint8 and mlx.mlx_array_dtype(sc) == .float32;
}

/// A row-scaled e4m3 weight decoded to bf16 `[R, K]`.
fn dequantRows(w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(w);
    const ssh = mlx.getShape(sc);
    if (sh.len != 2 or ssh.len != 2 or ssh[0] != sh[0] or ssh[1] * FP8_BLOCK != sh[1]) return error.InvalidFp8Scales;
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_from_fp8(&f, w, .bfloat16, s));
    var tiles = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tiles);
    try mlx.check(mlx.mlx_reshape(&tiles, f, &[_]c_int{ sh[0], ssh[1], FP8_BLOCK }, 3, s));
    var sb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sb);
    try mlx.check(mlx.mlx_astype(&sb, sc, .bfloat16, s));
    var grid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grid);
    try mlx.check(mlx.mlx_reshape(&grid, sb, &[_]c_int{ sh[0], ssh[1], 1 }, 3, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, tiles, grid, s));
    var dense = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_reshape(&dense, scaled, &[_]c_int{ sh[0], sh[1] }, 2, s));
    return dense;
}

/// `x @ W^T` for a row-scaled e4m3 weight. Decode and verify widths read the stored codes in
/// place (`gemv`); wider inputs decode to bf16 one projection at a time, so the resident bytes
/// stay the stored ones either way.
pub fn fp8Linear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (try gemv(x, w, sc, s)) |y| return y;
    const dense = try dequantRows(w, sc, s);
    defer _ = mlx.mlx_array_free(dense);
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, dense, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
    return out;
}

// e4m3 GEMV ported from sushi's fp8_block.zig (beamivalice/sushi, MIT): each simdgroup owns NR
// weight rows for all M input rows, lanes stride a row in 16-byte chunks, products sum in f32.
// The e4m3 decode is bit-exact: the code's low 7 bits as a half, times 256.
pub const GEMV_HEADER = kernelSource("kernels/quant/sushi_gemv_header.metal");
pub const GEMV_SOURCE = kernelSource("kernels/quant/sushi_gemv.metal");
const GEMV_MAX_ROWS = 8;
const GemvKey = struct { m: c_int, n: c_int, k: c_int, dt: mlx.mlx_dtype };
var gemv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemv_keys: [64]GemvKey = undefined;
var gemv_cfgs: [64]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var gemv_next: usize = 0;

fn gemvConfig(key: GemvKey) !mlx.mlx_fast_metal_kernel_config {
    for (gemv_cfgs, 0..) |c, i| if (c != null and std.meta.eql(gemv_keys[i], key)) return c.?;
    // Past four rows two weight rows per simdgroup share each x read.
    const nr: c_int = if (key.m <= 4) 1 else 2;
    const sgs: c_int = if (key.m <= 4) 2 else 8;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ key.m, key.n }, 2, key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * sgs * @divTrunc(key.n + nr * sgs - 1, nr * sgs), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * sgs, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", key.dt));
    inline for (.{ .{ "M", key.m }, .{ "N", key.n }, .{ "K", key.k }, .{ "NR", nr }, .{ "SGS", sgs } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    const slot = gemv_next % gemv_cfgs.len;
    if (gemv_cfgs[slot]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
    gemv_keys[slot] = key;
    gemv_cfgs[slot] = c;
    gemv_next += 1;
    return c;
}

/// `x @ W^T` over the stored codes for up to `GEMV_MAX_ROWS` input rows on the GPU; null otherwise.
fn gemv(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const wsh = mlx.getShape(w);
    const dt = mlx.mlx_array_dtype(x);
    if (!mlx.streamIsGpu(s) or xsh.len == 0 or wsh.len != 2 or xsh[xsh.len - 1] != wsh[1] or @rem(wsh[1], FP8_BLOCK) != 0) return null;
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    var m: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| m *= d;
    if (m < 1 or m > GEMV_MAX_ROWS) return null;
    if (gemv_kernel == null) {
        const ins = [_][*:0]const u8{ "x", "w", "scales" };
        const outs = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("msv_fp8_gemv", in_vec, out_vec, GEMV_SOURCE, GEMV_HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        gemv_kernel = k;
        log.info("[fp8] e4m3 GEMV engaged (up to {d} rows)\n", .{GEMV_MAX_ROWS});
    }
    const cfg = try gemvConfig(.{ .m = m, .n = wsh[0], .k = wsh[1], .dt = dt });
    const in_vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ x, w, sc }, 3);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, gemv_kernel.?, in_vec, cfg, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_vector_array_get(&flat, out_vec, 0));
    var lead: [8]c_int = undefined;
    if (xsh.len > lead.len) return null;
    @memcpy(lead[0..xsh.len], xsh);
    lead[xsh.len - 1] = wsh[0];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, flat, &lead, xsh.len, s));
    return out;
}

/// A tensor of a shard's header as the resident bill reads it.
pub const Stored = struct {
    key: []const u8,
    bytes: u64,
    dtype: []const u8,
    shape: []const std.json.Value,

    pub fn elements(t: Stored) u64 {
        var n: u64 = 1;
        for (t.shape) |d| n *= @intCast(d.integer);
        return n;
    }
};

/// What a load holds of the Sushi pack at `model_dir` once its family's adapter has run: every tensor of the shards
/// the index names, in the shard the index gives it, as `family.bill(tensor)` prices it.
pub fn residentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, family: anytype) !u64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var referenced = indexShardSet(io, dir);
    defer if (referenced) |*r| freeShardSet(r);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var owners = indexOwners(io, arena.allocator(), dir);
    var total: u64 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        var file = try dir.openFile(io, entry.name, .{});
        defer file.close(io);
        var rbuf: [8192]u8 = undefined;
        var rs = file.reader(io, &rbuf);
        const len = try rs.interface.takeInt(u64, .little);
        if (len > 64 * 1024 * 1024) return error.InvalidSafetensorsHeader;
        const raw = try rs.interface.readAlloc(allocator, @intCast(len));
        defer allocator.free(raw);
        total += try headerResidentBytes(allocator, raw, entry.name, if (owners) |*o| o else null, family);
    }
    return total;
}

/// One shard header's share of `residentBytes`.
pub fn headerResidentBytes(allocator: std.mem.Allocator, raw: []const u8, shard: []const u8, owners: ?*const Owners, family: anytype) !u64 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    var total: u64 = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__") or e.value_ptr.* != .object) continue;
        const key = e.key_ptr.*;
        if (owners) |o| if (o.get(key)) |owner| if (!std.mem.eql(u8, owner, shard)) continue;
        const offs = e.value_ptr.object.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
        if (offs != .array or offs.array.items.len != 2) return error.InvalidSafetensorsHeader;
        const shape = e.value_ptr.object.get("shape") orelse return error.InvalidSafetensorsHeader;
        if (shape != .array) return error.InvalidSafetensorsHeader;
        const dtype = if (e.value_ptr.object.get("dtype")) |d| (if (d == .string) d.string else "") else "";
        total += try family.bill(.{
            .key = key,
            .bytes = @intCast(offs.array.items[1].integer - offs.array.items[0].integer),
            .dtype = dtype,
            .shape = shape.array.items,
        });
    }
    return total;
}

/// `<prefix>.layers.N.<rest>` with N at or past the trunk's depth (an MTP layer the trunk never reads).
pub fn pastTrunk(key: []const u8, trunk_layers: u32) bool {
    const at = std.mem.indexOf(u8, key, ".layers.") orelse return false;
    const digits = key[at + ".layers.".len ..];
    const end = std.mem.indexOfScalar(u8, digits, '.') orelse return false;
    const n = std.fmt.parseInt(u32, digits[0..end], 10) catch return false;
    return n >= trunk_layers;
}

const testing = std.testing;

test "sushi pack: a tensor another shard owns is not billed" {
    const header =
        \\{"model.embed_tokens.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[0,700]},
        \\"lm_head.weight":{"dtype":"U32","shape":[1,1],"data_offsets":[700,800]}}
    ;
    const AsStored = struct {
        fn bill(_: @This(), t: Stored) !u64 {
            return t.bytes;
        }
    };
    var owners: Owners = .empty;
    defer owners.deinit(testing.allocator);
    try owners.put(testing.allocator, "model.embed_tokens.weight", "affine.safetensors");
    try testing.expectEqual(@as(u64, 100), try headerResidentBytes(testing.allocator, header, "source.safetensors", &owners, AsStored{}));
}

fn putNew(weights: *Weights, name: []const u8, arr: mlx.mlx_array) !void {
    try weights.map.put(try weights.allocator.dupe(u8, name), arr);
}

fn cpuStream() mlx.mlx_stream {
    return mlx.mlx_default_cpu_stream_new();
}

fn e4m3(code: u8) f32 {
    const sign: f32 = if (code & 0x80 != 0) -1 else 1;
    const e: i32 = (code >> 3) & 0xF;
    const m: f32 = @floatFromInt(code & 7);
    if (e == 0) return sign * m / 8 * std.math.pow(f32, 2, -6);
    return sign * (1 + m / 8) * std.math.pow(f32, 2, @floatFromInt(e - 7));
}

test "sushi pack: the rank-local QKV tensor-parallel count is solved from the scale grid" {
    // MiMo's sliding layers (116 scale rows) and global layers (108) are both TP 4.
    try testing.expectEqual(@as(u32, 4), try solveQkvTp(64 * 192, 8 * 192, 8 * 128, 116));
    try testing.expectEqual(@as(u32, 4), try solveQkvTp(64 * 192, 4 * 192, 4 * 128, 108));
    try testing.expectError(error.InvalidQkvGeometry, solveQkvTp(64 * 192, 4 * 192, 4 * 128, 109));
}

test "sushi pack: block-FP8 slabs dequantize with their own partial last tile" {
    const s = cpuStream();
    defer _ = mlx.mlx_stream_free(s);
    // 2 slabs of 130 rows (a full tile and a 2-row tile each), K = 256 (two column tiles).
    const tp = 2;
    const per = 130;
    const k = 256;
    var codes: [tp * per * k]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 37 + 11) % 0x78); // positive finite e4m3
    var scales: [tp * 2 * 2]f32 = undefined;
    for (&scales, 0..) |*sc, i| sc.* = 0.25 + @as(f32, @floatFromInt(i)) * 0.125;
    const c_arr = mlx.mlx_array_new_data(&codes, &[_]c_int{ tp * per, k }, 2, .uint8);
    defer _ = mlx.mlx_array_free(c_arr);
    const s_arr = mlx.mlx_array_new_data(&scales, &[_]c_int{ tp * 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(s_arr);
    const out = try dequantSlabs(c_arr, s_arr, tp, per, s);
    defer _ = mlx.mlx_array_free(out);
    try testing.expectEqualSlices(c_int, &.{ tp, per, k }, mlx.getShape(out));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, out, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const got = mlx.mlx_array_data_float32(f).?;
    for ([_][3]usize{ .{ 0, 0, 0 }, .{ 0, 127, 255 }, .{ 0, 128, 3 }, .{ 0, 129, 200 }, .{ 1, 5, 130 }, .{ 1, 129, 255 } }) |at| {
        const r = at[0] * per + at[1];
        const tile_row = at[0] * 2 + at[1] / 128;
        const want = e4m3(codes[r * k + at[2]]) * scales[tile_row * 2 + at[2] / 128];
        try testing.expectApproxEqRel(want, got[(at[0] * per + at[1]) * k + at[2]], 1e-2);
    }
}

test "sushi pack: a QKV part keeps its e4m3 rows and decodes as the slab dequantization does" {
    const s = cpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const tp = 2;
    const per = 130;
    const k = 256;
    var codes: [tp * per * k]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 29 + 5) % 0x78);
    var scales: [tp * 2 * 2]f32 = undefined;
    for (&scales, 0..) |*sc, i| sc.* = 0.25 + @as(f32, @floatFromInt(i)) * 0.125;
    const c_arr = mlx.mlx_array_new_data(&codes, &[_]c_int{ tp * per, k }, 2, .uint8);
    defer _ = mlx.mlx_array_free(c_arr);
    const s_arr = mlx.mlx_array_new_data(&scales, &[_]c_int{ tp * 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(s_arr);
    const ref = try dequantSlabs(c_arr, s_arr, tp, per, s);
    defer _ = mlx.mlx_array_free(ref);

    // Rows [100, 130) of each slab straddle the tile edge at 128.
    var slabs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slabs);
    try mlx.check(mlx.mlx_reshape(&slabs, c_arr, &[_]c_int{ tp, per, k }, 3, s));
    const part = try slabPart(slabs, tp, 100, 130, s);
    defer _ = mlx.mlx_array_free(part);
    const row_scales = try rowScales(s_arr, tp, per, 100, 130, s);
    defer _ = mlx.mlx_array_free(row_scales);
    try testing.expectEqualSlices(c_int, &.{ tp * 30, k }, mlx.getShape(part));
    try testing.expectEqualSlices(c_int, &.{ tp * 30, 2 }, mlx.getShape(row_scales));
    try testing.expect(isRowScaledFp8(part, row_scales));

    const got = try dequantRows(part, row_scales, s);
    defer _ = mlx.mlx_array_free(got);
    // The projection itself: x @ W^T over the decoded rows (GPU, as served), at both GEMV shapes.
    const gs = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gs);
    var dense_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense_f);
    try mlx.check(mlx.mlx_astype(&dense_f, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(dense_f));
    const dv = mlx.mlx_array_data_float32(dense_f).?;
    var xs: [5 * k]f32 = undefined;
    for (&xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.25;
    for ([_]c_int{ 2, 5 }) |rows| {
        const x32 = mlx.mlx_array_new_data(&xs, &[_]c_int{ 1, rows, k }, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var xb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xb);
        try mlx.check(mlx.mlx_astype(&xb, x32, .bfloat16, gs));
        const y = try fp8Linear(xb, part, row_scales, gs);
        defer _ = mlx.mlx_array_free(y);
        var yf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(yf);
        try mlx.check(mlx.mlx_astype(&yf, y, .float32, gs));
        try mlx.check(mlx.mlx_array_eval(yf));
        const yv = mlx.mlx_array_data_float32(yf).?;
        for (0..@intCast(rows)) |row| for (0..tp * 30) |r| {
            var acc: f64 = 0;
            var mag: f64 = 0;
            for (0..k) |c| {
                acc += @as(f64, xs[row * k + c]) * dv[r * k + c];
                mag += @abs(@as(f64, xs[row * k + c]) * dv[r * k + c]);
            }
            try testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), yv[row * tp * 30 + r], @as(f32, @floatCast(0.02 * mag + 1e-3)));
        };
    }

    var want_slab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want_slab);
    try mlx.check(mlx.mlx_slice(&want_slab, ref, &[_]c_int{ 0, 100, 0 }, 3, &[_]c_int{ tp, per, k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_reshape(&want, want_slab, &[_]c_int{ tp * 30, k }, 2, s));
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
    try mlx.check(mlx.mlx_astype(&wf, want, .float32, s));
    try mlx.check(mlx.mlx_array_eval(gf));
    try mlx.check(mlx.mlx_array_eval(wf));
    const g = mlx.mlx_array_data_float32(gf).?;
    const w = mlx.mlx_array_data_float32(wf).?;
    for (0..tp * 30 * k) |i| testing.expectApproxEqAbs(w[i], g[i], 0.02 * @abs(w[i]) + 1e-6) catch |e| {
        std.debug.print("at row {d} col {d}\n", .{ i / k, i % k });
        std.debug.print("want {any}\ngot {any}\n", .{ w[0..6], g[0..6] });
        std.debug.print("codes {any} {d}\n", .{ codes[100 * k .. 100 * k + 6], 100 * k });
        return e;
    };
}

test "sushi pack: hyper-connection tensors take the loader's names" {
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    for ([_][]const u8{ "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale", "input_layernorm.weight" }) |leaf| {
        var buf: [96]u8 = undefined;
        try putNew(&w, try std.fmt.bufPrint(&buf, "model.language_model.layers.3.{s}", .{leaf}), mlx.mlx_array_new_float(1));
    }
    try renameHyperConnections(&w);
    try testing.expectEqual(@as(u32, 7), w.count());
    for ([_][]const u8{ "attn_hc.fn", "attn_hc.base", "attn_hc.scale", "ffn_hc.fn", "ffn_hc.base", "ffn_hc.scale", "input_layernorm.weight" }) |name| {
        var buf: [96]u8 = undefined;
        try testing.expect(w.get(try std.fmt.bufPrint(&buf, "model.language_model.layers.3.{s}", .{name})) != null);
    }
}

test "sushi pack: kv_b_proj splits into the per-head halves the loader reads" {
    const s = cpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const d: KvB = .{ .heads = 2, .nope = 8, .v = 8, .latent = 64 };
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    var dense_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense_f);
    try mlx.check(mlx.mlx_random_normal(&dense_f, &[_]c_int{ 32, 64 }, 2, .float32, 0, 1, key, s));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_astype(&dense, dense_f, .bfloat16, s));

    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, dense, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(6), "affine", .{}, s));
    var q = [3]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    for (&q, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, parts, i));

    // Both halves are the dequantized rows of the stored matrix, head by head.
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_dequantize(&ref, q[0], q[1], q[2], mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(6), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
    var cube = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cube);
    try mlx.check(mlx.mlx_reshape(&cube, ref, &[_]c_int{ 2, 16, 64 }, 3, s));
    var want_v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want_v);
    try mlx.check(mlx.mlx_slice(&want_v, cube, &[_]c_int{ 0, 8, 0 }, 3, &[_]c_int{ 2, 16, 64 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var want_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want_k);
    try mlx.check(mlx.mlx_slice(&want_k, cube, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 2, 8, 64 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    try putNew(&w, "m.layers.0.self_attn.kv_b_proj.weight", q[0]);
    try putNew(&w, "m.layers.0.self_attn.kv_b_proj.scales", q[1]);
    try putNew(&w, "m.layers.0.self_attn.kv_b_proj.biases", q[2]);
    try splitKvB(&w, d, s);

    try testing.expectEqual(@as(u32, 2), w.count());
    const eq = w.get("m.layers.0.self_attn.embed_q.weight").?;
    const un = w.get("m.layers.0.self_attn.unembed_out.weight").?;
    try testing.expectEqualSlices(c_int, &.{ 2, 64, 8 }, mlx.getShape(eq));
    try testing.expectEqualSlices(c_int, &.{ 2, 8, 64 }, mlx.getShape(un));
    // The two halves are what `headerResidentBytes` bills for the split: rows x latent x 2 (bf16).
    try testing.expectEqual(@as(usize, d.heads * (d.nope + d.v) * d.latent * 2), (mlx.mlx_array_size(eq) + mlx.mlx_array_size(un)) * 2);

    var got_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_k);
    try mlx.check(mlx.mlx_swapaxes(&got_k, eq, 1, 2, s));
    try testing.expect(try allEqual(want_k, got_k, s));
    try testing.expect(try allEqual(want_v, un, s));
}

fn allEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_equal(&eq, a, b, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, eq, false, s));
    var out = false;
    try mlx.check(mlx.mlx_array_item_bool(&out, all));
    return out;
}
