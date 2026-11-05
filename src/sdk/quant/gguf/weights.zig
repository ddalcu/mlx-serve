//! GGUF tensors -> the name -> array map mlx-serve's Transformer binds from.
//! Quantized tensors keep their bytes (see kernels.Info for how they travel),
//! float tensors become bf16 like an mlx-community checkpoint.
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const gguf = @import("gguf.zig");
const kernels = @import("kernels.zig");
const affine = @import("affine.zig");
const arch_mod = @import("arch.zig");
const qwen35 = arch_mod.qwen35;
const Arch = arch_mod.Arch;

pub const WeightMap = std.StringHashMap(mlx.mlx_array);

/// Fills `out` (keys owned by `allocator`, the contract of mlx-serve's Weights).
pub fn load(allocator: std.mem.Allocator, f: *const gguf.File, out: *WeightMap, s: mlx.mlx_stream) !void {
    const arch = try Arch.read(f);
    // Only the GatedDeltaNet maps yield un-tiling transforms, the rest never look at this.
    const hp = arch.gdn();
    const mlx_layouts = @import("claim.zig").claimState().mlx_layouts;
    var name_buf: [160]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = arch.mapTensor(&name_buf, e.key_ptr.*) orelse return error.UnsupportedTensor;
        try loadTensor(allocator, f, e.key_ptr.*, e.value_ptr.*, mapped, arch, hp, mlx_layouts, out, s);
    }
    if (arch == .qwen4exp) for (0..arch.qwen4exp.base.n_layers) |li| {
        var lb: [64]u8 = undefined;
        const layer = std.fmt.bufPrint(&lb, "language_model.model.layers.{d}.", .{li}) catch unreachable;
        try joinIndexer(allocator, out, layer, s);
        try quantizeGdnGates(allocator, out, layer, s);
    };
    if (mlx_layouts) switch (arch) {
        .qwen4exp => try narrowALog(out, ".linear_attn.A_log", s),
        .nemotron_h => try narrowALog(out, ".mixer.A_log", s),
        else => {},
    };
}

/// One tensor of `f` under its mapped name, in the layout `arch` serves it in (nothing for a skipped one).
fn loadTensor(allocator: std.mem.Allocator, f: *const gguf.File, key: []const u8, tensor: gguf.Tensor, mapped: arch_mod.Mapped, arch: Arch, hp: qwen35.Hparams, mlx_layouts: bool, out: *WeightMap, s: mlx.mlx_stream) !void {
    if (mapped.name.len == 0) return;
    var t = tensor;
    const uncached = try f.readUncached(allocator, key);
    defer if (uncached) |r| allocator.free(r.buf);
    if (uncached) |r| t.data = r.data;
    if (t.ty.isQuantized()) {
        if (mlx_layouts and arch.transcodes(t.ty) and try (if (t.ty == .mxfp4) loadMxfp4(allocator, t, mapped, out) else loadAffine(allocator, t, mapped, hp, out))) return;
        try loadQuantized(allocator, t, mapped, hp, out, s);
    } else {
        try put(allocator, out, mapped.name, try loadFloat(allocator, t, mapped.transform, hp, s));
    }
}

/// A DeepSeek-V4.1 set's DSpark stage bundle (`dspark-*.gguf`, arch/deepseek41.zig `mapDsparkTensor`) into `out`
/// beside the trunk `hp`, each tensor in the layout the trunk's own would take.
pub fn loadDspark(allocator: std.mem.Allocator, f: *const gguf.File, hp: arch_mod.deepseek41.Hparams, out: *WeightMap, s: mlx.mlx_stream) !void {
    const ds = try arch_mod.deepseek41.Dspark.read(f, hp);
    const arch: Arch = .{ .deepseek41 = hp };
    const mlx_layouts = @import("claim.zig").claimState().mlx_layouts;
    var name_buf: [160]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = arch_mod.deepseek41.mapDsparkTensor(&name_buf, e.key_ptr.*, ds) orelse return error.UnsupportedTensor;
        try loadTensor(allocator, f, e.key_ptr.*, e.value_ptr.*, mapped, arch, arch.gdn(), mlx_layouts, out, s);
    }
}

/// `residentBytes` of a DSpark stage bundle (`loadDspark`).
pub fn dsparkResidentBytes(f: *const gguf.File, hp: arch_mod.deepseek41.Hparams) !u64 {
    const ds = try arch_mod.deepseek41.Dspark.read(f, hp);
    const arch: Arch = .{ .deepseek41 = hp };
    var name_buf: [160]u8 = undefined;
    var total: u64 = 0;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = arch_mod.deepseek41.mapDsparkTensor(&name_buf, e.key_ptr.*, ds) orelse return error.UnsupportedTensor;
        if (mapped.name.len != 0) total += loadedBytes(e.value_ptr.*, arch.transcodes(e.value_ptr.ty));
    }
    return total;
}

/// A DeepSeek-V4.1 set's vision tower GGUF (arch/deepseek41.zig `Vision`, `mapVisionTensor`) into `out` beside the
/// trunk `hp`, its floats as stored.
pub fn loadVision(allocator: std.mem.Allocator, f: *const gguf.File, hp: arch_mod.deepseek41.Hparams, out: *WeightMap, s: mlx.mlx_stream) !void {
    const vis = try arch_mod.deepseek41.Vision.read(f, hp, null);
    const arch: Arch = .{ .deepseek41 = hp };
    var buf: [96]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = arch_mod.deepseek41.mapVisionTensor(&buf, e.key_ptr.*) orelse return error.UnsupportedTensor;
        try loadTensor(allocator, f, e.key_ptr.*, e.value_ptr.*, mapped, arch, arch.gdn(), true, out, s);
    }
    try joinMmproj(allocator, out, vis.layers, s);
}

/// An mmproj's tower in the release's layout: each block's gate and up as one `mlp.w1` (gate rows, then up), and the
/// patch conv `[out, 3, p, p]` as the `[out, 3*p*p]` linear the tower reads (channel, row, column: its pixel order).
fn joinMmproj(allocator: std.mem.Allocator, out: *WeightMap, layers: u32, s: mlx.mlx_stream) !void {
    var nb: [96]u8 = undefined;
    for (0..layers) |i| {
        const g = out.fetchRemove(std.fmt.bufPrint(&nb, "vision.blocks.{d}.mlp.w1_gate.weight", .{i}) catch unreachable) orelse continue;
        defer allocator.free(g.key);
        defer _ = mlx.mlx_array_free(g.value);
        const u = out.fetchRemove(std.fmt.bufPrint(&nb, "vision.blocks.{d}.mlp.w1_up.weight", .{i}) catch unreachable) orelse return error.UnsupportedTensor;
        defer allocator.free(u.key);
        defer _ = mlx.mlx_array_free(u.value);
        const parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        _ = mlx.mlx_vector_array_append_value(parts, g.value);
        _ = mlx.mlx_vector_array_append_value(parts, u.value);
        var joined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&joined, parts, 0, s));
        try mlx.check(mlx.mlx_array_eval(joined));
        try put(allocator, out, std.fmt.bufPrint(&nb, "vision.blocks.{d}.mlp.w1.weight", .{i}) catch unreachable, joined);
    }
    const patch = out.getPtr("vision.patch_embed.proj.weight") orelse return;
    if (mlx.mlx_array_ndim(patch.*) != 4) return;
    const rows: c_int = mlx.mlx_array_shape(patch.*)[0];
    var flat = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&flat, patch.*, &[_]c_int{ rows, -1 }, 2, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    _ = mlx.mlx_array_free(patch.*);
    patch.* = flat;
}

/// `residentBytes` of a vision tower GGUF (`loadVision`).
pub fn visionResidentBytes(f: *const gguf.File, hp: arch_mod.deepseek41.Hparams) !u64 {
    _ = try arch_mod.deepseek41.Vision.read(f, hp, null);
    var total: u64 = 0;
    var buf: [96]u8 = undefined;
    for (f.tensors.keys(), f.tensors.values()) |name, t| {
        if (arch_mod.deepseek41.mapVisionTensor(&buf, name) == null) return error.UnsupportedTensor;
        total += loadedBytes(t, false);
    }
    return total;
}

/// Every `*<suffix>` A_log in bf16, as our qwen4_exp packs and mlx-lm's Nemotron-H packs ship it: the fused GDN decode
/// and Mamba2 step kernels read it in the activation dtype and decline an f32 one (Mamba2 then scans row by row).
fn narrowALog(out: *WeightMap, suffix: []const u8, s: mlx.mlx_stream) !void {
    var it = out.iterator();
    while (it.next()) |e| {
        if (!std.mem.endsWith(u8, e.key_ptr.*, suffix) or mlx.mlx_array_dtype(e.value_ptr.*) == .bfloat16) continue;
        var cast = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cast);
        try mlx.check(mlx.mlx_astype(&cast, e.value_ptr.*, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(cast));
        _ = mlx.mlx_array_free(e.value_ptr.*);
        e.value_ptr.* = cast;
    }
}

/// A GatedDeltaNet layer's a and b projections (dense in the file, 48 rows) quantized to the affine width its qkv
/// and z projections share, so the four join one row group as in our packs (the group's fused GDN arms need it).
fn quantizeGdnGates(allocator: std.mem.Allocator, out: *WeightMap, layer: []const u8, s: mlx.mlx_stream) !void {
    var nb: [160]u8 = undefined;
    const qkv_w = out.get(std.fmt.bufPrint(&nb, "{s}linear_attn.in_proj_qkv.weight", .{layer}) catch unreachable) orelse return;
    const qkv_s = out.get(std.fmt.bufPrint(&nb, "{s}linear_attn.in_proj_qkv.scales", .{layer}) catch unreachable) orelse return;
    if (mlx.mlx_array_dtype(qkv_s) != .bfloat16 or mlx.mlx_array_dtype(qkv_w) != .uint32) return;
    // The join needs z at qkv's layout; below 4 bits the gates would lose more than the join saves.
    const z_w = out.get(std.fmt.bufPrint(&nb, "{s}linear_attn.in_proj_z.weight", .{layer}) catch unreachable) orelse return;
    const z_s = out.get(std.fmt.bufPrint(&nb, "{s}linear_attn.in_proj_z.scales", .{layer}) catch unreachable) orelse return;
    if (mlx.getShape(z_w)[1] != mlx.getShape(qkv_w)[1] or mlx.getShape(z_s)[1] != mlx.getShape(qkv_s)[1] or mlx.mlx_array_dtype(z_s) != .bfloat16) return;
    const a_w = out.get(std.fmt.bufPrint(&nb, "{s}linear_attn.in_proj_a.weight", .{layer}) catch unreachable) orelse return;
    // A raw or quantized gate keeps its layout: its row holds bytes or codes, not its input width.
    switch (mlx.mlx_array_dtype(a_w)) {
        .float32, .float16, .bfloat16 => {},
        else => return,
    }
    if (@divExact(mlx.getShape(qkv_w)[1] * 32, mlx.getShape(a_w)[1]) < 4) return;
    for ([_][]const u8{ "in_proj_a", "in_proj_b" }) |part| {
        if (out.get(std.fmt.bufPrint(&nb, "{s}linear_attn.{s}.scales", .{ layer, part }) catch unreachable) != null) continue;
        const dense = out.fetchRemove(std.fmt.bufPrint(&nb, "{s}linear_attn.{s}.weight", .{ layer, part }) catch unreachable) orelse continue;
        defer allocator.free(dense.key);
        defer _ = mlx.mlx_array_free(dense.value);
        const in = mlx.getShape(dense.value)[1];
        const gs = @divExact(in, mlx.getShape(qkv_s)[1]);
        const bits = @divExact(mlx.getShape(qkv_w)[1] * 32, in);
        var q = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(q);
        try mlx.check(mlx.mlx_quantize(&q, dense.value, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, i| {
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&arr, q, i));
            try mlx.check(mlx.mlx_array_eval(arr));
            try put(allocator, out, std.fmt.bufPrint(&nb, "{s}linear_attn.{s}.{s}", .{ layer, part, leaf }) catch unreachable, arr);
        }
    }
}

/// A qwen4exp MTP sidecar's head (`mtp-*.gguf`, arch/qwen4exp.zig `mapMtpTensor`) into `out` beside the trunk.
pub fn loadMtp(allocator: std.mem.Allocator, f: *const gguf.File, out: *WeightMap, s: mlx.mlx_stream) !void {
    const q4 = arch_mod.qwen4exp;
    if (!std.mem.eql(u8, f.getString("general.architecture") orelse "", q4.arch_name)) return error.UnsupportedArch;
    if ((f.getInt(q4.arch_name ++ ".nextn_predict_layers") orelse 0) != 1) return error.BadMetadata;
    const base = try qwen35.Hparams.read(f, q4.arch_name);
    var name_buf: [160]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = q4.mapMtpTensor(&name_buf, e.key_ptr.*, base) orelse return error.UnsupportedTensor;
        if (mapped.name.len == 0) continue;
        const t = e.value_ptr.*;
        if (std.mem.endsWith(u8, mapped.name, "block_inject_weight.weight") and t.ty.isQuantized()) {
            // The trunk's injects are dense: the head's quantized ones join them as bf16.
            try put(allocator, out, mapped.name, try dequantized(allocator, t));
        } else if (t.ty.isQuantized()) {
            if (try loadAffine(allocator, t, mapped, base, out)) {
                if (headBits(mapped.name, t)) |bits| try requantHead(out, mapped.name, t, bits, s);
                continue;
            }
            try loadQuantized(allocator, t, mapped, base, out, s);
        } else try put(allocator, out, mapped.name, try loadFloat(allocator, t, mapped.transform, base, s));
    }
    try joinIndexer(allocator, out, q4.mtp_prefix ++ "layers.0.", s);
    try splitEhProj(allocator, out, s);
}

/// A qwen35 / qwen35moe file's own MTP block (`nextn_predict_layers` = 1, arch/qwen35.zig `mapMtpTensor`) into `out`
/// as the generic MTP loader's `mtp.*` weights, nothing when the file has none. bf16 (the loader requantizes the
/// head's linears itself), except what the qwen3_5 head binds only quantized: the expert banks (affine 4 bits) and
/// the shared-expert gate (8 bits), group 64.
pub fn loadQwen35Mtp(allocator: std.mem.Allocator, f: *const gguf.File, out: *WeightMap, s: mlx.mlx_stream) !void {
    const arch = Arch.read(f) catch return;
    if (arch != .qwen35) return;
    var name_buf: [160]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = qwen35.mapMtpTensor(&name_buf, e.key_ptr.*, arch.qwen35) orelse continue;
        if (mapped.name.len == 0) continue;
        const t = e.value_ptr.*;
        const dense = if (t.ty.isQuantized()) try dequantized(allocator, t) else try loadFloat(allocator, t, mapped.transform, arch.qwen35, s);
        const bits: ?u32 = if (std.mem.indexOf(u8, mapped.name, ".switch_mlp.") != null) 4 else if (std.mem.endsWith(u8, mapped.name, "shared_expert_gate.weight")) 8 else null;
        if (bits) |b| {
            defer _ = mlx.mlx_array_free(dense);
            try quantizeInto(allocator, out, mapped.name, dense, b, s);
        } else try put(allocator, out, mapped.name, dense);
    }
}

/// `dense` as MLX affine `bits` / group 64 under `name`'s `.weight` / `.scales` / `.biases`.
fn quantizeInto(allocator: std.mem.Allocator, out: *WeightMap, name: []const u8, dense: mlx.mlx_array, bits: u32, s: mlx.mlx_stream) !void {
    const stem = name[0 .. name.len - "weight".len];
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
    for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, i| {
        var arr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&arr, q, i));
        try mlx.check(mlx.mlx_array_eval(arr));
        var nb: [160]u8 = undefined;
        try put(allocator, out, std.fmt.bufPrint(&nb, "{s}{s}", .{ stem, leaf }) catch return error.NameTooLong, arr);
    }
}

/// Bytes of a qwen35 file's own MTP block as the GGUF stores it (`loadQwen35Mtp` lands it near that), 0 without one.
pub fn qwen35MtpBytes(f: *const gguf.File) u64 {
    const arch = Arch.read(f) catch return 0;
    if (arch != .qwen35) return 0;
    var name_buf: [160]u8 = undefined;
    var total: u64 = 0;
    var it = f.tensors.iterator();
    while (it.next()) |e| if (qwen35.mapMtpTensor(&name_buf, e.key_ptr.*, arch.qwen35) != null) {
        total += e.value_ptr.data.len;
    };
    return total;
}

/// The head only proposes (the trunk verifies every token), so it loads in our qwen4 packs' head layout and rides
/// their kernels: routed experts at 4 bits, every other affine projection at 8, group 64. The width `t` (loaded by
/// `loadAffine`) is requantized to, null when it already is that layout or narrower at group 64.
fn headBits(name: []const u8, t: gguf.Tensor) ?u32 {
    const from = affine.bitsOf(t.ty) orelse return null;
    if (t.dims[0] % 64 != 0) return null;
    const bits: u32 = if (std.mem.indexOf(u8, name, ".switch_mlp.") != null) 4 else 8;
    return if (affine.groupOf(t.ty) == 64 and from <= bits) null else bits;
}

fn requantHead(out: *WeightMap, name: []const u8, t: gguf.Tensor, bits: u32, s: mlx.mlx_stream) !void {
    const stem = name[0 .. name.len - "weight".len];
    var nb: [3][160]u8 = undefined;
    var slots: [3]*mlx.mlx_array = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }, &slots, &nb) |leaf, *slot, *buf|
        slot.* = out.getPtr(std.fmt.bufPrint(buf, "{s}{s}", .{ stem, leaf }) catch return error.NameTooLong) orelse return error.MissingWeight;
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    const from_g: c_int = @intCast(affine.groupOf(t.ty));
    const from_b: c_int = @intCast(affine.bitsOf(t.ty).?);
    try mlx.check(mlx.mlx_dequantize(&dense, slots[0].*, slots[1].*, slots[2].*, mlx.mlx_optional_int.some(from_g), mlx.mlx_optional_int.some(from_b), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
    for (slots, 0..) |slot, i| {
        var arr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&arr, q, i));
        try mlx.check(mlx.mlx_array_eval(arr));
        _ = mlx.mlx_array_free(slot.*);
        slot.* = arr;
    }
}

/// A small quantized tensor as bf16 (`[rows, in]`, an expert bank `[experts, rows, in]`), through the reference dequant.
fn dequantized(allocator: std.mem.Allocator, t: gguf.Tensor) !mlx.mlx_array {
    const n: usize = @intCast(t.elems());
    const f = try allocator.alloc(f32, n);
    defer allocator.free(f);
    try @import("quants.zig").dequantize(t.ty, t.data, f);
    const h = try allocator.alloc(u16, n);
    defer allocator.free(h);
    for (f, h) |v, *o| o.* = affine.bf16(v);
    var shape: [4]c_int = undefined;
    for (0..t.n_dims) |i| shape[i] = @intCast(t.dims[t.n_dims - 1 - i]);
    return mlx.mlx_array_new_data(h.ptr, &shape, @intCast(t.n_dims), .bfloat16);
}

/// The head's `[fc_embedding | fc_hidden]` combiner as our packs' two projections, its input split in half,
/// embedding first (the converter's `cat([fc_embedding, fc_hidden], dim=1)`). Affine layouts only.
fn splitEhProj(allocator: std.mem.Allocator, out: *WeightMap, s: mlx.mlx_stream) !void {
    const q4 = arch_mod.qwen4exp;
    const stem = q4.mtp_eh_proj[0 .. q4.mtp_eh_proj.len - "weight".len];
    if (out.get(stem ++ "scales")) |sc| if (mlx.mlx_array_dtype(sc) == .bool_) return error.UnsupportedTensor;
    for ([_][]const u8{ "weight", "scales", "biases" }) |leaf| {
        var nb: [96]u8 = undefined;
        const whole = out.fetchRemove(std.fmt.bufPrint(&nb, "{s}{s}", .{ stem, leaf }) catch unreachable) orelse return error.UnsupportedTensor;
        defer allocator.free(whole.key);
        defer _ = mlx.mlx_array_free(whole.value);
        const sh = mlx.getShape(whole.value);
        if (sh.len != 2 or @rem(sh[1], 2) != 0) return error.BadShape;
        const half = @divExact(sh[1], 2);
        for ([_][]const u8{ "fc_embedding", "fc_hidden" }, 0..) |part, i| {
            const lo: c_int = @intCast(i * @as(usize, @intCast(half)));
            var view = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(view);
            try mlx.check(mlx.mlx_slice(&view, whole.value, &[_]c_int{ 0, lo }, 2, &[_]c_int{ sh[0], lo + half }, 2, &[_]c_int{ 1, 1 }, 2, s));
            const copy = try @import("../../../mlx/array.zig").materializedOwnedCopy(s, view);
            try mlx.check(mlx.mlx_array_eval(copy));
            var pb: [96]u8 = undefined;
            try put(allocator, out, std.fmt.bufPrint(&pb, q4.mtp_prefix ++ "{s}.{s}", .{ part, leaf }) catch unreachable, copy);
        }
    }
}

/// One full layer's indexer q and k rows back into the one `index_qk_proj` our qwen4_exp packs carry, q first.
/// `layer` is the layer's name prefix; a layer without an indexer is left alone.
fn joinIndexer(allocator: std.mem.Allocator, out: *WeightMap, layer: []const u8, s: mlx.mlx_stream) !void {
    const q4 = arch_mod.qwen4exp;
    var qb: [160]u8 = undefined;
    var kb: [160]u8 = undefined;
    const q = out.fetchRemove(std.fmt.bufPrint(&qb, "{s}" ++ q4.idx_q, .{layer}) catch unreachable) orelse return;
    defer allocator.free(q.key);
    defer _ = mlx.mlx_array_free(q.value);
    const k = out.fetchRemove(std.fmt.bufPrint(&kb, "{s}" ++ q4.idx_k, .{layer}) catch unreachable) orelse return error.UnsupportedTensor;
    defer allocator.free(k.key);
    defer _ = mlx.mlx_array_free(k.value);
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    _ = mlx.mlx_vector_array_append_value(parts, q.value);
    _ = mlx.mlx_vector_array_append_value(parts, k.value);
    var joined = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&joined, parts, 0, s));
    try mlx.check(mlx.mlx_array_eval(joined));
    var nb: [160]u8 = undefined;
    try put(allocator, out, std.fmt.bufPrint(&nb, "{s}self_attn.indexer.index_qk_proj.weight", .{layer}) catch unreachable, joined);
}

/// Bytes `load` copies into MLX arrays (an upper bound: f32 tensors land as bf16); the tensors the arch skips stay in the file.
pub fn residentBytes(f: *const gguf.File) !u64 {
    const arch = try Arch.read(f);
    var name_buf: [160]u8 = undefined;
    var total: u64 = 0;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = arch.mapTensor(&name_buf, e.key_ptr.*) orelse return error.UnsupportedTensor;
        if (mapped.name.len != 0) total += loadedBytes(e.value_ptr.*, arch.transcodes(e.value_ptr.ty));
    }
    return total;
}

/// `residentBytes` of an MTP sidecar (`loadMtp`).
pub fn mtpResidentBytes(f: *const gguf.File) !u64 {
    const base = try qwen35.Hparams.read(f, arch_mod.qwen4exp.arch_name);
    var name_buf: [160]u8 = undefined;
    var total: u64 = 0;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = arch_mod.qwen4exp.mapMtpTensor(&name_buf, e.key_ptr.*, base) orelse return error.UnsupportedTensor;
        if (mapped.name.len == 0) continue;
        const t = e.value_ptr.*;
        total += if (headBits(mapped.name, t)) |bits| t.elems() * bits / 8 + t.elems() / 64 * 4 else loadedBytes(t, affine.bitsOf(t.ty) != null);
    }
    return total;
}

/// A tensor's bytes as loaded: an affine copy is the codes plus a bf16 scale and bias per group.
fn loadedBytes(t: gguf.Tensor, transcoded: bool) u64 {
    const bits = if (transcoded) affine.bitsOf(t.ty) else null;
    return if (bits) |b| t.elems() * b / 8 + t.elems() / affine.groupOf(t.ty) * 4 else t.data.len;
}

/// A tensor of an affine-coded type (affine.zig) in MLX's layout: `<base>.weight` u32 codes, `.scales` and `.biases`
/// bf16, one per `affine.groupOf` weights. False when its type or transform keeps the ggml blocks.
fn loadAffine(allocator: std.mem.Allocator, t: gguf.Tensor, mapped: arch_mod.Mapped, hp: qwen35.Hparams, out: *WeightMap) !bool {
    const bits = affine.bitsOf(t.ty) orelse return false;
    if (!std.mem.endsWith(u8, mapped.name, ".weight") or t.n_dims < 2 or t.n_dims > 3) return false;
    const in: usize = @intCast(t.dims[0]);
    const rows: usize = @intCast(t.elems() / t.dims[0]);
    const row_bytes = in / t.ty.blockElems() * t.ty.blockBytes();
    var src = t.data;
    var untiled: []u8 = &.{};
    defer allocator.free(untiled);
    switch (mapped.transform) {
        .none => {},
        .untile_rows => |u| {
            // Rows are whole blocks, so moving a row moves its bytes and nothing else.
            if (t.n_dims != 2 or u.start + hp.nv * u.unit != rows) return error.BadShape;
            untiled = try allocator.alloc(u8, t.data.len);
            untile(untiled, t.data, u.start * row_bytes, u.unit * row_bytes, hp);
            src = untiled;
        },
        // Value heads of whole groups move after the decode.
        .tiled_input => if (t.n_dims != 2 or hp.dv % affine.groupOf(t.ty) != 0 or in != hp.nv * hp.dv) return false,
        else => return false,
    }
    const job: AffineJob = .{
        .ty = t.ty,
        .bits = bits,
        .in = in,
        .row_bytes = row_bytes,
        .src = src,
        .tiled = mapped.transform == .tiled_input,
        .hp = hp,
        .w = try allocator.alloc(u32, rows * in * bits / 32),
        .sc = try allocator.alloc(u16, rows * in / affine.groupOf(t.ty)),
        .bi = try allocator.alloc(u16, rows * in / affine.groupOf(t.ty)),
    };
    defer allocator.free(job.w);
    defer allocator.free(job.sc);
    defer allocator.free(job.bi);
    // Large tensors (expert banks) decode on several threads, rows split evenly.
    const n_threads: usize = if (rows * in < 1 << 22) 1 else @min(16, std.Thread.getCpuCount() catch 1);
    var threads: [16]?std.Thread = @splat(null);
    const per = (rows + n_threads - 1) / n_threads;
    var ok = std.atomic.Value(bool).init(true);
    for (0..n_threads) |i| {
        const lo = @min(rows, i * per);
        const hi = @min(rows, lo + per);
        if (i + 1 == n_threads) {
            job.run(lo, hi, &ok);
        } else threads[i] = std.Thread.spawn(.{}, AffineJob.run, .{ &job, lo, hi, &ok }) catch blk: {
            job.run(lo, hi, &ok);
            break :blk null;
        };
    }
    for (threads) |th| if (th) |x| x.join();
    if (!ok.load(.acquire)) return error.OutOfMemory;

    var shape: [3]c_int = undefined;
    const nd: usize = t.n_dims;
    for (0..nd) |i| shape[i] = @intCast(t.dims[nd - 1 - i]);
    const base = mapped.name[0 .. mapped.name.len - "weight".len];
    var nb: [160]u8 = undefined;
    shape[nd - 1] = @intCast(in * bits / 32);
    try put(allocator, out, mapped.name, mlx.mlx_array_new_data(job.w.ptr, &shape, @intCast(nd), .uint32));
    shape[nd - 1] = @intCast(in / affine.groupOf(t.ty));
    try put(allocator, out, std.fmt.bufPrint(&nb, "{s}scales", .{base}) catch return error.NameTooLong, mlx.mlx_array_new_data(job.sc.ptr, &shape, @intCast(nd), .bfloat16));
    try put(allocator, out, std.fmt.bufPrint(&nb, "{s}biases", .{base}) catch return error.NameTooLong, mlx.mlx_array_new_data(job.bi.ptr, &shape, @intCast(nd), .bfloat16));
    return true;
}

/// An MXFP4 tensor in MLX's mxfp4 layout, losslessly: a ggml block is MLX's group (one E8M0 scale, 32 e2m1 codes), but
/// ggml keeps elements 0-15 in the low nibbles and 16-31 in the high ones where MLX packs them densely.
fn loadMxfp4(allocator: std.mem.Allocator, t: gguf.Tensor, mapped: arch_mod.Mapped, out: *WeightMap) !bool {
    if (t.ty != .mxfp4 or mapped.transform != .none or !std.mem.endsWith(u8, mapped.name, ".weight") or t.n_dims < 2 or t.n_dims > 3) return false;
    const in: usize = @intCast(t.dims[0]);
    const n_blocks: usize = @intCast(t.elems() / 32);
    const w = try allocator.alloc(u32, n_blocks * 4);
    defer allocator.free(w);
    const sc = try allocator.alloc(u8, n_blocks);
    defer allocator.free(sc);
    const codes = std.mem.sliceAsBytes(w);
    for (0..n_blocks) |b| {
        const blk = t.data[b * 17 ..][0..17];
        sc[b] = blk[0];
        for (0..8) |k| {
            codes[b * 16 + k] = (blk[1 + 2 * k] & 0xF) | (blk[2 + 2 * k] & 0xF) << 4;
            codes[b * 16 + 8 + k] = blk[1 + 2 * k] >> 4 | (blk[2 + 2 * k] >> 4) << 4;
        }
    }
    var shape: [3]c_int = undefined;
    const nd: usize = t.n_dims;
    for (0..nd) |i| shape[i] = @intCast(t.dims[nd - 1 - i]);
    shape[nd - 1] = @intCast(in / 8);
    try put(allocator, out, mapped.name, mlx.mlx_array_new_data(w.ptr, &shape, @intCast(nd), .uint32));
    shape[nd - 1] = @intCast(in / 32);
    var nb: [160]u8 = undefined;
    const base = mapped.name[0 .. mapped.name.len - "weight".len];
    try put(allocator, out, std.fmt.bufPrint(&nb, "{s}scales", .{base}) catch return error.NameTooLong, mlx.mlx_array_new_data(sc.ptr, &shape, @intCast(nd), .uint8));
    return true;
}

const AffineJob = struct {
    ty: @import("quants.zig").GgmlType,
    bits: u32,
    in: usize,
    row_bytes: usize,
    src: []const u8,
    /// The input's value heads are in llama.cpp's tiled order (arch/qwen35.zig): put them back grouped.
    tiled: bool,
    hp: qwen35.Hparams,
    w: []u32,
    sc: []u16,
    bi: []u16,

    fn run(self: *const AffineJob, lo: usize, hi: usize, ok: *std.atomic.Value(bool)) void {
        const a = std.heap.page_allocator;
        const n_groups = self.in / affine.groupOf(self.ty);
        const buf = a.alloc(u8, 2 * self.in) catch return ok.store(false, .release);
        defer a.free(buf);
        const fbuf = a.alloc(f32, 4 * n_groups) catch return ok.store(false, .release);
        defer a.free(fbuf);
        const words = self.in * self.bits / 32;
        for (lo..hi) |r| {
            var q = buf[0..self.in];
            var scale = fbuf[0..n_groups];
            var bias = fbuf[n_groups .. 2 * n_groups];
            affine.decodeRow(self.ty, self.src[r * self.row_bytes ..][0..self.row_bytes], q, scale, bias);
            if (self.tiled) {
                const dv = self.hp.dv;
                const gpv = dv / affine.groupOf(self.ty);
                const q2 = buf[self.in..];
                const s2 = fbuf[2 * n_groups .. 3 * n_groups];
                const b2 = fbuf[3 * n_groups ..];
                for (0..self.hp.nv) |tiled| {
                    const h = qwen35.groupedIndex(@intCast(tiled), self.hp.nk, self.hp.r());
                    @memcpy(q2[h * dv ..][0..dv], q[tiled * dv ..][0..dv]);
                    @memcpy(s2[h * gpv ..][0..gpv], scale[tiled * gpv ..][0..gpv]);
                    @memcpy(b2[h * gpv ..][0..gpv], bias[tiled * gpv ..][0..gpv]);
                }
                q = q2;
                scale = s2;
                bias = b2;
            }
            affine.packCodes(self.bits, q, self.w[r * words ..][0..words]);
            for (scale, bias, self.sc[r * n_groups ..][0..n_groups], self.bi[r * n_groups ..][0..n_groups]) |sv, bv, *so, *bo| {
                so.* = affine.bf16(sv);
                bo.* = affine.bf16(bv);
            }
        }
    }
};

/// Runs the decode and the prefill kernel of every distinct (type, shape) in
/// `map` once, so Metal compiles them during load and not inside the first
/// request (a few hundred ms there).
pub fn warmKernels(allocator: std.mem.Allocator, map: *const WeightMap, s: mlx.mlx_stream) !void {
    var seen = std.AutoHashMap([4]c_int, void).init(allocator);
    defer seen.deinit();
    var it = map.iterator();
    while (it.next()) |e| {
        if (!std.mem.endsWith(u8, e.key_ptr.*, ".weight")) continue;
        var sc_name: [160]u8 = undefined;
        const base = e.key_ptr.*[0 .. e.key_ptr.len - "weight".len];
        const sc = map.get(std.fmt.bufPrint(&sc_name, "{s}scales", .{base}) catch continue) orelse continue;
        const w = e.value_ptr.*;
        const info = kernels.infoOf(w, sc) orelse continue;
        const ws = mlx.getShape(w);
        const bank = ws.len == 3;
        const key = [4]c_int{ @intCast(@backingInt(info.ty)), @intFromBool(bank), ws[ws.len - 2], ws[ws.len - 1] };
        if ((try seen.getOrPut(key)).found_existing) continue;

        const in: c_int = @intCast(@as(usize, @intCast(ws[ws.len - 1])) / info.ty.blockBytes() * info.ty.blockElems());
        // One row = the decode matvec, 16 = the prefill tile kernel.
        for ([_]c_int{ 1, 16 }) |m| {
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            const x_shape: []const c_int = if (bank) &.{ m, 1, in } else &.{ m, in };
            try mlx.check(mlx.mlx_zeros(&x, x_shape.ptr, x_shape.len, .bfloat16, s));
            const y = if (bank) blk: {
                var experts = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(experts);
                try mlx.check(mlx.mlx_zeros(&experts, &[_]c_int{m}, 1, .uint32, s));
                break :blk try kernels.gatherLinear(.{ .ty = info.ty }, x, w, .{ .ctx = null }, experts, false, s);
            } else try kernels.linear(.{ .ty = info.ty }, x, w, s);
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_array_eval(y));
        }
    }
}

fn put(allocator: std.mem.Allocator, out: *WeightMap, name: []const u8, arr: mlx.mlx_array) !void {
    errdefer _ = mlx.mlx_array_free(arr);
    const key = try allocator.dupe(u8, name);
    errdefer allocator.free(key);
    try out.putNoClobber(key, arr);
}

fn loadQuantized(allocator: std.mem.Allocator, t: gguf.Tensor, mapped: arch_mod.Mapped, hp: qwen35.Hparams, out: *WeightMap, s: mlx.mlx_stream) !void {
    // Q2_0 has no ggml kernel: only its affine layout (`loadAffine`) serves it.
    if (!std.mem.endsWith(u8, mapped.name, ".weight") or t.ty == .q2_0) return error.UnsupportedTensor;
    const rows: usize = @intCast(t.dims[1]);
    if (t.n_dims == 3) {
        // An expert bank, experts outermost: [experts, rows, row bytes].
        const experts: usize = @intCast(t.dims[2]);
        const rb = t.data.len / experts / rows;
        switch (mapped.transform) {
            .none => {
                const bank = [_]c_int{ @intCast(experts), @intCast(rows), @intCast(rb) };
                try put(allocator, out, mapped.name, try packedRows(allocator, t.ty, t.data, &bank, s));
                return putSentinel(allocator, out, mapped.name, .{ .ty = t.ty }, s);
            },
            .split_gate_up => {
                const at = std.mem.indexOf(u8, mapped.name, ".gate_proj.") orelse return error.UnsupportedTensor;
                if (rows % 2 != 0) return error.BadShape;
                const half = rows / 2;
                const tmp = try allocator.alloc(u8, experts * half * rb);
                defer allocator.free(tmp);
                for (0..2) |k| {
                    for (0..experts) |e| @memcpy(tmp[e * half * rb ..][0 .. half * rb], t.data[(e * rows + k * half) * rb ..][0 .. half * rb]);
                    var name_buf: [160]u8 = undefined;
                    const name = if (k == 0) mapped.name else try std.fmt.bufPrint(&name_buf, "{s}.up_proj.{s}", .{ mapped.name[0..at], mapped.name[at + ".gate_proj.".len ..] });
                    const bank = [_]c_int{ @intCast(experts), @intCast(half), @intCast(rb) };
                    try put(allocator, out, name, try packedRows(allocator, t.ty, tmp, &bank, s));
                    try putSentinel(allocator, out, name, .{ .ty = t.ty }, s);
                }
                return;
            },
            else => return error.UnsupportedTensor,
        }
    }
    if (t.n_dims != 2) return error.UnsupportedTensor;
    const row_bytes = t.data.len / rows;
    const shape = [_]c_int{ @intCast(rows), @intCast(row_bytes) };

    if (mapped.transform == .split_ba) {
        const g = mapped.transform.split_ba;
        const nv = g.nk * g.r;
        if (rows != 2 * nv) return error.BadShape;
        const marker = ".in_proj_ba.";
        const at = std.mem.indexOf(u8, mapped.name, marker) orelse return error.UnsupportedTensor;
        const tmp = try allocator.alloc(u8, nv * row_bytes);
        defer allocator.free(tmp);
        for ([_][]const u8{ ".in_proj_b.", ".in_proj_a." }, 0..) |half_name, half| {
            for (0..g.nk) |k| for (0..g.r) |j| {
                const src = (k * 2 * g.r + half * g.r + j) * row_bytes;
                @memcpy(tmp[(k * g.r + j) * row_bytes ..][0..row_bytes], t.data[src..][0..row_bytes]);
            };
            var name_buf: [160]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "{s}{s}{s}", .{ mapped.name[0..at], half_name, mapped.name[at + marker.len ..] });
            try put(allocator, out, name, mlx.mlx_array_new_data(tmp.ptr, &[_]c_int{ @intCast(nv), @intCast(row_bytes) }, 2, .uint8));
            try putSentinel(allocator, out, name, .{ .ty = t.ty }, s);
        }
        return;
    }

    var info = kernels.Info{ .ty = t.ty };
    const w = switch (mapped.transform) {
        .none => try packedRows(allocator, t.ty, t.data, &shape, s),
        .tiled_input => blk: {
            info.tiled = .{ .nk = @intCast(hp.nk), .r = @intCast(hp.r()) };
            break :blk try packedRows(allocator, t.ty, t.data, &shape, s);
        },
        .untile_rows => |u| blk: {
            // Rows are whole blocks, so moving a row moves its bytes and nothing else.
            if (u.start + hp.nv * u.unit != rows) return error.BadShape;
            const tmp = try allocator.alloc(u8, t.data.len);
            defer allocator.free(tmp);
            untile(tmp, t.data, u.start * row_bytes, u.unit * row_bytes, hp);
            break :blk try packedRows(allocator, t.ty, tmp, &shape, s);
        },
        .rope_rows => |hd| blk: {
            const tmp = try allocator.alloc(u8, t.data.len);
            defer allocator.free(tmp);
            try unpermuteRope(tmp, t.data, row_bytes, hd);
            break :blk try packedRows(allocator, t.ty, tmp, &shape, s);
        },
        else => return error.UnsupportedTensor,
    };
    try put(allocator, out, mapped.name, w);
    try putSentinel(allocator, out, mapped.name, info, s);
}

/// `arch.Transform.rope_rows`: per head of `hd` rows, row `2j + p` of `src` lands at `p*hd/2 + j` of `dst`.
fn unpermuteRope(dst: []u8, src: []const u8, row_bytes: usize, hd: usize) !void {
    const rows = src.len / row_bytes;
    if (hd == 0 or hd % 2 != 0 or rows % hd != 0) return error.BadShape;
    for (0..rows / hd) |h| for (0..hd / 2) |j| for (0..2) |p| {
        @memcpy(dst[(h * hd + p * hd / 2 + j) * row_bytes ..][0..row_bytes], src[(h * hd + 2 * j + p) * row_bytes ..][0..row_bytes]);
    };
}

test "unpermuteRope puts llama.cpp's interleaved rope rows back in rotate-half order per head" {
    // Two heads of 4 rows, one byte per row: GGUF rows hold HF rows [0, 2, 1, 3] of each head.
    const src = [_]u8{ 0, 2, 1, 3, 4, 6, 5, 7 };
    var dst: [8]u8 = undefined;
    try unpermuteRope(&dst, &src, 1, 4);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &dst);
    try std.testing.expectError(error.BadShape, unpermuteRope(dst[0..6], src[0..6], 1, 4));
}

/// The weight as an MLX array, rows in the split layout when the kernels want it (see kernels.isSplit).
fn packedRows(allocator: std.mem.Allocator, ty: @import("quants.zig").GgmlType, bytes: []const u8, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    const row_bytes: usize = @intCast(shape[shape.len - 1]);
    const in = row_bytes / ty.blockBytes() * ty.blockElems();
    if (!kernels.isSplit(ty, in)) return mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), .uint8);
    const rows = bytes.len / row_bytes;
    const tmp = try allocator.alloc(u8, rows * kernels.rowBytes(ty, in));
    defer allocator.free(tmp);
    kernels.packRows(ty, in, tmp, bytes);
    var padded: [3]c_int = undefined;
    @memcpy(padded[0..shape.len], shape);
    padded[shape.len - 1] = @intCast(kernels.rowBytes(ty, in));
    return mlx.mlx_array_new_data(tmp.ptr, &padded, @intCast(shape.len), .uint8);
}

/// `<base>.scales` next to `<base>.weight`, see kernels.Info.
fn putSentinel(allocator: std.mem.Allocator, out: *WeightMap, weight_name: []const u8, info: kernels.Info, s: mlx.mlx_stream) !void {
    var sc_name: [160]u8 = undefined;
    const base = weight_name[0 .. weight_name.len - "weight".len];
    try put(allocator, out, try std.fmt.bufPrint(&sc_name, "{s}scales", .{base}), try kernels.sentinel(info, s));
}

/// Copy `src` to `dst` moving each tiled value-head chunk (`unit` bytes, the
/// region starting at `start`) to its grouped position.
fn untile(dst: []u8, src: []const u8, start: usize, unit: usize, hp: qwen35.Hparams) void {
    @memcpy(dst[0..start], src[0..start]);
    for (0..hp.nv) |tiled| {
        const grouped = qwen35.groupedIndex(@intCast(tiled), hp.nk, hp.r());
        @memcpy(dst[start + grouped * unit ..][0..unit], src[start + tiled * unit ..][0..unit]);
    }
}

fn loadFloat(allocator: std.mem.Allocator, t: gguf.Tensor, transform: arch_mod.Transform, hp: qwen35.Hparams, s: mlx.mlx_stream) !mlx.mlx_array {
    // ggml dims are innermost first.
    var shape_buf: [5]c_int = undefined;
    for (0..t.n_dims) |i| shape_buf[i] = @intCast(t.dims[t.n_dims - 1 - i]);
    var shape: []const c_int = shape_buf[0..t.n_dims];
    const dtype: mlx.mlx_dtype = switch (t.ty) {
        .f32 => .float32,
        .f16 => .float16,
        .bf16 => .bfloat16,
        else => unreachable,
    };

    const elem = t.ty.blockBytes();
    const raw = switch (transform) {
        .none, .keep_f32 => mlx.mlx_array_new_data(t.data.ptr, shape.ptr, @intCast(shape.len), dtype),
        .shared_gate => blk: {
            if (t.n_dims != 1) return error.BadShape;
            break :blk mlx.mlx_array_new_data(t.data.ptr, &[_]c_int{ 1, shape[0] }, 2, dtype);
        },
        .trailing_axis => blk: {
            if (t.n_dims + 1 > shape_buf.len) return error.BadShape;
            shape_buf[t.n_dims] = 1;
            shape = shape_buf[0 .. t.n_dims + 1];
            break :blk mlx.mlx_array_new_data(t.data.ptr, shape.ptr, @intCast(shape.len), dtype);
        },
        .minus_one, .neg_log => blk: {
            if (t.ty != .f32) return error.UnsupportedTensor;
            const tmp = try allocator.alloc(f32, t.elems());
            defer allocator.free(tmp);
            for (tmp, std.mem.bytesAsSlice(f32, t.data)) |*o, v| o.* = if (transform == .neg_log) @log(-v) else v - 1;
            const flat = [_]c_int{@intCast(t.elems())};
            break :blk mlx.mlx_array_new_data(tmp.ptr, if (transform == .neg_log) &flat else shape.ptr, if (transform == .neg_log) 1 else @intCast(shape.len), dtype);
        },
        .flatten => mlx.mlx_array_new_data(t.data.ptr, &[_]c_int{@intCast(t.elems())}, 1, dtype),
        .rope_rows => |hd| blk: {
            if (t.n_dims != 2) return error.BadShape;
            const tmp = try allocator.alignedAlloc(u8, .@"4", t.data.len);
            defer allocator.free(tmp);
            try unpermuteRope(tmp, t.data, t.data.len / @as(usize, @intCast(t.dims[1])), hd);
            break :blk mlx.mlx_array_new_data(tmp.ptr, shape.ptr, @intCast(shape.len), dtype);
        },
        .split_ba, .split_gate_up => return error.UnsupportedTensor,
        // Un-tiling only moves bytes, so it works the same on any float type.
        .untile_rows, .untile_vec, .a_log, .conv1d, .tiled_input => blk: {
            const tmp = try allocator.alignedAlloc(u8, .@"4", t.data.len);
            defer allocator.free(tmp);
            const row: usize = if (t.n_dims == 2) @intCast(t.dims[0]) else 1;
            switch (transform) {
                .untile_rows => |u| {
                    if (t.n_dims != 2 or u.start + hp.nv * u.unit != t.dims[1]) return error.BadShape;
                    untile(tmp, t.data, u.start * row * elem, u.unit * row * elem, hp);
                },
                // [channels, kernel]: q and k channels first, then the value channels.
                .conv1d => {
                    if (t.n_dims != 2 or 2 * hp.nk * hp.dk + hp.nv * hp.dv != t.dims[1]) return error.BadShape;
                    untile(tmp, t.data, 2 * hp.nk * hp.dk * row * elem, hp.dv * row * elem, hp);
                    shape_buf[t.n_dims] = 1;
                    shape = shape_buf[0 .. t.n_dims + 1];
                },
                // A float weight can simply have its columns moved, row by row.
                .tiled_input => {
                    if (t.n_dims != 2 or hp.nv * hp.dv != row) return error.BadShape;
                    for (0..@intCast(t.dims[1])) |r| untile(tmp[r * row * elem ..][0 .. row * elem], t.data[r * row * elem ..][0 .. row * elem], 0, hp.dv * elem, hp);
                },
                else => {
                    if (t.elems() != hp.nv) return error.BadShape;
                    untile(tmp, t.data, 0, elem, hp);
                },
            }
            if (transform == .a_log) {
                if (t.ty != .f32) return error.UnsupportedTensor;
                for (std.mem.bytesAsSlice(f32, tmp)) |*v| v.* = @log(-v.*);
            }
            break :blk mlx.mlx_array_new_data(tmp.ptr, shape.ptr, @intCast(shape.len), dtype);
        },
    };
    // A_log loads f32 like the reference checkpoints (`narrowALog` narrows it where a fused kernel wants bf16).
    if (transform == .a_log or transform == .neg_log or transform == .keep_f32 or dtype == .bfloat16) return raw;
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

test "an mmproj's gate and up join gate-first into mlp.w1, and its 4-D patch conv flattens to the tower's linear" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    const gate = [_]f32{ 1, 2, 3, 4 };
    const up = [_]f32{ 5, 6, 7, 8 };
    try put(a, &map, "vision.blocks.0.mlp.w1_gate.weight", mlx.mlx_array_new_data(&gate, &[_]c_int{ 2, 2 }, 2, .float32));
    try put(a, &map, "vision.blocks.0.mlp.w1_up.weight", mlx.mlx_array_new_data(&up, &[_]c_int{ 2, 2 }, 2, .float32));
    var px: [2 * 3 * 2 * 2]f32 = undefined;
    for (&px, 0..) |*v, i| v.* = @floatFromInt(i);
    try put(a, &map, "vision.patch_embed.proj.weight", mlx.mlx_array_new_data(&px, &[_]c_int{ 2, 3, 2, 2 }, 4, .float32));
    try joinMmproj(a, &map, 1, s);
    try std.testing.expect(map.get("vision.blocks.0.mlp.w1_gate.weight") == null and map.get("vision.blocks.0.mlp.w1_up.weight") == null);
    const w1 = map.get("vision.blocks.0.mlp.w1.weight").?;
    try std.testing.expectEqualSlices(c_int, &.{ 4, 2 }, mlx.mlx_array_shape(w1)[0..2]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, mlx.mlx_array_data_float32(w1).?[0..8]);
    const p = map.get("vision.patch_embed.proj.weight").?;
    try std.testing.expectEqualSlices(c_int, &.{ 2, 12 }, mlx.mlx_array_shape(p)[0..2]);
    try std.testing.expectEqualSlices(f32, px[0..], mlx.mlx_array_data_float32(p).?[0..24]);
}

test "Nemotron-H's and Flash Next's A_log narrow to bf16 for their fused kernels, other A_log stay f32" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    const v = [_]f32{ -0.5, 1.25 };
    for ([_][]const u8{ "backbone.layers.0.mixer.A_log", "model.layers.0.linear_attn.A_log" }) |n|
        try put(a, &map, n, mlx.mlx_array_new_data(&v, &[_]c_int{2}, 1, .float32));
    try narrowALog(&map, ".mixer.A_log", s);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(map.get("backbone.layers.0.mixer.A_log").?));
    try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(map.get("model.layers.0.linear_attn.A_log").?));
}

test "untile moves tiled value-head chunks to grouped order and keeps the prefix" {
    var hp = std.mem.zeroes(qwen35.Hparams);
    hp.nk = 2;
    hp.nv = 6;
    // prefix "PP", then heads tiled as g0j0 g1j0 g0j1 g1j1 g0j2 g1j2, two bytes each.
    const src = "PP" ++ "a0" ++ "b0" ++ "a1" ++ "b1" ++ "a2" ++ "b2";
    var dst: [src.len]u8 = undefined;
    untile(&dst, src, 2, 2, hp);
    try std.testing.expectEqualStrings("PPa0a1a2b0b1b2", &dst);
}

test "GDN gates join the qkv and z layout they share at 4 bits or more, else stay dense" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const layer = "language_model.model.layers.0.";
    const Layout = struct { bits: c_int, gs: c_int };
    const Case = struct { qkv: Layout, z: ?Layout, joins: bool };
    for ([_]Case{
        .{ .qkv = .{ .bits = 4, .gs = 32 }, .z = .{ .bits = 4, .gs = 32 }, .joins = true },
        .{ .qkv = .{ .bits = 2, .gs = 64 }, .z = .{ .bits = 2, .gs = 64 }, .joins = false },
        .{ .qkv = .{ .bits = 4, .gs = 32 }, .z = .{ .bits = 8, .gs = 32 }, .joins = false },
        .{ .qkv = .{ .bits = 4, .gs = 32 }, .z = null, .joins = false },
    }) |c| {
        var map = WeightMap.init(a);
        defer {
            var it = map.iterator();
            while (it.next()) |e| {
                _ = mlx.mlx_array_free(e.value_ptr.*);
                a.free(e.key_ptr.*);
            }
            map.deinit();
        }
        var prng = std.Random.DefaultPrng.init(0x9a7e);
        var buf: [16 * 128]f32 = undefined;
        for (&buf) |*v| v.* = prng.random().floatNorm(f32) * 0.02;
        const parts = [_]struct { name: []const u8, rows: c_int, q: ?Layout }{
            .{ .name = "in_proj_qkv", .rows = 16, .q = c.qkv },
            .{ .name = "in_proj_z", .rows = 16, .q = c.z },
            .{ .name = "in_proj_a", .rows = 4, .q = null },
            .{ .name = "in_proj_b", .rows = 4, .q = null },
        };
        for (parts) |part| {
            const f = mlx.mlx_array_new_data(&buf, &[_]c_int{ part.rows, 128 }, 2, .float32);
            defer _ = mlx.mlx_array_free(f);
            var bf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&bf, f, .bfloat16, s));
            var nb: [128]u8 = undefined;
            const q = part.q orelse {
                try put(a, &map, try std.fmt.bufPrint(&nb, "{s}linear_attn.{s}.weight", .{ layer, part.name }), bf);
                continue;
            };
            defer _ = mlx.mlx_array_free(bf);
            var v = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_quantize(&v, bf, mlx.mlx_optional_int.some(q.gs), mlx.mlx_optional_int.some(q.bits), "affine", .{}, s));
            for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, li| {
                var arr = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&arr, v, li));
                try mlx.check(mlx.mlx_array_eval(arr));
                try put(a, &map, try std.fmt.bufPrint(&nb, "{s}linear_attn.{s}.{s}", .{ layer, part.name, leaf }), arr);
            }
        }
        try quantizeGdnGates(a, &map, layer, s);
        for ([_][]const u8{ "in_proj_a", "in_proj_b" }) |gate| {
            var nb: [128]u8 = undefined;
            const sc = map.get(try std.fmt.bufPrint(&nb, "{s}linear_attn.{s}.scales", .{ layer, gate }));
            try std.testing.expectEqual(c.joins, sc != null);
            if (sc) |x| try std.testing.expectEqual(@divExact(@as(c_int, 128), c.qkv.gs), mlx.getShape(x)[1]);
        }
    }
}

test "the MTP head loads in our packs' head layout: experts 4-bit g64, the rest 8-bit g64, narrower g64 kept" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const quants = @import("quants.zig");
    const q4 = arch_mod.qwen4exp;
    // rel: the RMS error against ggml's weights the stored layout may cost (a re-rounding at 4 or 8 bits, else bf16 scales).
    const Case = struct { name: []const u8, ty: quants.GgmlType, fixture: []const u8, in: usize, bits: ?u32, rel: f32 };
    inline for ([_]Case{
        .{ .name = q4.mtp_prefix ++ "layers.0.mlp.switch_mlp.gate_proj.weight", .ty = .q8_0, .fixture = "q8_0.bin", .in = 128, .bits = 4, .rel = 0.15 },
        .{ .name = q4.mtp_prefix ++ "fc_embedding.weight", .ty = .q8_0, .fixture = "q8_0.bin", .in = 128, .bits = 8, .rel = 0.02 },
        .{ .name = q4.mtp_prefix ++ "layers.0.mlp.switch_mlp.down_proj.weight", .ty = .q2_0, .fixture = "q2_0.bin", .in = 256, .bits = null, .rel = 0.005 },
    }) |c| {
        const data = @embedFile("fixtures/" ++ c.fixture);
        const n_blocks = data.len / (c.ty.blockBytes() + 4 * c.ty.blockElems());
        const rows = n_blocks * c.ty.blockElems() / c.in;
        const t = gguf.Tensor{ .dims = .{ c.in, rows, 1, 1 }, .n_dims = 2, .ty = c.ty, .data = data[0 .. rows * c.in / c.ty.blockElems() * c.ty.blockBytes()] };
        const ref = try a.alloc(f32, rows * c.in);
        defer a.free(ref);
        try quants.dequantize(c.ty, t.data, ref);
        var map = WeightMap.init(a);
        defer {
            var it = map.iterator();
            while (it.next()) |e| {
                _ = mlx.mlx_array_free(e.value_ptr.*);
                a.free(e.key_ptr.*);
            }
            map.deinit();
        }
        try std.testing.expect(try loadAffine(a, t, .{ .name = c.name, .transform = .none }, std.mem.zeroes(qwen35.Hparams), &map));
        try std.testing.expectEqual(c.bits, headBits(c.name, t));
        if (c.bits) |bits| try requantHead(&map, c.name, t, bits, s);
        const stem = c.name[0 .. c.name.len - "weight".len];
        const w = map.get(c.name).?;
        const sc = map.get(stem ++ "scales").?;
        const bits: c_int = @intCast(c.bits orelse affine.bitsOf(c.ty).?);
        try std.testing.expectEqual(@divExact(@as(c_int, @intCast(c.in)) * bits, 32), mlx.getShape(w)[1]);
        try std.testing.expectEqual(@as(c_int, @intCast(c.in / 64)), mlx.getShape(sc)[1]);
        var back = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(back);
        try mlx.check(mlx.mlx_dequantize(&back, w, sc, map.get(stem ++ "biases").?, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(bits), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
        try mlx.check(mlx.mlx_array_eval(back));
        const got = mlx.mlx_array_data_float32(back).?;
        var err: f64 = 0;
        var mag: f64 = 0;
        for (ref, 0..) |r, i| {
            err += (got[i] - r) * (got[i] - r);
            mag += r * r;
        }
        try std.testing.expect(@sqrt(err / mag) < c.rel);
    }
}

test "the MTP combiner splits into fc_embedding | fc_hidden along its input, embedding first" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    // 2 rows of 64 inputs at 8 bits: 16 words and 2 groups a row; word i holds i, scale g holds g.
    var w: [32]u32 = undefined;
    for (&w, 0..) |*v, i| v.* = @intCast(i % 16);
    var sc: [4]u16 = undefined;
    for (&sc, 0..) |*v, i| v.* = affine.bf16(@floatFromInt(i % 2));
    const stem = arch_mod.qwen4exp.mtp_prefix ++ "eh_proj.";
    try put(a, &map, stem ++ "weight", mlx.mlx_array_new_data(&w, &[_]c_int{ 2, 16 }, 2, .uint32));
    try put(a, &map, stem ++ "scales", mlx.mlx_array_new_data(&sc, &[_]c_int{ 2, 2 }, 2, .bfloat16));
    try put(a, &map, stem ++ "biases", mlx.mlx_array_new_data(&sc, &[_]c_int{ 2, 2 }, 2, .bfloat16));
    try splitEhProj(a, &map, s);
    try std.testing.expect(map.get(stem ++ "weight") == null);
    for ([_][]const u8{ "fc_embedding", "fc_hidden" }, 0..) |part, half| {
        var nb: [96]u8 = undefined;
        const pw = map.get(try std.fmt.bufPrint(&nb, arch_mod.qwen4exp.mtp_prefix ++ "{s}.weight", .{part})).?;
        try std.testing.expectEqualSlices(c_int, &.{ 2, 8 }, mlx.getShape(pw));
        try std.testing.expectEqual(@as(u32, @intCast(half * 8)), mlx.mlx_array_data_uint32(pw).?[0]);
        const ps = map.get(try std.fmt.bufPrint(&nb, arch_mod.qwen4exp.mtp_prefix ++ "{s}.scales", .{part})).?;
        try std.testing.expectEqualSlices(c_int, &.{ 2, 1 }, mlx.getShape(ps));
    }
}

test "an affine-coded tensor loads as MLX affine: mlx's dequant equals ggml's, value heads un-tiled" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const quants = @import("quants.zig");
    const Case = struct { ty: quants.GgmlType, fixture: []const u8, in: usize };
    var hp = std.mem.zeroes(qwen35.Hparams);
    hp.nk = 2;
    hp.nv = 4;
    inline for ([_]Case{ .{ .ty = .q4_k, .fixture = "q4_k.bin", .in = 256 }, .{ .ty = .q5_1, .fixture = "q5_1.bin", .in = 128 }, .{ .ty = .q8_0, .fixture = "q8_0.bin", .in = 128 }, .{ .ty = .q2_0, .fixture = "q2_0.bin", .in = 256 } }) |c| {
        const g = affine.groupOf(c.ty);
        hp.dv = @intCast(c.in / hp.nv);
        const data = @embedFile("fixtures/" ++ c.fixture);
        const n_blocks = data.len / (c.ty.blockBytes() + 4 * c.ty.blockElems());
        const rows = n_blocks * c.ty.blockElems() / c.in;
        const row_bytes = c.in / c.ty.blockElems() * c.ty.blockBytes();
        const t = gguf.Tensor{ .dims = .{ c.in, rows, 1, 1 }, .n_dims = 2, .ty = c.ty, .data = data[0 .. rows * row_bytes] };
        const ref = try a.alloc(f32, rows * c.in);
        defer a.free(ref);
        try quants.dequantize(c.ty, t.data, ref);
        // The bound bf16 scales and biases allow: half a bf16 ulp of each term.
        const tol = try a.alloc(f32, rows * c.in);
        defer a.free(tol);
        {
            const q = try a.alloc(u8, rows * c.in);
            defer a.free(q);
            const sc = try a.alloc(f32, rows * c.in / g);
            defer a.free(sc);
            const bi = try a.alloc(f32, rows * c.in / g);
            defer a.free(bi);
            affine.decodeRow(c.ty, t.data, q, sc, bi);
            for (tol, q, 0..) |*o, code, i| o.* = (@abs(sc[i / g] * @as(f32, @floatFromInt(code))) + @abs(bi[i / g])) / 256 + 1e-6;
        }
        for ([_]arch_mod.Transform{ .none, .tiled_input }) |tr| {
            var map = WeightMap.init(a);
            defer {
                var it = map.iterator();
                while (it.next()) |e| {
                    _ = mlx.mlx_array_free(e.value_ptr.*);
                    a.free(e.key_ptr.*);
                }
                map.deinit();
            }
            try std.testing.expect(try loadAffine(a, t, .{ .name = "x.weight", .transform = tr }, hp, &map));
            // f32 scales: mlx's dequant computes in the scales' dtype, the matmul kernels in f32.
            var sc32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc32);
            try mlx.check(mlx.mlx_astype(&sc32, map.get("x.scales").?, .float32, s));
            var bi32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi32);
            try mlx.check(mlx.mlx_astype(&bi32, map.get("x.biases").?, .float32, s));
            var y = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_dequantize(&y, map.get("x.weight").?, sc32, bi32, mlx.mlx_optional_int.some(@intCast(g)), mlx.mlx_optional_int.some(@intCast(affine.bitsOf(c.ty).?)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
            try mlx.check(mlx.mlx_array_eval(y));
            const got = mlx.mlx_array_data_float32(y).?[0 .. rows * c.in];
            for (0..rows) |r| for (0..c.in) |col| {
                // Tiled input: grouped head h reads the tiled head holding it.
                const src_col = if (tr == .tiled_input) blk: {
                    var t_head: u32 = 0;
                    while (qwen35.groupedIndex(t_head, hp.nk, hp.r()) != col / hp.dv) t_head += 1;
                    break :blk t_head * hp.dv + col % hp.dv;
                } else col;
                try std.testing.expectApproxEqAbs(ref[r * c.in + src_col], got[r * c.in + col], tol[r * c.in + src_col]);
            };
        }
    }
}

test "an MXFP4 tensor loads as MLX mxfp4: mlx's dequant equals ggml's, bit for bit" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const quants = @import("quants.zig");
    const data = @embedFile("fixtures/mxfp4.bin");
    const n_blocks = data.len / (17 + 4 * 32);
    const in = 64;
    const rows = n_blocks * 32 / in;
    const t = gguf.Tensor{ .dims = .{ in, rows, 1, 1 }, .n_dims = 2, .ty = .mxfp4, .data = data[0 .. rows * in / 32 * 17] };
    const ref = try a.alloc(f32, rows * in);
    defer a.free(ref);
    try quants.dequantize(.mxfp4, t.data, ref);
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    try std.testing.expect(try loadMxfp4(a, t, .{ .name = "x.weight", .transform = .none }, &map));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_dequantize(&y, map.get("x.weight").?, map.get("x.scales").?, .{ .ctx = null }, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "mxfp4", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_array_eval(y));
    try std.testing.expectEqualSlices(f32, ref, mlx.mlx_array_data_float32(y).?[0 .. rows * in]);
}

test "a tiled-input weight of a split-layout type loads in the kernels' layout: its linear equals the un-tiled weights" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const quants = @import("quants.zig");
    const Case = struct { ty: quants.GgmlType, fixture: []const u8, in: usize };
    var hp = std.mem.zeroes(qwen35.Hparams);
    hp.nk = 2;
    hp.nv = 4;
    inline for ([_]Case{ .{ .ty = .q8_0, .fixture = "q8_0.bin", .in = 128 }, .{ .ty = .q6_k, .fixture = "q6_k.bin", .in = 256 } }) |c| {
        hp.dv = @intCast(c.in / hp.nv);
        const data = @embedFile("fixtures/" ++ c.fixture);
        const n_blocks = data.len / (c.ty.blockBytes() + 4 * c.ty.blockElems());
        const rows = n_blocks * c.ty.blockElems() / c.in;
        const t = gguf.Tensor{ .dims = .{ c.in, rows, 1, 1 }, .n_dims = 2, .ty = c.ty, .data = data[0 .. rows * c.in / c.ty.blockElems() * c.ty.blockBytes()] };
        const wref = try a.alloc(f32, rows * c.in);
        defer a.free(wref);
        try quants.dequantize(c.ty, t.data, wref);
        var map = WeightMap.init(a);
        defer {
            var it = map.iterator();
            while (it.next()) |e| {
                _ = mlx.mlx_array_free(e.value_ptr.*);
                a.free(e.key_ptr.*);
            }
            map.deinit();
        }
        try loadQuantized(a, t, .{ .name = "x.weight", .transform = .tiled_input }, hp, &map, s);
        // x in grouped head order (what the model hands the projection), exact in bf16.
        const M = 2;
        var xs: [M * 256]f32 = undefined;
        for (xs[0 .. M * c.in], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 7 % 17)) - 8)) / 8;
        const xf = mlx.mlx_array_new_data(&xs, &[_]c_int{ M, @intCast(c.in) }, 2, .float32);
        defer _ = mlx.mlx_array_free(xf);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
        const w = map.get("x.weight").?;
        const y = try kernels.linear(kernels.infoOf(w, map.get("x.scales").?).?, x, w, s);
        defer _ = mlx.mlx_array_free(y);
        var y32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y32);
        try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
        try mlx.check(mlx.mlx_array_eval(y32));
        const got = mlx.mlx_array_data_float32(y32).?;
        for (0..M) |m| for (0..rows) |r| {
            // Weight column c sits in tiled head c / dv; the activation it meets is that head's grouped position.
            var want: f32 = 0;
            var mag: f32 = 0;
            for (0..c.in) |col| {
                const gh = qwen35.groupedIndex(@intCast(col / hp.dv), hp.nk, hp.r());
                const xv = xs[m * c.in + gh * hp.dv + col % hp.dv];
                want += xv * wref[r * c.in + col];
                mag += @abs(xv * wref[r * c.in + col]);
            }
            try std.testing.expectApproxEqAbs(want, got[m * rows + r], mag / 64 + 1e-4);
        };
    }
}

test "a DeepSeek-V4.1 vision tower GGUF loads under the release's names, bf16 as stored, rows = output features" {
    const a = std.testing.allocator;
    const ds41 = arch_mod.deepseek41;
    var t = try ds41.testFileDs4(ds41.release_config);
    defer t.deinit();
    var v = try ds41.testVisionFileDs4();
    defer v.deinit();
    const w = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0 };
    for ([_][]const u8{ "vision.patch_embed.proj.weight", "aligner.w1.weight" }) |n|
        try v.tensors.put(v.arena.allocator(), n, .{ .dims = .{ 3, 2, 1, 1 }, .n_dims = 2, .ty = .bf16, .data = std.mem.sliceAsBytes(&w) });
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try loadVision(a, &v, try ds41.Hparams.read(&t), &map, s);
    try std.testing.expectEqual(@as(u32, 2), map.count());
    const got = map.get("aligner.w1.weight").?;
    try std.testing.expectEqualSlices(c_int, &.{ 2, 3 }, mlx.getShape(got));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(got));
    try mlx.check(mlx.mlx_array_eval(got));
    try std.testing.expectEqualSlices(u16, &w, mlx.mlx_array_data_bfloat16(got).?[0..6]);
    try std.testing.expectEqual(@as(u64, 24), try visionResidentBytes(&v, try ds41.Hparams.read(&t)));
}

test "real model loads into a weight map (a real GGUF, env-gated)" {
    const path = std.mem.span(env.get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var f = try gguf.File.open(a, path);
    defer f.deinit();
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    try load(a, &f, &map, s);
    try warmKernels(a, &map, s);

    const arch = try Arch.read(&f);
    const p = "language_model.model.";
    const prefix: []const u8 = switch (arch) {
        .gemma3, .lfm2, .gpt_oss, .qwen3next, .qwen3, .flint, .muse_glimmer => "model.",
        .nemotron_h => "backbone.",
        .deepseek41 => "",
        else => p,
    };
    const norm_name = if (arch == .qwen4exp) try a.dupe(u8, p ++ "hyper_connection_mixer.hc_norm.weight") else try std.fmt.allocPrint(a, "{s}{s}norm{s}.weight", .{ prefix, if (arch == .lfm2) "embedding_" else "", if (arch == .nemotron_h) "_f" else "" });
    defer a.free(norm_name);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(map.get(norm_name).?));
    switch (arch) {
        .qwen35 => |hp| {
            const qkv = map.get(p ++ "layers.0.linear_attn.in_proj_qkv.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(2 * hp.nk * hp.dk + hp.nv * hp.dv)), mlx.getShape(qkv)[0]);
            const out_sc = map.get(p ++ "layers.0.linear_attn.out_proj.scales").?;
            const out_w = map.get(p ++ "layers.0.linear_attn.out_proj.weight").?;
            try std.testing.expectEqual(hp.r() > 1, kernels.infoOf(out_w, out_sc).?.tiled != null);
            try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(map.get(p ++ "layers.0.linear_attn.A_log").?));
            try std.testing.expectEqualSlices(c_int, &.{ @intCast(2 * hp.nk * hp.dk + hp.nv * hp.dv), @intCast(hp.conv_kernel), 1 }, mlx.getShape(map.get(p ++ "layers.0.linear_attn.conv1d.weight").?));
        },
        .gemma4 => |hp| {
            try std.testing.expectEqual(hp.n_experts > 0, map.get(p ++ "layers.0.experts.switch_glu.up_proj.weight") != null);
            try std.testing.expectEqualSlices(c_int, &.{1}, mlx.getShape(map.get(p ++ "layers.0.layer_scalar").?));
            try std.testing.expect(map.get("rope_freqs.weight") == null);
            try std.testing.expectEqual(hp.ple_dim > 0, map.get(p ++ "embed_tokens_per_layer.weight") != null);
            try std.testing.expectEqual(hp.ple_dim > 0, map.get(p ++ "layers.0.per_layer_input_gate.weight") != null);
            const q = map.get(p ++ "layers.0.self_attn.q_proj.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(hp.n_heads * hp.head_dim)), mlx.getShape(q)[0]);
        },
        .lfm2 => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{ @intCast(hp.hidden), @intCast(hp.conv_cache), 1 }, mlx.getShape(map.get("model.layers.0.conv.conv.weight").?));
            try std.testing.expectEqual(hp.moe, map.get("model.layers.2.feed_forward.switch_mlp.up_proj.weight") != null);
        },
        .qwen3next => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.nv)}, mlx.getShape(map.get("model.layers.0.linear_attn.A_log").?));
            try std.testing.expectEqual(@as(c_int, @intCast(hp.nv)), mlx.getShape(map.get("model.layers.0.linear_attn.in_proj_b.weight").?)[0]);
            try std.testing.expect(map.get("model.layers.0.linear_attn.in_proj_ba.weight") == null);
        },
        .nemotron_h => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.mamba_heads)}, mlx.getShape(map.get("backbone.layers.0.mixer.A_log").?));
            try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(map.get("backbone.layers.0.mixer.A_log").?));
        },
        .gpt_oss => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.n_heads)}, mlx.getShape(map.get("model.layers.0.self_attn.sinks").?));
            try std.testing.expect(map.get("model.layers.0.mlp.experts.gate_proj.bias") != null);
        },
        .qwen4exp => |hp| {
            const li = hp.base.full_attn_interval - 1;
            var nb: [128]u8 = undefined;
            const qk = map.get(try std.fmt.bufPrint(&nb, p ++ "layers.{d}.self_attn.indexer.index_qk_proj.weight", .{li})).?;
            try std.testing.expectEqual(@as(c_int, @intCast((hp.idx_heads + 1) * hp.idx_dim)), mlx.getShape(qk)[0]);
            try std.testing.expect(map.get(arch_mod.qwen4exp.table_name) == null);
        },
        inline .gemma3, .qwen3, .muse_glimmer => |hp| {
            const q = map.get("model.layers.0.self_attn.q_proj.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(hp.n_heads * hp.head_dim)), mlx.getShape(q)[0]);
            try std.testing.expect(map.get("model.layers.0.input_layernorm.weight") != null);
        },
        .deepseek41 => |hp| {
            try std.testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(map.get("layers.0.ffn.experts.w1.weight").?));
            try std.testing.expectEqual(@as(c_int, @intCast(hp.n_experts)), mlx.getShape(map.get("layers.0.ffn.experts.w1.weight").?)[0]);
        },
        .flint => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.n_heads)}, mlx.getShape(map.get("model.layers.0.self_attn.softmax_offset").?));
            try std.testing.expect(map.get("model.layers.2.mlp.switch_mlp.up_proj.weight") != null);
        },
    }
}
