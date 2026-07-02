//! Runtime (unfused) LoRA adapters for the image backends (FLUX.2 / Krea-2).
//!
//! A LoRA safetensors file carries pairs `<module>.lora_A.weight` [r,in] /
//! `<module>.lora_B.weight` [out,r] (diffusers naming; `lora_down`/`lora_up`
//! aliases accepted) plus an optional scalar `<module>.alpha` (net scale
//! alpha/rank, kohya convention). Adapters are NOT fused into the base
//! weights — each attached linear computes y = base(x) + scale·(x@Aᵀ)@Bᵀ at
//! runtime. That keeps quantized checkpoints lossless (no dequant→requant
//! round-trip) and makes detach a pointer clear.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

/// Non-owning adapter reference installed on a linear layer. `at`/`bt` are
/// pre-transposed bf16 so the hot path is two plain matmuls.
pub const Ref = struct {
    at: mlx.mlx_array, // [in, r]
    bt: mlx.mlx_array, // [r, out]
    scale: f32,
};

pub const Role = enum { a, b, alpha };
pub const KeyInfo = struct { module: []const u8, role: Role };

/// Classify one safetensors key. Strips common wrapper prefixes
/// (`base_model.model.`, `transformer.`, `diffusion_model.`) and normalizes
/// diffusers' `to_out.0` to `to_out`. Returns null for non-LoRA keys.
pub fn parseKey(key: []const u8) ?KeyInfo {
    var k = key;
    inline for (.{ "base_model.model.", "transformer.", "diffusion_model." }) |pfx| {
        if (std.mem.startsWith(u8, k, pfx)) k = k[pfx.len..];
    }
    const suffixes = .{
        .{ ".lora_A.weight", Role.a },
        .{ ".lora_down.weight", Role.a },
        .{ ".lora_B.weight", Role.b },
        .{ ".lora_up.weight", Role.b },
        .{ ".alpha", Role.alpha },
    };
    inline for (suffixes) |sf| {
        if (std.mem.endsWith(u8, k, sf[0])) {
            var m = k[0 .. k.len - sf[0].len];
            if (std.mem.endsWith(u8, m, ".to_out.0")) m = m[0 .. m.len - 2];
            return .{ .module = m, .role = sf[1] };
        }
    }
    return null;
}

/// y_delta = scale · (x @ at) @ bt, returned in x's dtype.
pub fn delta(x: mlx.mlx_array, ref: Ref, s: mlx.mlx_stream) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    var xa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xa);
    try mlx.check(mlx.mlx_matmul(&xa, x, ref.at, s));
    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_matmul(&xb, xa, ref.bt, s));
    const sc = mlx.mlx_array_new_float(ref.scale);
    defer _ = mlx.mlx_array_free(sc);
    var scaled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, xb, sc, s));
    if (mlx.mlx_array_dtype(scaled) != dt) {
        var back = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(back);
        try mlx.check(mlx.mlx_astype(&back, scaled, dt, s));
        _ = mlx.mlx_array_free(scaled);
        return back;
    }
    return scaled;
}

/// One loaded adapter pair, keyed by the module it targets.
pub const Entry = struct {
    module: []u8, // owned
    at: mlx.mlx_array, // [in, r] bf16
    bt: mlx.mlx_array, // [r, out] bf16
    scale: f32, // alpha/rank when the file carries alpha, else 1.0
};

/// All adapters from one safetensors file. Owns the arrays the installed
/// `Ref`s point at — must outlive every attach until detach.
pub const File = struct {
    allocator: std.mem.Allocator,
    entries: []Entry,

    pub fn deinit(self: *File) void {
        for (self.entries) |*e| {
            self.allocator.free(e.module);
            _ = mlx.mlx_array_free(e.at);
            _ = mlx.mlx_array_free(e.bt);
        }
        self.allocator.free(self.entries);
    }

    pub fn find(self: *const File, module: []const u8) ?*const Entry {
        for (self.entries) |*e| {
            if (std.mem.eql(u8, e.module, module)) return e;
        }
        return null;
    }
};

const Partial = struct {
    a: mlx.mlx_array = .{ .ctx = null },
    b: mlx.mlx_array = .{ .ctx = null },
    alpha: ?f32 = null,
};

fn scalarValue(arr: mlx.mlx_array, s: mlx.mlx_stream) ?f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    mlx.check(mlx.mlx_astype(&f, arr, .float32, s)) catch return null;
    _ = mlx.mlx_array_eval(f);
    const d = mlx.mlx_array_data_float32(f) orelse return null;
    return d[0];
}

/// Load every complete A/B pair from a LoRA .safetensors file. A/B are
/// pre-transposed to [in,r]/[r,out], materialized, and cast to bf16.
/// All load-time ops run on a CPU stream — `Load::eval_gpu` is not
/// implemented, exactly like `model.loadWeights` (unified memory makes the
/// arrays GPU-usable afterwards).
pub fn loadFile(allocator: std.mem.Allocator, path: []const u8) !File {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return error.BadLoraPath;
    const pathz = try allocator.dupeZ(u8, path);
    defer allocator.free(pathz);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, pathz, s));

    var partials = std.StringHashMap(Partial).init(allocator);
    defer {
        var it = partials.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.a.ctx != null) _ = mlx.mlx_array_free(e.value_ptr.a);
            if (e.value_ptr.b.ctx != null) _ = mlx.mlx_array_free(e.value_ptr.b);
            allocator.free(e.key_ptr.*);
        }
        partials.deinit();
    }

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const info = parseKey(std.mem.span(key.?)) orelse {
            _ = mlx.mlx_array_free(value);
            continue;
        };
        const gop = try partials.getOrPut(info.module);
        if (!gop.found_existing) {
            gop.key_ptr.* = try allocator.dupe(u8, info.module);
            gop.value_ptr.* = .{};
        }
        switch (info.role) {
            .a => {
                if (gop.value_ptr.a.ctx != null) _ = mlx.mlx_array_free(gop.value_ptr.a);
                gop.value_ptr.a = value;
            },
            .b => {
                if (gop.value_ptr.b.ctx != null) _ = mlx.mlx_array_free(gop.value_ptr.b);
                gop.value_ptr.b = value;
            },
            .alpha => {
                gop.value_ptr.alpha = scalarValue(value, s);
                _ = mlx.mlx_array_free(value);
            },
        }
    }

    var entries: std.ArrayList(Entry) = .empty;
    errdefer {
        for (entries.items) |*e| {
            allocator.free(e.module);
            _ = mlx.mlx_array_free(e.at);
            _ = mlx.mlx_array_free(e.bt);
        }
        entries.deinit(allocator);
    }
    var it = partials.iterator();
    while (it.next()) |e| {
        const p = e.value_ptr;
        if (p.a.ctx == null or p.b.ctx == null) continue; // incomplete pair
        const rank: c_int = mlx.getShape(p.a)[0]; // A [r,in]
        const at = try prepTransposed(p.a, s);
        errdefer _ = mlx.mlx_array_free(at);
        const bt = try prepTransposed(p.b, s);
        errdefer _ = mlx.mlx_array_free(bt);
        const scale: f32 = if (p.alpha) |al| al / @as(f32, @floatFromInt(rank)) else 1.0;
        try entries.append(allocator, .{
            .module = try allocator.dupe(u8, e.key_ptr.*),
            .at = at,
            .bt = bt,
            .scale = scale,
        });
    }
    return .{ .allocator = allocator, .entries = try entries.toOwnedSlice(allocator) };
}

/// [o,i] → materialized bf16 [i,o].
fn prepTransposed(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const axes = [_]c_int{ 1, 0 };
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, w, &axes, 2, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, t, false, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, c, .bfloat16, s));
    _ = mlx.mlx_array_eval(out); // settle the Load graph on the CPU stream
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "parseKey classifies diffusers + kohya LoRA keys and strips wrappers" {
    // diffusers A/B with transformer. prefix
    const a = parseKey("transformer.transformer_blocks.3.attn.to_q.lora_A.weight").?;
    try testing.expectEqualStrings("transformer_blocks.3.attn.to_q", a.module);
    try testing.expectEqual(Role.a, a.role);
    const b = parseKey("transformer.transformer_blocks.3.attn.to_q.lora_B.weight").?;
    try testing.expectEqual(Role.b, b.role);
    // to_out.0 normalization
    const o = parseKey("transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight").?;
    try testing.expectEqualStrings("transformer_blocks.0.attn.to_out", o.module);
    // kohya-style down/up aliases, no prefix
    const d = parseKey("single_transformer_blocks.7.attn.to_out.lora_down.weight").?;
    try testing.expectEqual(Role.a, d.role);
    try testing.expectEqualStrings("single_transformer_blocks.7.attn.to_out", d.module);
    const u = parseKey("blocks.2.mlp.gate.lora_up.weight").?;
    try testing.expectEqual(Role.b, u.role);
    // alpha
    const al = parseKey("blocks.2.mlp.gate.alpha").?;
    try testing.expectEqual(Role.alpha, al.role);
    try testing.expectEqualStrings("blocks.2.mlp.gate", al.module);
    // non-LoRA keys are ignored
    try testing.expect(parseKey("blocks.2.mlp.gate.weight") == null);
    try testing.expect(parseKey("bn.running_mean") == null);
}

test "delta computes scale·(x@Aᵀ)@Bᵀ" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // x [1,2] = [1,2]; A [1,2] (r=1,in=2) → at [2,1] = [[3],[4]];
    // B [2,1] (out=2,r=1) → bt [1,2] = [[5,6]]. x@at = 11; delta = 2·[55,66].
    const xv = [_]f32{ 1, 2 };
    const xs = [_]c_int{ 1, 2 };
    const x = mlx.mlx_array_new_data(&xv, &xs, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    const atv = [_]f32{ 3, 4 };
    const ats = [_]c_int{ 2, 1 };
    const at = mlx.mlx_array_new_data(&atv, &ats, 2, .float32);
    defer _ = mlx.mlx_array_free(at);
    const btv = [_]f32{ 5, 6 };
    const bts = [_]c_int{ 1, 2 };
    const bt = mlx.mlx_array_new_data(&btv, &bts, 2, .float32);
    defer _ = mlx.mlx_array_free(bt);
    const d = try delta(x, .{ .at = at, .bt = bt, .scale = 2.0 }, s);
    defer _ = mlx.mlx_array_free(d);
    _ = mlx.mlx_array_eval(d);
    const dd = mlx.mlx_array_data_float32(d) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 110), dd[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 132), dd[1], 1e-4);
}

test "loadFile rejects relative/empty paths (openFileAbsolute UB class)" {
    try testing.expectError(error.BadLoraPath, loadFile(testing.allocator, ""));
    try testing.expectError(error.BadLoraPath, loadFile(testing.allocator, "rel/lora.safetensors"));
}
