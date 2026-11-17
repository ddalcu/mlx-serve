//! A pack an outside quant plugin claims (`sdk/quant/outside_model.zig`): the trunk linears it stores in its own
//! format and the routed banks it loads, keyed by the map array the layer binder bound in their place, so the trunk's
//! projection (`qmatmul`) and its MoE run them on the plugin's table. The trunk's forward is otherwise unchanged.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const model = @import("../../sdk/config/model.zig");
const plugin_abi = @import("../../abi/plugin.zig");
const outside = @import("../../sdk/quant/outside.zig");
const outside_model = @import("../../sdk/quant/outside_model.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const MoeMlpWeights = @import("../../sdk/blocks/moe/weights.zig").MoeMlpWeights;

pub const Outside = struct {
    a: std.mem.Allocator,
    /// The claiming table's load (closed at the load's end); every weight it loaded lives until `destroy`.
    load: outside_model.Load,
    /// The table's projection for each `.weight` the pack stores in its format, by that array's handle.
    linears: std.AutoHashMapUnmanaged(usize, outside.Weight) = .empty,
    /// The table's routed banks for each MoE block, by the block's bound gate bank (`switch_gate_w`).
    banks: std.AutoHashMapUnmanaged(usize, moe.FormatBanks) = .empty,

    /// The table claiming the pack at `ctx`, reading its tensors from `weights` (which must not move); null when no
    /// loaded table claims it.
    pub fn open(a: std.mem.Allocator, ctx: *const plugin_abi.LoadCtx, weights: *const model.Weights, s: mlx.mlx_stream) !?*Outside {
        var ld = try outside_model.Load.open(a, ctx, weights.source(), s);
        if (ld.format == null) {
            ld.deinit();
            return null;
        }
        errdefer ld.deinit();
        const o = try a.create(Outside);
        o.* = .{ .a = a, .load = ld };
        return o;
    }

    pub fn destroy(self: *Outside) void {
        self.linears.deinit(self.a);
        self.banks.deinit(self.a);
        self.load.deinit();
        self.a.destroy(self);
    }

    pub fn linearOf(self: *const Outside, w: mlx.mlx_array) ?*const outside.Weight {
        return self.linears.getPtr(@intFromPtr(w.ctx orelse return null));
    }

    pub fn banksOf(self: *const Outside, mw: *const MoeMlpWeights) ?*const moe.FormatBanks {
        return self.banks.getPtr(@intFromPtr(mw.switch_gate_w.ctx orelse return null));
    }

    /// Every `.weight` of `weights` in no MLX encoding, loaded on the table and keyed by that array; the count.
    /// Refused by name: one the trunk reads past `qmatmul`, one without 1-D side vectors (the layer binder would
    /// transpose or join it as an MLX weight), one the table does not read.
    pub fn bindLinears(self: *Outside, weights: *const model.Weights) !u32 {
        var n: u32 = 0;
        var it = weights.map.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            if (!std.mem.endsWith(u8, name, ".weight") or mlxEncoded(mlx.mlx_array_dtype(e.value_ptr.*))) continue;
            const base = name[0 .. name.len - ".weight".len];
            if (!projected(base)) {
                refused(name, "the trunk reads it past its projections");
                return error.Qwen4OutsideWeightUnserved;
            }
            var nb: [256]u8 = undefined;
            const sc = weights.get(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}));
            if (sc == null or mlx.mlx_array_ndim(sc.?) != 1) {
                refused(name, "it has no 1-D scales");
                return error.Qwen4OutsideWeightLayout;
            }
            const lin = (try self.load.linear(base, 0)) orelse {
                refused(name, "the quant plugin does not read it");
                return error.Qwen4OutsideWeightUnserved;
            };
            const slot = try self.linears.getOrPut(self.a, @intFromPtr(e.value_ptr.ctx));
            // Two names over one array are one linear.
            if (slot.found_existing) continue;
            slot.value_ptr.* = lin.ext.?;
            n += 1;
        }
        return n;
    }

    /// The table's routed banks `<base>.{gate,up,down}_proj` for the MoE block the binder bound as `mw`; false when
    /// the pack stores none in its format (the block keeps its own arms).
    pub fn bindBanks(self: *Outside, base: []const u8, mw: *const MoeMlpWeights, hidden: u32, inter: u32) !bool {
        const fb = (self.load.experts(base, hidden, inter) catch |e| {
            log.err("[qwen4] {s}: the quant plugin's banks: {s}\n", .{ base, @errorName(e) });
            return e;
        }) orelse return false;
        try self.banks.put(self.a, @intFromPtr(mw.switch_gate_w.ctx orelse return error.MissingWeight), fb);
        return true;
    }
};

fn refused(name: []const u8, why: []const u8) void {
    log.err("[qwen4] {s} is in an outside quant plugin's format, and {s}\n", .{ name, why });
}

/// `x` [B, S, D] through banks `fb`'s experts `inds` [B, S, K], weighted by `scores`, on the table (qwen4's SwiGLU is
/// unclamped: `unservedLayout`). A new array.
pub fn routed(fb: *const moe.FormatBanks, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const gate: outside.Weight = .{ .t = fb.table, .w = fb.gate };
    const up: outside.Weight = .{ .t = fb.table, .w = fb.up };
    const down: outside.Weight = .{ .t = fb.table, .w = fb.down };
    return outside.swiglu(&gate, &up, &down, x, inds, scores, 0, s);
}

/// MLX's own encodings of a `.weight`: a float table, the uint32 words of its quantized modes, the uint8 of e4m3 and
/// of raw ggml rows.
pub fn mlxEncoded(dtype: mlx.mlx_dtype) bool {
    return switch (dtype) {
        .float16, .bfloat16, .float32, .uint32, .uint8 => true,
        else => false,
    };
}

/// The first `.weight` of `weights` in no MLX encoding (a pack only an outside quant plugin reads); null when none.
pub fn firstUnencoded(weights: *const model.Weights) ?[]const u8 {
    var it = weights.map.iterator();
    while (it.next()) |e| if (std.mem.endsWith(u8, e.key_ptr.*, ".weight") and !mlxEncoded(mlx.mlx_array_dtype(e.value_ptr.*))) return e.key_ptr.*;
    return null;
}

/// The projections the trunk and its MTP head read through `qmatmul` alone, by their name's tail.
const projections = [_][]const u8{
    "self_attn.q_proj",            "self_attn.k_proj",            "self_attn.v_proj",
    "self_attn.o_proj",            "self_attn.indexer.index_qk_proj", "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",       "linear_attn.in_proj_a",       "linear_attn.in_proj_b",
    "linear_attn.in_proj_qkvz",    "linear_attn.in_proj_ba",      "linear_attn.out_proj",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",   "mlp.shared_expert.down_proj",
    "lm_head",                     "fc_embedding",                "fc_hidden",
};

fn projected(base: []const u8) bool {
    for (projections) |p| {
        if (std.mem.eql(u8, base, p)) return true;
        if (base.len > p.len and std.mem.endsWith(u8, base, p) and base[base.len - p.len - 1] == '.') return true;
    }
    return false;
}

/// The pack's f16 tables and side vectors narrowed to bf16, as the engine narrows every other pack's (a claimed pack
/// loads as stored); the table's own operands, 1-D `.scales` / `.biases`, stay f16.
pub fn narrowTables(weights: *model.Weights, s: mlx.mlx_stream) !void {
    var it = weights.map.iterator();
    while (it.next()) |e| {
        const v = e.value_ptr.*;
        const nd = mlx.mlx_array_ndim(v);
        if (!model.narrowsLoadedF16(e.key_ptr.*, nd, mlx.mlx_array_dtype(v)) or sideVector(e.key_ptr.*, nd)) continue;
        var narrowed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&narrowed, v, .bfloat16, s));
        _ = mlx.mlx_array_free(v);
        e.value_ptr.* = narrowed;
    }
}

fn sideVector(key: []const u8, ndim: usize) bool {
    return ndim == 1 and (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases"));
}

// ── tests ──

const testing = std.testing;
const quant_abi = @import("../../abi/quant.zig");
const common = @import("../../abi/common.zig");
const Trunk = @import("trunk.zig").Trunk;

/// Claims a `"toy"` block; loads any projection or bank but `bad*` (sharing the map's array), and runs both as a
/// copy of `x`, counted.
const Toy = struct {
    var matmuls: u32 = 0;
    var swiglus: u32 = 0;

    fn claims(block: quant_abi.Str, _: quant_abi.Str) callconv(.c) common.Claim {
        return if (std.mem.indexOf(u8, block.slice(), "\"toy\"") != null) .by_marker else .no;
    }
    fn validate(_: quant_abi.Str, _: quant_abi.Str, _: *const common.PackDir) callconv(.c) common.Verdict {
        return .ok;
    }
    fn bill(_: quant_abi.Str, _: quant_abi.Str, stored: u64) callconv(.c) u64 {
        return stored;
    }
    fn open(_: *quant_abi.LoadCtx) callconv(.c) common.Verdict {
        return .ok;
    }
    fn close(_: *quant_abi.LoadCtx) callconv(.c) void {}
    fn load(ctx: *const quant_abi.LoadCtx, name: quant_abi.Str, in_dim: u32, out: *quant_abi.Weight) callconv(.c) common.Verdict {
        const n = name.slice();
        if (std.mem.indexOf(u8, n, "bad") != null or std.mem.indexOf(u8, n, "none") != null) return common.Verdict.refuse(.failed, "MissingWeight");
        var nb: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&nb, "{s}.weight", .{n}) catch return common.Verdict.refuse(.failed, "ToyName");
        out.* = .{ .w = ctx.source.tensor(key), .block = in_dim };
        return .ok;
    }
    fn release(w: *quant_abi.Weight) callconv(.c) void {
        w.* = .{};
    }
    fn copy(x: quant_abi.Array, out: *quant_abi.Array) common.Verdict {
        var h = mlx.mlx_array_new();
        if (mlx.mlx_array_set(&h, .{ .ctx = x.ctx }) != 0) return common.Verdict.refuse(.failed, "ToyCopy");
        out.* = .{ .ctx = h.ctx };
        return .ok;
    }
    fn matmul(_: *const quant_abi.Weight, x: quant_abi.Array, _: quant_abi.Stream, out: *quant_abi.Array) callconv(.c) common.Verdict {
        matmuls += 1;
        return copy(x, out);
    }
    fn moeRun(_: *const quant_abi.Weight, _: quant_abi.Array, _: quant_abi.Array, _: quant_abi.Array, _: bool, _: quant_abi.Stream, _: *quant_abi.Array) callconv(.c) common.Verdict {
        return common.Verdict.refuse(.unsupported, "toy");
    }
    fn swiglu(_: *const quant_abi.Weight, _: *const quant_abi.Weight, _: *const quant_abi.Weight, x: quant_abi.Array, _: quant_abi.Array, _: quant_abi.Array, opts: *const quant_abi.Swiglu, _: quant_abi.Stream, out: *quant_abi.Array) callconv(.c) common.Verdict {
        if (opts.limit != 0) return common.Verdict.refuse(.failed, "ToyClamped");
        swiglus += 1;
        return copy(x, out);
    }
    const table: quant_abi.QuantTable = .{ .name = .of("toy"), .claims = &claims, .validate = &validate, .bill = &bill, .open = &open, .close = &close, .load_tensor = &load, .load_bank = &load, .release = &release, .matmul = &matmul, .moe = &moeRun, .swiglu = &swiglu };
};

fn toyOutside(weights: *const model.Weights) !*Outside {
    const f = try testing.allocator.create(outside.Format);
    f.* = .{ .a = testing.allocator, .t = &Toy.table, .config = try testing.allocator.alloc(u8, 0), .block = try testing.allocator.alloc(u8, 0), .source = weights.source(), .ctx = .{ .block_json = .{}, .source = undefined } };
    f.ctx.source = &f.source;
    const o = try testing.allocator.create(Outside);
    o.* = .{ .a = testing.allocator, .load = .{ .a = testing.allocator, .format = f } };
    return o;
}

fn putZeros(w: *model.Weights, name: []const u8, dt: mlx.mlx_dtype, shape: []const c_int) !mlx.mlx_array {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, s));
    try w.put(name, a);
    return a;
}

/// A trunk with nothing loaded but `o`: enough for its projections' routing.
fn bareTrunk(o: *Outside, s: mlx.mlx_stream) Trunk {
    const none: mlx.mlx_array = .{ .ctx = null };
    return .{ .config = .{}, .s = s, .allocator = testing.allocator, .emb_w = none, .emb_s = none, .emb_b = none, .lm_head_w = none, .lm_head_s = none, .lm_head_b = none, .moe_layers = &.{}, .qwen4_mixer = undefined, .ones_hidden = none, .one = none, .rms_eps_arr = none, .outside = o };
}

test "qwen4_exp outside: a trunk linear the pack stores in the plugin's format runs on its table, through qmatmul and the head" {
    var w = model.Weights.init(testing.allocator);
    defer w.deinit();
    const q = try putZeros(&w, "m.layers.3.self_attn.q_proj.weight", .uint16, &.{ 4, 4, 32 });
    _ = try putZeros(&w, "m.layers.3.self_attn.q_proj.scales", .float16, &.{64});
    const head = try putZeros(&w, "lm_head.weight", .uint16, &.{ 4, 4, 32 });
    _ = try putZeros(&w, "lm_head.scales", .float16, &.{64});
    const emb = try putZeros(&w, "m.embed_tokens.weight", .bfloat16, &.{ 8, 64 });
    const o = try toyOutside(&w);
    defer o.destroy();
    try testing.expectEqual(@as(u32, 2), try o.bindLinears(&w));
    o.load.close();
    try testing.expect(o.linearOf(q) != null and o.linearOf(head) != null and o.linearOf(emb) == null);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var t = bareTrunk(o, s);
    t.lm_head_w = head;
    t.lm_head_s = w.get("lm_head.scales").?;
    const x = try putZeros(&w, "x", .bfloat16, &.{ 1, 2, 64 });
    Toy.matmuls = 0;
    const y = try t.qmatmul(x, q, w.get("m.layers.3.self_attn.q_proj.scales").?, .{ .ctx = null });
    defer _ = mlx.mlx_array_free(y);
    const logits = try t.lmHeadProject(x);
    defer _ = mlx.mlx_array_free(logits);
    const rows = try t.mtpProjectionRows(x, q, w.get("m.layers.3.self_attn.q_proj.scales").?, .{ .ctx = null }, false);
    defer _ = mlx.mlx_array_free(rows);
    try testing.expectEqual(@as(u32, 3), Toy.matmuls);
}

test "qwen4_exp outside: a block's routed banks run on the table; a block it stores none of keeps its own arms" {
    var w = model.Weights.init(testing.allocator);
    defer w.deinit();
    const gate = try putZeros(&w, "m.layers.0.mlp.switch_mlp.gate_proj.trellis", .uint16, &.{ 4, 4, 4, 32 });
    const o = try toyOutside(&w);
    defer o.destroy();
    var mw: MoeMlpWeights = undefined;
    mw.switch_gate_w = gate;
    try testing.expect(try o.bindBanks("m.layers.0.mlp.switch_mlp", &mw, 64, 64));
    var other: MoeMlpWeights = undefined;
    other.switch_gate_w = try putZeros(&w, "m.layers.1.mlp.switch_mlp.gate_proj.trellis", .uint16, &.{ 4, 4, 4, 32 });
    try testing.expect(!try o.bindBanks("m.layers.1.none.switch_mlp", &other, 64, 64));
    try testing.expect(o.banksOf(&other) == null);
    // A copy of the block (the grouped verify's) finds the same banks.
    const copied = mw;
    const fb = o.banksOf(&copied).?;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const x = try putZeros(&w, "x", .bfloat16, &.{ 1, 1, 64 });
    Toy.swiglus = 0;
    const y = try routed(fb, x, x, x, s);
    defer _ = mlx.mlx_array_free(y);
    try testing.expectEqual(@as(u32, 1), Toy.swiglus);
}

test "qwen4_exp outside: a weight the trunk reads past qmatmul, without 1-D scales, or the table does not read is refused by name" {
    inline for (.{
        .{ "m.layers.0.attn_hyper_connection.input_mix_weight_down", true, error.Qwen4OutsideWeightUnserved },
        .{ "m.layers.0.self_attn.o_proj", false, error.Qwen4OutsideWeightLayout },
        .{ "m.layers.0.bad.self_attn.o_proj", true, error.Qwen4OutsideWeightUnserved },
    }) |c| {
        var w = model.Weights.init(testing.allocator);
        defer w.deinit();
        _ = try putZeros(&w, c[0] ++ ".weight", .uint16, &.{ 4, 4, 32 });
        if (c[1]) _ = try putZeros(&w, c[0] ++ ".scales", .float16, &.{64});
        const o = try toyOutside(&w);
        defer o.destroy();
        try testing.expectError(c[2], o.bindLinears(&w));
        try testing.expectEqualStrings(c[0] ++ ".weight", firstUnencoded(&w).?);
    }
}

test "qwen4_exp outside: a claimed pack's f16 tables narrow to bf16, the plugin's 1-D side vectors stay f16" {
    var w = model.Weights.init(testing.allocator);
    defer w.deinit();
    _ = try putZeros(&w, "m.norm.weight", .float16, &.{64});
    _ = try putZeros(&w, "m.q_proj.scales", .float16, &.{64});
    _ = try putZeros(&w, "m.embed_tokens.scales", .float16, &.{ 8, 2 });
    _ = try putZeros(&w, "m.ple.ple_embedding.head_bias", .float16, &.{ 2, 8 });
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try narrowTables(&w, s);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(w.get("m.norm.weight").?));
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(w.get("m.q_proj.scales").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(w.get("m.embed_tokens.scales").?));
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(w.get("m.ple.ple_embedding.head_bias").?));
}
