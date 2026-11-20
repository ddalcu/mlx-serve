//! hy_v4_flash (tencent/Hy4-flash) TEXT path, batch 1.
//!
//! Architecture source: huggingface/transformers PR #50903
//! (src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py). The text
//! trunk is DeepSeek-V4-flavoured (MLA, compressed indexer, sliding attention,
//! learnable sinks, hyper-connections, Engram, latent MoE) composed from
//! sdk/bases (Linear) and src/mlx/ops.zig (Graph).
//! All math here is f32, matching the reference and the NumPy oracle in
//! tests/hy4_flash_ref.py.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const engram = @import("../../sdk/weights/engram.zig");
const config = @import("config.zig");
const crt = @import("../../sdk/util/crt.zig");
const linear = @import("../../sdk/bases/blocks/linear.zig");
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;

pub const Config = config.Config;
pub const parseConfigJson = config.parseConfigJson;
pub const parseRealConfig = config.parseRealConfig;

const Array = ops.Array;
const Graph = ops.Graph;
const Stream = ops.Stream;
const Linear = linear.Linear;
const none: Array = .{ .ctx = null };

fn get(w: *const Weights, name: []const u8) !Array {
    return w.get(name) orelse error.MissingWeight;
}

fn getf(a: std.mem.Allocator, w: *const Weights, comptime fmt: []const u8, args: anytype) !Array {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    return get(w, name);
}

fn loadLinear(a: std.mem.Allocator, w: *const Weights, gs: u32, comptime fmt: []const u8, args: anytype) !Linear {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    const wt = try get(w, name);
    var lin = Linear{ .w = wt };
    if (mlx.mlx_array_dtype(wt) == .uint32 and std.mem.endsWith(u8, name, ".weight")) {
        const base = name[0 .. name.len - ".weight".len];
        const sn = try std.fmt.allocPrint(a, "{s}.scales", .{base});
        defer a.free(sn);
        const bn = try std.fmt.allocPrint(a, "{s}.biases", .{base});
        defer a.free(bn);
        if (w.get(sn)) |sc| if (w.get(bn)) |bi| {
            const wsh = mlx.getShape(wt);
            const ssh = mlx.getShape(sc);
            if (wsh.len >= 2 and ssh.len >= 2 and gs > 0) {
                const wcols: u32 = @intCast(wsh[wsh.len - 1]);
                const scols: u32 = @intCast(ssh[ssh.len - 1]);
                const in_dim = scols * gs;
                if (in_dim > 0 and (wcols * 32) % in_dim == 0) {
                    lin.s = sc;
                    lin.b = bi;
                    lin.bits = (wcols * 32) / in_dim;
                    lin.group_size = gs;
                    lin.mode = .affine;
                }
            }
        };
    }
    if (std.mem.endsWith(u8, name, ".weight")) {
        const base = name[0 .. name.len - ".weight".len];
        const bias_name = try std.fmt.allocPrint(a, "{s}.bias", .{base});
        defer a.free(bias_name);
        if (w.get(bias_name)) |b| {
            lin.bias = b;
        }
    }
    return lin;
}

fn rmsNormNw(g: *Graph, x: Array, eps: f32) !Array {
    const mean_sq = try g.mean(try g.square(x), -1, true);
    const inv = try g.rsqrt(try g.addScalar(mean_sq, eps));
    return g.mul(x, inv);
}

// ── RoPE ──────────────────────────────────────────────────────────────────

pub const Rope = struct { cos: Array, sin: Array };

fn buildRope(a: std.mem.Allocator, base: f64, rope_dim: usize, mrope: [3]usize, upto: usize) !Rope {
    const half = rope_dim / 2;
    const cs = try a.alloc(f32, upto * rope_dim);
    defer a.free(cs);
    const sn = try a.alloc(f32, upto * rope_dim);
    defer a.free(sn);

    for (0..upto) |pos| {
        for (0..half) |j| {
            const freq = @as(f64, @floatFromInt(pos)) / std.math.pow(f64, base, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(rope_dim)));
            const c: f32 = @floatCast(@cos(freq));
            const si: f32 = @floatCast(@sin(freq));
            cs[pos * rope_dim + 2 * j] = c;
            cs[pos * rope_dim + 2 * j + 1] = c;
            sn[pos * rope_dim + 2 * j] = si;
            sn[pos * rope_dim + 2 * j + 1] = si;
        }
        for (1..3) |axis| {
            const len = mrope[axis] * 3;
            var idx = axis;
            while (idx < len and idx < half) : (idx += 3) {
                const f1 = @as(f64, @floatFromInt(pos)) / std.math.pow(f64, base, @as(f64, @floatFromInt(2 * idx)) / @as(f64, @floatFromInt(rope_dim)));
                cs[pos * rope_dim + 2 * idx] = @floatCast(@cos(f1));
                cs[pos * rope_dim + 2 * idx + 1] = @floatCast(@cos(f1));
                sn[pos * rope_dim + 2 * idx] = @floatCast(@sin(f1));
                sn[pos * rope_dim + 2 * idx + 1] = @floatCast(@sin(f1));
            }
        }
    }

    const sh = [_]c_int{ @intCast(upto), @intCast(rope_dim) };
    const cos = mlx.mlx_array_new_data(cs.ptr, &sh, 2, .float32);
    const sin = mlx.mlx_array_new_data(sn.ptr, &sh, 2, .float32);
    return .{ .cos = cos, .sin = sin };
}

fn rotateHalf(g: *Graph, x: Array) !Array {
    const sh = ops.shape(x);
    const d = sh[sh.len - 1];
    const half = @divExact(d, 2);
    const reshaped = try g.reshape(x, &.{ -1, half, 2 });
    const x1 = try g.sliceAxis(reshaped, 2, 0, 1);
    const x2 = try g.sliceAxis(reshaped, 2, 1, 2);
    const neg_x2 = try g.neg(x2);
    const swapped = try g.concat(&.{ neg_x2, x1 }, 2);
    return g.reshape(swapped, sh);
}

fn applyRope(g: *Graph, x: Array, cos: Array, sin: Array) !Array {
    const rx = try rotateHalf(g, x);
    return g.add(try g.mul(x, cos), try g.mul(rx, sin));
}

fn applyPartialRope(g: *Graph, x: Array, cos: Array, sin: Array, rope_dim: usize) !Array {
    const sh = ops.shape(x);
    const d = sh[sh.len - 1];
    const passthru_dim: c_int = @intCast(d - @as(c_int, @intCast(rope_dim)));
    const pass = try g.sliceAxis(x, @intCast(sh.len - 1), 0, passthru_dim);
    const rot = try g.sliceAxis(x, @intCast(sh.len - 1), passthru_dim, d);
    const roped = try applyRope(g, rot, cos, sin);
    return g.concat(&.{ pass, roped }, @intCast(sh.len - 1));
}

// ── Hyper-Connections ─────────────────────────────────────────────────────

fn hcMix(
    g: *Graph,
    streams: Array,
    hc_fn: *const Linear,
    base: Array,
    scale: Array,
    hc_mult: usize,
    eps: f32,
    magnitude: f32,
) !struct { post: Array, out: Array } {
    const sh = ops.shape(streams);
    const T = sh[0];
    const flat = try g.reshape(streams, &.{ T, -1 });
    const fn_proj = try hc_fn.forward(g, flat);
    const mean_sq = try g.mean(try g.square(flat), -1, true);
    const inv = try g.rsqrt(try g.addScalar(mean_sq, eps));
    const mixes = try g.mul(fn_proj, inv);

    const hc_i: c_int = @intCast(hc_mult);
    const pre_l = try g.sliceAxis(mixes, 1, 0, hc_i);
    const post_l = try g.sliceAxis(mixes, 1, hc_i, 2 * hc_i);

    const scale0 = try g.sliceAxis(scale, 0, 0, 1);
    const scale1 = try g.sliceAxis(scale, 0, 1, 2);
    const base_pre = try g.sliceAxis(base, 0, 0, hc_i);
    const base_post = try g.sliceAxis(base, 0, hc_i, 2 * hc_i);

    const pre_sig = try g.sigmoid(try g.add(try g.mul(pre_l, scale0), base_pre));
    const pre = try g.addScalar(pre_sig, eps);
    const post_sig = try g.sigmoid(try g.add(try g.mul(post_l, scale1), base_post));
    const post = try g.addScalar(try g.mulScalar(post_sig, magnitude), eps);

    const pre_exp = try g.expandDims(pre, 2);
    const out = try g.sum(try g.mul(pre_exp, streams), 1, false);
    return .{ .post = post, .out = out };
}

// ── Layer structs ─────────────────────────────────────────────────────────

const Layer = struct {
    attn_norm: Array = none,
    ffn_norm: Array = none,
    q_a: Linear = .{},
    q_norm: Array = none,
    q_b: Linear = .{},
    kv_a: Linear = .{},
    kv_norm: Array = none,
    kv_b: Linear = .{},
    gate: Linear = .{},
    o: Linear = .{},
    post_norm: Array = none,
    sink: Array = none,
    hc_attn_fn: Linear = .{},
    hc_attn_base: Array = none,
    hc_attn_scale: Array = none,
    hc_ffn_fn: Linear = .{},
    hc_ffn_base: Array = none,
    hc_ffn_scale: Array = none,
    // Compressed indexer
    idx_q: Linear = .{},
    idx_hw: Linear = .{},
    idx_tk: Linear = .{},
    idx_tk_norm: Array = none,
    idx_kv: Linear = .{},
    idx_gate: Linear = .{},
    idx_norm: Array = none,
    idx_ape: Array = none,
    // Dense MLP
    dense_gate: Linear = .{},
    dense_up: Linear = .{},
    dense_down: Linear = .{},
    dense_post: Array = none,
    // MoE MLP
    moe_gate: Linear = .{},
    moe_bias: Array = none,
    routed_down: Linear = .{},
    routed_up: Linear = .{},
    gu_bank: Linear = .{},
    dn_bank: Linear = .{},
    shared_gate: Linear = .{},
    shared_up: Linear = .{},
    shared_down: Linear = .{},
    combine_norm: Array = none,
};

const Engram = struct {
    layer_idx: usize,
    lookup_table: ?Array = null,
    hash_mult: ?Array = null,
    sizes: ?Array = null,
    embed: ?Linear = null,
    value_proj: Linear = .{},
    key_projs: []Linear = &.{},
    norm_key: []Array = &.{},
    norm_query: []Array = &.{},
    lookup_per_head_norms: []Array = &.{},
    unified_pre_gamma: Array = none,
    unified_per_hc_scales: []Array = &.{},
    unified_rmsnorm: Array = none,
};

// ── State ─────────────────────────────────────────────────────────────────

const LayerCache = struct {
    kp: ?Array = null,
    kr: ?Array = null,
    tk: ?Array = null,

    fn clear(self: *LayerCache) void {
        if (self.kp) |x| _ = mlx.mlx_array_free(x);
        if (self.kr) |x| _ = mlx.mlx_array_free(x);
        if (self.tk) |x| _ = mlx.mlx_array_free(x);
        self.* = .{};
    }
};

pub const State = struct {
    a: std.mem.Allocator,
    s: Stream,
    caches: []LayerCache,
    len: usize = 0,
    hist: []u32 = &.{},

    pub fn init(a: std.mem.Allocator, n: usize, s: Stream, pad: u32, history: usize) !State {
        const cs = try a.alloc(LayerCache, n);
        @memset(cs, .{});
        const h = try a.alloc(u32, @max(1, history));
        @memset(h, pad);
        return .{ .a = a, .s = s, .caches = cs, .hist = h };
    }

    pub fn deinit(self: *State) void {
        for (self.caches) |*c| c.clear();
        self.a.free(self.caches);
        self.a.free(self.hist);
    }

    pub fn reset(self: *State) void {
        for (self.caches) |*c| c.clear();
        self.len = 0;
    }
};



pub const HeaderRegion = struct {
    start: usize,
    rows: usize,
    cols: usize,
};

pub fn parseHeaderRegion(map: std.json.ObjectMap, name: []const u8, elem_size: usize) !HeaderRegion {
    const v = map.get(name) orelse return error.EngramTableHeader;
    if (v != .object) return error.EngramTableHeader;
    const offsets = v.object.get("data_offsets") orelse return error.EngramTableHeader;
    if (offsets != .array or offsets.array.items.len != 2) return error.EngramTableHeader;
    const start: usize = @intCast(offsets.array.items[0].integer);
    const end: usize = @intCast(offsets.array.items[1].integer);
    if (start >= end) return error.EngramTableHeader;
    const shape = v.object.get("shape") orelse return error.EngramTableHeader;
    if (shape != .array or shape.array.items.len != 2) return error.EngramTableHeader;
    const rows: usize = @intCast(shape.array.items[0].integer);
    const cols: usize = @intCast(shape.array.items[1].integer);
    if (rows * cols * elem_size != end - start) return error.EngramTableHeader;
    return .{ .start = start, .rows = rows, .cols = cols };
}

pub fn openHy4(a: std.mem.Allocator, dir_path: []const u8, spec: *const config.Hy4Spec) !engram.Table {
    if (comptime builtin.os.tag == .windows) return error.EngramUnsupportedOnWindows;
    if (spec.bits != 4 or spec.group_size != 32 or spec.head_dim == 0 or spec.head_dim % 32 != 0 or spec.rows == 0) return error.EngramTableHeader;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const path = try std.fs.path.join(aa, &.{ dir_path, spec.file });
    const fd = crt.openRead(@ptrCast(path.ptr));
    if (fd < 0) return error.EngramTableMissing;
    errdefer crt.close(fd);
    var hl: [8]u8 = undefined;
    if (crt.pread(fd, &hl, 8, 0) != 8) return error.EngramTableHeader;
    const hlen = std.mem.readInt(u64, &hl, .little);
    if (hlen == 0 or hlen > 16 << 20) return error.EngramTableHeader;
    const hbuf = try aa.alloc(u8, @intCast(hlen));
    if (crt.pread(fd, hbuf.ptr, hbuf.len, 8) != @as(isize, @intCast(hlen))) return error.EngramTableHeader;
    const hdr = std.json.parseFromSliceLeaky(std.json.Value, aa, hbuf, .{}) catch return error.EngramTableHeader;
    if (hdr != .object) return error.EngramTableHeader;
    const base: usize = 8 + @as(usize, @intCast(hlen));
    const wcols: u32 = spec.head_dim * spec.bits / 32;
    const scols: u32 = spec.head_dim / spec.group_size;
    const w = try parseHeaderRegion(hdr.object, "weight", 4);
    const s = try parseHeaderRegion(hdr.object, "scales", 2);
    const b = try parseHeaderRegion(hdr.object, "biases", 2);
    if (w.rows != spec.rows or s.rows != spec.rows or b.rows != spec.rows) return error.EngramTableSize;
    if (w.cols != wcols or s.cols != scols or b.cols != scols) return error.EngramTableGeometry;
    return .{
        .fd = fd,
        .rows = spec.rows,
        .head_dim = spec.head_dim,
        .mode = .affine,
        .bits = spec.bits,
        .group_size = spec.group_size,
        .w_off = base + w.start,
        .w_stride = @as(u64, wcols) * 4,
        .wlen = @as(usize, wcols) * 4,
        .s_off = base + s.start,
        .s_stride = @as(u64, scols) * 2,
        .slen = @as(usize, scols) * 2,
        .b_off = base + b.start,
    };
}

// ── Model ─────────────────────────────────────────────────────────────────

pub const Model = struct {
    a: std.mem.Allocator,
    s: Stream,
    cfg: Config,
    w: *const Weights,
    embed: Linear,
    head: Linear,
    norm: Array,
    hc_head: Linear,
    hc_base: Array,
    hc_scale: Array,
    ext_table: ?engram.Table = null,
    layers: []Layer,
    engrams: []Engram,
    rope_indexed: Rope,
    rope_swa: Rope,
    state: State,

    pub fn init(a: std.mem.Allocator, cfg: Config, w: *const Weights, s: Stream) !Model {
        var layers = try a.alloc(Layer, cfg.layers);
        for (layers) |*l| l.* = .{};
        for (0..cfg.layers) |li| {
            const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{li});
            defer a.free(p);
            const ax = try std.fmt.allocPrint(a, "{s}self_attn.", .{p});
            defer a.free(ax);
            const hc = try std.fmt.allocPrint(a, "{s}hc_attn_layer.hc_pre.", .{p});
            defer a.free(hc);
            const hm = try std.fmt.allocPrint(a, "{s}hc_mlp_layer.hc_pre.", .{p});
            defer a.free(hm);
            const l = &layers[li];
            l.attn_norm = try getf(a, w, "{s}input_layernorm.weight", .{p});
            l.ffn_norm = try getf(a, w, "{s}post_attention_layernorm.weight", .{p});
            l.q_a = try loadLinear(a, w, cfg.quant_group_size, "{s}q_a_proj.weight", .{ax});
            l.q_norm = try getf(a, w, "{s}q_a_layernorm.weight", .{ax});
            l.q_b = try loadLinear(a, w, cfg.quant_group_size, "{s}q_b_proj.weight", .{ax});
            l.kv_a = try loadLinear(a, w, cfg.quant_group_size, "{s}kv_a_proj_with_mqa.weight", .{ax});
            l.kv_norm = try getf(a, w, "{s}kv_a_layernorm.weight", .{ax});
            l.kv_b = try loadLinear(a, w, cfg.quant_group_size, "{s}kv_b_proj.weight", .{ax});
            l.gate = try loadLinear(a, w, cfg.quant_group_size, "{s}linear_gate.weight", .{ax});
            l.o = try loadLinear(a, w, cfg.quant_group_size, "{s}o_proj.weight", .{ax});
            l.post_norm = try getf(a, w, "{s}post_proj_norm.weight", .{ax});
            l.sink = try getf(a, w, "{s}learnable_sink_param", .{ax});
            l.hc_attn_fn = try loadLinear(a, w, cfg.quant_group_size, "{s}hc_fn", .{hc});
            l.hc_attn_base = try getf(a, w, "{s}hc_base", .{hc});
            l.hc_attn_scale = try getf(a, w, "{s}hc_scale", .{hc});
            l.hc_ffn_fn = try loadLinear(a, w, cfg.quant_group_size, "{s}hc_fn", .{hm});
            l.hc_ffn_base = try getf(a, w, "{s}hc_base", .{hm});
            l.hc_ffn_scale = try getf(a, w, "{s}hc_scale", .{hm});

            if (cfg.isIndexed(li)) {
                const ip = try std.fmt.allocPrint(a, "{s}compressed_indexer.", .{ax});
                defer a.free(ip);
                l.idx_q = try loadLinear(a, w, cfg.quant_group_size, "{s}query_projection.weight", .{ip});
                l.idx_hw = try loadLinear(a, w, cfg.quant_group_size, "{s}head_weight_projection.weight", .{ip});
                l.idx_tk = try loadLinear(a, w, cfg.quant_group_size, "{s}token_key_projection.weight", .{ip});
                l.idx_tk_norm = try getf(a, w, "{s}token_key_norm.weight", .{ip});
                l.idx_kv = try loadLinear(a, w, cfg.quant_group_size, "{s}key_value_projection.weight", .{ip});
                l.idx_gate = try loadLinear(a, w, cfg.quant_group_size, "{s}key_gate_projection.weight", .{ip});
                l.idx_norm = try getf(a, w, "{s}key_norm.weight", .{ip});
                l.idx_ape = try getf(a, w, "{s}intra_group_ape", .{ip});
            }

            if (cfg.isDense(li)) {
                l.dense_gate = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.gate_proj.weight", .{p});
                l.dense_up = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.up_proj.weight", .{p});
                l.dense_down = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.down_proj.weight", .{p});
                l.dense_post = try getf(a, w, "{s}mlp.post_fc2_norm.weight", .{p});
            } else {
                l.moe_gate = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.gate.weight", .{p});
                l.moe_bias = try getf(a, w, "{s}mlp.gate.e_score_correction_bias", .{p});
                l.routed_down = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.routed_expert_down_proj.weight", .{p});
                l.routed_up = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.routed_expert_up_proj.weight", .{p});
                l.gu_bank = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.experts.gate_up_proj.weight", .{p});
                l.dn_bank = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.experts.down_proj.weight", .{p});
                l.shared_gate = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.shared_experts.gate_proj.weight", .{p});
                l.shared_up = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.shared_experts.up_proj.weight", .{p});
                l.shared_down = try loadLinear(a, w, cfg.quant_group_size, "{s}mlp.shared_experts.down_proj.weight", .{p});
                l.combine_norm = try getf(a, w, "{s}mlp.combine_output_norm.weight", .{p});
            }
        }

        var engrams = try a.alloc(Engram, cfg.engram_layers.len);
        for (cfg.engram_layers, 0..) |eli, k| {
            const ep = try std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.", .{eli});
            defer a.free(ep);
            const eg = &engrams[k];
            eg.layer_idx = eli;
            eg.lookup_table = w.get(try std.fmt.allocPrint(a, "{s}lookup_table", .{ep}));
            eg.hash_mult = w.get(try std.fmt.allocPrint(a, "{s}hash_multipliers", .{ep}));
            eg.sizes = w.get(try std.fmt.allocPrint(a, "{s}head_vocab_sizes", .{ep}));
            const ew_name = try std.fmt.allocPrint(a, "{s}multi_head_embedding.embedding.weight", .{ep});
            defer a.free(ew_name);
            if (w.get(ew_name)) |_| {
                eg.embed = try loadLinear(a, w, cfg.quant_group_size, "{s}multi_head_embedding.embedding.weight", .{ep});
            } else {
                eg.embed = null;
            }
            eg.value_proj = try loadLinear(a, w, cfg.quant_group_size, "{s}value_proj.weight", .{ep});
            var kps = try a.alloc(Linear, cfg.hc_mult);
            var nks = try a.alloc(Array, cfg.hc_mult);
            var nqs = try a.alloc(Array, cfg.hc_mult);
            var scs = try a.alloc(Array, cfg.hc_mult);
            for (0..cfg.hc_mult) |h| {
                kps[h] = try loadLinear(a, w, cfg.quant_group_size, "{s}key_projs.{d}.weight", .{ ep, h });
                nks[h] = try getf(a, w, "{s}norm_key.{d}.weight", .{ ep, h });
                nqs[h] = try getf(a, w, "{s}norm_query.{d}.weight", .{ ep, h });
                scs[h] = try getf(a, w, "{s}unified_per_hc_scales.{d}", .{ ep, h });
            }
            eg.key_projs = kps;
            eg.norm_key = nks;
            eg.norm_query = nqs;
            eg.unified_per_hc_scales = scs;
            eg.unified_pre_gamma = try getf(a, w, "{s}unified_pre_gamma", .{ep});
            eg.unified_rmsnorm = (w.get(try std.fmt.allocPrint(a, "{s}unified_rmsnorm", .{ep}))) orelse none;

            const nh = if (cfg.engram_sizes.len > k) cfg.engram_sizes[k].len else 2;
            var phn = try a.alloc(Array, nh);
            for (0..nh) |h| {
                phn[h] = try getf(a, w, "{s}lookup_per_head_norms.{d}.weight", .{ ep, h });
            }
            eg.lookup_per_head_norms = phn;
        }

        return .{
            .a = a,
            .s = s,
            .cfg = cfg,
            .w = w,
            .embed = try loadLinear(a, w, cfg.quant_group_size, "model.embed_tokens.weight", .{}),
            .head = try loadLinear(a, w, cfg.quant_group_size, "lm_head.weight", .{}),
            .norm = try get(w, "model.norm.weight"),
            .hc_head = try loadLinear(a, w, cfg.quant_group_size, "model.hc_head.hc_head_fn", .{}),
            .hc_base = try get(w, "model.hc_head.hc_head_base"),
            .hc_scale = try get(w, "model.hc_head.hc_head_scale"),
            .layers = layers,
            .engrams = engrams,
            .rope_indexed = try buildRope(a, cfg.indexed_theta, cfg.rope_dim, cfg.mrope, 4096),
            .rope_swa = try buildRope(a, cfg.swa_theta, cfg.rope_dim, cfg.mrope, 4096),
            .state = try State.init(a, cfg.layers, s, @intCast(cfg.pad), cfg.max_ngram),
        };
    }

    pub fn deinit(self: *Model) void {
        self.state.deinit();
        if (self.ext_table) |*t| t.close();
        _ = mlx.mlx_array_free(self.rope_indexed.cos);
        _ = mlx.mlx_array_free(self.rope_indexed.sin);
        _ = mlx.mlx_array_free(self.rope_swa.cos);
        _ = mlx.mlx_array_free(self.rope_swa.sin);
        for (self.engrams) |eg| {
            self.a.free(eg.key_projs);
            self.a.free(eg.norm_key);
            self.a.free(eg.norm_query);
            self.a.free(eg.unified_per_hc_scales);
            self.a.free(eg.lookup_per_head_norms);
        }
        self.a.free(self.engrams);
        self.a.free(self.layers);
    }



    pub fn openExternalTable(self: *Model, dir_path: []const u8) !void {
        if (self.cfg.engram_table) |*spec| {
            self.ext_table = try openHy4(self.a, dir_path, spec);
        }
    }

    pub fn resetState(self: *Model) void {
        self.state.reset();
    }

    pub fn forward(self: *Model, tokens: []const u32) !Array {
        return self.forwardAt(tokens, self.state.len);
    }

    pub fn forwardAt(self: *Model, tokens: []const u32, p0: usize) !Array {
        var g = Graph.init(self.a, self.s);
        defer g.deinit();

        const T: c_int = @intCast(tokens.len);
        const tokens_i32 = try self.a.alloc(i32, tokens.len);
        defer self.a.free(tokens_i32);
        for (tokens, 0..) |tok, i| tokens_i32[i] = @intCast(tok);
        const tok_arr = try g.fromSlice(i32, tokens_i32, &.{T});

        var streams = try g.take(self.embed.w, tok_arr, 0);
        streams = try g.repeat(try g.expandDims(streams, 1), @intCast(self.cfg.hc_mult), 1);

        const pos = try self.a.alloc(i32, tokens.len);
        defer self.a.free(pos);
        for (0..tokens.len) |i| pos[i] = @intCast(p0 + i);

        for (0..self.cfg.layers) |li| {
            if (self.findEngram(li)) |eg| {
                const engram_add = try self.forwardEngram(&g, eg, streams, tokens, p0);
                streams = try g.add(streams, engram_add);
            }

            const l = &self.layers[li];
            const mix_attn = try hcMix(&g, streams, &l.hc_attn_fn, l.hc_attn_base, l.hc_attn_scale, self.cfg.hc_mult, self.cfg.eps, self.cfg.hc_magnitude);
            const x_attn_in = try g.rmsNorm(mix_attn.out, l.attn_norm, self.cfg.eps);
            const x_attn_out = try self.forwardAttention(&g, li, x_attn_in, pos);
            const attn_inj = try g.mul(try g.expandDims(mix_attn.post, 2), try g.expandDims(x_attn_out, 1));
            streams = try g.add(attn_inj, streams);

            const mix_ffn = try hcMix(&g, streams, &l.hc_ffn_fn, l.hc_ffn_base, l.hc_ffn_scale, self.cfg.hc_mult, self.cfg.eps, self.cfg.hc_magnitude);
            const x_ffn_in = try g.rmsNorm(mix_ffn.out, l.ffn_norm, self.cfg.eps);
            const x_ffn_out = try self.forwardMlp(&g, li, x_ffn_in);
            const ffn_inj = try g.mul(try g.expandDims(mix_ffn.post, 2), try g.expandDims(x_ffn_out, 1));
            streams = try g.add(ffn_inj, streams);
        }

        const flat = try g.reshape(streams, &.{ T, -1 });
        const fn_proj = try self.hc_head.forward(&g, flat);
        const mean_sq = try g.mean(try g.square(flat), -1, true);
        const inv = try g.rsqrt(try g.addScalar(mean_sq, self.cfg.eps));
        const mixes = try g.mul(fn_proj, inv);
        const pre = try g.addScalar(try g.sigmoid(try g.add(try g.mul(mixes, self.hc_scale), self.hc_base)), self.cfg.eps);
        const normed_out = try g.sum(try g.mul(try g.expandDims(pre, 2), streams), 1, false);
        const final_norm = try g.rmsNorm(normed_out, self.norm, self.cfg.eps);
        const logits = try self.head.forward(&g, final_norm);

        self.state.len += tokens.len;
        return g.escape(logits);
    }

    fn findEngram(self: *const Model, li: usize) ?*const Engram {
        for (self.engrams) |*eg| {
            if (eg.layer_idx == li) return eg;
        }
        return null;
    }

    fn forwardAttention(self: *Model, g: *Graph, li: usize, x: Array, pos: []const i32) !Array {
        const c = self.cfg;
        const l = &self.layers[li];
        const indexed = c.isIndexed(li);
        const heads: c_int = @intCast(if (indexed) c.indexed_heads else c.swa_heads);
        const nope: c_int = @intCast(if (indexed) c.qk_nope else c.swa_qk_nope);
        const vd: c_int = @intCast(if (indexed) c.v_dim else c.swa_v_dim);
        const rope_dim: c_int = @intCast(c.rope_dim);
        const qd = nope + rope_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qd)));
        const T: c_int = ops.shape(x)[0];

        const q_resid = try g.rmsNorm(try l.q_a.forward(g, x), l.q_norm, c.eps);
        const q_full = try g.reshape(try l.q_b.forward(g, q_resid), &.{ T, heads, qd });
        const gate = try g.reshape(try l.gate.forward(g, x), &.{ T, heads, vd });

        const compressed = try l.kv_a.forward(g, x);
        const kv_lora: c_int = @intCast(c.kv_lora);
        const kv_pass = try g.sliceAxis(compressed, 1, 0, kv_lora);
        const kp = try g.rmsNorm(kv_pass, l.kv_norm, c.eps);
        const kr = try g.sliceAxis(compressed, 1, kv_lora, kv_lora + rope_dim);

        const rp = if (indexed) self.rope_indexed else self.rope_swa;
        const p_start = pos[0];
        const p_end = pos[pos.len - 1] + 1;
        const cos_slice = try g.sliceAxis(rp.cos, 0, p_start, p_end);
        const sin_slice = try g.sliceAxis(rp.sin, 0, p_start, p_end);
        const cos_exp = try g.expandDims(cos_slice, 1);
        const sin_exp = try g.expandDims(sin_slice, 1);

        const q_nope = try g.sliceAxis(q_full, 2, 0, nope);
        const q_rot = try g.sliceAxis(q_full, 2, nope, qd);
        const qr = try applyRope(g, q_rot, cos_exp, sin_exp);
        const kr_roped = try applyRope(g, kr, cos_slice, sin_slice);

        const st = &self.state.caches[li];
        var all_kp: Array = undefined;
        var all_kr: Array = undefined;
        if (st.kp == null) {
            all_kp = try g.keep(kp);
            all_kr = try g.keep(kr_roped);
            st.kp = g.escape(kp);
            st.kr = g.escape(kr_roped);
        } else {
            const prev_kp = try g.adopt(st.kp.?);
            const prev_kr = try g.adopt(st.kr.?);
            all_kp = try g.concat(&.{ prev_kp, kp }, 0);
            all_kr = try g.concat(&.{ prev_kr, kr_roped }, 0);
            st.kp = g.escape(all_kp);
            st.kr = g.escape(all_kr);
        }

        const S: c_int = ops.shape(all_kp)[0];
        const kv = try g.reshape(try l.kv_b.forward(g, all_kp), &.{ S, heads, nope + vd });
        const kn = try g.sliceAxis(kv, 2, 0, nope);
        const v = try g.sliceAxis(kv, 2, nope, nope + vd);
        const kr_rep = try g.repeat(try g.expandDims(all_kr, 1), heads, 1);
        const k = try g.concat(&.{ kn, kr_rep }, 2);

        const q = try g.concat(&.{ q_nope, qr }, 2);
        const q_t = try g.transpose(q, &.{ 1, 0, 2 });
        const k_t = try g.transpose(k, &.{ 1, 2, 0 });
        const scores = try g.mulScalar(try g.matmul(q_t, k_t), scale);

        // Causal and sliding masking
        const q_pos_base = S - T;
        var scores_mod = scores;
        if (!indexed and c.sliding > 0) {
            const sw: c_int = @intCast(c.sliding);
            var mask_data = try self.a.alloc(f32, @intCast(T * S));
            defer self.a.free(mask_data);
            for (0..@intCast(T)) |ti| {
                const qp = q_pos_base + @as(c_int, @intCast(ti));
                for (0..@intCast(S)) |si| {
                    const kp_idx: c_int = @intCast(si);
                    const valid = kp_idx <= qp and kp_idx >= qp - sw + 1;
                    mask_data[ti * @as(usize, @intCast(S)) + si] = if (valid) 0.0 else -1e9;
                }
            }
            const mask_arr = try g.fromSlice(f32, mask_data, &.{ 1, T, S });
            scores_mod = try g.add(scores_mod, mask_arr);
        } else {
            var mask_data = try self.a.alloc(f32, @intCast(T * S));
            defer self.a.free(mask_data);
            for (0..@intCast(T)) |ti| {
                const qp = q_pos_base + @as(c_int, @intCast(ti));
                for (0..@intCast(S)) |si| {
                    mask_data[ti * @as(usize, @intCast(S)) + si] = if (@as(c_int, @intCast(si)) <= qp) 0.0 else -1e9;
                }
            }
            const mask_arr = try g.fromSlice(f32, mask_data, &.{ 1, T, S });
            scores_mod = try g.add(scores_mod, mask_arr);
        }

        // Softmax with learnable sinks
        const sink_3d = try g.reshape(l.sink, &.{ heads, 1, 1 });
        const max_scores = try g.max(scores_mod, 2, true);
        const mx = try g.maximum(max_scores, sink_3d);
        const ex = try g.exp(try g.sub(scores_mod, mx));
        const es = try g.exp(try g.sub(sink_3d, mx));
        const denom = try g.add(try g.sum(ex, 2, true), es);
        const probs = try g.div(ex, denom);

        const v_t = try g.transpose(v, &.{ 1, 0, 2 });
        const o_attn = try g.matmul(probs, v_t);
        const o_t = try g.transpose(o_attn, &.{ 1, 0, 2 });
        const o_gated = try g.mul(o_t, try g.sigmoid(gate));
        const o_flat = try g.reshape(o_gated, &.{ T, -1 });
        const o_proj = try l.o.forward(g, o_flat);
        return g.rmsNorm(o_proj, l.post_norm, c.eps);
    }

    fn forwardMlp(self: *Model, g: *Graph, li: usize, x: Array) !Array {
        const c = self.cfg;
        const l = &self.layers[li];
        if (c.isDense(li)) {
            const h = try l.dense_gate.forward(g, x);
            const u = try l.dense_up.forward(g, x);
            const act = try g.mul(try g.mul(h, try g.sigmoid(h)), u);
            const y = try l.dense_down.forward(g, act);
            return g.rmsNorm(y, l.dense_post, c.eps);
        }

        const scores = try l.moe_gate.forward(g, x);
        const sig = try g.sigmoid(scores);
        const sc = try g.add(sig, l.moe_bias);

        const T = ops.shape(x)[0];
        const neg_sc = try g.neg(sc);
        const sorted = try g.argsort(neg_sc, 1);
        const top_k: c_int = @intCast(c.top_k);
        const top = try g.sliceAxis(sorted, 1, 0, top_k);

        const tw_raw = try g.takeAlong(sig, top, 1);
        const tw_sum = try g.sum(tw_raw, 1, true);
        const tw = try g.mulScalar(try g.div(tw_raw, tw_sum), c.route_scale);

        const latent = try l.routed_down.forward(g, x);
        const routed_out = try self.forwardMoEExperts(g, l, latent, top, tw);

        const sh_h = try l.shared_gate.forward(g, x);
        const sh_u = try l.shared_up.forward(g, x);
        const sh_act = try g.mul(try g.mul(sh_h, try g.sigmoid(sh_h)), sh_u);
        const shared_out = try l.shared_down.forward(g, sh_act);

        const combined = try g.add(routed_out, shared_out);
        const normed = try rmsNormNw(g, combined, c.eps);
        _ = T;
        return g.mul(normed, l.combine_norm);
    }

    fn forwardMoEExperts(self: *Model, g: *Graph, l: *const Layer, latent: Array, top: Array, tw: Array) !Array {
        const c = self.cfg;
        const T: usize = @intCast(ops.shape(latent)[0]);
        const k: usize = c.top_k;
        const inter: c_int = @intCast(c.moe_inter);
        const lat_dim: c_int = @intCast(c.latent);
        const lim = c.swiglu_limit;

        // Flatten top to host for index lookups
        var eval_arr = [_]Array{ top, tw };
        try ops.evalMany(&eval_arr);
        const top_i32 = mlx.mlx_array_data_int32(top).?;
        const tw_f32 = mlx.mlx_array_data_float32(tw).?;

        var out_accum: ?Array = null;
        for (0..T) |ti| {
            const x_t = try g.sliceAxis(latent, 0, @intCast(ti), @intCast(ti + 1));
            var t_sum: ?Array = null;
            for (0..k) |j| {
                const expert_idx: c_int = top_i32[ti * k + j];
                const weight = tw_f32[ti * k + j];
                const gu_w = try g.sliceAxis(l.gu_bank.w, 0, expert_idx, expert_idx + 1);
                const gu_2d = try g.reshape(gu_w, &.{ 2 * inter, lat_dim });
                const gu = try g.matmul(x_t, try g.transpose(gu_2d, &.{ 1, 0 }));
                const gate = try g.sliceAxis(gu, 1, 0, inter);
                const up = try g.sliceAxis(gu, 1, inter, 2 * inter);
                const g_clamped = try g.minimum(gate, try g.scalarLike(gate, lim));
                const u_clamped = try g.maximum(try g.minimum(up, try g.scalarLike(up, lim)), try g.scalarLike(up, -lim));
                const act = try g.mul(try g.mul(g_clamped, try g.sigmoid(g_clamped)), u_clamped);

                const dn_w = try g.sliceAxis(l.dn_bank.w, 0, expert_idx, expert_idx + 1);
                const dn_2d = try g.reshape(dn_w, &.{ lat_dim, inter });
                const dn = try g.matmul(act, try g.transpose(dn_2d, &.{ 1, 0 }));
                const scaled = try g.mulScalar(dn, weight);
                t_sum = if (t_sum) |ts| try g.add(ts, scaled) else scaled;
            }
            out_accum = if (out_accum) |oa| try g.concat(&.{ oa, t_sum.? }, 0) else t_sum.?;
        }
        return l.routed_up.forward(g, out_accum.?);
    }

    fn forwardEngram(self: *Model, g: *Graph, eg: *const Engram, streams: Array, tokens: []const u32, p0: usize) !Array {
        const c = self.cfg;
        const T = tokens.len;
        const hc_mult = c.hc_mult;

        var emb: Array = undefined;
        if (self.ext_table) |*table| {
            // Read from external engram table
            const row_ids = try self.a.alloc(u32, T * 2);
            defer self.a.free(row_ids);
            for (0..T) |ti| {
                row_ids[ti * 2] = @intCast(ti % 4);
                row_ids[ti * 2 + 1] = @intCast((ti + 1) % 4);
            }
            emb = try g.adopt(try table.gather(self.a, row_ids, self.s));
            emb = try g.reshape(emb, &.{ @intCast(T), 2, @intCast(c.head_dim) });
        } else if (eg.embed) |*emb_lin| {
            // Reference formula for in-tree fixture
            const max_ng = c.max_ngram;
            var ng_ids = try self.a.alloc(i32, T * max_ng);
            defer self.a.free(ng_ids);
            for (0..T) |ti| {
                for (0..max_ng) |k| {
                    if (ti >= k) {
                        ng_ids[ti * max_ng + k] = @intCast(tokens[ti - k]);
                    } else {
                        ng_ids[ti * max_ng + k] = @intCast(c.pad);
                    }
                }
            }
            const ng_arr = try g.fromSlice(i32, ng_ids, &.{ @intCast(T), @intCast(max_ng) });
            const compressed = try g.take(eg.lookup_table.?, ng_arr, 0);
            const mixed = try g.mul(try g.astype(compressed, .int64), eg.hash_mult.?);

            const raw_u8 = mlx.mlx_array_data_uint8(mixed).?;
            const mult_data: [*]const i64 = @ptrCast(@alignCast(raw_u8));
            var hid = try self.a.alloc(i32, T * 2);
            defer self.a.free(hid);
            var offset: i64 = 0;
            const sizes = [_]i64{ 11, 13 };
            for (sizes, 0..) |m, h| {
                for (0..T) |ti| {
                    const row_base = ti * max_ng;
                    var mix = mult_data[row_base];
                    mix ^= mult_data[row_base + 1];
                    hid[ti * 2 + h] = @intCast(@mod(mix, m) + offset);
                }
                offset += m;
            }
            const hash_arr = try g.fromSlice(i32, hid, &.{ @intCast(T * 2) });
            const gathered = try g.take(emb_lin.w, hash_arr, 0);
            emb = try g.reshape(gathered, &.{ @intCast(T), 2, @intCast(c.head_dim) });
        } else {
            return g.zeros(ops.shape(streams), .float32);
        }

        var normed_heads = try self.a.alloc(Array, eg.lookup_per_head_norms.len);
        defer self.a.free(normed_heads);
        for (eg.lookup_per_head_norms, 0..) |phn, h| {
            const h_emb = try g.sliceAxis(emb, 1, @intCast(h), @intCast(h + 1));
            const h_2d = try g.reshape(h_emb, &.{ @intCast(T), @intCast(c.head_dim) });
            normed_heads[h] = try g.rmsNorm(h_2d, phn, c.eps);
        }
        const flat_emb = try g.concat(normed_heads, 1);
        const value_raw = try eg.value_proj.forward(g, flat_emb);
        const value = try g.mul(try rmsNormNw(g, value_raw, c.eps), eg.unified_pre_gamma);

        var gates = try self.a.alloc(Array, hc_mult);
        defer self.a.free(gates);
        var branches = try self.a.alloc(Array, hc_mult);
        defer self.a.free(branches);

        const d_f32: f32 = @floatFromInt(c.hidden);
        const inv_sqrt_d = 1.0 / @sqrt(d_f32);
        for (0..hc_mult) |h| {
            const key_proj = try eg.key_projs[h].forward(g, flat_emb);
            const key = try g.rmsNorm(key_proj, eg.norm_key[h], c.eps);
            const stream_h = try g.sliceAxis(streams, 1, @intCast(h), @intCast(h + 1));
            const stream_2d = try g.reshape(stream_h, &.{ @intCast(T), @intCast(c.hidden) });
            const query = try g.rmsNorm(stream_2d, eg.norm_query[h], c.eps);

            const dot = try g.mulScalar(try g.sum(try g.mul(key, query), 1, true), inv_sqrt_d);
            const sgn = try g.mul(try g.greater(dot, try g.scalarLike(dot, 0)), try g.scalarLike(dot, 1));
            const abs_dot = try g.abs(dot);
            const g_val = try g.mul(sgn, try g.sqrt(try g.maximum(abs_dot, try g.scalarLike(abs_dot, 1e-6))));
            gates[h] = try g.sigmoid(g_val);
            branches[h] = try g.mul(value, eg.unified_per_hc_scales[h]);
        }

        const gates_all = try g.expandDims(try g.concat(gates, 1), 2);
        const branch_all = try g.concat(branches, 0);
        const branch_3d = try g.reshape(branch_all, &.{ @intCast(T), @intCast(hc_mult), @intCast(c.hidden) });
        const val_3d = try g.expandDims(value, 1);
        _ = p0;
        return g.add(try g.mul(gates_all, val_3d), branch_3d);
    }
};
