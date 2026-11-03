//! The DeepSeek plugins' tiny random packs, MLX-free: safetensors bytes in the pack's own dtypes (bf16 norms, f32
//! hyper-connections, affine-packed linears and expert banks, an int64 hash-routing table) and their config.json.

const std = @import("std");
const sdk = @import("../../sdk/root.zig");

const File = sdk.fixture.File;

/// Tensors being written: names, dtypes and bytes in an arena; seeded values.
pub const Pack = struct {
    a: std.mem.Allocator,
    rnd: std.Random.DefaultPrng,
    list: std.ArrayList(Tensor) = .empty,

    const Tensor = struct { name: []const u8, dtype: []const u8, shape: []const u64, bytes: []const u8 };

    pub fn init(a: std.mem.Allocator, seed: u64) Pack {
        return .{ .a = a, .rnd = .init(seed) };
    }

    pub fn put(p: *Pack, name: []const u8, dtype: []const u8, shape: []const u64, data: []const u8) !void {
        try p.list.append(p.a, .{ .name = try p.a.dupe(u8, name), .dtype = dtype, .shape = try p.a.dupe(u64, shape), .bytes = data });
    }

    fn count(shape: []const u64) usize {
        var n: usize = 1;
        for (shape) |d| n *= @intCast(d);
        return n;
    }

    fn uniform(p: *Pack, lo: f32, hi: f32) f32 {
        return lo + (hi - lo) * p.rnd.random().float(f32);
    }

    fn bf16Bits(v: f32) u16 {
        const b: u32 = @bitCast(v);
        return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }

    /// Uniform values in `offset ± scale`, as f32 or bf16.
    pub fn dense(p: *Pack, name: []const u8, shape: []const u64, scale: f32, offset: f32, bf16: bool) !void {
        const n = count(shape);
        if (bf16) {
            const v = try p.a.alloc(u16, n);
            for (v) |*x| x.* = bf16Bits(offset + p.uniform(-scale, scale));
            return p.put(name, "BF16", shape, std.mem.sliceAsBytes(v));
        }
        const v = try p.a.alloc(f32, n);
        for (v) |*x| x.* = offset + p.uniform(-scale, scale);
        try p.put(name, "F32", shape, std.mem.sliceAsBytes(v));
    }

    /// A norm weight near one, bf16.
    pub fn norm(p: *Pack, name: []const u8, n: u64) !void {
        try p.dense(name, &.{n}, 0.1, 1.0, true);
    }

    /// `<base>.weight` `[lead..., in]`, affine `bits`-packed in groups of `gs`, with bf16 scales and biases: random
    /// codes over `±sqrt(3 / in)`.
    pub fn affine(p: *Pack, base: []const u8, lead: []const u64, in: u64, bits: u32, gs: u64) !void {
        var buf: [160]u8 = undefined;
        const rows = count(lead);
        var shape: [4]u64 = undefined;
        @memcpy(shape[0..lead.len], lead);
        const words = try p.a.alloc(u32, rows * in * bits / 32);
        for (words) |*w| w.* = p.rnd.random().int(u32);
        shape[lead.len] = in * bits / 32;
        try p.put(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}), "U32", shape[0 .. lead.len + 1], std.mem.sliceAsBytes(words));
        const amp: f32 = @sqrt(3.0 / @as(f32, @floatFromInt(in)));
        const levels: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
        const groups = rows * in / gs;
        const sc = try p.a.alloc(u16, groups);
        const bi = try p.a.alloc(u16, groups);
        for (sc, bi) |*s, *b| {
            s.* = bf16Bits(2 * amp / levels * p.uniform(0.6, 1.4));
            b.* = bf16Bits(-amp * p.uniform(0.6, 1.4));
        }
        shape[lead.len] = in / gs;
        try p.put(try std.fmt.bufPrint(&buf, "{s}.scales", .{base}), "BF16", shape[0 .. lead.len + 1], std.mem.sliceAsBytes(sc));
        try p.put(try std.fmt.bufPrint(&buf, "{s}.biases", .{base}), "BF16", shape[0 .. lead.len + 1], std.mem.sliceAsBytes(bi));
    }

    /// `affine` whose rows all dequantize to zero but `up`'s (every element `bias`) and `down`'s (`-bias`): logits 0
    /// except those two, one of which leads whatever the hidden (a model whose drafts can agree with its trunk).
    pub fn peaked(p: *Pack, base: []const u8, rows: u64, in: u64, bits: u32, gs: u64, up: u64, down: u64, bias: f32) !void {
        var buf: [160]u8 = undefined;
        const words = try p.a.alloc(u32, rows * in * bits / 32);
        @memset(words, 0);
        try p.put(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}), "U32", &.{ rows, in * bits / 32 }, std.mem.sliceAsBytes(words));
        const per = in / gs;
        const sc = try p.a.alloc(u16, rows * per);
        const bi = try p.a.alloc(u16, rows * per);
        @memset(sc, 0);
        @memset(bi, 0);
        @memset(bi[up * per ..][0..per], bf16Bits(bias));
        @memset(bi[down * per ..][0..per], bf16Bits(-bias));
        try p.put(try std.fmt.bufPrint(&buf, "{s}.scales", .{base}), "BF16", &.{ rows, per }, std.mem.sliceAsBytes(sc));
        try p.put(try std.fmt.bufPrint(&buf, "{s}.biases", .{base}), "BF16", &.{ rows, per }, std.mem.sliceAsBytes(bi));
    }

    /// `<base>.weight` `[lead..., in]` as MXFP8: e4m3 codes four to a u32 word (exponents in a narrow band, never NaN)
    /// with one e8m0 scale per 32 inputs (`<base>.scales`, U8), values about `±sqrt(3 / in)`.
    pub fn mxfp8(p: *Pack, base: []const u8, lead: []const u64, in: u64) !void {
        var buf: [160]u8 = undefined;
        const rows = count(lead);
        var shape: [4]u64 = undefined;
        @memcpy(shape[0..lead.len], lead);
        const codes = try p.a.alloc(u8, rows * in);
        for (codes) |*c| c.* = (@as(u8, p.rnd.random().int(u1)) << 7) | (@as(u8, p.rnd.random().intRangeAtMost(u8, 6, 8)) << 3) | p.rnd.random().int(u3);
        shape[lead.len] = in / 4;
        try p.put(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}), "U32", shape[0 .. lead.len + 1], codes);
        const amp: f32 = @sqrt(3.0 / @as(f32, @floatFromInt(in)));
        const sc = try p.a.alloc(u8, rows * in / 32);
        for (sc) |*e| e.* = @intCast(127 + @as(i32, @intFromFloat(@round(@log2(amp)))) + p.rnd.random().intRangeAtMost(i32, -1, 0));
        shape[lead.len] = in / 32;
        try p.put(try std.fmt.bufPrint(&buf, "{s}.scales", .{base}), "U8", shape[0 .. lead.len + 1], sc);
    }

    /// Integers in `[0, max)`, int64.
    pub fn ints(p: *Pack, name: []const u8, shape: []const u64, max: u64) !void {
        const v = try p.a.alloc(i64, count(shape));
        for (v) |*x| x.* = @intCast(p.rnd.random().uintLessThan(u64, max));
        try p.put(name, "I64", shape, std.mem.sliceAsBytes(v));
    }

    /// The safetensors file: 8-byte header length, JSON header, data.
    pub fn bytes(p: *const Pack) ![]u8 {
        var hdr: std.ArrayList(u8) = .empty;
        try hdr.append(p.a, '{');
        var off: u64 = 0;
        for (p.list.items, 0..) |t, i| {
            try hdr.print(p.a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",", t.name, t.dtype });
            for (t.shape, 0..) |d, k| try hdr.print(p.a, "{s}{d}", .{ if (k == 0) "" else ",", d });
            try hdr.print(p.a, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t.bytes.len });
            off += t.bytes.len;
        }
        try hdr.append(p.a, '}');
        while (hdr.items.len % 8 != 0) try hdr.append(p.a, ' ');
        const out = try p.a.alloc(u8, @intCast(8 + hdr.items.len + off));
        std.mem.writeInt(u64, out[0..8], hdr.items.len, .little);
        @memcpy(out[8..][0..hdr.items.len], hdr.items);
        var at: usize = 8 + hdr.items.len;
        for (p.list.items) |t| {
            @memcpy(out[at..][0..t.bytes.len], t.bytes);
            at += t.bytes.len;
        }
        return out;
    }
};

// ── DeepSeek-V4: the oracle's miniature (tests/dsv4_mlx_ref.py); DSpark stages on request ──

const D = 64;
const H = 4;
const HD = 96;
const QL = 32;
const OL = 16;
const OG = 2;
const IH = 2;
const IHD = 32;
const E = 8;
const K = 2;
const MI = 32;
const V = 64;
const ratios = [_]u8{ 0, 4, 16, 4 };

pub const v4_config =
    \\{"model_type": "deepseek_v4", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
    \\ "num_key_value_heads": 1, "head_dim": 96, "qk_rope_head_dim": 32, "q_lora_rank": 32, "o_lora_rank": 16,
    \\ "o_groups": 2, "sliding_window": 8, "compress_ratios": [0, 4, 16, 4], "compress_rope_theta": 160000.0,
    \\ "rope_theta": 10000.0, "rope_scaling": {"factor": 16, "original_max_position_embeddings": 64, "beta_fast": 32,
    \\ "beta_slow": 1, "type": "yarn"}, "index_n_heads": 2, "index_head_dim": 32, "index_topk": 4,
    \\ "n_routed_experts": 8, "num_experts_per_tok": 2, "num_hash_layers": 1, "n_shared_experts": 1,
    \\ "moe_intermediate_size": 32, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "norm_topk_prob": true,
    \\ "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
    \\ "rms_norm_eps": 1e-6, "vocab_size": 64, "max_position_embeddings": 4096, "bos_token_id": 0, "eos_token_id": 1,
    \\ "quantization": {"group_size": 32, "bits": 8, "mode": "affine"}}
;

pub fn hc(p: *Pack, pfx: []const u8, dim: u64) !void {
    var buf: [160]u8 = undefined;
    const mix = (2 + 4) * 4;
    for ([_][]const u8{ "attn", "ffn" }) |tag| {
        try p.dense(try std.fmt.bufPrint(&buf, "{s}.hc_{s}_fn", .{ pfx, tag }), &.{ mix, 4 * dim }, 0.02, 0, false);
        try p.dense(try std.fmt.bufPrint(&buf, "{s}.hc_{s}_base", .{ pfx, tag }), &.{mix}, 0.1, 0, false);
        try p.dense(try std.fmt.bufPrint(&buf, "{s}.hc_{s}_scale", .{ pfx, tag }), &.{3}, 0.2, 0.6, false);
    }
}

pub fn v4Weights(a: std.mem.Allocator, seed: u64) ![]u8 {
    return v4WeightsWith(a, seed, 0);
}

/// The V4 miniature with `stages` DSpark stages (`v4_dspark_keys`): stage layers and the drafter's head.
pub fn v4WeightsWith(a: std.mem.Allocator, seed: u64, stages: usize) ![]u8 {
    var p = Pack.init(a, seed);
    var buf: [160]u8 = undefined;
    const q = struct {
        fn f(pk: *Pack, name: []const u8, out: u64, in: u64) !void {
            try pk.affine(name, &.{out}, in, 8, 32);
        }
    }.f;
    try q(&p, "embed", V, D);
    // With stages the head is peaked (token 5 or 7): drafts that can agree with the trunk, so rounds accept.
    if (stages > 0) try p.peaked("head", V, D, 8, 32, 5, 7, 1.0) else try q(&p, "head", V, D);
    try p.norm("norm.weight", D);
    try p.dense("hc_head_fn", &.{ 4, 4 * D }, 0.02, 0, false);
    try p.dense("hc_head_base", &.{4}, 0.1, 0, false);
    try p.dense("hc_head_scale", &.{1}, 0, 0.5, false);
    for (ratios, 0..) |ratio, li| {
        var pfx_buf: [32]u8 = undefined;
        try v4Layer(&p, try std.fmt.bufPrint(&pfx_buf, "layers.{d}", .{li}), ratio, li < 1);
    }
    if (stages > 0) {
        for (0..stages) |st| {
            var pfx_buf: [32]u8 = undefined;
            try v4Layer(&p, try std.fmt.bufPrint(&pfx_buf, "mtp.{d}", .{st}), 0, false);
        }
        const last = stages - 1;
        try p.affine("mtp.0.main_proj", &.{D}, D * ds_targets.len, 8, 32);
        try p.norm("mtp.0.main_norm.weight", D);
        try p.norm(try std.fmt.bufPrint(&buf, "mtp.{d}.norm.weight", .{last}), D);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.markov_head.markov_w1.weight", .{last}), &.{ V, ds_rank }, 0.01, 0, true);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.markov_head.markov_w2.weight", .{last}), &.{ V, ds_rank }, 0.01, 0, true);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.confidence_head.proj.weight", .{last}), &.{ 1, D + ds_rank }, 0.1, 0, true);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.hc_head_fn", .{last}), &.{ 4, 4 * D }, 0.02, 0, false);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.hc_head_base", .{last}), &.{4}, 0.1, 0, false);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.hc_head_scale", .{last}), &.{1}, 0, 0.5, false);
    }
    return p.bytes();
}

/// One V4 layer at `pfx` (`layers.N`, or `mtp.N` for a DSpark stage): `ratio` its compression, `hash` hash-routed.
fn v4Layer(p: *Pack, pfx: []const u8, ratio: u8, hash: bool) !void {
    var buf: [160]u8 = undefined;
    const q = struct {
        fn f(pk: *Pack, name: []const u8, out: u64, in: u64) !void {
            try pk.affine(name, &.{out}, in, 8, 32);
        }
    }.f;
    const at = struct {
        fn f(b: []u8, pf: []const u8, name: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "{s}.{s}", .{ pf, name });
        }
    }.f;
    try p.norm(try at(&buf, pfx, "attn_norm.weight"), D);
    try p.norm(try at(&buf, pfx, "ffn_norm.weight"), D);
    try hc(p, pfx, D);
    try q(p, try at(&buf, pfx, "attn.wq_a"), QL, D);
    try q(p, try at(&buf, pfx, "attn.wq_b"), H * HD, QL);
    try q(p, try at(&buf, pfx, "attn.wkv"), HD, D);
    try q(p, try at(&buf, pfx, "attn.wo_a"), OG * OL, H * HD / OG);
    try q(p, try at(&buf, pfx, "attn.wo_b"), D, OG * OL);
    try p.norm(try at(&buf, pfx, "attn.q_norm.weight"), QL);
    try p.norm(try at(&buf, pfx, "attn.kv_norm.weight"), HD);
    try p.dense(try at(&buf, pfx, "attn.attn_sink"), &.{H}, 0.5, 0, false);
    if (ratio != 0) {
        const coff: u64 = if (ratio == 4) 2 else 1;
        try compressor(p, pfx, "attn.compressor", coff * HD, HD, ratio);
        if (ratio == 4) {
            try q(p, try at(&buf, pfx, "attn.indexer.wq_b"), IH * IHD, QL);
            try p.dense(try at(&buf, pfx, "attn.indexer.weights_proj.weight"), &.{ IH, D }, 0.05, 0, true);
            try compressor(p, pfx, "attn.indexer.compressor", 2 * IHD, IHD, ratio);
        }
    }
    try p.dense(try at(&buf, pfx, "ffn.gate.weight"), &.{ E, D }, 0.05, 0, true);
    if (hash) try p.ints(try at(&buf, pfx, "ffn.gate.tid2eid"), &.{ V, K }, E) else try p.dense(try at(&buf, pfx, "ffn.gate.bias"), &.{E}, 0.1, 0, false);
    try p.affine(try at(&buf, pfx, "ffn.experts.w1"), &.{ E, MI }, D, 8, 32);
    try p.affine(try at(&buf, pfx, "ffn.experts.w3"), &.{ E, MI }, D, 8, 32);
    try p.affine(try at(&buf, pfx, "ffn.experts.w2"), &.{ E, D }, MI, 8, 32);
    try q(p, try at(&buf, pfx, "ffn.shared_experts.w1"), MI, D);
    try q(p, try at(&buf, pfx, "ffn.shared_experts.w3"), MI, D);
    try q(p, try at(&buf, pfx, "ffn.shared_experts.w2"), D, MI);
}

fn compressor(p: *Pack, pfx: []const u8, comptime sub: []const u8, width: u64, head: u64, ratio: u64) !void {
    var buf: [160]u8 = undefined;
    try p.dense(try std.fmt.bufPrint(&buf, "{s}." ++ sub ++ ".wkv.weight", .{pfx}), &.{ width, D }, 0.05, 0, true);
    try p.dense(try std.fmt.bufPrint(&buf, "{s}." ++ sub ++ ".wgate.weight", .{pfx}), &.{ width, D }, 0.05, 0, true);
    try p.dense(try std.fmt.bufPrint(&buf, "{s}." ++ sub ++ ".ape", .{pfx}), &.{ ratio, width }, 0.1, 0, false);
    try p.norm(try std.fmt.bufPrint(&buf, "{s}." ++ sub ++ ".norm.weight", .{pfx}), head);
}

pub fn v4Files(a: std.mem.Allocator, seed: u64) ![]const File {
    return a.dupe(File, &.{
        .{ .name = "config.json", .bytes = v4_config },
        .{ .name = "model.safetensors", .bytes = try v4Weights(a, seed) },
    });
}

/// The DSpark release's descriptor on the miniature: two stages drafting blocks of three from layers 2 and 3.
const ds_targets = [_]u8{ 2, 3 };
const ds_rank = 16;
const v4_dspark_keys = "\"compress_ratios\": [0, 4, 16, 4, 0, 0], \"num_nextn_predict_layers\": 2, \"dspark_block_size\": 3, " ++
    "\"dspark_noise_token_id\": 63, \"dspark_target_layer_ids\": [2, 3], \"dspark_markov_rank\": 16";

/// `v4Files` with two DSpark stages.
pub fn v4DsparkFiles(a: std.mem.Allocator, seed: u64) ![]const File {
    const cfg = try std.mem.replaceOwned(u8, a, v4_config, "\"compress_ratios\": [0, 4, 16, 4]", v4_dspark_keys);
    return a.dupe(File, &.{
        .{ .name = "config.json", .bytes = cfg },
        .{ .name = "model.safetensors", .bytes = try v4WeightsWith(a, seed, 2) },
    });
}

// ── DeepSeek-V4.1: tests/dump_dsv41_fixtures.py's pipe layout, without Engram; DSpark stages on request ──

pub const v41_config =
    \\{"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "bos_token_id": 0, "eos_token_id": 1,
    \\ "quantization": {"group_size": 64, "bits": 8},
    \\ "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 96, "hidden_size": 128,
    \\  "moe_intermediate_size": 128, "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 1,
    \\  "head_dim": 64, "qk_rope_head_dim": 16, "q_lora_rank": 64, "o_lora_rank": 32, "o_groups": 2, "swiglu_limit": 1.0,
    \\  "rms_norm_eps": 1e-20, "max_position_embeddings": 128, "rope_theta": 10000.0,
    \\  "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 64},
    \\  "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 3, "scoring_func": "sqrtsoftplus",
    \\  "topk_method": "noaux_tc", "norm_topk_prob": true, "routed_scaling_factor": 1.5, "sliding_window": 8,
    \\  "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1], "compress_rope_theta": 40000.0, "kv_source_layer_ids": [2, 5],
    \\  "index_source_layer_ids": [2, 5, 7], "index_n_heads": 8, "index_head_dim": 32, "index_topk": 4,
    \\  "candidate_source_layer_id": 5, "candidate_topk_blocks": 3, "candidate_block_size": 2, "hc_mult": 4,
    \\  "hc_sinkhorn_iters": 20, "hc_eps": 1e-6}}
;

pub fn v41Weights(a: std.mem.Allocator, seed: u64) ![]u8 {
    return v41WeightsWith(a, seed, 0);
}

const v41_d = 128;
const v41_vocab = 96;

/// The V4.1 pipe layout with `stages` DSpark stages (`v41_dspark_keys`): stage layers and the drafter's head.
pub fn v41WeightsWith(a: std.mem.Allocator, seed: u64, stages: usize) ![]u8 {
    const r41 = [_]u8{ 0, 0, 2, 2, 2, 1, 1, 1 };
    const kv_sources = [_]usize{ 2, 5 };
    const index_sources = [_]usize{ 2, 5, 7 };
    var p = Pack.init(a, seed);
    var buf: [160]u8 = undefined;
    try p.affine("embed", &.{v41_vocab}, v41_d, 8, 64);
    // With stages the head is peaked (token 5 or 7): drafts that can agree with the trunk, so rounds accept.
    if (stages > 0) try p.peaked("head", v41_vocab, v41_d, 8, 64, 5, 7, 1.0) else try p.affine("head", &.{v41_vocab}, v41_d, 8, 64);
    try p.norm("norm.weight", v41_d);
    for (r41, 0..) |ratio, li| {
        var pfx_buf: [32]u8 = undefined;
        const kv = std.mem.indexOfScalar(usize, &kv_sources, li) != null;
        const ix = std.mem.indexOfScalar(usize, &index_sources, li) != null;
        try v41Layer(&p, try std.fmt.bufPrint(&pfx_buf, "layers.{d}", .{li}), ratio, kv, ix);
    }
    if (stages > 0) {
        for (0..stages) |st| {
            var pfx_buf: [32]u8 = undefined;
            try v41Layer(&p, try std.fmt.bufPrint(&pfx_buf, "mtp.{d}", .{st}), 0, false, false);
        }
        const last = stages - 1;
        try p.affine("mtp.0.main_proj", &.{v41_d}, v41_d * ds_targets.len, 8, 64);
        try p.norm("mtp.0.main_norm.weight", v41_d);
        try p.norm(try std.fmt.bufPrint(&buf, "mtp.{d}.norm.weight", .{last}), v41_d);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.markov_head.embed.weight", .{last}), &.{ v41_vocab, ds_rank }, 0.01, 0, true);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.markov_head.head.weight", .{last}), &.{ v41_vocab, ds_rank }, 0.01, 0, true);
        try p.dense(try std.fmt.bufPrint(&buf, "mtp.{d}.confidence_head.proj.weight", .{last}), &.{ 1, v41_d + ds_rank }, 0.1, 0, true);
    }
    return p.bytes();
}

/// One V4.1 layer at `pfx` (`layers.N`, or `mtp.N` for a DSpark stage): `ratio` its compression, `kv` / `ix` whether
/// it is a KV / indexer source.
fn v41Layer(p: *Pack, pfx: []const u8, ratio: u8, kv: bool, ix: bool) !void {
    const d = v41_d;
    const hd = 64;
    const ql = 64;
    const h = 4;
    const og = 2;
    const ol = 32;
    const inter = 128;
    const e = 8;
    const ih = 8;
    const ihd = 32;
    var buf: [160]u8 = undefined;
    const at = struct {
        fn f(b: []u8, pf: []const u8, name: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "{s}.{s}", .{ pf, name });
        }
    }.f;
    try hc(p, pfx, d);
    try p.norm(try at(&buf, pfx, "attn_norm.weight"), d);
    try p.norm(try at(&buf, pfx, "ffn_norm.weight"), d);
    try p.affine(try at(&buf, pfx, "attn.wq_a"), &.{ql}, d, 8, 64);
    try p.norm(try at(&buf, pfx, "attn.q_norm.weight"), ql);
    try p.affine(try at(&buf, pfx, "attn.wq_b"), &.{h * hd}, ql, 8, 64);
    try p.affine(try at(&buf, pfx, "attn.wkv"), &.{hd}, d, 8, 64);
    try p.norm(try at(&buf, pfx, "attn.kv_norm.weight"), hd);
    try p.dense(try at(&buf, pfx, "attn.wo_a.weight"), &.{ og * ol, h * hd / og }, 0.1, 0, true);
    try p.affine(try at(&buf, pfx, "attn.wo_b"), &.{d}, og * ol, 8, 64);
    try p.dense(try at(&buf, pfx, "attn.attn_sink"), &.{h}, 0.5, 0, false);
    if (kv) {
        try p.dense(try at(&buf, pfx, "attn.compressor.wkv.weight"), &.{ hd, d }, 0.09, 0, true);
        if (ratio > 1) try p.dense(try at(&buf, pfx, "attn.compressor.wgate.weight"), &.{ hd, d }, 0.09, 0, true);
        try p.norm(try at(&buf, pfx, "attn.compressor.norm.weight"), hd);
    }
    if (ix) {
        try p.affine(try at(&buf, pfx, "attn.indexer.wq_b"), &.{ih * ihd}, ql, 8, 64);
        try p.dense(try at(&buf, pfx, "attn.indexer.weights_proj.weight"), &.{ ih, d }, 0.09, 0, true);
        if (kv) {
            try p.dense(try at(&buf, pfx, "attn.indexer.wk.weight"), &.{ ihd, hd }, 0.12, 0, true);
            try p.norm(try at(&buf, pfx, "attn.indexer.k_norm.weight"), ihd);
        }
    }
    try p.dense(try at(&buf, pfx, "ffn.gate.weight"), &.{ e, d }, 0.09, 0, true);
    try p.dense(try at(&buf, pfx, "ffn.gate.bias"), &.{e}, 0.3, 0, false);
    try p.affine(try at(&buf, pfx, "ffn.shared_experts.w1"), &.{inter}, d, 8, 64);
    try p.affine(try at(&buf, pfx, "ffn.shared_experts.w3"), &.{inter}, d, 8, 64);
    try p.affine(try at(&buf, pfx, "ffn.shared_experts.w2"), &.{d}, inter, 8, 64);
    try p.affine(try at(&buf, pfx, "ffn.experts.gate_proj"), &.{ e, inter }, d, 4, 64);
    try p.affine(try at(&buf, pfx, "ffn.experts.up_proj"), &.{ e, inter }, d, 4, 64);
    try p.affine(try at(&buf, pfx, "ffn.experts.down_proj"), &.{ e, d }, inter, 4, 64);
}

pub fn v41Files(a: std.mem.Allocator, seed: u64) ![]const File {
    return a.dupe(File, &.{
        .{ .name = "config.json", .bytes = v41_config },
        .{ .name = "model.safetensors", .bytes = try v41Weights(a, seed) },
    });
}

const v41_dspark_keys = "\"compress_ratios\": [0, 0, 2, 2, 2, 1, 1, 1, 0, 0], \"num_nextn_predict_layers\": 2, " ++
    "\"dspark_block_size\": 3, \"dspark_noise_token_id\": 95, \"dspark_target_layer_ids\": [2, 3], " ++
    "\"dspark_markov_rank\": 16, \"dspark_n_routed_experts\": 8, \"dspark_num_experts_per_tok\": 3";

/// `v41Files` with two DSpark stages.
pub fn v41DsparkFiles(a: std.mem.Allocator, seed: u64) ![]const File {
    const cfg = try std.mem.replaceOwned(u8, a, v41_config, "\"compress_ratios\": [0, 0, 2, 2, 2, 1, 1, 1]", v41_dspark_keys);
    return a.dupe(File, &.{
        .{ .name = "config.json", .bytes = cfg },
        .{ .name = "model.safetensors", .bytes = try v41WeightsWith(a, seed, 2) },
    });
}

// ── tests ──

test "deepseek fixture: the safetensors header names every tensor at its byte range" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var p = Pack.init(arena.allocator(), 1);
    try p.affine("w", &.{2}, 64, 8, 32);
    try p.ints("t", &.{3}, 4);
    const b = try p.bytes();
    const n = std.mem.readInt(u64, b[0..8], .little);
    const hdr = b[8..][0..n];
    try std.testing.expect(std.mem.indexOf(u8, hdr, "\"w.weight\":{\"dtype\":\"U32\",\"shape\":[2,16],\"data_offsets\":[0,128]}") != null);
    try std.testing.expect(std.mem.indexOf(u8, hdr, "\"t\":{\"dtype\":\"I64\",\"shape\":[3]") != null);
    try std.testing.expectEqual(@as(usize, 8 + n + 128 + 8 + 8 + 24), b.len);
}
