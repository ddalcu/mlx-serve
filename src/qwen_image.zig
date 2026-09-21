//! Qwen-Image-2.1 text-to-image (`Qwen/Qwen-Image-2.1`): a 7.1B single-stream
//! BLOCK-CAUSAL DiT over a Qwen3-VL-8B text encoder and a 64-channel /16 VAE.
//! Port of the pure-MLX reference (mflux PR #736). Sibling of `krea.zig` /
//! `mage_flow.zig`; the text encoder and the dense-or-quantized linear are
//! MageFlow's, everything else is this model's own.
//!
//! What is specific to this checkpoint:
//!  - the joint sequence is [text | image]; text attends causally to text, the
//!    image block attends to everything. Padding-free, so that is TWO sdpa
//!    calls (causal text, maskless image), never a dense mask;
//!  - `causal_condition`: ONE modulation shared by every block, read at t=0 for
//!    text tokens and at the sampled t for image tokens;
//!  - 3-axis RoPE on interleaved pairs: text advances all three axes, the image
//!    freezes the frame axis at the text length on a zero-centred h/w grid;
//!  - latents are consumed unpatched ([1, h·w, 64]);
//!  - the VAE's "3D" convs run per frame, so a single image is plain 2D convs;
//!    its `time_conv` weights are dead on this path and never loaded.
//!
//! Geometry comes from `transformer/config.json` + `vae/config.json`, which is
//! what lets the parity oracle (`tests/dump_qwen_image21_fixtures.py`) run a
//! tiny random-weight pack through the same code.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const sse = @import("gen_sse.zig");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const mage_flow = @import("mage_flow.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const MfLinear = mage_flow.MfLinear;
const TextEncoder = mage_flow.TextEncoder;

/// The reference's recommended sampling: 40 steps, no guidance.
pub const DEFAULT_STEPS: u32 = 40;
const VAE_DOWNSAMPLE: u32 = 16;
/// The DiT + text encoder run in bf16 like the checkpoint; the VAE stays f32.
const COMPUTE: mlx.mlx_dtype = .bfloat16;
const MAX_PROMPT_TOKENS: usize = 2048;

// Raw template string, not the chat template: the checkpoint was trained on it.
const SYSTEM_PREFIX = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const USER_PREFIX = "<|im_start|>user\n";
const PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";

// ── mlx primitives (file-local, mirroring krea.zig / mage_flow.zig) ──
inline fn free(a: A) void {
    _ = mlx.mlx_array_free(a);
}
inline fn addA(a: A, b: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
inline fn subA(a: A, b: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}
inline fn mulA(a: A, b: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
inline fn divA(a: A, b: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&o, a, b, s));
    return o;
}
inline fn reshape(x: A, shape: []const c_int, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
inline fn transpose(x: A, axes: []const c_int, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
inline fn astype(x: A, dt: mlx.mlx_dtype, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn contig(x: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
fn concat(arrs: []const A, axis: c_int, s: S) !A {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// `x[..., lo:hi, ...]` on one axis, materialized: a live slice pins its parent.
fn sliceAxis(x: A, axis: usize, lo_v: c_int, hi_v: c_int, s: S) !A {
    const sh = mlx.getShape(x);
    var lo: [8]c_int = @splat(0);
    var hi: [8]c_int = @splat(0);
    const st: [8]c_int = @splat(1);
    for (sh, 0..) |d, i| hi[i] = d;
    lo[axis] = lo_v;
    hi[axis] = hi_v;
    var o = mlx.mlx_array_new();
    defer free(o);
    try mlx.check(mlx.mlx_slice(&o, x, &lo, sh.len, &hi, sh.len, &st, sh.len, s));
    return contig(o, s);
}
/// A scalar in `ref`'s dtype: a bare f32 scalar promotes a bf16 chain to f32.
fn scalarLike(v: f32, ref: A, s: S) !A {
    const sc = mlx.mlx_array_new_float(v);
    if (mlx.mlx_array_dtype(ref) == .float32) return sc;
    defer free(sc);
    return astype(sc, mlx.mlx_array_dtype(ref), s);
}
fn addScalar(x: A, v: f32, s: S) !A {
    const c = try scalarLike(v, x, s);
    defer free(c);
    return addA(x, c, s);
}
fn mulScalar(x: A, v: f32, s: S) !A {
    const c = try scalarLike(v, x, s);
    defer free(c);
    return mulA(x, c, s);
}
fn silu(x: A, s: S) !A {
    var sig = mlx.mlx_array_new();
    defer free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}
fn tanhA(x: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tanh(&o, x, s));
    return o;
}
/// `nn.gelu_approx`: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
fn geluApprox(x: A, s: S) !A {
    const x2 = try mulA(x, x, s);
    defer free(x2);
    const x3 = try mulA(x2, x, s);
    defer free(x3);
    const kx3 = try mulScalar(x3, 0.044715, s);
    defer free(kx3);
    const inner = try addA(x, kx3, s);
    defer free(inner);
    const scaled = try mulScalar(inner, 0.7978845608028654, s);
    defer free(scaled);
    const t = try tanhA(scaled, s);
    defer free(t);
    const opt = try addScalar(t, 1.0, s);
    defer free(opt);
    const hx = try mulScalar(x, 0.5, s);
    defer free(hx);
    return mulA(hx, opt, s);
}
fn rmsNorm(x: A, w: A, eps: f32, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
/// LayerNorm over the last axis, no affine.
fn layerNorm(x: A, eps: f32, s: S) !A {
    const none = A{ .ctx = null };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, none, none, eps, s));
    return o;
}
fn sdpa(q: A, k: A, v: A, scale: f32, mode: [*:0]const u8, s: S) !A {
    const none = A{ .ctx = null };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, none, none, false, s));
    return o;
}

// ── Config ──

fn readJson(io: std.Io, a: std.mem.Allocator, path: []const u8) !std.json.Parsed(std.json.Value) {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch return error.QwenImageConfigMissing;
    defer a.free(bytes);
    return std.json.parseFromSlice(std.json.Value, a, bytes, .{});
}
fn jsonU32(v: std.json.Value, key: []const u8, default: u32) u32 {
    if (v != .object) return default;
    return switch (v.object.get(key) orelse return default) {
        .integer => |i| @intCast(i),
        else => default,
    };
}
fn jsonF32(v: std.json.Value) ?f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}
fn jsonArray(v: std.json.Value, key: []const u8) ?[]const std.json.Value {
    if (v != .object) return null;
    return switch (v.object.get(key) orelse return null) {
        .array => |arr| arr.items,
        else => null,
    };
}

pub const DitConfig = struct {
    layers: u32 = 32,
    heads: u32 = 32,
    head_dim: u32 = 128,
    in_ch: u32 = 64,
    out_ch: u32 = 64,
    context: u32 = 4096,
    mlp_ratio: u32 = 3,
    axes: [3]u32 = .{ 16, 56, 56 },
    eps: f32 = 1e-6,

    pub fn hidden(self: DitConfig) u32 {
        return self.heads * self.head_dim;
    }

    pub fn parse(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !DitConfig {
        const path = try std.fmt.allocPrint(a, "{s}/transformer/config.json", .{model_dir});
        defer a.free(path);
        var parsed = try readJson(io, a, path);
        defer parsed.deinit();
        const v = parsed.value;
        var c = DitConfig{};
        c.layers = jsonU32(v, "num_layers", c.layers);
        c.heads = jsonU32(v, "num_attention_heads", c.heads);
        c.head_dim = jsonU32(v, "attention_head_dim", c.head_dim);
        c.in_ch = jsonU32(v, "in_channels", c.in_ch);
        c.out_ch = jsonU32(v, "out_channels", c.out_ch);
        c.context = jsonU32(v, "context_in_dim", c.context);
        c.mlp_ratio = jsonU32(v, "mlp_ratio", c.mlp_ratio);
        if (jsonArray(v, "axes_dims_rope")) |axes| {
            if (axes.len != 3) return error.QwenImageBadConfig;
            for (axes, 0..) |ax, i| c.axes[i] = if (ax == .integer) @intCast(ax.integer) else return error.QwenImageBadConfig;
        }
        if (c.axes[0] + c.axes[1] + c.axes[2] != c.head_dim) return error.QwenImageBadConfig;
        return c;
    }
};

const MAX_STAGES = 8;

pub const VaeConfig = struct {
    base_dim: u32 = 96,
    dec_dim: u32 = 144,
    z_dim: u32 = 64,
    num_res_blocks: u32 = 2,
    mult: [MAX_STAGES]u32 = .{ 1, 2, 4, 8, 8, 0, 0, 0 },
    n_mult: usize = 5,
    /// `temperal_downsample[i]`: stage i's shortcut also folds the (single,
    /// zero-padded) frame axis.
    temporal: [MAX_STAGES]bool = .{ false, true, true, true, false, false, false, false },
    mean: []f32,
    std: []f32,

    pub fn parse(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !VaeConfig {
        const path = try std.fmt.allocPrint(a, "{s}/vae/config.json", .{model_dir});
        defer a.free(path);
        var parsed = try readJson(io, a, path);
        defer parsed.deinit();
        const v = parsed.value;
        var c = VaeConfig{ .mean = &.{}, .std = &.{} };
        c.base_dim = jsonU32(v, "base_dim", c.base_dim);
        c.dec_dim = jsonU32(v, "decoder_base_dim", c.dec_dim);
        c.z_dim = jsonU32(v, "z_dim", c.z_dim);
        c.num_res_blocks = jsonU32(v, "num_res_blocks", c.num_res_blocks);
        if (jsonArray(v, "dim_mult")) |m| {
            if (m.len == 0 or m.len > MAX_STAGES) return error.QwenImageBadConfig;
            c.n_mult = m.len;
            for (m, 0..) |e, i| c.mult[i] = if (e == .integer) @intCast(e.integer) else return error.QwenImageBadConfig;
        }
        if (jsonArray(v, "temperal_downsample")) |t| {
            c.temporal = @splat(false);
            for (t, 0..) |e, i| {
                if (i < MAX_STAGES) c.temporal[i] = (e == .bool and e.bool);
            }
        }
        c.mean = try parseFloats(a, v, "latents_mean", c.z_dim);
        errdefer a.free(c.mean);
        c.std = try parseFloats(a, v, "latents_std", c.z_dim);
        return c;
    }

    pub fn deinit(self: *VaeConfig, a: std.mem.Allocator) void {
        a.free(self.mean);
        a.free(self.std);
    }

    fn parseFloats(a: std.mem.Allocator, v: std.json.Value, key: []const u8, n: u32) ![]f32 {
        const items = jsonArray(v, key) orelse return error.QwenImageBadConfig;
        if (items.len != n) return error.QwenImageBadConfig;
        const out = try a.alloc(f32, n);
        errdefer a.free(out);
        for (items, 0..) |e, i| out[i] = jsonF32(e) orelse return error.QwenImageBadConfig;
        return out;
    }
};

// ── Scheduler ──

/// FlowMatchEuler sigmas [steps+1]: linspace(1, 1/steps), exponential time
/// shift with mu linear in the image sequence length, then stretched so the
/// last step lands on `shift_terminal`; a trailing 0 closes the schedule.
pub fn computeSigmas(a: std.mem.Allocator, steps: u32, image_seq_len: u32) ![]f32 {
    const base_shift: f64 = 0.5;
    const max_shift: f64 = 0.9;
    const base_seq: f64 = 256;
    const max_seq: f64 = 8192;
    const terminal: f64 = 0.02;
    const out = try a.alloc(f32, steps + 1);
    if (steps == 1) {
        out[0] = 1;
        out[1] = 0;
        return out;
    }
    const m = (max_shift - base_shift) / (max_seq - base_seq);
    const mu = m * @as(f64, @floatFromInt(image_seq_len)) + (base_shift - m * base_seq);
    const emu = @exp(mu);
    const n: f64 = @floatFromInt(steps);
    const shifted = struct {
        fn at(e: f64, steps_f: f64, i: f64) f64 {
            const sigma = if (steps_f <= 1) 1.0 else 1.0 + i * (1.0 / steps_f - 1.0) / (steps_f - 1.0);
            return e / (e + (1.0 / sigma - 1.0));
        }
    }.at;
    const scale = (1.0 - shifted(emu, n, n - 1)) / (1.0 - terminal);
    for (0..steps) |i| out[i] = @floatCast(1.0 - (1.0 - shifted(emu, n, @floatFromInt(i))) / scale);
    out[steps] = 0;
    return out;
}

// ── DiT ──

/// Per-request constants of the joint [text | image] sequence.
pub const Geometry = struct {
    text_len: c_int,
    cos: A, // [1, L, 1, head_dim/2, 1] f32
    sin: A,
    /// [L] i32: 1 for a text token (the t=0 modulation row), 0 for an image one.
    mod_row: A,

    pub fn init(a: std.mem.Allocator, cfg: DitConfig, text_len: usize, lat_h: usize, lat_w: usize) !Geometry {
        const L = text_len + lat_h * lat_w;
        const half: usize = cfg.head_dim / 2;
        const cosb = try a.alloc(f32, L * half);
        defer a.free(cosb);
        const sinb = try a.alloc(f32, L * half);
        defer a.free(sinb);
        const rows = try a.alloc(i32, L);
        defer a.free(rows);
        const h0: i64 = -@as(i64, @intCast(lat_h - lat_h / 2));
        const w0: i64 = -@as(i64, @intCast(lat_w - lat_w / 2));
        for (0..L) |p| {
            const is_text = p < text_len;
            rows[p] = @intFromBool(is_text);
            const q = p -| text_len;
            const pos: [3]i64 = if (is_text)
                .{ @intCast(p), @intCast(p), @intCast(p) }
            else
                .{ @intCast(text_len), h0 + @as(i64, @intCast(q / lat_w)), w0 + @as(i64, @intCast(q % lat_w)) };
            var col: usize = 0;
            for (cfg.axes, pos) |dim, ax_pos| {
                for (0..dim / 2) |k| {
                    // f32 throughout, like the reference's numpy tables.
                    const expo = @as(f32, @floatFromInt(2 * k)) / @as(f32, @floatFromInt(dim));
                    const omega: f32 = 1.0 / std.math.pow(f32, 10000.0, expo);
                    const ang: f32 = @as(f32, @floatFromInt(ax_pos)) * omega;
                    cosb[p * half + col] = @cos(ang);
                    sinb[p * half + col] = @sin(ang);
                    col += 1;
                }
            }
        }
        const sh = [_]c_int{ 1, @intCast(L), 1, @intCast(half), 1 };
        const rsh = [_]c_int{@intCast(L)};
        return .{
            .text_len = @intCast(text_len),
            .cos = mlx.mlx_array_new_data(cosb.ptr, &sh, sh.len, .float32),
            .sin = mlx.mlx_array_new_data(sinb.ptr, &sh, sh.len, .float32),
            .mod_row = mlx.mlx_array_new_data(rows.ptr, &rsh, 1, .int32),
        };
    }

    pub fn deinit(self: *Geometry) void {
        free(self.cos);
        free(self.sin);
        free(self.mod_row);
    }
};

const Block = struct {
    q: MfLinear,
    k: MfLinear,
    v: MfLinear,
    o: MfLinear,
    norm_q: A, // f32
    norm_k: A,
    proj: MfLinear,
    gate: MfLinear,
    out: MfLinear,

    fn deinit(self: *Block) void {
        inline for (.{ &self.q, &self.k, &self.v, &self.o, &self.proj, &self.gate, &self.out }) |l| l.deinit();
        free(self.norm_q);
        free(self.norm_k);
    }
};

/// The shared modulation of one denoise step, already per token: `1 + scale`
/// and `tanh(gate)` for the attention and MLP halves, each [1, L, hidden].
const StepMod = struct {
    scale1: A,
    gate1: A,
    scale2: A,
    gate2: A,

    fn deinit(self: *StepMod) void {
        inline for (.{ self.scale1, self.gate1, self.scale2, self.gate2 }) |x| free(x);
    }
};

fn loadVecF32(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, s: S) !A {
    const key = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(key);
    const raw = w.get(key) orelse {
        log.err("[qwen-image] missing weight: {s}\n", .{key});
        return error.MissingQwenImageWeight;
    };
    return astype(raw, .float32, s);
}

fn loadLinear(w: *const Weights, a: std.mem.Allocator, in_features: u32, dtype: mlx.mlx_dtype, s: S, comptime fmt: []const u8, args: anytype) !MfLinear {
    const prefix = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(prefix);
    return MfLinear.load(w, a, prefix, in_features, dtype, s);
}

pub const Dit = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: DitConfig,
    dtype: mlx.mlx_dtype,
    img_in: MfLinear,
    txt_norm: A, // f32, stored zero-centred: holds weight + 1
    txt_in: MfLinear,
    txt_out: MfLinear,
    t1: MfLinear,
    t2: MfLinear,
    modulation: MfLinear,
    blocks: []Block,
    norm_out: MfLinear,
    proj_out: MfLinear,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8, cfg: DitConfig, dtype: mlx.mlx_dtype) !Dit {
        const dir = try std.fmt.allocPrint(allocator, "{s}/transformer", .{model_dir});
        defer allocator.free(dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        const a = allocator;
        const H = cfg.hidden();

        var self: Dit = undefined;
        self.allocator = allocator;
        self.s = s;
        self.cfg = cfg;
        self.dtype = dtype;
        self.img_in = try loadLinear(&w, a, cfg.in_ch, dtype, s, "img_in", .{});
        const tn = try loadVecF32(&w, a, "txt_in.text_norm.weight", .{}, s);
        defer free(tn);
        self.txt_norm = try addScalar(tn, 1.0, s);
        self.txt_in = try loadLinear(&w, a, cfg.context, dtype, s, "txt_in.in_layer", .{});
        self.txt_out = try loadLinear(&w, a, H, dtype, s, "txt_in.out_layer", .{});
        self.t1 = try loadLinear(&w, a, 256, dtype, s, "time_text_embed.timestep_embedder.linear_1", .{});
        self.t2 = try loadLinear(&w, a, H, dtype, s, "time_text_embed.timestep_embedder.linear_2", .{});
        self.modulation = try loadLinear(&w, a, H, dtype, s, "modulation.1", .{});
        self.norm_out = try loadLinear(&w, a, H, dtype, s, "norm_out.linear", .{});
        self.proj_out = try loadLinear(&w, a, H, dtype, s, "proj_out", .{});

        self.blocks = try a.alloc(Block, cfg.layers);
        for (self.blocks, 0..) |*b, i| {
            const p = "transformer_blocks.{d}.";
            b.* = .{
                .q = try loadLinear(&w, a, H, dtype, s, p ++ "attn.to_q", .{i}),
                .k = try loadLinear(&w, a, H, dtype, s, p ++ "attn.to_k", .{i}),
                .v = try loadLinear(&w, a, H, dtype, s, p ++ "attn.to_v", .{i}),
                .o = try loadLinear(&w, a, H, dtype, s, p ++ "attn.to_out.0", .{i}),
                .norm_q = try loadVecF32(&w, a, p ++ "attn.norm_q.weight", .{i}, s),
                .norm_k = try loadVecF32(&w, a, p ++ "attn.norm_k.weight", .{i}, s),
                .proj = try loadLinear(&w, a, H, dtype, s, p ++ "img_mlp.proj", .{i}),
                .gate = try loadLinear(&w, a, H, dtype, s, p ++ "img_mlp.gate_layer", .{i}),
                .out = try loadLinear(&w, a, H * cfg.mlp_ratio, dtype, s, p ++ "img_mlp.out", .{i}),
            };
        }
        return self;
    }

    pub fn deinit(self: *Dit) void {
        inline for (.{ &self.img_in, &self.txt_in, &self.txt_out, &self.t1, &self.t2, &self.modulation, &self.norm_out, &self.proj_out }) |l| l.deinit();
        free(self.txt_norm);
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
    }

    /// Timestep embedding rows [2, hidden]: row 0 the sampled t, row 1 t = 0.
    /// The sinusoid (cos half first) is f32, cast to the compute dtype BEFORE
    /// the embedder so an f32 modulation never widens the hidden stream.
    fn timeEmbed(self: *const Dit, t: f32) !A {
        const s = self.s;
        var buf: [2 * 256]f32 = undefined;
        for ([_]f32{ t, 0 }, 0..) |tv, row| {
            for (0..128) |k| {
                const freq: f32 = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(k)) / 128.0);
                const arg = tv * 1000.0 * freq;
                buf[row * 256 + k] = @cos(arg);
                buf[row * 256 + 128 + k] = @sin(arg);
            }
        }
        const sh = [_]c_int{ 2, 256 };
        const raw = mlx.mlx_array_new_data(&buf, &sh, 2, .float32);
        defer free(raw);
        const proj = try astype(raw, self.dtype, s);
        defer free(proj);
        const h1 = try self.t1.forward(proj, null, s);
        defer free(h1);
        const act = try silu(h1, s);
        defer free(act);
        return self.t2.forward(act, null, s);
    }

    fn stepMod(self: *const Dit, temb_act: A, geo: *const Geometry) !StepMod {
        const s = self.s;
        const H: c_int = @intCast(self.cfg.hidden());
        const mod = try self.modulation.forward(temb_act, null, s); // [2, 4H]: scale1 gate1 scale2 gate2
        defer free(mod);
        const opm = try addScalar(mod, 1.0, s);
        defer free(opm);
        const th = try tanhA(mod, s);
        defer free(th);
        var out: [4]A = undefined;
        for (&out, 0..) |*o, i| {
            const src = if (i % 2 == 0) opm else th;
            const lo: c_int = @as(c_int, @intCast(i)) * H;
            const part = try sliceAxis(src, 1, lo, lo + H, s); // [2, H]
            defer free(part);
            var rows = mlx.mlx_array_new();
            defer free(rows);
            try mlx.check(mlx.mlx_take_axis(&rows, part, geo.mod_row, 0, s)); // [L, H]
            o.* = try reshape(rows, &[_]c_int{ 1, -1, H }, s);
        }
        return .{ .scale1 = out[0], .gate1 = out[1], .scale2 = out[2], .gate2 = out[3] };
    }

    /// Rotate interleaved pairs (2k, 2k+1) of x [1, L, heads, hd] in f32.
    fn applyRope(self: *const Dit, x: A, geo: *const Geometry) !A {
        const s = self.s;
        const sh = mlx.getShape(x);
        const half = @divExact(sh[3], 2);
        const xf = try astype(x, .float32, s);
        defer free(xf);
        const pairs = try reshape(xf, &[_]c_int{ sh[0], sh[1], sh[2], half, 2 }, s);
        defer free(pairs);
        const re = try sliceAxis(pairs, 4, 0, 1, s);
        defer free(re);
        const im = try sliceAxis(pairs, 4, 1, 2, s);
        defer free(im);
        const rc = try mulA(re, geo.cos, s);
        defer free(rc);
        const is = try mulA(im, geo.sin, s);
        defer free(is);
        const o_re = try subA(rc, is, s);
        defer free(o_re);
        const rs = try mulA(re, geo.sin, s);
        defer free(rs);
        const ic = try mulA(im, geo.cos, s);
        defer free(ic);
        const o_im = try addA(rs, ic, s);
        defer free(o_im);
        const joined = try concat(&.{ o_re, o_im }, 4, s);
        defer free(joined);
        const flat = try reshape(joined, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, s);
        defer free(flat);
        return astype(flat, self.dtype, s);
    }

    /// Project → per-head RMS norm → RoPE → [1, heads, L, hd].
    fn headsOf(self: *const Dit, lin: *const MfLinear, norm: ?A, x: A, geo: *const Geometry) !A {
        const s = self.s;
        const sh = mlx.getShape(x);
        const y = try lin.forward(x, null, s);
        defer free(y);
        const y4 = try reshape(y, &[_]c_int{ sh[0], sh[1], @intCast(self.cfg.heads), @intCast(self.cfg.head_dim) }, s);
        defer free(y4);
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const nw = norm orelse return transpose(y4, &perm, s);
        const yn = try rmsNorm(y4, nw, self.cfg.eps, s);
        defer free(yn);
        const yr = try self.applyRope(yn, geo);
        defer free(yr);
        return transpose(yr, &perm, s);
    }

    /// Block-causal attention as the reference segments it: causal over the
    /// text prefix, then the image block against the whole sequence.
    fn attention(self: *const Dit, b: *const Block, x: A, geo: *const Geometry) !A {
        const s = self.s;
        const sh = mlx.getShape(x);
        const T = geo.text_len;
        const q = try self.headsOf(&b.q, b.norm_q, x, geo);
        defer free(q);
        const k = try self.headsOf(&b.k, b.norm_k, x, geo);
        defer free(k);
        const v = try self.headsOf(&b.v, null, x, geo);
        defer free(v);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.cfg.head_dim)));

        const qi = try sliceAxis(q, 2, T, sh[1], s);
        defer free(qi);
        const img_out = try sdpa(qi, k, v, scale, "", s);
        defer free(img_out);
        const joined = if (T == 0) try contig(img_out, s) else blk: {
            const qt = try sliceAxis(q, 2, 0, T, s);
            defer free(qt);
            const kt = try sliceAxis(k, 2, 0, T, s);
            defer free(kt);
            const vt = try sliceAxis(v, 2, 0, T, s);
            defer free(vt);
            const txt_out = try sdpa(qt, kt, vt, scale, "causal", s);
            defer free(txt_out);
            break :blk try concat(&.{ txt_out, img_out }, 2, s);
        };
        defer free(joined);
        const back = try transpose(joined, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer free(back);
        const flat = try reshape(back, &[_]c_int{ sh[0], sh[1], sh[2] }, s);
        defer free(flat);
        return b.o.forward(flat, null, s);
    }

    fn blockForward(self: *const Dit, b: *const Block, x: A, mod: *const StepMod, geo: *const Geometry) !A {
        const s = self.s;
        const n1 = try layerNorm(x, self.cfg.eps, s);
        defer free(n1);
        const a_in = try mulA(n1, mod.scale1, s);
        defer free(a_in);
        const attn = try self.attention(b, a_in, geo);
        defer free(attn);
        const ga = try mulA(attn, mod.gate1, s);
        defer free(ga);
        const h = try addA(x, ga, s);
        defer free(h);

        const n2 = try layerNorm(h, self.cfg.eps, s);
        defer free(n2);
        const m_in = try mulA(n2, mod.scale2, s);
        defer free(m_in);
        const g = try b.gate.forward(m_in, null, s);
        defer free(g);
        const sg = try silu(g, s);
        defer free(sg);
        const p = try b.proj.forward(m_in, null, s);
        defer free(p);
        const gp = try mulA(sg, p, s);
        defer free(gp);
        const mlp = try b.out.forward(gp, null, s);
        defer free(mlp);
        const gm = try mulA(mlp, mod.gate2, s);
        defer free(gm);
        return addA(h, gm, s);
    }

    fn textIn(self: *const Dit, txt: A) !A {
        const s = self.s;
        const tf = try astype(txt, .float32, s);
        defer free(tf);
        const n = try rmsNorm(tf, self.txt_norm, self.cfg.eps, s);
        defer free(n);
        const h = try self.txt_in.forward(n, null, s);
        defer free(h);
        const act = try geluApprox(h, s);
        defer free(act);
        return self.txt_out.forward(act, null, s);
    }

    /// Velocity for one flow step. img [1, N, in_ch], txt [1, T, context] →
    /// [1, N, out_ch] in the compute dtype. Caller owns the result.
    pub fn forward(self: *const Dit, img: A, txt: A, t: f32, geo: *const Geometry) !A {
        const s = self.s;
        const temb = try self.timeEmbed(t);
        defer free(temb);
        const temb_act = try silu(temb, s);
        defer free(temb_act);
        var mod = try self.stepMod(temb_act, geo);
        defer mod.deinit();

        const th = try self.textIn(txt);
        defer free(th);
        const ih = try self.img_in.forward(img, null, s);
        defer free(ih);
        var x = try concat(&.{ th, ih }, 1, s);
        errdefer free(x);
        for (self.blocks) |*b| {
            const nx = try self.blockForward(b, x, &mod, geo);
            free(x);
            x = nx;
        }
        defer free(x);

        // Only image tokens leave the model, and LayerNorm is per token, so the
        // final norm reads the image rows and the sampled-t scale alone.
        const L = mlx.getShape(x)[1];
        const xi = try sliceAxis(x, 1, geo.text_len, L, s);
        defer free(xi);
        const n = try layerNorm(xi, self.cfg.eps, s);
        defer free(n);
        const sc2 = try self.norm_out.forward(temb_act, null, s); // [2, H]
        defer free(sc2);
        const sc = try sliceAxis(sc2, 0, 0, 1, s);
        defer free(sc);
        const opsc = try addScalar(sc, 1.0, s);
        defer free(opsc);
        const scaled = try mulA(n, opsc, s);
        defer free(scaled);
        return self.proj_out.forward(scaled, null, s);
    }
};

// ── VAE (f32, NHWC inside) ──

const Conv = struct {
    w: A, // OHWI
    b: A,

    fn load(w: *const Weights, a: std.mem.Allocator, s: S, comptime fmt: []const u8, args: anytype) !Conv {
        return (try loadOpt(w, a, s, fmt, args)) orelse {
            const prefix = try std.fmt.allocPrint(a, fmt, args);
            defer a.free(prefix);
            log.err("[qwen-image] missing VAE conv: {s}\n", .{prefix});
            return error.MissingQwenImageWeight;
        };
    }

    fn loadOpt(w: *const Weights, a: std.mem.Allocator, s: S, comptime fmt: []const u8, args: anytype) !?Conv {
        const wk = try std.fmt.allocPrint(a, fmt ++ ".weight", args);
        defer a.free(wk);
        const raw = w.get(wk) orelse return null;
        const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 1 }, s); // OIHW → OHWI
        defer free(t);
        const tc = try contig(t, s);
        defer free(tc);
        const wf = try astype(tc, .float32, s);
        errdefer free(wf);
        return .{ .w = wf, .b = try loadVecF32(w, a, fmt ++ ".bias", args, s) };
    }

    fn deinit(self: *Conv) void {
        free(self.w);
        free(self.b);
    }

    fn forward(self: *const Conv, x: A, stride: c_int, pad: c_int, s: S) !A {
        const xc = try contig(x, s);
        defer free(xc);
        const strips = stripCount(mlx.getShape(xc), mlx.getShape(self.w), stride);
        if (strips > 1) return self.forwardStrips(xc, strips, s);
        var o = mlx.mlx_array_new();
        defer free(o);
        try mlx.check(mlx.mlx_conv2d(&o, xc, self.w, stride, stride, pad, pad, 1, 1, 1, s));
        return addA(o, self.b, s);
    }

    /// A stride-1 3x3 conv in horizontal strips: rows are padded ONCE, each
    /// strip reads its own rows plus one above and below, so the result is the
    /// whole-image conv exactly. Each strip is evaluated before the next is built.
    fn forwardStrips(self: *const Conv, xc: A, strips: c_int, s: S) !A {
        const H = mlx.getShape(xc)[1];
        const axes = [_]c_int{1};
        const one = [_]c_int{1};
        const zero = mlx.mlx_array_new_float(0);
        defer free(zero);
        var padded = mlx.mlx_array_new();
        defer free(padded);
        try mlx.check(mlx.mlx_pad(&padded, xc, &axes, 1, &one, 1, &one, 1, zero, "constant", s));

        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        const rows = @divFloor(H + strips - 1, strips);
        var y: c_int = 0;
        while (y < H) : (y += rows) {
            const band = try sliceAxis(padded, 1, y, @min(y + rows, H) + 2, s);
            defer free(band);
            var o = mlx.mlx_array_new();
            defer free(o);
            try mlx.check(mlx.mlx_conv2d(&o, band, self.w, 1, 1, 0, 1, 1, 1, 1, s));
            const biased = try addA(o, self.b, s);
            defer free(biased);
            try mlx.check(mlx.mlx_array_eval(biased));
            _ = mlx.mlx_vector_array_append_value(vec, biased);
        }
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
        return out;
    }
};

/// MLX's 3x3 conv holds an unfolded copy of its input (H·W·9·C_in floats): 11 GB
/// for the decoder's 288-channel stage at 1024², invisible to MLX's own memory
/// counters. Past this budget a conv runs in strips.
const CONV_UNFOLD_BUDGET_BYTES: u64 = 512 << 20;

fn stripCount(x_shape: []const c_int, w_shape: []const c_int, stride: c_int) c_int {
    if (stride != 1 or w_shape[1] != 3 or w_shape[2] != 3) return 1;
    var unfold: u64 = 9 * @sizeOf(f32);
    for (x_shape) |d| unfold *= @intCast(d);
    const n = (unfold + CONV_UNFOLD_BUDGET_BYTES - 1) / CONV_UNFOLD_BUDGET_BYTES;
    return @intCast(@min(n, @as(u64, @intCast(x_shape[1]))));
}

/// Input rows per band of a stage (even, so a stride-2 stage cuts on cell
/// boundaries), sized so the stage's widest activation stays under the budget.
var band_budget_bytes: u64 = 256 << 20; // var: the parity test forces bands on a tiny pack

fn bandRows(x_shape: []const c_int, out_ch: c_int, grow: c_int) c_int {
    const width: u64 = @intCast(x_shape[2] * grow);
    const channels: u64 = @intCast(@max(x_shape[3], out_ch));
    const row_bytes = width * channels * @sizeOf(f32) * @as(u64, @intCast(grow));
    const rows = @max(8, band_budget_bytes / row_bytes);
    return @intCast(@min(rows & ~@as(u64, 1), @as(u64, @intCast(x_shape[1]))));
}

/// Wan-style norm gamma, flattened to [C].
fn loadVaeNorm(w: *const Weights, a: std.mem.Allocator, s: S, comptime fmt: []const u8, args: anytype) !A {
    const g = try loadVecF32(w, a, fmt ++ ".gamma", args, s);
    defer free(g);
    return reshape(g, &[_]c_int{-1}, s);
}

/// x/‖x‖₂ · √C · gamma over channels IS rms_norm (‖x‖₂/√C = rms), and the fused
/// kernel allocates one full-resolution tensor where the spelled-out chain
/// allocates five: at 1024² that chain was most of a 18 GB decode.
fn vaeNorm(x: A, gamma: A, s: S) !A {
    return rmsNorm(x, gamma, 1e-12, s);
}

const Res = struct {
    n1: A,
    c1: Conv,
    n2: A,
    c2: Conv,
    shortcut: ?Conv,

    fn load(w: *const Weights, a: std.mem.Allocator, s: S, comptime fmt: []const u8, args: anytype) !Res {
        return .{
            .n1 = try loadVaeNorm(w, a, s, fmt ++ ".norm1", args),
            .c1 = try Conv.load(w, a, s, fmt ++ ".conv1", args),
            .n2 = try loadVaeNorm(w, a, s, fmt ++ ".norm2", args),
            .c2 = try Conv.load(w, a, s, fmt ++ ".conv2", args),
            .shortcut = try Conv.loadOpt(w, a, s, fmt ++ ".conv_shortcut", args),
        };
    }

    fn deinit(self: *Res) void {
        free(self.n1);
        free(self.n2);
        self.c1.deinit();
        self.c2.deinit();
        if (self.shortcut) |*c| c.deinit();
    }

    fn forward(self: *const Res, x: A, s: S) !A {
        const n1 = try vaeNorm(x, self.n1, s);
        defer free(n1);
        const a1 = try silu(n1, s);
        defer free(a1);
        const h1 = try self.c1.forward(a1, 1, 1, s);
        defer free(h1);
        const n2 = try vaeNorm(h1, self.n2, s);
        defer free(n2);
        const a2 = try silu(n2, s);
        defer free(a2);
        const h2 = try self.c2.forward(a2, 1, 1, s);
        defer free(h2);
        const sc = self.shortcut orelse return addA(h2, x, s);
        const r = try sc.forward(x, 1, 0, s);
        defer free(r);
        return addA(h2, r, s);
    }
};

const Mid = struct {
    r0: Res,
    r1: Res,
    norm: A,
    qkv: Conv,
    proj: Conv,

    fn load(w: *const Weights, a: std.mem.Allocator, s: S, comptime side: []const u8) !Mid {
        const p = side ++ ".mid_block.";
        return .{
            .r0 = try Res.load(w, a, s, p ++ "resnets.0", .{}),
            .r1 = try Res.load(w, a, s, p ++ "resnets.1", .{}),
            .norm = try loadVaeNorm(w, a, s, p ++ "attentions.0.norm", .{}),
            .qkv = try Conv.load(w, a, s, p ++ "attentions.0.to_qkv", .{}),
            .proj = try Conv.load(w, a, s, p ++ "attentions.0.proj", .{}),
        };
    }

    fn deinit(self: *Mid) void {
        self.r0.deinit();
        self.r1.deinit();
        free(self.norm);
        self.qkv.deinit();
        self.proj.deinit();
    }

    /// Single-head self-attention over the flattened spatial axis.
    fn attn(self: *const Mid, x: A, s: S) !A {
        const sh = mlx.getShape(x); // [B,H,W,C]
        const C = sh[3];
        const n = try vaeNorm(x, self.norm, s);
        defer free(n);
        const qkv = try self.qkv.forward(n, 1, 0, s);
        defer free(qkv);
        const flat = try reshape(qkv, &[_]c_int{ sh[0], 1, sh[1] * sh[2], 3 * C }, s);
        defer free(flat);
        var parts: [3]A = undefined;
        for (&parts, 0..) |*p, i| p.* = try sliceAxis(flat, 3, @as(c_int, @intCast(i)) * C, @as(c_int, @intCast(i + 1)) * C, s);
        defer for (parts) |p| free(p);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(C)));
        const o = try sdpa(parts[0], parts[1], parts[2], scale, "", s);
        defer free(o);
        const grid = try reshape(o, &[_]c_int{ sh[0], sh[1], sh[2], C }, s);
        defer free(grid);
        const proj = try self.proj.forward(grid, 1, 0, s);
        defer free(proj);
        return addA(proj, x, s);
    }

    fn forward(self: *const Mid, x: A, s: S) !A {
        const a = try self.r0.forward(x, s);
        defer free(a);
        const b = try self.attn(a, s);
        defer free(b);
        return self.r1.forward(b, s);
    }
};

/// Parameterless up shortcut: repeat channels, then unfold the repeats into a
/// 2×2 nearest upsample. With a temporal fold the FIRST frame slot is dropped.
fn dupUp(x: A, out_ch: c_int, temporal: bool, s: S) !A {
    const sh = mlx.getShape(x); // [B,H,W,Cin]
    const ft: c_int = if (temporal) 2 else 1;
    const repeats = @divExact(out_ch * ft * 4, sh[3]);
    var rep = mlx.mlx_array_new();
    defer free(rep);
    try mlx.check(mlx.mlx_repeat_axis(&rep, x, repeats, 3, s));
    const folded = try reshape(rep, &[_]c_int{ sh[0], sh[1], sh[2], out_ch, ft, 2, 2 }, s);
    defer free(folded);
    const last = try sliceAxis(folded, 4, ft - 1, ft, s);
    defer free(last);
    const sq = try reshape(last, &[_]c_int{ sh[0], sh[1], sh[2], out_ch, 2, 2 }, s);
    defer free(sq);
    const t = try transpose(sq, &[_]c_int{ 0, 1, 4, 2, 5, 3 }, s);
    defer free(t);
    return reshape(t, &[_]c_int{ sh[0], sh[1] * 2, sh[2] * 2, out_ch }, s);
}

/// Parameterless down shortcut: fold `fs`×`fs` pixels (and, with a temporal
/// fold, one zero frame in FRONT of the image) into channels, then average
/// channel groups down to `out_ch`.
fn avgDown(x: A, out_ch: c_int, temporal: bool, fs: c_int, s: S) !A {
    const sh = mlx.getShape(x); // [B,H,W,C]
    const hh = @divExact(sh[1], fs);
    const ww = @divExact(sh[2], fs);
    const cells = try reshape(x, &[_]c_int{ sh[0], hh, fs, ww, fs, sh[3] }, s);
    defer free(cells);
    const t = try transpose(cells, &[_]c_int{ 0, 1, 3, 5, 2, 4 }, s); // [B,h,w,C,fs,fs]
    defer free(t);
    const framed = if (!temporal) try contig(t, s) else blk: {
        const f1 = try reshape(t, &[_]c_int{ sh[0], hh, ww, sh[3], 1, fs, fs }, s);
        defer free(f1);
        const zeros = try mulScalar(f1, 0.0, s);
        defer free(zeros);
        break :blk try concat(&.{ zeros, f1 }, 4, s);
    };
    defer free(framed);
    const grouped = try reshape(framed, &[_]c_int{ sh[0], hh, ww, out_ch, -1 }, s);
    defer free(grouped);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean_axis(&o, grouped, 4, false, s));
    return o;
}

const Stage = struct {
    res: []Res,
    resample: ?Conv,
    out_ch: c_int,
    temporal: bool,

    fn load(w: *const Weights, a: std.mem.Allocator, s: S, comptime fmt: []const u8, comptime sampler: []const u8, i: usize, n_res: usize, out_ch: u32, temporal: bool) !Stage {
        const res = try a.alloc(Res, n_res);
        for (res, 0..) |*r, j| r.* = try Res.load(w, a, s, fmt ++ ".resnets.{d}", .{ i, j });
        return .{
            .res = res,
            .resample = try Conv.loadOpt(w, a, s, fmt ++ "." ++ sampler ++ ".resample.1", .{i}),
            .out_ch = @intCast(out_ch),
            .temporal = temporal,
        };
    }

    fn deinit(self: *Stage, a: std.mem.Allocator) void {
        for (self.res) |*r| r.deinit();
        a.free(self.res);
        if (self.resample) |*c| c.deinit();
    }

    const Dir = enum { up, down };

    fn up(self: *const Stage, x: A, s: S) !A {
        return self.banded(x, .up, s);
    }
    fn down(self: *const Stage, x: A, s: S) !A {
        return self.banded(x, .down, s);
    }

    /// A stage in horizontal bands. Every op here is per-pixel (the channel
    /// norm, 1x1 convs, both shortcuts) or a 3x3 conv, so a band carrying
    /// `halo` extra rows each side computes its own rows exactly; the halo rows
    /// are cropped. What this bounds is the LIVE SET: a whole 1024² stage holds
    /// half a dozen 1.2 GB f32 tensors at once.
    fn banded(self: *const Stage, x: A, comptime dir: Dir, s: S) !A {
        const sh = mlx.getShape(x);
        const H = sh[1];
        const scaled = self.resample != null;
        const grow: c_int = if (dir == .up and scaled) 2 else 1; // output rows per input row
        const shrink: c_int = if (dir == .down and scaled) 2 else 1; // input rows per output row
        const rows = bandRows(sh, self.out_ch, grow);
        if (rows >= H) return self.whole(x, dir, s);
        const halo: c_int = @intCast(2 * self.res.len + 2); // two 3x3 convs per resnet + the resample conv, kept even

        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var y0: c_int = 0;
        while (y0 < H) : (y0 += rows) {
            const y1 = @min(y0 + rows, H);
            const lo = @max(0, y0 - halo);
            const band = try sliceAxis(x, 1, lo, @min(H, y1 + halo), s);
            defer free(band);
            const out = try self.whole(band, dir, s);
            defer free(out);
            const kept = try sliceAxis(out, 1, @divExact((y0 - lo) * grow, shrink), @divExact((y1 - lo) * grow, shrink), s);
            defer free(kept);
            try mlx.check(mlx.mlx_array_eval(kept));
            _ = mlx.mlx_vector_array_append_value(vec, kept);
        }
        var joined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, s));
        return joined;
    }

    fn whole(self: *const Stage, x: A, comptime dir: Dir, s: S) !A {
        return switch (dir) {
            .up => self.upWhole(x, s),
            .down => self.downWhole(x, s),
        };
    }

    fn resnets(self: *const Stage, x: A, s: S) !A {
        var h = try contig(x, s);
        errdefer free(h);
        for (self.res) |*r| {
            const nh = try r.forward(h, s);
            free(h);
            h = nh;
            // Full-resolution stages are GBs of f32: never let resnets stack lazily.
            try mlx.check(mlx.mlx_array_eval(h));
        }
        return h;
    }

    fn upWhole(self: *const Stage, x: A, s: S) !A {
        const h = try self.resnets(x, s);
        const conv = self.resample orelse return h;
        defer free(h);
        var r1 = mlx.mlx_array_new();
        defer free(r1);
        try mlx.check(mlx.mlx_repeat_axis(&r1, h, 2, 1, s));
        var r2 = mlx.mlx_array_new();
        defer free(r2);
        try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
        const c = try conv.forward(r2, 1, 1, s);
        defer free(c);
        const sc = try dupUp(x, self.out_ch, self.temporal, s);
        defer free(sc);
        return addA(c, sc, s);
    }

    fn downWhole(self: *const Stage, x: A, s: S) !A {
        const h = try self.resnets(x, s);
        defer free(h);
        const conv = self.resample orelse {
            const sc = try avgDown(x, self.out_ch, false, 1, s);
            defer free(sc);
            return addA(h, sc, s);
        };
        // ZeroPad2d((0,1,0,1)) then a stride-2 valid conv.
        const axes = [_]c_int{ 1, 2 };
        const lo = [_]c_int{ 0, 0 };
        const hi = [_]c_int{ 1, 1 };
        const zero = mlx.mlx_array_new_float(0);
        defer free(zero);
        var padded = mlx.mlx_array_new();
        defer free(padded);
        try mlx.check(mlx.mlx_pad(&padded, h, &axes, 2, &lo, 2, &hi, 2, zero, "constant", s));
        const c = try conv.forward(padded, 2, 0, s);
        defer free(c);
        const sc = try avgDown(x, self.out_ch, self.temporal, 2, s);
        defer free(sc);
        return addA(c, sc, s);
    }
};

fn loadVaeWeights(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Weights {
    const dir = try std.fmt.allocPrint(a, "{s}/vae", .{model_dir});
    defer a.free(dir);
    return model_mod.loadWeights(io, a, dir);
}

fn channelVec(vals: []const f32, s: S) !A {
    const sh = [_]c_int{@intCast(vals.len)};
    const raw = mlx.mlx_array_new_data(vals.ptr, &sh, 1, .float32);
    defer free(raw);
    return contig(raw, s);
}

/// Everything both halves of the VAE share: stem, stages, mid block, head.
const VaeHalf = struct {
    allocator: std.mem.Allocator,
    s: S,
    quant: Conv, // post_quant_conv (decoder) / quant_conv (encoder)
    conv_in: Conv,
    mid: Mid,
    stages: []Stage,
    norm_out: A,
    conv_out: Conv,
    mean: A, // [z]
    std: A,

    fn deinit(self: *VaeHalf) void {
        self.quant.deinit();
        self.conv_in.deinit();
        self.mid.deinit();
        for (self.stages) |*st| st.deinit(self.allocator);
        self.allocator.free(self.stages);
        free(self.norm_out);
        self.conv_out.deinit();
        free(self.mean);
        free(self.std);
    }

    fn head(self: *const VaeHalf, x: A) !A {
        const n = try vaeNorm(x, self.norm_out, self.s);
        defer free(n);
        const act = try silu(n, self.s);
        defer free(act);
        return self.conv_out.forward(act, 1, 1, self.s);
    }
};

pub const VaeDecoder = struct {
    h: VaeHalf,

    pub fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, cfg: VaeConfig) !VaeDecoder {
        var w = try loadVaeWeights(io, a, model_dir);
        defer w.deinit();
        // dims = dec_dim · [m[-1], m[-1], …, m[0]]: one stage per transition,
        // all but the last upsample.
        const n = cfg.n_mult;
        const stages = try a.alloc(Stage, n);
        for (stages, 0..) |*st, i| {
            const out_mult = cfg.mult[n - 1 - i];
            st.* = try Stage.load(&w, a, s, "decoder.up_blocks.{d}", "upsampler", i, cfg.num_res_blocks + 1, cfg.dec_dim * out_mult, i + 1 < n and cfg.temporal[n - 2 - i]);
        }
        return .{ .h = .{
            .allocator = a,
            .s = s,
            .quant = try Conv.load(&w, a, s, "post_quant_conv", .{}),
            .conv_in = try Conv.load(&w, a, s, "decoder.conv_in", .{}),
            .mid = try Mid.load(&w, a, s, "decoder"),
            .stages = stages,
            .norm_out = try loadVaeNorm(&w, a, s, "decoder.norm_out", .{}),
            .conv_out = try Conv.load(&w, a, s, "decoder.conv_out", .{}),
            .mean = try channelVec(cfg.mean, s),
            .std = try channelVec(cfg.std, s),
        } };
    }

    pub fn deinit(self: *VaeDecoder) void {
        self.h.deinit();
    }

    /// Normalized latent [1, z, h, w] → pixels [1, 3, 16h, 16w] f32 in [-1, 1].
    /// The 4th output channel carries edit masks, not image content.
    pub fn decode(self: *const VaeDecoder, latent: A) !A {
        const s = self.h.s;
        const lf = try astype(latent, .float32, s);
        defer free(lf);
        const nhwc = try transpose(lf, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer free(nhwc);
        const scaled = try mulA(nhwc, self.h.std, s);
        defer free(scaled);
        const z = try addA(scaled, self.h.mean, s);
        defer free(z);
        const pq = try self.h.quant.forward(z, 1, 0, s);
        defer free(pq);
        const stem = try self.h.conv_in.forward(pq, 1, 1, s);
        defer free(stem);
        var x = try self.h.mid.forward(stem, s);
        errdefer free(x);
        for (self.h.stages) |*st| {
            const nx = try st.up(x, s);
            free(x);
            x = nx;
            try mlx.check(mlx.mlx_array_eval(x));
        }
        defer free(x);
        const out = try self.h.head(x);
        defer free(out);
        const rgb = try sliceAxis(out, 3, 0, 3, s);
        defer free(rgb);
        return transpose(rgb, &[_]c_int{ 0, 3, 1, 2 }, s);
    }
};

pub const VaeEncoder = struct {
    h: VaeHalf,
    z_dim: c_int,

    pub fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, cfg: VaeConfig) !VaeEncoder {
        var w = try loadVaeWeights(io, a, model_dir);
        defer w.deinit();
        const n = cfg.n_mult;
        const stages = try a.alloc(Stage, n);
        for (stages, 0..) |*st, i|
            st.* = try Stage.load(&w, a, s, "encoder.down_blocks.{d}", "downsampler", i, cfg.num_res_blocks, cfg.base_dim * cfg.mult[i], cfg.temporal[i]);
        return .{ .z_dim = @intCast(cfg.z_dim), .h = .{
            .allocator = a,
            .s = s,
            .quant = try Conv.load(&w, a, s, "quant_conv", .{}),
            .conv_in = try Conv.load(&w, a, s, "encoder.conv_in", .{}),
            .mid = try Mid.load(&w, a, s, "encoder"),
            .stages = stages,
            .norm_out = try loadVaeNorm(&w, a, s, "encoder.norm_out", .{}),
            .conv_out = try Conv.load(&w, a, s, "encoder.conv_out", .{}),
            .mean = try channelVec(cfg.mean, s),
            .std = try channelVec(cfg.std, s),
        } };
    }

    pub fn deinit(self: *VaeEncoder) void {
        self.h.deinit();
    }

    /// Pixels [1, 3, H, W] f32 in [-1, 1] → normalized latent mean [1, z, H/16, W/16].
    pub fn encode(self: *const VaeEncoder, image: A) !A {
        const s = self.h.s;
        const nhwc = try transpose(image, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer free(nhwc);
        const alpha_src = try sliceAxis(nhwc, 3, 0, 1, s);
        defer free(alpha_src);
        const zeroed = try mulScalar(alpha_src, 0.0, s);
        defer free(zeroed);
        const alpha = try addScalar(zeroed, 1.0, s);
        defer free(alpha);
        const rgba = try concat(&.{ nhwc, alpha }, 3, s);
        defer free(rgba);
        var x = try self.h.conv_in.forward(rgba, 1, 1, s);
        errdefer free(x);
        for (self.h.stages) |*st| {
            const nx = try st.down(x, s);
            free(x);
            x = nx;
            try mlx.check(mlx.mlx_array_eval(x));
        }
        defer free(x);
        const m = try self.h.mid.forward(x, s);
        defer free(m);
        const moments = try self.h.head(m);
        defer free(moments);
        const q = try self.h.quant.forward(moments, 1, 0, s);
        defer free(q);
        const mu = try sliceAxis(q, 3, 0, self.z_dim, s);
        defer free(mu);
        const centered = try subA(mu, self.h.mean, s);
        defer free(centered);
        const normed = try divA(centered, self.h.std, s);
        defer free(normed);
        return transpose(normed, &[_]c_int{ 0, 3, 1, 2 }, s);
    }
};

// ── Engine ──

/// Debug line: MLX active / peak GB at a named point of a generation.
fn logMemory(stage: []const u8) void {
    var active: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_peak_memory(&peak);
    const gb = 1024.0 * 1024.0 * 1024.0;
    log.debug("[qwen-image] memory after {s}: active {d:.2} GB, peak {d:.2} GB\n", .{ stage, @as(f64, @floatFromInt(active)) / gb, @as(f64, @floatFromInt(peak)) / gb });
    _ = mlx.mlx_reset_peak_memory();
}

pub const GenOpts = struct {
    /// img2img source [1,3,H,W] f32 [0,1], already at the target size.
    init_image: ?A = null,
    start_step: u32 = 0,
    /// Real CFG: any scale but 1.0 costs a second forward per step, against
    /// `negative_prompt` (blank = the reference's empty-prompt encode).
    guidance_scale: f32 = 1.0,
    negative_prompt: []const u8 = "",
};

pub fn cfgActive(opts: GenOpts) bool {
    return opts.guidance_scale != 1.0;
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,
    model_dir: []u8,
    dit_cfg: DitConfig,
    vae_cfg: VaeConfig,
    dit: Dit,
    vae: VaeDecoder,
    tok: tok_mod.Tokenizer,
    /// Template tokens dropped from the front of the conditioning sequence.
    drop_tokens: usize,
    /// The text encoder is the largest part of a small pack and idle for the
    /// whole denoise, so a STAGED engine loads it per request and frees it
    /// before the first DiT forward.
    staged: bool,
    te: ?TextEncoder = null,
    /// Loaded on the first img2img request; txt2img never pays for it.
    vae_enc: ?VaeEncoder = null,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, staged: bool) !*Engine {
        const dit_cfg = try DitConfig.parse(io, allocator, model_dir);
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .s = mlx.mlx_default_gpu_stream_new(),
            .model_dir = try allocator.dupe(u8, model_dir),
            .dit_cfg = dit_cfg,
            .vae_cfg = undefined,
            .dit = undefined,
            .vae = undefined,
            .tok = undefined,
            .drop_tokens = 0,
            .staged = staged,
        };
        errdefer allocator.free(self.model_dir);
        self.vae_cfg = try VaeConfig.parse(io, allocator, model_dir);
        errdefer self.vae_cfg.deinit(allocator);

        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/processor", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        errdefer self.tok.deinit();
        const prefix_ids = try self.tok.encode(allocator, SYSTEM_PREFIX);
        self.drop_tokens = prefix_ids.len;
        allocator.free(prefix_ids);

        self.dit = try Dit.load(io, allocator, self.s, model_dir, self.dit_cfg, COMPUTE);
        errdefer self.dit.deinit();
        self.vae = try VaeDecoder.load(io, allocator, self.s, model_dir, self.vae_cfg);
        errdefer self.vae.deinit();
        if (!staged) self.te = try TextEncoder.load(io, allocator, self.s, model_dir, COMPUTE);
        logMemory("load");

        log.info("[image] Qwen-Image-2.1 ready (DiT {d}×{d}, VAE {d}ch /{d}, text encoder {s})\n", .{
            self.dit_cfg.layers,                                        self.dit_cfg.hidden(), self.vae_cfg.z_dim, VAE_DOWNSAMPLE,
            if (staged) "staged per request" else "resident",
        });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        if (self.te) |*t| t.deinit();
        if (self.vae_enc) |*e| e.deinit();
        self.vae.deinit();
        self.dit.deinit();
        self.tok.deinit();
        self.vae_cfg.deinit(self.allocator);
        self.allocator.free(self.model_dir);
        self.allocator.destroy(self);
    }

    /// Prompt → conditioning [1, n, context] (evaluated; caller frees).
    fn encodePrompt(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8) !A {
        const body = if (std.mem.trim(u8, prompt, " \t\r\n").len == 0) " " else prompt;
        const text = try std.fmt.allocPrint(allocator, SYSTEM_PREFIX ++ USER_PREFIX ++ "{s}" ++ PROMPT_SUFFIX, .{body});
        defer allocator.free(text);
        const enc = try self.tok.encode(allocator, text);
        defer allocator.free(enc);
        const n = @min(enc.len, MAX_PROMPT_TOKENS);
        const ids = try allocator.alloc(i32, n);
        defer allocator.free(ids);
        const mask = try allocator.alloc(i32, n);
        defer allocator.free(mask);
        for (0..n) |i| {
            ids[i] = @intCast(enc[i]);
            mask[i] = 1;
        }
        const te = if (self.te) |*t| t else return error.TextEncoderNotLoaded;
        const hidden = try te.encode(ids, mask);
        defer free(hidden);
        const out = try sliceAxis(hidden, 1, @intCast(@min(self.drop_tokens, n)), @intCast(n), self.s);
        errdefer free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    const Cond = struct { pos: A, neg: ?A };

    fn encodeConditioning(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, opts: GenOpts) !Cond {
        if (self.te == null) self.te = try TextEncoder.load(self.io, self.allocator, self.s, self.model_dir, COMPUTE);
        defer if (self.staged) {
            self.te.?.deinit();
            self.te = null;
            _ = mlx.mlx_clear_cache();
        };
        const pos = try self.encodePrompt(allocator, prompt);
        errdefer free(pos);
        const neg: ?A = if (cfgActive(opts)) try self.encodePrompt(allocator, opts.negative_prompt) else null;
        return .{ .pos = pos, .neg = neg };
    }

    /// Returns the image [1,3,H,W] f32 in [0,1] (owned; caller frees).
    pub fn generateImage(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: GenOpts, progress: ?sse.Progress) !A {
        const s = self.s;
        const n_steps: u32 = if (steps == 0) DEFAULT_STEPS else steps;
        const lat_h: usize = height / VAE_DOWNSAMPLE;
        const lat_w: usize = width / VAE_DOWNSAMPLE;
        const n_img: c_int = @intCast(lat_h * lat_w);
        const z: c_int = @intCast(self.dit_cfg.in_ch);

        log.info("[qwen-image] {d}x{d} steps={d} guidance={d:.1} ({s}){s}\n", .{
            width,                                                           height, n_steps, opts.guidance_scale,
            if (cfgActive(opts)) "two forwards per step" else "one forward per step",
            if (opts.init_image != null) " img2img" else "",
        });
        if (progress) |p| p.emit("Encoding prompt", 0, n_steps);
        const cond = try self.encodeConditioning(allocator, prompt, opts);
        defer free(cond.pos);
        defer if (cond.neg) |n| free(n);
        logMemory("prompt encode");

        var geo = try Geometry.init(allocator, self.dit_cfg, @intCast(mlx.getShape(cond.pos)[1]), lat_h, lat_w);
        defer geo.deinit();
        var neg_geo: ?Geometry = if (cond.neg) |n| try Geometry.init(allocator, self.dit_cfg, @intCast(mlx.getShape(n)[1]), lat_h, lat_w) else null;
        defer if (neg_geo) |*g| g.deinit();

        const sigmas = try computeSigmas(allocator, n_steps, @intCast(n_img));
        defer allocator.free(sigmas);
        const start: u32 = if (opts.init_image != null) @min(opts.start_step, n_steps - 1) else 0;

        var key = mlx.mlx_array_new();
        defer free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const nsh = [_]c_int{ 1, n_img, z };
        var noise = mlx.mlx_array_new();
        defer free(noise);
        try mlx.check(mlx.mlx_random_normal(&noise, &nsh, 3, .float32, 0.0, 1.0, key, s));
        var img = if (opts.init_image) |pix| try self.noisedSource(pix, noise, sigmas[start]) else try astype(noise, COMPUTE, s);
        defer free(img);

        const run = n_steps - start;
        for (start..n_steps) |i| {
            if (progress) |p| if (p.cancelled()) return error.Cancelled;
            var v = try self.dit.forward(img, cond.pos, sigmas[i], &geo);
            defer free(v);
            if (cond.neg) |neg| {
                // uncond + scale·(cond − uncond)
                const vn = try self.dit.forward(img, neg, sigmas[i], &neg_geo.?);
                defer free(vn);
                const diff = try subA(v, vn, s);
                defer free(diff);
                const scaled = try mulScalar(diff, opts.guidance_scale, s);
                defer free(scaled);
                const blended = try addA(vn, scaled, s);
                free(v);
                v = blended;
            }
            const dv = try mulScalar(v, sigmas[i + 1] - sigmas[i], s);
            defer free(dv);
            const next = try addA(img, dv, s);
            free(img);
            img = next;
            try mlx.check(mlx.mlx_array_eval(img));
            if (progress) |p| p.emit("Generating", @intCast(i + 1 - start), run);
        }

        logMemory("denoise");
        if (progress) |p| p.emit("Decoding image", run, run);
        const grid = try reshape(img, &[_]c_int{ 1, @intCast(lat_h), @intCast(lat_w), z }, s);
        defer free(grid);
        const latent = try transpose(grid, &[_]c_int{ 0, 3, 1, 2 }, s);
        defer free(latent);
        const decoded = try self.vae.decode(latent);
        defer free(decoded);
        try mlx.check(mlx.mlx_array_eval(decoded));
        logMemory("vae decode");
        return denormImage(decoded, s);
    }

    /// img2img start latent: (1 − σ)·encode(source) + σ·noise, packed [1, N, z].
    fn noisedSource(self: *Engine, pixels: A, noise: A, sigma: f32) !A {
        const s = self.s;
        if (self.vae_enc == null) self.vae_enc = try VaeEncoder.load(self.io, self.allocator, s, self.model_dir, self.vae_cfg);
        const doubled = try mulScalar(pixels, 2.0, s);
        defer free(doubled);
        const signed = try addScalar(doubled, -1.0, s);
        defer free(signed);
        const z0 = try self.vae_enc.?.encode(signed);
        defer free(z0);
        const nhwc = try transpose(z0, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer free(nhwc);
        const packed_z = try reshape(nhwc, mlx.getShape(noise), s);
        defer free(packed_z);
        const kept = try mulScalar(packed_z, 1.0 - sigma, s);
        defer free(kept);
        const added = try mulScalar(noise, sigma, s);
        defer free(added);
        const mixed = try addA(kept, added, s);
        defer free(mixed);
        return astype(mixed, COMPUTE, s);
    }
};

/// [-1,1] → clip(x·0.5 + 0.5, 0, 1).
fn denormImage(decoded: A, s: S) !A {
    const half = try mulScalar(decoded, 0.5, s);
    defer free(half);
    const shifted = try addScalar(half, 0.5, s);
    defer free(shifted);
    const lo = mlx.mlx_array_new_float(0.0);
    defer free(lo);
    const hi = mlx.mlx_array_new_float(1.0);
    defer free(hi);
    var floor = mlx.mlx_array_new();
    defer free(floor);
    try mlx.check(mlx.mlx_maximum(&floor, shifted, lo, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_minimum(&out, floor, hi, s));
    return out;
}

// ── Tests ──

const testing = std.testing;

test "QwenImage sigmas match the reference schedule" {
    // mflux LinearScheduler, 512x320 (seq 640), 8 steps.
    const want = [_]f32{ 1.0, 0.90480375, 0.79888034, 0.68030787, 0.54667729, 0.39492542, 0.22109789, 0.02, 0.0 };
    const got = try computeSigmas(testing.allocator, 8, 640);
    defer testing.allocator.free(got);
    for (want, got) |w, g| try testing.expectApproxEqAbs(w, g, 1e-5);
    // One step has no span to stretch.
    const one = try computeSigmas(testing.allocator, 1, 640);
    defer testing.allocator.free(one);
    try testing.expectEqualSlices(f32, &.{ 1.0, 0.0 }, one);
}

test "QwenImage convs strip only past the unfold budget" {
    const k3 = [_]c_int{ 144, 3, 3, 288 };
    try testing.expectEqual(@as(c_int, 1), stripCount(&.{ 1, 64, 64, 1152 }, &.{ 1152, 3, 3, 1152 }, 1));
    try testing.expectEqual(@as(c_int, 21), stripCount(&.{ 1, 1024, 1024, 288 }, &k3, 1));
    try testing.expectEqual(@as(c_int, 1), stripCount(&.{ 1, 1024, 1024, 288 }, &.{ 144, 1, 1, 288 }, 1));
    try testing.expectEqual(@as(c_int, 1), stripCount(&.{ 1, 1024, 1024, 288 }, &k3, 2));
}

test "QwenImage strip conv equals the whole-image conv" {
    const s = mlx.mlx_default_gpu_stream_new();
    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, 1));
    const draw = struct {
        fn f(shape: []const c_int, k: A, st: S) !A {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0.0, 1.0, k, st));
            return o;
        }
    }.f;
    var conv = Conv{ .w = try draw(&.{ 5, 3, 3, 4 }, key, s), .b = try draw(&.{5}, key, s) };
    defer conv.deinit();
    const x = try draw(&.{ 1, 13, 7, 4 }, key, s);
    defer free(x);
    const whole = try conv.forward(x, 1, 1, s);
    defer free(whole);
    // 13 rows in 4 strips: uneven bands, and both image borders.
    const strips = try conv.forwardStrips(x, 4, s);
    defer free(strips);
    try expectParity("strip conv", strips, whole, s);
}

test "QwenImage stages band only when their activations are large" {
    // 64x64 latent stage: far under the budget, one band.
    try testing.expectEqual(@as(c_int, 64), bandRows(&.{ 1, 64, 64, 1152 }, 1152, 2));
    // 512 -> 1024 rows at 288 channels: 1.2 GB whole, banded.
    try testing.expectEqual(@as(c_int, 56), bandRows(&.{ 1, 512, 512, 576 }, 288, 2));
    // Never below 8 rows, always even.
    try testing.expectEqual(@as(c_int, 8), bandRows(&.{ 1, 4096, 8192, 1152 }, 1152, 2));
}

test "QwenImage guidance 1.0 never pays for the second forward" {
    try testing.expect(!cfgActive(.{}));
    try testing.expect(!cfgActive(.{ .guidance_scale = 1.0, .negative_prompt = "blurry" }));
    try testing.expect(cfgActive(.{ .guidance_scale = 4.0 }));
}

const Parity = struct { cos: f32, rms_ratio: f32 };

/// Cosine AND rms ratio: a cosine alone cannot see a scale error.
fn parity(got: A, want: A, s: S) !Parity {
    const g32 = try astype(got, .float32, s);
    defer free(g32);
    const g = try reshape(g32, &[_]c_int{-1}, s);
    defer free(g);
    const w = try reshape(want, &[_]c_int{-1}, s);
    defer free(w);
    const dot = struct {
        fn f(x: A, y: A, st: S) !f32 {
            const p = try mulA(x, y, st);
            defer free(p);
            var o = mlx.mlx_array_new();
            defer free(o);
            try mlx.check(mlx.mlx_sum_axis(&o, p, 0, false, st));
            var v: f32 = 0;
            try mlx.check(mlx.mlx_array_item_float32(&v, o));
            return v;
        }
    }.f;
    const gg = try dot(g, g, s);
    const ww = try dot(w, w, s);
    return .{ .cos = (try dot(g, w, s)) / (@sqrt(gg) * @sqrt(ww)), .rms_ratio = @sqrt(gg / ww) };
}

fn expectParity(name: []const u8, got: A, want: A, s: S) !void {
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const p = try parity(got, want, s);
    std.debug.print("[qwen-image] {s}: cos={d:.6} rms_ratio={d:.6}\n", .{ name, p.cos, p.rms_ratio });
    try testing.expect(p.cos > 0.9999);
    try testing.expectApproxEqAbs(@as(f32, 1.0), p.rms_ratio, 1e-3);
}

const Fixture = struct {
    dir: []const u8,
    fx: Weights,

    // Both from tests/dump_qwen_image21_fixtures.py; skips when unset.
    fn open() !Fixture {
        const dir = std.mem.span(std.c.getenv("QWEN_IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
        const path = std.mem.span(std.c.getenv("QWEN_IMAGE_FIXTURE") orelse return error.SkipZigTest);
        return .{ .dir = dir, .fx = try model_mod.loadWeightsSingleFile(testing.allocator, path) };
    }
    fn get(self: *const Fixture, key: []const u8) !A {
        return self.fx.get(key) orelse error.MissingFixtureTensor;
    }
};

test "QwenImage DiT parity (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    const cfg = try DitConfig.parse(io, a, f.dir);
    var dit = try Dit.load(io, a, s, f.dir, cfg, .float32);
    defer dit.deinit();

    const txt = try f.get("dit_txt");
    var hw: [2]i32 = undefined;
    const hw_arr = try f.get("dit_lat_hw");
    try mlx.check(mlx.mlx_array_eval(hw_arr));
    @memcpy(&hw, mlx.mlx_array_data_int32(hw_arr).?[0..2]);
    var geo = try Geometry.init(a, cfg, @intCast(mlx.getShape(txt)[1]), @intCast(hw[0]), @intCast(hw[1]));
    defer geo.deinit();

    const half: c_int = @intCast(cfg.head_dim / 2);
    const table = try reshape(geo.cos, &[_]c_int{ -1, half }, s);
    defer free(table);
    try expectParity("rope cos", table, try f.get("dit_rope_cos"), s);

    var t: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&t, try f.get("dit_t")));
    const out = try dit.forward(try f.get("dit_img"), txt, t, &geo);
    defer free(out);
    try expectParity("dit", out, try f.get("dit_out"), s);

    // The serving dtype: no f32 scalar or table may widen the bf16 stream.
    var dit16 = try Dit.load(io, a, s, f.dir, cfg, .bfloat16);
    defer dit16.deinit();
    const out16 = try dit16.forward(try f.get("dit_img"), txt, t, &geo);
    defer free(out16);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(out16));
    try testing.expect((try parity(out16, try f.get("dit_out"), s)).cos > 0.99);
}

test "QwenImage VAE parity (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var cfg = try VaeConfig.parse(io, a, f.dir);
    defer cfg.deinit(a);
    var dec = try VaeDecoder.load(io, a, s, f.dir, cfg);
    defer dec.deinit();
    const decoded = try dec.decode(try f.get("vae_latent"));
    defer free(decoded);
    try expectParity("vae decode", decoded, try f.get("vae_decoded"), s);

    var enc = try VaeEncoder.load(io, a, s, f.dir, cfg);
    defer enc.deinit();
    const encoded = try enc.encode(try f.get("vae_image"));
    defer free(encoded);
    try expectParity("vae encode", encoded, try f.get("vae_encoded"), s);

    // Same oracle with every stage forced into 8-row bands: banding is exact.
    band_budget_bytes = 1;
    defer band_budget_bytes = 256 << 20;
    const banded_dec = try dec.decode(try f.get("vae_latent"));
    defer free(banded_dec);
    try expectParity("vae decode, banded", banded_dec, try f.get("vae_decoded"), s);
    const banded_enc = try enc.encode(try f.get("vae_image"));
    defer free(banded_enc);
    try expectParity("vae encode, banded", banded_enc, try f.get("vae_encoded"), s);
}

// Whole pipeline on a REAL converted pack (QWEN_IMAGE_E2E_MODEL), text encoder
// staged. Asserts a finite, non-flat image; QWEN_IMAGE_E2E_OUT=<file.png> keeps
// it for a look, QWEN_IMAGE_E2E_STEPS / _SIZE / _GUIDANCE override the run.
test "QwenImage e2e on a real pack (env-gated)" {
    const dir = std.mem.span(std.c.getenv("QWEN_IMAGE_E2E_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const envInt = struct {
        fn f(name: [*:0]const u8, default: u32) u32 {
            const v = std.c.getenv(name) orelse return default;
            return std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
        }
    }.f;
    const steps = envInt("QWEN_IMAGE_E2E_STEPS", 8);
    const size = envInt("QWEN_IMAGE_E2E_SIZE", 512);

    // The server caps the MLX buffer pool in main(); uncapped, freed step
    // buffers pile up and the process footprint says nothing about the need.
    var prev_cap: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev_cap, 1 << 30);
    defer _ = mlx.mlx_set_cache_limit(&prev_cap, prev_cap);
    const engine = try Engine.load(io, a, dir, true);
    defer engine.deinit();
    const gb = 1024.0 * 1024.0 * 1024.0;
    var load_peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&load_peak);
    _ = mlx.mlx_reset_peak_memory();
    const prompt = "A red fox sitting in fresh snow, holding a wooden sign that says \"MLX\", soft morning light, photograph";
    const img = try engine.generateImage(a, prompt, size, size, 42, steps, .{
        .guidance_scale = @floatFromInt(envInt("QWEN_IMAGE_E2E_GUIDANCE", 1)),
        .negative_prompt = "blurry, low quality",
    }, null);
    defer free(img);
    try testing.expectEqualSlices(c_int, &.{ 1, 3, @intCast(size), @intCast(size) }, mlx.getShape(img));
    try testing.expect(engine.te == null); // staged: freed before the denoise

    const flat = try reshape(img, &[_]c_int{-1}, engine.s);
    defer free(flat);
    var variance = mlx.mlx_array_new();
    defer free(variance);
    try mlx.check(mlx.mlx_var_axis(&variance, flat, 0, false, 0, engine.s));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, variance));
    var gen_peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&gen_peak);
    std.debug.print("[qwen-image] e2e {d}x{d} steps={d}: pixel variance {d:.5}, MLX peak {d:.2} GB at load, {d:.2} GB generating\n", .{
        size, size, steps, v, @as(f64, @floatFromInt(load_peak)) / gb, @as(f64, @floatFromInt(gen_peak)) / gb,
    });
    try testing.expect(std.math.isFinite(v) and v > 1e-3);

    if (std.c.getenv("QWEN_IMAGE_E2E_OUT")) |out| {
        const png = try @import("krea.zig").imageToPng(a, img, engine.s);
        defer a.free(png);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(out), .data = png });
    }
}
