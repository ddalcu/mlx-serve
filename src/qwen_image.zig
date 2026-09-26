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
const qwen_kernels = @import("qwen_image_kernels.zig");
const qwen_image_edit = @import("qwen_image_edit.zig");
const model_discovery = @import("model_discovery.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const MfLinear = mage_flow.MfLinear;
const TextEncoder = mage_flow.TextEncoder;
const VisionTower = mage_flow.VisionTower;
const stb = @import("stb");

/// The reference's recommended sampling: 40 steps, no guidance.
pub const DEFAULT_STEPS: u32 = 40;

pub fn resolveSteps(steps: u32) u32 {
    return if (steps == 0) DEFAULT_STEPS else steps;
}

const VAE_DOWNSAMPLE: u32 = 16;
/// Text encoding stays bf16; the DiT defaults to bf16 and the VAE stays f32.
const COMPUTE: mlx.mlx_dtype = .bfloat16;
/// Experimental DiT-only dtype override. Unrecognized values preserve the
/// checkpoint's default; text encoding and VAE decoding keep their own dtypes.
fn ditDtype(value: ?[*:0]const u8) mlx.mlx_dtype {
    return if (value) |v| if (std.mem.eql(u8, std.mem.span(v), "fp16")) .float16 else COMPUTE else COMPUTE;
}

fn fusedRopeEnabled(value: ?[*:0]const u8) bool {
    return if (value) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
}

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

/// sdpa with an explicit boolean mask ([.., q_len, kv_len], true = attend).
fn sdpaMasked(q: A, k: A, v: A, scale: f32, mask: A, s: S) !A {
    const none = A{ .ctx = null };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "", mask, none, false, s));
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

    /// Keep the original image positions; rebuilding with text_len=0 would
    /// change the frame-axis RoPE and therefore the model's output.
    fn imageOnly(self: *const Geometry, s: S) !Geometry {
        const L = mlx.getShape(self.cos)[1];
        const cos = try sliceAxis(self.cos, 1, self.text_len, L, s);
        errdefer free(cos);
        const sin = try sliceAxis(self.sin, 1, self.text_len, L, s);
        errdefer free(sin);
        return .{
            .text_len = 0,
            .cos = cos,
            .sin = sin,
            .mod_row = try sliceAxis(self.mod_row, 0, self.text_len, L, s),
        };
    }

    pub fn deinit(self: *Geometry) void {
        free(self.cos);
        free(self.sin);
        free(self.mod_row);
    }
};

/// The t=0 prefix cannot attend to the target, so its per-layer K/V is
/// invariant across steps. One cache belongs to one request and CFG branch.
const PrefixKV = struct {
    k: A,
    v: A,

    fn deinit(self: *PrefixKV) void {
        free(self.k);
        free(self.v);
    }

    fn capture(k: A, v: A, text_len: c_int, s: S) !PrefixKV {
        // sliceAxis requests row-contiguous storage. The pinned MLX runtime
        // also copies contiguous views when their backing allocation exceeds
        // the slice by 16 KiB, so image-sized buffers cannot stay pinned.
        // mlx_copy is intentionally not used: it shares the backing buffer.
        const kt = try sliceAxis(k, 2, 0, text_len, s);
        errdefer free(kt);
        return .{ .k = kt, .v = try sliceAxis(v, 2, 0, text_len, s) };
    }
};

const PrefixCache = struct {
    allocator: std.mem.Allocator,
    layers: []?PrefixKV,
    image_geo: Geometry,
    ready: bool = false,

    fn init(a: std.mem.Allocator, n_layers: usize, geo: *const Geometry, s: S) !PrefixCache {
        const layers = try a.alloc(?PrefixKV, n_layers);
        errdefer a.free(layers);
        @memset(layers, null);
        return .{ .allocator = a, .layers = layers, .image_geo = try geo.imageOnly(s) };
    }

    fn initEdit(a: std.mem.Allocator, n_layers: usize, geo: *const EditGeometry, s: S) !PrefixCache {
        return init(a, n_layers, &geo.ropeGeometry(), s);
    }

    fn deinit(self: *PrefixCache) void {
        for (self.layers) |*kv| if (kv.*) |*pair| pair.deinit();
        self.allocator.free(self.layers);
        self.image_geo.deinit();
    }

    fn materialize(self: *PrefixCache, output: A) !void {
        const arrays = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(arrays);
        try mlx.check(mlx.mlx_vector_array_append_value(arrays, output));
        for (self.layers) |kv| {
            const pair = kv orelse return error.IncompletePrefixCache;
            try mlx.check(mlx.mlx_vector_array_append_value(arrays, pair.k));
            try mlx.check(mlx.mlx_vector_array_append_value(arrays, pair.v));
        }
        // Evaluate together to share first-step computation and detach the
        // cache from the full lazy graph before the next denoising step.
        try mlx.check(mlx.mlx_eval(arrays));
        self.ready = true;
    }
};

pub fn prefixCacheEnabled(value: ?[*:0]const u8) bool {
    return if (value) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
}

// ── Edit (ti2i) geometry ──

/// One run of the edit joint prefix: a text run or a whole image block.
pub const EditSeg = struct { start: c_int, end: c_int, is_text: bool };

/// Per-request constants of the edit (ti2i) joint sequence: the VLM output
/// (text 1x, image slots 4x each — one slot is 2x2 latent tokens) with the
/// target's slots appended and expanded the same way. Block boundaries come
/// from the img_shapes token counts, never from mask runs (two adjacent
/// images are separate blocks); the layout is step-invariant, so it builds
/// once per request and is reused by every step and layer.
pub const EditGeometry = struct {
    allocator: std.mem.Allocator,
    joint_len: c_int,
    /// First target row; `segments` covers [0, target_start).
    target_start: c_int,
    target_tokens: c_int,
    cos: A, // [1, L, 1, head_dim/2, 1] f32
    sin: A,
    /// [L] i32: 1 on text/ref tokens (the t=0 modulation row), 0 on target.
    mod_row: A,
    /// [L] i32: -1 at text, a unique id per image block.
    image_ids: A,
    /// [L] i32: 1 on target tokens.
    target_mask: A,
    segments: []EditSeg,
    /// Expanded row j reads txt row txt_row[j] (appended target slots read the
    /// zero pad) and, where img_cond holds, latents row lat_row[j] instead.
    txt_row: A, // [L] i32
    lat_row: A, // [L] i32
    img_cond: A, // [1, L, 1] bool

    /// Borrowed tables; only imageOnly's slices are owned by the cache.
    fn ropeGeometry(self: *const EditGeometry) Geometry {
        return .{ .text_len = self.target_start, .cos = self.cos, .sin = self.sin, .mod_row = self.mod_row };
    }

    /// `n` is the VLM sequence length; `mask` is the pre-expansion joint mask
    /// (1 at a VLM image slot or an appended target slot, else 0); `shapes`
    /// are the per-image (frame, lat_h, lat_w) blocks, target last.
    pub fn init(a: std.mem.Allocator, cfg: DitConfig, n: usize, mask: []const i32, shapes: []const [3]u32) !EditGeometry {
        if (shapes.len == 0 or mask.len < n) return error.QwenImageBadConfig;
        if (cfg.axes[0] + cfg.axes[1] + cfg.axes[2] != cfg.head_dim) return error.QwenImageBadConfig;
        var total_slots: usize = 0;
        for (shapes) |sh| {
            // Single frame only: the walk below counts h*w tokens per block.
            if (sh[0] != 1) return error.QwenImageBadConfig;
            const tokens: usize = @as(usize, sh[1]) * sh[2];
            if (tokens == 0 or tokens % 4 != 0) return error.QwenImageBadConfig;
            total_slots += tokens / 4;
        }
        const target_tokens: usize = @as(usize, shapes[shapes.len - 1][1]) * shapes[shapes.len - 1][2];
        if (target_tokens / 4 != mask.len - n) return error.QwenImageBadConfig;
        var ones: usize = 0;
        for (mask) |m| if (m != 0) {
            ones += 1;
        };
        if (ones != total_slots) return error.QwenImageBadConfig;

        // Pre-expansion row -> first expanded row.
        const off = try a.alloc(usize, mask.len + 1);
        defer a.free(off);
        off[0] = 0;
        for (mask, 0..) |m, p| off[p + 1] = off[p] + @as(usize, if (m != 0) 4 else 1);
        const L: usize = off[mask.len];

        const frame = try a.alloc(i64, L);
        defer a.free(frame);
        const hpos = try a.alloc(i64, L);
        defer a.free(hpos);
        const wpos = try a.alloc(i64, L);
        defer a.free(wpos);
        const ids = try a.alloc(i32, L);
        defer a.free(ids);
        const tgtm = try a.alloc(i32, L);
        defer a.free(tgtm);
        const rows = try a.alloc(i32, L);
        defer a.free(rows);
        const trow = try a.alloc(i32, L);
        defer a.free(trow);
        const lrow = try a.alloc(i32, L);
        defer a.free(lrow);
        const icond = try a.alloc(bool, L);
        defer a.free(icond);
        @memset(ids, -1);
        @memset(tgtm, 0);
        @memset(lrow, 0);

        // The rope walk: text advances a shared position on all axes; an image
        // block freezes the frame axis there and lays out a zero-centred h/w
        // grid, then position += max(lat_h, lat_w).
        var segs: std.ArrayList(EditSeg) = .empty;
        defer segs.deinit(a);
        var cursor: usize = 0;
        var position: i64 = 0;
        var img_rank: usize = 0;
        var target_start: usize = 0;
        for (shapes, 0..) |sh, blk| {
            const h: usize = sh[1];
            const w: usize = sh[2];
            const slots: usize = h * w / 4;
            var bs: usize = cursor;
            while (bs < mask.len and mask[bs] == 0) bs += 1;
            if (bs + slots > mask.len) return error.QwenImageBadConfig;
            for (bs..bs + slots) |p| if (mask[p] == 0) return error.QwenImageBadConfig;
            if (bs > cursor) {
                for (cursor..bs) |p| {
                    const pos = position + @as(i64, @intCast(p - cursor));
                    frame[off[p]] = pos;
                    hpos[off[p]] = pos;
                    wpos[off[p]] = pos;
                    trow[off[p]] = @intCast(p);
                    icond[off[p]] = false;
                }
                position += @intCast(bs - cursor);
                try segs.append(a, .{ .start = @intCast(off[cursor]), .end = @intCast(off[bs]), .is_text = true });
            }
            const e0: usize = off[bs];
            const h0: i64 = -@as(i64, @intCast(h - h / 2));
            const w0: i64 = -@as(i64, @intCast(w - w / 2));
            for (0..h * w) |t| {
                const e = e0 + t;
                frame[e] = position;
                hpos[e] = h0 + @as(i64, @intCast(t / w));
                wpos[e] = w0 + @as(i64, @intCast(t % w));
                ids[e] = @intCast(blk);
                trow[e] = @intCast(bs + t / 4);
                lrow[e] = @intCast(img_rank + t);
                icond[e] = true;
            }
            if (blk + 1 == shapes.len) {
                target_start = e0;
                @memset(tgtm[e0..e0 + h * w], 1);
            } else {
                // The target block is not a prefix segment: its rows are the
                // maskless target query range.
                try segs.append(a, .{ .start = @intCast(e0), .end = @intCast(e0 + h * w), .is_text = false });
            }
            img_rank += h * w;
            position += @intCast(@max(h, w));
            cursor = bs + slots;
        }
        if (cursor < mask.len) {
            for (cursor..mask.len) |p| {
                const pos = position + @as(i64, @intCast(p - cursor));
                frame[off[p]] = pos;
                hpos[off[p]] = pos;
                wpos[off[p]] = pos;
                trow[off[p]] = @intCast(p);
                icond[off[p]] = false;
            }
            try segs.append(a, .{ .start = @intCast(off[cursor]), .end = @intCast(L), .is_text = true });
        }
        for (0..L) |p| rows[p] = 1 - tgtm[p];

        // f32 tables throughout, like the reference's numpy tables.
        const half: usize = cfg.head_dim / 2;
        const cosb = try a.alloc(f32, L * half);
        defer a.free(cosb);
        const sinb = try a.alloc(f32, L * half);
        defer a.free(sinb);
        for (0..L) |p| {
            const pos = [3]i64{ frame[p], hpos[p], wpos[p] };
            var col: usize = 0;
            for (cfg.axes, pos) |dim, ax_pos| {
                for (0..dim / 2) |kk| {
                    const expo = @as(f32, @floatFromInt(2 * kk)) / @as(f32, @floatFromInt(dim));
                    const omega: f32 = 1.0 / std.math.pow(f32, 10000.0, expo);
                    const ang: f32 = @as(f32, @floatFromInt(ax_pos)) * omega;
                    cosb[p * half + col] = @cos(ang);
                    sinb[p * half + col] = @sin(ang);
                    col += 1;
                }
            }
        }
        const segments = try segs.toOwnedSlice(a);
        errdefer a.free(segments);

        const tsh = [_]c_int{ 1, @intCast(L), 1, @intCast(half), 1 };
        const lsh = [_]c_int{@intCast(L)};
        const csh = [_]c_int{ 1, @intCast(L), 1 };
        return .{
            .allocator = a,
            .joint_len = @intCast(L),
            .target_start = @intCast(target_start),
            .target_tokens = @intCast(target_tokens),
            .cos = mlx.mlx_array_new_data(cosb.ptr, &tsh, tsh.len, .float32),
            .sin = mlx.mlx_array_new_data(sinb.ptr, &tsh, tsh.len, .float32),
            .mod_row = mlx.mlx_array_new_data(rows.ptr, &lsh, lsh.len, .int32),
            .image_ids = mlx.mlx_array_new_data(ids.ptr, &lsh, lsh.len, .int32),
            .target_mask = mlx.mlx_array_new_data(tgtm.ptr, &lsh, lsh.len, .int32),
            .segments = segments,
            .txt_row = mlx.mlx_array_new_data(trow.ptr, &lsh, lsh.len, .int32),
            .lat_row = mlx.mlx_array_new_data(lrow.ptr, &lsh, lsh.len, .int32),
            .img_cond = mlx.mlx_array_new_data(icond.ptr, &csh, csh.len, .bool_),
        };
    }

    pub fn deinit(self: *EditGeometry) void {
        free(self.cos);
        free(self.sin);
        free(self.mod_row);
        free(self.image_ids);
        free(self.target_mask);
        free(self.txt_row);
        free(self.lat_row);
        free(self.img_cond);
        self.allocator.free(self.segments);
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
    fused_rope: bool = false,
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
        self.fused_rope = fusedRopeEnabled(std.c.getenv("MLX_SERVE_QWEN_IMAGE_FUSED_ROPE"));
        self.img_in = try loadLinear(&w, a, cfg.in_ch, dtype, s, "img_in", .{});
        const tn = try loadVecF32(&w, a, "txt_in.text_norm.weight", .{}, s);
        defer free(tn);
        self.txt_norm = try addScalar(tn, 1.0, s);
        self.txt_in = try loadLinear(&w, a, cfg.context, dtype, s, "txt_in.in_layer", .{});
        self.txt_out = try loadLinear(&w, a, H, dtype, s, "txt_in.out_layer", .{});
        // ddalcu/diffusers spell the time embedder nested + modulation.1;
        // mlx-community flattens both — probe which naming is on disk.
        const t_nested = w.get("time_text_embed.timestep_embedder.linear_1.weight") != null;
        self.t1 = if (t_nested)
            try loadLinear(&w, a, 256, dtype, s, "time_text_embed.timestep_embedder.linear_1", .{})
        else
            try loadLinear(&w, a, 256, dtype, s, "time_text_embed.linear_1", .{});
        self.t2 = if (t_nested)
            try loadLinear(&w, a, H, dtype, s, "time_text_embed.timestep_embedder.linear_2", .{})
        else
            try loadLinear(&w, a, H, dtype, s, "time_text_embed.linear_2", .{});
        self.modulation = if (w.get("modulation.1.weight") != null)
            try loadLinear(&w, a, H, dtype, s, "modulation.1", .{})
        else
            try loadLinear(&w, a, H, dtype, s, "modulation.0", .{});
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
        if (self.fused_rope) return qwen_kernels.applyRope(x, geo.cos, geo.sin, self.dtype, s);
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
    fn attention(self: *const Dit, b: *const Block, x: A, geo: *const Geometry, cache: ?*?PrefixKV, cached: bool) !A {
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

        if (cache) |slot| if (!cached) {
            if (slot.*) |*old| old.deinit();
            slot.* = null;
            slot.* = try PrefixKV.capture(k, v, T, s);
        };
        const full_k = if (cached) try concat(&.{ cache.?.*.?.k, k }, 2, s) else null;
        defer if (full_k) |a| free(a);
        const full_v = if (cached) try concat(&.{ cache.?.*.?.v, v }, 2, s) else null;
        defer if (full_v) |a| free(a);
        const qi = try sliceAxis(q, 2, T, sh[1], s);
        defer free(qi);
        const img_out = try sdpa(qi, full_k orelse k, full_v orelse v, scale, "", s);
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

    fn blockForward(self: *const Dit, b: *const Block, x: A, mod: *const StepMod, geo: *const Geometry, cache: ?*?PrefixKV, cached: bool) !A {
        const s = self.s;
        const n1 = try layerNorm(x, self.cfg.eps, s);
        defer free(n1);
        const a_in = try mulA(n1, mod.scale1, s);
        defer free(a_in);
        const attn = try self.attention(b, a_in, geo, cache, cached);
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
        return self.forwardCached(img, txt, t, geo, null);
    }

    fn forwardCached(self: *const Dit, img: A, txt: A, t: f32, joint_geo: *const Geometry, cache: ?*PrefixCache) !A {
        const s = self.s;
        const cached = if (cache) |c| c.ready else false;
        const geo = if (cached) &cache.?.image_geo else joint_geo;
        const temb = try self.timeEmbed(t);
        defer free(temb);
        const temb_act = try silu(temb, s);
        defer free(temb_act);
        var mod = try self.stepMod(temb_act, geo);
        defer mod.deinit();

        const ih = try self.img_in.forward(img, null, s);
        defer free(ih);
        var x = if (cached) try contig(ih, s) else blk: {
            const th = try self.textIn(txt);
            defer free(th);
            break :blk try concat(&.{ th, ih }, 1, s);
        };
        defer free(x);
        for (self.blocks, 0..) |*b, i| {
            const slot = if (cache) |c| &c.layers[i] else null;
            const nx = try self.blockForward(b, x, &mod, geo, slot, cached);
            free(x);
            x = nx;
        }

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
        const output = try self.proj_out.forward(scaled, null, s);
        errdefer free(output);
        if (cache) |c| if (!cached) try c.materialize(output);
        return output;
    }

    // ── Edit (ti2i) forward ──

    /// `stepMod` with the edit row selection: target tokens read the sampled-t
    /// row, text and condition-image tokens the t=0 row.
    fn editStepMod(self: *const Dit, temb_act: A, mod_row: A) !StepMod {
        const s = self.s;
        const H: c_int = @intCast(self.cfg.hidden());
        const mod = try self.modulation.forward(temb_act, null, s);
        defer free(mod);
        const opm = try addScalar(mod, 1.0, s);
        defer free(opm);
        const th = try tanhA(mod, s);
        defer free(th);
        var out: [4]A = undefined;
        for (&out, 0..) |*o, i| {
            const src = if (i % 2 == 0) opm else th;
            const lo: c_int = @as(c_int, @intCast(i)) * H;
            const part = try sliceAxis(src, 1, lo, lo + H, s);
            defer free(part);
            var rows = mlx.mlx_array_new();
            defer free(rows);
            try mlx.check(mlx.mlx_take_axis(&rows, part, mod_row, 0, s));
            o.* = try reshape(rows, &[_]c_int{ 1, -1, H }, s);
        }
        return .{ .scale1 = out[0], .gate1 = out[1], .scale2 = out[2], .gate2 = out[3] };
    }

    /// `applyRope` over the edit tables (same layout as the t2i ones).
    fn editApplyRope(self: *const Dit, x: A, cos: A, sin: A) !A {
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
        const rc = try mulA(re, cos, s);
        defer free(rc);
        const is_ = try mulA(im, sin, s);
        defer free(is_);
        const o_re = try subA(rc, is_, s);
        defer free(o_re);
        const rs = try mulA(re, sin, s);
        defer free(rs);
        const ic = try mulA(im, cos, s);
        defer free(ic);
        const o_im = try addA(rs, ic, s);
        defer free(o_im);
        const joined = try concat(&.{ o_re, o_im }, 4, s);
        defer free(joined);
        const flat = try reshape(joined, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, s);
        defer free(flat);
        return astype(flat, self.dtype, s);
    }

    /// Project → per-head RMS norm → RoPE → [1, heads, L, hd], edit tables.
    fn editHeadsOf(self: *const Dit, lin: *const MfLinear, norm: ?A, x: A, geo: *const Geometry) !A {
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
        const yr = try self.editApplyRope(yn, geo.cos, geo.sin);
        defer free(yr);
        return transpose(yr, &perm, s);
    }

    fn editAttention(self: *const Dit, b: *const Block, x: A, geo: *const EditGeometry, rope: *const Geometry, slot: ?*?PrefixKV, cached: bool) !A {
        const s = self.s;
        const sh = mlx.getShape(x);
        const q = try self.editHeadsOf(&b.q, b.norm_q, x, rope);
        defer free(q);
        const k = try self.editHeadsOf(&b.k, b.norm_k, x, rope);
        defer free(k);
        const v = try self.editHeadsOf(&b.v, null, x, rope);
        defer free(v);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.cfg.head_dim)));
        const attn = if (cached) blk: {
            const pair = slot.?.*.?;
            const keys = try concat(&.{ pair.k, k }, 2, s);
            defer free(keys);
            const values = try concat(&.{ pair.v, v }, 2, s);
            defer free(values);
            break :blk try sdpa(q, keys, values, scale, "", s);
        } else blk: {
            if (slot) |kv| kv.* = try PrefixKV.capture(k, v, geo.target_start, s);
            break :blk try segmentSdpaWalk(q, k, v, geo, scale, s, self.allocator);
        };
        defer free(attn);
        const back = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer free(back);
        const flat = try reshape(back, &[_]c_int{ sh[0], sh[1], sh[2] }, s);
        defer free(flat);
        return b.o.forward(flat, null, s);
    }

    fn editBlockForward(self: *const Dit, b: *const Block, x: A, mod: *const StepMod, geo: *const EditGeometry, rope: *const Geometry, slot: ?*?PrefixKV, cached: bool) !A {
        const s = self.s;
        const n1 = try layerNorm(x, self.cfg.eps, s);
        defer free(n1);
        const a_in = try mulA(n1, mod.scale1, s);
        defer free(a_in);
        const attn = try self.editAttention(b, a_in, geo, rope, slot, cached);
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

    /// Velocity for one edit flow step: `latents` are the PACKED [refs |
    /// target] latent tokens, `hidden` the VLM output; the joint sequence
    /// (VLM slots expanded 4x, ref latents substituted in order, target rows
    /// appended) lives in `geo` — `mask` is what `geo` was built from.
    /// Returns the target rows [1, target_tokens, out_ch].
    pub fn forwardEdit(self: *Dit, latents: A, hidden: A, mask: []const i32, t: f32, geo: *const EditGeometry) !A {
        return self.forwardEditCached(latents, hidden, mask, t, geo, null);
    }

    /// First call receives [refs | target]; once ready, only target latents
    /// enter the DiT. The original block-causal prefix and RoPE stay fixed.
    fn forwardEditCached(self: *Dit, latents: A, hidden: A, mask: []const i32, t: f32, geo: *const EditGeometry, cache: ?*PrefixCache) !A {
        _ = mask;
        const s = self.s;
        const cached = if (cache) |c| c.ready else false;
        const full_rope = geo.ropeGeometry();
        const rope = if (cached) &cache.?.image_geo else &full_rope;
        const temb = try self.timeEmbed(t);
        defer free(temb);
        const temb_act = try silu(temb, s);
        defer free(temb_act);
        var mod = try self.editStepMod(temb_act, rope.mod_row);
        defer mod.deinit();

        const H: c_int = @intCast(self.cfg.hidden());
        var x = if (cached) try self.img_in.forward(latents, null, s) else blk: {
            const th = try self.textIn(hidden);
            defer free(th);
            var zeros = mlx.mlx_array_new();
            defer free(zeros);
            const zsh = [_]c_int{ 1, @divExact(geo.target_tokens, 4), H };
            try mlx.check(mlx.mlx_zeros(&zeros, &zsh, zsh.len, self.dtype, s));
            const joint_txt = try concat(&.{ th, zeros }, 1, s);
            defer free(joint_txt);
            var txt_rows = mlx.mlx_array_new();
            defer free(txt_rows);
            try mlx.check(mlx.mlx_take_axis(&txt_rows, joint_txt, geo.txt_row, 1, s));
            const lat = try self.img_in.forward(latents, null, s);
            defer free(lat);
            var lat_rows = mlx.mlx_array_new();
            defer free(lat_rows);
            try mlx.check(mlx.mlx_take_axis(&lat_rows, lat, geo.lat_row, 1, s));
            var joint = mlx.mlx_array_new();
            errdefer free(joint);
            try mlx.check(mlx.mlx_where(&joint, geo.img_cond, lat_rows, txt_rows, s));
            break :blk joint;
        };
        defer free(x);
        for (self.blocks, 0..) |*b, i| {
            const slot = if (cache) |c| &c.layers[i] else null;
            const nx = try self.editBlockForward(b, x, &mod, geo, rope, slot, cached);
            free(x);
            x = nx;
        }
        // Prefix rows use t=0; target-only steps retain their original t row.
        const n = try layerNorm(x, self.cfg.eps, s);
        defer free(n);
        const sc2 = try self.norm_out.forward(temb_act, null, s);
        defer free(sc2);
        var sel = mlx.mlx_array_new();
        defer free(sel);
        try mlx.check(mlx.mlx_take_axis(&sel, sc2, rope.mod_row, 0, s));
        const selr = try reshape(sel, &[_]c_int{ 1, -1, H }, s);
        defer free(selr);
        const opsc = try addScalar(selr, 1.0, s);
        defer free(opsc);
        const scaled = try mulA(n, opsc, s);
        defer free(scaled);
        const out = try self.proj_out.forward(scaled, null, s);
        defer free(out);
        const result = try sliceAxis(out, 1, if (cached) 0 else geo.target_start, mlx.getShape(out)[1], s);
        errdefer free(result);
        if (cache) |c| if (!cached) try c.materialize(result);
        return result;
    }
};

/// Block-causal attention over the edit joint sequence, as the reference
/// segments it: a text run is causal over kv[0..end) (mlx causal carries the
/// kv−q offset, so the fully visible prefix plus the run's own triangle is one
/// call), an image block and the target rows attend maskless up to their end.
fn segmentSdpaWalk(q: A, k: A, v: A, geo: *const EditGeometry, scale: f32, s: S, a: std.mem.Allocator) !A {
    const L: c_int = @intCast(mlx.getShape(q)[2]);
    const outs = try a.alloc(A, geo.segments.len + 1);
    defer a.free(outs);
    var filled: usize = 0;
    errdefer for (outs[0..filled]) |o| free(o);
    for (geo.segments) |seg| {
        const qs = try sliceAxis(q, 2, seg.start, seg.end, s);
        defer free(qs);
        const ks = try sliceAxis(k, 2, 0, seg.end, s);
        defer free(ks);
        const vs = try sliceAxis(v, 2, 0, seg.end, s);
        defer free(vs);
        const mode: [*:0]const u8 = if (seg.is_text) "causal" else "";
        outs[filled] = try sdpa(qs, ks, vs, scale, mode, s);
        filled += 1;
    }
    const qt = try sliceAxis(q, 2, geo.target_start, L, s);
    defer free(qt);
    outs[filled] = try sdpa(qt, k, v, scale, "", s);
    filled += 1;
    const joined = try concat(outs[0..filled], 2, s);
    for (outs[0..filled]) |o| free(o);
    return joined;
}

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
    pub fn decode(self: *const VaeDecoder, latent: A) !A {
        const rgba = try self.decodeRgba(latent);
        defer free(rgba);
        return sliceAxis(rgba, 1, 0, 3, self.h.s);
    }

    /// The fourth channel is native alpha; preserve it for transparent PNG output.
    pub fn decodeRgba(self: *const VaeDecoder, latent: A) !A {
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
        return transpose(out, &[_]c_int{ 0, 3, 1, 2 }, s);
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

    /// RGBA pixels [1, 4, H, W] f32 (alpha last, any range) → normalized
    /// latent mean [1, z, H/16, W/16]. Same body as `encode` minus the const-1
    /// alpha append: the input already carries a real mask channel.
    pub fn encodeRgba(self: *const VaeEncoder, image: A) !A {
        const s = self.h.s;
        const nhwc = try transpose(image, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer free(nhwc);
        var x = try self.h.conv_in.forward(nhwc, 1, 1, s);
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
    transparent: bool = false,
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

fn defaultDqGemmFloor(first: ?*const MfLinear, explicit_env: bool) ?usize {
    const q = first orelse return null;
    return if (!explicit_env and q.quantized and q.bits == 4) 1024 else null;
}

pub const EditOpts = struct {
    guidance_scale: f32 = 1.0,
    negative_prompt: []const u8 = "",
    /// Selected by request memory admission; direct callers default to uncached.
    prefix_cache: bool = false,
    /// Reference conditioning resolution — diffusers' per-call
    /// `output_resolution` knob. Lower = fewer joint tokens per ref (speed)
    /// at conditioning-fidelity cost; 1024 is the trained regime.
    ref_resolution: u32 = EDIT_REF_RESOLUTION,
};

/// diffusers calculate_dimensions: source aspect at output_resolution², round() each side.
pub fn editTargetSize(src_w: u32, src_h: u32, output_resolution: u32) struct { w: u32, h: u32 } {
    const wf: f64 = @floatFromInt(src_w);
    const hf: f64 = @floatFromInt(src_h);
    const res: f64 = @floatFromInt(output_resolution);
    const w: u32 = @intFromFloat(@round(@sqrt(res * res * (wf / hf))));
    // Multiply by src_h before dividing by src_w: an exact .5 quotient (a 4:3
    // source) stays exact in f64, so half-away rounding sees the true half.
    const h: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(w)) * hf / wf));
    return .{ .w = w, .h = h };
}

/// The pipeline's default output_resolution — every condition image resizes
/// at its square, whatever the target's own size.
const EDIT_REF_RESOLUTION: u32 = 1024;

/// round(x/32)*32 with a u64 intermediate (x+16 can wrap u32); floored at 32 —
/// the grid the slot math needs (smart_resize's own `max(factor, …)` floor).
/// gen.zig's `qwenSnap32` is the wire-side twin (aligned then capped); keep
/// the two in sync.
fn snap32(x: u32) u32 {
    const r: u64 = (@as(u64, x) + 16) / 32 * 32;
    const cap: u64 = @as(u64, std.math.maxInt(u32)) / 32 * 32;
    return @intCast(@min(@max(r, 32), cap));
}

/// A reference's resize target: its aspect at the request's ref resolution²
/// (`EditOpts.ref_resolution` — diffusers' `output_resolution` per-call knob),
/// each side onto the /32 grid. /32 keeps the grid the slot math needs
/// (smart_resize's own `max(factor, …)` floor).
fn refResizeDimsAt(src_w: u32, src_h: u32, res: u32) struct { w: u32, h: u32 } {
    const t = editTargetSize(src_w, src_h, res);
    return .{ .w = snap32(t.w), .h = snap32(t.h) };
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
    /// The pack's text_encoder/ carries the Qwen3-VL tower (edit capability),
    /// probed at load by `towerPresentIn`; the tower weights load on demand.
    has_tower: bool = false,

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
        self.has_tower = towerPresentIn(io, allocator, self.model_dir);
        self.vae_cfg = try VaeConfig.parse(io, allocator, model_dir);
        errdefer self.vae_cfg.deinit(allocator);

        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/processor", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        errdefer self.tok.deinit();
        const prefix_ids = try self.tok.encode(allocator, SYSTEM_PREFIX);
        self.drop_tokens = prefix_ids.len;
        allocator.free(prefix_ids);

        self.dit = try Dit.load(io, allocator, self.s, model_dir, self.dit_cfg, ditDtype(std.c.getenv("MLX_SERVE_QWEN_IMAGE_DTYPE")));
        errdefer self.dit.deinit();
        self.vae = try VaeDecoder.load(io, allocator, self.s, model_dir, self.vae_cfg);
        errdefer self.vae.deinit();
        if (!staged) self.te = try TextEncoder.load(io, allocator, self.s, model_dir, COMPUTE);
        logMemory("load");

        log.info("[image] Qwen-Image-2.1 ready (DiT {d}×{d} {s}, fused_rope={}, VAE {d}ch /{d}, text encoder {s})\n", .{
            self.dit_cfg.layers, self.dit_cfg.hidden(), @tagName(self.dit.dtype),                         self.dit.fused_rope,
            self.vae_cfg.z_dim,  VAE_DOWNSAMPLE,        if (staged) "staged per request" else "resident",
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

    /// One ti2i conditioning arm: template (the negative renders through it
    /// too, only the text differs) → tokenize (text-capped like t2i) →
    /// placeholder expansion → the joint VLM encode. Mask all-ones: one
    /// unpadded prompt, exactly what the processor emits for batch 1.
    fn encodeTi2iPrompt(
        self: *Engine,
        allocator: std.mem.Allocator,
        te: *TextEncoder,
        vit: *const VisionTower,
        prompt: []const u8,
        slots_per_image: []const usize,
        pixel_values: A,
        grids: []const [3]i64,
    ) !qwen_image_edit.EditCond {
        const a = allocator;
        const text = try qwen_image_edit.buildTi2iPrompt(a, prompt, slots_per_image.len);
        defer a.free(text);
        const enc = try self.tok.encode(a, text);
        defer a.free(enc);
        const n = @min(enc.len, MAX_PROMPT_TOKENS);
        const ids = try a.alloc(i32, n);
        defer a.free(ids);
        for (enc[0..n], 0..) |t, i| ids[i] = @intCast(t);
        const expanded = try qwen_image_edit.expandImagePads(a, ids, slots_per_image);
        defer a.free(expanded);
        const mask = try a.alloc(i32, expanded.len);
        defer a.free(mask);
        @memset(mask, 1);
        return qwen_image_edit.encodeTi2i(a, self.s, te, vit, expanded, mask, pixel_values, grids, self.drop_tokens);
    }

    /// Returns the image [1,3,H,W] f32 in [0,1] (owned; caller frees).
    pub fn generateImage(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: GenOpts, progress: ?sse.Progress) !A {
        const s = self.s;
        const n_steps = resolveSteps(steps);
        const lat_h: usize = height / VAE_DOWNSAMPLE;
        const lat_w: usize = width / VAE_DOWNSAMPLE;
        const n_img: c_int = @intCast(lat_h * lat_w);
        const z: c_int = @intCast(self.dit_cfg.in_ch);

        log.info("[qwen-image] {d}x{d} steps={d} guidance={d:.1} ({s}){s}\n", .{
            width,                                                                    height,                                          n_steps, opts.guidance_scale,
            if (cfgActive(opts)) "two forwards per step" else "one forward per step", if (opts.init_image != null) " img2img" else "",
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
        var img = if (opts.init_image) |pix| try self.noisedSource(pix, noise, sigmas[start]) else try astype(noise, self.dit.dtype, s);
        defer free(img);

        // The Q4 DiT's wide image rows favor dequantized GEMM over quantized matmul.
        const dq_override_prev = mage_flow.mf_dq_gemm_override;
        const first: ?*const MfLinear = if (self.dit.blocks.len != 0) &self.dit.blocks[0].q else null;
        if (defaultDqGemmFloor(first, std.c.getenv("MLX_SERVE_MF_DQ_GEMM") != null)) |floor|
            mage_flow.mf_dq_gemm_override = @as(?usize, floor);
        defer mage_flow.mf_dq_gemm_override = dq_override_prev;

        const run = n_steps - start;
        {
            // Request-local lifetime also covers cancellation and failed forwards.
            const use_prefix_cache = run > 1 and prefixCacheEnabled(std.c.getenv("MLX_SERVE_QWEN_IMAGE_KV_CACHE"));
            var pos_cache: ?PrefixCache = if (use_prefix_cache) try PrefixCache.init(allocator, self.dit.blocks.len, &geo, s) else null;
            defer if (pos_cache) |*c| c.deinit();
            var neg_cache: ?PrefixCache = if (use_prefix_cache and neg_geo != null) try PrefixCache.init(allocator, self.dit.blocks.len, &neg_geo.?, s) else null;
            defer if (neg_cache) |*c| c.deinit();
            for (start..n_steps) |i| {
                if (progress) |p| if (p.cancelled()) return error.Cancelled;
                var v = try self.dit.forwardCached(img, cond.pos, sigmas[i], &geo, if (pos_cache) |*c| c else null);
                defer free(v);
                if (cond.neg) |neg| {
                    // uncond + scale·(cond − uncond)
                    const vn = try self.dit.forwardCached(img, neg, sigmas[i], &neg_geo.?, if (neg_cache) |*c| c else null);
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
        }

        logMemory("denoise");
        if (progress) |p| p.emit("Decoding image", run, run);
        const grid = try reshape(img, &[_]c_int{ 1, @intCast(lat_h), @intCast(lat_w), z }, s);
        defer free(grid);
        const latent = try transpose(grid, &[_]c_int{ 0, 3, 1, 2 }, s);
        defer free(latent);
        const decoded = if (opts.transparent) try self.vae.decodeRgba(latent) else try self.vae.decode(latent);
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
        return astype(mixed, self.dit.dtype, s);
    }

    /// True when the pack's text_encoder/ carries the Qwen3-VL tower (edit capability).
    pub fn supportsEdit(self: *const Engine) bool {
        return self.has_tower;
    }

    /// Instruction edit: Qwen3-VL joint conditioning over the reference
    /// images, their clean VAE latents held constant beside the denoising
    /// target. Mirrors `generateImage`'s conventions (seeded noise, Euler
    /// sigma loop, real CFG blend, per-step progress, banded VAE decode).
    /// Returns the image [1,3,H,W] f32 in [0,1] (owned; caller frees).
    pub fn editImage(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8,
        image_bytes: []const []const u8, out_w: u32, out_h: u32, seed: u64, steps: u32,
        opts: EditOpts, progress: ?sse.Progress) !mlx.mlx_array
    {
        if (!self.has_tower) return error.QwenEditNotImplemented;
        if (image_bytes.len == 0) return error.NoReferenceImages;
        const a = allocator;
        const s = self.s;
        const n_steps = resolveSteps(steps);
        const lat_h: usize = out_h / VAE_DOWNSAMPLE;
        const lat_w: usize = out_w / VAE_DOWNSAMPLE;
        const target_tokens: usize = lat_h * lat_w;
        const n_img: c_int = @intCast(target_tokens);
        const z: c_int = @intCast(self.dit_cfg.in_ch);

        log.info("[qwen-image] edit {d}x{d} refs={d} steps={d} guidance={d:.1} refres={d} ({s})\n", .{
            out_w,                      out_h,  image_bytes.len, n_steps, opts.guidance_scale, opts.ref_resolution,
            if (opts.guidance_scale != 1.0) "two forwards per step" else "one forward per step",
        });
        if (progress) |p| p.emit("Encoding prompt", 0, n_steps);

        // 1. Per reference: decode, resize at its own aspect, patchify for the
        //    tower, VAE-encode the clean latent. The patch grid and the latent
        //    grid are the same pixels/16 (one DiT token per 16px tile).
        if (self.vae_enc == null) self.vae_enc = try VaeEncoder.load(self.io, self.allocator, s, self.model_dir, self.vae_cfg);
        var pv_chunks: std.ArrayList(A) = .empty;
        defer {
            for (pv_chunks.items) |p| free(p);
            pv_chunks.deinit(a);
        }
        var ref_lats: std.ArrayList(A) = .empty;
        defer {
            for (ref_lats.items) |l| free(l);
            ref_lats.deinit(a);
        }
        var grids: std.ArrayList([3]i64) = .empty;
        defer grids.deinit(a);
        var shapes: std.ArrayList([3]u32) = .empty;
        defer shapes.deinit(a);
        var slots: std.ArrayList(usize) = .empty;
        defer slots.deinit(a);
        // Capacity up front: the per-ref appends are assume-capacity, so a
        // failed ref leaks nothing it already handed to the lists.
        try pv_chunks.ensureTotalCapacity(a, image_bytes.len);
        try ref_lats.ensureTotalCapacity(a, image_bytes.len);
        try grids.ensureTotalCapacity(a, image_bytes.len);
        try shapes.ensureTotalCapacity(a, image_bytes.len);
        try slots.ensureTotalCapacity(a, image_bytes.len);

        for (image_bytes) |bytes| {
            var sw: c_int = 0;
            var sh: c_int = 0;
            var ch: c_int = 0;
            if (stb.stbi_info_from_memory(bytes.ptr, @intCast(bytes.len), &sw, &sh, &ch) == 0) return error.ImageDecodeFailed;
            const tgt = refResizeDimsAt(@intCast(sw), @intCast(sh), opts.ref_resolution);
            var e_owned = true;
            var e = try qwen_image_edit.prepareEditImage(a, s, bytes, tgt.w, tgt.h);
            errdefer if (e_owned) e.deinit();
            const gh: u32 = tgt.h / VAE_DOWNSAMPLE;
            const gw: u32 = tgt.w / VAE_DOWNSAMPLE;
            const pv = try qwen_image_edit.vlmPixelValues(a, s, e.rgb, gh, gw);
            errdefer free(pv);
            const lat_mean = try self.vae_enc.?.encodeRgba(e.vae_in); // [1, z, gh, gw]
            e.deinit();
            e_owned = false;
            // Pack [1, tokens, z] like `noisedSource`; refs stay CLEAN (constant
            // across steps — never noised, never re-encoded).
            const nhwc = try transpose(lat_mean, &[_]c_int{ 0, 2, 3, 1 }, s);
            free(lat_mean);
            const packed_z = try reshape(nhwc, &[_]c_int{ 1, @intCast(gh * gw), z }, s);
            free(nhwc);
            const lat = try astype(packed_z, COMPUTE, s);
            free(packed_z);
            pv_chunks.appendAssumeCapacity(pv);
            ref_lats.appendAssumeCapacity(lat);
            grids.appendAssumeCapacity(.{ 1, @intCast(gh), @intCast(gw) });
            shapes.appendAssumeCapacity(.{ 1, @intCast(gh), @intCast(gw) });
            slots.appendAssumeCapacity(@as(usize, gh) * gw / 4);
        }
        const pixel_values = try concat(pv_chunks.items, 0, s);
        defer free(pixel_values);
        const ref_latents = try concat(ref_lats.items, 1, s); // [1, ref_tokens, z]
        defer free(ref_latents);

        // 2. Conditioning (positive, + negative when guiding — a blank negative
        //    still encodes, t2i's convention): TE and tower load as ONE
        //    per-request unit and free before the denoise. A resident
        //    text-only TE frees first — exactly one LM is ever live, staged
        //    or not (the next t2i request reloads it lazily).
        if (self.te) |*te| {
            te.deinit();
            self.te = null;
            _ = mlx.mlx_clear_cache();
        }
        var vlm = try qwen_image_edit.loadTeWithTower(self.io, self.allocator, s, self.model_dir, COMPUTE);
        var vlm_freed = false;
        defer if (!vlm_freed) {
            vlm.te.deinit();
            vlm.vit.deinit();
        };
        var cond = try self.encodeTi2iPrompt(a, &vlm.te, &vlm.vit, prompt, slots.items, pixel_values, grids.items);
        defer cond.deinit(a);
        var neg_cond: ?qwen_image_edit.EditCond = if (opts.guidance_scale != 1.0)
            try self.encodeTi2iPrompt(a, &vlm.te, &vlm.vit, opts.negative_prompt, slots.items, pixel_values, grids.items)
        else
            null;
        defer if (neg_cond) |*nc| nc.deinit(a);
        vlm.te.deinit();
        vlm.vit.deinit();
        vlm_freed = true;
        _ = mlx.mlx_clear_cache();
        logMemory("edit conditioning");

        // 3. Target noise — the t2i seed semantics on the target's token grid.
        var key = mlx.mlx_array_new();
        defer free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const nsh = [_]c_int{ 1, n_img, z };
        var noise = mlx.mlx_array_new();
        defer free(noise);
        try mlx.check(mlx.mlx_random_normal(&noise, &nsh, 3, .float32, 0.0, 1.0, key, s));
        var target = try astype(noise, COMPUTE, s);
        defer free(target);

        // 4. Joint geometry: pre-expansion mask = VLM pad mask ++ target-slot
        //    ones; shapes = per-ref grids with the target LAST. The negative
        //    arm gets its OWN mask/geometry (its text length differs).
        const target_slots: usize = target_tokens / 4;
        try shapes.append(a, .{ 1, @intCast(lat_h), @intCast(lat_w) });
        const mask_pos = try a.alloc(i32, cond.n + target_slots);
        defer a.free(mask_pos);
        @memcpy(mask_pos[0..cond.n], cond.pad_mask);
        @memset(mask_pos[cond.n..], 1);
        var geo = try EditGeometry.init(a, self.dit_cfg, cond.n, mask_pos, shapes.items);
        defer geo.deinit();
        var neg_geo: ?EditGeometry = null;
        defer if (neg_geo) |*g| g.deinit();
        var mask_neg: ?[]i32 = null;
        defer if (mask_neg) |m| a.free(m);
        if (neg_cond) |*nc| {
            mask_neg = try a.alloc(i32, nc.n + target_slots);
            @memcpy(mask_neg.?[0..nc.n], nc.pad_mask);
            @memset(mask_neg.?[nc.n..], 1);
            neg_geo = try EditGeometry.init(a, self.dit_cfg, nc.n, mask_neg.?, shapes.items);
        }

        // 5. The mu/shift input counts TARGET tokens only (the reference
        //    pipeline feeds the target latents' length, never the joint's).
        const sigmas = try computeSigmas(a, n_steps, @intCast(target_tokens));
        defer a.free(sigmas);

        // 6. Euler denoise: the packed stream is [refs (constant clean) |
        //    target]; only the target rows step.
        {
            const use_cache = n_steps > 1 and opts.prefix_cache;
            var cache: ?PrefixCache = if (use_cache) try PrefixCache.initEdit(a, self.dit.blocks.len, &geo, s) else null;
            defer if (cache) |*c| c.deinit();
            var neg_cache: ?PrefixCache = if (use_cache and neg_geo != null) try PrefixCache.initEdit(a, self.dit.blocks.len, &neg_geo.?, s) else null;
            defer if (neg_cache) |*c| c.deinit();
            log.info("[qwen-image] edit prefix cache enabled={} target={d} prefix={d} negative_prefix={d}\n", .{
                use_cache, geo.target_tokens, geo.target_start, if (neg_geo) |ng| ng.target_start else @as(c_int, 0),
            });
            for (0..n_steps) |i| {
                if (progress) |p| if (p.cancelled()) return error.Cancelled;
                const model_input = if (use_cache and cache.?.ready) try contig(target, s) else try concat(&.{ ref_latents, target }, 1, s);
                defer free(model_input);
                var v = try self.dit.forwardEditCached(model_input, cond.hidden, mask_pos, sigmas[i], &geo, if (cache) |*c| c else null);
                defer free(v);
                if (neg_cond) |*nc| {
                    // uncond + scale·(cond − uncond)
                    const vn = try self.dit.forwardEditCached(model_input, nc.hidden, mask_neg.?, sigmas[i], &neg_geo.?, if (neg_cache) |*c| c else null);
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
                const next = try addA(target, dv, s);
                free(target);
                target = next;
                try mlx.check(mlx.mlx_array_eval(target));
                if (i == 0 and use_cache) log.info("[qwen-image] edit prefix cached: {d} layers, {d} branch(es); remaining steps target-only\n", .{
                    self.dit.blocks.len, @as(u32, if (neg_cache != null) 2 else 1),
                });
                if (progress) |p| p.emit("Generating", @intCast(i + 1), n_steps);
            }
        } // Release every branch's K/V before the VAE decode working set.

        // 7. Decode the target rows only — same unpack + banded path as t2i.
        logMemory("denoise");
        if (progress) |p| p.emit("Decoding image", n_steps, n_steps);
        const grid = try reshape(target, &[_]c_int{ 1, @intCast(lat_h), @intCast(lat_w), z }, s);
        defer free(grid);
        const latent = try transpose(grid, &[_]c_int{ 0, 3, 1, 2 }, s);
        defer free(latent);
        const decoded = try self.vae.decode(latent);
        defer free(decoded);
        try mlx.check(mlx.mlx_array_eval(decoded));
        logMemory("vae decode");
        return denormImage(decoded, s);
    }
};

/// Header-peek: does {model_dir}/text_encoder carry tower keys
/// (`model.visual.*` or `vision_tower.*`)?
pub fn towerPresentIn(io: anytype, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const te_path = std.fmt.allocPrint(allocator, "{s}/text_encoder", .{model_dir}) catch return false;
    defer allocator.free(te_path);
    var dir = std.Io.Dir.openDirAbsolute(io, te_path, .{ .iterate = true }) catch return false;
    defer dir.close(io);

    // The index names the shards; anything else is not peeked (indexShardSet policy).
    var referenced = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);

    var it = dir.iterate();
    while (it.next(io) catch return false) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        if (towerShardHasVisual(io, allocator, dir, entry.name)) return true;
    }
    return false;
}

const tower_header_limit: usize = 64 * 1024 * 1024;

/// Substring scan of one shard's safetensors JSON header for the tower keys —
/// `model.visual.` (diffusers/ddalcu) or `vision_tower.` (mlx-community).
fn towerShardHasVisual(io: anytype, allocator: std.mem.Allocator, dir: std.Io.Dir, name: []const u8) bool {
    const f = dir.openFile(io, name, .{}) catch return false;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    const header_len = rs.interface.takeInt(u64, .little) catch return false;
    if (header_len == 0 or header_len > tower_header_limit) return false;
    const header = allocator.alloc(u8, @intCast(header_len)) catch return false;
    defer allocator.free(header);
    rs.interface.readSliceAll(header) catch return false;
    return std.mem.indexOf(u8, header, "model.visual.") != null or
        std.mem.indexOf(u8, header, "\"vision_tower.") != null;
}

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

test "QwenImage enables wide GEMM for Q4 unless explicitly overridden" {
    const q4 = MfLinear{ .quantized = true, .w = .{ .ctx = null }, .dtype = .bfloat16, .bits = 4 };
    const q8 = MfLinear{ .quantized = true, .w = .{ .ctx = null }, .dtype = .bfloat16, .bits = 8 };
    const dense = MfLinear{ .quantized = false, .w = .{ .ctx = null }, .dtype = .bfloat16 };
    try testing.expectEqual(@as(?usize, 1024), defaultDqGemmFloor(&q4, false));
    try testing.expectEqual(@as(?usize, null), defaultDqGemmFloor(&q4, true));
    try testing.expectEqual(@as(?usize, null), defaultDqGemmFloor(&q8, false));
    try testing.expectEqual(@as(?usize, null), defaultDqGemmFloor(&dense, false));
    try testing.expectEqual(@as(?usize, null), defaultDqGemmFloor(null, false));
}

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

// Hand-pinned from calculate_dimensions: w = round(sqrt(res²·ratio)), h =
// round(w·src_h/src_w), round-half-away, before any /32 snapping.
test "QwenImage editTargetSize: source aspect at output_resolution squared, rounded" {
    const Case = struct { src_w: u32, src_h: u32, res: u32, w: u32, h: u32 };
    const cases = [_]Case{
        .{ .src_w = 512, .src_h = 512, .res = 1024, .w = 1024, .h = 1024 },
        .{ .src_w = 512, .src_h = 512, .res = 512, .w = 512, .h = 512 },
        // 4:3: h lands on the exact half 886.5 → 887 (floor would say 886).
        .{ .src_w = 4032, .src_h = 3024, .res = 1024, .w = 1182, .h = 887 },
        // 7:5: w = 1211.613 → 1212 (floor would say 1211) — round, not floor.
        .{ .src_w = 1400, .src_h = 1000, .res = 1024, .w = 1212, .h = 866 },
        .{ .src_w = 3024, .src_h = 4032, .res = 1024, .w = 887, .h = 1183 },
    };
    for (cases) |c| {
        const got = editTargetSize(c.src_w, c.src_h, c.res);
        try testing.expectEqual(c.w, got.w);
        try testing.expectEqual(c.h, got.h);
    }
}

test "QwenImage refResizeDimsAt: ref resolution is the per-request knob" {
    // (1184, 896) is the pinned 1024-regime reference for a 4:3 source; the
    // same source at 512² conditioning halves the joint tokens per ref.
    const d = refResizeDimsAt(512, 384, 1024);
    try testing.expectEqual(@as(u32, 1184), d.w);
    try testing.expectEqual(@as(u32, 896), d.h);
    const e = refResizeDimsAt(512, 384, 512);
    try testing.expectEqual(@as(u32, 576), e.w);
    try testing.expectEqual(@as(u32, 448), e.h);
    // Degenerate aspect still snaps to the /32 grid both sides.
    const f = refResizeDimsAt(100, 3000, 512);
    try testing.expectEqual(@as(u32, 96), f.w);
    try testing.expectEqual(@as(u32, 2784), f.h);
}

/// safetensors-shaped bytes: u64 LE header length + JSON + a small data pad.
fn stubSafetensors(a: std.mem.Allocator, header_json: []const u8) ![]u8 {
    const out = try a.alloc(u8, 8 + header_json.len + 16);
    std.mem.writeInt(u64, out[0..8], @intCast(header_json.len), .little);
    @memcpy(out[8 .. 8 + header_json.len], header_json);
    @memset(out[8 + header_json.len ..], 0);
    return out;
}

test "QwenImage towerPresentIn: header-peek of text_encoder" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];

    const lang = try stubSafetensors(a, "{\"model.language_model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}}");
    defer a.free(lang);
    const visual = try stubSafetensors(a, "{\"model.visual.blocks.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}}");
    defer a.free(visual);

    // No text_encoder dir at all.
    try testing.expect(!towerPresentIn(io, a, root));

    // Language-only shard: the tower is absent.
    try tmp.dir.createDirPath(io, "lang/text_encoder");
    try tmp.dir.writeFile(io, .{ .sub_path = "lang/text_encoder/model.safetensors", .data = lang });
    const lang_dir = try std.fmt.allocPrint(a, "{s}/lang", .{root});
    defer a.free(lang_dir);
    try testing.expect(!towerPresentIn(io, a, lang_dir));

    // A model.visual.* key in the header: the tower is present.
    try tmp.dir.createDirPath(io, "tower/text_encoder");
    try tmp.dir.writeFile(io, .{ .sub_path = "tower/text_encoder/model.safetensors", .data = visual });
    const tower_dir = try std.fmt.allocPrint(a, "{s}/tower", .{root});
    defer a.free(tower_dir);
    try testing.expect(towerPresentIn(io, a, tower_dir));

    // mlx-community's spelling: `vision_tower.*` keys mean the tower too.
    const vt = try stubSafetensors(a, "{\"vision_tower.patch_embed.proj.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}}");
    defer a.free(vt);
    try tmp.dir.createDirPath(io, "vt/text_encoder");
    try tmp.dir.writeFile(io, .{ .sub_path = "vt/text_encoder/model.safetensors", .data = vt });
    const vt_dir = try std.fmt.allocPrint(a, "{s}/vt", .{root});
    defer a.free(vt_dir);
    try testing.expect(towerPresentIn(io, a, vt_dir));

    // The index names only the language shard; the visual shard is not named
    // and never peeked — the index wins.
    try tmp.dir.createDirPath(io, "indexed/text_encoder");
    try tmp.dir.writeFile(io, .{ .sub_path = "indexed/text_encoder/shard-00001-of-00002.safetensors", .data = lang });
    try tmp.dir.writeFile(io, .{ .sub_path = "indexed/text_encoder/shard-00002-of-00002.safetensors", .data = visual });
    try tmp.dir.writeFile(io, .{ .sub_path = "indexed/text_encoder/model.safetensors.index.json", .data = "{\"metadata\":{\"total_size\":16},\"weight_map\":{\"model.language_model.layers.0.self_attn.q_proj.weight\":\"shard-00001-of-00002.safetensors\"}}" });
    const indexed_dir = try std.fmt.allocPrint(a, "{s}/indexed", .{root});
    defer a.free(indexed_dir);
    try testing.expect(!towerPresentIn(io, a, indexed_dir));
}

test "QwenImage edit scaffold: no capability by default, editImage refuses" {
    var e = Engine{
        .allocator = undefined,
        .io = undefined,
        .s = undefined,
        .model_dir = undefined,
        .dit_cfg = undefined,
        .vae_cfg = undefined,
        .dit = undefined,
        .vae = undefined,
        .tok = undefined,
        .drop_tokens = 0,
        .staged = false,
    };
    try testing.expect(!e.supportsEdit());
    try testing.expectError(error.QwenEditNotImplemented, e.editImage(
        testing.allocator, "make the sky red", &.{"stub bytes"}, 512, 512, 7, 4, .{}, null,
    ));
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

test "QwenImage experimental dtype and fused RoPE defaults are conservative" {
    try testing.expectEqual(COMPUTE, ditDtype(null));
    try testing.expectEqual(COMPUTE, ditDtype("bf16"));
    try testing.expectEqual(COMPUTE, ditDtype("unknown"));
    try testing.expectEqual(mlx.mlx_dtype.float16, ditDtype("fp16"));
    try testing.expect(!fusedRopeEnabled(null));
    try testing.expect(!fusedRopeEnabled("0"));
    try testing.expect(!fusedRopeEnabled("invalid"));
    try testing.expect(fusedRopeEnabled("1"));
}

test "QwenImage prefix cache is enabled unless explicitly disabled" {
    try testing.expect(prefixCacheEnabled(null));
    try testing.expect(prefixCacheEnabled("1"));
    try testing.expect(!prefixCacheEnabled("0"));
}

// Small deterministic nonzero weights exercise the actual full DiT without a
// downloaded fixture, including time modulation, causal attention and RoPE.
const TinyPrefixModel = struct {
    fn tensor(shape: []const c_int, phase: f32, scale: f32) !A {
        var n: usize = 1;
        for (shape) |d| n *= @intCast(d);
        const data = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(data);
        for (data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.173 + phase) * scale;
        return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    }

    fn linear(in: u32, out: u32, phase: f32, dtype: mlx.mlx_dtype, s: S) !MfLinear {
        const raw = try tensor(&.{ @intCast(in), @intCast(out) }, phase, 0.4 / @sqrt(@as(f32, @floatFromInt(in))));
        defer free(raw);
        return .{ .quantized = false, .w = try astype(raw, dtype, s), .dtype = dtype };
    }

    fn init(dtype: mlx.mlx_dtype, s: S) !Dit {
        const cfg: DitConfig = .{ .layers = 2, .heads = 2, .head_dim = 16, .in_ch = 8, .out_ch = 8, .context = 24, .axes = .{ 4, 6, 6 } };
        const h = cfg.hidden();
        const blocks = try testing.allocator.alloc(Block, cfg.layers);
        for (blocks, 0..) |*b, i| {
            const phase: f32 = @floatFromInt(i);
            b.* = .{
                .q = try linear(h, h, phase + 0.1, dtype, s),
                .k = try linear(h, h, phase + 0.2, dtype, s),
                .v = try linear(h, h, phase + 0.3, dtype, s),
                .o = try linear(h, h, phase + 0.4, dtype, s),
                .norm_q = try tensor(&.{@intCast(cfg.head_dim)}, 1.3, 1),
                .norm_k = try tensor(&.{@intCast(cfg.head_dim)}, 1.7, 1),
                .proj = try linear(h, h * cfg.mlp_ratio, phase + 0.5, dtype, s),
                .gate = try linear(h, h * cfg.mlp_ratio, phase + 0.6, dtype, s),
                .out = try linear(h * cfg.mlp_ratio, h, phase + 0.7, dtype, s),
            };
        }
        return .{
            .allocator = testing.allocator,
            .s = s,
            .cfg = cfg,
            .dtype = dtype,
            .img_in = try linear(cfg.in_ch, h, 0.3, dtype, s),
            .txt_norm = try tensor(&.{@intCast(cfg.context)}, 1.1, 1),
            .txt_in = try linear(cfg.context, h, 0.7, dtype, s),
            .txt_out = try linear(h, h, 1.1, dtype, s),
            .t1 = try linear(256, h, 1.5, dtype, s),
            .t2 = try linear(h, h, 1.9, dtype, s),
            .modulation = try linear(h, 4 * h, 2.3, dtype, s),
            .blocks = blocks,
            .norm_out = try linear(h, h, 2.7, dtype, s),
            .proj_out = try linear(h, cfg.out_ch, 3.1, dtype, s),
        };
    }
};

test "QwenImage prefix cache matches full forwards across timesteps and conditioning branches" {
    const s = mlx.mlx_default_gpu_stream_new();
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16, .float16 }) |dtype| {
        var dit = try TinyPrefixModel.init(dtype, s);
        defer dit.deinit();
        // Different prompt lengths/content model independent CFG branches or
        // successive requests; caches never survive beyond their owner.
        for ([_]c_int{ 1, 7 }) |text_len| {
            const txt = try TinyPrefixModel.tensor(&.{ 1, text_len, 24 }, @floatFromInt(text_len), 1);
            defer free(txt);
            var geo = try Geometry.init(testing.allocator, dit.cfg, @intCast(text_len), 2, 3);
            defer geo.deinit();
            var cache = try PrefixCache.init(testing.allocator, dit.blocks.len, &geo, s);
            defer cache.deinit();
            const head_input = try TinyPrefixModel.tensor(&.{ 1, text_len + 6, @intCast(dit.cfg.hidden()) }, 0.7, 1);
            defer free(head_input);
            // Per-head norm's f32 weights promote the intermediate to f32.
            // Both RoPE implementations must cast Q/K back to compute dtype.
            for ([_]bool{ false, true }) |fused| {
                dit.fused_rope = fused;
                const heads = try dit.headsOf(&dit.blocks[0].q, dit.blocks[0].norm_q, head_input, &geo);
                defer free(heads);
                try testing.expectEqual(dtype, mlx.mlx_array_dtype(heads));
            }
            dit.fused_rope = false;
            for ([_]f32{ 0.9, 0.5, 0.02 }, 0..) |t, step| {
                const img = try TinyPrefixModel.tensor(&.{ 1, 6, 8 }, t + 0.2, 1);
                defer free(img);
                const want = try dit.forward(img, txt, t, &geo);
                defer free(want);
                const want32 = try astype(want, .float32, s);
                defer free(want32);
                // Compare the primitive reference against fused RoPE with
                // both full first-step and cached later-step attention.
                dit.fused_rope = true;
                const got = try dit.forwardCached(img, txt, t, &geo, &cache);
                defer free(got);
                dit.fused_rope = false;
                const got32 = try astype(got, .float32, s);
                defer free(got32);
                try mlx.check(mlx.mlx_array_eval(got32));
                var elements: usize = 1;
                for (mlx.getShape(got32)) |d| elements *= @intCast(d);
                for (mlx.mlx_array_data_float32(got32).?[0..elements]) |v|
                    try testing.expect(std.math.isFinite(v));
                try testing.expect(cache.ready);
                try testing.expectEqual(dtype, mlx.mlx_array_dtype(got));
                try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
                const p = try parity(got, want32, s);
                std.debug.print("[qwen-image] prefix cache {s} T={d} step={d}: cos={d:.7} rms_ratio={d:.7}\n", .{ @tagName(dtype), text_len, step, p.cos, p.rms_ratio });
                try testing.expect(p.cos > 0.99999);
                try testing.expectApproxEqAbs(@as(f32, 1), p.rms_ratio, @as(f32, if (dtype == .float32) 1e-5 else 0.002));
                for (cache.layers) |kv| {
                    try testing.expectEqual(text_len, mlx.getShape(kv.?.k)[2]);
                    try testing.expectEqual(text_len, mlx.getShape(kv.?.v)[2]);
                }
            }
        }
    }
}

test "QwenImage edit prefix cache matches block-causal forwards across steps and reference layouts" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16, .float16 }) |dtype| {
        var dit = try TinyPrefixModel.init(dtype, s);
        defer dit.deinit();
        for ([_]usize{ 1, 2, 10 }) |refs| {
            // Unequal prompt lengths exercise independently owned CFG caches.
            for ([_]usize{ 1, 5 }) |text_len| {
                var shapes: std.ArrayList([3]u32) = .empty;
                defer shapes.deinit(a);
                var mask: std.ArrayList(i32) = .empty;
                defer mask.deinit(a);
                try mask.appendNTimes(a, 0, text_len);
                var ref_tokens: c_int = 0;
                for (0..refs) |r| {
                    const w: u32 = if (r % 2 == 0) 2 else 4;
                    try shapes.append(a, .{ 1, 2, w });
                    try mask.appendNTimes(a, 1, w / 2);
                    ref_tokens += @intCast(2 * w);
                    // Cover both adjacent image blocks and intervening text.
                    if (text_len == 5) try mask.append(a, 0);
                }
                try mask.append(a, 0);
                const n = mask.items.len;
                try shapes.append(a, .{ 1, 2, 4 });
                try mask.appendNTimes(a, 1, 2);
                var geo = try EditGeometry.init(a, dit.cfg, n, mask.items, shapes.items);
                defer geo.deinit();
                var cache = try PrefixCache.initEdit(a, dit.blocks.len, &geo, s);
                defer cache.deinit();
                const hidden = try TinyPrefixModel.tensor(&.{ 1, @intCast(n), 24 }, @floatFromInt(text_len), 1);
                defer free(hidden);
                const ref = try TinyPrefixModel.tensor(&.{ 1, ref_tokens, 8 }, @floatFromInt(refs + text_len), 1);
                defer free(ref);
                for ([_]f32{ 0.9, 0.5, 0.02 }, 0..) |t, step| {
                    const target = try TinyPrefixModel.tensor(&.{ 1, 8, 8 }, t + 0.2, 1);
                    defer free(target);
                    const full_input = try concat(&.{ ref, target }, 1, s);
                    defer free(full_input);
                    const want = try dit.forwardEdit(full_input, hidden, mask.items, t, &geo);
                    defer free(want);
                    const want32 = try astype(want, .float32, s);
                    defer free(want32);
                    const got = try dit.forwardEditCached(if (cache.ready) target else full_input, hidden, mask.items, t, &geo, &cache);
                    defer free(got);
                    const got32 = try astype(got, .float32, s);
                    defer free(got32);
                    try mlx.check(mlx.mlx_array_eval(got32));
                    for (mlx.mlx_array_data_float32(got32).?[0 .. 8 * 8]) |v|
                        try testing.expect(std.math.isFinite(v));
                    try testing.expect(cache.ready);
                    try testing.expectEqual(dtype, mlx.mlx_array_dtype(got));
                    try testing.expectEqualSlices(c_int, &.{ 1, 8, 8 }, mlx.getShape(got));
                    const p = try parity(got, want32, s);
                    const tolerance: f32 = if (dtype == .float32) 1e-5 else 0.002;
                    if (!(p.cos > 0.99999) or !(@abs(p.rms_ratio - 1) <= tolerance))
                        std.debug.print("[qwen-image] edit cache {s} refs={d} text={d} step={d}: cos={d:.7} rms_ratio={d:.7}\n", .{ @tagName(dtype), refs, text_len, step, p.cos, p.rms_ratio });
                    try testing.expect(p.cos > 0.99999);
                    try testing.expectApproxEqAbs(@as(f32, 1), p.rms_ratio, tolerance);
                    for (cache.layers) |kv| {
                        try testing.expectEqual(geo.target_start, mlx.getShape(kv.?.k)[2]);
                        try testing.expectEqual(geo.target_start, mlx.getShape(kv.?.v)[2]);
                    }
                }
            }
        }
    }
}

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

// ── Edit (ti2i) oracles: the edit half of tests/dump_qwen_image21_fixtures.py ──

/// i32 mlx array → owned host slice.
fn hostI32(a: std.mem.Allocator, arr: A) ![]i32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    const n = mlx.mlx_array_size(arr);
    const d = mlx.mlx_array_data_int32(arr) orelse return error.NoData;
    const out = try a.alloc(i32, n);
    @memcpy(out, d[0..n]);
    return out;
}

/// The edit fixture's shared setup: pack DiT config, img_shapes, and the
/// pre-expansion joint mask (the VLM pad mask with the appended target slots).
const EditSetup = struct {
    cfg: DitConfig,
    shapes: [][3]u32,
    mask: []i32,
    n: usize,

    fn deinit(self: *const EditSetup, a: std.mem.Allocator) void {
        a.free(self.shapes);
        a.free(self.mask);
    }
};

fn openEditSetup(a: std.mem.Allocator, io: std.Io, f: *const Fixture) !EditSetup {
    const cfg = try DitConfig.parse(io, a, f.dir);
    const raw = try hostI32(a, try f.get("edit_img_shapes"));
    defer a.free(raw);
    const shapes = try a.alloc([3]u32, raw.len / 3);
    errdefer a.free(shapes);
    for (0..shapes.len) |i| shapes[i] = .{ @intCast(raw[i * 3]), @intCast(raw[i * 3 + 1]), @intCast(raw[i * 3 + 2]) };
    const n: usize = @intCast(mlx.getShape(try f.get("edit_hidden"))[1]);
    const target_slots = @as(usize, shapes[shapes.len - 1][0]) * shapes[shapes.len - 1][1] * shapes[shapes.len - 1][2] / 4;
    const pad = try hostI32(a, try f.get("edit_pad_mask"));
    defer a.free(pad);
    const mask = try a.alloc(i32, pad.len + target_slots);
    @memcpy(mask[0..pad.len], pad);
    @memset(mask[pad.len..], 1);
    return .{ .cfg = cfg, .shapes = shapes, .mask = mask, .n = n };
}

test "QwenImage edit rope (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    var setup = try openEditSetup(a, io, &f);
    defer setup.deinit(a);

    var geo = try EditGeometry.init(a, setup.cfg, setup.n, setup.mask, setup.shapes);
    defer geo.deinit();
    const half: c_int = @intCast(setup.cfg.head_dim / 2);
    const cos_t = try reshape(geo.cos, &[_]c_int{ -1, half }, s);
    defer free(cos_t);
    const sin_t = try reshape(geo.sin, &[_]c_int{ -1, half }, s);
    defer free(sin_t);
    try expectParity("edit rope cos", cos_t, try f.get("edit_rope_cos"), s);
    try expectParity("edit rope sin", sin_t, try f.get("edit_rope_sin"), s);

    // The real checkpoint's (16,56,56) layout as a pure table: same walk,
    // different axes (the pack DiT's 16-wide head cannot run it).
    const real_cfg = DitConfig{ .head_dim = 128, .axes = .{ 16, 56, 56 } };
    var geo_real = try EditGeometry.init(a, real_cfg, setup.n, setup.mask, setup.shapes);
    defer geo_real.deinit();
    const cos_r = try reshape(geo_real.cos, &[_]c_int{ -1, 64 }, s);
    defer free(cos_r);
    const sin_r = try reshape(geo_real.sin, &[_]c_int{ -1, 64 }, s);
    defer free(sin_r);
    try expectParity("edit rope cos, real axes", cos_r, try f.get("edit_rope_cos_real"), s);
    try expectParity("edit rope sin, real axes", sin_r, try f.get("edit_rope_sin_real"), s);
}

test "QwenImage edit geometry (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    _ = s; // declared by the shared setup shape; this test reads host buffers only
    var setup = try openEditSetup(a, io, &f);
    defer setup.deinit(a);

    var geo = try EditGeometry.init(a, setup.cfg, setup.n, setup.mask, setup.shapes);
    defer geo.deinit();
    // The fixture's joint: 27 text + 2060 slots ×4 = 12363, target = last 4096.
    try testing.expectEqual(@as(c_int, 12363), geo.joint_len);
    try testing.expectEqual(@as(c_int, 8267), geo.target_start);
    try testing.expectEqual(@as(c_int, 4096), geo.target_tokens);

    const ids = try hostI32(a, geo.image_ids);
    defer a.free(ids);
    const ids_want = try hostI32(a, try f.get("edit_image_ids"));
    defer a.free(ids_want);
    try testing.expectEqualSlices(i32, ids_want, ids);

    const tgt = try hostI32(a, geo.target_mask);
    defer a.free(tgt);
    const tgt_want = try hostI32(a, try f.get("edit_target_token_mask"));
    defer a.free(tgt_want);
    try testing.expectEqualSlices(i32, tgt_want, tgt);

    // The reference walk's prefix segments (its own expanded-run structure).
    try testing.expectEqual(@as(usize, 5), geo.segments.len);
    const want = [_]EditSeg{
        .{ .start = 0, .end = 8, .is_text = true },
        .{ .start = 8, .end = 4152, .is_text = false },
        .{ .start = 4152, .end = 4158, .is_text = true },
        .{ .start = 4158, .end = 8254, .is_text = false },
        .{ .start = 8254, .end = 8267, .is_text = true },
    };
    for (geo.segments, want) |got, w| {
        try testing.expectEqual(w.start, got.start);
        try testing.expectEqual(w.end, got.end);
        try testing.expectEqual(w.is_text, got.is_text);
    }
}

test "QwenImage edit forward (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    var setup = try openEditSetup(a, io, &f);
    defer setup.deinit(a);

    var dit = try Dit.load(io, a, s, f.dir, setup.cfg, .float32);
    defer dit.deinit();
    var geo = try EditGeometry.init(a, setup.cfg, setup.n, setup.mask, setup.shapes);
    defer geo.deinit();
    var t_raw: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&t_raw, try f.get("edit_dit_t")));
    const out = try dit.forwardEdit(try f.get("edit_latents"), try f.get("edit_hidden"), setup.mask, t_raw / 1000.0, &geo);
    defer free(out);
    try expectParity("edit dit", out, try f.get("edit_dit_out"), s);
}

test "QwenImage edit VAE encode RGBA (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var cfg = try VaeConfig.parse(io, a, f.dir);
    defer cfg.deinit(a);
    var enc = try VaeEncoder.load(io, a, s, f.dir, cfg);
    defer enc.deinit();
    const out = try enc.encodeRgba(try f.get("edit_vae_image"));
    defer free(out);
    try expectParity("vae encode rgba", out, try f.get("edit_vae_encoded"), s);

    // Same oracle with every stage forced into 8-row bands: banding is exact.
    band_budget_bytes = 1;
    defer band_budget_bytes = 256 << 20;
    const banded = try enc.encodeRgba(try f.get("edit_vae_image"));
    defer free(banded);
    try expectParity("vae encode rgba, banded", banded, try f.get("edit_vae_encoded"), s);
}

// The equivalence invariant: the per-segment sdpa walk (what the DiT runs)
// against the dense block-causal mask (q_idx >= kv_idx or same image block)
// built explicitly — small random data, no pack.
test "QwenImage edit segment walk equals the dense block-causal mask" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    // [text 3 | ref1 (1 slot, 2x2) | text 2 | ref2 (2 slots, 2x4) | text 4 | target 4 slots, 4x4]
    const mask = [_]i32{ 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };
    const shapes = [_][3]u32{ .{ 1, 2, 2 }, .{ 1, 2, 4 }, .{ 1, 4, 4 } };
    const cfg = DitConfig{ .head_dim = 16, .axes = .{ 4, 6, 6 } };
    var geo = try EditGeometry.init(a, cfg, 12, &mask, &shapes);
    defer geo.deinit();
    // 3 + 4 + 2 + 8 + 4 + 16 = 37; target rows are [21, 37).
    try testing.expectEqual(@as(c_int, 37), geo.joint_len);
    try testing.expectEqual(@as(c_int, 21), geo.target_start);

    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, 11));
    const draw = struct {
        fn f(shape: []const c_int, k: A, st: S) !A {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0.0, 1.0, k, st));
            return o;
        }
    }.f;
    const q = try draw(&.{ 1, 2, 37, 16 }, key, s);
    defer free(q);
    const k = try draw(&.{ 1, 2, 37, 16 }, key, s);
    defer free(k);
    const v = try draw(&.{ 1, 2, 37, 16 }, key, s);
    defer free(v);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

    const walk = try segmentSdpaWalk(q, k, v, &geo, scale, s, a);
    defer free(walk);

    const ids = try hostI32(a, geo.image_ids);
    defer a.free(ids);
    const L: usize = @intCast(geo.joint_len);
    const mbuf = try a.alloc(bool, L * L);
    defer a.free(mbuf);
    for (0..L) |p| for (0..L) |j| {
        const same_block = ids[p] >= 0 and ids[p] == ids[j];
        mbuf[p * L + j] = p >= j or same_block;
    };
    const msh = [_]c_int{ 1, 1, @intCast(L), @intCast(L) };
    const marr = mlx.mlx_array_new_data(mbuf.ptr, &msh, 4, .bool_);
    defer free(marr);
    const dense = try sdpaMasked(q, k, v, scale, marr, s);
    defer free(dense);
    try expectParity("segment walk vs dense mask", walk, dense, s);
}

// The red-line case: two image blocks back-to-back in the mask (no text
// between) are still SEPARATE blocks — separate ids and segments, because
// the walk consumes blocks by img_shapes slot counts, never mask runs.
test "QwenImage edit segment walk: adjacent image blocks stay separate" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    // [text 3 | ref1 (1 slot, 2x2) | ref2 (2 slots, 2x4) | text 4 | target 4 slots, 4x4]
    const mask = [_]i32{ 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };
    const shapes = [_][3]u32{ .{ 1, 2, 2 }, .{ 1, 2, 4 }, .{ 1, 4, 4 } };
    const cfg = DitConfig{ .head_dim = 16, .axes = .{ 4, 6, 6 } };
    var geo = try EditGeometry.init(a, cfg, 10, &mask, &shapes);
    defer geo.deinit();
    // 3 + 4 + 8 + 4 + 16 = 35; target rows are [19, 35).
    try testing.expectEqual(@as(c_int, 35), geo.joint_len);
    try testing.expectEqual(@as(c_int, 19), geo.target_start);
    try testing.expectEqual(@as(usize, 4), geo.segments.len);
    // text [0,3) | ref1 [3,7) | ref2 [7,15) | text [15,19)
    try testing.expectEqual(EditSeg{ .start = 0, .end = 3, .is_text = true }, geo.segments[0]);
    try testing.expectEqual(EditSeg{ .start = 3, .end = 7, .is_text = false }, geo.segments[1]);
    try testing.expectEqual(EditSeg{ .start = 7, .end = 15, .is_text = false }, geo.segments[2]);
    try testing.expectEqual(EditSeg{ .start = 15, .end = 19, .is_text = true }, geo.segments[3]);
    const ids = try hostI32(a, geo.image_ids);
    defer a.free(ids);
    try testing.expectEqualSlices(i32, &([_]i32{ -1, -1, -1 }), ids[0..3]);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0, 0 }, ids[3..7]); // ref1
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 1, 1, 1, 1, 1, 1 }, ids[7..15]); // ref2, NOT ref1
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }, ids[19..35]);

    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, 23));
    const draw = struct {
        fn f(shape: []const c_int, k: A, st: S) !A {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0.0, 1.0, k, st));
            return o;
        }
    }.f;
    const q = try draw(&.{ 1, 2, 35, 16 }, key, s);
    defer free(q);
    const k = try draw(&.{ 1, 2, 35, 16 }, key, s);
    defer free(k);
    const v = try draw(&.{ 1, 2, 35, 16 }, key, s);
    defer free(v);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const walk = try segmentSdpaWalk(q, k, v, &geo, scale, s, a);
    defer free(walk);
    const L: usize = @intCast(geo.joint_len);
    const mbuf = try a.alloc(bool, L * L);
    defer a.free(mbuf);
    for (0..L) |p| for (0..L) |j| {
        const same_block = ids[p] >= 0 and ids[p] == ids[j];
        mbuf[p * L + j] = p >= j or same_block;
    };
    const msh = [_]c_int{ 1, 1, @intCast(L), @intCast(L) };
    const marr = mlx.mlx_array_new_data(mbuf.ptr, &msh, 4, .bool_);
    defer free(marr);
    const dense = try sdpaMasked(q, k, v, scale, marr, s);
    defer free(dense);
    try expectParity("adjacent-blocks walk vs dense mask", walk, dense, s);
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

// ── edit e2e on the tiny converted pack (QWEN_IMAGE_TEST_MODEL, the fixture
// oracles' own env): synthesized RGBA reference PNGs, smoke bar — shape,
// finiteness, same-seed determinism, and the CFG/negative divergence. ──

/// Deterministic RGBA test source: channel-varying bytes with an alpha that is
/// never 0 or 255, so the white-composite VLM copy and the alpha-carrying VAE
/// copy both see real values.
fn synthRgba(a: std.mem.Allocator, w: u32, h: u32, salt: u8) ![]u8 {
    const buf = try a.alloc(u8, @as(usize, w) * h * 4);
    for (buf, 0..) |*v, i| {
        const px: u32 = @intCast(i / 4);
        v.* = switch (i % 4) {
            3 => @truncate(120 + px % 136),
            else => salt +% @as(u8, @truncate(px >> 2)) +% @as(u8, @truncate(i % 4)) *% 37,
        };
    }
    return buf;
}

/// Two encoded references: 512x384 (landscape — the resize lands on a
/// non-square 74x56 grid) and 256x256. Caller frees each PNG.
fn editE2eRefs(a: std.mem.Allocator) ![2][]u8 {
    const png_mod = @import("png.zig");
    const r1 = try synthRgba(a, 512, 384, 0x5a);
    defer a.free(r1);
    const r2 = try synthRgba(a, 256, 256, 0xa7);
    defer a.free(r2);
    return .{
        try png_mod.encodeRgba(a, r1, 512, 384),
        try png_mod.encodeRgba(a, r2, 256, 256),
    };
}

/// Every element finite (NaN/Inf in the pipeline is the one hard failure a
/// random-weight smoke can catch).
fn expectFiniteImage(img: A, s: S) !void {
    var fin = mlx.mlx_array_new();
    defer free(fin);
    try mlx.check(mlx.mlx_isfinite(&fin, img, s));
    var allv = mlx.mlx_array_new();
    defer free(allv);
    try mlx.check(mlx.mlx_all(&allv, fin, false, s));
    var fv: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&fv, allv));
    try testing.expect(fv != 0.0);
}

/// Max |a − b| as a host f32 (0 == byte-identical images).
fn maxAbsDiff(x: A, y: A, s: S) !f32 {
    const d = try subA(x, y, s);
    defer free(d);
    var ad = mlx.mlx_array_new();
    defer free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var mx = mlx.mlx_array_new();
    defer free(mx);
    try mlx.check(mlx.mlx_max(&mx, ad, false, s));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, mx));
    return v;
}

test "QwenImage edit e2e (env-gated)" {
    const dir = std.mem.span(std.c.getenv("QWEN_IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var prev_cap: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev_cap, 1 << 30);
    defer _ = mlx.mlx_set_cache_limit(&prev_cap, prev_cap);
    const engine = try Engine.load(io, a, dir, true);
    defer engine.deinit();
    try testing.expect(engine.supportsEdit());

    const refs = try editE2eRefs(a);
    defer for (refs) |p| a.free(p);
    const prompt = "make the fox wear a tiny hat";

    const img1 = try engine.editImage(a, prompt, &refs, 256, 256, 42, 2, .{}, null);
    defer free(img1);
    try testing.expectEqualSlices(c_int, &.{ 1, 3, 256, 256 }, mlx.getShape(img1));
    try expectFiniteImage(img1, engine.s);
    try testing.expect(engine.te == null); // the edit's TE+tower freed before the denoise

    // Same seed ⇒ identical bytes.
    const img2 = try engine.editImage(a, prompt, &refs, 256, 256, 42, 2, .{}, null);
    defer free(img2);
    try testing.expectEqual(@as(f32, 0), try maxAbsDiff(img1, img2, engine.s));

    const flat = try reshape(img1, &[_]c_int{-1}, engine.s);
    defer free(flat);
    var variance = mlx.mlx_array_new();
    defer free(variance);
    try mlx.check(mlx.mlx_var_axis(&variance, flat, 0, false, 0, engine.s));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, variance));
    std.debug.print("[qwen-image] edit e2e 2 refs steps=2: pixel variance {d:.5}\n", .{v});
    try testing.expect(v > 1e-6);
}

test "QwenImage edit e2e CFG (env-gated)" {
    const dir = std.mem.span(std.c.getenv("QWEN_IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var prev_cap: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev_cap, 1 << 30);
    defer _ = mlx.mlx_set_cache_limit(&prev_cap, prev_cap);
    const engine = try Engine.load(io, a, dir, true);
    defer engine.deinit();

    const refs = try editE2eRefs(a);
    defer for (refs) |p| a.free(p);
    const prompt = "make the fox wear a tiny hat";

    const plain = try engine.editImage(a, prompt, &refs, 256, 256, 42, 2, .{}, null);
    defer free(plain);
    const guided = try engine.editImage(a, prompt, &refs, 256, 256, 42, 2, .{
        .guidance_scale = 2.5,
        .negative_prompt = "blurry",
    }, null);
    defer free(guided);
    try expectFiniteImage(guided, engine.s);

    // Real CFG: the second forward per step must move the output.
    const d = try maxAbsDiff(plain, guided, engine.s);
    std.debug.print("[qwen-image] edit e2e CFG 2.5: max |cfg - plain| {d:.5}\n", .{d});
    try testing.expect(d > 0.0);
}

test "QwenImage edit e2e towerless refuses (env-gated)" {
    const dir = std.mem.span(std.c.getenv("QWEN_IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const notower = try std.fmt.allocPrint(a, "{s}-notower", .{dir});
    defer a.free(notower);
    var d = std.Io.Dir.openDirAbsolute(io, notower, .{}) catch return error.SkipZigTest;
    d.close(io);

    const engine = try Engine.load(io, a, notower, true);
    defer engine.deinit();
    try testing.expect(!engine.supportsEdit());
    const refs = try editE2eRefs(a);
    defer for (refs) |p| a.free(p);
    try testing.expectError(error.QwenEditNotImplemented, engine.editImage(
        a, "make the fox wear a tiny hat", &refs, 256, 256, 42, 2, .{}, null,
    ));
}

// A pack that spells the DiT's time-embedder/modulation keys the
// mlx-community way (flattened) loads through the loader's probe.
test "QwenImage flattened DiT keys load (env-gated)" {
    const dir = std.mem.span(std.c.getenv("QWEN_IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const flat = try std.fmt.allocPrint(a, "{s}-flatdit", .{dir});
    defer a.free(flat);
    var d = std.Io.Dir.openDirAbsolute(io, flat, .{}) catch return error.SkipZigTest;
    d.close(io);

    const engine = try Engine.load(io, a, flat, true);
    defer engine.deinit();
    try testing.expect(engine.supportsEdit());
}
