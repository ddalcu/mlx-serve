//! MOSS-Image (`OpenMOSS-Team/MOSS-Image-{Base,SFT}`): a 5B single-stream DiT
//! over the Qwen3.5-9B text encoder and the FLUX.2 VAE with BFL's small
//! decoder. Port of the release's `pipeline_moss_image.py`, loading its
//! diffusers tree as published (the mlx-community bf16 mirror is byte-identical);
//! a quantized pack keeps the same names with MLX `.scales`/`.biases`.
//!
//! What is specific to this checkpoint:
//!  - the text encoder is the WHOLE Qwen3.5-9B trunk, final-normed, over the
//!    chat-templated prompt with thinking on; it runs on our qwen3_5 forward
//!    after mlx-lm's sanitize (norm `+1`, conv1d transpose), the vision tower dropped;
//!  - the joint sequence is [image | text], fully bidirectional, no mask. The text
//!    passes two refiner blocks first, which depend on nothing but the prompt,
//!    so they run once per request;
//!  - ONE adaLN modulation (shift/scale/gate x 2) shared by every block, no tanh;
//!    the FFN is SiTU-GLU: (4·tanh(g/4)·σ(g)) · (25·tanh(h/25));
//!  - 4-axis RoPE (r, f, h, w) on rotate-half pairs: image tokens sit at
//!    f = text length on a centred h/w grid, text tokens advance f;
//!  - CFG 3 whose negative branch SKIPS block 22 (the authors' default);
//!  - the sampler is Euler on a shift-5 schedule SNAPPED to the 1000-step
//!    training grid, so a step count can come back shorter than asked.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const sse = @import("../../sdk/media/progress.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const qwen3_5 = @import("../qwen3_5/plugin.zig");
const trunk_mod = @import("../../sdk/bases/trunk.zig");
const scope_mod = @import("../../mlx/scope.zig");
const ops = @import("../../mlx/ops.zig");
const lora_mod = @import("../../sdk/weights/lora.zig");
const dit_base = @import("../../sdk/bases/dit.zig");
const rope_blk = @import("../../sdk/blocks/dit/rope.zig");
const Linear = @import("../../sdk/blocks/dit/linear.zig").Linear;
const resnet_blk = @import("../../sdk/blocks/vae/resnet.zig");
const conv_blk = @import("../../sdk/blocks/vae/conv.zig");
const flow_match = @import("../../sdk/blocks/schedulers/flow_match.zig");

const Weights = @import("../../sdk/weights/resident.zig").Weights;
const Scope = scope_mod.Scope;
const Graph = ops.Graph;
const S = mlx.mlx_stream;
const A = mlx.mlx_array;

const config = @import("config.zig");

pub const DEFAULT_STEPS: u32 = 50;
pub const DEFAULT_GUIDANCE = config.DEFAULT_GUIDANCE;
/// Zero-based block the negative branch skips (README: 22, 24 and 26 all help).
const UNCOND_SKIP_LAYER: u32 = 22;
pub const MAX_PROMPT_TOKENS: usize = 4096;
pub const VAE_DOWNSAMPLE = config.VAE_DOWNSAMPLE;
const TRAIN_STEPS: usize = 1000;
const EPS: f32 = 1e-6;
/// Four RoPE axes of `head_dim / 8` bands each, built on the stack.
const MAX_HEAD_DIM: u32 = 512;
const COMPUTE: mlx.mlx_dtype = .bfloat16;

const PROMPT_PREFIX = "<|im_start|>user\n";
const PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n";

/// The release chat template over one user turn, thinking on: `render_content|trim` then the generation prompt.
pub fn formatPrompt(a: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, PROMPT_PREFIX ++ "{s}" ++ PROMPT_SUFFIX, .{std.mem.trim(u8, prompt, " \t\r\n\x0b\x0c")});
}

// ── config ──

fn readJson(io: std.Io, a: std.mem.Allocator, path: []const u8) !std.json.Parsed(std.json.Value) {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch return error.MossImageConfigMissing;
    defer a.free(bytes);
    return std.json.parseFromSlice(std.json.Value, a, bytes, .{});
}

/// A size from the config: absent is null, anything but an integer that fits a u32 is an error.
fn jsonU32(v: std.json.Value, key: []const u8) !?u32 {
    if (v != .object) return null;
    const x = v.object.get(key) orelse return null;
    if (x != .integer) return error.MossImageBadConfig;
    return std.math.cast(u32, x.integer) orelse error.MossImageBadConfig;
}

fn jsonNum(v: std.json.Value, key: []const u8) ?f64 {
    if (v != .object) return null;
    return switch (v.object.get(key) orelse return null) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

pub const DitConfig = struct {
    dim: u32 = 3072,
    ffn_dim: u32 = 8192,
    heads: u32 = 24,
    layers: u32 = 42,
    text_dim: u32 = 4096,
    latent_channels: u32 = 128,
    rope_base: f32 = 10000.0,
    shift: f32 = 5.0,
    training_shift: f32 = 1.0,

    pub fn headDim(self: DitConfig) u32 {
        return self.dim / self.heads;
    }

    pub fn parse(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !DitConfig {
        var c = DitConfig{};
        {
            const path = try std.fmt.allocPrint(a, "{s}/transformer/config.json", .{model_dir});
            defer a.free(path);
            var parsed = try readJson(io, a, path);
            defer parsed.deinit();
            const v = parsed.value;
            inline for (.{ "dim", "ffn_dim", "layers", "text_dim", "latent_channels", "heads" }, .{ "dim", "ffn_dim", "num_layers", "text_dim", "latent_channels", "num_heads" }) |field, key| {
                if (try jsonU32(v, key)) |n| @field(c, field) = n;
            }
            if (jsonNum(v, "rope_base")) |n| c.rope_base = @floatCast(n);
        }
        // The scheduler config is optional: the release defaults are the class defaults.
        const spath = try std.fmt.allocPrint(a, "{s}/scheduler/scheduler_config.json", .{model_dir});
        defer a.free(spath);
        if (readJson(io, a, spath)) |p| {
            var parsed = p;
            defer parsed.deinit();
            if (jsonNum(parsed.value, "shift")) |n| c.shift = @floatCast(n);
            if (jsonNum(parsed.value, "training_shift")) |n| c.training_shift = @floatCast(n);
        } else |_| {}
        if (c.heads == 0 or c.dim % c.heads != 0 or c.headDim() % 8 != 0 or c.headDim() > MAX_HEAD_DIM or c.latent_channels != 128 or !(c.shift > 0) or !(c.training_shift > 0))
            return error.MossImageBadConfig;
        return c;
    }
};

// ── schedule ──

/// torch.linspace in f32: the first half counts up from `start`, the second down from `end`.
fn linspaceTorch(out: []f32, start: f32, end: f32) void {
    if (out.len == 1) {
        out[0] = start;
        return;
    }
    const step: f32 = (end - start) / @as(f32, @floatFromInt(out.len - 1));
    const half = out.len / 2;
    for (out, 0..) |*o, i| o.* = if (i < half)
        start + step * @as(f32, @floatFromInt(i))
    else
        end - step * @as(f32, @floatFromInt(out.len - 1 - i));
}

/// `MOSSImageScheduler.set_timesteps.grid`: `count` shifted sigmas from 1 down.
fn shiftedGrid(out: []f32, shift: f32) void {
    const lower: f64 = @as(f64, shift) / (@as(f64, TRAIN_STEPS) + @as(f64, shift) - 1.0);
    linspaceTorch(out, 1.0, @floatCast(lower));
    for (out) |*v| v.* = flow_match.timeShift(f32, shift, v.*);
}

/// The release's sigmas for `steps`: the shifted grid snapped to the nearest
/// training sigma (ties to the lower), consecutive repeats dropped, then 0.
/// Caller owns the slice; its length minus one is the step count that runs.
pub fn scheduleSigmas(a: std.mem.Allocator, steps: u32, shift: f32, training_shift: f32) ![]f32 {
    var training: [TRAIN_STEPS]f32 = undefined;
    shiftedGrid(&training, training_shift);
    std.mem.reverse(f32, &training);
    const want = try a.alloc(f32, @max(steps, 1));
    defer a.free(want);
    shiftedGrid(want, shift);
    var out: std.ArrayList(f32) = .empty;
    errdefer out.deinit(a);
    for (want) |s| {
        var upper: usize = 0;
        while (upper < TRAIN_STEPS and training[upper] < s) upper += 1;
        upper = @min(upper, TRAIN_STEPS - 1);
        const lower = upper -| 1;
        const v = if (@abs(s - training[lower]) <= @abs(training[upper] - s)) training[lower] else training[upper];
        if (out.items.len == 0 or out.items[out.items.len - 1] != v) try out.append(a, v);
    }
    try out.append(a, 0.0);
    return out.toOwnedSlice(a);
}

// ── text encoder ──

/// mlx-lm's qwen3_5 sanitize over the release's `Qwen3_5Model` names: the
/// trunk moves under `language_model.model.`, the vision tower goes, and an
/// unsanitized checkpoint (conv1d stored [C, 1, K]) gets its depthwise convs
/// transposed and `+1` folded into the zero-centred norms.
fn sanitizeTextEncoder(a: std.mem.Allocator, src: *Weights, s: S) !Weights {
    var raw_conv = false;
    var it0 = src.map.iterator();
    while (it0.next()) |e| if (std.mem.endsWith(u8, e.key_ptr.*, "conv1d.weight") and mlx.getShape(e.value_ptr.*)[2] != 1) {
        raw_conv = true;
    };
    const norms = [_][]const u8{ ".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight", ".q_norm.weight", ".k_norm.weight" };
    var out = Weights.init(a);
    errdefer out.deinit();
    var it = src.map.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.startsWith(u8, k, "language_model.")) continue;
        const name = if (std.mem.startsWith(u8, k, "language_model.model."))
            try a.dupe(u8, k)
        else
            try std.fmt.allocPrint(a, "language_model.model.{s}", .{k["language_model.".len..]});
        errdefer a.free(name);
        var v = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(v);
        const sh = mlx.getShape(e.value_ptr.*);
        if (std.mem.endsWith(u8, k, "conv1d.weight") and sh[2] != 1) {
            try mlx.check(mlx.mlx_transpose_axes(&v, e.value_ptr.*, &[_]c_int{ 0, 2, 1 }, 3, s));
        } else if (raw_conv and sh.len == 1 and for (norms) |n| {
            if (std.mem.endsWith(u8, name, n)) break true;
        } else false) {
            const one = mlx.mlx_array_new_float(1.0);
            defer _ = mlx.mlx_array_free(one);
            var one_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(one_t);
            try mlx.check(mlx.mlx_astype(&one_t, one, mlx.mlx_array_dtype(e.value_ptr.*), s));
            try mlx.check(mlx.mlx_add(&v, e.value_ptr.*, one_t, s));
        } else {
            try mlx.check(mlx.mlx_array_set(&v, e.value_ptr.*));
        }
        try out.map.put(name, v);
    }
    return out;
}

const TextEncoder = struct {
    allocator: std.mem.Allocator,
    trunk: trunk_mod.Trunk,

    fn load(self: *TextEncoder, io: std.Io, a: std.mem.Allocator, model_dir: []const u8, s: S) !void {
        const dir = try std.fmt.allocPrint(a, "{s}/text_encoder", .{model_dir});
        defer a.free(dir);
        self.allocator = a;
        // The encoder ships no lm_head and only its final hidden state is read.
        const tied = "{\"tie_word_embeddings\":true,\"text_config\":{\"tie_word_embeddings\":true}}";
        self.trunk = trunk_mod.Trunk.parse(io, a, dir, &.{&qwen3_5.table}, tied) catch |e| return if (e == error.TrunkUnsupported) error.MossTextEncoderUnsupported else e;
        errdefer self.trunk.close(false);
        var raw = try weights_load.loadWeights(io, a, dir);
        defer raw.deinit();
        try self.trunk.load(io, a, dir, mlx.gpuStream(), try sanitizeTextEncoder(a, &raw, s));
    }

    fn deinit(self: *TextEncoder) void {
        self.trunk.close(true);
    }

    /// `last_hidden_state` [1, T, hidden] for one unpadded prompt (evaluated; caller frees).
    /// Every call runs on its own KV cache and GatedDeltaNet state.
    fn encode(self: *TextEncoder, ids: []const u32) !A {
        var cache = try trunk_mod.Cache.init(self.allocator, self.trunk.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        const shape = [_]c_int{ 1, @intCast(ids.len) };
        const arr = mlx.mlx_array_new_data(ids.ptr, &shape, 2, .uint32);
        defer _ = mlx.mlx_array_free(arr);
        const out = try self.trunk.forward(&cache, arr, .{});
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }
};

// ── DiT ──

const BlockLora = struct { q: lora_mod.Slot = .{}, k: lora_mod.Slot = .{}, v: lora_mod.Slot = .{}, o: lora_mod.Slot = .{}, up_gate: lora_mod.Slot = .{}, down: lora_mod.Slot = .{} };

const Block = struct {
    norm1: A,
    norm2: A,
    q_norm: A,
    k_norm: A,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up_gate: Linear,
    down: Linear,
    lora: BlockLora = .{},
};

const TextBlock = struct {
    norm1: A,
    norm2: A,
    q_norm: A,
    k_norm: A,
    qkv: Linear,
    proj: Linear,
    ffn_gate: Linear,
    ffn_up: Linear,
    ffn_down: Linear,
};

/// A checkpoint linear (dense or affine-quantized, optional bias) as the op layer's `Linear`.
fn linear(w: *const Weights, comptime fmt: []const u8, args: anytype, in_dim: u32) !Linear {
    const l = try scope_mod.linear(w, fmt, args, in_dim);
    return .{ .w = l.w, .scales = l.scales, .biases = l.biases, .bias = l.bias, .bits = l.bits, .group_size = l.group_size };
}

/// `lin(x)` plus every adapter attached to it.
fn linL(g: *Graph, l: *const Linear, slot: *const lora_mod.Slot, x: A) !A {
    var c = g.scope();
    defer c.deinit();
    const y = try l.forward(&c, x);
    if (slot.count == 0) return g.keep(y);
    return g.adopt(try lora_mod.addTo(y, x, slot.active(), g.s));
}

/// SiTU-GLU: (4·tanh(g/4)·σ(g)) · (25·tanh(h/25)), in the activation's dtype.
fn situ(g: *Graph, hidden: A, gate: A) !A {
    var c = g.scope();
    defer c.deinit();
    const gg = try c.mul(try c.mulScalar(try c.tanh(try c.mulScalar(gate, 0.25)), 4.0), try c.sigmoid(gate));
    const hh = try c.mulScalar(try c.tanh(try c.div(hidden, try c.scalarLike(hidden, 25.0))), 25.0);
    return g.keep(try c.mul(gg, hh));
}

/// [1, L, H·D] → per-head RMS norm and rotate-half RoPE (q, k) → [1, H, L, D]; cos/sin [1, L, 1, D/2].
fn heads(g: *Graph, y: A, n_heads: u32, norm: ?A, cos: A, sin: A) !A {
    var c = g.scope();
    defer c.deinit();
    const sh = ops.shape(y);
    const h: c_int = @intCast(n_heads);
    const x4 = try c.reshape(y, &.{ sh[0], sh[1], h, @divExact(sh[2], h) });
    const r = if (norm) |nw| try rope_blk.ropeSplitHalf(&c, try c.rmsNorm(x4, nw, EPS), cos, sin, @divExact(sh[2], 2 * h)) else x4;
    return g.keep(try c.transpose(r, &.{ 0, 2, 1, 3 }));
}

fn attend(g: *Graph, cfg: DitConfig, q: A, k: A, v: A) !A {
    var c = g.scope();
    defer c.deinit();
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.headDim())));
    return g.keep(try c.mergeHeads(try c.sdpa(q, k, v, scale, .none)));
}

/// One joint sequence's RoPE table (image rows then text rows), in the compute dtype.
pub const Rope = struct {
    cos: A,
    sin: A,
    n_img: c_int,

    pub fn init(a: std.mem.Allocator, cfg: DitConfig, lat_h: usize, lat_w: usize, text_len: usize, s: S) !Rope {
        const bands = cfg.headDim() / 8;
        const half = 4 * bands;
        const n = lat_h * lat_w + text_len;
        const cos_buf = try a.alloc(f32, n * half);
        defer a.free(cos_buf);
        const sin_buf = try a.alloc(f32, n * half);
        defer a.free(sin_buf);
        var band: [MAX_HEAD_DIM / 8]f32 = undefined;
        for (band[0..bands], 0..) |*b, j| {
            const e: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(bands));
            b.* = 1.0 / std.math.pow(f32, cfg.rope_base, e);
        }
        const ch: f32 = (@as(f32, @floatFromInt(lat_h)) - 1.0) / 2.0;
        const cw: f32 = (@as(f32, @floatFromInt(lat_w)) - 1.0) / 2.0;
        for (0..n) |row| {
            const coords: [4]f32 = if (row < lat_h * lat_w)
                .{ 0, @floatFromInt(text_len), @as(f32, @floatFromInt(row / lat_w)) - ch, @as(f32, @floatFromInt(row % lat_w)) - cw }
            else
                .{ 0, @floatFromInt(row - lat_h * lat_w), 0, 0 };
            for (0..4) |axis| for (0..bands) |j| {
                const phase = coords[axis] * band[j];
                cos_buf[row * half + axis * bands + j] = @cos(phase);
                sin_buf[row * half + axis * bands + j] = @sin(phase);
            };
        }
        const shape = [_]c_int{ 1, @intCast(n), 1, @intCast(half) };
        return .{
            .cos = try castNew(cos_buf, &shape, s),
            .sin = try castNew(sin_buf, &shape, s),
            .n_img = @intCast(lat_h * lat_w),
        };
    }

    pub fn deinit(self: *Rope) void {
        _ = mlx.mlx_array_free(self.cos);
        _ = mlx.mlx_array_free(self.sin);
    }
};

fn castNew(buf: []const f32, shape: []const c_int, s: S) !A {
    const raw = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(raw);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_astype(&o, raw, COMPUTE, s));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// The shared modulation of one step as [1, 1, dim] rows, `1 + scale` folded in:
/// shift_a, scale_a, gate_a, shift_f, scale_f, gate_f, then the head's shift, scale.
const Mod = struct {
    rows: [8]A,

    fn deinit(self: *Mod) void {
        for (self.rows) |x| _ = mlx.mlx_array_free(x);
    }
};

/// What MOSS-Image hands the sdk DiT skeleton: pre-norm adaLN blocks over one shared modulation (`folded`), its
/// SiTU-GLU and 4-axis rotate-half RoPE.
const Hooks = struct {
    pub const Layer = Block;
    pub const Model = Dit;
    pub const Input = struct { latents: A, text: A };
    pub const Ctx = struct { mod: *const Mod, rope: *const Rope, cfg: DitConfig };
    pub const mod: dit_base.Modulation = .folded;

    pub fn embed(g: *Graph, m: *const Dit, in: Input, _: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        return g.keep(try c.concat(&.{ try m.video_embedder.forward(&c, try c.astype(in.latents, COMPUTE)), in.text }, 1));
    }
    pub fn layers(m: *const Dit) usize {
        return m.blocks.len;
    }
    pub fn layer(m: *const Dit, i: usize) *const Block {
        return &m.blocks[i];
    }
    pub fn mods(ctx: Ctx, _: usize) dit_base.Mods {
        const r = ctx.mod.rows;
        return .{ .shift_a = r[0], .scale_a = r[1], .gate_a = r[2], .shift_f = r[3], .scale_f = r[4], .gate_f = r[5] };
    }
    pub fn norm1(g: *Graph, b: *const Block, x: A) !A {
        return g.rmsNorm(x, b.norm1, EPS);
    }
    pub fn norm2(g: *Graph, b: *const Block, x: A) !A {
        return g.rmsNorm(x, b.norm2, EPS);
    }
    pub fn attention(g: *Graph, b: *const Block, y: A, ctx: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        const cfg = ctx.cfg;
        const q = try heads(&c, try linL(&c, &b.q, &b.lora.q, y), cfg.heads, b.q_norm, ctx.rope.cos, ctx.rope.sin);
        const k = try heads(&c, try linL(&c, &b.k, &b.lora.k, y), cfg.heads, b.k_norm, ctx.rope.cos, ctx.rope.sin);
        const v = try heads(&c, try linL(&c, &b.v, &b.lora.v, y), cfg.heads, null, ctx.rope.cos, ctx.rope.sin);
        return g.keep(try linL(&c, &b.o, &b.lora.o, try attend(&c, cfg, q, k, v)));
    }
    pub fn ffn(g: *Graph, b: *const Block, y: A, ctx: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        const f: c_int = @intCast(ctx.cfg.ffn_dim);
        const hg = try linL(&c, &b.up_gate, &b.lora.up_gate, y);
        return g.keep(try linL(&c, &b.down, &b.lora.down, try situ(&c, try c.sliceAxis(hg, 2, 0, f), try c.sliceAxis(hg, 2, f, 2 * f))));
    }
    pub fn final(g: *Graph, m: *const Dit, x: A, ctx: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        const img = try c.sliceAxis(x, 1, 0, ctx.rope.n_img);
        const r = ctx.mod.rows;
        const y = try mod.apply(&c, try c.rmsNorm(img, m.final_norm, EPS), r[7], r[6]);
        return g.keep(try c.astype(try m.final_linear.forward(&c, y), .float32));
    }
};

pub const Dit = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: DitConfig,
    w: Weights,
    video_embedder: Linear,
    text_proj: Linear,
    text_blocks: [2]TextBlock,
    t1: Linear,
    t2: Linear,
    block_mod: Linear,
    head_mod: Linear,
    final_norm: A,
    final_linear: Linear,
    blocks: []Block,

    pub fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, cfg: DitConfig) !Dit {
        const dir = try std.fmt.allocPrint(a, "{s}/transformer", .{model_dir});
        defer a.free(dir);
        var self: Dit = undefined;
        self.allocator = a;
        self.s = s;
        self.cfg = cfg;
        self.w = try weights_load.loadWeights(io, a, dir);
        errdefer self.w.deinit();
        const w = &self.w;
        const d = cfg.dim;
        self.video_embedder = try linear(w, "video_embedder", .{}, cfg.latent_channels);
        self.text_proj = try linear(w, "text_embedder.proj", .{}, cfg.text_dim);
        for (&self.text_blocks, 0..) |*b, i| b.* = .{
            .norm1 = try scope_mod.getW(w, "text_embedder.blocks.{d}.norm1.weight", .{i}),
            .norm2 = try scope_mod.getW(w, "text_embedder.blocks.{d}.norm2.weight", .{i}),
            .q_norm = try scope_mod.getW(w, "text_embedder.blocks.{d}.q_norm.weight", .{i}),
            .k_norm = try scope_mod.getW(w, "text_embedder.blocks.{d}.k_norm.weight", .{i}),
            .qkv = try linear(w, "text_embedder.blocks.{d}.qkv", .{i}, d),
            .proj = try linear(w, "text_embedder.blocks.{d}.proj", .{i}, d),
            .ffn_gate = try linear(w, "text_embedder.blocks.{d}.ffn_gate", .{i}, d),
            .ffn_up = try linear(w, "text_embedder.blocks.{d}.ffn_up", .{i}, d),
            .ffn_down = try linear(w, "text_embedder.blocks.{d}.ffn_down", .{i}, cfg.ffn_dim),
        };
        self.t1 = try linear(w, "time_embedding.0", .{}, 256);
        self.t2 = try linear(w, "time_embedding.2", .{}, d);
        self.block_mod = try linear(w, "shared_block_modulation", .{}, d);
        self.head_mod = try linear(w, "video_head_modulation", .{}, d);
        self.final_norm = try scope_mod.getW(w, "final_norm_video.weight", .{});
        self.final_linear = try linear(w, "final_linear_video", .{}, d);
        if (countLayers(w) != cfg.layers) return error.MossImageBadConfig;
        self.blocks = try a.alloc(Block, cfg.layers);
        errdefer a.free(self.blocks);
        for (self.blocks, 0..) |*b, i| b.* = .{
            .norm1 = try scope_mod.getW(w, "layers.{d}.norm1.weight", .{i}),
            .norm2 = try scope_mod.getW(w, "layers.{d}.norm2.weight", .{i}),
            .q_norm = try scope_mod.getW(w, "layers.{d}.q_norm.weight", .{i}),
            .k_norm = try scope_mod.getW(w, "layers.{d}.k_norm.weight", .{i}),
            .q = try linear(w, "layers.{d}.proj.q", .{i}, d),
            .k = try linear(w, "layers.{d}.proj.k", .{i}, d),
            .v = try linear(w, "layers.{d}.proj.v", .{i}, d),
            .o = try linear(w, "layers.{d}.proj.o", .{i}, d),
            .up_gate = try linear(w, "layers.{d}.proj.ffn_up_gate", .{i}, d),
            .down = try linear(w, "layers.{d}.proj.ffn_down", .{i}, cfg.ffn_dim),
        };
        return self;
    }

    pub fn deinit(self: *Dit) void {
        self.allocator.free(self.blocks);
        self.w.deinit();
    }

    /// Attach every adapter in `stack` to its block linear (module names are the
    /// checkpoint's own, `layers.N.proj.q`). Returns the attachment count.
    pub fn attachLora(self: *Dit, stack: *const lora_mod.Stack) u32 {
        self.detachLora();
        var matched: u32 = 0;
        var kbuf: [64]u8 = undefined;
        var rbuf: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined;
        for (self.blocks, 0..) |*b, i| {
            inline for (.{ "q", "k", "v", "o", "ffn_up_gate", "ffn_down" }, .{ "q", "k", "v", "o", "up_gate", "down" }) |m, f| {
                const key = std.fmt.bufPrint(&kbuf, "layers.{d}.proj." ++ m, .{i}) catch unreachable;
                const refs = stack.findAll(key, &rbuf);
                if (refs.len > 0) {
                    @field(b.lora, f).set(refs);
                    matched += @intCast(refs.len);
                }
            }
        }
        return matched;
    }

    pub fn detachLora(self: *Dit) void {
        for (self.blocks) |*b| b.lora = .{};
    }

    /// The prompt's two refiner blocks, once per request: hidden [1, T, text_dim] → [1, T, dim].
    pub fn embedText(self: *const Dit, hidden: A, rope: *const Rope) !A {
        var g = Graph.init(self.allocator, self.s);
        defer g.deinit();
        const n_all = ops.shape(rope.cos)[1];
        const cos = try g.sliceAxis(rope.cos, 1, rope.n_img, n_all);
        const sin = try g.sliceAxis(rope.sin, 1, rope.n_img, n_all);
        var x = try self.text_proj.forward(&g, try g.astype(hidden, COMPUTE));
        const d: c_int = @intCast(self.cfg.dim);
        for (&self.text_blocks) |*b| {
            const qkv = try b.qkv.forward(&g, try g.rmsNorm(x, b.norm1, EPS));
            const q = try heads(&g, try g.sliceAxis(qkv, 2, 0, d), self.cfg.heads, b.q_norm, cos, sin);
            const k = try heads(&g, try g.sliceAxis(qkv, 2, d, 2 * d), self.cfg.heads, b.k_norm, cos, sin);
            const v = try heads(&g, try g.sliceAxis(qkv, 2, 2 * d, 3 * d), self.cfg.heads, null, cos, sin);
            x = try g.add(x, try b.proj.forward(&g, try attend(&g, self.cfg, q, k, v)));
            const y = try g.rmsNorm(x, b.norm2, EPS);
            x = try g.add(x, try b.ffn_down.forward(&g, try situ(&g, try b.ffn_up.forward(&g, y), try b.ffn_gate.forward(&g, y))));
        }
        const out = g.escape(x);
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    /// One step's modulation from the sampler timestep `t` (sigma x 1000).
    fn modulation(self: *const Dit, t: f32) !Mod {
        var g = Graph.init(self.allocator, self.s);
        defer g.deinit();
        // The reference computes the sinusoid in f64 and feeds f32 to the MLP.
        var buf: [256]f32 = undefined;
        for (0..128) |k| {
            const freq = std.math.pow(f64, 10000.0, -@as(f64, @floatFromInt(k)) / 128.0);
            const phase = @as(f64, t) * freq;
            buf[k] = @floatCast(@cos(phase));
            buf[128 + k] = @floatCast(@sin(phase));
        }
        const emb = try g.astype(try g.fromSlice(f32, &buf, &.{ 1, 256 }), COMPUTE);
        const time = try self.t2.forward(&g, try g.silu(try self.t1.forward(&g, emb)));
        const d: c_int = @intCast(self.cfg.dim);
        const bm = try self.block_mod.forward(&g, time);
        const hm = try self.head_mod.forward(&g, time);
        var rows: [8]A = undefined;
        for (&rows, 0..) |*r, i| {
            const j: c_int = @intCast(if (i < 6) i else i - 6);
            r.* = try g.reshape(try g.sliceAxis(if (i < 6) bm else hm, 1, j * d, (j + 1) * d), &.{ 1, 1, d });
            if (i == 1 or i == 4 or i == 7) r.* = try g.add(r.*, try g.scalarLike(r.*, 1.0));
        }
        try ops.eval(try g.concat(&rows, 0));
        var m: Mod = undefined;
        for (&m.rows, rows) |*o, r| o.* = g.escape(r);
        return m;
    }

    /// Velocity [1, HW, 128] f32 for latents [1, HW, 128] against a refined prompt
    /// `text` [1, T, dim], optionally skipping one block (the negative branch).
    pub fn forward(self: *const Dit, latents: A, text: A, m: *const Mod, rope: *const Rope, skip: ?u32) !A {
        var g = Graph.init(self.allocator, self.s);
        defer g.deinit();
        const ctx: Hooks.Ctx = .{ .mod = m, .rope = rope, .cfg = self.cfg };
        return g.escape(try dit_base.Dit(Hooks).forward(&g, self, .{ .latents = latents, .text = text }, ctx, if (skip) |k| k else null));
    }
};

fn countLayers(w: *const Weights) usize {
    var n: usize = 0;
    var buf: [48]u8 = undefined;
    while (w.get(std.fmt.bufPrint(&buf, "layers.{d}.norm1.weight", .{n}) catch return n) != null) n += 1;
    return n;
}

// ── VAE decoder (FLUX.2 small decoder, diffusers layout, f32) ──

const Res = resnet_blk.Resnet;
const vae_norm: resnet_blk.Norm = .{ .axes = .{ .eps = 1e-6 } };

const Vae = struct {
    allocator: std.mem.Allocator,
    s: S,
    w: Weights,
    /// OHWI f32 copies of the OIHW convs and the f32 vectors, owned.
    owned: std.ArrayList(A) = .empty,
    bn_mean: A,
    bn_std: A,
    pq_w: A,
    pq_b: A,
    conv_in_w: A,
    conv_in_b: A,
    mid: [2]Res,
    attn_gn: [2]A,
    attn: struct { q: Linear, k: Linear, v: Linear, o: Linear },
    up: [8][4]Res,
    up_res: [8]usize,
    up_conv: [8]?[2]A,
    n_up: usize,
    norm_out: [2]A,
    conv_out_w: A,
    conv_out_b: A,

    fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8) !Vae {
        const dir = try std.fmt.allocPrint(a, "{s}/vae", .{model_dir});
        defer a.free(dir);
        var self: Vae = undefined;
        self.allocator = a;
        self.s = s;
        self.owned = .empty;
        self.w = try weights_load.loadWeights(io, a, dir);
        errdefer self.deinit();
        const eps_bn: f32 = blk: {
            const path = try std.fmt.allocPrint(a, "{s}/config.json", .{dir});
            defer a.free(path);
            var p = readJson(io, a, path) catch break :blk 1e-4;
            defer p.deinit();
            break :blk if (jsonNum(p.value, "batch_norm_eps")) |e| @floatCast(e) else 1e-4;
        };
        self.bn_mean = try self.vec("bn.running_mean", .{});
        {
            var sc = Scope.init(a, s);
            defer sc.deinit();
            var o = mlx.mlx_array_new();
            const sq = try sc.res(mlx.mlx_sqrt(&o, try sc.addS(try self.vec("bn.running_var", .{}), eps_bn), s), &o);
            self.bn_std = try self.own(sc.out(sq));
        }
        self.pq_w = try self.conv("post_quant_conv.weight", .{});
        self.pq_b = try self.vec("post_quant_conv.bias", .{});
        self.conv_in_w = try self.conv("decoder.conv_in.weight", .{});
        self.conv_in_b = try self.vec("decoder.conv_in.bias", .{});
        for (&self.mid, 0..) |*r, i| r.* = try self.resnet("decoder.mid_block.resnets.{d}", .{i});
        self.attn_gn = .{ try self.vec("decoder.mid_block.attentions.0.group_norm.weight", .{}), try self.vec("decoder.mid_block.attentions.0.group_norm.bias", .{}) };
        inline for (.{ "to_q", "to_k", "to_v" }, .{ "q", "k", "v" }) |n, f| @field(self.attn, f) = try self.dense("decoder.mid_block.attentions.0." ++ n);
        self.attn.o = if (self.w.get("decoder.mid_block.attentions.0.to_out.0.weight") != null)
            try self.dense("decoder.mid_block.attentions.0.to_out.0")
        else
            try self.dense("decoder.mid_block.attentions.0.to_out");
        self.n_up = 0;
        var kbuf: [96]u8 = undefined;
        while (self.n_up < self.up.len and self.w.get(try std.fmt.bufPrint(&kbuf, "decoder.up_blocks.{d}.resnets.0.conv1.weight", .{self.n_up})) != null) : (self.n_up += 1) {
            const bi = self.n_up;
            self.up_res[bi] = 0;
            while (self.up_res[bi] < self.up[bi].len and self.w.get(try std.fmt.bufPrint(&kbuf, "decoder.up_blocks.{d}.resnets.{d}.conv1.weight", .{ bi, self.up_res[bi] })) != null) : (self.up_res[bi] += 1)
                self.up[bi][self.up_res[bi]] = try self.resnet("decoder.up_blocks.{d}.resnets.{d}", .{ bi, self.up_res[bi] });
            self.up_conv[bi] = if (self.w.get(try std.fmt.bufPrint(&kbuf, "decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi})) != null)
                .{ try self.conv("decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi}), try self.vec("decoder.up_blocks.{d}.upsamplers.0.conv.bias", .{bi}) }
            else
                null;
        }
        if (self.n_up == 0) return error.MissingWeight;
        self.norm_out = .{ try self.vec("decoder.conv_norm_out.weight", .{}), try self.vec("decoder.conv_norm_out.bias", .{}) };
        self.conv_out_w = try self.conv("decoder.conv_out.weight", .{});
        self.conv_out_b = try self.vec("decoder.conv_out.bias", .{});
        // The encoder never runs: drop its arrays so they are not resident.
        self.w.dropPrefix("encoder.");
        self.w.dropPrefix("quant_conv.");
        return self;
    }

    fn deinit(self: *Vae) void {
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.allocator);
        self.w.deinit();
    }

    fn own(self: *Vae, x: A) !A {
        errdefer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_array_eval(x));
        try self.owned.append(self.allocator, x);
        return x;
    }

    fn vec(self: *Vae, comptime fmt: []const u8, args: anytype) !A {
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_astype(&o, try scope_mod.getW(&self.w, fmt, args), .float32, self.s));
        return self.own(o);
    }

    /// OIHW (as published) → OHWI f32, materialized: MLX convs read channels-last weights.
    fn conv(self: *Vae, comptime fmt: []const u8, args: anytype) !A {
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const raw = try scope_mod.getW(&self.w, fmt, args);
        var o = mlx.mlx_array_new();
        const c = try sc.res(mlx.mlx_contiguous(&o, try sc.astype(try sc.transpose(raw, &.{ 0, 2, 3, 1 }), .float32), false, self.s), &o);
        return self.own(sc.out(c));
    }

    fn dense(self: *Vae, comptime prefix: []const u8) !Linear {
        return .{ .w = try self.vec(prefix ++ ".weight", .{}), .bias = try self.vec(prefix ++ ".bias", .{}) };
    }

    fn resnet(self: *Vae, comptime fmt: []const u8, args: anytype) !Res {
        var kbuf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&kbuf, fmt, args);
        var nbuf: [128]u8 = undefined;
        const has_sc = self.w.get(try std.fmt.bufPrint(&nbuf, "{s}.conv_shortcut.weight", .{p})) != null;
        return .{
            .n1w = try self.vec("{s}.norm1.weight", .{p}),
            .n1b = try self.vec("{s}.norm1.bias", .{p}),
            .c1w = try self.conv("{s}.conv1.weight", .{p}),
            .c1b = try self.vec("{s}.conv1.bias", .{p}),
            .n2w = try self.vec("{s}.norm2.weight", .{p}),
            .n2b = try self.vec("{s}.norm2.bias", .{p}),
            .c2w = try self.conv("{s}.conv2.weight", .{p}),
            .c2b = try self.vec("{s}.conv2.bias", .{p}),
            .sw = if (has_sc) try self.conv("{s}.conv_shortcut.weight", .{p}) else null,
            .sb = if (has_sc) try self.vec("{s}.conv_shortcut.bias", .{p}) else null,
        };
    }

    /// Packed latents [1, h, w, 128] (token layout, channel = c·4 + ph·2 + pw) →
    /// image [1, 3, 16h, 16w] f32 in [0, 1].
    fn decode(self: *const Vae, packed_lat: A) !A {
        const a = self.allocator;
        const s = self.s;
        var x = blk: {
            var g = Graph.init(a, s);
            defer g.deinit();
            const sh = ops.shape(packed_lat);
            const z = try g.add(try g.mul(try g.astype(packed_lat, .float32), self.bn_std), self.bn_mean);
            const u = try g.reshape(try g.transpose(try g.reshape(z, &.{ 1, sh[1], sh[2], 32, 2, 2 }), &.{ 0, 1, 4, 2, 5, 3 }), &.{ 1, sh[1] * 2, sh[2] * 2, 32 });
            const h = try conv_blk.conv2d(&g, try conv_blk.conv2d(&g, u, self.pq_w, self.pq_b, 1, 0, 1), self.conv_in_w, self.conv_in_b, 1, 1, 1);
            const m0 = try self.mid[0].forward(&g, vae_norm, h);
            const m1 = try resnet_blk.midAttention(&g, vae_norm, m0, self.attn_gn[0], self.attn_gn[1], self.attn);
            break :blk g.escape(try self.mid[1].forward(&g, vae_norm, m1));
        };
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_array_eval(x));
        for (0..self.n_up) |bi| {
            // The graph ends before the eval: a live handle would pin every full-resolution intermediate.
            const next = blk: {
                var g = Graph.init(a, s);
                defer g.deinit();
                var h = x;
                for (self.up[bi][0..self.up_res[bi]]) |*r| h = try r.forward(&g, vae_norm, h);
                if (self.up_conv[bi]) |uc| h = try conv_blk.upsample2x(&g, h, uc[0], uc[1]);
                break :blk g.escape(h);
            };
            _ = mlx.mlx_array_free(x);
            x = next;
            try mlx.check(mlx.mlx_array_eval(x));
        }
        var g = Graph.init(a, s);
        defer g.deinit();
        const n = try vae_norm.apply(&g, x, self.norm_out[0], self.norm_out[1]);
        const img = try conv_blk.conv2d(&g, try g.silu(n), self.conv_out_w, self.conv_out_b, 1, 1, 1);
        const unit = try g.mul(try g.add(img, try g.scalar(1.0, .float32)), try g.scalar(0.5, .float32));
        const clipped = try g.call(mlx.mlx_clip, .{ unit, try g.scalar(0, .float32), try g.scalar(1, .float32) });
        const out = g.escape(try g.contiguous(try g.transpose(clipped, &.{ 0, 3, 1, 2 })));
        return out;
    }
};

// ── engine ──

pub const GenOpts = struct {
    guidance_scale: f32 = DEFAULT_GUIDANCE,
    negative_prompt: []const u8 = "",
};

fn stageMs(io: std.Io, t0: *std.Io.Timestamp) u64 {
    const ns = t0.untilNow(io, .awake).nanoseconds;
    t0.* = std.Io.Timestamp.now(io, .awake);
    return @intCast(@divTrunc(ns, 1_000_000));
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,
    cfg: DitConfig,
    tok: tok_mod.Tokenizer,
    te: TextEncoder,
    dit: Dit,
    vae: Vae,
    /// The block the negative branch skips; a pack shallower than the default has none.
    uncond_skip: ?u32,

    /// Every array of the engine lives and runs on `s`.
    pub fn loadOn(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.io = io;
        self.s = s;
        self.cfg = try DitConfig.parse(io, allocator, model_dir);
        self.uncond_skip = if (self.cfg.layers > UNCOND_SKIP_LAYER) UNCOND_SKIP_LAYER else null;
        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        errdefer self.tok.deinit();
        self.dit = try Dit.load(io, allocator, self.s, model_dir, self.cfg);
        errdefer self.dit.deinit();
        self.vae = try Vae.load(io, allocator, self.s, model_dir);
        errdefer self.vae.deinit();
        try self.te.load(io, allocator, model_dir, self.s);
        errdefer self.te.deinit();
        // Resident now, so a request's memory check reads them in live memory.
        try scope_mod.evalAll(&self.dit.w);
        log.info("[image] MOSS-Image ready (DiT {d}x{d}, {d} heads, shift {d:.1}, text encoder {d} layers)\n", .{
            self.cfg.layers, self.cfg.dim, self.cfg.heads, self.cfg.shift, self.te.trunk.parsed.shell.num_hidden_layers,
        });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.te.deinit();
        self.vae.deinit();
        self.dit.deinit();
        self.tok.deinit();
        self.allocator.destroy(self);
    }

    /// The step count that runs for a request: the snapped schedule can be shorter.
    pub fn stepsFor(self: *const Engine, steps: u32) u32 {
        const sig = scheduleSigmas(self.allocator, if (steps == 0) DEFAULT_STEPS else steps, self.cfg.shift, self.cfg.training_shift) catch return if (steps == 0) DEFAULT_STEPS else steps;
        defer self.allocator.free(sig);
        return @intCast(sig.len - 1);
    }

    /// Prompt → the encoder's final hidden state [1, T, text_dim], at most 4096 tokens.
    fn encodePrompt(self: *Engine, prompt: []const u8) !A {
        const a = self.allocator;
        const text = try formatPrompt(a, prompt);
        defer a.free(text);
        const enc = try self.tok.encode(a, text);
        defer a.free(enc);
        return self.te.encode(enc[0..@min(enc.len, MAX_PROMPT_TOKENS)]);
    }

    const Branch = struct {
        text: A,
        rope: Rope,

        fn deinit(self: *Branch) void {
            _ = mlx.mlx_array_free(self.text);
            self.rope.deinit();
        }
    };

    fn branch(self: *Engine, hidden: A, lat_h: usize, lat_w: usize) !Branch {
        var rope = try Rope.init(self.allocator, self.cfg, lat_h, lat_w, @intCast(mlx.getShape(hidden)[1]), self.s);
        errdefer rope.deinit();
        return .{ .text = try self.dit.embedText(hidden, &rope), .rope = rope };
    }

    /// Returns the image [1,3,H,W] f32 in [0,1] (owned; caller frees).
    pub fn generateImage(self: *Engine, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: GenOpts, progress: ?sse.Progress) !A {
        const lat_n: c_int = @intCast((width / VAE_DOWNSAMPLE) * (height / VAE_DOWNSAMPLE));
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        var noise = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(noise);
        const shape = [_]c_int{ 1, lat_n, @intCast(self.cfg.latent_channels) };
        try mlx.check(mlx.mlx_random_normal(&noise, &shape, 3, .float32, 0.0, 1.0, key, self.s));
        return self.generateFrom(prompt, width, height, noise, steps, opts, progress);
    }

    /// The pipeline from given f32 latents [1, HW, 128] (borrowed).
    pub fn generateFrom(self: *Engine, prompt: []const u8, width: u32, height: u32, init: A, steps: u32, opts: GenOpts, progress: ?sse.Progress) !A {
        const s = self.s;
        const lat_h: usize = height / VAE_DOWNSAMPLE;
        const lat_w: usize = width / VAE_DOWNSAMPLE;
        const sigmas = try scheduleSigmas(self.allocator, if (steps == 0) DEFAULT_STEPS else steps, self.cfg.shift, self.cfg.training_shift);
        defer self.allocator.free(sigmas);
        const n_steps: u32 = @intCast(sigmas.len - 1);
        const cfg_on = opts.guidance_scale != 1.0;
        const skip = self.uncond_skip;
        log.info("[moss-image] {d}x{d} steps={d} guidance={d:.1} ({s})\n", .{
            width, height, n_steps, opts.guidance_scale, if (cfg_on) "two forwards per step, negative branch skips a block" else "one forward per step",
        });

        if (progress) |p| p.emit("Encoding prompt", 0, n_steps);
        var t0 = std.Io.Timestamp.now(self.io, .awake);
        var pos = blk: {
            const hidden = try self.encodePrompt(prompt);
            defer _ = mlx.mlx_array_free(hidden);
            break :blk try self.branch(hidden, lat_h, lat_w);
        };
        defer pos.deinit();
        var neg: ?Branch = if (cfg_on) blk: {
            const hidden = try self.encodePrompt(opts.negative_prompt);
            defer _ = mlx.mlx_array_free(hidden);
            break :blk try self.branch(hidden, lat_h, lat_w);
        } else null;
        defer if (neg) |*n| n.deinit();
        const prompt_ms = stageMs(self.io, &t0);

        var lat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lat);
        try mlx.check(mlx.mlx_array_set(&lat, init));
        for (0..n_steps) |i| {
            if (progress) |p| if (p.boundary()) return error.Cancelled;
            var m = try self.dit.modulation(sigmas[i] * @as(f32, TRAIN_STEPS));
            defer m.deinit();
            var sc = Scope.init(self.allocator, s);
            defer sc.deinit();
            var v = try sc.keep(try self.dit.forward(lat, pos.text, &m, &pos.rope, null));
            if (neg) |*n| {
                // CFG combines f32 predictions: uncond + scale · (cond − uncond).
                const vn = try sc.keep(try self.dit.forward(lat, n.text, &m, &n.rope, skip));
                v = try sc.add(vn, try sc.mulS(try sc.sub(v, vn), opts.guidance_scale));
            }
            var g = Graph.init(self.allocator, s);
            defer g.deinit();
            const next = g.escape(try flow_match.eulerStep(&g, lat, v, sigmas[i + 1] - sigmas[i], .float32));
            _ = mlx.mlx_array_free(lat);
            lat = next;
            try mlx.check(mlx.mlx_array_eval(lat));
            if (progress) |p| p.emit("Generating", @intCast(i + 1), n_steps);
        }
        const denoise_ms = stageMs(self.io, &t0);

        if (progress) |p| p.emit("Decoding image", n_steps, n_steps);
        var grid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(grid);
        try mlx.check(mlx.mlx_reshape(&grid, lat, &[_]c_int{ 1, @intCast(lat_h), @intCast(lat_w), @intCast(self.cfg.latent_channels) }, 4, s));
        const img = try self.vae.decode(grid);
        errdefer _ = mlx.mlx_array_free(img);
        try mlx.check(mlx.mlx_array_eval(img));
        log.info("[moss-image] stages: prompt {d} ms, denoise {d} ms ({d} steps), vae decode {d} ms\n", .{ prompt_ms, denoise_ms, n_steps, stageMs(self.io, &t0) });
        return img;
    }
};

// ── tests ──

const testing = std.testing;

test "MOSS-Image: a transformer config outside what the engine can hold is a named error" {
    const io = std.testing.io;
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "transformer");
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    for ([_][]const u8{
        "{\"dim\": 12288, \"num_heads\": 16}",
        "{\"dim\": -3072}",
        "{\"num_layers\": 1e12}",
        "{\"num_heads\": 4294967296}",
    }) |cfg| {
        try tmp.dir.writeFile(io, .{ .sub_path = "transformer/config.json", .data = cfg });
        try testing.expectError(error.MossImageBadConfig, DitConfig.parse(io, a, root));
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "transformer/config.json", .data = "{\"dim\": 3072, \"num_heads\": 24, \"num_layers\": 42}" });
    try testing.expectEqual(@as(u32, 128), (try DitConfig.parse(io, a, root)).headDim());
}

test "MOSS-Image: the schedule is the release scheduler's, snapped to the training grid" {
    const want8 = [_]f32{ 1.0, 0.967999995, 0.925999999, 0.870999992, 0.791000009, 0.671000004, 0.463000029, 0.0240000002, 0.0 };
    // torch's vectorized linspace rounds the grid's last ulp by vector width: 1e-6 is the bar.
    const got8 = try scheduleSigmas(testing.allocator, 8, 5.0, 1.0);
    defer testing.allocator.free(got8);
    try testing.expectEqual(want8.len, got8.len);
    for (want8, got8) |w, g| try testing.expectApproxEqAbs(w, g, 1e-6);
    const want4 = [_]f32{ 1.0, 0.909999967, 0.717000008, 0.0240000002, 0.0 };
    const got4 = try scheduleSigmas(testing.allocator, 4, 5.0, 1.0);
    defer testing.allocator.free(got4);
    try testing.expectEqual(want4.len, got4.len);
    for (want4, got4) |w, g| try testing.expectApproxEqAbs(w, g, 1e-6);
    const got50 = try scheduleSigmas(testing.allocator, 50, 5.0, 1.0);
    defer testing.allocator.free(got50);
    try testing.expectEqual(@as(usize, 51), got50.len);
    try testing.expectApproxEqAbs(@as(f32, 0.995999992), got50[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.115000002), got50[48], 1e-6);
    // Past the grid's resolution near sigma 1 the snap repeats; repeats run once.
    const dense = try scheduleSigmas(testing.allocator, 1000, 5.0, 1.0);
    defer testing.allocator.free(dense);
    try testing.expect(dense.len - 1 < 1000);
    for (dense[1..], dense[0 .. dense.len - 1]) |b, prev| try testing.expect(b < prev);
}

test "MOSS-Image: the prompt is the release template over one trimmed user turn, thinking on" {
    const p = try formatPrompt(testing.allocator, "  a cat\n");
    defer testing.allocator.free(p);
    try testing.expectEqualStrings("<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n", p);
}

// The parity oracle: a tiny random pack built from the release's own classes
// (`tests/dump_moss_image_fixtures.py`). Env-gated: MOSS_IMAGE_TEST_MODEL + MOSS_IMAGE_FIXTURE.
const Fixture = struct {
    w: Weights,
    model: []const u8,

    fn open(a: std.mem.Allocator) !Fixture {
        const model = std.mem.span(std.c.getenv("MOSS_IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
        const path = std.mem.span(std.c.getenv("MOSS_IMAGE_FIXTURE") orelse return error.SkipZigTest);
        return .{ .w = try weights_load.loadWeightsSingleFile(a, path), .model = model };
    }
};

fn parity(got: A, want: A, s: S) !struct { cos: f32, rms_ratio: f32 } {
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const g = try sc.reshape(try sc.astype(got, .float32), &.{-1});
    const w = try sc.reshape(try sc.astype(want, .float32), &.{-1});
    const dot = try sc.sumAxis(try sc.mul(g, w), 0);
    const gg = try sc.sumAxis(try sc.mul(g, g), 0);
    const ww = try sc.sumAxis(try sc.mul(w, w), 0);
    const all = try sc.concat(&.{ dot, gg, ww }, 0);
    try mlx.check(mlx.mlx_array_eval(all));
    const d = mlx.mlx_array_data_float32(all).?;
    return .{ .cos = d[0] / @sqrt(d[1] * d[2]), .rms_ratio = @sqrt(d[1] / d[2]) };
}

fn expectParity(name: []const u8, got: A, want: A, s: S, min_cos: f32) !void {
    const p = try parity(got, want, s);
    log.info("[moss-image test] {s}: cos {d:.6} rms_ratio {d:.4}\n", .{ name, p.cos, p.rms_ratio });
    try testing.expect(p.cos >= min_cos);
    try testing.expect(@abs(p.rms_ratio - 1.0) < 0.03);
}

/// NCHW [1, C, h, w] → tokens [1, h·w, C].
fn toTokens(sc: *Scope, x: A) !A {
    const sh = mlx.getShape(x);
    return sc.reshape(try sc.transpose(x, &.{ 0, 2, 3, 1 }), &.{ 1, sh[2] * sh[3], sh[1] });
}

test "MOSS-Image: text encoder, DiT, VAE and the whole pipeline match the reference on a tiny pack" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try Fixture.open(a);
    defer fx.w.deinit();
    const eng = try Engine.loadOn(io, a, fx.model, mlx.mlx_default_gpu_stream_new());
    defer eng.deinit();
    const s = eng.s;
    var sc = Scope.init(a, s);
    defer sc.deinit();
    const get = struct {
        fn f(w: *const Weights, k: []const u8) !A {
            return w.get(k) orelse error.MissingFixture;
        }
    }.f;

    // Prompt template + tokenizer: the reference's ids.
    const prompt = "A poster titled \"MOSS\" with a cat.";
    {
        const text = try formatPrompt(a, prompt);
        defer a.free(text);
        const ids = try eng.tok.encode(a, text);
        defer a.free(ids);
        const want = try get(&fx.w, "te_ids");
        try mlx.check(mlx.mlx_array_eval(want));
        const wd = mlx.mlx_array_data_int32(want).?;
        try testing.expectEqual(@as(usize, @intCast(mlx.getShape(want)[0])), ids.len);
        for (ids, 0..) |t, i| try testing.expectEqual(@as(i32, @intCast(t)), wd[i]);
    }
    const hidden = try sc.keep(try eng.encodePrompt(prompt));
    try expectParity("text encoder", hidden, try get(&fx.w, "te_hidden"), s, 0.999);
    const hidden_neg = try sc.keep(try eng.encodePrompt("worst quality"));
    try expectParity("text encoder (negative)", hidden_neg, try get(&fx.w, "te_hidden_neg"), s, 0.999);

    // One DiT forward on the reference's own encoder output, then with block 1 skipped.
    const lat_nchw = try get(&fx.w, "dit_latents");
    const sh = mlx.getShape(lat_nchw);
    const lh: usize = @intCast(sh[2]);
    const lw: usize = @intCast(sh[3]);
    const lat = try toTokens(&sc, lat_nchw);
    var m = try eng.dit.modulation(700.0);
    defer m.deinit();
    {
        var b = try eng.branch(try get(&fx.w, "te_hidden"), lh, lw);
        defer b.deinit();
        const v = try sc.keep(try eng.dit.forward(lat, b.text, &m, &b.rope, null));
        try expectParity("dit", v, try toTokens(&sc, try get(&fx.w, "dit_out")), s, 0.999);
    }
    {
        var b = try eng.branch(try get(&fx.w, "te_hidden_neg"), lh, lw);
        defer b.deinit();
        const v = try sc.keep(try eng.dit.forward(lat, b.text, &m, &b.rope, 1));
        try expectParity("dit skip", v, try toTokens(&sc, try get(&fx.w, "dit_out_skip")), s, 0.999);
    }

    // VAE: bn denorm + unpatchify + decoder, against the reference's own decode ([-1, 1] → [0, 1]).
    {
        const z = try sc.transpose(try get(&fx.w, "vae_in"), &.{ 0, 2, 3, 1 });
        const img = try sc.keep(try eng.vae.decode(z));
        var o = mlx.mlx_array_new();
        const want = try sc.res(mlx.mlx_clip(&o, try sc.mulS(try sc.addS(try get(&fx.w, "vae_out"), 1.0), 0.5), try sc.scalar(0), try sc.scalar(1), s), &o);
        try expectParity("vae", img, want, s, 0.9999);
    }

    // The whole pipeline: 4 steps, CFG 3 with block 1 skipped in the negative branch.
    {
        eng.uncond_skip = 1;
        const init = try toTokens(&sc, try get(&fx.w, "e2e_init"));
        const img = try sc.keep(try eng.generateFrom(prompt, @intCast(lw * 16), @intCast(lh * 16), init, 4, .{ .guidance_scale = 3.0, .negative_prompt = "worst quality" }, null));
        try expectParity("pipeline", img, try get(&fx.w, "e2e_image"), s, 0.999);
    }
}

// Bar: the text encoder's hidden rows bit-equal to the legacy trunk's on the tiny pack, recorded in
// `fixtures/text_encoder.tape`.
test "MOSS-Image: the text encoder's served trunk is bit-equal to the engine's own trunk" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tape_mod = @import("../../sdk/testing/tape.zig");
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fx = try @import("plugin.zig").conformanceFixture(arena.allocator());
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for (fx.files) |f| {
        if (std.fs.path.dirname(f.name)) |d| try tmp.dir.createDirPath(io, d);
        try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    }
    var pbuf: [512]u8 = undefined;
    const dir = pbuf[0..try tmp.dir.realPath(io, &pbuf)];
    const eng = try Engine.loadOn(io, a, dir, mlx.mlx_default_gpu_stream_new());
    defer eng.deinit();
    const h = try eng.encodePrompt("a red cat on a chair");
    defer _ = mlx.mlx_array_free(h);
    var tape = try tape_mod.Tape.player(@embedFile("fixtures/text_encoder.tape"));
    try tape.expectNext(h, false);
    try tape.expectDone();
}
