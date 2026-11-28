//! Ming-Image's prompt conditioning, the release's `get_condition_embeds_for_image_gen`:
//!  - the MLLM (`mllm/`, a Bailing-MoE-v2 decoder: GQA with q/k RMS norm and half-width RoPE, layer 0 dense, the
//!    rest sigmoid + expert-bias routed with group limits, a shared expert, and a second router for image rows)
//!    reads the chat-formatted prompt followed by `<image start> 256 learnable query tokens <image end>`, once,
//!    causally, with the release's video-RoPE positions for the query block;
//!  - the query rows of the final (normed) state go through `proj_in`, the non-causal Qwen2 `connector/` and
//!    `proj_out`: the DiT's caption features;
//!  - the prompt rows of the tapped states (`selected_hidden_states_layers`) are concatenated, RMS-normed and
//!    projected (`proj_directvlm`): the DiT's second caption stream, already at its width.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const log = @import("../../sdk/util/log.zig");
const params = @import("../../sdk/quant/params.zig");
const scope_mod = @import("../../mlx/scope.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const BLinear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const Linear = @import("../../sdk/blocks/dit/linear.zig").Linear;
const rope_blk = @import("../../sdk/blocks/dit/rope.zig");
const qwen3 = @import("../../sdk/blocks/text_encoders/qwen3.zig");

const Weights = weights_load.Weights;
const Graph = ops.Graph;
const Array = ops.Array;
const S = mlx.mlx_stream;

pub const DT: ops.Dtype = .bfloat16;

/// The release's system-turn content (its chat template's `detailed thinking off` follows it).
pub const DEFAULT_SYSTEM = "你是一个友好的AI助手。";
const PROMPT_SUFFIX = "<|role_end|><role>ASSISTANT</role>";

/// The chat template's system turn for `content`: `<role>SYSTEM</role>` + `content` + `\n\n` + the thinking option.
/// Empty content leaves the turn bare, as the template does with no system message.
pub fn systemTurn(a: std.mem.Allocator, content: []const u8) ![]u8 {
    return std.mem.concat(a, u8, &.{ "<role>SYSTEM</role>", content, if (content.len > 0) "\n\n" else "", "detailed thinking off<|role_end|>" });
}

/// `MLX_SERVE_MING_SYSTEM` replaces the system-turn content: the MLLM reads the prompt only (no assistant turn is
/// generated), so that line is all the instruction its caption conditioning is steered by. Empty drops it.
fn systemContent() []const u8 {
    return env.span(.MLX_SERVE_MING_SYSTEM) orelse DEFAULT_SYSTEM;
}

fn promptPrefix(a: std.mem.Allocator) ![]u8 {
    const sys = try systemTurn(a, systemContent());
    defer a.free(sys);
    return std.mem.concat(a, u8, &.{ sys, "<role>HUMAN</role>" });
}

/// The longest chat-formatted prompt a request may encode (the release truncates nowhere, so a longer one is refused:
/// cutting it would drop the assistant turn the caption streams are read after).
pub const MAX_PROMPT_TOKENS: usize = 4096;

// ── loading helpers (also the DiT's) ──

pub fn get(w: *const Weights, comptime fmt: []const u8, args: anytype) !Array {
    return scope_mod.getW(w, fmt, args);
}

/// A checkpoint linear (dense or affine-quantized, optional bias).
pub fn lin(w: *const Weights, comptime fmt: []const u8, args: anytype, in_dim: u32) !Linear {
    const l = try scope_mod.linear(w, fmt, args, in_dim);
    return .{ .w = l.w, .scales = l.scales, .biases = l.biases, .bias = l.bias, .bits = l.bits, .group_size = l.group_size };
}

/// Every float tensor but a quantized weight's scales and biases cast to `dt` in the map (the release runs each
/// component in bf16, the connector ships f32): the stored copy is freed once the cast evaluates.
pub fn castFloats(w: *Weights, dt: ops.Dtype, s: S) !void {
    var it = w.map.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (std.mem.endsWith(u8, k, ".scales") or std.mem.endsWith(u8, k, ".biases")) continue;
        const have = mlx.mlx_array_dtype(e.value_ptr.*);
        if (have == dt or !(have == .float32 or have == .float16 or have == .bfloat16)) continue;
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o, e.value_ptr.*, dt, s));
        _ = mlx.mlx_array_free(e.value_ptr.*);
        e.value_ptr.* = o;
    }
}

pub fn loadDir(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, sub: []const u8, opts: weights_load.LoadOpts) !Weights {
    const dir = try std.fmt.allocPrint(a, "{s}/{s}", .{ model_dir, sub });
    defer a.free(dir);
    return weights_load.loadWeightsOpt(io, a, dir, opts);
}

pub fn readJson(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, rel: []const u8) !std.json.Parsed(std.json.Value) {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ model_dir, rel });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20));
    defer a.free(bytes);
    return std.json.parseFromSlice(std.json.Value, a, bytes, .{});
}

pub fn num(v: std.json.Value, key: []const u8) !f64 {
    const x = v.object.get(key) orelse {
        log.err("[ming-image] config misses {s}\n", .{key});
        return error.MingImageBadConfig;
    };
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.MingImageBadConfig,
    };
}

pub fn int(v: std.json.Value, key: []const u8) !u32 {
    const f = try num(v, key);
    if (f < 0 or f > 1 << 24 or f != @floor(f)) return error.MingImageBadConfig;
    return @intFromFloat(f);
}

// ── config ──

pub const Config = struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    rope_theta: f32,
    eps: f32,
    experts: u32,
    top_k: u32,
    n_group: u32,
    topk_group: u32,
    routed_scale: f32,
    first_dense: u32,
    patch_token: u32,
    /// The learnable block's side (16: 256 query tokens).
    scale: u32,
    taps: [3]u32,
    conn: qwen3.Cfg,
    conn_layers: u32,
    conn_theta: f32,
    /// The DiT's caption width (`proj_out`) and model width (`proj_directvlm`).
    cap_dim: u32,
    inner_dim: u32,

    pub fn queries(c: Config) u32 {
        return c.scale * c.scale;
    }

    pub fn parse(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Config {
        var mllm = try readJson(io, a, model_dir, "mllm/config.json");
        defer mllm.deinit();
        const l = mllm.value.object.get("llm_config") orelse return error.MingImageBadConfig;
        var mlp = try readJson(io, a, model_dir, "mlp/config.json");
        defer mlp.deinit();
        var conn = try readJson(io, a, model_dir, "connector/config.json");
        defer conn.deinit();
        const m = mlp.value;
        // The released recipe; any other answer is a pipeline this port does not run.
        const flags = [_]struct { k: []const u8, want: bool }{
            .{ .k = "use_identity_mlp", .want = true },          .{ .k = "use_vlm_directvlm_condition", .want = true },
            .{ .k = "use_learnable_token_condition", .want = true }, .{ .k = "text_encoder_norm", .want = false },
            .{ .k = "connector_norm", .want = false },
        };
        for (flags) |f| {
            const got = m.object.get(f.k) orelse return error.MingImageBadConfig;
            if (got != .bool or got.bool != f.want) {
                log.err("[ming-image] mlp/config.json {s} must be {}\n", .{ f.k, f.want });
                return error.MingImageUnsupportedRecipe;
            }
        }
        const scales = m.object.get("img_gen_scales") orelse return error.MingImageBadConfig;
        const taps = m.object.get("selected_hidden_states_layers") orelse return error.MingImageBadConfig;
        if (scales != .array or scales.array.items.len != 1 or scales.array.items[0] != .integer) return error.MingImageUnsupportedRecipe;
        if (taps != .array or taps.array.items.len != 3) return error.MingImageUnsupportedRecipe;
        var c: Config = .{
            .hidden = try int(l, "hidden_size"),
            .layers = try int(l, "num_hidden_layers"),
            .heads = try int(l, "num_attention_heads"),
            .kv_heads = try int(l, "num_key_value_heads"),
            .head_dim = try int(l, "head_dim"),
            .rope_dim = 0,
            .rope_theta = @floatCast(try num(l, "rope_theta")),
            .eps = @floatCast(try num(l, "rms_norm_eps")),
            .experts = try int(l, "num_experts"),
            .top_k = try int(l, "num_experts_per_tok"),
            .n_group = try int(l, "n_group"),
            .topk_group = try int(l, "topk_group"),
            .routed_scale = @floatCast(try num(l, "routed_scaling_factor")),
            .first_dense = try int(l, "first_k_dense_replace"),
            .patch_token = try int(l, "image_patch_token"),
            .scale = @intCast(scales.array.items[0].integer),
            .taps = undefined,
            .conn = .{ .heads = @intCast(try int(conn.value, "num_attention_heads")), .kv_heads = @intCast(try int(conn.value, "num_key_value_heads")), .head_dim = 0, .eps = @floatCast(try num(conn.value, "rms_norm_eps")) },
            .conn_layers = try int(conn.value, "num_hidden_layers"),
            .conn_theta = @floatCast(try num(conn.value, "rope_theta")),
            .cap_dim = try int(m, "diffusion_c_input_dim"),
            .inner_dim = try int(m, "diffusion_inner_dim"),
        };
        c.rope_dim = @intFromFloat(@as(f64, @floatFromInt(c.head_dim)) * try num(l, "partial_rotary_factor"));
        c.conn.head_dim = @intCast(try int(conn.value, "hidden_size") / @as(u32, @intCast(c.conn.heads)));
        for (&c.taps, taps.array.items) |*t, v| {
            if (v != .integer or v.integer < 0 or v.integer > c.layers) return error.MingImageBadConfig;
            t.* = @intCast(v.integer);
        }
        if (c.heads == 0 or c.kv_heads == 0 or c.heads % c.kv_heads != 0 or c.rope_dim != 64 or c.rope_dim > c.head_dim or
            c.experts % c.n_group != 0 or c.top_k == 0 or c.scale == 0 or c.conn.head_dim == 0)
            return error.MingImageBadConfig;
        return c;
    }
};

/// Where an edit's picture sits in the prompt: its first patch row and its merged grid.
pub const Pic = struct { at: usize, hm: u32, wm: u32 };

/// `get_t_scale_rope_index` (scale 2, one frame per block) for `[prompt | start | query block | end]`, `pic` inside
/// the prompt: text runs 1-D; a block of merged grid (hm, wm) starting at s sits at t = s, h = s + r − ⌊(hm−1)/2⌋,
/// w = s + c − ⌊(wm−1)/2⌋, and the text after it resumes at s + 1. The query block is a 1 x n grid. A rotary
/// channel's frequency is `c mod (rope_dim/2)`; its axis h (even c) / w (odd c) over the first 3/8 of each half, t
/// over the rest.
pub fn ropeTables(g: *Graph, cfg: Config, n: usize, pic: ?Pic) !struct { cos: Array, sin: Array } {
    const q: usize = cfg.queries();
    const L = n + 1 + q + 1;
    const pos = try g.a.alloc([3]f64, L);
    defer g.a.free(pos);
    var next: f64 = 0; // the next text position
    var i: usize = 0;
    const blocks = [_]?struct { at: usize, hm: u32, wm: u32 }{
        if (pic) |pc| .{ .at = pc.at, .hm = pc.hm, .wm = pc.wm } else null,
        .{ .at = n + 1, .hm = 1, .wm = @intCast(q) },
    };
    for (blocks) |ob| if (ob) |b| {
        while (i < b.at) : (i += 1) {
            pos[i] = .{ next, next, next };
            next += 1;
        }
        const s0 = next;
        for (0..b.hm) |r| for (0..b.wm) |c| {
            const hh = s0 + @as(f64, @floatFromInt(r)) - @as(f64, @floatFromInt((b.hm - 1) / 2));
            const ww = s0 + @as(f64, @floatFromInt(c)) - @as(f64, @floatFromInt((b.wm - 1) / 2));
            pos[i] = .{ s0, hh, ww };
            i += 1;
        };
        next = s0 + 1;
    };
    while (i < L) : (i += 1) {
        pos[i] = .{ next, next, next };
        next += 1;
    }
    const rd: usize = cfg.rope_dim;
    const half = rd / 2;
    const hw = half * 3 / 4; // [8,12,12] sections × 2 over the 64-wide rotary: 24 of every 32 are h/w
    const cos = try g.a.alloc(f32, L * rd);
    defer g.a.free(cos);
    const sin = try g.a.alloc(f32, L * rd);
    defer g.a.free(sin);
    for (pos, 0..) |pp, row| for (0..rd) |c| {
        const k = c % half;
        const at = if (k >= hw) pp[0] else if (c % 2 == 0) pp[1] else pp[2];
        const inv = 1.0 / std.math.pow(f64, cfg.rope_theta, @as(f64, @floatFromInt(2 * k)) / @as(f64, @floatFromInt(rd)));
        const ang: f32 = @floatCast(at * @as(f64, @as(f32, @floatCast(inv))));
        cos[row * rd + c] = @cos(ang);
        sin[row * rd + c] = @sin(ang);
    };
    const sh = [_]c_int{ 1, 1, @intCast(L), @intCast(rd) };
    return .{ .cos = try g.astype(try g.fromSlice(f32, cos, &sh), DT), .sin = try g.astype(try g.fromSlice(f32, sin, &sh), DT) };
}

// ── the MLLM ──

const Mlp = struct { gate: Linear, up: Linear, down: Linear };

const Layer = struct {
    in_ln: Array,
    post_ln: Array,
    qkv: Linear,
    dense: Linear,
    q_norm: Array,
    k_norm: Array,
    mlp: union(enum) {
        dense: Mlp,
        /// One bank, two routers: text rows (`gate`), query rows (`image_gate`).
        moe: struct { text: moe.Moe.Weights, image: moe.Moe.Weights },
    },
};

fn bLinear(l: Linear) BLinear {
    if (l.scales) |sc| return .{ .w = l.w, .s = sc, .b = l.biases, .bits = l.bits, .group_size = l.group_size };
    return .{ .w = l.w };
}

/// Linears the MoE block reads keep their weights pre-transposed when dense (`[in, out]`).
fn bLinearT(g: *Graph, l: Linear) !BLinear {
    if (l.scales != null) return bLinear(l);
    return .{ .w = try g.transpose(l.w, &.{ 1, 0 }) };
}

/// The routed experts of one projection stacked into a bank `[E, out, in]` (quantized) or `[E, in, out]` (dense,
/// what `gather_mm` reads), from the per-expert tensors both the release and mlx-community's tree store. Only the
/// banks outlive the call: a transposed view kept in `g` would pin its expert's buffer after the map drops it.
fn bank(g: *Graph, w: *const Weights, layer: usize, comptime proj: []const u8, experts: u32, in_dim: u32) !moe.Bank {
    var c = g.scope();
    defer c.deinit();
    const fmt = "model.model.layers.{d}.mlp.experts.{d}." ++ proj;
    const first = try lin(w, fmt, .{ layer, @as(usize, 0) }, in_dim);
    const parts = try g.a.alloc(Array, experts);
    defer g.a.free(parts);
    if (first.scales != null) {
        const sp = try g.a.alloc(Array, experts);
        defer g.a.free(sp);
        const bp = try g.a.alloc(Array, experts);
        defer g.a.free(bp);
        for (0..experts) |e| {
            const l = try lin(w, fmt, .{ layer, e }, in_dim);
            if (l.bits != first.bits or l.group_size != first.group_size) return error.MingImageMixedExpertQuant;
            parts[e] = l.w;
            sp[e] = l.scales.?;
            bp[e] = l.biases;
        }
        return .{ .w = try g.keep(try stack(&c, parts)), .s = try g.keep(try stack(&c, sp)), .b = try g.keep(try stack(&c, bp)), .qp = .{ .bits = first.bits, .group_size = first.group_size } };
    }
    for (0..experts) |e| parts[e] = try c.transpose((try lin(w, fmt, .{ layer, e }, in_dim)).w, &.{ 1, 0 });
    return .{ .w = try g.keep(try stack(&c, parts)), .qp = .{ .bits = 0, .group_size = 64 } };
}

fn stack(g: *Graph, xs: []const Array) !Array {
    const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    return g.call(mlx.mlx_stack_axis, .{ v, @as(c_int, 0) });
}

/// A router's f32 weight `[hidden, E]` (the release computes the logits in f32) and selection bias.
fn router(g: *Graph, w: *const Weights, layer: usize, comptime name: []const u8) !struct { lin: BLinear, bias: Array } {
    const wt = try get(w, "model.model.layers.{d}.mlp." ++ name ++ ".weight", .{layer});
    const b = try get(w, "model.model.layers.{d}.mlp." ++ name ++ ".expert_bias", .{layer});
    var c = g.scope();
    defer c.deinit();
    return .{ .lin = .{ .w = try g.keep(try c.contiguous(try c.transpose(try c.astype(wt, .float32), &.{ 1, 0 }))) }, .bias = try g.astype(b, .float32) };
}

/// The MLLM's own tensors (the vision tower, LM head and audio router stay on disk).
fn skipMllm(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "vision.") or std.mem.startsWith(u8, key, "linear_proj.") or
        std.mem.startsWith(u8, key, "model.lm_head.") or std.mem.indexOf(u8, key, ".audio_gate.") != null;
}

pub const Encoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: Config,
    tok: tok_mod.Tokenizer,
    /// Owns every derived array (stacked banks, f32 routers, transposed views); `w` the loaded ones.
    owned: Graph,
    w: Weights,
    cw: Weights,
    mw: Weights,
    embed: Array,
    norm: Array,
    layers: []Layer,
    query: Array,
    proj_in: Linear,
    proj_out: Linear,
    direct_norm: Array,
    direct: Linear,
    conn: []ConnLayer,
    conn_norm: Array,
    start_id: u32,
    end_id: u32,

    const ConnLayer = struct { input_ln: Array, post_ln: Array, q: Linear, k: Linear, v: Linear, o: Linear, gate: Linear, up: Linear, down: Linear };

    pub fn load(self: *Encoder, io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, cfg: Config) !void {
        self.allocator = a;
        self.s = s;
        self.cfg = cfg;
        self.start_id = cfg.patch_token + 1;
        self.end_id = cfg.patch_token + 2;
        const tdir = try std.fmt.allocPrint(a, "{s}/mllm", .{model_dir});
        defer a.free(tdir);
        self.tok = try tok_mod.loadTokenizerAny(io, a, tdir);
        errdefer self.tok.deinit();
        self.owned = Graph.init(a, s);
        errdefer self.owned.deinit();
        self.w = try loadDir(io, a, model_dir, "mllm", .{ .skip = skipMllm });
        errdefer self.w.deinit();
        self.cw = try loadDir(io, a, model_dir, "connector", .{});
        errdefer self.cw.deinit();
        self.mw = try loadDir(io, a, model_dir, "mlp", .{});
        errdefer self.mw.deinit();
        try castFloats(&self.w, DT, s);
        try castFloats(&self.cw, DT, s);
        try castFloats(&self.mw, DT, s);
        const g = &self.owned;
        const w = &self.w;
        const h = cfg.hidden;
        self.embed = try get(w, "model.model.word_embeddings.weight", .{});
        self.norm = try get(w, "model.model.norm.weight", .{});
        self.layers = try a.alloc(Layer, cfg.layers);
        errdefer a.free(self.layers);
        for (self.layers, 0..) |*L, i| {
            L.* = .{
                .in_ln = try get(w, "model.model.layers.{d}.input_layernorm.weight", .{i}),
                .post_ln = try get(w, "model.model.layers.{d}.post_attention_layernorm.weight", .{i}),
                .qkv = try lin(w, "model.model.layers.{d}.attention.query_key_value", .{i}, h),
                .dense = try lin(w, "model.model.layers.{d}.attention.dense", .{i}, cfg.heads * cfg.head_dim),
                .q_norm = try get(w, "model.model.layers.{d}.attention.q_norm.weight", .{i}),
                .k_norm = try get(w, "model.model.layers.{d}.attention.k_norm.weight", .{i}),
                .mlp = undefined,
            };
            if (i < cfg.first_dense) {
                const up = try lin(w, "model.model.layers.{d}.mlp.up_proj", .{i}, h);
                L.mlp = .{ .dense = .{ .gate = try lin(w, "model.model.layers.{d}.mlp.gate_proj", .{i}, h), .up = up, .down = try lin(w, "model.model.layers.{d}.mlp.down_proj", .{i}, try outDim(up)) } };
                continue;
            }
            const sh_up = try lin(w, "model.model.layers.{d}.mlp.shared_experts.up_proj", .{i}, h);
            const shared: moe.SharedExpert = .{
                .gate = try bLinearT(g, try lin(w, "model.model.layers.{d}.mlp.shared_experts.gate_proj", .{i}, h)),
                .up = try bLinearT(g, sh_up),
                .down = try bLinearT(g, try lin(w, "model.model.layers.{d}.mlp.shared_experts.down_proj", .{i}, try outDim(sh_up))),
            };
            const e_up = try lin(w, "model.model.layers.{d}.mlp.experts.0.up_proj", .{i}, h);
            const inter = try outDim(e_up);
            const base: moe.Moe.Weights = .{
                .router = undefined,
                .route = undefined,
                .gate = try bank(g, w, i, "gate_proj", cfg.experts, h),
                .up = try bank(g, w, i, "up_proj", cfg.experts, h),
                .down = try bank(g, w, i, "down_proj", cfg.experts, inter),
                .shared = shared,
                .top_k = @intCast(cfg.top_k),
                .experts = @intCast(cfg.experts),
            };
            var text = base;
            var image = base;
            inline for (.{ .{ &text, "gate" }, .{ &image, "image_gate" } }) |pr| {
                const r = try router(g, w, i, pr[1]);
                pr[0].router = r.lin;
                pr[0].route = .{ .sigmoid_bias = .{ .expert_bias = r.bias, .norm = true, .scale = cfg.routed_scale, .n_group = @intCast(cfg.n_group), .topk_group = @intCast(cfg.topk_group) } };
            }
            L.mlp = .{ .moe = .{ .text = text, .image = image } };
            // The bank holds the experts now; the per-expert originals go once it evaluates.
            try evalBanks(&L.mlp.moe.text);
            dropExperts(w, i, cfg.experts);
        }

        const mw = &self.mw;
        var qb: [32]u8 = undefined;
        self.query = try get(mw, "query_tokens_dict.{s}", .{try std.fmt.bufPrint(&qb, "{d}x{d}", .{ cfg.scale, cfg.scale })});
        if (ops.shape(self.query)[0] != cfg.queries() or ops.shape(self.query)[1] != h) return error.MingImageBadConfig;
        self.proj_in = try lin(mw, "proj_in", .{}, h);
        const conn_h = try outDim(self.proj_in);
        self.proj_out = try lin(mw, "proj_out", .{}, conn_h);
        self.direct_norm = try get(mw, "proj_directvlm.0.weight", .{});
        self.direct = try lin(mw, "proj_directvlm.1", .{}, 3 * h);
        if (try outDim(self.proj_out) != cfg.cap_dim or try outDim(self.direct) != cfg.inner_dim) return error.MingImageBadConfig;

        const cw = &self.cw;
        self.conn_norm = try get(cw, "model.norm.weight", .{});
        self.conn = try a.alloc(ConnLayer, cfg.conn_layers);
        errdefer a.free(self.conn);
        for (self.conn, 0..) |*c, i| {
            const up = try lin(cw, "model.layers.{d}.mlp.up_proj", .{i}, conn_h);
            c.* = .{
                .input_ln = try get(cw, "model.layers.{d}.input_layernorm.weight", .{i}),
                .post_ln = try get(cw, "model.layers.{d}.post_attention_layernorm.weight", .{i}),
                .q = try lin(cw, "model.layers.{d}.self_attn.q_proj", .{i}, conn_h),
                .k = try lin(cw, "model.layers.{d}.self_attn.k_proj", .{i}, conn_h),
                .v = try lin(cw, "model.layers.{d}.self_attn.v_proj", .{i}, conn_h),
                .o = try lin(cw, "model.layers.{d}.self_attn.o_proj", .{i}, conn_h),
                .gate = try lin(cw, "model.layers.{d}.mlp.gate_proj", .{i}, conn_h),
                .up = up,
                .down = try lin(cw, "model.layers.{d}.mlp.down_proj", .{i}, try outDim(up)),
            };
        }
        self.cw.remove("model.embed_tokens.weight"); // the connector reads embeddings, never ids
        try scope_mod.evalAll(&self.w);
        try scope_mod.evalAll(&self.cw);
        try scope_mod.evalAll(&self.mw);
    }

    pub fn deinit(self: *Encoder) void {
        self.allocator.free(self.conn);
        self.allocator.free(self.layers);
        self.owned.deinit();
        self.mw.deinit();
        self.cw.deinit();
        self.w.deinit();
        self.tok.deinit();
    }

    /// The chat-formatted prompt's ids (the release's processor); an edit's picture is `<image>`, `picture` patch
    /// tokens, `</image>` and a newline before the prompt. `error.PromptTooLong` past `MAX_PROMPT_TOKENS`.
    pub fn promptIds(self: *const Encoder, a: std.mem.Allocator, prompt: []const u8, picture: u32) ![]u32 {
        const prefix = try promptPrefix(a);
        defer a.free(prefix);
        const ids = if (picture == 0) blk: {
            const text = try std.mem.concat(a, u8, &.{ prefix, prompt, PROMPT_SUFFIX });
            defer a.free(text);
            break :blk try self.tok.encode(a, text);
        } else blk: {
            // `</image>` is a special token: the text after it starts a new pretokenizer segment either way.
            const head = try self.tok.encode(a, prefix);
            defer a.free(head);
            const text = try std.mem.concat(a, u8, &.{ "\n", prompt, PROMPT_SUFFIX });
            defer a.free(text);
            const tail = try self.tok.encode(a, text);
            defer a.free(tail);
            const out = try a.alloc(u32, head.len + picture + 2 + tail.len);
            @memcpy(out[0..head.len], head);
            out[head.len] = self.start_id;
            @memset(out[head.len + 1 ..][0..picture], self.cfg.patch_token);
            out[head.len + 1 + picture] = self.end_id;
            @memcpy(out[head.len + picture + 2 ..], tail);
            break :blk out;
        };
        if (ids.len > MAX_PROMPT_TOKENS) {
            a.free(ids);
            return error.PromptTooLong;
        }
        return ids;
    }

    /// An edit's picture in the MLLM: the tower's rows and their merged grid.
    pub const Picture = struct { embeds: Array, hm: u32, wm: u32 };

    pub const Cond = struct {
        /// The connector's caption features [1, queries, cap_dim] and the projected direct stream [1, prompt, inner].
        learn: Array,
        direct: Array,
    };

    /// Both caption streams for the prompt `ids` (with `pic`'s patch tokens when editing), in `parent`.
    pub fn encode(self: *const Encoder, parent: *Graph, ids: []const u32, pic: ?Picture) !Cond {
        var g = parent.scope();
        defer g.deinit();
        const cfg = self.cfg;
        const n = ids.len;
        const p = n + 1; // the prompt and the block's start token
        const q: usize = cfg.queries();
        var head = try g.a.alloc(u32, p);
        defer g.a.free(head);
        @memcpy(head[0..n], ids);
        head[n] = self.start_id;
        const end = [_]u32{self.end_id};
        var emb = try g.take(self.embed, try g.fromSlice(u32, head, &.{@intCast(p)}), 0);
        var span: ?Pic = null;
        if (pic) |pc| {
            const at = std.mem.indexOfScalar(u32, ids, cfg.patch_token) orelse return error.MingImageBadPicture;
            const rows: usize = @as(usize, pc.hm) * pc.wm;
            if (at + rows > n or ops.shape(pc.embeds)[0] != rows or std.mem.count(u32, ids, &.{cfg.patch_token}) != rows)
                return error.MingImageBadPicture;
            emb = try g.concat(&.{ try g.sliceAxis(emb, 0, 0, @intCast(at)), try g.astype(pc.embeds, DT), try g.sliceAxis(emb, 0, @intCast(at + rows), @intCast(p)) }, 0);
            span = .{ .at = at, .hm = pc.hm, .wm = pc.wm };
        }
        const x0 = try g.concat(&.{ emb, self.query, try g.take(self.embed, try g.fromSlice(u32, &end, &.{1}), 0) }, 0);
        var x = try g.expandDims(x0, 0);
        const rt = try ropeTables(&g, cfg, n, span);
        // The image router reads the picture's rows and the query block; the text router the rest.
        var img_rows: std.ArrayList(u32) = .empty;
        defer img_rows.deinit(g.a);
        if (span) |sp| for (sp.at..sp.at + @as(usize, sp.hm) * sp.wm) |r| try img_rows.append(g.a, @intCast(r));
        for (p..p + q) |r| try img_rows.append(g.a, @intCast(r));
        var taps: [3]?Array = .{ null, null, null };
        for (self.layers, 0..) |*L, i| {
            for (cfg.taps, 0..) |t, ti| if (t == i) {
                taps[ti] = x;
            };
            var step = g.scope();
            defer step.deinit();
            const next = try g.keep(try self.layer(&step, L, x, rt.cos, rt.sin, img_rows.items));
            try mlx.check(mlx.mlx_array_eval(next));
            x = next;
        }
        const final = try g.rmsNorm(x, self.norm, cfg.eps);
        for (cfg.taps, 0..) |t, ti| if (t == cfg.layers) {
            taps[ti] = final;
        };

        // Direct stream: the prompt rows of the taps (the start token and the block are labelled, not read).
        var cat: [3]Array = undefined;
        for (&cat, taps) |*c, t| c.* = try g.sliceAxis(t.?, 1, 0, @intCast(n));
        const direct = try self.direct.forward(&g, try g.rmsNorm(try g.concat(&cat, 2), self.direct_norm, 1e-5));

        // Learnable stream: the query rows through the non-causal connector.
        var y = try self.proj_in.forward(&g, try g.sliceAxis(final, 1, @intCast(p), @intCast(p + q)));
        const ccfg = cfg.conn;
        for (self.conn) |*c| {
            var step = g.scope();
            defer step.deinit();
            y = try g.keep(try qwen3.layerForward(&step, c, ccfg, .{ .fast = .{ .theta = cfg.conn_theta } }, y, null));
        }
        const learn = try self.proj_out.forward(&g, try g.rmsNorm(y, self.conn_norm, ccfg.eps));
        // In the release's bf16: a pack with f32 quant scales (the f32 connector quantized as stored) would carry
        // f32 into every DiT block through the caption concat.
        return .{ .learn = try parent.keep(try g.astype(learn, DT)), .direct = try parent.keep(try g.astype(direct, DT)) };
    }

    fn layer(self: *const Encoder, parent: *Graph, L: *const Layer, x: Array, cos: Array, sin: Array, img_rows: []const u32) !Array {
        var g = parent.scope();
        defer g.deinit();
        const cfg = self.cfg;
        const seq = ops.shape(x)[1];
        const hd: c_int = @intCast(cfg.head_dim);
        const nh: c_int = @intCast(cfg.heads);
        const nkv: c_int = @intCast(cfg.kv_heads);
        const qkv = try g.reshape(try L.qkv.forward(&g, try g.rmsNorm(x, L.in_ln, cfg.eps)), &.{ 1, seq, nh + 2 * nkv, hd });
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const qh = try g.transpose(try g.rmsNorm(try g.sliceAxis(qkv, 2, 0, nh), L.q_norm, cfg.eps), &perm);
        const kh = try g.transpose(try g.rmsNorm(try g.sliceAxis(qkv, 2, nh, nh + nkv), L.k_norm, cfg.eps), &perm);
        const vh = try g.transpose(try g.sliceAxis(qkv, 2, nh + nkv, nh + 2 * nkv), &perm);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const at = try g.sdpa(try self.ropeHead(&g, qh, cos, sin), try self.ropeHead(&g, kh, cos, sin), vh, scale, .causal);
        const h1 = try g.add(x, try L.dense.forward(&g, try g.mergeHeads(at)));
        const hn = try g.rmsNorm(h1, L.post_ln, cfg.eps);
        const m = switch (L.mlp) {
            .dense => |d| try d.down.forward(&g, try g.mul(try g.silu(try d.gate.forward(&g, hn)), try d.up.forward(&g, hn))),
            .moe => |*r| try routedRows(&g, &r.text, &r.image, hn, img_rows),
        };
        return parent.keep(try g.add(h1, m));
    }

    /// Rotate-half RoPE over the first `rope_dim` channels of each head, the rest passed through.
    fn ropeHead(self: *const Encoder, parent: *Graph, x: Array, cos: Array, sin: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const rd: c_int = @intCast(self.cfg.rope_dim);
        const d: c_int = @intCast(self.cfg.head_dim);
        const r = try rope_blk.ropeRotateHalf(&g, try g.sliceAxis(x, 3, 0, rd), cos, sin, DT);
        if (rd == d) return parent.keep(r);
        return parent.keep(try g.concat(&.{ r, try g.sliceAxis(x, 3, rd, d) }, 3));
    }
};

/// The MoE over `x`'s rows, `img_rows` (ascending) routed by the image router and every other row by the text router
/// (rows are independent through the experts).
fn routedRows(parent: *Graph, text: *const moe.Moe.Weights, image: *const moe.Moe.Weights, x: Array, img_rows: []const u32) !Array {
    var g = parent.scope();
    defer g.deinit();
    const L: usize = @intCast(ops.shape(x)[1]);
    const order = try g.a.alloc(u32, L); // text rows, then image rows
    defer g.a.free(order);
    const back = try g.a.alloc(u32, L);
    defer g.a.free(back);
    var t: usize = 0;
    var k: usize = 0;
    for (0..L) |r| {
        if (k < img_rows.len and img_rows[k] == r) {
            k += 1;
            continue;
        }
        order[t] = @intCast(r);
        t += 1;
    }
    @memcpy(order[t..], img_rows);
    for (order, 0..) |r, i| back[r] = @intCast(i);
    const ti = try g.fromSlice(u32, order[0..t], &.{@intCast(t)});
    const ii = try g.fromSlice(u32, img_rows, &.{@intCast(img_rows.len)});
    const y = try g.concat(&.{ try moe.mlp(text, &g, try g.take(x, ti, 1)), try moe.mlp(image, &g, try g.take(x, ii, 1)) }, 1);
    return parent.keep(try g.take(y, try g.fromSlice(u32, back, &.{@intCast(L)}), 1));
}

fn outDim(l: Linear) !u32 {
    return @intCast(ops.shape(l.w)[0]);
}

fn evalBanks(m: *const moe.Moe.Weights) !void {
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    inline for (.{ "gate", "up", "down" }) |f| {
        const b = @field(m.*, f);
        _ = mlx.mlx_vector_array_append_value(v, b.w);
        if (b.s.ctx != null) _ = mlx.mlx_vector_array_append_value(v, b.s);
        if (b.b.ctx != null) _ = mlx.mlx_vector_array_append_value(v, b.b);
    }
    try mlx.check(mlx.mlx_eval(v));
}

fn dropExperts(w: *Weights, layer: usize, experts: u32) void {
    var buf: [128]u8 = undefined;
    for (0..experts) |e| inline for (.{ "gate_proj", "up_proj", "down_proj" }) |p| inline for (.{ "weight", "scales", "biases" }) |leaf| {
        if (std.fmt.bufPrint(&buf, "model.model.layers.{d}.mlp.experts.{d}." ++ p ++ "." ++ leaf, .{ layer, e })) |k| w.remove(k) else |_| {}
    };
}

// ── tests ──

const testing = std.testing;

// Bar: once the map drops the per-expert tensors, the stacked bank is the only copy left (a view kept past the
// call pinned all 30 GB of the release's experts).
test "ming_image llm: a stacked expert bank holds no reference to the per-expert tensors" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var w = Weights.init(a);
    defer w.deinit();
    const experts = 16;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    for (0..experts) |e| {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_random_normal(&r, &[_]c_int{ 512, 512 }, 2, .float32, 0, 1, key, s));
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, r, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(x));
        var kb: [96]u8 = undefined;
        try w.put(try std.fmt.bufPrint(&kb, "model.model.layers.0.mlp.experts.{d}.gate_proj.weight", .{e}), x);
    }
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    var g = Graph.init(a, s);
    defer g.deinit();
    const b = try bank(&g, &w, 0, "gate_proj", experts, 512);
    try mlx.check(mlx.mlx_array_eval(b.w));
    dropExperts(&w, 0, experts);
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    // The 8 MiB of originals moved into the bank: active memory stays put (it grew by 8 MiB with the view kept).
    try testing.expect(after <= before + (2 << 20));
    try testing.expectEqual(@as(u32, 0), w.count());
}

test "ming_image llm: the system turn is the release's line; an empty one leaves the template bare" {
    const a = std.testing.allocator;
    const sys = try systemTurn(a, DEFAULT_SYSTEM);
    defer a.free(sys);
    try std.testing.expectEqualStrings("<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|>", sys);
    const bare = try systemTurn(a, "");
    defer a.free(bare);
    try std.testing.expectEqualStrings("<role>SYSTEM</role>detailed thinking off<|role_end|>", bare);
    const prefix = try promptPrefix(a);
    defer a.free(prefix);
    try std.testing.expectEqualStrings("<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>", prefix);
}
