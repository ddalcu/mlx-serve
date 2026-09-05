//! SDXL's `UNet2DConditionModel` — the denoiser.
//!
//! 2.6B parameters over 1680 tensors, but structurally small: three down
//! blocks, a mid block, three up blocks, all built from two pieces (a
//! `ResnetBlock2D` and a `Transformer2DModel`) that `sdxl_nn.zig` and the
//! structs below define once each.
//!
//! Everything about the geometry is READ FROM THE CHECKPOINT's config rather
//! than hardcoded, because SDXL base and refiner disagree on nearly all of it
//! (the refiner is 384-wide, 4 down blocks, no 10-layer stage). What is fixed
//! is the SHAPE of the architecture — which block types appear in which order —
//! and that is asserted at load rather than assumed.
//!
//! TRAPS, each of which produces a running model and a wrong image:
//!
//!   `attention_head_dim` IS THE HEAD COUNT. Diffusers' own field name is
//!   wrong here and has been for years: SDXL's `[5, 10, 20]` are head COUNTS
//!   against block widths `[320, 640, 1280]`, so the head DIM is 64 throughout.
//!   Reading it as a dim gives 5 heads of 5 channels and a model that runs.
//!
//!   `use_linear_projection: true` REORDERS the transformer's entry. With it,
//!   the spatial tensor is flattened to tokens and THEN projected by a linear;
//!   without it, a 1x1 conv projects first and the flatten follows. The weights
//!   are the same numbers either way — `[640, 640]` is a valid conv kernel and a
//!   valid linear — so the wrong order is a silent transpose of the projection.
//!
//!   TWO GROUP-NORM EPSILONS. Resnets and the output norm use the config's
//!   `norm_eps` (1e-5); `Transformer2DModel`'s input norm hardcodes 1e-6
//!   upstream. See `sdxl_nn.groupNorm`.
//!
//!   THE FIRST SKIP IS `conv_in`'s OUTPUT. The up path pops nine skips; only
//!   eight come from the down blocks. Losing the ninth still type-checks
//!   because every up block pops the same count.
//!
//! ORACLE STATUS: pinned against diffusers' own `UNet2DConditionModel` by
//! `tests/dump_sdxl_unet_fixtures.py`, per block and end to end. See the
//! `sdxl unet parity` test at the bottom.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const nn = @import("sdxl_nn.zig");
const sdxl = @import("sdxl.zig");
const lora_mod = @import("lora.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;
const Linear = nn.Linear;
const Resnet = nn.Resnet;

/// `Transformer2DModel`'s input GroupNorm eps, hardcoded upstream rather than
/// read from the config that supplies every other eps in this file.
const TRANSFORMER_NORM_EPS: f32 = 1e-6;
/// `BasicTransformerBlock`'s LayerNorm eps (torch default).
const LAYER_NORM_EPS: f32 = 1e-5;

// ══════════════════════════════════════════════════════════════════
// Config
// ══════════════════════════════════════════════════════════════════

pub const UnetConfig = struct {
    in_channels: u32 = 4,
    out_channels: u32 = 4,
    /// Per-stage widths, e.g. `[320, 640, 1280]`.
    block_out_channels: []const u32,
    /// Head COUNT per stage — diffusers' misnamed `attention_head_dim`.
    num_heads: []const u32,
    /// `BasicTransformerBlock`s per `Transformer2DModel`, per stage.
    transformer_layers: []const u32,
    /// Which down stages carry attention. SDXL base: `[false, true, true]`.
    down_has_attn: []const bool,
    /// Which up stages carry attention. SDXL base: `[true, true, false]`.
    up_has_attn: []const bool,
    layers_per_block: u32 = 2,
    cross_attention_dim: u32 = 2048,
    /// `projection_class_embeddings_input_dim` — 2816 = pooled 1280 + 6x256.
    add_embed_input_dim: u32 = 2816,
    addition_time_embed_dim: u32 = 256,
    norm_eps: f32 = 1e-5,
    norm_num_groups: u32 = 32,
    /// True for SDXL's `add_embedding` micro-conditioning path (pooled text +
    /// crop/size ids). False for SD 1.x, whose `UNet2DConditionModel` has no
    /// `addition_embed_type` at all — same block/resnet/attention topology,
    /// just no augmentation on top of the plain time embedding.
    has_micro_conditioning: bool = true,

    pub fn timeEmbedDim(self: UnetConfig) u32 {
        return self.block_out_channels[0] * 4;
    }

    pub fn stages(self: UnetConfig) usize {
        return self.block_out_channels.len;
    }
};

/// SDXL base 1.0, verified against the checkpoint's own `unet/config.json`.
pub const BASE_CONFIG = UnetConfig{
    .block_out_channels = &[_]u32{ 320, 640, 1280 },
    .num_heads = &[_]u32{ 5, 10, 20 },
    .transformer_layers = &[_]u32{ 1, 2, 10 },
    .down_has_attn = &[_]bool{ false, true, true },
    .up_has_attn = &[_]bool{ true, true, false },
};

/// Parse the pieces of `unet/config.json` the forward actually reads.
pub fn parseConfig(allocator: std.mem.Allocator, json_bytes: []const u8) !UnetConfig {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    var cfg = UnetConfig{
        .block_out_channels = &[_]u32{},
        .num_heads = &[_]u32{},
        .transformer_layers = &[_]u32{},
        .down_has_attn = &[_]bool{},
        .up_has_attn = &[_]bool{},
    };

    const u32s = struct {
        fn read(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
            const arr = v.array;
            const out = try a.alloc(u32, arr.items.len);
            for (arr.items, out) |item, *o| o.* = @intCast(item.integer);
            return out;
        }
    };

    cfg.block_out_channels = try u32s.read(allocator, root.get("block_out_channels").?);
    errdefer allocator.free(cfg.block_out_channels);

    // `attention_head_dim` is the HEAD COUNT — see the file header. A scalar
    // spelling means the same count at every stage.
    const ahd = root.get("attention_head_dim") orelse return error.MissingAttentionHeadDim;
    cfg.num_heads = switch (ahd) {
        .array => try u32s.read(allocator, ahd),
        .integer => blk: {
            const out = try allocator.alloc(u32, cfg.block_out_channels.len);
            @memset(out, @intCast(ahd.integer));
            break :blk out;
        },
        else => return error.BadAttentionHeadDim,
    };
    errdefer allocator.free(cfg.num_heads);

    // SDXL's own config always states this explicitly (it varies per stage:
    // `[1, 2, 10]`), but diffusers defaults `transformer_layers_per_block` to
    // 1 when the key is absent — which is exactly SD 1.x's `unet/config.json`
    // (no per-stage depth ever needed there), so an absent key means 1 at
    // every stage rather than a load error.
    cfg.transformer_layers = if (root.get("transformer_layers_per_block")) |tlpb| switch (tlpb) {
        .array => try u32s.read(allocator, tlpb),
        .integer => blk: {
            const out = try allocator.alloc(u32, cfg.block_out_channels.len);
            @memset(out, @intCast(tlpb.integer));
            break :blk out;
        },
        else => return error.BadTransformerLayers,
    } else blk: {
        const out = try allocator.alloc(u32, cfg.block_out_channels.len);
        @memset(out, 1);
        break :blk out;
    };
    errdefer allocator.free(cfg.transformer_layers);

    // Attention presence comes from the BLOCK TYPE NAMES, which is the only
    // place the checkpoint states it. An unknown block type is refused by name
    // rather than silently treated as attention-free — that would load and
    // produce an image with two-thirds of the conditioning missing.
    const dbt = root.get("down_block_types").?.array;
    const dha = try allocator.alloc(bool, dbt.items.len);
    errdefer allocator.free(dha);
    for (dbt.items, dha) |item, *o| {
        const name = item.string;
        if (std.mem.eql(u8, name, "CrossAttnDownBlock2D")) {
            o.* = true;
        } else if (std.mem.eql(u8, name, "DownBlock2D")) {
            o.* = false;
        } else {
            log.err("[sdxl] unsupported down block type: {s}\n", .{name});
            return error.UnsupportedBlockType;
        }
    }
    cfg.down_has_attn = dha;

    const ubt = root.get("up_block_types").?.array;
    const uha = try allocator.alloc(bool, ubt.items.len);
    errdefer allocator.free(uha);
    for (ubt.items, uha) |item, *o| {
        const name = item.string;
        if (std.mem.eql(u8, name, "CrossAttnUpBlock2D")) {
            o.* = true;
        } else if (std.mem.eql(u8, name, "UpBlock2D")) {
            o.* = false;
        } else {
            log.err("[sdxl] unsupported up block type: {s}\n", .{name});
            return error.UnsupportedBlockType;
        }
    }
    cfg.up_has_attn = uha;

    // Every optional integer field below is guarded against a literal JSON
    // `null` (`if (root.get(...)) |v|` binds THAT too, not just an absent
    // key — see the `addition_embed_type` comment below): SD-Turbo's newer
    // diffusers version writes `addition_time_embed_dim` and
    // `projection_class_embeddings_input_dim` out as `null` rather than
    // omitting them, and an unguarded `v.integer` on a `.null` value is a
    // union-field-access panic, not a graceful skip.
    if (root.get("layers_per_block")) |v| if (v == .integer) {
        cfg.layers_per_block = @intCast(v.integer);
    };
    if (root.get("cross_attention_dim")) |v| if (v == .integer) {
        cfg.cross_attention_dim = @intCast(v.integer);
    };
    if (root.get("in_channels")) |v| if (v == .integer) {
        cfg.in_channels = @intCast(v.integer);
    };
    if (root.get("out_channels")) |v| if (v == .integer) {
        cfg.out_channels = @intCast(v.integer);
    };
    if (root.get("addition_time_embed_dim")) |v| if (v == .integer) {
        cfg.addition_time_embed_dim = @intCast(v.integer);
    };
    if (root.get("projection_class_embeddings_input_dim")) |v| if (v == .integer) {
        cfg.add_embed_input_dim = @intCast(v.integer);
    };
    if (root.get("norm_num_groups")) |v| if (v == .integer) {
        cfg.norm_num_groups = @intCast(v.integer);
    };
    if (root.get("norm_eps")) |v| cfg.norm_eps = switch (v) {
        .float => @floatCast(v.float),
        .integer => @floatFromInt(v.integer),
        else => 1e-5,
    };

    // `use_linear_projection` REORDERS the transformer entry (file header).
    // The conv spelling is a different forward, so refuse rather than run it.
    if (root.get("use_linear_projection")) |v| {
        if (v != .bool or !v.bool) return error.UnsupportedProjection;
    } else return error.UnsupportedProjection;

    // The additive time conditioning is what makes this SDXL rather than
    // SD 1.x/2.x: SDXL declares `addition_embed_type: "text_time"`. An older
    // SD 1.x config (pre-`0.7`-ish diffusers) omits the key entirely; a newer
    // one (SD-Turbo's, `_diffusers_version: 0.24.0.dev0`) writes it out as a
    // literal JSON `null` — `root.get` returns `Optional(.null)` for that,
    // NOT Zig's absent-key `null`, so it must be checked explicitly or a
    // Turbo config is refused as "unsupported" for a field it correctly
    // leaves empty. Any OTHER declared value is an architecture this port
    // has not built and is refused by name rather than silently treated as
    // either shape.
    if (root.get("addition_embed_type")) |v| {
        switch (v) {
            .null => cfg.has_micro_conditioning = false,
            .string => |s| {
                if (!std.mem.eql(u8, s, "text_time")) return error.UnsupportedAdditionEmbedType;
                cfg.has_micro_conditioning = true;
            },
            else => return error.UnsupportedAdditionEmbedType,
        }
    } else {
        cfg.has_micro_conditioning = false;
    }

    return cfg;
}

pub fn freeConfig(allocator: std.mem.Allocator, cfg: UnetConfig) void {
    allocator.free(cfg.block_out_channels);
    allocator.free(cfg.num_heads);
    allocator.free(cfg.transformer_layers);
    allocator.free(cfg.down_has_attn);
    allocator.free(cfg.up_has_attn);
}

// ══════════════════════════════════════════════════════════════════
// Timestep embedding
// ══════════════════════════════════════════════════════════════════

/// diffusers' `get_timestep_embedding` with `flip_sin_to_cos=True` and
/// `downscale_freq_shift=0`, computed on the CPU into `out` (length `dim`).
///
/// Host-side on purpose: it is a few hundred f32s once per forward, and the
/// alternative is four MLX ops whose only effect is to make a closed-form
/// constant harder to read. `flip_sin_to_cos` puts COS FIRST — the unflipped
/// order is the SD 1.x convention and is a real checkpoint difference, not a
/// stylistic one.
pub fn timestepEmbedding(t: f32, dim: usize, out: []f32) void {
    std.debug.assert(out.len == dim);
    const half = dim / 2;
    const max_period: f32 = 10000.0;
    for (0..half) |i| {
        const exponent = -@log(max_period) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half));
        const freq = @exp(exponent);
        const angle = t * freq;
        out[i] = @cos(angle);
        out[half + i] = @sin(angle);
    }
}

// ══════════════════════════════════════════════════════════════════
// Attention / transformer pieces
// ══════════════════════════════════════════════════════════════════

const Attention = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: c_int,

    fn deinit(self: *Attention) void {
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
    }

    /// `x` is `[1, N, inner]`. `ctx` is `x` for self-attention, or the text
    /// stream `[1, 77, cross_dim]` for cross-attention.
    fn forward(self: *const Attention, x: mlx.mlx_array, ctx: mlx.mlx_array, s: S) !mlx.mlx_array {
        const xsh = mlx.getShape(x);
        const n = xsh[1];
        const inner = xsh[2];
        const head_dim = @divExact(inner, self.heads);

        const q = try self.q.forward(x, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.k.forward(ctx, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try self.v.forward(ctx, s);
        defer _ = mlx.mlx_array_free(v);

        const ctx_n = mlx.getShape(k)[1];
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const qh = blk: {
            const r = try nn.reshape(q, &[_]c_int{ 1, n, self.heads, head_dim }, s);
            defer _ = mlx.mlx_array_free(r);
            break :blk try nn.transpose(r, &perm, s);
        };
        defer _ = mlx.mlx_array_free(qh);
        const kh = blk: {
            const r = try nn.reshape(k, &[_]c_int{ 1, ctx_n, self.heads, head_dim }, s);
            defer _ = mlx.mlx_array_free(r);
            break :blk try nn.transpose(r, &perm, s);
        };
        defer _ = mlx.mlx_array_free(kh);
        const vh = blk: {
            const r = try nn.reshape(v, &[_]c_int{ 1, ctx_n, self.heads, head_dim }, s);
            defer _ = mlx.mlx_array_free(r);
            break :blk try nn.transpose(r, &perm, s);
        };
        defer _ = mlx.mlx_array_free(vh);

        // NOT causal: the UNet's attention is bidirectional over image tokens
        // (and over text tokens on the cross path). CLIP's tower is the causal
        // one — the two live in the same pipeline and want opposite masks.
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qh, kh, vh, scale, "", null_a, null_a, false, s));

        const merged = blk: {
            const t = try nn.transpose(attn, &perm, s);
            defer _ = mlx.mlx_array_free(t);
            break :blk try nn.reshape(t, &[_]c_int{ 1, n, inner }, s);
        };
        defer _ = mlx.mlx_array_free(merged);
        return self.o.forward(merged, s);
    }
};

fn loadAttention(
    w: *const Weights,
    a: std.mem.Allocator,
    prefix: []const u8,
    heads: c_int,
    dt: mlx.mlx_dtype,
    s: S,
) !Attention {
    const kq = try nn.fmtKey(a, "{s}.to_q.weight", .{prefix});
    defer a.free(kq);
    const kk = try nn.fmtKey(a, "{s}.to_k.weight", .{prefix});
    defer a.free(kk);
    const kv = try nn.fmtKey(a, "{s}.to_v.weight", .{prefix});
    defer a.free(kv);
    const kow = try nn.fmtKey(a, "{s}.to_out.0.weight", .{prefix});
    defer a.free(kow);
    const kob = try nn.fmtKey(a, "{s}.to_out.0.bias", .{prefix});
    defer a.free(kob);

    // q/k/v carry NO bias in diffusers' attention; to_out.0 does. Loading a
    // bias that is not there is a missing-weight error, which is the good
    // direction — the reverse (dropping a real bias) is silent.
    var at = Attention{
        .q = try nn.loadLinear(w, kq, null, dt, s),
        .k = undefined,
        .v = undefined,
        .o = undefined,
        .heads = heads,
    };
    errdefer at.q.deinit();
    at.k = try nn.loadLinear(w, kk, null, dt, s);
    errdefer at.k.deinit();
    at.v = try nn.loadLinear(w, kv, null, dt, s);
    errdefer at.v.deinit();
    at.o = try nn.loadLinear(w, kow, kob, dt, s);
    return at;
}

/// diffusers `BasicTransformerBlock` — self-attn, cross-attn, GEGLU feed
/// forward, each pre-normed and residually added.
const BasicBlock = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    n3w: mlx.mlx_array,
    n3b: mlx.mlx_array,
    attn1: Attention,
    attn2: Attention,
    ff_proj: Linear,
    ff_out: Linear,

    fn deinit(self: *BasicBlock) void {
        inline for (.{ "n1w", "n1b", "n2w", "n2b", "n3w", "n3b" }) |f| {
            _ = mlx.mlx_array_free(@field(self, f));
        }
        self.attn1.deinit();
        self.attn2.deinit();
        self.ff_proj.deinit();
        self.ff_out.deinit();
    }

    fn forward(self: *const BasicBlock, x: mlx.mlx_array, ctx: mlx.mlx_array, s: S) !mlx.mlx_array {
        var h = try nn.dupA(x);
        errdefer _ = mlx.mlx_array_free(h);

        // Self-attention.
        {
            const normed = try nn.layerNorm(h, self.n1w, self.n1b, LAYER_NORM_EPS, s);
            defer _ = mlx.mlx_array_free(normed);
            const att = try self.attn1.forward(normed, normed, s);
            defer _ = mlx.mlx_array_free(att);
            nn.replace(&h, try nn.addA(h, att, s));
        }
        // Cross-attention against the text stream.
        {
            const normed = try nn.layerNorm(h, self.n2w, self.n2b, LAYER_NORM_EPS, s);
            defer _ = mlx.mlx_array_free(normed);
            const att = try self.attn2.forward(normed, ctx, s);
            defer _ = mlx.mlx_array_free(att);
            nn.replace(&h, try nn.addA(h, att, s));
        }
        // GEGLU feed-forward: one projection to 2x width, split into value and
        // gate, value * gelu(gate). The SPLIT ORDER matters and is not
        // symmetric — diffusers takes the FIRST half as the value.
        {
            const normed = try nn.layerNorm(h, self.n3w, self.n3b, LAYER_NORM_EPS, s);
            defer _ = mlx.mlx_array_free(normed);
            const proj = try self.ff_proj.forward(normed, s);
            defer _ = mlx.mlx_array_free(proj);

            const psh = mlx.getShape(proj); // [1, N, 2*inner]
            const half = @divExact(psh[2], 2);
            const val = blk: {
                const start = [_]c_int{ 0, 0, 0 };
                const stop = [_]c_int{ psh[0], psh[1], half };
                const stride = [_]c_int{ 1, 1, 1 };
                var o = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&o, proj, &start, 3, &stop, 3, &stride, 3, s));
                break :blk o;
            };
            defer _ = mlx.mlx_array_free(val);
            const gate = blk: {
                const start = [_]c_int{ 0, 0, half };
                const stop = [_]c_int{ psh[0], psh[1], psh[2] };
                const stride = [_]c_int{ 1, 1, 1 };
                var o = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&o, proj, &start, 3, &stop, 3, &stride, 3, s));
                break :blk o;
            };
            defer _ = mlx.mlx_array_free(gate);

            const g = try nn.gelu(gate, s);
            defer _ = mlx.mlx_array_free(g);
            const gated = try nn.mulA(val, g, s);
            defer _ = mlx.mlx_array_free(gated);
            const ff = try self.ff_out.forward(gated, s);
            defer _ = mlx.mlx_array_free(ff);
            nn.replace(&h, try nn.addA(h, ff, s));
        }
        return h;
    }
};

fn loadBasicBlock(
    w: *const Weights,
    a: std.mem.Allocator,
    prefix: []const u8,
    heads: c_int,
    dt: mlx.mlx_dtype,
    s: S,
) !BasicBlock {
    const K = struct {
        fn get(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, d: mlx.mlx_dtype, st: S) !mlx.mlx_array {
            const k = try nn.fmtKey(aa, "{s}.{s}", .{ p, sub });
            defer aa.free(k);
            return nn.dupWeight(ww, k, d, st);
        }
    };
    var b: BasicBlock = undefined;
    b.n1w = try K.get(w, a, prefix, "norm1.weight", dt, s);
    b.n1b = try K.get(w, a, prefix, "norm1.bias", dt, s);
    b.n2w = try K.get(w, a, prefix, "norm2.weight", dt, s);
    b.n2b = try K.get(w, a, prefix, "norm2.bias", dt, s);
    b.n3w = try K.get(w, a, prefix, "norm3.weight", dt, s);
    b.n3b = try K.get(w, a, prefix, "norm3.bias", dt, s);

    const p1 = try nn.fmtKey(a, "{s}.attn1", .{prefix});
    defer a.free(p1);
    b.attn1 = try loadAttention(w, a, p1, heads, dt, s);
    const p2 = try nn.fmtKey(a, "{s}.attn2", .{prefix});
    defer a.free(p2);
    b.attn2 = try loadAttention(w, a, p2, heads, dt, s);

    const kfw = try nn.fmtKey(a, "{s}.ff.net.0.proj.weight", .{prefix});
    defer a.free(kfw);
    const kfb = try nn.fmtKey(a, "{s}.ff.net.0.proj.bias", .{prefix});
    defer a.free(kfb);
    b.ff_proj = try nn.loadLinear(w, kfw, kfb, dt, s);
    // `net.1` is the dropout — no weights, and the index still advances.
    const kow = try nn.fmtKey(a, "{s}.ff.net.2.weight", .{prefix});
    defer a.free(kow);
    const kob = try nn.fmtKey(a, "{s}.ff.net.2.bias", .{prefix});
    defer a.free(kob);
    b.ff_out = try nn.loadLinear(w, kow, kob, dt, s);
    return b;
}

/// diffusers `Transformer2DModel` in its `use_linear_projection` spelling.
const Transformer2D = struct {
    norm_w: mlx.mlx_array,
    norm_b: mlx.mlx_array,
    proj_in: Linear,
    proj_out: Linear,
    blocks: []BasicBlock,
    groups: c_int,
    allocator: std.mem.Allocator,

    fn deinit(self: *Transformer2D) void {
        _ = mlx.mlx_array_free(self.norm_w);
        _ = mlx.mlx_array_free(self.norm_b);
        self.proj_in.deinit();
        self.proj_out.deinit();
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
    }

    /// `x` is NHWC `[1,H,W,C]`; returns the same shape.
    fn forward(self: *const Transformer2D, x: mlx.mlx_array, ctx: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        const h_ = sh[1];
        const w_ = sh[2];
        const c = sh[3];

        const normed = try nn.groupNorm(x, self.norm_w, self.norm_b, self.groups, TRANSFORMER_NORM_EPS, s);
        defer _ = mlx.mlx_array_free(normed);

        // Flatten to tokens FIRST, then project — the `use_linear_projection`
        // order. NHWC makes the flatten a pure reshape: the channel is already
        // the fastest axis, which is exactly what `permute(0,2,3,1)` achieves
        // on a NCHW tensor upstream.
        const tokens = try nn.reshape(normed, &[_]c_int{ 1, h_ * w_, c }, s);
        defer _ = mlx.mlx_array_free(tokens);
        var hh = try self.proj_in.forward(tokens, s);
        errdefer _ = mlx.mlx_array_free(hh);

        for (self.blocks) |*b| {
            const next = try b.forward(hh, ctx, s);
            nn.replace(&hh, next);
        }

        {
            const projected = try self.proj_out.forward(hh, s);
            nn.replace(&hh, projected);
        }
        const spatial = try nn.reshape(hh, &[_]c_int{ 1, h_, w_, c }, s);
        _ = mlx.mlx_array_free(hh);
        defer _ = mlx.mlx_array_free(spatial);
        return nn.addA(spatial, x, s);
    }
};

fn loadTransformer2D(
    w: *const Weights,
    a: std.mem.Allocator,
    prefix: []const u8,
    layers: u32,
    heads: c_int,
    groups: c_int,
    dt: mlx.mlx_dtype,
    s: S,
) !Transformer2D {
    const K = struct {
        fn get(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, d: mlx.mlx_dtype, st: S) !mlx.mlx_array {
            const k = try nn.fmtKey(aa, "{s}.{s}", .{ p, sub });
            defer aa.free(k);
            return nn.dupWeight(ww, k, d, st);
        }
    };
    var t: Transformer2D = undefined;
    t.allocator = a;
    t.groups = groups;
    t.norm_w = try K.get(w, a, prefix, "norm.weight", dt, s);
    t.norm_b = try K.get(w, a, prefix, "norm.bias", dt, s);

    const kiw = try nn.fmtKey(a, "{s}.proj_in.weight", .{prefix});
    defer a.free(kiw);
    const kib = try nn.fmtKey(a, "{s}.proj_in.bias", .{prefix});
    defer a.free(kib);
    t.proj_in = try nn.loadLinear(w, kiw, kib, dt, s);
    const kow = try nn.fmtKey(a, "{s}.proj_out.weight", .{prefix});
    defer a.free(kow);
    const kob = try nn.fmtKey(a, "{s}.proj_out.bias", .{prefix});
    defer a.free(kob);
    t.proj_out = try nn.loadLinear(w, kow, kob, dt, s);

    t.blocks = try a.alloc(BasicBlock, layers);
    errdefer a.free(t.blocks);
    for (t.blocks, 0..) |*b, i| {
        const bp = try nn.fmtKey(a, "{s}.transformer_blocks.{d}", .{ prefix, i });
        defer a.free(bp);
        b.* = try loadBasicBlock(w, a, bp, heads, dt, s);
    }
    return t;
}

// ══════════════════════════════════════════════════════════════════
// Blocks
// ══════════════════════════════════════════════════════════════════

const DownBlock = struct {
    resnets: []Resnet,
    attns: ?[]Transformer2D,
    down_w: ?mlx.mlx_array,
    down_b: ?mlx.mlx_array,
    allocator: std.mem.Allocator,

    fn deinit(self: *DownBlock) void {
        for (self.resnets) |*r| r.deinit();
        self.allocator.free(self.resnets);
        if (self.attns) |as_| {
            for (as_) |*t| t.deinit();
            self.allocator.free(as_);
        }
        if (self.down_w) |x| _ = mlx.mlx_array_free(x);
        if (self.down_b) |x| _ = mlx.mlx_array_free(x);
    }
};

const UpBlock = struct {
    resnets: []Resnet,
    attns: ?[]Transformer2D,
    up_w: ?mlx.mlx_array,
    up_b: ?mlx.mlx_array,
    allocator: std.mem.Allocator,

    fn deinit(self: *UpBlock) void {
        for (self.resnets) |*r| r.deinit();
        self.allocator.free(self.resnets);
        if (self.attns) |as_| {
            for (as_) |*t| t.deinit();
            self.allocator.free(as_);
        }
        if (self.up_w) |x| _ = mlx.mlx_array_free(x);
        if (self.up_b) |x| _ = mlx.mlx_array_free(x);
    }
};

const MidBlock = struct {
    resnets: [2]Resnet,
    attn: Transformer2D,

    fn deinit(self: *MidBlock) void {
        for (&self.resnets) |*r| r.deinit();
        self.attn.deinit();
    }
};

// ══════════════════════════════════════════════════════════════════
// The model
// ══════════════════════════════════════════════════════════════════

pub const Unet = struct {
    allocator: std.mem.Allocator,
    s: S,
    dtype: mlx.mlx_dtype,
    cfg: UnetConfig,
    owns_cfg: bool,

    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    time_1: Linear,
    time_2: Linear,
    /// Null on SD 1.x (`cfg.has_micro_conditioning == false`) — no checkpoint
    /// tensors to bind.
    add_1: ?Linear,
    add_2: ?Linear,
    down: []DownBlock,
    mid: MidBlock,
    up: []UpBlock,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Unet) void {
        _ = mlx.mlx_array_free(self.conv_in_w);
        _ = mlx.mlx_array_free(self.conv_in_b);
        self.time_1.deinit();
        self.time_2.deinit();
        if (self.add_1) |*l| l.deinit();
        if (self.add_2) |*l| l.deinit();
        for (self.down) |*d| d.deinit();
        self.allocator.free(self.down);
        self.mid.deinit();
        for (self.up) |*u| u.deinit();
        self.allocator.free(self.up);
        _ = mlx.mlx_array_free(self.norm_out_w);
        _ = mlx.mlx_array_free(self.norm_out_b);
        _ = mlx.mlx_array_free(self.conv_out_w);
        _ = mlx.mlx_array_free(self.conv_out_b);
        if (self.owns_cfg) freeConfig(self.allocator, self.cfg);
    }

    /// Build the time (+ SDXL micro-conditioning) embedding, `[1, timeEmbedDim]`.
    ///
    /// `text_embeds`/`time_ids` are SDXL-only: `text_embeds` is bigG's pooled
    /// vector and `time_ids` the six-number crop descriptor, sinusoidally
    /// expanded and concatenated. Null on `!cfg.has_micro_conditioning`
    /// (SD 1.x has no augmentation branch at all) — passing one there, or
    /// omitting one where the checkpoint expects it, is a programmer error
    /// (`error.MissingTextEmbeds`/`error.MissingTimeIds`), not a silent
    /// fallback, because getting it wrong does not error downstream; it
    /// shifts framing and composition, which just reads as a bad checkpoint.
    fn buildEmbedding(
        self: *const Unet,
        timestep: f32,
        text_embeds: ?mlx.mlx_array,
        time_ids: ?*const sdxl.TimeIds,
    ) !mlx.mlx_array {
        const s = self.s;
        const a = self.allocator;
        const t_dim = self.cfg.block_out_channels[0];

        // Sinusoidal timestep -> time_embedding MLP.
        const t_buf = try a.alloc(f32, t_dim);
        defer a.free(t_buf);
        timestepEmbedding(timestep, t_dim, t_buf);
        const t_shape = [_]c_int{ 1, @intCast(t_dim) };
        const t_arr = mlx.mlx_array_new_data(t_buf.ptr, &t_shape, 2, .float32);
        defer _ = mlx.mlx_array_free(t_arr);
        const t_cast = try nn.astype(t_arr, self.dtype, s);
        defer _ = mlx.mlx_array_free(t_cast);

        const emb = blk: {
            const l1 = try self.time_1.forward(t_cast, s);
            defer _ = mlx.mlx_array_free(l1);
            const act = try nn.silu(l1, s);
            defer _ = mlx.mlx_array_free(act);
            break :blk try self.time_2.forward(act, s);
        };
        errdefer _ = mlx.mlx_array_free(emb);

        if (!self.cfg.has_micro_conditioning) return emb;

        const te_in = text_embeds orelse return error.MissingTextEmbeds;
        const ti_in = time_ids orelse return error.MissingTimeIds;
        const add_1 = self.add_1 orelse return error.MissingAddEmbedding;
        const add_2 = self.add_2 orelse return error.MissingAddEmbedding;

        // Micro-conditioning: each of the six ids gets its OWN sinusoidal
        // expansion, and they are concatenated in order — the same flatten
        // diffusers does via `add_time_proj(time_ids.flatten())`.
        const add_dim = self.cfg.addition_time_embed_dim;
        const ids_buf = try a.alloc(f32, add_dim * ti_in.len);
        defer a.free(ids_buf);
        for (ti_in, 0..) |v, i| {
            timestepEmbedding(v, add_dim, ids_buf[i * add_dim .. (i + 1) * add_dim]);
        }
        const ids_shape = [_]c_int{ 1, @intCast(add_dim * ti_in.len) };
        const ids_arr = mlx.mlx_array_new_data(ids_buf.ptr, &ids_shape, 2, .float32);
        defer _ = mlx.mlx_array_free(ids_arr);
        const ids_cast = try nn.astype(ids_arr, self.dtype, s);
        defer _ = mlx.mlx_array_free(ids_cast);

        // Order is [pooled_text, time_ids] — the reverse loads and is wrong.
        const te = try nn.astype(te_in, self.dtype, s);
        defer _ = mlx.mlx_array_free(te);
        const cat = try nn.concat(&[_]mlx.mlx_array{ te, ids_cast }, 1, s);
        defer _ = mlx.mlx_array_free(cat);

        const aug = blk: {
            const l1 = try add_1.forward(cat, s);
            defer _ = mlx.mlx_array_free(l1);
            const act = try nn.silu(l1, s);
            defer _ = mlx.mlx_array_free(act);
            break :blk try add_2.forward(act, s);
        };
        defer _ = mlx.mlx_array_free(aug);

        const summed = try nn.addA(emb, aug, s);
        _ = mlx.mlx_array_free(emb);
        return summed;
    }

    /// One denoising forward.
    ///
    /// `sample` is NCHW `[1, 4, h, w]`, `ctx` the text stream (`[1, 77, 2048]`
    /// on SDXL, `[1, 77, 768]` on SD 1.x — `cross_attention_dim` is read from
    /// the checkpoint, not hardcoded). `text_embeds`/`time_ids` are SDXL's
    /// pooled-`[1,1280]`/six-number micro-conditioning; null on SD 1.x, whose
    /// `has_micro_conditioning` config is false. Returns NCHW `[1, 4, h, w]`.
    pub fn forward(
        self: *const Unet,
        sample: mlx.mlx_array,
        timestep: f32,
        ctx: mlx.mlx_array,
        text_embeds: ?mlx.mlx_array,
        time_ids: ?*const sdxl.TimeIds,
    ) !mlx.mlx_array {
        const s = self.s;
        const a = self.allocator;

        const emb = try self.buildEmbedding(timestep, text_embeds, time_ids);
        defer _ = mlx.mlx_array_free(emb);

        const ctx_cast = try nn.astype(ctx, self.dtype, s);
        defer _ = mlx.mlx_array_free(ctx_cast);

        const sample_cast = try nn.astype(sample, self.dtype, s);
        defer _ = mlx.mlx_array_free(sample_cast);
        const nhwc = try nn.nchwToNhwc(sample_cast, s);
        defer _ = mlx.mlx_array_free(nhwc);

        var h = try nn.conv2d(nhwc, self.conv_in_w, self.conv_in_b, 1, s);
        errdefer _ = mlx.mlx_array_free(h);

        // ── Down path. `conv_in`'s output is the FIRST skip; the up path pops
        // nine and only eight come from the blocks themselves.
        var skips: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (skips.items) |sk| _ = mlx.mlx_array_free(sk);
            skips.deinit(a);
        }
        try skips.append(a, try nn.dupA(h));

        for (self.down, 0..) |*blk, bi| {
            for (blk.resnets, 0..) |*r, ri| {
                nn.replace(&h, try r.forward(h, emb, s));
                if (blk.attns) |as_| {
                    nn.replace(&h, try as_[ri].forward(h, ctx_cast, s));
                }
                try skips.append(a, try nn.dupA(h));
            }
            if (blk.down_w) |dw| {
                nn.replace(&h, try nn.conv2dStride2(h, dw, blk.down_b.?, s));
                try skips.append(a, try nn.dupA(h));
            }
            _ = bi;
        }

        // ── Mid.
        nn.replace(&h, try self.mid.resnets[0].forward(h, emb, s));
        nn.replace(&h, try self.mid.attn.forward(h, ctx_cast, s));
        nn.replace(&h, try self.mid.resnets[1].forward(h, emb, s));

        // ── Up path. Each resnet consumes one skip, concatenated on the
        // CHANNEL axis (last, in NHWC) before the resnet rather than after.
        for (self.up) |*blk| {
            for (blk.resnets, 0..) |*r, ri| {
                const skip = skips.pop() orelse return error.SkipUnderflow;
                defer _ = mlx.mlx_array_free(skip);
                const joined = try nn.concat(&[_]mlx.mlx_array{ h, skip }, 3, s);
                nn.replace(&h, joined);
                nn.replace(&h, try r.forward(h, emb, s));
                if (blk.attns) |as_| {
                    nn.replace(&h, try as_[ri].forward(h, ctx_cast, s));
                }
            }
            if (blk.up_w) |uw| {
                nn.replace(&h, try nn.upsample2x(h, uw, blk.up_b.?, s));
            }
        }
        if (skips.items.len != 0) return error.SkipLeftover;

        // ── Out.
        {
            const n = try nn.groupNorm(h, self.norm_out_w, self.norm_out_b, @intCast(self.cfg.norm_num_groups), self.cfg.norm_eps, s);
            nn.replace(&h, n);
        }
        nn.replace(&h, try nn.silu(h, s));
        nn.replace(&h, try nn.conv2d(h, self.conv_out_w, self.conv_out_b, 1, s));

        defer _ = mlx.mlx_array_free(h);
        return nn.nhwcToNchw(h, s);
    }
};

/// Load the UNet from `<model_dir>/unet`, reading its own `config.json`.
pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: S,
    model_dir: []const u8,
    dtype: mlx.mlx_dtype,
) !Unet {
    const dir = try nn.fmtKey(allocator, "{s}/unet", .{model_dir});
    defer allocator.free(dir);

    const cfg_path = try nn.fmtKey(allocator, "{s}/config.json", .{dir});
    defer allocator.free(cfg_path);
    // `openFileAbsolute` ASSERTS its path is absolute, and a failed assert is
    // ReleaseFast UB that can miscompile the caller (the openDirAbsolute
    // gotcha). `model_dir` reaches here from a request, so guard first.
    if (cfg_path.len == 0 or !std.fs.path.isAbsolute(cfg_path)) return error.MissingUnetConfig;
    const cfg_file = std.Io.Dir.openFileAbsolute(io, cfg_path, .{}) catch |e| {
        log.err("[sdxl] cannot open {s}: {s}\n", .{ cfg_path, @errorName(e) });
        return error.MissingUnetConfig;
    };
    defer cfg_file.close(io);
    var cfg_rb: [4096]u8 = undefined;
    var cfg_rs = cfg_file.reader(io, &cfg_rb);
    const cfg_bytes = cfg_rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch {
        return error.MissingUnetConfig;
    };
    defer allocator.free(cfg_bytes);
    const cfg = try parseConfig(allocator, cfg_bytes);
    errdefer freeConfig(allocator, cfg);

    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();

    return loadFromWeights(allocator, s, &w, cfg, dtype, true);
}

/// The weight-binding half, split out so tests can drive it from an already
/// loaded `Weights` without a second read of a 5 GB file.
pub fn loadFromWeights(
    allocator: std.mem.Allocator,
    s: S,
    w: *Weights,
    cfg: UnetConfig,
    dtype: mlx.mlx_dtype,
    owns_cfg: bool,
) !Unet {
    var u: Unet = undefined;
    u.allocator = allocator;
    u.s = s;
    u.dtype = dtype;
    u.cfg = cfg;
    u.owns_cfg = owns_cfg;

    const groups: c_int = @intCast(cfg.norm_num_groups);

    u.conv_in_w = try nn.loadConvWeight(w, "conv_in.weight", dtype, s);
    u.conv_in_b = try nn.dupWeight(w, "conv_in.bias", dtype, s);
    u.time_1 = try nn.loadLinear(w, "time_embedding.linear_1.weight", "time_embedding.linear_1.bias", dtype, s);
    u.time_2 = try nn.loadLinear(w, "time_embedding.linear_2.weight", "time_embedding.linear_2.bias", dtype, s);
    if (cfg.has_micro_conditioning) {
        u.add_1 = try nn.loadLinear(w, "add_embedding.linear_1.weight", "add_embedding.linear_1.bias", dtype, s);
        u.add_2 = try nn.loadLinear(w, "add_embedding.linear_2.weight", "add_embedding.linear_2.bias", dtype, s);
    } else {
        u.add_1 = null;
        u.add_2 = null;
    }

    // ── Down blocks.
    u.down = try allocator.alloc(DownBlock, cfg.down_has_attn.len);
    errdefer allocator.free(u.down);
    for (u.down, 0..) |*blk, bi| {
        blk.allocator = allocator;
        blk.attns = null;
        blk.down_w = null;
        blk.down_b = null;
        blk.resnets = try allocator.alloc(Resnet, cfg.layers_per_block);
        for (blk.resnets, 0..) |*r, ri| {
            const pfx = try nn.fmtKey(allocator, "down_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            r.* = try nn.loadResnet(w, allocator, pfx, true, cfg.norm_eps, dtype, s);
        }
        if (cfg.down_has_attn[bi]) {
            const as_ = try allocator.alloc(Transformer2D, cfg.layers_per_block);
            for (as_, 0..) |*t, ai| {
                const pfx = try nn.fmtKey(allocator, "down_blocks.{d}.attentions.{d}", .{ bi, ai });
                defer allocator.free(pfx);
                t.* = try loadTransformer2D(w, allocator, pfx, cfg.transformer_layers[bi], @intCast(cfg.num_heads[bi]), groups, dtype, s);
            }
            blk.attns = as_;
        }
        // Every stage but the last downsamples.
        if (bi + 1 < cfg.down_has_attn.len) {
            const kw = try nn.fmtKey(allocator, "down_blocks.{d}.downsamplers.0.conv.weight", .{bi});
            defer allocator.free(kw);
            const kb = try nn.fmtKey(allocator, "down_blocks.{d}.downsamplers.0.conv.bias", .{bi});
            defer allocator.free(kb);
            blk.down_w = try nn.loadConvWeight(w, kw, dtype, s);
            blk.down_b = try nn.dupWeight(w, kb, dtype, s);
        }
    }

    // ── Mid block: resnet, attention, resnet.
    const deepest = cfg.stages() - 1;
    u.mid.resnets[0] = try nn.loadResnet(w, allocator, "mid_block.resnets.0", true, cfg.norm_eps, dtype, s);
    u.mid.resnets[1] = try nn.loadResnet(w, allocator, "mid_block.resnets.1", true, cfg.norm_eps, dtype, s);
    u.mid.attn = try loadTransformer2D(w, allocator, "mid_block.attentions.0", cfg.transformer_layers[deepest], @intCast(cfg.num_heads[deepest]), groups, dtype, s);

    // ── Up blocks. One MORE resnet per block than the down side, because the
    // up path consumes the extra `conv_in` skip.
    u.up = try allocator.alloc(UpBlock, cfg.up_has_attn.len);
    errdefer allocator.free(u.up);
    const up_layers = cfg.layers_per_block + 1;
    for (u.up, 0..) |*blk, bi| {
        blk.allocator = allocator;
        blk.attns = null;
        blk.up_w = null;
        blk.up_b = null;
        // Up stage `bi` mirrors down stage `stages-1-bi`.
        const mirrored = cfg.stages() - 1 - bi;
        blk.resnets = try allocator.alloc(Resnet, up_layers);
        for (blk.resnets, 0..) |*r, ri| {
            const pfx = try nn.fmtKey(allocator, "up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            r.* = try nn.loadResnet(w, allocator, pfx, true, cfg.norm_eps, dtype, s);
        }
        if (cfg.up_has_attn[bi]) {
            const as_ = try allocator.alloc(Transformer2D, up_layers);
            for (as_, 0..) |*t, ai| {
                const pfx = try nn.fmtKey(allocator, "up_blocks.{d}.attentions.{d}", .{ bi, ai });
                defer allocator.free(pfx);
                t.* = try loadTransformer2D(w, allocator, pfx, cfg.transformer_layers[mirrored], @intCast(cfg.num_heads[mirrored]), groups, dtype, s);
            }
            blk.attns = as_;
        }
        if (bi + 1 < cfg.up_has_attn.len) {
            const kw = try nn.fmtKey(allocator, "up_blocks.{d}.upsamplers.0.conv.weight", .{bi});
            defer allocator.free(kw);
            const kb = try nn.fmtKey(allocator, "up_blocks.{d}.upsamplers.0.conv.bias", .{bi});
            defer allocator.free(kb);
            blk.up_w = try nn.loadConvWeight(w, kw, dtype, s);
            blk.up_b = try nn.dupWeight(w, kb, dtype, s);
        }
    }

    u.norm_out_w = try nn.dupWeight(w, "conv_norm_out.weight", dtype, s);
    u.norm_out_b = try nn.dupWeight(w, "conv_norm_out.bias", dtype, s);
    u.conv_out_w = try nn.loadConvWeight(w, "conv_out.weight", dtype, s);
    u.conv_out_b = try nn.dupWeight(w, "conv_out.bias", dtype, s);

    log.info("[sdxl] loaded unet: stages={d} widths={any} tlayers={any} heads={any}\n", .{
        cfg.stages(), cfg.block_out_channels, cfg.transformer_layers, cfg.num_heads,
    });
    return u;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "sdxl unet: config parse reads the real checkpoint contract" {
    const a = testing.allocator;
    const json =
        \\{"_class_name":"UNet2DConditionModel","act_fn":"silu",
        \\"addition_embed_type":"text_time","addition_time_embed_dim":256,
        \\"attention_head_dim":[5,10,20],"block_out_channels":[320,640,1280],
        \\"cross_attention_dim":2048,
        \\"down_block_types":["DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"],
        \\"in_channels":4,"layers_per_block":2,"norm_eps":1e-05,"norm_num_groups":32,
        \\"out_channels":4,"projection_class_embeddings_input_dim":2816,
        \\"transformer_layers_per_block":[1,2,10],
        \\"up_block_types":["CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D"],
        \\"use_linear_projection":true}
    ;
    const cfg = try parseConfig(a, json);
    defer freeConfig(a, cfg);

    try testing.expectEqualSlices(u32, &[_]u32{ 320, 640, 1280 }, cfg.block_out_channels);
    // The head COUNT, not a head dim: 320/5 == 640/10 == 1280/20 == 64.
    try testing.expectEqualSlices(u32, &[_]u32{ 5, 10, 20 }, cfg.num_heads);
    for (cfg.block_out_channels, cfg.num_heads) |width, heads| {
        try testing.expectEqual(@as(u32, 64), width / heads);
    }
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 10 }, cfg.transformer_layers);
    try testing.expectEqualSlices(bool, &[_]bool{ false, true, true }, cfg.down_has_attn);
    try testing.expectEqualSlices(bool, &[_]bool{ true, true, false }, cfg.up_has_attn);
    try testing.expectEqual(@as(u32, 1280), cfg.timeEmbedDim());
    // 2816 = pooled 1280 + 6 ids x 256.
    try testing.expectEqual(@as(u32, 2816), cfg.add_embed_input_dim);
    try testing.expectEqual(@as(u32, 1280 + 6 * 256), cfg.add_embed_input_dim);
    try testing.expect(cfg.has_micro_conditioning);
}

test "sdxl unet: an SD 1.x config (no addition_embed_type, no transformer_layers_per_block) parses with micro-conditioning off" {
    const a = testing.allocator;
    // SD 1.5's real `unet/config.json` shape, trimmed to the keys the parser
    // reads: no `addition_embed_type` at all (the key CLIP-only SD 1.x never
    // declares), no `transformer_layers_per_block` either (SDXL is the family
    // that varies per stage and states it; SD 1.x has never needed it, and
    // diffusers defaults the field to 1 when absent — live 2026-09-05, a
    // real SD 1.5 pull threw `MissingTransformerLayers` because the parser
    // treated the key as mandatory), scalar `attention_head_dim` (8, not
    // per-stage), and a fourth attention-free stage the way diffusers' own
    // config ships it.
    const json =
        \\{"_class_name":"UNet2DConditionModel","act_fn":"silu",
        \\"attention_head_dim":8,"block_out_channels":[320,640,1280,1280],
        \\"cross_attention_dim":768,
        \\"down_block_types":["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
        \\"in_channels":4,"layers_per_block":2,"norm_eps":1e-05,"norm_num_groups":32,
        \\"out_channels":4,
        \\"up_block_types":["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"],
        \\"use_linear_projection":true}
    ;
    const cfg = try parseConfig(a, json);
    defer freeConfig(a, cfg);

    try testing.expect(!cfg.has_micro_conditioning);
    try testing.expectEqual(@as(u32, 768), cfg.cross_attention_dim);
    try testing.expectEqualSlices(u32, &[_]u32{ 320, 640, 1280, 1280 }, cfg.block_out_channels);
    // Scalar attention_head_dim broadcasts to every stage.
    try testing.expectEqualSlices(u32, &[_]u32{ 8, 8, 8, 8 }, cfg.num_heads);
    // Absent transformer_layers_per_block defaults to 1 at every stage, the
    // same value diffusers' own UNet2DConditionModel defaults it to.
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 1, 1, 1 }, cfg.transformer_layers);
    try testing.expectEqualSlices(bool, &[_]bool{ true, true, true, false }, cfg.down_has_attn);
    try testing.expectEqualSlices(bool, &[_]bool{ false, true, true, true }, cfg.up_has_attn);
}

test "sdxl unet: addition_embed_type as a literal JSON null also means no micro-conditioning" {
    // stabilityai/sd-turbo/unet/config.json, real bytes (fetched and verified
    // against the live repo) — a NEWER diffusers version than SD 1.5's, which
    // writes every declared-but-unset field out as `null` rather than
    // omitting the key. `root.get` returns `Optional(.null)` for this, which
    // is NOT Zig's absent-key `null` — a parser that only checks `if
    // (root.get(...)) |v|` treats it as present-and-wrong and refuses a
    // checkpoint that correctly declares no micro-conditioning.
    const a = testing.allocator;
    const json =
        \\{"_class_name":"UNet2DConditionModel","act_fn":"silu",
        \\"addition_embed_type":null,"addition_time_embed_dim":null,
        \\"attention_head_dim":[5,10,20,20],"block_out_channels":[320,640,1280,1280],
        \\"cross_attention_dim":1024,
        \\"down_block_types":["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
        \\"in_channels":4,"layers_per_block":2,"norm_eps":1e-05,"norm_num_groups":32,
        \\"out_channels":4,"projection_class_embeddings_input_dim":null,
        \\"transformer_layers_per_block":1,
        \\"up_block_types":["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"],
        \\"use_linear_projection":true}
    ;
    const cfg = try parseConfig(a, json);
    defer freeConfig(a, cfg);

    try testing.expect(!cfg.has_micro_conditioning);
    try testing.expectEqual(@as(u32, 1024), cfg.cross_attention_dim);
    try testing.expectEqualSlices(bool, &[_]bool{ true, true, true, false }, cfg.down_has_attn);
    try testing.expectEqualSlices(bool, &[_]bool{ false, true, true, true }, cfg.up_has_attn);
}

test "sdxl unet: an unsupported projection or embed type is refused by name" {
    const a = testing.allocator;
    const base =
        \\{"addition_embed_type":"text_time","attention_head_dim":[5,10,20],
        \\"block_out_channels":[320,640,1280],
        \\"down_block_types":["DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"],
        \\"transformer_layers_per_block":[1,2,10],
        \\"up_block_types":["CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D"],
    ;
    // The conv-projection spelling is a DIFFERENT forward, not a variant.
    {
        const json = try std.fmt.allocPrint(a, "{s}\"use_linear_projection\":false}}", .{base});
        defer a.free(json);
        try testing.expectError(error.UnsupportedProjection, parseConfig(a, json));
    }
    // SD 2.x's conditioning has no micro-conditioning vector at all.
    {
        const json = try std.fmt.allocPrint(a, "{s}\"use_linear_projection\":true}}", .{base});
        defer a.free(json);
        const cfg = try parseConfig(a, json);
        freeConfig(a, cfg);
    }
    {
        const bad =
            \\{"addition_embed_type":"text","attention_head_dim":[5],
            \\"block_out_channels":[320],"down_block_types":["DownBlock2D"],
            \\"transformer_layers_per_block":[1],"up_block_types":["UpBlock2D"],
            \\"use_linear_projection":true}
        ;
        try testing.expectError(error.UnsupportedAdditionEmbedType, parseConfig(a, bad));
    }
    // An architecture we have not built is named, never guessed at.
    {
        const bad =
            \\{"addition_embed_type":"text_time","attention_head_dim":[5],
            \\"block_out_channels":[320],"down_block_types":["KAttentionBlock"],
            \\"transformer_layers_per_block":[1],"up_block_types":["UpBlock2D"],
            \\"use_linear_projection":true}
        ;
        try testing.expectError(error.UnsupportedBlockType, parseConfig(a, bad));
    }
}

// Numerical PARITY against diffusers' own UNet2DConditionModel. The fixture is
// generated by `tests/dump_sdxl_unet_fixtures.py` over the SAME inputs this
// test reads back out of it, in float32 on CPU.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//   SDXL_UNET_FIXTURE=~/.mlx-serve/staging/sdxl_unet_fixture.safetensors \
//     zig build test -Dtest-filter="sdxl unet parity"
//
// Runs fp16 by default — the width the model actually serves at — so the bar
// is fp16-appropriate. `SDXL_UNET_F32=1` reruns in float32, which is how a
// failure gets split into "wrong math" and "accumulated precision": a logic
// error does not improve when the dtype widens.
test "sdxl unet parity: the forward matches diffusers" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("SDXL_UNET_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    const want_f32 = std.c.getenv("SDXL_UNET_F32") != null;
    const dtype: mlx.mlx_dtype = if (want_f32) .float32 else .float16;

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();

    // Every input comes FROM the fixture, so the two sides cannot describe
    // different forwards.
    const sample = fx.get("in.sample") orelse return error.MissingFixtureSample;
    const ctx = fx.get("in.encoder_hidden_states") orelse return error.MissingFixtureCtx;
    const text_embeds = fx.get("in.text_embeds") orelse return error.MissingFixtureEmbeds;

    const tid_arr = fx.get("in.time_ids") orelse return error.MissingFixtureTimeIds;
    _ = mlx.mlx_array_eval(tid_arr);
    const tid_ptr = mlx.mlx_array_data_float32(tid_arr) orelse return error.BadFixtureTimeIds;
    var time_ids: sdxl.TimeIds = undefined;
    for (&time_ids, 0..) |*v, i| v.* = tid_ptr[i];

    const ts_arr = fx.get("in.timestep") orelse return error.MissingFixtureTimestep;
    _ = mlx.mlx_array_eval(ts_arr);
    const timestep = (mlx.mlx_array_data_float32(ts_arr) orelse return error.BadFixtureTimestep)[0];

    var unet = try load(io, a, s, dir, dtype);
    defer unet.deinit();

    const out = try unet.forward(sample, timestep, ctx, text_embeds, &time_ids);
    defer _ = mlx.mlx_array_free(out);

    const ref = fx.get("out.noise_pred") orelse return error.MissingFixtureOutput;
    _ = mlx.mlx_array_eval(out);
    try testing.expectEqualSlices(c_int, mlx.getShape(ref), mlx.getShape(out));
    try expectClose("unet", "noise_pred", out, ref, s);
}

// The same forward, driven by REAL text conditioning instead of random
// `encoder_hidden_states`.
//
// This is not redundant with the test above, and the difference is the whole
// point: with a random context the cross-attention has nothing coherent to
// attend to, so its contribution is small and a defect on that path barely
// moves the output. Real embeddings put the cross-attention in charge of the
// image. A UNet fixture built only from noise scores ~1.0 either way.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//   SDXL_UNET_REAL_FIXTURE=~/.mlx-serve/staging/sdxl_unet_real_fixture.safetensors \
//     zig build test -Dtest-filter="sdxl unet real"
test "sdxl unet real conditioning: cross-attention matches diffusers" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("SDXL_UNET_REAL_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    const want_f32 = std.c.getenv("SDXL_UNET_F32") != null;
    const dtype: mlx.mlx_dtype = if (want_f32) .float32 else .float16;

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();

    const sample = fx.get("in.sample") orelse return error.MissingFixtureSample;
    const ctx = fx.get("in.ctx") orelse return error.MissingFixtureCtx;
    const pooled = fx.get("in.pooled") orelse return error.MissingFixturePooled;

    const tid_arr = fx.get("in.time_ids") orelse return error.MissingFixtureTimeIds;
    _ = mlx.mlx_array_eval(tid_arr);
    const tid_ptr = mlx.mlx_array_data_float32(tid_arr) orelse return error.BadFixtureTimeIds;
    var time_ids: sdxl.TimeIds = undefined;
    for (&time_ids, 0..) |*v, i| v.* = tid_ptr[i];

    const ts_arr = fx.get("in.timestep") orelse return error.MissingFixtureTimestep;
    _ = mlx.mlx_array_eval(ts_arr);
    const timestep = (mlx.mlx_array_data_float32(ts_arr) orelse return error.BadFixtureTimestep)[0];

    var unet = try load(io, a, s, dir, dtype);
    defer unet.deinit();

    const out = try unet.forward(sample, timestep, ctx, pooled, &time_ids);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);

    const ref = fx.get("out.eps") orelse return error.MissingFixtureOutput;
    try testing.expectEqualSlices(c_int, mlx.getShape(ref), mlx.getShape(out));
    try expectClose("unet", "eps (real cond)", out, ref, s);
}

/// Cosine + rms_ratio, shared by the parity tests in this file.
///
/// The rms_ratio half is not decoration: a cosine cannot see a scale error,
/// and the UNet's output is integrated by the sampler over many steps, so a
/// uniformly scaled epsilon prediction scores a perfect cosine here and
/// produces a washed or blown-out image.
fn expectClose(who: []const u8, what: []const u8, got: mlx.mlx_array, ref: mlx.mlx_array, s: S) !void {
    const g_arr = try flatF32(got, s);
    defer _ = mlx.mlx_array_free(g_arr);
    const r_arr = try flatF32(ref, s);
    defer _ = mlx.mlx_array_free(r_arr);

    const n = mlx.mlx_array_size(g_arr);
    try testing.expectEqual(n, mlx.mlx_array_size(r_arr));
    const g = mlx.mlx_array_data_float32(g_arr).?;
    const r = mlx.mlx_array_data_float32(r_arr).?;

    var dot: f64 = 0;
    var ng: f64 = 0;
    var nr: f64 = 0;
    var max_abs: f64 = 0;
    for (0..n) |i| {
        const gv: f64 = g[i];
        const rv: f64 = r[i];
        // Finiteness FIRST: `NaN > bar` is false, so an all-NaN tensor scores
        // zero and reads as a mismatch rather than as the crash it is.
        try testing.expect(std.math.isFinite(gv));
        dot += gv * rv;
        ng += gv * gv;
        nr += rv * rv;
        max_abs = @max(max_abs, @abs(gv - rv));
    }
    const cos = dot / (@sqrt(ng) * @sqrt(nr));
    const rms_ratio = @sqrt(ng) / @sqrt(nr);
    std.debug.print("[sdxl-parity] {s} {s}: cos={d:.6} rms_ratio={d:.6} max_abs={d:.5}\n", .{ who, what, cos, rms_ratio, max_abs });

    try testing.expect(cos > 0.999);
    try testing.expect(rms_ratio > 0.99 and rms_ratio < 1.01);
}

fn flatF32(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var f = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    const n: c_int = @intCast(mlx.mlx_array_size(x));
    const shape = [_]c_int{n};
    var flat = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&flat, f, &shape, 1, s));
    _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_array_eval(flat);
    return flat;
}

test "sdxl unet: timestep embedding puts cos first and is bounded" {
    var buf: [320]f32 = undefined;
    timestepEmbedding(501.0, 320, &buf);
    // flip_sin_to_cos: the FIRST half is cosine, so index 0 is cos(t*1) = cos(t)
    // at frequency 1. The unflipped SD 1.x order would put sin there.
    try testing.expectApproxEqAbs(@cos(@as(f32, 501.0)), buf[0], 1e-5);
    try testing.expectApproxEqAbs(@sin(@as(f32, 501.0)), buf[160], 1e-5);
    // The highest-index frequency is the slowest; at i = half-1 it is nearly
    // 1/10000, so cos is close to 1 and sin close to t/10000.
    try testing.expect(buf[159] > 0.99);
    for (buf) |v| try testing.expect(v >= -1.0001 and v <= 1.0001);
    // t = 0 is the degenerate case the fixture deliberately avoids.
    var zero: [320]f32 = undefined;
    timestepEmbedding(0.0, 320, &zero);
    for (0..160) |i| try testing.expectApproxEqAbs(@as(f32, 1.0), zero[i], 1e-6);
    for (160..320) |i| try testing.expectApproxEqAbs(@as(f32, 0.0), zero[i], 1e-6);
}

// ════════════════════════════════════════════════════════════════════════
// Runtime LoRA
// ════════════════════════════════════════════════════════════════════════

/// Install a LoRA stack on every linear the adapters name, and report how many
/// (module, adapter) attachments landed.
///
/// The module keys built here are the FLAT diffusers names
/// (`down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q`), which is
/// exactly what `lora.canonicalizeSdxl` translates a Kohya LDM key into. The
/// two must agree literally — a canonicalization that lands on a name nothing
/// queries is indistinguishable, from the outside, from an adapter that simply
/// did not match, so the count returned here is the only observable.
///
/// SCOPE: attention and feed-forward linears plus the transformer's
/// `proj_in`/`proj_out`. Resnet convolutions are deliberately NOT covered —
/// LoRA here rides `nn.Linear`, and a conv adapter would need its own path.
/// Every SDXL adapter in circulation trains this set (verified against
/// nerijs/pixel-art-xl: 722 modules, all of them in it), and a file that
/// carries resnet keys simply matches fewer modules rather than misapplying.
pub fn attachLora(u: *Unet, stack: *const lora_mod.Stack) u32 {
    detachLora(u);
    var matched: u32 = 0;
    var kbuf: [256]u8 = undefined;
    var rbuf: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined;

    const Bind = struct {
        fn one(st: *const lora_mod.Stack, key: []const u8, lin: *nn.Linear, rb: *[lora_mod.MAX_LORAS]lora_mod.Ref, acc: *u32) void {
            const refs = st.findAll(key, rb);
            if (refs.len == 0) return;
            lin.setLoraRefs(refs);
            acc.* += @intCast(refs.len);
        }

        fn transformer(st: *const lora_mod.Stack, prefix: []const u8, t: *Transformer2D, kb: *[256]u8, rb: *[lora_mod.MAX_LORAS]lora_mod.Ref, acc: *u32) void {
            {
                const k = std.fmt.bufPrint(kb, "{s}_proj_in", .{prefix}) catch return;
                one(st, k, &t.proj_in, rb, acc);
            }
            {
                const k = std.fmt.bufPrint(kb, "{s}_proj_out", .{prefix}) catch return;
                one(st, k, &t.proj_out, rb, acc);
            }
            for (t.blocks, 0..) |*b, bi| {
                const leaves = .{
                    .{ "attn1_to_q", &b.attn1.q },    .{ "attn1_to_k", &b.attn1.k },
                    .{ "attn1_to_v", &b.attn1.v },    .{ "attn1_to_out", &b.attn1.o },
                    .{ "attn2_to_q", &b.attn2.q },    .{ "attn2_to_k", &b.attn2.k },
                    .{ "attn2_to_v", &b.attn2.v },    .{ "attn2_to_out", &b.attn2.o },
                    .{ "ff_net_0_proj", &b.ff_proj }, .{ "ff_net_2", &b.ff_out },
                };
                inline for (leaves) |lf| {
                    // `catch continue` would be COMPTIME control flow inside an
                    // unrolled `inline for`; fall back to an empty key and skip
                    // it at runtime instead.
                    const k = std.fmt.bufPrint(kb, "{s}_transformer_blocks_{d}_{s}", .{ prefix, bi, lf[0] }) catch "";
                    if (k.len != 0) one(st, k, lf[1], rb, acc);
                }
            }
        }
    };

    for (u.down, 0..) |*blk, bi| {
        const attns = blk.attns orelse continue;
        for (attns, 0..) |*t, ai| {
            var pbuf: [64]u8 = undefined;
            const prefix = std.fmt.bufPrint(&pbuf, "down_blocks_{d}_attentions_{d}", .{ bi, ai }) catch continue;
            Bind.transformer(stack, prefix, t, &kbuf, &rbuf, &matched);
        }
    }
    Bind.transformer(stack, "mid_block_attentions_0", &u.mid.attn, &kbuf, &rbuf, &matched);
    for (u.up, 0..) |*blk, bi| {
        const attns = blk.attns orelse continue;
        for (attns, 0..) |*t, ai| {
            var pbuf: [64]u8 = undefined;
            const prefix = std.fmt.bufPrint(&pbuf, "up_blocks_{d}_attentions_{d}", .{ bi, ai }) catch continue;
            Bind.transformer(stack, prefix, t, &kbuf, &rbuf, &matched);
        }
    }

    log.info("[sdxl] lora attached: {d} module bindings\n", .{matched});
    return matched;
}

pub fn detachLora(u: *Unet) void {
    const clearT = struct {
        fn go(t: *Transformer2D) void {
            t.proj_in.clearLoraRefs();
            t.proj_out.clearLoraRefs();
            for (t.blocks) |*b| {
                b.attn1.q.clearLoraRefs();
                b.attn1.k.clearLoraRefs();
                b.attn1.v.clearLoraRefs();
                b.attn1.o.clearLoraRefs();
                b.attn2.q.clearLoraRefs();
                b.attn2.k.clearLoraRefs();
                b.attn2.v.clearLoraRefs();
                b.attn2.o.clearLoraRefs();
                b.ff_proj.clearLoraRefs();
                b.ff_out.clearLoraRefs();
            }
        }
    }.go;

    for (u.down) |*blk| if (blk.attns) |attns| for (attns) |*t| clearT(t);
    clearT(&u.mid.attn);
    for (u.up) |*blk| if (blk.attns) |attns| for (attns) |*t| clearT(t);
}
