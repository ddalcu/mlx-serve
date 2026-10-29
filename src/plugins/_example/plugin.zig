//! The example outside plugin, built from the kit as a dylib: a llama-shaped decoder (RMS norms, GQA attention with
//! RoPE through the engine's KV cache, a SwiGLU MLP, a tied head) in mlx-lm's weight names, on mlx-c alone.
//! `mlx-serve plugin new <name>` copies this folder.

const std = @import("std");
const kit = @import("mlx_serve");
const c = @import("mlx_c");
const abi = kit.abi.plugin;
pub const config = @import("config.zig");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{
    .name = "_example",
    .kind = .model,
    .claims = &config.model_types,
};

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{},
    .weights = &.{},
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

/// f32 weights; K and V rows at the activation width.
fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    _ = shape;
    const k = config.of(p);
    const h: u64 = k.hidden_size;
    const q: u64 = @as(u64, k.num_attention_heads) * k.head_dim;
    const kv: u64 = @as(u64, k.num_key_value_heads) * k.head_dim;
    const layer = 2 * h + 2 * h * q + 2 * h * kv + 3 * h * k.intermediate_size;
    out.* = .{
        .load_bytes = 4 * (@as(u64, k.vocab_size) * h + h + k.num_hidden_layers * layer),
        .kv_per_token = 4 * 2 * kv * k.num_hidden_layers,
    };
}

const Layer = struct { ln1: c.mlx_array, q: c.mlx_array, k: c.mlx_array, v: c.mlx_array, o: c.mlx_array, ln2: c.mlx_array, gate: c.mlx_array, up: c.mlx_array, down: c.mlx_array };

const Model = struct {
    cfg: config.Config,
    caps: abi.Caps,
    stream: c.mlx_stream,
    embed: c.mlx_array,
    norm: c.mlx_array,
    layers: []Layer,
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const src = ctx.weights orelse {
        out.* = Verdict.refuse(.unsupported, "_example: reads the engine's loaded tensors (LoadCtx.weights)");
        return null;
    };
    const cfg = config.of(ctx.parsed).*;
    const m = gpa.create(Model) catch {
        out.* = Verdict.refuse(.failed, "_example: out of memory");
        return null;
    };
    const layers = gpa.alloc(Layer, cfg.num_hidden_layers) catch {
        gpa.destroy(m);
        out.* = Verdict.refuse(.failed, "_example: out of memory");
        return null;
    };
    for (layers) |*l| inline for (@typeInfo(Layer).@"struct".field_names) |f| {
        @field(l, f) = c.mlx_array_new();
    };
    m.* = .{ .cfg = cfg, .caps = ctx.parsed.caps, .stream = .{ .ctx = ctx.stream.ctx }, .embed = c.mlx_array_new(), .norm = c.mlx_array_new(), .layers = layers };
    var name_buf: [128]u8 = undefined;
    var ok = take(src, "model.embed_tokens.weight", &m.embed) and take(src, "model.norm.weight", &m.norm);
    for (layers, 0..) |*l, i| inline for (.{
        .{ "ln1", "input_layernorm" }, .{ "q", "self_attn.q_proj" }, .{ "k", "self_attn.k_proj" },
        .{ "v", "self_attn.v_proj" },  .{ "o", "self_attn.o_proj" }, .{ "ln2", "post_attention_layernorm" },
        .{ "gate", "mlp.gate_proj" },  .{ "up", "mlp.up_proj" },     .{ "down", "mlp.down_proj" },
    }) |w| {
        const name = std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}.weight", .{ i, w[1] }) catch unreachable;
        ok = ok and take(src, name, &@field(l, w[0]));
    };
    if (!ok) {
        unload(m);
        out.* = Verdict.refuse(.failed, "_example: a weight is missing");
        return null;
    }
    return m;
}

/// A new reference on the source's tensor `name`; false when the pack has none.
fn take(src: *const kit.abi.quant.Source, name: []const u8, dst: *c.mlx_array) bool {
    const t = src.tensor(name);
    if (t.ctx == null) return false;
    return c.mlx_array_set(dst, .{ .ctx = t.ctx }) == 0;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    _ = c.mlx_array_free(m.embed);
    _ = c.mlx_array_free(m.norm);
    for (m.layers) |l| inline for (@typeInfo(Layer).@"struct".field_names) |f| {
        _ = c.mlx_array_free(@field(l, f));
    };
    gpa.free(m.layers);
    gpa.destroy(m);
}

/// The arrays of one forward, freed together.
const Graph = struct {
    s: c.mlx_stream,
    live: std.ArrayList(c.mlx_array) = .empty,

    fn deinit(g: *Graph) void {
        for (g.live.items) |x| _ = c.mlx_array_free(x);
        g.live.deinit(gpa);
    }
    /// `f(&out, args..., stream)`, the result kept until `deinit`.
    fn op(g: *Graph, comptime f: anytype, args: anytype) !c.mlx_array {
        var r = c.mlx_array_new();
        if (@call(.auto, f, .{&r} ++ args ++ .{g.s}) != 0) {
            _ = c.mlx_array_free(r);
            return error.MlxOpFailed;
        }
        try g.live.append(gpa, r);
        return r;
    }
    fn linear(g: *Graph, x: c.mlx_array, w: c.mlx_array) !c.mlx_array {
        const axes = [_]c_int{ 1, 0 };
        return g.op(c.mlx_matmul, .{ x, try g.op(c.mlx_transpose_axes, .{ w, &axes, axes.len }) });
    }
    fn reshape(g: *Graph, x: c.mlx_array, shape: []const c_int) !c.mlx_array {
        return g.op(c.mlx_reshape, .{ x, shape.ptr, shape.len });
    }
    fn swap12(g: *Graph, x: c.mlx_array) !c.mlx_array {
        const axes = [_]c_int{ 0, 2, 1, 3 };
        return g.op(c.mlx_transpose_axes, .{ x, &axes, axes.len });
    }
    /// A new reference for the host.
    fn out(x: c.mlx_array) abi.Array {
        var r = c.mlx_array_new();
        _ = c.mlx_array_set(&r, x);
        return .{ .ctx = r.ctx };
    }
};

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "_example: forward mode not declared");
    if (f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "_example: no layer capture");
    const ops = f.cache_ops orelse return Verdict.refuse(.failed, "_example: a decoder needs the engine's cache");
    const slots = f.caches.len;
    if (slots == 0 or f.rows == 0 or f.positions.len != slots) return Verdict.refuse(.failed, "_example: tokens are not rows x slots");
    // The engine hands its ids either on the host (`tokens`) or as a device array `[slots, rows]` (`ids`).
    if (f.ids.ctx == null) {
        const ids = f.tokens.slice();
        if (ids.len != f.rows * slots) return Verdict.refuse(.failed, "_example: tokens are not rows x slots");
        for (ids) |t| if (t >= m.cfg.vocab_size) return Verdict.refuse(.failed, "_example: token id past the vocabulary");
    }
    run(m, f, ops, out) catch return Verdict.refuse(.failed, "_example: an MLX op failed");
    return .ok;
}

/// Each slot runs on its own cache and position; the rows stack slot after slot.
fn run(m: *Model, f: *const abi.Forward, ops: *const abi.CacheOps, out: *abi.ForwardOut) !void {
    var g: Graph = .{ .s = m.stream };
    defer g.deinit();
    const rows: c_int = @intCast(f.rows);
    var hiddens: std.ArrayList(c.mlx_array) = .empty;
    defer hiddens.deinit(gpa);
    for (f.caches.slice(), f.positions.slice(), 0..) |cache, pos, i| {
        const slot: c_int = @intCast(i);
        const ids = if (f.ids.ctx != null)
            try g.reshape(try g.op(c.mlx_slice, .{ c.mlx_array{ .ctx = f.ids.ctx }, &[_]c_int{ slot, 0 }, 2, &[_]c_int{ slot + 1, rows }, 2, &[_]c_int{ 1, 1 }, 2 }), &.{rows})
        else blk: {
            const toks = f.tokens.slice()[i * f.rows ..][0..f.rows];
            const a = c.mlx_array_new_data(toks.ptr, &[_]c_int{rows}, 1, c.MLX_UINT32);
            try g.live.append(gpa, a);
            break :blk a;
        };
        const x = try decoder(m, &g, ops, cache, ids, @intCast(pos));
        const h = if (f.logits == .last_row) try g.op(c.mlx_slice, .{ x, &[_]c_int{ rows - 1, 0 }, 2, &[_]c_int{ rows, @intCast(m.cfg.hidden_size) }, 2, &[_]c_int{ 1, 1 }, 2 }) else x;
        try hiddens.append(gpa, h);
    }
    const vec = c.mlx_vector_array_new_data(hiddens.items.ptr, hiddens.items.len);
    defer _ = c.mlx_vector_array_free(vec);
    const h = try g.op(c.mlx_concatenate_axis, .{ vec, 0 });
    if (f.logits != .none) out.logits = Graph.out(try g.linear(h, m.embed));
    if (f.want_hidden) out.hidden = Graph.out(h);
}

/// One slot's rows through the trunk: the final-normed hidden `[rows, hidden]`.
fn decoder(m: *Model, g: *Graph, ops: *const abi.CacheOps, cache: abi.Handle, ids: c.mlx_array, pos: c_int) !c.mlx_array {
    const k = m.cfg;
    const t: c_int = @intCast(c.mlx_array_size(ids));
    var x = try g.reshape(try g.op(c.mlx_take_axis, .{ m.embed, ids, 0 }), &.{ 1, t, @intCast(k.hidden_size) });
    const nh: c_int = @intCast(k.num_attention_heads);
    const nkv: c_int = @intCast(k.num_key_value_heads);
    const hd: c_int = @intCast(k.head_dim);
    const base: c.mlx_optional_float = .{ .value = k.rope_theta, .has_value = true };
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(k.head_dim)));
    const no_array = c.mlx_array_new();
    for (m.layers, 0..) |l, li| {
        const h = try g.op(c.mlx_fast_rms_norm, .{ x, l.ln1, k.rms_norm_eps });
        const q = try g.op(c.mlx_fast_rope, .{ try g.swap12(try g.reshape(try g.linear(h, l.q), &.{ 1, t, nh, hd })), hd, false, base, 1.0, pos, no_array });
        const kk = try g.op(c.mlx_fast_rope, .{ try g.swap12(try g.reshape(try g.linear(h, l.k), &.{ 1, t, nkv, hd })), hd, false, base, 1.0, pos, no_array });
        const v = try g.swap12(try g.reshape(try g.linear(h, l.v), &.{ 1, t, nkv, hd }));
        var view: abi.KvView = .{};
        const app = ops.append(cache, @intCast(li), .{ .ctx = kk.ctx }, .{ .ctx = v.ctx }, 0, .{ .ctx = g.s.ctx }, &view);
        if (!app.isOk()) return error.CacheRefused;
        defer ops.release(&view);
        const attn = try g.op(c.mlx_fast_scaled_dot_product_attention, .{ q, c.mlx_array{ .ctx = view.k.ctx }, c.mlx_array{ .ctx = view.v.ctx }, scale, if (t > 1) "causal" else "", no_array, no_array, false });
        const o = try g.linear(try g.reshape(try g.swap12(attn), &.{ 1, t, nh * hd }), l.o);
        x = try g.op(c.mlx_add, .{ x, o });
        const h2 = try g.op(c.mlx_fast_rms_norm, .{ x, l.ln2, k.rms_norm_eps });
        const gate = try g.linear(h2, l.gate);
        const act = try g.op(c.mlx_multiply, .{ gate, try g.op(c.mlx_sigmoid, .{gate}) });
        const mlp = try g.linear(try g.op(c.mlx_multiply, .{ act, try g.linear(h2, l.up) }), l.down);
        x = try g.op(c.mlx_add, .{ x, mlp });
    }
    const normed = try g.op(c.mlx_fast_rms_norm, .{ x, m.norm, k.rms_norm_eps });
    return g.reshape(normed, &.{ t, @intCast(k.hidden_size) });
}

/// The tiny pack the conformance suite runs, complete enough to serve (`mlx-serve plugin fixture`): the 256 bytes
/// plus four specials, width 16, two layers, GQA 2/1, seeded random weights, a role-tag chat template.
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    const fx = kit.sdk.fixture;
    const v, const h, const ff, const q, const kv = .{ 260, 16, 32, 16, 8 };
    var tensors: std.ArrayList(fx.Tensor) = .empty;
    const ones = try a.alloc(f32, h);
    @memset(ones, 1);
    try tensors.append(a, .{ .name = "model.embed_tokens.weight", .shape = &.{ v, h }, .data = try fx.randomF32(a, 1, v * h, 0.5) });
    try tensors.append(a, .{ .name = "model.norm.weight", .shape = &.{h}, .data = ones });
    for (0..2) |i| {
        const seed: u64 = 10 + 10 * i;
        inline for (.{
            .{ "input_layernorm", &[_]u64{h}, 0 },        .{ "self_attn.q_proj", &[_]u64{ q, h }, 1 },
            .{ "self_attn.k_proj", &[_]u64{ kv, h }, 2 }, .{ "self_attn.v_proj", &[_]u64{ kv, h }, 3 },
            .{ "self_attn.o_proj", &[_]u64{ h, q }, 4 },  .{ "post_attention_layernorm", &[_]u64{h}, 0 },
            .{ "mlp.gate_proj", &[_]u64{ ff, h }, 5 },    .{ "mlp.up_proj", &[_]u64{ ff, h }, 6 },
            .{ "mlp.down_proj", &[_]u64{ h, ff }, 7 },
        }) |w| {
            var n: usize = 1;
            for (w[1]) |d| n *= d;
            try tensors.append(a, .{
                .name = try std.fmt.allocPrint(a, "model.layers.{d}.{s}.weight", .{ i, w[0] }),
                .shape = w[1],
                .data = if (w[2] == 0) ones else try fx.randomF32(a, seed + w[2], n, 0.3),
            });
        }
    }
    const specials = [_]kit.sdk.byte_tokenizer.Special{
        .{ .id = 256, .content = "<|end|>" },  .{ .id = 257, .content = "<|system|>" },
        .{ .id = 258, .content = "<|user|>" }, .{ .id = 259, .content = "<|assistant|>" },
    };
    return .{ .files = try a.dupe(fx.File, &.{
        .{ .name = "config.json", .bytes = "{\"model_type\":\"_example\",\"vocab_size\":260,\"hidden_size\":16,\"intermediate_size\":32,\"num_hidden_layers\":2,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"rms_norm_eps\":1e-5,\"tie_word_embeddings\":true,\"eos_token_id\":256}" },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.7}" },
        .{ .name = "tokenizer.json", .bytes = try kit.sdk.byte_tokenizer.json(a, &specials) },
        .{ .name = "tokenizer_config.json", .bytes =
        \\{"eos_token":"<|end|>","chat_template":"{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"}
        },
        .{ .name = "model.safetensors", .bytes = try fx.safetensors(a, tensors.items) },
    }) };
}
